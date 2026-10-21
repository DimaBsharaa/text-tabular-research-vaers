"""Traceable, outcome-blind E5-budget passage selection for Cornell A.

Only tokenizer assets are loaded, from an isolated external cache. No weights,
embeddings, fine-tuning or changes to the reference implementation are involved.
"""
import functools
import hashlib
import json
import re
import sys

from tennis.scripts.common import CACHE, digest

MODEL_REVISION = "ffb93f3bd4047442299a41ebb6fa998a38507c52"
TOKENIZER_DIR = CACHE / "tokenizer_audit"
MAX_TOKENS = 512
COLS = ["player_a_text", "player_b_text"]
POLICIES = ["legacy300", "token_head", "head_tail", "topical_answers", "topical_qa"]
TOPICS = {
    "condition": r"\b(injur\w*|pain\w*|hurt\w*|tired\w*|fatigue\w*|recover\w*|physio\w*|cramp\w*|shoulder\w*|wrist\w*|ankle\w*|knee\w*)\b",
    "tactics": r"\b(tactic\w*|strateg\w*|forehand\w*|backhand\w*|serv(?:e|ing|ice)\w*|return\w*|volley\w*|slice\w*|aggress\w*|baseline\w*|rall(?:y|ies))\b",
    "preparation": r"\b(practic\w*|train\w*|prepar\w*|rest\w*|sleep\w*|schedule\w*|adapt\w*|adjust\w*)\b",
    "mentality": r"\b(confiden\w*|nerv\w*|focus\w*|concentrat\w*|pressure\w*)\b",
    "surface": r"\b(grass|clay|surface\w*|hard court|indoor\w*|outdoor\w*|bounce\w*)\b",
}
WEIGHTS = {"condition": 3, "tactics": 2, "preparation": 2, "mentality": 1, "surface": 1}
RULE_VERSION = "cornell-extractive-v1"


def normalize(text):
    return " ".join(str(text).split())


@functools.lru_cache(maxsize=1)
def tokenizer():
    try:
        from tokenizers import Tokenizer
    except ImportError:
        vendor = str(TOKENIZER_DIR / "vendor")
        if vendor not in sys.path:
            sys.path.insert(0, vendor)
        from tokenizers import Tokenizer
    value = Tokenizer.from_file(str(TOKENIZER_DIR / "tokenizer.json"))
    value.no_padding(); value.no_truncation()
    return value


def formatted(text, column=COLS[0]):
    # Matches format_e5_passage in the pinned reference; TAR builds the same
    # string via 'column: value' followed by TextLabelDataset's 'passage: '.
    return f"passage: {column}: {text}"


def count_tokens(text, column=COLS[0]):
    return len(tokenizer().encode(formatted(text, column)).ids)


def body_tokens(text):
    return len(tokenizer().encode(text, add_special_tokens=False).ids)


def cap_head(text, limit=MAX_TOKENS):
    text = normalize(text)
    prefix = formatted("")
    encoded = tokenizer().encode(prefix + text)
    if len(encoded.ids) <= limit:
        return text
    end = encoded.offsets[limit - 2][1] - len(prefix)
    clipped = text[:max(0, end)].rstrip()
    # Cutting a WordPiece at a character offset can change re-tokenization.
    while clipped and max(count_tokens(clipped, col) for col in COLS) > limit:
        clipped = clipped.rsplit(" ", 1)[0] if " " in clipped else clipped[:-1]
    return clipped


def topic_flags(text):
    return {topic: bool(re.search(pattern, text, re.I)) for topic, pattern in TOPICS.items()}


def units(qa):
    output = []
    for pair_idx, (question, answer) in enumerate(qa):
        answer, question = normalize(answer), normalize(question)
        cursor = 0
        for sentence_idx, text in enumerate(re.split(r"(?<=[.!?])\s+", answer)):
            start = answer.find(text, cursor); end = start + len(text); cursor = end
            if not text:
                continue
            output.append({"pair_idx": pair_idx, "sentence_idx": sentence_idx,
                "start": start, "end": end, "text": text, "question": question,
                "answer_sha256": digest(answer), "question_sha256": digest(question)})
    return output


def select(qa, policy):
    """No target/date/opponent/ranking parameters: same document, same passage."""
    if policy not in POLICIES:
        raise ValueError(policy)
    source = " ".join(normalize(pair[1]) for pair in qa if normalize(pair[1]))
    lineage = []
    legacy_over = False
    if policy in {"legacy300", "token_head"}:
        value = " ".join(source.split()[:300]) if policy == "legacy300" else source
        legacy_over = count_tokens(value) > MAX_TOKENS
        text = cap_head(value)
        lineage = [{"kind": "answer_prefix", "joined_answer_start": 0, "joined_answer_end": len(text)}]
    elif policy == "head_tail":
        head = cap_head(source, 256)
        # Keep an original answer suffix; the explicit omission marker is not a
        # generated claim. Finish with a strict formatted-input length check.
        offsets = tokenizer().encode(source, add_special_tokens=False).offsets
        start = offsets[max(0, len(offsets) - 240)][0] if offsets else 0
        tail = source[start:]
        if count_tokens(source) <= MAX_TOKENS:
            text = source
            lineage = [{"kind": "answer_prefix", "joined_answer_start": 0, "joined_answer_end": len(text)}]
        else:
            text = cap_head(head + " [...] " + tail)
            lineage = [{"kind": "head_tail", "head_end": len(head), "tail_start": start,
                        "assembled_before_cap_sha256": digest(head + " [...] " + tail)}]
    else:
        candidates = units(qa)
        for item in candidates:
            flags = topic_flags(item["text"])
            short = len(item["text"].split()) < 20
            # For elliptical short answers, question context may carry the
            # subject and polarity. Both remain verbatim source spans.
            use_q = policy == "topical_qa" and short
            evidence = item["question"] + " " + item["text"] if use_q else item["text"]
            flags = topic_flags(evidence)
            item["topics"] = [k for k, v in flags.items() if v]
            item["rendered"] = ("Q: " + item["question"] + " A: " + item["text"]) if use_q else item["text"]
            item["question_included"] = use_q
            item["tokens"] = body_tokens(item["rendered"])
            item["base_score"] = sum(WEIGHTS[k] for k in item["topics"])
        remaining = [v for v in candidates if v["base_score"] > 0]
        chosen, covered, used_tokens = [], set(), 0
        budget = MAX_TOKENS - count_tokens("")
        pair_counts = {}
        while remaining:
            remaining.sort(key=lambda v: (
                -(v["base_score"] + 2 * len(set(v["topics"]) - covered)) / max(v["tokens"], 1) ** .35,
                v["pair_idx"], v["sentence_idx"]))
            item = remaining.pop(0)
            if pair_counts.get(item["pair_idx"], 0) >= 3 or item["tokens"] + used_tokens > budget:
                continue
            chosen.append(item); used_tokens += item["tokens"]
            covered.update(item["topics"]); pair_counts[item["pair_idx"]] = pair_counts.get(item["pair_idx"], 0) + 1
        if chosen:
            chosen.sort(key=lambda v: (v["pair_idx"], v["sentence_idx"]))
            text = cap_head(" ".join(v["rendered"] for v in chosen))
            lineage = [{k: v[k] for k in ["pair_idx", "sentence_idx", "start", "end", "answer_sha256",
                "question_sha256", "question_included"]} for v in chosen]
        else:
            text = cap_head(source)
            lineage = [{"kind": "fallback_answer_prefix", "joined_answer_start": 0, "joined_answer_end": len(text)}]
    return {"text": text, "policy": policy, "rule_version": RULE_VERSION, "source_answer_sha256": digest(source),
        "selected_sha256": digest(text), "selected_tokens_a": count_tokens(text, COLS[0]),
        "selected_tokens_b": count_tokens(text, COLS[1]), "full_formatted_tokens": count_tokens(source),
        "legacy_was_over_token_limit": legacy_over, "lineage": lineage,
        "full_topics": topic_flags(source), "selected_topics": topic_flags(text)}


def verify_selected(qa, selected):
    source = " ".join(normalize(pair[1]) for pair in qa if normalize(pair[1]))
    if selected["source_answer_sha256"] != digest(source):
        return False
    for line in selected["lineage"]:
        if "pair_idx" in line:
            question, answer = qa[line["pair_idx"]]
            if digest(normalize(answer)) != line["answer_sha256"] or digest(normalize(question)) != line["question_sha256"]:
                return False
            span = normalize(answer)[line["start"]:line["end"]]
            if not span:
                return False
    # Re-run the fixed extraction to validate head/tail assembly and output;
    # source span hashes alone would not detect arbitrary output modification.
    expected = select(qa, selected["policy"])
    return expected["selected_sha256"] == selected["selected_sha256"] == digest(selected["text"])
