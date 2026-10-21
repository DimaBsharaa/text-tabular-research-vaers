"""Audit the actual Cornell JSON and original dates without installing ConvoKit.

Heavy parsed SpaCy data is deliberately never extracted. Raw answers and pilot
CSVs stay in the external cache. Tracked outputs contain metadata/counts only.
"""
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
from zipfile import ZipFile

import pandas as pd

from tennis.scripts.common import CACHE, OUT, PRIVATE, digest, write_json

TOPICS = {"tactics": r"\b(tactic|strategy|aggress|slice|volley|forehand|backhand|rall|net\b|game plan)",
          "serve_return": r"\b(serv|return|break point)",
          "condition": r"\b(injur|pain|hurt|fatigue|tired|physio|recover|cramp|shoulder|knee|ankle)",
          "mentality": r"\b(confiden|nerv|mental|pressure|focus|concentrat)",
          "surface": r"\b(clay|grass|hard court|surface|bounce)"}


def main():
    OUT.mkdir(parents=True, exist_ok=True); PRIVATE.mkdir(parents=True, exist_ok=True)
    folder = CACHE / "cornell"
    archive_metadata = []
    for archive in ["tennis-corpus.zip", "tennis_data.zip"]:
        with ZipFile(folder / archive) as z:
            archive_metadata.append({"archive": archive, "compressed_bytes": (folder / archive).stat().st_size,
                                     "uncompressed_bytes": sum(f.file_size for f in z.infolist()),
                                     "sha256": digest((folder / archive).read_bytes()), "files": z.namelist()})
            for file in z.namelist():
                base = Path(file).name
                if file.startswith("__MACOSX"):
                    continue
                wanted = {"utterances.jsonl", "conversations.json", "users.json", "index.json", "corpus.json"} if archive == "tennis-corpus.zip" else {"transcripts_matchinfo.json", "README.txt"}
                if base in wanted:
                    target = folder / (("convokit_" if archive == "tennis-corpus.zip" else "") + base)
                    if not target.exists(): target.write_bytes(z.read(file))
    original = json.loads((folder / "transcripts_matchinfo.json").read_text())
    conversations = json.loads((folder / "convokit_conversations.json").read_text())
    users = json.loads((folder / "convokit_users.json").read_text())
    pair_counts, speaker_counts, flags, timestamp_counts = Counter(), Counter(), Counter(), Counter()
    date_mismatches, answer_mismatches, reply_mismatches = [], [], []
    per_interview_dates = defaultdict(set)
    answer_counts = Counter(); answer_hashes = Counter(); long_answer_hashes = Counter(); total_answers = short_answers = empty_answers = boilerplate = 0
    for line in (folder / "convokit_utterances.jsonl").open():
        u = json.loads(line); meta = u["meta"]; cid = u["root"]; iid = str(conversations[cid]["match_id"])
        pair_counts[cid] += 1; speaker_counts[u["user"]] += 1; flags.update(k for k in ["answer" if meta["is_answer"] else "question"])
        per_interview_dates[iid].add(u["timestamp"]); timestamp_counts[str(u["timestamp"])] += 1
        if u["timestamp"] != original[iid]["date"]: date_mismatches.append(u["id"])
        if meta["is_answer"]:
            answer_counts[iid] += 1; total_answers += 1
            words = u["text"].split(); short_answers += len(words) <= 5; empty_answers += not u["text"].strip()
            boilerplate += bool(re.fullmatch(r"\s*(yes|no|yeah|thank you|thanks|okay|ok)[.!? ]*", u["text"], re.I))
            answer_hashes[digest(u["text"].strip().lower())] += 1
            if len(words) >= 30: long_answer_hashes[digest(" ".join(u["text"].lower().split()))] += 1
            pair = int(meta["pair_idx"].rsplit("_", 1)[1])
            if u["text"] != original[iid]["QandA"][pair][1]: answer_mismatches.append(u["id"])
            if u["reply-to"] != cid: reply_mismatches.append(u["id"])
    records = []
    for iid, v in original.items():
        answers = [str(pair[1]).strip() for pair in v["QandA"]]
        text = "\n\n".join(a for a in answers if a)
        rec = {"interview_id": iid, **{k: v[k] for k in v if k != "QandA"},
               "qa_pairs": len(v["QandA"]), "answer_words": len(text.split()), "answer_chars": len(text),
               "answer_sha256": digest(text), "convokit_answers": answer_counts[iid],
               "original_date_matches_all_utterances": per_interview_dates[iid] == {v["date"]},
               "opponent_known": v["opponent"] not in {"nan", "", "None"},
               "answer_text": text}
        for topic, pattern in TOPICS.items(): rec["keyword_" + topic] = bool(re.search(pattern, text, re.I))
        records.append(rec)
    data = pd.DataFrame(records); data["year"] = data.date.str[:4].astype(int)
    data.to_csv(PRIVATE / "interviews_with_answers.csv", index=False)
    metadata = data.drop(columns="answer_text")
    metadata.to_csv(OUT / "cornell_interview_metadata.csv", index=False)
    for col in ["gender", "year", "stage", "tournament", "player", "result"]:
        data.groupby(col, dropna=False).agg(interviews=("interview_id", "size"), median_answer_words=("answer_words", "median")).reset_index().to_csv(OUT / f"cornell_by_{col}.csv", index=False)
    # Purposive short/medium-length review: multiple genders, years, outcomes, stages.
    pool = data[(data.answer_words >= 70) & (data.answer_words <= 350)].copy()
    pool["order"] = pool.interview_id.map(lambda x: digest("manual-review-v1:" + str(x)))
    sample = pool.sort_values("order").groupby(["year", "gender", "result"], group_keys=False).head(1).sort_values(["year", "gender", "result"])
    sample[["interview_id", "date", "player", "gender", "stage", "result", "answer_words"]].to_csv(OUT / "manual_review_sample.csv", index=False)
    write_json(PRIVATE / "manual_review_texts.json", [{"interview_id": r.interview_id, "player": r.player, "date": r.date,
                   "gender": r.gender, "stage": r.stage, "result": r.result, "qa": original[r.interview_id]["QandA"]} for r in sample.itertuples()])
    info = {"archives": archive_metadata, "speakers_including_reporter": len(users), "players": data.player.nunique(),
            "utterances": sum(flags.values()), "qa_conversations": len(conversations), "interviews": len(data),
            "unique_source_match_id_values": len({v["match_id"] for v in conversations.values()}),
            "source_match_id_is_interview_key": True, "all_interview_keys_in_original": set(original) == {str(v["match_id"]) for v in conversations.values()},
            "gender_interviews": data.gender.value_counts().to_dict(), "gender_players": data.groupby("gender").player.nunique().to_dict(),
            "source_result_counts": data.result.value_counts().to_dict(), "date_min": data.date.min(), "date_max": data.date.max(),
            "missing_dates": int(data.date.isna().sum()), "original_vs_utterance_date_mismatches": len(date_mismatches),
            "answer_reconstruction_mismatches": len(answer_mismatches), "reply_link_mismatches": len(reply_mismatches),
            "non_two_utterance_conversations": sum(n != 2 for n in pair_counts.values()),
            "missing_opponent": int((~data.opponent_known).sum()), "missing_rank": int(data.ranking.isna().sum()),
            "duplicate_full_answer_interviews": int(data.answer_sha256.duplicated().sum()),
            "answers": total_answers, "empty_answers": empty_answers, "answers_at_most_5_words": short_answers,
            "exact_boilerplate_answers": boilerplate, "duplicate_answer_occurrences_beyond_first": sum(n - 1 for n in answer_hashes.values()),
            "duplicate_long_answer_sections_beyond_first": sum(n - 1 for n in long_answer_hashes.values()),
            "answer_words_quantiles": data.answer_words.quantile([0,.1,.25,.5,.75,.9,1]).to_dict(),
            "keyword_flags_NOT_human_labels": {k: float(data["keyword_" + k].mean()) for k in TOPICS},
            "manual_review_n": len(sample), "manual_review_sampling": "purposive 70–350 answer words; one per year/gender/result stratum; not representative prevalence",
            "publication_timestamp_available": False, "explicit_redistribution_license_found_in_archives": False}
    write_json(OUT / "cornell_audit.json", info)
    print(json.dumps({k:v for k,v in info.items() if k != "archives"}, indent=2))


if __name__ == "__main__": main()
