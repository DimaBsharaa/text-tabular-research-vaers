"""Prepare private, human-readable views of the existing A candidate.

No new sources, model training, permission requests or benchmark changes. Keep
text-bearing CSVs/notebook outside Git; published summaries contain metadata only.
"""
import json
import math
import re

import pandas as pd

from research.scripts.tennis.common import OUT, PRIVATE, digest, text_is_detected, write_json

REVIEW = PRIVATE / "candidate_A_v0_review"
CUES = {
    "condition": r"\b(injur\w*|pain\w*|hurt\w*|fatigue\w*|tired\w*|physio\w*|recover\w*|cramp\w*|shoulder\w*|knee\w*|ankle\w*|wrist\w*)\b",
    "tactics": r"\b(tactic\w*|strateg\w*|aggress\w*|slice\w*|volley\w*|forehand\w*|backhand\w*|baseline\w*)\b",
    "surface": r"\b(clay|grass|hard court|surface\w*|bounce\w*)\b",
}


def chronological_partitions(frame):
    """Fix date-blocked 60/20/20 roles by row positions without reading targets."""
    ordered = frame.sort_values(["match_date", "match_id"])
    development_start = ordered.match_date.iloc[int(.6 * len(ordered))]
    reserved_test_start = ordered.match_date.iloc[int(.8 * len(ordered))]
    split = pd.Series("train", index=ordered.index)
    split.loc[ordered.match_date >= development_start] = "development"
    split.loc[ordered.match_date >= reserved_test_start] = "reserved_test"
    result = ordered[["match_id", "match_date"]].copy()
    result["split"] = split
    return result, {"development_start": development_start, "reserved_test_start": reserved_test_start}


def clipping_audit(interviews):
    rows = []
    for row in interviews.itertuples():
        words = row.answer_text.split()
        first = " ".join(words[:300])
        item = {"interview_id": row.interview_id, "full_answer_words": len(words),
                "first_passage_words": min(300, len(words)), "word_chunks_of_300": math.ceil(len(words) / 300)}
        for topic, pattern in CUES.items():
            item[topic + "_cue_full"] = bool(re.search(pattern, row.answer_text, re.I))
            item[topic + "_cue_first300"] = bool(re.search(pattern, first, re.I))
        rows.append(item)
    return pd.DataFrame(rows)


def prepare():
    REVIEW.mkdir(parents=True, exist_ok=True)
    original_schema = json.loads((PRIVATE / "variant_A_schema.json").read_text())
    frame = pd.read_csv(PRIVATE / "variant_A.csv")
    columns = original_schema["structured_columns"] + original_schema["text_columns"]
    selected = frame[["match_id"] + columns + [original_schema["target"]]]
    selected.to_csv(REVIEW / "features.csv", index=False)
    audit_columns = [c for c in frame if c not in columns + [original_schema["target"]]]
    frame[audit_columns].to_csv(REVIEW / "audit.csv", index=False)
    partitions, cutoffs = chronological_partitions(frame)
    partitions.to_csv(REVIEW / "splits.csv", index=False)
    by_split = frame.merge(partitions[["match_id", "split"]], on="match_id", validate="one_to_one")
    train = by_split[by_split.split == "train"]
    training_doc_ids = set(train.a_interview_id.astype(int)) | set(train.b_interview_id.astype(int))
    split_rows = []
    for name in ["train", "development", "reserved_test"]:
        group = by_split[by_split.split == name]
        previous = by_split[by_split.match_date < group.match_date.min()]
        current_ids = set(group.match_id)
        refs = set(previous.a_source_match_id) | set(previous.b_source_match_id)
        split_rows.append({"split": name, "rows": len(group), "first_day": group.match_date.min(),
            "last_day": group.match_date.max(), "a_wins": int(group.target.sum()),
            "b_wins": int(len(group) - group.target.sum()),
            "current_targets_referenced_by_earlier_partitions": len(refs & current_ids)})
    pd.DataFrame(split_rows).to_csv(OUT / "interview_candidate_split_plan.csv", index=False)
    used_ids = set(frame.a_interview_id.astype(int)) | set(frame.b_interview_id.astype(int))
    full = pd.read_csv(PRIVATE / "interviews_with_answers.csv")
    full = full[full.interview_id.isin(used_ids)].copy()
    clips = clipping_audit(full)
    clips["in_training_partition"] = clips.interview_id.isin(training_doc_ids)
    clips.to_csv(OUT / "interview_candidate_clipping.csv", index=False)
    training_clips = clips[clips.in_training_partition]
    mentions = []
    for topic in CUES:
        full_flag, clipped_flag = training_clips[topic + "_cue_full"], training_clips[topic + "_cue_first300"]
        mentions.append({"topic": topic, "scope": "training_partition_interviews_only",
            "documents_with_cue_in_full": int(full_flag.sum()),
            "documents_with_cue_in_first300": int(clipped_flag.sum()),
            "documents_with_cue_only_after_first300": int((full_flag & ~clipped_flag).sum()),
            "fraction_of_full_cue_documents_lost": float((full_flag & ~clipped_flag).sum() / full_flag.sum()) if full_flag.any() else 0})
    pd.DataFrame(mentions).to_csv(OUT / "interview_candidate_clipping_summary.csv", index=False)
    # Three concise excerpts are shown only in the private notebook, with exact
    # publisher attribution. No interview excerpt is saved in repo metadata.
    examples = []
    for iid, publisher in [(854, 45075), (2557, 61198), (3406, 69159)]:
        row = train[(train.a_interview_id == iid) | (train.b_interview_id == iid)].sort_values(["match_date", "match_id"]).iloc[0]
        side = "a" if int(row.a_interview_id) == iid else "b"
        text = row[f"player_{side}_text"]
        words = text.split()
        cue = next((i for i, word in enumerate(words) if re.search(CUES["condition"], word, re.I)), None)
        start = max(0, cue - 4) if cue is not None else 0
        examples.append({"match_id": row.match_id, "target_date": row.match_date,
            "player_a": row.player_a, "player_b": row.player_b, "a_rank": row.a_rank, "b_rank": row.b_rank,
            "a_form10": row.a_form10, "b_form10": row.b_form10, "surface": row.surface,
            "selected_text_side": side, "source_interview_id": iid, "source_date": row[f"{side}_text_date"],
            "actual_input_excerpt_12_words": " ".join(words[start:start + 12]),
            "publisher_url": f"https://www.asapsports.com/show_interview.php?id={publisher}",
            "player_a_wins": row.target, "publication_verified": bool(row.publication_verified)})
    pd.DataFrame(examples).to_csv(REVIEW / "examples_private.csv", index=False)
    nulls = selected[columns].isna().sum().rename("missing_rows").rename_axis("column").reset_index()
    nulls["missing_fraction"] = nulls.missing_rows / len(frame)
    nulls.to_csv(OUT / "interview_candidate_missingness.csv", index=False)
    schema = original_schema | {
        "csv_sha256": digest((REVIEW / "features.csv").read_bytes()),
        "audit_columns": ["match_id"], "purpose": "private_candidate_review_only",
        "strict_publication_temporal_gate_passed": False, "redistribution_cleared": False,
        "preprocessing": "Original fixed first-300-answer-word input; missing numerical values preserved. No fitted imputer/scaler.",
        "split_file": "splits.csv", "source_csv_sha256": digest((PRIVATE / "variant_A.csv").read_bytes())}
    write_json(REVIEW / "schema.json", schema)
    summary = {"rows": len(frame), "structured_columns": len(original_schema["structured_columns"]),
        "text_columns": len(original_schema["text_columns"]), "target": original_schema["target"],
        "csv_columns_including_audit_id_and_target": len(selected.columns), "unique_interviews": len(full),
        "training_unique_interviews": len(training_clips),
        "median_full_interview_words": float(clips.full_answer_words.median()),
        "median_word_chunks_of_300": float(clips.word_chunks_of_300.median()),
        "documents_over_300_words": int((clips.full_answer_words > 300).sum()),
        "cutoffs": cutoffs, "split_rows": split_rows,
        "text_detection_on_training_partition": {c: text_is_detected(train[c]) for c in original_schema["text_columns"]},
        "clipping_summary": mentions, "publication_verification_complete": False,
        "model_fits": 0, "official_model_code_modified": False, "outreach_sent": False,
        "feature_csv_sha256": schema["csv_sha256"], "files_private": ["features.csv", "audit.csv", "splits.csv", "schema.json", "examples_private.csv", "candidate_workbench.ipynb"]}
    write_json(OUT / "interview_candidate_review.json", summary)
    (REVIEW / "README.md").write_text("""# Tennis interview candidate v0 — private review

Open candidate_workbench.ipynb to see executed tables, plots and three concise real text excerpts.
features.csv contains 4,052 provisional rows: 16 structured columns, two text columns, target and an audit match_id.
match_id is never a predictor. audit.csv holds player/source identities and date evidence; join by match_id.
splits.csv fixes chronological training/development/reserved-test roles before any A model fits.
schema.json explicitly retains unresolved publication/reuse flags. This is a review export, not a cleared benchmark input.

Text is the existing first 300 player-answer words per side, without invented summaries or segmentation-based selection.
Missing values remain missing. A future imputer/vocabulary/PCA must be fitted only on the appropriate training rows.
All transcript-bearing files stay in this external cache and must not be committed/pushed as a public dataset.
No new sources, training or messages to anyone were part of this preparation.
""")
    return summary


def notebook():
    from research.scripts.tennis.notebooks import code, markdown, save_execute
    cells = [markdown('''
        # Tennis interview candidate v0: local review workbench

        Executed from existing cached data. This notebook and its text-bearing CSV stay outside Git.
        It displays three concise source-attributed excerpts, not full interviews. No new model has been trained.
        The rows remain provisional; source-use and historical publication questions have not been silently cleared.
    '''), code(f'''
        import json
        from pathlib import Path
        import pandas as pd
        import matplotlib.pyplot as plt
        from IPython.display import display
        review = Path({str(REVIEW)!r})
        out = Path({str(OUT)!r})
        info = json.loads((out / "interview_candidate_review.json").read_text())
        schema = json.loads((review / "schema.json").read_text())
        display(pd.DataFrame([{{"Rows": info["rows"], "Structured": info["structured_columns"],
                              "Text": info["text_columns"], "Target": info["target"], "Model fits": 0}}]))
        display(pd.DataFrame([{{"role": "structured", "column": c}} for c in schema["structured_columns"]] +
                             [{{"role": "text", "column": c}} for c in schema["text_columns"]] +
                             [{{"role": "target", "column": "target"}}, {{"role": "audit only", "column": "match_id"}}]))
        print("Private feature CSV:", review / "features.csv")
    '''), code('''
        examples = pd.read_csv(review / "examples_private.csv")
        display(examples)
    '''), markdown('''
        **What to inspect:** these examples are from the training partition. Rankings/form/context describe prior measurable strength. The actual prior interview words
        can discuss condition or preparation. Their complementary predictive value is still a hypothesis.
        These excerpts come from the current clipped input, not an invented summary. Player A/B orientation is independent
        of outcome; the previous interview is never used to predict its own source match.
    '''), code('''
        splits = pd.read_csv(out / "interview_candidate_split_plan.csv")
        display(splits)
        ax = splits.plot.bar(x="split", y="rows", legend=False, rot=0, figsize=(7, 3), color="#4f86a8")
        ax.set(ylabel="Target match rows", title="Chronological roles fixed before A model fitting")
        plt.tight_layout(); plt.show()
        nulls = pd.read_csv(out / "interview_candidate_missingness.csv")
        display(nulls[nulls.missing_rows > 0])
    '''), markdown('''
        **Insight:** whole dates stay in one partition. An earlier partition cannot use a later partition's target match
        as an interview source. This is a proposed diagnostic split, not a change to the official random benchmark.
        A rolling test can use genuinely earlier observed match results, including earlier test-period matches;
        it must not be described as predicting the whole test block at one frozen date. Missing values are preserved
        for training-only handling. Publication verification remains incomplete.
    '''), code('''
        clipping = pd.read_csv(out / "interview_candidate_clipping_summary.csv")
        display(clipping)
        plot = clipping.set_index("topic")[["documents_with_cue_in_first300", "documents_with_cue_only_after_first300"]]
        ax = plot.plot.bar(stacked=True, figsize=(8, 3), rot=0, color=["#4f86a8", "#c47a55"])
        ax.set(ylabel="Unique training interviews", title="Lexical cues retained or lost by the first-300-word policy")
        ax.legend(["Cue retained", "Cue only in omitted answers"], loc="upper right")
        plt.tight_layout(); plt.show()
        print("Unique interviews:", info["unique_interviews"])
        print("Training interviews used for passage-policy diagnostics:", info["training_unique_interviews"])
        print("Median full words:", info["median_full_interview_words"])
        print("Median chunks of 300 words:", info["median_word_chunks_of_300"])
        print("Text cardinality rule on proposed training pool:", info["text_detection_on_training_partition"])
    '''), markdown('''
        **Insight:** this passage-policy summary uses training interviews only. Lexical cues are screening flags, not verified injuries/tactics or evidence of predictive value.
        Clipping can discard relevant-looking language. The diagnostic did not choose passages using targets or test scores.
        Three-hundred-word chunks are not E5 tokens: the official encoder has a token limit. Pooling multiple chunk embeddings
        would change the representation pipeline, so it is not silently introduced. Keep the unchanged benchmark and decide
        on one fixed, label-blind input passage policy before a real evaluation.

        **Current use:** inspect this CSV and notebook; no need to start Colab yet. No outreach is part of this work.
    ''')]
    save_execute("tennis_candidate_workbench", cells, output_path=REVIEW / "candidate_workbench.ipynb")


def main():
    info = prepare()
    notebook()
    print(json.dumps(info, indent=2))
    print("Private workbench:", REVIEW / "candidate_workbench.ipynb")


if __name__ == "__main__":
    main()
