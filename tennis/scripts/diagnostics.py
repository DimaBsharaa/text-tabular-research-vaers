"""Small CPU TF-IDF/logistic diagnostics; never official MulTaBench results.

Fixed chronological 80/20 day split, no tuning and no TAR approximation. A is
excluded from modeling until publication availability is established. B/C share
the same target matches for their comparable run; C also has a bounded pilot.
"""
import json
import time

import numpy as np
import pandas as pd
import psutil
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from tennis.scripts.common import OUT, PRIVATE, digest, text_is_detected, write_json
from tennis.scripts.prototypes import NUMERIC, CONTEXT


def chronological_split(df):
    ordered = df.sort_values(["match_date", "match_id"]).reset_index(drop=True)
    cutoff = ordered.match_date.iloc[int(len(ordered) * .8)]
    train = ordered[ordered.match_date < cutoff].copy()
    test = ordered[ordered.match_date >= cutoff].copy()
    if min(len(train), len(test)) < 20 or min(train.target.nunique(), test.target.nunique()) != 2:
        raise ValueError("Insufficient dates/rows/classes for an interpretable diagnostic")
    assert train.match_date.max() < test.match_date.min()
    assert not set(train.match_id) & set(test.match_id)
    return train, test, cutoff


def preprocessor(structured, numeric, text):
    pieces = []
    nums = [c for c in structured if c in numeric]
    cats = [c for c in structured if c not in numeric]
    if nums:
        pieces.append(("numeric", make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler()), nums))
    if cats:
        pieces.append(("categorical", make_pipeline(SimpleImputer(strategy="constant", fill_value="missing"), OneHotEncoder(handle_unknown="ignore")), cats))
    for i, col in enumerate(text):
        pieces.append((f"text{i}", TfidfVectorizer(ngram_range=(1,2), min_df=1, max_features=4000, sublinear_tf=True), col))
    return ColumnTransformer(pieces, sparse_threshold=1.0)


def run_cohort(name, df, structured, text, numeric=NUMERIC):
    train, test, cutoff = chronological_split(df)
    # String missing values handled with fixed category, not corpus-fitted global statistics.
    for d in [train, test]:
        for c in [c for c in structured if c not in numeric]: d[c] = d[c].fillna("missing").astype(str)
    splits = {"train_ids":train.match_id.tolist(), "test_ids":test.match_id.tolist(), "cutoff":cutoff,
              "train_max_date":train.match_date.max(), "test_min_date":test.match_date.min(),
              "player_overlap":len((set(train.player_a_key)|set(train.player_b_key)) & (set(test.player_a_key)|set(test.player_b_key)))}
    write_json(OUT / f"diagnostic_{name}_split.json", splits)
    rows = []
    conditions = {"structured":(structured,[]), "text_tfidf":([],text), "joint_tfidf":(structured,text)}
    if name.startswith("B"):
        fine = [f"{s}_{k}" for s in ["a","b"] for k in ["wide_fraction","t_fraction","charted_n","coded_serves"]]
        conditions["all_numeric_tactics_control"] = (structured + fine, [])
    if name.startswith("C"):
        conditions["all_structured_context_control"] = (NUMERIC + CONTEXT, [])
    # Shuffle training text only, keeping real test text. A fixed negative control,
    # not a null-hypothesis p-value and never used to choose a final model.
    conditions["joint_shuffled_train_text"] = (structured,text)
    for condition, (s,t) in conditions.items():
        started = time.perf_counter()
        fit_train = train.copy()
        if condition == "joint_shuffled_train_text":
            indices = np.random.default_rng(1729).permutation(len(fit_train))
            fit_train.loc[:,t] = fit_train[t].iloc[indices].to_numpy()
        numeric_for_fit = numeric + ([c for c in s if c not in structured] if condition == "all_numeric_tactics_control" else [])
        prep = preprocessor(s, numeric_for_fit, t)
        model = make_pipeline(prep, LogisticRegression(C=1.0, solver="liblinear", max_iter=500, random_state=42))
        model.fit(fit_train, train.target)
        pred = model.predict_proba(test)[:,1]
        score = float(roc_auc_score(test.target, pred))
        pd.DataFrame({"match_id":test.match_id,"target":test.target,"probability":pred}).to_csv(OUT / f"diagnostic_{name}_{condition}_predictions.csv", index=False)
        row = {"cohort":name,"condition":condition,"learner":"LogisticRegression CPU; TF-IDF where applicable; NOT MulTaBench",
               "auroc":score,"n_train":len(train),"n_test":len(test),"cutoff":cutoff,
               "runtime_seconds":time.perf_counter()-started,"rss_mb_after_fit":psutil.Process().memory_info().rss/1e6,
               "detected_text_on_fit_pool":json.dumps({c:text_is_detected(train[c]) for c in text}),"tar_score":None}
        rows.append(row); print(name,condition,round(score,4),len(train),len(test),flush=True)
    return rows


def main():
    b = pd.read_csv(PRIVATE / "variant_B.csv")
    c = pd.read_csv(PRIVATE / "variant_C.csv")
    a = pd.read_csv(PRIVATE / "variant_A.csv")
    c_common = c[c.match_id.isin(b.match_id)].copy()
    assert set(b.match_id) == set(c_common.match_id)
    results = []
    results += run_cohort("B_same415", b, NUMERIC + CONTEXT, ["player_a_text","player_b_text"])
    results += run_cohort("C_same415", c_common, NUMERIC, ["context_text"])
    # Fixed bounded pilot, latest 1,500 chronologically eligible context rows.
    c_pilot = c.sort_values(["match_date","match_id"]).tail(1500)
    results += run_cohort("C_latest1500", c_pilot, NUMERIC, ["context_text"])
    pd.DataFrame(results).to_csv(OUT / "diagnostic_scores.csv", index=False)
    # Prepare strict shared-population A/B/C table without training A on an
    # unverified publication assumption. This is an engineering comparison only.
    common = sorted(set(a.match_id)&set(b.match_id)&set(c.match_id))
    write_json(OUT / "diagnostic_status.json", {"A":"not trained: publication-availability gate unresolved",
        "B":"CPU TF-IDF/logistic on calendar-valid, pre-target public chart snapshot",
        "C":"CPU TF-IDF/logistic; categorical verbalization control only",
        "E5":"not installed locally; no frozen E5 experiment run", "TAR":"not run; GPU submission only after review",
        "common_ABC_calendar_candidate_ids":common,"common_ABC_count":len(common),
        "B_C_same_target_population":True,"final_test_optimization":False})


if __name__ == "__main__": main()
