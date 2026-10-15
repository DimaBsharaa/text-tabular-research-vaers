"""Cheap CPU sanity screen for a built candidate (NOT the official benchmark).

1. Profile the CSV against what passing MulTaBench datasets look like (notebook 01 medians).
2. Fit TF-IDF + logistic regression for structured / text / joint on one stratified 80/20 split of a sample,
   to see whether text is strong and the two modalities look complementary before spending GPU time.
3. Leakage audit: list the text n-grams the text-only model trusts most for the positive class. Outcome words
   (hospital, died, ICU, ...) at the top would mean the label leaks through the text.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Medians of passing datasets in notebook 01 (research/data, 21 profiled passes).
WINNER_PROFILE = {"rows": 18843, "structured_features": 6, "text_features": 3, "longest_text_col_mean_chars": 121,
                  "text_minus_structured_score": 0.109}


def profile(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    text = schema["text_columns"]
    rows = {"rows": len(df), "structured_features": len(schema["structured_columns"]), "text_features": len(text),
            "longest_text_col_mean_chars": round(max(df[c].astype(str).str.len().mean() for c in text), 1),
            "positive_rate": round(float(df[schema["target"]].mean()), 4)}
    for c in text:
        rows[f"{c}: distinct share"] = round(df[c].nunique() / len(df), 3)
        rows[f"{c}: 'none reported' share"] = round(float((df[c] == "none reported").mean()), 3)
    for c in schema["structured_columns"]:
        rows[f"{c}: missing share"] = round(float(df[c].isna().mean()), 3)
    return pd.DataFrame({"candidate": rows, "passing-dataset median": WINNER_PROFILE}).fillna("")


def _pipeline(schema, columns, numeric):
    parts = []
    num = [c for c in columns if c in numeric]
    cat = [c for c in columns if c in schema["structured_columns"] and c not in numeric]
    txt = [c for c in columns if c in schema["text_columns"]]
    if num:
        parts.append(("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num))
    if cat:
        parts.append(("cat", make_pipeline(SimpleImputer(strategy="constant", fill_value="NA"),
                                           OneHotEncoder(handle_unknown="ignore", min_frequency=5)), cat))
    for c in txt:
        parts.append((f"txt_{c}", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=20000, sublinear_tf=True), c))
    return make_pipeline(ColumnTransformer(parts), LogisticRegression(C=1.0, max_iter=2000, solver="liblinear"))


def run(csv: Path, schema_path: Path, sample: int = 20000, seed: int = 0) -> dict:
    schema = json.loads(Path(schema_path).read_text())
    df = pd.read_csv(csv)
    if len(df) > sample:
        df = df.sample(sample, random_state=seed)
    y = df[schema["target"]].values
    tr, te = train_test_split(np.arange(len(df)), test_size=0.2, stratify=y, random_state=seed)
    numeric = set(schema["numeric_columns"])
    conds = {"structured_only": schema["structured_columns"], "text_only": schema["text_columns"],
             "joint": schema["structured_columns"] + schema["text_columns"]}
    scores, models = {}, {}
    for name, cols in conds.items():
        pipe = _pipeline(schema, cols, numeric)
        pipe.fit(df.iloc[tr][cols], y[tr])
        scores[name] = round(roc_auc_score(y[te], pipe.predict_proba(df.iloc[te][cols])[:, 1]), 4)
        models[name] = pipe
    scores["joint_gain_vs_best_unimodal"] = round(scores["joint"] - max(scores["structured_only"], scores["text_only"]), 4)
    scores["text_minus_structured"] = round(scores["text_only"] - scores["structured_only"], 4)
    # Leakage audit: strongest positive n-grams of the text-only model.
    ct, lr = models["text_only"].steps[0][1], models["text_only"].steps[1][1]
    names = ct.get_feature_names_out()
    top = np.argsort(lr.coef_[0])[::-1][:40]
    scores["top_positive_ngrams"] = [f"{names[i]} ({lr.coef_[0][i]:.2f})" for i in top]
    return scores


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    ap.add_argument("schema", type=Path)
    a = ap.parse_args()
    print(json.dumps(run(a.csv, a.schema), indent=2))
