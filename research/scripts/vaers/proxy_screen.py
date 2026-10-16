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


def column_audit(csv: Path, schema_path: Path, seed: int = 0) -> pd.DataFrame:
    """Leak audit: AUROC of EACH column alone (cross-fitted), and of its 'is missing / none reported' flag alone.
    A single column near 0.9, or a strong missingness flag, points to a reporting artifact worth explaining."""
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    schema = json.loads(Path(schema_path).read_text())
    df = pd.read_csv(csv)
    y = df[schema["target"]].values
    folds = StratifiedKFold(5, shuffle=True, random_state=seed)

    def auc(score):
        a = roc_auc_score(y, score)
        return round(max(a, 1 - a), 3)

    rows = []
    for c in schema["structured_columns"]:
        x = df[c]
        if c in schema["numeric_columns"]:
            # Linear ranking AND an out-of-fold rate per value bin (catches non-monotonic effects, e.g. one year
            # far more serious than the years around it); the larger is reported.
            linear = auc(x.fillna(x.median()))
            bins = x.rank(method="dense") if x.nunique() <= 20 else pd.qcut(x, 10, labels=False, duplicates="drop")
            bins = bins.fillna(-1).astype(str)
            oof = np.zeros(len(df))
            for tr, te in folds.split(df, y):
                rate = pd.Series(y[tr]).groupby(bins.iloc[tr].values).mean()
                oof[te] = bins.iloc[te].map(rate).fillna(y[tr].mean()).values
            alone = max(linear, auc(oof))
            flag = x.isna()
        else:  # out-of-fold category rate: no row sees its own label
            oof = np.zeros(len(df))
            for tr, te in folds.split(df, y):
                rate = pd.Series(y[tr]).groupby(x.iloc[tr].astype(str).values).mean()
                oof[te] = x.iloc[te].astype(str).map(rate).fillna(y[tr].mean()).values
            alone = auc(oof)
            flag = x.astype(str).str.upper().isin(["UNKNOWN", "UNK", "U", "NAN"])
        rows.append({"column": c, "kind": "table", "auroc_alone": alone,
                     "auroc_missing_flag": auc(flag) if flag.any() and not flag.all() else None})
    for c in schema["text_columns"]:
        model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=20000, sublinear_tf=True),
                              LogisticRegression(max_iter=2000, solver="liblinear"))
        prob = cross_val_predict(model, df[c].astype(str), y, cv=folds, method="predict_proba")[:, 1]
        flag = df[c].eq("none reported")
        rows.append({"column": c, "kind": "text", "auroc_alone": auc(prob),
                     "auroc_missing_flag": auc(flag) if flag.any() and not flag.all() else None})
    return pd.DataFrame(rows).sort_values("auroc_alone", ascending=False).reset_index(drop=True)
