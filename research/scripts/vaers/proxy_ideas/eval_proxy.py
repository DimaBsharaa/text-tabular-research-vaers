"""CPU proxy of the four MulTaBench conditions (development diagnostic only; not E5, not official).

frozen-like text  = per-column TF-IDF -> TruncatedSVD(30), fit on training rows (label-free compression,
                    like frozen E5 -> PCA-30)
TAR-like text     = the same 30 dims + a supervised text score per column (logistic regression on TF-IDF;
                    out-of-fold on training rows, fit-on-all-train for test rows), mimicking an encoder whose
                    embedding has been pulled toward the label
Splits follow multabench.preprocessing.splits (seed 42 + fold, one example per class forced into test,
stratified 10% test capped at 2,000) with training capped at 10,000 (stratified).
"""
import sys
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

STRUCT_NUM = ["age_years", "onset_days", "n_vaccines", "report_year"]
STRUCT_CAT = ["sex", "state", "vax_type", "vax_manu", "vax_dose_series", "vax_route", "vax_site", "admin_by"]
SEED = 42


def split(y, fold):
    rs = SEED + fold
    rng = np.random.RandomState(rs)
    forced = [rng.choice(y.index[y == v].tolist()) for v in sorted(set(y))]
    rest = y.drop(index=forced)
    test_size = min(int(len(y) * 0.1), 2000)
    tr, te = train_test_split(rest.index, test_size=test_size, random_state=rs, stratify=rest)
    te = list(te) + forced
    if len(tr) > 10000:
        tr, _ = train_test_split(tr, train_size=10000, random_state=rs, stratify=y.loc[tr])
    return pd.Index(tr), pd.Index(te)


def struct(df):
    x = pd.DataFrame(index=df.index)
    for c in STRUCT_NUM:
        x[c] = pd.to_numeric(df[c], errors="coerce")
    for c in STRUCT_CAT:
        x[c] = df[c].astype("category")
    return x


def tfidf(tr_text, te_text):
    v = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=60000, sublinear_tf=True)
    return v.fit_transform(tr_text), v.transform(te_text)


def text_feats(df, tr, te, col, y_tr):
    Xtr, Xte = tfidf(df.loc[tr, col], df.loc[te, col])
    k = min(30, Xtr.shape[1] - 1)
    svd = TruncatedSVD(k, random_state=SEED).fit(Xtr)
    f_tr = pd.DataFrame(svd.transform(Xtr), index=tr).add_prefix(f"{col}_svd")
    f_te = pd.DataFrame(svd.transform(Xte), index=te).add_prefix(f"{col}_svd")
    oof = np.zeros(len(tr))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xtr, y_tr):
        m = LogisticRegression(C=2.0, max_iter=2000).fit(Xtr[a], y_tr[a])
        oof[b] = m.decision_function(Xtr[b])
    m = LogisticRegression(C=2.0, max_iter=2000).fit(Xtr, y_tr)
    s_tr = pd.Series(oof, index=tr, name=f"{col}_score")
    s_te = pd.Series(m.decision_function(Xte), index=te, name=f"{col}_score")
    return f_tr, f_te, s_tr, s_te


def gbm(xtr, ytr, xte, yte):
    m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=20,
                           colsample_bytree=0.8, subsample=0.8, subsample_freq=1, random_state=SEED, verbose=-1)
    m.fit(xtr, ytr)
    return roc_auc_score(yte, m.predict_proba(xte)[:, 1])


def evaluate(path, folds=(0, 1, 2, 3, 4)):
    df = pd.read_csv(path, keep_default_na=False, dtype=str).reset_index(drop=True)
    text_cols = [c for c in df.columns if c not in STRUCT_NUM + STRUCT_CAT + ["serious", "report_id"]]
    y = df.serious.astype(int)
    xs = struct(df)
    rows = []
    for f in folds:
        tr, te = split(y, f)
        ytr, yte = y.loc[tr].to_numpy(), y.loc[te].to_numpy()
        fz_tr, fz_te, sc_tr, sc_te = [], [], [], []
        for c in text_cols:
            a, b, s1, s2 = text_feats(df, tr, te, c, ytr)
            fz_tr.append(a); fz_te.append(b); sc_tr.append(s1); sc_te.append(s2)
        FZtr, FZte = pd.concat(fz_tr, axis=1), pd.concat(fz_te, axis=1)
        SCtr, SCte = pd.concat(sc_tr, axis=1), pd.concat(sc_te, axis=1)
        r = {"fold": f, "n": len(df), "pos": round(y.mean(), 3)}
        r["S"] = gbm(xs.loc[tr], ytr, xs.loc[te], yte)
        r["T"] = gbm(FZtr, ytr, FZte, yte)
        r["T_tar"] = gbm(pd.concat([FZtr, SCtr], axis=1), ytr, pd.concat([FZte, SCte], axis=1), yte)
        r["JF"] = gbm(pd.concat([xs.loc[tr], FZtr], axis=1), ytr, pd.concat([xs.loc[te], FZte], axis=1), yte)
        r["JT"] = gbm(pd.concat([xs.loc[tr], FZtr, SCtr], axis=1), ytr,
                      pd.concat([xs.loc[te], FZte, SCte], axis=1), yte)
        r["JS"] = r["JF"] - max(r["S"], r["T"])
        r["TG"] = r["JT"] - r["JF"]
        rows.append(r)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    out = {}
    for p in sys.argv[1:]:
        res = evaluate(p)
        name = p.split("/")[-1].replace(".csv", "")
        res.insert(0, "variant", name)
        res.to_csv(f"proxy_{name}.csv", index=False)
        m = res.mean(numeric_only=True)
        print(f"{name:24s} n={int(m.n):6d} pos={m.pos:.3f}  S={m.S:.4f} T={m['T']:.4f} T_tar={m.T_tar:.4f} "
              f"JF={m.JF:.4f} JT={m.JT:.4f}  JS={m.JS:+.4f} TG={m.TG:+.4f}  TG per fold={np.round(res.TG.values, 4)}",
              flush=True)
