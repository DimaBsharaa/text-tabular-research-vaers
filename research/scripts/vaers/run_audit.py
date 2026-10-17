"""Independent audit of a finished official-code screen (answers the reviewer's open questions).

Inputs: the exact candidate CSV + schema.json used by the run, the run's results folder (benchmark-results/<plan_id>,
which holds one sub-folder per run with config.json and splits.json), and optionally the raw yearly VAERS ZIPs.

Sections (each prints PASS/FLAG lines and numbers; nothing here changes the dataset):
  1. Mechanics    - same CSV hash in every run; identical outer test IDs for all conditions of a fold; train/test
                    disjoint; TAR encoder rows inside learner_fit only; Joint Frozen and Joint TAR use the same columns;
                    detected text columns equal the schema's.
  2. Near-duplicate events across train/test - pairs that share age, sex, state, vaccine type, onset days and report
                    year and whose excerpts are near-identical (token Jaccard >= 0.6), counted per fold where one is in
                    outer_train and the other in outer_test, with label agreement.
  3. Duplicate sensitivity - a cheap proxy model (TF-IDF + one-hot + logistic regression) trained on each fold's
                    outer_train, AUROC on outer_test with and without the near-duplicate test rows.
  4. Strata      - proxy text-only AUROC within report year and within COVID vs non-COVID vaccines: does the text
                    signal survive inside groups, or is it only a between-group (reporting) effect?
  5. Style       - AUROC of writing-style features alone (length, first person, capitals, digits, 'pt'/'patient'):
                    how much of the text signal is reporter style rather than content?
  6. Original vs retained text - random reports of both labels: the original narrative opening next to the retained
                    excerpt and background (needs the raw ZIPs).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TOKEN = re.compile(r"[a-z0-9]+")


def load_runs(results_dir: Path) -> list[dict]:
    runs = []
    for cfg in sorted(Path(results_dir).glob("*/config.json")):
        splits = cfg.parent / "splits.json"
        if splits.exists():
            runs.append({"config": json.loads(cfg.read_text()), "splits": json.loads(splits.read_text())})
    if not runs:
        raise FileNotFoundError(f"No */config.json + splits.json under {results_dir}")
    return runs


def mechanics(runs: list[dict], csv_sha: str, schema: dict) -> list[str]:
    out = []
    shas = {r["config"]["csv_sha256"] for r in runs}
    out.append(f"{'PASS' if shas == {csv_sha} else 'FLAG'} every run used this CSV (sha {csv_sha[:12]}…): {len(runs)} runs")
    text_cols = set(schema["text_columns"])
    for fold in sorted({r["config"]["fold"] for r in runs}):
        R = [r for r in runs if r["config"]["fold"] == fold]
        tests = {frozenset(r["splits"]["outer_test"]) for r in R}
        trains = {frozenset(r["splits"]["outer_train"]) for r in R}
        test, train = next(iter(tests)), next(iter(trains))
        ok_same = len(tests) == 1 and len(trains) == 1
        out.append(f"{'PASS' if ok_same else 'FLAG'} fold {fold}: {len(R)} runs share one outer_test ({len(test)}) "
                   f"and one outer_train ({len(train)})")
        out.append(f"{'PASS' if not test & train else 'FLAG'} fold {fold}: outer_train ∩ outer_test = {len(test & train)}")
        for r in R:
            s = {k: set(v) for k, v in r["splits"].items() if isinstance(v, list)}
            if s["learner_fit"] & s["learner_validation"] or (s["learner_fit"] | s["learner_validation"]) != s["outer_train"]:
                out.append(f"FLAG fold {fold} {r['config']['model_key']}/{r['config']['condition']}: learner split")
            if "encoder_train" in s:
                enc = s["encoder_train"] | s["encoder_validation"]
                ok = enc <= s["learner_fit"] and not enc & s["outer_test"] and not enc & s["learner_validation"]
                out.append(f"{'PASS' if ok else 'FLAG'} fold {fold} {r['config']['model_key']} TAR encoder rows "
                           f"({len(enc)}) inside learner_fit, none in test or learner validation")
            detected = set(r["splits"].get("detected_text_columns", []))
            cond = r["config"]["condition"]
            expect = set() if cond == "structured_only" else text_cols
            if detected != expect:
                out.append(f"FLAG fold {fold} {cond}: detected text {sorted(detected)} != expected {sorted(expect)}")
    cols = defaultdict(set)
    for r in runs:
        cols[r["config"]["condition"]].add(tuple(r["config"]["columns"]))
    same = cols.get("joint_frozen") == cols.get("joint_tar")
    out.append(f"{'PASS' if same else 'FLAG'} Joint Frozen and Joint TAR use identical predictor columns")
    leaked = {c for cs in cols.values() for t in cs for c in t} & {schema["target"], schema.get("id_column", "")}
    out.append(f"{'PASS' if not leaked else 'FLAG'} target / id column not among predictors {sorted(leaked)}")
    return out


def _tokens(text: str) -> frozenset:
    return frozenset(TOKEN.findall(str(text).lower()))


def near_duplicate_groups(df: pd.DataFrame, text_col: str = "narrative_onset", threshold: float = 0.6) -> list[set]:
    """Clusters of rows that agree on the event key and have near-identical excerpts (union-find over pairs)."""
    key = ["age_years", "sex", "state", "vax_type", "onset_days", "report_year"]
    ids = df.report_id.astype(str).to_numpy()
    toks = df[text_col].map(_tokens).to_numpy()
    parent = {i: i for i in range(len(df))}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for _, idx in df.reset_index(drop=True).groupby(key, dropna=False).groups.items():
        idx = list(idx)
        if len(idx) < 2 or len(idx) > 400:
            continue
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                ta, tb = toks[idx[a]], toks[idx[b]]
                if ta and tb and len(ta & tb) / len(ta | tb) >= threshold:
                    parent[find(idx[a])] = find(idx[b])
    groups = defaultdict(set)
    for i in range(len(df)):
        groups[find(i)].add(ids[i])
    return [g for g in groups.values() if len(g) > 1]


def cross_split_duplicates(groups: list[set], runs: list[dict], labels: dict) -> list[str]:
    out = [f"near-duplicate clusters in the whole CSV: {len(groups)} "
           f"({sum(len(g) for g in groups)} rows; label-consistent clusters "
           f"{sum(len({labels[i] for i in g}) == 1 for g in groups)})"]
    for fold in sorted({r["config"]["fold"] for r in runs}):
        s = next(r["splits"] for r in runs if r["config"]["fold"] == fold)
        train, test = set(s["outer_train"]), set(s["outer_test"])
        straddle = [g for g in groups if g & train and g & test]
        test_rows = {i for g in straddle for i in g & test}
        agree = [len({labels[i] for i in g}) == 1 for g in straddle]
        out.append(f"fold {fold}: clusters with members in both outer_train and outer_test = {len(straddle)}; "
                   f"test rows involved = {len(test_rows)} of {len(test)} "
                   f"({len(test_rows) / len(test):.2%}); same label in {np.mean(agree) if agree else float('nan'):.2f} of them")
    return out


def _proxy(schema: dict, columns: list[str]):
    num = [c for c in columns if c in schema["numeric_columns"]]
    cat = [c for c in columns if c in schema["structured_columns"] and c not in num]
    txt = [c for c in columns if c in schema["text_columns"]]
    parts = []
    if num:
        parts.append(("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num))
    if cat:
        parts.append(("cat", make_pipeline(SimpleImputer(strategy="constant", fill_value="NA"),
                                           OneHotEncoder(handle_unknown="ignore", min_frequency=5)), cat))
    for c in txt:
        parts.append((f"t_{c}", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=20000, sublinear_tf=True), c))
    return make_pipeline(ColumnTransformer(parts), LogisticRegression(max_iter=2000, solver="liblinear"))


def _fold_frames(df, runs):
    by_id = df.set_index(df.report_id.astype(str))
    for fold in sorted({r["config"]["fold"] for r in runs}):
        s = next(r["splits"] for r in runs if r["config"]["fold"] == fold)
        yield fold, by_id.loc[s["outer_train"]], by_id.loc[s["outer_test"]]


def duplicate_sensitivity(df, schema, runs, groups) -> list[str]:
    dup_ids = {i for g in groups for i in g}
    cols = schema["structured_columns"] + schema["text_columns"]
    out = []
    for fold, tr, te in _fold_frames(df, runs):
        model = _proxy(schema, cols).fit(tr[cols], tr[schema["target"]])
        p = pd.Series(model.predict_proba(te[cols])[:, 1], index=te.index)
        keep = ~te.index.isin(list(dup_ids))
        a_all = roc_auc_score(te[schema["target"]], p)
        if te[schema["target"]][keep].nunique() < 2:
            out.append(f"fold {fold}: proxy joint AUROC all test {a_all:.4f}; too few rows left without duplicates")
            continue
        a_clean = roc_auc_score(te[schema["target"]][keep], p[keep])
        out.append(f"fold {fold}: proxy joint AUROC all test {a_all:.4f}; without near-duplicate rows {a_clean:.4f} "
                   f"(Δ {a_clean - a_all:+.4f}, removed {int((~keep).sum())})")
    return out


def strata(df, schema, runs) -> list[str]:
    y = schema["target"]
    preds = []
    for fold, tr, te in _fold_frames(df, runs):
        m = _proxy(schema, schema["text_columns"]).fit(tr[schema["text_columns"]], tr[y])
        preds.append(te.assign(_p=m.predict_proba(te[schema["text_columns"]])[:, 1], _fold=fold))
    P = pd.concat(preds)
    P["vaccine_group"] = np.where(P.vax_type.astype(str).str.startswith("COVID"), "COVID", "non-COVID")
    out = [f"text-only proxy AUROC, all test rows of all folds: {roc_auc_score(P[y], P._p):.4f}"]
    for col in ["report_year", "vaccine_group", "admin_by"]:
        for g, sub in P.groupby(col):
            if sub[y].nunique() == 2 and sub[y].sum() >= 20:
                out.append(f"  within {col}={g}: AUROC {roc_auc_score(sub[y], sub._p):.4f} "
                           f"(n={len(sub)}, serious={int(sub[y].sum())})")
    return out


STYLE = {
    "chars": lambda s: s.str.len(),
    "sentences": lambda s: s.str.count(r"[.!?]") + 1,
    "first_person": lambda s: s.str.count(r"(?i)\b(?:i|my|me|i'm|i've)\b"),
    "third_party": lambda s: s.str.count(r"(?i)\b(?:pt|patient|client|she|he|her|his)\b"),
    "caps_share": lambda s: s.map(lambda t: sum(c.isupper() for c in t) / max(1, sum(c.isalpha() for c in t))),
    "digit_share": lambda s: s.map(lambda t: sum(c.isdigit() for c in t) / max(1, len(t))),
    "parent": lambda s: s.str.count(r"(?i)\b(?:my (?:son|daughter|child|baby|wife|husband|mother|father))\b"),
}


def style(df, schema, runs, text_col="narrative_onset") -> list[str]:
    y = schema["target"]
    out = []
    for fold, tr, te in _fold_frames(df, runs):
        f = lambda d: pd.DataFrame({k: fn(d[text_col].astype(str)) for k, fn in STYLE.items()})
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(f(tr), tr[y])
        a_style = roc_auc_score(te[y], m.predict_proba(f(te))[:, 1])
        mt = _proxy(schema, [text_col]).fit(tr[[text_col]], tr[y])
        a_text = roc_auc_score(te[y], mt.predict_proba(te[[text_col]])[:, 1])
        out.append(f"fold {fold}: style-only AUROC {a_style:.4f} vs full {text_col} TF-IDF {a_text:.4f}")
    return out


def originals(df, schema, raw_paths, n=12, seed=0) -> list[str]:
    from research.scripts.vaers import build_vaers as bv
    data = bv.load_year_files([Path(p) for p in raw_paths])["DATA"].drop_duplicates("VAERS_ID")
    data["VAERS_ID"] = data.VAERS_ID.astype(str)
    raw = data.set_index("VAERS_ID").SYMPTOM_TEXT
    out = []
    for label in (1, 0):
        sample = df[df[schema["target"]] == label].sample(n, random_state=seed)
        out.append(f"--- label {label} ({'serious' if label else 'not serious'}): original opening | retained excerpt")
        for _, row in sample.iterrows():
            orig = str(raw.get(str(row.report_id), ""))[:400].replace("\n", " ")
            out.append(f"[{row.report_id}] ORIGINAL: {orig}")
            out.append(f"           RETAINED: {row.narrative_onset}")
            bg = [f"{c}={row[c]}" for c in schema["text_columns"] if c != "narrative_onset"]
            out.append(f"           BACKGROUND: {'; '.join(bg)[:300]}")
    return out


def run(csv: Path, schema_path: Path, results_dir: Path, raw_paths=(), out_path: Path | None = None) -> str:
    schema = json.loads(Path(schema_path).read_text())
    csv_sha = hashlib.sha256(Path(csv).read_bytes()).hexdigest()
    df = pd.read_csv(csv)
    df["report_id"] = df.report_id.astype(str)
    runs = load_runs(results_dir)
    labels = dict(zip(df.report_id, df[schema["target"]]))
    sections = [("1. Mechanics", mechanics(runs, csv_sha, schema))]
    groups = near_duplicate_groups(df)
    sections.append(("2. Near-duplicate events across train/test", cross_split_duplicates(groups, runs, labels)))
    sections.append(("3. Duplicate sensitivity (proxy model)", duplicate_sensitivity(df, schema, runs, groups)))
    sections.append(("4. Text signal within strata (proxy, text only)", strata(df, schema, runs)))
    sections.append(("5. Reporter style vs content", style(df, schema, runs)))
    if raw_paths:
        sections.append(("6. Original vs retained text (random reports)", originals(df, schema, raw_paths)))
    text = "\n".join(f"\n## {title}\n" + "\n".join(lines) for title, lines in sections)
    flags = text.count("FLAG")
    text = f"# Audit of {Path(results_dir).name} on {Path(csv).name} (csv sha {csv_sha[:12]}…)\nFLAG lines: {flags}\n" + text
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(text + "\n")
    return text


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    ap.add_argument("schema", type=Path)
    ap.add_argument("results_dir", type=Path)
    ap.add_argument("--raw", nargs="*", default=[])
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    print(run(a.csv, a.schema, a.results_dir, a.raw, a.out))
