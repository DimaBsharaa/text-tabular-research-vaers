"""Local data diagnostics for a finished official-code screen.

Inputs: the exact candidate CSV + schema.json used by the run, the run's results folder (benchmark-results/<plan_id>,
which holds one sub-folder per run with config.json and splits.json), and optionally the raw yearly VAERS ZIPs.

Sections (each prints PASS/FLAG lines and numbers; nothing here changes the dataset):
  1. Mechanics    - same CSV hash in every run; identical outer test IDs for all conditions of a fold; train/test
                    disjoint; TAR encoder rows inside learner_fit only; Joint Frozen and Joint TAR use the same columns;
                    detected text columns equal the schema's.
  2. Candidate duplicate clusters across train/test - pairs that share age, sex, state, vaccine type, onset days and report
                    year and whose excerpts are near-identical (token Jaccard >= 0.6), counted per fold where one is in
                    outer_train and the other in outer_test, with label agreement.
  3. Duplicate sensitivity - a cheap proxy model (TF-IDF + one-hot + logistic regression) trained on each fold's
                    outer_train, AUROC on outer_test with and without the near-duplicate test rows.
  4. Strata      - per-fold proxy text-only AUROC within year, vaccine, and year x vaccine: does the text
                    signal survive inside groups, or is it only a between-group (reporting) effect?
  5. Style       - AUROC of writing-style features alone (length, first person, capitals, digits, 'pt'/'patient'):
                    how much of the text signal is reporter style rather than content?
  6. Original vs retained text - matched source reports, source hashes and full local JSON examples (raw ZIPs needed).

PASS concerns a named implemented check only. Candidate clusters are not confirmed
clinical events. All fitted models here are CPU proxies, not E5/TAR reproductions.
Recorded splits are adapter reconstructions, not independent runtime tensor traces.
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
CONDITIONS = {"structured_only", "text_only", "joint_frozen", "joint_tar"}
AUDIT_VERSION = 'update15-corrected-2'


class AuditInputError(ValueError):
    """Input identity or recorded mechanics failed; do not fit diagnostic models."""


def load_runs(results_dir: Path) -> list[dict]:
    runs = []
    for cfg in sorted(Path(results_dir).glob("*/config.json")):
        splits = cfg.parent / "splits.json"
        if not splits.exists():
            raise FileNotFoundError(f'Missing split evidence: {splits}')
        if splits.exists():
            result = cfg.parent / "result.json"
            record = json.loads(result.read_text()) if result.exists() else {}
            runs.append({"config": json.loads(cfg.read_text()), "splits": json.loads(splits.read_text()),
                         "result_status": record.get("status", "missing"), "run_folder": str(cfg.parent)})
    if not runs:
        raise FileNotFoundError(f"No */config.json + splits.json under {results_dir}")
    return runs


def mechanics(runs: list[dict], csv_sha: str, schema: dict, schema_sha: str | None = None) -> list[str]:
    out = []

    def check(ok, message):
        out.append(f"{'PASS' if ok else 'FLAG'} {message}")

    shas = {r["config"]["csv_sha256"] for r in runs}
    check(bool(runs) and shas == {csv_sha}, f"every run used this CSV (sha {csv_sha[:12]}…): {len(runs)} runs")
    check(schema.get("csv_sha256") == csv_sha, "schema CSV fingerprint matches actual CSV bytes")
    if schema_sha is not None:
        check(all(r['config'].get('schema_sha256') == schema_sha for r in runs), "every run used this exact schema file")
    else:
        out.append("NOT_VERIFIED schema-file fingerprint was not supplied")
    check(all(r.get('result_status') == 'ok' for r in runs), "every selected run has a successful result.json")
    keys = [(r['config']['model_key'], r['config']['fold'], r['config']['condition']) for r in runs]
    check(len(set(keys)) == len(keys), "no duplicate learner/fold/condition runs")
    for model, fold in sorted({k[:2] for k in keys}):
        conditions = {k[2] for k in keys if k[:2] == (model, fold)}
        check(conditions == CONDITIONS, f"{model}/fold {fold}: all four conditions present")
    text_cols = set(schema["text_columns"])
    for fold in sorted({r["config"]["fold"] for r in runs}):
        R = [r for r in runs if r["config"]["fold"] == fold]
        tests = {frozenset(map(str, r["splits"]["outer_test"])) for r in R}
        trains = {frozenset(map(str, r["splits"]["outer_train"])) for r in R}
        ok_same = len(tests) == 1 and len(trains) == 1
        check(ok_same, f"fold {fold}: {len(R)} runs share identical outer training/test sets")
        for r in R:
            cond = r["config"]["condition"]
            name = f"fold {fold} {r['config']['model_key']}/{cond}"
            required = ['outer_train', 'outer_test', 'learner_fit', 'learner_validation']
            if cond == 'joint_tar':
                required += ['encoder_train', 'encoder_validation']
            check(all(k in r['splits'] for k in required), name + ": required partitions recorded")
            s = {k: set(map(str, r['splits'].get(k, []))) for k in required}
            check(all(len(s[k]) == len(r['splits'].get(k, [])) for k in required), name + ": partition IDs are unique")
            check(bool(s['outer_train']) and bool(s['outer_test']) and not s['outer_train'] & s['outer_test'], name + ": outer train/test disjoint and nonempty")
            check(not s['learner_fit'] & s['learner_validation'] and s['learner_fit'] | s['learner_validation'] == s['outer_train'], name + ": learner partitions exactly divide training")
            if cond == 'joint_tar':
                enc = s["encoder_train"] | s["encoder_validation"]
                ok = (bool(s['encoder_train']) and bool(s['encoder_validation']) and
                      not s['encoder_train'] & s['encoder_validation'] and enc == s['learner_fit'] and
                      not enc & (s['outer_test'] | s['learner_validation']))
                check(ok, name + ": encoder partitions disjoint, exactly cover learner fitting, exclude test/learner validation")
            detected = set(r["splits"].get("detected_text_columns", []))
            expect = set() if cond == "structured_only" else text_cols
            check(detected == expect, name + ": detected text agrees with schema")
            expected_columns = (schema['structured_columns'] if cond == 'structured_only' else
                                schema['text_columns'] if cond == 'text_only' else
                                schema['structured_columns'] + schema['text_columns'])
            check(r['config']['columns'] == expected_columns, name + ": exact predictor whitelist")
    cols = defaultdict(set)
    for r in runs:
        cols[r["config"]["condition"]].add(tuple(r["config"]["columns"]))
    same = bool(cols.get('joint_frozen')) and cols.get("joint_frozen") == cols.get("joint_tar")
    out.append(f"{'PASS' if same else 'FLAG'} Joint Frozen and Joint TAR use identical predictor columns")
    leaked = {c for cs in cols.values() for t in cs for c in t} & {schema["target"], schema.get("id_column", "")}
    out.append(f"{'PASS' if not leaked else 'FLAG'} target / id column not among predictors {sorted(leaked)}")
    return out


def _tokens(text: str) -> frozenset:
    return frozenset(TOKEN.findall(str(text).lower()))


def near_duplicate_groups(df: pd.DataFrame, text_col: str = "narrative_onset", threshold: float = 0.6,
                          coverage: dict | None = None) -> list[set]:
    """Candidate clusters, not confirmed events. Large blocks retain exact-token matches.

    Approximate comparisons in blocks above 400 are explicitly unverified. All
    blocking metadata must match; changed metadata can hide actual secondary reports.
    """
    key = ["age_years", "sex", "state", "vax_type", "onset_days", "report_year"]
    ids = df.report_id.astype(str).to_numpy()
    toks = df[text_col].map(_tokens).to_numpy()
    parent = {i: i for i in range(len(df))}
    stats = {"rows": len(df), "blocks": 0, "large_blocks_exact_only": 0,
             "rows_in_large_blocks": 0, "approximate_pair_comparisons": 0}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for _, idx in df.reset_index(drop=True).groupby(key, dropna=False).groups.items():
        idx = list(idx)
        stats['blocks'] += 1
        if len(idx) < 2:
            continue
        # Exact token-set matches are inexpensive, including in large blocks.
        seen = {}
        for j in idx:
            if not toks[j]:
                continue
            if toks[j] in seen:
                parent[find(j)] = find(seen[toks[j]])
            else:
                seen[toks[j]] = j
        if len(idx) > 400:
            stats['large_blocks_exact_only'] += 1
            stats['rows_in_large_blocks'] += len(idx)
            continue
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                stats['approximate_pair_comparisons'] += 1
                ta, tb = toks[idx[a]], toks[idx[b]]
                if ta and tb and len(ta & tb) / len(ta | tb) >= threshold:
                    parent[find(idx[a])] = find(idx[b])
    groups = defaultdict(set)
    for i in range(len(df)):
        groups[find(i)].add(ids[i])
    if coverage is not None:
        coverage.update(stats)
    return [g for g in groups.values() if len(g) > 1]


def cross_split_test_ids(groups, train_ids, test_ids) -> set:
    train, test = set(map(str, train_ids)), set(map(str, test_ids))
    return {i for g in groups if g & train and g & test for i in g & test}


def cross_split_duplicates(groups: list[set], runs: list[dict], labels: dict) -> list[str]:
    out = [f"Candidate text/metadata clusters in the whole CSV: {len(groups)} "
           f"({sum(len(g) for g in groups)} rows; label-consistent clusters "
           f"{sum(len({labels[i] for i in g}) == 1 for g in groups)})",
           "NOT_VERIFIED these are shared coarse metadata plus similar text, not confirmed clinical-event identities"]
    for fold in sorted({r["config"]["fold"] for r in runs}):
        s = next(r["splits"] for r in runs if r["config"]["fold"] == fold)
        train, test = set(map(str, s["outer_train"])), set(map(str, s["outer_test"]))
        straddle = [g for g in groups if g & train and g & test]
        test_rows = cross_split_test_ids(groups, train, test)
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
        yield fold, by_id.loc[list(map(str, s["outer_train"]))], by_id.loc[list(map(str, s["outer_test"]))]


def duplicate_sensitivity(df, schema, runs, groups) -> list[str]:
    cols = schema["structured_columns"] + schema["text_columns"]
    out = ["CPU proxy only: test subsets differ; this is not official E5/TAR performance after deduplication."]
    for fold, tr, te in _fold_frames(df, runs):
        model = _proxy(schema, cols).fit(tr[cols], tr[schema["target"]])
        p = pd.Series(model.predict_proba(te[cols])[:, 1], index=te.index)
        dup_ids = cross_split_test_ids(groups, tr.index, te.index)
        keep = ~te.index.isin(dup_ids)
        a_all = roc_auc_score(te[schema["target"]], p)
        if te[schema["target"]][keep].nunique() < 2:
            out.append(f"fold {fold}: proxy joint AUROC all test {a_all:.4f}; too few rows left without duplicates")
            continue
        a_clean = roc_auc_score(te[schema["target"]][keep], p[keep])
        removed_y = te.loc[~keep, schema['target']]
        out.append(f"fold {fold}: proxy joint AUROC all test {a_all:.4f}; excluding cross-split candidate test members {a_clean:.4f} "
                   f"(Δ {a_clean - a_all:+.4f}, removed {int((~keep).sum())}: "
                   f"serious={int(removed_y.sum())}, non-serious={int((removed_y == 0).sum())})")
    return out


def _stratum_scores(frame, target, columns, fold, min_class=20) -> list[str]:
    out = []
    for group, sub in frame.groupby(columns, dropna=False):
        pos = int((sub[target] == 1).sum())
        neg = int((sub[target] == 0).sum())
        label = f"fold {fold} within {' x '.join(columns)}={group}: n={len(sub)}, serious={pos}, non-serious={neg}"
        if min(pos, neg) < min_class:
            out.append(f"NOT_ESTIMATED {label}; need at least {min_class} of each class")
        else:
            out.append(f"{label}; AUROC {roc_auc_score(sub[target], sub['_p']):.4f}")
    return out


def strata(df, schema, runs) -> list[str]:
    y = schema["target"]
    out = ["Per-fold CPU text proxy; predictions from overlapping folds are not pooled.",
           "NOT_VERIFIED a surviving conditional association does not prove absence of reporting or outcome proxies"]
    for fold, tr, te in _fold_frames(df, runs):
        m = _proxy(schema, schema["text_columns"]).fit(tr[schema["text_columns"]], tr[y])
        P = te.assign(_p=m.predict_proba(te[schema["text_columns"]])[:, 1])
        P["vaccine_group"] = np.where(P.vax_type.astype(str).str.startswith("COVID"), "COVID", "non-COVID")
        out.append(f"fold {fold}: text-only proxy AUROC {roc_auc_score(P[y], P['_p']):.4f}; n={len(P)}, serious={int(P[y].sum())}")
        for cols in [['report_year'], ['vax_type'], ['report_year', 'vax_type'], ['vaccine_group'], ['admin_by']]:
            out.extend(_stratum_scores(P, y, cols, fold))
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
    out = ["Style features are imperfect proxies and can encode patient population or event content.",
           "Do not subtract standalone AUROCs to assign a percentage of signal to style."]
    for fold, tr, te in _fold_frames(df, runs):
        f = lambda d: pd.DataFrame({k: fn(d[text_col].astype(str)) for k, fn in STYLE.items()})
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(f(tr), tr[y])
        a_style = roc_auc_score(te[y], m.predict_proba(f(te))[:, 1])
        mt = _proxy(schema, [text_col]).fit(tr[[text_col]], tr[y])
        a_text = roc_auc_score(te[y], mt.predict_proba(te[[text_col]])[:, 1])
        out.append(f"fold {fold}: style-only AUROC {a_style:.4f} vs full {text_col} TF-IDF {a_text:.4f}")
        names = [f'__audit_style_{k}' for k in STYLE]
        train, test = tr.copy(), te.copy()
        train[names], test[names] = f(tr).to_numpy(), f(te).to_numpy()
        extended = {**schema, 'numeric_columns': schema['numeric_columns'] + names,
                    'structured_columns': schema['structured_columns'] + names}
        base_cols = extended['structured_columns']
        base = _proxy(extended, base_cols).fit(train[base_cols], train[y])
        full_cols = base_cols + [text_col]
        full = _proxy(extended, full_cols).fit(train[full_cols], train[y])
        a_base = roc_auc_score(test[y], base.predict_proba(test[base_cols])[:, 1])
        a_full = roc_auc_score(test[y], full.predict_proba(test[full_cols])[:, 1])
        out.append(f"fold {fold}: structured+style {a_base:.4f}; structured+style+excerpt {a_full:.4f}; "
                   f"incremental proxy gain {a_full-a_base:+.4f} (association, not causal attribution)")
    return out


def originals(df, schema, raw_paths, n=12, seed=0, samples_path=None) -> list[str]:
    from research.scripts.vaers import build_vaers as bv
    tables = bv.load_year_files([Path(p) for p in raw_paths])
    data = tables['DATA'].drop_duplicates('VAERS_ID')
    out, examples = [], []
    identity = lambda rows: {r['file'].split(':')[-1]: r['sha256'] for r in rows}
    if schema.get('sources'):
        same = identity(schema['sources']) == identity(tables['sources'])
        out.append(f"{'PASS' if same else 'FLAG'} raw source member hashes match construction schema")
        if not same:
            return out + ['NOT_VERIFIED original-text comparisons skipped because raw source identity differs']
    else:
        out.append('NOT_VERIFIED no raw source fingerprints in schema; ID matches alone do not verify source version')
    # VAERS 2019 stores IDs such as 0811848, while pandas infers 811848
    # from the candidate CSV. Match their numeric identity without changing files.
    canonical_id = lambda value: (str(value).strip().lstrip('0') or '0') if str(value).strip().isdigit() else str(value).strip()
    data['VAERS_ID'] = data.VAERS_ID.map(canonical_id)
    if data.VAERS_ID.duplicated().any():
        return out + ['FLAG raw source IDs collide after leading-zero normalization; original comparisons stopped']
    raw = data.set_index('VAERS_ID')
    out.append('PASS raw source IDs unique after leading-zero normalization for CSV matching')
    for label in (1, 0):
        candidates = df[df[schema['target']] == label]
        sample = candidates.sample(min(n, len(candidates)), random_state=seed)
        out.append(f"--- label {label} ({'serious' if label else 'not serious'}): original opening | retained excerpt")
        for _, row in sample.iterrows():
            rid = canonical_id(row.report_id)
            if rid not in raw.index:
                out.append(f'FLAG [{rid}] raw report ID not found; original comparison unavailable')
                continue
            source = raw.loc[rid]
            orig = str(source.get('SYMPTOM_TEXT', ''))
            display = orig[:1200].replace('\n', ' ')
            if len(orig) > 1200:
                display += ' [display truncated; full original in examples JSON]'
            out.append(f"[{row.report_id}] ORIGINAL: {display}")
            out.append(f"           RETAINED: {row.narrative_onset}")
            if bv.onset_narrative(orig) != row.narrative_onset:
                out.append(f'FLAG [{rid}] current extractor differs from retained excerpt; check construction-code version')
            flags = [c for c in bv.OUTCOME_FLAGS if c in source.index]
            if flags and int(source[flags].fillna('').eq('Y').any()) != label:
                out.append(f'FLAG [{rid}] raw recorded seriousness flags disagree with CSV label')
            bg = [f"{c}={row[c]}" for c in schema["text_columns"] if c != "narrative_onset"]
            out.append(f"           BACKGROUND: {'; '.join(bg)[:300]}")
            examples.append(dict(report_id=rid, label=int(label), original_narrative=orig,
                                 retained_excerpt=row.narrative_onset,
                                 original_background={c: str(source.get(c, '')) for c in bv.TEXT_SOURCES.values()},
                                 retained_background={c: str(row[c]) for c in schema['text_columns'] if c != 'narrative_onset'}))
    if samples_path:
        Path(samples_path).parent.mkdir(parents=True, exist_ok=True)
        Path(samples_path).write_text(json.dumps(examples, indent=2) + '\n')
        out.append(f'Full matched examples saved locally: {samples_path}')
    return out


def run(csv: Path, schema_path: Path, results_dir: Path, raw_paths=(), out_path: Path | None = None) -> str:
    missing = [str(p) for p in [csv, schema_path, results_dir] if not Path(p).exists()]
    if missing:
        raise FileNotFoundError('Missing audit inputs: ' + ', '.join(missing))
    schema = json.loads(Path(schema_path).read_text())
    csv_sha = hashlib.sha256(Path(csv).read_bytes()).hexdigest()
    schema_sha = hashlib.sha256(Path(schema_path).read_bytes()).hexdigest()
    df = pd.read_csv(csv)
    df["report_id"] = df.report_id.astype(str)
    runs = load_runs(results_dir)
    labels = dict(zip(df.report_id, df[schema["target"]]))
    lines = mechanics(runs, csv_sha, schema, schema_sha)
    fields = schema['structured_columns'] + schema['text_columns'] + [schema['target']] + schema['audit_columns']
    if len(fields) != len(set(fields)) or set(fields) != set(df.columns):
        lines.append('FLAG CSV columns do not partition into the declared predictors, target and audit fields')
    if not df.report_id.is_unique or df.report_id.eq('nan').any():
        lines.append('FLAG missing or repeated report IDs in CSV')
    if df[schema['target']].isna().any() or set(df[schema['target']].unique()) != {0, 1}:
        lines.append('FLAG expected both binary labels and no missing labels')
    if df[schema['text_columns']].isna().any().any():
        lines.append('FLAG missing text cells in CSV')
    all_ids = set(df.report_id)
    if any(not set(map(str, r['splits'].get(k, []))) <= all_ids for r in runs for k in ['outer_train', 'outer_test']):
        lines.append('FLAG a recorded split refers to IDs absent from the CSV')
    sections = [("1. Mechanics", lines)]
    plan_file = Path(results_dir) / 'plan.json'
    if plan_file.exists():
        from itertools import product
        plan = json.loads(plan_file.read_text())
        jobs = [(r['config']['model_key'], r['config']['fold'], r['config']['condition']) for r in runs]
        expected = set(product(plan['models'], plan['folds'], plan['conditions']))
        if set(jobs) != expected or len(jobs) != len(expected):
            lines.append('FLAG run folders do not match the exact jobs in plan.json')
        if plan.get('csv_sha256') != csv_sha:
            lines.append('FLAG plan.json dataset fingerprint does not match CSV')
    else:
        lines.append('NOT_VERIFIED plan.json missing; requested job coverage cannot be independently checked')

    def save():
        text = '\n'.join(f'\n## {title}\n' + '\n'.join(items) for title, items in sections)
        flags = sum(line.startswith('FLAG ') for _, items in sections for line in items)
        text = (f'# Audit of {Path(results_dir).name} on {Path(csv).name} (csv sha {csv_sha[:12]}…)\n'
                f'Audit version: {AUDIT_VERSION}; script sha256: {hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}\n'
                f'FLAG lines: {flags}\nNo overall leakage-free verdict is assigned. PASS applies only to its named check.\n'
                'Recorded partitions are adapter reconstructions; CPU proxy scores are not official E5/TAR scores.\n' + text)
        if out_path:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            Path(out_path).write_text(text + '\n')
        return text

    save()
    if any(line.startswith('FLAG ') for line in lines):
        raise AuditInputError('Input identity/mechanics failed; proxy fitting stopped. See saved report for FLAG lines.')
    print('Input identity/mechanics passed. Searching candidate clusters...', flush=True)
    coverage = {}
    groups = near_duplicate_groups(df, coverage=coverage)
    duplicate_lines = cross_split_duplicates(groups, runs, labels)
    duplicate_lines.append('Duplicate-search coverage: ' + json.dumps(coverage, sort_keys=True))
    if coverage['large_blocks_exact_only']:
        duplicate_lines.append('NOT_VERIFIED approximate matching inside large blocks was not searched; exact token-set matches were checked')
    duplicate_lines.append('NOT_VERIFIED changed metadata or dissimilar secondary descriptions can escape this detector')
    sections.append(("2. Candidate duplicate clusters across train/test", duplicate_lines))
    save()
    print('Running duplicate-sensitivity CPU proxies...', flush=True)
    sections.append(("3. Duplicate sensitivity (proxy model)", duplicate_sensitivity(df, schema, runs, groups)))
    save()
    print('Running per-fold conditional-stratum CPU proxies...', flush=True)
    sections.append(("4. Text signal within strata (proxy, text only)", strata(df, schema, runs)))
    save()
    print('Running style and incremental-content CPU proxies...', flush=True)
    sections.append(("5. Reporter style vs content", style(df, schema, runs)))
    save()
    if raw_paths:
        samples = Path(out_path).with_suffix('.examples.json') if out_path else None
        sections.append(("6. Original vs retained text (random reports)", originals(df, schema, raw_paths, samples_path=samples)))
    else:
        sections.append(('6. Original vs retained text', ['NOT_VERIFIED no raw yearly files supplied; original-text and label reconstruction were not checked']))
    return save()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    ap.add_argument("schema", type=Path)
    ap.add_argument("results_dir", type=Path)
    ap.add_argument("--raw", nargs="*", default=[])
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    print(run(a.csv, a.schema, a.results_dir, a.raw, a.out))
