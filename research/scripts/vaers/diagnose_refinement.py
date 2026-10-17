"""Matched CPU diagnostics for v9 -> v10. Never substitutes for official E5/TAR.

The existing folds are development folds. Policy/data hashes are frozen before
fitting, transformations fit on training rows, and row-level predictions are kept
locally for matched comparisons. No selection of a new policy from these scores.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from research.scripts.vaers.refine_v9 import read_base, canonical_id, EXTRA_CARE, EVENT_BACKGROUND
from research.scripts.vaers.run_audit import _proxy


def read_csv(path):
    df = pd.read_csv(path)
    df.report_id = df.report_id.map(canonical_id)
    return df.set_index('report_id', drop=False)


def run(base, build_dir, results_zip, out_dir):
    parent, parent_schema, _ = read_base(base)
    # Match pandas NA interpretation used by notebook 02.
    parent = read_csv(io.StringIO(parent.to_csv(index=False)))
    root = Path(build_dir)
    paths = {v: root / f'vaers_{v}' / f'vaers_{v}.csv' for v in ['v10', 'v10_matched', 'v9_control']}
    frames = {v: read_csv(p) for v, p in paths.items()}
    for v, path in paths.items():
        s = json.loads(path.with_name('schema.json').read_text())
        if hashlib.sha256(path.read_bytes()).hexdigest() != s['csv_sha256']:
            raise ValueError('A corrected CSV differs from its frozen schema')
    eligible = set(frames['v10'].index)
    protected = parent_schema['structured_columns'] + ['serious', 'report_id']
    if not frames['v10_matched'][protected].equals(parent[protected]):
        raise ValueError('Matched input rows/protected fields differ')
    if not frames['v9_control'].index.equals(frames['v10'].index):
        raise ValueError('Candidate and control IDs/order differ')
    with zipfile.ZipFile(results_zip) as z:
        plan = json.loads(z.read('plan.json'))
        if plan['csv_sha256'] != parent_schema['csv_sha256']:
            raise ValueError('Reference result splits belong to a different parent CSV')
        splits = {}
        for name in z.namelist():
            if name.endswith('/config.json'):
                cfg = json.loads(z.read(name))
                split = json.loads(z.read(name[:-12] + '/splits.json'))
                old = splits.setdefault(cfg['fold'], split)
                if any(old[k] != split[k] for k in ['outer_train', 'outer_test']):
                    raise ValueError('Reference outer partitions differ across conditions')
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    if (out / 'summary.json').exists():
        raise FileExistsError('Diagnostics already exist; choose a new output directory instead of overwriting')
    frozen = {'parent_csv_sha256': parent_schema['csv_sha256'],
              'data_sha256': {v: hashlib.sha256(p.read_bytes()).hexdigest() for v, p in paths.items()},
              'reference_archive_sha256': hashlib.sha256(Path(results_zip).read_bytes()).hexdigest(),
              'proxy': 'TF-IDF (train-only vocabulary/IDF) + logistic; no frozen E5 or TAR',
              'protocols': ['all_parent_rows', 'same_eligible_rows'], 'folds': sorted(splits),
              'negative_controls': ['fold0_permuted_training_labels', 'fold0_independently_shuffled_text'],
              'diagnostic_code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out / 'plan.json').write_text(json.dumps(frozen, indent=2) + '\n')
    (out / 'diagnostic_source.py').write_bytes(Path(__file__).read_bytes())
    rows, predictions = [], []
    cols = {'structured': parent_schema['structured_columns'], 'text': parent_schema['text_columns'],
            'joint': parent_schema['structured_columns'] + parent_schema['text_columns']}

    def fit_score(variant, protocol, fold, train, test, condition, shuffle=None):
        features = cols[condition]
        train, test = train.copy(), test.copy()
        y = train.serious.copy()
        rng = np.random.default_rng(715)
        if shuffle == 'labels':
            y = rng.permutation(y)
        elif shuffle == 'text':
            for c in parent_schema['text_columns']:
                train[c] = rng.permutation(train[c].to_numpy())
                test[c] = rng.permutation(test[c].to_numpy())
        model = _proxy(parent_schema, features).fit(train[features], y)
        p = model.predict_proba(test[features])[:, 1]
        score = roc_auc_score(test.serious, p)
        rows.append({'variant': variant, 'protocol': protocol, 'fold': fold, 'condition': condition,
                     'auroc': float(score), 'n_train': len(train), 'n_test': len(test),
                     'train_serious': int(train.serious.sum()), 'test_serious': int(test.serious.sum())})
        predictions.append(pd.DataFrame({'variant': variant, 'protocol': protocol, 'fold': fold,
                                         'condition': condition, 'report_id': test.index,
                                         'y': test.serious.to_numpy(), 'probability': p}))
        print(f'{protocol} / {variant} / fold {fold} / {condition}: {score:.4f}', flush=True)

    for fold, split in sorted(splits.items()):
        tr, te = [list(map(canonical_id, split[k])) for k in ['outer_train', 'outer_test']]
        if set(tr) & set(te) or not set(tr + te) <= set(parent.index):
            raise ValueError('Invalid recorded split')
        for protocol in ['all_parent_rows', 'same_eligible_rows']:
            train = tr if protocol == 'all_parent_rows' else [i for i in tr if i in eligible]
            test = te if protocol == 'all_parent_rows' else [i for i in te if i in eligible]
            base_train, base_test = parent.loc[train], parent.loc[test]
            fixed = frames['v10_matched'] if protocol == 'all_parent_rows' else frames['v10']
            for condition in cols:
                fit_score('v9', protocol, fold, base_train, base_test, condition)
            for condition in ['text', 'joint']:
                fit_score('v10', protocol, fold, fixed.loc[train], fixed.loc[test], condition)
        if fold == 0:
            for name, shuffle in [('label_permutation', 'labels'), ('text_permutation', 'text')]:
                fit_score(name, 'negative_control', fold, frames['v10'].loc[train], frames['v10'].loc[test], 'joint', shuffle)
    scores = pd.DataFrame(rows)
    scores.to_csv(out / 'scores.csv', index=False)
    pd.concat(predictions, ignore_index=True).to_csv(out / 'paired_predictions.csv.gz', index=False)
    means = scores.groupby(['protocol', 'variant', 'condition']).auroc.mean().unstack()
    summary = means.astype(object).where(pd.notna(means), None).reset_index().to_dict('records')
    flags = {}
    ledger = pd.read_json(root / 'row_lineage.jsonl', lines=True)
    y = parent.loc[ledger.report_id.map(canonical_id)].serious.to_numpy()
    for col in ['post_event_background', 'eligible', 'excerpt_changed']:
        f = ledger[col].astype(int)
        a = roc_auc_score(y, f) if f.nunique() > 1 else None
        flags[col] = {'audit_only_flag_auroc': a, 'orientation_free_auroc': max(a, 1-a) if a is not None else None}
    validation = {'protected_fields_unchanged': True, 'candidate_control_same_ordered_ids': True,
                  'remaining_extra_care_matches_in_candidate_excerpts': int(frames['v10'].narrative_onset.map(lambda s: bool(EXTRA_CARE.search(s))).sum()),
                  'remaining_explicit_post_event_matches_in_candidate_background': int(frames['v10'].patient_background.map(lambda s: bool(EVENT_BACKGROUND.search(s))).sum())}
    result = {'plan': frozen, 'means': summary, 'scores': rows, 'quality_checks': validation,
              'selection_flags': flags,
              'limitations': ['CPU proxy only; E5 Joint Frozen and TAR require notebook 02.',
                              'Previously inspected development folds; not a fresh confirmatory evaluation.',
                              'Eligible-row comparison fits both texts on the same restricted train/test rows.',
                              'Zero pattern hits validates implemented rules, not universal semantic absence of leakage.']}
    (out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ['base', 'build-dir', 'results-zip', 'out-dir']:
        ap.add_argument('--' + name, required=True, type=Path)
    a = ap.parse_args()
    run(a.base, a.build_dir, a.results_zip, a.out_dir)
