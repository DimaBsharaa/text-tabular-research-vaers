"""Read-only comparison of supplied v8/v9 screen ZIPs and exact candidate ZIPs.

No model fitting, source downloads or official benchmark changes. The separate
v7 archive auditor remains unchanged; its v7-specific constants are not reused.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import itertools
import json
import math
from pathlib import Path
import re
import zipfile

import pandas as pd

from research.scripts.vaers import build_vaers as bv
from research.scripts.vaers.run_audit import mechanics


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read_candidate(path):
    with zipfile.ZipFile(path) as z:
        if z.testzip() is not None or len(z.namelist()) != len(set(z.namelist())):
            raise ValueError('Invalid candidate archive')
        schema_bytes = z.read(next(n for n in z.namelist() if n.endswith('/schema.json')))
        schema = json.loads(schema_bytes)
        csv_bytes = z.read(next(n for n in z.namelist() if n.endswith(f"/vaers_{schema['variant']}.csv")))
    if sha(csv_bytes) != schema['csv_sha256']:
        raise ValueError('Candidate hash differs from its schema')
    df = pd.read_csv(io.BytesIO(csv_bytes), dtype=str, keep_default_na=False).set_index('report_id', drop=False)
    if not df.index.is_unique or len(df) != schema['rows']:
        raise ValueError('Repeated IDs or wrong candidate size')
    return df, schema, sha(schema_bytes)


def inspect_run(path, candidate, schema, schema_sha, tree):
    checks, records, runs = [], [], []

    def check(name, ok):
        checks.append({'check': name, 'status': 'PASS' if bool(ok) else 'FAIL'})

    with zipfile.ZipFile(path) as z:
        check('archive_crc_and_unique_names', z.testzip() is None and len(z.namelist()) == len(set(z.namelist())))
        plan = json.loads(z.read('plan.json'))
        check('plan_matches_dataset', plan['csv_sha256'] == schema['csv_sha256'])
        check('no_process_errors', json.loads(z.read('process_errors.json')) == [])
        check('plan_id_matches_archive_name', sha(json.dumps(plan, sort_keys=True).encode())[:16] in Path(path).name)
        exported = pd.read_csv(io.BytesIO(z.read('results.csv')))
        summary_export = pd.read_csv(io.BytesIO(z.read('summary.csv'))).set_index('learner')
        check('unique_exported_run_ids', exported.run_id.is_unique)
        roots = sorted(n[:-12] for n in z.namelist() if n.endswith('/config.json'))
        for root in roots:
            cfg, split, result, proof = [json.loads(z.read(root + '/' + n)) for n in ['config.json', 'splits.json', 'result.json', 'reference_proof.json']]
            summary = result['official_summary']
            runs.append({'config': cfg, 'splits': split, 'result_status': result.get('status')})
            label = f"{cfg['model_key']}/fold{cfg['fold']}/{cfg['condition']}"
            check(label + ': configuration_and_run_identity', result['config'] == cfg and result['run_id'] == root and sha(json.dumps(cfg, sort_keys=True).encode())[:24] == root)
            check(label + ': plan_and_environment', all(cfg[k] == plan[k] for k in ['csv_sha256', 'benchmark_commit', 'adapter_sha256', 'environment_sha256', 'mode']) and sha(z.read(root + '/pip_freeze.txt')) == cfg['environment_sha256'])
            score = summary['test_score']
            row = exported[exported.run_id == root]
            check(label + ': score_export', len(row) == 1 and math.isclose(float(row.iloc[0].auroc), score, abs_tol=1e-14) and summary['metrics_dict']['metrics']['auc'] == score and summary['dataset'] == schema['dataset_id'])
            check(label + ': recorded_ids_in_candidate', all(set(map(str, split[k])) <= set(candidate.index.str.lstrip('0')) for k in ['outer_train', 'outer_test']))
            check(label + ': sizes', summary['n_train'] == len(split['outer_train']) and summary['n_test'] == len(split['outer_test']))
            proof_files = {f['path']: f for f in proof['tracked_files']}
            check(label + ': pinned_reference_inventory', proof['commit'] == plan['benchmark_commit'] == tree['sha'] and {p: f['git_blob'] for p, f in proof_files.items()} == {r['path']: r['sha'] for r in tree['tree'] if r['type'] == 'blob'} and all(f['matches_commit'] for f in proof_files.values()))
            imports_ok = True
            for item in proof['imported_modules']:
                rel = item['loaded_from'].removeprefix('/content/MulTaBench-reference/')
                imports_ok &= rel in proof_files and item['sha256'] == proof_files.get(rel, {}).get('sha256') and item['matches_commit']
            check(label + ': imported_reference_hashes', imports_ok)
            log = z.read(f"{cfg['model_key']}_{cfg['condition']}_{cfg['fold']}.log").decode(errors='replace')
            features = len(schema['structured_columns']) if cfg['condition'] == 'structured_only' else 30 * len(schema['text_columns']) + (len(schema['structured_columns']) if cfg['condition'] != 'text_only' else 0)
            check(label + ': logged_features', bool(re.search(rf'🔭 {features} features', log)))
            if cfg['condition'] == 'joint_tar':
                passages = re.findall(r'\[Dataset\] train: n=(\d+), val: n=(\d+)', log)
                check(label + ': TAR_passage_counts', passages == [(str(len(split['encoder_train']) * len(schema['text_columns'])), str(len(split['encoder_validation']) * len(schema['text_columns'])))])
                check(label + ': TAR_settings', cfg['e5_train_kwargs'] == dict(lora_rank=16, text_layers=3, learning_rate=.0001, epochs=50, patience=3, weight_decay=.001, batch_size=256))
            records.append({'model': cfg['model_key'], 'fold': cfg['fold'], 'condition': cfg['condition'], 'auroc': score})
        lines = mechanics(runs, schema['csv_sha256'], schema, schema_sha)
        checks.extend({'check': line[5:], 'status': 'PASS' if line.startswith('PASS ') else 'FAIL'} for line in lines if line.startswith(('PASS ', 'FLAG ')))
        jobs = {(r['model'], r['fold'], r['condition']) for r in records}
        check('requested_jobs_complete', len(jobs) == len(records) == len(exported) and jobs == set(itertools.product(plan['models'], plan['folds'], plan['conditions'])) and set(exported.run_id) == set(roots))
        frame = pd.DataFrame(records)
        means = frame.groupby(['model', 'condition']).auroc.mean().unstack()
        means['joint_signal_gain'] = means.joint_frozen - means[['structured_only', 'text_only']].max(axis=1)
        means['tar_gain'] = means.joint_tar - means.joint_frozen
        for model, row in means.iterrows():
            check(model + ': summary_arithmetic', all(math.isclose(row[c], summary_export.loc[model, c], abs_tol=1e-14) for c in means.columns))
        folded = frame.pivot(index=['model', 'fold'], columns='condition', values='auroc')
        folded['joint_signal_gain'] = folded.joint_frozen - folded[['structured_only', 'text_only']].max(axis=1)
        folded['tar_gain'] = folded.joint_tar - folded.joint_frozen
        # Canonicalize only for linkage to recorded pandas-inferred numeric IDs.
        linked = candidate.copy(); linked.index = linked.index.str.lstrip('0')
        partitions = []
        splits_by_fold = {}
        for fold in plan['folds']:
            split = next(r['splits'] for r in runs if r['config']['fold'] == fold)
            splits_by_fold[fold] = split
            partitions.append({'fold': fold, **{k: {'rows': len(split[k]), 'serious': int(linked.loc[list(map(str, split[k])), 'serious'].astype(int).sum())} for k in ['outer_train', 'outer_test']}})
        overlaps = [{'folds': [a, b], 'shared_test_ids': len(set(splits_by_fold[a]['outer_test']) & set(splits_by_fold[b]['outer_test']))} for a, b in itertools.combinations(plan['folds'], 2)]
    return {'archive': str(path), 'archive_sha256': sha(Path(path).read_bytes()), 'variant': schema['variant'],
            'plan': plan, 'checks': checks, 'check_counts': {s: sum(c['status'] == s for c in checks) for s in ['PASS', 'FAIL']},
            'means': means.reset_index().to_dict('records'), 'per_fold': folded.reset_index().to_dict('records'),
            'partitions': partitions, 'cross_fold_test_overlap': overlaps}, splits_by_fold


def main(args):
    tree = json.loads(args.tree.read_text())
    data, info, splits = {}, {}, {}
    for variant, archive, candidate_zip in [('v8', args.v8_run, args.v8_data), ('v9', args.v9_run, args.v9_data)]:
        df, schema, schema_sha = read_candidate(candidate_zip)
        if schema['variant'] != variant:
            raise ValueError('Variant does not match supplied argument')
        data[variant] = df
        info[variant], splits[variant] = inspect_run(archive, df, schema, schema_sha, tree)
        info[variant]['candidate'] = {'rows': len(df), 'serious': int(df.serious.astype(int).sum()), 'csv_sha256': schema['csv_sha256'], 'schema_sha256': schema_sha, 'source_members': schema['sources']}
    v7 = pd.read_csv(args.v7_csv, dtype=str, keep_default_na=False).set_index('report_id', drop=False)
    v8, v9 = data['v8'], data['v9']
    common = sorted(set(v8.index) & set(v9.index))
    comparisons = {
        'v7_v8_identical_ordered_ids': v7.index.tolist() == v8.index.tolist(),
        'v7_v8_identical_structured_labels_excerpt': v7.drop(columns=list(bv.TEXT_SOURCES)).equals(v8.drop(columns=['patient_background'])),
        'v8_merged_background_matches_v7': bool(v7.apply(bv.merge_background, axis=1).eq(v8.patient_background).all()),
        'v8_v9_common_ids': len(common), 'v9_ids_not_in_v8': len(set(v9.index) - set(v8.index)),
        'v8_v9_identical_all_fields_on_common_ids': v8.loc[common].equals(v9.loc[common]),
        'v7_v8_same_outer_ids_by_fold': {},
    }
    with zipfile.ZipFile(args.v7_run) as z:
        old_splits = {}
        for name in z.namelist():
            if name.endswith('/config.json'):
                cfg = json.loads(z.read(name))
                old_splits.setdefault(cfg['fold'], json.loads(z.read(name[:-12] + '/splits.json')))
        for fold in splits['v8']:
            old = old_splits.get(fold)
            comparisons['v7_v8_same_outer_ids_by_fold'][fold] = old is not None and all(old[k] == splits['v8'][fold][k] for k in ['outer_train', 'outer_test'])
    source_checks = {}
    if args.raw:
        tables = bv.load_year_files(args.raw)
        hashes = lambda members: {r['file'].split(':')[-1]: r['sha256'] for r in members}
        source = tables['DATA'].drop_duplicates('VAERS_ID').set_index('VAERS_ID')
        for variant, df in data.items():
            same_sources = hashes(tables['sources']) == hashes(info[variant]['candidate']['source_members'])
            if not same_sources or not set(df.index) <= set(source.index):
                raise ValueError('Source fingerprints or report IDs differ')
            selected = source.loc[df.index]
            expected_labels = selected[bv.OUTCOME_FLAGS].fillna('').eq('Y').any(axis=1).astype(int)
            background = pd.DataFrame({col: selected[src].map(bv.clean_text).map(bv.mask_outcome_sentences) for col, src in bv.TEXT_SOURCES.items()})
            source_checks[variant] = {'rows_checked': len(df), 'source_hashes_match': same_sources,
                'label_mismatches': int((expected_labels != df.serious.astype(int)).sum()),
                'excerpt_mismatches': int((selected.SYMPTOM_TEXT.map(bv.onset_narrative) != df.narrative_onset).sum()),
                'merged_background_mismatches': int((background.apply(bv.merge_background, axis=1) != df.patient_background).sum()),
                'narrative_surgery_word_matches': int(df.narrative_onset.str.contains(r'\bsurger\w*\b|\bsurgical\w*\b', case=False, regex=True).sum()),
                'narrative_blood_transfusion_matches': int(df.narrative_onset.str.contains(r'\bblood transfusion\b', case=False, regex=True).sum())}
    result = {'date': '2026-10-17', 'runs': info, 'data_comparisons': comparisons,
              'source_replay': source_checks,
              'limitations': ['Recorded partitions are adapter reconstructions, not independent tensor traces.',
                              'V9 changes both train and test populations; comparisons with v8 are not on a common test set.',
                              'Repeated folds overlap, development selection is adaptive, and paired predictions are unavailable.',
                              'No full semantic leakage clearance is implied; the shared extraction policy has documented care-text gaps.']}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'checks': {v: info[v]['check_counts'] for v in info}, 'means': {v: info[v]['means'] for v in info}, 'comparisons': comparisons}, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ['v8-run', 'v9-run', 'v8-data', 'v9-data', 'v7-csv', 'v7-run', 'tree', 'out']:
        ap.add_argument('--' + name, type=Path, required=True)
    ap.add_argument('--raw', nargs='+', type=Path)
    main(ap.parse_args())
