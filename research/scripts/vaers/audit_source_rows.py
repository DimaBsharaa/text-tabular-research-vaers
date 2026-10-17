"""Read-only source reconstruction and independent text review for an exact v7 CSV.

Reapplying the extractor checks reproducibility, not the semantic adequacy of its
rules. Independent text screens return review candidates, not leakage verdicts.
Detailed examples stay in the caller's local output, outside the repository.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from research.scripts.vaers import build_vaers as bv
from research.scripts.vaers.run_audit import load_runs


def canonical_id(value):
    text = str(value).strip()
    return (text.lstrip('0') or '0') if text.isdigit() else text


REVIEW_PATTERNS = {
    'surgery_words': r'\b(?:surger\w*|surgical\w*)\b',
    'procedure_words': r'\b(?:transfusion|dialysis|catheter\w*|stent\w*|defibrillat\w*|echocardiogra\w*|electrocardiogra\w*)\b',
    'care_actions': r'\b(?:rescue squad|life flight|airlift\w*|medevac|operating room|underwent|outpatient|in-patient|intensive-care)\b',
    'document_pointers': r'\b(?:see attached|attached records|medical records|discharge summary)\b',
    'later_course_words': r'\b(?:months? later|years? later|to date|ongoing|since (?:receiving|getting)|after receiving)\b',
}


def audit(csv, schema_path, raw_paths, results_dir=None):
    schema = json.loads(Path(schema_path).read_text())
    if schema.get('variant') != 'v7':
        raise ValueError('All-row source reconstruction currently supports v7; do not apply it silently to another variant')
    csv_sha = hashlib.sha256(Path(csv).read_bytes()).hexdigest()
    if csv_sha != schema['csv_sha256']:
        raise ValueError('Candidate bytes do not match schema')
    tables = bv.load_year_files(list(map(Path, raw_paths)))
    source_hashes = lambda rows: {r['file'].split(':')[-1]: r['sha256'] for r in rows}
    if source_hashes(tables['sources']) != source_hashes(schema['sources']):
        raise ValueError('Raw source member hashes do not match schema')
    df = pd.read_csv(csv, keep_default_na=False, dtype={'report_id': str})
    df['report_id'] = df.report_id.map(canonical_id)
    if not df.report_id.is_unique:
        raise ValueError('Candidate IDs collide after normalization')
    df = df.set_index('report_id', drop=False)
    raw = tables['DATA'].drop_duplicates('VAERS_ID').copy()
    raw['VAERS_ID'] = raw.VAERS_ID.map(canonical_id)
    if not raw.VAERS_ID.is_unique:
        raise ValueError('Raw source IDs collide after normalization')
    raw = raw.set_index('VAERS_ID')
    missing = sorted(set(df.index) - set(raw.index))
    if missing:
        raise ValueError(f'{len(missing)} candidate IDs missing from source')
    raw = raw.loc[df.index]
    checks = []

    def compare(name, expected, actual, numeric=False):
        if numeric:
            a, b = pd.to_numeric(actual, errors='coerce'), pd.to_numeric(expected, errors='coerce')
            equal = np.isclose(a, b, equal_nan=True)
        else:
            equal = actual.astype(str).to_numpy() == expected.astype(str).to_numpy()
        ids = df.index[~equal].tolist()
        checks.append({'check': name, 'status': 'PASS' if not ids else 'FAIL',
                       'rows_checked': len(df), 'mismatches': len(ids), 'example_ids': ids[:10]})

    flags = raw[bv.OUTCOME_FLAGS].fillna('').eq('Y').any(axis=1).astype(int)
    compare('target_from_raw_seriousness_flags', flags, df[schema['target']], numeric=True)
    compare('onset_excerpt_reproduced_by_current_extractor', raw.SYMPTOM_TEXT.map(bv.onset_narrative), df.narrative_onset)
    for col, source in bv.TEXT_SOURCES.items():
        compare(f'{col}_reproduced_by_current_mask', raw[source].map(bv.clean_text).map(bv.mask_outcome_sentences), df[col])
    age = pd.to_numeric(raw.AGE_YRS, errors='coerce').fillna(pd.to_numeric(raw.CAGE_YR, errors='coerce'))
    compare('age_years_from_source', age.where(age.between(0, 120)), df.age_years, numeric=True)
    onset = pd.to_numeric(raw.NUMDAYS, errors='coerce')
    compare('onset_days_from_source', onset.where(onset.between(0, 3650)), df.onset_days, numeric=True)
    compare('report_year_from_receipt_date', pd.to_datetime(raw.RECVDATE, errors='coerce').dt.year, df.report_year, numeric=True)
    compare('sex_from_source', raw.SEX.fillna('U').replace('', 'U'), df.sex)
    vax = tables['VAX'].copy()
    vax['VAERS_ID'] = vax.VAERS_ID.map(canonical_id)
    first = vax.drop_duplicates('VAERS_ID').set_index('VAERS_ID').loc[df.index]
    compare('n_vaccines_from_source_rows', vax.groupby('VAERS_ID').size().loc[df.index], df.n_vaccines, numeric=True)
    for col, source in {'state': 'STATE', 'admin_by': 'V_ADMINBY',
                        'vax_type': 'VAX_TYPE', 'vax_manu': 'VAX_MANU', 'vax_dose_series': 'VAX_DOSE_SERIES',
                        'vax_route': 'VAX_ROUTE', 'vax_site': 'VAX_SITE'}.items():
        values = raw[source] if col in {'state', 'admin_by'} else first[source]
        values = values.fillna('UNKNOWN').str.strip().replace('', 'UNKNOWN').str.upper()
        expected = values.where(values.isin(schema['category_vocabulary'][col]), 'OTHER')
        compare(f'{col}_from_source_and_declared_vocabulary', expected, df[col])

    combined = pd.DataFrame({'VAX_DATE': raw.VAX_DATE, 'ONSET_DATE': raw.ONSET_DATE,
                             'AGE_YRS': raw.AGE_YRS, 'SEX': raw.SEX, 'STATE': raw.STATE,
                             'VAX_TYPE': first.VAX_TYPE, 'VAX_LOT': first.VAX_LOT})
    complete = combined.notna().all(axis=1) & combined.astype(str).apply(lambda c: c.str.strip().ne('')).all(axis=1)
    event_groups = [set(map(str, ids)) for ids in combined[complete].groupby(list(combined.columns), dropna=False).groups.values() if len(ids) > 1]
    cross = []
    if results_dir:
        runs = load_runs(Path(results_dir))
        for fold in sorted({r['config']['fold'] for r in runs}):
            split = next(r['splits'] for r in runs if r['config']['fold'] == fold)
            train, test = set(map(canonical_id, split['outer_train'])), set(map(canonical_id, split['outer_test']))
            shared = [g for g in event_groups if g & train and g & test]
            cross.append({'fold': fold, 'candidate_clusters': len(shared),
                          'test_members': sum(len(g & test) for g in shared),
                          'groups': [sorted(g) for g in shared]})

    reviews = []
    for name, pattern in REVIEW_PATTERNS.items():
        for col in schema['text_columns']:
            mask = df[col].str.contains(pattern, flags=re.I, regex=True, na=False)
            sub = df[mask]
            examples = []
            # Deterministic class-stratified review; no feature/rule is changed.
            for y in [1, 0]:
                part = sub[sub[schema['target']] == y]
                for rid, row in part.head(3).iterrows():
                    examples.append({'report_id': rid, 'serious': int(y), 'retained': row[col],
                                     'original_narrative': str(raw.at[rid, 'SYMPTOM_TEXT'])})
            reviews.append({'pattern': name, 'column': col, 'matches': len(sub),
                            'serious': int(sub[schema['target']].sum()), 'examples': examples})
    weak_excerpt = ~df.narrative_onset.str.contains(r'[A-Za-z]{3,}', regex=True)
    quality = {'no_ascii_word_of_three_letters': int(weak_excerpt.sum()),
               'serious_among_these': int(df.loc[weak_excerpt, schema['target']].sum()),
               'placeholder_cells_by_column': {c: int(df[c].eq(bv.EMPTY_TEXT).sum()) for c in schema['text_columns']},
               'examples': df.loc[weak_excerpt, ['report_id', 'narrative_onset']].head(6).to_dict('records'),
               'interpretation': 'A narrow review trigger, not a language detector or definitive informativeness test.'}
    return {'csv_sha256': csv_sha, 'schema_sha256': hashlib.sha256(Path(schema_path).read_bytes()).hexdigest(),
            'source_member_hashes_match': True, 'rows': len(df), 'serious': int(df[schema['target']].sum()),
            'source_reconstruction_checks': checks, 'existing_rule_counters': bv.outcome_mentions(df, schema['text_columns']),
            'raw_event_key_candidates': {'complete_key_rows': int(complete.sum()), 'clusters': len(event_groups),
                                          'member_rows': sum(map(len, event_groups)), 'cross_split': cross},
            'independent_text_review': reviews,
            'text_quality': quality,
            'limitations': ['Extractor replay is not independent semantic validation.',
                            'Review-pattern matches include legitimate prior history, negations and false positives.',
                            'Shared complete event keys are candidates, not proven same-patient events; changed or missing metadata can conceal repeats.',
                            'No predictor, label, cohort or model result has been modified.']}


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('csv', type=Path)
    ap.add_argument('schema', type=Path)
    ap.add_argument('--raw', nargs='+', required=True, type=Path)
    ap.add_argument('--results-dir', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    result = audit(args.csv, args.schema, args.raw, args.results_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'rows': result['rows'], 'serious': result['serious'],
                      'source_checks': result['source_reconstruction_checks'],
                      'raw_event_keys': result['raw_event_key_candidates'], 'output': str(args.out)}, indent=2))
