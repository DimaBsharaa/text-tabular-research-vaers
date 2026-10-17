"""Build v10 from a frozen v9 candidate and verified raw sources, without rebuilding v9.

Policy is fixed before diagnostics. No label is inspected by a text transformation
or eligibility rule. Two artifacts are produced: v10_matched (all v9 rows, for
paired sensitivity only) and v10 (eligible rows, with a new immutable build ID).
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile

import pandas as pd

from research.scripts.vaers import build_vaers as bv

BUILD_VERSION = 9
POLICY_VERSION = 'care-boundary-2'
EMPTY = bv.EMPTY_TEXT
# The inherited outcome/chart/vital checks still apply. These are additional
# care classes, fixed from the audit's semantic failures, not fitted vocabulary.
EXTRA_CARE = re.compile(
    r'\b(?:surger(?:y|ies)|surgical(?:ly)?(?!\s+tape)|transfusions?|dialysis|h(?:ae|e)modialysis|'
    r'catheteri[sz]\w*|catheters?|stents?|defibrillat\w*|ablation|bypass|'
    r'arthroscop\w*|debridement|cardioversion|lumbar fusion|bone marrow transplant|'
    r'operating room|rescue squad|life flight|airlift\w*|medevac|'
    r'medical leave|sick leave|physical therapy|occupational therapy|underwent|patholog\w*)\b|'
    r'\bseen for (?:an? )?(?:outpatient|acute|appointment|visit)\b|'
    r'\b(?:was|were) (?:then )?found to have\b|'
    r'\b(?:the )?(?:tissue|clot|mass|lesion|implants?|tumou?r) (?:was|were) '
    r'(?:then )?(?:removed|excised|resected|decompressed)\b|'
    r'\b(?:out of|off|absent from) work\b|'
    r'\b(?:monitored|monitoring) (?:vitals|vital signs)\b|'
    r'\b(?:dentist|specialist|therapist|cardiologist|neurologist|dermatologist)\b|'
    r'\b(?:took|taking|given|received) (?:some |a |an |oral |otc )?'
    r'(?:tylenol|acetaminophen|paracetamol|ibuprofen|benadryl|diphenhydramine|naproxen|pain medication)\b', re.I)
DOC_POINTER = re.compile(r'\b(?:see attached|attached (?:records|letter|report)|medical records|discharge summary)\b', re.I)
EVENT_BACKGROUND = re.compile(
    r'\b(?:after|since|following)\s+(?:(?:receiving|received|getting|given|having|the|this|that|these|those|my|his|her|their|a|'
    r'first|second|third|last|booster|dose|round|series|of)\s+){0,7}'
    r'(?:vaccin\w*|immuni\w*|shots?|injections?|adverse (?:event|reaction)|'
    r'(?:this |the )?reaction|shingrix|abrysvo|prevnar|moderna|pfizer|covid\w*)\b|'
    r'\b(?:given|received|administered)\b.{0,65}\b(?:vaccin\w*|shots?|hep\s*b|heb\s*b)\b|'
    r'\bfollowing this adverse reaction\b', re.I)
PRIOR_CONTEXT = re.compile(r'\b(?:prior to|before|pre-existing|previously|history of|years? ago|months? ago|'
                           r'weeks? (?:prior|before)|days? (?:prior|before)|last year|in childhood)\b', re.I)
# Connectors introducing intervention belong to the removed suffix too. Keep
# the meaningful observed reaction before it, rather than discard the sentence.
TRAILING_CONNECTOR = re.compile(
    r'(?:\s*[,;:-]?\s*)\b(?:(?:that |which )?(?:resulted|resulting|led|leading) (?:in|to)|'
    r'(?:ultimately )?(?:requiring|required|requires)|(?:and )?(?:I |she |he |patient |child )?'
    r'(?:underwent|needed|needs|had)|followed by|removed (?:via|by)|with|for|on|in|a|an|'
    r'extensive arthroscopic|triple|open heart|blood|and (?:was|were|is|has|had|have))\s*$', re.I)
BOILERPLATE_WORDS = set('the and patient reported reports received vaccine vaccination immunization dose after before '
                        'following immediately hours days weeks month year none noted unknown see attached records '
                        'outside facility manufacturer given administered first second third booster shot '
                        'was were are is has had been being did does'.split())


def meaningful(text):
    return any(w not in BOILERPLATE_WORDS for w in re.findall(r'[a-z]{3,}', text.lower()))


def strict_excerpt(text):
    """Extract a literal prefix of the frozen v9 excerpt; never add or rewrite facts."""
    text = bv.clean_text(text)
    hits = [p.search(text) for p in [bv.OUTCOME_SENTENCE, bv.CARE_SENTENCE, bv.CHART_SENTENCE,
                                   bv.VITALS, bv.CAPS_RUN, EXTRA_CARE, DOC_POINTER]]
    hits = [m for m in hits if m]
    reason = 'unchanged'
    if hits:
        first = min(hits, key=lambda m: m.start())
        text = text[:first.start()].rstrip(' ,;:-')
        reason = 'care_or_record_suffix'
        # Strip several stacked connectors, e.g. "requiring blood".
        while TRAILING_CONNECTOR.search(text):
            text = TRAILING_CONNECTOR.sub('', text).rstrip(' ,;:-')
        # A new sentence that consists only of a dangling care-introduction
        # ("She", "Patient has seen her", "I had to go see") carries no reaction.
        start = max(text.rfind('.'), text.rfind('!'), text.rfind('?')) + 1
        tail = re.findall(r'[a-z]+', text[start:].lower())
        if tail and set(tail) <= set('i she he they we patient has have had was were is are been to go see seen her his the a an'.split()):
            text = text[:start].rstrip()
    if not meaningful(text):
        return EMPTY, 'no_informative_excerpt'
    return text, reason


def background_field(text):
    """Retain background clauses; quarantine explicit post-event history at row level.

    Legitimate medication names/doses, negation and clearly prior procedures are
    retained. Ambiguous invasive-care clauses without a prior anchor are omitted.
    """
    text = bv.clean_text(text)
    if text == EMPTY:
        return EMPTY, False, []
    # Detect temporal contamination before legacy masking can erase its evidence.
    if EVENT_BACKGROUND.search(text):
        return EMPTY, True, ['explicit_post_event_background']
    # Use the already frozen legacy mask: do not restore previously excluded text.
    text = bv.mask_outcome_sentences(text)
    if EVENT_BACKGROUND.search(text):
        return EMPTY, True, ['explicit_post_event_background']
    contaminated = False
    kept, reasons = [], []
    for clause in re.split(r'(?<=[.!?])\s+|[;\n]+', text):
        clause = clause.strip()
        if not clause:
            continue
        if EVENT_BACKGROUND.search(clause):
            contaminated = True
            reasons.append('explicit_post_event_background')
            break  # Later clauses may continue that same event.
        if DOC_POINTER.search(clause):
            reasons.append('record_pointer')
            continue
        if EXTRA_CARE.search(clause) and not PRIOR_CONTEXT.search(clause):
            reasons.append('unanchored_care_background')
            continue
        kept.append(clause)
    return bv.clean_text('; '.join(kept)), contaminated, reasons


def canonical_id(value):
    s = str(value).strip()
    return (s.lstrip('0') or '0') if s.isdigit() else s


def read_base(path):
    path = Path(path)
    if path.suffix == '.zip':
        with zipfile.ZipFile(path) as z:
            schema_bytes = z.read(next(n for n in z.namelist() if n.endswith('/schema.json')))
            csv_bytes = z.read(next(n for n in z.namelist() if n.endswith('/vaers_v9.csv')))
    else:
        csv_bytes, schema_bytes = path.read_bytes(), path.with_name('schema.json').read_bytes()
    schema = json.loads(schema_bytes)
    if schema.get('variant') != 'v9' or schema.get('build_version') != 8:
        raise ValueError('Use the frozen build-8 v9 as the parent')
    if hashlib.sha256(csv_bytes).hexdigest() != schema['csv_sha256']:
        raise ValueError('Parent CSV identity differs from its schema')
    df = pd.read_csv(io.BytesIO(csv_bytes), keep_default_na=False, dtype=str)
    df.report_id = df.report_id.map(canonical_id)
    if not df.report_id.is_unique:
        raise ValueError('Parent report IDs are not unique')
    return df, schema, hashlib.sha256(schema_bytes).hexdigest()


def refine(parent, raw):
    """No row labels are accessed here: return transformed fields and eligibility."""
    if not set(parent.report_id) <= set(raw.index):
        raise ValueError('A parent ID is absent from the raw sources')
    out = parent.copy()
    ledger = []
    for i, row in parent.iterrows():
        rid = canonical_id(row.report_id)
        source = raw.loc[rid]
        excerpt, reason = strict_excerpt(row.narrative_onset)
        fields, bg_reasons, contaminated = {}, [], False
        for col, src in bv.TEXT_SOURCES.items():
            value, post_event, reasons = background_field(source.get(src, EMPTY))
            fields[col] = value
            contaminated |= post_event
            bg_reasons.extend(f'{col}:{r}' for r in reasons)
        if contaminated:
            # A history list can repeat later conditions described in another
            # field. Do not leave such a list behind after dropping its evidence.
            fields = {col: EMPTY for col in bv.TEXT_SOURCES}
        background = bv.merge_background(fields)
        if EVENT_BACKGROUND.search(background):
            contaminated = True
            background = EMPTY
            bg_reasons.append('merged:explicit_post_event_background')
        out.at[i, 'narrative_onset'] = excerpt
        out.at[i, 'patient_background'] = background
        ledger.append({'report_id': rid, 'excerpt_reason': reason,
                       'background_reasons': bg_reasons, 'post_event_background': contaminated,
                       'eligible': excerpt != EMPTY and not contaminated,
                       'excerpt_changed': excerpt != row.narrative_onset,
                       'background_changed': background != row.patient_background})
    return out, pd.DataFrame(ledger)


def build(base, raw_paths, out_dir):
    parent, schema, parent_schema_sha = read_base(base)
    tables = bv.load_year_files(list(map(Path, raw_paths)))
    member_hashes = lambda members: {r['file'].split(':')[-1]: r['sha256'] for r in members}
    if member_hashes(tables['sources']) != member_hashes(schema['sources']):
        raise ValueError('Raw source version differs from frozen v9')
    raw = tables['DATA'].drop_duplicates('VAERS_ID').copy()
    raw.VAERS_ID = raw.VAERS_ID.map(canonical_id)
    if not raw.VAERS_ID.is_unique:
        raise ValueError('Source IDs collide after normalization')
    raw = raw.set_index('VAERS_ID')
    # Verify alignment before applying any correction. Labels are audit-only.
    labels = raw.loc[parent.report_id, bv.OUTCOME_FLAGS].fillna('').eq('Y').any(axis=1).astype(int).to_numpy()
    if not (labels == parent.serious.astype(int).to_numpy()).all():
        raise ValueError('Parent target does not match source flags')
    corrected, ledger = refine(parent, raw)
    protected = schema['structured_columns'] + ['serious', 'report_id']
    if not corrected[protected].equals(parent[protected]):
        raise AssertionError('A protected field changed')
    target = parent.serious.astype(int)
    counts = {'parent_rows': len(parent), 'kept_rows': int(ledger.eligible.sum()),
              'removed_rows': int((~ledger.eligible).sum()),
              'excerpt_changed': int(ledger.excerpt_changed.sum()),
              'background_changed': int(ledger.background_changed.sum()),
              'post_event_background_rows': int(ledger.post_event_background.sum()),
              'uninformative_excerpt_rows': int(corrected.narrative_onset.eq(EMPTY).sum()),
              'by_label': {str(y): {'parent': int(target.eq(y).sum()),
                                   'retained': int((target.eq(y) & ledger.eligible).sum()),
                                   'removed': int((target.eq(y) & ~ledger.eligible).sum())} for y in [0, 1]}}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if any((out_dir / f'vaers_{v}').exists() for v in ['v10', 'v10_matched', 'v9_control']):
        raise FileExistsError('Choose a new output directory; frozen outputs are never overwritten')
    code_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    paths = {}
    for variant, data, purpose in [('v10_matched', corrected, 'diagnostic_sensitivity'),
                                    ('v10', corrected[ledger.eligible].copy(), 'candidate_dataset'),
                                    ('v9_control', parent[ledger.eligible].copy(), 'diagnostic_sensitivity')]:
        folder = out_dir / f'vaers_{variant}'
        folder.mkdir()
        csv = folder / f'vaers_{variant}.csv'
        data.to_csv(csv, index=False)
        new_schema = deepcopy(schema)
        new_schema.update(dataset_id=f'BIN_TEXT_VAERS_SERIOUS_{variant.upper()}', variant=variant,
                          build_version=BUILD_VERSION, purpose=purpose, rows=len(data),
                          positive_rate=float(data.serious.astype(int).mean()),
                          csv_sha256=hashlib.sha256(csv.read_bytes()).hexdigest(),
                          counts=counts, narrative_onset_policy='Literal prefix of frozen v9 excerpt, cut before care, functional outcome or record pointers; keep observed reaction before the cut. No generated facts.',
                          background_policy='Legacy outcome masking, omit unanchored invasive-care clauses and record pointers; explicit post-event background quarantines all background for matched sensitivity and excludes the row from v10.',
                          population=schema['population'] + f' Further restricted by the label-blind {POLICY_VERSION} eligibility rules. The resulting class ratio may differ from the parent 25%.',
                          sampling='Frozen case-enriched v9 IDs; no new sampling or rebalancing after text correction.',
                          policy_version=POLICY_VERSION,
                          provenance={'parent_csv_sha256': schema['csv_sha256'], 'parent_schema_sha256': parent_schema_sha,
                                      'refinement_sha256': code_sha, 'legacy_builder_sha256': hashlib.sha256(Path(bv.__file__).read_bytes()).hexdigest()})
        if variant == 'v10_matched':
            new_schema['population'] = 'Exactly the frozen v9 rows and labels; newly censored excerpts use none reported. Diagnostic sensitivity only, not the final candidate cohort.'
        if variant == 'v9_control':
            new_schema['narrative_onset_policy'] = schema['narrative_onset_policy']
            new_schema['background_policy'] = 'Original frozen v9 background, retained only as a diagnostic control on the exact v10 cohort.'
        (folder / 'schema.json').write_text(json.dumps(new_schema, indent=2) + '\n')
        paths[variant] = str(csv)
    # Full before/after texts remain in the local dataset artifacts, not Git.
    ledger.to_json(out_dir / 'row_lineage.jsonl', orient='records', lines=True)
    (out_dir / 'build_summary.json').write_text(json.dumps({'policy': POLICY_VERSION, 'counts': counts, 'paths': paths}, indent=2) + '\n')
    print(json.dumps({'policy': POLICY_VERSION, 'counts': counts, 'paths': paths}, indent=2), flush=True)
    return paths, counts


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base', type=Path, required=True)
    ap.add_argument('--raw', type=Path, nargs='+', required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    args = ap.parse_args()
    build(args.base, args.raw, args.out_dir)
