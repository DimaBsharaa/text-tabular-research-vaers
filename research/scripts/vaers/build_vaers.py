"""Build the VAERS severity-triage candidate (Track 2) from official VAERS yearly files.

Task: from what is known when a vaccine adverse-event report is filed — the coded symptoms, the patient's
background (history, current illness, medications, allergies) and structured facts (age, sex, vaccine, onset
interval) — predict whether the event was SERIOUS (death, life-threatening, hospitalisation, prolonged
hospitalisation, disability or birth defect). A triage analogue of a caseworker reading a file and grading it.

Inputs: the yearly ZIPs from https://vaers.hhs.gov/data/datasets.html (e.g. 2019VAERSData.zip) or the three
yearly CSVs (YYYYVAERSDATA.csv, YYYYVAERSVAX.csv, YYYYVAERSSYMPTOMS.csv). Downloads need a verification
code in a browser, so they are fetched manually and passed in by path.

Leakage rules (all enforced in code and recorded in schema.json):
  * Outcome flags (DIED, L_THREAT, HOSPITAL, X_STAY, DISABLE, BIRTH_DEFECT) build the label and are never predictors.
    Other post-event fields are also excluded: ER/office visits, HOSPDAYS, RECOVD, dates of death, LAB_DATA.
  * MedDRA symptom terms that name an outcome or a care setting (Death, Hospitalisation, Intensive care,
    Endotracheal intubation, ...) are removed from the symptom text; the removed list is saved.
  * v1/v3 exclude the free narrative SYMPTOM_TEXT. v4/v5 include it after removing reporter boilerplate,
    identifiers and every sentence that mentions an outcome or care setting (the same masking is applied to the
    four background fields in every variant).
  * Manufacturer reports (non-empty SPLTTYPE) are dropped: they are mandated mainly for serious events and use a
    templated text format, which would let the model learn the reporter type instead of the medicine.
  * Likely secondary/duplicate reports (same vaccination date, onset date, age, sex, state, vaccine type and lot)
    are collapsed to the earliest VAERS_ID.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

BUILD_VERSION = 5  # 4 = scope: vaccine reactions only (no breakthrough/infection reports), onset 0-42 days.
                   # 5 = onset excerpt also cut at chart/diagnosis-list language; joined excerpt re-checked.
# 3 = background/narrative outcome masking, symptom-aware dedup, duplicate-input guard.
# Reports of the infection the vaccine targets (breakthrough COVID etc.) or of vaccine failure are not reactions to
# the vaccine, and serious breakthrough cases were reported preferentially (hospitalised/fatal cases were the ones
# required to be reported), so in those reports the label follows from how the report was collected.
OUT_OF_SCOPE_TERM = re.compile(r"(?:^|; )(?:vaccine breakthrough infection|breakthrough covid-19|covid-19|"
                               r"covid-19 pneumonia|sars-cov-2 test positive|sars-cov-2 infection|"
                               r"vaccination failure|drug ineffective|lack of vaccination site reaction)(?:;|$)", re.I)
MAX_ONSET_DAYS = 42  # Usual risk window for adverse events following immunisation; unknown onset is kept.
# v1: coded symptoms + 4 background fields. v3: background merged into one column.
# v4: v1 + the reporter's free-text narrative (outcome sentences removed). v5: narrative instead of coded symptoms.
# (v2, an earlier narrative variant, was dropped after the first screen.)
# v6: v1 + only the OPENING of the narrative (the reporter's account of the reaction), cut at the first sentence
# about care, treatment, tests or outcome. v4/v5 were rejected: the full narrative carries hospital-chart text
# (remdesivir, IV, O2, ICD "unspecified", "PMHx", dates) that reveals hospital care without naming it.
# v7: NO coded symptoms (they are coded from the whole report, including the hospital course). Text = the patient's
# background (pre-existing by nature) + the opening narrative only: "from the patient's background and their first
# description of the reaction, did it turn out serious?"
VARIANTS = ("v1", "v3", "v4", "v5", "v6", "v7")
OUTCOME_FLAGS = ["DIED", "L_THREAT", "HOSPITAL", "X_STAY", "DISABLE", "BIRTH_DEFECT"]
TEXT_SOURCES = {"history": "HISTORY", "current_illness": "CUR_ILL", "other_meds": "OTHER_MEDS", "allergies": "ALLERGIES"}
EMPTY_TEXT = "none reported"
# Strings reporters use for "nothing" — normalised to one token so the model does not learn reporter style.
NULL_LIKE = re.compile(r"^\s*(none|no|n/?a|na|nka|nkda|nkfa|unknown|unk|not known|none known|none reported|"
                       r"no known allergies|no known drug allergies|denies|nil|-+|\.+|0)\s*\.?\s*$", re.I)
# MedDRA preferred terms that reveal the outcome or the care setting rather than the reaction itself.
OUTCOME_TERM = re.compile(r"death|died|autopsy|post ?mortem|hospital|intensive care|intubation|mechanical ventilation|"
                          r"resuscitation|life support|hospice|palliative|emergency care|bedridden|disabilit|dialysis|"
                          r"tracheostomy|oxygen therapy|transfusion|surgery|induction of coma|critical illness|"
                          r"life.?threatening|congenital anomaly|birth defect|fatal", re.I)
OUTCOME_TERM_KEEP = {"hyperventilation", "hypoventilation", "fear of death", "near death experience",
                     "death of relative", "hypertensive emergency"}
# Care-process terms: tests, scans and treatments say how intensively the patient was handled (a proxy for
# hospital care), not what the reaction was. Removed unless they carry a finding (abnormal, increased, ...).
CARE_TERM = re.compile(r"\btests?\b|laboratory|imaging|tomogram|\bscan\b|x-ray|ultrasound|echocardiogram|"
                       r"electrocardiogram|biopsy|angiogram|monitoring|investigation|examination|culture|\bcount\b|"
                       r"analysis|puncture|catheterisation|endoscopy|therapy|treatment", re.I)
CARE_FINDING = re.compile(r"abnormal|increased|decreased|positive|negative|prolonged|elevat|depress|inversion|"
                          r"\blow\b|\bhigh\b|present|absent|non-responder|noncompliance|refusal", re.I)
# Vaccination-process terms (errors, storage, schedule, "No adverse event"): these reports describe how a vaccine
# was given, not a reaction. They are removed from the text; reports left with no reaction term are dropped.
ADMIN_TERM = re.compile(r"^no adverse event$|product|administ|vaccination error|medication error|underdose|overdose|"
                        r"off label|unapproved|dose omission|interchange of vaccine|exposure (?:via|during)|"
                        r"circumstance or information|accidental exposure", re.I)
ADMIN_TERM_KEEP = {"productive cough", "shoulder injury related to vaccine administration",
                   "administration related reaction"}
ADMIN_TERM_KEEP_PREFIX = "administration site"  # Local reactions (pain, swelling, ...) are real reactions, keep them.
# Sentences that state the outcome or the care setting of THIS event. Used on every free-text field (background
# fields and the narrative). Deliberately broad: an old admission mentioned in the history is also removed, which
# costs some legitimate background but keeps the label out of the text. Removal counts are saved in the schema.
OUTCOME_SENTENCE = re.compile(r"hospital|admit|admission|discharg|\bicu\b|intensive care|\ber\b|emergency|\bed\b|"
                              r"urgent care|died|death|dead|deceased|passed away|expired|autopsy|fatal|intubat|"
                              r"ventilat|life.?threat|resuscitat|\bcpr\b|disab|inpatient|ambulance|\bems\b|911|"
                              r"paramedic|transport|taken to|transferred|observation unit|telemetry|life support|"
                              r"hospice|code blue|serious|birth defect|congenital|permanent|\bward\b|overnight|"
                              r"\bstay(?:ed)?\b|\bbedside\b", re.I)  # ward/overnight/stay: inpatient stays without the word "hospital"
NARRATIVE_OUTCOME = OUTCOME_SENTENCE  # Backwards-compatible name.
# Reporter-system boilerplate seen in narratives ("Reported Symptoms: 10016256:FATIGUE; Narrative: ...") and
# identifiers (lot numbers, long code numbers) that describe the report, not the patient.
NARRATIVE_HEADER = re.compile(r"^.*?\bnarrative:\s*", re.I | re.S)
IDENTIFIER = re.compile(r"\blot\s*(?:#|no\.?|number)?\s*:?\s*[A-Z0-9-]{4,}\b|\b\d{6,}\b", re.I)
STRUCTURED = ["age_years", "sex", "state", "onset_days", "vax_type", "vax_manu", "vax_dose_series", "vax_route",
              "vax_site", "n_vaccines", "admin_by", "report_year"]
NUMERIC = ["age_years", "onset_days", "n_vaccines", "report_year"]
MAX_CATEGORIES = 99  # Values are kept as-is; grouping happens only if a column would reach the official
# 100-distinct text-detector threshold (then the rarest values become OTHER). Label-free, fixed at build time.


def _read(name: str, raw: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(raw), encoding="latin-1", dtype=str, keep_default_na=False, na_values=[""],
                       low_memory=False)


def load_year_files(paths: list[Path]) -> dict[str, pd.DataFrame]:
    """Concatenate DATA / VAX / SYMPTOMS tables from yearly ZIPs or CSVs."""
    parts = {"DATA": [], "VAX": [], "SYMPTOMS": []}
    sources = []
    seen_files, seen_hashes = set(), set()
    for path in paths:
        members = []
        if path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as zf:
                members = [(n, zf.read(n)) for n in zf.namelist() if n.lower().endswith(".csv")]
        else:
            members = [(path.name, path.read_bytes())]
        for name, raw in members:
            upper = Path(name).name.upper()
            if "NONDOMESTIC" in upper:
                continue  # Foreign reports have blanked narratives and different reporting rules.
            kind = next((k for k in ("SYMPTOMS", "VAX", "DATA") if upper.endswith(f"VAERS{k}.CSV")), None)
            if kind is None:
                continue
            digest = hashlib.sha256(raw).hexdigest()
            file_key = Path(name).name.upper()
            if file_key in seen_files or digest in seen_hashes:
                # e.g. a yearly ZIP AND its extracted CSVs: would double-count vaccines and reports.
                raise ValueError(f"Duplicate VAERS input {name} (pass each year once: ZIP or CSVs, not both)")
            seen_files.add(file_key); seen_hashes.add(digest)
            parts[kind].append(_read(name, raw))
            sources.append({"file": f"{path.name}:{name}", "sha256": digest, "bytes": len(raw)})
    missing = [k for k, v in parts.items() if not v]
    if missing:
        raise FileNotFoundError(f"No VAERS {missing} files among {paths}")
    tables = {k: pd.concat(v, ignore_index=True) for k, v in parts.items()}
    tables["sources"] = sources
    return tables


def clean_text(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return EMPTY_TEXT
    text = re.sub(r"\s+", " ", str(value)).strip()
    return EMPTY_TEXT if not text or NULL_LIKE.match(text) else text


def symptom_text(symptoms: pd.DataFrame) -> tuple[pd.Series, dict[str, list[str]]]:
    """Join each report's reaction terms; return removed terms per category (outcome / care_process / administration)."""
    cols = [c for c in symptoms.columns if re.fullmatch(r"SYMPTOM\d+", c)]
    long = symptoms.melt(id_vars="VAERS_ID", value_vars=cols, value_name="term").dropna(subset=["term"])
    long["term"] = long.term.str.strip()
    long = long[long.term != ""]
    lower = long.term.str.lower()
    is_outcome = long.term.str.contains(OUTCOME_TERM) & ~lower.isin(OUTCOME_TERM_KEEP)
    is_care = long.term.str.contains(CARE_TERM) & ~long.term.str.contains(CARE_FINDING) & ~is_outcome
    is_admin = (long.term.str.contains(ADMIN_TERM) & ~lower.isin(ADMIN_TERM_KEEP)
                & ~lower.str.startswith(ADMIN_TERM_KEEP_PREFIX) & ~is_outcome & ~is_care)
    removed = {"outcome": sorted(long.loc[is_outcome, "term"].unique()),
               "care_process": sorted(long.loc[is_care, "term"].unique()),
               "administration": sorted(long.loc[is_admin, "term"].unique())}
    kept = long[~(is_outcome | is_care | is_admin)].drop_duplicates(["VAERS_ID", "term"])
    joined = kept.sort_values("term").groupby("VAERS_ID").term.agg("; ".join)
    return joined, removed


BACKGROUND_LABELS = {"history": "history", "current_illness": "current illness", "other_meds": "medications",
                     "allergies": "allergies"}


def merge_background(row) -> str:
    """'history: ...; medications: ...' using only the fields that say something; EMPTY_TEXT if none do."""
    parts = [f"{label}: {row[col]}" for col, label in BACKGROUND_LABELS.items() if row[col] != EMPTY_TEXT]
    return "; ".join(parts) if parts else EMPTY_TEXT


def mask_outcome_sentences(text) -> str:
    """Drop every sentence/clause that mentions an outcome or care setting; EMPTY_TEXT if nothing is left."""
    text = clean_text(text)
    if text == EMPTY_TEXT:
        return text
    tokens = re.split(r"(\s*[;\n]\s*|(?<=[.!?])\s+)", text)  # pieces at even positions, separators at odd
    out, sep = [], ""
    for i, tok in enumerate(tokens):
        if i % 2:
            sep = tok
            continue
        if tok.strip() and not OUTCOME_SENTENCE.search(tok):
            out.append((sep if out else "") + tok)  # keep the original separator ("; " or ". ") between kept pieces
    return clean_text("".join(out))


def mask_narrative(text) -> str:
    """Narrative: strip reporter boilerplate and identifiers, then outcome sentences."""
    text = clean_text(text)
    if text == EMPTY_TEXT:
        return text
    text = NARRATIVE_HEADER.sub("", text)
    text = IDENTIFIER.sub(" ", text)
    return mask_outcome_sentences(text)


# Care, treatment and testing language. The onset narrative stops at the first sentence that contains any of it,
# because what follows is usually the course of care, which reveals seriousness without stating it.
# ("given" alone is not a cut point: "Vaccine was given at noon." is part of the onset story.)
CARE_SENTENCE = re.compile(r"\btreat|\btx\b|\bgiven (?:iv|o2|oxygen|fluids|epi|epinephrine|benadryl|steroids?|"
                           r"medication|medicine|antibiotics?|a shot of|an injection of)|\biv\b|\bo2\b|oxygen|remdesivir|dexamethasone|steroid|"
                           r"epinephrine|epi-?pen|\bepi\b|antibiotic|\bheparin|\bcpap|bipap|\btests?\b|\blab|\bct\b|"
                           r"\bmri\b|x-?ray|\bscan|\bekg\b|\becg\b|echocardiogram|\becho\b|culture|biopsy|"
                           r"\bdoctor|\bdr\.?\s|physician|\bpcp\b|clinic|\bnurse|provider|presented|went to|"
                           r"seen (?:by|at|in)|diagnos|\bdx\b|prescribed|\bmg\b", re.I)
DATE_OR_CODE = re.compile(r"\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b|\b(?:19|20)\d{2}\b|\b[A-Z]\d{2}\.\d{1,4}\b|"
                          r"\b(?:pmhx?|hx|y/?o)\b", re.I)
MAX_ONSET_SENTENCES = 3
# Chart and diagnosis-list language (ICD-10 descriptors, EHR markers, clinical shorthand). Build 4's n-gram audit
# showed "unspecified" and "init" (as in "..., unspecified, init encntr") among the strongest serious-class words:
# hospital staff paste discharge diagnoses into the narrative, which reveals the hospital course without naming it.
CHART_SENTENCE = re.compile(r"unspecified|\binit\b|encntr|\bencounter\b|sequela|\bnos\b|elsewhere classified|"
                            r"without complication|\bhcc\b|\bs/p\b|\bc/o\b|\bh/o\b|chief complaint|\bhpi\b|"
                            r"\bassessment\b|\bimpression\b|\bplan\s*:|\bpatient\s+is\s+a\b", re.I)


def _cuts(sentence: str) -> bool:
    return bool(OUTCOME_SENTENCE.search(sentence) or CARE_SENTENCE.search(sentence) or CHART_SENTENCE.search(sentence))


def onset_narrative(text) -> str:
    """The reporter's opening account of the reaction: boilerplate, identifiers, dates and chart codes removed,
    cut at the first sentence that mentions care, testing, treatment or an outcome; at most 3 sentences."""
    text = clean_text(text)
    if text == EMPTY_TEXT:
        return text
    text = NARRATIVE_HEADER.sub("", text)
    text = DATE_OR_CODE.sub(" ", IDENTIFIER.sub(" ", text))
    kept = []
    for sentence in re.split(r"(?<=[.!?;])\s+|\n+", text):
        if not sentence.strip():
            continue
        if _cuts(sentence):
            break
        kept.append(sentence.strip())
        if len(kept) == MAX_ONSET_SENTENCES:
            break
    # The sentence splitter can break inside a match ("seen by Dr. Lee" splits after "Dr."), so the joined excerpt
    # is checked again and shortened until it is clean. The leak counter is then 0 by construction.
    while kept and _cuts(" ".join(kept)):
        kept.pop()
    return clean_text(re.sub(r"\s+", " ", " ".join(kept)))


def outcome_mentions(df: pd.DataFrame, columns) -> dict:
    """Leak counter: cells per text column that still match the outcome/care-setting pattern (and, for the onset
    narrative, the care/treatment/testing pattern too)."""
    counts = {c: int(df[c].astype(str).str.contains(OUTCOME_SENTENCE).sum()) for c in columns}
    if "narrative_onset" in columns:
        counts["narrative_onset_care_words"] = int(df.narrative_onset.astype(str).str.contains(CARE_SENTENCE).sum())
        counts["narrative_onset_chart_words"] = int(df.narrative_onset.astype(str).str.contains(CHART_SENTENCE).sum())
    return counts


def cap_categories(s: pd.Series, k: int = MAX_CATEGORIES) -> pd.Series:
    s = s.fillna("UNKNOWN").astype(str).str.strip().replace("", "UNKNOWN").str.upper()
    if s.nunique() <= k:
        return s  # no grouping needed
    top = s.value_counts().index[:k - 1]
    return s.where(s.isin(top), "OTHER")


def build(paths: list[Path], variant: str = "v1", n_rows: int = 60000, seed: int = 42) -> tuple[pd.DataFrame, dict]:
    if variant not in VARIANTS:
        raise ValueError(variant)
    t = load_year_files(paths)
    data, vax, sym = t["DATA"], t["VAX"], t["SYMPTOMS"]
    counts = {"raw_reports": int(data.VAERS_ID.nunique())}
    data = data.drop_duplicates("VAERS_ID")

    # Manufacturer reports out (see module docstring).
    manufacturer = data.get("SPLTTYPE", pd.Series("", index=data.index)).fillna("").str.strip() != ""
    data = data[~manufacturer].copy()
    counts["after_dropping_manufacturer_reports"] = len(data)

    # Vaccines: first listed row = primary vaccine; count all.
    vax = vax.copy()
    vax["_order"] = np.arange(len(vax))
    first = vax.sort_values("_order").drop_duplicates("VAERS_ID").set_index("VAERS_ID")
    n_vax = vax.groupby("VAERS_ID").size()
    data = data.set_index("VAERS_ID")
    data = data.join(first[["VAX_TYPE", "VAX_MANU", "VAX_LOT", "VAX_DOSE_SERIES", "VAX_ROUTE", "VAX_SITE"]], how="inner")
    data["n_vaccines"] = n_vax.reindex(data.index).astype(float)
    counts["with_vaccine_record"] = len(data)

    symptoms, removed_terms = symptom_text(sym)  # dict: outcome / care_process / administration
    raw_terms = sym.melt(id_vars="VAERS_ID", value_vars=[c for c in sym.columns if re.fullmatch(r"SYMPTOM\d+", c)],
                         value_name="t").dropna(subset=["t"])
    raw_terms = raw_terms[raw_terms.t.str.strip() != ""]
    data["_coded_terms"] = raw_terms.groupby("VAERS_ID").t.agg(lambda x: "; ".join(sorted(set(x)))).reindex(data.index)

    # Scope: reactions to the vaccine (see OUT_OF_SCOPE_TERM / MAX_ONSET_DAYS).
    infection = data["_coded_terms"].fillna("").str.contains(OUT_OF_SCOPE_TERM)
    onset_num = pd.to_numeric(data.NUMDAYS, errors="coerce")
    late = onset_num > MAX_ONSET_DAYS
    negative = onset_num < 0  # onset before vaccination: a data error, not "unknown"
    counts["removed_infection_or_failure_reports"] = int(infection.sum())
    counts["removed_onset_over_42_days"] = int((late & ~infection).sum())
    counts["removed_negative_onset"] = int((negative & ~infection).sum())
    data = data[~infection & ~late & ~negative]
    counts["after_scope_vaccine_reactions"] = len(data)

    # Likely secondary reports of the same event: same vaccination/onset dates, patient, vaccine lot AND the same
    # coded terms (two different patients rarely match on all of these). Keep the earliest VAERS_ID; count clusters
    # whose seriousness label disagrees, for the audit.
    key = ["VAX_DATE", "ONSET_DATE", "AGE_YRS", "SEX", "STATE", "VAX_TYPE", "VAX_LOT", "_coded_terms"]
    data["_id_num"] = pd.to_numeric(data.index, errors="coerce")
    keyed = data[key].notna().all(axis=1)
    flags_present = [c for c in OUTCOME_FLAGS if c in data.columns]
    data["_serious"] = data[flags_present].fillna("").eq("Y").any(axis=1)
    clusters = data[keyed].groupby(key, dropna=False)
    sizes = clusters.size()
    counts["dedup_clusters_merged"] = int((sizes > 1).sum())
    counts["dedup_reports_removed"] = int((sizes - 1).clip(lower=0).sum())
    counts["dedup_clusters_with_label_conflict"] = int((clusters._serious.nunique() > 1).sum())
    dup = keyed & data.sort_values("_id_num").duplicated(subset=key).reindex(data.index)
    data = data[~dup]
    counts["after_secondary_dedup"] = len(data)
    out = pd.DataFrame(index=data.index)
    out["report_id"] = data.index.astype(str)
    flags = [c for c in OUTCOME_FLAGS if c in data.columns]  # BIRTH_DEFECT exists only on the VAERS 2 form.
    out["serious"] = data[flags].fillna("").eq("Y").any(axis=1).astype(int)
    age = pd.to_numeric(data.AGE_YRS, errors="coerce").fillna(pd.to_numeric(data.get("CAGE_YR"), errors="coerce"))
    out["age_years"] = age.where((age >= 0) & (age <= 120))
    out["sex"] = data.SEX.fillna("U").replace("", "U")
    out["state"] = cap_categories(data.STATE, 60)
    onset = pd.to_numeric(data.NUMDAYS, errors="coerce")
    out["onset_days"] = onset.where((onset >= 0) & (onset <= 3650))
    out["vax_type"] = cap_categories(data.VAX_TYPE)
    out["vax_manu"] = cap_categories(data.VAX_MANU)
    out["vax_dose_series"] = cap_categories(data.VAX_DOSE_SERIES)
    out["vax_route"] = cap_categories(data.VAX_ROUTE)
    out["vax_site"] = cap_categories(data.VAX_SITE)
    out["n_vaccines"] = data.n_vaccines
    out["admin_by"] = cap_categories(data.V_ADMINBY)
    out["report_year"] = pd.to_datetime(data.RECVDATE, errors="coerce").dt.year.astype(float)
    out["symptoms"] = symptoms.reindex(out.index).fillna(EMPTY_TEXT)
    masked = {}
    for col, src in TEXT_SOURCES.items():
        raw = data[src].map(clean_text) if src in data.columns else pd.Series(EMPTY_TEXT, index=data.index)
        out[col] = raw.map(mask_outcome_sentences)
        masked[col] = int((out[col] != raw).sum())
    counts["background_cells_with_outcome_text_removed"] = masked
    # Masking also removes legitimate past history; report how often per label, so any imbalance is visible.
    serious_now = data[[c for c in OUTCOME_FLAGS if c in data.columns]].fillna("").eq("Y").any(axis=1)
    any_masked = pd.Series(False, index=out.index)
    for col, src in TEXT_SOURCES.items():
        if src in data.columns:
            any_masked |= out[col] != data[src].map(clean_text)
    counts["background_masked_share_by_label"] = {
        "serious": round(float(any_masked[serious_now].mean()), 4) if serious_now.any() else None,
        "not_serious": round(float(any_masked[~serious_now].mean()), 4) if (~serious_now).any() else None}
    text_cols = ["symptoms", *TEXT_SOURCES]
    if variant == "v3":
        # One richer cell per patient instead of four mostly-empty ones (57-86% "none reported" in v1),
        # so target-aware fine-tuning sees real background text instead of the same filler phrase.
        out["patient_background"] = out.apply(merge_background, axis=1)
        out = out.drop(columns=list(TEXT_SOURCES))
        text_cols = ["symptoms", "patient_background"]
    if variant in ("v6", "v7"):
        out["narrative_onset"] = data.SYMPTOM_TEXT.map(onset_narrative)
        counts["narrative_onset_empty"] = int((out.narrative_onset == EMPTY_TEXT).sum())
        sev = data[[c for c in OUTCOME_FLAGS if c in data.columns]].fillna("").eq("Y").any(axis=1)
        empty = out.narrative_onset == EMPTY_TEXT
        counts["narrative_onset_empty_share_by_label"] = {"serious": round(float(empty[sev].mean()), 4),
                                                          "not_serious": round(float(empty[~sev].mean()), 4)}
        text_cols = (["symptoms"] if variant == "v6" else []) + [*TEXT_SOURCES, "narrative_onset"]
    if variant in ("v4", "v5"):
        raw = data.SYMPTOM_TEXT.map(clean_text)
        out["narrative"] = data.SYMPTOM_TEXT.map(mask_narrative)
        counts["narrative_changed_by_masking"] = int((out.narrative != raw).sum())
        counts["narrative_empty_after_masking"] = int((out.narrative == EMPTY_TEXT).sum())
        text_cols = (["symptoms", *TEXT_SOURCES, "narrative"] if variant == "v4" else [*TEXT_SOURCES, "narrative"])

    # Same row scope for every variant: a report needs at least one coded reaction term (even in v5, where the
    # coded terms are not given to the model), so all variants describe the same population.
    out = out[out.symptoms != EMPTY_TEXT]
    counts["with_symptom_terms"] = len(out)
    # Population is defined WITHOUT reference to the variant, so every variant has the same report IDs:
    # exact duplicates are judged on all source fields, and the sample is a fixed hash of the report ID.
    source_fields = data.reindex(out.index)[[c for c in ["AGE_YRS", "SEX", "STATE", "NUMDAYS", "VAX_TYPE", "VAX_MANU",
                                             "VAX_DOSE_SERIES", "VAX_ROUTE", "VAX_SITE", "V_ADMINBY", "RECVDATE",
                                             "VAX_DATE", "ONSET_DATE", "VAX_LOT", "n_vaccines",
                                             "SYMPTOM_TEXT", *TEXT_SOURCES.values(), "_coded_terms"]
                                            if c in data.columns]].fillna("")
    out = out[~source_fields.duplicated()]
    counts["after_exact_duplicate_rows"] = len(out)
    if len(out) > n_rows:  # Deterministic, variant-independent sample: smallest hash(seed, report_id).
        key = out.report_id.map(lambda r: hashlib.sha256(f"{seed}:{r}".encode()).hexdigest())
        out = out.loc[key.sort_values().index[:n_rows]]
    out = out.sort_values("report_id", key=lambda s: pd.to_numeric(s, errors="coerce")).reset_index(drop=True)
    counts["final_rows"] = len(out)
    out = out[STRUCTURED + text_cols + ["serious", "report_id"]]
    counts["outcome_mentions_left"] = outcome_mentions(out, text_cols)
    vocab = {c: sorted(out[c].dropna().astype(str).unique().tolist()) for c in STRUCTURED if c not in NUMERIC}
    meta = {"counts": counts, "removed_terms": removed_terms, "sources": t["sources"], "text_columns": text_cols,
            "category_vocabulary": vocab}
    return out, meta


def write(out: pd.DataFrame, meta: dict, out_dir: Path, variant: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv = out_dir / f"vaers_{variant}.csv"
    out.to_csv(csv, index=False)
    schema = {
        "dataset_id": f"BIN_TEXT_VAERS_SERIOUS_{variant.upper()}",
        "build_version": BUILD_VERSION,
        "purpose": "candidate_dataset",
        "task": "binary",
        "target": "serious",
        "target_definition": "1 if any of " + ", ".join(OUTCOME_FLAGS) + " == 'Y' in VAERSDATA",
        "id_column": "report_id",
        "structured_columns": STRUCTURED,
        "text_columns": meta["text_columns"],
        "audit_columns": ["report_id"],
        "numeric_columns": NUMERIC,
        "csv_sha256": hashlib.sha256(csv.read_bytes()).hexdigest(),
        "rows": len(out),
        "positive_rate": round(float(out.serious.mean()), 4),
        "counts": meta["counts"],
        "excluded_fields": ["SYMPTOM_TEXT (v1, v3; masked narrative in v4, v5)", "LAB_DATA", "ER_VISIT", "ER_ED_VISIT", "OFC_VISIT", "HOSPDAYS", "RECOVD",
                            "DATEDIED", "RECVDATE/VAX_DATE/ONSET_DATE (raw dates)", "SPLTTYPE", "FORM_VERS", "VAX_LOT",
                            "VAX_NAME", "PRIOR_VAX", "TODAYS_DATE", "RPT_DATE", "V_FUNDBY"],
        "removed_terms": meta["removed_terms"],
        "category_vocabulary": meta["category_vocabulary"],
        "category_grouping": f"Label-free dataset definition: per categorical column, the {MAX_CATEGORIES - 1} most "
                             "frequent values over the source reports are kept and the rest become OTHER (state: 59). "
                             "Fixed once when the dataset is built, before any benchmark split; no labels used.",
        "task_framing": "Retrospective classification of recorded report seriousness from restricted report "
                        "contents. Not a forecast at vaccination time and not evidence about vaccine causation.",
        "population": f"Domestic, non-manufacturer reports of reactions with onset 0-{MAX_ONSET_DAYS} days after "
                      "vaccination (or unknown onset); infection/breakthrough and vaccine-failure reports excluded.",
        "scope": "Reports with at least one reaction term left after removing outcome, care-process and "
                 "vaccination-process terms (pure administration-error reports are out of scope).",
        "sources": meta["sources"],
        "source_url": "https://vaers.hhs.gov/data/datasets.html",
        "source_terms": "U.S. government public data (CDC/FDA). Reports are unverified and do not establish causality.",
        "variant": variant,
    }
    (out_dir / "schema.json").write_text(json.dumps(schema, indent=2) + "\n")
    return csv


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", type=Path, help="Yearly VAERS ZIPs or CSVs")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--variant", choices=VARIANTS, default="v1")
    ap.add_argument("--rows", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    out, meta = build(args.inputs, args.variant, args.rows, args.seed)
    csv = write(out, meta, args.out_dir, args.variant)
    print(json.dumps({"csv": str(csv), **meta["counts"], "positive_rate": round(float(out.serious.mean()), 4),
                      "removed_terms": {k: len(v) for k, v in meta["removed_terms"].items()}}, indent=2))


if __name__ == "__main__":
    main()
