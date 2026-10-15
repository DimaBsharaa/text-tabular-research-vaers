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
  * Variant v1 (default) excludes the free narrative SYMPTOM_TEXT. Variant v2 adds it with every sentence that
    mentions an outcome/care-setting keyword removed (an experimental, riskier variant).
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

VARIANTS = ("v1", "v2")
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
                          r"life threatening", re.I)
OUTCOME_TERM_KEEP = {"hyperventilation", "hypoventilation", "fear of death", "near death experience",
                     "death of relative", "hypertensive emergency"}
NARRATIVE_OUTCOME = re.compile(r"hospital|admit|admission|discharg|\bicu\b|intensive care|\ber\b|emergency|\bed\b|"
                               r"urgent care|died|death|dead|deceased|passed away|expired|autopsy|fatal|intubat|"
                               r"ventilat|life.?threat|resuscitat|\bcpr\b|disab|inpatient|ambulance|\bems\b|911|"
                               r"transferred|observation unit|telemetry|life support|hospice|code blue|serious", re.I)
STRUCTURED = ["age_years", "sex", "state", "onset_days", "vax_type", "vax_manu", "vax_dose_series", "vax_route",
              "vax_site", "n_vaccines", "admin_by", "report_year"]
NUMERIC = ["age_years", "onset_days", "n_vaccines", "report_year"]
MAX_CATEGORIES = 40  # Keeps categoricals below the official 100-distinct text detector threshold.


def _read(name: str, raw: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(raw), encoding="latin-1", dtype=str, keep_default_na=False, na_values=[""],
                       low_memory=False)


def load_year_files(paths: list[Path]) -> dict[str, pd.DataFrame]:
    """Concatenate DATA / VAX / SYMPTOMS tables from yearly ZIPs or CSVs."""
    parts = {"DATA": [], "VAX": [], "SYMPTOMS": []}
    sources = []
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
            parts[kind].append(_read(name, raw))
            sources.append({"file": f"{path.name}:{name}", "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)})
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


def symptom_text(symptoms: pd.DataFrame) -> tuple[pd.Series, list[str]]:
    cols = [c for c in symptoms.columns if re.fullmatch(r"SYMPTOM\d+", c)]
    long = symptoms.melt(id_vars="VAERS_ID", value_vars=cols, value_name="term").dropna(subset=["term"])
    long["term"] = long.term.str.strip()
    long = long[long.term != ""]
    is_outcome = long.term.str.contains(OUTCOME_TERM) & ~long.term.str.lower().isin(OUTCOME_TERM_KEEP)
    removed = sorted(long.loc[is_outcome, "term"].unique())
    kept = long[~is_outcome].drop_duplicates(["VAERS_ID", "term"])
    joined = kept.sort_values("term").groupby("VAERS_ID").term.agg("; ".join)
    return joined, removed


def mask_narrative(text) -> str:
    text = clean_text(text)
    if text == EMPTY_TEXT:
        return text
    sentences = re.split(r"(?<=[.!?;])\s+", text)
    kept = [s for s in sentences if not NARRATIVE_OUTCOME.search(s)]
    return " ".join(kept).strip() or EMPTY_TEXT


def cap_categories(s: pd.Series, k: int = MAX_CATEGORIES) -> pd.Series:
    s = s.fillna("UNKNOWN").astype(str).str.strip().replace("", "UNKNOWN").str.upper()
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

    # Likely secondary reports of the same event: keep the earliest VAERS_ID.
    key = ["VAX_DATE", "ONSET_DATE", "AGE_YRS", "SEX", "STATE", "VAX_TYPE", "VAX_LOT"]
    data["_id_num"] = pd.to_numeric(data.index, errors="coerce")
    keyed = data[key].notna().all(axis=1)
    dup = keyed & data.sort_values("_id_num").duplicated(subset=key).reindex(data.index)
    data = data[~dup]
    counts["after_secondary_dedup"] = len(data)

    symptoms, removed_terms = symptom_text(sym)
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
    for col, src in TEXT_SOURCES.items():
        out[col] = data[src].map(clean_text) if src in data.columns else EMPTY_TEXT
    text_cols = ["symptoms", *TEXT_SOURCES]
    if variant == "v2":
        out["narrative_masked"] = data.SYMPTOM_TEXT.map(mask_narrative)
        text_cols.append("narrative_masked")

    out = out[out.symptoms != EMPTY_TEXT]  # A report with no reaction term left carries no event description.
    counts["with_symptom_terms"] = len(out)
    pred = STRUCTURED + text_cols
    out = out.drop_duplicates(subset=pred + ["serious"])
    counts["after_exact_duplicate_rows"] = len(out)
    if len(out) > n_rows:  # Uniform random sample; the benchmark samples train/test from this per fold.
        out = out.sample(n=n_rows, random_state=seed)
    out = out.sort_values("report_id", key=lambda s: pd.to_numeric(s, errors="coerce")).reset_index(drop=True)
    counts["final_rows"] = len(out)
    out = out[STRUCTURED + text_cols + ["serious", "report_id"]]
    meta = {"counts": counts, "removed_outcome_terms": removed_terms, "sources": t["sources"], "text_columns": text_cols}
    return out, meta


def write(out: pd.DataFrame, meta: dict, out_dir: Path, variant: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv = out_dir / f"vaers_{variant}.csv"
    out.to_csv(csv, index=False)
    schema = {
        "dataset_id": f"BIN_TEXT_VAERS_SERIOUS_{variant.upper()}",
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
        "excluded_fields": ["SYMPTOM_TEXT (v1)", "LAB_DATA", "ER_VISIT", "ER_ED_VISIT", "OFC_VISIT", "HOSPDAYS", "RECOVD",
                            "DATEDIED", "RECVDATE/VAX_DATE/ONSET_DATE (raw dates)", "SPLTTYPE", "FORM_VERS", "VAX_LOT",
                            "VAX_NAME", "PRIOR_VAX", "TODAYS_DATE", "RPT_DATE", "V_FUNDBY"],
        "removed_outcome_terms": meta["removed_outcome_terms"],
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
                      "removed_outcome_terms": len(meta["removed_outcome_terms"])}, indent=2))


if __name__ == "__main__":
    main()
