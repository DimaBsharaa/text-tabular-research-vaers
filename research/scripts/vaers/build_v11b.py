"""Build the submitted VAERS dataset (v11b) from the official raw VAERS files, end to end.

    python -m research.scripts.vaers.build_v11b 2019VAERSData.zip 2023VAERSData.zip 2024VAERSData.zip \
        --out-dir out/

Steps (each step's output hash is checked against the frozen value):
  1. Full cohort: every eligible report from the three yearly files, built with the frozen build-8 text
     policy (build_vaers.py, onset excerpt capped at 3 sentences) and refined with care-boundary-2
     (refine_v9.py). No enrichment: every non-serious report is kept at this stage.
  2. v11a, matched controls: every serious report plus up to 3 non-serious reports per serious report
     within each stratum vax_type x report_year x age band x sex; non-serious reports are chosen by the
     smallest sha256("42:" + report_id). Uses only the label and these structured fields, never text.
  3. v11b, care-boundary-3: applies the recorded edit ledger research/data/vaers_v11b/care_boundary_3_edits.csv.
     503 symptom excerpts are cut to a literal prefix (removing named medications, clinical care and
     procedures, e.g. "Gave TYLENOL ...", "resolved with prednisone", "IVIG", "NICU"), and the 4 reports
     whose excerpt has no informative content left are excluded. Nothing is added or rewritten; the
     ledger stores, per report, how many leading characters are kept and the exact removed text.
     The ledger is label-blind (decided from text only) and is checked row by row before it is applied.

Provenance note: the original care-boundary-3 rule script was lost with a deleted branch. Its exact
output was preserved, and the ledger reproduces it byte for byte (verified by the final hash).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import pandas as pd

from research.scripts.vaers import build_vaers as bv
from research.scripts.vaers import refine_v9 as rv

HERE = Path(__file__).resolve().parents[2]
LEDGER = HERE / "data" / "vaers_v11b" / "care_boundary_3_edits.csv"
SHA_V11A = "7b05de4288beb4df0e07dbdc8d9250c4a7d99cc0d8ad61be27f47ced56f73dab"
SHA_V11B = "8e59f09b19d0f7f174b38413ffe48052b005491fa1464877764d405911e16690"


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode()


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def as_str(df: pd.DataFrame) -> pd.DataFrame:
    d = pd.read_csv(io.StringIO(df.to_csv(index=False)), keep_default_na=False, dtype=str)
    d.report_id = d.report_id.map(rv.canonical_id)
    return d


def full_cohort(raw_zips: list[Path]) -> pd.DataFrame:
    tables = bv.load_year_files(raw_zips)
    raw = tables["DATA"].drop_duplicates("VAERS_ID").copy()
    raw.VAERS_ID = raw.VAERS_ID.map(rv.canonical_id)
    raw = raw.set_index("VAERS_ID")
    bv.MAX_ONSET_SENTENCES = 3
    bv.ENRICHED_POSITIVE_SHARE = 0.01  # n_neg exceeds what exists, so every non-serious report is kept
    out, _ = bv.build(raw_zips, "v9", 60000, 42)
    corrected, ledger = rv.refine(as_str(out), raw)
    return corrected[ledger.eligible.to_numpy()].reset_index(drop=True)


def age_band(a: str) -> str:
    try:
        a = float(a)
    except ValueError:
        return "unk"
    return "0-17" if a < 18 else "18-49" if a < 50 else "50-64" if a < 65 else "65+"


def matched_controls(full: pd.DataFrame, ratio: int = 3) -> pd.DataFrame:
    full = full.copy()
    full["_s"] = full.vax_type + "|" + full.report_year + "|" + full.age_years.map(age_band) + "|" + full.sex
    full["_h"] = full.report_id.map(lambda rid: hashlib.sha256(f"42:{rid}".encode()).hexdigest())
    keep = []
    for _, g in full.groupby("_s"):
        pos = g[g.serious == "1"]
        keep += [pos, g[g.serious == "0"].sort_values("_h").iloc[: ratio * len(pos)]]
    d = pd.concat(keep).drop(columns=["_s", "_h"])
    d = d.sort_values("report_id", key=lambda s: pd.to_numeric(s, errors="coerce"))
    return d.reset_index(drop=True)


def apply_care_boundary_3(v11a: pd.DataFrame, ledger: pd.DataFrame) -> pd.DataFrame:
    d = v11a.set_index("report_id", drop=False).copy()
    for row in ledger.itertuples(index=False):
        old = d.at[row.report_id, "narrative_onset"]
        if row.action == "exclude":
            assert old == row.removed_text, f"ledger mismatch for excluded report {row.report_id}"
            continue
        k = int(row.kept_chars)
        assert old[k:] == row.removed_text, f"ledger mismatch for report {row.report_id}"
        d.at[row.report_id, "narrative_onset"] = old[:k]
    excluded = set(ledger.loc[ledger.action == "exclude", "report_id"])
    return d[~d.report_id.isin(excluded)].reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raw_zips", nargs=3, type=Path, help="2019, 2023 and 2024 VAERSData.zip from vaers.hhs.gov")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    full = full_cohort(sorted(args.raw_zips))
    print(f"1. full cohort: {len(full)} reports")

    v11a = matched_controls(full)
    b = csv_bytes(v11a)
    print(f"2. v11a: {len(v11a)} reports, sha256 {sha(b)}")
    assert sha(b) == SHA_V11A, "v11a does not match the frozen file"
    v11a = pd.read_csv(io.BytesIO(b), keep_default_na=False, dtype=str)

    ledger = pd.read_csv(LEDGER, keep_default_na=False, dtype=str)
    v11b = apply_care_boundary_3(v11a, ledger)
    b = csv_bytes(v11b)
    print(f"3. v11b: {len(v11b)} reports, {int((v11b.serious == '1').sum())} serious, sha256 {sha(b)}")
    assert sha(b) == SHA_V11B, "v11b does not match the frozen file"

    (args.out_dir / "vaers_v11b.csv").write_bytes(b)
    schema = json.loads((LEDGER.parent / "schema.json").read_text())
    (args.out_dir / "schema.json").write_text(json.dumps(schema, indent=1))
    print("OK: rebuilt the submitted dataset exactly ->", args.out_dir / "vaers_v11b.csv")


if __name__ == "__main__":
    main()
