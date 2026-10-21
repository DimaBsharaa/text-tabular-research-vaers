# VAERS v11b — the dataset

| | |
|---|---|
| File | `vaers_v11b.csv` (UTF-8, comma-separated, header row) |
| Rows | 18,859 reports · 4,900 serious (26.0%) |
| Columns | 12 structured + 2 text + target + `report_id` |
| sha256 | `8e59f09b19d0f7f174b38413ffe48052b005491fa1464877764d405911e16690` |
| MulTaBench name | `BIN_TEXT_VAERS_SERIOUS_V11B` (binary classification, metric AUROC) |
| Source | [VAERS](https://vaers.hhs.gov/data/datasets.html) yearly files 2019, 2023, 2024 — public U.S. government data |

## Columns

| Column | Type | Description |
|---|---|---|
| `age_years` | numeric | Age at vaccination (empty if unknown) |
| `sex` | categorical | F, M or U |
| `state` | categorical | U.S. state or territory; UNKNOWN if missing |
| `onset_days` | numeric | Days from vaccination to first symptom (empty if unknown) |
| `vax_type` | categorical | Type of the first listed vaccine (e.g. COVID19, VARZOS = shingles, FLU4) |
| `vax_manu` | categorical | Manufacturer |
| `vax_dose_series` | categorical | Dose number in the series (UNK if unknown) |
| `vax_route` | categorical | Route code (IM intramuscular, SC subcutaneous, SYR syringe, …) |
| `vax_site` | categorical | Injection site code (LA left arm, RA right arm, …) |
| `n_vaccines` | numeric | Number of vaccines given at the visit |
| `admin_by` | categorical | Facility type that gave the vaccine (PHM pharmacy, PVT private practice, PUB public clinic, …) |
| `report_year` | numeric | Year the report was received |
| `patient_background` | text | Medical history, allergies, current medications and illness at vaccination, merged into one field; "none reported" if all empty |
| `narrative_onset` | text | Opening of the reporter's symptom description, up to 3 sentences, cut before any care, test, treatment or outcome |
| `serious` | target | 1 if any outcome flag is set: died, life-threatening, hospitalised, prolonged stay, disability, birth defect |
| `report_id` | audit | VAERS ID, dropped by the benchmark recipe |

The label flags themselves, dates, lab data, emergency/office visits, hospital days, recovery status and coded symptom terms
are **not** included, because each of them reveals or is recorded after the outcome. Rare categorical values (beyond the 98
most frequent per column) are grouped as OTHER, decided without the label. `schema.json` lists every excluded
field, every removed term and the exact sampling rule.

## How it is built

Run [`research/scripts/vaers/build_v11b.py`](../../scripts/vaers/build_v11b.py) on the raw files in
[`../vaers_raw/`](../vaers_raw/README.md); it reproduces this file byte for byte and checks both hashes.

1. **Full cohort** (`build_vaers.py`, `refine_v9.py`): domestic, non-manufacturer reports of reactions with onset 0–42 days (or unknown),
   duplicates merged, outcome and care terms removed, narrative cut to its opening and cleaned of care text: 62,754 reports.
2. **Matched controls (v11a)**: every serious report plus up to 3 non-serious reports in the same stratum of
   vaccine type × report year × age band × sex, chosen by a fixed hash of the report ID. Uses no text. 18,863 reports,
   sha256 `7b05de42…`.
3. **Care cleanup 3 (v11b)**: [`care_boundary_3_edits.csv`](care_boundary_3_edits.csv) records each edit — 503 excerpts
   cut to a shorter prefix (removing medication, treatment and procedure phrases such as "Gave TYLENOL…", "resolved with
   prednisone", "IVIG") and 4 reports removed because nothing informative was left. Edits only shorten text, never add or
   rewrite it, and were decided from the text alone, without the label. The rule script that produced them was lost with a
   deleted branch; the ledger stores its exact output and the build checks every row against it.

Notebook [02](../../notebooks/02_vaers_eda_and_preprocessing.ipynb) runs these steps and shows the evidence behind each rule.
