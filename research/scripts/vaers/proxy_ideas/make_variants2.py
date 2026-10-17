"""Round 2: matching variants and separate background columns (development only)."""
import sys
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from make_variants import (OUT, W, bv, rv, raw_table, as_str, matched, age_band, save, RAW)  # noqa: E402


def onset_band(d):
    try:
        d = float(d)
    except ValueError:
        return "unk"
    return "0" if d < 1 else "1-2" if d < 3 else "3-7" if d < 8 else "8-42"


def full_cohort_split_background(raw):
    """Full cohort with care-boundary-2, keeping the four background fields as separate columns."""
    captured = []
    orig = bv.merge_background

    def capture(fields):
        captured.append(dict(fields))
        return orig(fields)

    bv.ENRICHED_POSITIVE_SHARE = 0.01
    out, _ = bv.build(RAW, "v9", 60000, 42)
    parent = as_str(out)
    bv.merge_background = capture
    try:
        corrected, ledger = rv.refine(parent, raw)
    finally:
        bv.merge_background = orig
    sep = pd.DataFrame(captured, index=corrected.index)
    corrected = pd.concat([corrected.drop(columns=["patient_background"]), sep], axis=1)
    return corrected[ledger.eligible.to_numpy()].reset_index(drop=True)


if __name__ == "__main__":
    full3 = pd.read_csv(OUT / "full3.csv", keep_default_na=False, dtype=str)
    v10 = pd.read_csv(OUT / "v10.csv", keep_default_na=False, dtype=str)
    base = full3.vax_type + "|" + full3.report_year + "|" + full3.age_years.map(age_band)
    save(matched(full3, base + "|" + full3.sex, name="A2_plus_sex"), "A2_plus_sex")
    save(matched(full3, base + "|" + full3.onset_days.map(onset_band), name="A3_plus_onset"), "A3_plus_onset")
    save(matched(full3, full3.vax_type + "|" + full3.report_year, name="A4_vax_year"), "A4_vax_year")

    raw = raw_table()
    split = full_cohort_split_background(raw)
    cols = [c for c in v10.columns if c != "patient_background"]
    g = split[split.report_id.isin(set(v10.report_id))]
    g = g[[c for c in cols if c not in ("serious", "report_id")] + list(bv.TEXT_SOURCES) + ["serious", "report_id"]]
    print("G rows", len(g), "same ids as v10:", set(g.report_id) == set(v10.report_id))
    save(g, "G_split_background")
    am = pd.read_csv(OUT / "A_matched.csv", keep_default_na=False, dtype=str)
    ag = split[split.report_id.isin(set(am.report_id))][g.columns]
    print("AG rows", len(ag), "of", len(am))
    save(ag, "AG_matched_split")
