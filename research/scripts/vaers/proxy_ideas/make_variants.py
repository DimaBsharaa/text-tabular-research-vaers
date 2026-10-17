"""Build the idea-test variants from the raw VAERS ZIPs (development only, not committed data).

Every variant reuses the frozen build-8 text policy and the care-boundary-2 refinement unchanged;
only the sampling design, the excerpt sentence cap or the text columns differ.
"""
import hashlib
import os
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from research.scripts.vaers import build_vaers as bv  # noqa: E402
from research.scripts.vaers import refine_v9 as rv  # noqa: E402

W = Path(os.environ.get("VAERS_WORK", "vaers_work"))
RAW = [W / "raw" / f"{y}VAERSData.zip" for y in (2019, 2023, 2024)]
OUT = W / "variants"
OUT.mkdir(exist_ok=True)
TEXT = ["patient_background", "narrative_onset"]


def as_str(df):
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    d = pd.read_csv(buf, keep_default_na=False, dtype=str)
    d.report_id = d.report_id.map(rv.canonical_id)
    return d


def raw_table():
    tables = bv.load_year_files(RAW)
    raw = tables["DATA"].drop_duplicates("VAERS_ID").copy()
    raw.VAERS_ID = raw.VAERS_ID.map(rv.canonical_id)
    return raw.set_index("VAERS_ID")


def full_cohort(max_sentences, raw):
    """All eligible reports (no enrichment), refined with care-boundary-2."""
    bv.MAX_ONSET_SENTENCES = max_sentences
    bv.ENRICHED_POSITIVE_SHARE = 0.01  # n_neg exceeds what exists, so every non-serious report is kept
    out, meta = bv.build(RAW, "v9", 60000, 42)
    parent = as_str(out)
    corrected, ledger = rv.refine(parent, raw)
    full = corrected[ledger.eligible.to_numpy()].reset_index(drop=True)
    bv.MAX_ONSET_SENTENCES = 3
    return full


def h(rid, salt="42"):
    return hashlib.sha256(f"{salt}:{rid}".encode()).hexdigest()


def age_band(a):
    try:
        a = float(a)
    except ValueError:
        return "unk"
    return "0-17" if a < 18 else "18-49" if a < 50 else "50-64" if a < 65 else "65+"


def matched(full, strata, ratio=3, name=""):
    """Frequency matching: in each stratum keep every serious report and up to `ratio` non-serious per serious
    (smallest hash first). Uses only the label and the stratum, never text."""
    full = full.copy()
    full["_s"] = strata
    full["_h"] = full.report_id.map(h)
    keep = []
    short = 0
    for _, g in full.groupby("_s"):
        pos = g[g.serious == "1"]
        neg = g[g.serious == "0"].sort_values("_h")
        want = ratio * len(pos)
        short += max(0, want - len(neg))
        keep.append(pos)
        keep.append(neg.iloc[:want])
    d = pd.concat(keep)
    d = d.drop(columns=["_s", "_h"]).sort_values("report_id", key=lambda s: pd.to_numeric(s, errors="coerce"))
    print(f"{name}: rows {len(d)}, serious share {d.serious.astype(int).mean():.3f}, controls short {short}")
    return d.reset_index(drop=True)


def save(d, name):
    d.to_csv(OUT / f"{name}.csv", index=False)
    return d


if __name__ == "__main__":
    raw = raw_table()
    v10 = pd.read_csv(W / "b9/vaers_v10/vaers_v10.csv", keep_default_na=False, dtype=str)
    v10.report_id = v10.report_id.map(rv.canonical_id)
    save(v10, "v10")

    full3 = full_cohort(3, raw)
    save(full3, "full3")
    # Sanity: the full cohort restricted to v10's IDs must reproduce v10 exactly.
    chk = full3.set_index("report_id").loc[v10.report_id].reset_index()[v10.columns]
    print("full3 reproduces v10 on its IDs:", chk.equals(v10))

    # B: excerpt only (no background column)
    save(v10.drop(columns=["patient_background"]), "B_excerpt_only")

    # C: up to 5 sentences (same cut rules), restricted to v10 IDs so rows match
    full5 = full_cohort(5, raw)
    c = full5[full5.report_id.isin(set(v10.report_id))]
    save(c, "C_long5")
    save(v10[v10.report_id.isin(set(c.report_id))], "C_base_same_rows")
    print("C rows", len(c), "excerpt changed", int((c.set_index("report_id").narrative_onset !=
                                                   v10.set_index("report_id").loc[c.report_id].narrative_onset).sum()))

    # D: 50% serious vs a same-size random 25% control
    pos = v10[v10.serious == "1"]
    neg = v10[v10.serious == "0"].assign(_h=lambda d: d.report_id.map(h)).sort_values("_h")
    save(pd.concat([pos, neg.iloc[:len(pos)].drop(columns="_h")]), "D_bal50")
    n = 2 * len(pos)
    rng = np.random.RandomState(7)
    pos_s = pos.sample(int(round(n * 0.25)), random_state=rng)
    neg_s = v10[v10.serious == "0"].sample(n - len(pos_s), random_state=rng)
    save(pd.concat([pos_s, neg_s]), "D_ctrl25_same_size")

    # A: matched on vaccine type x report year x age band
    strata_a = full3.vax_type + "|" + full3.report_year + "|" + full3.age_years.map(age_band)
    save(matched(full3, strata_a, name="A_matched"), "A_matched")

    # F: matched on the primary coded symptom (coded terms are used to choose rows only, never as inputs)
    v1, _ = bv.build(RAW, "v1", 10**9, 42)
    v1 = as_str(v1)
    sym = v1.set_index("report_id").symptoms.reindex(full3.report_id).fillna("none")
    terms = sym.str.lower().str.split(r"\s*;\s*")
    freq = pd.Series([t for ts in terms for t in set(ts)]).value_counts()
    # Most specific term the report carries (rarest term that still occurs in >= 50 reports): the same "topic"
    # for cases and controls, so only how the reaction is described can separate them.
    def specific(ts):
        ok = [t for t in set(ts) if freq.get(t, 0) >= 50]
        return min(ok, key=lambda t: (freq[t], t)) if ok else "other"
    first = terms.map(specific)
    print("F strata:", first.nunique(), "top:", first.value_counts().head(10).to_dict())
    (OUT / "F_strata.json").write_text(json.dumps(first.value_counts().to_dict()))
    save(matched(full3, first.to_numpy(), name="F_symptom_matched"), "F_symptom_matched")
    strata_af = strata_a.to_numpy() + "|" + first.to_numpy()
    save(matched(full3, strata_af, name="AF_matched"), "AF_matched")
    json.dump({"full3_rows": len(full3), "full5_rows": len(full5)}, open(OUT / "meta.json", "w"))
