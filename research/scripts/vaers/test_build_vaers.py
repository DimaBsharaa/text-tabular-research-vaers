"""Offline tests for the VAERS candidate builder on a synthetic fixture in the official file format.

Run: python3 -m unittest research.scripts.vaers.test_build_vaers -v
"""
import io
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from research.scripts.vaers import build_vaers as bv  # noqa: E402
from evaluation.colab_runner import load_csv  # noqa: E402

DATA_COLS = ["VAERS_ID", "RECVDATE", "STATE", "AGE_YRS", "CAGE_YR", "CAGE_MO", "SEX", "RPT_DATE", "SYMPTOM_TEXT", "DIED",
             "DATEDIED", "L_THREAT", "ER_VISIT", "HOSPITAL", "HOSPDAYS", "X_STAY", "DISABLE", "RECOVD", "VAX_DATE",
             "ONSET_DATE", "NUMDAYS", "LAB_DATA", "V_ADMINBY", "V_FUNDBY", "OTHER_MEDS", "CUR_ILL", "HISTORY",
             "PRIOR_VAX", "SPLTTYPE", "FORM_VERS", "TODAYS_DATE", "BIRTH_DEFECT", "OFC_VISIT", "ER_ED_VISIT", "ALLERGIES"]
TERMS_MILD = ["Injection site pain", "Pyrexia", "Headache", "Fatigue", "Chills", "Myalgia", "Nausea", "Rash"]
TERMS_SEVERE = ["Myocarditis", "Pulmonary embolism", "Guillain-Barre syndrome", "Anaphylactic reaction", "Seizure"]
HIST = ["Hypertension", "Type 2 diabetes", "Asthma", "COPD, CHF", "None", "N/A", "", "Obesity; hyperlipidemia",
        "Atrial fibrillation on warfarin", "no known allergies", "Breast cancer 2015", "Depression"]


def fixture(n=2400, seed=0, year=2023):
    rng = np.random.default_rng(seed)
    rows, vax, sym = [], [], []
    for i in range(n):
        vid = str(1_000_000 * (year - 2000) + i)
        severe = rng.random() < 0.15
        age = int(rng.integers(1, 95))
        terms = list(rng.choice(TERMS_SEVERE if severe else TERMS_MILD, size=int(rng.integers(1, 4)), replace=False))
        if severe and rng.random() < 0.5:
            terms += ["Hospitalisation", "Intensive care"] if rng.random() < .5 else ["Death"]
        hist = HIST[int(rng.integers(len(HIST)))]
        rows.append({c: "" for c in DATA_COLS} | {
            "VAERS_ID": vid, "RECVDATE": f"03/{1 + i % 28:02d}/{year}", "STATE": ["CA", "NY", "TX", "FL", ""][i % 5],
            "AGE_YRS": str(age), "SEX": "MF"[i % 2],
            "SYMPTOM_TEXT": (f"Patient {i} developed {terms[0].lower()} two days after dose. "
                             + ("She was admitted to the hospital and treated. " if severe else "")
                             + "Symptoms resolved with rest. Café visit noted."),
            "HOSPITAL": "Y" if severe else "", "DIED": "Y" if severe and "Death" in terms else "",
            "RECOVD": "N" if severe else "Y", "HOSPDAYS": "4" if severe else "",
            "VAX_DATE": f"02/{1 + i % 28:02d}/{year}", "ONSET_DATE": f"02/{3 + i % 25:02d}/{year}",
            "NUMDAYS": str(int(rng.integers(0, 30))), "V_ADMINBY": ["PHM", "PVT", "PUB", "OTH"][i % 4],
            "OTHER_MEDS": ["lisinopril 10mg", "metformin", "None", "", f"med {i % 150}"][i % 5],
            "CUR_ILL": ["None", "cold", "", f"illness {i % 130}"][i % 4],
            "HISTORY": hist if i % 7 else f"{hist}; detail {i % 120}",
            "ALLERGIES": ["NKDA", "penicillin", "", f"allergen {i % 140}"][i % 4],
            "SPLTTYPE": "USPFIZER INC2023" if i % 50 == 0 else "", "FORM_VERS": "2"})
        vax.append({"VAERS_ID": vid, "VAX_TYPE": ["COVID19", "FLU4", "VARZOS", "RSV"][i % 4], "VAX_MANU": "PFIZER\\BIONTECH",
                    "VAX_LOT": f"L{i}", "VAX_DOSE_SERIES": str(1 + i % 3), "VAX_ROUTE": "IM", "VAX_SITE": "LA",
                    "VAX_NAME": "x"})
        if i % 10 == 0:
            vax.append({"VAERS_ID": vid, "VAX_TYPE": "FLU4", "VAX_MANU": "SANOFI", "VAX_LOT": "Z", "VAX_DOSE_SERIES": "1",
                        "VAX_ROUTE": "IM", "VAX_SITE": "RA", "VAX_NAME": "y"})
        rec = {"VAERS_ID": vid}
        for k in range(5):
            rec[f"SYMPTOM{k + 1}"] = terms[k] if k < len(terms) else ""
            rec[f"SYMPTOMVERSION{k + 1}"] = "26.1" if k < len(terms) else ""
        sym.append(rec)
    # One secondary report duplicating report 1 (same event keys, later ID, different text).
    dup = dict(rows[1]); dup["VAERS_ID"] = str(int(rows[1]["VAERS_ID"]) + 900_000); dup["HISTORY"] = "secondary copy"
    rows.append(dup)
    vax += [dict(v) | {"VAERS_ID": dup["VAERS_ID"]} for v in vax if v["VAERS_ID"] == rows[1]["VAERS_ID"]]
    sym += [dict(r) | {"VAERS_ID": dup["VAERS_ID"]} for r in sym if r["VAERS_ID"] == rows[1]["VAERS_ID"]]
    return pd.DataFrame(rows), pd.DataFrame(vax), pd.DataFrame(sym)


def write_zip(path, year=2023, **kw):
    data, vax, sym = fixture(year=year, **kw)
    with zipfile.ZipFile(path, "w") as zf:
        for name, df in [("VAERSDATA", data), ("VAERSVAX", vax), ("VAERSSYMPTOMS", sym)]:
            buf = io.BytesIO(); df.to_csv(buf, index=False, encoding="latin-1"); zf.writestr(f"{year}{name}.csv", buf.getvalue())
    return data, vax, sym


class BuildVaersTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.zip = cls.tmp / "2023VAERSData.zip"
        cls.raw = write_zip(cls.zip)
        cls.out, cls.meta = bv.build([cls.zip], "v1", n_rows=100000)

    def test_label_matches_outcome_flags(self):
        data = self.raw[0].set_index("VAERS_ID")
        expected = data[bv.OUTCOME_FLAGS[:-1]].eq("Y").any(axis=1).astype(int)
        got = self.out.set_index("report_id").serious
        self.assertTrue((expected.reindex(got.index) == got).all())
        self.assertTrue(0 < got.mean() < 1)

    def test_no_outcome_or_post_event_field_in_predictors(self):
        predictors = set(bv.STRUCTURED) | set(self.meta["text_columns"])
        forbidden = {c.lower() for c in bv.OUTCOME_FLAGS + ["HOSPDAYS", "RECOVD", "LAB_DATA", "ER_ED_VISIT", "OFC_VISIT",
                                                            "SYMPTOM_TEXT", "DATEDIED", "SPLTTYPE"]}
        self.assertFalse(predictors & forbidden)
        self.assertNotIn("report_id", predictors)

    def test_outcome_terms_removed_from_symptoms(self):
        for t in ["Death", "Hospitalisation", "Intensive care"]:
            self.assertFalse(self.out.symptoms.str.contains(t, regex=False).any(), t)
        self.assertIn("Death", self.meta["removed_terms"]["outcome"])
        self.assertTrue(self.out.symptoms.str.contains("Myocarditis|Pyrexia").any())

    def test_keep_list_terms_are_not_removed(self):
        s = pd.DataFrame({"VAERS_ID": ["1", "2"], "SYMPTOM1": ["Hyperventilation", "Sudden death"]})
        joined, removed = bv.symptom_text(s)
        self.assertEqual(joined.get("1"), "Hyperventilation")
        self.assertEqual(removed["outcome"], ["Sudden death"])

    def test_care_and_administration_terms_removed_findings_kept(self):
        s = pd.DataFrame({"VAERS_ID": ["1", "1", "1", "1", "2", "2", "3", "3", "4"],
                          "SYMPTOM1": ["Blood test", "Troponin increased", "Computerised tomogram", "Anticoagulant therapy",
                                       "No adverse event", "Expired product administered",
                                       "Productive cough", "Electrocardiogram ST segment elevation",
                                       "Administration site pain"]})
        joined, removed = bv.symptom_text(s)
        self.assertEqual(joined.get("1"), "Troponin increased")
        self.assertNotIn("2", joined.index)  # pure administration report -> no reaction term -> dropped later
        self.assertEqual(joined.get("3"), "Electrocardiogram ST segment elevation; Productive cough")
        self.assertEqual(joined.get("4"), "Administration site pain")  # a real local reaction, not a process term
        self.assertEqual(removed["care_process"], ["Anticoagulant therapy", "Blood test", "Computerised tomogram"])
        self.assertEqual(removed["administration"], ["Expired product administered", "No adverse event"])

    def test_manufacturer_reports_and_secondary_duplicates_dropped(self):
        ids = set(self.out.report_id)
        self.assertFalse(ids & {str(23_000_000 + i) for i in range(0, 2400, 50)})  # SPLTTYPE rows
        self.assertNotIn(str(23_000_001 + 900_000), ids)  # later secondary copy collapsed
        self.assertIn(str(23_000_001), ids)
        self.assertEqual(self.meta["counts"]["after_secondary_dedup"],
                         self.meta["counts"]["with_vaccine_record"] - 1)

    def test_null_like_text_normalised_and_nonempty(self):
        for col in self.meta["text_columns"]:
            self.assertFalse(self.out[col].isna().any() or self.out[col].str.strip().eq("").any(), col)
            self.assertFalse(self.out[col].isin(["None", "N/A", "NKDA", "no known allergies"]).any(), col)
        self.assertEqual(bv.clean_text("  None. "), bv.EMPTY_TEXT)
        self.assertEqual(bv.clean_text("Asthma"), "Asthma")

    def test_multi_vaccine_count_and_primary_vaccine(self):
        r = self.out.set_index("report_id").loc[str(23_000_010)]
        self.assertEqual(r.n_vaccines, 2)
        self.assertEqual(r.vax_type, "VARZOS")  # first listed row, not the extra FLU4

    def test_predictors_invariant_to_outcome_flags(self):
        data, vax, sym = self.raw
        flipped = data.copy()
        flipped[["HOSPITAL", "DIED", "L_THREAT"]] = ""
        flipped["HOSPDAYS"] = ""; flipped["RECOVD"] = "Y"
        z = self.tmp / "flipped.zip"
        with zipfile.ZipFile(z, "w") as zf:
            for name, df in [("VAERSDATA", flipped), ("VAERSVAX", vax), ("VAERSSYMPTOMS", sym)]:
                buf = io.BytesIO(); df.to_csv(buf, index=False, encoding="latin-1"); zf.writestr(f"2023{name}.csv", buf.getvalue())
        out2, _ = bv.build([z], "v1", n_rows=100000)
        a = self.out.drop(columns="serious").set_index("report_id")
        b = out2.drop(columns="serious").set_index("report_id")
        common = a.index.intersection(b.index)
        self.assertGreater(len(common), 0.95 * len(a))
        pd.testing.assert_frame_equal(a.loc[common], b.loc[common])
        self.assertEqual(out2.serious.sum(), 0)

    def test_v2_narrative_masks_outcome_sentences(self):
        out, meta = bv.build([self.zip], "v2", n_rows=100000)
        self.assertIn("narrative_masked", meta["text_columns"])
        self.assertFalse(out.narrative_masked.str.contains("admitted|hospital", case=False).any())
        self.assertTrue(out.narrative_masked.str.contains("resolved with rest").any())
        self.assertEqual(bv.mask_narrative("Pt died. Fever began day 2."), "Fever began day 2.")

    def test_schema_loads_with_evaluation_adapter(self):
        out_dir = self.tmp / "v1"
        csv = bv.write(self.out, self.meta, out_dir, "v1")
        df, schema = load_csv(csv, out_dir / "schema.json")
        self.assertEqual(len(df), len(self.out))
        self.assertEqual(schema["id_column"], "report_id")
        self.assertEqual(set(schema["structured_columns"] + schema["text_columns"]) & set(schema["audit_columns"]), set())
        self.assertTrue(json.loads((out_dir / "schema.json").read_text())["removed_terms"]["outcome"])

    def test_sampling_is_deterministic_and_capped(self):
        a, _ = bv.build([self.zip], "v1", n_rows=500, seed=7)
        b, _ = bv.build([self.zip], "v1", n_rows=500, seed=7)
        self.assertEqual(len(a), 500)
        pd.testing.assert_frame_equal(a, b)

    def test_categoricals_stay_below_text_detector(self):
        for col in ["state", "vax_type", "vax_manu", "vax_dose_series", "vax_route", "vax_site", "admin_by", "sex"]:
            self.assertLess(self.out[col].nunique(), 100, col)


if __name__ == "__main__":
    unittest.main()
