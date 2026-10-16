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

    def test_v4_v5_narrative_variants_mask_outcomes_and_boilerplate(self):
        out4, meta4 = bv.build([self.zip], "v4", n_rows=100000)
        out5, meta5 = bv.build([self.zip], "v5", n_rows=100000)
        self.assertEqual(meta4["text_columns"], ["symptoms", *bv.TEXT_SOURCES, "narrative"])
        self.assertEqual(meta5["text_columns"], [*bv.TEXT_SOURCES, "narrative"])
        self.assertNotIn("symptoms", out5.columns)
        for out in (out4, out5):
            self.assertFalse(out.narrative.str.contains("admitted|hospital", case=False).any())
            self.assertTrue(out.narrative.str.contains("resolved with rest").any())
        self.assertEqual(sorted(out4.report_id), sorted(out5.report_id))  # same population as each other
        self.assertEqual(meta4["counts"]["outcome_mentions_left"]["narrative"], 0)
        self.assertEqual(bv.mask_narrative("Pt died. Fever began day 2."), "Fever began day 2.")
        self.assertEqual(bv.mask_narrative("Reported Symptoms: 10016256:FATIGUE; Narrative: Tired all day. Lot# EW0182 given."),
                         "Tired all day. given.")
        self.assertEqual(bv.mask_narrative("Taken to the ER by ambulance; rash on arm"), "rash on arm")

    def test_scope_excludes_breakthrough_and_late_onset(self):
        data, vax, sym = self.raw
        data, sym = data.copy(), sym.copy()
        ids = data.VAERS_ID.iloc[[3, 4, 6]].tolist()
        sym.loc[sym.VAERS_ID == ids[0], "SYMPTOM1"] = "Vaccine breakthrough infection"
        sym.loc[sym.VAERS_ID == ids[1], "SYMPTOM1"] = "COVID-19"
        data.loc[data.VAERS_ID == ids[2], "NUMDAYS"] = "120"
        z = self.tmp / "scope.zip"
        with zipfile.ZipFile(z, "w") as zf:
            for name, df in [("VAERSDATA", data), ("VAERSVAX", vax), ("VAERSSYMPTOMS", sym)]:
                buf = io.BytesIO(); df.to_csv(buf, index=False, encoding="latin-1"); zf.writestr(f"2023{name}.csv", buf.getvalue())
        out, meta = bv.build([z], "v1", n_rows=100000)
        self.assertFalse(set(ids) & set(out.report_id))
        self.assertGreaterEqual(meta["counts"]["removed_infection_or_failure_reports"], 2)
        self.assertGreaterEqual(meta["counts"]["removed_onset_over_42_days"], 1)
        self.assertTrue((out.onset_days.dropna() <= bv.MAX_ONSET_DAYS).all())
        # "COVID-19 vaccine"-like reaction terms that merely contain the word are not excluded
        self.assertFalse(bv.OUT_OF_SCOPE_TERM.search("Injection site pain; Pyrexia"))
        self.assertTrue(bv.OUT_OF_SCOPE_TERM.search("Fatigue; COVID-19; Pyrexia"))
        self.assertFalse(bv.OUT_OF_SCOPE_TERM.search("Post-acute COVID-19 syndrome"))

    def test_v6_onset_narrative_stops_before_care(self):
        f = bv.onset_narrative
        self.assertEqual(f("Tingling of lips and tongue. Started 10 min after. Given epinephrine at pharmacy. Resolved."),
                         "Tingling of lips and tongue. Started 10 min after.")
        self.assertEqual(f("On 1/5/21 pt developed SOB and fever. Started on remdesivir and O2."),
                         "On pt developed SOB and fever.")
        self.assertEqual(f("Reported Symptoms: 10016256:FATIGUE; Narrative: Tired all day. Went to the ER."), "Tired all day.")
        self.assertEqual(f("Patient was admitted. Fever."), bv.EMPTY_TEXT)  # cut at the first outcome sentence
        self.assertEqual(f("A. B. C. D. E."), "A. B. C.")
        out, meta = bv.build([self.zip], "v6", n_rows=100000)
        self.assertEqual(meta["text_columns"], ["symptoms", *bv.TEXT_SOURCES, "narrative_onset"])
        left = meta["counts"]["outcome_mentions_left"]
        self.assertEqual(left["narrative_onset"], 0)
        self.assertEqual(left["narrative_onset_care_words"], 0)
        self.assertFalse(out.narrative_onset.str.contains("admitted|hospital", case=False).any())

    def test_variants_share_identical_report_ids(self):
        ids = {v: sorted(bv.build([self.zip], v, n_rows=800, seed=3)[0].report_id) for v in ("v1", "v3", "v6", "v7")}
        self.assertEqual(len({tuple(x) for x in ids.values()}), 1)
        self.assertEqual(len(ids["v1"]), 800)
        self.assertEqual(bv.mask_outcome_sentences("Asthma. An overnight stay on the ward was required."), "Asthma.")

    def test_review_fixes_negative_onset_dedup_fields_given(self):
        self.assertEqual(bv.onset_narrative("Vaccine was given at noon. Tingling began that evening."),
                         "Vaccine was given at noon. Tingling began that evening.")
        self.assertEqual(bv.onset_narrative("Arm swelled. Given epinephrine at pharmacy."), "Arm swelled.")
        data, vax, sym = self.raw
        data = data.copy()
        neg_id = data.VAERS_ID.iloc[7]
        data.loc[data.VAERS_ID == neg_id, "NUMDAYS"] = "-2"
        z = self.tmp / "neg.zip"
        with zipfile.ZipFile(z, "w") as zf:
            for name, df in [("VAERSDATA", data), ("VAERSVAX", vax), ("VAERSSYMPTOMS", sym)]:
                buf = io.BytesIO(); df.to_csv(buf, index=False, encoding="latin-1"); zf.writestr(f"2023{name}.csv", buf.getvalue())
        out, meta = bv.build([z], "v1", n_rows=100000)
        self.assertNotIn(neg_id, set(out.report_id))
        self.assertEqual(meta["counts"]["removed_negative_onset"], 1)
        s = pd.Series(["A", "B", "A", None])
        self.assertEqual(bv.cap_categories(s).tolist(), ["A", "B", "A", "UNKNOWN"])  # no grouping below the cap

    def test_onset_cut_at_chart_language_and_rechecked(self):
        # ICD-style diagnosis lists end the excerpt.
        self.assertEqual(bv.onset_narrative("Fever and chills that night. Acute respiratory failure with hypoxia, "
                                            "unspecified. Hypotension."), "Fever and chills that night.")
        self.assertEqual(bv.onset_narrative("Myocarditis, init encntr. Chest pain."), bv.EMPTY_TEXT)
        self.assertEqual(bv.onset_narrative("Pt c/o chest pain."), bv.EMPTY_TEXT)
        # A split inside "Dr. Lee" must not let care language through.
        out = bv.onset_narrative("Arm was sore. She was seen by Dr. Lee the next day.")
        self.assertEqual(out, "Arm was sore.")
        self.assertFalse(bv.CARE_SENTENCE.search(out) or bv.CHART_SENTENCE.search(out))
        # Ordinary first-person accounts survive.
        self.assertEqual(bv.onset_narrative("My arm hurt within an hour. I felt dizzy."),
                         "My arm hurt within an hour. I felt dizzy.")

    def test_build6_reports_without_onset_excerpt_are_excluded_everywhere(self):
        # Gaps left by removed dates must not hide care phrases ("Went 12/3 to" -> "Went to").
        out = bv.onset_narrative("Rash appeared. Went 12/3 to urgent place.")
        self.assertEqual(out, "Rash appeared.")
        data, vax, sym = self.raw
        data = data.copy()
        gone = data.VAERS_ID.iloc[3]
        data.loc[data.VAERS_ID == gone, "SYMPTOM_TEXT"] = "Patient was hospitalized on arrival. Rash."
        z = self.tmp / "noexcerpt.zip"
        with zipfile.ZipFile(z, "w") as zf:
            for name, df in [("VAERSDATA", data), ("VAERSVAX", vax), ("VAERSSYMPTOMS", sym)]:
                buf = io.BytesIO(); df.to_csv(buf, index=False, encoding="latin-1"); zf.writestr(f"2023{name}.csv", buf.getvalue())
        for variant in ("v1", "v6", "v7"):
            built, meta = bv.build([z], variant, n_rows=100000)
            self.assertNotIn(gone, set(built.report_id), variant)
            self.assertGreaterEqual(meta["counts"]["with_symptom_terms"] - meta["counts"]["with_onset_excerpt"], 1)
        self.assertTrue((built.narrative_onset != bv.EMPTY_TEXT).all())

    def test_v7_has_no_coded_symptoms(self):
        out, meta = bv.build([self.zip], "v7", n_rows=100000)
        self.assertEqual(meta["text_columns"], [*bv.TEXT_SOURCES, "narrative_onset"])
        self.assertNotIn("symptoms", out.columns)
        v6, _ = bv.build([self.zip], "v6", n_rows=100000)
        self.assertEqual(sorted(out.report_id), sorted(v6.report_id))  # same population

    def test_background_outcome_statements_are_removed(self):
        self.assertEqual(bv.mask_outcome_sentences("Hospitalized after this event; patient died."), bv.EMPTY_TEXT)
        self.assertEqual(bv.mask_outcome_sentences("Asthma. Admitted to ICU after vaccine. Diabetes"), "Asthma. Diabetes")
        data, vax, sym = self.raw
        data = data.copy()
        data.loc[5, "HISTORY"] = "Asthma; Hospitalized after this event; patient died."
        z = self.tmp / "bg.zip"
        with zipfile.ZipFile(z, "w") as zf:
            for name, df in [("VAERSDATA", data), ("VAERSVAX", vax), ("VAERSSYMPTOMS", sym)]:
                buf = io.BytesIO(); df.to_csv(buf, index=False, encoding="latin-1"); zf.writestr(f"2023{name}.csv", buf.getvalue())
        for variant in ("v1", "v3"):
            out, meta = bv.build([z], variant, n_rows=100000)
            row = out.set_index("report_id").loc[data.loc[5, "VAERS_ID"]]
            text = " ".join(str(row[c]) for c in meta["text_columns"])
            self.assertNotRegex(text.lower(), "hospitali|died")
            self.assertIn("Asthma", text)
            self.assertTrue(all(v == 0 for k, v in meta["counts"]["outcome_mentions_left"].items() if k != "symptoms"))

    def test_symptom_filter_covers_life_threatening_and_birth_defect(self):
        s = pd.DataFrame({"VAERS_ID": ["1", "1", "1"],
                          "SYMPTOM1": ["Life-threatening illness", "Congenital anomaly", "Rash"]})
        joined, removed = bv.symptom_text(s)
        self.assertEqual(joined.get("1"), "Rash")
        self.assertEqual(removed["outcome"], ["Congenital anomaly", "Life-threatening illness"])

    def test_dedup_keeps_distinct_patients_with_same_keys(self):
        data, vax, sym = self.raw
        twin = dict(data.iloc[2]); twin["VAERS_ID"] = str(int(twin["VAERS_ID"]) + 800_000)
        twin_sym = {"VAERS_ID": twin["VAERS_ID"], "SYMPTOM1": "Tinnitus"}  # same keys, different coded event
        data2 = pd.concat([data, pd.DataFrame([twin])], ignore_index=True)
        vax2 = pd.concat([vax, pd.DataFrame([dict(v) | {"VAERS_ID": twin["VAERS_ID"]}
                                              for v in vax.to_dict("records") if v["VAERS_ID"] == data.iloc[2]["VAERS_ID"]])])
        sym2 = pd.concat([sym, pd.DataFrame([twin_sym])], ignore_index=True)
        z = self.tmp / "twin.zip"
        with zipfile.ZipFile(z, "w") as zf:
            for name, df in [("VAERSDATA", data2), ("VAERSVAX", vax2), ("VAERSSYMPTOMS", sym2)]:
                buf = io.BytesIO(); df.to_csv(buf, index=False, encoding="latin-1"); zf.writestr(f"2023{name}.csv", buf.getvalue())
        out, meta = bv.build([z], "v1", n_rows=100000)
        self.assertIn(twin["VAERS_ID"], set(out.report_id))
        self.assertEqual(meta["counts"]["dedup_reports_removed"], 1)  # only the true secondary copy

    def test_zip_plus_extracted_csv_is_rejected(self):
        d = self.tmp / "extracted"; d.mkdir(exist_ok=True)
        with zipfile.ZipFile(self.zip) as zf:
            zf.extractall(d)
        with self.assertRaises(ValueError):
            bv.build([self.zip, *sorted(d.glob("*.csv"))], "v1")

    def test_v3_merges_background_into_one_column(self):
        out, meta = bv.build([self.zip], "v3", n_rows=100000)
        self.assertEqual(meta["text_columns"], ["symptoms", "patient_background"])
        self.assertFalse(set(bv.TEXT_SOURCES) & set(out.columns))
        v1 = self.out.set_index("report_id")
        v3 = out.set_index("report_id")
        common = v1.index.intersection(v3.index)
        pd.testing.assert_series_equal(v1.loc[common, "serious"], v3.loc[common, "serious"])
        row = pd.Series({"history": "Asthma", "current_illness": bv.EMPTY_TEXT, "other_meds": "metformin",
                         "allergies": bv.EMPTY_TEXT})
        self.assertEqual(bv.merge_background(row), "history: Asthma; medications: metformin")
        self.assertEqual(bv.merge_background(row.map(lambda _: bv.EMPTY_TEXT)), bv.EMPTY_TEXT)
        self.assertLess((v3.patient_background == bv.EMPTY_TEXT).mean(), (v1.history == bv.EMPTY_TEXT).mean() + 1e-9)

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
