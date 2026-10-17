"""Regression cases from observed care leaks, and label-blind construction checks."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from research.scripts.vaers import build_vaers as bv
from research.scripts.vaers import refine_v9 as rv
from research.scripts.vaers.test_build_vaers import write_zip


class RefineTextTest(unittest.TestCase):
    def test_transfusion_suffix_preserves_observed_anemia(self):
        text = 'Extreme inflammatory response resulting in hyperglycemia and hemolytic anemia requiring blood transfusion x 2.'
        out, reason = rv.strict_excerpt(text)
        self.assertEqual(out, 'Extreme inflammatory response resulting in hyperglycemia and hemolytic anemia')
        self.assertEqual(reason, 'care_or_record_suffix')

    def test_bypass_suffix_preserves_breathlessness(self):
        self.assertEqual(rv.strict_excerpt('Shortness of breath that resulted in triple bypass surgery')[0], 'Shortness of breath')

    def test_surgery_removes_intervention_not_symptom(self):
        self.assertEqual(rv.strict_excerpt('vomiting, lethargy, diarrhea- intussusception requiring surgery.')[0], 'vomiting, lethargy, diarrhea- intussusception')

    def test_medical_leave_is_removed(self):
        out, _ = rv.strict_excerpt('My left foot hurt. I was out on medical leave for months.')
        self.assertNotIn('medical leave', out)
        self.assertIn('My left foot hurt.', out)

    def test_cut_does_not_leave_a_dangling_care_sentence(self):
        self.assertEqual(rv.strict_excerpt('Patient experienced swelling. She underwent surgery.')[0], 'Patient experienced swelling.')
        self.assertEqual(rv.strict_excerpt('Rash. I had to go see a specialist.')[0], 'Rash.')

    def test_informative_severe_symptoms_and_negation_are_kept(self):
        for text in ['I had a stroke', 'Rash', 'No fever, no chest pain; mild redness.', 'Difficulty breathing with facial swelling']:
            self.assertEqual(rv.strict_excerpt(text)[0], text)

    def test_timestamps_and_record_pointers_are_not_meaningful(self):
        for text in ['- : 14:00.', 'Dr.', 'A', 'See attached records.', 'Vaccine was given.']:
            self.assertEqual(rv.strict_excerpt(text)[0], rv.EMPTY)

    def test_surgical_tape_allergy_is_not_invasive_care(self):
        self.assertEqual(rv.background_field('Allergic to surgical tape')[0], 'Allergic to surgical tape')

    def test_prior_procedure_and_medications_remain(self):
        self.assertEqual(rv.background_field('Surgery years ago; metformin 500mg; no drug allergies')[0], 'Surgery years ago; metformin 500mg; no drug allergies')

    def test_background_following_adverse_reaction_is_quarantined(self):
        text = 'He underwent extensive allergy testing following this adverse reaction.'
        value, post, reasons = rv.background_field(text)
        self.assertTrue(post)
        self.assertEqual(value, rv.EMPTY)

    def test_post_vaccine_background_brand_is_detected(self):
        self.assertTrue(rv.background_field('Atrial fibrillation returned after receiving SHINGRIX.')[1])
        self.assertTrue(rv.background_field('Severe allergies since these vaccines were given.')[1])
        self.assertTrue(rv.background_field('Received RIG; Rabies vaccine then additional doses.')[1])
        self.assertTrue(rv.background_field('He was hospitalized after vaccination.')[1])

    def test_actual_child_background_later_conditions_not_retained(self):
        raw = pd.DataFrame([{'HISTORY': 'Neurodevelopmental delay', 'CUR_ILL': 'Child was born healthy. Child was given the Heb B vaccine. Child underwent surgery for ears 3x.',
                             'OTHER_MEDS': 'None', 'ALLERGIES': 'Now yes'}], index=['1'])
        row = pd.DataFrame([{'report_id': '1', 'serious': '1', 'patient_background': 'history: Neurodevelopmental delay',
                             'narrative_onset': 'Child vomited after the shot.'}])
        out, ledger = rv.refine(row, raw)
        self.assertFalse(ledger.eligible.iloc[0])
        self.assertEqual(out.patient_background.iloc[0], rv.EMPTY)

    def test_row_eligibility_and_text_do_not_depend_on_label(self):
        raw = pd.DataFrame([{'HISTORY': 'Diabetes', 'CUR_ILL': 'None', 'OTHER_MEDS': 'metformin', 'ALLERGIES': 'penicillin'}], index=['1'])
        parent = pd.DataFrame([{'report_id': '1', 'serious': '0', 'patient_background': 'history: Diabetes',
                                'narrative_onset': 'Rash requiring surgery.'}])
        first, a = rv.refine(parent, raw)
        changed = parent.copy(); changed.serious = '1'
        second, b = rv.refine(changed, raw)
        pd.testing.assert_frame_equal(first.drop(columns='serious'), second.drop(columns='serious'))
        pd.testing.assert_frame_equal(a, b)


class RefineBuildTest(unittest.TestCase):
    def test_build_keeps_protected_fields_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root / '2023VAERSData.zip'
            write_zip(source, n=200)
            data, meta = bv.build([source], 'v9')
            csv = bv.write(data, meta, root / 'base', 'v9')
            paths, counts = rv.build(csv, [source], root / 'new')
            matched = pd.read_csv(paths['v10_matched'], keep_default_na=False, dtype=str)
            parent, _, _ = rv.read_base(csv)
            protected = bv.STRUCTURED + ['report_id', 'serious']
            pd.testing.assert_frame_equal(matched[protected], parent[protected])
            self.assertEqual(len(matched), len(parent))
            self.assertEqual(json.loads(Path(paths['v10']).with_name('schema.json').read_text())['build_version'], 9)
            with self.assertRaises(FileExistsError):
                rv.build(csv, [source], root / 'new')
            before = csv.read_bytes(); csv.write_bytes(before + b'\n')
            with self.assertRaisesRegex(ValueError, 'Parent CSV identity'):
                rv.read_base(csv)


if __name__ == '__main__':
    unittest.main()
