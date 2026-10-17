"""Offline test of run_audit on the synthetic VAERS fixture with a fabricated results folder."""
import hashlib
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from research.scripts.vaers import build_vaers as bv, run_audit as ra  # noqa: E402
from research.scripts.vaers import test_build_vaers as tbv  # noqa: E402


class RunAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tbv.BuildVaersTest.setUpClass()
        cls.zip, cls.tmp = tbv.BuildVaersTest.zip, Path(tempfile.mkdtemp())
        out, meta = bv.build([cls.zip], "v7", n_rows=100000)
        big = pd.concat([out] * 6, ignore_index=True)  # repeated rows = guaranteed near-duplicates
        big["report_id"] = [f"{i}" for i in range(len(big))]
        cls.csv = bv.write(big, meta, cls.tmp / "v7", "v7")
        cls.schema = json.loads((cls.csv.parent / "schema.json").read_text())
        sha = hashlib.sha256(cls.csv.read_bytes()).hexdigest()
        schema_sha = hashlib.sha256((cls.csv.parent / 'schema.json').read_bytes()).hexdigest()
        ids = big.report_id.tolist()
        rng = np.random.default_rng(0)
        cls.results = cls.tmp / "results"
        cols_joint = cls.schema["structured_columns"] + cls.schema["text_columns"]
        for fold in range(2):
            perm = rng.permutation(ids).tolist()
            test, train = perm[: len(ids) // 5], perm[len(ids) // 5:]
            fit, val = train[: int(len(train) * .9)], train[int(len(train) * .9):]
            for cond, cols in [("structured_only", cls.schema["structured_columns"]),
                               ("text_only", cls.schema["text_columns"]),
                               ("joint_frozen", cols_joint), ("joint_tar", cols_joint)]:
                d = cls.results / f"{fold}_{cond}"
                d.mkdir(parents=True)
                splits = {"outer_train": train, "outer_test": test, "learner_fit": fit, "learner_validation": val,
                          "detected_text_columns": [] if cond == "structured_only" else cls.schema["text_columns"]}
                if cond == "joint_tar":
                    splits |= {"encoder_train": fit[: int(len(fit) * .9)], "encoder_validation": fit[int(len(fit) * .9):]}
                (d / "splits.json").write_text(json.dumps(splits))
                (d / "config.json").write_text(json.dumps({"csv_sha256": sha, "fold": fold, "condition": cond,
                                                           "schema_sha256": schema_sha,
                                                           "model_key": "light", "columns": cols}))
                (d / 'result.json').write_text(json.dumps({'status': 'ok'}))

    def test_full_audit_runs_and_mechanics_pass(self):
        # This fixture deliberately gives cloned reports new IDs. Original-linkage is
        # tested separately on authentic source IDs, not on these synthetic IDs.
        text = ra.run(self.csv, self.csv.parent / "schema.json", self.results, (), self.tmp / "audit.md")
        self.assertIn("FLAG lines: 0", text)
        self.assertIn("Candidate duplicate", text)
        self.assertIn("NOT_VERIFIED no raw yearly files", text)
        self.assertTrue((self.tmp / "audit.md").exists())

    def test_mechanics_flags_a_leaking_encoder(self):
        runs = ra.load_runs(self.results)
        bad = json.loads(json.dumps(runs))
        for r in bad:
            if "encoder_train" in r["splits"]:
                r["splits"]["encoder_train"] = r["splits"]["encoder_train"] + r["splits"]["outer_test"][:3]
        sha = runs[0]["config"]["csv_sha256"]
        self.assertTrue(any(l.startswith("FLAG") for l in ra.mechanics(bad, sha, self.schema)))

    def test_near_duplicates_found(self):
        df = pd.read_csv(self.csv); df["report_id"] = df.report_id.astype(str)
        self.assertTrue(ra.near_duplicate_groups(df))

    def test_encoder_training_validation_overlap_is_flagged(self):
        runs = json.loads(json.dumps(ra.load_runs(self.results)))
        tar = next(r for r in runs if r['config']['condition'] == 'joint_tar')
        tar['splits']['encoder_train'].append(tar['splits']['encoder_validation'][0])
        lines = ra.mechanics(runs, runs[0]['config']['csv_sha256'], self.schema)
        self.assertTrue(any(l.startswith('FLAG') and 'encoder partitions' in l for l in lines))

    def test_missing_encoder_partition_and_missing_conditions_are_flagged(self):
        runs = json.loads(json.dumps(ra.load_runs(self.results)))
        tar = next(r for r in runs if r['config']['condition'] == 'joint_tar')
        del tar['splits']['encoder_validation']
        lines = ra.mechanics(runs, runs[0]['config']['csv_sha256'], self.schema)
        self.assertTrue(any(l.startswith('FLAG') and 'required partitions' in l for l in lines))
        only = [r for r in runs if r['config']['condition'] == 'structured_only']
        lines = ra.mechanics(only, only[0]['config']['csv_sha256'], self.schema)
        self.assertTrue(any(l.startswith('FLAG') and 'all four conditions' in l for l in lines))
        self.assertTrue(any(l.startswith('FLAG') and 'identical predictor columns' in l for l in lines))

    def test_large_exact_match_block_is_detected_with_coverage(self):
        df = pd.read_csv(self.csv).iloc[[0]].copy()
        big = pd.concat([df] * 401, ignore_index=True)
        big['report_id'] = [str(i) for i in range(401)]
        coverage = {}
        groups = ra.near_duplicate_groups(big, coverage=coverage)
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]), 401)
        self.assertEqual(coverage['large_blocks_exact_only'], 1)
        self.assertEqual(coverage['rows_in_large_blocks'], 401)

    def test_only_cross_split_duplicate_members_are_removed(self):
        groups = [{'test_a', 'test_b'}, {'train_a', 'test_c'}, {'unused', 'test_d'}]
        self.assertEqual(ra.cross_split_test_ids(groups, ['train_a'], ['test_a', 'test_b', 'test_c', 'test_d']), {'test_c'})

    def test_duplicate_reporting_normalizes_numeric_split_ids(self):
        groups = [{'1', '2'}]
        runs = [{'config': {'fold': 0}, 'splits': {'outer_train': [1], 'outer_test': [2]}}]
        lines = ra.cross_split_duplicates(groups, runs, {'1': 0, '2': 0})
        self.assertIn('test rows involved = 1 of 1', '\n'.join(lines))

    def test_duplicate_sensitivity_keeps_test_only_cluster(self):
        frame = pd.DataFrame({'report_id': ['train0', 'train1', 'test_a', 'test_b'],
                              'serious': [0, 1, 0, 1], 'value': [0, 1, 0, 1]}).set_index('report_id', drop=False)
        class Model:
            def fit(self, x, y):
                return self
            def predict_proba(self, x):
                return np.array([[.8, .2], [.2, .8]])
        with patch.object(ra, '_fold_frames', return_value=[(0, frame.iloc[:2], frame.iloc[2:])]), patch.object(ra, '_proxy', return_value=Model()):
            lines = ra.duplicate_sensitivity(frame, {'target': 'serious', 'structured_columns': ['value'], 'text_columns': []}, [], [{'test_a', 'test_b'}])
        self.assertIn('removed 0', '\n'.join(lines))

    def test_joint_stratum_support_is_reported(self):
        df = pd.DataFrame({'report_year': [2023] * 5, 'vax_type': ['A'] * 3 + ['B'] * 2,
                           'serious': [0, 0, 1, 1, 0], '_p': [.1, .2, .9, .8, .3]})
        lines = ra._stratum_scores(df, 'serious', ['report_year', 'vax_type'], 0)
        self.assertEqual(len(lines), 2)
        self.assertTrue(all(l.startswith('NOT_ESTIMATED') and 'report_year x vax_type' in l for l in lines))

    def test_raw_examples_are_actually_linked_and_full_text_saved(self):
        data, _ = bv.build([self.zip], 'v7', n_rows=100000)
        dest = self.tmp / 'matched.examples.json'
        lines = ra.originals(data, self.schema, [self.zip], n=2, samples_path=dest)
        self.assertFalse(any(l.startswith('FLAG') for l in lines), '\n'.join(lines))
        examples = json.loads(dest.read_text())
        self.assertEqual(len(examples), 4)
        self.assertTrue(all(e['original_narrative'] and e['retained_excerpt'] for e in examples))
        bad = data.head(1).copy(); bad['report_id'] = 'not-in-the-source'
        self.assertTrue(any(l.startswith('FLAG') and 'not found' in l for l in ra.originals(bad, self.schema, [self.zip], n=1)))

    def test_identity_failure_stops_before_proxy_fitting(self):
        broken = self.tmp / 'changed-schema.json'
        broken.write_text((self.csv.parent / 'schema.json').read_text() + '\n')
        with patch.object(ra, '_proxy') as fit:
            with self.assertRaises(ra.AuditInputError):
                ra.run(self.csv, broken, self.results, out_path=self.tmp / 'failed.md')
            fit.assert_not_called()
        self.assertIn('FLAG', (self.tmp / 'failed.md').read_text())

    def test_raw_ids_with_leading_zeros_match_csv_numeric_ids(self):
        data, _ = bv.build([self.zip], 'v7', n_rows=100000)
        tables = bv.load_year_files([self.zip])
        tables['DATA']['VAERS_ID'] = tables['DATA'].VAERS_ID.astype(str).str.zfill(12)
        data['report_id'] = data.report_id.astype(int)
        with patch.object(bv, 'load_year_files', return_value=tables):
            lines = ra.originals(data, self.schema, [self.zip], n=2)
        self.assertFalse(any(l.startswith('FLAG') for l in lines), '\n'.join(lines))
        self.assertEqual(sum('ORIGINAL:' in l for l in lines), 4)

    def test_raw_id_normalization_collision_stops_examples(self):
        data, _ = bv.build([self.zip], 'v7', n_rows=100000)
        tables = bv.load_year_files([self.zip])
        extra = tables['DATA'].iloc[[0]].copy()
        extra['VAERS_ID'] = '0' + str(extra.iloc[0].VAERS_ID)
        tables['DATA'] = pd.concat([tables['DATA'], extra], ignore_index=True)
        with patch.object(bv, 'load_year_files', return_value=tables):
            lines = ra.originals(data, self.schema, [self.zip], n=2)
        self.assertTrue(any(l.startswith('FLAG') and 'collide' in l for l in lines))
        self.assertFalse(any('ORIGINAL:' in l for l in lines))


if __name__ == "__main__":
    unittest.main()
