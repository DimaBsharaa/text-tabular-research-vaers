"""Offline test of run_audit on the synthetic VAERS fixture with a fabricated results folder."""
import hashlib
import json
import sys
import tempfile
import unittest
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
                                                           "model_key": "light", "columns": cols}))

    def test_full_audit_runs_and_mechanics_pass(self):
        text = ra.run(self.csv, self.csv.parent / "schema.json", self.results, [self.zip], self.tmp / "audit.md")
        self.assertIn("FLAG lines: 0", text)
        self.assertIn("Near-duplicate", text)
        self.assertIn("ORIGINAL:", text)
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


if __name__ == "__main__":
    unittest.main()
