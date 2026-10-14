"""Offline checks for leakage boundaries and honest result aggregation (no fits)."""
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import types
import unittest

import pandas as pd

from evaluation.colab_runner import load_csv, selected_columns, verify_reference, verify_import_origins, OFFICIAL_REPOSITORY
from evaluation.results import CONDITIONS, summarize
from research.scripts.prepare_tennis_smoke import build

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "research/data/tennis_smoke"


class TennisHistoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = pd.read_csv(DATA / "kaggle_v1219_prefix.csv")
        cls.prepared = build(cls.raw)

    def test_slots_independent_of_current_outcome(self):
        swapped = self.raw.copy()
        swapped["Winner"] = swapped.apply(lambda r: r.Player_2 if r.Winner == r.Player_1 else r.Player_1, axis=1)
        other = build(swapped)
        pd.testing.assert_frame_equal(self.prepared[["match_id", "player_a", "player_b"]], other[["match_id", "player_a", "player_b"]])
        self.assertTrue((self.prepared.player_a_wins == 1 - other.player_a_wins).all())

    def test_target_day_outcomes_cannot_change_same_day_features(self):
        dates = sorted(self.raw.Date.unique())
        cutoff = dates[len(dates) // 2]
        changed = self.raw.copy()
        affected = changed.Date >= cutoff
        changed.loc[affected, "Winner"] = changed[affected].apply(lambda r: r.Player_2 if r.Winner == r.Player_1 else r.Player_1, axis=1)
        other = build(changed)
        mask = self.prepared.match_date <= cutoff
        predictors = [c for c in self.prepared if c != "player_a_wins"]
        pd.testing.assert_frame_equal(self.prepared.loc[mask, predictors], other.loc[mask, predictors])

    def test_original_slot_swap_preserves_dataset(self):
        swapped = self.raw.copy()
        for prefix in ["Player", "Rank", "Pts", "Odd"]:
            swapped[[prefix + "_1", prefix + "_2"]] = swapped[[prefix + "_2", prefix + "_1"]].to_numpy()
        pd.testing.assert_frame_equal(self.prepared, build(swapped))

    def test_current_score_odds_unused(self):
        changed = self.raw.copy()
        changed["Score"] = "deliberately different current score"
        changed[["Odd_1", "Odd_2"]] = 9999
        pd.testing.assert_frame_equal(self.prepared, build(changed))

    def test_schema_selection_and_hash_guard(self):
        df, schema = load_csv(DATA / "tennis_smoke.csv", DATA / "schema.json")
        for condition in CONDITIONS:
            selected = selected_columns(schema, condition)
            self.assertFalse(set(selected) & set(schema["audit_columns"] + [schema["target"]]))
        self.assertEqual(selected_columns(schema, "joint_frozen"), selected_columns(schema, "joint_tar"))
        self.assertEqual(len(df), 488)
        with tempfile.TemporaryDirectory() as folder:
            csv = Path(folder) / "modified.csv"
            csv.write_bytes((DATA / "tennis_smoke.csv").read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "hash"):
                load_csv(csv, DATA / "schema.json")


def records(folds, mode="full", purpose="candidate_dataset"):
    return pd.DataFrame([{"model_key": "light", "condition": c, "fold": f, "status": "ok", "auroc": score,
                          "mode": mode, "purpose": purpose, "csv_sha256": "dataset", "benchmark_commit": "ref", "experiment_id": "same"}
                         for f in folds for c, score in zip(CONDITIONS, [.6, .65, .7, .73])])


class ResultTests(unittest.TestCase):
    def test_smoke_does_not_claim_pass(self):
        result = summarize(records([0], "smoke", "smoke_fixture"), [0], "smoke", "smoke_fixture").iloc[0]
        self.assertEqual(result.status, "complete")
        self.assertAlmostEqual(result.joint_signal_gain, .05)
        self.assertIsNone(result.assignment_pass)

    def test_missing_condition_never_zero_or_pass(self):
        raw = records(range(5))
        raw = raw[raw.condition != "text_only"]
        result = summarize(raw, list(range(5)), "full", "candidate_dataset").iloc[0]
        self.assertEqual(result.status, "incomplete")
        self.assertTrue(pd.isna(result.text_only))
        self.assertTrue(pd.isna(result.joint_signal_gain))
        self.assertIsNone(result.assignment_pass)

    def test_complete_candidate_and_error(self):
        raw = records(range(5))
        result = summarize(raw, list(range(5)), "full", "candidate_dataset").iloc[0]
        self.assertTrue(result.assignment_pass)
        self.assertTrue(result.paper_pass)
        raw.loc[0, "status"] = "error"
        result = summarize(raw, list(range(5)), "full", "candidate_dataset").iloc[0]
        self.assertEqual(result.status, "incomplete")
        self.assertIsNone(result.assignment_pass)

    def test_duplicate_and_mixed_experiments_rejected(self):
        raw = records([0])
        with self.assertRaisesRegex(ValueError, "Multiple successful"):
            summarize(pd.concat([raw, raw.iloc[:1]]), [0], "full", "candidate_dataset")
        raw.loc[0, "experiment_id"] = "different settings"
        with self.assertRaisesRegex(ValueError, "Mixed"):
            summarize(raw, [0], "full", "candidate_dataset")


class ReferenceProofTests(unittest.TestCase):
    """A tiny local fixture tests proof checks, not the official GPU environment."""
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ref = Path(self.tmp.name) / "reference"
        self.ref.mkdir()
        self.run_git("init", "--quiet")
        self.run_git("remote", "add", "origin", OFFICIAL_REPOSITORY + ".git")
        (self.ref / "sample.py").write_text("VALUE = 1\n")
        self.run_git("add", "sample.py")
        self.run_git("-c", "user.name=Proof fixture", "-c", "user.email=fixture@example.invalid", "commit", "--quiet", "-m", "fixture")
        self.commit = self.run_git("rev-parse", "HEAD").strip()

    def run_git(self, *args):
        return subprocess.check_output(["git", "-C", str(self.ref), *args], text=True)

    def test_byte_proof_and_hidden_modification(self):
        proof = verify_reference(self.ref, self.commit)
        self.assertEqual([r["path"] for r in proof], ["sample.py"])
        self.assertTrue(proof[0]["matches_commit"])
        self.run_git("update-index", "--assume-unchanged", "sample.py")
        (self.ref / "sample.py").write_text("VALUE = 2\n")
        self.assertFalse(self.run_git("status", "--porcelain").strip())
        with self.assertRaisesRegex(ValueError, "bytes differ"):
            verify_reference(self.ref, self.commit)

    def test_wrong_origin_or_commit_rejected(self):
        with self.assertRaisesRegex(ValueError, "pinned"):
            verify_reference(self.ref, "0" * 40)
        self.run_git("remote", "set-url", "origin", "https://example.invalid/unofficial.git")
        with self.assertRaisesRegex(ValueError, "official"):
            verify_reference(self.ref, self.commit)

    def test_imported_source_location_checked(self):
        proof = verify_reference(self.ref, self.commit)
        name = "multabench._provenance_test_fixture"
        self.assertNotIn(name, sys.modules)
        module = types.ModuleType(name)
        module.__file__ = str(self.ref / "sample.py")
        sys.modules[name] = module
        self.addCleanup(sys.modules.pop, name, None)
        self.assertEqual(verify_import_origins(self.ref, proof)[0]["loaded_from"], str(Path(module.__file__).resolve()))
        module.__file__ = str(Path(self.tmp.name) / "outside.py")
        Path(module.__file__).write_text("VALUE = 1\n")
        with self.assertRaisesRegex(ValueError, "outside"):
            verify_import_origins(self.ref, proof)


if __name__ == "__main__":
    unittest.main()
