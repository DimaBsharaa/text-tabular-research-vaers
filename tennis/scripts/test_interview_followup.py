"""Boundary checks for source selection and outcome-reference auditing."""
import unittest

import pandas as pd

from tennis.scripts.common import OUT
from tennis.scripts.interview_followup import latest_unique_before, split_audit, visible_text
from tennis.scripts.prepare_interview_review import chronological_partitions, clipping_audit


class FollowupTests(unittest.TestCase):
    def test_partition_dates_remain_together_and_ignore_targets(self):
        frame = pd.DataFrame({"match_id": list("abcdefghij"),
            "match_date": ["2015-01-01", "2015-01-02", "2015-01-03", "2015-01-04", "2015-01-05",
                           "2015-01-06", "2015-01-06", "2015-01-07", "2015-01-08", "2015-01-08"],
            "target": [0, 1] * 5})
        first, cutoffs = chronological_partitions(frame)
        changed = frame.copy(); changed.target = 1 - changed.target
        second, _ = chronological_partitions(changed)
        pd.testing.assert_frame_equal(first, second)
        self.assertTrue((first.groupby("match_date").split.nunique() == 1).all())
        groups = {name: group for name, group in first.groupby("split")}
        self.assertLess(groups["train"].match_date.max(), groups["development"].match_date.min())
        self.assertLess(groups["development"].match_date.max(), groups["reserved_test"].match_date.min())

    def test_clipping_distinguishes_late_cues_from_absent_cues(self):
        frame = pd.DataFrame({"interview_id": [1, 2],
            "answer_text": ["ordinary " * 301 + "My shoulder was painful.", "I practiced my forehand."]})
        result = clipping_audit(frame).set_index("interview_id")
        self.assertTrue(result.loc[1, "condition_cue_full"])
        self.assertFalse(result.loc[1, "condition_cue_first300"])
        self.assertFalse(result.loc[2, "condition_cue_full"])

    def test_same_day_and_future_documents_are_excluded(self):
        docs = pd.DataFrame({"date": pd.to_datetime(["2015-01-09", "2015-01-10", "2015-01-11"]),
                             "interview_id": [1, 2, 3]})
        self.assertEqual(latest_unique_before(docs, pd.Timestamp("2015-01-10")).interview_id, 1)

    def test_earlier_cutoff_does_not_extend_freshness_window(self):
        docs = pd.DataFrame({"date": pd.to_datetime(["2015-01-01"]), "interview_id": [1]})
        self.assertIsNone(latest_unique_before(docs, pd.Timestamp("2015-03-25"),
                                               age_reference=pd.Timestamp("2015-04-05")))

    def test_ambiguous_latest_interview_is_not_chosen(self):
        docs = pd.DataFrame({"date": pd.to_datetime(["2015-01-08", "2015-01-09", "2015-01-09"]),
                             "interview_id": [1, 2, 3]})
        self.assertIsNone(latest_unique_before(docs, pd.Timestamp("2015-01-10")))

    def test_source_reference_and_document_reuse_are_different(self):
        train = pd.DataFrame([{"match_id": "later", "match_date": "2015-02-01",
            "a_interview_id": 10, "b_interview_id": 11, "a_source_match_id": "heldout",
            "b_source_match_id": "other", "player_a_key": "p1", "player_b_key": "p2"}])
        test = pd.DataFrame([{"match_id": "heldout", "match_date": "2015-01-25",
            "a_interview_id": 11, "b_interview_id": 20, "a_source_match_id": "old1",
            "b_source_match_id": "old2", "player_a_key": "p2", "player_b_key": "p3"}])
        row, examples = split_audit(train, test, "synthetic")
        self.assertEqual(row["train_test_target_overlap"], 0)
        self.assertEqual(row["heldout_targets_referenced_in_training_source_matches"], 1)
        self.assertEqual(row["shared_interview_documents"], 1)
        self.assertEqual(examples.iloc[0].referenced_test_match_date, "2015-01-25")

    def test_publisher_encoding_repair_preserves_words(self):
        self.assertEqual(visible_text("<p>Good.\u00c2&nbsp;I felt pain.</p><script>bad</script>"), "Good. I felt pain.")

    def test_measured_chronological_split_has_no_reverse_reference(self):
        rows = pd.read_csv(OUT / "interview_split_risks.csv").set_index("split")
        self.assertEqual(rows.loc["chronological_80_20", "heldout_targets_referenced_in_training_source_matches"], 0)
        self.assertGreater(rows.loc["random_80_20_seed42_diagnostic", "heldout_targets_referenced_in_training_source_matches"], 0)

    def test_same_day_negative_control_is_rejected(self):
        rows = pd.read_csv(OUT / "interview_publisher_validation.csv")
        bad = rows[rows.purpose == "deliberately_wrong_same_day_interview"].iloc[0]
        self.assertTrue(bad.date_agrees)
        self.assertTrue(bad.player_agrees)
        self.assertFalse(bad.first_long_answer_prefix_matches)
        self.assertFalse(rows.historical_availability_verified.any())


if __name__ == "__main__":
    unittest.main()
