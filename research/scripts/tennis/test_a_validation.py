"""Adversarial boundary tests for Cornell A's new construction and audit."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from evaluation.tennis_temporal_pilot import load_temporal_candidate
from research.scripts.tennis.a_study import STUDY, assign_split
from research.scripts.tennis.a_temporal import asof_interviews, next_match_audit, ordinal
from research.scripts.tennis.a_text import POLICIES, count_tokens, select, verify_selected
from research.scripts.tennis.common import digest


class PassageTests(unittest.TestCase):
    def test_whole_block_keeps_qualified_negative_answer_together(self):
        from research.scripts.tennis.a_context_blocks import select_blocks
        result = select_blocks([["Were you tired?", "No. I was not tired. My shoulder felt fine."]])
        self.assertIn("No. I was not tired. My shoulder felt fine.", result["text"])
        self.assertLessEqual(result["selected_tokens_a"], 512)

    def test_exact_budget_includes_format_and_special_tokens(self):
        result = select([["How did you play?", "tennis " * 2000]], "token_head")
        self.assertEqual(result["selected_tokens_a"], 512)
        self.assertLessEqual(result["selected_tokens_b"], 512)

    def test_short_head_tail_does_not_repeat_the_middle(self):
        text = "tennis " * 350
        self.assertLess(count_tokens(text), 512)
        result = select([["Question", text]], "head_tail")
        self.assertEqual(result["text"], text.strip())

    def test_every_method_preserves_budget_on_long_interview(self):
        qa = [["How is your wrist?", "I have no wrist pain. " * 100],
              ["What changed?", "I adjusted my backhand and practiced on grass. " * 100]]
        for policy in POLICIES:
            result = select(qa, policy)
            self.assertLessEqual(result["selected_tokens_a"], 512, policy)
            self.assertTrue(verify_selected(qa, result), policy)

    def test_negation_and_short_answer_context_survive(self):
        qa = [["Did you have wrist pain?", "No."], ["Anything else?", "I practiced my return."]]
        result = select(qa, "topical_qa")
        self.assertIn("Q: Did you have wrist pain? A: No.", result["text"])

    def test_invented_output_fails_provenance(self):
        qa = [["How was it?", "My shoulder felt fine, and I served well."]]
        result = select(qa, "topical_answers")
        result["text"] += " The player will win the next match."
        self.assertFalse(verify_selected(qa, result))

    def test_changed_source_answer_fails_provenance(self):
        qa = [["How was it?", "My shoulder felt fine, and I served well."]]
        result = select(qa, "topical_answers")
        altered = copy.deepcopy(qa); altered[0][1] = "My shoulder was injured."
        self.assertFalse(verify_selected(altered, result))


class DateRuleTests(unittest.TestCase):
    def targets(self):
        return pd.DataFrame([{"match_id": "target", "match_date": "2015-01-10", "gender": "M",
                              "player_a_key": "a", "player_b_key": "b"}])

    def docs(self):
        rows = []
        for iid, player, day in [(1, "a", "2015-01-05"), (2, "a", "2015-01-09"),
                                 (3, "b", "2015-01-07"), (4, "b", "2015-01-12")]:
            rows.append({"interview_id": iid, "gender": "M", "player_key": player, "date": day,
                "date_ordinal": ordinal(day), "answer_words": 100, "source_date_agrees": True,
                "source_played": True, "source_comment": "Completed", "match_id": f"source-{iid}"})
        return pd.DataFrame(rows)

    def test_delay_falls_back_to_older_source(self):
        frame, excluded = asof_interviews(self.targets(), self.docs(), lag=3)
        self.assertEqual(len(frame), 1)
        self.assertEqual(int(frame.a_interview_id.iloc[0]), 1)
        self.assertEqual(int(frame.b_interview_id.iloc[0]), 3)
        self.assertFalse(frame.publication_verified.iloc[0])

    def test_same_day_and_future_sources_never_qualify(self):
        docs = self.docs(); docs.loc[docs.player_key == "a", "date"] = "2015-01-10"
        docs.loc[docs.player_key == "a", "date_ordinal"] = ordinal("2015-01-10")
        frame, excluded = asof_interviews(self.targets(), docs, lag=1)
        self.assertTrue(frame.empty)
        self.assertIn("a_no_eligible", excluded.reason.iloc[0])

    def test_withdrawal_is_not_silently_called_a_played_match(self):
        docs = self.docs(); docs.loc[docs.interview_id == 1, "source_played"] = False
        frame, excluded = asof_interviews(self.targets(), docs, lag=3)
        self.assertTrue(frame.empty)

    def test_next_record_does_not_skip_retirement_to_find_favorable_completed_target(self):
        docs = self.docs().iloc[[0]].copy(); docs["player"] = "Player A"
        matches = pd.DataFrame([
            {"match_id": "next-retired", "match_date": "2015-01-06", "gender": "M", "player_a_key": "a", "player_b_key": "b", "comment": "Retired"},
            {"match_id": "later-completed", "match_date": "2015-01-10", "gender": "M", "player_a_key": "a", "player_b_key": "b", "comment": "Completed"}])
        result = next_match_audit(docs, matches)
        self.assertEqual(result.next_match_id.iloc[0], "next-retired")
        self.assertEqual(result.next_comment.iloc[0], "Retired")


class PreparedStudyTests(unittest.TestCase):
    def test_documented_assumption_passes_without_false_publication_claim(self):
        _, schema, train, dev, basis = load_temporal_candidate(STUDY / "features_topical_answers.csv", STUDY / "schema.json")
        self.assertFalse(schema["strict_publication_temporal_gate_passed"])
        self.assertIn("NOT verified", basis)
        self.assertLess(train.match_date.max(), dev.match_date.min())
        self.assertTrue(set(dev.split) == {"development"})

    def test_tampered_sidecar_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            schema = json.loads((STUDY / "schema.json").read_text())
            (root / "schema.json").write_text(json.dumps(schema))
            (root / "audit.csv").write_text("match_id,match_date\ncorrupt,2030-01-01\n")
            with self.assertRaisesRegex(ValueError, "sidecar hash"):
                load_temporal_candidate(STUDY / "features_topical_answers.csv", root / "schema.json")

    def test_three_way_cutoffs_do_not_depend_on_outcomes(self):
        data = pd.read_csv(STUDY / "metadata.csv")
        expected = assign_split(data)
        data.target = 1 - data.target
        pd.testing.assert_series_equal(expected, assign_split(data))

    def test_reserved_test_was_not_scored(self):
        from research.scripts.tennis.common import OUT
        scores = pd.read_csv(OUT / "a_v1_development_scores.csv")
        self.assertTrue(scores.reserved_test_rows_scored.eq(0).all())
        self.assertEqual(scores.config_sha256.nunique(), 1)


if __name__ == "__main__":
    unittest.main()
