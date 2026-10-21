"""Dataset invariants; no model fitting, package installation or reference edits."""
import json
import unittest

import pandas as pd

from tennis.scripts.common import OUT, PRIVATE, aliases, normalized, text_is_detected
from tennis.scripts.diagnostics import chronological_split
from tennis.scripts.prototypes import derive_matches


class HelperTests(unittest.TestCase):
    def test_future_and_same_day_outcomes_cannot_change_predictors(self):
        rows=[]
        for i,day in enumerate(["2015-01-01","2015-01-02","2015-01-02","2015-01-03"]):
            rows.append({"match_id":str(i),"Date":day,"winner_key":"playera","loser_key":"playerb",
                         "Winner":"Player A","Loser":"Player B","WRank":1,"LRank":2,
                         "gender":"M","Tournament":"Event","Surface":"Hard","Court":"Outdoor",
                         "Round":str(i),"Best of":3,"Series":"ATP250","Tier":None,"Comment":"Completed",
                         "source_file":"fixture","source_row":i})
        raw=pd.DataFrame(rows); expected=derive_matches(raw)
        changed=raw.copy(); affected=changed.Date >= "2015-01-02"
        for a,b in [("winner_key","loser_key"),("Winner","Loser"),("WRank","LRank")]:
            changed.loc[affected,[a,b]]=changed.loc[affected,[b,a]].to_numpy()
        actual=derive_matches(changed)
        mask=expected.match_date <= "2015-01-02"
        pd.testing.assert_frame_equal(expected.loc[mask].drop(columns="target"),actual.loc[mask].drop(columns="target"))

    def test_deterministic_name_aliases(self):
        self.assertIn(normalized("Del Potro J.M."), aliases("Juan Martin Del Potro"))
        self.assertIn(normalized("Tsonga J.W."), aliases("Jo-Wilfried Tsonga"))
        self.assertIn(normalized("Li N."), aliases("Na Li"))

    def test_repetition_does_not_become_text_by_column_name(self):
        self.assertFalse(text_is_detected(["The match is on clay"] * 200))
        self.assertTrue(text_is_detected([f"Example distinct text {n}" for n in range(100)] * 2))


@unittest.skipUnless((PRIVATE / "variant_A.csv").exists(), "Run the local data preparation first")
class PreparedDataTests(unittest.TestCase):
    def test_interviews_are_previous_events_and_day_strict(self):
        df = pd.read_csv(PRIVATE / "variant_A.csv")
        for side in ["a","b"]:
            self.assertTrue((df[f"{side}_text_date"] < df.match_date).all())
            self.assertTrue((df[f"{side}_source_match_id"] != df.match_id).all())
            self.assertFalse((df[f"{side}_interview_id"] == 1499).any())
        self.assertFalse(df.publication_verified.any())

    def test_tactical_snapshot_and_source_dates_precede_target(self):
        df = pd.read_csv(PRIVATE / "variant_B.csv")
        self.assertTrue((pd.to_datetime(df.source_available_at,utc=True) < pd.to_datetime(df.match_date,utc=True)).all())
        for side in ["a","b"]:
            self.assertTrue((df[f"{side}_latest_source_date"] < df.match_date).all())
            self.assertTrue((df[f"{side}_charted_n"] >= 2).all())
            self.assertTrue((df[f"{side}_wide_fraction"] + df[f"{side}_t_fraction"] <= 1.00000001).all())

    def test_feature_partition_and_no_outcome_role_inputs(self):
        for name in ["A","B","C"]:
            schema = json.loads((PRIVATE / f"variant_{name}_schema.json").read_text())
            features = schema["structured_columns"] + schema["text_columns"]
            self.assertFalse(set(features) & set(schema["audit_columns"] + ["target"]))
            self.assertFalse(any("winner" in c or "loser" in c for c in features))
            if name == "C":
                self.assertFalse(set(features) & {"surface","court","level","round","best_of","gender"})

    def test_orientation_rank_assignment_and_label(self):
        raw = pd.read_csv(PRIVATE / "structured_source_rows.csv",low_memory=False).set_index("match_id")
        matches = pd.read_csv(PRIVATE / "matches.csv").sample(100,random_state=10)
        for r in matches.itertuples():
            source = raw.loc[r.match_id]
            self.assertEqual(r.target, int(r.player_a_key == source.winner_key))
            self.assertEqual({r.player_a_key,r.player_b_key},{source.winner_key,source.loser_key})
            for side in ["a","b"]:
                field="WRank" if getattr(r,f"player_{side}_key") == source.winner_key else "LRank"
                expected = pd.to_numeric(source[field],errors="coerce")
                actual=getattr(r,f"{side}_rank")
                if pd.notna(expected) and expected > 0: self.assertEqual(actual,expected)

    def test_chronological_comparison_uses_identical_matches(self):
        b = pd.read_csv(PRIVATE / "variant_B.csv")
        c = pd.read_csv(PRIVATE / "variant_C.csv")
        c = c[c.match_id.isin(b.match_id)]
        btr,bte,_=chronological_split(b); ctr,cte,_=chronological_split(c)
        self.assertEqual(btr.match_id.tolist(),ctr.match_id.tolist())
        self.assertEqual(bte.match_id.tolist(),cte.match_id.tolist())
        self.assertFalse(set(btr.match_id)&set(bte.match_id))

    def test_review_sample_and_manual_labels_agree(self):
        sample=pd.read_csv(OUT/"manual_review_sample.csv")
        labels=pd.read_csv(OUT/"manual_content_review.csv")
        self.assertEqual(set(sample.interview_id),set(labels.interview_id))
        self.assertGreaterEqual(len(labels),30)
        self.assertEqual(sample.date.str[:4].nunique(),9)


if __name__ == "__main__": unittest.main()
