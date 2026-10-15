"""Flatten measured construction and diagnostic facts into a reviewable table."""
import json
import pandas as pd
from research.scripts.tennis.common import OUT, PRIVATE


def main():
    stats=json.loads((OUT/"variant_statistics.json").read_text())
    scores=pd.read_csv(OUT/"diagnostic_scores.csv")
    rows=[];missing=[];text_profiles=[]
    for stat in stats:
        variant=stat["variant"]
        frame=pd.read_csv(PRIVATE/f"variant_{variant}.csv")
        schema=json.loads((PRIVATE/f"variant_{variant}_schema.json").read_text())
        for col in schema["structured_columns"]:
            missing.append({"variant":variant,"column":col,"missing_fraction":float(frame[col].isna().mean())})
        for col in schema["text_columns"]:
            words=frame[col].str.split().str.len()
            text_profiles.append({"variant":variant,"column":col,"rows":len(frame),"unique_texts":frame[col].nunique(),
                "words_min":words.min(),"words_p10":words.quantile(.1),"words_median":words.median(),"words_p90":words.quantile(.9),"words_max":words.max()})
        cohort={"A":None,"B":"B_same415","C":"C_latest1500"}[variant]
        runs=scores[scores.cohort==cohort].set_index("condition") if cohort else pd.DataFrame()
        def score(condition):return float(runs.loc[condition,"auroc"]) if len(runs) else None
        structured,text,joint=score("structured"),score("text_tfidf"),score("joint_tfidf")
        rows.append({"variant":variant,"constructed_rows":stat["rows"],"unique_matches":stat["unique_matches"],
            "players":stat["players"],"structured_columns":stat["structured_columns"],"text_columns":stat["text_columns"],
            "text_unique_by_column":json.dumps(stat["text_unique"]),"median_words_by_column":json.dumps(stat["median_words"]),
            "missing_text_percent_by_column":json.dumps(stat["missing_text_percent"]),
            "date_min":stat["date_min"],"date_max":stat["date_max"],"target_a_win_fraction":stat["a_win_fraction"],
            "temporal_status":"provisional calendar rule; publication gate unresolved" if variant=="A" else "pre-target Git snapshot and earlier dates" if variant=="B" else "known context; earlier-day histories",
            "publication_verified_A_rows":0 if variant=="A" else None,"redistribution_cleared":False,
            "diagnostic_cohort":cohort,"diagnostic_learner":"CPU logistic regression + TF-IDF; NOT MulTaBench" if cohort else None,
            "diagnostic_n_train":int(runs.iloc[0].n_train) if len(runs) else None,
            "diagnostic_n_test":int(runs.iloc[0].n_test) if len(runs) else None,
            "structured_auroc":structured,"text_tfidf_auroc":text,"joint_tfidf_auroc":joint,
            "joint_gain":joint-max(structured,text) if joint is not None else None,
            "frozen_E5_auroc":None,"TAR_auroc":None,"TAR_gain":None,
            "official_training_pool_text_gate":"not run: availability blocked" if variant=="A" else "does not trigger text in measured training pools",
            "recommendation":"investigate rights/timing next; no final commitment" if variant=="A" else "deprioritize main NLP construction" if variant=="B" else "keep as representation control only"})
    pd.DataFrame(rows).to_csv(OUT/"variant_comparison.csv",index=False)
    pd.DataFrame(missing).to_csv(OUT/"feature_missingness.csv",index=False)
    pd.DataFrame(text_profiles).to_csv(OUT/"prototype_text_profiles.csv",index=False)


if __name__ == "__main__":main()
