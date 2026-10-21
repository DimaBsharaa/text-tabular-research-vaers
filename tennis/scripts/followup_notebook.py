"""Execute the small interview follow-up notebook from saved metadata only."""
from tennis.scripts.notebooks import code, markdown, save_execute
from tennis.scripts.common import OUT


def main():
    if (OUT / "a_v1_summary.json").exists():
        from tennis.scripts.a_review_outputs import collect, notebook
        notebook(*collect())
        return
    cells = [markdown('''
        # Tennis interview validation: publication, source terms and split risks

        Executed locally October 15, 2026. This is the next bounded investigation of variant A.
        It contains actual metadata audits and coverage counts, with **no model fitting**.
        The official benchmark and notebook 02 remain unchanged. No transcript text is published here.

        Read [the report](../reports/tennis_interview_followup.md) for source links and the decision.
        Reproduce the saved-data audit: `python3 -m tennis.scripts.interview_followup`,
        then `python3 -m tennis.scripts.followup_notebook`.
    '''), code('''
        import json
        from pathlib import Path
        import sys
        project = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "tennis/scripts").is_dir())
        if str(project) not in sys.path: sys.path.insert(0, str(project))
        import pandas as pd
        import matplotlib.pyplot as plt
        from IPython.display import display
        from tennis.scripts.common import OUT
        audit = json.loads((OUT / "interview_followup_audit.json").read_text())
        publisher = pd.read_csv(OUT / "interview_publisher_validation.csv")
        lags = pd.read_csv(OUT / "interview_lag_sensitivity.csv")
        splits = pd.read_csv(OUT / "interview_split_risks.csv")
        coverage = pd.read_csv(OUT / "interview_alternative_source_coverage.csv")
        display(pd.DataFrame([{
            "Calendar-candidate rows": audit["provisional_A_rows"],
            "Strict publication-verified rows": audit["strict_publication_verified_A_rows"],
            "Raw-text redistribution cleared": audit["raw_text_redistribution_cleared"],
            "New learner fits": 0,
        }]))
    '''), markdown('''
        **Interpretation:** the 4,052 count measures provisional row coverage. Publication evidence and reuse permission
        remain unresolved. This audit does not convert an event date into a first-publication time.
    '''), code('''
        display(publisher[["interview_id", "publisher_id", "player", "corpus_event_date", "purpose",
            "date_agrees", "player_agrees", "first_long_answer_prefix_matches",
            "matched_answer_prefixes", "tested_answer_prefixes", "historical_availability_verified"]])
        correct = publisher[publisher.purpose != "deliberately_wrong_same_day_interview"]
        print("Correct-page substantial answer prefixes:", int(correct.matched_answer_prefixes.sum()),
              "/", int(correct.tested_answer_prefixes.sum()))
        print("Recognized publication metadata tags:", int(publisher.publication_meta_tag_found.sum()))
        print("Archive probe:", audit["archive_probe"])
    '''), markdown('''
        **Finding:** nine content corroborations support the existence of the real language source.
        The same-player/same-day negative control fails, so those two keys alone are insufficient.
        Wozniak 1499 remains excluded because its source result is wrong; Serena 5019 is a withdrawal interview.
        Prefix agreement is not full transcript identity or proof of historical public availability.
        The one-URL archive probe returned no usable evidence, not proof that historical captures do not exist.
    '''), code('''
        display(pd.DataFrame([
            {"Source": "Tennis-Data", "Observed": "Current homepage restricts data-training products using bots/scrapers/AI",
             "Next requirement": "Clarify academic training and derived-data redistribution"},
            {"Source": "Cornell / ASAP transcripts", "Observed": "No explicit transcript redistribution grant found in inspected sources",
             "Next requirement": "Clarify rights and historical publication evidence"},
            {"Source": "Sackmann archive", "Observed": "Preserved author README and archive LICENSE state CC BY-NC-SA 4.0",
             "Next requirement": "Credit source; resolve date/pairing validity and separate text rights"},
        ]))
        print("Original ATP/WTA GitHub API status:", audit["sackmann_upstream_github_http_status"])
        print("Pinned mirror:", "8ac86f7c62889b5d5c4cfec4c6403edcb8c5d993")
        print("Recorded new source body bytes:", audit["new_download_body_bytes"])
    '''), markdown('''
        **Finding:** the structured-source restriction is concrete new evidence. A free download is not treated as
        permission for our training/public release. The mirror provides a clearer noncommercial structured-data license,
        but its tournament-week dates do not supply exact match days. No message has been sent to any source.
    '''), code('''
        ax = lags.plot.bar(x="minimum_calendar_gap_days", y="candidate_rows", figsize=(8, 3.5),
                          color="#4f86a8", legend=False, rot=0)
        for index, value in enumerate(lags.candidate_rows):
            ax.text(index, value + 50, f"{value:,}", ha="center")
        ax.set(xlabel="Minimum calendar-day gap on both sides", ylabel="Provisional target matches",
               title="Calendar-delay sensitivity; strict verified availability remains zero", ylim=(0, 4600))
        plt.tight_layout(); plt.show()
        display(lags)
    '''), markdown('''
        **Insight:** a seven-day gap retains 2,056 of 4,052 candidates and discards many fresh interviews.
        A delay buffer is an assumption with a coverage cost. It supplies no missing historical upload timestamp.
    '''), code('''
        display(splits)
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
        labels = ["Chronological", "Random diagnostic"]
        axes[0].bar(labels, splits.test_fraction_with_training_interview * 100, color=["#4f86a8", "#c47a55"])
        axes[0].set(ylabel="Percent of test rows", title="Test rows reusing a training interview", ylim=(0, 100))
        axes[1].bar(labels, splits.heldout_targets_referenced_in_training_source_matches, color=["#4f86a8", "#c47a55"])
        axes[1].set(ylabel="Distinct held-out targets", title="Training interview sources that are test targets")
        plt.tight_layout(); plt.show()
        display(pd.read_csv(OUT / "interview_split_reference_examples.csv"))
        display(pd.DataFrame([audit["source_target_disjoint_coverage_control"]]))
    '''), markdown('''
        **Insight:** random splitting yields 82.1% interview reuse and 242 held-out target matches referenced by
        later training interviews; chronological splitting yields 7.9% reuse and zero reverse-time references.
        Genuine earlier-document reuse is not automatically leakage. A source reference flags possible exposure
        to a held-out outcome, without proving every clipped text states the result or quantifying model inflation.
        A fixed source/target-disjoint coverage control leaves 1,265 rows but does not resolve other dependencies.

        These are 80/20 metadata diagnostics, **not runs of the official evaluator**. Keep official benchmark scores
        separate from a chronological assessment of rolling future prediction. No benchmark code is changed.
    '''), code('''
        display(coverage)
        first = coverage[coverage.scope == "first_round_completed"]
        grouped = first.groupby("margin_days_before_week_anchor").both_recent_interviews.sum()
        ax = grouped.plot.bar(figsize=(6, 3.5), color="#4f86a8", rot=0)
        for index, value in enumerate(grouped): ax.text(index, value + 1, str(value), ha="center")
        ax.set(xlabel="Days before tournament-week anchor", ylabel="Both-player interview candidates",
               title="Alternative structured source: 2015 first-round coverage", ylim=(0, 65))
        plt.tight_layout(); plt.show()
        display(pd.read_csv(OUT / "interview_alternative_source_sample_metadata.csv"))
    '''), markdown('''
        **Insight:** the two 2015 files yield only 54 first-round completed candidates with two recent interviews
        before the week anchor; a seven-day-earlier scenario leaves 48. All-round conditional-pairing coverage is 423
        at the anchor. Neither count is certified: event/draw publication and transcript availability remain unresolved.
        The study did not infer match dates, build a replacement training CSV or extrapolate one year to nine.

        **Next decision:** review the report and unsent clarification drafts. Resolve source rights, availability and
        the acceptable evaluation claim before training A. If those cannot be settled promptly, examine another domain
        with explicit dataset rights. No final dataset has been selected; no new model scores, commit or push exist.
    ''')]
    save_execute("03_tennis_interview_validation", cells)


if __name__ == "__main__":
    main()
