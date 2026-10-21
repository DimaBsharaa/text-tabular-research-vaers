"""Build and execute two ordinary-Python research notebooks; no GPU or network."""
import base64
import contextlib
import io
import json
import os
from pathlib import Path
import textwrap

os.environ.setdefault("MPLCONFIGDIR","/tmp/text-tabular-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import IPython.display

from tennis.scripts.common import ROOT


def markdown(s):return {"cell_type":"markdown","metadata":{},"source":textwrap.dedent(s).strip()+"\n"}
def code(s):return {"cell_type":"code","metadata":{},"source":textwrap.dedent(s).strip()+"\n","execution_count":None,"outputs":[]}


def save_execute(name,cells,output_path=None):
    nb={"nbformat":4,"nbformat_minor":5,"metadata":{"kernelspec":{"name":"python3","display_name":"Python 3","language":"python"},"language_info":{"name":"python"},
        "research_execution":{"network":False,"gpu_training":False,"official_benchmark_modified":False}},"cells":cells}
    ns={};count=0
    old_display,old_show=IPython.display.display,plt.show
    try:
        for i,cell in enumerate(cells):
            cell["id"]=name[:2]+"-research-"+str(i)
            if cell["cell_type"]!="code":continue
            count+=1;cell["execution_count"]=count;outputs=[];stream=io.StringIO()
            def flush():
                if stream.getvalue():outputs.append({"output_type":"stream","name":"stdout","text":stream.getvalue()})
                stream.seek(0);stream.truncate(0)
            def display(*objects,**kwargs):
                flush()
                for obj in objects:
                    data={"text/plain":str(obj)}
                    if hasattr(obj,"_repr_html_"):data["text/html"]=obj._repr_html_()
                    outputs.append({"output_type":"display_data","metadata":{},"data":data})
            def show(*args,**kwargs):
                flush()
                for n in plt.get_fignums():
                    buf=io.BytesIO();plt.figure(n).savefig(buf,format="png",dpi=105,bbox_inches="tight")
                    outputs.append({"output_type":"display_data","metadata":{},"data":{"image/png":base64.b64encode(buf.getvalue()).decode(),"text/plain":"Executed research figure"}})
                plt.close("all")
            IPython.display.display,plt.show=display,show;ns["display"]=display
            with contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):exec(compile(cell["source"],name,"exec"),ns)
            flush();cell["outputs"]=outputs
    finally:IPython.display.display,plt.show=old_display,old_show
    path=Path(output_path) if output_path else ROOT/"tennis/notebooks"/(name+".ipynb")
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(nb,indent=1,ensure_ascii=False)+"\n")
    print(name,count,"executed cells")


def setup():
    return code('''
        import json
        from pathlib import Path
        import sys
        project = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "tennis/scripts").is_dir())
        if str(project) not in sys.path: sys.path.insert(0, str(project))
        import pandas as pd
        import matplotlib.pyplot as plt
        from IPython.display import display
        from tennis.scripts.common import OUT, PRIVATE
        audit = json.loads((OUT / "cornell_audit.json").read_text())
        joins = json.loads((OUT / "join_summary.json").read_text())
        extra = json.loads((OUT / "additional_audit.json").read_text())
        meta = pd.read_csv(OUT / "cornell_interview_metadata.csv")
    ''')


def main():
    n3=[markdown('''
        # Cornell tennis interviews: source audit and EDA

        Executed locally, October 15, 2026. Source: [ConvoKit tennis](https://convokit.cornell.edu/documentation/tennis.html)
        and [the original Cornell release](https://www.cs.cornell.edu/~liye/tennis.html).
        We inspect the actual legacy native JSON, without installing ConvoKit/SpaCy or extracting parsed annotations.
        No learner training runs here. No raw transcripts are published in notebook outputs.

        Reproduce from the repository root: `python3 -m tennis.scripts.fetch`, then `audit`, `prototypes`, `additional`, `diagnostics`.
        These scripts keep raw data and full candidate CSVs in the external `tennis-research-cache` folder.
        The current setup needs existing numpy/pandas/sklearn/matplotlib/openpyxl/IPython/psutil.
    '''),setup(),code('''
        display(pd.DataFrame({"Count": {"Players":audit["players"],"Users including REPORTER":audit["speakers_including_reporter"],
            "Utterances":audit["utterances"],"Question-answer conversations":audit["qa_conversations"],"Interview records":audit["interviews"],
            "Accepted distinct sporting matches":joins["unique_interview_source_matches"]}}))
        display(pd.DataFrame({"Interviews":audit["gender_interviews"],"Players":audit["gender_players"]}))
        print("Legacy keys: user / root / reply-to / timestamp; interview key = conversations[root]['match_id'].")
        print("Date mismatches:",audit["original_vs_utterance_date_mismatches"],"Answer reconstruction mismatches:",audit["answer_reconstruction_mismatches"])
    '''),markdown('''
        `match_id` in this source identifies an interview, not a unique sporting match.
        Zero reconstruction/date mismatches verify conversion consistency, not transcript-to-match truth or original online availability.
    '''),code('''
        yearly=meta.groupby(["year","gender"]).size().unstack()
        yearly.plot.bar(stacked=True,figsize=(9,3),color=["#c0789c","#4f86a8"])
        plt.ylabel("Interview records");plt.title("Interview coverage by year and gender");plt.tight_layout();plt.show()
    '''),markdown('''
        **Insight:** Both genders span all nine years, but coverage is uneven. Counts measure interviews, not all matches played or equally sampled players.
        Separate men's/women's summaries are essential before interpreting pooled prediction performance.
    '''),code('''
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        meta.player.value_counts().head(15).sort_values().plot.barh(ax=axes[0],color="#4f86a8")
        meta.stage.value_counts().plot.bar(ax=axes[1],color="#4f86a8",rot=60)
        axes[0].set_title("Most interviewed players");axes[1].set_title("Interview stage distribution")
        plt.tight_layout();plt.show()
        print("Top-ten player share:",round(extra["corpus_top10_player_interview_share"]*100,1),"%")
        print("Grand Slam share:",round(extra["corpus_grandslam_interview_share"]*100,1),"%")
        print("Winning-player source interview share:",round(extra["corpus_source_winner_interview_share"]*100,1),"%")
    '''),markdown('''
        **Insight:** Stars account for a substantial share of source documents and 68% of interviews follow wins.
        The final A/B label can still be balanced because its orientation is independent of the winner.
        Interview selection bias and target class balance are different questions.
    '''),code('''
        fig,axes=plt.subplots(1,2,figsize=(11,3))
        meta.answer_words.plot.hist(bins=45,ax=axes[0],color="#4f86a8")
        meta.qa_pairs.plot.hist(bins=30,ax=axes[1],color="#4f86a8")
        axes[0].set(xlabel="Player-answer words per complete interview",title="Answer length")
        axes[1].set(xlabel="Question-answer pairs per interview",title="Interview structure")
        plt.tight_layout();plt.show()
        display(pd.DataFrame([{"measure":"Answers ≤5 words","count":audit["answers_at_most_5_words"]},
            {"measure":"Exact narrow boilerplate","count":audit["exact_boilerplate_answers"]},
            {"measure":"Empty answers","count":audit["empty_answers"]},
            {"measure":"Duplicate full answer documents","count":audit["duplicate_full_answer_interviews"]},
            {"measure":"Repeated long sections (≥30 words)","count":audit["duplicate_long_answer_sections_beyond_first"]}]))
    '''),markdown('''
        **Insight:** Median full answers total 769 words; most exceed 512 words, which is not the same as the E5 token limit.
        A fixed passage policy is necessary. Many short answers need their question context; removing every short answer could remove useful evidence.
        Exact-duplicate checks do not establish absence of near-duplicates.
    '''),code('''
        reviews=pd.read_csv(OUT/"manual_content_review.csv")
        sample=pd.read_csv(OUT/"manual_review_sample.csv")
        reviewed=sample.merge(reviews,on="interview_id",validate="one_to_one")
        display(reviewed[["interview_id","player","date","gender","result","specific_context_0_to_2","review_note"]])
        display(reviews.specific_context_0_to_2.value_counts().rename_axis("Subjective specificity").to_frame("Reviewed interviews"))
        print("Full raw answers remain private at:",PRIVATE/"manual_review_texts.json")
        print("Automated lexical flags are NOT human labels:",audit["keyword_flags_NOT_human_labels"])
    '''),markdown('''
        **Insight:** The 36 reviewed answer sets cover every year/gender/outcome stratum but are restricted to 70–350 words.
        Human review found 22 detailed, 9 somewhat contextual and 5 generic/off-topic examples. This is not a population-prevalence estimate.
        Interview 1499 explicitly describes a loss despite a source win label; it is excluded from construction.
    '''),code('''
        linked=pd.read_csv(OUT/"cornell_match_join_audit.csv")
        display(linked.join_status.value_counts().to_frame("Interviews"))
        display(linked[linked.join_status!="matched"].head(12))
        display(pd.read_json(OUT/"publisher_spot_checks.json"))
        print("Original interview date range:",audit["date_min"],"to",audit["date_max"])
        print("Missing original dates:",audit["missing_dates"],"Unknown opponents:",audit["missing_opponent"])
    '''),markdown('''
        **Insight:** 6,366/6,467 records have accepted deterministic links, but 98.44% link coverage is not independently measured accuracy.
        The source originally used Tennis-Data too. Two publisher-content checks corroborate dated transcripts; the same-day wrong semifinal is rejected.
        No first-publication times or open transcript redistribution license were established.

        **Gate decision:** genuine language and thousands of provisional calendar candidates justify a focused source-rights/timing check.
        A is not yet cleared for training as a strict historical-availability task. See `tennis_cornell_eda.md` and notebook 04.
    ''')]
    save_execute("01_tennis_interviews_eda",n3)
    n4=[markdown('''
        # Tennis constructions A/B/C: real prototypes and CPU diagnostics

        Executed locally, October 15, 2026. This notebook reuses the source audit and leaves the successful Colab notebook 02 unchanged.
        Scores here are **TF-IDF/logistic regression**, not MulTaBench or E5/TAR. No GPU training is launched.
        Full text CSVs and schemas stay outside Git; samples below preserve source identities without publishing interview answers.
    '''),setup(),code('''
        stats=pd.read_json(OUT/"variant_statistics.json")
        display(stats[["variant","rows","unique_matches","players","structured_columns","text_columns","date_min","date_max","text_unique","median_words","duplicate_text_fraction"]])
        stats.set_index("variant")["rows"].plot.bar(figsize=(6,3),color="#4f86a8",rot=0,logy=True)
        plt.ylabel("Constructed rows (log scale)");plt.title("Coverage, with A publication gate unresolved");plt.tight_layout();plt.show()
    '''),markdown('''
        **Insight:** A has 4,052 provisional two-interview rows; B has 415 snapshot-valid tactical rows; C has 44,359 verbalization rows.
        These are different populations. A has zero independently publication-certified rows, so its numerical size does not mean it is cleared for modeling.
        B's fixed historical snapshot is much smaller than today's retrospective chart archive.
    '''),code('''
        for variant in ["A","B","C"]:
            print("Variant",variant)
            df=pd.read_csv(PRIVATE/f"variant_{variant}.csv")
            manifest=json.loads((PRIVATE/f"variant_{variant}_schema.json").read_text())
            cols=["match_id","match_date","player_a","player_b","a_rank","b_rank","target"]
            source_cols=[c for c in df if c.endswith("interview_id") or c.endswith("source_match_ids") or c=="source_available_at"]
            examples=df[cols+source_cols].head(3).copy()
            examples["prediction_cutoff"]=examples.match_date.map(lambda day:"Before reported day "+day+"; same-day evidence excluded")
            display(examples)
            print("Structured:",manifest["structured_columns"])
            print("Text:",manifest["text_columns"])
            print("Private 50-row CSV:",PRIVATE/f"variant_{variant}_sample50.csv")
            if variant=="A":print("Interview answers withheld from public notebook outputs pending rights/timing review.")
            elif variant=="B":print("One measured-template example:",df.player_a_text.iloc[0])
            else:print("One context-verbalization example:",df.context_text.iloc[0])
    '''),markdown('''
        A uses previous player answers, clipped to the first 300 words per side, with questions retained separately in the private source.
        B's sentences describe wide/T/middle serve-placement frequencies from old charts; they do not claim forehand quality, injuries or psychology.
        C moves gender/surface/court/level/round/best-of into text and removes those fields from structured input.
        Source IDs, match dates and winner/loser source roles never enter model predictors.
        A raw point-log prefix contains 292 complete points from two matches; its 8 nonempty Notes are terse challenge annotations,
        not a verified analysis corpus. This bounded prefix cannot estimate the full project's note quality or prevalence.
    '''),code('''
        missing=pd.read_csv(OUT/"feature_missingness.csv")
        display(missing[missing.missing_fraction>0].sort_values(["variant","missing_fraction"],ascending=[True,False]))
        display(pd.read_csv(OUT/"prototype_text_profiles.csv"))
        print("No whole-dataset numerical imputation was applied; diagnostic imputation fits training data only.")
    '''),code('''
        a=pd.read_csv(PRIVATE/"variant_A.csv")
        gaps=pd.DataFrame({side:(pd.to_datetime(a.match_date)-pd.to_datetime(a[f"{side}_text_date"])).dt.days for side in ["a","b"]})
        gaps.plot.hist(bins=25,alpha=.6,figsize=(7,3))
        plt.xlabel("Previous interview age in days");plt.title("A: 90-day policy and strict earlier-day rule");plt.tight_layout();plt.show()
        print("No/one/two recent sides before completed-target filtering:",joins["a_interview_sides_coverage"])
        print("Player-centric unique next-match candidates:",extra["playercentric_unique_target_matches"])
    '''),markdown('''
        **Insight:** Median source age is 19 days, and a fixed 90-day limit prevents indefinite profile reuse.
        This validates the implemented day rule, not publication availability. The player-centric alternative changes coverage and must deduplicate target matches.
        Neither larger candidate count resolves that timing/licensing gate.
    '''),code('''
        results=pd.read_csv(OUT/"diagnostic_scores.csv")
        display(results[["cohort","condition","auroc","n_train","n_test","cutoff","runtime_seconds","rss_mb_after_fit"]].round(4))
        common=results[results.cohort.isin(["B_same415","C_same415"])]
        common.pivot(index="cohort",columns="condition",values="auroc").plot.bar(figsize=(12,4),rot=0)
        plt.ylim(0,1);plt.ylabel("Test AUROC");plt.title("Same 415 matches: CPU TF-IDF/logistic diagnostics");plt.tight_layout();plt.show()
    '''),markdown('''
        **Insight:** Neither B nor C improves the joint score over its stronger unimodal baseline on this split.
        B's raw-numeric tactical control performs similarly to its verbalized joint condition. C's full-structured-context control equals B's structured baseline.
        This is a small chronological diagnostic; it neither proves impossibility nor supplies official curation evidence.
        The separate C-latest1500 scores use another population and should not be ranked directly against B as a fair construction comparison.
    '''),code('''
        display(results[["cohort","condition","detected_text_on_fit_pool"]].drop_duplicates())
        print("All A/B/C calendar intersection:",joins["common_ABC_rows"])
        print("A strict publication-verified rows:",joins["a_strict_publication_verified_rows"])
        print("TAR scores: unavailable; no GPU diagnostic was run.")
        print("Local predictions and chronological split IDs are in",OUT)
    '''),markdown('''
        **Insight:** The pinned detector requires at least 100 distinct strings or a uniqueness ratio ≥0.8.
        B/C pass on their full populations but fail on the measured chronological training subsets.
        Our explicitly forced TF-IDF diagnostic is therefore not proof the official pipeline would use E5.
        We do not modify the detector or add artificial IDs. A's timing gate remains unresolved and A has no model score.
    '''),markdown('''
        ## Prepared A100 diagnostic, deferred pending review

        Reuse notebook 02's setup cells and isolated `/content/multabench-env` environment; do not replace its working smoke-test CSV or training cells.
        After review, upload the relevant private CSV/schema and our new `tennis/scripts/tennis_temporal_pilot.py` (plus existing `colab_runner.py`) to your private Colab project.
        No new files are available from GitHub until a push is explicitly approved.

        First check, without training:

        ```python
        command([str(BENCH_PY), "-u", str(PROJECT / "tennis/scripts/tennis_temporal_pilot.py"),
                 "--reference", str(REFERENCE), "--csv", "/content/tennis-private/variant_A.csv",
                 "--schema", "/content/tennis-private/variant_A_schema.json", "--output", str(OUTPUT / "interview-pilot")])
        ```

        Current A will stop at publication validation. Current B/C will stop at actual official fitting-pool text detection.
        Do not spend GPU time on those known blockers. If the gates are genuinely resolved and GPU work is approved, append `--run`.
        That uses unchanged official LightGBM/E5/TAR/metrics with four conditions, a chronological outer holdout and two TAR epochs/batch 32.
        It is a diagnostic departure from the five-random-holdout protocol; results must keep that label.

        **Recommendation:** prioritize A's source availability/rights and a small independent transcript-identity audit.
        B is currently weak as the main NLP direction; C is a control. If A's gates remain unresolved, pivot away from tennis rather than force passage.
    ''')]
    n4 += [markdown('''
        ## Practical review of interview candidate A

        The local review export separates model inputs, audit metadata and chronological roles. Its text-bearing files
        and an executed workbench with three concise real examples stay outside Git in
        `tennis-research-cache/derived/candidate_A_v0_review/` (a sibling cache of this repository).
        This section displays metadata only. No new sources, model fits or outreach are part of this step.
    '''),code('''
        review_file = OUT / "interview_candidate_review.json"
        if review_file.exists():
            review = json.loads(review_file.read_text())
            print("Private workbench:", PRIVATE / "candidate_A_v0_review/candidate_workbench.ipynb")
            print("Private feature CSV:", PRIVATE / "candidate_A_v0_review/features.csv")
            display(pd.DataFrame([{"rows":review["rows"], "structured":review["structured_columns"],
                "text":review["text_columns"], "unique_interviews":review["unique_interviews"], "new_model_fits":0}]))
            display(pd.read_csv(OUT / "interview_candidate_split_plan.csv"))
            clip = pd.read_csv(OUT / "interview_candidate_clipping_summary.csv")
            display(clip)
            ax = clip.set_index("topic")[["documents_with_cue_in_first300", "documents_with_cue_only_after_first300"]].plot.bar(
                stacked=True,figsize=(8,3),rot=0,color=["#4f86a8","#c47a55"])
            ax.set(ylabel="Unique training interviews",title="Does the current first-300-word input retain lexical cues?")
            ax.legend(["Cue retained", "Cue only after first 300 words"])
            plt.tight_layout();plt.show()
        else:
            print("Prepare this optional local review: python3 -m tennis.scripts.prepare_interview_review")
    '''),markdown('''
        **Insight:** of 1,296 training-source interviews with a condition-related lexical cue, 541 (41.7%) have no such cue
        in the first 300 words. These counts are screening indicators, not verified injury labels or model results.
        Fixed clipping may discard potentially relevant language. Passage-policy decisions should use training data,
        preserve provenance and respect the official token limit; multi-chunk embedding pooling is not silently added.

        The review keeps 16 structured columns, two real-text columns and the outcome; `match_id` is audit-only.
        The chronological plan has 2,430 training, 809 development and 813 reserved-test rows, with whole dates kept together.
        This fixes the reverse-time interview-source route across those partitions but does not certify historical publication.
        Source-use and availability statuses remain explicit. No email or other outreach is part of the current work.
    ''')]
    n4.insert(1,markdown('''
        **Current A follow-up:** this notebook preserves the original A/B/C and v0 review results.
        Updated [notebook 05](03_tennis_interview_validation.ipynb) compares six A passage policies, constructs 3,627
        rows under a declared three-day calendar-delay assumption, runs training/development CPU diagnostics and records
        the full leakage audit. Unknown upload times are documented assumptions rather than an automatic research blocker.
        The official implementation and notebook 02 remain unchanged; no outreach or push occurred.
    '''))
    save_execute("02_tennis_dataset_prototypes",n4)


if __name__ == "__main__":main()
