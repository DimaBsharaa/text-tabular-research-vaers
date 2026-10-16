"""Read-only audit of exported v7 diagnostic ZIPs; no extraction, fitting or network.

Checks recorded partitions, provenance and result arithmetic. Recorded partitions are
adapter reconstructions, not instrumented traces of tensors actually used at runtime.
This cannot audit report contents or duplicate clinical events without source data.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import itertools
import json
import math
from pathlib import Path
import re
import statistics
import zipfile

PIN = "3bb95079a6146ed35d45187e98bcad5b063273a1"
STRUCTURED = "age_years sex state onset_days vax_type vax_manu vax_dose_series vax_route vax_site n_vaccines admin_by report_year".split()
TEXT = "history current_illness other_meds allergies narrative_onset".split()
CONDITIONS = ["structured_only", "text_only", "joint_frozen", "joint_tar"]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def audit(paths, tree_path, source_dir=None):
    tree = json.loads(Path(tree_path).read_text())
    expected_blobs = {r["path"]: r["sha"] for r in tree["tree"] if r["type"] == "blob"}
    checks, runs, archives = [], [], []

    def check(scope, name, passed, detail=""):
        checks.append(dict(scope=scope, check=name, status="PASS" if passed else "FAIL", detail=detail))

    check("global", "independent_tree_pin", tree["sha"] == PIN and not tree.get("truncated"))
    outer_by_fold, fit_by_model_fold, proof_sha_by_path = {}, {}, {}
    source_checks = {}
    if source_dir:
        for p in sorted(Path(source_dir).rglob("*.py")):
            relative = p.relative_to(source_dir).as_posix()
            raw = p.read_bytes()
            blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
            check(relative, "downloaded_source_matches_pinned_git_blob", expected_blobs.get(relative) == blob)
            source_checks[relative] = digest(raw)

    for path in map(Path, paths):
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            scope = path.name
            check(scope, "unique_zip_member_names", len(names) == len(set(names)))
            check(scope, "zip_crc_integrity", z.testzip() is None)
            plan = json.loads(z.read("plan.json"))
            check(scope, "no_process_errors", json.loads(z.read("process_errors.json")) == [])
            roots = sorted(n[:-len("/result.json")] for n in names if n.endswith("/result.json"))
            expected_jobs = set(itertools.product(plan["models"], plan["folds"], plan["conditions"]))
            observed_jobs = []
            check(scope, "four_conditions_requested", plan["conditions"] == CONDITIONS)
            plan_id = digest(json.dumps(plan, sort_keys=True).encode())[:16]
            check(scope, "plan_hash_matches_archive_name", f"_{plan_id}_" in path.name)
            archive_csv = list(csv.DictReader(io.StringIO(z.read("results.csv").decode())))
            archive_by_id = {r["run_id"]: r for r in archive_csv}
            check(scope, "csv_has_unique_run_ids", len(archive_by_id) == len(archive_csv))
            archives.append(dict(name=scope, sha256=digest(path.read_bytes()), runs=len(roots),
                                 dataset_or_schema_present=any(Path(n).name in {"schema.json", "vaers_v7.csv", "audit.txt"} for n in names)))

            for root in roots:
                cfg = json.loads(z.read(root + "/config.json"))
                result = json.loads(z.read(root + "/result.json"))
                splits = json.loads(z.read(root + "/splits.json"))
                proof = json.loads(z.read(root + "/reference_proof.json"))
                summary = result["official_summary"]
                key = (cfg["model_key"], cfg["fold"], cfg["condition"])
                observed_jobs.append(key)
                s = f"{cfg['model_key']}/{cfg['fold']}/{cfg['condition']}"
                check(s, "completed", result["status"] == "ok")
                check(s, "configuration_identity", result["config"] == cfg and result["run_id"] == root)
                check(s, "run_id_hash", digest(json.dumps(cfg, sort_keys=True).encode())[:24] == root)
                common = {k: v for k, v in cfg.items() if k not in {"model_key", "condition", "fold", "columns", "e5_train_kwargs", "research_commit", "experiment_id"}}
                check(s, "experiment_id_hash", digest(json.dumps(common, sort_keys=True).encode())[:24] == cfg["experiment_id"])
                check(s, "matches_plan", all(cfg[k] == plan[k] for k in ["csv_sha256", "benchmark_commit", "adapter_sha256", "environment_sha256", "mode"]))
                check(s, "environment_bytes_hash", digest(z.read(root + "/pip_freeze.txt")) == cfg["environment_sha256"])
                cols = STRUCTURED if cfg["condition"] == "structured_only" else TEXT if cfg["condition"] == "text_only" else STRUCTURED + TEXT
                check(s, "predictor_whitelist_and_no_target_or_id", cfg["columns"] == cols)
                expected_text = [] if cfg["condition"] == "structured_only" else sorted(TEXT)
                check(s, "recorded_text_detection", sorted(splits["detected_text_columns"]) == expected_text)
                check(s, "dataset_and_score_identity", summary["dataset"] == "BIN_TEXT_VAERS_SERIOUS_V7" and
                      math.isfinite(summary["test_score"]) and 0 <= summary["test_score"] <= 1 and
                      summary["metrics_dict"]["metrics"]["auc"] == summary["test_score"])
                row = archive_by_id.get(root, {})
                check(s, "exported_score_matches_json", bool(row) and float(row["auroc"]) == summary["test_score"] and
                      (row["model_key"], int(row["fold"]), row["condition"]) == key)
                raw_sets = {k: set(map(str, splits[k])) for k in ["outer_train", "outer_test", "learner_fit", "learner_validation"]}
                check(s, "unique_partition_ids", all(len(v) == len(splits[k]) for k, v in raw_sets.items()))
                tr, te, fit, val = (raw_sets[k] for k in ["outer_train", "outer_test", "learner_fit", "learner_validation"])
                check(s, "recorded_outer_train_test_disjoint", not tr & te)
                check(s, "recorded_learner_partition", not fit & val and fit | val == tr)
                check(s, "summary_split_sizes", summary["n_train"] == len(tr) and summary["n_test"] == len(te))
                outer = (tuple(splits["outer_train"]), tuple(splits["outer_test"]))
                check(s, "same_outer_rows_across_conditions_and_learners", outer_by_fold.setdefault(cfg["fold"], outer) == outer)
                inner = (tuple(splits["learner_fit"]), tuple(splits["learner_validation"]))
                check(s, "same_learner_partition_across_conditions", fit_by_model_fold.setdefault(key[:2], inner) == inner)
                tar = cfg["condition"] == "joint_tar"
                check(s, "tuning_switches", summary["tune_e5"] == tar and not summary["tune_dino"] and
                      summary["target_override"] is None and summary["pca_components"] == 30 and not summary["no_pca"])
                log = z.read(f"{cfg['model_key']}_{cfg['condition']}_{cfg['fold']}.log").decode(errors="replace")
                feature_count = len(STRUCTURED) if cfg["condition"] == "structured_only" else 30 * len(TEXT) if cfg["condition"] == "text_only" else len(STRUCTURED) + 30 * len(TEXT)
                check(s, "logged_feature_count", bool(re.search(rf"🔭 {feature_count} features", log)))
                counts = {k: len(v) for k, v in raw_sets.items()}
                if tar:
                    et, ev = (set(map(str, splits[k])) for k in ["encoder_train", "encoder_validation"])
                    check(s, "unique_encoder_ids", len(et) == len(splits["encoder_train"]) and len(ev) == len(splits["encoder_validation"]))
                    check(s, "recorded_encoder_partition", not et & ev and et | ev == fit)
                    check(s, "recorded_encoder_excludes_test_and_learner_validation", not (et | ev) & (te | val))
                    passages = re.findall(r"\[Dataset\] train: n=(\d+), val: n=(\d+)", log)
                    check(s, "logged_passages_match_encoder_rows_times_five", passages == [(str(len(et) * len(TEXT)), str(len(ev) * len(TEXT)))])
                    check(s, "official_tar_schedule", cfg["e5_train_kwargs"] == dict(lora_rank=16, text_layers=3, learning_rate=0.0001, epochs=50, patience=3, weight_decay=0.001, batch_size=256))
                    counts.update(encoder_train=len(et), encoder_validation=len(ev))
                else:
                    check(s, "no_tar_configuration_or_training_log", cfg["e5_train_kwargs"] is None and "Finetuning single E5" not in log)
                files = proof["tracked_files"]
                by_path = {r["path"]: r for r in files}
                check(s, "reference_inventory_matches_independent_tree", proof["commit"] == PIN and
                      proof["repository"] == "https://github.com/alanarazi7/MulTaBench" and len(by_path) == len(files) and
                      {p: r["git_blob"] for p, r in by_path.items()} == expected_blobs and all(r["matches_commit"] for r in files))
                imports_ok = True
                for item in proof["imported_modules"]:
                    prefix = "/content/MulTaBench-reference/"
                    rel = item["loaded_from"].removeprefix(prefix)
                    imports_ok &= item["loaded_from"].startswith(prefix) and rel in by_path and item["matches_commit"] and item["sha256"] == by_path.get(rel, {}).get("sha256")
                check(s, "recorded_import_locations_and_hashes", imports_ok)
                check(s, "selected_source_hashes_verified_independently", all(by_path.get(p, {}).get("sha256") == sha for p, sha in source_checks.items()), f"{len(source_checks)} downloaded pinned files")
                check(s, "requirements_hash_matches_reference_record", cfg["requirements_sha256"] == by_path["requirements.txt"]["sha256"])
                for p, item in by_path.items():
                    proof_sha_by_path.setdefault(p, set()).add(item["sha256"])
                runs.append(dict(archive=path.name, run_id=root, model=cfg["model_key"], fold=cfg["fold"], condition=cfg["condition"],
                                 score=summary["test_score"], csv_sha256=cfg["csv_sha256"], schema_sha256=cfg["schema_sha256"],
                                 benchmark_commit=cfg["benchmark_commit"], research_commit=cfg["research_commit"],
                                 experiment_id=cfg["experiment_id"], adapter_sha256=cfg["adapter_sha256"], counts=counts))
            check(scope, "complete_unique_requested_jobs", len(observed_jobs) == len(set(observed_jobs)) and set(observed_jobs) == expected_jobs)
            check(scope, "csv_and_json_run_sets_match", set(archive_by_id) == set(roots))

    check("global", "no_duplicate_logical_jobs", len({(r['model'], r['fold'], r['condition']) for r in runs}) == len(runs))
    for field in ["csv_sha256", "schema_sha256", "benchmark_commit", "research_commit", "experiment_id", "adapter_sha256"]:
        check("global", "same_" + field, len({r[field] for r in runs}) == 1)
    check("global", "recorded_reference_hashes_agree_across_jobs", all(len(v) == 1 for v in proof_sha_by_path.values()))
    if runs:
        commit = runs[0]["research_commit"]
        import subprocess
        root = Path(__file__).resolve().parents[3]
        raw = subprocess.check_output(["git", "show", commit + ":evaluation/colab_runner.py"], cwd=root)
        check("global", "adapter_bytes_match_recorded_project_commit", digest(raw) == runs[0]["adapter_sha256"])
    overlaps = []
    for a, b in itertools.combinations(sorted(outer_by_fold), 2):
        overlaps.append(dict(folds=[a, b], shared_test_report_ids=len(set(outer_by_fold[a][1]) & set(outer_by_fold[b][1]))))
    by_model = {}
    for r in runs:
        by_model.setdefault(r["model"], {}).setdefault(r["condition"], []).append(r["score"])
    gains = []
    for model, conditions in sorted(by_model.items()):
        means = {k: statistics.mean(conditions[k]) for k in CONDITIONS}
        gains.append(dict(model=model, folds=len(conditions["joint_tar"]), **means,
                          joint_signal=means["joint_frozen"] - max(means["structured_only"], means["text_only"]),
                          tar_gain=means["joint_tar"] - means["joint_frozen"]))
    return dict(archives=archives, checks=checks, runs=runs, gains=gains, cross_fold_test_overlap=overlaps,
                limitations=["Recorded split IDs are adapter reconstructions, not independent runtime tensor traces.",
                             "No dataset CSV, schema.json or construction audit in either archive.",
                             "Cannot assess outcome-bearing text, repeated events, source/label accuracy or cohort bias.",
                             "Proof records and logs corroborate execution; they are not external execution attestations.",
                             "This is the older pinned diagnostic protocol, not the required final registered run."])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("archives", nargs="+", type=Path)
    p.add_argument("--tree", type=Path, required=True)
    p.add_argument("--source-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = audit(args.archives, args.tree, args.source_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    totals = {s: sum(c["status"] == s for c in result["checks"]) for s in ["PASS", "FAIL"]}
    print(json.dumps(dict(runs=len(result["runs"]), checks=totals, output=str(args.output)), indent=2))
    for item in result["checks"]:
        if item["status"] == "FAIL":
            print(json.dumps(item))
    raise SystemExit(bool(totals["FAIL"]))
