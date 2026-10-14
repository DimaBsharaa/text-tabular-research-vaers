"""Collect official scores; never invent results for unfinished conditions."""
import json
from pathlib import Path

import pandas as pd

CONDITIONS = ["structured_only", "text_only", "joint_frozen", "joint_tar"]
MODELS = ["light", "cat", "tabm", "tabpfnv2", "tabpfnv2p5"]


def collect(folder):
    rows = []
    for path in sorted(Path(folder).glob("*/result.json")):
        result = json.loads(path.read_text())
        config = result["config"]
        summary = result.get("official_summary", {})
        rows.append({"run_id": result["run_id"], **{k: config[k] for k in ["model_key", "condition", "fold", "mode", "purpose", "csv_sha256", "benchmark_commit", "experiment_id"]},
                     "status": result["status"], "auroc": summary.get("test_score"),
                     "n_train": summary.get("n_train"), "n_test": summary.get("n_test"),
                     "runtime_s": summary.get("runtime"), "gpu": summary.get("gpu_type"),
                     "train_peak_gpu_gb": summary.get("train_peak_gpu_gb"), "error": result.get("error"),
                     "result_path": str(path)})
    return pd.DataFrame(rows)


def summarize(raw, folds, mode, purpose):
    if len(raw):
        for column in ["csv_sha256", "benchmark_commit", "mode", "purpose", "experiment_id"]:
            if raw[column].nunique(dropna=False) != 1:
                raise ValueError(f"Mixed {column} values cannot form one comparison")
        if raw["mode"].iloc[0] != mode or raw["purpose"].iloc[0] != purpose:
            raise ValueError("Results do not match requested evaluation mode/purpose")
    rows = []
    for model in MODELS:
        row = {"learner": model}
        model_runs = raw[raw.model_key == model] if len(raw) else pd.DataFrame()
        complete = True
        for condition in CONDITIONS:
            runs = model_runs[(model_runs.condition == condition) & (model_runs.status == "ok")] if len(model_runs) else pd.DataFrame()
            # Never average retries or differing configurations as extra folds.
            if len(runs) and runs.fold.duplicated().any():
                raise ValueError("Multiple successful configurations for a fold; select one run directory")
            complete &= len(runs) == len(folds) and (set(runs.fold) == set(folds) if len(runs) else False)
            row[condition] = runs.auroc.mean() if len(runs) else float("nan")
            row[condition + "_sd"] = runs.auroc.std() if len(runs) > 1 else float("nan")
        row["joint_signal_gain"] = row["joint_frozen"] - max(row["structured_only"], row["text_only"]) if all(pd.notna(row[c]) for c in CONDITIONS[:3]) else float("nan")
        row["tar_gain"] = row["joint_tar"] - row["joint_frozen"]
        eligible = complete and mode == "full" and purpose == "candidate_dataset" and set(folds) == set(range(5))
        row["status"] = "complete" if complete else ("incomplete" if len(model_runs) else "not run")
        row["assignment_pass"] = bool(row["joint_signal_gain"] > 0 and row["tar_gain"] > 0) if eligible else None
        row["paper_pass"] = bool(row["joint_signal_gain"] > 0.001 and row["tar_gain"] > 0.001) if eligible else None
        rounded = pd.Series({c: row[c] for c in CONDITIONS}).round(3)
        legacy_joint = round(rounded.joint_frozen - max(rounded.structured_only, rounded.text_only), 6)
        legacy_tar = round(rounded.joint_tar - rounded.joint_frozen, 6)
        row["legacy_pass"] = bool(legacy_joint >= 0.001 and legacy_tar >= 0.001) if eligible else None
        rows.append(row)
    return pd.DataFrame(rows)
