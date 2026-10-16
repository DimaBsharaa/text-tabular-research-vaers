"""Per-fold paired statistics for an official-code screen (results.csv from notebook 02, or a 4-column extract
model,condition,fold,auroc). Prints Joint Signal and TAR Gain per learner and fold, with mean, sd, se and the
number of positive folds; also the Hanley-McNeil standard error of one test-set AUROC, to judge fold noise."""
from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd


def paired_table(results: pd.DataFrame) -> pd.DataFrame:
    df = results.rename(columns={"model_key": "model"})
    w = df.pivot_table(index=["model", "fold"], columns="condition", values="auroc")
    w["joint_signal"] = w.joint_frozen - w[["structured_only", "text_only"]].max(axis=1)
    w["tar_gain"] = w.joint_tar - w.joint_frozen
    w["text_minus_table"] = w.text_only - w.structured_only
    return w


def summary(w: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, g in w.groupby(level=0):
        for c in ["joint_signal", "tar_gain", "text_minus_table"]:
            x = g[c].to_numpy()
            sd = x.std(ddof=1) if len(x) > 1 else float("nan")
            rows.append({"model": model, "metric": c, "mean": x.mean(), "sd": sd, "se": sd / math.sqrt(len(x)),
                         "positive_folds": f"{int((x > 0).sum())}/{len(x)}", "min": x.min()})
    return pd.DataFrame(rows)


def hanley_mcneil_se(auc: float, n_pos: int, n_neg: int) -> float:
    q1, q2 = auc / (2 - auc), 2 * auc ** 2 / (1 + auc)
    return math.sqrt((auc * (1 - auc) + (n_pos - 1) * (q1 - auc ** 2) + (n_neg - 1) * (q2 - auc ** 2)) / (n_pos * n_neg))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--prevalence", type=float, required=True)
    ap.add_argument("--n-test", type=int, default=2000)
    a = ap.parse_args()
    w = paired_table(pd.read_csv(a.csv))
    print(w.round(4).to_string())
    print(summary(w).round(4).to_string(index=False))
    pos = round(a.n_test * a.prevalence)
    print(f"Hanley-McNeil SE of one AUROC at 0.85 with {pos} positives / {a.n_test}: "
          f"{hanley_mcneil_se(0.85, pos, a.n_test - pos):.4f}")
