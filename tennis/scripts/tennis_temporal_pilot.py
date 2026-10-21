"""Optional Colab diagnostic using unmodified official classes.

Default is checks only. --run performs four LightGBM fits, 2 TAR epochs. This
chronological outer split deliberately differs from official random holdouts.
No final-candidate qualification is computed. A documented calendar-delay policy
can support explicitly conditional research without claiming verified uploads.
"""
import argparse
from enum import Enum
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from evaluation.colab_runner import load_csv, verify_reference, verify_import_origins, worker_environment, write_json


def load_temporal_candidate(csv, schema_path, evaluation_partition="development"):
    """Validate sidecar provenance before importing any model/transformer stack."""
    import hashlib
    import pandas as pd
    df, schema = load_csv(csv, schema_path)
    folder = Path(schema_path).resolve().parent
    if schema.get("audit_file"):
        audit_path = folder / schema["audit_file"]
        if hashlib.sha256(audit_path.read_bytes()).hexdigest() != schema["audit_csv_sha256"]:
            raise ValueError("Audit sidecar hash differs from frozen schema")
        audit = pd.read_csv(audit_path)
        if not audit.match_id.is_unique or set(audit.match_id) != set(df.match_id):
            raise ValueError("Audit sidecar and feature rows do not align uniquely")
        audit_cols = [c for c in audit if c == "match_id" or c not in df.columns]
        df = df.reset_index(drop=True).merge(audit[audit_cols], on="match_id", validate="one_to_one")
        df.index = df.match_id.astype(str)
    verified = bool(schema.get("strict_publication_temporal_gate_passed"))
    assumed = schema.get("availability_policy") == "documented_calendar_delay_assumption" and schema.get("conditional_calendar_policy_passed") is True
    if not (verified or assumed):
        raise ValueError("Declare verified source timing or an explicit conservative calendar-delay research policy")
    if assumed:
        lag = int(schema.get("availability_lag_days", 0))
        if lag < 2:
            raise ValueError("Conditional calendar policy requires a declared buffer of at least two days")
        for side in ["a", "b"]:
            ages = (pd.to_datetime(df.match_date) - pd.to_datetime(df[f"{side}_text_date"])).dt.days
            if not ages.between(lag, 90).all() or not df[f"{side}_source_match_id"].ne(df.match_id).all():
                raise ValueError("Row-level source dates/identity violate the declared calendar policy")
        availability = "assumed calendar delay; NOT verified historical publication"
    else:
        if "publication_verified" in df and not df.publication_verified.eq(True).all():
            raise ValueError("Verified publication claim lacks complete row-level support")
        availability = "manifest and row-level publication verification claim"
    if "split" in df:
        if not set(df.split) <= {"train", "development", "reserved_test"}:
            raise ValueError("Unexpected chronological partition label")
        train_names = ["train"] if evaluation_partition == "development" else ["train", "development"]
        train = df[df.split.isin(train_names)].copy()
        test = df[df.split == evaluation_partition].copy()
    else:
        ordered = df.sort_values(["match_date", "match_id"])
        cutoff = ordered.match_date.iloc[int(.8 * len(ordered))]
        train, test = ordered[ordered.match_date < cutoff].copy(), ordered[ordered.match_date >= cutoff].copy()
    if train.empty or test.empty or train.match_date.max() >= test.match_date.min() or set(train.match_id) & set(test.match_id):
        raise ValueError("Training/evaluation dates or target IDs overlap")
    if "source_available_at" in df:
        if not (pd.to_datetime(df.source_available_at, utc=True) < pd.to_datetime(df.match_date, utc=True)).all():
            raise ValueError("Source snapshot is not strictly before each target day")
    return df, schema, train, test, availability


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["reference", "csv", "schema", "output"]: parser.add_argument("--" + name, required=True)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--evaluation-partition", choices=["development", "reserved_test"], default="development")
    args = parser.parse_args()
    import os
    os.environ.update(worker_environment())
    df, schema, train, test, availability = load_temporal_candidate(args.csv, args.schema, args.evaluation_partition)
    print("Availability basis:", availability, flush=True)
    ref = Path(args.reference).resolve(); proof = verify_reference(ref)
    sys.dont_write_bytecode = True; sys.path.insert(0, str(ref))
    import torch
    from tabstar.preprocessing.splits import split_to_val
    from tabstar.preprocessing.feat_types import detect_numerical_features
    from multabench.baselines.preprocessing.feature_types import transform_feature_types
    from multabench.preprocessing.feat_types import classify_semantic_features
    from multabench.baselines.lgbm import LightGBM
    from multabench.datasets.objects import SupervisedTask
    from multabench.finetune.train_args import E5TrainArgs
    df.index.name = "_audit_row_id"
    cutoff = test.match_date.min()
    if min(len(train),len(test)) < 20 or min(train[schema["target"]].nunique(),test[schema["target"]].nunique()) < 2:
        raise ValueError("Not enough chronological rows/classes for this diagnostic")
    cols = schema["structured_columns"] + schema["text_columns"]
    fit, _, _, _ = split_to_val(x=train[cols], y=train[schema["target"]], is_cls=True)
    numeric = set(detect_numerical_features(fit))
    converted = transform_feature_types(fit, numerical_features=numeric, image_features=set())
    detected = classify_semantic_features(converted, numerical_features=numeric).text_features
    if detected != set(schema["text_columns"]):
        raise ValueError(f"Official training-pool text detection is {sorted(detected)}. Declared text would not reliably receive E5/TAR. Stop this pilot.")
    print(f"Checks passed: {len(train)} prior training rows / {len(test)} later test rows; cutoff {cutoff}")
    if not args.run:
        print("Check only. No models fitted. --run executes the labeled diagnostic on the selected partition.")
        return
    if not torch.cuda.is_available(): raise RuntimeError("Colab GPU required")
    out = Path(args.output).resolve(); out.mkdir(parents=True,exist_ok=True); os.chdir(out)
    train_args = E5TrainArgs().to_dict(); train_args.update(epochs=2,batch_size=32)
    records=[]
    for condition in ["structured_only","text_only","joint_frozen","joint_tar"]:
        selected = schema["structured_columns"] if condition == "structured_only" else schema["text_columns"] if condition == "text_only" else cols
        started=time.perf_counter()
        model=LightGBM(problem_type=SupervisedTask.BINARY, device=torch.device("cuda"),
             dataset=Enum("PilotDataset",{schema["dataset_id"]:"local-pilot"})[schema["dataset_id"]],
             tune_e5=condition=="joint_tar",e5_train_kwargs=train_args if condition=="joint_tar" else None,pca_components=30)
        model.fit(train[selected],train[schema["target"]])
        metrics=model.score_all_metrics(X=test[selected],y=test[schema["target"]])
        row={"condition":condition,"auroc":metrics.score,"cutoff":cutoff,"n_train":len(train),"n_test":len(test),
             "runtime_seconds":time.perf_counter()-started,"protocol":"chronological diagnostic; NOT official random-holdout curation",
             "availability_basis":availability, "evaluation_partition":args.evaluation_partition,
             "e5_train_args":train_args if condition=="joint_tar" else None}
        records.append(row);write_json(out/(condition+".json"),row)
        del model;torch.cuda.empty_cache()
    verify_reference(ref)
    write_json(out/"source_proof.json",{"files":proof,"imports":verify_import_origins(ref,proof)})
    write_json(out/"scores.json",records)


if __name__ == "__main__": main()
