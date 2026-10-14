"""CSV/JSON adapter to the unmodified paper-version MulTaBench evaluator.

All imports of learners are deferred to Colab's isolated benchmark environment.
This file never patches a benchmark class, registers a dataset upstream, or scores
predictions itself. The reference supplies preprocessing, splitting, fit and AUROC.
"""
import argparse
from enum import Enum
import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback

PAPER_COMMIT = "3bb95079a6146ed35d45187e98bcad5b063273a1"
OFFICIAL_REPOSITORY = "https://github.com/alanarazi7/MulTaBench"
MODELS = {"light": ("multabench.baselines.lgbm", "LightGBM"),
          "cat": ("multabench.baselines.catboost", "CatBoost"),
          "tabm": ("multabench.baselines.tabm", "TabM"),
          "tabpfnv2": ("multabench.baselines.tabpfnv2", "TabPFNv2"),
          "tabpfnv2p5": ("multabench.baselines.tabpfnv2", "TabPFNv2p5")}
CONDITIONS = ("structured_only", "text_only", "joint_frozen", "joint_tar")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(ref, *args):
    return subprocess.check_output(["git", "-C", str(ref), *args], text=True).strip()


def verify_reference(ref, expected_commit=PAPER_COMMIT):
    ref = Path(ref).resolve()
    if git(ref, "remote", "get-url", "origin").rstrip("/").removesuffix(".git") != OFFICIAL_REPOSITORY:
        raise ValueError("Reference origin must be the official alanarazi7/MulTaBench repository")
    if git(ref, "rev-parse", "HEAD") != expected_commit:
        raise ValueError("Reference checkout differs from pinned paper-version commit")
    # Check tracked source plus unexpected untracked files. No writes into checkout.
    if git(ref, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("Reference checkout must be clean; keep outputs outside it")
    # Compare actual bytes to every Git blob, even if a local Git index flag hides changes.
    tree = subprocess.check_output(["git", "-C", str(ref), "ls-tree", "-rz", expected_commit])
    proof = []
    for entry in tree.split(b"\0"):
        if not entry:
            continue
        header, raw_name = entry.split(b"\t", 1)
        _, kind, expected_blob = header.decode().split()
        name = os.fsdecode(raw_name)
        path = ref / name
        if kind != "blob":
            raise ValueError(f"Unsupported reference entry: {name} ({kind})")
        content = os.fsencode(os.readlink(path)) if path.is_symlink() else path.read_bytes()
        actual_blob = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
        if actual_blob != expected_blob:
            raise ValueError(f"Reference bytes differ from pinned commit: {name}")
        proof.append({"path": name, "git_blob": expected_blob,
                      "sha256": hashlib.sha256(content).hexdigest(), "matches_commit": True})
    return proof


def verify_import_origins(ref, file_proof):
    """Check the modules actually imported by this worker, not just a spare checkout."""
    ref = Path(ref).resolve()
    by_path = {r["path"]: r for r in file_proof}
    imported = []
    for name, module in list(sys.modules.items()):
        if name != "multabench" and not name.startswith("multabench."):
            continue
        if not getattr(module, "__file__", None):
            continue  # Namespace packages contain no source to execute.
        path = Path(inspect.getsourcefile(module) or module.__file__).resolve()
        if not path.is_relative_to(ref) or path.relative_to(ref).as_posix() not in by_path:
            raise ValueError(f"Imported {name} from outside the verified reference: {path}")
        row = by_path[path.relative_to(ref).as_posix()]
        if sha(path) != row["sha256"]:
            raise ValueError(f"Imported module changed after verification: {path}")
        imported.append({"module": name, "loaded_from": str(path), "sha256": row["sha256"], "matches_commit": True})
    return sorted(imported, key=lambda r: r["module"])


def load_csv(csv, schema_path):
    import pandas as pd
    schema = json.loads(Path(schema_path).read_text())
    if schema["task"] != "binary":
        raise ValueError("This initial adapter implements binary classification")
    if schema["csv_sha256"] != sha(csv):
        raise ValueError("CSV hash differs from frozen schema")
    df = pd.read_csv(csv)
    structured, text = schema["structured_columns"], schema["text_columns"]
    target, audit = schema["target"], schema["audit_columns"]
    columns = structured + text + [target] + audit
    if len(columns) != len(set(columns)) or set(columns) != set(df.columns):
        raise ValueError("Manifest must partition all fields without overlap or undeclared columns")
    if df[target].isna().any() or set(df[target].unique()) != {0, 1}:
        raise ValueError("Expected both binary outcome classes and no missing targets")
    if not df.match_id.is_unique or df.match_id.isna().any():
        raise ValueError("Stable unique match_id is required")
    for col in schema["numeric_columns"]:
        df[col] = pd.to_numeric(df[col], errors="raise")
    for col in structured + text:
        if col not in schema["numeric_columns"]:
            df[col] = df[col].astype(object)
        if df[col].isna().all():
            raise ValueError(f"All-missing predictor: {col}")
    if any(df[c].isna().any() or df[c].astype(str).str.strip().eq("").any() for c in text):
        raise ValueError("This first adapter requires nonempty text on every retained row")
    df.index = df["match_id"].astype(str)
    return df, schema


def selected_columns(schema, condition):
    if condition == "structured_only":
        return schema["structured_columns"]
    if condition == "text_only":
        return schema["text_columns"]
    if condition in {"joint_frozen", "joint_tar"}:
        return schema["structured_columns"] + schema["text_columns"]
    raise ValueError(condition)


def import_model(name):
    module, cls = MODELS[name]
    return getattr(importlib.import_module(module), cls)


def audit_split(df, schema, model_cls, fold, train_examples, condition):
    from tabstar.preprocessing.splits import split_to_test, split_to_val
    from multabench.preprocessing.splits import split_to_val as encoder_split
    from multabench.baselines.preprocessing.sampling import subsample_dataset
    from multabench.baselines.preprocessing.feature_types import transform_feature_types
    from tabstar.preprocessing.feat_types import detect_numerical_features
    from multabench.preprocessing.feat_types import classify_semantic_features

    x = df[selected_columns(schema, condition)].copy()
    y = df[schema["target"]].copy()
    x, y = subsample_dataset(x=x, y=y, is_cls=True, train_examples=train_examples, fold=fold)
    x_tr, x_te, y_tr, _ = split_to_test(x=x, y=y, is_cls=True, fold=fold, train_examples=train_examples)
    fit_x, fit_y, val_x = x_tr, y_tr, None
    if model_cls.USE_VAL_SPLIT:
        fit_x, val_x, fit_y, _ = split_to_val(x=x_tr, y=y_tr, is_cls=True)
    numerical = set(detect_numerical_features(fit_x))
    converted = transform_feature_types(fit_x, numerical_features=numerical, image_features=set())
    detected = classify_semantic_features(converted, numerical_features=numerical).text_features
    expected = set() if condition == "structured_only" else set(schema["text_columns"])
    if detected != expected:
        raise ValueError(f"Official text detection {sorted(detected)} != manifest {sorted(expected)} in actual fitting pool")
    if expected and len(fit_x) < 30:
        raise ValueError("Insufficient rows for the official 30-component PCA")
    splits = {"outer_train": list(x_tr.index), "outer_test": list(x_te.index),
              "learner_fit": list(fit_x.index), "learner_validation": list(val_x.index) if val_x is not None else [],
              "detected_text_columns": sorted(detected)}
    assert not set(splits["outer_train"]) & set(splits["outer_test"])
    if condition == "joint_tar":
        e_tr, e_val, _, _ = encoder_split(x=fit_x, y=fit_y, is_cls=True)
        splits.update(encoder_train=list(e_tr.index), encoder_validation=list(e_val.index))
    return splits


def write_json(path, value):
    # Atomic completion marker; interrupted fits cannot be mistaken for results.
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, default=lambda v: v.value if isinstance(v, Enum) else str(v)) + "\n")
    tmp.replace(path)


def execute(args):
    ref, out = Path(args.reference).resolve(), Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    # Never import from the small source snapshots already stored in research/data.
    file_proof = verify_reference(ref)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(ref))
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch
    from multabench.finetune.train_args import E5TrainArgs
    from multabench.baselines.benchmarks.evaluate import evaluate_on_loaded_dataset
    from multabench.datasets.curation import MultimodalDataset
    from multabench.datasets.objects import SupervisedTask
    from multabench.datasets.multimodal import MultimodalState
    from multabench.e5.constants import E5_SMALL_V2

    if not torch.cuda.is_available():
        raise RuntimeError("Select a Colab GPU runtime before running the four conditions")
    df, schema = load_csv(args.csv, args.schema)
    model_cls = import_model(args.model)
    splits = audit_split(df, schema, model_cls, args.fold, args.train_examples, args.condition)
    if args.check:
        for name in MODELS:
            import_model(name)  # Import only; never initialize models or download weights.
        imported = verify_import_origins(ref, file_proof)
        write_json(out / "reference_proof.json", {"repository": OFFICIAL_REPOSITORY,
                   "commit": PAPER_COMMIT, "tracked_files": file_proof, "imported_modules": imported})
        print(json.dumps({"check": "PASS", "gpu": torch.cuda.get_device_name(0),
                          "rows": len(df), "split_counts": {k: len(v) for k, v in splits.items()},
                          "verified_tracked_files": len(file_proof), "verified_imported_modules": len(imported),
                          "reference_commit": PAPER_COMMIT, "note": "No model fit or checkpoint access verified"}, indent=2))
        verify_reference(ref)
        return
    if args.mode == "full" and schema.get("purpose") != "candidate_dataset":
        raise ValueError("Full evaluation requires a real candidate_dataset manifest; smoke fixture is ineligible")
    if args.mode == "full" and (args.train_examples != 10000 or args.fold not in range(5)):
        raise ValueError("Full paper evaluation uses train_examples=10000 and folds 0–4")
    e5_kwargs = E5TrainArgs().to_dict()
    if args.mode == "smoke":
        e5_kwargs.update(epochs=2, batch_size=32)  # Parameters only; reference implementation unchanged.
    freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    project = Path(__file__).resolve().parents[1]
    config = {"benchmark_commit": PAPER_COMMIT, "research_commit": git(project, "rev-parse", "HEAD"),
              "adapter_sha256": sha(__file__),
              "schema_sha256": sha(args.schema), "csv_sha256": sha(args.csv),
              "requirements_sha256": sha(ref / "requirements.txt"),
              "environment_sha256": hashlib.sha256(freeze.encode()).hexdigest(),
              "model_key": args.model, "condition": args.condition, "fold": args.fold, "mode": args.mode,
              "train_examples": args.train_examples, "e5_model_name": E5_SMALL_V2,
              "e5_train_kwargs": e5_kwargs if args.condition == "joint_tar" else None,
              "pca_components": 30, "purpose": schema.get("purpose"), "columns": selected_columns(schema, args.condition)}
    # Same scientific settings across models/conditions/folds; exact jobs still get individual hashes.
    common_config = {k: v for k, v in config.items() if k not in {"model_key", "condition", "fold", "columns", "e5_train_kwargs", "research_commit"}}
    config["experiment_id"] = hashlib.sha256(json.dumps(common_config, sort_keys=True).encode()).hexdigest()[:24]
    run_id = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:24]
    folder = out / run_id
    folder.mkdir(exist_ok=True)
    result_path = folder / "result.json"
    if result_path.exists():
        prior = json.loads(result_path.read_text())
        if prior.get("status") == "ok" and prior.get("config") == config:
            print(f"Reusing completed run {run_id}")
            return
    write_json(folder / "config.json", config)
    write_json(folder / "splits.json", splits)
    write_json(folder / "reference_proof.json", {"repository": OFFICIAL_REPOSITORY, "commit": PAPER_COMMIT,
               "tracked_files": file_proof, "imported_modules": verify_import_origins(ref, file_proof)})
    (folder / "pip_freeze.txt").write_text(freeze)
    # Scratch stays in our ignored tree; persistent result files may live in Drive.
    # This also lets the official cwd-based `git` field resolve our commit correctly.
    scratch = project / "evaluation/runs" / run_id
    scratch.mkdir(parents=True, exist_ok=True)
    os.chdir(scratch)
    local_id = Enum("LocalDatasetID", {schema["dataset_id"]: "local-csv"})[schema["dataset_id"]]
    dataset = MultimodalDataset(x=df[config["columns"]].copy(), y=df[schema["target"]].copy(),
                                task_type=SupervisedTask.BINARY, dataset_id=local_id)
    state = {"structured_only": MultimodalState.NO_TEXT, "text_only": MultimodalState.TEXT_ONLY,
             "joint_frozen": MultimodalState.ALL, "joint_tar": MultimodalState.ALL}[args.condition]
    record = {"run_id": run_id, "config": config, "status": "running"}
    try:
        # This is the official entry point, not a reimplementation of its pipeline.
        summary = evaluate_on_loaded_dataset(model_cls=model_cls, dataset=dataset,
                  fold=args.fold, device=torch.device("cuda"), train_examples=args.train_examples,
                  multimodal_state=state, tune_e5=args.condition == "joint_tar",
                  e5_train_kwargs=config["e5_train_kwargs"], e5_model_name=E5_SMALL_V2,
                  pca_components=30, no_pca=False)
        if summary["n_train"] != len(splits["outer_train"]) or summary["n_test"] != len(splits["outer_test"]):
            raise ValueError("Official evaluator split sizes differ from audit")
        if not math.isfinite(float(summary["test_score"])) or not 0 <= float(summary["test_score"]) <= 1:
            raise ValueError("Official evaluator did not return a finite binary AUROC in [0, 1]")
        verify_reference(ref)
        write_json(folder / "reference_proof.json", {"repository": OFFICIAL_REPOSITORY, "commit": PAPER_COMMIT,
                   "tracked_files": file_proof, "imported_modules": verify_import_origins(ref, file_proof)})
        record.update(status="ok", official_summary=summary)
    except Exception as exc:
        record.update(status="error", error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        write_json(result_path, record)
    print(f"Saved {args.model} / {args.condition}: AUROC={summary['test_score']:.4f} to {result_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("reference", "csv", "schema", "output"):
        parser.add_argument(f"--{key}", required=True)
    parser.add_argument("--model", choices=MODELS, default="light")
    parser.add_argument("--condition", choices=CONDITIONS, default="joint_tar")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--train-examples", type=int, default=10000)
    parser.add_argument("--mode", choices=["smoke", "diagnostic", "full"], default="smoke")
    parser.add_argument("--check", action="store_true")
    execute(parser.parse_args())


if __name__ == "__main__":
    main()
