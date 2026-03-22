"""
benchmark_runner_external.py

Patched external-benchmark runner for BWORF on:
- breast_cancer
- heart_failure
- diabetes
- thyroid

Models:
- rf
- orf_style      -> patched BWORF with weighted_bootstrap=False, MI off
- bworf_no_mi    -> patched BWORF with weighted_bootstrap=True, MI off
- bworf_mi       -> patched BWORF with weighted_bootstrap=True, MI on

Outputs per dataset/model:
- fold_results.csv
- oof_predictions.csv.gz
- summary.csv
- run_config.json
"""

from __future__ import annotations

import argparse
import contextlib
import gzip
import json
import os
import platform
import sys
import time
from typing import Any
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, label_binarize

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from bworf_with_mi import BWORFClassifier, apply_mutual_information_filtering  # noqa: E402

REGISTRY: dict[str, dict] = {
    "breast_cancer": dict(
        file="breast_cancer.csv.gz",
        target="class",
        task="binary",
        categorical_cols=[],
        default_top_k=6,
    ),
    "heart_failure": dict(
        file="heart_failure.csv.gz",
        target="DEATH_EVENT",
        task="binary",
        categorical_cols=[],
        default_top_k=8,
    ),
    "diabetes": dict(
        file="diabetes.csv.gz",
        target="Outcome",
        task="binary",
        categorical_cols=[],
        default_top_k=6,
    ),
    "thyroid": dict(
        file="thyroid.csv.gz",
        target="class",
        task="multiclass",
        categorical_cols=[],
        default_top_k=4,
    ),
}

BINARY_METRICS = [
    "accuracy",
    "balanced_accuracy",
    "f1_pos",
    "precision_pos",
    "recall_pos",
    "mcc",
    "auroc",
    "auprc",
    "log_loss",
    "brier",
]

MULTI_METRICS = [
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "macro_precision",
    "macro_recall",
    "mcc",
    "auroc_macro_ovr",
    "auprc_macro_ovr",
    "log_loss",
    "brier_multiclass",
]


def _str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ("true", "1", "yes", "y"):
        return True
    if v.lower() in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {v!r}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="External benchmark runner for patched BWORF")
    p.add_argument("--data_dir", default="datasets")
    p.add_argument("--out_dir", default="outputs")
    p.add_argument("--datasets", default="breast_cancer,heart_failure,diabetes,thyroid")
    p.add_argument("--models", default="rf,orf_style,bworf_no_mi,bworf_mi")

    p.add_argument("--n_splits", type=int, default=10)
    p.add_argument("--seeds", default="42,43,44,45,46")
    p.add_argument("--n_jobs", type=int, default=None)

    p.add_argument("--rf_n_estimators", type=int, default=100)
    p.add_argument("--rf_max_depth", type=int, default=6)

    p.add_argument("--bworf_n_estimators", type=int, default=100)
    p.add_argument("--bworf_max_depth", type=int, default=6)
    p.add_argument("--bworf_min_samples_split", type=int, default=10)
    p.add_argument("--bworf_min_samples_leaf", type=int, default=1)
    p.add_argument("--bworf_l1_strength", type=float, default=0.2)
    p.add_argument("--bworf_bootstrap_temperature", type=float, default=1.0)
    p.add_argument("--bworf_n_tries", type=int, default=10)

    p.add_argument(
        "--bworf_top_k_override",
        type=int,
        default=None,
        help="If set, override dataset-specific MI top-k for bworf_mi.",
    )
    p.add_argument("--drop_incomplete_oof", type=_str2bool, default=False)
    return p.parse_args()


def build_preprocessor(spec: dict, X_df: pd.DataFrame) -> Pipeline:
    declared_cat = [c for c in spec["categorical_cols"] if c in X_df.columns]

    auto_cat = [
        c for c in X_df.columns
        if pd.api.types.is_object_dtype(X_df[c])
        or pd.api.types.is_categorical_dtype(X_df[c])
        or pd.api.types.is_bool_dtype(X_df[c])
    ]

    cat_cols = sorted(set(declared_cat + auto_cat))
    num_cols = [c for c in X_df.columns if c not in cat_cols]

    transformers = []
    if num_cols:
        transformers.append((
            "num",
            SimpleImputer(strategy="median"),
            num_cols,
        ))

    if cat_cols:
        transformers.append((
            "cat",
            Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            cat_cols,
        ))

    ct = ColumnTransformer(transformers, remainder="drop")
    return Pipeline([("prep", ct)])


def _build_rf(seed: int, n_jobs: int, args: argparse.Namespace):
    return RandomForestClassifier(
        n_estimators=args.rf_n_estimators,
        max_depth=args.rf_max_depth,
        class_weight="balanced",
        random_state=seed,
        n_jobs=n_jobs,
    )


def _build_oblique_variant(seed: int, args: argparse.Namespace, *, weighted_bootstrap: bool):
    return BWORFClassifier(
        n_estimators=args.bworf_n_estimators,
        max_depth=args.bworf_max_depth,
        min_samples_split=args.bworf_min_samples_split,
        min_samples_leaf=args.bworf_min_samples_leaf,
        l1_strength=args.bworf_l1_strength,
        weighted_bootstrap=weighted_bootstrap,
        bootstrap_temperature=args.bworf_bootstrap_temperature,
        n_tries=args.bworf_n_tries,
        random_state=seed,
    )


def build_model(model_name: str, seed: int, n_jobs: int, args: argparse.Namespace):
    if model_name == "rf":
        return _build_rf(seed, n_jobs, args)
    if model_name == "orf_style":
        return _build_oblique_variant(seed, args, weighted_bootstrap=False)
    if model_name in {"bworf_no_mi", "bworf_mi"}:
        return _build_oblique_variant(seed, args, weighted_bootstrap=True)
    raise ValueError(f"Unknown model: {model_name}")


def model_params_json(model_name: str, args: argparse.Namespace, seed: int, top_k_used: int | None) -> str:
    if model_name == "rf":
        d = dict(
            n_estimators=args.rf_n_estimators,
            max_depth=args.rf_max_depth,
            class_weight="balanced",
            random_state=seed,
        )
    else:
        d = dict(
            n_estimators=args.bworf_n_estimators,
            max_depth=args.bworf_max_depth,
            min_samples_split=args.bworf_min_samples_split,
            min_samples_leaf=args.bworf_min_samples_leaf,
            l1_strength=args.bworf_l1_strength,
            weighted_bootstrap=(model_name != "orf_style"),
            bootstrap_temperature=args.bworf_bootstrap_temperature,
            n_tries=args.bworf_n_tries,
            top_k_features=top_k_used,
            random_state=seed,
        )
    return json.dumps(d, separators=(",", ":"))


def sanitize_proba(P: np.ndarray, n_classes: int, context: str = "") -> np.ndarray:
    P = np.asarray(P, dtype=np.float64)
    if P.ndim == 1:
        if n_classes == 2:
            P = np.column_stack([1.0 - P, P])
        else:
            raise ValueError(f"1-D proba with n_classes={n_classes}. {context}")
    if P.shape[1] != n_classes:
        raise ValueError(f"proba shape {P.shape} but expected n_classes={n_classes}. {context}")
    P = np.clip(P, 1e-15, 1.0 - 1e-15)
    row_sums = P.sum(axis=1, keepdims=True)
    return P / row_sums


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray, task: str, n_classes: int) -> dict[str, float]:
    m: dict[str, float] = {}
    m["accuracy"] = accuracy_score(y_true, y_pred)
    m["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
    m["mcc"] = matthews_corrcoef(y_true, y_pred)
    m["log_loss"] = log_loss(y_true, proba, labels=list(range(n_classes)))

    if task == "binary":
        m["f1_pos"] = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        m["precision_pos"] = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        m["recall_pos"] = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        p1 = proba[:, 1]
        m["auroc"] = roc_auc_score(y_true, p1)
        m["auprc"] = average_precision_score(y_true, p1)
        m["brier"] = float(np.mean((p1 - y_true) ** 2))
    else:
        m["macro_f1"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
        m["macro_precision"] = precision_score(y_true, y_pred, average="macro", zero_division=0)
        m["macro_recall"] = recall_score(y_true, y_pred, average="macro", zero_division=0)
        y_bin = label_binarize(y_true, classes=list(range(n_classes)))
        m["auroc_macro_ovr"] = roc_auc_score(y_bin, proba, multi_class="ovr", average="macro")
        m["auprc_macro_ovr"] = float(np.mean([average_precision_score(y_bin[:, k], proba[:, k]) for k in range(n_classes)]))
        y_oh = label_binarize(y_true, classes=list(range(n_classes)))
        m["brier_multiclass"] = float(np.mean(np.sum((proba - y_oh) ** 2, axis=1)))
    return m


@contextlib.contextmanager
def suppress_stdout_stderr():
    devnull = open(os.devnull, "w")
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = devnull, devnull
        yield
    finally:
        sys.stdout, sys.stderr = old_out, old_err
        devnull.close()


def _pkg_versions() -> dict[str, str]:
    import sklearn
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
    }


def top_k_for_model(ds_name: str, spec: dict, model_name: str, args: argparse.Namespace) -> int:
    if model_name != "bworf_mi":
        return 0
    if args.bworf_top_k_override is not None:
        return args.bworf_top_k_override
    return int(spec["default_top_k"])


def run_dataset_model(ds_name: str, spec: dict, model_name: str, seeds: list[int], args: argparse.Namespace, n_jobs: int) -> None:
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir) / ds_name / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    task = spec["task"]
    target = spec["target"]

    df = pd.read_csv(data_dir / spec["file"])
    n_samples = len(df)
    y_raw = df[target].values
    X_df = df.drop(columns=[target])
    feature_cols = list(X_df.columns)

    for c in spec["categorical_cols"]:
        if c in X_df.columns:
            X_df[c] = X_df[c].astype(str)

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    n_classes = len(le.classes_)
    label_map = {str(orig): int(enc) for orig, enc in zip(le.classes_, le.transform(le.classes_))}

    top_k_used = top_k_for_model(ds_name, spec, model_name, args)
    print(f"\n{'='*72}")
    print(f"Dataset: {ds_name} | Model: {model_name} | n={n_samples} | p={len(feature_cols)} | classes={n_classes} | task={task}")
    print(f"Seeds: {seeds} | Folds: {args.n_splits} | top_k_used: {top_k_used}")
    print(f"{'='*72}")

    metric_names = BINARY_METRICS if task == "binary" else MULTI_METRICS
    fold_rows: list[dict] = []

    oof_path = out_dir / "oof_predictions.csv.gz"
    oof_header_written = False
    oof_total_rows = 0

    for seed in seeds:
        oof_pred = np.full(n_samples, -1, dtype=int)
        oof_proba = np.full((n_samples, n_classes), np.nan)
        skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=seed)

        for fold_id, (train_idx, test_idx) in enumerate(skf.split(X_df, y)):
            X_train_df = X_df.iloc[train_idx]
            X_test_df = X_df.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            preprocessor = build_preprocessor(spec, X_train_df)
            X_train = preprocessor.fit_transform(X_train_df)
            X_test = preprocessor.transform(X_test_df)

            if model_name == "bworf_mi":
                X_train, X_test, selected_idx, _ = apply_mutual_information_filtering(
                    X_train, X_test, y_train, top_k_features=top_k_used, plot_mi=False
                )
                note = f"mi_top_k={len(selected_idx)}"
            else:
                note = ""

            model = build_model(model_name, seed, n_jobs, args)
            ctx = f"ds={ds_name} model={model_name} seed={seed} fold={fold_id}"

            t0 = time.perf_counter()
            with suppress_stdout_stderr():
                model.fit(X_train, y_train)
            fit_sec = time.perf_counter() - t0

            t0 = time.perf_counter()
            with suppress_stdout_stderr():
                raw_proba = model.predict_proba(X_test)
                y_pred_model = model.predict(X_test)
            pred_sec = time.perf_counter() - t0

            if raw_proba.ndim == 1 and n_classes == 2:
                raw_proba = np.column_stack([1.0 - raw_proba, raw_proba])

            proba = sanitize_proba(raw_proba, n_classes, context=ctx)
            y_pred = np.argmax(proba, axis=1).astype(int)

            if y_pred_model is not None:
                mismatch_rate = float(np.mean(y_pred_model != y_pred))
                if mismatch_rate > 0.0:
                    note = (note + f" pred_mismatch_rate={mismatch_rate:.3f}").strip()

            mvals = compute_metrics(y_test, y_pred, proba, task, n_classes)

            row = {
                "dataset_name": ds_name,
                "seed": seed,
                "fold_id": fold_id,
                "model": model_name,
                "task_type": task,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "n_classes": n_classes,
            }
            for mk in metric_names:
                row[mk] = mvals.get(mk, np.nan)
            row["fit_seconds"] = fit_sec
            row["pred_seconds"] = pred_sec
            row["model_params_json"] = model_params_json(model_name, args, seed, top_k_used)
            row["notes"] = note
            fold_rows.append(row)

            oof_pred[test_idx] = y_pred.astype(int)
            oof_proba[test_idx] = proba

        has_nan = np.isnan(oof_proba).any()
        if has_nan and not args.drop_incomplete_oof:
            raise ValueError(f"OOF incomplete for {ds_name}/{model_name}/seed={seed}. NaN found in OOF proba.")

        proba_cols = {f"p{k}": oof_proba[:, k] for k in range(n_classes)}
        oof_df = pd.DataFrame({
            "sample_id": np.arange(n_samples),
            "seed": seed,
            "model": model_name,
            "y_true": y,
            "y_pred": oof_pred,
            **proba_cols,
        })
        oof_total_rows += len(oof_df)

        with gzip.open(oof_path, "at", encoding="utf-8") as fh:
            oof_df.to_csv(fh, index=False, header=(not oof_header_written), lineterminator="\n")
        oof_header_written = True

        print(f"  {model_name:<12} seed={seed} done", flush=True)

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(out_dir / "fold_results.csv", index=False)

    sub = fold_df[fold_df["model"] == model_name]
    row_s = {"model": model_name}
    for mk in metric_names:
        if mk in sub.columns:
            row_s[f"{mk}_mean"] = sub[mk].mean()
            row_s[f"{mk}_std"] = sub[mk].std()
    row_s["fit_seconds_mean"] = sub["fit_seconds"].mean()
    row_s["fit_seconds_std"] = sub["fit_seconds"].std()
    pd.DataFrame([row_s]).to_csv(out_dir / "summary.csv", index=False)

    config = {
        "dataset_name": ds_name,
        "file": spec["file"],
        "target": target,
        "categorical_cols": spec["categorical_cols"],
        "task": task,
        "n_samples": n_samples,
        "n_features": len(feature_cols),
        "n_classes": n_classes,
        "label_mapping": label_map,
        "seeds": seeds,
        "n_splits": args.n_splits,
        "model": model_name,
        "top_k_used": top_k_used,
        "cli_args": {k: v for k, v in vars(args).items()},
        "package_versions": _pkg_versions(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "run_config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")

    expected_fold_rows = len(seeds) * args.n_splits
    actual_fold_rows = len(fold_df)
    expected_oof_rows = n_samples * len(seeds)

    print(f"\n  Fold rows: expected={expected_fold_rows} actual={actual_fold_rows}")
    print(f"  OOF rows:  expected={expected_oof_rows} actual={oof_total_rows}")

    if actual_fold_rows != expected_fold_rows:
        raise ValueError(f"[{ds_name}/{model_name}] fold_results row mismatch: expected {expected_fold_rows}, got {actual_fold_rows}")
    if oof_total_rows != expected_oof_rows:
        raise ValueError(f"[{ds_name}/{model_name}] OOF row mismatch: expected {expected_oof_rows}, got {oof_total_rows}")

    print(f"  {ds_name}/{model_name} OK")


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    args = parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    if args.n_jobs is None:
        env_val = os.environ.get("SLURM_CPUS_PER_TASK")
        n_jobs = int(env_val) if env_val else -1
    else:
        n_jobs = args.n_jobs

    ds_names = [s.strip() for s in args.datasets.split(",") if s.strip()]
    model_names = [s.strip() for s in args.models.split(",") if s.strip()]

    for dn in ds_names:
        if dn not in REGISTRY:
            raise ValueError(f"Unknown dataset: {dn}")

    valid_models = {"rf", "orf_style", "bworf_no_mi", "bworf_mi"}
    for mn in model_names:
        if mn not in valid_models:
            raise ValueError(f"Unknown model: {mn}. Valid choices: {sorted(valid_models)}")

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    print(f"Datasets: {ds_names}")
    print(f"Models:   {model_names}")
    print(f"Seeds:    {seeds}")
    print(f"Folds:    {args.n_splits}")
    print(f"n_jobs:   {n_jobs}")

    for dn in ds_names:
        for mn in model_names:
            run_dataset_model(dn, REGISTRY[dn], mn, seeds, args, n_jobs)

    print("\nEXTERNAL BENCHMARK RUN COMPLETE")


if __name__ == "__main__":
    main()
