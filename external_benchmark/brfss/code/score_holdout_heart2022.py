"""
score_holdout_heart2022.py — refit each (model, seed) on the full modeling set
and score on the held-out 20% test set.

This is the confirmatory test-set evaluation we set aside in prep_heart2022.py.
The holdout was not seen by any CV fold; it's touched exactly once, here, after
all CV is complete.

For each model + seed, we:
    1. Load the modeling set (train) and the holdout set (test).
    2. Fit a fresh model with locked hyperparameters on the entire modeling set.
    3. Predict on the holdout.
    4. Compute the same 10 binary metrics the runner reports.

Output:
    <out_dir>/_aggregated/<dataset>/holdout_scores.csv  (one row per model+seed)

Usage:

    cd ~/external_benchmark
    # subsample (matches CV dataset name "heart_attack_2022_brfss"):
    python score_holdout_heart2022.py \\
        --modeling_csv data/cleaned/heart_attack_2022_brfss.csv.gz \\
        --holdout_csv  data/cleaned/heart_attack_2022_brfss_holdout.csv.gz \\
        --models lr,rf,xgb,orf,bworf \\
        --out_dir outputs_heart2022_subsample \\
        --dataset heart_attack_2022_brfss

    # full-N (no BWORF):
    python score_holdout_heart2022.py \\
        --modeling_csv data/cleaned/heart_attack_2022_brfss_full.csv.gz \\
        --holdout_csv  data/cleaned/heart_attack_2022_brfss_full_holdout.csv.gz \\
        --models lr,rf,xgb,orf \\
        --out_dir outputs_heart2022_full \\
        --dataset heart_attack_2022_brfss_full
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    f1_score, log_loss, matthews_corrcoef, precision_score, recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

# Optional deps (same lazy-import policy as the runner)
try:
    import xgboost as xgb
    _HAS_XGB = True
except (ImportError, OSError):
    _HAS_XGB = False

# Use the same models as the runner
_models_dir = str(Path(__file__).resolve().parent / "models")
if _models_dir not in sys.path:
    sys.path.insert(0, _models_dir)
from bworf_with_mi import BWORFClassifier  # noqa: E402
from bworf_parallel import BWORFParallelClassifier  # noqa: E402
from orf_treeple import check_orf_available, make_orf, predict_proba_aligned  # noqa: E402
_HAS_ORF = check_orf_available()


# Matches the runner's REGISTRY entries for these datasets.
# Hard-coded here to avoid importing the runner module.
DATASET_SPECS = {
    "heart_attack_2022_brfss": {
        "target": "HadHeartAttack",
        "categorical_cols": ["RaceEthnicityCategory", "TetanusLast10Tdap", "CovidPos"],
    },
    "heart_attack_2022_brfss_full": {
        "target": "HadHeartAttack",
        "categorical_cols": ["RaceEthnicityCategory", "TetanusLast10Tdap", "CovidPos"],
    },
}


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


def build_preprocessor(spec: dict, X_columns, for_lr: bool) -> Pipeline:
    cat_cols = [c for c in spec["categorical_cols"] if c in X_columns]
    num_cols = [c for c in X_columns if c not in cat_cols]
    transformers = []
    if num_cols:
        transformers.append(("num", SimpleImputer(strategy="median"), num_cols))
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
    steps = [("prep", ct)]
    if for_lr:
        steps.append(("scale", StandardScaler(with_mean=False)))
    return Pipeline(steps)


def build_model(model_name, seed, n_jobs, n_classes, y_train):
    if model_name == "lr":
        return LogisticRegression(max_iter=2000, class_weight="balanced",
                                  solver="lbfgs")
    if model_name == "rf":
        return RandomForestClassifier(n_estimators=100, max_depth=5,
                                      class_weight="balanced",
                                      random_state=seed, n_jobs=n_jobs)
    if model_name == "xgb":
        if not _HAS_XGB:
            return None
        params = dict(n_estimators=100, max_depth=5, n_jobs=n_jobs,
                      random_state=seed, verbosity=0,
                      objective="binary:logistic", eval_metric="logloss")
        n_neg = int((y_train == 0).sum())
        n_pos = int((y_train == 1).sum())
        if n_pos > 0:
            params["scale_pos_weight"] = n_neg / n_pos
        return xgb.XGBClassifier(**params)
    if model_name == "orf":
        if not _HAS_ORF:
            return None
        return make_orf(random_state=seed, n_estimators=100, max_depth=5,
                        n_jobs=n_jobs, class_weight="balanced")
    if model_name == "bworf":
        return BWORFClassifier(
            n_estimators=100, max_depth=5, l1_strength=1.0,
            weighted_bootstrap=True, bootstrap_temperature=1.0,
            n_tries=2, random_state=seed,
        )
    if model_name == "bworf_parallel":
        return BWORFParallelClassifier(
            n_estimators=100, max_depth=5, l1_strength=1.0,
            weighted_bootstrap=True, bootstrap_temperature=1.0,
            n_tries=2, random_state=seed,
            n_jobs=n_jobs,
        )
    raise ValueError(f"Unknown model: {model_name}")


def compute_metrics(y_true, y_pred, proba):
    p1 = proba[:, 1]
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_pos": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "precision_pos": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_pos": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "auroc": roc_auc_score(y_true, p1),
        "auprc": average_precision_score(y_true, p1),
        "log_loss": log_loss(y_true, proba, labels=[0, 1]),
        "brier": float(np.mean((p1 - y_true) ** 2)),
    }


def parse_args():
    p = argparse.ArgumentParser(description="Holdout scorer")
    p.add_argument("--modeling_csv", required=True)
    p.add_argument("--holdout_csv", required=True)
    p.add_argument("--models", default="lr,rf,xgb,orf,bworf")
    p.add_argument("--seeds", default="13,42,77,123,2025")
    p.add_argument("--n_jobs", type=int, default=None)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--dataset", required=True,
                   help="Dataset name (key into DATASET_SPECS)")
    return p.parse_args()


def main():
    args = parse_args()
    if args.dataset not in DATASET_SPECS:
        raise ValueError(f"Unknown dataset: {args.dataset}. "
                         f"Known: {list(DATASET_SPECS)}")
    spec = DATASET_SPECS[args.dataset]
    target = spec["target"]

    if args.n_jobs is None:
        env_val = os.environ.get("SLURM_CPUS_PER_TASK")
        n_jobs = int(env_val) if env_val else -1
    else:
        n_jobs = args.n_jobs

    seeds = [int(s) for s in args.seeds.split(",")]
    model_names = [s.strip() for s in args.models.split(",")]

    # Load data
    df_mod = pd.read_csv(args.modeling_csv)
    df_hold = pd.read_csv(args.holdout_csv)
    print(f"Modeling set: {len(df_mod)} rows ({df_mod[target].mean()*100:.2f}% pos)")
    print(f"Holdout set:  {len(df_hold)} rows ({df_hold[target].mean()*100:.2f}% pos)")

    y_train_raw = df_mod[target].values
    X_train_df = df_mod.drop(columns=[target])
    y_test_raw = df_hold[target].values
    X_test_df = df_hold.drop(columns=[target])
    feature_cols = list(X_train_df.columns)

    # Cast categorical cols to string (consistent with runner)
    for c in spec["categorical_cols"]:
        if c in X_train_df.columns:
            X_train_df[c] = X_train_df[c].astype(str)
            X_test_df[c] = X_test_df[c].astype(str)

    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)
    y_test = le.transform(y_test_raw)
    n_classes = len(le.classes_)
    if n_classes != 2:
        raise ValueError(f"Expected binary task, got {n_classes} classes")

    # Iterate
    rows = []
    for model_name in model_names:
        for seed in seeds:
            ctx = f"{model_name}/seed={seed}"

            preprocessor = build_preprocessor(spec, feature_cols,
                                              for_lr=(model_name == "lr"))
            X_tr = preprocessor.fit_transform(X_train_df)
            X_te = preprocessor.transform(X_test_df)

            model = build_model(model_name, seed, n_jobs, n_classes, y_train)
            if model is None:
                print(f"  {ctx}: skipped (missing dependency)")
                continue

            t0 = time.perf_counter()
            try:
                if model_name in ("bworf", "bworf_parallel"):
                    with suppress_stdout_stderr():
                        model.fit(X_tr, y_train)
                else:
                    model.fit(X_tr, y_train)
            except Exception as e:
                print(f"  {ctx}: fit FAILED: {type(e).__name__}: {e}")
                continue
            fit_sec = time.perf_counter() - t0

            t0 = time.perf_counter()
            try:
                if model_name in ("bworf", "bworf_parallel"):
                    with suppress_stdout_stderr():
                        proba = model.predict_proba(X_te)
                elif model_name == "orf":
                    proba = predict_proba_aligned(model, X_te, n_classes)
                else:
                    proba = model.predict_proba(X_te)
            except Exception as e:
                print(f"  {ctx}: predict FAILED: {type(e).__name__}: {e}")
                continue
            pred_sec = time.perf_counter() - t0

            proba = np.asarray(proba, dtype=np.float64)
            if proba.ndim == 1:
                proba = np.column_stack([1.0 - proba, proba])
            proba = np.clip(proba, 1e-15, 1.0 - 1e-15)
            y_pred = np.argmax(proba, axis=1).astype(int)

            m = compute_metrics(y_test, y_pred, proba)
            row = {
                "dataset": args.dataset,
                "model": model_name,
                "seed": seed,
                "n_train": len(y_train),
                "n_holdout": len(y_test),
                "fit_seconds": fit_sec,
                "predict_seconds": pred_sec,
                **m,
            }
            rows.append(row)
            print(f"  {ctx}: AUROC={m['auroc']:.4f} BalAcc={m['balanced_accuracy']:.4f} "
                  f"MCC={m['mcc']:.4f} fit={fit_sec:.1f}s")

    # Write output
    out_dir = Path(args.out_dir)
    agg_dir = out_dir / "_aggregated" / args.dataset
    agg_dir.mkdir(parents=True, exist_ok=True)
    df_out = pd.DataFrame(rows)
    out_path = agg_dir / "holdout_scores.csv"
    df_out.to_csv(out_path, index=False)
    print(f"\nWrote {out_path} ({len(df_out)} rows)")

    # Summary
    print("\n=== Holdout summary (mean across seeds) ===")
    if len(df_out):
        summary = df_out.groupby("model")[
            ["auroc", "balanced_accuracy", "mcc", "f1_pos", "auprc"]
        ].agg(["mean", "std"])
        print(summary.round(4).to_string())

    # Manifest
    manifest = {
        "dataset": args.dataset,
        "modeling_csv": args.modeling_csv,
        "holdout_csv": args.holdout_csv,
        "models": model_names,
        "seeds": seeds,
        "n_train": len(y_train),
        "n_holdout": len(y_test),
        "label_mapping": {str(c): int(le.transform([c])[0]) for c in le.classes_},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    (agg_dir / "holdout_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
