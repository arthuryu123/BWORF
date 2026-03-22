#!/usr/bin/env python3
"""
Parameterized multiclass DILI benchmark runner for patched BWORF on Sol.

Key upgrade from the original runner:
- oblique-family hyperparameters are now configurable via CLI
- optional robust scaling for the oblique family
- writes a header-only mi_selected_features.csv when no MI model is run

Example:
python code/run_multiclass_dili_patched_benchmark_v2.py \
  --data_path data/dili_multiclass.csv \
  --feature_names_path data/dili_multiclass_feature_names.csv \
  --output_root outputs \
  --run_name bworf_rescue_mi100 \
  --n_splits 10 \
  --seeds 42,43,44,45,46 \
  --models bworf_mi \
  --mi_top_k 100 \
  --oblique_n_estimators 300 \
  --oblique_max_depth 8 \
  --oblique_l1_strength 0.05 \
  --oblique_n_tries 20 \
  --n_jobs 64
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
except Exception as exc:
    raise ImportError("xgboost is required for this runner") from exc

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from bworf_with_mi import ObliqueRandomForestMulti  # noqa: E402


ALL_MODELS = [
    "rf",
    "orf_style",
    "lr",
    "svm_rbf",
    "xgb",
    "naive_bayes",
    "bworf_no_mi",
    "bworf_mi",
]
CLASS_ORDER = np.array([0, 1, 2], dtype=int)
DATASET_NAME = "dili_multiclass"
ID_COL = "USER_ID"
TARGET_COL = "y_3class"
EXCLUDE_COLS = {
    "USER_ID",
    "Likelihood_letter_raw",
    "Likelihood_letter_base",
    "y_3class",
    "y_3class_name",
}
FEATURE_PATTERN = re.compile(r"D\d{3}$")


@dataclass
class ProcessedFold:
    X_train: np.ndarray
    X_test: np.ndarray
    feature_names: List[str]
    mi_rows: List[Dict[str, object]]
    feature_space_row: Dict[str, object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run parameterized patched multiclass DILI benchmark.")
    parser.add_argument("--data_path", required=True, help="Path to dili_multiclass.csv")
    parser.add_argument("--feature_names_path", default=None, help="Optional path to feature names csv")
    parser.add_argument("--output_root", required=True, help="Root directory for run outputs")
    parser.add_argument("--run_name", default=None, help="Run name prefix")
    parser.add_argument("--n_splits", type=int, default=10, help="Number of CV folds")
    parser.add_argument("--seeds", default="42,43,44,45,46", help="Comma-separated seed list")
    parser.add_argument("--models", default=",".join(ALL_MODELS), help="Comma-separated model names")
    parser.add_argument("--n_jobs", type=int, default=1, help="Parallel workers for supported models")
    parser.add_argument("--mi_top_k", type=int, default=50, help="Top-k features for bworf_mi")
    parser.add_argument("--keep_model_stdout", action="store_true", help="Keep verbose BWORF fit logs")
    parser.add_argument("--smoke", action="store_true", help="Smoke test mode")

    # Oblique-family tunable hyperparameters
    parser.add_argument("--oblique_n_estimators", type=int, default=100)
    parser.add_argument("--oblique_max_depth", type=int, default=6)
    parser.add_argument("--oblique_min_samples_split", type=int, default=10)
    parser.add_argument("--oblique_min_samples_leaf", type=int, default=1)
    parser.add_argument("--oblique_l1_strength", type=float, default=0.2)
    parser.add_argument("--oblique_n_tries", type=int, default=10)
    parser.add_argument("--oblique_bootstrap_temperature", type=float, default=1.0)
    parser.add_argument(
        "--oblique_scale_mode",
        choices=["standard", "robust", "none"],
        default="standard",
        help="Scaling strategy for ORF/BWORF family",
    )
    return parser.parse_args()


def load_dataset(data_path: str) -> Tuple[pd.DataFrame, List[str]]:
    df = pd.read_csv(data_path)
    feature_cols = [c for c in df.columns if FEATURE_PATTERN.fullmatch(c)]
    if len(feature_cols) != 777:
        raise ValueError(f"Expected 777 descriptor columns, found {len(feature_cols)}")
    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COL}")
    if ID_COL not in df.columns:
        raise ValueError(f"Missing ID column: {ID_COL}")
    y_values = set(pd.unique(df[TARGET_COL]))
    if y_values != {0, 1, 2}:
        raise ValueError(f"Expected target classes {{0,1,2}}, got {sorted(y_values)}")
    return df, feature_cols


def maybe_read_feature_name_map(feature_names_path: str | None) -> pd.DataFrame | None:
    if not feature_names_path:
        return None
    path = Path(feature_names_path)
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def parse_model_list(raw: str) -> List[str]:
    models = [m.strip() for m in raw.split(",") if m.strip()]
    invalid = [m for m in models if m not in ALL_MODELS]
    if invalid:
        raise ValueError(f"Invalid model names: {invalid}")
    if len(set(models)) != len(models):
        raise ValueError("Duplicate model names are not allowed")
    return models


def parse_seeds(raw: str) -> List[int]:
    seeds = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not seeds:
        raise ValueError("At least one seed is required")
    return seeds


def ensure_run_dir(output_root: str, run_name: str | None) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = run_name if run_name else "multiclass_dili_patched"
    run_dir = Path(output_root) / f"{prefix}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def select_feature_columns(feature_cols: Sequence[str]) -> List[str]:
    selected = [c for c in feature_cols if c not in EXCLUDE_COLS]
    if len(selected) != 777:
        raise ValueError(f"Expected 777 usable descriptors, got {len(selected)}")
    return selected


def fit_imputer(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    imputer = SimpleImputer(strategy="median")
    return imputer.fit_transform(X_train), imputer.transform(X_test)


def apply_train_fold_mi(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    feature_names: Sequence[str],
    top_k: int,
    seed: int,
    fold: int,
    model_name: str,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[Dict[str, object]]]:
    n_features = X_train.shape[1]
    mi_scores = mutual_info_classif(X_train, y_train, random_state=seed)
    ranked = np.argsort(mi_scores)[::-1]
    mi_rows = []
    for rank, idx in enumerate(ranked, start=1):
        mi_rows.append(
            {
                "seed": seed,
                "fold": fold,
                "model_name": model_name,
                "rank": rank,
                "feature_name": feature_names[idx],
                "mi_score": float(mi_scores[idx]),
            }
        )

    if top_k <= 0 or top_k >= n_features:
        return X_train, X_test, list(feature_names), mi_rows

    selected_idx = ranked[:top_k]
    selected_features = [feature_names[i] for i in selected_idx]
    return X_train[:, selected_idx], X_test[:, selected_idx], selected_features, mi_rows


def maybe_scale(
    X_train: np.ndarray,
    X_test: np.ndarray,
    scale_mode: str,
) -> Tuple[np.ndarray, np.ndarray]:
    if scale_mode == "none":
        return X_train, X_test
    if scale_mode == "robust":
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test)


def preprocess_for_model(
    model_name: str,
    X_train_df: pd.DataFrame,
    X_test_df: pd.DataFrame,
    y_train: np.ndarray,
    feature_names: List[str],
    mi_top_k: int,
    seed: int,
    fold: int,
    oblique_scale_mode: str,
) -> ProcessedFold:
    X_train_imp, X_test_imp = fit_imputer(X_train_df, X_test_df)

    mi_rows: List[Dict[str, object]] = []
    current_features = list(feature_names)
    mi_used = False

    if model_name == "bworf_mi":
        X_train_imp, X_test_imp, current_features, mi_rows = apply_train_fold_mi(
            X_train_imp, X_test_imp, y_train, current_features, mi_top_k, seed, fold, model_name
        )
        mi_used = True

    if model_name in {"orf_style", "bworf_no_mi", "bworf_mi"}:
        do_scale = oblique_scale_mode != "none"
        X_train_proc, X_test_proc = maybe_scale(X_train_imp, X_test_imp, oblique_scale_mode)
        used_scaling = oblique_scale_mode
    elif model_name in {"lr", "svm_rbf"}:
        do_scale = True
        X_train_proc, X_test_proc = maybe_scale(X_train_imp, X_test_imp, "standard")
        used_scaling = "standard"
    else:
        do_scale = False
        X_train_proc, X_test_proc = X_train_imp, X_test_imp
        used_scaling = "none"

    feature_space_row = {
        "seed": seed,
        "fold": fold,
        "model_name": model_name,
        "original_feature_count": len(feature_names),
        "selected_feature_count": len(current_features),
        "used_mi": bool(mi_used),
        "used_scaling": used_scaling,
        "used_median_imputation": True,
    }

    return ProcessedFold(
        X_train=X_train_proc,
        X_test=X_test_proc,
        feature_names=current_features,
        mi_rows=mi_rows,
        feature_space_row=feature_space_row,
    )


def build_model(model_name: str, seed: int, n_jobs: int, args: argparse.Namespace):
    if model_name == "rf":
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            min_samples_split=10,
            min_samples_leaf=1,
            max_features="sqrt",
            random_state=seed,
            n_jobs=n_jobs,
        )
    if model_name in {"orf_style", "bworf_no_mi", "bworf_mi"}:
        weighted_bootstrap = model_name != "orf_style"
        return ObliqueRandomForestMulti(
            n_estimators=args.oblique_n_estimators,
            max_depth=args.oblique_max_depth,
            min_samples_split=args.oblique_min_samples_split,
            min_samples_leaf=args.oblique_min_samples_leaf,
            l1_strength=args.oblique_l1_strength,
            random_state=seed,
            weighted_bootstrap=weighted_bootstrap,
            bootstrap_temperature=args.oblique_bootstrap_temperature,
            n_tries=args.oblique_n_tries,
        )
    if model_name == "lr":
        return LogisticRegression(
            solver="lbfgs",
            C=1.0,
            max_iter=4000,
            multi_class="auto",
            random_state=seed,
        )
    if model_name == "svm_rbf":
        return SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,
            random_state=seed,
        )
    if model_name == "xgb":
        return XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=1.0,
            colsample_bytree=1.0,
            objective="multi:softprob",
            eval_metric="mlogloss",
            num_class=3,
            random_state=seed,
            n_jobs=n_jobs,
            verbosity=0,
        )
    if model_name == "naive_bayes":
        return GaussianNB()
    raise ValueError(f"Unsupported model: {model_name}")


def align_proba_columns(proba: np.ndarray, model, class_order: np.ndarray, fallback_classes: np.ndarray) -> np.ndarray:
    model_classes = getattr(model, "classes_", None)
    if model_classes is None:
        model_classes = fallback_classes
    model_classes = np.asarray(model_classes)

    out = np.zeros((proba.shape[0], len(class_order)), dtype=float)
    for src_idx, cls in enumerate(model_classes):
        dst_idx = int(np.where(class_order == cls)[0][0])
        out[:, dst_idx] = proba[:, src_idx]
    return out


def compute_macro_auprc_ovr(y_true: np.ndarray, proba: np.ndarray, class_order: np.ndarray) -> float:
    vals = []
    for i, cls in enumerate(class_order):
        y_bin = (y_true == cls).astype(int)
        vals.append(average_precision_score(y_bin, proba[:, i]))
    return float(np.mean(vals))


def compute_fold_metrics(y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "macro_auroc_ovr": float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro", labels=CLASS_ORDER)),
        "macro_auprc_ovr": compute_macro_auprc_ovr(y_true, proba, CLASS_ORDER),
        "log_loss": float(log_loss(y_true, proba, labels=CLASS_ORDER)),
    }


def fit_and_predict(model, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, keep_stdout: bool):
    fit_start = time.perf_counter()
    if keep_stdout:
        model.fit(X_train, y_train)
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            model.fit(X_train, y_train)
    fit_time = time.perf_counter() - fit_start

    pred_start = time.perf_counter()
    y_pred = model.predict(X_test)
    proba = model.predict_proba(X_test)
    pred_time = time.perf_counter() - pred_start
    return y_pred, proba, fit_time, pred_time


def build_split_manifest_rows(df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray, seed: int, fold: int):
    rows = []
    for split_role, idxs in (("train", train_idx), ("test", test_idx)):
        chunk = df.iloc[idxs][[ID_COL, TARGET_COL]].copy()
        chunk["sample_index"] = idxs
        chunk["seed"] = seed
        chunk["fold"] = fold
        chunk["split_role"] = split_role
        chunk["dataset"] = DATASET_NAME
        rows.extend(
            chunk.rename(columns={TARGET_COL: "y_true"})[
                ["dataset", "sample_index", ID_COL, "seed", "fold", "split_role", "y_true"]
            ].to_dict(orient="records")
        )
    return rows


def aggregate_summary(fold_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "macro_recall",
        "weighted_f1",
        "mcc",
        "macro_auroc_ovr",
        "macro_auprc_ovr",
        "log_loss",
    ]
    grouped = fold_df.groupby("model_name", dropna=False)[metric_cols].agg(["mean", "std"])
    grouped.columns = [f"{metric}_{stat}" for metric, stat in grouped.columns]
    return grouped.reset_index()


def build_confusion_payload(oof_df: pd.DataFrame) -> Dict[str, object]:
    payload = {"per_seed": {}, "pooled": {}}
    for seed in sorted(oof_df["seed"].unique().tolist()):
        payload["per_seed"][str(seed)] = {}
        sub_seed = oof_df[oof_df["seed"] == seed]
        for model_name in sorted(sub_seed["model_name"].unique().tolist()):
            sub = sub_seed[sub_seed["model_name"] == model_name]
            cm = confusion_matrix(sub["y_true"], sub["y_pred"], labels=CLASS_ORDER)
            row_norm = cm.astype(float) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
            payload["per_seed"][str(seed)][model_name] = {
                "raw": cm.tolist(),
                "row_normalized": row_norm.tolist(),
            }
    for model_name in sorted(oof_df["model_name"].unique().tolist()):
        sub = oof_df[oof_df["model_name"] == model_name]
        cm = confusion_matrix(sub["y_true"], sub["y_pred"], labels=CLASS_ORDER)
        row_norm = cm.astype(float) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
        payload["pooled"][model_name] = {
            "raw": cm.tolist(),
            "row_normalized": row_norm.tolist(),
        }
    return payload


def build_sanity_payload(df: pd.DataFrame, fold_df: pd.DataFrame, oof_df: pd.DataFrame, models: Sequence[str], seeds: Sequence[int], n_splits: int):
    expected_fold_rows = len(models) * len(seeds) * n_splits
    expected_oof_rows = len(df) * len(models) * len(seeds)

    duplicate_oof_key_count = int(oof_df.duplicated(subset=["sample_index", "seed", "model_name"]).sum())
    proba_sums = oof_df[["proba_0", "proba_1", "proba_2"]].sum(axis=1).to_numpy(dtype=float) if len(oof_df) else np.array([1.0])
    max_abs_error = float(np.max(np.abs(proba_sums - 1.0)))

    return {
        "dataset": DATASET_NAME,
        "n_samples": int(len(df)),
        "n_features": 777,
        "n_models": int(len(models)),
        "n_seeds": int(len(seeds)),
        "n_folds": int(n_splits),
        "class_counts": df[TARGET_COL].value_counts().sort_index().to_dict(),
        "expected_fold_rows": int(expected_fold_rows),
        "actual_fold_rows": int(len(fold_df)),
        "fold_rows_match_expected": bool(len(fold_df) == expected_fold_rows),
        "expected_oof_rows": int(expected_oof_rows),
        "actual_oof_rows": int(len(oof_df)),
        "oof_rows_match_expected": bool(len(oof_df) == expected_oof_rows),
        "duplicate_oof_key_count": duplicate_oof_key_count,
        "oof_probability_sum_max_abs_error": max_abs_error,
    }


def run_benchmark(args: argparse.Namespace) -> Path:
    df, feature_cols = load_dataset(args.data_path)
    feature_cols = select_feature_columns(feature_cols)
    _feature_names = maybe_read_feature_name_map(args.feature_names_path)

    models = parse_model_list(args.models)
    seeds = parse_seeds(args.seeds)
    n_splits = int(args.n_splits)

    if args.smoke:
        seeds = seeds[:1]
        n_splits = 3
        models = ["rf", "naive_bayes", "bworf_no_mi"]

    run_dir = ensure_run_dir(args.output_root, args.run_name)

    X_df = df[feature_cols].copy()
    y = df[TARGET_COL].to_numpy(dtype=int)

    fold_rows: List[Dict[str, object]] = []
    oof_rows: List[Dict[str, object]] = []
    split_rows: List[Dict[str, object]] = []
    mi_rows_all: List[Dict[str, object]] = []
    feature_space_rows: List[Dict[str, object]] = []

    for seed in seeds:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_df, y), start=1):
            X_train_df = X_df.iloc[train_idx].copy()
            X_test_df = X_df.iloc[test_idx].copy()
            y_train = y[train_idx]
            y_test = y[test_idx]

            split_rows.extend(build_split_manifest_rows(df, train_idx, test_idx, seed, fold_idx))
            train_counts = pd.Series(y_train).value_counts().reindex(CLASS_ORDER, fill_value=0)
            test_counts = pd.Series(y_test).value_counts().reindex(CLASS_ORDER, fill_value=0)

            for model_name in models:
                processed = preprocess_for_model(
                    model_name=model_name,
                    X_train_df=X_train_df,
                    X_test_df=X_test_df,
                    y_train=y_train,
                    feature_names=feature_cols,
                    mi_top_k=args.mi_top_k,
                    seed=seed,
                    fold=fold_idx,
                    oblique_scale_mode=args.oblique_scale_mode,
                )
                feature_space_rows.append(processed.feature_space_row)
                if processed.mi_rows:
                    mi_rows_all.extend(processed.mi_rows)

                model = build_model(model_name, seed=seed, n_jobs=args.n_jobs, args=args)
                y_pred, proba_raw, fit_time, pred_time = fit_and_predict(
                    model, processed.X_train, y_train, processed.X_test, keep_stdout=args.keep_model_stdout
                )
                proba = align_proba_columns(proba_raw, model, CLASS_ORDER, np.unique(y_train))
                metrics = compute_fold_metrics(y_test, y_pred, proba)

                fold_rows.append(
                    {
                        "dataset": DATASET_NAME,
                        "seed": seed,
                        "fold": fold_idx,
                        "model_name": model_name,
                        "n_train": int(len(train_idx)),
                        "n_test": int(len(test_idx)),
                        "class0_train": int(train_counts.loc[0]),
                        "class1_train": int(train_counts.loc[1]),
                        "class2_train": int(train_counts.loc[2]),
                        "class0_test": int(test_counts.loc[0]),
                        "class1_test": int(test_counts.loc[1]),
                        "class2_test": int(test_counts.loc[2]),
                        "fit_time_sec": float(fit_time),
                        "predict_time_sec": float(pred_time),
                        **metrics,
                    }
                )

                user_ids = df.iloc[test_idx][ID_COL].tolist()
                sample_indices = test_idx.tolist()
                for row_pos, sample_index in enumerate(sample_indices):
                    oof_rows.append(
                        {
                            "dataset": DATASET_NAME,
                            "sample_index": int(sample_index),
                            "USER_ID": int(user_ids[row_pos]),
                            "seed": seed,
                            "fold": fold_idx,
                            "model_name": model_name,
                            "y_true": int(y_test[row_pos]),
                            "y_pred": int(y_pred[row_pos]),
                            "proba_0": float(proba[row_pos, 0]),
                            "proba_1": float(proba[row_pos, 1]),
                            "proba_2": float(proba[row_pos, 2]),
                        }
                    )

    fold_df = pd.DataFrame(fold_rows)
    oof_df = pd.DataFrame(oof_rows)
    split_df = pd.DataFrame(split_rows)
    mi_columns = ["seed", "fold", "model_name", "rank", "feature_name", "mi_score"]
    mi_df = pd.DataFrame(mi_rows_all, columns=mi_columns)
    feature_space_df = pd.DataFrame(feature_space_rows)
    summary_df = aggregate_summary(fold_df)
    confusion_payload = build_confusion_payload(oof_df)
    sanity_payload = build_sanity_payload(df, fold_df, oof_df, models=models, seeds=seeds, n_splits=n_splits)

    fold_df.to_csv(run_dir / "fold_results.csv", index=False)
    summary_df.to_csv(run_dir / "summary_metrics.csv", index=False)
    oof_df.to_csv(run_dir / "oof_predictions.csv", index=False)
    split_df.to_csv(run_dir / "split_manifest.csv", index=False)
    mi_df.to_csv(run_dir / "mi_selected_features.csv", index=False)
    feature_space_df.to_csv(run_dir / "feature_space_manifest.csv", index=False)

    with open(run_dir / "confusion_matrices.json", "w", encoding="utf-8") as f:
        json.dump(confusion_payload, f, indent=2)
    with open(run_dir / "sanity_check.json", "w", encoding="utf-8") as f:
        json.dump(sanity_payload, f, indent=2)
    with open(run_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": DATASET_NAME,
                "data_path": str(Path(args.data_path).resolve()),
                "feature_names_path": str(Path(args.feature_names_path).resolve()) if args.feature_names_path else None,
                "output_root": str(Path(args.output_root).resolve()),
                "run_dir": str(run_dir.resolve()),
                "seeds": seeds,
                "n_splits": n_splits,
                "models": models,
                "mi_top_k": int(args.mi_top_k),
                "target_col": TARGET_COL,
                "id_col": ID_COL,
                "feature_rule": "columns matching D###",
                "excluded_from_modeling": sorted(EXCLUDE_COLS),
                "class_order": CLASS_ORDER.tolist(),
                "notes": {
                    "runner_plotting_separate": True,
                    "hard_predictions_from_model_predict": True,
                    "probabilities_from_predict_proba": True,
                    "full_split_manifest": True,
                    "oblique_scale_mode": args.oblique_scale_mode,
                    "header_only_mi_file_when_no_mi_model": True,
                },
                "hyperparameters": {
                    "rf": {
                        "n_estimators": 100,
                        "max_depth": 6,
                        "min_samples_split": 10,
                        "min_samples_leaf": 1,
                        "max_features": "sqrt",
                    },
                    "oblique_family": {
                        "n_estimators": args.oblique_n_estimators,
                        "max_depth": args.oblique_max_depth,
                        "min_samples_split": args.oblique_min_samples_split,
                        "min_samples_leaf": args.oblique_min_samples_leaf,
                        "l1_strength": args.oblique_l1_strength,
                        "bootstrap_temperature": args.oblique_bootstrap_temperature,
                        "n_tries": args.oblique_n_tries,
                        "scale_mode": args.oblique_scale_mode,
                    },
                    "bworf_mi": {"mi_top_k": int(args.mi_top_k)},
                    "lr": {"solver": "lbfgs", "C": 1.0, "max_iter": 4000},
                    "svm_rbf": {"kernel": "rbf", "C": 1.0, "gamma": "scale", "probability": True},
                    "xgb": {
                        "n_estimators": 100,
                        "max_depth": 6,
                        "learning_rate": 0.1,
                        "subsample": 1.0,
                        "colsample_bytree": 1.0,
                        "objective": "multi:softprob",
                        "eval_metric": "mlogloss",
                        "num_class": 3,
                    },
                    "naive_bayes": {"type": "GaussianNB"},
                },
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "script_name": Path(__file__).name,
            },
            f,
            indent=2,
        )

    return run_dir


def main() -> None:
    args = parse_args()
    run_dir = run_benchmark(args)
    print(f"Run completed: {run_dir}")


if __name__ == "__main__":
    main()
