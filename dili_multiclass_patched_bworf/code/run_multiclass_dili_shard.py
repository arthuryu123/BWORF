#!/usr/bin/env python3
"""
Shard runner for multiclass DILI benchmark.

Purpose:
- run exactly one model on exactly one (seed, fold) shard
- preserve the same logic as the parameterized v2 runner
- write shard-local outputs that can later be merged into the usual final artifacts

Typical Slurm array usage:
- seeds = 42,43,44,45,46
- n_splits = 10
- task_id 0..49 mapped as:
    seed_idx = task_id // 10
    fold     = task_id % 10 + 1
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
    p = argparse.ArgumentParser(description="Run one multiclass DILI shard (one seed-fold for one model).")
    p.add_argument("--data_path", required=True)
    p.add_argument("--feature_names_path", default=None)
    p.add_argument("--output_root", required=True)
    p.add_argument("--run_name", required=True, help="Config name prefix; shards will go under <run_name>_shards/")
    p.add_argument("--n_splits", type=int, default=10)
    p.add_argument("--seeds", default="42,43,44,45,46")
    p.add_argument("--model", required=True, choices=ALL_MODELS)

    p.add_argument("--seed", type=int, default=None, help="Seed to run (required unless --task_id is used)")
    p.add_argument("--fold", type=int, default=None, help="1-based fold index (required unless --task_id is used)")
    p.add_argument("--task_id", type=int, default=None, help="Optional array task id used to derive seed/fold")

    p.add_argument("--n_jobs", type=int, default=1)
    p.add_argument("--mi_top_k", type=int, default=50)
    p.add_argument("--keep_model_stdout", action="store_true")

    p.add_argument("--oblique_n_estimators", type=int, default=100)
    p.add_argument("--oblique_max_depth", type=int, default=6)
    p.add_argument("--oblique_min_samples_split", type=int, default=10)
    p.add_argument("--oblique_min_samples_leaf", type=int, default=1)
    p.add_argument("--oblique_l1_strength", type=float, default=0.2)
    p.add_argument("--oblique_n_tries", type=int, default=10)
    p.add_argument("--oblique_bootstrap_temperature", type=float, default=1.0)
    p.add_argument("--oblique_scale_mode", choices=["standard", "robust", "none"], default="standard")
    return p.parse_args()


def parse_seeds(raw: str) -> List[int]:
    seeds = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not seeds:
        raise ValueError("At least one seed is required")
    return seeds


def resolve_seed_fold(args: argparse.Namespace) -> Tuple[int, int, List[int]]:
    seeds = parse_seeds(args.seeds)
    if args.task_id is not None:
        total = len(seeds) * args.n_splits
        if args.task_id < 0 or args.task_id >= total:
            raise ValueError(f"task_id must be in [0, {total-1}], got {args.task_id}")
        seed = seeds[args.task_id // args.n_splits]
        fold = (args.task_id % args.n_splits) + 1
        return seed, fold, seeds

    if args.seed is None or args.fold is None:
        raise ValueError("Either provide --task_id or both --seed and --fold")
    if args.seed not in seeds:
        raise ValueError(f"Seed {args.seed} not found in --seeds {seeds}")
    if args.fold < 1 or args.fold > args.n_splits:
        raise ValueError(f"Fold must be in [1, {args.n_splits}]")
    return args.seed, args.fold, seeds


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


def maybe_scale(X_train: np.ndarray, X_test: np.ndarray, scale_mode: str) -> Tuple[np.ndarray, np.ndarray]:
    if scale_mode == "none":
        return X_train, X_test
    scaler = RobustScaler() if scale_mode == "robust" else StandardScaler()
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
        X_train_proc, X_test_proc = maybe_scale(X_train_imp, X_test_imp, oblique_scale_mode)
        used_scaling = oblique_scale_mode
    elif model_name in {"lr", "svm_rbf"}:
        X_train_proc, X_test_proc = maybe_scale(X_train_imp, X_test_imp, "standard")
        used_scaling = "standard"
    else:
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
        return SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=seed)
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


def ensure_shard_dir(output_root: str, run_name: str, seed: int, fold: int) -> Path:
    root = Path(output_root) / f"{run_name}_shards" / f"seed_{seed}_fold_{fold:02d}"
    root.mkdir(parents=True, exist_ok=False)
    return root


def main():
    args = parse_args()
    seed, fold, seeds = resolve_seed_fold(args)

    df, feature_cols = load_dataset(args.data_path)
    feature_cols = select_feature_columns(feature_cols)
    _feature_names_ref = maybe_read_feature_name_map(args.feature_names_path)

    shard_dir = ensure_shard_dir(args.output_root, args.run_name, seed, fold)

    X_df = df[feature_cols].copy()
    y = df[TARGET_COL].to_numpy(dtype=int)

    cv = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=seed)
    splits = list(cv.split(X_df, y))
    train_idx, test_idx = splits[fold - 1]

    X_train_df = X_df.iloc[train_idx].copy()
    X_test_df = X_df.iloc[test_idx].copy()
    y_train = y[train_idx]
    y_test = y[test_idx]

    processed = preprocess_for_model(
        model_name=args.model,
        X_train_df=X_train_df,
        X_test_df=X_test_df,
        y_train=y_train,
        feature_names=feature_cols,
        mi_top_k=args.mi_top_k,
        seed=seed,
        fold=fold,
        oblique_scale_mode=args.oblique_scale_mode,
    )

    model = build_model(args.model, seed=seed, n_jobs=args.n_jobs, args=args)
    y_pred, proba_raw, fit_time, pred_time = fit_and_predict(
        model, processed.X_train, y_train, processed.X_test, keep_stdout=args.keep_model_stdout
    )
    proba = align_proba_columns(proba_raw, model, CLASS_ORDER, np.unique(y_train))
    metrics = compute_fold_metrics(y_test, y_pred, proba)

    train_counts = pd.Series(y_train).value_counts().reindex(CLASS_ORDER, fill_value=0)
    test_counts = pd.Series(y_test).value_counts().reindex(CLASS_ORDER, fill_value=0)

    fold_df = pd.DataFrame([
        {
            "dataset": DATASET_NAME,
            "seed": seed,
            "fold": fold,
            "model_name": args.model,
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
    ])

    user_ids = df.iloc[test_idx][ID_COL].tolist()
    sample_indices = test_idx.tolist()
    oof_rows = []
    for row_pos, sample_index in enumerate(sample_indices):
        oof_rows.append(
            {
                "dataset": DATASET_NAME,
                "sample_index": int(sample_index),
                "USER_ID": int(user_ids[row_pos]),
                "seed": seed,
                "fold": fold,
                "model_name": args.model,
                "y_true": int(y_test[row_pos]),
                "y_pred": int(y_pred[row_pos]),
                "proba_0": float(proba[row_pos, 0]),
                "proba_1": float(proba[row_pos, 1]),
                "proba_2": float(proba[row_pos, 2]),
            }
        )
    oof_df = pd.DataFrame(oof_rows)

    split_df = pd.DataFrame(build_split_manifest_rows(df, train_idx, test_idx, seed, fold))
    feature_space_df = pd.DataFrame([processed.feature_space_row])

    mi_columns = ["seed", "fold", "model_name", "rank", "feature_name", "mi_score"]
    mi_df = pd.DataFrame(processed.mi_rows, columns=mi_columns)

    fold_df.to_csv(shard_dir / "fold_results.csv", index=False)
    oof_df.to_csv(shard_dir / "oof_predictions.csv", index=False)
    split_df.to_csv(shard_dir / "split_manifest.csv", index=False)
    feature_space_df.to_csv(shard_dir / "feature_space_manifest.csv", index=False)
    mi_df.to_csv(shard_dir / "mi_selected_features.csv", index=False)

    shard_metadata = {
        "dataset": DATASET_NAME,
        "run_name": args.run_name,
        "model": args.model,
        "seed": seed,
        "fold": fold,
        "task_id": args.task_id,
        "n_splits": args.n_splits,
        "seeds_all": seeds,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "class_counts_train": train_counts.to_dict(),
        "class_counts_test": test_counts.to_dict(),
        "fit_time_sec": float(fit_time),
        "predict_time_sec": float(pred_time),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "success": True,
        "hyperparameters": {
            "mi_top_k": args.mi_top_k,
            "oblique_n_estimators": args.oblique_n_estimators,
            "oblique_max_depth": args.oblique_max_depth,
            "oblique_min_samples_split": args.oblique_min_samples_split,
            "oblique_min_samples_leaf": args.oblique_min_samples_leaf,
            "oblique_l1_strength": args.oblique_l1_strength,
            "oblique_n_tries": args.oblique_n_tries,
            "oblique_bootstrap_temperature": args.oblique_bootstrap_temperature,
            "oblique_scale_mode": args.oblique_scale_mode,
            "n_jobs": args.n_jobs,
        },
    }
    with open(shard_dir / "shard_metadata.json", "w", encoding="utf-8") as f:
        json.dump(shard_metadata, f, indent=2)

    print(f"Shard completed: {shard_dir}")
    print(f"Model={args.model} Seed={seed} Fold={fold}")
    print(f"fold_results rows: {len(fold_df)}")
    print(f"oof_predictions rows: {len(oof_df)}")
    print(f"split_manifest rows: {len(split_df)}")
    print(f"mi_selected_features rows: {len(mi_df)}")


if __name__ == "__main__":
    main()
