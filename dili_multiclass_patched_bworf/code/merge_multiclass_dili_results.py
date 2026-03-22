#!/usr/bin/env python3
"""
Merge baseline and oblique multiclass DILI benchmark outputs into one combined run folder.

Usage example:
python code/merge_multiclass_dili_results.py \
  --baseline_dir outputs/multiclass_dili_baselines_20260312_092014 \
  --oblique_dir outputs/multiclass_dili_oblique_20260312_092910 \
  --output_root outputs \
  --run_name multiclass_dili_merged
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError
from sklearn.metrics import confusion_matrix

CLASS_ORDER = np.array([0, 1, 2], dtype=int)
EXPECTED_ALL_MODELS = {
    "rf",
    "lr",
    "svm_rbf",
    "xgb",
    "naive_bayes",
    "orf_style",
    "bworf_no_mi",
    "bworf_mi",
}
REQUIRED_FILES = [
    "fold_results.csv",
    "oof_predictions.csv",
    "summary_metrics.csv",
    "split_manifest.csv",
    "feature_space_manifest.csv",
    "confusion_matrices.json",
    "sanity_check.json",
    "run_config.json",
]


def parse_args():
    p = argparse.ArgumentParser(description="Merge multiclass DILI baseline and oblique benchmark outputs.")
    p.add_argument("--baseline_dir", required=True)
    p.add_argument("--oblique_dir", required=True)
    p.add_argument("--output_root", required=True)
    p.add_argument("--run_name", default="multiclass_dili_merged")
    return p.parse_args()


def ensure_input_dir(path_str: str) -> Path:
    p = Path(path_str).resolve()
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(f"Missing input directory: {p}")
    missing = [name for name in REQUIRED_FILES if not (p / name).exists()]
    if not (p / "mi_selected_features.csv").exists():
        missing.append("mi_selected_features.csv")
    if missing:
        raise FileNotFoundError(f"Missing required files in {p}: {missing}")
    return p


def read_csv_allow_empty(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


def make_output_dir(output_root: str, run_name: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(output_root).resolve() / f"{run_name}_{ts}"
    out.mkdir(parents=True, exist_ok=False)
    return out


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


def build_confusion_payload(oof_df: pd.DataFrame) -> dict:
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


def build_sanity_payload(fold_df: pd.DataFrame, oof_df: pd.DataFrame, split_df: pd.DataFrame) -> dict:
    models = sorted(fold_df["model_name"].unique().tolist())
    seeds = sorted(fold_df["seed"].unique().tolist())
    n_folds = int(fold_df["fold"].nunique())
    n_samples = int(oof_df["sample_index"].nunique())

    expected_fold_rows = len(models) * len(seeds) * n_folds
    expected_oof_rows = n_samples * len(models) * len(seeds)

    duplicate_oof_keys = int(oof_df.duplicated(subset=["sample_index", "seed", "model_name"]).sum())
    proba_sums = oof_df[["proba_0", "proba_1", "proba_2"]].sum(axis=1).to_numpy(dtype=float)
    max_abs_error = float(np.max(np.abs(proba_sums - 1.0)))

    class_counts = (
        oof_df[["sample_index", "y_true"]]
        .drop_duplicates()
        .sort_values("sample_index")["y_true"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    return {
        "dataset": str(fold_df["dataset"].iloc[0]) if len(fold_df) else "dili_multiclass",
        "n_samples": n_samples,
        "n_features": 777,
        "n_models": len(models),
        "n_seeds": len(seeds),
        "n_folds": n_folds,
        "class_counts": class_counts,
        "expected_fold_rows": expected_fold_rows,
        "actual_fold_rows": int(len(fold_df)),
        "fold_rows_match_expected": bool(len(fold_df) == expected_fold_rows),
        "expected_oof_rows": expected_oof_rows,
        "actual_oof_rows": int(len(oof_df)),
        "oof_rows_match_expected": bool(len(oof_df) == expected_oof_rows),
        "duplicate_oof_key_count": duplicate_oof_keys,
        "oof_probability_sum_max_abs_error": max_abs_error,
        "split_manifest_rows": int(len(split_df)),
    }


def normalized_split_manifest(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["dataset", "sample_index", "USER_ID", "seed", "fold", "split_role", "y_true"]
    out = df[cols].copy()
    out = out.sort_values(cols).reset_index(drop=True)
    return out


def main():
    args = parse_args()
    baseline_dir = ensure_input_dir(args.baseline_dir)
    oblique_dir = ensure_input_dir(args.oblique_dir)
    out_dir = make_output_dir(args.output_root, args.run_name)

    fold_base = pd.read_csv(baseline_dir / "fold_results.csv")
    fold_oblq = pd.read_csv(oblique_dir / "fold_results.csv")
    oof_base = pd.read_csv(baseline_dir / "oof_predictions.csv")
    oof_oblq = pd.read_csv(oblique_dir / "oof_predictions.csv")
    split_base = pd.read_csv(baseline_dir / "split_manifest.csv")
    split_oblq = pd.read_csv(oblique_dir / "split_manifest.csv")
    feat_base = pd.read_csv(baseline_dir / "feature_space_manifest.csv")
    feat_oblq = pd.read_csv(oblique_dir / "feature_space_manifest.csv")
    mi_base = read_csv_allow_empty(baseline_dir / "mi_selected_features.csv")
    mi_oblq = read_csv_allow_empty(oblique_dir / "mi_selected_features.csv")

    base_models = set(fold_base["model_name"].unique().tolist())
    oblq_models = set(fold_oblq["model_name"].unique().tolist())
    overlap = base_models & oblq_models
    if overlap:
        raise ValueError(f"Model overlap between input runs: {sorted(overlap)}")
    combined_models = base_models | oblq_models
    missing_expected = EXPECTED_ALL_MODELS - combined_models
    extra_models = combined_models - EXPECTED_ALL_MODELS
    if missing_expected:
        raise ValueError(f"Merged runs are missing expected models: {sorted(missing_expected)}")
    if extra_models:
        raise ValueError(f"Merged runs contain unexpected models: {sorted(extra_models)}")

    fold_merged = pd.concat([fold_base, fold_oblq], ignore_index=True)
    oof_merged = pd.concat([oof_base, oof_oblq], ignore_index=True)
    feat_merged = pd.concat([feat_base, feat_oblq], ignore_index=True)
    mi_frames = [df for df in [mi_base, mi_oblq] if len(df) > 0]
    mi_merged = pd.concat(mi_frames, ignore_index=True) if mi_frames else pd.DataFrame()

    split_base_norm = normalized_split_manifest(split_base)
    split_oblq_norm = normalized_split_manifest(split_oblq)
    if not split_base_norm.equals(split_oblq_norm):
        union_split = pd.concat([split_base_norm, split_oblq_norm], ignore_index=True).drop_duplicates()
        if len(union_split) != len(split_base_norm):
            raise ValueError("Split manifests are not identical and cannot be safely reconciled")
        split_merged = union_split.sort_values(list(union_split.columns)).reset_index(drop=True)
    else:
        split_merged = split_base_norm

    summary_merged = aggregate_summary(fold_merged)
    confusion_payload = build_confusion_payload(oof_merged)
    sanity_payload = build_sanity_payload(fold_merged, oof_merged, split_merged)

    fold_merged.to_csv(out_dir / "fold_results.csv", index=False)
    summary_merged.to_csv(out_dir / "summary_metrics.csv", index=False)
    oof_merged.to_csv(out_dir / "oof_predictions.csv", index=False)
    split_merged.to_csv(out_dir / "split_manifest.csv", index=False)
    feat_merged.to_csv(out_dir / "feature_space_manifest.csv", index=False)
    if len(mi_merged) > 0:
        mi_merged.to_csv(out_dir / "mi_selected_features.csv", index=False)
    else:
        (out_dir / "mi_selected_features.csv").write_text("", encoding="utf-8")

    with open(out_dir / "confusion_matrices.json", "w", encoding="utf-8") as f:
        json.dump(confusion_payload, f, indent=2)
    with open(out_dir / "sanity_check.json", "w", encoding="utf-8") as f:
        json.dump(sanity_payload, f, indent=2)

    merge_metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "baseline_dir": str(baseline_dir),
        "oblique_dir": str(oblique_dir),
        "output_dir": str(out_dir),
        "merged_models": sorted(combined_models),
        "notes": {
            "fold_results": "concatenated",
            "oof_predictions": "concatenated",
            "summary_metrics": "recomputed from merged fold_results",
            "confusion_matrices": "recomputed from merged oof_predictions",
            "sanity_check": "recomputed from merged fold_results/oof_predictions",
            "split_manifest": "verified against both runs; single canonical copy retained",
            "mi_selected_features": "carried from oblique run only; baseline MI file may be empty",
        },
    }
    with open(out_dir / "merge_metadata.json", "w", encoding="utf-8") as f:
        json.dump(merge_metadata, f, indent=2)

    print(f"Merged outputs written to: {out_dir}")
    print(f"fold_results rows: {len(fold_merged)}")
    print(f"oof_predictions rows: {len(oof_merged)}")
    print(f"summary_metrics rows: {len(summary_merged)}")
    print(f"split_manifest rows: {len(split_merged)}")
    print(f"feature_space_manifest rows: {len(feat_merged)}")
    print(f"mi_selected_features rows: {len(mi_merged)}")
    print(f"models: {sorted(combined_models)}")
    print("Sanity:")
    print(json.dumps(sanity_payload, indent=2))


if __name__ == "__main__":
    main()
