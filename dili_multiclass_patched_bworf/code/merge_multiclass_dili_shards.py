#!/usr/bin/env python3
"""
Merge shard outputs for one multiclass DILI config back into the usual final artifact format.

Expected input layout:
outputs/<run_name>_shards/
  seed_42_fold_01/
  seed_42_fold_02/
  ...
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


def parse_args():
    p = argparse.ArgumentParser(description="Merge multiclass DILI shard outputs.")
    p.add_argument("--shards_dir", required=True, help="Path to <run_name>_shards directory")
    p.add_argument("--output_root", required=True, help="Root directory for merged outputs")
    p.add_argument("--run_name", required=True, help="Merged run name prefix")
    p.add_argument("--expected_shards", type=int, default=50)
    return p.parse_args()


def read_csv_allow_empty(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


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


def build_sanity_payload(fold_df: pd.DataFrame, oof_df: pd.DataFrame, split_df: pd.DataFrame, n_features: int = 777) -> dict:
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
        "n_features": n_features,
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


def main():
    args = parse_args()
    shards_dir = Path(args.shards_dir).resolve()
    if not shards_dir.exists() or not shards_dir.is_dir():
        raise FileNotFoundError(f"Missing shards directory: {shards_dir}")

    shard_dirs = sorted([p for p in shards_dir.iterdir() if p.is_dir() and (p / "shard_metadata.json").exists()])
    if len(shard_dirs) != args.expected_shards:
        print(f"WARNING: expected {args.expected_shards} shards, found {len(shard_dirs)}")

    fold_parts = []
    oof_parts = []
    split_parts = []
    feat_parts = []
    mi_parts = []
    shard_meta = []

    for sd in shard_dirs:
        fold_parts.append(pd.read_csv(sd / "fold_results.csv"))
        oof_parts.append(pd.read_csv(sd / "oof_predictions.csv"))
        split_parts.append(pd.read_csv(sd / "split_manifest.csv"))
        feat_parts.append(pd.read_csv(sd / "feature_space_manifest.csv"))
        mi_df = read_csv_allow_empty(sd / "mi_selected_features.csv")
        if len(mi_df) > 0:
            mi_parts.append(mi_df)
        with open(sd / "shard_metadata.json", "r", encoding="utf-8") as f:
            shard_meta.append(json.load(f))

    if not fold_parts:
        raise RuntimeError("No shard outputs found to merge")

    fold_df = pd.concat(fold_parts, ignore_index=True).sort_values(["model_name", "seed", "fold"]).reset_index(drop=True)
    oof_df = pd.concat(oof_parts, ignore_index=True).sort_values(["model_name", "seed", "sample_index"]).reset_index(drop=True)
    split_df = pd.concat(split_parts, ignore_index=True).sort_values(["seed", "fold", "split_role", "sample_index"]).reset_index(drop=True)
    feat_df = pd.concat(feat_parts, ignore_index=True).sort_values(["model_name", "seed", "fold"]).reset_index(drop=True)
    mi_columns = ["seed", "fold", "model_name", "rank", "feature_name", "mi_score"]
    mi_df = pd.concat(mi_parts, ignore_index=True) if mi_parts else pd.DataFrame(columns=mi_columns)

    summary_df = aggregate_summary(fold_df)
    confusion_payload = build_confusion_payload(oof_df)
    sanity_payload = build_sanity_payload(fold_df, oof_df, split_df)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_root).resolve() / f"{args.run_name}_merged_{ts}"
    out_dir.mkdir(parents=True, exist_ok=False)

    fold_df.to_csv(out_dir / "fold_results.csv", index=False)
    summary_df.to_csv(out_dir / "summary_metrics.csv", index=False)
    oof_df.to_csv(out_dir / "oof_predictions.csv", index=False)
    split_df.to_csv(out_dir / "split_manifest.csv", index=False)
    feat_df.to_csv(out_dir / "feature_space_manifest.csv", index=False)
    mi_df.to_csv(out_dir / "mi_selected_features.csv", index=False)

    with open(out_dir / "confusion_matrices.json", "w", encoding="utf-8") as f:
        json.dump(confusion_payload, f, indent=2)
    with open(out_dir / "sanity_check.json", "w", encoding="utf-8") as f:
        json.dump(sanity_payload, f, indent=2)
    with open(out_dir / "merge_metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "shards_dir": str(shards_dir),
                "output_dir": str(out_dir),
                "n_shards_found": len(shard_dirs),
                "expected_shards": args.expected_shards,
                "notes": {
                    "fold_results": "concatenated from shard fold_results.csv",
                    "oof_predictions": "concatenated from shard oof_predictions.csv",
                    "summary_metrics": "recomputed from merged fold_results",
                    "confusion_matrices": "recomputed from merged oof_predictions",
                    "sanity_check": "recomputed from merged fold_results/oof_predictions",
                },
            },
            f,
            indent=2,
        )

    # Carry a representative run config from the first shard metadata for convenience
    rep = shard_meta[0].copy()
    rep["merged_from_shards"] = True
    rep["n_shards_found"] = len(shard_dirs)
    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2)

    print(f"Merged shard outputs written to: {out_dir}")
    print(f"fold_results rows: {len(fold_df)}")
    print(f"oof_predictions rows: {len(oof_df)}")
    print(f"summary_metrics rows: {len(summary_df)}")
    print(f"split_manifest rows: {len(split_df)}")
    print(f"feature_space_manifest rows: {len(feat_df)}")
    print(f"mi_selected_features rows: {len(mi_df)}")
    print("Sanity:")
    print(json.dumps(sanity_payload, indent=2))


if __name__ == "__main__":
    main()
