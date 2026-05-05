"""
aggregate_heart2022.py — concatenate per-task outputs from a sharded run.

Each (model, seed) shard wrote its own
    <out_dir>/<model>/seed_<S>/<dataset_name>/{fold_results.csv,
                                                 oof_predictions.csv.gz,
                                                 summary.csv,
                                                 run_config.json}

This script walks the tree, concatenates fold_results across all shards,
concatenates OOF predictions, recomputes summary stats from pooled fold
results (mean/std across folds, per model), and writes the aggregated
output to <out_dir>/_aggregated/<dataset_name>/, mirroring the layout
your existing externals use (so it slots into Table 4.7 as a peer row).

Usage:

    cd ~/external_benchmark
    # After subsample fast models finish:
    python aggregate_heart2022.py \\
        --out_dir outputs_heart2022_subsample \\
        --dataset heart_attack_2022_brfss

    # Once BWORF subsample finishes, re-run -- it picks up everything found:
    python aggregate_heart2022.py \\
        --out_dir outputs_heart2022_subsample \\
        --dataset heart_attack_2022_brfss

    # After full-N fast models finish:
    python aggregate_heart2022.py \\
        --out_dir outputs_heart2022_full \\
        --dataset heart_attack_2022_brfss_full

Output:
    <out_dir>/_aggregated/<dataset_name>/fold_results.csv         # all shards concatenated
    <out_dir>/_aggregated/<dataset_name>/oof_predictions.csv.gz   # all shards concatenated
    <out_dir>/_aggregated/<dataset_name>/summary.csv              # recomputed mean/std per model
    <out_dir>/_aggregated/<dataset_name>/aggregation_manifest.json
"""

from __future__ import annotations

import argparse
import gzip
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

BINARY_METRICS = [
    "accuracy", "balanced_accuracy", "f1_pos", "precision_pos", "recall_pos",
    "mcc", "auroc", "auprc", "log_loss", "brier",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate sharded heart_2022 outputs")
    p.add_argument("--out_dir", required=True,
                   help="Top-level output dir, e.g. outputs_heart2022_subsample")
    p.add_argument("--dataset", required=True,
                   help="Dataset name as used in REGISTRY, e.g. heart_attack_2022_brfss")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    if not out_dir.exists():
        raise FileNotFoundError(f"out_dir does not exist: {out_dir}")

    agg_dir = out_dir / "_aggregated" / args.dataset
    agg_dir.mkdir(parents=True, exist_ok=True)

    # Discover per-shard fold_results files
    fold_files = sorted(out_dir.glob(f"*/seed_*/{args.dataset}/fold_results.csv"))
    if not fold_files:
        raise FileNotFoundError(
            f"No fold_results.csv found under {out_dir}/*/seed_*/{args.dataset}/"
        )

    print(f"Found {len(fold_files)} shard(s):")
    for p in fold_files:
        # path is .../<model>/seed_<S>/<dataset>/fold_results.csv
        rel = p.relative_to(out_dir)
        print(f"  {rel}")

    # ---- Concatenate fold_results ----
    fold_dfs = [pd.read_csv(p) for p in fold_files]
    fold_all = pd.concat(fold_dfs, ignore_index=True)
    fold_all_path = agg_dir / "fold_results.csv"
    fold_all.to_csv(fold_all_path, index=False)
    print(f"\nWrote {fold_all_path}: {len(fold_all)} rows")

    # ---- Concatenate OOF predictions ----
    oof_files = sorted(out_dir.glob(f"*/seed_*/{args.dataset}/oof_predictions.csv.gz"))
    oof_all_path = agg_dir / "oof_predictions.csv.gz"
    total_oof_rows = 0
    header_written = False
    if oof_all_path.exists():
        oof_all_path.unlink()
    for p in oof_files:
        df = pd.read_csv(p)
        with gzip.open(oof_all_path, "at", encoding="utf-8") as fh:
            df.to_csv(fh, index=False, header=(not header_written),
                      lineterminator="\n")
        header_written = True
        total_oof_rows += len(df)
    print(f"Wrote {oof_all_path}: {total_oof_rows} rows")

    # ---- Recompute summary.csv (mean/std across folds per model) ----
    summary_rows = []
    models = sorted(fold_all["model"].unique())
    for mn in models:
        sub = fold_all[fold_all["model"] == mn]
        n_folds_completed = len(sub.dropna(subset=["accuracy"]))
        row = {
            "model": mn,
            "n_fold_results_rows": int(len(sub)),
            "n_folds_completed": int(n_folds_completed),
            "n_seeds": int(sub["seed"].nunique()),
        }
        for mk in BINARY_METRICS:
            if mk in sub.columns:
                row[f"{mk}_mean"] = float(sub[mk].mean())
                row[f"{mk}_std"] = float(sub[mk].std())
        if "fit_seconds" in sub.columns:
            row["fit_seconds_mean"] = float(sub["fit_seconds"].mean())
            row["fit_seconds_std"] = float(sub["fit_seconds"].std())
        if "pred_seconds" in sub.columns:
            row["pred_seconds_mean"] = float(sub["pred_seconds"].mean())
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    summary_path = agg_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}")

    # ---- Aggregation manifest ----
    manifest = {
        "dataset_name": args.dataset,
        "out_dir": str(out_dir),
        "n_shards": len(fold_files),
        "shard_paths": [str(p.relative_to(out_dir)) for p in fold_files],
        "models_found": models,
        "seeds_found": sorted(int(s) for s in fold_all["seed"].unique().tolist()),
        "total_fold_results_rows": int(len(fold_all)),
        "total_oof_rows": int(total_oof_rows),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = agg_dir / "aggregation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {manifest_path}")

    # ---- Console table ----
    print("\n=== Per-model summary ===")
    cols_show = ["model", "n_folds_completed", "n_seeds",
                 "auroc_mean", "auroc_std",
                 "balanced_accuracy_mean", "balanced_accuracy_std",
                 "mcc_mean", "f1_pos_mean", "auprc_mean"]
    cols_show = [c for c in cols_show if c in summary_df.columns]
    print(summary_df[cols_show].to_string(index=False))

    print("\nAggregation complete.")


if __name__ == "__main__":
    main()
