"""
aggregate_full_bworf.py — concatenate per-fold shards from the full-N
parallel BWORF run.

The per-fold sbatch writes outputs to:
    outputs_heart2022_full_bworf/seed_<S>_fold_<F>/<DATASET>/{fold_results.csv,
                                                                oof_predictions.csv.gz,
                                                                run_config.json,
                                                                summary.csv}

This script walks all 50 such directories (5 seeds * 10 folds) and
produces the canonical aggregated layout:
    outputs_heart2022_full_bworf/_aggregated/<DATASET>/{fold_results.csv,
                                                          oof_predictions.csv.gz,
                                                          summary.csv,
                                                          aggregation_manifest.json}

This mirrors what aggregate_heart2022.py produces, so downstream code
(the holdout scorer, README references) works without modification.

Usage:
    cd ~/external_benchmark
    python aggregate_full_bworf.py
"""

from __future__ import annotations

import gzip
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path("outputs_heart2022_full_bworf")
DATASET = "heart_attack_2022_brfss_full"
AGG_DIR = OUT_DIR / "_aggregated" / DATASET

BINARY_METRICS = [
    "accuracy", "balanced_accuracy", "f1_pos", "precision_pos", "recall_pos",
    "mcc", "auroc", "auprc", "log_loss", "brier",
]


def main() -> None:
    if not OUT_DIR.exists():
        raise FileNotFoundError(f"{OUT_DIR} does not exist")

    AGG_DIR.mkdir(parents=True, exist_ok=True)

    # --- Discover shards: outputs_heart2022_full_bworf/seed_*_fold_*/<DATASET>/fold_results.csv ---
    shards = sorted(OUT_DIR.glob(f"seed_*_fold_*/{DATASET}/fold_results.csv"))
    if not shards:
        raise FileNotFoundError(f"No fold_results.csv found under {OUT_DIR}/seed_*_fold_*/")

    print(f"Found {len(shards)} shard(s):")
    for p in shards:
        print(f"  {p.relative_to(OUT_DIR)}")

    # --- Concatenate fold_results ---
    fold_dfs = [pd.read_csv(p) for p in shards]
    fold_all = pd.concat(fold_dfs, ignore_index=True)
    fold_path = AGG_DIR / "fold_results.csv"
    fold_all.to_csv(fold_path, index=False)
    print(f"\nWrote {fold_path}: {len(fold_all)} rows")

    # --- Concatenate OOF predictions (streamed via gzip) ---
    oof_files = sorted(OUT_DIR.glob(f"seed_*_fold_*/{DATASET}/oof_predictions.csv.gz"))
    oof_path = AGG_DIR / "oof_predictions.csv.gz"
    if oof_path.exists():
        oof_path.unlink()
    total_oof_rows = 0
    header_written = False
    for p in oof_files:
        df = pd.read_csv(p)
        with gzip.open(oof_path, "at", encoding="utf-8") as fh:
            df.to_csv(fh, index=False, header=(not header_written),
                      lineterminator="\n")
        header_written = True
        total_oof_rows += len(df)
    print(f"Wrote {oof_path}: {total_oof_rows} rows")

    # --- Recompute summary.csv across all 50 fold rows ---
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
    summary_path = AGG_DIR / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}")

    # --- Aggregation manifest ---
    manifest = {
        "dataset_name": DATASET,
        "out_dir": str(OUT_DIR),
        "n_shards": len(shards),
        "shard_paths": [str(p.relative_to(OUT_DIR)) for p in shards],
        "models_found": models,
        "seeds_found": sorted(int(s) for s in fold_all["seed"].unique().tolist()),
        "folds_found": sorted(int(f) for f in fold_all["fold_id"].unique().tolist()),
        "total_fold_results_rows": int(len(fold_all)),
        "total_oof_rows": int(total_oof_rows),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = AGG_DIR / "aggregation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {manifest_path}")

    # --- Console summary ---
    print("\n=== Per-model summary ===")
    cols_show = ["model", "n_folds_completed", "n_seeds",
                 "auroc_mean", "auroc_std",
                 "balanced_accuracy_mean", "balanced_accuracy_std",
                 "mcc_mean", "f1_pos_mean", "auprc_mean"]
    cols_show = [c for c in cols_show if c in summary_df.columns]
    print(summary_df[cols_show].to_string(index=False))

    if len(shards) < 50:
        missing = 50 - len(shards)
        print(f"\nNote: {missing} shard(s) missing (expected 50: 5 seeds * 10 folds).")
        print("Either some tasks haven't finished yet, or some failed.")
        print("Re-run this aggregator after all tasks complete.")
    else:
        print(f"\nAll 50 shards present. Aggregation complete.")


if __name__ == "__main__":
    main()
