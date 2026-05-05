"""
aggregate_temp_sweep.py — aggregate the BRFSS BWORF temperature-sweep
sensitivity analysis.

Walks outputs_heart2022_subsample_temp_sweep/T<T>/seed_<S>/heart_attack_2022_brfss/
and produces a summary CSV with one row per temperature value, summarizing
mean +/- SD across the 50 fold-results (10 folds x 5 seeds) for each metric.

Also concatenates the existing T=1.0 run (which lives under
outputs_heart2022_subsample/bworf/seed_<S>/heart_attack_2022_brfss/) so the
final summary covers all 5 temperatures.

Usage:

    cd ~/external_benchmark
    python aggregate_temp_sweep.py

Output:
    outputs_heart2022_subsample_temp_sweep/_aggregated/
        temp_sweep_summary.csv
        fold_results_all.csv  (250 rows: 5 temps x 5 seeds x 10 folds)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

DATASET = "heart_attack_2022_brfss"
SWEEP_DIR = Path("outputs_heart2022_subsample_temp_sweep")
T1_DIR = Path("outputs_heart2022_subsample")  # existing main run with T=1.0

OUTPUT_DIR = SWEEP_DIR / "_aggregated"
METRICS = [
    "auroc", "balanced_accuracy", "f1_pos", "precision_pos", "recall_pos",
    "mcc", "auprc", "log_loss", "brier", "accuracy",
]


def main() -> None:
    if not SWEEP_DIR.exists():
        raise FileNotFoundError(f"{SWEEP_DIR} does not exist; run the sweep first")

    # --- Discover sweep shards (T != 1.0) ---
    fold_dfs = []
    for shard in sorted(SWEEP_DIR.glob(f"T*/seed_*/{DATASET}/fold_results.csv")):
        # path: outputs_heart2022_subsample_temp_sweep/T0.5/seed_13/<DATASET>/fold_results.csv
        temp_str = shard.parts[-4][1:]  # strip leading "T"
        temp = float(temp_str)
        df = pd.read_csv(shard)
        df["temperature"] = temp
        fold_dfs.append(df)

    if not fold_dfs:
        raise FileNotFoundError(f"No fold_results found under {SWEEP_DIR}/T*/seed_*/")

    # --- Pull in existing T=1.0 from main subsample run ---
    t1_dfs = []
    for shard in sorted(T1_DIR.glob(f"bworf/seed_*/{DATASET}/fold_results.csv")):
        df = pd.read_csv(shard)
        df["temperature"] = 1.0
        t1_dfs.append(df)

    if t1_dfs:
        fold_dfs.extend(t1_dfs)
        print(f"Including {len(t1_dfs)} shard(s) from T=1.0 main run")
    else:
        print(f"WARNING: no T=1.0 shards found in {T1_DIR}/bworf/")

    fold_all = pd.concat(fold_dfs, ignore_index=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fold_path = OUTPUT_DIR / "fold_results_all.csv"
    fold_all.to_csv(fold_path, index=False)
    print(f"Wrote {fold_path}: {len(fold_all)} rows across "
          f"{fold_all['temperature'].nunique()} temperature(s)")

    # --- Per-temperature summary (mean +/- std across fold rows) ---
    summary_rows = []
    for temp in sorted(fold_all["temperature"].unique()):
        sub = fold_all[fold_all["temperature"] == temp]
        n_folds_completed = len(sub.dropna(subset=["accuracy"]))
        row = {
            "temperature": temp,
            "n_fold_rows": int(len(sub)),
            "n_folds_completed": int(n_folds_completed),
            "n_seeds": int(sub["seed"].nunique()),
        }
        for mk in METRICS:
            if mk in sub.columns:
                row[f"{mk}_mean"] = float(sub[mk].mean())
                row[f"{mk}_std"] = float(sub[mk].std())
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).sort_values("temperature").reset_index(drop=True)
    summary_path = OUTPUT_DIR / "temp_sweep_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}")

    # --- Manifest ---
    manifest = {
        "experiment": "BWORF bootstrap_temperature sensitivity sweep on BRFSS subsample",
        "dataset": DATASET,
        "temperatures": sorted(fold_all["temperature"].unique().tolist()),
        "seeds": sorted(int(s) for s in fold_all["seed"].unique()),
        "n_splits_per_seed": int(fold_all.groupby(["temperature", "seed"]).size().mean()),
        "total_fold_rows": int(len(fold_all)),
        "main_run_T1_dir": str(T1_DIR),
        "sweep_dir": str(SWEEP_DIR),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    (OUTPUT_DIR / "temp_sweep_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    # --- Console table ---
    cols_show = ["temperature", "n_folds_completed", "n_seeds",
                 "auroc_mean", "auroc_std",
                 "balanced_accuracy_mean", "balanced_accuracy_std",
                 "mcc_mean",
                 "f1_pos_mean", "f1_pos_std",
                 "precision_pos_mean", "recall_pos_mean"]
    cols_show = [c for c in cols_show if c in summary_df.columns]
    print("\n=== Temperature sweep summary ===")
    print(summary_df[cols_show].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
