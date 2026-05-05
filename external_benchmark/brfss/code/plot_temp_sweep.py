"""
plot_temp_sweep.py — produce a two-panel supplement figure for the
BWORF bootstrap-temperature sensitivity analysis.

Reads the aggregated CSV from aggregate_temp_sweep.py and writes a PNG
suitable for the BMC Bioinformatics supplement.

Panel A: Metric values vs. temperature (six lines: AUROC, balanced
         accuracy, MCC, F1_pos, precision_pos, recall_pos). AUROC line
         is annotated to highlight flatness.
Panel B: Precision vs. recall trajectory across temperatures, with
         each temperature labeled. Visualizes the operating-point shift.

Usage (login node, after the aggregator has produced
temp_sweep_summary.csv):

    cd ~/external_benchmark
    python plot_temp_sweep.py

Output:
    outputs_heart2022_subsample_temp_sweep/_aggregated/temp_sweep_figure.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless / no display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INPUT_CSV = Path(
    "outputs_heart2022_subsample_temp_sweep/_aggregated/temp_sweep_summary.csv"
)
OUTPUT_PNG = Path(
    "outputs_heart2022_subsample_temp_sweep/_aggregated/temp_sweep_figure.png"
)


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"{INPUT_CSV} not found. Run aggregate_temp_sweep.py first."
        )
    df = pd.read_csv(INPUT_CSV).sort_values("temperature").reset_index(drop=True)
    print(f"Loaded {len(df)} temperature rows from {INPUT_CSV}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)

    # --- Panel A: metrics vs temperature ---
    metric_cols = [
        ("auroc", "AUROC", "tab:blue", "-", "o"),
        ("balanced_accuracy", "Balanced accuracy", "tab:orange", "-", "s"),
        ("mcc", "MCC", "tab:green", "-", "^"),
        ("f1_pos", "F1 (positive)", "tab:red", "-", "D"),
        ("precision_pos", "Precision (positive)", "tab:purple", "--", "v"),
        ("recall_pos", "Recall (positive)", "tab:brown", "--", "P"),
    ]
    T = df["temperature"].values
    for col, label, color, style, marker in metric_cols:
        mean_col = f"{col}_mean"
        std_col = f"{col}_std"
        if mean_col not in df.columns:
            continue
        y = df[mean_col].values
        ax1.plot(T, y, label=label, color=color, linestyle=style,
                 marker=marker, markersize=7, linewidth=1.8)
        if std_col in df.columns:
            ax1.fill_between(T, y - df[std_col].values, y + df[std_col].values,
                             color=color, alpha=0.10)

    # Highlight AUROC's flatness with a band
    auroc_y = df["auroc_mean"].values
    auroc_min, auroc_max = auroc_y.min(), auroc_y.max()
    ax1.axhspan(auroc_min, auroc_max, color="tab:blue", alpha=0.06, zorder=0)
    ax1.text(0.5, (auroc_min + auroc_max) / 2,
             f"AUROC range:\n{auroc_min:.3f}-{auroc_max:.3f}\n(span {auroc_max-auroc_min:.3f})",
             color="tab:blue", fontsize=9, ha="left", va="center",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                       edgecolor="tab:blue", alpha=0.85))

    # Mark the locked T=1.0 used in the main benchmark
    ax1.axvline(1.0, color="black", linestyle=":", linewidth=1.2, alpha=0.6)
    ax1.text(1.05, 0.05, "locked T = 1.0\n(main benchmark)",
             fontsize=9, color="black", ha="left", va="bottom",
             rotation=0, transform=ax1.get_xaxis_transform())

    ax1.set_xlabel("BWORF bootstrap temperature  $T$", fontsize=11)
    ax1.set_ylabel("Metric value (10-fold CV \u00d7 5 seeds, mean)", fontsize=11)
    ax1.set_title("(A) BWORF metrics vs. bootstrap temperature",
                  fontsize=12, loc="left")
    ax1.set_xticks(T)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="center right", fontsize=9, frameon=True,
               framealpha=0.95)
    ax1.set_ylim(0.0, 1.0)

    # --- Panel B: precision-recall trajectory ---
    if "precision_pos_mean" in df.columns and "recall_pos_mean" in df.columns:
        prec = df["precision_pos_mean"].values
        rec = df["recall_pos_mean"].values
        # Color points by temperature
        sc = ax2.scatter(rec, prec, c=T, cmap="viridis", s=140,
                         edgecolors="black", linewidths=1.2, zorder=3)
        # Connect with arrows
        ax2.plot(rec, prec, color="gray", linewidth=1.2, linestyle="-",
                 alpha=0.5, zorder=2)
        # Label each point with its temperature
        for ti, ri, pi in zip(T, rec, prec):
            offset_x, offset_y = 0.01, 0.012
            ax2.annotate(f"T={ti:g}",
                         xy=(ri, pi),
                         xytext=(ri + offset_x, pi + offset_y),
                         fontsize=10, fontweight="bold")
        cbar = plt.colorbar(sc, ax=ax2, shrink=0.8)
        cbar.set_label("Temperature  $T$", fontsize=10)

    ax2.set_xlabel("Recall on positive class", fontsize=11)
    ax2.set_ylabel("Precision on positive class", fontsize=11)
    ax2.set_title("(B) Operating-point trajectory across temperature",
                  fontsize=12, loc="left")
    ax2.set_xlim(0, 1.05)
    ax2.set_ylim(0, 0.55)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        "BWORF sensitivity to bootstrap temperature on BRFSS 2022 (subsample, N=4,000)",
        fontsize=13, y=1.02,
    )

    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=180, bbox_inches="tight")
    print(f"Wrote {OUTPUT_PNG}")
    plt.close(fig)


if __name__ == "__main__":
    main()
