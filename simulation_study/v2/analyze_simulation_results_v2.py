"""
Analyze Simulation Results v2

Purpose
-------
Read v2 runner outputs (simulation_results_v2.csv) and generate publication-ready
figures + a compact summary table comparing BWORF vs RF.

Outputs
-------
- fig_auc_vs_h2_v2.png         : Line plot of AUC vs h2 (2x2 facets: n_samples x signal_type)
- fig_brier_vs_h2_v2.png       : Line plot of Brier vs h2 (2x2 facets: n_samples x signal_type)
- fig_mean_rank_boxplot_v2.png : Boxplot of mean_rank_causal (rows=n_samples, cols=rho)
- fig_delta_auc_heatmap_v2.png : Heatmap of Delta AUC (BWORF - RF), Interaction only
- simulation_summary_table_v2.csv : Summary statistics by scenario

Usage
-----
python analyze_simulation_results_v2.py --csv <path> --outdir <path> --heatmap_h2 0.15
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ==============================================================================
# Constants
# ==============================================================================

REQUIRED_COLUMNS = [
    "method",
    "n_samples",
    "n_snps",
    "rho",
    "n_causal_pairs",
    "h2",
    "signal_ratio",
    "auc_roc",
    "brier",
    "mean_rank_causal",
    "recall_at_20_causal",
    "runtime_seconds",
]

SCENARIO_COLS = ["n_samples", "n_snps", "rho", "n_causal_pairs", "h2", "signal_ratio"]

METRIC_COLS = ["auc_roc", "brier", "mean_rank_causal", "recall_at_20_causal", "runtime_seconds"]

# Consistent color mapping across all figures
METHOD_COLORS = {"RF": "#1f77b4", "BWORF": "#d62728"}
METHOD_MARKERS = {"RF": "o", "BWORF": "s"}

# ==============================================================================
# Helpers
# ==============================================================================


def normalize_method(m: str) -> str:
    """Normalize method name to uppercase."""
    return str(m).strip().upper()


def add_signal_type_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived signal_type column based on signal_ratio."""
    df = df.copy()
    df["signal_type"] = df["signal_ratio"].apply(
        lambda x: "Pure Linear" if np.isinf(x) else "Interaction"
    )
    return df


def load_and_validate_csv(csv_path: Path) -> pd.DataFrame:
    """Load CSV and validate required columns exist."""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")

    # Check required columns
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Normalize method names
    df["method"] = df["method"].apply(normalize_method)

    # Check methods include RF and BWORF
    methods = set(df["method"].unique())
    if "RF" not in methods:
        raise ValueError("Dataset must contain method='RF'")
    if "BWORF" not in methods:
        raise ValueError("Dataset must contain method='BWORF'")

    # Keep only RF and BWORF
    df = df[df["method"].isin(["RF", "BWORF"])].copy()
    print(f"Filtered to RF/BWORF: {len(df)} rows")

    # Add derived signal_type column
    df = add_signal_type_column(df)
    print(f"Signal types: {sorted(df['signal_type'].unique())}")

    return df


def aggregate_by_scenario(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate metrics by (method, scenario_cols, signal_type) computing mean and std.
    """
    group_cols = ["method", "signal_type"] + SCENARIO_COLS

    agg_dict = {}
    for col in METRIC_COLS:
        if col in df.columns:
            agg_dict[col] = ["mean", "std", "count"]

    agg_df = df.groupby(group_cols, dropna=False).agg(agg_dict).reset_index()

    # Flatten column names
    new_cols = []
    for col in agg_df.columns:
        if isinstance(col, tuple):
            if col[1] == "":
                new_cols.append(col[0])
            else:
                new_cols.append(f"{col[0]}_{col[1]}")
        else:
            new_cols.append(col)
    agg_df.columns = new_cols

    # Check for NaNs
    nan_counts = agg_df[[c for c in agg_df.columns if "_mean" in c]].isna().sum()
    if nan_counts.sum() > 0:
        warnings.warn(f"NaN counts after aggregation:\n{nan_counts[nan_counts > 0]}")

    return agg_df


def compute_delta_auc(df: pd.DataFrame, group_cols: list) -> pd.DataFrame:
    """
    Compute Delta AUC = mean(AUC_BWORF) - mean(AUC_RF) for specified grouping.
    """
    # First aggregate by method and group_cols
    agg = df.groupby(["method"] + group_cols)["auc_roc"].mean().reset_index()

    # Pivot to get BWORF and RF side by side
    pivot_df = agg.pivot_table(
        index=group_cols,
        columns="method",
        values="auc_roc",
        aggfunc="first"
    ).reset_index()

    if "BWORF" not in pivot_df.columns or "RF" not in pivot_df.columns:
        raise ValueError("Cannot compute delta AUC: missing BWORF or RF in pivoted data")

    pivot_df["delta_auc"] = pivot_df["BWORF"] - pivot_df["RF"]
    return pivot_df


def style_axis(ax):
    """Apply consistent styling: remove top/right spines, light horizontal gridlines."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", alpha=0.3, linestyle="-", linewidth=0.5)
    ax.grid(False, axis="x")


# ==============================================================================
# Plotting Functions
# ==============================================================================


def plot_metric_vs_h2_gemini(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    save_path: Path,
):
    """
    Create 2x2 faceted line plot: metric vs h2.
    Rows: n_samples (500, 2000)
    Cols: signal_type (Pure Linear, Interaction)
    Lines: method (RF, BWORF) with error bars (SD)
    """
    # Get unique values
    n_samples_vals = sorted(df["n_samples"].dropna().unique())
    signal_types = ["Pure Linear", "Interaction"]
    methods = ["RF", "BWORF"]

    # Filter to available signal types
    signal_types = [st for st in signal_types if st in df["signal_type"].unique()]
    n_samples_vals = [ns for ns in n_samples_vals if ns in [500, 2000]]
    if len(n_samples_vals) == 0:
        n_samples_vals = sorted(df["n_samples"].dropna().unique())[:2]

    n_rows = len(n_samples_vals)
    n_cols = len(signal_types)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 8), squeeze=False, sharey=True)

    # Aggregate data for plotting
    group_cols = ["method", "signal_type", "n_samples", "h2"]
    agg = df.groupby(group_cols)[metric].agg(["mean", "std", "count"]).reset_index()

    # Track y-limits for consistent scaling
    y_min, y_max = np.inf, -np.inf

    for i, n_samp in enumerate(n_samples_vals):
        for j, sig_type in enumerate(signal_types):
            ax = axes[i, j]

            subset = agg[(agg["n_samples"] == n_samp) & (agg["signal_type"] == sig_type)]

            for method in methods:
                data = subset[subset["method"] == method].sort_values("h2")
                if len(data) == 0:
                    continue

                h2_vals = data["h2"].values
                y_vals = data["mean"].values
                y_std = data["std"].values

                color = METHOD_COLORS[method]
                marker = METHOD_MARKERS[method]

                ax.errorbar(
                    h2_vals, y_vals, yerr=y_std,
                    color=color, marker=marker, markersize=7,
                    linewidth=1.5, capsize=3, capthick=1.2,
                    label=method if (i == 0 and j == 0) else None
                )

                # Update y-limits
                valid_mask = ~np.isnan(y_vals) & ~np.isnan(y_std)
                if np.any(valid_mask):
                    y_min = min(y_min, np.min(y_vals[valid_mask] - y_std[valid_mask]))
                    y_max = max(y_max, np.max(y_vals[valid_mask] + y_std[valid_mask]))

            # Panel title
            ax.set_title(f"N={int(n_samp)}, {sig_type}", fontsize=12)
            ax.set_xlabel("Heritability (h²)", fontsize=12)
            if j == 0:
                ax.set_ylabel(ylabel, fontsize=12)

            style_axis(ax)
            ax.tick_params(labelsize=10)

    # Set consistent y-limits with padding
    if np.isfinite(y_min) and np.isfinite(y_max):
        y_pad = (y_max - y_min) * 0.1
        for ax_row in axes:
            for ax in ax_row:
                ax.set_ylim(y_min - y_pad, y_max + y_pad)

    # Create shared legend outside panels on the right
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", bbox_to_anchor=(0.98, 0.5),
               fontsize=11, frameon=True, edgecolor="gray")

    fig.suptitle(title, fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 0.88, 0.95])

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_mean_rank_boxplot_gemini(
    df: pd.DataFrame,
    save_path: Path,
):
    """
    Boxplot of mean_rank_causal using raw replicate-level data.
    Rows: n_samples (500, 2000)
    Cols: rho (0.0, 0.8)
    X-axis: signal_type (Pure Linear, Interaction)
    Hue: method (RF, BWORF)
    """
    n_samples_vals = sorted([ns for ns in df["n_samples"].unique() if ns in [500, 2000]])
    if len(n_samples_vals) == 0:
        n_samples_vals = sorted(df["n_samples"].dropna().unique())[:2]

    rho_vals = sorted([r for r in df["rho"].unique() if r in [0.0, 0.8]])
    if len(rho_vals) == 0:
        rho_vals = sorted(df["rho"].dropna().unique())[:2]

    signal_types = ["Pure Linear", "Interaction"]
    signal_types = [st for st in signal_types if st in df["signal_type"].unique()]
    methods = ["RF", "BWORF"]

    rho_labels = {0.0: "ρ=0.0 (No LD)", 0.8: "ρ=0.8 (High LD)"}

    n_rows = len(n_samples_vals)
    n_cols = len(rho_vals)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 8), squeeze=False, sharey=True)

    # Track y-limits for consistent scaling
    y_min, y_max = np.inf, -np.inf

    for i, n_samp in enumerate(n_samples_vals):
        for j, rho in enumerate(rho_vals):
            ax = axes[i, j]

            subset = df[(df["n_samples"] == n_samp) & (df["rho"] == rho)]

            # Prepare data for grouped boxplot
            positions = []
            box_data = []
            box_colors = []
            tick_positions = []
            tick_labels = []

            for k, sig_type in enumerate(signal_types):
                base_pos = k * 3  # spacing between signal_type groups
                tick_positions.append(base_pos + 0.5)
                tick_labels.append(sig_type)

                for m_idx, method in enumerate(methods):
                    vals = subset[
                        (subset["signal_type"] == sig_type) & (subset["method"] == method)
                    ]["mean_rank_causal"].dropna().values

                    if len(vals) > 0:
                        box_data.append(vals)
                        positions.append(base_pos + m_idx * 0.7)
                        box_colors.append(METHOD_COLORS[method])
                        y_min = min(y_min, np.min(vals))
                        y_max = max(y_max, np.max(vals))
                    else:
                        box_data.append([np.nan])
                        positions.append(base_pos + m_idx * 0.7)
                        box_colors.append(METHOD_COLORS[method])

            if len(box_data) > 0:
                bp = ax.boxplot(
                    box_data, positions=positions, widths=0.5,
                    patch_artist=True, manage_ticks=False
                )
                for patch, color in zip(bp["boxes"], box_colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                for median in bp["medians"]:
                    median.set_color("black")
                    median.set_linewidth(1.5)

            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, fontsize=10)

            rho_label = rho_labels.get(rho, f"ρ={rho}")
            ax.set_title(f"N={int(n_samp)}, {rho_label}", fontsize=12)
            ax.set_xlabel("Signal Type", fontsize=12)
            if j == 0:
                ax.set_ylabel("Mean Causal SNP Rank", fontsize=12)

            style_axis(ax)
            ax.tick_params(labelsize=10)

    # Set consistent y-limits
    if np.isfinite(y_min) and np.isfinite(y_max):
        y_pad = (y_max - y_min) * 0.1
        for ax_row in axes:
            for ax in ax_row:
                ax.set_ylim(max(0, y_min - y_pad), y_max + y_pad)

    # Create legend manually
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=METHOD_COLORS["RF"], alpha=0.7, label="RF"),
        Patch(facecolor=METHOD_COLORS["BWORF"], alpha=0.7, label="BWORF"),
    ]
    fig.legend(handles=legend_elements, loc="center right", bbox_to_anchor=(0.98, 0.5),
               fontsize=11, frameon=True, edgecolor="gray")

    fig.suptitle("Causal Discovery: Mean Rank of Causal SNPs", fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 0.88, 0.95])

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_delta_auc_heatmap_gemini(
    df: pd.DataFrame,
    save_path: Path,
    fixed_h2: float = 0.15,
):
    """
    Heatmap of Delta AUC (BWORF - RF).
    - Subset to Interaction only
    - Fixed h2 value
    - x-axis: n_samples
    - y-axis: combined label "m=X | ρ=Y"
    """
    # Filter to Interaction only
    df_int = df[df["signal_type"] == "Interaction"].copy()
    if len(df_int) == 0:
        warnings.warn("No 'Interaction' data for heatmap. Skipping.")
        return

    # Filter to fixed h2
    h2_vals = sorted(df_int["h2"].unique())
    if fixed_h2 not in h2_vals:
        # Fall back to closest
        closest_h2 = min(h2_vals, key=lambda x: abs(x - fixed_h2))
        warnings.warn(f"h2={fixed_h2} not found. Using h2={closest_h2}.")
        fixed_h2 = closest_h2

    df_h2 = df_int[df_int["h2"] == fixed_h2].copy()
    if len(df_h2) == 0:
        warnings.warn(f"No data for h2={fixed_h2}. Skipping heatmap.")
        return

    # Compute delta AUC grouped by (n_samples, rho, n_causal_pairs)
    group_cols = ["n_samples", "rho", "n_causal_pairs"]
    delta_df = compute_delta_auc(df_h2, group_cols)

    # Create combined y-axis label
    delta_df["y_label"] = delta_df.apply(
        lambda r: f"m={int(r['n_causal_pairs'])} | ρ={r['rho']:.1f}", axis=1
    )

    # Pivot for heatmap: y_label vs n_samples
    pivot = delta_df.pivot(index="y_label", columns="n_samples", values="delta_auc")

    # Sort y-axis labels
    y_order = sorted(pivot.index, key=lambda x: (
        int(x.split("|")[0].split("=")[1].strip()),
        float(x.split("|")[1].split("=")[1].strip())
    ))
    pivot = pivot.reindex(y_order)

    # Sort x-axis (n_samples)
    x_order = sorted(pivot.columns)
    pivot = pivot[x_order]

    if pivot.empty:
        warnings.warn("Heatmap pivot is empty. Skipping.")
        return

    # Set up figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Diverging colormap centered at 0
    vmin = pivot.values[~np.isnan(pivot.values)].min() if not np.all(np.isnan(pivot.values)) else -0.1
    vmax = pivot.values[~np.isnan(pivot.values)].max() if not np.all(np.isnan(pivot.values)) else 0.1
    vabs = max(abs(vmin), abs(vmax), 0.01)
    norm = mcolors.TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)

    im = ax.imshow(pivot.values, cmap="RdBu_r", norm=norm, aspect="auto")

    # Set ticks
    ax.set_xticks(range(len(x_order)))
    ax.set_xticklabels([str(int(x)) for x in x_order], fontsize=11)
    ax.set_yticks(range(len(y_order)))
    ax.set_yticklabels(y_order, fontsize=11)

    ax.set_xlabel("N (samples)", fontsize=12)
    ax.set_ylabel("Scenario (m | ρ)", fontsize=12)

    # Annotate cells with ΔAUC values
    for i in range(len(y_order)):
        for j in range(len(x_order)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                text_color = "white" if abs(val) > vabs * 0.5 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=10, color=text_color, fontweight="bold")

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("ΔAUC (BWORF − RF)", fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    # Remove top/right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle(f"ΔAUC Heatmap: Interaction Regime (h²={fixed_h2})", fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {save_path}")


def create_summary_table(df: pd.DataFrame, save_path: Path):
    """
    Create summary table with mean metrics by scenario.
    """
    agg_df = aggregate_by_scenario(df)

    # Select columns for output
    output_cols = ["method", "signal_type"] + SCENARIO_COLS
    for metric in METRIC_COLS:
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"
        count_col = f"{metric}_count"
        if mean_col in agg_df.columns:
            output_cols.append(mean_col)
        if std_col in agg_df.columns:
            output_cols.append(std_col)
        if count_col in agg_df.columns:
            output_cols.append(count_col)

    output_cols = [c for c in output_cols if c in agg_df.columns]
    summary_df = agg_df[output_cols].copy()

    # Sort for readability
    sort_cols = ["method", "signal_type"] + [c for c in SCENARIO_COLS if c in summary_df.columns]
    summary_df = summary_df.sort_values(sort_cols).reset_index(drop=True)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(save_path, index=False)
    print(f"Saved: {save_path}")

    return summary_df


# ==============================================================================
# Main
# ==============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Simulation Results v2 - Generate figures and summary table"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=r"C:\Users\Arthur\Desktop\ORF\Dili Fixed\Simulation study\v2\outputs\simulation_results_v2.csv",
        help="Path to simulation_results_v2.csv",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=r"C:\Users\Arthur\Desktop\ORF\Dili Fixed\Simulation study\v2\outputs",
        help="Output directory for plots and tables",
    )
    parser.add_argument(
        "--heatmap_h2",
        type=float,
        default=0.15,
        help="Fixed h2 value for delta AUC heatmap (default: 0.15)",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    outdir = Path(args.outdir)
    plots_dir = outdir / "plots"
    tables_dir = outdir / "tables"

    print("=" * 70)
    print("Simulation Study v2 - Analysis & Visualization (Gemini Style)")
    print("=" * 70)
    print(f"CSV: {csv_path}")
    print(f"Output dir: {outdir}")
    print(f"Heatmap h2: {args.heatmap_h2}")
    print()

    # Load and validate
    df = load_and_validate_csv(csv_path)

    # Report unique scenarios
    unique_scenarios = df.groupby(SCENARIO_COLS).ngroups
    print(f"Unique scenarios: {unique_scenarios}")
    print(f"Unique methods: {sorted(df['method'].unique())}")
    print(f"n_samples values: {sorted(df['n_samples'].unique())}")
    print(f"rho values: {sorted(df['rho'].unique())}")
    print(f"h2 values: {sorted(df['h2'].unique())}")
    print(f"n_causal_pairs values: {sorted(df['n_causal_pairs'].unique())}")
    print()

    # Create output directories
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots
    print("-" * 70)
    print("Generating plots...")
    print("-" * 70)

    # Figure 1: AUC vs h2 (2x2 facet: n_samples x signal_type)
    plot_metric_vs_h2_gemini(
        df,
        metric="auc_roc",
        ylabel="AUC",
        title="Predictive Performance: AUC vs Heritability",
        save_path=plots_dir / "fig_auc_vs_h2_v2.png",
    )

    # Figure 2: Brier vs h2 (2x2 facet: n_samples x signal_type)
    plot_metric_vs_h2_gemini(
        df,
        metric="brier",
        ylabel="Brier Score",
        title="Predictive Performance: Brier Score vs Heritability",
        save_path=plots_dir / "fig_brier_vs_h2_v2.png",
    )

    # Figure 3: Mean rank boxplot (rows=n_samples, cols=rho, x=signal_type, hue=method)
    plot_mean_rank_boxplot_gemini(
        df,
        save_path=plots_dir / "fig_mean_rank_boxplot_v2.png",
    )

    # Figure 4: Delta AUC heatmap (Interaction only, fixed h2)
    plot_delta_auc_heatmap_gemini(
        df,
        save_path=plots_dir / "fig_delta_auc_heatmap_v2.png",
        fixed_h2=args.heatmap_h2,
    )

    # Summary table
    print("\n" + "-" * 70)
    print("Generating summary table...")
    print("-" * 70)
    summary_df = create_summary_table(df, tables_dir / "simulation_summary_table_v2.csv")

    # Final report
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Rows loaded: {len(df)}")
    print(f"Unique scenarios: {unique_scenarios}")
    print()
    print("Output files:")
    print(f"  - {plots_dir / 'fig_auc_vs_h2_v2.png'}")
    print(f"  - {plots_dir / 'fig_brier_vs_h2_v2.png'}")
    print(f"  - {plots_dir / 'fig_mean_rank_boxplot_v2.png'}")
    print(f"  - {plots_dir / 'fig_delta_auc_heatmap_v2.png'}")
    print(f"  - {tables_dir / 'simulation_summary_table_v2.csv'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
