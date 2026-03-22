#!/usr/bin/env python3
"""
Refined multiclass DILI plotting script.

Key differences from the earlier plotting script:
- Writes to a NEW subfolder by default: plots_refined/
- Main metric boxplot includes ONLY comparable metrics:
    Accuracy, Balanced Accuracy, Macro F1, Macro Recall, Macro AUROC, Macro AUPRC
- MCC is plotted separately
- Log loss is not plotted
- Uses seaborn for boxplots and heatmaps

Usage:
python code/plot_multiclass_dili_results_v3.py \
  --merged_dir outputs/multiclass_dili_merged_20260313_053906
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

CLASS_ORDER = [0, 1, 2]
CLASS_LABELS = {0: "Low", 1: "Medium", 2: "High"}

MODEL_ORDER = [
    "rf",
    "lr",
    "svm_rbf",
    "xgb",
    "naive_bayes",
    "orf_style",
    "bworf_no_mi",
    "bworf_mi",
]

MODEL_LABELS = {
    "rf": "RF",
    "lr": "Logistic Regression",
    "svm_rbf": "SVM (RBF)",
    "xgb": "XGBoost",
    "naive_bayes": "Naive Bayes",
    "orf_style": "ORF-style",
    "bworf_no_mi": "BWORF",
    "bworf_mi": "BWORF + MI",
}

MAIN_METRICS = [
    ("accuracy", "Accuracy"),
    ("balanced_accuracy", "Balanced Accuracy"),
    ("macro_f1", "Macro F1"),
    ("macro_recall", "Macro Recall"),
    ("macro_auroc_ovr", "Macro AUROC"),
    ("macro_auprc_ovr", "Macro AUPRC"),
]

REQUIRED_FILES = [
    "fold_results.csv",
    "summary_metrics.csv",
    "oof_predictions.csv",
    "mi_selected_features.csv",
]


def parse_args():
    p = argparse.ArgumentParser(description="Create refined plots for merged multiclass DILI results.")
    p.add_argument("--merged_dir", required=True, help="Merged benchmark output folder")
    p.add_argument("--plots_subdir", default="plots_refined", help="Subfolder name to write plots into")
    p.add_argument("--top_mi_n", type=int, default=20, help="Top N MI features to show")
    p.add_argument("--dpi", type=int, default=300, help="PNG export DPI")
    return p.parse_args()


def ensure_inputs(merged_dir: str) -> Path:
    root = Path(merged_dir).resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Merged directory not found: {root}")
    missing = [name for name in REQUIRED_FILES if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files in merged directory: {missing}")
    return root


def configure_theme():
    sns.set_theme(
        style="whitegrid",
        context="talk",
        font_scale=0.95,
        rc={
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.8,
            "axes.edgecolor": "#444444",
            "axes.linewidth": 1.0,
            "legend.frameon": True,
        },
    )


def get_color_map(models_in_data):
    palette = {
        "rf": "#4C78A8",
        "lr": "#72B7B2",
        "svm_rbf": "#54A24B",
        "xgb": "#B279A2",
        "naive_bayes": "#E3BA22",
        "orf_style": "#F58518",
        "bworf_no_mi": "#E45756",
        "bworf_mi": "#C73E1D",
    }
    return {m: palette[m] for m in MODEL_ORDER if m in models_in_data}


def save_figure(fig, out_base: Path, dpi: int):
    fig.savefig(out_base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def make_long_metric_df(fold_df: pd.DataFrame, metric_pairs):
    rows = []
    for metric_key, metric_label in metric_pairs:
        tmp = fold_df[["model_name", metric_key]].copy()
        tmp["Metric"] = metric_label
        tmp["Score"] = tmp[metric_key]
        rows.append(tmp[["model_name", "Metric", "Score"]])
    long_df = pd.concat(rows, ignore_index=True)
    long_df["model_name"] = pd.Categorical(long_df["model_name"], categories=MODEL_ORDER, ordered=True)
    long_df["Metric"] = pd.Categorical(
        long_df["Metric"],
        categories=[label for _, label in metric_pairs],
        ordered=True,
    )
    return long_df


def plot_metric_boxplots_main(fold_df: pd.DataFrame, plots_dir: Path, dpi: int, color_map: dict):
    long_df = make_long_metric_df(fold_df, MAIN_METRICS)
    model_order_present = [m for m in MODEL_ORDER if m in set(long_df["model_name"].dropna().astype(str))]

    fig, ax = plt.subplots(figsize=(14, 7))
    sns.boxplot(
        data=long_df,
        x="Metric",
        y="Score",
        hue="model_name",
        order=[label for _, label in MAIN_METRICS],
        hue_order=model_order_present,
        palette=color_map,
        width=0.82,
        linewidth=1.0,
        fliersize=3,
        ax=ax,
    )

    ax.set_title("Multiclass DILI Benchmark: Fold-Level Metric Distributions", pad=12)
    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")

    # Set a sensible y-axis based on the actual comparable metrics.
    score_min = float(long_df["Score"].min())
    lower = max(0.0, score_min - 0.03)
    upper = min(1.02, max(1.0, float(long_df["Score"].max()) + 0.01))
    ax.set_ylim(lower, upper)

    ax.tick_params(axis="x", rotation=28)
    ax.legend(
        title="Model",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
    )
    save_figure(fig, plots_dir / "metric_boxplots_main_refined", dpi=dpi)


def plot_mcc_boxplot(fold_df: pd.DataFrame, plots_dir: Path, dpi: int, color_map: dict):
    sub = fold_df[["model_name", "mcc"]].copy().rename(columns={"mcc": "Score"})
    sub["model_name"] = pd.Categorical(sub["model_name"], categories=MODEL_ORDER, ordered=True)
    order_present = [m for m in MODEL_ORDER if m in set(sub["model_name"].dropna().astype(str))]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(
        data=sub,
        x="model_name",
        y="Score",
        order=order_present,
        palette=color_map,
        width=0.65,
        linewidth=1.0,
        fliersize=3,
        ax=ax,
    )
    ax.set_title("Multiclass DILI Benchmark: MCC Distributions", pad=12)
    ax.set_xlabel("Model")
    ax.set_ylabel("MCC")
    ax.set_xticklabels([MODEL_LABELS[m] for m in order_present], rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.30)
    save_figure(fig, plots_dir / "mcc_boxplot_refined", dpi=dpi)


def plot_roc_ovr(oof_df: pd.DataFrame, plots_dir: Path, dpi: int, color_map: dict):
    models = [m for m in MODEL_ORDER if m in set(oof_df["model_name"])]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4), sharex=True, sharey=True)

    for ax, cls in zip(axes, CLASS_ORDER):
        for model_name in models:
            sub = oof_df[oof_df["model_name"] == model_name]
            y_true_bin = (sub["y_true"].to_numpy(dtype=int) == cls).astype(int)
            score = sub[f"proba_{cls}"].to_numpy(dtype=float)
            fpr, tpr, _ = roc_curve(y_true_bin, score)
            roc_auc = auc(fpr, tpr)
            ax.plot(
                fpr, tpr,
                linewidth=2.2,
                color=color_map[model_name],
                label=f"{MODEL_LABELS[model_name]} ({roc_auc:.3f})",
            )
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2, color="#777777")
        ax.set_title(f"{CLASS_LABELS[cls]} vs Rest", pad=8)
        ax.set_xlabel("False Positive Rate")
        if cls == 0:
            ax.set_ylabel("True Positive Rate")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.28)

    axes[-1].legend(loc="lower left", bbox_to_anchor=(1.02, 0.0), frameon=True, title="Model (AUC)")
    fig.suptitle("One-vs-Rest ROC Curves for Multiclass DILI", y=1.04)
    save_figure(fig, plots_dir / "roc_ovr_3panel_refined", dpi=dpi)


def plot_pr_ovr(oof_df: pd.DataFrame, plots_dir: Path, dpi: int, color_map: dict):
    models = [m for m in MODEL_ORDER if m in set(oof_df["model_name"])]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4), sharex=True, sharey=True)

    for ax, cls in zip(axes, CLASS_ORDER):
        positive_rate = (oof_df["y_true"] == cls).mean()
        for model_name in models:
            sub = oof_df[oof_df["model_name"] == model_name]
            y_true_bin = (sub["y_true"].to_numpy(dtype=int) == cls).astype(int)
            score = sub[f"proba_{cls}"].to_numpy(dtype=float)
            precision, recall, _ = precision_recall_curve(y_true_bin, score)
            ap = average_precision_score(y_true_bin, score)
            ax.plot(
                recall, precision,
                linewidth=2.2,
                color=color_map[model_name],
                label=f"{MODEL_LABELS[model_name]} ({ap:.3f})",
            )
        ax.axhline(positive_rate, linestyle="--", linewidth=1.2, color="#777777")
        ax.set_title(f"{CLASS_LABELS[cls]} vs Rest", pad=8)
        ax.set_xlabel("Recall")
        if cls == 0:
            ax.set_ylabel("Precision")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.28)

    axes[-1].legend(loc="lower left", bbox_to_anchor=(1.02, 0.0), frameon=True, title="Model (AP)")
    fig.suptitle("One-vs-Rest Precision-Recall Curves for Multiclass DILI", y=1.04)
    save_figure(fig, plots_dir / "pr_ovr_3panel_refined", dpi=dpi)


def plot_confusion_matrices(oof_df: pd.DataFrame, plots_dir: Path, dpi: int):
    models = [m for m in MODEL_ORDER if m in set(oof_df["model_name"])]
    fig, axes = plt.subplots(2, 4, figsize=(17, 8.8))
    axes = axes.ravel()
    cmap = sns.color_palette("Blues", as_cmap=True)

    for ax, model_name in zip(axes, models):
        sub = oof_df[oof_df["model_name"] == model_name]
        cm = confusion_matrix(sub["y_true"], sub["y_pred"], labels=CLASS_ORDER)
        cm_norm = cm.astype(float) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)

        sns.heatmap(
            cm_norm,
            vmin=0.0,
            vmax=1.0,
            cmap=cmap,
            annot=True,
            fmt=".2f",
            square=True,
            cbar=False,
            linewidths=0.6,
            linecolor="white",
            xticklabels=[CLASS_LABELS[c] for c in CLASS_ORDER],
            yticklabels=[CLASS_LABELS[c] for c in CLASS_ORDER],
            ax=ax,
        )
        ax.set_title(MODEL_LABELS[model_name], pad=6)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    for ax in axes[len(models):]:
        ax.axis("off")

    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_clim(0, 1)
    cbar = fig.colorbar(sm, ax=axes.tolist(), fraction=0.02, pad=0.02)
    cbar.set_label("Row-normalized proportion")
    fig.suptitle("Pooled Row-Normalized Confusion Matrices", y=1.02)
    save_figure(fig, plots_dir / "confusion_matrices_pooled_refined", dpi=dpi)


def plot_mi_top_features(mi_df: pd.DataFrame, plots_dir: Path, dpi: int, top_n: int):
    if len(mi_df) == 0:
        return None

    top = (
        mi_df.groupby("feature_name", as_index=False)["mi_score"]
        .mean()
        .sort_values("mi_score", ascending=False)
        .head(top_n)
        .sort_values("mi_score", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(
        data=top,
        x="mi_score",
        y="feature_name",
        color="#C73E1D",
        edgecolor="black",
        linewidth=0.8,
        ax=ax,
    )
    ax.set_xlabel("Mean mutual information score")
    ax.set_ylabel("Feature")
    ax.set_title(f"Top {len(top)} MI-Ranked Features for BWORF + MI", pad=10)
    ax.grid(axis="x", alpha=0.28)
    save_figure(fig, plots_dir / "mi_top_features_barplot_refined", dpi=dpi)
    return top


def main():
    args = parse_args()
    configure_theme()
    root = ensure_inputs(args.merged_dir)
    plots_dir = root / args.plots_subdir
    plots_dir.mkdir(parents=True, exist_ok=True)

    fold_df = pd.read_csv(root / "fold_results.csv")
    summary_df = pd.read_csv(root / "summary_metrics.csv")
    oof_df = pd.read_csv(root / "oof_predictions.csv")
    mi_df = pd.read_csv(root / "mi_selected_features.csv")

    models_in_data = sorted(set(fold_df["model_name"].unique().tolist()) | set(oof_df["model_name"].unique().tolist()))
    color_map = get_color_map(models_in_data)

    plot_metric_boxplots_main(fold_df, plots_dir, args.dpi, color_map)
    plot_mcc_boxplot(fold_df, plots_dir, args.dpi, color_map)
    plot_roc_ovr(oof_df, plots_dir, args.dpi, color_map)
    plot_pr_ovr(oof_df, plots_dir, args.dpi, color_map)
    plot_confusion_matrices(oof_df, plots_dir, args.dpi)
    top_mi = plot_mi_top_features(mi_df, plots_dir, args.dpi, args.top_mi_n)

    summary_df.to_csv(plots_dir / "summary_metrics_table.csv", index=False)
    with open(plots_dir / "model_color_map.json", "w", encoding="utf-8") as f:
        json.dump({MODEL_LABELS[k]: v for k, v in color_map.items()}, f, indent=2)

    manifest = {
        "source_merged_dir": str(root),
        "plots_dir": str(plots_dir),
        "main_boxplot_metrics": [label for _, label in MAIN_METRICS],
        "excluded_from_main_boxplot": ["MCC", "Log Loss"],
        "created_files": [
            "metric_boxplots_main_refined.png",
            "metric_boxplots_main_refined.pdf",
            "mcc_boxplot_refined.png",
            "mcc_boxplot_refined.pdf",
            "roc_ovr_3panel_refined.png",
            "roc_ovr_3panel_refined.pdf",
            "pr_ovr_3panel_refined.png",
            "pr_ovr_3panel_refined.pdf",
            "confusion_matrices_pooled_refined.png",
            "confusion_matrices_pooled_refined.pdf",
            "summary_metrics_table.csv",
            "model_color_map.json",
            "plot_manifest.json",
        ],
        "optional_files": [
            "mi_top_features_barplot_refined.png",
            "mi_top_features_barplot_refined.pdf",
        ],
        "top_mi_n": args.top_mi_n,
    }
    with open(plots_dir / "plot_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Plots written to: {plots_dir}")
    print("Main boxplot metrics:", [label for _, label in MAIN_METRICS])
    print("Excluded from main boxplot: ['MCC', 'Log Loss']")
    print("Created files:")
    for name in manifest["created_files"]:
        print(f"  - {name}")
    if top_mi is not None:
        print("Top MI features:")
        print(top_mi.to_string(index=False))


if __name__ == "__main__":
    main()
