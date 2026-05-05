#!/usr/bin/env python3
"""Run thesis-consistent multiclass LIME for the locked BWORF+MI seed-43 artifact.

This script expects:
- a saved model bundle from save_final_lime_bworf_mi_seed43.py
- a selected instance CSV from select_lime_instances_bworf_mi_seed43.py

It explains the selected class for each case (OOF-selected predicted class or validated final-refit class)
and writes both raw explanation tables and thesis-ready bar plots.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from lime.lime_tabular import LimeTabularExplainer
except Exception as exc:  # pragma: no cover
    raise ImportError(
        'The lime package is required for this script. Install it in the same environment, e.g. `pip install lime`.'
    ) from exc

CLASS_NAME_MAP = {0: 'Low', 1: 'Moderate', 2: 'High'}
CATEGORY_TITLE_MAP = {
    'high_confidence_correct': 'High-confidence correct',
    'borderline_correct': 'Borderline correct',
    'notable_error': 'Notable error',
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='Run LIME on the locked multiclass BWORF+MI seed-43 artifact.')
    ap.add_argument('--bundle', required=True, help='Path to lime_final_bworf_mi_seed43.joblib')
    ap.add_argument('--selected_csv', required=True, help='Path to selected LIME instances CSV')
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--num_features', type=int, default=10)
    ap.add_argument('--num_samples', type=int, default=5000)
    return ap.parse_args()


def load_selected_feature_matrix(bundle) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str]]:
    df = pd.read_csv(bundle['data_path'])
    feature_cols = bundle['feature_cols']
    X_raw = df[feature_cols].copy()
    y = df[bundle['target_col']].to_numpy(dtype=int)
    X_imp = bundle['imputer'].transform(X_raw)
    selected_idx = np.array(bundle['selected_feature_indices'], dtype=int)
    selected_features = bundle['selected_features']
    X_sel_raw = X_imp[:, selected_idx]
    return df, X_sel_raw, y, selected_features


def predict_fn_from_selected_raw(bundle):
    scaler = bundle['scaler']
    model = bundle['model']
    def _predict(selected_raw_np: np.ndarray) -> np.ndarray:
        X_proc = scaler.transform(selected_raw_np) if scaler is not None else selected_raw_np
        return model.predict_proba(X_proc)
    return _predict


def final_pred_from_selected_raw(bundle, X_row: np.ndarray) -> tuple[int, np.ndarray]:
    scaler = bundle['scaler']
    model = bundle['model']
    X_proc = scaler.transform(X_row.reshape(1, -1)) if scaler is not None else X_row.reshape(1, -1)
    pred = int(model.predict(X_proc)[0])
    proba = model.predict_proba(X_proc)[0]
    return pred, proba


def sanitize_name(s: str) -> str:
    return s.replace(' ', '_').replace('/', '_').replace('-', '_').lower()


def make_plot(exp_pairs: List[tuple[str, float]], title: str, outpath: Path) -> None:
    # preserve LIME order; plot negatives left, positives right with diverging colors
    features = [x[0] for x in exp_pairs]
    weights = np.array([x[1] for x in exp_pairs], dtype=float)
    colors = ['#1f77b4' if w < 0 else '#d62728' for w in weights]
    fig_h = max(4.5, 0.42 * len(features) + 1.5)
    fig, ax = plt.subplots(figsize=(9, fig_h))
    y = np.arange(len(features))
    ax.barh(y, weights, color=colors, alpha=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(features)
    ax.axvline(0, color='#555555', linewidth=1)
    ax.set_xlabel('Local contribution weight')
    ax.set_title(title)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(outpath.with_suffix('.png'), dpi=300, bbox_inches='tight')
    fig.savefig(outpath.with_suffix('.pdf'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    args = parse_args()
    bundle = joblib.load(args.bundle)
    selected = pd.read_csv(args.selected_csv)
    outdir = Path(args.output_dir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    df_full, X_sel_raw, y, selected_features = load_selected_feature_matrix(bundle)
    predict_fn = predict_fn_from_selected_raw(bundle)

    explainer = LimeTabularExplainer(
        training_data=X_sel_raw,
        feature_names=selected_features,
        class_names=[CLASS_NAME_MAP[c] for c in bundle['class_order']],
        discretize_continuous=True,
        mode='classification',
        random_state=bundle['seed'],
    )

    raw_rows = []
    summary_rows = []
    for _, row in selected.iterrows():
        sample_index = int(row['sample_index'])
        X_row = X_sel_raw[sample_index]
        # explain the selected predicted class if present, else the final model prediction
        explain_class = int(row['final_refit_pred']) if 'final_refit_pred' in row and not pd.isna(row['final_refit_pred']) else int(row['y_pred'])
        final_pred, final_proba = final_pred_from_selected_raw(bundle, X_row)
        exp = explainer.explain_instance(
            X_row,
            predict_fn,
            labels=[explain_class],
            num_features=args.num_features,
            num_samples=args.num_samples,
        )
        pairs = exp.as_list(label=explain_class)

        # save raw rows
        for rank, (feature_expr, weight) in enumerate(pairs, start=1):
            raw_rows.append({
                'sample_index': sample_index,
                'USER_ID': int(row['USER_ID']),
                'selection_category': row['selection_category'],
                'true_class': int(row['y_true']),
                'selected_oof_pred_class': int(row['y_pred']),
                'explained_class': int(explain_class),
                'final_refit_pred_class': int(final_pred),
                'final_refit_pred_proba': float(final_proba[final_pred]),
                'rank': rank,
                'feature_expression': feature_expr,
                'weight': float(weight),
            })

        summary_rows.append({
            'sample_index': sample_index,
            'USER_ID': int(row['USER_ID']),
            'selection_category': row['selection_category'],
            'true_class': int(row['y_true']),
            'selected_oof_pred_class': int(row['y_pred']),
            'final_refit_pred_class': int(final_pred),
            'explained_class': int(explain_class),
            'final_refit_pred_proba': float(final_proba[final_pred]),
            'n_features_shown': len(pairs),
        })

        title = (
            f"{CATEGORY_TITLE_MAP.get(row['selection_category'], row['selection_category'])}: "
            f"sample {sample_index} (ID {int(row['USER_ID'])}) | "
            f"true={CLASS_NAME_MAP[int(row['y_true'])]}, "
            f"explained={CLASS_NAME_MAP[int(explain_class)]}"
        )
        stub = f"lime_{sanitize_name(str(row['selection_category']))}_sample{sample_index}_id{int(row['USER_ID'])}"
        make_plot(pairs, title, outdir / stub)

    raw_df = pd.DataFrame(raw_rows)
    raw_df.to_csv(outdir / 'lime_explanations_topk.csv', index=False)
    pd.DataFrame(summary_rows).to_csv(outdir / 'lime_explanations_summary.csv', index=False)

    metadata = {
        'bundle_path': str(Path(args.bundle).resolve()),
        'selected_csv': str(Path(args.selected_csv).resolve()),
        'num_features': int(args.num_features),
        'num_samples': int(args.num_samples),
        'n_cases': int(len(selected)),
    }
    with open(outdir / 'lime_run_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    print(f'LIME outputs written to: {outdir}')
    print(f'Generated explanations for {len(selected)} cases.')


if __name__ == '__main__':
    main()
