#!/usr/bin/env python3
"""Refit and save the thesis-consistent multiclass BWORF+MI explanation model.

This script reproduces the exact multiclass thesis configuration used for
`multiclass_dili_merged_20260313_053906` and saves a single full-data refit
artifact for downstream LIME.

Locked configuration:
- dataset: data/dili_multiclass.csv
- target: y_3class
- ID: USER_ID
- features: D001..D777 only
- seed: 43
- imputation: median
- MI top-k: 50 (computed on the full imputed training matrix)
- scaling: StandardScaler (applied after MI selection)
- model: bworf_mi
- hyperparameters:
    n_estimators=100
    max_depth=6
    min_samples_split=10
    min_samples_leaf=1
    l1_strength=0.2
    weighted_bootstrap=True
    bootstrap_temperature=1.0
    n_tries=10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# project-local BWORF implementation

import sys

SCRIPT_DIR = Path(__file__).resolve().parent



# When this script lives in <project_root>/LIME/, the code dir is one level up.

PROJECT_CODE_DIR = SCRIPT_DIR.parent / 'code'

if PROJECT_CODE_DIR.exists() and str(PROJECT_CODE_DIR) not in sys.path:

    sys.path.insert(0, str(PROJECT_CODE_DIR))



from bworf_with_mi import ObliqueRandomForestMulti  # type: ignore
ID_COL = 'USER_ID'
TARGET_COL = 'y_3class'
TARGET_NAME_COL = 'y_3class_name'
EXCLUDE_COLS = {ID_COL, 'Likelihood_letter_raw', 'Likelihood_letter_base', TARGET_COL, TARGET_NAME_COL}
CLASS_ORDER = [0, 1, 2]
CLASS_NAME_MAP = {0: 'Low', 1: 'Moderate', 2: 'High'}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='Refit and save final LIME model artifact for multiclass BWORF+MI (seed 43).')
    ap.add_argument('--project_root', required=True, help='Path to dili_multiclass_patched_bworf project root')
    ap.add_argument('--seed', type=int, default=43, help='Locked final refit seed (default: 43)')
    ap.add_argument('--mi_top_k', type=int, default=50, help='Locked MI top-k (default: 50)')
    ap.add_argument('--output_dir', default=None, help='Optional output directory; default <project_root>/lime_artifacts/bworf_mi_seed<seed>')
    return ap.parse_args()


def load_data(project_root: Path) -> tuple[pd.DataFrame, List[str]]:
    data_path = project_root / 'data' / 'dili_multiclass.csv'
    if not data_path.exists():
        raise FileNotFoundError(f'Missing dataset: {data_path}')
    df = pd.read_csv(data_path)
    feature_cols = [c for c in df.columns if c.startswith('D') and len(c) == 4 and c[1:].isdigit() and c not in EXCLUDE_COLS]
    if len(feature_cols) != 777:
        raise ValueError(f'Expected 777 descriptor columns, found {len(feature_cols)}')
    if TARGET_COL not in df.columns:
        raise ValueError(f'Missing target column {TARGET_COL}')
    if ID_COL not in df.columns:
        raise ValueError(f'Missing ID column {ID_COL}')
    return df, feature_cols


def build_model(seed: int) -> ObliqueRandomForestMulti:
    return ObliqueRandomForestMulti(
        n_estimators=100,
        max_depth=6,
        min_samples_split=10,
        min_samples_leaf=1,
        l1_strength=0.2,
        random_state=seed,
        weighted_bootstrap=True,
        bootstrap_temperature=1.0,
        n_tries=10,
    )


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    if not project_root.exists():
        raise FileNotFoundError(f'Project root not found: {project_root}')

    # Ensure project code is importable when script is copied outside project root
    proj_code = project_root / 'code'
    if str(proj_code) not in sys.path:
        sys.path.insert(0, str(proj_code))

    df, feature_cols = load_data(project_root)
    X_df = df[feature_cols].copy()
    y = df[TARGET_COL].to_numpy(dtype=int)

    output_dir = Path(args.output_dir).resolve() if args.output_dir else (project_root / 'lime_artifacts' / f'bworf_mi_seed{args.seed}')
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) median imputation on full data
    imputer = SimpleImputer(strategy='median')
    X_imp = imputer.fit_transform(X_df)

    # 2) full-data MI ranking (locked top-50)
    mi_scores = mutual_info_classif(X_imp, y, random_state=args.seed)
    ranked = np.argsort(mi_scores)[::-1]
    selected_idx = ranked[: args.mi_top_k]
    selected_features = [feature_cols[i] for i in selected_idx]
    X_sel = X_imp[:, selected_idx]

    mi_rank_df = pd.DataFrame({
        'rank': np.arange(1, len(feature_cols) + 1),
        'feature_name': [feature_cols[i] for i in ranked],
        'mi_score': [float(mi_scores[i]) for i in ranked],
        'selected_for_final_model': [i in set(selected_idx) for i in ranked],
    })
    mi_rank_df.to_csv(output_dir / f'lime_selected_features_seed{args.seed}.csv', index=False)

    # 3) standard scaling after MI selection
    scaler = StandardScaler()
    X_proc = scaler.fit_transform(X_sel)

    # 4) fit final full-data model
    model = build_model(args.seed)
    model.fit(X_proc, y)

    # 5) save bundle
    bundle: Dict[str, object] = {
        'model_name': 'bworf_mi',
        'artifact_role': 'final_explanation_model',
        'artifact_version': 'thesis_lime_v1',
        'source_results_bundle': 'multiclass_dili_merged_20260313_053906',
        'source_oblique_run': 'multiclass_dili_oblique_20260312_092910',
        'project_root': str(project_root),
        'data_path': str((project_root / 'data' / 'dili_multiclass.csv').resolve()),
        'id_col': ID_COL,
        'target_col': TARGET_COL,
        'target_name_col': TARGET_NAME_COL,
        'class_order': CLASS_ORDER,
        'class_name_map': CLASS_NAME_MAP,
        'feature_cols': feature_cols,
        'selected_feature_indices': [int(i) for i in selected_idx.tolist()],
        'selected_features': selected_features,
        'seed': int(args.seed),
        'preprocessing': {
            'imputation': 'median',
            'mi_filtering': True,
            'mi_top_k': int(args.mi_top_k),
            'scaling': 'standard',
        },
        'hyperparameters': {
            'n_estimators': 100,
            'max_depth': 6,
            'min_samples_split': 10,
            'min_samples_leaf': 1,
            'l1_strength': 0.2,
            'weighted_bootstrap': True,
            'bootstrap_temperature': 1.0,
            'n_tries': 10,
        },
        'model': model,
        'imputer': imputer,
        'scaler': scaler,
        'n_samples': int(len(df)),
    }
    joblib.dump(bundle, output_dir / f'lime_final_bworf_mi_seed{args.seed}.joblib')

    # 6) save metadata JSON separately for transparency
    metadata = {k: v for k, v in bundle.items() if k not in {'model', 'imputer', 'scaler'}}
    with open(output_dir / f'lime_model_metadata_seed{args.seed}.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    # 7) save a full-data prediction snapshot for sanity checks
    proba = model.predict_proba(X_proc)
    pred = model.predict(X_proc)
    sanity_df = pd.DataFrame({
        'sample_index': np.arange(len(df)),
        ID_COL: df[ID_COL].to_numpy(),
        'y_true': y,
        'y_pred_full_refit': pred,
        'proba_0': proba[:, 0],
        'proba_1': proba[:, 1],
        'proba_2': proba[:, 2],
    })
    sanity_df.to_csv(output_dir / f'lime_full_refit_predictions_seed{args.seed}.csv', index=False)

    print(f'Saved final LIME model artifact to: {output_dir}')
    print(f'Selected {len(selected_features)} features for final explanation model.')


if __name__ == '__main__':
    main()
