#!/usr/bin/env python3
"""Select thesis-consistent LIME cases from the merged multiclass OOF file.

Selection source-of-truth:
- merged results: multiclass_dili_merged_20260313_053906
- model: bworf_mi
- seed: 43

Selection rule:
- high-confidence correct: highest predicted-class probability among correct cases, stratified by true class
- borderline correct: smallest positive decision margin among correct cases, stratified by true class
- notable error: highest predicted-class probability among incorrect cases, stratified by true class

To avoid ambiguity between hard predictions and probability ranking, candidates are restricted to cases where
argmax(probabilities) == y_pred from the benchmark output. Optionally, if a final saved model bundle is supplied,
selected cases are validated against the final refit model, and the script advances to the next candidate if the
final refit prediction does not match the OOF-selected predicted class.
"""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

CLASS_NAME_MAP = {0: 'Low', 1: 'Moderate', 2: 'High'}
CATEGORIES = ['high_confidence_correct', 'borderline_correct', 'notable_error']


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='Select thesis-consistent LIME cases from merged OOF results.')
    ap.add_argument('--merged_zip', required=True, help='Path to multiclass_dili_merged_20260313_053906.zip')
    ap.add_argument('--model_name', default='bworf_mi')
    ap.add_argument('--seed', type=int, default=43)
    ap.add_argument('--bundle', default=None, help='Optional final explanation model bundle for validation')
    ap.add_argument('--output_csv', required=True)
    ap.add_argument('--candidate_csv', default=None, help='Optional path to write ranked candidate pools')
    return ap.parse_args()


def load_merged_oof(path: Path, model_name: str, seed: int) -> pd.DataFrame:
    with zipfile.ZipFile(path) as z:
        candidates = [
            n for n in z.namelist()
            if n.endswith('oof_predictions.csv') and not n.endswith('/')
        ]
        if not candidates:
            raise KeyError(
                f"No oof_predictions.csv found in archive. First entries were: {z.namelist()[:20]}"
            )
        oof_name = sorted(candidates, key=len)[0]
        print(f"[INFO] Reading OOF predictions from zip member: {oof_name}")
        oof = pd.read_csv(io.BytesIO(z.read(oof_name)))
    df = oof[(oof['model_name'] == model_name) & (oof['seed'] == seed)].copy()
    if df.empty:
        raise ValueError(f'No OOF rows found for model={model_name}, seed={seed}')
    P = df[['proba_0', 'proba_1', 'proba_2']].to_numpy(dtype=float)
    df['pred_from_probs'] = P.argmax(axis=1)
    pred = df['y_pred'].to_numpy(dtype=int)
    df['predicted_probability'] = np.array([row[p] for row, p in zip(P, pred)])
    df['decision_margin'] = df['predicted_probability'].to_numpy() - np.array([np.max(np.delete(row, p)) for row, p in zip(P, pred)])
    df['correct'] = (df['y_true'].to_numpy(dtype=int) == pred)
    df['argmax_matches_predict'] = (df['pred_from_probs'].to_numpy(dtype=int) == pred)
    df['true_class_name'] = df['y_true'].map(CLASS_NAME_MAP)
    df['predicted_class_name'] = df['y_pred'].map(CLASS_NAME_MAP)
    return df


def load_bundle(path: Path):
    return joblib.load(path)


def final_refit_predict(bundle, sample_rows: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    feature_cols = bundle['feature_cols']
    selected_idx = np.array(bundle['selected_feature_indices'], dtype=int)
    imputer = bundle['imputer']
    scaler = bundle['scaler']
    model = bundle['model']
    X_raw = sample_rows[feature_cols].copy()
    X_imp = imputer.transform(X_raw)
    X_sel = X_imp[:, selected_idx]
    X_proc = scaler.transform(X_sel) if scaler is not None else X_sel
    pred = model.predict(X_proc)
    proba = model.predict_proba(X_proc)
    return pred, proba


def build_candidate_pool(df: pd.DataFrame) -> Dict[Tuple[int, str], pd.DataFrame]:
    pool: Dict[Tuple[int, str], pd.DataFrame] = {}
    # restrict to cases where hard prediction aligns with probability ordering for clean explanation semantics
    clean = df[df['argmax_matches_predict']].copy()
    for cls in [0, 1, 2]:
        sub = clean[clean['y_true'] == cls].copy()
        pool[(cls, 'high_confidence_correct')] = sub[sub['correct']].sort_values(
            ['predicted_probability', 'decision_margin'], ascending=[False, False]
        )
        # borderline: smallest nonnegative margins among correct cases
        border = sub[sub['correct']].copy()
        border = border.sort_values(['decision_margin', 'predicted_probability'], ascending=[True, True])
        pool[(cls, 'borderline_correct')] = border
        pool[(cls, 'notable_error')] = sub[~sub['correct']].sort_values(
            ['predicted_probability', 'decision_margin'], ascending=[False, False]
        )
    return pool


def select_cases(df: pd.DataFrame, pool: Dict[Tuple[int, str], pd.DataFrame], bundle=None) -> pd.DataFrame:
    selected_rows = []
    used_sample_idx = set()
    # optional full-data validation
    full_df = None
    if bundle is not None:
        data_path = Path(bundle['data_path'])
        full_df = pd.read_csv(data_path)

    for cls in [0, 1, 2]:
        for category in CATEGORIES:
            candidates = pool[(cls, category)].copy()
            if candidates.empty:
                raise ValueError(f'No candidates available for class={cls}, category={category}')
            chosen = None
            for _, row in candidates.iterrows():
                if int(row['sample_index']) in used_sample_idx:
                    continue
                if bundle is not None and full_df is not None:
                    sample_row = full_df.iloc[[int(row['sample_index'])]].copy()
                    final_pred, final_proba = final_refit_predict(bundle, sample_row)
                    final_pred = int(final_pred[0])
                    if final_pred != int(row['y_pred']):
                        continue
                    row = row.copy()
                    row['final_refit_pred'] = final_pred
                    row['final_refit_pred_name'] = CLASS_NAME_MAP[final_pred]
                    row['final_refit_pred_proba'] = float(final_proba[0, final_pred])
                chosen = row.copy()
                break
            if chosen is None:
                raise ValueError(f'Could not validate any candidate for class={cls}, category={category}.')
            chosen['selection_category'] = category
            chosen['selection_rule'] = 'argmax-matched OOF candidate; validated against final refit' if bundle is not None else 'argmax-matched OOF candidate'
            selected_rows.append(chosen)
            used_sample_idx.add(int(chosen['sample_index']))
    selected = pd.DataFrame(selected_rows)
    selected = selected[[
        'sample_index','USER_ID','y_true','true_class_name','y_pred','predicted_class_name',
        'predicted_probability','decision_margin','selection_category','selection_rule',
        *([c for c in ['final_refit_pred','final_refit_pred_name','final_refit_pred_proba'] if c in selected.columns])
    ]].sort_values(['y_true','selection_category'])
    return selected


def main() -> None:
    args = parse_args()
    merged_zip = Path(args.merged_zip).resolve()
    df = load_merged_oof(merged_zip, args.model_name, args.seed)
    pool = build_candidate_pool(df)

    if args.candidate_csv:
        cand_rows = []
        for (cls, category), cdf in pool.items():
            tmp = cdf.head(25).copy()
            tmp['selection_category'] = category
            cand_rows.append(tmp)
        pd.concat(cand_rows, ignore_index=True).to_csv(args.candidate_csv, index=False)

    bundle = load_bundle(Path(args.bundle).resolve()) if args.bundle else None
    selected = select_cases(df, pool, bundle=bundle)
    output_csv = Path(args.output_csv).resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output_csv, index=False)
    print(f'Wrote selected LIME cases to: {output_csv}')
    print(selected.to_string(index=False))


if __name__ == '__main__':
    main()
