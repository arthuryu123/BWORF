# Final thesis-consistent LIME workflow (bworf_mi, seed 43)

## Locked source of truth
- merged thesis result bundle: `multiclass_dili_merged_20260313_053906`
- upstream oblique run: `multiclass_dili_oblique_20260312_092910`
- target model family: `bworf_mi`
- final refit seed: `43`

## Locked config
- target column: `y_3class`
- ID column: `USER_ID`
- feature columns: `D001`..`D777`
- median imputation
- MI top-k = 50
- StandardScaler after MI selection
- BWORF hyperparameters:
  - n_estimators = 100
  - max_depth = 6
  - min_samples_split = 10
  - min_samples_leaf = 1
  - l1_strength = 0.2
  - weighted_bootstrap = True
  - bootstrap_temperature = 1.0
  - n_tries = 10

## Files
- `save_final_lime_bworf_mi_seed43.py` : full-data refit + save artifact
- `run_save_final_lime_bworf_mi_seed43.sbatch` : Sol job wrapper for the refit/save step
- `select_lime_instances_bworf_mi_seed43.py` : OOF-based case selection (optionally validates against final refit bundle)
- `lime_preselected_bworf_mi_seed43.csv` : provisional 9-case set from OOF seed 43
- `run_lime_bworf_mi_seed43.py` : generates raw LIME outputs and thesis-ready plots
- `run_lime_bworf_mi_seed43.sbatch` : Sol job wrapper for the LIME step

## Sol commands
### 1) Build final explanation artifact
sbatch run_save_final_lime_bworf_mi_seed43.sbatch \
  /ABS/PATH/TO/dili_multiclass_patched_bworf \
  /ABS/PATH/TO/save_final_lime_bworf_mi_seed43.py

### 2) Select final 9 cases (validated against saved artifact)
python select_lime_instances_bworf_mi_seed43.py \
  --merged_zip /ABS/PATH/TO/multiclass_dili_merged_20260313_053906.zip \
  --bundle /ABS/PATH/TO/dili_multiclass_patched_bworf/lime_artifacts/bworf_mi_seed43/lime_final_bworf_mi_seed43.joblib \
  --output_csv /ABS/PATH/TO/dili_multiclass_patched_bworf/lime_artifacts/bworf_mi_seed43/lime_selected_instances.csv \
  --candidate_csv /ABS/PATH/TO/dili_multiclass_patched_bworf/lime_artifacts/bworf_mi_seed43/lime_candidate_pool.csv

### 3) Run LIME
sbatch run_lime_bworf_mi_seed43.sbatch \
  /ABS/PATH/TO/dili_multiclass_patched_bworf \
  /ABS/PATH/TO/dili_multiclass_patched_bworf/lime_artifacts/bworf_mi_seed43/lime_final_bworf_mi_seed43.joblib \
  /ABS/PATH/TO/dili_multiclass_patched_bworf/lime_artifacts/bworf_mi_seed43/lime_selected_instances.csv \
  /ABS/PATH/TO/run_lime_bworf_mi_seed43.py

## Important nuance
In the merged OOF file, `y_pred` is not always the argmax of `predict_proba` for BWORF+MI.
The selection script therefore restricts candidate cases to those where `argmax(probabilities) == y_pred`
and can also validate that the full-data refit predicts the same class before finalizing the 9 cases.
