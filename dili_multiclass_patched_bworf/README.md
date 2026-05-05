# DILI Multiclass Benchmark — Patched BWORF

This directory contains the primary multiclass DILI (Drug-Induced Liver Injury) classification benchmark for an in-progress master's thesis on Bootstrap-Weighted Oblique Random Forest (BWORF) by Chung-Yuan Yu (Arizona State University). The directory holds the patched implementation of BWORF, the benchmark runners, the merged baseline + oblique results that constitute the headline multiclass benchmark, and the LIME interpretability pipeline used to generate the per-case explanation figures.

This README documents the contents of the directory, explains the meaning of the "patched" naming, identifies the canonical thesis result, and describes how to reproduce it.

---

## 1. What is in this directory

```
dili_multiclass_patched_bworf/
├── README.md                       this file
├── code/
│   ├── bworf_with_mi.py           PATCHED BWORF model + utilities (see §3)
│   ├── run_multiclass_dili_patched_benchmark.py
│   ├── run_multiclass_dili_patched_benchmark_v2.py
│   ├── run_multiclass_dili_shard.py
│   ├── merge_multiclass_dili_results.py
│   ├── merge_multiclass_dili_shards.py
│   └── plot_multiclass_dili_results_v3.py
├── data/
│   ├── dili_multiclass.csv                574 compounds × Mold2 descriptors + label
│   └── dili_multiclass_feature_names.csv  feature schema
├── refs/                           reference label files
├── configs/                        per-run JSON configs (currently empty placeholder)
├── LIME/                           LIME pipeline scripts (see §6)
├── lime_artifacts/                 saved LIME refit model + selected cases (see §6)
├── lime_outputs/                   LIME explanation files and figures (see §6)
├── outputs/                        all benchmark outputs (see §5)
├── logs/                           SLURM log files
├── tmp/                            scratch directory (empty)
├── run_multiclass_dili_baselines.sbatch    SLURM wrapper, baseline panel
├── run_multiclass_dili_oblique.sbatch      SLURM wrapper, oblique panel
├── run_multiclass_dili_patched.sbatch      SLURM wrapper, original patched run
├── run_bworf_mi_k100_sharded.sbatch        exploratory hyperparameter run (see §5)
├── run_bworf_mi_k100_robust_sharded.sbatch exploratory hyperparameter run, refined
├── run_bworf_mi_k150_sharded.sbatch        exploratory hyperparameter run, k=150
└── run_bworf_rescue.sbatch                 SLURM rescue/re-run wrapper
```

---

## 2. The dataset

The benchmark uses 574 compounds from the DILIrank 1.0 reference list, each represented as a 777-dimensional Mold2 descriptor vector and labeled with a three-tier multiclass DILI risk endpoint derived from LiverTox likelihood categories.

| Class | Meaning                              | LiverTox source | N   | Prevalence |
|-------|--------------------------------------|-----------------|-----|------------|
| 0     | No DILI concern                      | E, E*           | 333 | 58.0%      |
| 1     | Likely DILI concern                  | C, D            | 101 | 17.6%      |
| 2     | Most DILI concern                    | A, B            | 140 | 24.4%      |

Predictors are the 777 Mold2 descriptor columns (D001–D777). Compound identifier (USER_ID) and label-provenance fields are retained for auditing but excluded from the predictor matrix during model fitting.

Dataset construction, label harmonization, and descriptor generation are described in Chapter 2 of the thesis.

---

## 3. What "patched" means

This directory uses a revised BWORF implementation in which leaf-level prediction probabilities use the same class-balanced weighting logic as the rest of the model.

### 3.1 The inconsistency that motivated the patch

The original BWORF design uses class-balanced weighting at two stages of training:

1. **Bootstrap sampling** (`ObliqueRandomForestMulti.fit`): when `weighted_bootstrap=True`, samples are drawn into each tree's bootstrap with probabilities proportional to inverse class frequency.
2. **Split selection** (`ObliqueDecisionTreeMulti._find_best_split_logistic`): the L1-regularized logistic regression that learns each oblique split is fitted with class-balanced sample weights, and Gini impurity is computed against weighted class probabilities.

In the original (unpatched) code, terminal-leaf probabilities were computed from raw class counts in the leaf:

```
p_k = count_k / total_samples_in_leaf
```

This is inconsistent with the class-balancing applied earlier: a leaf with 20 majority-class and 3 minority-class samples would return roughly `[0.87, 0.13]`, even though both the bootstrap and split-finding logic were specifically designed to compensate for this kind of class imbalance.

### 3.2 What the patch changes

The patched implementation adds explicit class-balanced weighted statistics inside `ObliqueDecisionTreeMulti`. Each terminal node now stores `class_weight_sums` and `class_probabilities` computed under the same per-node `n / (n_classes × count_k)` weighting used for split evaluation. At prediction time, `predict_proba` returns these weighted leaf probabilities rather than raw class counts.

The key modified members in `code/bworf_with_mi.py` are:

- `compute_node_weights(...)`
- `_weighted_class_totals(...)`
- `_gini_impurity(...)`
- `_build_tree(...)`
- `_predict_proba_sample(...)`
- `test_weighted_leaf_consistency(...)`

A fallback path that uses raw class counts is preserved for compatibility, but is not exercised by the benchmark runs.

### 3.3 Effect on metrics

The patch changes probability outputs only. Hard-class predictions in `ObliqueRandomForestMulti.predict()` are still computed from per-tree majority votes. As a consequence:

- **Probability-based metrics** (macro AUROC, macro AUPRC, log loss, Brier) are sensitive to the patch.
- **Hard-decision metrics** (accuracy, balanced accuracy, macro F1, MCC) are also sensitive but indirectly, since `argmax(predict_proba)` may not equal `predict()`.
- **LIME explanations** are sensitive, because LIME queries `predict_proba`.

The patch was applied because the thesis evaluation primarily uses probability-based metrics; without the patch, the model's probabilistic outputs would not reflect the class-balancing applied during training.

### 3.4 Recovering the unpatched version

A unified diff documenting the exact change is preserved at:

```
../patch_external/bworf_with_mi_patch.diff
```

The diff is between `bworf_with_mi_backup.py` (unpatched, dated 2026-03-06 16:00 UTC) and `bworf_with_mi_patched.py` (patched, dated 2026-03-06 16:13 UTC). To reconstruct the unpatched version from the patched source:

```bash
cp code/bworf_with_mi.py /tmp/bworf_with_mi.py
cd /tmp
patch -R -i /path/to/patch_external/bworf_with_mi_patch.diff bworf_with_mi.py
# /tmp/bworf_with_mi.py is now the unpatched version
```

---

## 4. Models compared

Eight model families are evaluated under a single common protocol. The eight are organized in three layers:

**Conventional baselines** (from `run_multiclass_dili_baselines.sbatch`):
1. `lr` — multinomial logistic regression
2. `rf` — random forest
3. `svm_rbf` — support vector machine with RBF kernel
4. `xgb` — XGBoost
5. `naive_bayes` — Gaussian Naive Bayes

**Oblique comparator** (from `run_multiclass_dili_oblique.sbatch`):
6. `orf_style` — oblique random forest comparator (axis-aligned + oblique splits)

**BWORF variants** (also from `run_multiclass_dili_oblique.sbatch`):
7. `bworf_no_mi` — patched BWORF without mutual-information feature filtering
8. `bworf_mi` — patched BWORF with fold-local top-100 mutual-information feature filtering

The thesis reports a single BWORF row in the headline benchmark table, selected per the convention defined in the thesis Methods chapter (highest seed-level accuracy among coherent operating points).

---

## 5. Outputs and the canonical thesis result

The `outputs/` directory contains every benchmark run produced during the project, including exploratory hyperparameter sweeps. Not all runs are cited in the thesis. This section identifies which is canonical.

### 5.1 The canonical merged result

```
outputs/multiclass_dili_merged_20260313_053906/
```

This directory contains the eight-model benchmark used as the headline multiclass result. It was produced by `merge_multiclass_dili_results.py` from two independent SLURM runs:

| Source run                                                | Models merged                                            |
|-----------------------------------------------------------|----------------------------------------------------------|
| `outputs/multiclass_dili_baselines_20260312_092014/`     | lr, rf, svm_rbf, xgb, naive_bayes                       |
| `outputs/multiclass_dili_oblique_20260312_092910/`       | orf_style, bworf_no_mi, bworf_mi                        |

Sanity check (from `sanity_check.json`):
- 574 samples × 8 models × 5 seeds = 22,960 OOF rows ✓
- 8 models × 5 seeds × 10 folds = 400 fold rows ✓
- 0 duplicate OOF keys ✓

Files inside:

| File                          | Description                                              |
|-------------------------------|----------------------------------------------------------|
| `fold_results.csv`           | per-fold metrics for every (model, seed, fold) cell      |
| `oof_predictions.csv`         | pooled OOF predictions (USER_ID, y_true, y_pred, proba_0, proba_1, proba_2) |
| `summary_metrics.csv`         | aggregated metrics across seeds                          |
| `confusion_matrices.json`     | per-model confusion matrices                             |
| `mi_selected_features.csv`    | per-fold MI-selected feature lists for `bworf_mi`        |
| `feature_space_manifest.csv`  | feature audit metadata                                   |
| `split_manifest.csv`          | per-fold split assignments for reproducibility           |
| `merge_metadata.json`         | which source runs were merged and when                   |
| `sanity_check.json`           | row-count and uniqueness assertions                      |
| `plots/`                      | initial figure set (ROC OvR, PR OvR, confusion heatmaps) |
| `plots_refined/`              | thesis-final refined figure set                          |

A zip archive of this directory exists at `outputs/multiclass_dili_merged_20260313_053906.zip` (5.4 MB) for portability.

### 5.2 Other outputs

The remaining directories under `outputs/` are not the canonical headline result but are retained for provenance:

| Directory                                                       | Description                                                   |
|-----------------------------------------------------------------|---------------------------------------------------------------|
| `multiclass_dili_baselines_20260312_092014/`                   | source run for baseline rows of the merged result            |
| `multiclass_dili_oblique_20260312_092910/`                     | source run for oblique and BWORF rows of the merged result   |
| `bworf_mi_k100_l005_t20_d8_e300_shards/`                       | exploratory hyperparameter run, sharded by seed              |
| `bworf_mi_k100_l005_t20_d8_e300_robust_shards/`                | refined exploratory run with stricter checks                  |
| `bworf_mi_k150_l005_t20_d8_e300_shards/`                       | exploratory variant with k=150 MI features                   |
| `bworf_mi_k100_l005_t20_d8_e300_merged_20260321_190430/`       | merged result of the k=100 exploratory run                   |

The exploratory BWORF runs (`l005_t20_d8_e300`) used different hyperparameters than the canonical run: `l1_strength=0.05`, `n_tries=20`, `max_depth=8`, `n_estimators=300`. The canonical run uses the configuration reported in the thesis Methods chapter.

---

## 6. LIME interpretability pipeline

Local interpretability is reported in the thesis using the LIME framework applied to the canonical `bworf_mi` model at seed 43 (the seed-level operating point selected as the headline BWORF entry).

The LIME workflow is a three-stage pipeline distributed across three sibling directories:

| Stage | Directory                                          | Contents                                                         |
|-------|----------------------------------------------------|------------------------------------------------------------------|
| 1     | `LIME/`                                            | scripts and SLURM wrappers driving the pipeline                  |
| 2     | `lime_artifacts/bworf_mi_seed43/`                  | locked refit model, MI feature list, selected cases              |
| 3     | `lime_outputs/bworf_mi_seed43/`                    | per-case LIME explanation tables and thesis-ready figures        |

### Stage 1: scripts (`LIME/`)

| Script                                       | Purpose                                                                 |
|---------------------------------------------|-------------------------------------------------------------------------|
| `select_lime_instances_bworf_mi_seed43.py`  | identifies the nine validated LIME case instances from pooled OOF       |
| `save_final_lime_bworf_mi_seed43.py`        | refits the BWORF+MI model on all 574 compounds at seed 43 and saves it  |
| `run_lime_bworf_mi_seed43.py`               | computes LIME explanations for each selected instance                   |
| `*.sbatch`                                  | SLURM wrappers for the above                                            |
| `LIME_NEXT_STEPS.md`                        | working notes from pipeline development                                 |
| `lime_preselected_bworf_mi_seed43.csv`      | candidate-pool intermediate for case selection                          |

### Stage 2: artifacts (`lime_artifacts/bworf_mi_seed43/`)

| File                                          | Description                                                |
|----------------------------------------------|------------------------------------------------------------|
| `lime_final_bworf_mi_seed43.joblib`          | full-data BWORF+MI refit model (locked source-of-truth)    |
| `lime_model_metadata_seed43.json`            | refit configuration and provenance                         |
| `lime_selected_features_seed43.csv`          | MI-selected feature list used during refit                 |
| `lime_full_refit_predictions_seed43.csv`     | predictions from the refit model on all 574 compounds      |
| `lime_selected_instances.csv`                | the nine validated case instances                          |
| `lime_candidate_pools.csv`                   | full candidate pool from which cases were selected         |

### Stage 3: outputs (`lime_outputs/bworf_mi_seed43/`)

| File                                | Description                                                  |
|------------------------------------|--------------------------------------------------------------|
| `lime_explanations_summary.csv`    | per-case top-feature summaries                               |
| `lime_explanations_topk.csv`       | per-case top-K feature contributions (longer-form table)     |
| `lime_run_metadata.json`           | LIME run configuration and seed                              |
| `*.pdf`, `*.png`                   | nine per-case figure panels (one per validated case)         |

The case-selection logic, refit protocol, and explanation aggregation are described in the thesis under "Interpretability Analyses."

---

## 7. Methodology summary

The full protocol is described in Chapter 3 of the thesis. The benchmark in this directory implements the following:

| Aspect                  | Setting                                                              |
|------------------------|----------------------------------------------------------------------|
| Cross-validation        | Stratified 10-fold                                                   |
| Repetitions             | 5 fixed seeds: 41, 42, 43, 44, 45                                    |
| Preprocessing           | fold-local imputation; standardization for oblique models only       |
| Feature filtering       | for `bworf_mi`: fold-local top-100 mutual information                |
| Reporting               | seed-level pooled OOF; mean ± SD across seeds                        |
| Threshold-free metrics  | macro AUROC (OvR), macro AUPRC (OvR), log loss                      |
| Operational metrics     | accuracy, balanced accuracy, macro F1, MCC under argmax decision     |
| Confidence intervals    | Wilson 95% for accuracy on selected BWORF operating points          |

BWORF hyperparameters used in the canonical run match the thesis Methods chapter; the exploratory `l005_t20_d8_e300` variants in §5.2 use a different configuration and are not the thesis-cited result.

---

## 8. Reproducing the canonical run

The canonical merged result was produced by running two SLURM jobs and merging their outputs.

### 8.1 Baseline panel

```bash
sbatch run_multiclass_dili_baselines.sbatch
```

This runs the five baseline models (lr, rf, svm_rbf, xgb, naive_bayes) under stratified 10-fold CV across all five seeds. Output goes to `outputs/multiclass_dili_baselines_<TIMESTAMP>/`.

### 8.2 Oblique + BWORF panel

```bash
sbatch run_multiclass_dili_oblique.sbatch
```

This runs the three oblique models (`orf_style`, `bworf_no_mi`, `bworf_mi`) under the same protocol. Output goes to `outputs/multiclass_dili_oblique_<TIMESTAMP>/`.

### 8.3 Merge

After both panels complete:

```bash
python code/merge_multiclass_dili_results.py \
    --baselines_dir outputs/multiclass_dili_baselines_<TIMESTAMP>/ \
    --oblique_dir   outputs/multiclass_dili_oblique_<TIMESTAMP>/ \
    --out_dir       outputs/multiclass_dili_merged_<NEW_TIMESTAMP>/
```

The merge script verifies that both source runs use compatible split manifests, deduplicates OOF predictions, and writes the canonical eight-model output set.

### 8.4 Plots

Final plots (used in the thesis) are produced by:

```bash
python code/plot_multiclass_dili_results_v3.py \
    --merged_dir outputs/multiclass_dili_merged_<NEW_TIMESTAMP>/
```

This writes both the initial `plots/` directory and the refined `plots_refined/` directory.

---

## 9. Environment

The benchmark was run on the ASU Sol HPC under:

- Python 3.10 (cluster module)
- numpy, pandas, scikit-learn, xgboost, lightgbm
- LIME (for the interpretability pipeline)

A specific dependency manifest is not preserved in this directory; package versions follow the standard HPC `module load` environment current at the time of the run (March 2026). For exact version pinning, see the BMC paper's external benchmark suite under `external_benchmark_brfss/` which uses an explicit `pydeps/` directory.

---

## 10. Provenance and known caveats

- **Single-commit history.** This directory was committed in one commit during initial GitHub migration. Per-file modification dates reflect Sol filesystem mtimes, not git history.
- **Original BWORF source not preserved as a separate file.** The unpatched version is recoverable via `patch_external/bworf_with_mi_patch.diff` (see §3.4) but no `bworf_with_mi_unpatched.py` is committed standalone.
- **`configs/` is currently empty.** Per-run configurations are embedded inside `outputs/<run>/merge_metadata.json` and inside the sbatch scripts themselves rather than in standalone JSON config files.
- **Merge timestamp drift.** The merged directory's timestamp (`20260313_053906`) is the merge time, not the time of the underlying source runs (`20260312_092014` and `20260312_092910`). The split-manifest verification inside the merge script ensures consistency despite the timestamp gap.

---

## 11. Citation

Citation to be updated after publication.
