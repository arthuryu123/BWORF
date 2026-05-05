# External Benchmark Suite — Classical Datasets

This directory contains the classical external biomedical tabular benchmarks for an in-progress master's thesis on Bootstrap-Weighted Oblique Random Forest (BWORF) by Chung-Yuan Yu (Arizona State University). Four small public datasets — breast cancer recurrence, heart failure, diabetes (Pima Indians), and thyroid disease — are evaluated under a common protocol to triangulate BWORF's behavior outside the DILI domain on which the method was developed.

A second external benchmark, the BRFSS 2022 heart-attack dataset, lives in the sibling directory `../brfss/` and provides a large, modern, severely imbalanced binary task. Together, the two directories under `external_benchmark/` constitute the full external suite.

This README documents the contents of the classical directory, the methodology and hyperparameters, the per-dataset outputs, and how to reproduce the runs.

---

## 1. About the directory name

This directory was originally named `patch_external/` during the project because it was the working area where the patched BWORF implementation (see `bworf_with_mi_patch.diff` and §3) was first deployed to external (non-DILI) benchmarks. It was renamed to `external_benchmark/classical/` for clarity during repository preparation. The "patch" in the original name referred to the model-implementation patch documented in §3, not to any data correction.

---

## 2. What is in this directory

```
external_benchmark/classical/
├── README.md                       this file
├── bworf_with_mi_patch.diff       unified diff: original BWORF -> patched BWORF
├── code/
│   ├── bworf_with_mi.py           PATCHED BWORF model (see §3)
│   ├── benchmark_runner_external.py    main runner used for the canonical results
│   └── benchmark_runner_patchcheck.py  alternate runner used for patch-check experiments
├── configs/                        empty placeholder; configs are embedded in run_config.json files
├── datasets/
│   ├── breast_cancer.csv.gz        UCI Ljubljana breast cancer recurrence (286 rows, 9 features)
│   ├── diabetes.csv.gz             Pima Indians diabetes (768 rows, 8 features)
│   ├── heart_failure.csv.gz        UCI heart failure clinical records (299 rows, 12 features)
│   └── thyroid.csv.gz              UCI new-thyroid 215-instance subset (215 rows, 5 features)
├── sbatch/
│   ├── run_external.sbatch         per-(dataset,model) SLURM wrapper for canonical runs
│   └── run_patchcheck.sbatch       SLURM wrapper for patch-check experiments
├── submit_all_external.sh          submits all 16 (dataset × model) SLURM jobs for the canonical run
├── submit_all_patchcheck.sh        submits patch-check verification jobs (historical, not canonical)
├── logs/                           SLURM .err logs from the canonical run
└── outputs/
    ├── breast_cancer/
    ├── diabetes/
    ├── heart_failure/
    └── thyroid/
        └── <model>/
            ├── fold_results.csv         per-fold metrics
            ├── oof_predictions.csv.gz   pooled out-of-fold predictions
            ├── run_config.json          configuration metadata
            └── summary.csv              aggregated metrics
```

---

## 3. The patched BWORF model

This directory uses the same patched BWORF implementation documented in `../../dili_multiclass_patched_bworf/README.md` §3, where leaf-level prediction probabilities use class-balanced weighting consistent with the rest of the model rather than reverting to raw terminal-node class counts.

The unified diff that documents the change between the unpatched and patched implementations lives at the top of this directory:

```
bworf_with_mi_patch.diff
```

The diff is between `bworf_with_mi_backup.py` (unpatched, 2026-03-06 16:00 UTC) and `bworf_with_mi_patched.py` (patched, 2026-03-06 16:13 UTC). To recover the unpatched version:

```bash
cp code/bworf_with_mi.py /tmp/bworf_with_mi.py
cd /tmp
patch -R -i /path/to/bworf_with_mi_patch.diff bworf_with_mi.py
```

For full background on what the patch changes and why, see the dili README §3.

---

## 4. The four datasets

| Dataset                         | Source                | N    | Features | Task    | Target          |
|--------------------------------|-----------------------|------|----------|---------|-----------------|
| Breast cancer recurrence       | UCI (Ljubljana, 1988) | 286  | 9        | Binary  | `class`         |
| Heart failure clinical records | UCI                   | 299  | 12       | Binary  | `DEATH_EVENT`   |
| Diabetes (Pima Indians)        | UCI                   | 768  | 8        | Binary  | `Outcome`       |
| Thyroid disease (new-thyroid)  | UCI                   | 215  | 5        | 3-class | `class`         |

All four are well-established small biomedical tabular benchmarks. They were chosen for the thesis specifically to cover a range of feature counts and task types (three binary tasks plus one multiclass) at small sample sizes, which contrasts with the larger BRFSS 2022 dataset in the sibling `../brfss/` directory.

### 4.1 Note on breast cancer recurrence

The breast cancer recurrence dataset is included in this directory because it was part of the thesis external suite. The peer-reviewed paper extension of this work omits this dataset in favor of the BRFSS 2022 dataset documented in `../brfss/`. The breast cancer outputs in this directory remain as thesis-era artifacts.

### 4.2 Preprocessing

Each dataset is converted to a purely numerical feature matrix prior to model fitting. Categorical fields, where present, are one-hot encoded. Missing values are handled using fold-local imputation: imputation parameters are estimated using only the training partition of each fold and applied to the held-out partition to prevent leakage. RF is fit on the resulting numeric matrix without scaling. The oblique models (`orf_style`, `bworf_no_mi`, `bworf_mi`) use fold-local standardization to stabilize oblique split learning, consistent with the patched DILI multiclass pipeline.

---

## 5. Models compared

The external suite uses an intentionally compact, forest-based panel of four models:

1. **`rf`** — random forest (axis-aligned, scikit-learn baseline)
2. **`orf_style`** — oblique random forest comparator
3. **`bworf_no_mi`** — patched BWORF without mutual-information feature filtering
4. **`bworf_mi`** — patched BWORF with fold-local top-k mutual-information feature filtering

This is narrower than the DILI multiclass benchmark (which uses 8 model families) because the external suite is methodological triangulation rather than a broad model bake-off.

---

## 6. Methodology and hyperparameters

| Aspect                      | Setting                                                          |
|-----------------------------|------------------------------------------------------------------|
| Cross-validation            | Stratified 10-fold                                               |
| Repetitions                 | 5 fixed seeds: 42, 43, 44, 45, 46                                |
| Preprocessing               | fold-local imputation; standardization for oblique models only   |
| Feature filtering (`bworf_mi`) | fold-local top-k mutual information (k varies by dataset)    |
| Reporting                   | seed-level pooled OOF; mean ± SD across seeds                    |
| Threshold-free metrics      | AUROC, AUPRC (binary) or macro AUROC, macro AUPRC (3-class thyroid) |

### 6.1 BWORF hyperparameters (locked across all four datasets)

| Parameter                       | Value |
|---------------------------------|-------|
| `n_estimators`                  | 100   |
| `max_depth`                     | 6     |
| `min_samples_split`             | 10    |
| `min_samples_leaf`              | 1     |
| `l1_strength`                   | 0.2   |
| `bootstrap_temperature`         | 1.0   |
| `weighted_bootstrap`            | True  |
| `n_tries`                       | 10    |

These values are identical for all four datasets and for both `bworf_no_mi` and `bworf_mi` runs. They differ from the BWORF settings used in the `../brfss/` directory, which uses a separately-tuned configuration appropriate to that dataset's larger N and different feature dimensionality.

### 6.2 MI feature-filter top-k by dataset

The `bworf_mi` model selects the top-k features by mutual information per fold. Because the four datasets have different total feature counts, the selected k differs per dataset:

| Dataset        | Total features | `top_k_used` |
|----------------|----------------|--------------|
| Heart failure  | 12             | 8            |
| Diabetes       | 8              | 6            |
| Breast cancer  | 9              | 6            |
| Thyroid        | 5              | 4            |

In each case, k is set to keep most of the available features while still filtering out the least informative dimensions, in proportion to the dataset's total feature count.

---

## 7. Outputs

For every (dataset, model) combination, the runner writes four files into `outputs/<dataset>/<model>/`:

| File                    | Description                                                              |
|------------------------|--------------------------------------------------------------------------|
| `fold_results.csv`      | one row per (seed, fold), 23 columns of per-fold metrics and metadata    |
| `oof_predictions.csv.gz`| pooled out-of-fold predictions across all seeds (one row per sample)     |
| `run_config.json`       | full configuration: dataset metadata, hyperparameters, package versions  |
| `summary.csv`           | aggregated metrics (mean and SD across the 50 fold rows)                 |

The summary file reports threshold-free metrics (AUROC, AUPRC), threshold-derived metrics under the default argmax decision rule (accuracy, balanced accuracy, F1 on the positive class for binary tasks, MCC), calibration metrics (log loss, Brier score), and timing.

For the multiclass thyroid task, the same column names are used, but binary-specific columns (`f1_pos`, `precision_pos`, `recall_pos`) are computed under the default argmax decision rule rather than being task-specific.

---

## 8. Reproducing the canonical run

The canonical run consists of 16 SLURM jobs (4 datasets × 4 models). They were submitted via:

```bash
bash submit_all_external.sh
```

This script issues 16 `sbatch sbatch/run_external.sbatch <dataset> <model>` calls. Each task fits and evaluates one (dataset, model) cell across 5 seeds and 10 folds, then writes the four output files.

### 8.1 Path note

The committed `sbatch/run_external.sbatch` and `submit_all_external.sh` reference `/home/chungyua/patch_external/`, the original directory path used during the thesis. The files are preserved verbatim for provenance. To re-run from this repository's path, the `ROOT` variable in `run_external.sbatch` and the path in `submit_all_external.sh` must be updated to point at the cloned `external_benchmark/classical/` location.

### 8.2 Patch-check experiments (historical)

`submit_all_patchcheck.sh` and `sbatch/run_patchcheck.sbatch` are preserved from a separate verification workflow that compared patched and unpatched BWORF on a subset of datasets (heart failure and thyroid) under a reduced 5-fold protocol with 3 seeds. Their outputs are not in this directory; the verification was conducted in a separate working directory (`/home/chungyua/patch_check_heart_thyroid/`) that is not part of this repository. The scripts are retained as historical artifacts of the patch-validation process.

---

## 9. Environment

The classical external runs used a different Python environment than the BRFSS work in `../brfss/`. Specifically:

- Python 3.10.19
- numpy 1.26.4, pandas 2.3.3, scikit-learn 1.6.1
- Conda environment named `sim_orf`, located at `~/.conda/envs/sim_orf/` on the original Sol filesystem
- Loaded via `module load mamba/latest`

The BRFSS directory uses a Python 3.9.2 environment with locally installed dependencies under `pydeps/`. The two environments produce numerically consistent results because the BWORF computations rely on standard scikit-learn and numpy primitives that are stable across the version ranges involved.

---

## 10. Provenance and known caveats

- **Single-commit repository history.** This directory was committed in one commit during initial GitHub migration. Per-file modification dates reflect Sol filesystem mtimes, not git history.
- **`configs/` directory is empty.** Per-run configurations are embedded inside each `outputs/<dataset>/<model>/run_config.json` file rather than in standalone config files.
- **Sbatch hardcoded paths are stale post-rename.** See §8.1.
- **Patch-check working directory is external.** See §8.2; the validation runs were conducted outside this repository.
- **Original BWORF source not preserved as a separate file.** The unpatched version is recoverable via `bworf_with_mi_patch.diff` (see §3); no `bworf_with_mi_unpatched.py` is committed standalone.

---

## 11. Citation

Citation to be updated after publication.
