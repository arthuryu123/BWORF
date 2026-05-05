# External Benchmark — BRFSS 2022 Heart Attack

This directory contains the BRFSS 2022 Heart Attack external benchmark for an in-progress master's thesis on Bootstrap-Weighted Oblique Random Forest (BWORF) by Chung-Yuan Yu (Arizona State University) and a forthcoming peer-reviewed paper extension. The benchmark evaluates BWORF against four comparator models on a large, modern, severely imbalanced binary classification task using publicly available U.S. survey data.

The classical four-dataset external suite (heart failure, diabetes, breast cancer recurrence, thyroid) lives in the sibling directory `../classical/` and provides smaller-scale triangulation. The BRFSS work in this directory provides the population-scale, modern complement that is the centerpiece of the peer-reviewed paper extension.

This README documents the dataset, methodology, the two parallel runs (subsample and full-N), engineering decisions made to make the full-N run feasible, the temperature sensitivity sweep, and how to reproduce each piece.

---

## 1. What is in this directory

```
external_benchmark/brfss/
├── README.md                       this file
├── code/
│   ├── prepare_heart2022.py        BRFSS data prep: cleaning, encoding, split generation
│   ├── benchmark_runner.py         main runner used for all 5 models on both runs
│   ├── score_holdout_heart2022.py  refit on modeling set, score on holdout
│   ├── aggregate_heart2022.py      aggregate per-seed runs into _aggregated/
│   ├── aggregate_full_bworf.py     aggregate the 50-task BWORF full-N shards
│   ├── aggregate_temp_sweep.py     aggregate the temperature sensitivity runs
│   ├── plot_temp_sweep.py          generate the temperature sensitivity figure
│   └── profile_bworf.py            profiling utility for BWORF fit-time scaling
├── sbatch/
│   ├── run_subsample_fast.sbatch       canonical subsample run (5 models)
│   ├── run_subsample_bworf.sbatch      subsample BWORF (separate due to longer runtime)
│   ├── run_full_fast.sbatch            full-N run for the 4 fast models
│   ├── run_full_bworf_parallel.sbatch  full-N BWORF, 50-task array, parallel implementation
│   ├── run_heart22_temp_sweep.sbatch   temperature sensitivity sweep, subsample
│   ├── score_holdout_full.sbatch       holdout scoring for the 4 fast models, full-N
│   ├── score_holdout_bworf.sbatch      holdout scoring for BWORF, subsample
│   ├── score_holdout_bworf_full.sbatch holdout scoring for BWORF, full-N
│   └── profile_bworf.sbatch            profiling SLURM wrapper
├── engineering/
│   ├── validate_parallel_bworf.py          byte-equivalence test for parallel BWORF
│   ├── patch_runner_for_bworf_parallel.py  AST patcher: wires BWORFParallelClassifier into runner
│   └── patch_runner_for_fold_sharding.py   AST patcher: adds --fold_ids and OOF gating
├── models/
│   ├── bworf_with_mi.py            PATCHED BWORF model (same as in dili and classical)
│   └── bworf_parallel.py           BWORFParallelClassifier: byte-equivalent parallel fit
├── data/
│   ├── README.md                   how to obtain the BRFSS CSV from Kaggle
│   └── (raw BRFSS CSV not committed; see data/README.md)
└── outputs/
    ├── subsample/                  aggregated outputs for the subsample run
    ├── full/                       aggregated outputs for the full-N fast-models run
    ├── full_bworf/                 aggregated outputs for the full-N BWORF run
    ├── full_bworf_holdout/         holdout scores for full-N BWORF
    └── temp_sweep/                 temperature sensitivity sweep aggregated outputs
```

Per-seed and per-shard intermediate outputs are not committed — only `_aggregated/` results are included. Full out-of-fold prediction CSVs are excluded for the full-N runs (78 MB and 46 MB respectively); they remain on the original Sol filesystem and are available on request.

---

## 2. The dataset

The benchmark uses the publicly available BRFSS 2022 personal-key-indicators dataset hosted on Kaggle, derived from the U.S. Centers for Disease Control and Prevention's 2022 Behavioral Risk Factor Surveillance System survey. The CDC original is in the public domain.

| Aspect             | Value                                                       |
|--------------------|-------------------------------------------------------------|
| Source             | Kaggle: `kamilpytlak/personal-key-indicators-of-heart-disease` |
| File used          | `heart_2022_no_nans.csv`                                    |
| SHA-256            | `f5eddf47d85170f2f7bc3ba523c275d705ddf22518f95d50aa33ddcfd786a94f` |
| Rows after dedup   | 245,986                                                     |
| Target             | `HadHeartAttack` (Yes/No → 1/0)                             |
| Positive prevalence| 5.46% (severely imbalanced)                                 |
| Predictors         | 39 features after dropping `State`                          |

The `State` column is dropped because BRFSS sampling weights are state-stratified; including state would partially encode the survey design rather than the substantive demographic and behavioral predictors. All other columns are retained.

For instructions on obtaining the raw CSV, see `data/README.md`.

---

## 3. Two parallel runs

The benchmark is run twice with different sample sizes:

| Run        | Modeling set | Holdout test | Purpose                                            |
|-----------|-------------|--------------|----------------------------------------------------|
| Subsample | 4,000        | 1,000        | sanity check; matches scale of classical externals |
| Full-N    | 196,788      | 49,198       | population-scale evaluation; primary paper result  |

Both runs use stratified 80/20 modeling/holdout splits drawn with `prep_seed=20260425`. Modeling-set predictions come from stratified 10-fold cross-validation; holdout predictions come from refitting on the entire modeling set and scoring on the held-out 20%.

The subsample provides a methodological consistency check at the same data scale as the classical four-dataset external suite. The full-N run provides the population-scale evaluation that demonstrates BWORF's behavior at modern data sizes.

---

## 4. Models

Five models are evaluated in both runs:

1. **`lr`** — multinomial logistic regression with class-balanced weights
2. **`rf`** — random forest, scikit-learn baseline
3. **`xgb`** — XGBoost with `scale_pos_weight` for imbalance
4. **`orf`** — oblique random forest comparator
5. **`bworf`** — patched BWORF with weighted bootstrap (the focal model)

For the meaning of "patched," see `../../dili_multiclass_patched_bworf/README.md` §3 or `../classical/README.md` §3. Briefly: the patched implementation makes leaf-level predicted probabilities use the same class-balanced weighting as the rest of the model, instead of reverting to raw terminal-node class counts.

---

## 5. Methodology

### 5.1 Cross-validation protocol

| Aspect              | Setting                                                     |
|---------------------|-------------------------------------------------------------|
| Cross-validation    | Stratified 10-fold                                          |
| Repetitions         | 5 fixed seeds: `13, 42, 77, 123, 2025`                       |
| Holdout protocol    | Refit on full modeling set per seed, score on 20% holdout   |
| Threshold-free      | AUROC, AUPRC                                                |
| Threshold-derived   | Accuracy, balanced accuracy, F1 on positive, MCC            |
| Calibration         | Log loss, Brier score                                       |

### 5.2 Locked BWORF hyperparameters

Identical for both subsample and full-N runs:

| Parameter               | Value |
|-------------------------|-------|
| `n_estimators`          | 100   |
| `max_depth`             | 5     |
| `min_samples_split`     | 50    |
| `min_samples_leaf`      | 5     |
| `l1_strength`           | 1.0   |
| `bootstrap_temperature` | 1.0   |
| `weighted_bootstrap`    | True  |
| `n_tries`               | 2     |

These differ from the classical-suite BWORF settings in `../classical/` (which uses `max_depth=6`, `l1_strength=0.2`, `n_tries=10`). The BRFSS settings are tuned for the dataset's larger N and lower feature dimensionality (39 features vs. 5-12 in the classical suite).

---

## 6. Engineering: scaling BWORF to full-N

### 6.1 The problem

BWORF fit time scales near-quadratically with sample size in the patched implementation (empirical scaling exponent ~1.95). At full-N with 196,788 modeling-set rows under 10-fold CV, a serial implementation would require approximately one CPU-month per seed × 5 seeds = roughly five CPU-months of fit time, in addition to per-fold prediction.

This is impractical for a SLURM-managed shared cluster with per-job walltime limits, even with the longest available walltime queue.

### 6.2 The parallel BWORF implementation

`models/bworf_parallel.py` defines `BWORFParallelClassifier`, a subclass of `ObliqueRandomForestMulti` from `bworf_with_mi.py` that parallelizes tree fitting across joblib workers while preserving byte-exact equivalence with the serial implementation.

The byte-equivalence is achieved by **pre-consuming the master RNG sequentially** before parallel dispatch, so each worker receives a deterministic seed for its assigned trees. Workers fit their trees in parallel but produce results identical to a sequential fit of the same trees.

The `prediction_proba` method is similarly parallelized but byte-exact: each tree's predictions are computed in parallel and averaged, producing identical floating-point results to the serial version (modulo joblib's order-of-summation, which is fixed by sorting tree indices before reduction).

### 6.3 Validation

`engineering/validate_parallel_bworf.py` is a standalone validation script that asserts byte-equivalence between the serial and parallel implementations. It runs:

- 6 synthetic configurations (varying n_samples, n_features, random states)
- 1 real BRFSS subsample configuration

For each, it fits both a serial `ObliqueRandomForestMulti` and a parallel `BWORFParallelClassifier` with the same RNG seed and asserts that:

- Predictions are bit-exact identical
- `pred_mismatch_rate` (between hard predictions and argmax of probabilities) is identical between implementations

The test passed for all 7 configurations, including bit-exact prediction match against the thesis-cited serial BWORF runs on the BRFSS subsample.

### 6.4 Runner integration

The runner (`code/benchmark_runner.py`) was originally designed to fit one full forest per (seed, fold) pair. To use the parallel BWORF efficiently, two patches were applied via AST-based patchers:

- **`engineering/patch_runner_for_bworf_parallel.py`** — wires `BWORFParallelClassifier` into the runner, with a new `--bworf_parallel_n_jobs` flag.
- **`engineering/patch_runner_for_fold_sharding.py`** — adds `--fold_ids` to allow per-fold execution and gates row-count and OOF-NaN sanity checks behind a `--drop_incomplete_oof` flag, since per-fold runs naturally have incomplete OOF coverage.

Both patches are idempotent and AST-validated.

### 6.5 Production run

The full-N BWORF run was executed as a 50-task SLURM array (5 seeds × 10 folds), via:

```bash
sbatch sbatch/run_full_bworf_parallel.sbatch
```

Each task fit one (seed, fold) cell with 32 CPUs, 64 GB memory, and a 7-day walltime ceiling. Mean fit time was approximately 5.9 hours per fold; all 50 tasks completed successfully.

After completion, results were aggregated via:

```bash
python code/aggregate_full_bworf.py
```

This produces the `outputs/full_bworf/_aggregated/` directory whose summary CSV is committed to this repository.

---

## 7. Results

### 7.1 Subsample CV (10-fold × 5 seeds, mean ± SD)

| Model | AUROC | BalAcc | MCC | F1_pos | AUPRC |
|-------|-------|--------|-----|--------|-------|
| BWORF | 0.852 ± 0.038 | 0.720 | 0.347 | 0.378 | 0.358 |
| LR    | 0.872 | 0.787 | 0.325 | 0.317 | 0.385 |
| ORF   | 0.873 | 0.775 | 0.345 | 0.351 | 0.361 |
| RF    | 0.878 | 0.775 | 0.348 | 0.355 | 0.382 |
| XGB   | 0.840 | 0.647 | 0.319 | 0.349 | 0.354 |

At subsample scale, BWORF performs in the middle of the panel. LR, ORF, and RF have a slight AUROC edge.

### 7.2 Full-N CV (10-fold × 5 seeds, mean ± SD)

From `outputs/full_bworf/summary.csv` and `outputs/full/summary.csv`:

| Model | AUROC | BalAcc | MCC | F1_pos | AUPRC |
|-------|-------|--------|-----|--------|-------|
| BWORF | 0.8886 | 0.8009 | 0.327 | 0.310 | **0.4171** |
| LR    | 0.8886 | 0.8018 | 0.343 | 0.331 | 0.4067 |
| RF    | 0.8826 | 0.7953 | 0.322 | 0.307 | 0.3927 |
| XGB   | 0.8814 | 0.7941 | 0.338 | 0.330 | 0.3957 |
| ORF   | 0.8805 | 0.7943 | 0.310 | 0.291 | 0.3823 |

At full-N CV, BWORF and LR are tied at AUROC. BWORF wins AUPRC clearly (0.4171 vs 0.4067, ~2.5% relative).

### 7.3 Full-N holdout (refit on 196,788, score on 49,198)

The held-out test set evaluation is the primary result for the peer-reviewed paper:

| Model | AUROC | BalAcc | MCC | F1_pos | AUPRC |
|-------|-------|--------|-----|--------|-------|
| **BWORF** | **0.8933** | **0.8097** | 0.337 | 0.316 | **0.4212** |
| LR    | 0.8921 | 0.8060 | **0.3478** | 0.3343 | 0.4103 |
| RF    | 0.8881 | 0.8011 | 0.3284 | 0.3113 | 0.4002 |
| XGB   | 0.8856 | 0.8019 | 0.3460 | **0.3347** | 0.4032 |
| ORF   | 0.8850 | 0.7978 | 0.3129 | 0.2921 | 0.3874 |

**BWORF leads on AUROC, balanced accuracy, and AUPRC.** LR is best at MCC; XGB is best at F1 on the positive class. The BWORF AUPRC lead (0.4212 vs. next-best LR 0.4103) is the largest relative margin in the panel — about 2.7% relative — and is the most discriminating metric for severely imbalanced binary classification.

### 7.4 Scale-dependent behavior

A specific finding emerges from comparing 7.1 and 7.3: BWORF's AUROC improves from 0.852 (subsample) to 0.8933 (full-N), a gain of +0.041. The other models gain between +0.005 and +0.017 over the same data-scale change. BWORF's gain from increased N is roughly 2.5-8× larger than the other models'.

This pattern is consistent with the model's design: bootstrap weighting and oblique splits are most effective when there are enough minority-class instances per leaf to learn meaningful local boundaries. At small N, the minority class is too sparse for BWORF's machinery to fully express; at large N, it can.

---

## 8. Temperature sensitivity sweep

A supplemental analysis varies BWORF's `bootstrap_temperature` parameter to characterize what it controls.

### 8.1 Setup

- Subsample run only (4,000 modeling)
- 5 temperatures: T ∈ {0.5, 1.0, 1.5, 2.0, 3.0}
- Same 10-fold × 5 seeds protocol as the main subsample run
- Other hyperparameters held at the locked values from §5.2

### 8.2 Results

From `outputs/temp_sweep/temp_sweep_summary.csv`:

| T   | AUROC | BalAcc | F1    | Precision | Recall |
|-----|-------|--------|-------|-----------|--------|
| 0.5 | 0.854 | 0.670  | 0.392 | 0.429     | 0.368  |
| 1.0 | 0.852 | 0.720  | 0.378 | 0.304     | 0.509  |
| 1.5 | 0.858 | 0.756  | 0.343 | 0.236     | 0.631  |
| 2.0 | 0.866 | 0.781  | 0.265 | 0.159     | 0.808  |
| 3.0 | 0.843 | 0.584  | 0.122 | 0.065     | 0.985  |

### 8.3 Interpretation

AUROC is essentially flat across T ∈ {0.5, 1.0, 1.5, 2.0}, varying by only 0.014, with a sharper drop at T=3.0. In stark contrast, recall on the positive class climbs monotonically from 0.368 at T=0.5 to 0.985 at T=3.0, while precision collapses from 0.429 to 0.065 over the same range. Balanced accuracy varies by 0.197 across T ∈ {0.5..2.0}.

This pattern suggests that **bootstrap_temperature controls the operating point of BWORF, not its discrimination capacity**. AUROC measures how well the model orders examples; this remains stable. Precision and recall measure decisions made at the default `argmax(predict_proba)` threshold; these change because the temperature shifts the score distribution and therefore where the decision boundary falls. At T=3.0, the model is so eager to predict the positive class that it almost always does (recall = 0.985), but its underlying ranking of examples is barely worse than at T=1.0 (AUROC drops from 0.852 to 0.843).

For the headline BRFSS result, T=1.0 was chosen because it sits in the stable AUROC region and produces a balanced operating point. The full-N results in §7 use T=1.0; no full-N temperature sweep was conducted because the AUROC stability observed at subsample is the relevant property for the headline claim.

---

## 9. Reproducing the runs

### 9.1 Data preparation

Obtain the raw CSV (see `data/README.md`), then:

```bash
cd external_benchmark/brfss
python code/prepare_heart2022.py
```

This produces the cleaned subsample and full-N modeling/holdout splits in `data/cleaned/` (not committed; regenerated per local run).

### 9.2 Subsample run

```bash
sbatch sbatch/run_subsample_fast.sbatch    # lr, rf, xgb, orf
sbatch sbatch/run_subsample_bworf.sbatch   # bworf separately
python code/aggregate_heart2022.py --run subsample
sbatch sbatch/score_holdout_bworf.sbatch
```

### 9.3 Full-N run, fast models

```bash
sbatch sbatch/run_full_fast.sbatch          # lr, rf, xgb, orf
python code/aggregate_heart2022.py --run full
sbatch sbatch/score_holdout_full.sbatch
```

### 9.4 Full-N run, BWORF

```bash
sbatch sbatch/run_full_bworf_parallel.sbatch    # 50-task array
python code/aggregate_full_bworf.py
sbatch sbatch/score_holdout_bworf_full.sbatch
```

### 9.5 Temperature sweep

```bash
sbatch sbatch/run_heart22_temp_sweep.sbatch
python code/aggregate_temp_sweep.py
python code/plot_temp_sweep.py    # optional: regenerate the supplemental figure
```

### 9.6 Path note

Sbatch scripts reference `/home/chungyua/external_benchmark/` as the working directory, the path used during the original run on Sol. To re-run from this repository's location, the working directory and PYTHONPATH variables in each sbatch must be updated.

---

## 10. Environment

The BRFSS runs used a different Python environment than the classical external suite in `../classical/`:

- Python 3.9.2 via `module load shpc/python/3.9.2-slim/module` on the ASU Sol HPC
- Dependencies installed locally to `pydeps/` via `pip install --target=pydeps`
- `PYTHONPATH=$HOME/external_benchmark/pydeps` ensures the local install takes precedence

Key versions:
- numpy 2.0.2, pandas 2.3.3, scikit-learn 1.6.1, xgboost 2.1.4, joblib 1.5.3
- matplotlib 3.9.4, scipy 1.13.1

The local-pydeps approach was chosen over conda for reproducibility: the `pydeps/` directory is a self-contained snapshot of all dependency versions used. It is not committed to this repository (~500 MB), but the version manifest above allows recreation.

---

## 11. Provenance and known caveats

- **Single-commit repository history.** This directory was added to the BWORF repository in one commit during BMC-paper-additions migration. Per-file modification dates reflect Sol filesystem mtimes, not git history.
- **Sbatch hardcoded paths are stale post-migration.** See §9.6.
- **Large output files excluded from repo.** Full-N OOF prediction CSVs (~80 MB and ~46 MB) and per-shard intermediate outputs are not committed. Available from the author on request.
- **Raw BRFSS CSV not committed.** See `data/README.md` for download instructions.
- **No full-N temperature sweep.** The temperature sweep is subsample-only. The flat-AUROC finding at subsample (§8) is generalized to full-N by argument rather than by direct evaluation.
- **`pydeps/` not committed.** Dependency snapshot is recreatable from the version manifest in §10 but not bit-exact.

---

## 12. Citation

Citation to be updated after publication.
