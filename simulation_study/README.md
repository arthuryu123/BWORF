# Mechanistic Simulation Study

This directory contains the synthetic-data simulation study from Chapter 3 of an in-progress master's thesis on Bootstrap-Weighted Oblique Random Forest (BWORF) by Chung-Yuan Yu (Arizona State University). The study evaluates whether BWORF recovers causal features under known generative structure, controlling for heritability, sample size, number of causal features, between-feature correlation, and signal-to-noise ratio.

The simulation uses a GWAS-flavored synthetic data generator: features mimic genotype-style allele counts, the response is a binary case/control endpoint generated from a small set of causal features under controlled heritability, and between-feature correlation is induced through block-structured covariance to mimic linkage-disequilibrium-style dependencies. The framing is mechanistic, not biological: the goal is to test BWORF on data with known causal structure and tunable difficulty, not to claim genomic application.

This README documents the version history, the canonical v4 run, the simulation grid, and how to reproduce it.

---

## 1. What is in this directory

```
simulation_study/
├── README.md                       this file
├── models/
│   ├── bworf_with_mi.py           PATCHED BWORF model (shared across all versions)
│   └── orf_treeple.py             ORF wrapper using the treeple library
├── v1/                             initial version (script only, no preserved outputs)
├── v2/                             second iteration (script + analysis only, no outputs)
├── v3/                             third iteration with outputs (precursor to v4)
└── v4/                             CANONICAL thesis-cited version (see §3)
    ├── run_simulation_study_v4.py
    ├── v4_full.sbatch              single-task SLURM wrapper
    ├── v4_sharded.sbatch           4-task array SLURM wrapper (used for production)
    ├── outputs/                    initial v4 run (March 8, 2026), pre-patch
    ├── outputs_patched/            re-run with patched BWORF (March 14-15, 2026), CANONICAL
    └── logs/                       SLURM log files
```

The shared `models/` directory at the top level holds the BWORF and ORF implementations imported by every version's runner.

---

## 2. Version history

The simulation went through four iterations during thesis development. All four are preserved in the repository for provenance, but only v4 is cited in the thesis.

| Version | Date         | Status       | Description                                                        |
|---------|--------------|--------------|--------------------------------------------------------------------|
| v1      | January 2026 | exploratory  | Initial implementation; runner and utility scripts only            |
| v2      | January 2026 | exploratory  | Refactored runner with separate analysis script                    |
| v3      | February 2026 | exploratory | First version with full SLURM run and outputs; imbalance addon     |
| v4      | March 2026   | **canonical** | Sharded production run; re-executed with patched BWORF (§4)       |

Versions v1-v3 are not used in the thesis. They are retained because they document the development trajectory and because v3's parameter grid informed v4's design.

### 2.1 Why two output directories under v4?

v4 was executed twice:

- **`v4/outputs/`** (March 8, 2026): initial single-task run using the original (unpatched) BWORF implementation.
- **`v4/outputs_patched/`** (March 14-15, 2026): production sharded run using the patched BWORF implementation that became the canonical model for the thesis.

The thesis cites `v4/outputs_patched/` exclusively. The earlier unpatched run is retained as a reference point against which the patch's effect can be measured if needed.

For the meaning of "patched," see `../dili_multiclass_patched_bworf/README.md` §3 or `../external_benchmark/classical/README.md` §3. Briefly: the patched implementation makes leaf-level predicted probabilities use the same class-balanced weighting as the rest of the model, instead of reverting to raw terminal-node class counts.

---

## 3. The canonical v4 run

### 3.1 Methods

Three methods are compared in the canonical run, intentionally restricted to the conceptual progression from axis-aligned to oblique to bootstrap-weighted oblique:

1. **`rf`** — random forest (axis-aligned, scikit-learn baseline)
2. **`orf`** — oblique random forest from the treeple library
3. **`bworf`** — patched BWORF with weighted bootstrap enabled, no MI feature filtering

This three-method panel isolates two specific effects: the gain from oblique splits (RF → ORF) and the additional gain from bootstrap weighting on top of oblique splits (ORF → BWORF).

### 3.2 Scenario grid

The canonical run sweeps a five-dimensional grid of generative parameters:

| Parameter        | Values                                  | Meaning                                                       |
|------------------|----------------------------------------|---------------------------------------------------------------|
| `h2`             | 0.005, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2 | heritability (variance of response explained by causal features) |
| `n_samples`      | 500, 2500                              | sample size                                                   |
| `m`              | 2, 5, 10                               | number of causal features                                     |
| `rho`            | 0.2, 0.8                               | between-feature correlation within blocks                     |
| `signal_ratio`   | inf, 1.0, 0.5                          | signal-to-noise ratio (inf = no noise; 1.0/0.5 = noisy)       |

Held constant across all scenarios:

- `p = 1000` total features
- `maf_type = fixed`, `maf_fixed = 0.3` (uniform minor allele frequency)
- `case_control_ratio = 1.0` (balanced binary outcome)
- `n_blocks = 50` (correlation block structure inducing approximate LD)

The full Cartesian product yields:

> 7 (h2) × 2 (n) × 3 (m) × 2 (rho) × 3 (signal_ratio) = **252 scenarios**

### 3.3 Replication and evaluation

For each scenario × method combination:

| Aspect                  | Setting                                                                              |
|------------------------|--------------------------------------------------------------------------------------|
| Holdout seeds           | 5 fixed: `42, 0, 12, 100, 90`                                                        |
| Replicates per seed     | 5                                                                                    |
| Test split              | 30% held out for evaluation                                                          |
| Master simulation seed  | 12345 (controls the data-generation RNG)                                             |
| Threshold-free metrics  | AUC, Brier score                                                                     |
| Operational metrics     | accuracy, balanced accuracy                                                          |
| Feature recovery        | mean and median rank of true causal features in the model's importance ranking       |
| Top-k recovery          | recall@20 (proportion of true causal features within the top 20 ranked features)     |

Total result rows for the canonical run:

> 252 scenarios × 3 methods × 5 holdout seeds × 5 replicates = **18,900 rows**

Verified against `outputs_patched/simulation_results_v4.csv` (18,900 data rows + 1 header = 18,901 lines total).

### 3.4 Model hyperparameters in the canonical run

| Method | Parameter            | Value |
|--------|----------------------|-------|
| `rf`   | `n_estimators`       | 100   |
| `orf`  | `n_estimators`       | 100   |
| `bworf`| `n_estimators`       | 100   |
| `bworf`| `l1_strength`        | 1.0   |
| `bworf`| `weighted_bootstrap` | True  |

Other BWORF parameters use the patched implementation's defaults. Notably, MI feature filtering is **not** used in the simulation study (`bworf` here is closer to the `bworf_no_mi` configuration in the classical external benchmarks).

### 3.5 Permutation importance

Feature importance for ranking-based metrics is computed via permutation importance with these settings:

| Parameter         | Value      |
|-------------------|------------|
| `n_repeats`       | 1          |
| `subsample_n`     | 500        |
| `scoring`         | `roc_auc`  |
| `topk`            | 20         |

The single-repeat setting is chosen for computational tractability across 18,900 trials; the resulting per-trial rankings are noisier than a typical multi-repeat permutation importance, but the noise averages out across 25 (5 holdout × 5 replicate) trials per (scenario, method) cell.

---

## 4. Outputs

### 4.1 Canonical `outputs_patched/`

Contains 4 shard CSVs and the merged total:

| File                                  | Rows  | Description                                  |
|---------------------------------------|-------|----------------------------------------------|
| `simulation_results_v4_shard0.csv`   | 4,725 | scenarios 0-62                                |
| `simulation_results_v4_shard1.csv`   | 4,725 | scenarios 63-125                              |
| `simulation_results_v4_shard2.csv`   | 4,725 | scenarios 126-188                             |
| `simulation_results_v4_shard3.csv`   | 4,725 | scenarios 189-251                             |
| `simulation_results_v4.csv`          | 18,900 | concatenation of all four shards              |
| `run_config_v4_shard<i>.json`        | -     | per-shard configuration metadata              |

Each shard processes 63 of the 252 scenarios (252 / 4 = 63), with 75 result rows per scenario (3 methods × 5 holdout seeds × 5 replicates).

### 4.2 Result schema

Per-row columns in `simulation_results_v4.csv`:

| Column                            | Description                                                     |
|-----------------------------------|-----------------------------------------------------------------|
| `scenario_id`                     | hash-derived scenario identifier                                |
| `scenario_json`                   | full scenario specification as JSON                             |
| `method`                          | `rf`, `orf`, or `bworf`                                         |
| `seed`                            | holdout seed                                                    |
| `replicate`                       | replicate index (0-4)                                           |
| `n_samples`, `p`, `m`, `h2`, ...  | scenario parameters (also embedded in `scenario_json`)          |
| `auc`, `brier`                    | threshold-free metrics                                          |
| `accuracy`, `balanced_accuracy`   | argmax-decision metrics                                         |
| `mean_rank_causal`, `median_rank_causal` | ranking position of true causal features                  |
| `recall_at_20`                    | fraction of causal features in top-20 by permutation importance |
| `fit_seconds`, `perm_seconds`     | timing                                                          |
| `status`, `error_msg`             | row-level error flag (rare; populated only when a trial fails)  |

### 4.3 The earlier `outputs/` directory

Contains the single-task pre-patch run:

| File                                | Rows  | Description                                      |
|-------------------------------------|-------|--------------------------------------------------|
| `simulation_results_v4.csv`         | 18,900 | unpatched-BWORF run from March 8, 2026           |
| `run_config_v4.json`                | -     | configuration for the unpatched run              |

This directory is not the thesis-cited result. It exists for comparison with the canonical patched run.

---

## 5. Reproducing the canonical run

The canonical run is submitted as a 4-task SLURM array on a private partition with exclusive node access:

```bash
sbatch v4_sharded.sbatch
```

This issues 4 array tasks (`SLURM_ARRAY_TASK_ID = 0..3`), each running:

```bash
python run_simulation_study_v4.py --n-cores <SLURM_CPUS_ON_NODE> --shard-id <i> --n-shards 4
```

Each shard processes its 63 scenarios using all available CPU cores on the allocated node, then writes its `simulation_results_v4_shard<i>.csv` and `run_config_v4_shard<i>.json` into `outputs_patched/`. After all 4 shards complete, the runner produces the concatenated `simulation_results_v4.csv` for downstream analysis.

### 5.1 Resource and partition note

The committed `v4_sharded.sbatch` requests:

- `--partition=general` and `--qos=private` (account-specific access not available to public reviewers)
- `--exclusive --mem=0 --time=7-00:00:00` (full-node exclusive access for up to 7 days)

These settings reflect the original run on the ASU Sol HPC and would need adjustment for any other compute environment.

### 5.2 Python environment

The simulation uses the `sim_orf` conda environment from the original Sol filesystem:

- Python via `~/.conda/envs/sim_orf/bin/python`
- scikit-learn 1.6.1
- treeple library (for the ORF comparator)
- standard numpy, pandas, scipy

Environment threading variables are explicitly set inside the sbatch to `1` for OMP, OPENBLAS, MKL, NUMEXPR, and VECLIB to prevent nested parallelism conflicts with joblib.

---

## 6. Earlier versions (v1, v2, v3)

These are retained for provenance but are not part of the thesis-cited results.

### 6.1 v1 (`v1/`)

| File                       | Description                                |
|---------------------------|--------------------------------------------|
| `run_simulation_study_v1.py` | initial monolithic runner                |
| `simulation_utils_v1.py`   | utility functions split out from the runner |

No outputs preserved.

### 6.2 v2 (`v2/`)

| File                                | Description                                |
|------------------------------------|--------------------------------------------|
| `run_simulation_study_v2.py`       | refactored runner                          |
| `analyze_simulation_results_v2.py` | analysis script for v2 outputs             |

No outputs preserved.

### 6.3 v3 (`v3/`)

| Path                                        | Description                                     |
|--------------------------------------------|-------------------------------------------------|
| `v3/run_simulation_study_v3.py`            | base v3 runner                                  |
| `v3/run_simulation_study_v3_imbalance_addon.py` | extension with class-imbalance scenarios   |
| `v3/v3_full.sbatch`                        | single-task SLURM wrapper                       |
| `v3/outputs/`                              | results from a Feb 12 run                       |
| `v3/logs/`                                 | SLURM logs from the v3 run                      |

v3 introduced the imbalance scenarios and the SLURM-based execution model that v4 inherited. The v3 grid is similar in structure to v4 but covers a smaller set of scenarios.

---

## 7. Provenance and known caveats

- **Single-commit repository history.** This directory was committed in one commit during initial GitHub migration. Per-file modification dates reflect Sol filesystem mtimes, not git history.
- **Sbatch hardcoded paths are stale post-rename.** `v4/v4_sharded.sbatch` and `v4/v4_full.sbatch` reference `/home/chungyua/simulation_study/v4/`. The paths would need updating to re-run from this repository's location.
- **Private QoS.** The canonical run used `--qos=private`, which is account-specific. Reviewers without that access cannot reproduce the run with identical resource shape; the standard `public` partition with adjusted walltime would still produce the same numerical result.
- **Two parallel output sets in v4.** `outputs/` (unpatched, single-task) and `outputs_patched/` (patched, sharded) coexist. Only `outputs_patched/` is canonical (§3).
- **Single permutation-importance repeat.** Per-trial feature rankings are computed with `n_repeats=1` for tractability across 18,900 trials. Per-cell averaging across 25 trials reduces but does not eliminate the per-trial ranking noise.
- **No analysis or plotting scripts in v4.** The thesis figures derived from this run were produced by separate analysis code not preserved in this directory.

---

## 8. Citation

Citation to be updated after publication.
