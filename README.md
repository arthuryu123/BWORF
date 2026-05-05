# BWORF: Bootstrap-Weighted Oblique Random Forest

Repository for the master's thesis "Bootstrap-Weighted Oblique Random Forest" by Chung-Yuan Yu (Arizona State University) and a forthcoming peer-reviewed paper extension.

BWORF is a tree-based ensemble method for tabular classification that combines three ideas: oblique decision splits learned via L1-regularized logistic regression at each node, weighted bootstrap sampling to compensate for class imbalance, and class-balanced leaf-level probability estimation that maintains the imbalance handling consistently throughout the model. The method is evaluated on a primary multiclass DILI (Drug-Induced Liver Injury) classification task, four classical small biomedical tabular benchmarks, a population-scale modern benchmark on BRFSS 2022 heart-attack prediction, and a controlled mechanistic simulation study.

---

## Repository contents

This repository contains four self-contained sub-projects, each with its own README documenting that sub-project's protocol, hyperparameters, results, and reproduction instructions.

| Directory                                    | What it contains                                                                       |
|----------------------------------------------|----------------------------------------------------------------------------------------|
| [`dili_multiclass_patched_bworf/`](dili_multiclass_patched_bworf/) | Primary multiclass DILI benchmark (574 compounds, 8 model families, LIME interpretability) |
| [`external_benchmark/classical/`](external_benchmark/classical/)   | Four small classical biomedical tabular benchmarks (heart failure, diabetes, breast cancer, thyroid) |
| [`external_benchmark/brfss/`](external_benchmark/brfss/)           | BRFSS 2022 heart-attack benchmark (population-scale, severely imbalanced binary task)   |
| [`simulation_study/`](simulation_study/)     | Mechanistic simulation with known causal structure (252 scenarios × 3 methods)         |

The two external benchmark directories together form the external evaluation suite. The classical directory contains the four datasets used in the master's thesis; the BRFSS directory contains the population-scale dataset added for the peer-reviewed paper extension.

---

## The patched BWORF model

This repository uses a "patched" version of BWORF in which leaf-level prediction probabilities use the same class-balanced weighting as the rest of the model, instead of reverting to raw terminal-node class counts. The patch makes probability outputs consistent with the imbalance handling applied during bootstrap sampling and oblique split selection. Three of the four sub-projects use this patched version:

- `dili_multiclass_patched_bworf/code/bworf_with_mi.py`
- `external_benchmark/classical/code/bworf_with_mi.py`
- `external_benchmark/brfss/models/bworf_with_mi.py`

Each is a copy of the same patched implementation, retained at the sub-project level for self-contained reproducibility. A unified diff documenting the difference between the unpatched and patched versions is preserved at `external_benchmark/classical/bworf_with_mi_patch.diff`.

The simulation study (`simulation_study/`) uses the patched implementation under `simulation_study/models/bworf_with_mi.py`.

For the rationale behind the patch, see §3 of the dili README, the classical README, or the BRFSS README. The DILI README has the most detailed explanation.

---

## Quick start

Each sub-project is self-contained. To explore a specific result:

```bash
git clone https://github.com/arthuryu123/BWORF.git
cd BWORF/<sub_project_directory>
cat README.md
```

To regenerate any sub-project's results from scratch, follow the reproduction instructions in that sub-project's README. Be aware that:

- The DILI multiclass and classical external runs were performed with a `sim_orf` conda environment (Python 3.10).
- The BRFSS run was performed with a `module load shpc/python/3.9.2-slim/module` environment plus a local `pydeps/` directory (Python 3.9.2).
- The simulation study uses the same `sim_orf` environment as the DILI work.
- All runs targeted the ASU Sol HPC; SLURM submission scripts will need account/partition adjustments for other environments.

Specific dataset acquisition steps are documented in each sub-project's README. The BRFSS data must be downloaded from Kaggle (see `external_benchmark/brfss/data/README.md`); the others are committed directly.

---

## License

MIT License (see `LICENSE`).

---

## Citation

Citation to be updated after publication.
