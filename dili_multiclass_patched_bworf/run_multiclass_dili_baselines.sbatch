#!/bin/bash

#SBATCH --account=grp_gbmostaf

#SBATCH --qos=public

#SBATCH --partition=public

#SBATCH --job-name=mc_dili_base

#SBATCH --output=/home/chungyua/dili_multiclass_patched_bworf/logs/%x_%j.out

#SBATCH --error=/home/chungyua/dili_multiclass_patched_bworf/logs/%x_%j.err

#SBATCH --time=24:00:00

#SBATCH --cpus-per-task=32

#SBATCH --mem=64G



set -euo pipefail



ROOT="/home/chungyua/dili_multiclass_patched_bworf"

PYTHON_BIN="$HOME/.conda/envs/sim_orf/bin/python"

RUNNER="$ROOT/code/run_multiclass_dili_patched_benchmark.py"

DATA_PATH="$ROOT/data/dili_multiclass.csv"

FEATURE_NAMES_PATH="$ROOT/data/dili_multiclass_feature_names.csv"

OUTPUT_ROOT="$ROOT/outputs"



mkdir -p "$ROOT/logs" "$ROOT/outputs"

cd "$ROOT"



echo "HOST: $(hostname)"

echo "DATE: $(date)"

echo "GROUP: baselines"

echo "ROOT: $ROOT"

echo "PYTHON_BIN: $PYTHON_BIN"

echo "CPUS: ${SLURM_CPUS_PER_TASK:-32}"



module purge

module load mamba/latest



"$PYTHON_BIN" "$RUNNER" --data_path "$DATA_PATH" --feature_names_path "$FEATURE_NAMES_PATH" --output_root "$OUTPUT_ROOT" --run_name multiclass_dili_baselines --n_splits 10 --seeds 42,43,44,45,46 --models rf,lr,svm_rbf,xgb,naive_bayes --n_jobs "${SLURM_CPUS_PER_TASK:-32}"
