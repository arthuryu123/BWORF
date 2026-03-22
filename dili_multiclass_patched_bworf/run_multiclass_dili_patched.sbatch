#!/bin/bash

#SBATCH --job-name=mc_dili_patch

#SBATCH --output=/home/chungyua/dili_multiclass_patched_bworf/logs/%x_%j.out

#SBATCH --error=/home/chungyua/dili_multiclass_patched_bworf/logs/%x_%j.err

#SBATCH --time=24:00:00

#SBATCH --cpus-per-task=8

#SBATCH --mem=32G



set -euo pipefail



MODE="${1:-smoke}"   # smoke or full



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

echo "MODE: $MODE"

echo "ROOT: $ROOT"

echo "PYTHON_BIN: $PYTHON_BIN"

echo "CPUS: ${SLURM_CPUS_PER_TASK:-8}"



module purge

module load mamba/latest



if [[ "$MODE" == "smoke" ]]; then

    "$PYTHON_BIN" "$RUNNER" --data_path "$DATA_PATH" --feature_names_path "$FEATURE_NAMES_PATH" --output_root "$OUTPUT_ROOT" --run_name smoke_multiclass_patch --smoke --n_jobs "${SLURM_CPUS_PER_TASK:-8}"

elif [[ "$MODE" == "full" ]]; then

    "$PYTHON_BIN" "$RUNNER" --data_path "$DATA_PATH" --feature_names_path "$FEATURE_NAMES_PATH" --output_root "$OUTPUT_ROOT" --run_name multiclass_dili_patched_main --n_splits 10 --seeds 42,43,44,45,46 --models rf,orf_style,lr,svm_rbf,xgb,naive_bayes,bworf_no_mi,bworf_mi --n_jobs "${SLURM_CPUS_PER_TASK:-8}"

else

    echo "Invalid MODE: $MODE"

    echo "Usage: sbatch run_multiclass_dili_patched.sbatch [smoke|full]"

    exit 1

fi
