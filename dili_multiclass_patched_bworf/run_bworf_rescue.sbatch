#!/bin/bash

#SBATCH --account=grp_gbmostaf

#SBATCH --qos=public

#SBATCH --partition=public

#SBATCH --job-name=bworf_rescue

#SBATCH --output=/home/chungyua/dili_multiclass_patched_bworf/logs/%x_%j.out

#SBATCH --error=/home/chungyua/dili_multiclass_patched_bworf/logs/%x_%j.err

#SBATCH --time=36:00:00

#SBATCH --cpus-per-task=64

#SBATCH --mem=128G



set -euo pipefail



RUN_NAME="${1:?Need run name}"

MODEL="${2:?Need model name}"

MI_TOP_K="${3:?Need mi_top_k (use 0 for no_mi)}"

L1="${4:-0.05}"

NTRIES="${5:-20}"

DEPTH="${6:-8}"

ESTIMATORS="${7:-300}"

SCALE_MODE="${8:-standard}"



ROOT="/home/chungyua/dili_multiclass_patched_bworf"

PYTHON_BIN="$HOME/.conda/envs/sim_orf/bin/python"

RUNNER="$ROOT/code/run_multiclass_dili_patched_benchmark_v2.py"

DATA_PATH="$ROOT/data/dili_multiclass.csv"

FEATURE_NAMES_PATH="$ROOT/data/dili_multiclass_feature_names.csv"

OUTPUT_ROOT="$ROOT/outputs"



mkdir -p "$ROOT/logs" "$ROOT/outputs"

cd "$ROOT"



echo "HOST: $(hostname)"

echo "DATE: $(date)"

echo "RUN_NAME: $RUN_NAME"

echo "MODEL: $MODEL"

echo "MI_TOP_K: $MI_TOP_K"

echo "L1: $L1"

echo "NTRIES: $NTRIES"

echo "DEPTH: $DEPTH"

echo "ESTIMATORS: $ESTIMATORS"

echo "SCALE_MODE: $SCALE_MODE"

echo "CPUS: ${SLURM_CPUS_PER_TASK:-64}"



module purge

module load mamba/latest



"$PYTHON_BIN" "$RUNNER" --data_path "$DATA_PATH" --feature_names_path "$FEATURE_NAMES_PATH" --output_root "$OUTPUT_ROOT" --run_name "$RUN_NAME" --n_splits 10 --seeds 42,43,44,45,46 --models "$MODEL" --mi_top_k "$MI_TOP_K" --oblique_n_estimators "$ESTIMATORS" --oblique_max_depth "$DEPTH" --oblique_min_samples_split 10 --oblique_min_samples_leaf 1 --oblique_l1_strength "$L1" --oblique_n_tries "$NTRIES" --oblique_bootstrap_temperature 1.0 --oblique_scale_mode "$SCALE_MODE" --n_jobs "${SLURM_CPUS_PER_TASK:-64}"
