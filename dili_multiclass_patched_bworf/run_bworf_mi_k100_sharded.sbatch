#!/bin/bash

#SBATCH --account=grp_gbmostaf

#SBATCH --qos=public

#SBATCH --partition=public

#SBATCH --job-name=bworf_mi_k100

#SBATCH --output=/home/chungyua/dili_multiclass_patched_bworf/logs/%x_%A_%a.out

#SBATCH --error=/home/chungyua/dili_multiclass_patched_bworf/logs/%x_%A_%a.err

#SBATCH --time=1-00:00:00

#SBATCH --cpus-per-task=2

#SBATCH --mem=8G

#SBATCH --array=0-49%20



set -euo pipefail



ROOT="/home/chungyua/dili_multiclass_patched_bworf"

PYTHON_BIN="$HOME/.conda/envs/sim_orf/bin/python"

RUNNER="$ROOT/code/run_multiclass_dili_shard.py"

DATA_PATH="$ROOT/data/dili_multiclass.csv"

FEATURE_NAMES_PATH="$ROOT/data/dili_multiclass_feature_names.csv"

OUTPUT_ROOT="$ROOT/outputs"



RUN_NAME="bworf_mi_k100_l005_t20_d8_e300"



mkdir -p "$ROOT/logs" "$ROOT/outputs"

cd "$ROOT"



echo "HOST: $(hostname)"

echo "DATE: $(date)"

echo "JOBID: ${SLURM_JOB_ID}"

echo "TASK_ID: ${SLURM_ARRAY_TASK_ID}"

echo "RUN_NAME: ${RUN_NAME}"

echo "CPUS: ${SLURM_CPUS_PER_TASK:-2}"



module purge

module load mamba/latest



"$PYTHON_BIN" "$RUNNER" --data_path "$DATA_PATH" --feature_names_path "$FEATURE_NAMES_PATH" --output_root "$OUTPUT_ROOT" --run_name "$RUN_NAME" --n_splits 10 --seeds 42,43,44,45,46 --model bworf_mi --task_id "${SLURM_ARRAY_TASK_ID}" --mi_top_k 100 --oblique_n_estimators 300 --oblique_max_depth 8 --oblique_min_samples_split 10 --oblique_min_samples_leaf 1 --oblique_l1_strength 0.05 --oblique_n_tries 20 --oblique_bootstrap_temperature 1.0 --oblique_scale_mode standard --n_jobs "${SLURM_CPUS_PER_TASK:-2}"
