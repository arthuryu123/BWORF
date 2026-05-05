#!/bin/bash

#SBATCH --job-name=patchcheck

#SBATCH --output=/home/chungyua/patch_check_heart_thyroid/logs/%x_%j.out

#SBATCH --error=/home/chungyua/patch_check_heart_thyroid/logs/%x_%j.err

#SBATCH --time=04:00:00

#SBATCH --cpus-per-task=8

#SBATCH --mem=24G



set -euo pipefail



DATASET="${1:?Need dataset name}"

MODEL="${2:?Need model name}"



ROOT="/home/chungyua/patch_check_heart_thyroid"

cd "$ROOT"



echo "HOST: $(hostname)"

echo "DATE: $(date)"

echo "DATASET: ${DATASET}"

echo "MODEL: ${MODEL}"

echo "CPUS: ${SLURM_CPUS_PER_TASK:-8}"



module purge

module load mamba/latest



~/.conda/envs/sim_orf/bin/python code/benchmark_runner_patchcheck.py --data_dir datasets --out_dir outputs --datasets "$DATASET" --models "$MODEL" --n_splits 5 --seeds 42,43,44 --n_jobs "${SLURM_CPUS_PER_TASK:-8}"

