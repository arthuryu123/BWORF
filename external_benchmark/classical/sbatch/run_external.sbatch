#!/bin/bash

#SBATCH --job-name=externalbw

#SBATCH --output=/home/chungyua/patch_external/logs/%x_%j.out

#SBATCH --error=/home/chungyua/patch_external/logs/%x_%j.err

#SBATCH --time=08:00:00

#SBATCH --cpus-per-task=8

#SBATCH --mem=24G



set -euo pipefail



DATASET="${1:?Need dataset name}"

MODEL="${2:?Need model name}"



ROOT="/home/chungyua/patch_external"

cd "$ROOT"



echo "HOST: $(hostname)"

echo "DATE: $(date)"

echo "DATASET: ${DATASET}"

echo "MODEL: ${MODEL}"

echo "CPUS: ${SLURM_CPUS_PER_TASK:-8}"



module purge

module load mamba/latest



~/.conda/envs/sim_orf/bin/python code/benchmark_runner_external.py --data_dir datasets --out_dir outputs --datasets "$DATASET" --models "$MODEL" --n_splits 10 --seeds 42,43,44,45,46 --n_jobs "${SLURM_CPUS_PER_TASK:-8}"
