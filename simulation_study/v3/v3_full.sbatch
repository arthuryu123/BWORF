#!/bin/bash

#SBATCH --job-name=sim_v3_imb

#SBATCH --output=logs/%x_%j.out

#SBATCH --error=logs/%x_%j.err

#SBATCH --time=24:00:00

#SBATCH --cpus-per-task=16

#SBATCH --mem=128G

#SBATCH --partition=public

#SBATCH --qos=class

#SBATCH --account=class_lsc519fall2025b



set -euo pipefail

mkdir -p logs

cd ~/simulation_study/v3



PY="$HOME/.conda/envs/sim_orf/bin/python"



export OMP_NUM_THREADS=1

export OPENBLAS_NUM_THREADS=1

export MKL_NUM_THREADS=1

export NUMEXPR_NUM_THREADS=1

export VECLIB_MAXIMUM_THREADS=1



"$PY" run_simulation_study_v3_imbalance_addon.py --shard-id 0 --n-shards 1


