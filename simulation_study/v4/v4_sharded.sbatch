#!/bin/bash
#SBATCH --job-name=sim_v4_shard
#SBATCH --account=grp_gbmostaf
#SBATCH --partition=general
#SBATCH --qos=private
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=7-00:00:00
#SBATCH --array=0-3
#SBATCH --output=/home/chungyua/simulation_study/v4/logs/%x_%A_%a.out
#SBATCH --error=/home/chungyua/simulation_study/v4/logs/%x_%A_%a.err

set -euo pipefail

cd /home/chungyua/simulation_study/v4
mkdir -p logs outputs_patched

PY="$HOME/.conda/envs/sim_orf/bin/python"

# Prevent nested threading / oversubscription
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# Use the whole allocated node when possible
NCORES="${SLURM_CPUS_ON_NODE:-${SLURM_CPUS_PER_TASK:-1}}"
SHARD_ID="${SLURM_ARRAY_TASK_ID}"
N_SHARDS=4

echo "=============================================="
echo "Job ID:        ${SLURM_JOB_ID:-NA}"
echo "Array Job ID:  ${SLURM_ARRAY_JOB_ID:-NA}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID:-NA}"
echo "Job name:      ${SLURM_JOB_NAME:-NA}"
echo "Node:          $(hostname)"
echo "Start time:    $(date)"
echo "Work dir:      $(pwd)"
echo "Account:       ${SLURM_JOB_ACCOUNT:-NA}"
echo "Partition:     ${SLURM_JOB_PARTITION:-NA}"
echo "QOS:           ${SLURM_JOB_QOS:-NA}"
echo "CPUs on node:  ${SLURM_CPUS_ON_NODE:-NA}"
echo "CPUs per task: ${SLURM_CPUS_PER_TASK:-NA}"
echo "Memory/node:   ${SLURM_MEM_PER_NODE:-NA}"
echo "NCORES used:   ${NCORES}"
echo "Shard:         ${SHARD_ID}/${N_SHARDS}"
echo "Python:        $PY"
echo "=============================================="

"$PY" run_simulation_study_v4.py --n-cores "$NCORES" --shard-id "$SHARD_ID" --n-shards "$N_SHARDS"

echo "=============================================="
echo "End time:      $(date)"
echo "Done."
echo "=============================================="
