#!/bin/bash
#SBATCH --job-name=heart22_full_bworf
#SBATCH --partition=public
#SBATCH --account=grp_gbmostaf
#SBATCH --qos=public
#SBATCH --output=logs/heart22_full_bworf_%A_%a.out
#SBATCH --error=logs/heart22_full_bworf_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --array=0-49

# 5 seeds x 10 folds = 50 array tasks
# task_id =  seed_idx * 10 + fold_id
#   seed_idx 0..4 -> seeds [13, 42, 77, 123, 2025]
#   fold_id  0..9 -> folds [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
#
# Each task runs ONE fold of ONE seed of full-N parallel BWORF.
# Wallclock estimate per task: ~13-27 hours at 32 CPUs.

SEEDS=(13 42 77 123 2025)
SEED_IDX=$((SLURM_ARRAY_TASK_ID / 10))
FOLD_ID=$((SLURM_ARRAY_TASK_ID % 10))
SEED=${SEEDS[$SEED_IDX]}

echo "Task ${SLURM_ARRAY_TASK_ID}: seed=${SEED} fold=${FOLD_ID}"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "CPUs allocated: ${SLURM_CPUS_PER_TASK}"

module load shpc/python/3.9.2-slim/module
cd $HOME/external_benchmark
export PYTHONPATH=$HOME/external_benchmark/pydeps:$PYTHONPATH

# Each (seed, fold) shard gets its own output directory so they don't collide
OUT_DIR="outputs_heart2022_full_bworf/seed_${SEED}_fold_${FOLD_ID}"

time python benchmark_runner.py --datasets heart_attack_2022_brfss_full --models bworf_parallel --seeds ${SEED} --n_splits 10 --fold_ids ${FOLD_ID} --drop_incomplete_oof True --n_jobs 1 --bworf_n_estimators 100 --bworf_max_depth 5 --bworf_parallel_n_jobs ${SLURM_CPUS_PER_TASK} --out_dir ${OUT_DIR} > /dev/null 2>> logs/heart22_full_bworf_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_stderr.log

echo "EXIT CODE: $?"
echo "Done: $(date)"
