#!/bin/bash
#SBATCH --job-name=heart22_sub_bworf
#SBATCH --partition=public
#SBATCH --account=grp_gbmostaf
#SBATCH --qos=public
#SBATCH --output=logs/heart22_sub_bworf_%A_%a.out
#SBATCH --error=logs/heart22_sub_bworf_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH --array=0-4

# 1 model x 5 seeds = 5 array tasks
# task_id 0..4 -> bworf with seeds 13,42,77,123,2025

SEEDS=(13 42 77 123 2025)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
MODEL=bworf

echo "Task ${SLURM_ARRAY_TASK_ID}: model=${MODEL} seed=${SEED}"
echo "Node: $(hostname)"
echo "Date: $(date)"

module load shpc/python/3.9.2-slim/module
cd $HOME/external_benchmark
export PYTHONPATH=$HOME/external_benchmark/pydeps:$PYTHONPATH

OUT_DIR="outputs_heart2022_subsample/${MODEL}/seed_${SEED}"

# stdout redirected to /dev/null because BWORF prints heavily during fit;
# stderr goes to a per-task log so we still see real errors.
time python benchmark_runner.py \
    --datasets heart_attack_2022_brfss \
    --models ${MODEL} \
    --seeds ${SEED} \
    --n_splits 10 \
    --n_jobs 1 \
    --bworf_n_estimators 100 \
    --bworf_max_depth 5 \
    --out_dir ${OUT_DIR} \
    > /dev/null 2>> logs/heart22_sub_bworf_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_stderr.log

echo "EXIT CODE: $?"
echo "Done: $(date)"
