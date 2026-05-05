#!/bin/bash
#SBATCH --job-name=heart22_sub_fast
#SBATCH --partition=htc
#SBATCH --account=grp_gbmostaf
#SBATCH --qos=public
#SBATCH --output=logs/heart22_sub_fast_%A_%a.out
#SBATCH --error=logs/heart22_sub_fast_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --array=0-19

# 4 models x 5 seeds = 20 array tasks
# task_id 0-4   -> lr  with seeds 13,42,77,123,2025
# task_id 5-9   -> rf
# task_id 10-14 -> xgb
# task_id 15-19 -> orf

MODELS=(lr rf xgb orf)
SEEDS=(13 42 77 123 2025)
MODEL_IDX=$((SLURM_ARRAY_TASK_ID / 5))
SEED_IDX=$((SLURM_ARRAY_TASK_ID % 5))
MODEL=${MODELS[$MODEL_IDX]}
SEED=${SEEDS[$SEED_IDX]}

echo "Task ${SLURM_ARRAY_TASK_ID}: model=${MODEL} seed=${SEED}"
echo "Node: $(hostname)"
echo "Date: $(date)"

module load shpc/python/3.9.2-slim/module
cd $HOME/external_benchmark
export PYTHONPATH=$HOME/external_benchmark/pydeps:$PYTHONPATH

OUT_DIR="outputs_heart2022_subsample/${MODEL}/seed_${SEED}"

time python benchmark_runner.py \
    --datasets heart_attack_2022_brfss \
    --models ${MODEL} \
    --seeds ${SEED} \
    --n_splits 10 \
    --n_jobs ${SLURM_CPUS_PER_TASK} \
    --rf_n_estimators 100 \
    --xgb_n_estimators 100 \
    --orf_n_estimators 100 \
    --out_dir ${OUT_DIR} \
    > /dev/null 2>> logs/heart22_sub_fast_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_stderr.log

echo "EXIT CODE: $?"
echo "Done: $(date)"
