#!/bin/bash
#SBATCH --job-name=heart22_temp_sweep
#SBATCH --partition=public
#SBATCH --account=grp_gbmostaf
#SBATCH --qos=public
#SBATCH --output=logs/heart22_temp_sweep_%A_%a.out
#SBATCH --error=logs/heart22_temp_sweep_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH --array=0-19

# BWORF bootstrap_temperature sensitivity analysis on BRFSS subsample.
# 4 temperatures (T=0.5, 1.5, 2.0, 3.0) x 5 seeds = 20 array tasks.
# T=1.0 is already in outputs_heart2022_subsample/ from the main run.
#
# task_id 0-4   -> T=0.5  with seeds 13,42,77,123,2025
# task_id 5-9   -> T=1.5
# task_id 10-14 -> T=2.0
# task_id 15-19 -> T=3.0
#
# Side experiment / sensitivity analysis. Main table results use T=1.0 only.

TEMPS=(0.5 1.5 2.0 3.0)
SEEDS=(13 42 77 123 2025)
TEMP_IDX=$((SLURM_ARRAY_TASK_ID / 5))
SEED_IDX=$((SLURM_ARRAY_TASK_ID % 5))
TEMP=${TEMPS[$TEMP_IDX]}
SEED=${SEEDS[$SEED_IDX]}

echo "Task ${SLURM_ARRAY_TASK_ID}: T=${TEMP} seed=${SEED}"
echo "Node: $(hostname)"
echo "Date: $(date)"

module load shpc/python/3.9.2-slim/module
cd $HOME/external_benchmark
export PYTHONPATH=$HOME/external_benchmark/pydeps:$PYTHONPATH

OUT_DIR="outputs_heart2022_subsample_temp_sweep/T${TEMP}/seed_${SEED}"

time python benchmark_runner.py --datasets heart_attack_2022_brfss --models bworf --seeds ${SEED} --n_splits 10 --n_jobs 1 --bworf_n_estimators 100 --bworf_max_depth 5 --bworf_bootstrap_temperature ${TEMP} --out_dir ${OUT_DIR} > /dev/null 2>> logs/heart22_temp_sweep_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_stderr.log

echo "EXIT CODE: $?"
echo "Done: $(date)"
