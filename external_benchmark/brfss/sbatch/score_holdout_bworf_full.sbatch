#!/bin/bash
#SBATCH --job-name=holdout_bworf_full
#SBATCH --partition=public
#SBATCH --account=grp_gbmostaf
#SBATCH --qos=public
#SBATCH --output=logs/holdout_bworf_full_%A_%a.out
#SBATCH --error=logs/holdout_bworf_full_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-4

SEEDS=(13 42 77 123 2025)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "Task ${SLURM_ARRAY_TASK_ID}: seed=${SEED}"
echo "Node: $(hostname)"
echo "Date: $(date)"

module load shpc/python/3.9.2-slim/module
cd $HOME/external_benchmark
export PYTHONPATH=$HOME/external_benchmark/pydeps:$PYTHONPATH

time python score_holdout_heart2022.py --modeling_csv data/cleaned/heart_attack_2022_brfss_full.csv.gz --holdout_csv data/cleaned/heart_attack_2022_brfss_full_holdout.csv.gz --models bworf_parallel --seeds ${SEED} --out_dir outputs_heart2022_full_bworf_holdout/seed_${SEED} --dataset heart_attack_2022_brfss_full

echo "EXIT CODE: $?"
echo "Done: $(date)"
