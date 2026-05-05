#!/bin/bash
#SBATCH --job-name=holdout_full
#SBATCH --partition=public
#SBATCH --account=grp_gbmostaf
#SBATCH --qos=public
#SBATCH --output=logs/holdout_full_%j.out
#SBATCH --error=logs/holdout_full_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=4:00:00

module load shpc/python/3.9.2-slim/module
cd $HOME/external_benchmark
export PYTHONPATH=$HOME/external_benchmark/pydeps:$PYTHONPATH

python score_holdout_heart2022.py --modeling_csv data/cleaned/heart_attack_2022_brfss_full.csv.gz --holdout_csv data/cleaned/heart_attack_2022_brfss_full_holdout.csv.gz --models lr,rf,xgb,orf --out_dir outputs_heart2022_full --dataset heart_attack_2022_brfss_full
