#!/bin/bash
#SBATCH --job-name=holdout_bworf
#SBATCH --partition=public
#SBATCH --account=grp_gbmostaf
#SBATCH --qos=public
#SBATCH --output=logs/holdout_bworf_%j.out
#SBATCH --error=logs/holdout_bworf_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=2:00:00

module load shpc/python/3.9.2-slim/module
cd $HOME/external_benchmark
export PYTHONPATH=$HOME/external_benchmark/pydeps:$PYTHONPATH

python score_holdout_heart2022.py --modeling_csv data/cleaned/heart_attack_2022_brfss.csv.gz --holdout_csv data/cleaned/heart_attack_2022_brfss_holdout.csv.gz --models bworf --out_dir outputs_heart2022_subsample --dataset heart_attack_2022_brfss
