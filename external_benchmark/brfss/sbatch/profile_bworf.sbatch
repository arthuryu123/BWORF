#!/bin/bash
#SBATCH --job-name=profile_bworf
#SBATCH --partition=htc
#SBATCH --account=grp_gbmostaf
#SBATCH --qos=public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=logs/profile_bworf_%j.out
#SBATCH --error=logs/profile_bworf_%j.err

module load shpc/python/3.9.2-slim/module
cd $HOME/external_benchmark
export PYTHONPATH=$HOME/external_benchmark/pydeps:$PYTHONPATH
python -u profile_bworf.py
