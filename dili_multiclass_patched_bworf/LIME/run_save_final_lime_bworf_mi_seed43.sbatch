#!/bin/bash
#SBATCH --account=grp_gbmostaf
#SBATCH --qos=public
#SBATCH --partition=public
#SBATCH --job-name=lime_refit_mc43
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: sbatch $0 <PROJECT_ROOT> <PYTHON_SCRIPT> [OUTPUT_DIR]"
  exit 1
fi

PROJECT_ROOT="$1"
PY_SCRIPT="$2"
OUTPUT_DIR="${3:-$PROJECT_ROOT/lime_artifacts/bworf_mi_seed43}"
PYTHON_BIN="$HOME/.conda/envs/sim_orf/bin/python"

module purge
module load mamba/latest

mkdir -p "$OUTPUT_DIR"
cd "$PROJECT_ROOT"

echo "HOST: $(hostname)"
echo "DATE: $(date)"
echo "PROJECT_ROOT: $PROJECT_ROOT"
echo "PYTHON_BIN: $PYTHON_BIN"
echo "OUTPUT_DIR: $OUTPUT_DIR"

"$PYTHON_BIN" "$PY_SCRIPT" \
  --project_root "$PROJECT_ROOT" \
  --seed 43 \
  --mi_top_k 50 \
  --output_dir "$OUTPUT_DIR"
