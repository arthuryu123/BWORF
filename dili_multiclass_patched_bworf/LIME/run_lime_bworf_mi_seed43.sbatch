#!/bin/bash
#SBATCH --account=grp_gbmostaf
#SBATCH --qos=public
#SBATCH --partition=public
#SBATCH --job-name=lime_mc43
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: sbatch $0 <PROJECT_ROOT> <BUNDLE_PATH> <SELECTED_CSV> <PYTHON_SCRIPT> [OUTPUT_DIR]"
  exit 1
fi

PROJECT_ROOT="$1"
BUNDLE_PATH="$2"
SELECTED_CSV="$3"
PY_SCRIPT="$4"
OUTPUT_DIR="${5:-$PROJECT_ROOT/lime_outputs/bworf_mi_seed43}"
PYTHON_BIN="$HOME/.conda/envs/sim_orf/bin/python"

module purge
module load mamba/latest

cd "$PROJECT_ROOT"
mkdir -p "$OUTPUT_DIR"
export PYTHONPATH="$PROJECT_ROOT/code:$PYTHONPATH"

echo "HOST: $(hostname)"
echo "DATE: $(date)"
echo "PROJECT_ROOT: $PROJECT_ROOT"
echo "BUNDLE_PATH: $BUNDLE_PATH"
echo "SELECTED_CSV: $SELECTED_CSV"
echo "OUTPUT_DIR: $OUTPUT_DIR"

"$PYTHON_BIN" "$PY_SCRIPT" \
  --bundle "$BUNDLE_PATH" \
  --selected_csv "$SELECTED_CSV" \
  --output_dir "$OUTPUT_DIR" \
  --num_features 10 \
  --num_samples 5000
