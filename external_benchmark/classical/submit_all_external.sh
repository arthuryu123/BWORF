#!/bin/bash

set -euo pipefail



ROOT="/home/chungyua/patch_external/sbatch/run_external.sbatch"



sbatch "$ROOT" breast_cancer rf

sbatch "$ROOT" breast_cancer orf_style

sbatch "$ROOT" breast_cancer bworf_no_mi

sbatch "$ROOT" breast_cancer bworf_mi



sbatch "$ROOT" heart_failure rf

sbatch "$ROOT" heart_failure orf_style

sbatch "$ROOT" heart_failure bworf_no_mi

sbatch "$ROOT" heart_failure bworf_mi



sbatch "$ROOT" diabetes rf

sbatch "$ROOT" diabetes orf_style

sbatch "$ROOT" diabetes bworf_no_mi

sbatch "$ROOT" diabetes bworf_mi



sbatch "$ROOT" thyroid rf

sbatch "$ROOT" thyroid orf_style

sbatch "$ROOT" thyroid bworf_no_mi

sbatch "$ROOT" thyroid bworf_mi
