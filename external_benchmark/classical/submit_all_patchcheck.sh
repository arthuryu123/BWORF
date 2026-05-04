#!/bin/bash

set -euo pipefail



ROOT="/home/chungyua/patch_check_heart_thyroid/sbatch/run_patchcheck.sbatch"



sbatch "$ROOT" heart_failure rf

sbatch "$ROOT" heart_failure orf_style

sbatch "$ROOT" heart_failure bworf_no_mi

sbatch "$ROOT" heart_failure bworf_mi



sbatch "$ROOT" thyroid rf

sbatch "$ROOT" thyroid orf_style

sbatch "$ROOT" thyroid bworf_no_mi

sbatch "$ROOT" thyroid bworf_mi
