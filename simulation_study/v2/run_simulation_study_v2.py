"""
Simulation Study v2 - Runner Script

Purpose
-------
This runner performs a mechanistic binary genetics simulation benchmark comparing:
  - Random Forest (RF)
  - BWORF (ObliqueRandomForestMulti from bworf_with_mi.py)

Key upgrade from v1
-------------------
v2 adds VALID + FAST model-agnostic permutation importance on a TEST set to compute
“causal discovery quality” metrics for BOTH RF and BWORF:
  - mean_rank_causal
  - median_rank_causal
  - recall_at_20_causal (fraction of causal SNPs in top-20)

Speed optimizations:
  - PERM_N_REPEATS defaults to 1
  - permutation scoring is evaluated on a STRATIFIED row subsample of the test set

Outputs
-------
All artifacts are written under this script's folder:
  - outputs/simulation_results_v2.csv
  - outputs/run_config_v2.json
  - outputs/run_summary_v2.md

Notes
-----
- This script is deterministic given the config + MASTER_SEED.
- Parallelism is at the replicate-task level using joblib (loky).
- Thread oversubscription is prevented via OMP/MKL/OpenBLAS env vars set before numpy import.
"""

# ==============================================================================
# CRITICAL: Set thread limits BEFORE importing numpy/scipy/sklearn
# ==============================================================================
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import sys
import json
import hashlib
import time
import threading
from pathlib import Path
from datetime import datetime, timezone
from itertools import product

import numpy as np

# ==============================================================================
# Configuration
# ==============================================================================

QUICK_TEST = False  # Set to False for full run

MASTER_SEED = 12345
N_CORES = 10
VERSION = "v2"

# Evaluation protocol: repeated stratified holdout (replaces k-fold CV)
HOLDOUT_SEEDS = [42, 0, 12, 100, 90]
HOLDOUT_TEST_SIZE = 0.3

# Optional: write per-seed (per split) rows to a separate CSV
WRITE_PER_SEED_ROWS = False

# Batch size for incremental writes (how many tasks to run before writing)
BATCH_SIZE = N_CORES  # checkpoint after each wave of workers

# BWORF Model Configuration
BWORF_L1_STRENGTH = 1.0
BWORF_N_ESTIMATORS = 100
BWORF_N_TRIES = 10
BWORF_MAX_DEPTH = None  # None = unlimited depth

# Permutation-importance (causal discovery) configuration
ENABLE_PERM_IMPORTANCE = True
PERM_N_REPEATS = 1
PERM_SUBSAMPLE_N = 500
PERM_SCORING = "auc"
PERM_TOPK = 20

# Base directories
BASE_DIR = Path(__file__).resolve().parent  # .../Simulation study/v2
OUTPUTS_DIR = BASE_DIR / "outputs"
LOGS_DIR = BASE_DIR / "logs"

# ==============================================================================
# Path setup for imports
# ==============================================================================

# Engine lives in v1 directory (do NOT copy/modify simulation_utils_v1.py)
V1_DIR = BASE_DIR.parent / "v1"
ENGINE_PATH = V1_DIR / "simulation_utils_v1.py"
if not ENGINE_PATH.exists():
    raise FileNotFoundError(f"Expected engine at: {ENGINE_PATH}")
sys.path.insert(0, str(V1_DIR))

# BWORF directory (same as v1)
BWORF_DIR = Path(r"C:\Users\Arthur\Desktop\ORF\bworf\model")
if not BWORF_DIR.exists():
    # Try relative path from project root
    PROJECT_ROOT = BASE_DIR.parent.parent.parent  # .../ORF
    BWORF_DIR = PROJECT_ROOT / "bworf" / "model"
    if not BWORF_DIR.exists():
        BWORF_DIR = PROJECT_ROOT / "bworf"
sys.path.insert(0, str(BWORF_DIR))
sys.path.insert(0, str(BWORF_DIR.parent))

# ==============================================================================
# Imports (after path setup)
# ==============================================================================

try:
    from simulation_utils_v1 import simulate_dataset_v1
except ImportError as e:
    print(f"ERROR: Cannot import simulation engine: {e}")
    print(f"  Expected at: {ENGINE_PATH}")
    sys.exit(1)

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression as SklearnLR  # calibration
    from sklearn.metrics import (
        roc_auc_score,
        brier_score_loss,
        precision_score,
        recall_score,
        f1_score,
    )
    import sklearn

    SKLEARN_VERSION = sklearn.__version__
except ImportError as e:
    print(f"ERROR: Cannot import sklearn: {e}")
    sys.exit(1)

try:
    from joblib import Parallel, delayed
except ImportError as e:
    print(f"ERROR: Cannot import joblib: {e}")
    sys.exit(1)

try:
    from joblib.externals.loky.process_executor import TerminatedWorkerError
except Exception:
    TerminatedWorkerError = None  # type: ignore

# BWORF import (fail loudly if missing)
BWORF_CLASS = None
BWORF_MODULE_PATH = None
try:
    from bworf_with_mi import ObliqueRandomForestMulti as BWORF_CLASS

    BWORF_MODULE_PATH = str(BWORF_DIR / "bworf_with_mi.py")
except ImportError:
    pass

if BWORF_CLASS is None:
    try:
        from bworf.bworf_with_mi import ObliqueRandomForestMulti as BWORF_CLASS

        BWORF_MODULE_PATH = "bworf.bworf_with_mi"
    except ImportError:
        pass

if BWORF_CLASS is None:
    print("=" * 60)
    print("ERROR: Cannot import BWORF (ObliqueRandomForestMulti)")
    print(f"  Searched in: {BWORF_DIR}")
    print("  Please ensure bworf_with_mi.py exists and contains ObliqueRandomForestMulti")
    print("=" * 60)
    sys.exit(1)

print(f"[INFO] BWORF imported from: {BWORF_MODULE_PATH}")

# ==============================================================================
# Grid Definition
# ==============================================================================

if QUICK_TEST:
    # Small grid to validate pipeline quickly (including perm importance)
    GRID = {
        "n_snps": [1000],
        "n_samples": [500],
        "n_causal_pairs": [2],
        "h2": [0.10],
        "signal_ratio": [1.0],
        "rho": [0.5],
        "maf_type": ["fixed"],
        "case_control_ratio": [1.0],
    }
    N_REPLICATES = 2
else:
    # Default “focused” full grid (edit as needed)
    GRID = {
        "n_snps": [1000],
        "n_samples": [500, 2000],
        "n_causal_pairs": [2, 5],
        "h2": [0.05, 0.15],
        "signal_ratio": [np.inf, 1.0, 0.5],
        "rho": [0.0, 0.8],
        "maf_type": ["fixed"],
        "case_control_ratio": [1.0],
    }
    N_REPLICATES = 5

# Fixed parameters
N_BLOCKS_PER_1000_SNPS = 50

# ==============================================================================
# Thread-safe CSV writer lock
# ==============================================================================

CSV_LOCK = threading.Lock()

# ==============================================================================
# Helper functions
# ==============================================================================


def stable_hash_from_parts(*parts) -> int:
    """Stable 32-bit hash from arbitrary parts (do NOT use Python's built-in hash())."""
    data = "_".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.md5(data).hexdigest()
    return int(digest[:8], 16)


def stable_hash_seed(master_seed: int, scenario_idx: int, replicate_id: int) -> int:
    """Deterministic seed per (scenario_idx, replicate_id)."""
    return stable_hash_from_parts(master_seed, scenario_idx, replicate_id)


def nanmean_std(values) -> tuple:
    """Return (nanmean, nanstd(ddof=1)) or (nan, nan) if all nan."""
    arr = np.asarray(values, dtype=np.float64)
    mask = ~np.isnan(arr)
    if not np.any(mask):
        return np.nan, np.nan
    mean = float(np.nanmean(arr))
    std = float(np.nanstd(arr, ddof=1)) if np.sum(mask) >= 2 else np.nan
    return mean, std


def build_scenario_id(params: dict) -> str:
    ratio_str = "inf" if np.isinf(params["signal_ratio"]) else f"{params['signal_ratio']:.1f}"
    return (
        f"P{params['n_snps']}_N{params['n_samples']}_M{params['n_causal_pairs']}_"
        f"h2{params['h2']:.2f}_r{ratio_str}_rho{params['rho']:.1f}_"
        f"{params['maf_type']}_bal{params['case_control_ratio']:.1f}"
    )


def compute_n_blocks_from_n_snps(n_snps: int) -> int:
    """
    Choose n_blocks that divides n_snps (required by GenomeSimulator).
    Aim for block size ~ 1000/50 = 20 SNPs per block.
    """
    if n_snps <= 0:
        raise ValueError(f"n_snps must be positive, got {n_snps}")
    target_block_size = max(1, int(round(1000 / N_BLOCKS_PER_1000_SNPS)))
    approx_blocks = max(1, int(round(n_snps / target_block_size)))
    approx_blocks = min(approx_blocks, n_snps)
    for n_blocks in range(approx_blocks, 0, -1):
        if n_snps % n_blocks == 0:
            return n_blocks
    return 1


def generate_scenarios() -> list:
    keys = list(GRID.keys())
    values = [GRID[k] for k in keys]
    scenarios = []
    for combo in product(*values):
        params = dict(zip(keys, combo))
        params["n_blocks"] = compute_n_blocks_from_n_snps(int(params["n_snps"]))
        scenarios.append(params)
    return scenarios


def create_rf_model(seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=200,
        random_state=seed,
        n_jobs=1,  # avoid nested parallelism
        class_weight=None,
    )


def create_bworf_model(seed: int, n_features: int):
    """Create BWORF model (adapts to available constructor args)."""
    kwargs = {"n_estimators": BWORF_N_ESTIMATORS, "random_state": seed}

    import inspect

    try:
        sig = inspect.signature(BWORF_CLASS.__init__)
        param_names = set(sig.parameters.keys())
    except Exception:
        param_names = set()

    optional_params = {
        "max_depth": BWORF_MAX_DEPTH,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "n_tries": BWORF_N_TRIES,
        "l1_strength": BWORF_L1_STRENGTH,
        "weighted_bootstrap": False,
        "bootstrap_temperature": 1.0,
        "n_jobs": 1,
    }

    for k, v in optional_params.items():
        if k in param_names:
            kwargs[k] = v

    return BWORF_CLASS(**kwargs)


def get_proba_class1(model, X: np.ndarray) -> np.ndarray:
    """
    Robustly return P(y=1) for a fitted binary classifier.

    Rules:
    - If model has classes_ and includes class 1: use that column.
    - Else if predict_proba is (n, 2): use column 1.
    - Else raise with diagnostics.
    """
    proba = model.predict_proba(X)
    if proba.ndim != 2:
        raise RuntimeError(f"predict_proba must return 2D array, got shape {proba.shape}")
    if hasattr(model, "classes_"):
        classes = list(model.classes_)
        if 1 in classes:
            return proba[:, classes.index(1)]
    if proba.shape[1] == 2:
        return proba[:, 1]
    raise RuntimeError(
        f"Cannot determine class-1 probability: classes_={getattr(model, 'classes_', None)}, proba_shape={proba.shape}"
    )


def compute_calibration(y_true: np.ndarray, y_proba: np.ndarray) -> tuple:
    """Calibration intercept/slope via logistic regression on logit(p) -> y."""
    try:
        p = np.clip(y_proba, 1e-6, 1 - 1e-6)
        logit_p = np.log(p / (1 - p))
        if len(np.unique(y_true)) < 2:
            return np.nan, np.nan
        clf = SklearnLR(solver="lbfgs", max_iter=1000)
        clf.fit(logit_p.reshape(-1, 1), y_true)
        return float(clf.intercept_[0]), float(clf.coef_[0, 0])
    except Exception:
        return np.nan, np.nan


def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray) -> dict:
    """Predictive + class-specific + calibration metrics on a test set."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = (y_proba >= 0.5).astype(int)

    try:
        auc_roc = roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) >= 2 else np.nan
    except Exception:
        auc_roc = np.nan

    try:
        brier = brier_score_loss(y_true, y_proba)
    except Exception:
        brier = np.nan

    try:
        precision_pos = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        recall_pos = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        f1_pos = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    except Exception:
        precision_pos = np.nan
        recall_pos = np.nan
        f1_pos = np.nan

    calib_intercept, calib_slope = compute_calibration(y_true, y_proba)

    return {
        "auc_roc": auc_roc,
        "brier": brier,
        "precision_pos": precision_pos,
        "recall_pos": recall_pos,
        "f1_pos": f1_pos,
        "calib_intercept": calib_intercept,
        "calib_slope": calib_slope,
    }


def stratified_subsample_indices(y: np.ndarray, n_target: int, rng: np.random.Generator) -> np.ndarray:
    """
    Stratified subsample indices from y (no replacement), ensuring both classes if possible.
    If len(y) <= n_target: returns all indices.
    """
    y = np.asarray(y).astype(int)
    n = len(y)
    if n_target is None or n_target <= 0 or n <= n_target:
        return np.arange(n, dtype=int)

    classes = np.unique(y)
    if len(classes) < 2:
        return np.arange(n, dtype=int)

    idx_by_class = {c: np.where(y == c)[0] for c in classes}
    # Only support binary here (simulation is binary)
    c0, c1 = classes[0], classes[1]
    n0_total = len(idx_by_class[c0])
    n1_total = len(idx_by_class[c1])

    # Proportional allocation with at least 1 from each
    n0 = max(1, int(round(n_target * n0_total / n)))
    n1 = n_target - n0
    if n1 < 1:
        n1 = 1
        n0 = n_target - 1

    # Clip to available and re-balance if needed
    n0 = min(n0, n0_total)
    n1 = min(n1, n1_total)
    if n0 + n1 < n_target:
        remaining = n_target - (n0 + n1)
        # Fill remaining from the class with more leftover capacity
        cap0 = n0_total - n0
        cap1 = n1_total - n1
        add0 = min(remaining, cap0) if cap0 >= cap1 else 0
        add1 = remaining - add0
        add1 = min(add1, cap1)
        add0 = remaining - add1
        n0 += add0
        n1 += add1

    # Final fallback if still short (extreme imbalance)
    if n0 < 1 or n1 < 1:
        return np.arange(n, dtype=int)

    sel0 = rng.choice(idx_by_class[c0], size=n0, replace=False)
    sel1 = rng.choice(idx_by_class[c1], size=n1, replace=False)
    sel = np.concatenate([sel0, sel1])
    rng.shuffle(sel)
    return sel.astype(int)


def permutation_importance_auc(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    perm_seed: int,
    subsample_idx: np.ndarray,
) -> tuple[np.ndarray | None, float | None]:
    """
    Compute permutation importance over ALL features using AUC on a stratified row subsample.

    Returns:
        importances: np.ndarray shape (P,) or None if cannot compute (e.g., single-class y_sub)
        auc_base: baseline AUC on subsample or None
    """
    if PERM_SCORING != "auc":
        raise ValueError(f"Unsupported PERM_SCORING: {PERM_SCORING}")

    y_test = np.asarray(y_test).astype(int)
    if subsample_idx is None or len(subsample_idx) == 0:
        subsample_idx = np.arange(len(y_test), dtype=int)

    # Build subsample (small, contiguous) for speed
    X_sub = np.ascontiguousarray(X_test[subsample_idx])
    y_sub = y_test[subsample_idx]

    # Ensure both classes present; if not, fall back to full test
    if len(np.unique(y_sub)) < 2:
        X_sub = np.ascontiguousarray(X_test)
        y_sub = y_test
        if len(np.unique(y_sub)) < 2:
            print("  [PERM] WARNING: test set has a single class; skipping permutation importance.")
            return None, None

    # Baseline AUC on subsample
    try:
        proba_base = get_proba_class1(model, X_sub)
        auc_base = float(roc_auc_score(y_sub, proba_base))
    except Exception:
        return None, None

    n_rows, n_features = X_sub.shape
    importances = np.zeros(n_features, dtype=np.float64)

    # Validate causal indices later; here ensure we have full feature scan
    assert n_features > 0

    # Permute one feature at a time in-place (restore after), avoiding large copies.
    for j in range(n_features):
        col_orig = X_sub[:, j].copy()
        drops = []
        for r in range(PERM_N_REPEATS):
            feature_seed = stable_hash_from_parts(perm_seed, j, r)
            rng = np.random.default_rng(feature_seed)
            perm = rng.permutation(n_rows)
            X_sub[:, j] = col_orig[perm]
            try:
                proba_perm = get_proba_class1(model, X_sub)
                auc_perm = float(roc_auc_score(y_sub, proba_perm))
                drops.append(auc_base - auc_perm)
            except Exception:
                drops.append(0.0)
        X_sub[:, j] = col_orig  # restore
        importances[j] = float(np.mean(drops)) if drops else 0.0

    # Sanity: length == P
    if importances.shape[0] != X_test.shape[1]:
        raise RuntimeError(f"Permutation importance length mismatch: {importances.shape[0]} vs P={X_test.shape[1]}")

    return importances, auc_base


def compute_causal_rank_metrics(importances: np.ndarray, causal_snps: np.ndarray) -> tuple[float, float, float]:
    """
    Convert importance scores into causal rank metrics.

    recall_at_20_causal is the FRACTION of causal SNPs that appear in top PERM_TOPK.
    """
    P = int(importances.shape[0])
    causal_snps = np.asarray(causal_snps, dtype=int)

    if np.any(causal_snps < 0) or np.any(causal_snps >= P):
        raise ValueError(f"Causal SNP indices out of bounds for P={P}: {causal_snps}")

    # Stable tie-breaking: mergesort is stable; NaNs are pushed to the end.
    imp = np.asarray(importances, dtype=np.float64)
    imp_sort = np.where(np.isnan(imp), -np.inf, imp)
    order = np.argsort(-imp_sort, kind="mergesort")

    ranks = np.empty(P, dtype=int)
    ranks[order] = np.arange(1, P + 1)

    causal_ranks = ranks[causal_snps]
    mean_rank = float(np.mean(causal_ranks))
    median_rank = float(np.median(causal_ranks))

    topk = int(PERM_TOPK)
    recall_at_k = float(np.mean(causal_ranks <= topk)) if len(causal_ranks) > 0 else np.nan
    if not (0.0 <= recall_at_k <= 1.0):
        raise RuntimeError(f"recall_at_{topk}_causal out of range: {recall_at_k}")

    return mean_rank, median_rank, recall_at_k


def evaluate_repeated_holdout(
    method_name: str,
    X: np.ndarray,
    y: np.ndarray,
    causal_snps: np.ndarray,
    scenario_id: str,
    scenario_idx: int,
    replicate_id: int,
    base_seed: int,
    n_features: int,
) -> tuple[dict, list]:
    """
    Repeated stratified holdout evaluation over HOLDOUT_SEEDS.

    Adds permutation-importance based causal metrics if ENABLE_PERM_IMPORTANCE=True.
    """
    per_seed_rows = []

    # per-seed values
    auc_vals, brier_vals, prec_vals, rec_vals, f1_vals = [], [], [], [], []
    calib_i_vals, calib_s_vals = [], []
    mean_rank_vals, median_rank_vals, recall20_vals = [], [], []
    runtime_vals = []

    for split_seed in HOLDOUT_SEEDS:
        # Deterministic model seed per split + method
        model_seed = stable_hash_from_parts(MASTER_SEED, scenario_id, replicate_id, split_seed, method_name, "model")

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=HOLDOUT_TEST_SIZE,
            stratify=y,
            random_state=split_seed,
        )

        t0 = time.perf_counter()

        if method_name == "RF":
            model = create_rf_model(model_seed)
        elif method_name == "BWORF":
            model = create_bworf_model(model_seed, n_features)
        else:
            raise ValueError(f"Unknown method_name: {method_name}")

        model.fit(X_train, y_train)
        y_proba = get_proba_class1(model, X_test)
        runtime = float(time.perf_counter() - t0)

        m = compute_metrics(y_test, y_proba)
        auc_vals.append(m["auc_roc"])
        brier_vals.append(m["brier"])
        prec_vals.append(m["precision_pos"])
        rec_vals.append(m["recall_pos"])
        f1_vals.append(m["f1_pos"])
        calib_i_vals.append(m["calib_intercept"])
        calib_s_vals.append(m["calib_slope"])
        runtime_vals.append(runtime)

        # Permutation-importance based causal discovery metrics (apples-to-apples)
        mean_rank = np.nan
        median_rank = np.nan
        recall20 = np.nan

        if ENABLE_PERM_IMPORTANCE:
            # Method-independent seed for permutation mechanics:
            perm_seed = stable_hash_from_parts(MASTER_SEED, scenario_id, replicate_id, split_seed, "perm")
            perm_rng = np.random.default_rng(perm_seed)

            # Stratified row subsample (shared across methods because seed is method-independent)
            sub_idx = stratified_subsample_indices(y_test, PERM_SUBSAMPLE_N, perm_rng)

            try:
                importances, auc_base_sub = permutation_importance_auc(
                    model=model,
                    X_test=X_test,
                    y_test=y_test,
                    perm_seed=perm_seed,
                    subsample_idx=sub_idx,
                )
                if importances is not None:
                    # Validate importance vector length and causal indices
                    if importances.shape[0] != X_test.shape[1]:
                        raise RuntimeError("importance vector length != P")
                    mean_rank, median_rank, recall20 = compute_causal_rank_metrics(importances, causal_snps)
            except Exception as e:
                # Do not crash run; keep NaNs
                print(f"  [PERM] WARNING: permutation importance failed ({method_name}, split_seed={split_seed}): {e}")

        mean_rank_vals.append(mean_rank)
        median_rank_vals.append(median_rank)
        recall20_vals.append(recall20)

        if WRITE_PER_SEED_ROWS:
            per_seed_rows.append(
                {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "version": VERSION,
                    "scenario_id": scenario_id,
                    "replicate_id": replicate_id,
                    "seed": base_seed,
                    "split_seed": split_seed,
                    "method": method_name,
                    "auc_roc": m["auc_roc"],
                    "brier": m["brier"],
                    "precision_pos": m["precision_pos"],
                    "recall_pos": m["recall_pos"],
                    "f1_pos": m["f1_pos"],
                    "calib_intercept": m["calib_intercept"],
                    "calib_slope": m["calib_slope"],
                    "mean_rank_causal": mean_rank,
                    "median_rank_causal": median_rank,
                    "recall_at_20_causal": recall20,
                    "runtime_seconds": runtime,
                    "error": "",
                }
            )

    auc_mean, auc_std = nanmean_std(auc_vals)
    brier_mean, brier_std = nanmean_std(brier_vals)
    prec_mean, prec_std = nanmean_std(prec_vals)
    rec_mean, rec_std = nanmean_std(rec_vals)
    f1_mean, f1_std = nanmean_std(f1_vals)
    ci_mean, ci_std = nanmean_std(calib_i_vals)
    cs_mean, cs_std = nanmean_std(calib_s_vals)
    mr_mean, mr_std = nanmean_std(mean_rank_vals)
    medr_mean, medr_std = nanmean_std(median_rank_vals)
    r20_mean, r20_std = nanmean_std(recall20_vals)
    rt_mean, rt_std = nanmean_std(runtime_vals)

    agg = {
        "auc_roc": auc_mean,
        "auc_roc_std": auc_std,
        "brier": brier_mean,
        "brier_std": brier_std,
        "precision_pos": prec_mean,
        "precision_pos_std": prec_std,
        "recall_pos": rec_mean,
        "recall_pos_std": rec_std,
        "f1_pos": f1_mean,
        "f1_pos_std": f1_std,
        "calib_intercept": ci_mean,
        "calib_intercept_std": ci_std,
        "calib_slope": cs_mean,
        "calib_slope_std": cs_std,
        "mean_rank_causal": mr_mean,
        "mean_rank_causal_std": mr_std,
        "median_rank_causal": medr_mean,
        "median_rank_causal_std": medr_std,
        "recall_at_20_causal": r20_mean,
        "recall_at_20_causal_std": r20_std,
        "runtime_seconds": rt_mean,
        "runtime_seconds_std": rt_std,
    }

    return agg, per_seed_rows


# ==============================================================================
# BWORF Sanity Probe
# ==============================================================================


def run_bworf_sanity_probe():
    """Lightweight check that BWORF behaves as binary classifier and returns 2-column proba."""
    print("[SANITY PROBE] Verifying BWORF binary classification behavior...")
    print(
        f"  BWORF config: n_estimators={BWORF_N_ESTIMATORS}, max_depth={BWORF_MAX_DEPTH}, "
        f"n_tries={BWORF_N_TRIES}, l1_strength={BWORF_L1_STRENGTH}"
    )
    if BWORF_L1_STRENGTH is None or BWORF_L1_STRENGTH <= 0:
        raise RuntimeError(f"BWORF_L1_STRENGTH must be > 0, got {BWORF_L1_STRENGTH}")

    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 20))
    y = rng.integers(0, 2, size=100)
    model = create_bworf_model(seed=42, n_features=20)
    model.fit(X, y)
    proba = model.predict_proba(X)
    if proba.ndim != 2 or proba.shape[1] != 2:
        raise RuntimeError(f"BWORF predict_proba shape unexpected: {proba.shape}")
    row_sums = proba.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        raise RuntimeError(
            f"BWORF predict_proba rows do not sum to 1. Range [{row_sums.min():.4f}, {row_sums.max():.4f}]"
        )
    _ = get_proba_class1(model, X)
    print("[SANITY PROBE] OK")


# ==============================================================================
# CSV I/O
# ==============================================================================


CSV_COLUMNS = [
    "timestamp_utc",
    "version",
    "scenario_id",
    "replicate_id",
    "seed",
    "n_samples",
    "n_snps",
    "n_blocks",
    "rho",
    "maf_type",
    "n_causal_pairs",
    "h2",
    "signal_ratio",
    "case_control_ratio",
    "method",
    "auc_roc",
    "auc_roc_std",
    "brier",
    "brier_std",
    "precision_pos",
    "precision_pos_std",
    "recall_pos",
    "recall_pos_std",
    "f1_pos",
    "f1_pos_std",
    "calib_intercept",
    "calib_intercept_std",
    "calib_slope",
    "calib_slope_std",
    "mean_rank_causal",
    "mean_rank_causal_std",
    "median_rank_causal",
    "median_rank_causal_std",
    "recall_at_20_causal",
    "recall_at_20_causal_std",
    "runtime_seconds",
    "runtime_seconds_std",
    "error",
]

PER_SEED_CSV_COLUMNS = [
    "timestamp_utc",
    "version",
    "scenario_id",
    "replicate_id",
    "seed",
    "split_seed",
    "n_samples",
    "n_snps",
    "n_blocks",
    "rho",
    "maf_type",
    "n_causal_pairs",
    "h2",
    "signal_ratio",
    "case_control_ratio",
    "method",
    "auc_roc",
    "brier",
    "precision_pos",
    "recall_pos",
    "f1_pos",
    "calib_intercept",
    "calib_slope",
    "mean_rank_causal",
    "median_rank_causal",
    "recall_at_20_causal",
    "runtime_seconds",
    "error",
]


def write_csv_header(csv_path: Path, columns: list):
    expected_header = ",".join(columns)
    with CSV_LOCK:
        if csv_path.exists():
            with open(csv_path, "r", encoding="utf-8") as f:
                first_line = f.readline().rstrip("\n\r")
            if first_line and first_line != expected_header:
                raise RuntimeError(
                    f"Existing CSV schema mismatch for: {csv_path}\n"
                    f"Expected header:\n{expected_header}\n\n"
                    f"Found header:\n{first_line}\n\n"
                    f"Please move/delete the existing file before running with the new schema."
                )
            if not first_line:
                with open(csv_path, "w", encoding="utf-8") as f:
                    f.write(expected_header + "\n")
        else:
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write(expected_header + "\n")


def append_csv_rows(csv_path: Path, rows: list, columns: list):
    if not rows:
        return
    with CSV_LOCK:
        with open(csv_path, "a", encoding="utf-8") as f:
            for row in rows:
                values = []
                for col in columns:
                    val = row.get(col, "")
                    if isinstance(val, float):
                        if np.isnan(val):
                            val = ""
                        elif np.isinf(val):
                            val = "inf"
                        else:
                            val = f"{val:.6f}"
                    elif isinstance(val, (int, np.integer)):
                        val = str(int(val))
                    else:
                        val = str(val)
                    if "," in val or '"' in val:
                        val = '"' + val.replace('"', '""') + '"'
                    values.append(val)
                f.write(",".join(values) + "\n")


def count_csv_data_rows(csv_path: Path) -> int:
    if not csv_path.exists():
        return 0
    with open(csv_path, "r", encoding="utf-8") as f:
        n_lines = sum(1 for _ in f)
    return max(0, n_lines - 1)


def load_completed_tasks(csv_path: Path) -> set:
    """Resume support: only count rows with empty error as completed."""
    if not csv_path.exists():
        return set()
    completed = set()
    try:
        import pandas as pd

        try:
            df = pd.read_csv(csv_path, usecols=["scenario_id", "replicate_id", "method", "error"])
        except Exception:
            df = pd.read_csv(csv_path, usecols=["scenario_id", "replicate_id", "method"])
            df["error"] = ""

        for _, row in df.iterrows():
            err = row.get("error", "")
            err_str = "" if pd.isna(err) else str(err).strip()
            if err_str:
                continue
            completed.add((str(row["scenario_id"]), int(row["replicate_id"]), str(row["method"])))
    except ImportError:
        import csv

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if str(row.get("error", "")).strip():
                    continue
                completed.add((row["scenario_id"], int(row["replicate_id"]), row["method"]))
    return completed


# ==============================================================================
# Single replicate runner
# ==============================================================================


def run_single_replicate(
    scenario_idx: int,
    scenario_params: dict,
    replicate_id: int,
    scenario_id: str,
    skip_methods: set | None = None,
) -> tuple[list, list]:
    if skip_methods is None:
        skip_methods = set()

    seed = stable_hash_seed(MASTER_SEED, scenario_idx, replicate_id)
    timestamp_utc = datetime.now(timezone.utc).isoformat()

    agg_rows = []
    per_seed_rows_all = []

    methods_to_run = [m for m in ["RF", "BWORF"] if m not in skip_methods]
    if not methods_to_run:
        return agg_rows, per_seed_rows_all

    def make_nan_row(method: str, error_msg: str) -> dict:
        row = {
            "timestamp_utc": timestamp_utc,
            "version": VERSION,
            "scenario_id": scenario_id,
            "replicate_id": replicate_id,
            "seed": seed,
            "n_samples": scenario_params["n_samples"],
            "n_snps": scenario_params["n_snps"],
            "n_blocks": scenario_params["n_blocks"],
            "rho": scenario_params["rho"],
            "maf_type": scenario_params["maf_type"],
            "n_causal_pairs": scenario_params["n_causal_pairs"],
            "h2": scenario_params["h2"],
            "signal_ratio": scenario_params["signal_ratio"],
            "case_control_ratio": scenario_params["case_control_ratio"],
            "method": method,
            "error": error_msg,
        }
        # Fill numeric columns as NaN for schema consistency
        for col in CSV_COLUMNS:
            if col not in row and col not in {"timestamp_utc", "version", "scenario_id", "method", "error", "maf_type"}:
                row[col] = np.nan
        return row

    # Generate dataset once per replicate
    try:
        X, y, causal_pairs, _meta = simulate_dataset_v1(
            n_samples=scenario_params["n_samples"],
            n_snps=scenario_params["n_snps"],
            n_blocks=scenario_params["n_blocks"],
            rho=scenario_params["rho"],
            maf_type=scenario_params["maf_type"],
            n_causal_pairs=scenario_params["n_causal_pairs"],
            h2=scenario_params["h2"],
            signal_ratio=scenario_params["signal_ratio"],
            case_control_ratio=scenario_params["case_control_ratio"],
            random_state=seed,
        )
    except Exception as e:
        for method in methods_to_run:
            agg_rows.append(make_nan_row(method, f"Data generation failed: {e}"))
        return agg_rows, per_seed_rows_all

    y = np.asarray(y).astype(int)
    causal_snps = np.unique(np.asarray(causal_pairs).flatten()).astype(int)
    n_features = int(X.shape[1])

    # Basic validation
    if ENABLE_PERM_IMPORTANCE:
        if np.any(causal_snps < 0) or np.any(causal_snps >= n_features):
            for method in methods_to_run:
                agg_rows.append(make_nan_row(method, "Invalid causal SNP indices for P"))
            return agg_rows, per_seed_rows_all

    base_info = {
        "timestamp_utc": timestamp_utc,
        "version": VERSION,
        "scenario_id": scenario_id,
        "replicate_id": replicate_id,
        "seed": seed,
        "n_samples": scenario_params["n_samples"],
        "n_snps": scenario_params["n_snps"],
        "n_blocks": scenario_params["n_blocks"],
        "rho": scenario_params["rho"],
        "maf_type": scenario_params["maf_type"],
        "n_causal_pairs": scenario_params["n_causal_pairs"],
        "h2": scenario_params["h2"],
        "signal_ratio": scenario_params["signal_ratio"],
        "case_control_ratio": scenario_params["case_control_ratio"],
        "error": "",
    }

    for method_name in methods_to_run:
        try:
            agg, per_seed_rows = evaluate_repeated_holdout(
                method_name=method_name,
                X=X,
                y=y,
                causal_snps=causal_snps,
                scenario_id=scenario_id,
                scenario_idx=scenario_idx,
                replicate_id=replicate_id,
                base_seed=seed,
                n_features=n_features,
            )
            row = base_info.copy()
            row["method"] = method_name
            row.update(agg)
            agg_rows.append(row)

            if WRITE_PER_SEED_ROWS and per_seed_rows:
                for r in per_seed_rows:
                    r.update(
                        {
                            "n_samples": scenario_params["n_samples"],
                            "n_snps": scenario_params["n_snps"],
                            "n_blocks": scenario_params["n_blocks"],
                            "rho": scenario_params["rho"],
                            "maf_type": scenario_params["maf_type"],
                            "n_causal_pairs": scenario_params["n_causal_pairs"],
                            "h2": scenario_params["h2"],
                            "signal_ratio": scenario_params["signal_ratio"],
                            "case_control_ratio": scenario_params["case_control_ratio"],
                        }
                    )
                per_seed_rows_all.extend(per_seed_rows)
        except Exception as e:
            agg_rows.append(make_nan_row(method_name, f"{method_name} failed: {e}"))

    return agg_rows, per_seed_rows_all


# ==============================================================================
# Config + Summary writers
# ==============================================================================


def write_run_config(config_path: Path, scenarios: list, start_time: datetime):
    rf_params = {"n_estimators": 200, "n_jobs": 1, "class_weight": None}
    bworf_params = {
        "n_estimators": BWORF_N_ESTIMATORS,
        "max_depth": BWORF_MAX_DEPTH,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "n_tries": BWORF_N_TRIES,
        "l1_strength": BWORF_L1_STRENGTH,
        "weighted_bootstrap": False,
        "bootstrap_temperature": 1.0,
        "n_jobs": 1,
    }
    config = {
        "version": VERSION,
        "quick_test": QUICK_TEST,
        "master_seed": MASTER_SEED,
        "n_cores": N_CORES,
        "batch_size": BATCH_SIZE,
        "engine_path": str(ENGINE_PATH),
        "bworf_module_path": BWORF_MODULE_PATH,
        "evaluation": {
            "protocol": "repeated_holdout",
            "holdout_seeds": HOLDOUT_SEEDS,
            "test_size": HOLDOUT_TEST_SIZE,
            "n_splits": len(HOLDOUT_SEEDS),
            "write_per_seed_rows": WRITE_PER_SEED_ROWS,
        },
        "perm_importance": {
            "enabled": ENABLE_PERM_IMPORTANCE,
            "n_repeats": PERM_N_REPEATS,
            "subsample_n": PERM_SUBSAMPLE_N,
            "scoring": PERM_SCORING,
            "topk": PERM_TOPK,
            "notes": "Uses stratified row subsample and deterministic per-feature permutations (method-independent).",
        },
        "grid": {k: [("inf" if (isinstance(v, float) and np.isinf(v)) else v) for v in vals] for k, vals in GRID.items()},
        "n_scenarios": len(scenarios),
        "n_replicates": N_REPLICATES,
        "model_hyperparams": {"RF": rf_params, "BWORF": bworf_params},
        "timestamp_start_utc": start_time.isoformat(),
        "python_version": sys.version,
        "package_versions": {"numpy": np.__version__, "sklearn": SKLEARN_VERSION},
        "thread_limits": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        },
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def write_run_summary(
    summary_path: Path,
    scenarios: list,
    start_time: datetime,
    end_time: datetime | None = None,
    actual_rows: int | None = None,
    skipped_tasks: int = 0,
):
    n_scenarios = len(scenarios)
    expected_rows = n_scenarios * N_REPLICATES * 2

    content = f"""# Simulation Study v2 - Run Summary

## Configuration

- **Mode**: {"QUICK_TEST" if QUICK_TEST else "FULL"}
- **Version**: {VERSION}
- **Master Seed**: {MASTER_SEED}
- **Evaluation**: Repeated stratified holdout (n_splits={len(HOLDOUT_SEEDS)}, test_size={HOLDOUT_TEST_SIZE})
- **Holdout seeds**: {HOLDOUT_SEEDS}
- **Replicates per Scenario**: {N_REPLICATES}
- **Parallel Cores**: {N_CORES}
- **Batch Size**: {BATCH_SIZE}

## Permutation importance (causal discovery)

- **ENABLE_PERM_IMPORTANCE**: {ENABLE_PERM_IMPORTANCE}
- **PERM_N_REPEATS**: {PERM_N_REPEATS}
- **PERM_SUBSAMPLE_N**: {PERM_SUBSAMPLE_N} (stratified subsample of test rows)
- **PERM_TOPK**: {PERM_TOPK}

**Optimizations**:
1. n_repeats = {PERM_N_REPEATS}
2. Stratified row subsampling (max {PERM_SUBSAMPLE_N} rows) for AUC drop evaluation

## Grid

| Parameter | Values |
|-----------|--------|
"""
    for key, vals in GRID.items():
        content += f"| {key} | {', '.join(str(v) for v in vals)} |\n"

    content += f"""
## Scale

- **Number of Scenarios**: {n_scenarios}
- **Total Replicates**: {n_scenarios * N_REPLICATES}
- **Expected Rows**: {expected_rows} (scenarios × replicates × 2 methods)
"""

    if skipped_tasks > 0:
        content += f"- **Skipped (already completed)**: {skipped_tasks} task-method pairs\n"
    if actual_rows is not None:
        content += f"- **Actual Rows in CSV**: {actual_rows}\n"

    content += f"""
## Reproducibility

Seeds are derived deterministically via MD5 hashing of stable identifiers.

## Outputs

- `simulation_results_v2.csv` - aggregated results (mean±std across holdout seeds)
- `run_config_v2.json` - configuration snapshot
- `run_summary_v2.md` - this file
"""

    content += f"\n## Timestamps\n\n- **Start**: {start_time.isoformat()}\n"
    if end_time is not None:
        duration = (end_time - start_time).total_seconds()
        content += f"- **End**: {end_time.isoformat()}\n"
        content += f"- **Duration**: {duration:.1f} seconds ({duration/60:.1f} minutes)\n"

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(content)


# ==============================================================================
# Main
# ==============================================================================


def main():
    print("=" * 70)
    print("Simulation Study v2 - Runner (checkpointing + perm importance)")
    print("=" * 70)
    print(f"Mode: {'QUICK_TEST' if QUICK_TEST else 'FULL'}")
    print(f"Master Seed: {MASTER_SEED}")
    print(f"Cores: {N_CORES}")
    print(f"Evaluation: repeated holdout | n_splits={len(HOLDOUT_SEEDS)} | test_size={HOLDOUT_TEST_SIZE}")
    print(f"Holdout seeds: {HOLDOUT_SEEDS}")
    print(f"Replicates: {N_REPLICATES}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Perm importance: enabled={ENABLE_PERM_IMPORTANCE}, repeats={PERM_N_REPEATS}, subsample_n={PERM_SUBSAMPLE_N}")
    print()

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = OUTPUTS_DIR / "simulation_results_v2.csv"
    config_path = OUTPUTS_DIR / "run_config_v2.json"
    summary_path = OUTPUTS_DIR / "run_summary_v2.md"
    per_seed_csv_path = OUTPUTS_DIR / "simulation_results_v2_per_seed.csv"

    run_bworf_sanity_probe()
    print()

    completed_tasks = load_completed_tasks(csv_path)

    scenarios = generate_scenarios()
    n_scenarios = len(scenarios)
    expected_rows = n_scenarios * N_REPLICATES * 2
    print(f"Scenarios: {n_scenarios}")
    print(f"Total replicates: {n_scenarios * N_REPLICATES}")
    print(f"Expected rows: {expected_rows}")
    print()

    start_time = datetime.now(timezone.utc)
    write_run_config(config_path, scenarios, start_time)
    write_run_summary(summary_path, scenarios, start_time, skipped_tasks=len(completed_tasks))
    print(f"Config written: {config_path}")
    print(f"Summary written: {summary_path}")
    print()

    write_csv_header(csv_path, CSV_COLUMNS)
    print(f"CSV (aggregate): {csv_path}")
    if WRITE_PER_SEED_ROWS:
        write_csv_header(per_seed_csv_path, PER_SEED_CSV_COLUMNS)
        print(f"CSV (per-seed): {per_seed_csv_path}")
    print()

    existing_rows_before_run = count_csv_data_rows(csv_path)
    if existing_rows_before_run > 0:
        print(f"[RESUME] Existing rows in CSV (incl. errors): {existing_rows_before_run}")
        print()

    tasks = []
    skipped_count = 0
    for scenario_idx, scenario_params in enumerate(scenarios):
        scenario_id = build_scenario_id(scenario_params)
        for replicate_id in range(N_REPLICATES):
            skip_methods = set()
            for method in ["RF", "BWORF"]:
                key = (scenario_id, replicate_id, method)
                if key in completed_tasks:
                    skip_methods.add(method)
                    skipped_count += 1
            if len(skip_methods) < 2:
                tasks.append((scenario_idx, scenario_params, replicate_id, scenario_id, skip_methods))

    total_tasks = len(tasks)
    print(f"Tasks to run: {total_tasks} (skipped {skipped_count} already-completed method runs)")
    if total_tasks == 0:
        print("All tasks already completed. Nothing to do.")
        end_time = datetime.now(timezone.utc)
        actual_rows_in_file = count_csv_data_rows(csv_path)
        write_run_summary(summary_path, scenarios, start_time, end_time, actual_rows_in_file, skipped_count)
        return

    print("-" * 70)

    rows_written_this_run = 0
    success_keys_written = set()

    def process_task(task):
        scenario_idx, scenario_params, replicate_id, scenario_id, skip_methods = task
        return run_single_replicate(scenario_idx, scenario_params, replicate_id, scenario_id, skip_methods)

    n_batches = (total_tasks + BATCH_SIZE - 1) // BATCH_SIZE
    effective_n_jobs = N_CORES

    for batch_idx in range(n_batches):
        batch_start = batch_idx * BATCH_SIZE
        batch_end = min(batch_start + BATCH_SIZE, total_tasks)
        batch_tasks = tasks[batch_start:batch_end]

        print(f"[Batch {batch_idx + 1}/{n_batches}] Processing {len(batch_tasks)} tasks...")

        # Retry fallback on worker death
        batch_results = None
        n_jobs = effective_n_jobs
        while True:
            try:
                batch_results = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
                    delayed(process_task)(task) for task in batch_tasks
                )
                effective_n_jobs = n_jobs
                break
            except Exception as e:
                is_terminated = (
                    (TerminatedWorkerError is not None and isinstance(e, TerminatedWorkerError))
                    or (e.__class__.__name__ == "TerminatedWorkerError")
                )
                if not is_terminated:
                    raise
                if n_jobs <= 1:
                    print(
                        f"[Batch {batch_idx + 1}/{n_batches}] WARNING: TerminatedWorkerError even with n_jobs=1. "
                        f"Falling back to sequential execution for this batch."
                    )
                    batch_results = [process_task(task) for task in batch_tasks]
                    effective_n_jobs = 1
                    break
                new_jobs = max(1, n_jobs // 2)
                if new_jobs == n_jobs:
                    new_jobs = n_jobs - 1
                print(
                    f"[Batch {batch_idx + 1}/{n_batches}] WARNING: TerminatedWorkerError (worker died). "
                    f"Retrying batch with n_jobs={new_jobs} (was {n_jobs})."
                )
                n_jobs = new_jobs

        # batch_results: list of (agg_rows, per_seed_rows)
        all_rows = []
        all_seed_rows = []
        for agg_rows, per_seed_rows in batch_results:
            all_rows.extend(agg_rows)
            if WRITE_PER_SEED_ROWS and per_seed_rows:
                all_seed_rows.extend(per_seed_rows)

            for r in agg_rows:
                err = str(r.get("error", "")).strip()
                if err:
                    continue
                try:
                    key = (str(r.get("scenario_id")), int(r.get("replicate_id")), str(r.get("method")))
                    success_keys_written.add(key)
                except Exception:
                    pass

        append_csv_rows(csv_path, all_rows, CSV_COLUMNS)
        if WRITE_PER_SEED_ROWS and all_seed_rows:
            append_csv_rows(per_seed_csv_path, all_seed_rows, PER_SEED_CSV_COLUMNS)
        rows_written_this_run += len(all_rows)

        # Progress log
        for r in all_rows:
            method = r.get("method", "?")
            err = str(r.get("error", "")).strip()
            if err:
                print(f"  {method:5s} ERROR: {err}")
            else:
                auc = r.get("auc_roc", np.nan)
                auc_str = f"{auc:.3f}" if not np.isnan(auc) else "ERR"
                print(f"  {method:5s} AUC={auc_str}")

        print(f"  -> Checkpoint: {rows_written_this_run} rows written to CSV")

    end_time = datetime.now(timezone.utc)
    duration = (end_time - start_time).total_seconds()
    print("-" * 70)
    print(f"Completed in {duration:.1f} seconds ({duration/60:.1f} minutes)")
    print(f"Rows written this run: {rows_written_this_run}")

    total_rows_in_file = existing_rows_before_run + rows_written_this_run
    unique_completed_success = len(completed_tasks.union(success_keys_written))
    print(f"Total rows in CSV file: {total_rows_in_file} (may include prior error rows)")
    print(f"Unique completed (error-free) rows: {unique_completed_success} (expected: {expected_rows})")

    write_run_summary(summary_path, scenarios, start_time, end_time, total_rows_in_file, skipped_count)

    if unique_completed_success != expected_rows:
        print(f"WARNING: Incomplete run! UniqueCompleted={unique_completed_success}, Expected={expected_rows}")
    else:
        print("Row count validated OK.")

    print()
    print("=" * 70)
    print("Outputs:")
    print(f"  - {csv_path}")
    print(f"  - {config_path}")
    print(f"  - {summary_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()

