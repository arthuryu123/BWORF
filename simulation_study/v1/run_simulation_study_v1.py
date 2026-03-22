"""
Simulation Study v1 - Runner Script

Binary mechanistic simulation validation comparing RF vs BWORF.
Focus: LD ON (rho > 0), unique causal SNP pairs per dataset.

Outputs:
- outputs/simulation_results_v1.csv
- outputs/run_config_v1.json
- outputs/run_summary_v1.md

Usage:
    python run_simulation_study_v1.py

Set QUICK_TEST = True for a fast sanity check.
"""

# ==============================================================================
# CRITICAL: Set thread limits BEFORE importing numpy/scipy/sklearn
# This prevents BLAS/OpenMP thread oversubscription when using multiprocessing
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
import traceback
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
N_CORES = 6
VERSION = "v1"

# Legacy (no longer used): kept to avoid breaking older helper functions.
CV_FOLDS = 5

# ==============================================================================
# Evaluation protocol: Repeated stratified holdout splits (replaces k-fold CV)
# ==============================================================================
HOLDOUT_SEEDS = [42, 0, 12, 100, 90]
HOLDOUT_TEST_SIZE = 0.3

# Optional: write per-seed (per split) rows to a separate CSV
WRITE_PER_SEED_ROWS = False

# Batch size for incremental writes (how many tasks to run before writing)
# Smaller = more frequent checkpoints but more I/O overhead
BATCH_SIZE = N_CORES  # Process one batch of N_CORES tasks, then write results

# ==============================================================================
# BWORF Model Configuration
# ==============================================================================
# L1 regularization strength for BWORF oblique splits
# MUST be > 0 to avoid division by zero in LogisticRegression C parameter
# Higher = more sparse splits; 1.0 is a reasonable default
BWORF_L1_STRENGTH = 1.0
BWORF_N_ESTIMATORS = 100
BWORF_N_TRIES = 10
BWORF_MAX_DEPTH = None  # None = unlimited depth

# Base directory (this script's folder)
BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"
LOGS_DIR = BASE_DIR / "logs"

# ==============================================================================
# Path setup for imports
# ==============================================================================

# Add engine directory to path
sys.path.insert(0, str(BASE_DIR))

# Add BWORF directory to path
BWORF_DIR = Path(r"C:\Users\Arthur\Desktop\ORF\bworf\model")
if not BWORF_DIR.exists():
    # Try relative path from project root
    PROJECT_ROOT = BASE_DIR.parent.parent.parent  # Go up to ORF
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
    print(f"  Expected at: {BASE_DIR / 'simulation_utils_v1.py'}")
    sys.exit(1)

# Import sklearn
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, train_test_split
    from sklearn.linear_model import LogisticRegression as SklearnLR  # For calibration
    from sklearn.metrics import (
        roc_auc_score,
        brier_score_loss,
        accuracy_score,
        f1_score,
        balanced_accuracy_score,
        precision_score,
        recall_score
    )
    import sklearn
    SKLEARN_VERSION = sklearn.__version__
except ImportError as e:
    print(f"ERROR: Cannot import sklearn: {e}")
    sys.exit(1)

# Import joblib
try:
    from joblib import Parallel, delayed
except ImportError as e:
    print(f"ERROR: Cannot import joblib: {e}")
    sys.exit(1)

# TerminatedWorkerError is raised when a process is killed/crashes (OOM, segfault, etc.).
# We use it for a safe retry fallback with fewer workers.
try:
    from joblib.externals.loky.process_executor import TerminatedWorkerError
except Exception:
    TerminatedWorkerError = None  # type: ignore

# Import BWORF
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
    # Minimal grid for quick sanity check (~2-5 min)
    GRID = {
        "n_snps": [200],          # Small for speed
        "n_samples": [300],
        "n_causal_pairs": [2],
        "h2": [0.10],
        "signal_ratio": [1.0],    # With interaction
        "rho": [0.5],
        "maf_type": ["fixed"],
        "case_control_ratio": [1.0],
    }
    N_REPLICATES = 2
else:
    # Focused grid for thesis - tests critical regimes (kept intentionally small).
    GRID = {
        # P (SNPs)
        "n_snps": [1000],
        # N (samples)
        "n_samples": [500, 2000],
        # M (causal pairs) - set to [2] if you want an even smaller grid
        "n_causal_pairs": [2, 5],
        # h^2
        "h2": [0.05, 0.15],
        # ratio (beta_main / beta_int): no interaction vs interaction vs interaction-dominant
        "signal_ratio": [np.inf, 1.0, 0.5],
        # LD (rho): no LD vs strong LD
        "rho": [0.0, 0.8],
        # MAF and balance (keep fixed/balanced for v1)
        "maf_type": ["fixed"],
        "case_control_ratio": [1.0],
    }
    # Scenarios: 1 × 2 × 2 × 2 × 3 × 2 × 1 × 1 = 72
    # Replicates: 72 × 5 = 360 total (rows = 360 × 2 methods = 720)
    N_REPLICATES = 5

# Fixed parameters
N_BLOCKS_PER_1000_SNPS = 50  # Scale blocks with SNPs

# ==============================================================================
# Thread-safe CSV writer lock (for safety, though we write in main thread)
# ==============================================================================
CSV_LOCK = threading.Lock()

# ==============================================================================
# Helper Functions
# ==============================================================================

def stable_hash_seed(master_seed: int, scenario_idx: int, replicate_id: int) -> int:
    """
    Derive a deterministic seed using MD5 hash.
    Returns an integer in [0, 2^32).
    """
    data = f"{master_seed}_{scenario_idx}_{replicate_id}".encode("utf-8")
    digest = hashlib.md5(data).hexdigest()
    return int(digest[:8], 16)  # First 8 hex chars -> 32-bit int


def stable_hash_from_parts(*parts) -> int:
    """
    Stable 32-bit hash for arbitrary parts (do NOT use Python's built-in hash()).

    Useful for deriving deterministic but distinct seeds for:
    - model RNG per holdout split
    - any other reproducible sub-seeding needs
    """
    data = "_".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.md5(data).hexdigest()
    return int(digest[:8], 16)


def nanmean_std(values) -> tuple:
    """
    Compute (nanmean, nanstd) with ddof=1 when possible.
    Returns (np.nan, np.nan) if all values are nan.
    """
    arr = np.asarray(values, dtype=np.float64)
    mask = ~np.isnan(arr)
    if not np.any(mask):
        return np.nan, np.nan
    mean = float(np.nanmean(arr))
    std = float(np.nanstd(arr, ddof=1)) if np.sum(mask) >= 2 else np.nan
    return mean, std


def build_scenario_id(params: dict) -> str:
    """Build a stable scenario identifier string."""
    ratio_str = "inf" if np.isinf(params["signal_ratio"]) else f"{params['signal_ratio']:.1f}"
    return (
        f"P{params['n_snps']}_N{params['n_samples']}_M{params['n_causal_pairs']}_"
        f"h2{params['h2']:.2f}_r{ratio_str}_rho{params['rho']:.1f}_"
        f"{params['maf_type']}_bal{params['case_control_ratio']:.1f}"
    )


def compute_n_blocks_from_n_snps(n_snps: int) -> int:
    """
    Choose a valid n_blocks that divides n_snps (required by GenomeSimulator).

    We aim to keep a roughly constant block size consistent with:
        n_snps=1000 and N_BLOCKS_PER_1000_SNPS=50 => block size ~20.

    For small n_snps (e.g., QUICK_TEST), the old formula could produce n_blocks=0.
    This helper guarantees n_blocks >= 1 and n_snps % n_blocks == 0.
    """
    if n_snps <= 0:
        raise ValueError(f"n_snps must be positive, got {n_snps}")

    # Target block size implied by the 1000-SNP reference.
    target_block_size = max(1, int(round(1000 / N_BLOCKS_PER_1000_SNPS)))
    approx_blocks = max(1, int(round(n_snps / target_block_size)))
    approx_blocks = min(approx_blocks, n_snps)

    # Find the closest divisor <= approx_blocks (fast and deterministic).
    for n_blocks in range(approx_blocks, 0, -1):
        if n_snps % n_blocks == 0:
            return n_blocks
    return 1


def generate_scenarios() -> list:
    """Generate all scenario parameter combinations."""
    keys = list(GRID.keys())
    values = [GRID[k] for k in keys]
    scenarios = []
    for combo in product(*values):
        params = dict(zip(keys, combo))
        # Compute n_blocks based on n_snps (must divide n_snps; never 0)
        params["n_blocks"] = compute_n_blocks_from_n_snps(int(params["n_snps"]))
        scenarios.append(params)
    return scenarios


def create_rf_model(seed: int) -> RandomForestClassifier:
    """Create a RandomForest model with n_jobs=1 (for parallel safety)."""
    return RandomForestClassifier(
        n_estimators=200,
        random_state=seed,
        n_jobs=1,  # Single-threaded to avoid nested parallelism
        class_weight=None
    )


def create_bworf_model(seed: int, n_features: int):
    """
    Create a BWORF model with conservative, stable config.
    Adapts to available constructor arguments.
    Uses global config variables: BWORF_L1_STRENGTH, BWORF_N_ESTIMATORS, etc.
    """
    # Base kwargs that should always work
    kwargs = {
        "n_estimators": BWORF_N_ESTIMATORS,
        "random_state": seed,
    }
    
    # Try to add optional parameters
    import inspect
    try:
        sig = inspect.signature(BWORF_CLASS.__init__)
        param_names = set(sig.parameters.keys())
    except Exception:
        param_names = set()
    
    # Use config variables for key parameters
    # l1_strength MUST be > 0 to avoid division by zero
    optional_params = {
        "max_depth": BWORF_MAX_DEPTH,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "n_tries": BWORF_N_TRIES,
        "l1_strength": BWORF_L1_STRENGTH,  # Must be > 0!
        "weighted_bootstrap": False,
        "bootstrap_temperature": 1.0,
        "n_jobs": 1,  # Single-threaded to avoid nested parallelism
    }
    
    for param, value in optional_params.items():
        if param in param_names:
            kwargs[param] = value
    
    # Note: top_k is NOT passed to model; MI filtering is handled separately
    # BWORF in this study runs on raw features (no MI filtering)
    
    return BWORF_CLASS(**kwargs)


def get_proba_for_class_1(model, proba: np.ndarray) -> np.ndarray:
    """
    Robustly extract probability for class 1 from predict_proba output.
    
    Handles:
    - models with classes_ attribute
    - models without classes_ (fallback to column 1 if shape is (n,2))
    - raises clear error if cannot determine correct column
    """
    n_samples = proba.shape[0]
    n_cols = proba.shape[1] if proba.ndim == 2 else 1
    
    # Check for classes_ attribute
    if hasattr(model, 'classes_'):
        classes = list(model.classes_)
        if 1 in classes:
            idx_1 = classes.index(1)
            return proba[:, idx_1]
        elif n_cols == 2:
            # Binary but classes might be [0, something_else]; assume col 1 is positive
            return proba[:, 1]
        else:
            raise RuntimeError(
                f"Model classes_={classes} does not contain class 1, "
                f"and proba has {n_cols} columns (not binary). Cannot determine positive class."
            )
    else:
        # No classes_ attribute - use fallback logic
        if n_cols == 2:
            # Assume standard binary: col 0 = class 0, col 1 = class 1
            return proba[:, 1]
        elif n_cols == 1:
            # Single column - treat as P(y=1) directly
            return proba.ravel()
        else:
            raise RuntimeError(
                f"Model lacks classes_ attribute and proba has {n_cols} columns. "
                f"Cannot determine which column is P(y=1)."
            )


def compute_oof_predictions(model, X: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    """
    Compute out-of-fold predicted probabilities using K-fold CV.
    Returns array of shape (n_samples,) with P(y=1).
    
    Uses robust class detection via get_proba_for_class_1().
    """
    n_samples = len(y)
    oof_proba = np.full(n_samples, np.nan, dtype=np.float64)
    
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
    
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train = y[train_idx]
        
        # Clone model for this fold
        model_fold = model.__class__(**model.get_params()) if hasattr(model, 'get_params') else create_model_copy(model)
        model_fold.fit(X_train, y_train)
        
        # Predict probabilities and extract P(y=1) robustly
        proba = model_fold.predict_proba(X_val)
        oof_proba[val_idx] = get_proba_for_class_1(model_fold, proba)
    
    return oof_proba


def evaluate_repeated_holdout(
    method_name: str,
    X: np.ndarray,
    y: np.ndarray,
    causal_snps: np.ndarray,
    scenario_idx: int,
    replicate_id: int,
    base_seed: int,
    n_features: int,
) -> tuple[dict, list]:
    """
    Repeated stratified holdout evaluation over HOLDOUT_SEEDS.

    Returns:
        agg: dict with mean/std metrics for the method (mean columns match CSV_COLUMNS names)
        per_seed_rows: optional per-seed rows (for separate CSV) if WRITE_PER_SEED_ROWS=True
    """
    per_seed_rows = []

    # Collect per-seed metric values
    auc_vals = []
    brier_vals = []
    prec_vals = []
    rec_vals = []
    f1_vals = []
    calib_i_vals = []
    calib_s_vals = []
    mean_rank_vals = []
    median_rank_vals = []
    recall20_vals = []
    runtime_vals = []

    for split_seed in HOLDOUT_SEEDS:
        # Deterministic model seed per split + method
        model_seed = stable_hash_from_parts(MASTER_SEED, scenario_idx, replicate_id, split_seed, method_name)

        # Stratified train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=HOLDOUT_TEST_SIZE,
            stratify=y,
            random_state=split_seed,
        )

        t0 = time.perf_counter()

        # Create model for this split
        if method_name == "RF":
            model = create_rf_model(model_seed)
        elif method_name == "BWORF":
            model = create_bworf_model(model_seed, n_features)
        else:
            raise ValueError(f"Unknown method_name: {method_name}")

        # Fit + predict on test
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)
        y_proba = get_proba_for_class_1(model, proba)

        runtime = time.perf_counter() - t0

        # Metrics on test
        m = compute_metrics(y_test, y_proba)
        auc_vals.append(m["auc_roc"])
        brier_vals.append(m["brier"])
        prec_vals.append(m["precision_pos"])
        rec_vals.append(m["recall_pos"])
        f1_vals.append(m["f1_pos"])
        calib_i_vals.append(m["calib_intercept"])
        calib_s_vals.append(m["calib_slope"])
        runtime_vals.append(runtime)

        # Importance quality:
        # - RF: use built-in feature_importances_
        # - BWORF: skipped (NaN)
        if method_name == "RF":
            try:
                importances = model.feature_importances_
                mean_rank, median_rank, recall20 = compute_feature_metrics(importances, causal_snps)
            except Exception:
                mean_rank, median_rank, recall20 = np.nan, np.nan, np.nan
        else:
            mean_rank, median_rank, recall20 = np.nan, np.nan, np.nan

        mean_rank_vals.append(mean_rank)
        median_rank_vals.append(median_rank)
        recall20_vals.append(recall20)

        if WRITE_PER_SEED_ROWS:
            per_seed_rows.append(
                {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "version": VERSION,
                    "scenario_id": None,  # filled by caller
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

    # Aggregate mean/std across split seeds
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


def create_model_copy(model):
    """Create a copy of BWORF model for fold fitting."""
    # Get seed and recreate
    seed = getattr(model, 'random_state', 42)
    n_features = getattr(model, 'n_features_in_', 1000)
    return create_bworf_model(seed, n_features)


def compute_permutation_importance(model, X: np.ndarray, y: np.ndarray, 
                                    y_proba_baseline: np.ndarray, n_repeats: int,
                                    rng: np.random.Generator) -> np.ndarray:
    """
    Compute permutation importance for all features using AUC-ROC as the metric.
    
    Works for any model (RF, BWORF) regardless of whether it has feature_importances_.
    
    Args:
        model: Fitted model with predict_proba method
        X: Feature matrix (n_samples, n_features)
        y: True labels
        y_proba_baseline: Baseline predicted probabilities for class 1
        n_repeats: Number of permutation repeats per feature
        rng: Random number generator for reproducibility
    
    Returns:
        importances: Array of shape (n_features,) with importance scores
                    (higher = more important, i.e., larger AUC drop when permuted)
                    Returns None if baseline AUC cannot be computed
    """
    n_samples, n_features = X.shape
    
    # Compute baseline AUC
    try:
        if len(np.unique(y)) < 2:
            return None  # Single class - cannot compute AUC
        baseline_auc = roc_auc_score(y, y_proba_baseline)
    except Exception:
        return None
    
    importances = np.zeros(n_features, dtype=np.float64)
    
    for j in range(n_features):
        auc_drops = []
        for _ in range(n_repeats):
            # Create a copy of X with feature j permuted
            X_permuted = X.copy()
            X_permuted[:, j] = rng.permutation(X_permuted[:, j])
            
            # Predict with permuted feature
            try:
                proba_permuted = model.predict_proba(X_permuted)
                y_proba_permuted = get_proba_for_class_1(model, proba_permuted)
                auc_permuted = roc_auc_score(y, y_proba_permuted)
                auc_drops.append(baseline_auc - auc_permuted)
            except Exception:
                auc_drops.append(0.0)  # No drop if prediction fails
        
        importances[j] = np.mean(auc_drops)
    
    return importances


def compute_feature_metrics(importances: np.ndarray, causal_snps: np.ndarray) -> tuple:
    """
    Compute feature discovery metrics from importance scores.
    
    Returns:
        mean_rank: Mean rank of causal SNPs (1-indexed, lower is better)
        median_rank: Median rank of causal SNPs (1-indexed, lower is better)
        recall_at_20: Count of causal SNPs in top 20
    """
    if importances is None:
        return np.nan, np.nan, np.nan
    
    n_features = len(importances)
    # Rank: argsort descending (highest importance = rank 1)
    sorted_indices = np.argsort(importances)[::-1]
    # Create rank array: rank[feature_idx] = 1-indexed rank
    ranks = np.empty(n_features, dtype=int)
    ranks[sorted_indices] = np.arange(1, n_features + 1)
    
    # Ranks of causal SNPs
    causal_ranks = ranks[causal_snps]
    mean_rank = np.mean(causal_ranks)
    median_rank = np.median(causal_ranks)
    
    # Recall@20: how many causal SNPs in top 20
    top_20 = set(sorted_indices[:20])
    recall_at_20 = sum(1 for snp in causal_snps if snp in top_20)
    
    return mean_rank, median_rank, recall_at_20


def compute_calibration(y_true: np.ndarray, y_proba: np.ndarray) -> tuple:
    """
    Compute calibration slope and intercept via logistic regression.
    
    Fits: logit(p) -> y using logistic regression.
    Returns (intercept, slope) or (NaN, NaN) if fitting fails.
    """
    try:
        # Clip probabilities to avoid log(0)
        p_clipped = np.clip(y_proba, 1e-6, 1 - 1e-6)
        logit_p = np.log(p_clipped / (1 - p_clipped))
        
        # Check for single-class y
        if len(np.unique(y_true)) < 2:
            return np.nan, np.nan
        
        # Fit logistic regression: logit(p) as single feature -> y
        clf = SklearnLR(solver='lbfgs', max_iter=1000)
        clf.fit(logit_p.reshape(-1, 1), y_true)
        
        intercept = clf.intercept_[0]
        slope = clf.coef_[0, 0]
        return intercept, slope
    except Exception:
        return np.nan, np.nan


def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray) -> dict:
    """
    Compute all classification metrics including:
    - Predictive: AUC-ROC, Brier score
    - Class-specific (pos=1): precision, recall, F1
    - Calibration: intercept, slope
    """
    y_pred = (y_proba >= 0.5).astype(int)
    
    # Handle single-class case for AUC
    try:
        if len(np.unique(y_true)) < 2:
            auc_roc = np.nan
        else:
            auc_roc = roc_auc_score(y_true, y_proba)
    except Exception:
        auc_roc = np.nan
    
    # Brier score
    try:
        brier = brier_score_loss(y_true, y_proba)
    except Exception:
        brier = np.nan
    
    # Class-specific metrics for positive class (label=1)
    try:
        precision_pos = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        recall_pos = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        f1_pos = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    except Exception:
        precision_pos = np.nan
        recall_pos = np.nan
        f1_pos = np.nan
    
    # Calibration
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


# ==============================================================================
# BWORF Sanity Probe (CORRECTNESS check)
# ==============================================================================

def run_bworf_sanity_probe():
    """
    Verify BWORF predict_proba behavior on a tiny binary dataset.
    Raises RuntimeError if BWORF doesn't behave correctly for binary classification.
    """
    print("[SANITY PROBE] Verifying BWORF binary classification behavior...")
    print(f"  BWORF config: n_estimators={BWORF_N_ESTIMATORS}, max_depth={BWORF_MAX_DEPTH}, "
          f"n_tries={BWORF_N_TRIES}, l1_strength={BWORF_L1_STRENGTH}")
    
    # Validate l1_strength is positive (safety check)
    if BWORF_L1_STRENGTH <= 0:
        raise RuntimeError(
            f"BWORF_L1_STRENGTH must be > 0 to avoid division by zero. "
            f"Current value: {BWORF_L1_STRENGTH}"
        )
    
    # Generate tiny test data
    rng = np.random.default_rng(42)
    n_samples, n_features = 100, 20
    X_test = rng.standard_normal((n_samples, n_features))
    y_test = rng.integers(0, 2, size=n_samples)  # Binary {0, 1}
    
    # Create and fit BWORF
    try:
        model = create_bworf_model(seed=42, n_features=n_features)
        model.fit(X_test, y_test)
    except Exception as e:
        raise RuntimeError(f"BWORF failed to fit on binary data: {e}")
    
    # Check predict_proba output
    try:
        proba = model.predict_proba(X_test)
    except Exception as e:
        raise RuntimeError(f"BWORF predict_proba failed: {e}")
    
    # Validate shape
    if proba.ndim != 2:
        raise RuntimeError(
            f"BWORF predict_proba returned {proba.ndim}D array, expected 2D. "
            f"Shape: {proba.shape}"
        )
    
    if proba.shape[0] != n_samples:
        raise RuntimeError(
            f"BWORF predict_proba returned {proba.shape[0]} rows, expected {n_samples}"
        )
    
    if proba.shape[1] != 2:
        raise RuntimeError(
            f"BWORF predict_proba returned {proba.shape[1]} columns, expected 2 for binary. "
            f"This may indicate BWORF is treating labels as multiclass. "
            f"Ensure y contains only {{0, 1}} and BWORF handles binary correctly."
        )
    
    # Check classes_ attribute
    if hasattr(model, 'classes_'):
        classes = set(model.classes_)
        if classes != {0, 1}:
            raise RuntimeError(
                f"BWORF classes_ = {model.classes_}, expected {{0, 1}}. "
                f"This may cause incorrect probability column selection."
            )
        print(f"  classes_ = {list(model.classes_)} (OK)")
    else:
        print("  WARNING: BWORF lacks classes_ attribute; assuming column order [0, 1]")
    
    # Validate probabilities sum to 1
    row_sums = proba.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        raise RuntimeError(
            f"BWORF probabilities don't sum to 1. Row sums range: "
            f"[{row_sums.min():.4f}, {row_sums.max():.4f}]"
        )
    
    # Test extraction function
    try:
        proba_class1 = get_proba_for_class_1(model, proba)
        assert proba_class1.shape == (n_samples,), f"Unexpected shape: {proba_class1.shape}"
        assert np.all((proba_class1 >= 0) & (proba_class1 <= 1)), "Probabilities out of [0,1]"
    except Exception as e:
        raise RuntimeError(f"get_proba_for_class_1 failed: {e}")
    
    print(f"  predict_proba shape: {proba.shape} (OK)")
    print(f"  Probabilities sum to 1: OK")
    print(f"  P(y=1) extraction: OK")
    print("[SANITY PROBE] BWORF binary classification verified successfully.")
    return True


# ==============================================================================
# Resume Support: Load completed tasks from existing CSV
# ==============================================================================

def load_completed_tasks(csv_path: Path) -> set:
    """
    Load already-completed (scenario_id, replicate_id, method) tuples from CSV.
    Returns empty set if file doesn't exist.
    
    Uses pandas for efficiency if available, otherwise falls back to csv module.
    """
    if not csv_path.exists():
        return set()
    
    completed = set()
    
    try:
        import pandas as pd
        # Read only the columns we need for efficiency.
        # IMPORTANT: only treat rows as completed if error is empty.
        try:
            df = pd.read_csv(csv_path, usecols=['scenario_id', 'replicate_id', 'method', 'error'])
        except Exception:
            df = pd.read_csv(csv_path, usecols=['scenario_id', 'replicate_id', 'method'])
            df['error'] = ''

        for _, row in df.iterrows():
            err = row.get('error', '')
            err_str = '' if pd.isna(err) else str(err).strip()
            if err_str != '':
                continue  # keep failed rows eligible for rerun
            key = (str(row['scenario_id']), int(row['replicate_id']), str(row['method']))
            completed.add(key)
        print(f"[RESUME] Loaded {len(completed)} completed (error-free) task-method pairs from existing CSV")
    except ImportError:
        # Fallback to csv module
        import csv
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                err_str = str(row.get('error', '')).strip()
                if err_str != '':
                    continue  # keep failed rows eligible for rerun
                key = (row['scenario_id'], int(row['replicate_id']), row['method'])
                completed.add(key)
        print(f"[RESUME] Loaded {len(completed)} completed (error-free) task-method pairs (csv fallback)")
    except Exception as e:
        print(f"[RESUME] Warning: Could not load existing CSV for resume: {e}")
        return set()
    
    return completed


# ==============================================================================
# Single Replicate Runner
# ==============================================================================

def run_single_replicate(
    scenario_idx: int,
    scenario_params: dict,
    replicate_id: int,
    scenario_id: str,
    skip_methods: set = None
) -> tuple[list, list]:
    """
    Run a single replicate for both RF and BWORF.
    Returns list of result dicts (one per method).
    
    skip_methods: set of method names to skip (for resume support)
    """
    if skip_methods is None:
        skip_methods = set()
    
    seed = stable_hash_seed(MASTER_SEED, scenario_idx, replicate_id)
    timestamp_utc = datetime.now(timezone.utc).isoformat()
    
    results = []
    per_seed_rows_all = []  # optional (written to separate CSV when enabled)
    
    # Check if both methods should be skipped
    methods_to_run = [m for m in ["RF", "BWORF"] if m not in skip_methods]
    if not methods_to_run:
        return results, per_seed_rows_all  # Nothing to do
    
    # NaN row template for errors
    def make_nan_row(method: str, error_msg: str) -> dict:
        return {
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
            "auc_roc": np.nan,
            "auc_roc_std": np.nan,
            "brier": np.nan,
            "brier_std": np.nan,
            "precision_pos": np.nan,
            "precision_pos_std": np.nan,
            "recall_pos": np.nan,
            "recall_pos_std": np.nan,
            "f1_pos": np.nan,
            "f1_pos_std": np.nan,
            "calib_intercept": np.nan,
            "calib_intercept_std": np.nan,
            "calib_slope": np.nan,
            "calib_slope_std": np.nan,
            "mean_rank_causal": np.nan,
            "mean_rank_causal_std": np.nan,
            "median_rank_causal": np.nan,
            "median_rank_causal_std": np.nan,
            "recall_at_20_causal": np.nan,
            "recall_at_20_causal_std": np.nan,
            "runtime_seconds": np.nan,
            "runtime_seconds_std": np.nan,
            "error": error_msg
        }
    
    # Generate dataset
    try:
        X, y, causal_pairs, meta = simulate_dataset_v1(
            n_samples=scenario_params["n_samples"],
            n_snps=scenario_params["n_snps"],
            n_blocks=scenario_params["n_blocks"],
            rho=scenario_params["rho"],
            maf_type=scenario_params["maf_type"],
            n_causal_pairs=scenario_params["n_causal_pairs"],
            h2=scenario_params["h2"],
            signal_ratio=scenario_params["signal_ratio"],
            case_control_ratio=scenario_params["case_control_ratio"],
            random_state=seed
        )
    except Exception as e:
        # Data generation failed - write error rows for methods we need to run
        for method in methods_to_run:
            results.append(make_nan_row(method, f"Data generation failed: {str(e)}"))
        return results, per_seed_rows_all
    
    # Extract causal SNP indices (unique, sorted)
    causal_snps = np.unique(causal_pairs.flatten())
    n_features = X.shape[1]
    
    # Base result template (scenario info only)
    base_row = {
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
        "error": ""
    }
    
    # Repeated holdout evaluation over HOLDOUT_SEEDS:
    # - one dataset per replicate
    # - 5 stratified train/test splits
    # - aggregate mean/std into ONE row per method
    if "RF" in methods_to_run:
        try:
            agg, per_seed_rows = evaluate_repeated_holdout(
                method_name="RF",
                X=X,
                y=y,
                causal_snps=causal_snps,
                scenario_idx=scenario_idx,
                replicate_id=replicate_id,
                base_seed=seed,
                n_features=n_features,
            )
            row_rf = base_row.copy()
            row_rf.update({"method": "RF"})
            row_rf.update(agg)
            results.append(row_rf)

            if WRITE_PER_SEED_ROWS and per_seed_rows:
                for r in per_seed_rows:
                    r["scenario_id"] = scenario_id
                    r["n_samples"] = scenario_params["n_samples"]
                    r["n_snps"] = scenario_params["n_snps"]
                    r["n_blocks"] = scenario_params["n_blocks"]
                    r["rho"] = scenario_params["rho"]
                    r["maf_type"] = scenario_params["maf_type"]
                    r["n_causal_pairs"] = scenario_params["n_causal_pairs"]
                    r["h2"] = scenario_params["h2"]
                    r["signal_ratio"] = scenario_params["signal_ratio"]
                    r["case_control_ratio"] = scenario_params["case_control_ratio"]
                per_seed_rows_all.extend(per_seed_rows)
        except Exception as e:
            results.append(make_nan_row("RF", f"RF failed: {str(e)}"))

    if "BWORF" in methods_to_run:
        try:
            agg, per_seed_rows = evaluate_repeated_holdout(
                method_name="BWORF",
                X=X,
                y=y,
                causal_snps=causal_snps,
                scenario_idx=scenario_idx,
                replicate_id=replicate_id,
                base_seed=seed,
                n_features=n_features,
            )
            row_bworf = base_row.copy()
            row_bworf.update({"method": "BWORF"})
            row_bworf.update(agg)
            results.append(row_bworf)

            if WRITE_PER_SEED_ROWS and per_seed_rows:
                for r in per_seed_rows:
                    r["scenario_id"] = scenario_id
                    r["n_samples"] = scenario_params["n_samples"]
                    r["n_snps"] = scenario_params["n_snps"]
                    r["n_blocks"] = scenario_params["n_blocks"]
                    r["rho"] = scenario_params["rho"]
                    r["maf_type"] = scenario_params["maf_type"]
                    r["n_causal_pairs"] = scenario_params["n_causal_pairs"]
                    r["h2"] = scenario_params["h2"]
                    r["signal_ratio"] = scenario_params["signal_ratio"]
                    r["case_control_ratio"] = scenario_params["case_control_ratio"]
                per_seed_rows_all.extend(per_seed_rows)
        except Exception as e:
            results.append(make_nan_row("BWORF", f"BWORF failed: {str(e)}"))

    return results, per_seed_rows_all


# ==============================================================================
# CSV I/O
# ==============================================================================

CSV_COLUMNS = [
    "timestamp_utc", "version", "scenario_id", "replicate_id", "seed",
    "n_samples", "n_snps", "n_blocks", "rho", "maf_type",
    "n_causal_pairs", "h2", "signal_ratio", "case_control_ratio",
    "method",
    # Predictive performance
    "auc_roc", "auc_roc_std",
    "brier", "brier_std",
    # Class-specific (pos=1)
    "precision_pos", "precision_pos_std",
    "recall_pos", "recall_pos_std",
    "f1_pos", "f1_pos_std",
    # Calibration
    "calib_intercept", "calib_intercept_std",
    "calib_slope", "calib_slope_std",
    # Variable importance quality
    "mean_rank_causal", "mean_rank_causal_std",
    "median_rank_causal", "median_rank_causal_std",
    "recall_at_20_causal", "recall_at_20_causal_std",
    # Computational cost
    "runtime_seconds", "runtime_seconds_std",
    "error"
]

# Optional per-seed output (one row per holdout split). Written only when WRITE_PER_SEED_ROWS=True.
PER_SEED_CSV_COLUMNS = [
    "timestamp_utc", "version", "scenario_id", "replicate_id", "seed", "split_seed",
    "n_samples", "n_snps", "n_blocks", "rho", "maf_type",
    "n_causal_pairs", "h2", "signal_ratio", "case_control_ratio",
    "method",
    "auc_roc", "brier",
    "precision_pos", "recall_pos", "f1_pos",
    "calib_intercept", "calib_slope",
    "mean_rank_causal", "median_rank_causal", "recall_at_20_causal",
    "runtime_seconds",
    "error",
]


def write_csv_header(csv_path: Path, columns: list):
    """
    Ensure CSV exists and has the expected header.

    Safety: if the file exists with a different header, raise a clear error
    (prevents corrupt mixed-schema CSVs).
    """
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
                # Empty file: write header
                with open(csv_path, "w", encoding="utf-8") as f:
                    f.write(expected_header + "\n")
        else:
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write(expected_header + "\n")


def append_csv_rows(csv_path: Path, rows: list, columns: list):
    """
    Append rows to CSV file with thread safety.
    Uses lock to prevent corruption if called from multiple threads.
    """
    if not rows:
        return
    
    with CSV_LOCK:
        with open(csv_path, "a", encoding="utf-8") as f:
            for row in rows:
                values = []
                for col in columns:
                    val = row.get(col, "")
                    # Handle special values
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
                    # Escape commas in strings
                    if "," in val or '"' in val:
                        val = '"' + val.replace('"', '""') + '"'
                    values.append(val)
                f.write(",".join(values) + "\n")


def count_csv_data_rows(csv_path: Path) -> int:
    """Count data rows in CSV (excluding header)."""
    if not csv_path.exists():
        return 0
    with open(csv_path, "r", encoding="utf-8") as f:
        n_lines = sum(1 for _ in f)
    return max(0, n_lines - 1)


# ==============================================================================
# Config and Summary Writers
# ==============================================================================

def write_run_config(config_path: Path, scenarios: list, start_time: datetime):
    """Write run configuration JSON."""
    # Model hyperparameters - use config variables
    rf_params = {
        "n_estimators": 200,
        "n_jobs": 1,
        "class_weight": None,
    }
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
        "evaluation": {
            "protocol": "repeated_holdout",
            "holdout_seeds": HOLDOUT_SEEDS,
            "test_size": HOLDOUT_TEST_SIZE,
            "n_splits": len(HOLDOUT_SEEDS),
            "write_per_seed_rows": WRITE_PER_SEED_ROWS,
        },
        "n_replicates": N_REPLICATES,
        "n_scenarios": len(scenarios),
        "batch_size": BATCH_SIZE,
        "feature_importance": {
            "RF": "built-in Gini importance (free)",
            "BWORF": "skipped (too expensive, not primary goal)",
        },
        "grid": {k: [float(v) if isinstance(v, (float, np.floating)) and not np.isinf(v) 
                     else ("inf" if isinstance(v, (float, np.floating)) and np.isinf(v) else v) 
                     for v in vals] 
                for k, vals in GRID.items()},
        "n_blocks_per_1000_snps": N_BLOCKS_PER_1000_SNPS,
        "model_hyperparams": {
            "RF": rf_params,
            "BWORF": bworf_params,
        },
        "bworf_module_path": BWORF_MODULE_PATH,
        "timestamp_start_utc": start_time.isoformat(),
        "python_version": sys.version,
        "package_versions": {
            "numpy": np.__version__,
            "sklearn": SKLEARN_VERSION,
        },
        "thread_limits": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
    }
    
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def write_run_summary(summary_path: Path, scenarios: list, start_time: datetime, 
                      end_time: datetime = None, actual_rows: int = None,
                      skipped_tasks: int = 0):
    """Write human-readable summary markdown."""
    n_scenarios = len(scenarios)
    expected_rows = n_scenarios * N_REPLICATES * 2  # 2 methods
    
    content = f"""# Simulation Study v1 - Run Summary

## Configuration

- **Mode**: {"QUICK_TEST" if QUICK_TEST else "FULL"}
- **Version**: {VERSION}
- **Master Seed**: {MASTER_SEED}
- **Evaluation**: Repeated stratified holdout (n_splits={len(HOLDOUT_SEEDS)}, test_size={HOLDOUT_TEST_SIZE})
- **Holdout seeds**: {HOLDOUT_SEEDS}
- **Replicates per Scenario**: {N_REPLICATES}
- **Parallel Cores**: {N_CORES}
- **Batch Size**: {BATCH_SIZE} (tasks per checkpoint)

## Grid

| Parameter | Values |
|-----------|--------|
"""
    for key, vals in GRID.items():
        vals_str = ", ".join(str(v) for v in vals)
        content += f"| {key} | {vals_str} |\n"
    
    content += f"""
## Scale

- **Number of Scenarios**: {n_scenarios}
- **Total Replicates**: {n_scenarios * N_REPLICATES}
- **Expected Rows**: {expected_rows} (scenarios x replicates x 2 methods)
"""
    
    if skipped_tasks > 0:
        content += f"- **Skipped (already completed)**: {skipped_tasks} task-method pairs\n"
    
    if actual_rows is not None:
        content += f"- **Actual Rows in CSV**: {actual_rows}\n"
    
    content += f"""
## Reproducibility

Seeds are derived deterministically using MD5 hash:
```
seed = md5(f"{{MASTER_SEED}}_{{scenario_idx}}_{{replicate_id}}")[:8] (as int)
```

This ensures identical results when re-running with the same configuration.

## Thread Safety

Environment variables set to prevent BLAS/OpenMP oversubscription:
- OMP_NUM_THREADS=1
- MKL_NUM_THREADS=1
- OPENBLAS_NUM_THREADS=1

## Outputs

- `simulation_results_v1.csv` - Main results (long format, incremental writes)
- `run_config_v1.json` - Full configuration
- `run_summary_v1.md` - This file

## Timestamps

- **Start**: {start_time.isoformat()}
"""
    
    if end_time is not None:
        duration = (end_time - start_time).total_seconds()
        content += f"- **End**: {end_time.isoformat()}\n"
        content += f"- **Duration**: {duration:.1f} seconds ({duration/60:.1f} minutes)\n"
    
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(content)


# ==============================================================================
# Main Runner with Batch Processing & Checkpointing
# ==============================================================================

def main():
    print("=" * 70)
    print("Simulation Study v1 - Runner (with checkpointing)")
    print("=" * 70)
    print(f"Mode: {'QUICK_TEST' if QUICK_TEST else 'FULL'}")
    print(f"Master Seed: {MASTER_SEED}")
    print(f"Cores: {N_CORES}")
    print(f"Evaluation: repeated holdout | n_splits={len(HOLDOUT_SEEDS)} | test_size={HOLDOUT_TEST_SIZE}")
    print(f"Holdout seeds: {HOLDOUT_SEEDS}")
    print(f"Replicates: {N_REPLICATES}")
    print(f"Batch Size: {BATCH_SIZE}")
    print()
    
    # Create output directories
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    # File paths
    csv_path = OUTPUTS_DIR / "simulation_results_v1.csv"
    config_path = OUTPUTS_DIR / "run_config_v1.json"
    summary_path = OUTPUTS_DIR / "run_summary_v1.md"
    per_seed_csv_path = OUTPUTS_DIR / "simulation_results_v1_per_seed.csv"
    
    # ==== BWORF Sanity Probe (runs once at startup) ====
    run_bworf_sanity_probe()
    print()
    
    # ==== Load completed tasks for resume support ====
    completed_tasks = load_completed_tasks(csv_path)
    
    # Generate scenarios
    scenarios = generate_scenarios()
    n_scenarios = len(scenarios)
    expected_rows = n_scenarios * N_REPLICATES * 2
    
    print(f"Scenarios: {n_scenarios}")
    print(f"Total replicates: {n_scenarios * N_REPLICATES}")
    print(f"Expected rows: {expected_rows}")
    print()
    
    # Start time
    start_time = datetime.now(timezone.utc)
    
    # Write initial config and summary
    write_run_config(config_path, scenarios, start_time)
    write_run_summary(summary_path, scenarios, start_time, skipped_tasks=len(completed_tasks))
    print(f"Config written: {config_path}")
    print(f"Summary written: {summary_path}")
    print()
    
    # Initialize CSV header (and validate schema if file exists)
    write_csv_header(csv_path, CSV_COLUMNS)
    print(f"CSV (aggregate): {csv_path}")

    if WRITE_PER_SEED_ROWS:
        write_csv_header(per_seed_csv_path, PER_SEED_CSV_COLUMNS)
        print(f"CSV (per-seed): {per_seed_csv_path}")
    print()

    # Count existing rows (includes any prior error rows)
    existing_rows_before_run = count_csv_data_rows(csv_path)
    if existing_rows_before_run > 0:
        print(f"[RESUME] Existing rows in CSV (incl. errors): {existing_rows_before_run}")
        print()
    
    # Build task list with skip info: (scenario_idx, scenario_params, replicate_id, scenario_id, skip_methods)
    tasks = []
    skipped_count = 0
    for scenario_idx, scenario_params in enumerate(scenarios):
        scenario_id = build_scenario_id(scenario_params)
        for replicate_id in range(N_REPLICATES):
            # Determine which methods to skip (already completed)
            skip_methods = set()
            for method in ["RF", "BWORF"]:
                key = (scenario_id, replicate_id, method)
                if key in completed_tasks:
                    skip_methods.add(method)
                    skipped_count += 1
            
            # Only add task if at least one method needs to run
            if len(skip_methods) < 2:
                tasks.append((scenario_idx, scenario_params, replicate_id, scenario_id, skip_methods))
    
    total_tasks = len(tasks)
    print(f"Tasks to run: {total_tasks} (skipped {skipped_count} already-completed method runs)")
    
    if total_tasks == 0:
        print("All tasks already completed. Nothing to do.")
        end_time = datetime.now(timezone.utc)
        # Count actual rows in CSV file (may include prior error rows)
        actual_rows_in_file = count_csv_data_rows(csv_path)
        write_run_summary(summary_path, scenarios, start_time, end_time, actual_rows_in_file, skipped_count)
        return
    
    print("-" * 70)
    
    # ==== Batch Processing with Incremental Writes ====
    # Process tasks in batches to checkpoint progress frequently
    rows_written_this_run = 0
    success_keys_written = set()  # track newly completed (error-free) keys for validation
    
    def process_task(task):
        """Worker function for parallel execution."""
        scenario_idx, scenario_params, replicate_id, scenario_id, skip_methods = task
        agg_rows, per_seed_rows = run_single_replicate(
            scenario_idx, scenario_params, replicate_id, scenario_id, skip_methods
        )
        return (scenario_idx, replicate_id, agg_rows, per_seed_rows)
    
    # Process in batches
    n_batches = (total_tasks + BATCH_SIZE - 1) // BATCH_SIZE
    effective_n_jobs = N_CORES  # may be reduced if workers crash
    
    for batch_idx in range(n_batches):
        batch_start = batch_idx * BATCH_SIZE
        batch_end = min(batch_start + BATCH_SIZE, total_tasks)
        batch_tasks = tasks[batch_start:batch_end]
        
        print(f"[Batch {batch_idx + 1}/{n_batches}] Processing {len(batch_tasks)} tasks...")
        
        # Run batch in parallel with retry fallback.
        # If a loky worker is killed (e.g., OOM), retry with fewer processes.
        batch_results = None
        n_jobs = effective_n_jobs
        while True:
            try:
                batch_results = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
                    delayed(process_task)(task) for task in batch_tasks
                )
                effective_n_jobs = n_jobs  # keep reduced value for subsequent batches
                break
            except Exception as e:
                is_terminated_worker = (
                    (TerminatedWorkerError is not None and isinstance(e, TerminatedWorkerError))
                    or (e.__class__.__name__ == "TerminatedWorkerError")
                )
                if not is_terminated_worker:
                    raise

                if n_jobs <= 1:
                    # Last resort: run this batch sequentially so the run can continue.
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
        
        # Immediately write results to CSV (CHECKPOINT)
        all_rows = []
        all_seed_rows = []
        for scenario_idx, replicate_id, agg_rows, per_seed_rows in batch_results:
            all_rows.extend(agg_rows)
            if WRITE_PER_SEED_ROWS and per_seed_rows:
                all_seed_rows.extend(per_seed_rows)

            # Track successful keys for row-count validation (ignore error rows)
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
        
        # Print progress for this batch
        for scenario_idx, replicate_id, agg_rows, _per_seed_rows in batch_results:
            if len(agg_rows) >= 1:
                for res in agg_rows:
                    method = res.get("method", "?")
                    err = str(res.get("error", "")).strip()
                    if err:
                        print(f"  Scen {scenario_idx+1:3d}/{n_scenarios} | "
                              f"Rep {replicate_id+1:2d}/{N_REPLICATES} | "
                              f"{method:5s} ERROR: {err}")
                    else:
                        auc = res.get("auc_roc", np.nan)
                        auc_str = f"{auc:.3f}" if not np.isnan(auc) else "ERR"
                        print(f"  Scen {scenario_idx+1:3d}/{n_scenarios} | "
                              f"Rep {replicate_id+1:2d}/{N_REPLICATES} | "
                              f"{method:5s} AUC={auc_str}")
        
        print(f"  -> Checkpoint: {rows_written_this_run} rows written to CSV")
    
    # End time
    end_time = datetime.now(timezone.utc)
    duration = (end_time - start_time).total_seconds()
    
    print("-" * 70)
    print(f"Completed in {duration:.1f} seconds ({duration/60:.1f} minutes)")
    print(f"Rows written this run: {rows_written_this_run}")
    
    # Row counting:
    # - total rows in file can include prior error rows
    # - unique completed keys should match expected_rows when run is complete
    total_rows_in_file = existing_rows_before_run + rows_written_this_run
    unique_completed_success = len(completed_tasks.union(success_keys_written))
    print(f"Total rows in CSV file: {total_rows_in_file} (may include prior error rows)")
    print(f"Unique completed (error-free) rows: {unique_completed_success} (expected: {expected_rows})")
    
    # Update summary with final stats
    write_run_summary(summary_path, scenarios, start_time, end_time, total_rows_in_file, skipped_count)
    
    # Validation
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
