#!/usr/bin/env python3
"""
Simulation Study v3 - Runner Script (Sol/HPC Ready)

How to run:
    # Local (6 cores default)
    python run_simulation_study_v3.py

    # Override cores
    python run_simulation_study_v3.py --n-cores 12

    # Slurm array job (sharding)
    python run_simulation_study_v3.py --shard-id 0 --n-shards 10

    # Or use SLURM env vars automatically:
    #   SLURM_CPUS_PER_TASK -> n_cores
    #   SLURM_ARRAY_TASK_ID -> shard_id
    #   SLURM_ARRAY_TASK_COUNT -> n_shards

Purpose
-------
This runner performs a mechanistic binary genetics simulation benchmark comparing:
  - rf: sklearn RandomForestClassifier
  - orf: Oblique Random Forest via Treeple (models/orf_treeple.py)
  - bworf: BWORF from models/bworf_with_mi.py

Metrics computed for ALL methods:
  Predictive: auc, brier
  Causal recovery (via permutation importance):
    - mean_rank_causal, median_rank_causal, recall_at_20

Outputs
-------
  - v3/outputs/simulation_results_v3.csv
  - v3/outputs/run_config_v3.json
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
import argparse
import threading
from pathlib import Path
from datetime import datetime, timezone
from itertools import product

import numpy as np

# ==============================================================================
# V3 GRID - LOCKED (Single Source of Truth)
# ==============================================================================

H2_LIST = [0.005, 0.010, 0.025, 0.050, 0.100, 0.150, 0.200]
P_LIST = [1000]
N_LIST = [500, 2500]
M_LIST = [2, 5, 10]
RHO_LIST = [0.2, 0.8]
SIGNAL_RATIO_LIST = [float("inf"), 1.0, 0.5]
MAF_TYPE = "fixed"
MAF_FIXED = 0.3
CASE_CONTROL_RATIO = 1.0

HOLDOUT_SEEDS = [42, 0, 12, 100, 90]
HOLDOUT_TEST_SIZE = 0.3
N_REPLICATES = 5

# ==============================================================================
# Configuration
# ==============================================================================

MASTER_SEED = 12345
VERSION = "v3"

# Model hyperparameters
N_ESTIMATORS = 2000  # Shared across rf and orf
BWORF_N_ESTIMATORS = 100
BWORF_L1_STRENGTH = 1.0
BWORF_N_TRIES = 10
BWORF_MAX_DEPTH = None

# Permutation importance settings (v2 fast settings)
PERM_N_REPEATS = 1
PERM_SUBSAMPLE_N = 500
PERM_SCORING = "roc_auc"
PERM_TOPK = 20

# Fixed parameters for genome simulation
N_BLOCKS_PER_1000_SNPS = 50

# ==============================================================================
# Paths
# ==============================================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
OUTPUTS_DIR = BASE_DIR / "outputs"

V1_DIR = PROJECT_ROOT / "v1"
ENGINE_PATH = V1_DIR / "simulation_utils_v1.py"
if not ENGINE_PATH.exists():
    raise FileNotFoundError(f"Expected engine at: {ENGINE_PATH}")
sys.path.insert(0, str(V1_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

# ==============================================================================
# Imports (after path setup)
# ==============================================================================

try:
    from simulation_utils_v1 import simulate_dataset_v1
except ImportError as e:
    print(f"ERROR: Cannot import simulation engine: {e}")
    sys.exit(1)

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, brier_score_loss, make_scorer
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
    TerminatedWorkerError = None

# ORF import
ORF_AVAILABLE = False
ORF_IMPORT_ERROR = None
try:
    from models.orf_treeple import make_orf, predict_proba_binary, check_orf_available
    ORF_AVAILABLE = check_orf_available()
    if not ORF_AVAILABLE:
        ORF_IMPORT_ERROR = "treeple not installed"
except ImportError as e:
    ORF_IMPORT_ERROR = str(e)

# BWORF import
BWORF_CLASS = None
BWORF_IMPORT_ERROR = None
try:
    from models.bworf_with_mi import BWORFClassifier as BWORF_CLASS
except ImportError as e:
    BWORF_IMPORT_ERROR = str(e)

# ==============================================================================
# CLI Argument Parsing
# ==============================================================================


def parse_args():
    parser = argparse.ArgumentParser(description="Simulation Study v3 Runner")
    parser.add_argument(
        "--n-cores",
        type=int,
        default=None,
        help="Number of parallel cores (default: SLURM_CPUS_PER_TASK or 6)",
    )
    parser.add_argument(
        "--shard-id",
        type=int,
        default=None,
        help="Shard ID for array jobs (default: SLURM_ARRAY_TASK_ID or 0)",
    )
    parser.add_argument(
        "--n-shards",
        type=int,
        default=None,
        help="Total number of shards (default: SLURM_ARRAY_TASK_COUNT or 1)",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run minimal smoke test only",
    )
    return parser.parse_args()


def get_runtime_config(args):
    """Resolve runtime config from CLI args and environment."""
    # n_cores: CLI > SLURM_CPUS_PER_TASK > 6
    if args.n_cores is not None:
        n_cores = args.n_cores
    else:
        n_cores = int(os.environ.get("SLURM_CPUS_PER_TASK", "6"))

    # shard_id: CLI > SLURM_ARRAY_TASK_ID > 0
    if args.shard_id is not None:
        shard_id = args.shard_id
    else:
        shard_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))

    # n_shards: CLI > SLURM_ARRAY_TASK_COUNT > 1
    if args.n_shards is not None:
        n_shards = args.n_shards
    else:
        n_shards = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", "1"))

    return {
        "n_cores": n_cores,
        "shard_id": shard_id,
        "n_shards": n_shards,
        "smoke_test": args.smoke_test,
    }


# ==============================================================================
# Scenario ID and JSON
# ==============================================================================


def compute_scenario_id(scenario_dict: dict) -> str:
    """Deterministic 12-char hex ID from scenario parameters."""
    # Convert inf to string for JSON serialization
    clean_dict = {}
    for k, v in scenario_dict.items():
        if isinstance(v, float) and np.isinf(v):
            clean_dict[k] = "inf"
        else:
            clean_dict[k] = v
    json_str = json.dumps(clean_dict, sort_keys=True)
    return hashlib.sha1(json_str.encode()).hexdigest()[:12]


def scenario_to_json(scenario_dict: dict) -> str:
    """Compact JSON string for scenario (for CSV column)."""
    clean_dict = {}
    for k, v in scenario_dict.items():
        if isinstance(v, float) and np.isinf(v):
            clean_dict[k] = "inf"
        else:
            clean_dict[k] = v
    return json.dumps(clean_dict, sort_keys=True, separators=(",", ":"))


# ==============================================================================
# Grid and Scenario Generation
# ==============================================================================


def compute_n_blocks(n_snps: int) -> int:
    """Choose n_blocks that divides n_snps evenly."""
    target_block_size = max(1, int(round(1000 / N_BLOCKS_PER_1000_SNPS)))
    approx_blocks = max(1, int(round(n_snps / target_block_size)))
    approx_blocks = min(approx_blocks, n_snps)
    for n_blocks in range(approx_blocks, 0, -1):
        if n_snps % n_blocks == 0:
            return n_blocks
    return 1


def generate_all_scenarios() -> list:
    """Generate full scenario list from locked v3 grid (deterministic order)."""
    scenarios = []
    # Sorted product for determinism
    for p in sorted(P_LIST):
        for n in sorted(N_LIST):
            for m in sorted(M_LIST):
                for h2 in sorted(H2_LIST):
                    for rho in sorted(RHO_LIST):
                        for sr in sorted(SIGNAL_RATIO_LIST, key=lambda x: (x == float("inf"), x)):
                            scenario = {
                                "p": p,
                                "n_samples": n,
                                "m": m,
                                "h2": h2,
                                "rho": rho,
                                "signal_ratio": sr,
                                "maf_type": MAF_TYPE,
                                "maf_fixed": MAF_FIXED,
                                "case_control_ratio": CASE_CONTROL_RATIO,
                                "n_blocks": compute_n_blocks(p),
                            }
                            scenarios.append(scenario)
    return scenarios


def filter_scenarios_by_shard(scenarios: list, shard_id: int, n_shards: int) -> list:
    """Select scenarios for this shard."""
    if n_shards <= 1:
        return scenarios
    return [s for i, s in enumerate(scenarios) if i % n_shards == shard_id]


# ==============================================================================
# Seed Generation
# ==============================================================================


def stable_hash(*parts) -> int:
    """Stable 32-bit hash from parts."""
    data = "_".join(str(p) for p in parts).encode("utf-8")
    return int(hashlib.md5(data).hexdigest()[:8], 16)


# ==============================================================================
# Model Creation
# ==============================================================================


def create_model(method: str, seed: int, n_features: int):
    """Create model for given method."""
    if method == "rf":
        return RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            max_features="sqrt",
            random_state=seed,
            n_jobs=1,
        )
    elif method == "orf":
        if not ORF_AVAILABLE:
            raise RuntimeError(f"ORF not available: {ORF_IMPORT_ERROR}")
        return make_orf(
            random_state=seed,
            n_estimators=N_ESTIMATORS,
            max_depth=None,
            n_jobs=1,
        )
    elif method == "bworf":
        if BWORF_CLASS is None:
            raise RuntimeError(f"BWORF not available: {BWORF_IMPORT_ERROR}")
        import inspect
        kwargs = {"n_estimators": BWORF_N_ESTIMATORS, "random_state": seed}
        try:
            sig = inspect.signature(BWORF_CLASS.__init__)
            param_names = set(sig.parameters.keys())
        except Exception:
            param_names = set()
        optional = {
            "max_depth": BWORF_MAX_DEPTH,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "n_tries": BWORF_N_TRIES,
            "l1_strength": BWORF_L1_STRENGTH,
            "weighted_bootstrap": False,
            "n_jobs": 1,
        }
        for k, v in optional.items():
            if k in param_names:
                kwargs[k] = v
        return BWORF_CLASS(**kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")


# ==============================================================================
# Probability Extraction (Robust)
# ==============================================================================


def get_proba_class1(model, X: np.ndarray, method: str) -> np.ndarray:
    """
    Get P(y=1) probabilities with validation.
    Returns shape (n,) array.
    """
    n_samples = X.shape[0]

    if method == "orf":
        proba = predict_proba_binary(model, X)
    else:
        if not hasattr(model, "predict_proba"):
            raise RuntimeError(f"{method}: model lacks predict_proba")
        proba = model.predict_proba(X)

    # Validate shape
    if proba.ndim != 2 or proba.shape[1] != 2:
        raise RuntimeError(f"{method}: predict_proba shape {proba.shape}, expected (n, 2)")

    if proba.shape[0] != n_samples:
        raise RuntimeError(f"{method}: predict_proba has {proba.shape[0]} rows, expected {n_samples}")

    # Check for NaN
    if np.any(np.isnan(proba)):
        raise RuntimeError(f"{method}: predict_proba contains NaN")

    # Validate rows sum to ~1
    row_sums = proba.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-4):
        print(f"  WARNING [{method}]: row sums range [{row_sums.min():.4f}, {row_sums.max():.4f}]")

    # Find P(y=1) column using classes_
    if hasattr(model, "classes_"):
        classes = list(model.classes_)
        if 1 in classes:
            return proba[:, classes.index(1)]
        elif len(classes) == 2:
            return proba[:, 1]
    return proba[:, 1]


# ==============================================================================
# Metrics
# ==============================================================================


def compute_auc_brier(y_true: np.ndarray, y_proba: np.ndarray) -> dict:
    """Compute AUC and Brier score."""
    y_true = np.asarray(y_true).astype(int)
    result = {"auc": np.nan, "brier": np.nan, "status": "ok", "error_msg": ""}

    # AUC
    try:
        if len(np.unique(y_true)) < 2:
            result["auc"] = np.nan
            result["status"] = "fail_auc"
            result["error_msg"] = "single class in y_true"
        else:
            result["auc"] = float(roc_auc_score(y_true, y_proba))
    except Exception as e:
        result["auc"] = np.nan
        result["status"] = "fail_auc"
        result["error_msg"] = str(e)

    # Brier
    try:
        result["brier"] = float(brier_score_loss(y_true, y_proba))
    except Exception:
        result["brier"] = np.nan

    return result


# ==============================================================================
# Permutation Importance
# ==============================================================================


def stratified_subsample(y: np.ndarray, n_target: int, rng: np.random.Generator) -> np.ndarray:
    """Stratified subsample indices."""
    y = np.asarray(y).astype(int)
    n = len(y)
    if n_target is None or n_target <= 0 or n <= n_target:
        return np.arange(n)

    classes = np.unique(y)
    if len(classes) < 2:
        return np.arange(n)

    idx_by_class = {c: np.where(y == c)[0] for c in classes}
    c0, c1 = classes[0], classes[1]
    n0_total, n1_total = len(idx_by_class[c0]), len(idx_by_class[c1])

    n0 = max(1, int(round(n_target * n0_total / n)))
    n1 = n_target - n0
    if n1 < 1:
        n1 = 1
        n0 = n_target - 1

    n0 = min(n0, n0_total)
    n1 = min(n1, n1_total)

    sel0 = rng.choice(idx_by_class[c0], size=n0, replace=False)
    sel1 = rng.choice(idx_by_class[c1], size=n1, replace=False)
    return np.concatenate([sel0, sel1])


def permutation_importance_auc(
    model, X: np.ndarray, y: np.ndarray, method: str, perm_seed: int
) -> tuple:
    """
    Compute permutation importance using AUC drop on test subsample.
    Returns (importances array, baseline_auc) or (None, None) on failure.
    """
    rng = np.random.default_rng(perm_seed)
    sub_idx = stratified_subsample(y, PERM_SUBSAMPLE_N, rng)

    X_sub = np.ascontiguousarray(X[sub_idx])
    y_sub = y[sub_idx]

    if len(np.unique(y_sub)) < 2:
        return None, None

    try:
        proba_base = get_proba_class1(model, X_sub, method)
        auc_base = float(roc_auc_score(y_sub, proba_base))
    except Exception:
        return None, None

    n_rows, n_features = X_sub.shape
    importances = np.zeros(n_features, dtype=np.float64)

    for j in range(n_features):
        col_orig = X_sub[:, j].copy()
        drops = []
        for r in range(PERM_N_REPEATS):
            feat_seed = stable_hash(perm_seed, j, r)
            feat_rng = np.random.default_rng(feat_seed)
            X_sub[:, j] = col_orig[feat_rng.permutation(n_rows)]
            try:
                proba_perm = get_proba_class1(model, X_sub, method)
                auc_perm = float(roc_auc_score(y_sub, proba_perm))
                drops.append(auc_base - auc_perm)
            except Exception:
                drops.append(0.0)
        X_sub[:, j] = col_orig
        importances[j] = float(np.mean(drops)) if drops else 0.0

    return importances, auc_base


def compute_rank_metrics(importances: np.ndarray, causal_snps: np.ndarray) -> dict:
    """Compute causal rank metrics from importance scores."""
    P = len(importances)
    causal_snps = np.asarray(causal_snps, dtype=int)

    if len(causal_snps) == 0 or np.any(causal_snps < 0) or np.any(causal_snps >= P):
        return {
            "mean_rank_causal": np.nan,
            "median_rank_causal": np.nan,
            "recall_at_20": np.nan,
        }

    imp = np.where(np.isnan(importances), -np.inf, importances)
    order = np.argsort(-imp, kind="mergesort")
    ranks = np.empty(P, dtype=int)
    ranks[order] = np.arange(1, P + 1)

    causal_ranks = ranks[causal_snps]
    return {
        "mean_rank_causal": float(np.mean(causal_ranks)),
        "median_rank_causal": float(np.median(causal_ranks)),
        "recall_at_20": float(np.mean(causal_ranks <= PERM_TOPK)),
    }


# ==============================================================================
# Single Evaluation
# ==============================================================================


def evaluate_single(
    method: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    causal_snps: np.ndarray,
    model_seed: int,
    perm_seed: int,
) -> dict:
    """Evaluate single method on single train/test split."""
    result = {
        "auc": np.nan,
        "brier": np.nan,
        "mean_rank_causal": np.nan,
        "median_rank_causal": np.nan,
        "recall_at_20": np.nan,
        "fit_seconds": np.nan,
        "perm_seconds": np.nan,
        "status": "ok",
        "error_msg": "",
    }

    n_features = X_train.shape[1]

    # Fit
    t0 = time.perf_counter()
    try:
        model = create_model(method, model_seed, n_features)
        model.fit(X_train, y_train)
    except Exception as e:
        result["status"] = "fail_fit"
        result["error_msg"] = str(e)
        return result
    result["fit_seconds"] = time.perf_counter() - t0

    # Predict
    try:
        y_proba = get_proba_class1(model, X_test, method)
    except Exception as e:
        result["status"] = "fail_predict"
        result["error_msg"] = str(e)
        return result

    # Metrics
    m = compute_auc_brier(y_test, y_proba)
    result["auc"] = m["auc"]
    result["brier"] = m["brier"]
    if m["status"] != "ok":
        result["status"] = m["status"]
        result["error_msg"] = m["error_msg"]

    # Permutation importance
    t1 = time.perf_counter()
    try:
        importances, _ = permutation_importance_auc(model, X_test, y_test, method, perm_seed)
        if importances is not None:
            ranks = compute_rank_metrics(importances, causal_snps)
            result.update(ranks)
    except Exception as e:
        result["error_msg"] = f"perm_failed: {e}"
    result["perm_seconds"] = time.perf_counter() - t1

    return result


# ==============================================================================
# CSV I/O
# ==============================================================================

CSV_COLUMNS = [
    "scenario_id",
    "scenario_json",
    "method",
    "seed",
    "replicate",
    "n_samples",
    "p",
    "m",
    "h2",
    "rho",
    "signal_ratio",
    "maf_type",
    "case_control_ratio",
    "n_features",
    "n_causal_snps",
    "auc",
    "brier",
    "mean_rank_causal",
    "median_rank_causal",
    "recall_at_20",
    "fit_seconds",
    "perm_seconds",
    "status",
    "error_msg",
]

CSV_LOCK = threading.Lock()


def write_csv_header(csv_path: Path):
    header = ",".join(CSV_COLUMNS)
    with CSV_LOCK:
        if csv_path.exists():
            with open(csv_path, "r") as f:
                existing = f.readline().strip()
            if existing and existing != header:
                raise RuntimeError(f"CSV schema mismatch: {csv_path}")
        else:
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            with open(csv_path, "w") as f:
                f.write(header + "\n")


def append_csv_row(csv_path: Path, row: dict):
    with CSV_LOCK:
        with open(csv_path, "a") as f:
            values = []
            for col in CSV_COLUMNS:
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
                if "," in val or '"' in val or "\n" in val:
                    val = '"' + val.replace('"', '""') + '"'
                values.append(val)
            f.write(",".join(values) + "\n")


def load_completed_keys(csv_path: Path) -> set:
    """Load completed (scenario_id, seed, replicate, method) tuples."""
    if not csv_path.exists():
        return set()
    completed = set()
    try:
        import csv
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                status = row.get("status", "")
                if status.startswith("fail"):
                    continue
                key = (
                    row.get("scenario_id", ""),
                    int(row.get("seed", 0)),
                    int(row.get("replicate", 0)),
                    row.get("method", ""),
                )
                completed.add(key)
    except Exception:
        pass
    return completed


# ==============================================================================
# Config Writer
# ==============================================================================


def write_config(config_path: Path, cfg: dict, scenarios: list, shard_scenarios: list):
    config = {
        "version": VERSION,
        "master_seed": MASTER_SEED,
        "n_cores": cfg["n_cores"],
        "shard_id": cfg["shard_id"],
        "n_shards": cfg["n_shards"],
        "grid": {
            "h2": H2_LIST,
            "p": P_LIST,
            "n": N_LIST,
            "m": M_LIST,
            "rho": RHO_LIST,
            "signal_ratio": ["inf" if np.isinf(x) else x for x in SIGNAL_RATIO_LIST],
            "maf_type": MAF_TYPE,
            "maf_fixed": MAF_FIXED,
            "case_control_ratio": CASE_CONTROL_RATIO,
        },
        "holdout_seeds": HOLDOUT_SEEDS,
        "test_size": HOLDOUT_TEST_SIZE,
        "n_replicates": N_REPLICATES,
        "total_scenarios": len(scenarios),
        "shard_scenarios": len(shard_scenarios),
        "methods": get_available_methods(),
        "model_params": {
            "rf": {"n_estimators": N_ESTIMATORS},
            "orf": {"n_estimators": N_ESTIMATORS, "available": ORF_AVAILABLE},
            "bworf": {
                "n_estimators": BWORF_N_ESTIMATORS,
                "l1_strength": BWORF_L1_STRENGTH,
                "available": BWORF_CLASS is not None,
            },
        },
        "perm_importance": {
            "n_repeats": PERM_N_REPEATS,
            "subsample_n": PERM_SUBSAMPLE_N,
            "scoring": PERM_SCORING,
            "topk": PERM_TOPK,
        },
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": SKLEARN_VERSION,
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)


# ==============================================================================
# Available Methods
# ==============================================================================


def get_available_methods() -> list:
    methods = ["rf"]
    if ORF_AVAILABLE:
        methods.append("orf")
    if BWORF_CLASS is not None:
        methods.append("bworf")
    return methods


# ==============================================================================
# Task Runner
# ==============================================================================


def run_scenario_replicate_method(
    scenario: dict,
    scenario_id: str,
    scenario_json: str,
    replicate: int,
    method: str,
    holdout_seed: int,
) -> dict:
    """Run single (scenario, replicate, method, holdout_seed) task."""
    # Derive seeds
    base_seed = stable_hash(MASTER_SEED, scenario_id, replicate)
    model_seed = stable_hash(base_seed, holdout_seed, method, "model")
    perm_seed = stable_hash(base_seed, holdout_seed, "perm")

    row = {
        "scenario_id": scenario_id,
        "scenario_json": scenario_json,
        "method": method,
        "seed": holdout_seed,
        "replicate": replicate,
        "n_samples": scenario["n_samples"],
        "p": scenario["p"],
        "m": scenario["m"],
        "h2": scenario["h2"],
        "rho": scenario["rho"],
        "signal_ratio": scenario["signal_ratio"],
        "maf_type": scenario["maf_type"],
        "case_control_ratio": scenario["case_control_ratio"],
        "status": "ok",
        "error_msg": "",
    }

    # Generate data
    try:
        X, y, causal_pairs, _meta = simulate_dataset_v1(
            n_samples=scenario["n_samples"],
            n_snps=scenario["p"],
            n_blocks=scenario["n_blocks"],
            rho=scenario["rho"],
            maf_type=scenario["maf_type"],
            n_causal_pairs=scenario["m"],
            h2=scenario["h2"],
            signal_ratio=scenario["signal_ratio"],
            case_control_ratio=scenario["case_control_ratio"],
            random_state=base_seed,
        )
    except Exception as e:
        row["status"] = "fail_data"
        row["error_msg"] = str(e)
        return row

    y = np.asarray(y).astype(int)
    causal_snps = np.unique(np.asarray(causal_pairs).flatten()).astype(int)
    row["n_features"] = X.shape[1]
    row["n_causal_snps"] = len(causal_snps)

    # Train/test split
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=HOLDOUT_TEST_SIZE, stratify=y, random_state=holdout_seed
        )
    except Exception as e:
        row["status"] = "fail_split"
        row["error_msg"] = str(e)
        return row

    # Evaluate
    result = evaluate_single(
        method=method,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        causal_snps=causal_snps,
        model_seed=model_seed,
        perm_seed=perm_seed,
    )
    row.update(result)
    return row


# ==============================================================================
# Main
# ==============================================================================


def main():
    args = parse_args()
    cfg = get_runtime_config(args)

    print("=" * 70)
    print(f"Simulation Study v3 - Runner (rf vs orf vs bworf)")
    print("=" * 70)
    print(f"n_cores: {cfg['n_cores']}")
    print(f"shard: {cfg['shard_id']} / {cfg['n_shards']}")
    print(f"methods: {get_available_methods()}")
    print()

    # Generate scenarios
    all_scenarios = generate_all_scenarios()
    shard_scenarios = filter_scenarios_by_shard(all_scenarios, cfg["shard_id"], cfg["n_shards"])

    print(f"Total scenarios: {len(all_scenarios)}")
    print(f"This shard: {len(shard_scenarios)} scenarios")
    print(f"Replicates: {N_REPLICATES}")
    print(f"Holdout seeds: {HOLDOUT_SEEDS}")
    print()

    # Paths
    csv_path = OUTPUTS_DIR / "simulation_results_v3.csv"
    config_path = OUTPUTS_DIR / "run_config_v3.json"

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    write_csv_header(csv_path)
    write_config(config_path, cfg, all_scenarios, shard_scenarios)

    # Load completed
    completed = load_completed_keys(csv_path)
    print(f"Already completed: {len(completed)} rows")

    # Build task list
    methods = get_available_methods()
    tasks = []
    for scenario in shard_scenarios:
        scenario_id = compute_scenario_id(scenario)
        scenario_json = scenario_to_json(scenario)
        for replicate in range(N_REPLICATES):
            for holdout_seed in HOLDOUT_SEEDS:
                for method in methods:
                    key = (scenario_id, holdout_seed, replicate, method)
                    if key not in completed:
                        tasks.append((scenario, scenario_id, scenario_json, replicate, method, holdout_seed))

    print(f"Tasks to run: {len(tasks)}")
    if len(tasks) == 0:
        print("Nothing to do.")
        return

    print("-" * 70)

    # Smoke test mode
    if cfg["smoke_test"]:
        print("SMOKE TEST: Running first 3 tasks only")
        tasks = tasks[:3]

    # Process tasks
    def process_task(task):
        scenario, scenario_id, scenario_json, replicate, method, holdout_seed = task
        return run_scenario_replicate_method(
            scenario, scenario_id, scenario_json, replicate, method, holdout_seed
        )

    n_cores = cfg["n_cores"]
    batch_size = max(1, n_cores)
    n_batches = (len(tasks) + batch_size - 1) // batch_size
    rows_written = 0

    for batch_idx in range(n_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(tasks))
        batch_tasks = tasks[batch_start:batch_end]

        print(f"[Batch {batch_idx + 1}/{n_batches}] {len(batch_tasks)} tasks...")

        try:
            results = Parallel(n_jobs=n_cores, backend="loky", verbose=0)(
                delayed(process_task)(t) for t in batch_tasks
            )
        except Exception as e:
            if TerminatedWorkerError is not None and isinstance(e, TerminatedWorkerError):
                print(f"  Worker died, falling back to sequential")
                results = [process_task(t) for t in batch_tasks]
            else:
                raise

        for row in results:
            append_csv_row(csv_path, row)
            rows_written += 1
            status = row.get("status", "ok")
            method = row.get("method", "?")
            auc = row.get("auc", np.nan)
            if status == "ok":
                auc_str = f"{auc:.3f}" if not np.isnan(auc) else "nan"
                print(f"  {method:5s} AUC={auc_str}")
            else:
                print(f"  {method:5s} {status}: {row.get('error_msg', '')[:50]}")

        print(f"  -> {rows_written} rows written")

    print("-" * 70)
    print(f"Done. Total rows written: {rows_written}")
    print(f"Output: {csv_path}")


if __name__ == "__main__":
    main()
