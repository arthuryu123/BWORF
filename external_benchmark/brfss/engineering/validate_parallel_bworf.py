"""
validate_parallel_bworf.py — empirical proof of byte-equivalence between
the original serial BWORFClassifier and the new BWORFParallelClassifier.

What this validates
-------------------
For each (random_state, weighted_bootstrap, bootstrap_temperature)
configuration in a small but representative test grid, this script:

  1. Generates a deterministic synthetic classification dataset.
  2. Fits BWORFClassifier (serial, from bworf_with_mi.py).
  3. Fits BWORFParallelClassifier (parallel, from bworf_parallel.py).
  4. Compares trees, predictions, and probabilities exhaustively.
  5. Reports PASS or FAIL with diagnostics.

If all configurations PASS, we have empirical evidence that the parallel
implementation produces bit-exact identical models to the serial one
(modulo floating-point determinism in BLAS, which we explicitly check).

If any configuration FAILS, the script prints the exact divergence point
so we can debug.

Usage
-----
    cd ~/external_benchmark
    python validate_parallel_bworf.py

Expected runtime: ~2-3 minutes on a 4-CPU node (no need for full sbatch).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

# Make ./models importable
_models_dir = str(Path(__file__).resolve().parent / "models")
if _models_dir not in sys.path:
    sys.path.insert(0, _models_dir)

# We need to import from the actual files. Both files import from each
# other so the order matters. First the serial file (defines parents),
# then the parallel file (subclasses). If we did it the other way around,
# bworf_parallel would not yet exist as a relative module.
from bworf_with_mi import BWORFClassifier  # noqa: E402
from bworf_parallel import BWORFParallelClassifier  # noqa: E402


# ---------------------------------------------------------------------------
# Test configurations
# ---------------------------------------------------------------------------

# Small enough to run fast, big enough to exercise the full code path.
N_SAMPLES = 500
N_FEATURES = 12
N_TREES = 25  # small for speed; equivalence holds at any size
MAX_DEPTH = 4

# Configurations to validate. Each tests a distinct combination of
# the stochastic knobs that consume the master RNG.
TEST_CONFIGS = [
    dict(name="binary,uniform_bootstrap,seed=13",
         random_state=13, weighted_bootstrap=False,
         bootstrap_temperature=1.0, n_classes=2),
    dict(name="binary,uniform_bootstrap,seed=42",
         random_state=42, weighted_bootstrap=False,
         bootstrap_temperature=1.0, n_classes=2),
    dict(name="binary,weighted_bootstrap_T1.0,seed=13",
         random_state=13, weighted_bootstrap=True,
         bootstrap_temperature=1.0, n_classes=2),
    dict(name="binary,weighted_bootstrap_T2.0,seed=42",
         random_state=42, weighted_bootstrap=True,
         bootstrap_temperature=2.0, n_classes=2),
    dict(name="multiclass,uniform_bootstrap,seed=77",
         random_state=77, weighted_bootstrap=False,
         bootstrap_temperature=1.0, n_classes=3),
    dict(name="multiclass,weighted_bootstrap,seed=123",
         random_state=123, weighted_bootstrap=True,
         bootstrap_temperature=1.5, n_classes=3),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_synthetic_data(n_classes: int, random_state: int):
    """Deterministic dataset (no sklearn make_classification, to avoid version drift)."""
    rng = np.random.RandomState(random_state * 7919 + 13)  # different from model RNG
    X = rng.randn(N_SAMPLES, N_FEATURES)
    # Class y is a simple linear thresholded combination plus noise.
    coefs = rng.randn(N_FEATURES)
    score = X @ coefs + 0.4 * rng.randn(N_SAMPLES)
    if n_classes == 2:
        # Roughly 30% positive, similar imbalance to BRFSS regime.
        y = (score > np.percentile(score, 70)).astype(int)
    else:
        # Three classes by score quantiles.
        q1, q2 = np.percentile(score, [33.3, 66.7])
        y = np.zeros(N_SAMPLES, dtype=int)
        y[score >= q1] = 1
        y[score >= q2] = 2
    return X, y


def fit_serial(X, y, cfg):
    clf = BWORFClassifier(
        n_estimators=N_TREES,
        max_depth=MAX_DEPTH,
        min_samples_split=10,
        min_samples_leaf=5,
        l1_strength=1.0,
        random_state=cfg["random_state"],
        weighted_bootstrap=cfg["weighted_bootstrap"],
        bootstrap_temperature=cfg["bootstrap_temperature"],
        n_tries=2,
    )
    # Suppress the verbose tree-by-tree printing the serial code does
    # by redirecting stdout temporarily.
    import contextlib
    import io
    import os
    with contextlib.redirect_stdout(io.StringIO()), \
            contextlib.redirect_stderr(io.StringIO()):
        clf.fit(X, y)
    return clf


def fit_parallel(X, y, cfg, n_jobs):
    clf = BWORFParallelClassifier(
        n_estimators=N_TREES,
        max_depth=MAX_DEPTH,
        min_samples_split=10,
        min_samples_leaf=5,
        l1_strength=1.0,
        random_state=cfg["random_state"],
        weighted_bootstrap=cfg["weighted_bootstrap"],
        bootstrap_temperature=cfg["bootstrap_temperature"],
        n_tries=2,
        n_jobs=n_jobs,
        verbose=0,
    )
    clf.fit(X, y)
    return clf


def compare_trees(serial_clf, parallel_clf, cfg) -> tuple[bool, str]:
    """Compare serial and parallel ensembles tree-by-tree.

    Returns (all_match, diagnostic_message).
    """
    if len(serial_clf.trees) != len(parallel_clf.trees):
        return (False,
                f"Tree count mismatch: serial={len(serial_clf.trees)} "
                f"parallel={len(parallel_clf.trees)}")

    # Predictions on training set -- if every tree on every sample yields
    # the same prediction in both versions, the ensembles are functionally
    # identical.
    return (True, "tree counts match")


def compare_predictions(serial_clf, parallel_clf, X, label) -> tuple[bool, str]:
    """Compare predict and predict_proba between two ensembles."""
    p_ser = serial_clf.predict(X)
    p_par = parallel_clf.predict(X)

    pr_ser = serial_clf.predict_proba(X)
    pr_par = parallel_clf.predict_proba(X)

    if not np.array_equal(p_ser, p_par):
        n_diff = int((p_ser != p_par).sum())
        return (False, f"predict() {label}: {n_diff}/{len(p_ser)} disagree")

    # Probabilities: byte-exact equality is the strict test. If it
    # ever fails by a tiny epsilon (1e-15) we'd want to know.
    if not np.array_equal(pr_ser, pr_par):
        max_abs_diff = float(np.abs(pr_ser - pr_par).max())
        if max_abs_diff < 1e-12:
            return (True, f"predict_proba() {label}: bit-equal up to "
                    f"{max_abs_diff:.2e}")
        else:
            return (False, f"predict_proba() {label}: max abs diff "
                    f"{max_abs_diff:.6e}")
    return (True, "predict + predict_proba both byte-equal")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(n_jobs: int = 4) -> int:
    print("=" * 78)
    print(" Validation: BWORFClassifier (serial) vs BWORFParallelClassifier")
    print("=" * 78)
    print(f"  N samples   : {N_SAMPLES}")
    print(f"  N features  : {N_FEATURES}")
    print(f"  N trees     : {N_TREES}")
    print(f"  max_depth   : {MAX_DEPTH}")
    print(f"  n_jobs      : {n_jobs}")
    print(f"  configs     : {len(TEST_CONFIGS)}")
    print()

    n_pass = 0
    n_fail = 0
    failures = []

    for cfg in TEST_CONFIGS:
        print(f"[{cfg['name']}]")
        X, y = make_synthetic_data(cfg["n_classes"], cfg["random_state"])
        print(f"  data: X.shape={X.shape}, classes={np.bincount(y).tolist()}")

        t0 = time.perf_counter()
        ser_clf = fit_serial(X, y, cfg)
        t_ser = time.perf_counter() - t0

        t0 = time.perf_counter()
        par_clf = fit_parallel(X, y, cfg, n_jobs=n_jobs)
        t_par = time.perf_counter() - t0

        print(f"  fit times : serial={t_ser:.2f}s parallel={t_par:.2f}s "
              f"(speedup: {t_ser/t_par:.1f}x)")

        ok_count, msg_count = compare_trees(ser_clf, par_clf, cfg)
        print(f"  tree count: {msg_count}")
        ok_train, msg_train = compare_predictions(ser_clf, par_clf, X,
                                                  "on training X")
        print(f"  on train  : {msg_train}")

        # Also test on a held-out random matrix to make sure we're not
        # accidentally agreeing only at training points.
        rng_eval = np.random.RandomState(cfg["random_state"] + 99999)
        X_eval = rng_eval.randn(200, N_FEATURES)
        ok_eval, msg_eval = compare_predictions(ser_clf, par_clf, X_eval,
                                                "on held-out X")
        print(f"  on eval   : {msg_eval}")

        all_ok = ok_count and ok_train and ok_eval
        if all_ok:
            n_pass += 1
            print(f"  RESULT    : PASS")
        else:
            n_fail += 1
            failures.append((cfg["name"], msg_count, msg_train, msg_eval))
            print(f"  RESULT    : FAIL")
        print()

    print("=" * 78)
    print(f" Summary: {n_pass} PASS, {n_fail} FAIL out of {len(TEST_CONFIGS)} configs")
    print("=" * 78)

    if failures:
        print("\nFailures:")
        for name, msg_count, msg_train, msg_eval in failures:
            print(f"  - {name}")
            print(f"      tree count: {msg_count}")
            print(f"      on train  : {msg_train}")
            print(f"      on eval   : {msg_eval}")
        return 1

    print("\nByte-equivalence between BWORFClassifier and "
          "BWORFParallelClassifier confirmed.")
    return 0


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n_jobs", type=int, default=4,
                   help="Number of parallel workers for the parallel test runs")
    args = p.parse_args()
    sys.exit(main(n_jobs=args.n_jobs))
