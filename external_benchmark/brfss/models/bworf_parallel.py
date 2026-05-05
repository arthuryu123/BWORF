"""
bworf_parallel.py — drop-in parallel implementation of BWORF for large datasets.

This module provides BWORFParallelClassifier, a subclass of
ObliqueRandomForestMulti that overrides .fit() and .predict_proba() to use
joblib-based parallelism across trees. It is functionally byte-identical
to the original BWORFClassifier in bworf_with_mi.py for the same
random_state — only the wallclock changes.

Design contract
---------------
For any (X, y, random_state, hyperparameters), the trees fitted by
BWORFParallelClassifier are bit-exactly the same as those fitted by
BWORFClassifier. This is achieved by:

  1. Sequentially consuming the master np.random.RandomState in the same
     order as the original .fit() loop (one rng.choice + one rng.randint
     per tree), producing N bootstrap-index arrays and N tree seeds.
  2. Dispatching tree-fitting to N independent joblib workers, each
     receiving a (X[indices], y[indices], tree_seed, hyperparameters)
     tuple and returning a fitted ObliqueDecisionTreeMulti.
  3. Each worker uses ONLY its own RandomState and sklearn random_state
     ints, never touching np.random global state, so worker processes
     cannot cross-contaminate.

The validation script validate_parallel_bworf.py asserts byte-equivalence
empirically across multiple random_state values.

Why a separate file?
--------------------
The thesis-validated bworf_with_mi.py is left byte-identical so that all
existing experiments (DILI multiclass, smaller external benchmarks)
continue to use the originally-defended code without modification.

Usage
-----
    from models.bworf_parallel import BWORFParallelClassifier
    clf = BWORFParallelClassifier(
        n_estimators=100, max_depth=5, l1_strength=1.0,
        weighted_bootstrap=True, bootstrap_temperature=1.0,
        n_tries=2, random_state=13,
        n_jobs=-1,                # -1 = all CPUs (default)
    )
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)
"""

from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed

# Reuse the thesis-validated tree and ensemble classes.
# Use absolute import (matching how benchmark_runner.py expects models/
# to be on sys.path as a flat module directory).
from bworf_with_mi import ObliqueDecisionTreeMulti, ObliqueRandomForestMulti


# ---------------------------------------------------------------------------
# Module-level worker functions
#
# These MUST be at module level (not class methods) so joblib can pickle
# them across worker processes without dragging an entire ensemble instance
# along with each tree job.
# ---------------------------------------------------------------------------

def _fit_one_tree(
    X_bootstrap: np.ndarray,
    y_bootstrap: np.ndarray,
    tree_seed: int,
    max_depth: int,
    min_samples_split: int,
    min_samples_leaf: int,
    l1_strength: float,
    n_tries: int,
) -> ObliqueDecisionTreeMulti:
    """Fit a single oblique tree in an isolated worker process.

    Mirrors lines 671-682 of ObliqueRandomForestMulti.fit() exactly,
    minus the verbose printing (which would flood per-tree-process logs).
    """
    tree = ObliqueDecisionTreeMulti(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        l1_strength=l1_strength,
        random_state=tree_seed,
        debug_trace=False,
        debug_max_nodes=0,
        n_tries=n_tries,
    )
    tree.fit(X_bootstrap, y_bootstrap)
    return tree


def _predict_proba_one_tree(tree: ObliqueDecisionTreeMulti, X: np.ndarray) -> np.ndarray:
    """Score one tree on a feature matrix (worker function)."""
    return tree.predict_proba(X)


# ---------------------------------------------------------------------------
# BWORFParallelClassifier
# ---------------------------------------------------------------------------

class BWORFParallelClassifier(ObliqueRandomForestMulti):
    """Parallel-fitting BWORF classifier.

    Inherits everything from ObliqueRandomForestMulti and adds:
      - n_jobs parameter (default -1 = all CPUs)
      - parallel .fit() implementation
      - parallel .predict_proba() implementation

    For the same (X, y, random_state, hyperparameters), produces
    bit-exact identical predictions to BWORFClassifier from
    bworf_with_mi.py. Use BWORFClassifier when n is small (parallel
    overhead dominates); use this class when n is large enough that
    serial tree-fitting is the bottleneck.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        min_samples_split: int = 10,
        min_samples_leaf: int = 5,
        l1_strength: float = 1.0,
        random_state: int | None = None,
        debug_trace: bool = False,
        debug_max_nodes: int = 3,
        weighted_bootstrap: bool = False,
        bootstrap_weight_mode: str = "balanced",
        bootstrap_temperature: float = 1.0,
        n_tries: int = 2,
        n_jobs: int = -1,
        verbose: int = 0,
    ):
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            l1_strength=l1_strength,
            random_state=random_state,
            debug_trace=debug_trace,
            debug_max_nodes=debug_max_nodes,
            weighted_bootstrap=weighted_bootstrap,
            bootstrap_weight_mode=bootstrap_weight_mode,
            bootstrap_temperature=bootstrap_temperature,
            n_tries=n_tries,
        )
        self.n_jobs = n_jobs
        self.verbose = verbose

    # -----------------------------------------------------------------
    # Override fit() with parallel implementation.
    # -----------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the ensemble in parallel across trees.

        Pre-computes all bootstrap indices and tree seeds sequentially
        from the master RNG (preserving byte-equivalence with serial fit),
        then dispatches tree fitting to joblib workers.
        """
        rng = np.random.RandomState(self.random_state)
        self.trees = []
        n_samples = len(X)

        # ----- Compute bootstrap probabilities (mirrors lines 622-650) -----
        bootstrap_probs = None
        if self.weighted_bootstrap:
            if self.bootstrap_weight_mode == "balanced":
                unique_classes, class_counts = np.unique(y, return_counts=True)
                n_classes_local = len(unique_classes)
                class_weights_dict = {
                    cls: n_samples / (n_classes_local * count)
                    for cls, count in zip(unique_classes, class_counts)
                }
                sample_weights = np.array([class_weights_dict[label] for label in y])
                sample_weights_temp = sample_weights ** self.bootstrap_temperature
                bootstrap_probs = sample_weights_temp / sample_weights_temp.sum()

        # ----- Sequential RNG consumption (mirrors loop body of original) -----
        # CRITICAL: this loop must consume rng in the same order as the original
        # serial fit (one rng.choice, then one rng.randint, per tree). That is
        # what guarantees byte-equivalence with BWORFClassifier.
        bootstrap_indices_list = []
        tree_seeds = []
        for _ in range(self.n_estimators):
            if self.weighted_bootstrap and bootstrap_probs is not None:
                indices = rng.choice(
                    n_samples, size=n_samples, replace=True, p=bootstrap_probs
                )
            else:
                indices = rng.choice(n_samples, size=n_samples, replace=True)
            bootstrap_indices_list.append(indices)
            tree_seeds.append(int(rng.randint(0, 100000)))

        # ----- Parallel tree fitting -----
        # Each worker is fully independent. They share no state, so the order
        # in which they finish doesn't affect the final result; we collect
        # by index, preserving the ensemble's tree order.
        if self.verbose:
            print(
                f"BWORFParallelClassifier.fit: dispatching {self.n_estimators} "
                f"trees across n_jobs={self.n_jobs}"
            )
        fitted_trees = Parallel(
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            backend="loky",
            # Joblib will memory-map large arrays automatically; this just
            # raises the threshold so small smoke-test datasets don't trigger
            # mmap overhead. For full-N runs, mmap kicks in correctly.
            max_nbytes="100M",
        )(
            delayed(_fit_one_tree)(
                X[indices],
                y[indices],
                tree_seed,
                self.max_depth,
                self.min_samples_split,
                self.min_samples_leaf,
                self.l1_strength,
                self.n_tries,
            )
            for indices, tree_seed in zip(bootstrap_indices_list, tree_seeds)
        )

        self.trees = list(fitted_trees)
        return self

    # -----------------------------------------------------------------
    # Override predict_proba() with parallel implementation.
    # -----------------------------------------------------------------
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Average per-tree class probabilities, scoring trees in parallel."""
        if not self.trees:
            raise ValueError("Forest not fitted. Call fit() first.")

        # For very small forests or X, parallel overhead dominates; fall
        # back to serial. Threshold of 16 trees * 1000 rows is well below
        # any production scale.
        if len(self.trees) < 16 or X.shape[0] < 1000:
            all_probas = np.array(
                [tree.predict_proba(X) for tree in self.trees]
            )
            return np.mean(all_probas, axis=0)

        proba_list = Parallel(
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            backend="loky",
            max_nbytes="100M",
        )(
            delayed(_predict_proba_one_tree)(tree, X) for tree in self.trees
        )
        all_probas = np.array(proba_list)
        return np.mean(all_probas, axis=0)
