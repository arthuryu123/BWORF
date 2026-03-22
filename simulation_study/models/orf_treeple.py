"""
Wrapper module for classic Oblique Random Forest (ORF) baseline using scikit-tree.

Provides a scikit-learn compatible interface for scikit-tree's ObliqueRandomForestClassifier.

Install: pip install scikit-tree
"""

import numpy as np

# ---------------------------------------------------------------------------
# scikit-tree import
# ---------------------------------------------------------------------------
_SKTREE_AVAILABLE = False
_IMPORT_ERROR = None
ObliqueRandomForestClassifier = None

try:
    from sktree.ensemble import ObliqueRandomForestClassifier
    _SKTREE_AVAILABLE = True
except ImportError as e:
    _IMPORT_ERROR = str(e)
except Exception as e:
    _IMPORT_ERROR = str(e)


def check_orf_available() -> bool:
    """
    Check if scikit-tree's Oblique Random Forest is available.
    
    Returns:
        True if sktree can be imported successfully, False otherwise.
        This function never raises an exception.
    """
    return _SKTREE_AVAILABLE


def _raise_if_unavailable():
    """Raise ImportError with helpful message if scikit-tree is not available."""
    if not _SKTREE_AVAILABLE:
        raise ImportError(
            f"scikit-tree is not installed or could not be imported.\n"
            f"Install with: pip install scikit-tree\n"
            f"Error: {_IMPORT_ERROR}"
        )


def make_orf(
    random_state: int,
    n_estimators: int = 100,
    max_depth=None,
    n_jobs: int = -1
):
    """
    Create a scikit-tree Oblique Random Forest classifier ready for fitting.
    
    This returns an unfitted estimator that supports .fit(X, y) and .predict_proba(X)
    for binary classification tasks.
    
    Parameters
    ----------
    random_state : int
        Random seed for reproducibility.
    n_estimators : int, default=100
        Number of trees in the forest.
    max_depth : int or None, default=None
        Maximum depth of each tree. None means unlimited depth.
    n_jobs : int, default=-1
        Number of parallel jobs. -1 uses all available cores.
    
    Returns
    -------
    estimator : ObliqueRandomForestClassifier
        An unfitted scikit-learn compatible classifier.
    
    Raises
    ------
    ImportError
        If scikit-tree is not installed.
    
    Examples
    --------
    >>> orf = make_orf(random_state=42, n_estimators=50, max_depth=10)
    >>> orf.fit(X_train, y_train)
    >>> probas = orf.predict_proba(X_test)  # shape (n_samples, 2)
    """
    _raise_if_unavailable()
    
    return ObliqueRandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=n_jobs,
    )


def predict_proba_binary(estimator, X) -> np.ndarray:
    """
    Get binary class probabilities with guaranteed shape (n_samples, 2).
    
    Validates that the estimator was trained on binary classes {0, 1} and
    returns probabilities ordered as [P(y=0), P(y=1)].
    
    Parameters
    ----------
    estimator : fitted classifier
        A fitted classifier with predict_proba method and classes_ attribute.
    X : array-like of shape (n_samples, n_features)
        Samples to predict.
    
    Returns
    -------
    probas : ndarray of shape (n_samples, 2)
        Probability of class 0 in column 0, probability of class 1 in column 1.
    
    Raises
    ------
    ValueError
        If estimator.classes_ is not exactly {0, 1}.
    """
    n_samples = X.shape[0]
    
    # Validate binary classes {0, 1}
    if hasattr(estimator, 'classes_'):
        classes = estimator.classes_
        classes_set = set(classes)
        if classes_set != {0, 1}:
            raise ValueError(
                f"Expected binary classes {{0, 1}}. Got classes_={list(classes)}"
            )
    # If classes_ missing, proceed but validate output shape below
    
    probas = estimator.predict_proba(X)
    
    # Validate shape
    if probas.ndim == 1:
        probas = probas.reshape(-1, 1)
    
    if probas.shape != (n_samples, 2):
        raise ValueError(
            f"Expected predict_proba shape ({n_samples}, 2), got {probas.shape}"
        )
    
    # Reorder columns if classes_ is [1, 0] instead of [0, 1]
    if hasattr(estimator, 'classes_'):
        classes = list(estimator.classes_)
        if classes == [1, 0]:
            # Swap columns to get [P(y=0), P(y=1)]
            probas = probas[:, ::-1].copy()
    
    return probas


# ---------------------------------------------------------------------------
# Smoke test helper (call explicitly, not run on import)
# ---------------------------------------------------------------------------
def _smoke_test():
    """
    Minimal smoke test for ORF wrapper.
    
    Trains on tiny synthetic data, validates predict_proba_binary output.
    Raises AssertionError on failure.
    """
    if not check_orf_available():
        raise ImportError("scikit-tree not available for smoke test")
    
    # Tiny synthetic dataset
    rng = np.random.default_rng(42)
    X = rng.standard_normal((50, 5))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    # Fit model
    orf = make_orf(random_state=42, n_estimators=10, max_depth=5, n_jobs=1)
    orf.fit(X, y)
    
    # Predict
    probas = predict_proba_binary(orf, X)
    
    # Validate shape
    assert probas.shape == (50, 2), f"Shape mismatch: {probas.shape}"
    
    # Validate row sums ~1
    row_sums = probas.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-6), f"Row sums not ~1: {row_sums}"
    
    # Validate values in [0, 1]
    assert np.all(probas >= 0) and np.all(probas <= 1), "Probas outside [0,1]"
    
    print("_smoke_test passed.")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("ORF scikit-tree Wrapper - Availability Check")
    print("=" * 60)
    
    print(f"\nscikit-tree available: {check_orf_available()}")
    
    if not check_orf_available():
        print(f"\nERROR: scikit-tree not installed or import failed.")
        print(f"  Error: {_IMPORT_ERROR}")
        print("\nInstall with: pip install scikit-tree")
        exit(1)
    
    print(f"ObliqueRandomForestClassifier: {ObliqueRandomForestClassifier}")
    print("\nAll imports OK.")
    print("=" * 60)
