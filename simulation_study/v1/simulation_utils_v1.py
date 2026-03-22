"""
Simulation Utilities v1 - Binary Liability-Threshold Genetics Simulation Engine

This module provides deterministic utilities for simulating genetic data with:
- Block-diagonal LD structure (compound symmetry within blocks)
- Liability-threshold model for binary phenotypes
- Coefficient-ratio signal model (v1) for epistasis

All randomness is controlled via numpy's default_rng for reproducibility.

Dependencies: numpy, scipy (no sklearn)
"""

import numpy as np
from scipy.stats import norm


class GenomeSimulator:
    """
    Simulates genotype matrices with block-diagonal LD structure.
    
    Genotypes are integers in {0, 1, 2} representing minor allele counts.
    LD is induced via latent multivariate normal sampling with compound
    symmetry correlation within blocks, then discretized to genotypes
    using Hardy-Weinberg equilibrium thresholds.
    
    Parameters
    ----------
    n_snps : int
        Total number of SNPs to simulate.
    n_blocks : int
        Number of LD blocks. Must divide n_snps evenly.
    rho : float
        Within-block correlation coefficient in [0, 1).
    maf_type : str
        Either "fixed" (all SNPs have MAF=0.3) or "uniform" (MAF ~ U(0.05, 0.5)).
    random_state : int or None
        Seed for reproducibility.
    
    Attributes
    ----------
    rng : np.random.Generator
        Local random number generator.
    maf : np.ndarray
        Minor allele frequency for each SNP, shape (n_snps,).
    block_size : int
        Number of SNPs per block.
    chol_block : np.ndarray
        Cholesky factor of the block covariance matrix.
    """
    
    def __init__(
        self,
        n_snps: int = 1000,
        n_blocks: int = 50,
        rho: float = 0.5,
        maf_type: str = "fixed",
        random_state=None
    ):
        # Validate n_snps divisibility
        if n_snps % n_blocks != 0:
            raise ValueError(
                f"n_snps ({n_snps}) must be divisible by n_blocks ({n_blocks}). "
                f"Remainder is {n_snps % n_blocks}."
            )
        
        # Validate rho
        if not (0.0 <= rho < 1.0):
            raise ValueError(f"rho must be in [0, 1), got {rho}")
        
        # Validate maf_type
        if maf_type not in {"fixed", "uniform"}:
            raise ValueError(f"maf_type must be 'fixed' or 'uniform', got '{maf_type}'")
        
        self.n_snps = n_snps
        self.n_blocks = n_blocks
        self.rho = rho
        self.maf_type = maf_type
        self.block_size = n_snps // n_blocks
        
        # Initialize RNG
        self.rng = np.random.default_rng(random_state)
        
        # Precompute MAF vector
        if maf_type == "fixed":
            self.maf = np.full(n_snps, 0.3, dtype=np.float64)
        else:  # uniform
            self.maf = self.rng.uniform(0.05, 0.5, size=n_snps)
        
        # Precompute Cholesky factor for block covariance
        # Block covariance: compound symmetry with 1 on diagonal, rho off-diagonal
        b = self.block_size
        if rho == 0.0:
            # Identity matrix - Cholesky is identity
            self.chol_block = np.eye(b, dtype=np.float64)
        else:
            # Compound symmetry: Sigma = (1-rho)*I + rho*J
            # where J is the all-ones matrix
            # Cholesky can be computed analytically but we use numpy for clarity
            cov_block = np.full((b, b), rho, dtype=np.float64)
            np.fill_diagonal(cov_block, 1.0)
            self.chol_block = np.linalg.cholesky(cov_block)
        
        # Precompute HWE thresholds for each SNP
        # P(0) = (1-maf)^2, P(1) = 2*maf*(1-maf), P(2) = maf^2
        p0 = (1.0 - self.maf) ** 2
        p1 = 2.0 * self.maf * (1.0 - self.maf)
        # Thresholds in standard normal
        self._thr1 = norm.ppf(p0)  # Z <= thr1 => genotype 0
        self._thr2 = norm.ppf(p0 + p1)  # thr1 < Z <= thr2 => genotype 1
    
    def generate_genotypes(self, n_samples: int) -> np.ndarray:
        """
        Generate a genotype matrix with LD structure.
        
        Parameters
        ----------
        n_samples : int
            Number of individuals to simulate.
        
        Returns
        -------
        X : np.ndarray
            Genotype matrix of shape (n_samples, n_snps) with values in {0, 1, 2}.
            Dtype is int8 for memory efficiency.
        """
        # Step (a): Generate latent correlated normals via block sampling
        Z = np.empty((n_samples, self.n_snps), dtype=np.float64)
        
        b = self.block_size
        for block_idx in range(self.n_blocks):
            start = block_idx * b
            end = start + b
            # Draw standard normals for this block
            Z_raw = self.rng.standard_normal((n_samples, b))
            # Apply Cholesky to induce correlation: Z_corr = Z_raw @ L.T
            Z[:, start:end] = Z_raw @ self.chol_block.T
        
        # Step (b): Discretize to genotypes using precomputed thresholds
        # Vectorized: compare Z against thresholds broadcasted over SNPs
        # thr1, thr2 have shape (n_snps,), Z has shape (n_samples, n_snps)
        X = np.zeros((n_samples, self.n_snps), dtype=np.int8)
        
        # genotype 1 where thr1 < Z <= thr2
        mask_1 = (Z > self._thr1) & (Z <= self._thr2)
        X[mask_1] = 1
        
        # genotype 2 where Z > thr2
        mask_2 = Z > self._thr2
        X[mask_2] = 2
        
        return X


def calculate_liability_v1(
    X: np.ndarray,
    n_causal_pairs: int = 2,
    h2: float = 0.05,
    signal_ratio: float = 1.0,
    random_state=None
) -> tuple:
    """
    Compute liability scores using the v1 coefficient-ratio signal model.
    
    The genetic score G is computed as:
        G = sum over causal pairs (a, b):
            beta_main * X[:, a] + beta_main * X[:, b] + beta_int * (X[:, a] * X[:, b])
    
    Noise variance is scaled to achieve the target heritability h2 based on
    the realized variance of G.
    
    Parameters
    ----------
    X : np.ndarray
        Genotype matrix of shape (n_samples, n_snps) with values in {0, 1, 2}.
    n_causal_pairs : int
        Number of causal SNP pairs for epistasis. Must have 2*n_causal_pairs <= n_snps.
    h2 : float
        Target heritability in (0, 1).
    signal_ratio : float
        Ratio of main effects to interaction: beta_main / beta_int.
        If np.isinf(signal_ratio), marginal-only model (beta_int=0).
    random_state : int or None
        Seed for reproducibility.
    
    Returns
    -------
    Q : np.ndarray
        Liability scores of shape (n_samples,).
    causal_pairs : np.ndarray
        Array of shape (n_causal_pairs, 2) with causal SNP index pairs.
    metadata : dict
        Dictionary with keys: "beta_main", "beta_int", "var_g", "sigma_e2", "h2_target".
    
    Raises
    ------
    ValueError
        If input validation fails.
    RuntimeError
        If realized Var(G) is effectively zero.
    """
    # Validate X
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")
    
    n_samples, n_snps = X.shape
    unique_vals = np.unique(X)
    if not np.all(np.isin(unique_vals, [0, 1, 2])):
        raise ValueError(f"X must contain only values in {{0, 1, 2}}, got {unique_vals}")
    
    # Validate n_causal_pairs
    if n_causal_pairs < 1:
        raise ValueError(f"n_causal_pairs must be >= 1, got {n_causal_pairs}")
    if 2 * n_causal_pairs > n_snps:
        raise ValueError(
            f"Need 2*n_causal_pairs <= n_snps, but 2*{n_causal_pairs}={2*n_causal_pairs} > {n_snps}"
        )
    
    # Validate h2
    if not (0.0 < h2 < 1.0):
        raise ValueError(f"h2 must be in (0, 1), got {h2}")
    
    # Initialize RNG
    rng = np.random.default_rng(random_state)
    
    # Select unique causal SNP indices
    causal_indices = rng.choice(n_snps, size=2 * n_causal_pairs, replace=False)
    causal_pairs = causal_indices.reshape(n_causal_pairs, 2)
    
    # Determine betas based on signal_ratio
    if np.isinf(signal_ratio):
        # Marginal-only model
        beta_int = 0.0
        beta_main = 1.0
    else:
        beta_int = 1.0
        beta_main = float(signal_ratio)
    
    # Compute genetic score G
    G = np.zeros(n_samples, dtype=np.float64)
    for a, b in causal_pairs:
        xa = X[:, a].astype(np.float64)
        xb = X[:, b].astype(np.float64)
        G += beta_main * xa + beta_main * xb + beta_int * (xa * xb)
    
    # Compute realized Var(G)
    var_g = np.var(G, ddof=1)
    if var_g < 1e-12:
        raise RuntimeError(
            f"Realized Var(G) is effectively zero ({var_g:.2e}). "
            "Check causal SNP selection or genotype data."
        )
    
    # Noise scaling to enforce target h2
    # h2 = Var(G) / (Var(G) + sigma_e2)
    # => sigma_e2 = Var(G) * (1 - h2) / h2
    sigma_e2 = var_g * (1.0 - h2) / h2
    
    # Generate noise
    epsilon = rng.normal(0.0, np.sqrt(sigma_e2), size=n_samples)
    
    # Compute liability
    Q = G + epsilon
    
    metadata = {
        "beta_main": beta_main,
        "beta_int": beta_int,
        "var_g": var_g,
        "sigma_e2": sigma_e2,
        "h2_target": h2
    }
    
    return Q, causal_pairs, metadata


def assign_phenotypes(
    Q: np.ndarray,
    case_control_ratio: float = 1.0
) -> tuple:
    """
    Assign binary phenotypes using a liability threshold model.
    
    Parameters
    ----------
    Q : np.ndarray
        Liability scores of shape (n_samples,).
    case_control_ratio : float
        Ratio of controls to cases. E.g., ratio=1 means equal numbers (prevalence=0.5).
        ratio=9 means 9 controls per case (prevalence=0.1).
    
    Returns
    -------
    y : np.ndarray
        Binary phenotype array of shape (n_samples,) with dtype int8.
        1 = case, 0 = control.
    metadata : dict
        Dictionary with keys: "threshold", "prevalence_target", "prevalence_achieved".
    
    Raises
    ------
    ValueError
        If case_control_ratio <= 0.
    """
    if case_control_ratio <= 0:
        raise ValueError(f"case_control_ratio must be > 0, got {case_control_ratio}")
    
    # Prevalence K = 1 / (1 + ratio)
    # ratio = controls/cases, so if ratio=1, K=0.5
    prevalence_target = 1.0 / (1.0 + case_control_ratio)
    
    # Threshold at (1 - K) quantile: individuals above threshold are cases
    threshold = np.quantile(Q, 1.0 - prevalence_target)
    
    # Assign phenotypes: y=1 if Q > threshold (strict inequality)
    y = (Q > threshold).astype(np.int8)
    
    # Compute achieved prevalence
    prevalence_achieved = np.mean(y)
    
    metadata = {
        "threshold": float(threshold),
        "prevalence_target": prevalence_target,
        "prevalence_achieved": float(prevalence_achieved)
    }
    
    return y, metadata


def simulate_dataset_v1(
    n_samples: int,
    n_snps: int,
    n_blocks: int,
    rho: float,
    maf_type: str,
    n_causal_pairs: int,
    h2: float,
    signal_ratio: float,
    case_control_ratio: float,
    random_state: int
) -> tuple:
    """
    Convenience function to simulate a complete dataset.
    
    Parameters
    ----------
    n_samples : int
        Number of individuals.
    n_snps : int
        Number of SNPs.
    n_blocks : int
        Number of LD blocks.
    rho : float
        Within-block LD correlation.
    maf_type : str
        "fixed" or "uniform" MAF distribution.
    n_causal_pairs : int
        Number of causal SNP pairs.
    h2 : float
        Target heritability.
    signal_ratio : float
        Ratio beta_main / beta_int.
    case_control_ratio : float
        Ratio of controls to cases.
    random_state : int
        Master seed for full reproducibility.
    
    Returns
    -------
    X : np.ndarray
        Genotype matrix (n_samples, n_snps).
    y : np.ndarray
        Binary phenotype vector (n_samples,).
    causal_pairs : np.ndarray
        Causal SNP pairs (n_causal_pairs, 2).
    metadata : dict
        Combined metadata from all simulation steps.
    """
    # Create master RNG to derive sub-seeds deterministically
    master_rng = np.random.default_rng(random_state)
    
    # Derive sub-seeds
    seed_genome = int(master_rng.integers(0, 2**31))
    seed_liability = int(master_rng.integers(0, 2**31))
    
    # Step 1: Generate genotypes
    simulator = GenomeSimulator(
        n_snps=n_snps,
        n_blocks=n_blocks,
        rho=rho,
        maf_type=maf_type,
        random_state=seed_genome
    )
    X = simulator.generate_genotypes(n_samples)
    
    # Step 2: Compute liability
    Q, causal_pairs, liability_meta = calculate_liability_v1(
        X,
        n_causal_pairs=n_causal_pairs,
        h2=h2,
        signal_ratio=signal_ratio,
        random_state=seed_liability
    )
    
    # Step 3: Assign phenotypes
    y, pheno_meta = assign_phenotypes(Q, case_control_ratio=case_control_ratio)
    
    # Combine metadata
    metadata = {
        "n_samples": n_samples,
        "n_snps": n_snps,
        "n_blocks": n_blocks,
        "rho": rho,
        "maf_type": maf_type,
        "n_causal_pairs": n_causal_pairs,
        "random_state": random_state,
        **liability_meta,
        **pheno_meta
    }
    
    return X, y, causal_pairs, metadata


# =============================================================================
# Self-test
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("simulation_utils_v1.py - Self-Test")
    print("=" * 60)
    
    # Run a tiny simulation
    X, y, causal_pairs, meta = simulate_dataset_v1(
        n_samples=200,
        n_snps=200,
        n_blocks=20,
        rho=0.5,
        maf_type="fixed",
        n_causal_pairs=2,
        h2=0.05,
        signal_ratio=1.0,
        case_control_ratio=1.0,
        random_state=123
    )
    
    print(f"\n[Genotypes]")
    print(f"  X.shape       : {X.shape}")
    print(f"  X.dtype       : {X.dtype}")
    print(f"  Unique values : {np.unique(X)}")
    
    print(f"\n[Phenotypes]")
    print(f"  y.shape            : {y.shape}")
    print(f"  Prevalence achieved: {meta['prevalence_achieved']:.4f}")
    print(f"  Prevalence target  : {meta['prevalence_target']:.4f}")
    
    print(f"\n[Liability Model]")
    print(f"  Var(G)    : {meta['var_g']:.6f}")
    print(f"  sigma_e2  : {meta['sigma_e2']:.6f}")
    print(f"  h2 target : {meta['h2_target']}")
    print(f"  beta_main : {meta['beta_main']}")
    print(f"  beta_int  : {meta['beta_int']}")
    
    print(f"\n[Causal Pairs]")
    print(f"  {causal_pairs}")
    
    # Sanity checks
    print(f"\n[Sanity Checks]")
    assert X.shape == (200, 200), "X shape mismatch"
    assert set(np.unique(X)).issubset({0, 1, 2}), "Invalid genotype values"
    assert y.shape == (200,), "y shape mismatch"
    assert set(np.unique(y)) == {0, 1}, "y should be binary"
    assert causal_pairs.shape == (2, 2), "causal_pairs shape mismatch"
    print("  All sanity checks PASSED.")
    
    print("\n" + "=" * 60)
    print("Self-test complete. No files written.")
    print("=" * 60)
