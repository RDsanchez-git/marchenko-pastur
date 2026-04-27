import importlib.metadata
import time

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from marchenko_pastur.api import run_mp

"""
EXPERIMENT 011: ORTHOGONAL INVARIANCE AND STRUCTURAL ROBUSTNESS
---------------------------------------------

Objective:
---------
Demonstrate that the spectral detection algorithm depends strictly on the topology 
of the covariance matrix and is completely blind to orthogonal rotations (basis changes) 
in the variable space. Validate that the transformation X' = XQ preserves the exact 
number of extracted latent factors, confirming the geometric coherence of the pipeline.

Methodology:
------------
- Data: Matrices with k=3 latent factors injected into Gaussian noise, manually 
  standardized to isolate scale effects.
- Dimensions: n=500 observations, p=300 variables.
- Transformation: Multiplication by a random orthogonal matrix Q generated via QR decomposition.
- Execution: Parallel Monte Carlo simulations. The exact same seed controls the generation 
  and the Bootstrap resampling for both X and XQ to ensure an isolated topological comparison.
- Metrics: Absolute differences in the analytic limit (Delta lambda_+), empirical 
  threshold (Delta lambda_boot), and spike consistency.

Key Findings:
-------------
1. Absolute Structural Invariance: The detector identifies the exact same factors (k=3) 
   in 100% of the cases, regardless of the orthogonal basis representing the data.
2. Analytic Precision: The difference in the theoretical Marchenko-Pastur limit between 
   the original and rotated space is absolute zero (~1e-16, machine precision), confirming 
   that the spectra of Sigma and Sigma' = Q^T Sigma Q are strictly identical.
3. Resampling Marginal Sensitivity: The Bootstrap threshold differs significantly after 
   rotation. This evidences a theoretical limitation of independent column permutation: 
   the Q matrix spreads the original signal variance across all marginal distributions. 
   Permuting these "inflated" marginals causes Bootstrap to build an artificially heavier null distribution.

Interpretation:
---------------
The statistical pipeline is geometrically robust and spectrally invariant. Even though 
permutation resampling reacts to rotation-induced marginal deformation by raising a 
higher empirical barrier, the algorithm maintains perfect statistical power, consistently 
detecting the true latent subspace dimensionality. This certifies that the tool relies 
on the global RMT spectral structure, not the data parameterization.
"""

# ======================================================================
# PRIVATE HELPERS
# ======================================================================

def _random_orthogonal_matrix(rng: np.random.Generator, p: int):
    """Generates a random orthogonal matrix Q using QR decomposition."""
    A = rng.normal(size=(p, p))
    Q, _ = np.linalg.qr(A)
    return Q

def _generate_data(rng: np.random.Generator, n: int, p: int, k_true: int):
    """Generates pure noise, injects latent subspace, and manually standardizes."""
    Z = rng.normal(size=(n, p))
    U = rng.normal(size=(n, k_true))
    V = rng.normal(size=(p, k_true))
    
    signal = U @ V.T
    X = Z + signal

    # Manual standardization to strictly control scale geometry
    X = X - X.mean(axis=0)
    X = X / X.std(axis=0, ddof=1)
    
    return X

def _run_single_sim(n: int, p: int, k_true: int, B: int, alpha_sig: float, seed: int):
    """Isolated step: Generates X, rotates to XQ, and tests both in parallel."""
    rng = np.random.default_rng(seed)
    
    X = _generate_data(rng, n, p, k_true)
    Q = _random_orthogonal_matrix(rng, p)
    X_rot = X @ Q

    # Crucial: Using the exact same seed for the Bootstrap resampling guarantees 
    # that any difference comes purely from the data geometry, not stochastic noise.
    res_X = run_mp(
        X,
        covariance="classical",
        threshold="bootstrap",
        alpha=alpha_sig,
        bootstrap_samples=B,
        standardize_data=False,
        random_state=seed
    )

    res_XQ = run_mp(
        X_rot,
        covariance="classical",
        threshold="bootstrap",
        alpha=alpha_sig,
        bootstrap_samples=B,
        standardize_data=False,
        random_state=seed
    )

    k_orig = int(res_X.k_effective)
    k_rot = int(res_XQ.k_effective)
    l_plus_diff = float(abs(res_X.lambda_plus - res_XQ.lambda_plus))
    t_diff = float(abs(res_X.spike_threshold - res_XQ.spike_threshold))

    return (k_orig, k_rot, l_plus_diff, t_diff)

# ======================================================================
# EXPERIMENT FUNCTION
# ======================================================================

def exp_011_mp_orthogonal_invariance(n=500, p=300, k_true=3, M=200, B=250, alpha_sig=0.05, seed=42):
    """
    Executes the orthogonal invariance validation experiment.

    Parameters
    ----------
    n : int
        Number of observations.
    p : int
        Number of features.
    k_true : int
        Number of latent factors.
    M : int
        Number of Monte Carlo iterations.
    B : int
        Number of Bootstrap resamples per iteration.
    alpha_sig : float
        Target significance level for FPR.
    seed : int
        Master random seed.

    Returns
    -------
    dict
        Structured output containing:
        - 'results': Computed validation metrics and differentials.
        - 'meta': Traceability configuration and model version.
    """
    try:
        model_version = importlib.metadata.version("marchenko_pastur")
    except importlib.metadata.PackageNotFoundError:
        model_version = "unknown"

    start_time = time.time()
    sq = np.random.SeedSequence(seed)
    
    child_seeds = [int(s) for s in sq.generate_state(M)]

    results = Parallel(n_jobs=-1)(
        delayed(_run_single_sim)(n, p, k_true, B, alpha_sig, child_seeds[i])
        for i in tqdm(range(M), desc="Validating Invariance")
    )

    # type: ignore for Pylance over joblib
    k_orig = np.array([r[0] for r in results])    # type: ignore
    k_rot = np.array([r[1] for r in results])     # type: ignore
    l_diff = np.array([r[2] for r in results])    # type: ignore
    t_diff = np.array([r[3] for r in results])    # type: ignore

    consistency_rate = float(np.mean(k_orig == k_rot))

    elapsed = time.time() - start_time

    # ==================================================
    # RESULTS & META CONTRACT
    # ==================================================
    return {
        "results": {
            "consistency_rate": consistency_rate,
            "mean_lambda_plus_diff": float(np.mean(l_diff)),
            "mean_lambda_boot_diff": float(np.mean(t_diff)),
            "mean_k_original": float(np.mean(k_orig)),
            "mean_k_rotated": float(np.mean(k_rot)),
            "raw_k_original": k_orig.tolist(),
            "raw_k_rotated": k_rot.tolist(),
            "raw_l_diff": l_diff.tolist()
        },
        "meta": {
            "n": n,
            "p": p,
            "k_true": k_true,
            "M": M,
            "B": B,
            "seed": seed,
            "execution_time_minutes": round(elapsed / 60, 2),
            "model_version": model_version
        }
    }


if __name__ == "__main__":
    print("=== QUICK RUN: EXP_011_MP_ORTHOGONAL_INVARIANCE ===")
    out = exp_011_mp_orthogonal_invariance(M=5, B=10)
    
    print("\n=== RESULTS ===")
    print(f"Consistency: {out['results']['consistency_rate'] * 100}%")
    
    print("\n=== META ===")
    for k, v in out['meta'].items():
        print(f"{k}: {v}")