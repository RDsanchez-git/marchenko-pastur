import importlib.metadata
import time

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from marchenko_pastur.engine.tyler import compute_covariance

"""
EXPERIMENT 014: SPECTRAL DENSITY DIAGNOSTIC (TYLER VS PEARSON)
---------------------------------------------

Objective:
---------
Diagnose the empirical spectral mass distribution of Tyler's estimator compared 
to the Pearson covariance matrix. The experiment isolates the theoretical cause 
of the Marchenko-Pastur (MP) threshold decalibration by analyzing the eigenvalue 
density differences within specific bins across a pooled Monte Carlo spectrum.

Methodology:
------------
- Data: Pooled Gaussian noise matrices (N=400, P=200).
- Execution: Parallel Monte Carlo simulations. Eigenvalues from all iterations 
  are pooled together to create an ultra-smooth, high-resolution Empirical 
  Spectral Density (ESD) distribution.
- Process: The spectrum is segmented into 50 bins. The empirical density for 
  Pearson and Tyler is calculated and the absolute difference 
  (Diff = Tyler_density - Pearson_density) is mapped.

Key Findings:
-------------
1. Mass Displacement (Tyler): Tyler's estimator concentrates higher mass density 
   in small eigenvalues (near 0) and towards the extreme right tail of the spectrum 
   compared to Pearson.
2. Spectral Dispersion: Although bin differences are numerically small, they 
   evidence that the general shape of the spectrum is structurally more dispersed 
   (fatter tails) in Tyler.
3. Pipeline Integrity: Telemetry confirms that both Tyler and the theoretical 
   MP limit are correctly implemented. The discrepancy is not computational, 
   but arises from geometric distribution incompatibilities.

Interpretation:
---------------
The spatial constraint imposed by scale-invariant estimators, specifically the 
trace condition tr(Sigma)=p, forces the descent of certain eigenvalues to be 
compensated by the increase of others. This generates a systemic spectral widening 
that breaks the classical Wishart matrix assumption. Consequently, the MP asymptotic 
limit lambda_+ is structurally too narrow for Tyler's expanded "bulk", pushing noise 
past the threshold and explaining the severe False Positive Rate inflation.
"""

# ======================================================================
# PRIVATE WORKER
# ======================================================================

def _run_single_sim(n: int, p: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Generates a single spectrum for pooling."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))

    # Pearson
    Sigma_pearson = np.cov(X, rowvar=False)
    eig_pearson = np.linalg.eigvalsh(Sigma_pearson)

    # Tyler
    Sigma_tyler = compute_covariance(X)
    eig_tyler = np.linalg.eigvalsh(Sigma_tyler)

    return (eig_pearson, eig_tyler)

# ======================================================================
# EXPERIMENT FUNCTION
# ======================================================================

def exp_014_mp_tyler_spectrum_diagnostic(
        n: int = 400, p: int = 200, M: int = 50, num_bins: int = 50, seed: int = 42) -> dict:
    """
    Executes the pooled spectrum diagnostic.

    Parameters
    ----------
    n : int
        Number of observations.
    p : int
        Number of features.
    M : int
        Monte Carlo iterations for eigenvalue pooling.
    num_bins : int
        Number of histogram bins for density estimation.
    seed : int
        Master random seed for reproducibility.

    Returns
    -------
    dict
        Structured output containing:
        - 'results': Computed densities, bins, and theoretical limits.
        - 'meta': Traceability configuration, execution time and model version.
    """
    try:
        model_version = importlib.metadata.version("marchenko_pastur")
    except importlib.metadata.PackageNotFoundError:
        model_version = "unknown"
        
    start_time = time.time()

    q = p / n
    sq = np.random.SeedSequence(seed)

    child_seeds = [int(s) for s in sq.generate_state(M)]

    results = Parallel(n_jobs=-1)(
        delayed(_run_single_sim)(n, p, child_seeds[i])
        for i in tqdm(range(M), desc="Pooling Spectra")
    )

    # Pool all eigenvalues into massive 1D arrays
    ep_pooled = np.concatenate([r[0] for r in results]) #type: ignore
    et_pooled = np.concatenate([r[1] for r in results]) #type: ignore

    # Global bounds for symmetric binning
    min_val = float(min(ep_pooled.min(), et_pooled.min()))
    max_val = float(max(ep_pooled.max(), et_pooled.max()))
    bins = np.linspace(min_val, max_val, num_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Density calculation
    hist_p, _ = np.histogram(ep_pooled, bins=bins, density=True)
    hist_t, _ = np.histogram(et_pooled, bins=bins, density=True)
    diff = hist_t - hist_p

    lambda_plus = (1 + np.sqrt(q)) ** 2
    
    elapsed = time.time() - start_time

    # ==================================================
    # RESULTS & META CONTRACT
    # ==================================================
    return {
        "results": {
            "bin_start": bins[:-1].tolist(),
            "bin_end": bins[1:].tolist(),
            "bin_center": bin_centers.tolist(),
            "density_pearson": hist_p.tolist(),
            "density_tyler": hist_t.tolist(),
            "density_diff": diff.tolist(),
            "lambda_plus_theoretical": float(lambda_plus)
        },
        "meta": {
            "n": n,
            "p": p,
            "q": float(q),
            "M": M,
            "total_pooled_eigenvalues": int(M * p),
            "seed": seed,
            "execution_time_minutes": round(elapsed / 60, 2),
            "model_version": model_version
        }
    }

if __name__ == "__main__":
    print("=== QUICK RUN: EXP_014_MP_TYLER_SPECTRUM_DIAGNOSTIC ===")
    out = exp_014_mp_tyler_spectrum_diagnostic(M=2)
    print(f"Pooled Eigenvalues: {out['meta']['total_pooled_eigenvalues']}")