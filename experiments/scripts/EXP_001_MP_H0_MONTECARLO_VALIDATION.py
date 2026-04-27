import importlib.metadata
import time

import numpy as np

from marchenko_pastur.api import run_mp

"""
EXPERIMENT 001: MP H0 MONTECARLO
---------------------------------------------

Objective:
---------
Validate the behavior of the Marchenko-Pastur engine under the null
hypothesis (pure Wishart noise).

Methodology:
------------
Monte Carlo simulation across M repetitions using Gaussian random matrices.
Each iteration computes MP statistics on synthetic noise.

Key Findings:
-------------
- σ² estimator remains stable (~1.0) using mean (trace consistency)
- Small false positive rate due to finite-sample effects
- Evidence of Tracy-Widom fluctuations above λ+

Interpretation:
---------------
This experiment defines the baseline calibration of the MP engine.
It quantifies expected false positives under H0.
"""


def exp_001_mp_h0_montecarlo(n=500, p=300, M=200, seed=42):
    """
    Run Monte Carlo validation under Wishart null hypothesis.

    Parameters
    ----------
    n : int
        Number of observations.
    p : int
        Number of features.
    M : int
        Number of Monte Carlo iterations.
    seed : int
        Master random seed for reproducibility.

    Returns
    -------
    dict
        Structured output containing:
        - 'results': Computed metrics (k_effective, ratio_apex, sigma2_hat).
        - 'meta': Traceability configuration, execution time, and model version.
    """
    try:
        model_version = importlib.metadata.version("marchenko_pastur")
    except importlib.metadata.PackageNotFoundError:
        model_version = "unknown"

    start_time = time.time()
    
    rng = np.random.default_rng(seed)

    k_vals = []
    ratio_vals = []
    sigma_vals = []

    for _ in range(M):
        X = rng.normal(size=(n, p))
        result = run_mp(X)
        k_vals.append(result.k_effective)
        ratio_vals.append(result.ratio_apex)
        sigma_vals.append(result.sigma2_hat)

    elapsed = time.time() - start_time
    
    return {
        "results": {
            "mean_k": float(np.mean(k_vals)),
            "std_k": float(np.std(k_vals)),
            "mean_ratio": float(np.mean(ratio_vals)),
            "mean_sigma2": float(np.mean(sigma_vals)),
        },
        "meta": {
            "n": n,
            "p": p,
            "M": M,
            "seed": seed,
            "execution_time_minutes": round(elapsed / 60, 2),
            "model_version": model_version, 
        }
    }


if __name__ == "__main__":
    print("=== EXP_001_MP_H0_MONTECARLO ===")
    print(exp_001_mp_h0_montecarlo(n=500, p=300))