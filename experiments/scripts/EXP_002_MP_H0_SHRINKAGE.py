import importlib.metadata
import time

import numpy as np

from marchenko_pastur.api import run_mp

"""
EXPERIMENT 002: MP H0 SHRINKAGE
---------------------------------------------

Objective:
---------
Evaluate the behavior of the MP framework under the null hypothesis
when using shrinkage covariance estimation (Ledoit-Wolf).

Methodology:
------------
Monte Carlo simulation with Gaussian random matrices.
Each iteration applies the MP pipeline using shrinkage covariance
instead of classical Pearson estimation.

Key Findings:
-------------
- Strong compression of the spectral right tail (ratio_apex collapse)
- Elimination of apparent Tracy-Widom leakage (k_effective ≈ 0)
- Breakdown of MP theoretical assumptions under shrinkage

Interpretation:
---------------
Shrinkage alters the eigenvalue distribution, making the classical
MP threshold (λ+) invalid. While useful for numerical stability,
it requires alternative thresholding methods (e.g., bootstrap).
"""
def exp_002_mp_h0_shrinkage(n=500, p=300, M=200, seed=42):
    """
    Monte Carlo validation under H0 using shrinkage covariance.

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

        result = run_mp(
            X,
            covariance="shrinkage",
            shrinkage_method="lw",
        )

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
    print("=== EXP_002_MP_H0_SHRINKAGE ===")

    print("\n[Standard regime p < n]")
    print(exp_002_mp_h0_shrinkage(n=500, p=300))

    print("\n[Near-square regime p ≈ n]")
    print(exp_002_mp_h0_shrinkage(n=400, p=380))

    print("\n[Singular regime p > n]")
    print(exp_002_mp_h0_shrinkage(n=300, p=500))