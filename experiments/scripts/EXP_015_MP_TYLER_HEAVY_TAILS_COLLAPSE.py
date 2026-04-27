import importlib.metadata
import time

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from collections import Counter

from marchenko_pastur.api import run_mp

"""
EXPERIMENT 015: BAI-YIN COLLAPSE AND SPATIAL ROBUSTNESS (TYLER VS PEARSON)
---------------------------------------------

Objective:
---------
Empirically evaluate the behavior of Tyler's M-estimator against classical Pearson 
covariance under heavy-tailed multivariate elliptical noise (t-Student, df=3). 
Contrast Pearson's asymptotic collapse (violation of Bai-Yin theorem) with Tyler's 
geometric stability in finite samples.

Methodology:
------------
- Data: Heavy-tailed elliptical noise matrices (multivariate t-Student, df=3).
- Dimensions: Finite-sample regime: N=400, P=200 (q=0.5).
- Execution: Parallel Monte Carlo (M=200 simulations).
- Process: Pearson (standardized to isolate scale bias) vs Tyler (unstandardized).
- Evaluated Metrics: FPR, False spike count, analytical limit (lambda_plus), 
  and inferred variance (sigma^2).

Key Findings:
-------------
1. Classical Collapse (Pearson Explosive Error): Violating the finite fourth 
   moment assumption renders Pearson unusable. Bulk variance collapses 
   (sigma^2 ~ 0.407), sinking the theoretical threshold (lambda_plus ~ 1.186). 
   This causes an FPR of 1.0 and massive hallucination (~45.7 false spikes/sim).
2. Structural Containment (Tyler Bounded Error): Tyler neutralizes massive radial 
   shocks and preserves bulk integrity (sigma^2 ~ 0.989, lambda_plus ~ 2.882). 
   It reduces error magnitude by ~45x compared to Pearson, limiting false 
   detections to ~1.04 spikes on average.
3. Finite-Sample Effect (Tyler Inflated FPR): Although stable in magnitude, Tyler 
   yields a high empirical FPR. Extreme outlier spatial normalization creates 
   small agglomerations on the unit sphere, pushing the leading eigenvalue 
   marginally past the rigid MP wall.

Interpretation:
---------------
A fundamental paradigm shift in robust RMT: under heavy tails, Pearson suffers 
an explosive, distribution-dependent error, while Tyler guarantees a structurally 
bounded (distribution-free magnitude) error. However, Tyler's high FPR proves 
its scale invariance is not asymptotically perfect against MP in finite samples, 
making empirical Bootstrap calibration mandatory for exact Type I error control.
"""

# ======================================================================
# PRIVATE HELPERS
# ======================================================================

def _generate_multivariate_t(rng: np.random.Generator, n: int, p: int, df: int) -> np.ndarray:
    """Generates heavy-tailed elliptical noise (multivariate t-Student) to violate Bai-Yin assumptions."""
    Z = rng.standard_normal(size=(n, p))
    W = rng.chisquare(df=df, size=(n, 1))
    X = Z * np.sqrt(df / W)
    return X

def _run_single_sim(n: int, p: int, df: int, seed: int) -> tuple[int, int, float, float, float, float]:
    """Isolated Monte Carlo step: Contrass Pearson's explosive error vs Tyler's containment."""
    rng = np.random.default_rng(seed)
    X = _generate_multivariate_t(rng, n, p, df)

    res_p = run_mp(X, covariance="classical", threshold="mp", standardize_data=True)
    res_t = run_mp(X, covariance="tyler", threshold="mp", standardize_data=False)

    return (
        int(res_p.k_effective), int(res_t.k_effective),
        float(res_p.lambda_plus), float(res_t.lambda_plus),
        float(res_p.sigma2_hat), float(res_t.sigma2_hat)
    )

# ======================================================================
# EXPERIMENT FUNCTION
# ======================================================================

def exp_015_mp_tyler_heavy_tails_collapse(n=400, p=200, df=3, M=200, seed=42):
    """
    Executes the Bai-Yin collapse and spatial robustness Monte Carlo simulation.

    Parameters
    ----------
    n : int
        Number of observations.
    p : int
        Number of features.
    df : int
        Degrees of freedom for the t-Student distribution (controls tail heaviness).
    M : int
        Number of Monte Carlo iterations.
    seed : int
        Master random seed for reproducibility.

    Returns
    -------
    dict
        Structured output containing:
        - 'results': Computed FPR, error magnitude, and internal telemetry.
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
        delayed(_run_single_sim)(n, p, df, child_seeds[i])
        for i in tqdm(range(M), desc="Simulating Heavy Tails Collapse")
    )

    k_p = np.array([r[0] for r in results]) # type: ignore
    k_t = np.array([r[1] for r in results]) # type: ignore
    lp_p = np.array([r[2] for r in results]) # type: ignore
    lp_t = np.array([r[3] for r in results]) # type: ignore
    s2_p = np.array([r[4] for r in results]) # type: ignore
    s2_t = np.array([r[5] for r in results]) # type: ignore

    elapsed = time.time() - start_time

    return {
        "results": {
            "fpr_pearson": float(np.mean(k_p > 0)),
            "fpr_tyler": float(np.mean(k_t > 0)),
            "mean_spikes_pearson": float(np.mean(k_p)),
            "mean_spikes_tyler": float(np.mean(k_t)),
            "lambda_plus_pearson": float(np.mean(lp_p)),
            "lambda_plus_tyler": float(np.mean(lp_t)),
            "sigma2_pearson": float(np.mean(s2_p)),
            "sigma2_tyler": float(np.mean(s2_t)),
            "dist_pearson": {str(k): int(v) for k, v in sorted(Counter(k_p.tolist()).items())},
            "dist_tyler": {str(k): int(v) for k, v in sorted(Counter(k_t.tolist()).items())},
            "raw_k_pearson": k_p.tolist(),
            "raw_k_tyler": k_t.tolist()
        },
        "meta": {
            "n": n, "p": p, "df": df, "M": M, "seed": seed,
            "execution_time_minutes": round(elapsed / 60, 2),
            "model_version": model_version, 
        }
    }

if __name__ == "__main__":
    print("=== QUICK RUN: EXP_015_MP_TYLER_HEAVY_TAILS_COLLAPSE ===")
    out = exp_015_mp_tyler_heavy_tails_collapse(M=10)
    print(f"Mean Spikes - Pearson: {out['results']['mean_spikes_pearson']}")
    print(f"Mean Spikes - Tyler:   {out['results']['mean_spikes_tyler']}")