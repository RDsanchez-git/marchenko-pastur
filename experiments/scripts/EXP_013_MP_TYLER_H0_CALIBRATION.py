import importlib.metadata
import time

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from collections import Counter

from marchenko_pastur.api import run_mp

"""
EXPERIMENT 013: SPECTRAL DECALIBRATION (TYLER VS PEARSON UNDER H0)
---------------------------------------------

Objective:
---------
Empirically evaluate the spectral Relative Asymptotic Efficiency of Tyler's 
M-estimator against classical Pearson covariance under Gaussian white noise (H0). 
Demonstrate if the classical analytic Marchenko-Pastur (MP) threshold is sufficient 
to control the False Positive Rate (FPR) in robust spatial estimators.

Methodology:
------------
- Data: Pure Gaussian noise matrices (n=400, p=200, q=0.5).
- Execution: Parallel Monte Carlo (M=200 simulations).
- Process: Paired data comparison between Pearson (standardized) and Tyler (unstandardized).
- Metrics: Comparison of FPR, estimated theoretical limit (lambda_plus), and 
  inferred variance (sigma^2).

Key Findings:
-------------
1. Pearson + MP (Control): Nearly perfect calibration. FPR ~ 0.06, aligned with 
   nominal alpha=0.05 plus expected Tracy-Widom finite-sample fluctuations.
2. Tyler + MP (Decalibration): Exhibits chronically inflated FPR (~0.35). 
   The spectral bulk of Tyler's M-estimator is wider than the MP prediction 
   due to the angular dependence induced by the trace constraint.

Interpretation:
---------------
Tyler's ensemble does not strictly follow the Marchenko-Pastur law in finite samples. 
The spatial projection onto the unit sphere creates sample dependencies that 
push eigenvalues beyond the analytic limit. This necessitates Bootstrap calibration.
"""

# ======================================================================
# PRIVATE WORKER
# ======================================================================

def _run_single_sim(n: int, p: int, seed: int) -> tuple[int, int, float, float, float, float]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))

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

def exp_013_mp_tyler_h0_calibration(n: int = 400, p: int = 200, M: int = 200, seed: int = 42) -> dict:
    """
    Executes the Tyler vs Pearson H0 duel and returns a JSON-safe contract.

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
        - 'results': Computed metrics, FPR, and telemetry.
        - 'meta': Traceability configuration, execution time and model version.
    """
    try:
        model_version = importlib.metadata.version("marchenko_pastur")
    except importlib.metadata.PackageNotFoundError:
        model_version = "unknown"

    start_time = time.time()

    sq = np.random.SeedSequence(seed)

    # Pure integer seeds to avoid TypeError in run_mp
    child_seeds = [int(s) for s in sq.generate_state(M)]

    results = Parallel(n_jobs=-1)(
        delayed(_run_single_sim)(n, p, child_seeds[i])
        for i in tqdm(range(M), desc="Running Tyler H0 Duel")
    )

    k_p = np.array([r[0] for r in results]) #type: ignore
    k_t = np.array([r[1] for r in results]) #type: ignore
    lp_p = np.array([r[2] for r in results]) #type: ignore
    lp_t = np.array([r[3] for r in results]) #type: ignore
    s2_p = np.array([r[4] for r in results]) #type: ignore
    s2_t = np.array([r[5] for r in results]) #type: ignore

    tyler_counts = Counter(k_t.tolist())

    elapsed = time.time() - start_time

    return {
        "results": {
            "fpr_pearson": float(np.mean(k_p > 0)),
            "fpr_tyler": float(np.mean(k_t > 0)),
            "mean_spikes_tyler": float(np.mean(k_t)),
            "distribution_tyler": {str(k): int(v) for k, v in sorted(tyler_counts.items())},
            "telemetry": {
                "lambda_plus_pearson": float(np.mean(lp_p)),
                "lambda_plus_tyler": float(np.mean(lp_t)),
                "sigma2_pearson": float(np.mean(s2_p)),
                "sigma2_tyler": float(np.mean(s2_t))
            }
        },
        "meta": {
            "n": n, "p": p, "q": float(p/n), "M": M, "seed": seed,
            "execution_time_minutes": round(elapsed / 60, 2),
            "model_version": model_version
        }
    }

if __name__ == "__main__":
    print("=== RUNNING EXPERIMENT 013 ===")
    out = exp_013_mp_tyler_h0_calibration(M=10)
    
    print("\n=== RESULTS ===")
    print(f"FPR Tyler: {out['results']['fpr_tyler']}")
    
    print("\n=== META ===")
    for k, v in out['meta'].items():
        print(f"{k}: {v}")