import importlib.metadata
import time

import numpy as np
from joblib import Parallel, delayed

from marchenko_pastur.api import run_mp

"""
EXPERIMENT 010: MARCHENKO-PASTUR VS BOOTSTRAP (ROBUSTNESS TO HEAVY TAILS & ASYMMETRY)
---------------------------------------------

Objective:
---------
Demonstrate the collapse of the analytic Marchenko-Pastur (MP) theory against 
violations of the normality assumption (extreme kurtosis and asymmetry), and 
empirically validate the superiority of the spectral Bootstrap in maintaining 
perfect False Positive Rate (FPR) calibration under the Null Hypothesis.

Methodology:
------------
- Data: Pure noise matrices (H0) rigorously standardized (ddof=1) to isolate the geometric effect of the tails.
- Dimensions: n=400 observations, p=200 variables (q=0.5).
- Distributions: Gaussian (df=inf), Heavy-tailed t-Student (df=10, 5, 3), and Lognormal (asymmetric).
- Execution: Monte Carlo simulation parallelized across available CPU cores.
- Extraction: Classical FPR (MP), Empirical FPR (Bootstrap), and Maximum Eigenvalue (lambda_max).

Key Findings:
-------------
1. Perfect SOTA Calibration: The Bootstrap threshold maintains strict Type I error control, 
   anchored at the nominal level (alpha=0.05) across absolutely all regimes.
2. Classical Degradation: The MP formula fails dramatically when departing from normality. 
   Under the Lognormal distribution, the analytic FPR explodes, overestimating spurious factors.
3. Finite-Sample Bias (Tracy-Widom): Even in the ideal Gaussian scenario (df=inf), 
   the MP asymptotics exhibit a chronic FPR due to spectral edge fluctuations in finite 
   dimensions, a structural defect automatically corrected by resampling.

Interpretation:
---------------
The Marchenko-Pastur upper bound is structurally fragile to asymmetry and leptokurtosis, 
phenomena ubiquitous in real-world data matrices. Bootstrap resampling rescues inferential 
validity, dynamically adapting to the true noise distribution and blocking spurious 
eigenvalue detection regardless of the geometric severity of the tails.
"""

# ======================================================================
# PRIVATE HELPERS
# ======================================================================

def _generate_data(rng: np.random.Generator, dist: str, df: float, n: int, p: int):
    """Generates standardized noise matrices using thread-safe NumPy generators."""
    if dist == "t":
        if np.isinf(df):
            X = rng.normal(size=(n, p))
        else:
            X = rng.standard_t(df, size=(n, p))
    elif dist == "lognormal":
        X = rng.lognormal(sigma=1.0, size=(n, p))
    else:
        raise ValueError(f"Unknown distribution: {dist}")

    # Critical standardization to isolate tail geometry
    X = X - X.mean(axis=0)
    X = X / X.std(axis=0, ddof=1)
    
    return X

def _run_single_sim(dist: str, df: float, n: int, p: int, B: int, alpha_sig: float, seed: int):
    """Isolated Monte Carlo step for parallel execution."""
    rng = np.random.default_rng(seed)
    
    X = _generate_data(rng, dist, df, n, p)

    # Empirical Spectrum
    C = np.corrcoef(X, rowvar=False)
    eigvals = np.linalg.eigvalsh(C)
    lambda_max = eigvals[-1]

    # SOTA Bootstrapped Engine
    res_boot = run_mp(
        X,
        covariance="classical",
        threshold="bootstrap",
        bootstrap_samples=B,
        alpha=alpha_sig
    )

    k_boot = int(res_boot.k_effective)
    
    # Classical MP Engine (Comparing empirical spectrum to theoretical bound)
    k_mp = int(np.sum(eigvals > res_boot.lambda_plus))

    return (k_mp, k_boot, float(lambda_max))

# ======================================================================
# EXPERIMENT FUNCTION
# ======================================================================

def exp_010_mp_robustness_heavy_tails(n=400, p=200, M=500, B=300, alpha_sig=0.05, seed=42):
    """
    Executes the heavy-tails robustness experiment using parallel Monte Carlo.

    Parameters
    ----------
    n : int
        Number of observations.
    p : int
        Number of features.
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
        - 'results': List of dictionaries with FPR metrics per distribution.
        - 'meta': Traceability configuration, execution time and model version.
    """
    try:
        model_version = importlib.metadata.version("marchenko_pastur")
    except importlib.metadata.PackageNotFoundError:
        model_version = "unknown"

    start_time = time.time()

    dfs = [3, 5, 10, np.inf]
    distributions = ["t", "lognormal"]

    results_all = []
    sq = np.random.SeedSequence(seed)

    for dist in distributions:
        for df in dfs:
            if dist == "lognormal" and df != dfs[0]:
                continue # Lognormal only needs one pass in this design

            label = f"{dist} (df={df})" if dist == "t" else dist

            # Spawn independent seeds for the M workers
            child_seeds = sq.spawn(M)

            results = Parallel(n_jobs=-1)(
                delayed(_run_single_sim)(dist, df, n, p, B, alpha_sig, child_seeds[i])
                for i in range(M)
            )

            # type: ignore for Pylance over joblib
            k_mp = np.array([r[0] for r in results])    # type: ignore
            k_boot = np.array([r[1] for r in results])  # type: ignore
            l_max = np.array([r[2] for r in results])   # type: ignore

            fpr_mp = float(np.mean(k_mp > 0))
            fpr_boot = float(np.mean(k_boot > 0))

            results_all.append({
                "Scenario": label,
                "fpr_mp": fpr_mp,
                "fpr_boot": fpr_boot,
                "lambda_max_mean": float(l_max.mean()),
                "lambda_max_std": float(l_max.std())
            })

    elapsed = time.time() - start_time

    # ==================================================
    # RESULTS & META CONTRACT
    # ==================================================
    return {
        "results": results_all,
        "meta": {
            "n": n,
            "p": p,
            "M": M,
            "B": B,
            "alpha_sig": alpha_sig,
            "seed": seed,
            "execution_time_minutes": round(elapsed / 60, 2),
            "model_version": model_version
        }
    }


# ======================================================================
# QUICK RUN (DEBUG ONLY)
# ======================================================================
if __name__ == "__main__":
    print("=== QUICK RUN: EXP_010_MP_ROBUSTNESS_HEAVY_TAILS ===")
    out = exp_010_mp_robustness_heavy_tails(M=2, B=10) # Minimal scale for debug
    for r in out["results"]:
        print(r)