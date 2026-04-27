import importlib.metadata
import time

import numpy as np
from joblib import Parallel, delayed
from scipy.linalg import toeplitz
from tqdm import tqdm

from marchenko_pastur.api import run_mp

"""
EXPERIMENT 009: MP VS BOOTSTRAP ROBUSTNESS TO AR1 CORRELATION
---------------------------------------------

Objective:
---------
Demonstrate the systematic collapse of the asymptotic Marchenko-Pastur (MP) limit 
when the i.i.d. noise hypothesis is violated. Validate the superiority of the 
empirical Bootstrap threshold in maintaining False Positive Rate (FPR) control 
under structured elliptical noise (Toeplitz AR1).

Methodology:
------------
- Data: Noise matrix with AR(1) covariance structure and k=3 latent factors.
- Dimensions: n=400 observations, p=200 variables (q=0.5).
- Correlation Sweep: rho parameter evaluated across the interval [0.0, 0.6].
- Execution: 250 parallel Monte Carlo simulations per rho level (using SeedSequence).
- Extraction: Classical threshold (lambda_plus), empirical threshold (lambda_star), and FPR.

Key Findings:
-------------
1. Threshold Divergence: As autocorrelation (rho) increases, the theoretical MP threshold 
   erroneously drops (from ~2.56 to ~0.82), while the Bootstrap threshold remains stable 
   and resilient (~2.92), adapting to the actual bulk deformation.
2. Classical Fragility: The MP asymptotic theory is highly fragile. At a minimal 
   correlation of rho=0.1, FPR explodes to ~63%. By rho=0.2, error control collapses entirely (FPR=100%).
3. SOTA Robustness & Breakpoint: The Bootstrap detector maintains a perfect shield 
   (FPR = 0.000) under moderate correlation (rho <= 0.2). A formal topological breakdown 
   occurs at rho=0.3, marking the validity domain of classical row-resampling against extreme serial dependence.

Interpretation:
---------------
The Marchenko-Pastur limit becomes unusable in empirical matrices exhibiting cross-correlation 
in their residuals. The Bootstrap framework corrects this geometric deficiency, replacing 
the rigid limit with an adaptive topological boundary that protects latent subspace dimensionality.
"""

# ======================================================================
# PRIVATE HELPERS
# ======================================================================

def _generate_correlated_data(rng: np.random.Generator, n: int, p: int, k: int, alpha: float, rho: float):
    """Generates AR(1) correlated Wishart noise with latent factors."""
    cov_matrix = toeplitz(rho ** np.arange(p))

    Z = rng.multivariate_normal(
        mean=np.zeros(p),
        cov=cov_matrix,
        size=n
    )

    factors = np.zeros((n, p))
    for _ in range(k):
        u = rng.normal(size=(n, 1))
        v = rng.normal(size=(1, p))
        v = v / np.linalg.norm(v)
        factors += u @ v

    return Z + alpha * factors

def _run_single_sim(rho: float, n: int, p: int, k: int, alpha: float, seed: int):
    """Isolated Monte Carlo step for joblib parallelization."""
    rng = np.random.default_rng(seed)
    
    X = _generate_correlated_data(rng, n, p, k, alpha, rho)

    res_mp = run_mp(X, covariance="classical", threshold="mp", standardize_data=True)
    res_boot = run_mp(X, covariance="classical", threshold="bootstrap", standardize_data=True, bootstrap_samples=100)

    fp_mp = max(0, int(res_mp.k_effective) - k)
    fp_boot = max(0, int(res_boot.k_effective) - k)

    return (fp_mp, fp_boot, float(res_mp.lambda_plus), float(res_boot.spike_threshold))

# ======================================================================
# EXPERIMENT FUNCTION
# ======================================================================

def exp_009_mp_robustness_ar1_sweep(n=400, p=200, k=3, alpha=3.0, simulations=250, seed=42):
    """
    Executes the AR(1) correlation sweep to compare MP vs Bootstrap robustness.

    Parameters
    ----------
    n : int
        Number of observations.
    p : int
        Number of features.
    k : int
        Number of latent factors.
    alpha : float
        Factor strength.
    simulations : int
        Monte Carlo iterations per rho level.
    seed : int
        Master random seed.

    Returns
    -------
    dict
        Structured output containing:
        - 'results': Computed metrics, thresholds, and FPR across the rho grid.
        - 'meta': Traceability configuration ,execution time and model version.
    """
    try:
        model_version = importlib.metadata.version("marchenko_pastur")
    except importlib.metadata.PackageNotFoundError:
        model_version = "unknown"

    rhos = np.linspace(0.0, 0.6, 7).tolist()
    
    out_rho, out_fpr_mp, out_fpr_boot = [], [], []
    out_se_mp, out_se_boot = [], []
    out_l_mp, out_l_boot = [], []

    start_time = time.time()
    sq = np.random.SeedSequence(seed)

    # 1. Visual Duel (Run once for interpretation context at rho=0.3)
    rng_visual = np.random.default_rng(seed)
    X_vis = _generate_correlated_data(rng_visual, n, p, k, alpha, rho=0.3)
    vis_mp = run_mp(X_vis, threshold="mp", standardize_data=True)
    vis_boot = run_mp(X_vis, threshold="bootstrap", standardize_data=True, bootstrap_samples=100)
    
    visual_payload = {
        "rho": 0.3,
        "k_mp": int(vis_mp.k_effective),
        "k_boot": int(vis_boot.k_effective),
        "lambda_mp": float(vis_mp.lambda_plus),
        "lambda_boot": float(vis_boot.spike_threshold)
    }

    # 2. Main Monte Carlo Sweep
    for rho in tqdm(rhos, desc="Evaluating AR1 Sweep"):
        child_seeds = sq.spawn(simulations)

        results = Parallel(n_jobs=-1)(
            delayed(_run_single_sim)(rho, n, p, k, alpha, child_seeds[i])
            for i in range(simulations)
        )

        # type: ignore to silence pylance over joblib inferences
        fp_mp = np.array([r[0] for r in results])    # type: ignore 
        fp_boot = np.array([r[1] for r in results])  # type: ignore
        l_mp = np.array([r[2] for r in results])     # type: ignore
        l_boot = np.array([r[3] for r in results])   # type: ignore

        fpr_mp = float(np.mean(fp_mp > 0))
        fpr_boot = float(np.mean(fp_boot > 0))

        out_rho.append(rho)
        out_fpr_mp.append(fpr_mp)
        out_fpr_boot.append(fpr_boot)
        out_se_mp.append(float(np.sqrt(fpr_mp * (1 - fpr_mp) / simulations)))
        out_se_boot.append(float(np.sqrt(fpr_boot * (1 - fpr_boot) / simulations)))
        out_l_mp.append(float(l_mp.mean()))
        out_l_boot.append(float(l_boot.mean()))

    elapsed = time.time() - start_time

    # ==================================================
    # RESULTS & META CONTRACT
    # ==================================================
    return {
        "results": {
            "rho": out_rho,
            "fpr_mp": out_fpr_mp,
            "fpr_boot": out_fpr_boot,
            "se_mp": out_se_mp,
            "se_boot": out_se_boot,
            "lambda_mp_mean": out_l_mp,
            "lambda_boot_mean": out_l_boot,
            "visual_duel": visual_payload
        },
        "meta": {
            "n": n,
            "p": p,
            "k_true": k,
            "alpha": alpha,
            "simulations": simulations,
            "seed": seed,
            "execution_time_minutes": round(elapsed / 60, 2),
            "model_version": model_version
        }
    }

if __name__ == "__main__":
    print("=== QUICK RUN: EXP_009_MP_ROBUSTNESS_AR1_SWEEP ===")
    out = exp_009_mp_robustness_ar1_sweep(simulations=5)
    print("\nVisual Duel:")
    print(out["results"]["visual_duel"])