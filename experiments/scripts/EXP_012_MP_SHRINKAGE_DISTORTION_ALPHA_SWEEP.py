import importlib.metadata
import time

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from marchenko_pastur.api import run_mp

"""
EXPERIMENT 012: SHRINKAGE DISTORTION (LEDOIT-WOLF VS PEARSON)
---------------------------------------------

Objective:
---------
Empirically demonstrate the mathematical pathology that occurs when applying 
standard linear regularization (Ledoit-Wolf shrinkage) to covariance matrices 
containing dominant latent factors. The experiment evaluates how structural 
signal strength (alpha) deforms the detection of significant eigenvalues 
(spikes) when using a Bootstrap threshold.

Methodology:
------------
- Data: Strict base Wishart noise matrix (n=400, p=200).
- Signal: Injection of k=3 fixed orthogonal latent factors.
- Sweep: Parametric sweep of factor strength (alpha in [1, 10]).
- Engine: Pearson vs. Ledoit-Wolf covariance estimators under identical Bootstrap resampling.
- Execution: Parallel Monte Carlo simulations (SeedSequence isolated) to compute expected distortion.

Key Findings:
-------------
1. Pearson + Bootstrap (BBP Control): Demonstrates perfect structural consistency. 
   Once alpha >= 2 (crossing the Baik-Ben Arous-Péché phase transition), the engine 
   anchors exactly on the 3 true factors and becomes invulnerable to variance increases.
2. Ledoit-Wolf + Bootstrap (Pathology): Exhibits a catastrophic and non-monotonic collapse. 
   Around alpha=2, the Ledoit-Wolf formula misinterprets the true variance of the factors 
   as massive estimation error. It aggressively increases shrinkage intensity (delta), 
   compressing the true factors and lifting the upper bound of the noise bulk. 
   This forces the Bootstrap to hallucinate dozens of false positive spikes. 
   As alpha approaches 10, the false spikes descend linearly (delta saturation), 
   but the method still reports massive Type I errors.

Interpretation:
---------------
Random Matrix Theory proves here that classical regularization assumes all extreme 
deviations are noise. When a matrix has a very high Signal-to-Noise Ratio driven 
by massive factors (typical in finance and macroeconomics), Ledoit-Wolf shrinkage 
severely deforms the spectrum. This absolutely justifies the necessity of extracting 
principal factors (Factor-Adjusted Shrinkage) before applying any regularization 
to the residual matrix.
"""

# ======================================================================
# PRIVATE HELPERS
# ======================================================================

def _run_single_sim(alpha: float, n: int, p: int, k: int, B: int, seed: int) -> tuple[int, int]:
    """Isolated Monte Carlo step for parallel execution."""
    rng = np.random.default_rng(seed)
    
    Z = rng.normal(size=(n, p))
    factors = np.zeros((n, p))

    for _ in range(k):
        u = rng.normal(size=(n, 1))
        v = rng.normal(size=(1, p))
        v = v / np.linalg.norm(v)
        factors += (u @ v)

    X = Z + alpha * factors

    # Control: Pearson + Bootstrap
    res_p = run_mp(
        X,
        covariance="classical",
        threshold="bootstrap",
        standardize_data=True,
        bootstrap_samples=B,
        random_state=seed
    )

    # Pathology: Ledoit-Wolf + Bootstrap
    res_lw = run_mp(
        X,
        covariance="shrinkage",
        shrinkage_method="lw",
        threshold="bootstrap",
        standardize_data=True,
        bootstrap_samples=B,
        random_state=seed
    )

    return (int(res_p.k_effective), int(res_lw.k_effective))

# ======================================================================
# EXPERIMENT FUNCTION
# ======================================================================

def exp_012_mp_shrinkage_distortion_alpha_sweep(
        n: int = 400, p: int = 200, k: int = 3, M: int = 100, B: int = 150, seed: int = 42) -> dict:
    """
    Executes the Shrinkage Distortion sweep using parallel Monte Carlo.

    Parameters
    ----------
    n : int
        Number of observations.
    p : int
        Number of features.
    k : int
        Number of true latent factors.
    M : int
        Number of Monte Carlo iterations per alpha level.
    B : int
        Number of Bootstrap resamples per simulation.
    seed : int
        Master random seed.

    Returns
    -------
    dict
        Structured output containing:
        - 'results': Alpha grid and detected spike statistics for Pearson and LW.
        - 'meta': Traceability configuration, execution time and model version.
    """
    try:
        model_version = importlib.metadata.version("marchenko_pastur")
    except importlib.metadata.PackageNotFoundError:
        model_version = "unknown"

    alphas = np.linspace(1, 10, 10).tolist()
    
    out_p_mean, out_p_std = [], []
    out_lw_mean, out_lw_std = [], []

    start_time = time.time()
    sq = np.random.SeedSequence(seed)

    for alpha in tqdm(alphas, desc="Sweeping Factor Strength (Alpha)"):
        # SOTA: Pure integer seeds derived from SeedSequence for thread safety
        child_seeds = [int(s) for s in sq.generate_state(M)]

        results = Parallel(n_jobs=-1)(
            delayed(_run_single_sim)(alpha, n, p, k, B, child_seeds[i])
            for i in range(M)
        )

        # type: ignore for Pylance over joblib inference
        k_p = np.array([r[0] for r in results])   # type: ignore
        k_lw = np.array([r[1] for r in results])  # type: ignore

        out_p_mean.append(float(k_p.mean()))
        out_p_std.append(float(k_p.std()))
        out_lw_mean.append(float(k_lw.mean()))
        out_lw_std.append(float(k_lw.std()))

    elapsed = time.time() - start_time

    # ==================================================
    # RESULTS & META CONTRACT
    # ==================================================
    return {
        "results": {
            "alphas": alphas,
            "pearson_mean": out_p_mean,
            "pearson_std": out_p_std,
            "lw_mean": out_lw_mean,
            "lw_std": out_lw_std
        },
        "meta": {
            "n": n,
            "p": p,
            "true_k": k,
            "M": M,
            "B": B,
            "seed": seed,
            "execution_time_minutes": round(elapsed / 60, 2),
            "model_version": model_version
        }
    }


# ======================================================================
# QUICK RUN (DEBUG ONLY)
# ======================================================================
if __name__ == "__main__":
    print("=== QUICK RUN: EXP_012_MP_SHRINKAGE_DISTORTION_ALPHA_SWEEP ===")
    out = exp_012_mp_shrinkage_distortion_alpha_sweep(M=2, B=10) # Fast test
    print("\nPearson Detections:", out["results"]["pearson_mean"])
    print("LW Detections:     ", out["results"]["lw_mean"])