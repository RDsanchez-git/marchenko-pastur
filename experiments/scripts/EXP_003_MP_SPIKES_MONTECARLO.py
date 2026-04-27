import importlib.metadata
import time

import numpy as np

from marchenko_pastur.api import run_mp

"""
EXPERIMENT 003: MP SPIKES MONTECARLO
---------------------------------------------

Objective:
---------
Evaluate the ability of the MP engine to detect low-rank signal
(spikes) injected into a Wishart noise background.

Methodology:
------------
Monte Carlo simulation where Gaussian noise matrices are augmented
with synthetic low-rank structure (spikes). Detection is performed
using the classical MP threshold.

Key Findings:
-------------
- Z-score standardization fixes trace → σ² ≈ 1
- Bulk eigenvalues shrink to compensate spike energy
- MP threshold becomes artificially high
- Weak signals become undetectable

Interpretation:
---------------
Working in correlation space (implicit standardization) distorts
the noise structure. This creates a mismatch between the true
background variance and the MP threshold, reducing sensitivity
to weak signals.
"""


# =========================================================
# DATA GENERATION (PURE FUNCTIONS)
# =========================================================

def generate_wishart(rng, n, p):
    return rng.normal(size=(n, p))


def inject_spikes(rng, X, n_spikes=3, strength=10):
    n, p = X.shape

    for _ in range(n_spikes):
        factor = rng.normal(size=(n, 1))
        loadings = rng.normal(size=(1, p))
        X = X + strength * (factor @ loadings) / np.sqrt(n)

    return X


# =========================================================
# MAIN EXPERIMENT
# =========================================================

def exp_003_mp_spikes_montecarlo(
    n=500,
    p=300,
    n_spikes=3,
    strength=10,
    M=200,
    seed=42,
):
    """
    Run Monte Carlo simulation to evaluate spike detection in Wishart noise.

    Parameters
    ----------
    n : int
        Number of observations.
    p : int
        Number of features.
    n_spikes : int
        Number of synthetic latent factors (spikes) to inject.
    strength : float
        Signal strength multiplier for the injected spikes.
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

    detected_spikes = []
    sigma_vals = []
    ratios = []

    for _ in range(M):
        X = generate_wishart(rng, n, p)
        X = inject_spikes(rng, X, n_spikes=n_spikes, strength=strength)

        result = run_mp(
            X,
            covariance="classical",
            threshold="mp",
        )

        detected_spikes.append(result.k_effective)
        sigma_vals.append(result.sigma2_hat)
        ratios.append(result.ratio_apex)

    elapsed = time.time() - start_time

    return {
        "results": {
            "mean_detected_spikes": float(np.mean(detected_spikes)),
            "std_detected_spikes": float(np.std(detected_spikes)),
            "mean_sigma2": float(np.mean(sigma_vals)),
            "mean_ratio": float(np.mean(ratios)),
        },
        "meta": {
            "n": n,
            "p": p,
            "n_spikes": n_spikes,
            "strength": strength,
            "M": M,
            "seed": seed,
            "execution_time_minutes": round(elapsed / 60, 2),
            "model_version": model_version,
        }
    }


if __name__ == "__main__":
    print("=== EXP_003_MP_SPIKES_MONTECARLO ===")

    output = exp_003_mp_spikes_montecarlo(
        n=500,
        p=300,
        n_spikes=3,
        strength=12,
        M=300,
    )

    print(output)