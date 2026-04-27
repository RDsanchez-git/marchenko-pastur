import importlib.metadata
import time

import numpy as np
from marchenko_pastur.api import run_mp

"""
EXPERIMENT 004: BBP Phase Transition Validation
---------------------------------------------

Objective:
---------
Empirically validate the Baik-Ben Arous-Péché (BBP) phase transition
in the covariance setting by varying the spike strength (theta).

Methodology:
------------
- Generate a single base noise matrix and spike direction (Z, u, v)
- Sweep theta over a grid of values
- Inject signal with controlled intensity over the SAME base structure
- Apply MP inference (run_mp)
- Record detected spikes (k_effective)

Key Findings:
-------------
- For theta < sqrt(q): spike is absorbed → k = 0
- For theta > sqrt(q): spike detaches → k = 1
- Produces a sharp (Heaviside-like) transition when design is correct

Interpretation:
---------------
This experiment validates the BBP phase transition predicted by
Random Matrix Theory. It confirms that detectability of signal
depends critically on the signal-to-noise ratio relative to sqrt(q).
"""


# =========================================================
# BASE COMPONENT GENERATION (SOTA FIX)
# =========================================================
def _generate_base_components(rng, n, p):
    Z = rng.normal(size=(n, p))
    u = rng.normal(size=(n, 1))
    v = rng.normal(size=(1, p))

    # Normalización para controlar geometría del spike
    v = v / np.linalg.norm(v)

    return Z, u, v


# =========================================================
# MAIN EXPERIMENT (NAME MUST MATCH FILE)
# =========================================================
def exp_004_mp_spikes_bbp_transition(
    n=400,
    p=200,
    n_points=30,
    seed=42,
):
    """
    Executes the BBP phase transition sweep by varying signal strength (theta).

    Parameters
    ----------
    n : int
        Number of observations.
    p : int
        Number of features.
    n_points : int
        Number of grid points for the theta sweep.
    seed : int
        Random seed for reproducible noise and signal direction generation.

    Returns
    -------
    dict
        Structured output containing:
        - 'results': The theta grid and the corresponding number of detected spikes.
        - 'meta': Traceability configuration, dimensional ratios (q, sqrt_q), and model version.
    """

    try:
        model_version = importlib.metadata.version("marchenko_pastur")
    except importlib.metadata.PackageNotFoundError:
        model_version = "unknown"

    start_time = time.time()

    rng = np.random.default_rng(seed)

    q = p / n
    sqrt_q = np.sqrt(q)

    Z, u, v = _generate_base_components(rng, n, p)

    thetas = np.linspace(0.1, 4.0, n_points)
    detected = []


    for _, theta in enumerate(thetas):
        # Inyección controlada
        X = Z + np.sqrt(theta) * (u @ v)

        result = run_mp(X, standardize_data=False)
        detected.append(int(result.k_effective))

    elapsed = time.time() - start_time

    return {
        "results": {
            "thetas": thetas.tolist(),              
            "detected_spikes": detected,            
        },
        "meta": {
            "n": n,
            "p": p,
            "n_points": n_points,
            "q": float(q),
            "sqrt_q": float(sqrt_q),
            "seed": seed,
            "execution_time_minutes": round(elapsed / 60, 2),
            "model_version": model_version, 
        }
    }


if __name__ == "__main__":
    print("=== EXP_004_MP_SPIKES_BBP_TRANSITION ===")

    output = exp_004_mp_spikes_bbp_transition()

    print("\n=== RESULTS ===")
    print(output["results"])

    print("\n=== META ===")
    print(output["meta"])