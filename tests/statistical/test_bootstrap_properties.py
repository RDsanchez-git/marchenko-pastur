import numpy as np
import pytest

from marchenko_pastur.api import run_mp


# =====================================================================
# 1. Orthogonal Rotation Invariance
# =====================================================================
@pytest.mark.slow
def test_rotation_invariance(rng):
    """Bootstrap inference must be invariant to orthogonal rotations of the data."""
    n, p = 400, 200
    X = rng.normal(size=(n, p))

    # Inject arbitrary structure
    for _ in range(2):
        u = rng.normal(size=(n, 1))
        v = rng.normal(size=(1, p))
        v /= np.linalg.norm(v)
        X += 4.0 * (u @ v)

    # Random orthogonal matrix
    Q, _ = np.linalg.qr(rng.normal(size=(p, p)))
    X_rot = X @ Q

    res1 = run_mp(
        X,
        covariance="classical",
        threshold="bootstrap",
        bootstrap_samples=50,
        standardize_data=True,
        random_state=42,
    )
    res2 = run_mp(
        X_rot,
        covariance="classical",
        threshold="bootstrap",
        bootstrap_samples=50,
        standardize_data=True,
        random_state=42,
    )

    assert res1.k_effective == res2.k_effective


# =====================================================================
# 2. Signal Strength Monotonicity
# =====================================================================
@pytest.mark.slow
def test_signal_strength_monotonicity(rng):
    """Stronger latent factors must yield equal or greater spike detections."""
    n, p = 400, 200
    Z = rng.normal(size=(n, p))

    u = rng.normal(size=(n, 1))
    v = rng.normal(size=(1, p))
    v /= np.linalg.norm(v)

    X1 = Z + 3.0 * (u @ v)
    X2 = Z + 6.0 * (u @ v)

    res1 = run_mp(
        X1,
        covariance="classical",
        threshold="bootstrap",
        bootstrap_samples=50,
        standardize_data=True,
        random_state=42,
    )
    res2 = run_mp(
        X2,
        covariance="classical",
        threshold="bootstrap",
        bootstrap_samples=50,
        standardize_data=True,
        random_state=42,
    )

    assert res2.k_effective >= res1.k_effective


# =====================================================================
# 3. MP Scaling (q = p/n)
# =====================================================================
@pytest.mark.slow
def test_mp_scaling(rng):
    """Larger dimensional ratios (q) must strictly increase the theoretical MP edge."""
    n = 400
    X1 = rng.normal(size=(n, 100))  # q = 0.25
    X2 = rng.normal(size=(n, 300))  # q = 0.75

    res1 = run_mp(X1, covariance="classical", threshold="mp", standardize_data=True)
    res2 = run_mp(X2, covariance="classical", threshold="mp", standardize_data=True)

    assert res2.lambda_plus > res1.lambda_plus


# =====================================================================
# 4. Bootstrap Reproducibility
# =====================================================================
@pytest.mark.slow
def test_bootstrap_reproducibility(rng):
    """Identical random states must yield mathematically identical distributions and detections."""
    X = rng.normal(size=(400, 200))

    res1 = run_mp(
        X, covariance="classical", threshold="bootstrap", bootstrap_samples=40, random_state=123
    )
    res2 = run_mp(
        X, covariance="classical", threshold="bootstrap", bootstrap_samples=40, random_state=123
    )

    assert np.isclose(res1.lambda_plus, res2.lambda_plus, atol=1e-10)
    # SOTA FIX: Reproducibility must guarantee identical spike detection count
    assert res1.k_effective == res2.k_effective
