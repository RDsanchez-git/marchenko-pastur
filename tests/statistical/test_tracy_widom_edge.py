import numpy as np
import pytest

from marchenko_pastur.core.mp_theory import mp_bounds
from marchenko_pastur.utils.eigen import compute_empirical_spectrum


@pytest.mark.slow
def test_tracy_widom_edge_fluctuations(rng: np.random.Generator):
    """
    Verify that the maximum eigenvalue of a Wishart covariance matrix follows
    Tracy-Widom TW1 fluctuations around the MP edge (Johnstone, 2001).
    """
    n = 1200
    p = 800
    q = p / n
    sigma2 = 1.0
    trials = 120

    _, lambda_plus = mp_bounds(q, sigma2)

    # Johnstone (2001) Tracy-Widom scaling factor
    scale = sigma2 * (1 + np.sqrt(q)) ** (4 / 3) * (q ** (-1 / 6)) * (n ** (-2 / 3))

    tw_statistics = []

    for _ in range(trials):
        X = rng.normal(size=(n, p))
        eigvals = compute_empirical_spectrum(X, assume_centered=True)

        lambda_max = eigvals[-1]
        tw = (lambda_max - lambda_plus) / scale
        tw_statistics.append(tw)

    tw_statistics = np.array(tw_statistics)

    # Tracy-Widom TW1 Properties (Theoretical Mean ≈ -1.206, Std ≈ 1.268)
    # Relaxed bounds due to finite 'n' and limited Monte Carlo trials
    empirical_mean = np.mean(tw_statistics)
    empirical_std = np.std(tw_statistics)

    assert -2.5 < empirical_mean < 0.2
    assert 0.6 < empirical_std < 2.0

    # Additional check: raw max eigenvalue must concentrate near lambda_plus
    raw_lambda_max = lambda_plus + scale * tw_statistics

    assert np.mean(raw_lambda_max) > lambda_plus - 5 * scale
    assert np.mean(raw_lambda_max) < lambda_plus + 5 * scale

