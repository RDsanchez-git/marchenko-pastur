import numpy as np
import pytest

from marchenko_pastur.api import run_mp


@pytest.mark.slow
def test_tw_false_positive_rate(rng: np.random.Generator):
    """Verify the empirical False Positive Rate (FPR) of TW matches the target alpha."""
    n = 500
    p = 200
    alpha = 0.05
    simulations = 400
    detections = 0

    for _ in range(simulations):
        # SOTA FIX: Fixture estricto para reproducibilidad garantizada
        X = rng.standard_normal((n, p))
        res = run_mp(X, covariance="classical", threshold="tw", alpha=alpha)

        if res.k_effective > 0:
            detections += 1

    fpr = detections / simulations
    se = np.sqrt(alpha * (1 - alpha) / simulations)
    tolerance = 3 * se

    assert abs(fpr - alpha) < tolerance
