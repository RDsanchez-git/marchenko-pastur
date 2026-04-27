import numpy as np
import pytest

from marchenko_pastur.api import run_mp


def test_shrinkage_runs(data_noise):
    """Basic test: Ledoit-Wolf shrinkage engine runs without crashing."""
    result = run_mp(
        data_noise,
        covariance="shrinkage",
        shrinkage_method="lw",
        threshold="mp",
        standardize_data=False,
    )

    assert result.lambda_plus > 0
    assert result.sigma2_hat > 0
    assert isinstance(result.k_effective, int)

@pytest.mark.slow
def test_shrinkage_wishart_h0(rng):
    """Under pure noise, shrinkage must strictly minimize false positive spikes."""
    n, p = 600, 300
    detections = []

    for _ in range(50):
        # Local RNG usage ensures trials are independent but the run is deterministic
        X = rng.normal(size=(n, p))
        result = run_mp(
            X, covariance="shrinkage", shrinkage_method="lw", threshold="mp", standardize_data=False
        )
        detections.append(result.k_effective)

    mean_detection = np.mean(detections)
    # SOTA FIX: Statistically robust threshold for finite trials
    assert mean_detection < 0.5


def test_shrinkage_p_greater_n(data_high_dim):
    """Shrinkage must survive and condition matrices in singular dimensions (P > N)."""
    result = run_mp(
        data_high_dim,
        covariance="shrinkage",
        shrinkage_method="lw",
        threshold="mp",
        standardize_data=False,
    )

    assert result.lambda_plus > 0
    assert result.sigma2_hat > 0
