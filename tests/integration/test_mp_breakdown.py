import numpy as np
import pytest

from marchenko_pastur.api import run_mp
from marchenko_pastur.pipelines.defactored_mp import defactored_mp


# ======================================================
# FIXTURES (Specific Data Generating Process)
# ======================================================
def generate_factor_model(n, p, k, strength=5.0, noise_std=1.0, seed=42):
    """Synthetic data generator injecting massive macro-factors."""
    rng = np.random.default_rng(seed)
    F = rng.standard_normal((n, k))
    Lambda = rng.standard_normal((k, p)) * strength
    E = rng.standard_normal((n, p)) * noise_std
    return F @ Lambda + E


@pytest.fixture(scope="module")
def factor_data():
    """Module-level fixture caching a matrix with a single massive factor (strength=8.0)."""
    return generate_factor_model(n=1000, p=200, k=1, strength=8.0)


# ======================================================
# REGRESSION TESTS: BREAKDOWN AND RECOVERY
# ======================================================
def test_original_disaster_iterative_collapse(factor_data):
    """Standard MP combined with iterative estimator must collapse under strong factors
    if standardized."""
    res = run_mp(
        factor_data,
        covariance="classical",
        threshold="tw",
        alpha=0.01,
        standardize_data=True,
        mp_sigma_estimator="iterative",
    )
    # Evaluate macroscopic collapse relative to dimension P, not a magic number
    p = factor_data.shape[1]
    assert res.k_effective > 0.5 * p
    assert res.sigma2_hat < 0.05


def test_unstandardized_success(factor_data):
    """Pure covariance without standardization must correctly isolate the true factor."""
    res = run_mp(
        factor_data,
        covariance="classical",
        threshold="tw",
        alpha=0.01,
        standardize_data=False,
        mp_sigma_estimator="iterative",
    )
    assert res.k_effective == 1
    assert 0.95 < res.sigma2_hat < 1.05


def test_median_estimator_fails(factor_data):
    """Median estimator alone cannot rescue a severely deformed standardized matrix."""
    res = run_mp(
        factor_data,
        covariance="classical",
        threshold="tw",
        alpha=0.01,
        standardize_data=True,
        mp_sigma_estimator="median",
    )
    assert 30 < res.k_effective < 80


def test_tyler_fails_on_standardized_data(factor_data):
    """Tyler estimator collapses if the input data is pre-standardized forcefully."""
    res = run_mp(
        factor_data,
        covariance="tyler",
        threshold="tw",
        alpha=0.01,
        standardize_data=True,
        mp_sigma_estimator="iterative",
    )
    assert res.k_effective > 100


def test_auto_circuit_breaker_triggers(factor_data):
    """AUTO mode must detect the collapse, fallback to trimmed, and log it in results."""
    # Act
    result = run_mp(factor_data, mp_sigma_estimator="auto")

    # Assert: We verify that the fallback occurred and was captured by the API
    has_breakdown_msg = any(
        "breakdown" in w.lower() for w in result.warnings
    )
    assert has_breakdown_msg, "The API should have caught the breakdown warning."
    assert result.sigma2_hat > 0


def test_sota_defactored_pipeline_perfection(factor_data):
    """The Two-Pass orchestrator must perfectly isolate the factor and restore i.i.d noise."""
    res = defactored_mp(factor_data, threshold="tw", alpha=0.01, max_factors=10, verbose=False)
    assert res.k_effective == 1
    assert 0.95 < res.sigma2_hat < 1.05
