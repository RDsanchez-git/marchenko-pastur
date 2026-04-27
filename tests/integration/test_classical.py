import numpy as np
import pytest

from marchenko_pastur.api import run_mp


# =================================================
# TEST 1: No spikes under H0
# =================================================
@pytest.mark.parametrize("estimator", ["iterative", "median"])
def test_no_spikes_under_h0(data_noise, estimator):
    """Expect zero or minimal TW-fluctuation spikes under pure noise."""
    result = run_mp(data_noise, mp_sigma_estimator=estimator)
    assert result.k_effective <= 2


# =================================================
# TEST 2: Variance consistency (Iterative ONLY)
# =================================================
def test_sigma2_consistency_iterative(data_noise):
    """Iterative estimator should converge near true sigma^2 (1.0)."""
    result = run_mp(data_noise, mp_sigma_estimator="iterative")
    assert abs(result.sigma2_hat - 1.0) < 0.05


# =================================================
# TEST 3: Apex Ratio (Iterative ONLY)
# =================================================
def test_ratio_apex_close_to_one_iterative(data_noise):
    """Under pure noise, the largest eigenvalue should tightly bound the MP edge."""
    result = run_mp(data_noise, mp_sigma_estimator="iterative")
    assert 0.95 < result.ratio_apex < 1.05


# =================================================
# TEST 4: Median estimator sanity check
# =================================================
def test_median_estimator_returns_positive_sigma(data_noise):
    """Median heuristic must yield valid, finite, strictly positive variance."""
    result = run_mp(data_noise, mp_sigma_estimator="median")
    assert result.sigma2_hat > 0
    assert np.isfinite(result.sigma2_hat)


# =================================================
# TEST 5: Regimes
# =================================================
@pytest.mark.parametrize("estimator", ["iterative", "median"])
def test_regime_standard(data_noise, estimator):
    """Classical regime detection (N > P)."""
    result = run_mp(data_noise, mp_sigma_estimator=estimator)
    assert result.regime == "standard"
    assert result.zero_mass == 0.0


@pytest.mark.parametrize("estimator", ["iterative", "median"])
def test_singular_regime(data_high_dim, estimator):
    """Singular regime detection (P > N)."""
    result = run_mp(data_high_dim, mp_sigma_estimator=estimator)
    assert result.regime == "singular"
    assert result.zero_mass > 0.0


# =================================================
# TEST 6: Estimator consistency
# =================================================
def test_sigma_estimators_reasonable_difference(data_noise):
    """Iterative and median estimators should bound the same neighborhood."""
    r_iter = run_mp(data_noise, mp_sigma_estimator="iterative")
    r_med = run_mp(data_noise, mp_sigma_estimator="median")
    diff = abs(r_iter.sigma2_hat - r_med.sigma2_hat)
    assert diff < 1.0


# =================================================
# TEST 7: Invalid Input
# =================================================
def test_invalid_sigma_estimator_raises_error(data_noise):
    """Strict rejection of non-existent sigma estimators."""
    with pytest.raises(ValueError):
        run_mp(data_noise, mp_sigma_estimator="ols")

