from marchenko_pastur.api import run_mp


# ==========================================================
# 1. API Consistency across dimensional regimes
# ==========================================================
def test_run_mp_api_consistency(data_high_dim):
    """API must consistently process singular regimes (P > N) without raising errors."""
    result = run_mp(data_high_dim, covariance="classical", standardize_data=True)

    assert result.lambda_plus > 0
    assert result.sigma2_hat > 0
    assert result.k_effective >= 0
    assert result.regime in {"singular", "standard"}


# ==========================================================
# 2. End-to-end Pipeline with Latent Factors
# ==========================================================
def test_mp_pipeline_with_spikes(data_spiked):
    """Full pipeline must successfully detect injected latent factors from conftest baseline."""
    result = run_mp(data_spiked, covariance="classical", standardize_data=False)

    # The data_spiked baseline matrix is designed with exactly 2 spikes
    assert result.k_effective >= 2
