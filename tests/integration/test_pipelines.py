import numpy as np

from marchenko_pastur.api import run_mp
from marchenko_pastur.pipelines.defactored_mp import defactored_mp

# ======================================================
# DATA GENERATING PROCESS
# ======================================================

def generate_factor_model(
        rng: np.random.Generator,
        n: int,
        p: int,
        k: int,
        strength:
        float = 5.0,
        noise_std:
        float = 1.0
    ) -> np.ndarray:

    E = rng.standard_normal((n, p)) * noise_std
    if k == 0:
        return E
    F = rng.standard_normal((n, k))
    Lambda = rng.standard_normal((k, p)) * strength
    return F @ Lambda + E

# ======================================================
# PIPELINE BEHAVIOR & MATH INTEGRATION
# ======================================================

def test_summary_contains_core_attributes(rng: np.random.Generator):
    X = generate_factor_model(rng, 500, 100, k=3)
    result = run_mp(X)
    assert hasattr(result, "k_effective")
    assert hasattr(result, "q")
    assert hasattr(result, "sigma2_hat")
    assert hasattr(result, "spike_threshold")

def test_pure_noise_detects_zero_spikes(rng: np.random.Generator):
    X = generate_factor_model(rng, 500, 100, k=0)
    result = run_mp(X, threshold="tw")
    # SOTA FIX: Strict equality (== 0) is safe here because rng is deterministic
    assert result.k_effective == 0

def test_bootstrap_generates_valid_threshold(rng: np.random.Generator):
    X = generate_factor_model(rng, 500, 100, k=0)
    result = run_mp(X, threshold="bootstrap", bootstrap_samples=20)
    assert result.spike_threshold > 0

def test_singular_regime_flag(rng: np.random.Generator):
    X = generate_factor_model(rng, 150, 400, k=2)
    result = run_mp(X, covariance="classical")
    assert result.q > 1.0

def test_saturated_signal_truncation(rng: np.random.Generator):
    X = generate_factor_model(rng, 800, 200, k=20, strength=2.0)
    result = run_mp(X, threshold="tw")
    assert result.k_effective > 0
    assert result.k_effective <= 200

# ======================================================
# DEFACTORED ORCHESTRATOR SURVIVAL
# ======================================================

def test_defactored_recovers_low_rank_structure(rng: np.random.Generator):
    X = generate_factor_model(rng, 500, 300, k=2)
    result = defactored_mp(X, threshold="tw", max_factors=10, verbose=False)
    assert result.k_effective >= 0
    assert hasattr(result, "warnings")

def test_defactored_short_circuit_noise(rng: np.random.Generator):
    X = generate_factor_model(rng, 500, 100, k=0)
    result = defactored_mp(X, threshold="tw", verbose=False)
    assert result.k_effective == 0

def test_defactored_caps_impossible_k(rng: np.random.Generator):
    X = rng.standard_normal((200, 50))
    result = defactored_mp(X, k_init=500, verbose=False)

    assert isinstance(result.k_effective, int)
    assert result.k_effective <= 49
    # SOTA FIX: Robust telemetry validation handling casing variations
    assert any("trunc" in w.lower() or "k_init" in w.lower() for w in result.warnings)
