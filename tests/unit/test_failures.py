import numpy as np
import pytest

from marchenko_pastur.api import run_mp

# ======================================================
# DATA GENERATING PROCESSES (Locals for Isolation)
# ======================================================

def generate_collinear_data(rng: np.random.Generator, n: int, p: int) -> np.ndarray:
    """Generates a perfectly rank-deficient matrix (exact column clones)."""
    X = rng.standard_normal((n, p // 2))
    return np.hstack([X, X])

def generate_nan_data(rng: np.random.Generator, n: int, p: int) -> np.ndarray:
    """Injects a single lethal NaN into an otherwise healthy matrix."""
    X = rng.standard_normal((n, p))
    X[n // 2, p // 2] = np.nan
    return X

# ======================================================
# HARD FAILS (API Input Validation & Gatekeeping)
# ======================================================

def test_nan_input_raises_value_error(rng):
    """Ensures the API completely blocks corrupted data before SVD execution."""
    X = generate_nan_data(rng, 100, 50)
    with pytest.raises(ValueError, match="NaN"):
        run_mp(X)

def test_invalid_shrinkage_config_raises_value_error(rng):
    """Ensures shrinkage is not assumed without an explicitly declared method."""
    X = rng.standard_normal((100, 50))
    with pytest.raises(ValueError, match="method"):
        run_mp(X, covariance="shrinkage")

def test_invalid_threshold_enum_raises_value_error(rng):
    """Ensures the internal Registry intercepts non-existent parameter strings."""
    X = rng.standard_normal((100, 50))
    with pytest.raises(ValueError, match="threshold"):
        run_mp(X, threshold="magic_formula")

def test_bootstrap_zero_samples_raises_value_error(rng):
    """Ensures mathematical paradoxes (empirical inference with B=0) are blocked."""
    X = rng.standard_normal((100, 50))
    with pytest.raises(ValueError):
        run_mp(X, threshold="bootstrap", bootstrap_samples=0)

# ======================================================
# SOFT FAILS (Graceful Degradation & Algorithmic Survival)
# ======================================================

def test_tyler_collinearity_triggers_warning(rng):
    """
    Ensures Tyler's estimator does not crash silently on singular matrices,
    but catches the convergence failure and reports it via warnings.
    """
    X = generate_collinear_data(rng, 200, 100)
    result = run_mp(X, covariance="tyler", standardize_data=False)

    assert hasattr(result, "warnings")
    assert any("convergence" in w.lower() or "collinear" in w.lower() for w in result.warnings)

def test_singular_regime_handled_safely(rng):
    """Ensures Pearson covariance scales correctly into P > N without crashing."""
    X = rng.standard_normal((50, 200))
    result = run_mp(X, covariance="classical", standardize_data=True)

    assert result is not None
    assert hasattr(result, "q")
    assert result.q > 1.0  # Mathematically confirms singular regime detection

def test_dense_signal_does_not_crash(rng):
    """Ensures extreme signal saturation does not break eigenvalue calculation."""
    E = rng.standard_normal((200, 100))
    F = rng.standard_normal((200, 20))
    Lambda = rng.standard_normal((20, 100)) * 8.0
    X = F @ Lambda + E

    result = run_mp(X, covariance="classical", threshold="tw", alpha=0.01, standardize_data=True)

    assert isinstance(result.k_effective, int)
    assert result.k_effective >= 0

def test_api_rejects_3d_tensors(rng: np.random.Generator):
    """Ensures RMT algorithms are protected against multi-dimensional tensors."""
    tensor_3d = rng.standard_normal((10, 10, 10))
    with pytest.raises(ValueError):
        run_mp(tensor_3d)

