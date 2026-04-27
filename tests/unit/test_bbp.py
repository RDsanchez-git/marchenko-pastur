import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_less

from marchenko_pastur.core.bbp import bbp_population_eigenvalue
from marchenko_pastur.core.mp_theory import mp_bounds


# ==========================================================
# 1. Scalar inversion + Round-Trip
# ==========================================================
def test_bbp_scalar_inversion_and_roundtrip():
    """Verify exact inversion and reconstruction of a single spike."""
    q = 0.5
    sigma2 = 1.0
    lambda_sample = 5.0

    ell = bbp_population_eigenvalue(lambda_sample, q, sigma2)

    assert np.isscalar(ell)

    # SOTA FIX: Coerción explícita a float para resolver Pylance (reportOperatorIssue)
    ell_val = float(np.atleast_1d(ell)[0])
    assert ell_val > sigma2
    assert ell_val < lambda_sample

    reconstructed_lambda = ell_val * (1 + (q * sigma2) / (ell_val - sigma2))
    assert_allclose(reconstructed_lambda, lambda_sample, rtol=1e-7)


# ==========================================================
# 2. Vectorized behaviour
# ==========================================================
def test_bbp_vectorized():
    """Ensure the BBP inversion handles NumPy arrays correctly without NaNs."""
    q = 0.3
    sigma2 = 1.0
    spikes = np.array([3.0, 4.5, 7.0])

    ell = bbp_population_eigenvalue(spikes, q, sigma2)

    assert isinstance(ell, np.ndarray)
    assert ell.shape == spikes.shape
    assert np.all(~np.isnan(ell))
    assert_array_less(ell, spikes)


# ==========================================================
# 3. Below bulk and exact boundary
# ==========================================================
def test_bbp_below_and_at_threshold():
    """Verify NaNs for noise eigenvalues and exact boundary matching."""
    q = 0.4
    sigma2 = 1.0
    _, lambda_plus = mp_bounds(q, sigma2)

    # Inside bulk
    lambda_below = lambda_plus * 0.95
    ell_below = bbp_population_eigenvalue(lambda_below, q, sigma2)
    assert np.isnan(ell_below)

    # Exact edge
    ell_edge = bbp_population_eigenvalue(lambda_plus, q, sigma2)
    ell_crit = sigma2 * (1 + np.sqrt(q))
    assert_allclose(ell_edge, ell_crit, atol=1e-7)


# ==========================================================
# 4. Near-edge numerical stability
# ==========================================================
def test_bbp_near_edge_stability():
    """Stress test the inversion formula extremely close to the MP edge."""
    q = 0.4
    sigma2 = 1.0
    _, lambda_plus = mp_bounds(q, sigma2)

    lam = lambda_plus * (1 + 1e-8)
    ell = bbp_population_eigenvalue(lam, q, sigma2)

    assert np.isfinite(ell)
    assert ell > sigma2
    assert ell < lam


# ==========================================================
# 5. Mixed vector input
# ==========================================================
def test_bbp_mixed_values():
    """Ensure vectorized inputs correctly map noise to NaNs and spikes to values."""
    q = 0.5
    sigma2 = 1.0
    _, lambda_plus = mp_bounds(q, sigma2)

    values = np.array([lambda_plus * 0.9, lambda_plus * 1.5, lambda_plus * 3.0])

    ell = bbp_population_eigenvalue(values, q, sigma2)

    # SOTA FIX: Coerción estricta a NDArray para habilitar el indexing [i]
    # y evitar que Pylance asuma que es un 'float' no subscriptable.
    ell_arr = np.atleast_1d(np.asarray(ell, dtype=np.float64))

    assert np.isnan(ell_arr[0])
    assert not np.isnan(ell_arr[1])
    assert not np.isnan(ell_arr[2])

    # SOTA FIX: Casting explícito a float puro para comparaciones estrictas
    assert float(ell_arr[1]) < float(values[1])
    assert float(ell_arr[2]) < float(values[2])


# ==========================================================
# 6. Monotonicity
# ==========================================================
def test_bbp_monotonicity():
    """Higher sample eigenvalues must yield higher population eigenvalues."""
    q = 0.4
    sigma2 = 1.0
    spikes = np.array([3.0, 4.0, 5.0, 6.0])

    ell = bbp_population_eigenvalue(spikes, q, sigma2)

    assert np.all(~np.isnan(ell))
    assert np.all(np.diff(ell) > 0)


# ==========================================================
# 7. Parameter validation
# ==========================================================
def test_bbp_invalid_parameters():
    """Ensure invalid inputs raise appropriate ValueErrors."""
    with pytest.raises(ValueError):
        bbp_population_eigenvalue(5.0, q=-0.1, sigma2=1.0)

    with pytest.raises(ValueError):
        bbp_population_eigenvalue(5.0, q=0.5, sigma2=0)
