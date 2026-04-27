import numpy as np
import pytest
import scipy.integrate as integrate

from marchenko_pastur.core.mp_fit import exact_mp_median

# ======================================================================
# EXACT MP MEDIAN PROPERTIES
# ======================================================================

def test_mp_median_limit_q_zero():
    """As q -> 0, MP collapses to a Dirac at 1."""
    assert exact_mp_median(0.0) == 1.0
    assert abs(exact_mp_median(1e-6) - 1.0) < 1e-3

def test_mp_median_monotonicity():
    """Median should decrease monotonically as dimensional ratio q increases."""
    q_vals = [0.1, 0.3, 0.5, 0.8]
    medians = [exact_mp_median(q) for q in q_vals]
    assert all(medians[i] > medians[i+1] for i in range(len(medians)-1))

def test_mp_median_within_support():
    """Median must lie strictly within the theoretical MP support bounds."""
    q_vals = [0.1, 0.5, 1.0]
    for q in q_vals:
        m = exact_mp_median(q)
        l_min = (1 - np.sqrt(q))**2
        l_max = (1 + np.sqrt(q))**2
        assert l_min <= m <= l_max

def test_mp_median_less_than_mean():
    """Due to right-skewness of MP density, median < mean (which is 1.0)."""
    for q in [0.1, 0.5, 1.0]:
        assert exact_mp_median(q) < 1.0

def test_mp_median_deterministic():
    """Function should be deterministic (validating LRU cache and numeric stability)."""
    q = 0.4
    m1 = exact_mp_median(q)
    m2 = exact_mp_median(q)
    assert abs(m1 - m2) < 1e-12

def test_mp_median_invalid_input():
    """Ensures negative ratios are strictly rejected."""
    with pytest.raises(ValueError, match="non-negative"):
        exact_mp_median(-0.1)

@pytest.mark.parametrize("q", [0.25, 0.5, 0.75, 1.0])
def test_mp_median_integral_definition(q):
    """
    The median MUST satisfy the strict mathematical definition: CDF(m) = 0.5.
    This invariant must hold regardless of the underlying implementation.
    """
    m = exact_mp_median(q)
    l_min = (1 - np.sqrt(q))**2
    l_max = (1 + np.sqrt(q))**2

    def mp_pdf(x):
        # Shields applied to the test environment as well
        x_safe = max(x, 1e-12)
        inside_sqrt = max((l_max - x_safe) * (x_safe - l_min), 0.0)
        return np.sqrt(inside_sqrt) / (2 * np.pi * q * x_safe)

    val, _ = integrate.quad(mp_pdf, l_min, m, limit=100)

    # 1e-4 tolerance allows for SciPy quadrature floating-point noise
    assert abs(val - 0.5) < 1e-4
