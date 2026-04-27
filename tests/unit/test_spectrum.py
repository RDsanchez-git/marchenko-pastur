import numpy as np
import pytest

from marchenko_pastur.utils.eigen import compute_empirical_spectrum


# ==========================================================
# 1. Mathematical Equivalence of the Spectrum (3 Regimes)
# ==========================================================
@pytest.mark.parametrize(
    "n,p",
    [
        (300, 200),  # p <= n (Classical)
        (200, 500),  # n < p <= 5n (Singular)
        (100, 1200),  # p > 5n (Ultra high-dimensional)
    ],
)
def test_empirical_spectrum_equivalence(rng, n, p):
    """HPC spectrum must mathematically match exact sample covariance eigenvalues."""
    X = rng.normal(size=(n, p))

    Sigma = np.cov(X, rowvar=False, ddof=0)
    eig_true = np.linalg.eigvalsh(Sigma)

    # Numerical safeguard: remove tiny negative eigenvalues due to floating point precision
    eig_true = np.maximum(eig_true, 0)
    eig_true.sort()

    eig_hpc = compute_empirical_spectrum(X)

    assert eig_hpc.shape == eig_true.shape
    assert np.allclose(eig_true, eig_hpc, rtol=1e-5, atol=1e-7)


# ==========================================================
# 2. Numerical Stability in Ultra-High Dimension
# ==========================================================
def test_ultra_high_dimensional_regime(rng):
    """The dual trick must not introduce numerical instability in extreme P >> N regimes."""
    n = 80
    p = 4000

    X = rng.normal(size=(n, p))
    eigvals = compute_empirical_spectrum(X)

    assert eigvals.shape[0] == p
    assert np.all(eigvals >= -1e-12)
    assert np.all(np.diff(eigvals) >= -1e-12)
