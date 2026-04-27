import numpy as np
import pytest

# ==========================================================
# BASE INFRASTRUCTURE
# ==========================================================


@pytest.fixture
def rng():
    """
    Isolated generator to guarantee determinism per test.
    Use this for local data generation where specific properties
    (like collinearity or BBP calibration) are needed.
    """
    return np.random.default_rng(42)


# ==========================================================
# GLOBAL BASELINES (DATA GENERATING PROCESSES)
# ==========================================================


@pytest.fixture
def data_noise(rng):
    """
    Classical regime (N > P), pure Wishart noise.
    Standard baseline for H0 tests.
    """
    return rng.normal(size=(300, 100))


@pytest.fixture
def data_high_dim(rng):
    """
    Singular regime (P > N), pure noise.
    Baseline for high-dimensional survival tests.
    """
    return rng.normal(size=(100, 500))


@pytest.fixture
def data_heteroskedastic(rng):
    """
    Classical regime with 2 dominant latent factors injected.
    Baseline for API and end-to-end pipeline verification.
    """
    n, p = 400, 150
    spikes = [15.0, 10.0]

    # Generate diagonal eigenvalue matrix
    eigvals = np.ones(p)
    eigvals[: len(spikes)] = spikes

    # Scale random normal data by the square root of the eigenvalues
    return rng.normal(size=(n, p)) * np.sqrt(eigvals)

@pytest.fixture
def data_spiked():
    """Generates a rank-2 Spiked Covariance Model (Johnstone, 2001)."""
    rng = np.random.default_rng(42)
    n, p = 200, 100
    X = rng.standard_normal((n, p))

    # Inyección de Factor 1 (Spike principal)
    v1 = rng.standard_normal((p, 1))
    v1 /= np.linalg.norm(v1)
    X += 6.0 * (rng.standard_normal((n, 1)) @ v1.T)

    # Inyección de Factor 2 (Spike secundario)
    v2 = rng.standard_normal((p, 1))
    v2 /= np.linalg.norm(v2)
    X += 4.5 * (rng.standard_normal((n, 1)) @ v2.T)

    return X
