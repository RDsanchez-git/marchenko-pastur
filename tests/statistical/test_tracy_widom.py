import numpy as np
import pytest

from marchenko_pastur.engine.classical import compute_covariance
from marchenko_pastur.utils.eigen import largest_eigenvalue


@pytest.mark.slow
def test_tracy_widom_edge(rng: np.random.Generator):
    """Verify that the maximum eigenvalue converges to the theoretical MP edge."""
    n = 600
    p = 300
    max_eigs = []

    for _ in range(40):
        # SOTA FIX: Fixture estricto para evitar Flaky Tests en CI
        X = rng.normal(size=(n, p))
        Sigma = compute_covariance(X)
        lam_max = largest_eigenvalue(Sigma)
        max_eigs.append(lam_max)

    max_eigs = np.array(max_eigs)

    q = p / n
    lambda_plus = (1 + np.sqrt(q)) ** 2
    mean_edge = max_eigs.mean()

    assert abs(mean_edge - lambda_plus) < 0.25
