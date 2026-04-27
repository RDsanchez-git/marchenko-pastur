import numpy as np

from marchenko_pastur.api import run_mp

# ==========================================================
# BBP Phase Transition Test (SOTA Exact)
# ==========================================================

def test_bbp_phase_transition_regions(rng: np.random.Generator):
    """
    Evaluate the BBP phase transition using strict subcritical and supercritical regions
    to prevent flaky tests caused by finite-size Tracy-Widom fluctuations near the edge.
    """
    n, p = 500, 800

    # Test subcritical region (Theta = 0.5) -> Must be fully absorbed by bulk
    X_sub = rng.normal(size=(n, p))
    u_sub = rng.normal(size=(n, 1))
    v_sub = rng.normal(size=(1, p))
    u_sub -= np.mean(u_sub)
    v_sub -= np.mean(v_sub)
    u_sub /= np.linalg.norm(u_sub)
    v_sub /= np.linalg.norm(v_sub)
    X_sub += np.sqrt(0.5 * n) * (u_sub @ v_sub)

    res_sub = run_mp(X_sub, covariance="classical", standardize_data=False)
    assert res_sub.k_effective == 0

    # Test supercritical region (Theta = 2.0) -> Must clearly separate from bulk
    X_super = rng.normal(size=(n, p))
    u_super = rng.normal(size=(n, 1))
    v_super = rng.normal(size=(1, p))
    u_super -= np.mean(u_super)
    v_super -= np.mean(v_super)
    u_super /= np.linalg.norm(u_super)
    v_super /= np.linalg.norm(v_super)
    X_super += np.sqrt(2.0 * n) * (u_super @ v_super)

    res_super = run_mp(X_super, covariance="classical", standardize_data=False)
    assert res_super.k_effective >= 1

def test_bbp_strict_inversion_accuracy(rng: np.random.Generator):
    """Ensure exact recovery of the injected signal magnitude, reversing asymptotic deformation."""
    theta_injected = 5.0
    n, p = 500, 800  # Singular regime (q = 1.6)

    X = rng.normal(0, 1.0, size=(n, p))
    u = rng.normal(size=(n, 1))
    v = rng.normal(size=(1, p))

    u -= np.mean(u)
    v -= np.mean(v)
    u /= np.linalg.norm(u)
    v /= np.linalg.norm(v)

    X += np.sqrt(theta_injected * n) * (u @ v)

    result = run_mp(X, covariance="classical", standardize_data=False)

    assert 0.95 < result.sigma2_hat < 1.05

    recovered_pop_eig = result.population_eigenvalues[0]
    expected_pop_eig = 1.0 + theta_injected

    relative_error = abs(recovered_pop_eig - expected_pop_eig) / expected_pop_eig
    assert relative_error < 0.10
