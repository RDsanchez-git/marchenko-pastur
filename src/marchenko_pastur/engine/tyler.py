r"""
Robust covariance estimation via Tyler's M-estimator.

This module implements Tyler's M-estimator, a robust and
distribution-free estimator of scatter matrices under
elliptical distributions.

Unlike classical covariance estimators, Tyler's estimator:

- is invariant to scale
- is robust to outliers
- does not require finite moments (e.g., heavy-tailed data)

This makes it particularly suitable in high-dimensional
settings where the assumptions of Gaussian noise are violated.

The estimator is defined implicitly through a fixed-point
equation and is normalized to ensure identifiability.

Notes
-----
Tyler's estimator is a key component of the library's
robust pipeline and serves as a mitigation strategy for
failure modes of Random Matrix Theory (RMT), particularly:

- heavy-tailed distributions
- violation of finite fourth moment assumptions

However, it comes with important constraints:

- requires strictly n > p
- does not estimate scale (only shape)

See Also
--------
classical : Empirical covariance estimator (non-robust).
shrinkage : Regularized covariance estimators.
bootstrap : Empirical thresholding under weak assumptions.
"""

# ======================================================================
# STANDARD LIBRARY IMPORTS
# ======================================================================

import warnings

# ======================================================================
# THIRD-PARTY IMPORTS
# ======================================================================
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import cho_factor, cho_solve

# ======================================================================
# CONSTANTS
# ======================================================================

EPS = 1e-12

# ======================================================================
# PUBLIC API
# ======================================================================

__all__ = ["compute_covariance"]

# ======================================================================
# FUNCTIONS
# ======================================================================


def compute_covariance(
    X: ArrayLike,
    *,
    assume_centered: bool = False,
    max_iter: int = 100,
    tol: float = 1e-6,
    warm_start: bool = True,
) -> NDArray[np.float64]:
    r"""
    Estimate a robust scatter matrix using Tyler's M-estimator.

    This estimator provides a robust alternative to classical
    covariance estimation under elliptical distributions,
    particularly in the presence of heavy tails or outliers.

    Parameters
    ----------
    X : ArrayLike of shape (n_samples, n_features)
        Input data matrix.

    assume_centered : bool, default=False
        If True, assumes columns of X are already centered.

    max_iter : int, default=100
        Maximum number of fixed-point iterations.

    tol : float, default=1e-6
        Convergence tolerance based on relative Frobenius norm.

    warm_start : bool, default=True
        If True, initializes iterations using the empirical
        covariance estimator. Otherwise uses identity matrix.

    Returns
    -------
    Sigma : ndarray of shape (n_features, n_features)
        Estimated scatter matrix, normalized such that:

        .. math::
            \mathrm{trace}(\Sigma) = p

    Raises
    ------
    ValueError
        If input is invalid or if p >= n.

    Warns
    -----
    RuntimeWarning
        If the algorithm fails to converge within `max_iter`.

    Notes
    -----
    **Theory**

    Tyler's estimator is defined implicitly as the solution to:

    .. math::
        \Sigma = \frac{p}{n} \sum_{i=1}^{n} \frac{x_i x_i^T}{x_i^T \Sigma^{-1} x_i}

    The solution is unique up to a scaling constant, which is fixed
    via trace normalization:

    .. math::
        \mathrm{trace}(\Sigma) = p

    **Assumptions**

    - Observations follow an elliptical distribution.
    - Data is centered (or approximately so).
    - Finite moments are *not required*.

    **Algorithm**

    1. Initialize :math:`\Sigma` (empirical covariance or identity).
    2. Iterate fixed-point update.
    3. Normalize trace at each step.
    4. Stop when convergence criterion is met.

    **Interpretation**

    Tyler's estimator captures the *shape* of the data distribution
    while discarding scale information, making it robust to extreme
    observations.

    **Failure Modes**

    - Requires strictly :math:`n > p` (otherwise no unique solution).
      Mitigation: use ``shrinkage.compute_covariance`` when :math:`p \ge n`.

    - Slow or non-convergence in ill-conditioned settings.
      Mitigation: enable ``warm_start=True`` or increase ``max_iter``.

    - Loss of scale information (only shape is estimated).
      Mitigation: combine with external variance estimation
      (e.g., MP bulk fitting).

    - Not optimal under Gaussian assumptions.
      Mitigation: use ``classical.compute_covariance`` or
      ``shrinkage.compute_covariance``.

    See Also
    --------
    classical.compute_covariance : Empirical covariance estimator.
    shrinkage.compute_covariance : Regularized covariance estimator.
    bootstrap_lambda_crit : Empirical thresholding under weak assumptions.

    References
    ----------
    Tyler, D. E. (1987).
    A distribution-free M-estimator of multivariate scatter.
    Annals of Statistics, 15(1), 234-251.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> # Heavy-tailed data (Cauchy)
    >>> X = rng.standard_cauchy(size=(100, 5))
    >>> Sigma = compute_covariance(X, max_iter=50)
    >>> Sigma.shape
    (5, 5)
    """

    X_arr = np.asarray(X, dtype=float)

    if X_arr.ndim != 2:
        raise ValueError("X must be a 2D array.")

    if not np.isfinite(X_arr).all():
        raise ValueError("X contains non-finite values (NaN or Inf).")

    n, p = X_arr.shape

    if n == 0 or p == 0:
        raise ValueError("Data matrix X cannot be empty.")

    # ---------------------------------------------------------------
    # Defensa Matemática: Condición de Tyler (n > p)
    # ---------------------------------------------------------------
    if p >= n:
        raise ValueError(
            "Tyler's estimator strictly requires n > p. "
            "For p >= n, please use shrinkage estimators."
        )

    # ---------------------------------------------------------------
    # Centrado de datos (Defensa Estadística)
    # ---------------------------------------------------------------
    if not assume_centered:
        X_arr = X_arr - X_arr.mean(axis=0, keepdims=True)

    # ---------------------------------------------------------------
    # Initial guess
    # ---------------------------------------------------------------
    if warm_start:
        Sigma = np.cov(X_arr, rowvar=False)
        Sigma *= p / max(np.trace(Sigma), EPS)
    else:
        Sigma = np.eye(p)

    # ---------------------------------------------------------------
    # Fixed-point iterations
    # ---------------------------------------------------------------
    # ---------------------------------------------------------------
    # Fixed-point iterations
    # ---------------------------------------------------------------
    converged = False
    collinearity_warned = False

    for _ in range(max_iter):
        Sigma_old = Sigma.copy()

        try:
            c, lower = cho_factor(Sigma, check_finite=False)
            S_inv_X_T = cho_solve((c, lower), X_arr.T, check_finite=False)
        except np.linalg.LinAlgError:

            if not collinearity_warned:
                warnings.warn(
                    "Convergence warning: singular matrix detected due to collinear data. "
                    "Falling back to pseudo-inverse.",
                    UserWarning,
                    stacklevel=2,
                )
                collinearity_warned = True
            S_inv_X_T = np.linalg.pinv(Sigma) @ X_arr.T

        # Quadratic forms
        Q = np.sum(X_arr * S_inv_X_T.T, axis=1)
        Q = np.maximum(Q, EPS)

        # Tyler weights
        weights = p / Q

        # Weighted covariance update
        Sigma = (X_arr.T * weights) @ X_arr / n

        # Trace normalization
        Sigma *= p / max(np.trace(Sigma), EPS)

        # Convergence test
        if np.linalg.norm(Sigma - Sigma_old, ord="fro") / p < tol:
            converged = True
            break

    # ---------------------------------------------------------------
    # Software Defense: Silent Failure
    # ---------------------------------------------------------------
    if not converged:
        # Semantic telemetry required by the integration test
        warnings.warn(
            f"Tyler's estimator failed convergence after {max_iter} iterations, "
            "likely due to collinearity or singular structure.",
            UserWarning,
            stacklevel=2,
        )

    return Sigma
