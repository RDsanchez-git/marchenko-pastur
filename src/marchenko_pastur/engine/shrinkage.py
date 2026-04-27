r"""
Shrinkage covariance estimators.

This module provides regularized covariance estimators designed for
high-dimensional settings where the empirical covariance matrix is
ill-conditioned or singular.

Two widely used estimators are supported:

- Ledoit-Wolf (LW)
- Oracle Approximating Shrinkage (OAS)

These estimators improve numerical stability by shrinking the sample
covariance matrix toward a structured target, typically a scaled
identity matrix.

The functions in this module are part of the public API and can be
used interchangeably via the ``covariance_fn`` interface across the
library.

This module is particularly relevant when:

- p \approx n (moderate ill-conditioning)
- p > n (singular covariance)
- numerical stability is critical for spectral decomposition

Notes
-----
Shrinkage improves conditioning but introduces bias. In the context
of Random Matrix Theory (RMT), excessive shrinkage may distort the
spectral structure and affect spike detection.

For heavy-tailed or non-Gaussian data, consider robust estimators
(e.g., ``tyler``) instead.

See Also
--------
classical : Empirical covariance estimator without regularization.
tyler : Robust covariance estimator for elliptical distributions.
"""

# ======================================================================
# STANDARD LIBRARY IMPORTS
# ======================================================================

from typing import Literal

# ======================================================================
# THIRD-PARTY IMPORTS
# ======================================================================
import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.covariance import OAS, LedoitWolf

# ======================================================================
# PUBLIC API
# ======================================================================

__all__ = ["compute_covariance"]

# ======================================================================
# FUNCTIONS
# ======================================================================


def compute_covariance(
    X: ArrayLike,
    method: Literal["lw", "oas"] = "lw",
    assume_centered: bool = False,
) -> NDArray[np.float64]:
    r"""
    Estimate a regularized (shrinkage) covariance matrix.

    This function computes a covariance estimator that improves
    numerical conditioning by shrinking the empirical covariance
    toward a structured target.

    Parameters
    ----------
    X : ArrayLike of shape (n_samples, n_features)
        Input data matrix.

    method : {"lw", "oas"}, default="lw"
        Shrinkage estimator:

        - "lw"  : Ledoit-Wolf (asymptotically optimal)
        - "oas" : Oracle Approximating Shrinkage (small-sample optimal)

    assume_centered : bool, default=False
        If True, assumes columns of X are already centered.

    Returns
    -------
    Sigma : ndarray of shape (n_features, n_features)
        Regularized covariance matrix.

    Raises
    ------
    ValueError
        If inputs are invalid or method is unknown.

    Notes
    -----
    **Theory**

    Shrinkage estimators take the form:

    .. math::
        \hat{\Sigma} = (1 - \alpha) S + \alpha F

    where:

    - :math:`S` is the sample covariance
    - :math:`F` is a structured target (typically scaled identity)
    - :math:`\alpha` is the shrinkage intensity

    **Ledoit-Wolf (LW)**

    - Minimizes Frobenius risk asymptotically
    - Robust in high-dimensional regimes

    **OAS**

    - Assumes Gaussian data
    - Lower bias in small samples

    **Interpretation**

    Shrinkage reduces variance at the cost of bias,
    producing more stable eigenvalues and improving
    numerical robustness in spectral methods.

    **Failure Modes**

    - Over-shrinkage may compress eigenvalue dispersion,
      masking weak spikes and reducing sensitivity in
      spectral detection tasks.

    - Gaussian assumption (OAS) may be violated in
      heavy-tailed data.
      Mitigation: use ``covariance='tyler'`` for
      robust estimation.

    - Shrinkage alters spectral geometry, potentially
      biasing RMT-based thresholds.
      Mitigation: prefer ``method="lw"`` or use
      classical covariance when theoretical fidelity
      is critical.

    See Also
    --------
    classical.compute_covariance : Empirical covariance estimator.
    tyler.compute_covariance : Robust covariance estimator.

    References
    ----------
    Ledoit, O., & Wolf, M. (2004).
    A well-conditioned estimator for large-dimensional covariance matrices.

    Chen, Y., Wiesel, A., Eldar, Y. C., & Hero, A. O. (2010).
    Shrinkage algorithms for MMSE covariance estimation.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> X = rng.normal(size=(50, 100))
    >>> Sigma = compute_covariance(X, method="lw")
    >>> Sigma.shape
    (100, 100)
    """

    # ---------------------------------------------------------------
    # Validation and coercion
    # ---------------------------------------------------------------

    X_arr = np.asarray(X, dtype=float)

    if X_arr.ndim != 2:
        raise ValueError("X must be a 2D array.")

    n, p = X_arr.shape

    if n == 0 or p == 0:
        raise ValueError("Data matrix X cannot be empty.")

    if not np.isfinite(X_arr).all():
        raise ValueError("X contains non-finite values (NaN or Inf).")

    # ---------------------------------------------------------------
    # Estimator selection and instantiation
    # ---------------------------------------------------------------

    if method == "lw":
        estimator = LedoitWolf(store_precision=False, assume_centered=assume_centered)

    elif method == "oas":
        estimator = OAS(store_precision=False, assume_centered=assume_centered)

    else:
        raise ValueError("Unknown shrinkage method. Use 'lw' (Ledoit-Wolf) or 'oas'.")

    # ---------------------------------------------------------------
    # Fit (Delegates to scikit-learn optimized C subroutines)
    # ---------------------------------------------------------------

    estimator.fit(X_arr)

    return estimator.covariance_
