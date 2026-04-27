r"""
Classical covariance estimators.

This module provides the standard empirical covariance estimator
used as a baseline within the spectral analysis pipeline.

The implementation is optimized for dense numerical linear algebra
and is designed to support both Random Matrix Theory (RMT)-consistent
and classical statistical covariance formulations.

This estimator supports both the maximum likelihood normalization
(:math:`1 / n`) required for strict consistency with the Marchenko-Pastur law,
and the unbiased sample normalization (:math:`1 / (n - 1)`) typical of
traditional statistical implementations.

The functions in this module are part of the public API and can be
used as drop-in covariance estimators via the ``covariance_fn``
interface across the library.

Typical usage includes:

- Baseline covariance estimation
- Input to spectral decomposition (PCA / eigen-analysis)
- Integration with Marchenko-Pastur filtering
- Bootstrap-based threshold estimation

Notes
-----
This module assumes that data is reasonably well-behaved (e.g.,
finite variance). For heavy-tailed or elliptical distributions,
consider using robust alternatives such as ``tyler`` estimators.

See Also
--------
tyler : Robust covariance estimator for heavy-tailed data.
shrinkage : Regularized covariance estimators.
"""
# ======================================================================
# THIRD-PARTY IMPORTS
# ======================================================================
import numpy as np
from numpy.typing import ArrayLike, NDArray

# ======================================================================
# LOCAL IMPORTS
# ======================================================================
from marchenko_pastur.enums.enums import CovarianceMethod

# ======================================================================
# PUBLIC API
# ======================================================================
__all__ = ["compute_covariance"]

# ======================================================================
# FUNCTIONS
# ======================================================================

def compute_covariance(
    X: ArrayLike,
    method: CovarianceMethod = CovarianceMethod.CLASSICAL,
    assume_centered: bool = False,
) -> NDArray[np.float64]:
    r"""
    Compute the empirical covariance matrix with exact degrees of freedom.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data matrix where rows represent observations and columns represent variables.
    method : CovarianceMethod, default=CovarianceMethod.CLASSICAL
        The estimator normalization method:
        - "classical": Maximum Likelihood Estimator (:math:`1/n`).
          Required for theoretical consistency with the Marchenko-Pastur law.
        - "pearson": Unbiased sample covariance (:math:`1/(n-1)`).
          Matches standard statistical practice but introduces a scaling mismatch
          with MP theory. Use only if consistency with sample covariance is strictly required.
    assume_centered : bool, default=False
        If True, assumes that `X` is already column-centered to avoid recomputing the mean.

    Returns
    -------
    S : ndarray of shape (n_features, n_features)
        Estimated covariance matrix.

    Raises
    ------
    ValueError
        If `X` is not 2D, is empty, contains non-finite values, or if :math:`n \le 1` under Pearson
        normalization.

    Notes
    -----
    **Theory**

    The covariance estimator computes:

    .. math::
        S = \frac{1}{d} X_c^T X_c

    where :math:`X_c = X - \bar{X}` denotes the column-centered data matrix.
    Centering is required for the unbiased (Pearson) estimator and is
    assumed unless explicitly disabled via ``assume_centered=True``.

    - **Classical (Wishart)**: :math:`d = n`. Required for strict Marchenko-Pastur asymptotics.
    - **Pearson (Sample)**: :math:`d = n - 1`. The standard unbiased estimator.

    Pearson normalization produces a uniformly rescaled covariance matrix
    relative to the classical estimator. This directly scales the
    subsequent Marchenko-Pastur boundaries.

    **Failure Modes**

    Statistical:

    - Mixing Pearson normalization (:math:`1/(n-1)`) with MP theory (which assumes :math:`1/n`)
      rescales the entire spectrum by a factor:

      .. math::
          \frac{n}{n-1}

      shifting both :math:`\lambda_-` and :math:`\lambda_+`.
      Mitigation: use ``method='classical'`` for strict MP inference, or adjust the MP bounds
      accordingly.

    Numerical:

    - Degenerate sample size (:math:`n \le 1`) leads to undefined or
      degenerate covariance estimates, particularly under Pearson normalization.
      Mitigation: ensure :math:`n \gg 1`.

    - Non-finite values in the input data matrix.
      Mitigation: clean data upstream before passing to the engine.

    **Complexity**

    - Time: :math:`O(n p^2)`
    - Memory: :math:`O(p^2)`

    See Also
    --------
    marchenko_pastur.utils.eigen.compute_empirical_spectrum : Fast-path spectrum extraction.
    marchenko_pastur.api.run_mp : High-level orchestrator.

    Examples
    --------
    >>> import numpy as np
    >>> from marchenko_pastur.enums.enums import CovarianceMethod
    >>> np.random.seed(42)
    >>> X = np.random.randn(100, 5)
    >>> # Classical (Wishart) normalization (1/n)
    >>> S_classical = compute_covariance(X, method=CovarianceMethod.CLASSICAL)
    >>> # Pearson (Sample) normalization (1/(n-1))
    >>> S_pearson = compute_covariance(X, method=CovarianceMethod.PEARSON)
    >>> np.allclose(S_pearson, S_classical * (100 / 99))
    True
    """

    # ---------------------------------------------------------------
    # Validation and coercion
    # ---------------------------------------------------------------
    X_arr = np.asarray(X, dtype=np.float64)

    if X_arr.ndim != 2:
        raise ValueError("X must be a 2D array.")

    n, p = X_arr.shape

    if n == 0 or p == 0:
        raise ValueError("The input matrix X cannot be empty.")

    if not np.isfinite(X_arr).all():
        raise ValueError("X contains NaN or infinite values.")

    is_pearson = (method == CovarianceMethod.PEARSON)

    if is_pearson and n <= 1:
        raise ValueError(
            "Pearson covariance requires at least 2 samples (n > 1) to avoid division by zero.")

    # ---------------------------------------------------------------
    # Data centering
    # ---------------------------------------------------------------
    if not assume_centered:
        X_arr = X_arr - X_arr.mean(axis=0, keepdims=True)

    # ---------------------------------------------------------------
    # Covariance Estimator (Applying strict degrees of freedom)
    # ---------------------------------------------------------------
    divisor = float(n - 1) if is_pearson else float(n)

    return (X_arr.T @ X_arr) / divisor
