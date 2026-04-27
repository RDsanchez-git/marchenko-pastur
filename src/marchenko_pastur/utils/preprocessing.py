r"""
Preprocessing utilities for Random Matrix Theory (RMT) analysis.

This module provides core data transformations required before applying
spectral methods such as Marchenko-Pastur inference.

The focus is on ensuring strict consistency with the assumptions of
high-dimensional random matrix theory, particularly:

- Mean-centered data.
- Variance normalization using Maximum Likelihood Estimation (MLE).
- Robust handling of degenerate features and factor structures.

The module includes:

- ``standardize``: column-wise Z-score normalization under MLE scaling.
- ``defactor_data``: removal of latent factors via PCA with idiosyncratic scaling.

Notes
-----
All transformations are designed to be consistent with covariance estimators
of the form:

.. math::
    \Sigma = \frac{1}{n} X^\top X

This ensures compatibility with the asymptotic limits of the
Marchenko-Pastur distribution.

All functions enforce:

- Input immutability (defensive copying where required).
- Strict validation (shape and numerical integrity).
- Explicit handling of edge cases (zero variance, rank deficiency).
"""

# ======================================================================
# STANDARD LIBRARY IMPORTS
# ======================================================================
import logging
import warnings

# ======================================================================
# THIRD-PARTY IMPORTS
# ======================================================================
import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.decomposition import PCA

# ======================================================================
# MODULE CONFIGURATION
# ======================================================================
logger = logging.getLogger(__name__)

# ======================================================================
# PUBLIC API EXPORTS
# ======================================================================
__all__ = [
    "standardize",
    "defactor_data",
]

# ======================================================================
# FUNCTIONS
# ======================================================================


def standardize(X: ArrayLike) -> NDArray[np.float64]:
    r"""
    Standardize data column-wise using Z-score normalization.

    Each feature is centered and scaled to unit variance using
    Maximum Likelihood Estimation (MLE), i.e., normalization by ``n``.

    Parameters
    ----------
    X : ArrayLike of shape (n_samples, n_features)
        Input data matrix.

    Returns
    -------
    X_std : ndarray of shape (n_samples, n_features)
        Standardized data matrix.

    Raises
    ------
    ValueError
        If ``X`` is not 2D or contains non-finite values.

    Notes
    -----
    The variance is computed with ``ddof=0``:

    .. math::
        \sigma_j^2 = \frac{1}{n} \sum_{i=1}^n (x_{ij} - \bar{x}_j)^2

    This ensures compatibility with Random Matrix Theory (RMT),
    where covariance matrices are defined as:

    .. math::
        \Sigma = \frac{1}{n} X^\top X

    Columns with zero variance are detected and handled by replacing
    their standard deviation with 1.0, effectively performing only
    mean-centering.

    A ``UserWarning`` is emitted in such cases.
    """
    X_arr = np.asarray(X, dtype=np.float64)

    # --------------------------------------------------
    # O(1) Structural Validations
    # --------------------------------------------------
    if X_arr.ndim != 2:
        raise ValueError("X must be a 2D array.")

    # --------------------------------------------------
    # O(n*p) Content Validations
    # --------------------------------------------------
    if not np.isfinite(X_arr).all():
        raise ValueError("X contains non-finite values (NaN or Inf).")

    # --------------------------------------------------
    # MLE Standardization (RMT Consistency: ddof=0)
    # --------------------------------------------------
    mean = np.mean(X_arr, axis=0, keepdims=True)
    std = np.std(X_arr, axis=0, ddof=0, keepdims=True)

    zero_var = std <= np.finfo(std.dtype).eps

    if np.any(zero_var):
        warnings.warn(
            "Zero variance columns detected during standardization. "
            "These columns will only be mean-centered.",
            UserWarning,
            stacklevel=2,
        )
        std[zero_var] = 1.0

    return (X_arr - mean) / std


def defactor_data(
    X: ArrayLike,
    k: int,
    verbose: bool = True,
) -> NDArray[np.float64]:
    r"""
    Remove latent factors via PCA and scale by idiosyncratic variance.

    This function decomposes the data matrix into a low-rank factor
    structure and a residual component, then rescales the data using
    the idiosyncratic (residual) variance.

    Parameters
    ----------
    X : ArrayLike of shape (n_samples, n_features)
        Input data matrix.

    k : int
        Number of latent factors to remove.

    verbose : bool, default=True
        If True, logs progress messages.

    Returns
    -------
    X_clean : ndarray of shape (n_samples, n_features)
        Defactored and rescaled data matrix.

    Raises
    ------
    ValueError
        If ``X`` is not 2D, contains non-finite values, or if
        ``k`` is outside the valid range ``[0, min(n_samples, n_features)]``.

    Notes
    -----
    **Purpose**

    In high-dimensional settings, strong common factors inflate
    the largest eigenvalues of the covariance matrix, violating
    the assumptions of the Marchenko-Pastur law.

    This function removes such factors to recover the bulk spectrum.

    **Algorithm**

    1. Mean-center the data:

       .. math::
           X_c = X - \bar{X}

    2. Extract top ``k`` principal components via PCA.

    3. Reconstruct the low-rank approximation:

       .. math::
           \hat{X} = F \Lambda^\top

    4. Compute residuals:

       .. math::
           R = X_c - \hat{X}

    5. Estimate idiosyncratic variance (MLE scaling):

       .. math::
           \sigma_j^2 = \frac{1}{n} \sum_{i=1}^n R_{ij}^2

    6. Rescale:

       .. math::
           X_{\text{clean}} = \frac{X_c}{\sigma}

    **RMT Consistency**

    All variance estimates use ``ddof=0`` to ensure compatibility with:

    .. math::
        \Sigma = \frac{1}{n} X^\top X

    **PCA Scaling Note**

    The internal implementation of PCA (via ``scikit-learn``) uses
    a normalization proportional to ``1/(n-1)`` when reporting explained
    variance. However, the factor reconstruction:

    .. math::
        \hat{X} = F \Lambda^\top

    is purely geometric (SVD-based) and independent of this scaling.
    Final variance normalization restores full MLE consistency.

    **Failure Modes**

    Statistical:

    - Strong latent factors inflate the spectrum.
      Mitigation: increase ``k``.

    Numerical:

    - Perfect factor explanation leads to zero residual variance.
      Mitigation: affected columns are detected, a warning is raised,
      and scaling defaults to 1.0 to avoid division by zero.

    Data:

    - Non-finite values (NaN/Inf) break PCA.
      Mitigation: input validation raises ``ValueError``.

    **Design Decisions**

    - Input immutability is enforced via defensive copying.
    - Factor selection (``k``) is external to avoid circular dependency
      with spectral estimation routines.

    See Also
    --------
    standardize
    marchenko_pastur.pipelines.defactored_mp

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> X = np.random.randn(100, 20)
    >>> X_clean = defactor_data(X, k=2)
    >>> X_clean.shape
    (100, 20)
    """
    # SOTA FIX: Strict immutability to prevent downstream in-place corruption
    X_arr = np.array(X, dtype=np.float64, copy=True)

    # --------------------------------------------------
    # O(1) Structural Validations
    # --------------------------------------------------
    if X_arr.ndim != 2:
        raise ValueError("X must be a 2D array.")

    n, p = X_arr.shape
    if k < 0 or k > min(n, p):
        raise ValueError(
        f"The number of factors 'k' must be between 0 and min(n_samples, n_features)={min(n, p)}."
        )

    # --------------------------------------------------
    # O(n*p) Content Validations
    # --------------------------------------------------
    if not np.isfinite(X_arr).all():
        raise ValueError("X contains non-finite values (NaN or Inf).")

    # --------------------------------------------------
    # 1. Mean Centering
    # --------------------------------------------------
    mu = np.mean(X_arr, axis=0, keepdims=True)
    X_centered = X_arr - mu

    # --------------------------------------------------
    # Base Case: No defactoring requested (k=0)
    # --------------------------------------------------
    if k == 0:
        if verbose:
            logger.info("[DEFAC] k=0 -> Returning standard Z-score matrix.")

        # SOTA FIX: Strict ddof=0 for RMT consistency
        sigma = np.std(X_centered, axis=0, ddof=0, keepdims=True)
        sigma = np.clip(sigma, 1e-12, None)
        return X_centered / sigma

    if verbose:
        logger.info(f"[DEFAC] Removing {k} factors via PCA...")

    # --------------------------------------------------
    # 2. Factor Extraction (PCA)
    # --------------------------------------------------
    pca = PCA(n_components=k)
    F = pca.fit_transform(X_centered)
    Lambda = pca.components_.T

    # --------------------------------------------------
    # 3. Residual Computation
    # --------------------------------------------------
    X_hat = F @ Lambda.T
    R = X_centered - X_hat

    # --------------------------------------------------
    # 4. Idiosyncratic Variance Scaling
    # --------------------------------------------------
    # SOTA FIX: Strict ddof=0 for RMT consistency
    sigma_idio = np.std(R, axis=0, ddof=0, keepdims=True)

    zero_var = sigma_idio <= np.finfo(sigma_idio.dtype).eps
    if np.any(zero_var):
        warnings.warn(
            "Zero idiosyncratic variance detected after defactoring. "
            "The removed factors perfectly explained one or more features.",
            UserWarning,
            stacklevel=2,
        )
        # SOTA FIX: Fallback to 1.0 to prevent division by zero, just like standardize
        sigma_idio[zero_var] = 1.0

    X_clean = X_centered / sigma_idio

    if verbose:
        logger.info("[DEFAC] Cleaned idiosyncratic matrix successfully generated.")

    return X_clean
