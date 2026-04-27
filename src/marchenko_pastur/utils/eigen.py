r"""
Spectral utilities for covariance and random matrix analysis.

This module provides efficient and numerically robust tools for extracting
eigenvalues from covariance matrices and data matrices under different
dimensional regimes.

The implementation is aligned with Random Matrix Theory (RMT), ensuring
consistency with the Marchenko-Pastur framework by:

- Supporting both Maximum Likelihood Estimator (MLE) normalization (:math:`1/n`)
  for strict RMT compliance, and unbiased Pearson normalization (:math:`1/(n-1)`).
- Preserving exact zero eigenvalues in high-dimensional regimes.
- Avoiding unnecessary data mutation and numerical artifacts.

The module automatically selects the most efficient computational strategy
depending on the dimensionality of the problem:

- Classical regime (:math:`p \le n`): covariance-based eigendecomposition.
- High-dimensional regime (:math:`p > n`): singular value decomposition (SVD).
- Ultra-high-dimensional regime (:math:`p \gg n`): Gram matrix trick.

These utilities act as the **primary engine for spectral extraction**
across the library, ensuring consistency between inference, diagnostics,
and visualization pipelines.

Notes
-----
This module is designed for internal reuse but exposes a stable public API
for advanced users requiring direct spectral access.

See Also
--------
marchenko_pastur.api.run_mp
marchenko_pastur.engine.classical.compute_covariance
"""

# ======================================================================
# STANDARD LIBRARY IMPORTS
# ======================================================================
# (Empty - retained for structural consistency)

# ======================================================================
# THIRD-PARTY IMPORTS
# ======================================================================
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import eigh, svd

# ======================================================================
# LOCAL IMPORTS
# ======================================================================
from marchenko_pastur.enums.enums import CovarianceMethod

# ======================================================================
# PUBLIC API EXPORTS
# ======================================================================
__all__ = [
    "sorted_eigenvalues",
    "largest_eigenvalue",
    "compute_empirical_spectrum",
]

# ======================================================================
# FUNCTIONS
# ======================================================================


def sorted_eigenvalues(Sigma: ArrayLike) -> NDArray[np.float64]:
    r"""
    Return the eigenvalues of a symmetric matrix in ascending order.

    Parameters
    ----------
    Sigma : ArrayLike of shape (p, p)
        Symmetric matrix (typically covariance or scatter matrix).

    Returns
    -------
    eigvals : ndarray of shape (p,)
        Eigenvalues sorted in ascending order.

    Raises
    ------
    ValueError
        If the input is not a 2D square matrix or contains non-finite values.

    Notes
    -----
    This function is a thin wrapper around ``numpy.linalg.eigvalsh``, which
    exploits symmetry for numerical stability and efficiency.

    It performs strict validation to guarantee:

    - Square matrix structure
    - Finite numerical values

    This function is intended for low-level spectral extraction and is used
    internally across the library.

    Examples
    --------
    >>> import numpy as np
    >>> Sigma = np.array([[2.0, 0.5], [0.5, 1.0]])
    >>> sorted_eigenvalues(Sigma)
    array([0.79289322, 2.20710678])
    """
    Sigma_arr = np.asarray(Sigma, dtype=np.float64)

    # --------------------------------------------------
    # O(1) Structural Validations
    # --------------------------------------------------
    if Sigma_arr.ndim != 2:
        raise ValueError("Sigma must be a 2D array.")
    if Sigma_arr.shape[0] != Sigma_arr.shape[1]:
        raise ValueError("Sigma must be a square matrix.")

    # --------------------------------------------------
    # O(p^2) Content Validations
    # --------------------------------------------------
    if not np.isfinite(Sigma_arr).all():
        raise ValueError("Sigma contains non-finite values (NaN or Inf).")

    return np.linalg.eigvalsh(Sigma_arr)


def largest_eigenvalue(Sigma: ArrayLike) -> float:
    r"""
    Compute the largest eigenvalue of a symmetric matrix efficiently.

    Parameters
    ----------
    Sigma : ArrayLike of shape (p, p)
        Symmetric matrix (typically covariance matrix).

    Returns
    -------
    lambda_max : float
        Largest eigenvalue of the matrix.

    Raises
    ------
    ValueError
        If the input is not a 2D square matrix or contains non-finite values.

    Notes
    -----
    This function uses a partial eigendecomposition via LAPACK (``driver='evr'``),
    which computes only the largest eigenvalue instead of the full spectrum.

    This reduces computational complexity compared to full eigendecomposition:

    - Full spectrum: :math:`O(p^3)`
    - Largest eigenvalue only: approximately :math:`O(p^2)`

    This function is particularly useful in iterative procedures such as:

    - Bootstrap methods
    - Threshold estimation
    - Spectral edge detection

    Examples
    --------
    >>> import numpy as np
    >>> Sigma = np.array([[2.0, 0.5], [0.5, 1.0]])
    >>> largest_eigenvalue(Sigma)
    2.2071067811865475
    """
    Sigma_arr = np.asarray(Sigma, dtype=np.float64)

    # --------------------------------------------------
    # O(1) Structural Validations
    # --------------------------------------------------
    if Sigma_arr.ndim != 2:
        raise ValueError("Sigma must be a 2D array.")
    if Sigma_arr.shape[0] != Sigma_arr.shape[1]:
        raise ValueError("Sigma must be a square matrix.")

    # --------------------------------------------------
    # O(p^2) Content Validations
    # --------------------------------------------------
    if not np.isfinite(Sigma_arr).all():
        raise ValueError("Sigma contains non-finite values (NaN or Inf).")

    p = Sigma_arr.shape[0]

    val = eigh(
        Sigma_arr,
        subset_by_index=[p - 1, p - 1],
        eigvals_only=True,
        check_finite=False,
        driver="evr",
    )

    return float(val[0])


def compute_empirical_spectrum(
    X: ArrayLike,
    method: CovarianceMethod = CovarianceMethod.CLASSICAL,
    assume_centered: bool = False,
) -> NDArray[np.float64]:
    r"""
    Compute the empirical eigenvalue spectrum of a data matrix.

    This function extracts the eigenvalues of the covariance matrix:

    .. math::
        \Sigma = \frac{1}{d} X^\top X

    using an adaptive algorithm that selects the most efficient method
    depending on the dimensional regime.

    Parameters
    ----------
    X : ArrayLike of shape (n_samples, n_features)
        Data matrix where rows correspond to observations and columns
        to variables.

    method : CovarianceMethod, default=CovarianceMethod.CLASSICAL
        The estimator normalization method:
        - "classical": Maximum Likelihood Estimator (:math:`1/n`).
          Required for strict consistency with the Marchenko-Pastur law.
        - "pearson": Unbiased sample covariance (:math:`1/(n-1)`).
          Matches standard statistical practice but introduces a scaling mismatch.

    assume_centered : bool, default=False
        If True, assumes that the data is already column-centered, skipping
        the mean subtraction step.

    Returns
    -------
    spectrum : ndarray of shape (p,)
        Sorted eigenvalues of the covariance matrix.

    Raises
    ------
    ValueError
        If the input is not a 2D array, contains non-finite values, or if
        :math:`n \le 1` under Pearson normalization.

    Notes
    -----
    **Theory**

    The covariance matrix is defined as:

    .. math::
        \Sigma = \frac{1}{d} X_c^\top X_c

    where :math:`X_c` is the column-centered data matrix and :math:`d` is the
    degrees of freedom:

    - **Classical (Wishart)**: :math:`d = n`.
    - **Pearson (Sample)**: :math:`d = n - 1`.

    **Algorithm**

    All computational paths (covariance, SVD, Gram) are mathematically
    equivalent and yield identical non-zero eigenvalues.

    The function dynamically selects one of three equivalent formulations:

    1. **Classical regime (:math:`p \le n`)**

       - Compute covariance explicitly:

         .. math::
             \Sigma = \frac{1}{d} X^\top X

       - Extract eigenvalues via eigendecomposition.

    2. **High-dimensional regime (:math:`p > n`)**

       - Use singular values:

         .. math::
             \lambda_i = \frac{s_i^2}{d}

    3. **Ultra-high-dimensional regime (:math:`p \gg n`)**

       - Use Gram matrix:

         .. math::
             G = \frac{1}{d} X X^\top

       - Non-zero eigenvalues of :math:`G` match those of :math:`\Sigma`.
       - Remaining eigenvalues are exactly zero.

    **Interpretation**

    - Large eigenvalues indicate signal (factors).
    - Small eigenvalues indicate noise.
    - Exact zeros (when :math:`p > n`) indicate rank deficiency.

    **Failure Modes**

    Statistical:

    - Spectral edge shift due to degrees of freedom:
      Using ``method='pearson'`` rescales the spectrum by :math:`n/(n-1)`.
      Mitigation: use ``method='classical'`` for strict theoretical MP bounds.

    - Heavy-tailed data violates MP assumptions.
      Mitigation: apply robust covariance preprocessing (e.g., Tyler estimator)
      before spectral analysis.

    Numerical:

    - Degenerate dimensions (:math:`n \le 1`) when calculating Pearson covariance.
      Mitigation: ensure sufficient sample size.

    - Floating-point errors may produce small negative eigenvalues.
      Mitigation: values are clipped to zero.

    - Extremely large matrices may cause memory pressure.
      Mitigation: dimensionality-aware algorithm selection handles up to
      moderate dimensions natively.

    **Complexity**

    - Classical: :math:`O(p^2 n + p^3)`
    - SVD: :math:`O(n p^2)`
    - Gram: :math:`O(n^2 p + n^3)`

    See Also
    --------
    marchenko_pastur.engine.classical.compute_covariance : Direct covariance computation.

    Examples
    --------
    >>> import numpy as np
    >>> from marchenko_pastur.enums.enums import CovarianceMethod
    >>> np.random.seed(42)

    >>> # High-dimensional case (p > n)
    >>> X = np.random.randn(10, 100)

    >>> spectrum = compute_empirical_spectrum(
    ...     X, method=CovarianceMethod.CLASSICAL
    ... )

    >>> spectrum.shape
    (100,)

    >>> # Rank deficiency: at most n non-zero eigenvalues
    >>> (spectrum > 0).sum() <= 10
    True
    """
    X_arr = np.array(X, dtype=np.float64, copy=True)

    if X_arr.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if not np.isfinite(X_arr).all():
        raise ValueError("X contains non-finite values (NaN or Inf).")

    n, p = X_arr.shape

    is_pearson = (method == CovarianceMethod.PEARSON)
    if is_pearson and n <= 1:
        raise ValueError("Pearson covariance requires n > 1.")

    if not assume_centered:
        X_arr = X_arr - X_arr.mean(axis=0, keepdims=True)

    # ---------------------------------------------------------
    # Dinamyc Divisor
    # ---------------------------------------------------------
    divisor = float(n - 1) if is_pearson else float(n)

    # ---------------------------------------------------------
    # Ultra-high dimensional regime (Gram Matrix)
    # ---------------------------------------------------------
    if p > 5 * n:
        G = (X_arr @ X_arr.T) / divisor
        eigvals = np.linalg.eigvalsh(G)

    # ---------------------------------------------------------
    # Moderate high-dimensional regime (SVD)
    # ---------------------------------------------------------
    elif p > n:
        # SVD of X gives singular values s. Eigenvalues of cov are s^2 / divisor
        # SOTA FIX: Coerción explícita a NDArray para resolver la ambigüedad
        # del stub de 'svd' (Union[tuple, NDArray]) ante Pylance.
        raw_s = svd(X_arr, compute_uv=False, overwrite_a=False, check_finite=False)
        s = np.asarray(raw_s, dtype=np.float64)
        eigvals = (s**2) / divisor

    # ---------------------------------------------------------
    # Classical regime (Standard Covariance)
    # ---------------------------------------------------------
    else:
        Sigma = (X_arr.T @ X_arr) / divisor
        eigvals = np.linalg.eigvalsh(Sigma)

    # ---------------------------------------------------------
    # Post-processing and padding
    # ---------------------------------------------------------
    eigvals = np.maximum(eigvals, 0.0)
    eigvals.sort()

    if p > len(eigvals):
        spectrum = np.zeros(p, dtype=np.float64)
        spectrum[p - len(eigvals) :] = eigvals
        return spectrum

    return eigvals
