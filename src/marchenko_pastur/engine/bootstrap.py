r"""
Permutation bootstrap engine for empirical spectral thresholds.

This module implements a non-parametric bootstrap procedure based on
column-wise permutation to estimate the distribution of the largest
eigenvalue under the null hypothesis of independence.

The method generates synthetic datasets by independently permuting
each column of the data matrix, thereby destroying cross-sectional
dependence while preserving marginal distributions.

This provides a robust alternative to Tracy-Widom-based thresholds
when theoretical assumptions are violated.

Notes
-----
This module is intended as a fallback inference engine when:

- Tracy-Widom approximation is unreliable
- Sample size is small
- Data exhibits heavy tails or non-Gaussian behavior

See Also
--------
tracy_widom_threshold : Asymptotic parametric threshold
fit_mp : Noise variance estimation
run_mp : Full spectral analysis pipeline
"""

# ======================================================================
# STANDARD LIBRARY IMPORTS
# ======================================================================

from typing import Any, Callable, Mapping, Optional, Tuple

# ======================================================================
# THIRD-PARTY IMPORTS
# ======================================================================
import numpy as np
from joblib import Parallel, delayed
from numpy.typing import NDArray

# ======================================================================
# LOCAL IMPORTS
# ======================================================================
from marchenko_pastur.utils.eigen import largest_eigenvalue

# ======================================================================
# PUBLIC API EXPORTS
# ======================================================================
__all__ = [
    "bootstrap_lambda_crit",
]

# ======================================================================
# INTERNAL UTILITIES
# ======================================================================


def _permute_columns(
    X: NDArray[np.float64],
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """
    Independently permute each column of the input data matrix.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input data matrix.

    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    X_perm : ndarray of shape (n_samples, n_features)
        Matrix with independently permuted columns.

    Notes
    -----
    Each column is permuted independently, which:

    - destroys cross-sectional dependence between variables
    - preserves marginal distributions

    This operation generates synthetic data consistent with the
    null hypothesis of independence used in the bootstrap procedure.
    """

    n, p = X.shape

    X_perm = np.empty_like(X)

    for j in range(p):
        idx = rng.permutation(n)
        X_perm[:, j] = X[idx, j]

    return X_perm


def _bootstrap_single(
    X: NDArray[np.float64],
    covariance_fn: Callable,
    seed: np.random.SeedSequence,
    cov_kwargs: Mapping[str, Any],
) -> float:
    """
    Perform a single bootstrap replication.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input data matrix.

    covariance_fn : callable
        Function used to estimate the covariance matrix.

    seed : numpy.random.SeedSequence
        Seed for reproducible random number generation.

    cov_kwargs : dict
        Additional keyword arguments passed to `covariance_fn`.

    Returns
    -------
    lambda_max : float
        Largest eigenvalue of the bootstrap covariance matrix.

    Notes
    -----
    This function performs the following steps:

    1. Generate a permuted dataset via `_permute_columns`
    2. Estimate covariance using `covariance_fn`
    3. Compute the largest eigenvalue

    It is designed to be executed in parallel across bootstrap samples.
    """

    rng = np.random.default_rng(seed)

    X_boot = _permute_columns(X, rng)

    Sigma_boot = covariance_fn(X_boot, **cov_kwargs)

    return float(largest_eigenvalue(Sigma_boot))


# ======================================================================
# BOOSTRAP
# ======================================================================


def bootstrap_lambda_crit(
    X: NDArray[np.float64],
    covariance_fn: Callable,
    *,
    alpha: float = 0.05,
    B: int = 250,
    random_state: Optional[int] = None,
    n_jobs: int = -1,
    **cov_kwargs: Any,
) -> Tuple[float, NDArray[np.float64]]:
    r"""
    Estimate spike detection threshold via permutation bootstrap.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input data matrix.

    covariance_fn : callable
        Function used to estimate the covariance matrix.

    alpha : float, default=0.05
        Significance level.

    B : int, default=250
        Number of bootstrap replications.

    random_state : int, optional
        Random seed for reproducibility.

    n_jobs : int, default=-1
        Number of parallel workers.

    **cov_kwargs
        Additional keyword arguments passed to `covariance_fn`.

    Returns
    -------
    lambda_crit : float
        Empirical critical threshold for the largest eigenvalue.

    lambda_max_dist : ndarray of shape (B,)
        Bootstrap distribution of the largest eigenvalue.

    Raises
    ------
    ValueError
        If inputs are invalid.

    Notes
    -----
    **Theory**

    Under the null hypothesis of independence:

    .. math::
        H_0: \text{variables are independent}

    the largest eigenvalue reflects only sampling variability.

    By permuting each column independently, we generate synthetic
    datasets consistent with this null hypothesis.

    The empirical threshold is defined as:

    .. math::
        \lambda_{\text{crit}} = \text{Quantile}_{1-\alpha}(\lambda_{\max})

    **Interpretation**

    - :math:`\lambda_{\max} > \lambda_{\text{crit}}` → evidence of structure
    - :math:`\lambda_{\max} \leq \lambda_{\text{crit}}` → consistent with noise

    **Failure Modes**

    Statistical:

    - Bootstrap assumes exchangeability within columns

      **Mitigation:** avoid time series or structured dependencies

    - Small B leads to unstable quantile estimates

      **Mitigation:** increase ``B`` (≥ 500 recommended)

    Numerical:

    - Expensive for large p (covariance + eigenvalue per iteration)

      **Mitigation:** reduce B or use ``tracy_widom_threshold``

    - Covariance estimator instability propagates to threshold

      **Mitigation:** use robust estimators (e.g. ``covariance_fn=tyler``)

    **Complexity**

    - Time: :math:`O(B \cdot p^3)` (dominated by eigenvalue computation)
    - Memory: :math:`O(B)`

    See Also
    --------
    tracy_widom_threshold : Parametric alternative
    tw_scale : Asymptotic scaling
    fit_mp : Noise estimation

    References
    ----------
    Efron, B. (1979). Bootstrap methods.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> X = rng.standard_normal((100, 50))

    >>> lambda_crit, dist = bootstrap_lambda_crit(
    ...     X,
    ...     covariance_fn=np.cov,
    ...     B=50,
    ...     rowvar=False
    ... )

    >>> isinstance(lambda_crit, float)
    True
    >>> dist.shape
    (50,)
    """

    # ---------------------------------------------------------------
    # Parameter validation
    # ---------------------------------------------------------------

    if not isinstance(B, int) or B <= 0:
        raise ValueError("B must be a strictly positive integer.")

    if not (0 < alpha < 1):
        raise ValueError("alpha must be in the interval (0, 1).")

    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")

    # ---------------------------------------------------------------
    # RNG Configuration (NumPy parallel streams)
    # ---------------------------------------------------------------

    ss = np.random.SeedSequence(random_state)

    child_seeds = ss.spawn(B)

    # ---------------------------------------------------------------
    # Parallel bootstrap execution
    # ---------------------------------------------------------------

    lambda_max_dist = Parallel(
        n_jobs=n_jobs,
        prefer="processes",
    )(delayed(_bootstrap_single)(X, covariance_fn, child_seeds[b], cov_kwargs) for b in range(B))

    lambda_max_dist = np.asarray(lambda_max_dist, dtype=float)

    # ---------------------------------------------------------------
    # Empirical quantile
    # ---------------------------------------------------------------

    lambda_crit = float(np.percentile(lambda_max_dist, 100 * (1 - alpha)))

    return lambda_crit, lambda_max_dist
