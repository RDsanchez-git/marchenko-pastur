r"""
Tracy-Widom thresholds for spike detection.

This module implements statistical thresholds based on the Tracy-Widom
distribution to detect significant eigenvalues (spikes) beyond the
Marchenko-Pastur bulk.

In high-dimensional regimes:

.. math::
    n, p \to \infty \quad \text{with} \quad \frac{p}{n} \to q

the largest eigenvalue of a Wishart covariance matrix fluctuates
around the upper edge :math:`\lambda_+` with scale :math:`n^{-2/3}`,
following the Tracy-Widom distribution (β = 1).

This enables formal hypothesis testing for the presence of latent
factors in empirical covariance matrices.

Notes
-----
This module provides:

- Asymptotic scaling of edge fluctuations (`tw_scale`)
- Critical values for spike detection (`tracy_widom_threshold`)

The implementation assumes Gaussian or light-tailed data. Deviations
from these assumptions may invalidate the theoretical thresholds.

See Also
--------
mp_bounds : Deterministic spectral support
fit_mp : Estimation of noise variance
run_mp : Full spectral analysis pipeline

References
----------
Johnstone, I. M. (2001).

Tracy, C. A., & Widom, H. (1994).
"""

# ======================================================================
# STANDARD LIBRARY IMPORTS
# ======================================================================

from functools import lru_cache
from typing import Dict

# ======================================================================
# THIRD-PARTY IMPORTS
# ======================================================================
import numpy as np

# ======================================================================
# LOCAL IMPORTS
# ======================================================================
from .mp_theory import mp_bounds

# ======================================================================
# PUBLIC API
# ======================================================================

__all__ = [
    "tracy_widom_threshold",
    "tw_scale",
]

# ======================================================================
# CONSTANTES
# ======================================================================

TW_QUANTILES: Dict[float, float] = {
    0.10: 0.450,
    0.05: 0.979,
    0.01: 2.023,
}

# ======================================================================
# ESCALA TRACY–WIDOM
# ======================================================================


def tw_scale(n: int, p: int, sigma2: float) -> float:
    r"""
    Compute the Tracy–Widom scaling factor for the largest eigenvalue.

    Parameters
    ----------
    n : int
        Number of observations.

    p : int
        Number of variables.

    sigma2 : float
        Noise variance.

    Returns
    -------
    scale : float
        Asymptotic fluctuation scale of the largest eigenvalue.

    Raises
    ------
    ValueError
        If inputs are invalid.

    Notes
    -----
    **Theory**

    Under the null hypothesis of pure noise:

    .. math::
        \lambda_{\max} \approx \lambda_+ + s \cdot TW_1

    where:

    .. math::
        \lambda_+ = \sigma^2 (1 + \sqrt{q})^2

    and the scaling factor is:

    .. math::
        s = \sigma^2 n^{-2/3} q^{-1/6} (1 + \sqrt{q})^{4/3}

    derived in Johnstone (2001).

    **Interpretation**

    The scale determines the magnitude of stochastic fluctuations
    around the deterministic edge :math:`\lambda_+`.

    **Failure Modes**

    - Small sample size (n too small) → asymptotic approximation breaks

      **Mitigation:** prefer bootstrap-based thresholds

    - Heavy-tailed data → Tracy–Widom no longer valid

      **Mitigation:** use robust covariance estimators (e.g. Tyler)

    **Complexity**

    - Time: :math:`O(1)`
    - Memory: :math:`O(1)`

    See Also
    --------
    tracy_widom_threshold : Full detection threshold
    mp_bounds : Deterministic spectral edge

    References
    ----------
    Johnstone, I. M. (2001).

    Examples
    --------
    >>> tw_scale(n=1000, p=500, sigma2=1.0)
    >>> round(scale, 4)
    0.0435
    """

    # ---------------------------------------------------------------
    # Strict type validation (safeguard against bools, allow np.integer)
    # ---------------------------------------------------------------

    if type(n) is not int or type(p) is not int:
        raise ValueError("n and p must be integers.")

    if n <= 0 or p <= 0:
        raise ValueError("n and p must be strictly positive.")

    # ---------------------------------------------------------------
    # Numerical validation
    # ---------------------------------------------------------------

    if not np.isfinite(sigma2):
        raise ValueError("sigma2 must be finite.")

    if sigma2 <= 0:
        raise ValueError("sigma2 must be strictly positive..")

    # ---------------------------------------------------------------
    # Scale factor computation (Johnstone, 2001)
    # ---------------------------------------------------------------

    q = p / n
    sqrt_q = np.sqrt(q)

    scale = sigma2 * n ** (-2 / 3) * q ** (-1 / 6) * (1 + sqrt_q) ** (4 / 3)

    return float(scale)


# ======================================================================
# TRACY-WIDOM THRESHOLD
# ======================================================================


@lru_cache(maxsize=128)
def tracy_widom_threshold(
    n: int,
    p: int,
    sigma2: float,
    alpha: float = 0.05,
) -> float:
    r"""
    Compute the Tracy–Widom critical threshold for spike detection.

    Parameters
    ----------
    n : int
        Number of observations.

    p : int
        Number of variables.

    sigma2 : float
        Estimated noise variance.

    alpha : float, default=0.05
        Significance level.

    Returns
    -------
    lambda_crit : float
        Critical threshold for the largest eigenvalue.

    Raises
    ------
    ValueError
        If `alpha` is not supported or parameters are invalid.

    Notes
    -----
    **Theory**

    Under the null hypothesis:

    .. math::
        H_0: \Sigma = \sigma^2 I

    the largest eigenvalue fluctuates as:

    .. math::
        \lambda_{\max} \approx \lambda_+ + s \cdot TW_\alpha

    where:

    .. math::
        \lambda_+ = \sigma^2 (1 + \sqrt{q})^2

    .. math::
        s = \text{tw_scale}(n, p, \sigma^2)

    and :math:`TW_\alpha` is the Tracy–Widom quantile.

    The detection threshold is:

    .. math::
        \lambda_{\text{crit}} = \lambda_+ + s \cdot TW_\alpha

    **Interpretation**

    - :math:`\lambda_{\max} > \lambda_{\text{crit}}` → significant spike
    - :math:`\lambda_{\max} \leq \lambda_{\text{crit}}` → consistent with noise

    **Failure Modes**

    Statistical:

    - Heavy-tailed distributions violate Tracy–Widom assumptions

      **Mitigation:** use ``covariance='tyler'`` or bootstrap thresholds

    - Small sample size invalidates asymptotic approximation

      **Mitigation:** increase n or use empirical resampling

    Numerical:

    - Incorrect σ² estimation shifts the threshold

      **Mitigation:** use ``fit_mp(method='auto')`` for robust estimation

    **Complexity**

    - Time: :math:`O(1)`
    - Memory: :math:`O(1)`

    See Also
    --------
    tw_scale : Scaling factor for fluctuations
    mp_bounds : Deterministic spectral edge
    run_mp : Full spike detection pipeline

    References
    ----------
    Johnstone, I. M. (2001).

    Tracy, C. A., & Widom, H. (1994).

    Examples
    --------
    >>> thresh = tracy_widom_threshold(n=1000, p=500, sigma2=1.0, alpha=0.05)
    >>> round(thresh, 4)
    2.957
    """

    # ---------------------------------------------------------------
    # Type normalization (assists with NumPy floats)
    # ---------------------------------------------------------------

    alpha = float(alpha)
    sigma2 = float(sigma2)

    # ---------------------------------------------------------------
    # Validations
    # ---------------------------------------------------------------

    if alpha not in TW_QUANTILES:
        raise ValueError(
            f"alpha={alpha}is not supported. Available values: {list(TW_QUANTILES.keys())}"
        )

    if not np.isfinite(sigma2):
        raise ValueError("sigma2 must be finite.")

    if sigma2 <= 0:
        raise ValueError("sigma2 must be strictly positive.")

    # ---------------------------------------------------------------
    # Threshold computation
    # ---------------------------------------------------------------

    q = p / n

    _, lambda_plus = mp_bounds(q, sigma2)

    scale = tw_scale(n, p, sigma2)

    tw_q = TW_QUANTILES[alpha]

    lambda_crit = lambda_plus + scale * tw_q

    return float(lambda_crit)
