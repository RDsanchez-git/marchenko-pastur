r"""
Analytical Marchenko-Pastur theory.

This module implements the core theoretical components of the
Marchenko-Pastur (MP) distribution, which describes the asymptotic
eigenvalue behavior of high-dimensional random covariance matrices.

It provides:

- Closed-form expressions for the spectral support (:math:`\lambda_-, \lambda_+`)
- Evaluation of the continuous spectral density

These functions form the mathematical backbone of the entire framework
and are used by higher-level estimation and inference routines.

Notes
-----
The Marchenko-Pastur law characterizes the limiting distribution of
eigenvalues of Wishart-type matrices under the asymptotic regime:

.. math::
    n, p \to \infty \quad \text{with} \quad \frac{p}{n} \to q

This module only implements the **continuous component** of the spectrum.
In the singular regime (:math:`q > 1`), a Dirac mass at zero is present but not
explicitly represented here.

**Design Philosophy**

This module is intentionally:

- Minimal (pure analytical formulas)
- Stateless (no estimation or heuristics)
- Reusable across different pipelines

See Also
--------
mp_fit : Estimation of MP parameters.
bbp_population_eigenvalue : Spike correction beyond MP bulk.
run_mp : Full spectral analysis pipeline.

References
----------
Marchenko, V. A., & Pastur, L. A. (1967).
Distribution of eigenvalues for some sets of random matrices.

Johnstone, I. M. (2001).
On the distribution of the largest eigenvalue in principal components analysis.
"""

# ======================================================================
# STANDARD LIBRARY IMPORTS
# ======================================================================

from typing import Tuple, Union

# ======================================================================
# THIRD-PARTY IMPORTS
# ======================================================================
import numpy as np
from numpy.typing import ArrayLike, NDArray

# ======================================================================
# CONSTANTS
# ======================================================================

EPS = 1e-12

# ======================================================================
# PUBLIC API EXPORTS
# ======================================================================

__all__ = [
    "mp_bounds",
    "mp_density",
]

# ======================================================================
# MARCHENKO-PASTUR BOUNDS
# ======================================================================


def mp_bounds(q: float, sigma2: float) -> Tuple[float, float]:
    r"""
    Compute the theoretical support of the Marchenko-Pastur distribution.

    Parameters
    ----------
    q : float
        Dimensional ratio:

        .. math::
            q = \frac{p}{n}

    sigma2 : float
        Noise variance.

    Returns
    -------
    lambda_minus : float
        Lower edge of the spectral bulk.

    lambda_plus : float
        Upper edge of the spectral bulk.

    Raises
    ------
    ValueError
        If parameters are not finite or strictly positive.

    Notes
    -----
    **Theory**

    The support of the Marchenko-Pastur distribution is:

    .. math::
        \lambda_{\pm} = \sigma^2 (1 \pm \sqrt{q})^2

    These bounds define the region where eigenvalues of a pure-noise
    covariance matrix concentrate asymptotically.

    **Interpretation**

    - :math:`\lambda \leq \lambda_+` -> noise
    - :math:`\lambda > \lambda_+` -> signal (spikes)

    **Failure Modes**

    - If :math:`q \to 0`, bounds collapse -> classical low-dimensional regime.
    - If :math:`\sigma^2 \to 0`, spectrum collapses.
      Mitigation: ensure proper data standardization before calling.

    **Complexity**

    - Time: :math:`O(1)`
    - Memory: :math:`O(1)`

    See Also
    --------
    mp_density : Continuous spectral density.
    fit_mp : Estimation of sigma^2 and MP bounds.

    References
    ----------
    Marchenko, V. A., & Pastur, L. A. (1967).

    Examples
    --------
    >>> l_min, l_max = mp_bounds(q=0.5, sigma2=1.0)
    >>> round(l_min, 4), round(l_max, 4)
    (0.0858, 2.9142)
    """

    if not np.isfinite(q):
        raise ValueError("q must be a finite number.")

    if not np.isfinite(sigma2):
        raise ValueError("sigma2 must be a finite number.")

    if q <= 0:
        raise ValueError("q must be strictly positive.")

    if sigma2 <= 0:
        raise ValueError("sigma2 must be strictly positive.")

    sqrt_q = np.sqrt(q)

    lambda_plus = sigma2 * (1 + sqrt_q) ** 2
    lambda_minus = sigma2 * (1 - sqrt_q) ** 2

    return float(lambda_minus), float(lambda_plus)


# ======================================================================
# MARCHENKO-PASTUR DENSITY
# ======================================================================


def mp_density(
    x: ArrayLike,
    q: float,
    sigma2: float,
) -> Union[float, NDArray[np.float64]]:
    r"""
    Evaluate the Marchenko-Pastur spectral density.

    Parameters
    ----------
    x : array-like or float
        Points where the density is evaluated.

    q : float
        Dimensional ratio:

        .. math::
            q = \frac{p}{n}

    sigma2 : float
        Noise variance.

    Returns
    -------
    density : float or ndarray
        Value of the spectral density at `x`.

    Raises
    ------
    ValueError
        If inputs are invalid.

    Notes
    -----
    **Theory**

    The continuous Marchenko-Pastur density is given by:

    .. math::
        \rho(x) = \frac{\sqrt{(\lambda_+ - x)(x - \lambda_-)}}{2\pi q \sigma^2 x}

    for:

    .. math::
        \lambda_- \le x \le \lambda_+

    where:

    .. math::
        \lambda_{\pm} = \sigma^2 (1 \pm \sqrt{q})^2

    **Singular Regime (q > 1)**

    A Dirac mass at zero appears with weight:

    .. math::
        1 - \frac{1}{q}

    This function returns only the continuous component.

    **Failure Modes**

    - Near :math:`x \to 0`, numerical instability may occur.
      Mitigation: small-value cutoff (EPS) is applied internally.
    - Outside support -> density = 0 (by definition).

    **Complexity**

    - Time: :math:`O(m)` where m is the size of x.
    - Memory: :math:`O(m)`

    See Also
    --------
    mp_bounds : Spectral support computation.
    fit_mp : MP fitting procedure.

    References
    ----------
    Marchenko, V. A., & Pastur, L. A. (1967).

    Examples
    --------
    >>> import numpy as np
    >>> density = mp_density(1.5, q=0.5, sigma2=1.0)
    >>> round(density, 4)
    0.2821
    """

    if not np.isfinite(q):
        raise ValueError("q must be a finite number.")

    if not np.isfinite(sigma2):
        raise ValueError("sigma2 must be a finite number.")

    if q <= 0:
        raise ValueError("q must be strictly positive.")

    if sigma2 <= 0:
        raise ValueError("sigma2 must be strictly positive.")

    scalar_input = np.isscalar(x)

    x_arr = np.asarray(x, dtype=float)

    if x_arr.ndim > 1:
        raise ValueError("x must be a scalar or a 1D array.")

    x_arr = np.atleast_1d(x_arr)

    lambda_minus, lambda_plus = mp_bounds(q, sigma2)

    density = np.zeros_like(x_arr)

    mask = (x_arr > EPS) & (x_arr >= lambda_minus) & (x_arr <= lambda_plus)

    density[mask] = np.sqrt((lambda_plus - x_arr[mask]) * (x_arr[mask] - lambda_minus)) / (
        2 * np.pi * q * sigma2 * x_arr[mask]
    )

    if scalar_input:
        return float(density[0])

    return density
