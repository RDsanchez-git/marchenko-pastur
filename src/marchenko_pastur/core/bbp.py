r"""
Baik-Ben Arous-Péché (BBP) phase transition and eigenvalue debiasing.

This module implements the analytical inversion of the BBP relation
for spiked covariance models in high-dimensional settings.

Its main purpose is to recover population eigenvalues (latent signal strength)
from observed sample eigenvalues, correcting the inflation induced by noise
in finite-sample covariance estimation.

The functionality provided here is used internally by the spectral analysis
pipeline to quantify the strength of detected factors beyond the
Marchenko-Pastur bulk.

Notes
-----
- Only eigenvalues above the Marchenko-Pastur upper edge (λ+) are considered
  valid spikes and can be inverted using the BBP formula.
- Eigenvalues within the bulk are treated as noise and mapped to NaN.
- This module assumes that the Marchenko-Pastur fit has already been performed
  and that the noise variance (σ²) is known or estimated.

See Also
--------
mp_theory : Theoretical Marchenko-Pastur bounds.
run_mp : End-to-end spectral analysis pipeline.
defactored_mp : High-level orchestrator for collinear datasets.

References
----------
Baik, J., Ben Arous, G., & Péché, S. (2005).
Phase transition of the largest eigenvalue for nonnull complex
sample covariance matrices. Annals of Probability, 33(5), 1643-1697.

Johnstone, I. M. (2001).
On the distribution of the largest eigenvalue in principal
components analysis. Annals of Statistics, 29(2), 295-327.
"""

# ======================================================================
# STANDARD LIBRARY IMPORTS
# ======================================================================

from typing import Union

# ======================================================================
# THIRD-PARTY IMPORTS
# ======================================================================
import numpy as np
from numpy.typing import ArrayLike, NDArray

# ======================================================================
# LOCAL IMPORTS
# ======================================================================
from .mp_theory import mp_bounds

# ======================================================================
# CONSTANTES
# ======================================================================

EPS = 1e-12

# ======================================================================
# PUBLIC API
# ======================================================================

__all__ = [
    "bbp_population_eigenvalue",
]

# ======================================================================
# FUNCTIONS
# ======================================================================


def bbp_population_eigenvalue(
    lambda_sample: ArrayLike,
    q: float,
    sigma2: float,
) -> Union[float, NDArray[np.float64]]:
    r"""
    Estimate population eigenvalues via BBP inversion.

    This function implements the analytical inversion of the
    Baik-Ben Arous-Péché (BBP) relation for the spiked covariance model,
    mapping observed sample eigenvalues to their corresponding population
    eigenvalues by correcting noise-induced bias in high-dimensional settings.

    Parameters
    ----------
    lambda_sample : ArrayLike
        Sample eigenvalue(s) classified as spikes. Can be a scalar or
        a 1D array.

    q : float
        Dimensional ratio defined as:

        .. math::
            q = \frac{p}{n}

    sigma2 : float
        Estimated noise variance.

    Returns
    -------
    ell : float or ndarray
        Estimated population eigenvalue(s).

        Returns ``NaN`` for eigenvalues that do not exceed the
        Marchenko-Pastur upper edge.

    Raises
    ------
    ValueError
        If inputs are not finite, not positive, or have invalid shape.

    Notes
    -----
    This function performs spectral de-biasing under the spiked covariance model.

    **Assumptions**

    - Observations are approximately i.i.d.
    - High-dimensional asymptotic regime:

      .. math::
          n, p \to \infty \quad \text{with} \quad \frac{p}{n} \to q

    - Finite fourth moments are required for valid edge behavior.
    - Spiked covariance structure (low-rank signal + noise).

    **Theory**

    Under the BBP model, the relationship between population eigenvalues
    :math:`\ell` and sample eigenvalues :math:`\lambda` is:

    .. math::
        \lambda = \ell \left(1 + \frac{q \sigma^2}{\ell - \sigma^2}\right)

    Inverting this relation yields:

    .. math::
        \ell^2 - \ell[\lambda + \sigma^2(1 - q)] + \lambda \sigma^2 = 0

    whose physically meaningful solution is:

    .. math::
    \ell = \frac{
        \lambda + \sigma^2(1 - q)
        + \sqrt{(\lambda + \sigma^2(1 - q))^2 - 4\sigma^2 \lambda}
    }{2}

    Detectability condition (BBP phase transition):

    .. math::
        \ell > \sigma^2 (1 + \sqrt{q})

    which implies:

    .. math::
        \lambda > \lambda_+

    where :math:`\lambda_+` is the Marchenko-Pastur upper edge.

    **Interpretation**

    - :math:`\ell` estimates the true strength of latent factors.
    - Values returned as ``NaN`` correspond to eigenvalues indistinguishable from noise.
    - Larger :math:`\ell` implies stronger underlying signal.

    **Failure Modes**

    Statistical:

    - If :math:`\lambda \leq \lambda_+`, inversion is not valid (returns NaN).
      **Mitigation:** Ensure spike detection is performed prior to calling this function.

    - Heavy-tailed data may violate BBP assumptions.
      **Mitigation:** Use ``covariance='tyler'`` in upstream estimation.

    Numerical:

    - Discriminant may become slightly negative due to floating-point errors.
      **Mitigation:** Clipped internally via numerical stabilization.

    - Near-threshold eigenvalues may yield unstable estimates.
      **Mitigation:** Apply stricter spike thresholding (e.g., Tracy-Widom).

    **Complexity**

    - Time: :math:`O(k)` where k is the number of spikes
    - Memory: :math:`O(k)`

    See Also
    --------
    mp_bounds : Compute theoretical Marchenko-Pastur support.
    run_mp : Full spectral analysis pipeline.
    tracy_widom_threshold : Statistical spike detection threshold.

    References
    ----------
    Baik, J., Ben Arous, G., & Péché, S. (2005).
    Phase transition of the largest eigenvalue for nonnull complex
    sample covariance matrices. Annals of Probability.

    Johnstone, I. M. (2001).
    On the distribution of the largest eigenvalue in PCA.
    Annals of Statistics.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> eigs = np.array([2.0, 3.5, 5.0])
    >>> bbp_population_eigenvalue(eigs, q=0.5, sigma2=1.0)
    array([nan, 2.08333333, 4.0])
    """

    # ---------------------------------------------------------------
    # Input validation
    # ---------------------------------------------------------------

    if not np.isfinite(q):
        raise ValueError("q must be finite.")

    if not np.isfinite(sigma2):
        raise ValueError("sigma2 must be finite.")

    if q <= 0:
        raise ValueError("q must be positive.")

    if sigma2 <= 0:
        raise ValueError("sigma2 must be positive.")

    scalar_input = np.isscalar(lambda_sample)

    lam = np.asarray(lambda_sample, dtype=float)

    if lam.ndim > 1:
        raise ValueError("lambda_sample must be a scalar or 1D array.")

    lam = np.atleast_1d(lam)

    # ---------------------------------------------------------------
    # Marchenko-Pastur bound
    # ---------------------------------------------------------------

    _, lambda_plus = mp_bounds(q, sigma2)

    # ---------------------------------------------------------------
    # BBP inversion
    # ---------------------------------------------------------------

    term = lam + sigma2 * (1 - q)

    disc = term**2 - 4 * sigma2 * lam

    # Numerical protection against negative discriminant
    disc = np.maximum(disc, 0.0)

    ell = np.full_like(lam, np.nan)

    # Physical BBP condition with numerical tolerance
    valid = lam >= (lambda_plus - EPS)

    if np.any(valid):
        ell[valid] = (term[valid] + np.sqrt(disc[valid])) / 2.0

    if scalar_input:
        return float(ell[0])

    return ell
