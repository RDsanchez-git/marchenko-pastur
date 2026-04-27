r"""
Marchenko-Pastur fitting and noise variance estimation.

This module implements the core procedures required to fit the
Marchenko-Pastur (MP) distribution to an empirical eigenvalue spectrum.

Its primary responsibilities are:

- Estimation of the noise variance :math:`\sigma^2`
- Identification of the effective noise bulk
- Computation of the theoretical spectral support

The implementation is designed for high-dimensional settings where
classical covariance estimation fails.

Notes
-----
**Theory**

The Marchenko-Pastur law describes the asymptotic distribution of
eigenvalues of sample covariance matrices:

.. math::
   \rho(\lambda) = \frac{\sqrt{(\lambda_+ - \lambda)(\lambda - \lambda_-)}}{2\pi q \sigma^2 \lambda}

where:

.. math::
    \lambda_\pm = \sigma^2 (1 \pm \sqrt{q})^2

and:

.. math::
    q = \frac{p}{n}

**Design Decisions**

- Noise variance is estimated from the bulk of the spectrum.
- The median-based estimator uses numerical inversion of the MP CDF
  (Gavish & Donoho, 2014), avoiding unstable heuristics.
- Iterative refinement separates bulk eigenvalues from spikes.
- A circuit breaker detects model breakdown and switches to robust trimming.

**Assumptions**

- High-dimensional regime:

  .. math::
      n, p \to \infty \quad \text{with} \quad \frac{p}{n} \to q

- Noise eigenvalues follow the MP distribution
- Finite fourth moments

  Mitigation: if violated, use ``covariance='tyler'`` upstream.

**Failure Modes**

Statistical:

- Strong latent factors (spikes) contaminate the bulk
  → Overestimation of :math:`\sigma^2`

  Mitigation: use ``method="auto"`` (robust trimming fallback)

- Heavy-tailed distributions violate MP assumptions

  Mitigation: use robust covariance estimators (e.g. Tyler)

Numerical:

- Floating-point instability near spectral edges

  Mitigation: epsilon padding and safeguarded integration

- Root-finding non-convergence in median estimation

  Mitigation: explicit convergence checks with exception

Structural:

- Singular covariance matrices (:math:`p > n`) introduce zero eigenvalues

  Mitigation: zero-mass correction in variance estimation

**Complexity**

- Time: :math:`O(p \log p)` + iterative refinement
- Memory: :math:`O(p)`

See Also
--------
mp_theory : Theoretical MP laws and spectral bounds
estimate_sigma2_bulk : Noise variance estimator
fit_mp : High-level MP fitting interface
run_mp : Full spectral analysis pipeline
defactored_mp : Preprocessing for collinear datasets

References
----------
Marchenko, V. A., & Pastur, L. A. (1967).
Distribution of eigenvalues for some sets of random matrices.

Gavish, M., & Donoho, D. L. (2014).
The optimal hard threshold for singular values.

Ledoit, O., & Wolf, M. (2004).
A well-conditioned estimator for large-dimensional covariance matrices.

Baik, J., Ben Arous, G., & Péché, S. (2005).
Phase transition of the largest eigenvalue for nonnull covariance matrices.
"""

# ======================================================================
# STANDARD LIBRARY IMPORTS
# ======================================================================
import warnings
from functools import lru_cache
from typing import Tuple

# ======================================================================
# THIRD-PARTY IMPORTS
# ======================================================================
import numpy as np
import scipy.integrate as integrate
import scipy.optimize as optimize

# ======================================================================
# LOCAL IMPORTS
# ======================================================================
from .mp_theory import mp_bounds

# ======================================================================
# CONSTANTS
# ======================================================================
EPS = 1e-12
MAX_ITER = 50

# ======================================================================
# PUBLIC API EXPORTS
# ======================================================================
__all__ = ["fit_mp"]

#======================================================================
# CORE ALGORITHMS
#======================================================================

# -------------------------------------------------
# Estimator theoretical median of MP distribution
# -------------------------------------------------
@lru_cache(maxsize=128)
def exact_mp_median(q: float) -> float:
    r"""
    Compute the theoretical median of the Marchenko-Pastur distribution.

    This function evaluates the exact median of the Marchenko-Pastur (MP)
    distribution for a given dimensional ratio :math:`q = p/n`, assuming
    unit variance (:math:`\sigma^2 = 1`).

    The median is obtained numerically by solving the implicit equation:

    .. math::
        \int_{\lambda_-}^{m_q} \rho(x) dx = 0.5

    where :math:`\rho(x)` is the MP density.

    Parameters
    ----------
    q : float
        Dimensional ratio :math:`q = p/n`. Must be non-negative.

    Returns
    -------
    median : float
        Theoretical median of the MP distribution under :math:`\sigma^2 = 1`.

    Raises
    ------
    ValueError
        If ``q < 0``.

    RuntimeError
        If the numerical root-finding procedure does not converge.

    Notes
    -----
    **Theory**

    The Marchenko-Pastur density is defined as:

    .. math::
        \rho(x) = \frac{\sqrt{(\lambda_+ - x)(x - \lambda_-)}}{2 \pi q x}

    with support:

    .. math::
        \lambda_- = (1 - \sqrt{q})^2, \quad
        \lambda_+ = (1 + \sqrt{q})^2

    The median has no closed-form solution and must be computed numerically.

    **Practical Usage**

    In empirical settings, this function is often evaluated at:

    .. math::
        q_{\text{eff}} = \frac{\text{rank}}{n}

    instead of :math:`q = p/n`, to account for:

    - Singular covariance matrices (:math:`p > n`)
    - Deflation of strong spikes

    **Algorithm**

    1. Define MP density for :math:`\sigma^2 = 1`
    2. Integrate density from :math:`\lambda_-` to candidate median
    3. Solve for root where cumulative mass equals 0.5
    4. Use Brent's method for robust convergence

    **Failure Modes**

    Numerical:

    - Instability near support edges due to floating-point precision
      Mitigation: epsilon padding in integration bounds and safe evaluation

    - Division by zero when :math:`x \to 0`
      Mitigation: enforce lower bound on :math:`x`

    Statistical:

    - Misinterpretation of :math:`q` in singular or defactored settings
      Mitigation: use :math:`q_{\text{eff}}` when appropriate

    **Complexity**

    - Time: :math:`O(K)` per new ``q`` (numerical integration)
    - Memory: :math:`O(1)` with :func:`functools.lru_cache`

    See Also
    --------
    estimate_sigma2_bulk : Uses this median for variance estimation.
    mp_bounds : Computes MP support bounds.

    References
    ----------
    Marchenko, V. A., & Pastur, L. A. (1967).

    Gavish, M., & Donoho, D. L. (2014).
    The optimal hard threshold for singular values is :math:`4/\sqrt{3}`.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> exact_mp_median(0.5)
    0.8526  # doctest: +SKIP
    """
    if q < 0:
        raise ValueError("The dimensional ratio 'q' must be non-negative.")

    # Numerical stability for q -> 0 (Pure noise, infinite samples)
    if q < 1e-8:
        return 1.0

    l_min = (1 - np.sqrt(q))**2
    l_max = (1 + np.sqrt(q))**2

    def mp_pdf(x):
        # Prevent division by zero and math domain errors near integration bounds
        x_safe = max(x, 1e-12)
        inside_sqrt = max((l_max - x_safe) * (x_safe - l_min), 0.0)
        return np.sqrt(inside_sqrt) / (2 * np.pi * q * x_safe)

    def cdf_diff(m):
        # We want the point 'm' where the cumulative integral is exactly 0.5.
        # limit=100 increases subdivision limit for stability near MP edges.
        val, _ = integrate.quad(mp_pdf, l_min, m, limit=100)
        return val - 0.5

    # Bracket with epsilon padding to prevent floating-point edge singularities
    eps = 1e-10
    res = optimize.root_scalar(
        cdf_diff,
        bracket=[l_min + eps, l_max - eps],
        method='brentq'
    )

    if not res.converged: # type: ignore
        raise RuntimeError(f"MP median root finding did not converge for q={q}")

    return res.root # type: ignore

# -------------------------------------------------
# Estimator Variance from eigenvalue spectrum
# -------------------------------------------------

def estimate_sigma2_bulk(
    eigenvalues: np.ndarray,
    n: int,
    p: int,
    method: str = "auto",
    max_iter: int = MAX_ITER,
    tol: float = 1e-7,
) -> float:
    r"""
    Estimate noise variance (:math:`\sigma^2`) from the eigenvalue spectrum.

    This function estimates the variance of the noise component in a
    high-dimensional covariance matrix using the Marchenko-Pastur framework.

    It supports robust estimation under model misspecification, including
    singular regimes (:math:`p > n`) and strong factor contamination.

    Parameters
    ----------
    eigenvalues : ndarray
        Eigenvalues of the empirical covariance matrix.

    n : int
        Number of observations.

    p : int
        Number of variables.

    method : {"median", "iterative", "auto"}, default="auto"
        Estimation strategy:

        - "median": Consistent estimator under the MP model using numerical
          inversion of the theoretical MP median (Gavish & Donoho, 2014)
        - "iterative": Fixed-point MP fitting using bulk separation
        - "auto": Iterative with breakdown detection and fallback

    max_iter : int, default=50
        Maximum number of iterations for the iterative method.

    tol : float, default=1e-7
        Convergence tolerance for iterative updates.

    Returns
    -------
    sigma2 : float
        Estimated noise variance.

    Raises
    ------
    ValueError
        If method is not recognized.

    Notes
    -----
    **Assumptions**

    - High-dimensional regime:

      .. math::
          n, p \to \infty \quad \text{with} \quad \frac{p}{n} \to q

    - Finite fourth moments for valid MP behavior.
    - Noise eigenvalues follow the Marchenko-Pastur distribution.

    **Algorithm**

    1. Initialize :math:`\sigma^2` using a median-based estimator.
    2. Iteratively:
       - Compute MP upper bound :math:`\lambda_+`.
       - Separate noise and spikes.
       - Update :math:`\sigma^2` using noise eigenvalues.
    3. Detect breakdown conditions (see below).
    4. Apply fallback if needed ("auto" mode).

    **Failure Modes**

    Statistical:

    - Strong factors contaminate bulk estimation
      (leads to overestimation of :math:`\sigma^2`).

      Mitigation: use ``method="auto"`` (robust fallback).

    - Heavy-tailed distributions violate MP assumptions.

      Mitigation: use robust covariance (e.g. ``covariance='tyler'`` upstream).

    Numerical:

    - Spectral collapse (:math:`\text{median} \le \epsilon`).

      Mitigation: Ex-ante detection intercepts the collapse, preventing
      solver execution, and initializes with a trimmed mean.

    - MP Structural Breakdown (Solver failure).
      The empirical spectrum is distorted and does not intersect the
      theoretical MP median, causing root-finding to fail.

      Mitigation: Surgical exception handling catches the sign mismatch
      and safely falls back to the trimmed estimator.

    - Variance collapse (:math:`\sigma^2 \to 0`).

      Mitigation: circuit breaker triggers trimmed estimator.

    - Excessive spike detection (>25%).

      Mitigation: fallback to quantile-based trimming.

    Structural:

    - Singular regime (:math:`p > n`) introduces zero eigenvalues.

      Mitigation: zero-mass correction included in estimator.

    **Complexity**

    - Time: :math:`O(p \log p)` (sorting) + iterations
    - Memory: :math:`O(p)`

    See Also
    --------
    fit_mp : Full MP fitting (:math:`\sigma^2` + spectral bounds).
    mp_bounds : Theoretical MP support.

    References
    ----------
    Marchenko, V. A., & Pastur, L. A. (1967).
    Ledoit, O., & Wolf, M. (2004).

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> eigs = np.abs(np.random.randn(100))
    >>> estimate_sigma2_bulk(eigs, n=200, p=100) # doctest: +SKIP
    1.0
    """

    eig = np.asarray(eigenvalues, dtype=np.float64)
    eig = np.maximum(eig, 0.0)
    eig.sort()

    valid_methods = {"median", "iterative", "auto"}
    if method not in valid_methods:
        raise ValueError(f"Unknown estimator method '{method}'. Valid options are {valid_methods}")

    # =========================================================
    # 1. PARAMETERIZATION AND SINGULAR REGIME
    # =========================================================
    q = p / n  # Theoretical q is never altered for bounds calculation
    is_singular = p > n
    zero_mass_count = max(0, p - n)

    positive = eig[eig > EPS]

    if positive.size == 0:
        return float(max(np.mean(eig), EPS))

   # =========================================================
    # 2. ROBUST INITIALIZATION (Median-based)
    # =========================================================
    q_eff = positive.size / n
    median_emp = np.median(positive)

    # SOTA FIX: Protección Ex-Ante. Si la mediana empírica colapsó,
    # no tiene sentido llamar a SciPy.
    if median_emp <= EPS:
        warnings.warn(
            "Spectral collapse detected (median ~ 0). Initializing with trimmed mean.",
            RuntimeWarning, stacklevel=2
        )
        sigma2 = np.mean(np.sort(positive)[:int(0.9 * positive.size)])
    else:
        try:
            mp_median = exact_mp_median(q_eff)
            sigma2 = median_emp / mp_median
        except ValueError as e:
            if "signs" in str(e):
                # BREAKDOWN ESTRUCTURAL: El espectro está distorsionado
                warnings.warn(
                    "MP Structural Breakdown: empirical median is incompatible with MP support. "
                    "Initializing with trimmed mean.",
                    RuntimeWarning, stacklevel=2
                )
                sigma2 = np.mean(np.sort(positive)[:int(0.9 * positive.size)])
            else:
                raise e

    if sigma2 <= EPS:
        sigma2 = np.sum(positive) / p

    if method == "median":
        return float(max(sigma2, EPS))

    # =========================================================
    # 3. ITERATIVE / AUTO (Circuit Breaker + Singular Fix)
    # =========================================================
    if method in ("iterative", "auto"):
        for _ in range(max_iter):
            _, lambda_plus = mp_bounds(q, float(sigma2))
            noise = positive[positive <= lambda_plus]
            spikes = positive[positive > lambda_plus]

            # CIRCUIT BREAKER MLOps
            spike_ratio = spikes.size / max(positive.size, 1)
            variance_collapse = sigma2 < (0.1 * median_emp)

            if spike_ratio > 0.25 or variance_collapse:
                if method == "auto":
                    warnings.warn(
                    "MP breakdown detected: Standardization + Strong factors destroyed i.i.d. " \
                    "noise structure."
                    f"Unrealistic spike ratio ({spike_ratio:.1%}) or variance collapse. "
                    "Falling back to robust 'trimmed' estimator.",
                        RuntimeWarning,
                    )
                    # More aggressive trimming in singular regime (p > n)
                    trim_quantile = 0.85 if is_singular else 0.90
                    lambda_cut = np.quantile(positive, trim_quantile)
                    noise = positive[positive <= lambda_cut]

                    noise_dims = noise.size + zero_mass_count
                    sigma2 = max(np.sum(noise) / max(noise_dims, 1), EPS)
                    break
                else:
                    pass  # 'Pure 'iterative' has no safety net (Research mode)

            # Variance averages positive mass + zero mass
            noise_dims = noise.size + zero_mass_count
            new_sigma2 = np.sum(noise) / max(noise_dims, 1)

            if abs(new_sigma2 - sigma2) < tol:
                sigma2 = new_sigma2
                break

            sigma2 = max(new_sigma2, EPS)

    return float(max(sigma2, EPS))

# -------------------------------------------------
# Fit the MP distribution
# -------------------------------------------------

def fit_mp(
    eigenvalues: np.ndarray,
    n: int,
    p: int,
    method: str = "auto",
) -> Tuple[float, float, float]:
    r"""
    Fit the Marchenko-Pastur distribution to empirical eigenvalues.

    This function estimates the noise variance and computes the
    theoretical support of the Marchenko-Pastur distribution.

    Parameters
    ----------
    eigenvalues : ndarray
        Eigenvalues of the empirical covariance matrix.

    n : int
        Number of observations.

    p : int
        Number of variables.

    method : {"median", "iterative", "auto"}, default="auto"
        Estimation method for :math:`\sigma^2`.

    Returns
    -------
    sigma2_hat : float
        Estimated noise variance.

    lambda_minus : float
        Lower bound of the MP support.

    lambda_plus : float
        Upper bound of the MP support.

    Notes
    -----
    **Theory**

    The edges of the Marchenko-Pastur support are:

    .. math::
        \lambda_- = \sigma^2 (1 - \sqrt{q})^2

    .. math::
        \lambda_+ = \sigma^2 (1 + \sqrt{q})^2

    **Interpretation**

    - Eigenvalues below :math:`\lambda_+` → noise bulk
    - Eigenvalues above :math:`\lambda_+` → signal (spikes)

    **Failure Modes**

    Statistical:

    - Strong latent factors distort MP fit
      Mitigation: use ``method="auto"``

    - Heavy-tailed data violates MP assumptions
      Mitigation: use ``covariance='tyler'`` upstream

    Numerical:

    - Poor variance initialization affects convergence
      Mitigation: median-based initialization via ``exact_mp_median``

    **Complexity**

    - Time: :math:`O(p \log p)` + iteration
    - Memory: :math:`O(p)`

    See Also
    --------
    estimate_sigma2_bulk : Noise variance estimator.
    mp_bounds : Theoretical MP limits.
    run_mp : Full spectral analysis pipeline.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> eigs = np.abs(np.random.randn(100))
    >>> fit_mp(eigs, n=200, p=100) # doctest: +SKIP
    (1.0, 0.2, 2.8)
    """

    sigma2_hat = estimate_sigma2_bulk(eigenvalues, n, p, method=method)
    q = p / n
    lambda_minus, lambda_plus = mp_bounds(q, sigma2_hat)

    return float(sigma2_hat), float(lambda_minus), float(lambda_plus)
