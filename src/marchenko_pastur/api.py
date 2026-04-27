r"""
Public API for the marchenko_pastur spectral analysis framework.

This module exposes the high-level interface for performing
random matrix theory (RMT) based spectral analysis, including:

- Noise-signal separation via Marchenko-Pastur theory
- Detection of significant eigenvalue spikes
- Robust covariance estimation
- Thresholding via MP, Tracy-Widom, or bootstrap methods

The API is designed with a separation of concerns:

- Low-level statistical engines (e.g., `fit_mp`)
- Core estimator (`run_mp`)
- High-level orchestrators (e.g., `defactored_mp`)
- Result containers (`MPResult`)

Users should typically interact with this module rather than
internal submodules.

---

## Design Philosophy

This framework follows a **scientific-first engineering approach**:

- Mathematical rigor is preserved
- Model assumptions are explicit
- Failure modes are surfaced, not hidden
- Invalid inference is degraded gracefully

The API does not attempt to "fix" invalid data silently.
Instead, it exposes diagnostics and encourages correct usage.

---

## Input Configuration

Categorical configuration parameters (e.g., covariance method, thresholding)
can be provided in two equivalent forms:

- **String-based (recommended for quick usage):**

>>> run_mp(X, covariance="pearson", threshold="tw")

- **Enum-based (recommended for production and type safety):**

>>> import marchenko_pastur as mp
>>> run_mp(X, covariance=mp.enums.CovarianceMethod.TYLER)

Strings must match the canonical enum values (case-insensitive).
Invalid inputs raise explicit and informative errors.

Internally, all inputs are validated and converted to strongly-typed enums.

---

## Typical Workflow

Basic usage:

>>> import numpy as np
>>> from marchenko_pastur import run_mp
>>> np.random.seed(42)
>>> X = np.random.randn(200, 50)
>>> result = run_mp(X, covariance="classical")
>>> result.k_effective
0

For datasets with strong collinearity or latent structure:

>>> from marchenko_pastur import defactored_mp
>>> result = defactored_mp(X)

---

## When to Use Each Function

- `run_mp`:
  Use for direct spectral analysis under standard assumptions.

- `defactored_mp`:
  Use when the dataset exhibits strong collinearity or
  latent low-rank structure.

---

## Failure Philosophy

This API follows a **fail-fast + graceful degradation** strategy:

- Invalid inputs → raise exceptions
- Numerical instability → warnings + degraded metrics
- Model breakdown → explicitly flagged in results
- Theoretical mismatch → warnings (e.g., MP bounds on non-linear estimators)

The user is never given misleading statistical conclusions.

---

## Notes

This module is part of a high-dimensional inference framework
based on:

- Marchenko-Pastur law
- BBP phase transition
- Tracy-Widom distribution

For theoretical details, see the documentation of individual functions.

---

## See Also

run_mp : Core spectral estimator
defactored_mp : High-level orchestrator for structured data
fit_mp : Marchenko-Pastur fitting engine

---

## References

Marchenko, V. A., & Pastur, L. A. (1967).
Distribution of eigenvalues of random matrices.

Baik, J., Ben Arous, G., & Péché, S. (2005).
Phase transition of the largest eigenvalue.

Johnstone, I. M. (2001).
On the distribution of the largest eigenvalue in PCA.
"""
# ======================================================================
# STANDARD LIBRARY IMPORTS
# ======================================================================

import time
import warnings
from datetime import datetime
from typing import Any, Callable, Dict, NamedTuple, Optional

# ======================================================================
# THIRD-PARTY IMPORTS
# ======================================================================
import numpy as np
from numpy.typing import ArrayLike, NDArray

# ======================================================================
# LOCAL IMPORTS
# ======================================================================
from marchenko_pastur.core.bbp import bbp_population_eigenvalue
from marchenko_pastur.core.metrics import _compute_metrics
from marchenko_pastur.core.mp_fit import fit_mp
from marchenko_pastur.core.tw_threshold import tracy_widom_threshold
from marchenko_pastur.engine import classical, shrinkage, tyler
from marchenko_pastur.engine.bootstrap import bootstrap_lambda_crit
from marchenko_pastur.enums.enums import (
    CovarianceMethod,
    MPSigmaEstimator,
    ShrinkageMethod,
    ThresholdMethod,
)
from marchenko_pastur.results.results import MPResult
from marchenko_pastur.utils.eigen import compute_empirical_spectrum, sorted_eigenvalues
from marchenko_pastur.utils.parsing import parse_enum
from marchenko_pastur.utils.preprocessing import standardize

# ======================================================================
# INTERNAL TYPES & REGISTRY
# ======================================================================

class CovarianceSpec(NamedTuple):
    estimator: Callable[..., NDArray[np.float64]]
    spectrum_fn: Callable[..., NDArray[np.float64]]

def _generic_spectrum(
    X: NDArray[np.float64],
    *,
    estimator: Callable[..., NDArray[np.float64]],
    **kwargs: Any,
) -> NDArray[np.float64]:
    Sigma = estimator(X, **kwargs)
    return sorted_eigenvalues(Sigma)


COVARIANCE_REGISTRY: Dict[CovarianceMethod, CovarianceSpec] = {
    CovarianceMethod.CLASSICAL: CovarianceSpec(
        estimator=classical.compute_covariance,
        spectrum_fn=lambda X, **kw: compute_empirical_spectrum(
            X, method=CovarianceMethod.CLASSICAL, **kw
        ),
    ),
    CovarianceMethod.PEARSON: CovarianceSpec(
        estimator=classical.compute_covariance,
        spectrum_fn=lambda X, **kw: compute_empirical_spectrum(
            X, method=CovarianceMethod.PEARSON, **kw
        ),
    ),
    CovarianceMethod.SHRINKAGE: CovarianceSpec(
        estimator=shrinkage.compute_covariance,
        spectrum_fn=lambda X, **kw: _generic_spectrum(X, estimator=shrinkage.compute_covariance,
                                                      **kw),
    ),
    CovarianceMethod.TYLER: CovarianceSpec(
        estimator=tyler.compute_covariance,
        spectrum_fn=lambda X, **kw: _generic_spectrum(X, estimator=tyler.compute_covariance, **kw),
    ),
}

# ======================================================================
# MAIN API
# ======================================================================


def run_mp(
    X: ArrayLike,
    *,
    covariance: CovarianceMethod | str = CovarianceMethod.CLASSICAL,
    shrinkage_method: Optional[ShrinkageMethod | str] = None,
    threshold: ThresholdMethod | str = ThresholdMethod.MP,
    alpha: Optional[float] = None,
    standardize_data: bool = True,
    bootstrap_samples: int = 250,
    random_state: Optional[int] = None,
    mp_sigma_estimator: MPSigmaEstimator | str = MPSigmaEstimator.ITERATIVE,
) -> MPResult:
    r"""
    Perform full spectral analysis using the Marchenko-Pastur framework.

    This function implements an end-to-end pipeline for high-dimensional
    covariance analysis. It estimates the noise spectrum using random
    matrix theory, detects significant eigenvalue outliers (spikes),
    and computes derived metrics such as effective dimensionality and SNR.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data matrix. Must be 2D and contain only finite values.

    covariance : {'classical', 'pearson', 'shrinkage', 'tyler'}, default='classical'
        Method used to estimate the covariance matrix.
        - 'classical': MLE (1/n normalization), strict theoretical MP compliance.
        - 'pearson': Unbiased (1/(n-1) normalization), standard statistical practice.
        - 'shrinkage': Linear or non-linear spectrum shrinkage.
        - 'tyler': Robust elliptical estimation for heavy-tailed data.

    shrinkage_method : {'lw', 'oas'}, optional
        Shrinkage estimator used when ``covariance='shrinkage'``.

    threshold : {'mp', 'tw', 'bootstrap'}, default='mp'
        Method used to determine the upper spectral edge (:math:`\lambda_+`).

    alpha : float, optional
        Significance level used for Tracy-Widom or bootstrap thresholds.

    standardize_data : bool, default=True
        If True, standardizes each feature (zero mean, unit variance).

    bootstrap_samples : int, default=250
        Number of bootstrap iterations when ``threshold='bootstrap'``.

    random_state : int, optional
        Random seed for reproducibility in stochastic procedures.

    mp_sigma_estimator : {'median', 'iterative', 'auto'}, default='iterative'
        Method used to estimate the noise variance :math:`\sigma^2`:

        - ``'median'``: Consistent estimator under the Marchenko-Pastur model
          using numerical inversion of the theoretical MP median
          (Gavish & Donoho, 2014)
        - ``'iterative'``: Fixed-point estimator based on MP bulk separation
        - ``'auto'``: Iterative method with breakdown detection and robust fallback

    Returns
    -------
    MPResult
        Immutable result object containing:

        - Estimated MP parameters (:math:`\sigma^2`, :math:`\lambda_-`, :math:`\lambda_+`)
        - Detected spikes (sample and population eigenvalues)
        - Effective rank and SNR
        - Threshold values and diagnostics
        - Execution metadata

    Raises
    ------
    ValueError
        If input data is not 2D, contains non-finite values,
        or if parameter configuration is inconsistent.

    Notes
    -----
    This function implements a spectral inference pipeline based on
    high-dimensional random matrix theory.

    **Theory**

    Under the asymptotic regime:

    .. math::
        n, p \to \infty \quad \text{with} \quad \frac{p}{n} \to q

    the empirical eigenvalue distribution of a noise covariance matrix
    converges to the Marchenko-Pastur law with support:

    .. math::
        \lambda_- = \sigma^2 (1 - \sqrt{q})^2

    .. math::
        \lambda_+ = \sigma^2 (1 + \sqrt{q})^2

    Eigenvalues above :math:`\lambda_+` indicate the presence of low-rank
    signal components (BBP phase transition).

    **Assumptions**

    - Observations are approximately i.i.d.
    - Finite fourth moments (required for Tracy-Widom edge universality).
      If data exhibits heavy tails violating this assumption, use
      ``covariance='tyler'`` for robust elliptical estimation (see caveats in Failure Modes).
    - High-dimensional regime (p/n not negligible).

    **Algorithm**

    1. Validate and optionally standardize input data
    2. Estimate covariance matrix using selected method
    3. Compute eigenvalue spectrum
    4. Fit the Marchenko-Pastur distribution to estimate :math:`\sigma^2` and bulk edges
    5. Compute threshold (MP, Tracy-Widom, or bootstrap)
    6. Detect spectral spikes (:math:`\lambda > \text{threshold}`)
    7. Compute derived metrics (SNR, effective rank, etc.)
    8. Map sample spikes to population eigenvalues (BBP inversion)

    **Interpretation**

    - :math:`\lambda \le \lambda_+` → noise (bulk spectrum)
    - :math:`\lambda > \text{threshold}` → signal (spikes)
    - k_effective → estimated intrinsic dimensionality
    - SNR → signal-to-noise strength

    Regimes:

    - :math:`p \len` → standard regime
    - :math:`p > n` → singular regime (mass at zero eigenvalues)

    **Failure Modes**

    Statistical:

    - **Theoretical mismatch via non-linear estimators:**
      Using ``covariance='tyler'`` or ``'shrinkage'`` alters the assumptions
      underlying the standard Marchenko-Pastur model, making the classical
      asymptotic bounds inexact.
      - Tyler requires a generalized MP framework due to induced dependence.
      - Shrinkage deforms the spectrum, so MP bounds become approximate.
      Mitigation: bounds computed under these methods should be treated as heuristics or used
      strictly for robust initialization.

    - **Incompatible Estimator Policy (Shrinkage + Median):**
      Shrinkage regularization destroys the pure Wishart structure required by
      the exact MP median estimator.
      Mitigation: The API enforces a strict theoretical policy. If this combination
      is detected, it emits a ``UserWarning`` and automatically coerces the
      variance estimator to ``'auto'`` to prevent solver breakdown.

    - **Spectral edge shift due to degrees of freedom:**
      Mixing ``covariance='pearson'`` (:math:`1/(n-1)`) with strict MP theory (:math:`1/n`)
      rescales the entire spectrum by a factor of :math:`n/(n-1)`, shifting both
      the bulk support and spike detection thresholds.
      Mitigation: use ``covariance='classical'`` for pure theoretical inference.

    - :math:`\lambda_+ \to 0` leads to spectrum collapse (no noise separation).
      Mitigation: use robust variance estimation (``mp_sigma_estimator='auto'``).

    - Small sample sizes make Tracy-Widom approximation unreliable.
      Mitigation: use ``threshold='bootstrap'``.

    Numerical:

    - **Degenerate dimensions:**
      :math:`n \le 1` causes division by zero under Pearson normalization.
      Mitigation: ensure sufficient sample size (:math:`n \gg 1`).
    - Perfect multicollinearity induces severe rank deficiency and variance collapse.
      Mitigation: do not use ``run_mp`` directly; use ``defactored_mp`` to
      extract dominant structural factors first.
    - Ill-conditioned covariance matrices may produce unstable eigenvalues.
    - Extreme :math:`p \gg n` may introduce floating-point artifacts.

    **Complexity**

    - Time: :math:`O(p^3)` (standard regime) or :math:`O(n^2 p)` / :math:`O(n p^2)` depending on
      dimensionality
    - Memory: :math:`O(p^2)` (or reduced in high-dimensional regimes)

    **RMT Normalization Standard**

    All internal spectral computations strictly utilize the Random Matrix Theory
    covariance normalization scaled by 1/N (equivalent to `ddof=0`). This ensures
    mathematical consistency with the Marchenko-Pastur asymptotic limits.
    Users manually comparing outputs should note this differs from the unbiased
    sample covariance estimator scaled by 1/(N-1) used by default in `numpy.cov`
    and `pandas.DataFrame.cov`.

    See Also
    --------
    defactored_mp : High-level orchestrator for collinear datasets.
    fit_mp : Core Marchenko-Pastur distribution fitting.
    tracy_widom_threshold : Theoretical spectral edge correction.
    bbp_population_eigenvalue : Population eigenvalue recovery (BBP inversion).

    References
    ----------
    Marchenko, V. A., & Pastur, L. A. (1967).
    Distribution of eigenvalues for some sets of random matrices.

    Baik, J., Ben Arous, G., & Péché, S. (2005).
    Phase transition of the largest eigenvalue.

    Tracy, C. A., & Widom, H. (1994).
    Level-spacing distributions and the Airy kernel.

    Examples
    --------
    >>> import numpy as np
    >>> # Pure noise matrix (no latent structure)
    >>> np.random.seed(42)
    >>> X = np.random.randn(200, 50)
    >>> result = run_mp(X)
    >>> result.k_effective
    0
    """

    # --------------------------------------------------
    # TIMER (MLOps)
    # --------------------------------------------------
    start_time = time.perf_counter()

    # --------------------------------------------------
    # DATA VALIDATION (Fail-Fast)
    # --------------------------------------------------
    X_arr = np.asarray(X, dtype=np.float64)

    if X_arr.ndim != 2:
        raise ValueError(f"X must be a 2D matrix. Current shape: {X_arr.shape}")

    if not np.isfinite(X_arr).all():
        raise ValueError("X contains non-finite values (NaN or Inf). Clean the data first.")

    n, p = X_arr.shape
    if n < 2 or p < 2:
        raise ValueError(f"Dimensions of X (n={n}, p={p}) are too small for spectral inference.")

    q = p / n

    # --------------------------------------------------
    # PARAMETER COERCION AND LOGICAL VALIDATION
    # --------------------------------------------------
    def _parse_with_context(val, enum_class, param_name: str):
        try:
            return parse_enum(val, enum_class)
        except ValueError as e:
            raise ValueError(f"Invalid parameter '{param_name}': {str(e)}") from None

    cov_enum = _parse_with_context(covariance, CovarianceMethod, "covariance")
    thresh_enum = _parse_with_context(threshold, ThresholdMethod, "threshold")
    sigma_enum = _parse_with_context(mp_sigma_estimator, MPSigmaEstimator, "mp_sigma_estimator")

    shrink_enum = None
    if shrinkage_method is not None:
        shrink_enum = _parse_with_context(shrinkage_method, ShrinkageMethod, "shrinkage_method")

    if cov_enum == CovarianceMethod.SHRINKAGE and shrink_enum is None:
        raise ValueError(
        "Invalid configuration: 'shrinkage_method' must be specified when covariance='shrinkage'."
        )

    if alpha is not None and not (0 < alpha < 1):
        raise ValueError(f"Significance level 'alpha' must be in (0, 1). Received: {alpha}")

    if thresh_enum == ThresholdMethod.BOOTSTRAP and bootstrap_samples < 1:
        raise ValueError(f"'bootstrap_samples' must be >= 1. Received: {bootstrap_samples}")

    # --------------------------------------------------
    # PREPROCESSING AND COMPUTATION (MLOPS WRAPPED)
    # --------------------------------------------------
    with warnings.catch_warnings(record=True) as captured_warnings:
        warnings.simplefilter("always")

        if standardize_data:
            X_arr = standardize(X_arr)

        # --------------------------------------------------
        # COVARIANCE AND SPECTRUM EXTRACTION
        # --------------------------------------------------
        spec = COVARIANCE_REGISTRY[cov_enum]

        is_centered = standardize_data
        cov_kwargs: Dict[str, Any] = {"assume_centered": is_centered}

        if cov_enum == CovarianceMethod.SHRINKAGE and shrink_enum is not None:
            cov_kwargs["method"] = shrink_enum.value

        eigenvalues = spec.spectrum_fn(X_arr, **cov_kwargs)

        # --------------------------------------------------
        # MARCHENKO-PASTUR FIT & THEORETICAL POLICY
        # --------------------------------------------------
        current_sigma_method = sigma_enum.value
        # SOTA FIX: Shrinkage destroys the Wishart structure required by the median estimator.
        if cov_enum == CovarianceMethod.SHRINKAGE and current_sigma_method == "median":
            warnings.warn(
                f"The '{MPSigmaEstimator.MEDIAN.value}' MP estimator is theoretically incompatible "
                f"with '{CovarianceMethod.SHRINKAGE.value}' covariance. "
                f"Switching to '{MPSigmaEstimator.AUTO.value}' robust estimator.",
                UserWarning,
                stacklevel=2
            )
            current_sigma_method = MPSigmaEstimator.AUTO.value

        sigma2_hat, lambda_minus, lambda_plus = fit_mp(
            eigenvalues, n, p, method=current_sigma_method
        )

        # --------------------------------------------------
        # THRESHOLDING (EDGE DETECTION)
        # --------------------------------------------------
        bootstrap_dist = None

        if thresh_enum == ThresholdMethod.MP:
            spike_threshold = lambda_plus

        elif thresh_enum == ThresholdMethod.TW:
            a_val = 0.05 if alpha is None else alpha
            spike_threshold = tracy_widom_threshold(n, p, sigma2_hat, a_val)

        elif thresh_enum == ThresholdMethod.BOOTSTRAP:
            a_val = 0.05 if alpha is None else alpha
            spike_threshold, bootstrap_dist = bootstrap_lambda_crit(
                X_arr,
                covariance_fn=spec.estimator,
                alpha=a_val,
                B=bootstrap_samples,
                random_state=random_state,
                **cov_kwargs,
            )
        else:
            raise RuntimeError("Unsupported threshold method. Check the Registry.")

        # --------------------------------------------------
        # METRICS AND SPIKES
        # --------------------------------------------------
        k_effective, ratio_apex, snr, deltas = _compute_metrics(
            eigenvalues, lambda_plus, spike_threshold
        )

        spikes = eigenvalues[eigenvalues > spike_threshold]

        # --------------------------------------------------
        # BBP INVERSION (POPULATION SIGNALS)
        # --------------------------------------------------
        if spikes.size > 0:
            # Explicitly cast to a 1D NDArray to satisfy the strict contract of MPResult
            raw_eigs = bbp_population_eigenvalue(spikes, q=q, sigma2=sigma2_hat)
            population_eigs = np.atleast_1d(np.asarray(raw_eigs, dtype=np.float64))
        else:
            population_eigs = np.empty(0, dtype=np.float64)

        # --------------------------------------------------
        # MATRIX REGIME
        # --------------------------------------------------
        regime = "singular" if p > n else "standard"
        zero_mass = max(0.0, 1 - n / p) if p > n else 0.0

    # --------------------------------------------------
    # CIERRE DEL TIMER Y EXTRACCIÓN SOTA DE WARNINGS
    # --------------------------------------------------
    exec_time = time.perf_counter() - start_time

    warning_msgs = tuple(f"{w.category.__name__}: {str(w.message)}" for w in captured_warnings)

    # SOTA FIX: Combina políticas teóricas (ex-ante) con colapsos del motor (ex-post)
    policy_triggered = (cov_enum == CovarianceMethod.SHRINKAGE and current_sigma_method == "median")
    engine_fallback = any("breakdown" in w.lower() or "fallback" in w.lower() for w in warning_msgs)
    is_robust = bool(policy_triggered or engine_fallback)

    # --------------------------------------------------
    # RESULT PACKAGING (IMMUTABLE)
    # --------------------------------------------------
    return MPResult(
        # Dimensions
        n=n,
        p=p,
        q=q,
        regime=regime,
        zero_mass=zero_mass,
        # SOTA Configuration (Pure Enums)
        covariance_method=cov_enum,
        threshold_method=thresh_enum,
        mp_sigma_estimator=sigma_enum,
        shrinkage_method=shrink_enum,
        alpha=alpha,
        bootstrap_samples=bootstrap_samples,
        # Reproducibility
        standardize_data=standardize_data,
        random_state=random_state,
        # MP Fit
        sigma2_hat=sigma2_hat,
        lambda_minus=lambda_minus,
        lambda_plus=lambda_plus,
        bulk_width=lambda_plus - lambda_minus,
        # Spikes and Spectrum (avoiding full eigenvalue load)
        spike_threshold=spike_threshold,
        spikes_sample=spikes,
        population_eigenvalues=population_eigs,
        deltas=deltas,
        bootstrap_distribution=bootstrap_dist,
        # Metrics
        k_effective=k_effective,
        snr=snr,
        ratio_apex=ratio_apex,
        # MLOps Metadata
        timestamp=datetime.now(),
        execution_time_sec=exec_time,
        model_version="0.1.0",
        warnings=warning_msgs,
        robust_fallback_used=is_robust,
    )
