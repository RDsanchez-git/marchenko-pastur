r"""
High-level pipeline for Marchenko-Pastur spectral analysis.

This module provides a user-facing pipeline that orchestrates the full
random matrix theory (RMT) workflow, combining preprocessing,
defactoring, and spectral inference into a single interface.

The primary goal of this module is to bridge the gap between:

- High-performance statistical inference (``run_mp``)
- Practical data analysis workflows (visualization, diagnostics)

Notes
-----
**Design Philosophy**

This module follows a two-layer architecture:

- Core inference layer -> ``run_mp`` (HPC-optimized, minimal memory footprint).
- Pipeline layer -> this module (user-oriented, state reconstruction).

Unlike the core API, this pipeline may:

- Reconstruct intermediate objects (e.g., processed data, eigenvalues).
- Duplicate certain computations intentionally.
- Trade performance for usability and interpretability.

This is a deliberate design decision to support exploratory data analysis
and visualization workflows without compromising the performance of the
core engine.

**Key Features**

- Optional defactoring (low-rank signal removal).
- Consistent eigenvalue extraction aligned with RMT assumptions.
- Full access to processed data used in inference.
- Compatibility with visualization workflows (e.g., scree plots).

**Theoretical Consistency**

This pipeline preserves strict consistency with the
Marchenko-Pastur framework:

- Covariance estimation uses MLE normalization (:math:`1 / n`).
- Eigenvalues are computed from the same estimator used internally.
- No use of ``np.corrcoef`` or Bessel-corrected estimators.

This ensures that:

    Visualization == Inference

i.e., the eigenvalues returned by the pipeline match exactly those used
in statistical decision-making.

**When to Use This Module**

Use this pipeline when you need:

- Eigenvalues for plotting or diagnostics.
- Access to the processed dataset (e.g., after defactoring).
- A simplified interface for exploratory analysis.

Avoid using this module when:

- Memory efficiency is critical.
- Only inference results are needed.

In those cases, use ``run_mp`` directly.

**Failure Philosophy**

This pipeline follows the same principles as the core API:

- Invalid inputs -> raise exceptions (fail-fast).
- Numerical issues -> handled upstream in ``run_mp``.
- Statistical breakdowns -> mitigated via defactoring or robust estimators.

However, the pipeline may:

- Perform additional preprocessing steps.
- Introduce extra computational cost due to state reconstruction.

This module is intentionally **not HPC-optimized**. It exists to provide
interpretability, debuggability, and visualization compatibility while
preserving full alignment with the underlying statistical theory.

See Also
--------
run_mp : Core spectral inference engine.
defactored_mp : High-level defactoring pipeline.
compute_covariance : Classical covariance estimator (RMT-consistent).

References
----------
Marchenko, V. A., & Pastur, L. A. (1967).
Distribution of eigenvalues of random matrices.

Baik, J., Ben Arous, G., & Péché, S. (2005).
Phase transition of the largest eigenvalue.

Tracy, C. A., & Widom, H. (1994).
Level-spacing distributions and the Airy kernel.
"""

# ======================================================================
# STANDARD LIBRARY IMPORTS
# ======================================================================
import logging
from typing import Optional, Tuple

# ======================================================================
# THIRD-PARTY IMPORTS
# ======================================================================
import numpy as np
from numpy.typing import ArrayLike, NDArray

# ======================================================================
# LOCAL IMPORTS
# ======================================================================
from marchenko_pastur.api import run_mp
from marchenko_pastur.engine.classical import compute_covariance
from marchenko_pastur.enums.enums import ThresholdMethod
from marchenko_pastur.results.results import MPResult
from marchenko_pastur.utils.eigen import sorted_eigenvalues
from marchenko_pastur.utils.preprocessing import defactor_data, standardize

logger = logging.getLogger(__name__)

# ======================================================================
# PUBLIC API EXPORTS
# ======================================================================
__all__ = [
    "mp_pipeline",
]

# ======================================================================
# PIPELINE FUNCTION (VISUALIZATION HELPER)
# ======================================================================


def mp_pipeline(
    X: ArrayLike,
    defactor: bool = False,
    k_init: Optional[int] = None,
    alpha: float = 0.01,
    threshold: ThresholdMethod | str = ThresholdMethod.TW,
    max_factors: int = 10,
    standardize_data: bool = True,
    covariance: str = "classical",
    verbose: bool = True,
) -> Tuple[MPResult, NDArray[np.float64], NDArray[np.float64]]:
    r"""
    High-level pipeline for Marchenko-Pastur spectral analysis with full spectrum extraction.

    This function provides an end-to-end workflow for performing random matrix
    theory (RMT) based spectral inference while also returning the processed
    dataset and its full eigenvalue spectrum for visualization purposes.

    Unlike ``run_mp``, which is optimized for high-performance inference and
    does not retain large intermediate objects, this pipeline explicitly
    reconstructs the internal state (data and spectrum) to support downstream
    tasks such as plotting and diagnostic analysis.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data matrix. Must be 2D and contain only finite values.

    defactor : bool, default=False
        Whether to apply defactoring (low-rank signal removal) before
        performing spectral analysis.

    k_init : int, optional
        Initial number of factors to remove. If None, the effective rank
        is estimated automatically via a preliminary MP analysis.

    alpha : float, default=0.01
        Significance level used for thresholding when applicable.

    threshold : {'mp', 'tw', 'bootstrap'} or ThresholdMethod, default='tw'
        Method used to determine the spectral edge.

    max_factors : int, default=10
        Maximum number of factors to remove during defactoring.

    standardize_data : bool, default=True
        If True, standardizes each feature (zero mean, unit variance)
        before spectral analysis.

    covariance : {"classical", "pearson", "shrinkage"}, default="classical"
        The covariance estimator to use. If "pearson", the returned eigenvalue
        spectrum is automatically scaled by :math:'1/(n-1)' to match the unbiased bounds.

    verbose : bool, default=True
        If True, emits informational messages during execution.

    Returns
    -------
    result : MPResult
        Result object containing spectral inference outputs (spikes,
        thresholds, metrics, etc.).

    X_used : ndarray of shape (n_samples, n_features)
        Data matrix actually used for spectral analysis after optional
        defactoring and preprocessing.

    eig : ndarray of shape (n_features,)
        Sorted eigenvalues of the covariance matrix consistent with the
        internal MP framework (MLE normalization).

    Raises
    ------
    ValueError
        If input data is not 2D or contains non-finite values.

    Notes
    -----
    This function acts as a **visualization-oriented wrapper** around
    ``run_mp``.

    It intentionally duplicates parts of the defactoring logic instead of
    calling ``defactored_mp`` directly. This design choice is required
    because ``MPResult`` does not retain the processed data or full
    eigenvalue spectrum for memory efficiency (HPC-aware design).

    **Theory**

    The pipeline is based on the Marchenko-Pastur law, which describes the
    asymptotic distribution of eigenvalues of sample covariance matrices
    under high-dimensional regimes:

    .. math::
        \lambda_+ = \sigma^2 (1 + \sqrt{q})^2

    where:

    .. math::
        q = \frac{p}{n}

    Eigenvalues exceeding this threshold indicate the presence of
    low-rank signal components (spikes).

    **Assumptions**

    - Observations are approximately i.i.d.
    - Finite fourth moments are required for Tracy-Widom corrections.
      If violated (heavy tails), use ``covariance='tyler'`` in ``run_mp``.
    - High-dimensional regime where :math:`p/n` is non-negligible.

    **Algorithm**

    1. Validate input data
    2. Optionally estimate structural rank via preliminary MP fit
    3. Remove dominant factors (defactoring)
    4. Perform final MP spectral inference using ``run_mp``
    5. Reconstruct covariance matrix using MLE normalization
    6. Compute eigenvalues consistent with the MP framework

    **Interpretation**

    - Eigenvalues below threshold -> noise (bulk spectrum)
    - Eigenvalues above threshold -> signal (spikes)
    - ``X_used`` reflects the data after structural noise removal
    - ``eig`` can be used for visualization (e.g., scree plots)

    **Failure Modes**

    Statistical:

    - Heavy-tailed distributions violate MP assumptions.
      Mitigation: use ``covariance='tyler'`` in ``run_mp``.

    - Strong latent structure inflates eigenvalues and biases MP fit.
      Mitigation: enable ``defactor=True``.

    Numerical:

    - Double standardization or centering inconsistencies can distort the spectrum.
      Mitigation: this pipeline enforces consistency via MLE covariance estimation.

    - Perfect multicollinearity leads to rank deficiency.
      Mitigation: use ``defactor=True`` or ``defactored_mp``.

    Computational:

    - Large feature dimension (p) leads to cubic complexity.
      Mitigation: reduce dimensionality or use approximate methods.

    **Complexity**

    - Time: :math:`O(p^3)` due to eigen-decomposition (twice if defactoring is enabled)
    - Memory: :math:`O(p^2)`

    See Also
    --------
    run_mp : Core spectral inference engine
    defactored_mp : High-level defactoring pipeline
    compute_covariance : Classical covariance estimator (MLE normalization)

    References
    ----------
    Marchenko, V. A., & Pastur, L. A. (1967).
    Distribution of eigenvalues of random matrices.

    Baik, J., Ben Arous, G., & Péché, S. (2005).
    Phase transition of the largest eigenvalue.

    Tracy, C. A., & Widom, H. (1994).
    Level-spacing distributions and the Airy kernel.

    Examples
    --------
    >>> import numpy as np
    >>> from marchenko_pastur.pipelines import mp_pipeline
    >>> np.random.seed(42)
    >>> X = np.random.randn(200, 50)
    >>> result, X_used, eig = mp_pipeline(X, defactor=False)
    >>> result.k_effective
    0
    >>> eig.shape
    (50,)
    """

    # --------------------------------------------------
    # 1. Validation (Fail-Fast)
    # --------------------------------------------------
    X_arr = np.asarray(X, dtype=np.float64)

    if X_arr.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if not np.isfinite(X_arr).all():
        raise ValueError("X contains non-finite values (NaN or Inf).")

    # Local state flag to preserve input immutability
    standardize_flag = standardize_data

    # --------------------------------------------------
    # 2. Preprocessing & Defactoring
    # --------------------------------------------------
    # Note: We duplicate the defactoring discovery logic here instead of calling
    # defactored_mp() because we strictly need to extract and return `X_used`
    # and the full spectrum for plotting purposes, which MPResult discards for HPC.

    if defactor:
        if verbose:
            logger.info("Applying defactoring procedure for visualization...")

        if k_init is None:
            pre_res = run_mp(
                X_arr,
                standardize_data=standardize_flag,
                covariance=covariance,
                threshold=threshold,
                alpha=alpha,
                mp_sigma_estimator="auto",
            )
            k = min(pre_res.k_effective, max_factors)
        else:
            k = k_init

        _, p = X_arr.shape
        k = max(0, min(k, p - 1))

        if k > 0:
            X_used = defactor_data(X_arr, k=k, verbose=verbose)
        else:
            X_used = X_arr.copy()

        # CRITICAL: Prevent double standardization down the pipeline
        standardize_flag = False
    else:
        X_used = X_arr.copy()

    # --------------------------------------------------
    # 3. Model Inference
    # --------------------------------------------------
    result = run_mp(
        X_used,
        standardize_data=standardize_flag,
        covariance=covariance,
        threshold=threshold,
        alpha=alpha,
    )

    ## --------------------------------------------------
    # 4. Consistent Spectrum Extraction
    # --------------------------------------------------
    X_eval = standardize(X_used) if standardize_flag else X_used

    # Pure extraction (1/n scaling)
    cov_matrix = compute_covariance(X_eval, assume_centered=standardize_flag)
    eig = sorted_eigenvalues(cov_matrix)

    # Structural Metric Alignment
    # If the engine used Pearson, we correct the exported spectrum so that
    # it numerically matches the theoretical limits of MPResult.
    engine_cov = result.covariance_method.value
    if engine_cov == "pearson":
        n_samples = X_eval.shape[0]
        if n_samples > 1:
            eig = eig * (n_samples / (n_samples - 1))
    elif engine_cov == "shrinkage":
        logger.warning(
            "Pipeline exports classical spectrum. Alignment with shrinkage bounds "
            "is not mathematically guaranteed."
        )

    return result, X_used, eig
