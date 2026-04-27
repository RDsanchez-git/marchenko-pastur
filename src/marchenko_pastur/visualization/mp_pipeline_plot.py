r"""
High-level visualization pipeline for Marchenko-Pastur (MP) spectral analysis.

This module provides a user-facing API that integrates preprocessing,
Random Matrix Theory (RMT) estimation, and diagnostic visualization into
a single workflow. It is designed for exploratory spectral analysis and
visual validation of factor structures in high-dimensional datasets.

The pipeline optionally performs factor removal (defactoring) prior to
final MP estimation, enabling comparison between raw and idiosyncratic
spectra. It supports multiple visualization modes, including spectral
density (PDF), cumulative distribution (CDF), and scree plots.

Notes
-----
**Main entry point:**
- ``plot_mp_defactored``

**Architecture Notes**

This module intentionally reconstructs the empirical spectrum at plotting time,
instead of retrieving it from the MP estimation result object (``MPResult``).

This design follows a strict separation of concerns:

- The inference engine (``run_mp``) is optimized for high-performance computing (HPC)
  and returns a lightweight result object without storing large intermediate data
  such as the full eigenvalue spectrum or transformed matrices.

- The visualization layer (this module) recomputes the necessary quantities
  (standardized data and eigenvalues) on demand, only when plotting is requested.

This trade-off ensures:

- Minimal memory footprint during large-scale simulations or batch estimation.
- No hidden state or large object retention inside result containers.
- Deterministic and reproducible visualization consistent with the inference pipeline.
- Exact metric alignment: The pipeline dynamically scales the reconstructed empirical
  spectrum to match the chosen covariance estimator (e.g., Pearson), ensuring visual
  integrity without coupling the plotting logic to the underlying matrix state.

While this introduces a small computational overhead during plotting, this cost is
negligible compared to the benefits in scalability and architectural clarity.

This module is intended for exploratory analysis and visual diagnostics.
It should not be used as a substitute for formal statistical inference.
"""

# ======================================================================
# STANDARD LIBRARY IMPORTS
# ======================================================================
import logging
from typing import Any, Optional

# ======================================================================
# THIRD-PARTY IMPORTS
# ======================================================================
import numpy as np
from numpy.typing import ArrayLike

# ======================================================================
# LOCAL IMPORTS
# ======================================================================
from marchenko_pastur.api import run_mp
from marchenko_pastur.utils.eigen import compute_empirical_spectrum
from marchenko_pastur.utils.preprocessing import defactor_data
from marchenko_pastur.utils.preprocessing import standardize as std_data

from .plot_scree import plot_scree
from .plot_spectral_fit import plot_spectral_fit

# ======================================================================
# MODULE CONFIGURATION
# ======================================================================
logger = logging.getLogger(__name__)

# ======================================================================
# PUBLIC API EXPORTS
# ======================================================================
__all__ = [
    "plot_mp_defactored",
]


# ======================================================================
# PUBLIC API FUNCTIONS
# ======================================================================
def plot_mp_defactored(
    X: ArrayLike,
    defactor: bool = False,
    k_init: Optional[int] = None,
    max_factors: int = 10,
    plot_type: str = "pdf",
    covariance: str = "classical",
    **plot_kwargs: Any,
) -> Any:
    r"""
    Run a full Marchenko-Pastur (MP) spectral analysis pipeline with optional
    defactoring and visualization.

    This function orchestrates a complete workflow for high-dimensional
    spectral diagnostics:

    1. (Optional) Estimate the number of latent factors using MP theory.
    2. Remove detected factors via PCA (defactoring).
    3. Re-estimate MP parameters on the processed data.
    4. Reconstruct the empirical spectrum (RMT-consistent).
    5. Generate diagnostic plots (PDF, CDF, or scree).

    The function is designed for exploratory analysis and visual validation
    of factor structures, particularly in econometrics and quantitative finance.

    Parameters
    ----------
    X : ArrayLike
        Input data matrix of shape (n_samples, n_features).
    defactor : bool, default=False
        Whether to remove latent factors before the final MP fit.
        If True, a preliminary MP estimation is used to infer the number
        of factors (unless ``k_init`` is provided).
    k_init : int or None, default=None
        User-specified number of factors to remove.
        If None, the number of factors is estimated using MP theory.
    max_factors : int, default=10
        Upper bound on the number of factors removed when ``k_init`` is None.
        Prevents overfitting and numerical instability.
    plot_type : {"pdf", "cdf", "scree"}, default="pdf"
        Type of visualization to produce:
        - "pdf": Spectral density with MP fit.
        - "cdf": Empirical vs theoretical CDF comparison.
        - "scree": Eigenvalue scree plot with spike detection.
    covariance : {"classical", "pearson"}, default="classical"
        The covariance estimator to use for MP bounds. If "pearson",
        the reconstructed spectrum is automatically scaled by :math:`1/n`
        to maintain visual alignment with theoretical bounds.
    **plot_kwargs
        Additional keyword arguments passed to the underlying plotting
        functions (``plot_spectral_fit`` or ``plot_scree``).

    Returns
    -------
    Any
        Matplotlib Axes object(s) returned by the selected plotting function.
        The exact structure depends on the chosen visualization mode.

    Raises
    ------
    ValueError
        If ``plot_type`` is not one of {"pdf", "cdf", "scree"}.

    Notes
    -----
    - When ``defactor=True``, the data is transformed using a PCA-based factor model
      and scaled by idiosyncratic variance. No additional standardization is applied
      during MP estimation.
    - When ``defactor=False``, the data is standardized before MP estimation, and the
      spectrum is computed from the corresponding correlation structure.
    - The empirical eigenvalue spectrum used for visualization is recomputed within
      this function using the internal spectral engine (``compute_empirical_spectrum``)
      to ensure strict consistency with the RMT assumptions (normalization by :math:`1/n`).
    - If ``covariance='pearson'``, the recomputed spectrum is dynamically scaled to
      match the unbiased bounds returned by the MP engine.
    - This recomputation is intentional: the MP estimation engine (``run_mp``) does not
      store intermediate data such as eigenvalues in order to remain lightweight and
      suitable for high-performance computing workflows.
    - This design ensures that inference remains memory-efficient, while visualization
      remains exact and reproducible.

    Examples
    --------
    >>> axes = plot_mp_defactored(X, defactor=True, plot_type="pdf")
    >>> axes = plot_mp_defactored(X, plot_type="scree", max_factors=5)
    """
    X_arr = np.asarray(X, dtype=np.float64)

    # ======================================================
    # STEP 1: PREPROCESSING
    # ======================================================
    if defactor:
        # ------------------------------------------
        # 1A. Detectar k (macro-factores)
        # ------------------------------------------
        if k_init is None:
            res_pre = run_mp(X_arr, standardize_data=False, covariance=covariance)
            k = min(res_pre.k_effective, max_factors)
        else:
            k = k_init

        k = min(k, X_arr.shape[1] - 1)

        logger.info(f"[PIPELINE] Defactoring with k={k}")

        # ------------------------------------------
        # 1B. Defactorizar
        # ------------------------------------------
        X_used = defactor_data(X_arr, k=k, verbose=True)

        # La matriz X_used ya está escalada por varianza idiosincrática
        standardize = False
    else:
        X_used = X_arr
        standardize = True

    # ======================================================
    # STEP 2: MP FIT
    # ======================================================
    res = run_mp(X_used, standardize_data=standardize, covariance=covariance)

    # ======================================================
    # STEP 3: EXACT RMT-CONSISTENT EIGENVALUES
    # ======================================================
    # SOTA FIX: Uso estricto del motor interno (normalización 1/n)
    # en lugar de np.cov / np.corrcoef (normalización 1/(n-1)).
    if standardize:
        X_eval = std_data(X_used)
    else:
        X_eval = X_used

    eig = compute_empirical_spectrum(X_eval, assume_centered=False)

    # Alineación Métrica Visual basada en el contrato del motor
    engine_cov = res.covariance_method.value
    if engine_cov == "pearson":
        n_samples = X_eval.shape[0]
        if n_samples > 1:
            eig = eig * (n_samples / (n_samples - 1))
    elif engine_cov == "shrinkage":
        logger.warning(
            "Plotting pipeline recomputes the classical spectrum. "
            "Visual alignment with shrinkage MP bounds is not mathematically guaranteed."
        )

    # ======================================================
    # STEP 4: PLOT DISPATCHER
    # ======================================================
    if plot_type == "pdf":
        return plot_spectral_fit(eig, res, **plot_kwargs)
    elif plot_type == "cdf":
        return plot_spectral_fit(eig, res, plot_cdf=True, **plot_kwargs)
    elif plot_type == "scree":
        return plot_scree(eig, res, **plot_kwargs)
    else:
        raise ValueError("plot_type must be 'pdf', 'cdf', or 'scree'.")
