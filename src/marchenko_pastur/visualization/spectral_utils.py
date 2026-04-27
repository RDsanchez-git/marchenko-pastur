r"""
Utility functions for spectral visualization preprocessing.

This module provides numerical helpers used across visualization routines,
primarily focused on stable and data-adaptive histogram construction for
empirical eigenvalue distributions.

Notes
-----
**Key Features**

- Robust bin selection using the Freedman-Diaconis rule.
- Automatic fallback to Scott's rule under IQR degeneracy.
- Defensive handling of near-zero variance data to prevent numerical instability.
- Hard clipping of bin counts to avoid memory or rendering issues.

**Design Principles**

- Numerical stability first: safeguards against floating-point collapse and
  degenerate distributions (e.g., near-Dirac spectra).
- Visualization consistency: ensures histogram resolution adapts smoothly
  across different sample sizes and dispersion regimes.
- Lightweight utilities: no dependency on model state or MP inference objects.
"""

# ======================================================================
# STANDARD LIBRARY IMPORTS
# ======================================================================
import math

# ======================================================================
# THIRD-PARTY IMPORTS
# ======================================================================
import numpy as np
from matplotlib.axes import Axes
from numpy.typing import ArrayLike

# ======================================================================
# LOCAL IMPORTS
# ======================================================================
from marchenko_pastur.results.results import MPResult

# ======================================================================
# MODULE CONFIGURATION
# ======================================================================
# Define a numerical threshold to prevent memory explosions from near-zero bin widths
EPS = 1e-12

# ======================================================================
# PUBLIC API EXPORTS
# ======================================================================
__all__ = [
    "optimal_bins",
]


# ======================================================================
# PUBLIC API FUNCTIONS
# ======================================================================
def optimal_bins(data: ArrayLike, min_bins: int = 20, max_bins: int = 200) -> int:
    r"""
    Compute an optimal number of histogram bins using robust statistical rules.

    This function implements a hierarchy of bin-width selection methods to
    ensure stable and meaningful histogram representations of empirical data,
    particularly for eigenvalue spectra.

    Parameters
    ----------
    data : array-like
        Input data array (e.g., eigenvalues) for which the histogram will be computed.
    min_bins : int, default=20
        Minimum number of bins to enforce, preventing under-resolution.
    max_bins : int, default=200
        Maximum number of bins to enforce, preventing excessive fragmentation
        and memory overhead.

    Returns
    -------
    int
        Optimal number of histogram bins, clipped within the specified bounds.

    Notes
    -----
    **Estimation Strategy**

    1. **Primary rule (Freedman-Diaconis)**:
       Uses interquartile range (IQR) for robust scaling:

       .. math::
           h = 2 \cdot \text{IQR} \cdot n^{-1/3}

    2. **Fallback 1 (Scott's rule)**:
       Activated when IQR collapses (e.g., highly concentrated data):

       .. math::
           h = 3.5 \cdot \sigma \cdot n^{-1/3}

    3. **Fallback 2 (degenerate case)**:
       If dispersion is numerically zero, returns ``min_bins``.

    **Numerical Safeguards**

    - A strict threshold (``EPS``) is used to detect near-zero bin widths.
    - ``math.ceil`` ensures full coverage of the data range.
    - Final bin count is clipped to ``[min_bins, max_bins]`` to avoid
      pathological histogram resolutions.

    **Use Case**

    This function is primarily designed for spectral density estimation
    in high-dimensional settings, where naive binning rules often fail
    due to heavy-tailed or highly concentrated eigenvalue distributions.
    """
    arr = np.asarray(data, dtype=np.float64)
    n = len(arr)

    if n < 2:
        return min_bins

    q75, q25 = np.percentile(arr, [75, 25])
    iqr = q75 - q25

    # --------------------------------------------------
    # Primary rule: Freedman-Diaconis
    # --------------------------------------------------
    h = 2.0 * iqr / (n ** (1.0 / 3.0))

    # --------------------------------------------------
    # Fallback 1: Scott's rule if IQR collapses
    # --------------------------------------------------
    # SOTA FIX: Protect against floating-point near-zero explosions
    if h <= EPS:
        std = np.std(arr, ddof=0)
        h = 3.5 * std / (n ** (1.0 / 3.0))

    # --------------------------------------------------
    # Fallback 2: Absolute collapse (Dirac mass)
    # --------------------------------------------------
    if h <= EPS:
        return min_bins

    # --------------------------------------------------
    # Bin computation and clamping
    # --------------------------------------------------
    # SOTA FIX: using math.ceil to ensure coverage, standard practice for bin counts
    raw_bins = math.ceil((arr.max() - arr.min()) / h)

    return int(np.clip(raw_bins, min_bins, max_bins))

# ======================================================================
# DIAGNOSTICS HELPERS
# ======================================================================
def extract_robust_diagnostics(result: MPResult) -> tuple[bool, str]:
    """Extrae el estado del fallback y el mensaje semántico de forma segura (DRY)."""
    is_robust = getattr(result, "robust_fallback_used", False)

    diag_msg = None
    if is_robust and hasattr(result, "warnings") and result.warnings:
        for w in result.warnings:
            w_str = str(w).lower()
            if any(key in w_str for key in ("breakdown", "fallback", "incompatible")):
                # Micro-fix: Resiliencia ante cambios de formato de la librería 'warnings'
                diag_msg = str(w).partition(": ")[-1] or str(w)
                break

    if diag_msg is None:
        diag_msg = "Robust fallback applied to preserve numerical stability."

    return is_robust, diag_msg

def inject_watermark(
        ax: Axes, is_robust: bool, diag_msg: str, show: bool, y_pos: float = 0.05) -> None:
    """Inyecta la marca de agua estandarizada aislando la UI del parsing."""
    if is_robust and show:
        ax.text(
            0.99, y_pos,
            f"⚠️ {diag_msg}",
            transform=ax.transAxes,
            ha='right', va='bottom',
            fontsize=8, color='gray', alpha=0.7
        )
