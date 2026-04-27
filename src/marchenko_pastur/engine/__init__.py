r"""
Covariance estimation engines for high-dimensional statistical inference.

This module provides a unified interface to multiple covariance
estimators used throughout the spectral analysis pipeline.

The estimators differ in their statistical assumptions, robustness,
and numerical stability properties.

Notes
-----
**Available engines**

- ``classical_covariance``: Standard empirical covariance (MLE, no regularization).
- ``shrinkage_covariance``: Regularized covariance estimators (Ledoit-Wolf, OAS).
- ``tyler_covariance``: Robust M-estimator for heavy-tailed elliptical distributions.
- ``bootstrap_lambda_crit``: Permutation-based bootstrap estimator for empirical spike detection
    thresholds.

**Design Philosophy**

Each estimator represents a different point in the bias-variance-robustness
trade-off:

- Classical: unbiased but unstable in high dimensions.
- Shrinkage: biased but well-conditioned.
- Tyler: robust to heavy tails, scale-invariant.

The choice of estimator directly impacts the validity of Random Matrix
Theory (RMT) assumptions used in downstream analysis.

**Actionable Insight**

- If your data violates Gaussian assumptions (e.g., heavy tails or outliers),
  prefer ``tyler_covariance``.
- If :math:`p \approx n` or :math:`p > n` (ill-conditioned covariance),
  prefer ``shrinkage_covariance``.
- Otherwise, use ``classical_covariance``.

See Also
--------
core.run_mp : Main spectral analysis pipeline.
core.defactored_mp : Robust orchestrator for collinear datasets.
"""

# ======================================================================
# LOCAL IMPORTS AND NAMESPACING
# ======================================================================
from .bootstrap import bootstrap_lambda_crit
from .classical import compute_covariance as classical_covariance
from .shrinkage import compute_covariance as shrinkage_covariance
from .tyler import compute_covariance as tyler_covariance

# ======================================================================
# PUBLIC API EXPORTS
# ======================================================================
__all__ = [
    "classical_covariance",
    "shrinkage_covariance",
    "tyler_covariance",
    "bootstrap_lambda_crit",
]
