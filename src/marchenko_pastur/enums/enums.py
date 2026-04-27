r"""
Enumerations defining the public configuration interface of the library.

This module centralizes all user-facing categorical options used across
the API, ensuring consistency, type safety, and discoverability.

These enums are used to specify:

- Covariance estimation methods
- Shrinkage strategies
- Thresholding procedures
- Variance estimation strategies within the MP framework

Notes
-----
**Design Philosophy**

Enums act as a formal contract between the user and the library.

They provide:

- Safer alternatives to raw strings
- Autocompletion support in IDEs
- Reduced risk of silent misconfiguration

Although string inputs are still supported for convenience,
the use of enums is recommended for production and research code.
"""

# ======================================================================
# STANDARD LIBRARY IMPORTS
# ======================================================================
from enum import Enum

# ======================================================================
# PUBLIC API EXPORTS
# ======================================================================
__all__ = [
    "CovarianceMethod",
    "ShrinkageMethod",
    "ThresholdMethod",
    "MPSigmaEstimator",
]

# ======================================================================
# ENUMERATIONS
# ======================================================================


class CovarianceMethod(str, Enum):
    r"""
    Available covariance estimation methods.

    Defines the statistical model used to estimate the covariance matrix.

    Attributes
    ----------
    CLASSICAL : str
        Classical empirical covariance estimator (MLE).
        Uses :math:`1/n` normalization for strict theoretical MP compliance.
    PEARSON : str
        Unbiased sample covariance estimator.
        Uses :math:`1/(n-1)` normalization. Matches standard statistical practice.
    SHRINKAGE : str
        Regularized covariance estimators (Ledoit-Wolf or OAS).
    TYLER : str
        Robust M-estimator for heavy-tailed elliptical distributions.

    Notes
    -----
    Choice of covariance estimator directly affects the validity
    of Random Matrix Theory assumptions used in spectral analysis.

    - CLASSICAL is fully consistent with the standard Marchenko-Pastur model.
    - PEARSON introduces a uniform scaling of the spectrum.
    - SHRINKAGE and TYLER alter the structure of the covariance matrix,
      making classical MP asymptotic results only approximate.
    """

    CLASSICAL = "classical"
    PEARSON = "pearson"
    SHRINKAGE = "shrinkage"
    TYLER = "tyler"


class ShrinkageMethod(str, Enum):
    r"""
    Shrinkage strategies for covariance regularization.

    Attributes
    ----------
    LEDOIT_WOLF : str
        Ledoit-Wolf linear shrinkage estimator with analytically
        optimal shrinkage intensity.
    OAS : str
        Oracle Approximating Shrinkage estimator optimized for
        Gaussian data under finite samples.

    Notes
    -----
    These methods improve conditioning of covariance matrices in
    high-dimensional settings (:math:`p \approx n` or :math:`p > n`).
    """

    LEDOIT_WOLF = "lw"
    OAS = "oas"


class ThresholdMethod(str, Enum):
    r"""
    Methods for determining spectral detection thresholds.

    Attributes
    ----------
    MP : str
        Marchenko-Pastur theoretical threshold.
    TW : str
        Tracy-Widom statistical threshold.
    BOOTSTRAP : str
        Empirical threshold via permutation bootstrap.

    Notes
    -----
    Threshold selection determines how spikes (signal eigenvalues)
    are separated from noise in spectral analysis.

    - MP assumes exact adherence to Marchenko-Pastur theory.
    - TW provides finite-sample corrections under Gaussian assumptions.
    - Bootstrap is model-free but computationally expensive.

    The validity of these thresholds depends on the covariance
    estimator used upstream.
    """

    MP = "mp"
    TW = "tw"
    BOOTSTRAP = "bootstrap"


class MPSigmaEstimator(str, Enum):
    r"""
    Methods for estimating the noise variance (:math:`\sigma^2`) in MP fitting.

    Attributes
    ----------
    ITERATIVE : str
        Fixed-point estimator based on MP bulk separation.
    MEDIAN : str
        Estimator based on numerical inversion of the theoretical
        MP median (Gavish & Donoho, 2014).
    TRIMMED : str
        Quantile-based estimator removing extreme eigenvalues.
        Typically an internal fallback used by ``AUTO``, but exposed
        for advanced users requiring explicit manual trimming.
    AUTO : str
        Adaptive strategy that detects breakdown conditions and
        switches to a trimmed estimator when necessary.
    """

    ITERATIVE = "iterative"
    MEDIAN = "median"
    TRIMMED = "trimmed"
    AUTO = "auto"
