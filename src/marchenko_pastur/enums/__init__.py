r"""
Public enumerations for the marchenko_pastur API.

This module exposes all categorical configuration options used across
the library, including covariance estimators, thresholding methods,
and internal estimation strategies.

Users are encouraged to import enums from this namespace instead of
accessing submodules directly.

Examples
--------
>>> from marchenko_pastur.enums import CovarianceMethod
>>> CovarianceMethod.CLASSICAL
<CovarianceMethod.CLASSICAL: 'classical'>
"""

# ======================================================================
# LOCAL IMPORTS
# ======================================================================
from .enums import (
    CovarianceMethod,
    MPSigmaEstimator,
    ShrinkageMethod,
    ThresholdMethod,
)

# ======================================================================
# PUBLIC API EXPORTS
# ======================================================================
__all__ = [
    "CovarianceMethod",
    "ShrinkageMethod",
    "ThresholdMethod",
    "MPSigmaEstimator",
]
