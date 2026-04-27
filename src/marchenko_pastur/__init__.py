r"""
Marchenko-Pastur spectral analysis toolkit.

This package provides high-performance tools for analyzing the eigenvalue
spectrum of high-dimensional datasets using Random Matrix Theory (RMT),
with a focus on the Marchenko-Pastur law.

Notes
-----
**Main Features**

- Detection of latent factors via spectral methods.
- Robust statistical inference (MP, Tracy-Widom, Bootstrap).
- High-performance design (HPC-ready, memory-efficient).
- Integrated visualization and reporting tools.

**API Overview**

- ``run_mp`` : Main entry point for MP spectral analysis.
- ``MPResult`` : Immutable container with inference results.
- ``Summary`` : Formatter for model diagnostics.
- ``visualization`` : Submodule containing plotting utilities.
- ``enums`` : Submodule with strict type configurations for production.

Internal modules (``engine``, ``utils``, ``core``) are not part of the public API.
The library is designed for both research and production environments.

Examples
--------
>>> import marchenko_pastur as mp
>>> res = mp.run_mp(X)
>>> print(res.summary())
"""

# ======================================================================
# PUBLIC API EXPORTS
# ======================================================================
from . import enums, visualization
from .api import run_mp
from .results.results import MPResult
from .results.summary import Summary

__all__ = [
    "run_mp",
    "MPResult",
    "Summary",
    "visualization",
    "enums",
]
