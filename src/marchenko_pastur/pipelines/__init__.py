r"""
High-level pipelines for Marchenko-Pastur spectral analysis.

This subpackage provides user-facing workflows that orchestrate
multiple components of the framework, combining preprocessing,
estimation, and post-processing steps into convenient pipelines.

The goal of these pipelines is to simplify common tasks while
preserving transparency and control over the underlying methodology.

Notes
-----
**Available Pipelines**

- ``defactored_mp``: Two-pass estimation procedure for datasets with strong
  low-rank structure or multicollinearity.
- ``mp_pipeline``: Visualization-oriented helper that returns the MP result,
  the processed dataset, and the corresponding eigenvalue spectrum.

**Design Philosophy**

- Pipelines are convenience wrappers built on top of the core API.
- They may trade strict minimalism for usability and workflow clarity.
- For full control and reproducibility, use ``run_mp`` directly.

See Also
--------
marchenko_pastur.api.run_mp : Core spectral estimation engine.
marchenko_pastur.engine : Low-level covariance estimators.
"""

# ======================================================================
# LOCAL IMPORTS
# ======================================================================
from .defactored_mp import defactored_mp
from .mp_pipeline import mp_pipeline

# ======================================================================
# PUBLIC API EXPORTS
# ======================================================================
__all__ = [
    "defactored_mp",
    "mp_pipeline",
]
