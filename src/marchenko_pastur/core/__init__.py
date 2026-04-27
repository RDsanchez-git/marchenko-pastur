r"""
Core mathematical components for Marchenko-Pastur analysis.

This submodule contains low-level implementations of:

- Spectral theory (Marchenko-Pastur bounds and density)
- Spike correction (BBP inversion)
- Noise estimation and bulk fitting
- Tracy-Widom thresholds and scaling

These functions are primarily intended for internal use by the main
estimators, but selected mathematical primitives are exposed here
for advanced researchers and custom pipeline building.
"""

from .bbp import bbp_population_eigenvalue
from .mp_fit import fit_mp
from .mp_theory import mp_bounds, mp_density
from .tw_threshold import tracy_widom_threshold, tw_scale

__all__ = [
    "mp_bounds",
    "mp_density",
    "bbp_population_eigenvalue",
    "tracy_widom_threshold",
    "tw_scale",
    "fit_mp",
]
