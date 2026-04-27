r"""
Utility functions for numerical operations, preprocessing, and input validation.

This subpackage provides reusable low-level components used across the
Marchenko-Pastur analysis pipeline. It includes spectral computations,
data preprocessing, and API boundary sanitization.

Notes
-----
These utilities are designed to be:

- Numerically stable.
- Reusable across modules.
- Independent of high-level orchestration logic.

Advanced users may import directly from this subpackage for custom workflows.

Examples
--------
>>> import numpy as np
>>> from marchenko_pastur.utils import standardize, sorted_eigenvalues
>>> X = np.array([[1.0, 2.0], [3.0, 4.0]])
>>> X_std = standardize(X)
>>> eigvals = sorted_eigenvalues(X_std)

>>> from marchenko_pastur.enums import CovarianceMethod
>>> from marchenko_pastur.utils import parse_enum
>>> parse_enum("tyler", CovarianceMethod)
<CovarianceMethod.TYLER: 'tyler'>
"""

# ======================================================================
# LOCAL IMPORTS
# ======================================================================
from .eigen import (
    compute_empirical_spectrum,
    largest_eigenvalue,
    sorted_eigenvalues,
)
from .parsing import (
    parse_enum,
)
from .preprocessing import (
    defactor_data,
    standardize,
)

# ======================================================================
# PUBLIC API EXPORTS
# ======================================================================
__all__ = [
    # eigen
    "sorted_eigenvalues",
    "largest_eigenvalue",
    "compute_empirical_spectrum",
    # preprocessing
    "standardize",
    "defactor_data",
    # parsing
    "parse_enum",
]
