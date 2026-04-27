# Docstring Standard

All core algorithms must follow the official docstring template:
[Core Docstring Guide](docs/dev/docstring_guide.md)

All experiments must follow the official experiment template:
[Experiment Guidelines](docs/dev/experiment_guidelines.md)

---

# Import Policy

## Philosophy
This project follows a strict import policy designed to:
- Maximize code clarity and readability.
- Avoid path-related errors and namespace collisions.
- Facilitate large-scale refactoring.
- Ensure compatibility with modern Python tooling (Ruff, isort, pytest).

### Rule 1: Absolute Imports
Prefer **absolute imports** throughout the codebase to maintain a clear dependency tree:

```python
# CORRECT
from marchenko_pastur.utils.parsing import parse_enum

# INCORRECT
from ..utils.parsing import parse_enum
```
### Rule 2: Strict Grouping (PEP 8)

Imports must be explicitly separated into three blocks, each ordered alphabetically:

1. Standard Library imports

2. Third-party imports (e.g., numpy, scipy)

3. Local application/library specific imports (marchenko_pastur)

```python
# ======================================================================
# STANDARD LIBRARY IMPORTS
# ======================================================================
import warnings
from typing import Optional, Union

# ======================================================================
# THIRD-PARTY IMPORTS
# ======================================================================
import numpy as np
from scipy.linalg import cho_factor

# ======================================================================
# LOCAL IMPORTS
# ======================================================================
from marchenko_pastur.utils.parsing import parse_enum

```

### Rule 3: Zero Wildcards

The use of wildcard imports (*) is strictly forbidden. It pollutes the namespace and 
breaks static analyzers (Pylance/MyPy).

```python
# INCORRECT
from numpy import *

# CORRECT
import numpy as np

```
