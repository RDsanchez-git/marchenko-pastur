# Documentation Standards (Scientific Python)

## Purpose

This document defines the official documentation standards for this repository.

The goal is to ensure:

* Scientific rigor
* Consistency across modules
* Compatibility with NumPyDoc and Sphinx
* Readability for both researchers and engineers
* Actionable documentation (theory → decisions)

---

## General Principles

1. **Docstrings are the contract of the API**
2. **If an algorithm comes from a paper → it must be cited**
3. **Clarity > verbosity**
4. **Consistency is mandatory**
5. **Not all functions deserve full documentation**
6. **Document decisions, not just parameters**

---

## Language Convention

All documentation must be written in **English**.

* Code: English
* Docstrings: English
* Comments: English
* Errors & warnings: English

---

## Docstring Standard (NumPyDoc)

All docstrings must follow the NumPyDoc format.

### Required Sections

* Summary
* Parameters
* Returns

### Optional Sections

* Raises
* Notes
* See Also
* References
* Examples

⚠️ Only these section headers are allowed at top level.
All custom structure must live inside **Notes**.

---

## Scientific Documentation Philosophy

This repository follows:

> “Readable code is good.
> Reproducible science is better.
> Actionable theory is best.”

This means:

✔ Explain theory
✔ Explain when theory breaks
✔ Explain what to do when it breaks

---

## 🔬 Mandatory Conventions (SOTA)

### 1. Raw Strings (REQUIRED)

All scientific docstrings MUST use raw strings:

```python
def function(...):
    r"""
    ...
    """
```

This prevents escaping issues and ensures LaTeX renders correctly.

---

### 2. Math Rendering (REQUIRED)

All mathematical expressions must use Sphinx MathJax:

```rst
.. math::
    \lambda_+ = \sigma^2 (1 + \sqrt{q})^2
```

❌ Forbidden:

* Unicode math (λ, σ², √)
* Inline hacks

✔ Only LaTeX via `.. math::`

---

### 3. Actionable Theory (CRITICAL)

Theoretical assumptions MUST be tied to user decisions.

❌ Bad:

* Finite fourth moments required

✔ Good:

* Finite fourth moments required. If violated, use `covariance='tyler'`.

👉 Every limitation must include a mitigation path if available.

---

### 4. Failure Modes with Mitigation (MANDATORY)

All core functions MUST include **Failure Modes**.

Each failure mode must include:

* Cause
* Effect
* Mitigation

Example:

```
**Failure Modes**

Statistical:

- Heavy-tailed data violates MP assumptions.
  Mitigation: use ``covariance='tyler'``.

Numerical:

- Perfect multicollinearity causes rank deficiency.
  Mitigation: use ``defactored_mp``.
```

---

### 5. Deterministic Examples (CI/CD SAFE)

All examples MUST:

* Be executable
* Be deterministic
* Reflect correct theoretical behavior

✔ Use:

```python
np.random.seed(42)
```

❌ Never hardcode incorrect outputs

---

## 📄 FULL TEMPLATE (CORE FUNCTIONS)

```python
def function_name(X: np.ndarray, q: float) -> float:
    r"""
    One-line summary.

    Extended description explaining what the function does.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Description.

    q : float
        Description.

    Returns
    -------
    result : float
        Description.

    Raises
    ------
    ValueError
        If inputs are invalid.

    Notes
    -----
    Brief explanation of context.

    **Theory**

    .. math::
        \lambda_+ = \sigma^2 (1 + \sqrt{q})^2

    **Assumptions**

    - i.i.d observations
    - Finite fourth moments. If violated, use ``covariance='tyler'``
    - High-dimensional regime (p/n not negligible)

    **Algorithm**

    1. Standardize data
    2. Estimate covariance
    3. Compute eigenvalues
    4. Fit MP law

    **Interpretation**

    - Eigenvalues above threshold → signal
    - Eigenvalues below → noise

    **Failure Modes**

    Statistical:

    - Heavy-tailed distributions violate assumptions
      Mitigation: use ``covariance='tyler'``

    Numerical:

    - Perfect multicollinearity → rank deficiency
      Mitigation: use ``defactored_mp``

    **Complexity**

    - Time: O(p^3)
    - Memory: O(p^2)

    See Also
    --------
    related_function

    References
    ----------
    Author (Year). Paper title.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> X = np.random.randn(100, 50)
    >>> function_name(X, q=0.5)
    0.0
    """
```

---

## Documentation Levels

### 1. Core Algorithms (MAX DETAIL)

Examples:

* run_mp
* fit_mp
* tracy_widom_threshold

✔ Full template required
✔ Theory (MathJax) required
✔ Failure Modes required
✔ References required
✔ Actionable mitigation required

---

### 2. Public API (MEDIUM DETAIL)

Examples:

* defactored_mp
* summary classes

✔ Focus on behavior
✔ Minimal theory
✔ Include guidance if misuse is likely

---

### 3. Helpers / Utilities (MINIMAL)

Examples:

* standardize
* internal helpers

✔ Only:

* Summary
* Parameters
* Returns

❌ No theory
❌ No references
❌ No failure modes

---

## 🔗 See Also (MANDATORY FOR CORE)

Every core function must reference related functions.

This ensures:

* Discoverability
* Navigation
* Cohesive API design

---

## References Policy

* Use original papers when possible
* Use consistent format
* Avoid over-citation

---

## Anti-Patterns (STRICTLY FORBIDDEN)

* Mixing languages
* Unicode math instead of LaTeX
* Missing parameter descriptions
* Fake or incorrect examples
* Failure modes without mitigation
* Over-documenting trivial functions
* Adding custom top-level sections (breaks NumPyDoc)

---

## CI/CD Integration

This standard is enforced through:

* Code review (PR)
* Sphinx build validation
* Doctest validation (`pytest --doctest-modules`)
* Optional docstring linting

---

## Final Rule

If a function implements theory:

→ Document the theory
→ Document when it fails
→ Document how to handle failure

If it does not:

→ Keep it simple
