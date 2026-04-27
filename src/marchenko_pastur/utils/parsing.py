r"""
Input validation and parsing utilities for the public API.

This module provides robust sanitization functions to bridge the gap
between flexible user inputs (e.g., strings) and the strict, type-safe
internal core (e.g., Enums).

Notes
-----
**Role in the Architecture**

- Acts as a sanitization gateway for all user-facing functions.
- Prevents invalid or ambiguous configurations from reaching the numerical
  estimation engines.

**Design Philosophy**

- Fails fast and explicitly on invalid user inputs.
- Suppresses internal tracebacks for cleaner user-facing error messages.
- Ensures absolute type safety for downstream analytical components.
"""

# ======================================================================
# STANDARD LIBRARY IMPORTS
# ======================================================================
from enum import Enum
from typing import Type, TypeVar, Union

# ======================================================================
# TYPE VARIABLES
# ======================================================================
E = TypeVar("E", bound=Enum)

# ======================================================================
# PUBLIC API EXPORTS
# ======================================================================
__all__ = [
    "parse_enum",
]


# ======================================================================
# PUBLIC API FUNCTIONS
# ======================================================================
def parse_enum(value: Union[str, E], enum_cls: Type[E]) -> E:
    r"""
    Normalize and validate user input into a strict Enum instance.

    This function acts as a sanitization gateway between user-facing APIs
    and the internal typed core of the library. It ensures that all downstream
    components receive validated Enum instances, regardless of whether the
    user provided a raw string or an Enum.

    The matching is performed strictly against the Enum ``.value`` field,
    enforcing a deterministic and explicit contract.

    Parameters
    ----------
    value : str or Enum
        User-provided configuration value. Can be either:
        - A string matching one of the Enum values (case-insensitive).
        - An already constructed Enum instance.
    enum_cls : Type[Enum]
        The Enum class against which the value should be validated.

    Returns
    -------
    Enum
        A validated Enum instance of type ``enum_cls``.

    Raises
    ------
    ValueError
        If the string does not match any valid Enum value.
    TypeError
        If the input is neither a string nor an instance of the expected Enum.

    Notes
    -----
    - Matching is performed against ``Enum.value`` (not ``Enum.name``).
    - Strings are normalized via ``strip().lower()`` before validation.
    - Exception chaining is suppressed (``from None``) to provide clean,
      user-facing error messages without internal tracebacks.

    **Design Principles**

    - **Explicit over implicit**: No fuzzy matching or name inference.
    - **Fail fast**: Invalid inputs raise immediately with clear guidance.
    - **Separation of concerns**: Keeps parsing logic outside API orchestration.

    Examples
    --------
    >>> parse_enum("tyler", CovarianceMethod)
    <CovarianceMethod.TYLER: 'tyler'>

    >>> parse_enum(CovarianceMethod.CLASSICAL, CovarianceMethod)
    <CovarianceMethod.CLASSICAL: 'classical'>

    >>> parse_enum("invalid", CovarianceMethod)
    ValueError: Invalid value 'invalid'. Expected one of: classical, shrinkage, tyler
    """
    # --------------------------------------------------
    # Case 1: Already correct Enum
    # --------------------------------------------------
    if isinstance(value, enum_cls):
        return value

    # --------------------------------------------------
    # Case 2: String -> normalize and validate
    # --------------------------------------------------
    if isinstance(value, str):
        key = value.strip().lower()
        try:
            return enum_cls(key)
        except ValueError:
            valid = ", ".join(str(e.value) for e in enum_cls)
            raise ValueError(f"Invalid value '{value}'. Expected one of: {valid}") from None

    # --------------------------------------------------
    # Case 3: Invalid type
    # --------------------------------------------------
    raise TypeError(f"Expected str or {enum_cls.__name__}, got {type(value).__name__}")
