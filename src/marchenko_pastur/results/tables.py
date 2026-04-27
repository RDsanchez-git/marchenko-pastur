r"""
Lightweight table formatting utilities for textual statistical summaries.

This module provides a minimal, dependency-free formatter for rendering
aligned tables in plain text, inspired by statistical packages such as
statsmodels.

Notes
-----
**Design Goals**

- Zero dependencies (pure Python, no pandas/tabulate).
- Deterministic alignment for reproducible reports.
- Flexible typing (automatic casting to string).
- Separation of indentation and column spacing for layout control.

This module is intentionally simple and optimized for integration with
``Summary``, where consistent formatting and readability are critical.
"""

# ======================================================================
# STANDARD LIBRARY IMPORTS
# ======================================================================
from typing import Any, Sequence

# ======================================================================
# PUBLIC API EXPORTS
# ======================================================================
__all__ = [
    "format_table",
]


# ======================================================================
# PUBLIC API FUNCTIONS
# ======================================================================
def format_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[Any]],
    title: str = "",
    col_sep: int = 2,
    indent: int = 2,
) -> str:
    r"""
    Lightweight table formatting utilities for textual statistical summaries.

    This module provides a minimal, dependency-free formatter for rendering
    aligned tables in plain text, inspired by statistical packages such as
    statsmodels.

    Notes
    -----
    **Design Goals**

    - Zero dependencies (pure Python, no pandas/tabulate).
    - Deterministic alignment for reproducible reports.
    - Flexible typing (automatic casting to string).
    - Separation of indentation and column spacing for layout control.

    This module is intentionally simple and optimized for integration with
    ``Summary``, where consistent formatting and readability are critical.
    """
    if not headers:
        return ""

    # Calcular ancho máximo de cada columna
    cols = list(zip(*([headers] + list(rows))))
    widths = [max(len(str(cell)) for cell in col) for col in cols]

    sep_str = " " * col_sep
    indent_str = " " * indent

    def format_row(row: Sequence[Any]) -> str:
        return indent_str + sep_str.join(str(cell).rjust(width) for cell, width in zip(row, widths))

    lines = []

    if title:
        lines.append(indent_str + title)
        lines.append(indent_str + "-" * len(title))

    lines.append(format_row(headers))
    lines.append(format_row(["-" * w for w in widths]))

    for row in rows:
        lines.append(format_row(row))

    return "\n".join(lines)
