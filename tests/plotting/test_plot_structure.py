import numpy as np
import pytest
from matplotlib.axes import Axes

from marchenko_pastur.api import run_mp
from marchenko_pastur.visualization.mp_pipeline_plot import plot_mp_defactored
from marchenko_pastur.visualization.plot_spectral_fit import plot_spectral_fit

# ======================================================
# DOM SEMANTICS & UI INTEGRATION
# ======================================================

def test_spectral_fit_semantics_and_lines(rng: np.random.Generator):
    """Verify that the DOM instantiates the correct empirical and theoretical densities."""
    X = rng.standard_normal((300, 100))
    res = run_mp(X)
    eig = np.linalg.eigvalsh(np.cov(X, rowvar=False, ddof=0))

    ax = plot_spectral_fit(eig, res)

    # El test ahora pasará porque ax ya no es una tupla
    assert isinstance(ax, Axes)

    # SOTA FIX LINTER: Casting explícito a str() para get_label()
    labels = [str(line.get_label()).lower() for line in ax.lines]
    assert any("theoretical" in lbl or "mp" in lbl for lbl in labels), (
        "The theoretical MP curve is missing from the DOM."
    )

def test_scree_plot_applies_log_scale(rng: np.random.Generator):
    """Verify that the scaling arguments modify the matplotlib DOM."""
    X = rng.standard_normal((300, 100))

    ax = plot_mp_defactored(X, plot_type="scree", yscale="log")
    target_ax = ax[0] if isinstance(ax, list) else ax

    assert target_ax.get_yscale() == "log"

def test_defactored_plot_creates_broken_axis(rng: np.random.Generator):
    """Check the implementation of the axis-breaking mechanism (POET-style correction)."""
    X = rng.standard_normal((300, 100))
    X += 5.0 * (rng.standard_normal((300, 1)) @ rng.standard_normal((1, 100)))

    ax = plot_mp_defactored(X, defactor=True, plot_type="scree")

    assert isinstance(ax, list)
    assert len(ax) >= 2  # Upper (outliers) and lower (bulk)

def test_invalid_plot_type_raises_value_error(rng: np.random.Generator):
    """SOTA FIX: Entry barrier for the visualization API."""
    X = rng.standard_normal((100, 50))
    with pytest.raises(ValueError):
        plot_mp_defactored(X, plot_type="invalid_type")
