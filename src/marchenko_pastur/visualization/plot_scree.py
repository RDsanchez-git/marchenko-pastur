r"""
Scree plot visualization for spectral diagnostics under Random Matrix Theory.

This module implements a State-of-the-Art (SOTA) scree plot tailored for
high-dimensional econometric and financial data, enabling robust identification
of latent factor structures via the Marchenko-Pastur (MP) framework.

The visualization explicitly separates:

- Geometric structure: eigenvalues exceeding the theoretical noise edge (:math:`\lambda_+`).
- Statistical inference: effective number of factors determined by
  Tracy-Widom-based thresholds.

Notes
-----
**Key Features**

- **BBP gap detection**: Identifies the phase transition between signal and noise
  using the Baik-Ben Arous-Peche (BBP) framework.
- **Dynamic broken-axis rendering**: Automatically splits the y-axis when a
  dominant factor compresses the remaining spectrum, preserving interpretability.
- **Semantic annotation control**: Allows explicit distinction between
  inferential anchoring (``k_eff``) and geometric anchoring (number of spikes).
- **Numerical robustness**: Ensures stability under log-scale transformations
  via lower-bound clamping.

**Design Principles**

- **Inference != Geometry**: The module enforces a strict conceptual separation
  between observed spectral structure and statistically validated factors.
- **Visualization as diagnosis**: This plot is not decorative-it is a diagnostic
  tool for validating Random Matrix Theory assumptions in empirical data.

This module assumes that eigenvalues are computed consistently with the
same normalization used in the MP estimation pipeline. If the `MPResult` was
generated using `covariance='pearson'`, the input eigenvalues MUST also be
scaled by :math:`1/(n-1)` to maintain visual alignment with theoretical bounds.
"""

# ======================================================================
# STANDARD LIBRARY IMPORTS
# ======================================================================
from typing import List, Optional, Union

import matplotlib.pyplot as plt

# ======================================================================
# THIRD-PARTY IMPORTS
# ======================================================================
import numpy as np
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpecFromSubplotSpec

# ======================================================================
# LOCAL IMPORTS
# ======================================================================
from marchenko_pastur.results.results import MPResult

from .spectral_utils import extract_robust_diagnostics, inject_watermark

# ======================================================================
# PUBLIC API EXPORTS
# ======================================================================
__all__ = [
    "plot_scree",
]


# ======================================================================
# PUBLIC API FUNCTIONS
# ======================================================================
def plot_scree(
    eigenvalues: Union[np.ndarray, list],
    result: MPResult,
    max_components: int = 50,
    yscale: str = "linear",
    ax: Optional[Axes] = None,
    auto_break: bool = True,
    break_threshold: float = 0.4,
    annotate_mode: str = "inferential",
    show_diagnostics: bool = True,
) -> Union[Axes, List[Axes]]:
    r"""
    Render a scree plot with RMT-based factor diagnostics and structural gap detection.

    This function visualizes the leading eigenvalues of a dataset in descending
    order and contrasts them with theoretical noise bounds derived from the
    Marchenko-Pastur (MP) distribution. It provides advanced diagnostics for
    detecting latent factor structures in high-dimensional settings.

    Parameters
    ----------
    eigenvalues : array-like
        Empirical eigenvalues of the covariance or correlation matrix.
        Must be consistent with the normalization used in ``result``.
    result : MPResult
        Output of the MP estimation pipeline containing theoretical bounds
        (:math:`\lambda_+`), inference thresholds, and the effective number
        of factors (``k_effective``).
    max_components : int, default=50
        Maximum number of largest eigenvalues to display.
    yscale : {"linear", "log"}, default="linear"
        Scale of the y-axis. Log scale is stabilized against zero values.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on. If None, a new figure is created.
    auto_break : bool, default=True
        Whether to automatically activate a broken y-axis when a dominant
        factor creates a large separation (BBP gap).
    break_threshold : float, default=0.4
        Minimum relative size of the BBP gap (as a fraction of total spectral
        range) required to trigger axis breaking.
    annotate_mode : {"inferential", "geometric"}, default="inferential"
        Controls the semantic interpretation of the BBP gap annotation:
        - ``"inferential"``: Anchors annotation relative to the inferred number
          of factors (``k_eff``), emphasizing statistical decision-making.
        - ``"geometric"``: Anchors annotation to the actual number of eigenvalues
          exceeding the noise edge, emphasizing raw spectral structure.

    Returns
    -------
    Union[matplotlib.axes.Axes, List[matplotlib.axes.Axes]]
        - Single Axes if no axis break is applied.
        - List of two Axes [top, bottom] if broken-axis visualization is used.

    Raises
    ------
    ValueError
        If ``annotate_mode`` is not one of {"inferential", "geometric"}.

    Notes
    -----
    **BBP Gap Interpretation**

    The BBP gap is defined as the difference between the smallest spike
    eigenvalue and the theoretical noise edge (:math:`\lambda_+`). It quantifies
    the strength of factor separation.

    **Broken-Axis Logic**

    The axis is split only when at least one spike exists and the BBP gap
    dominates the spectrum based on the relative ``break_threshold``.

    **Numerical Stability**

    Eigenvalues are clamped to a minimum of 1e-12 to prevent failures
    in log-scale rendering.

    **Design Philosophy**

    This function intentionally separates geometry (what the data shows) from
    inference (what the model concludes). The annotation system exposes this
    distinction explicitly to the user.
    """
    if annotate_mode not in ["inferential", "geometric"]:
        raise ValueError("annotate_mode must be 'inferential' or 'geometric'.")

    eig = np.sort(np.asarray(eigenvalues, dtype=np.float64))[::-1]
    p = len(eig)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.figure

    k = min(max_components, p)
    eig_plot = eig[:k]

    # SOTA FIX: Avoid strict zero or negative for log-scale
    eig_plot_safe = np.maximum(eig_plot, 1e-12)

    indices = np.arange(1, k + 1)
    marker_size = 6 if p < 200 else 3

    λ_plus = result.lambda_plus
    threshold = result.spike_threshold
    k_eff = max(1, min(result.k_effective, k))

    # =========================================================
    # 1. STRUCTURAL DETECTION (BBP GAP REFERENCED)
    # =========================================================
    do_break = False
    val_top: float = 0.0
    val_bottom: float = 0.0
    spikes_geom = eig_plot[eig_plot > λ_plus]

    if yscale == "linear" and auto_break and len(spikes_geom) > 0:
        smallest_spike = spikes_geom[-1]
        largest_noise = eig_plot[len(spikes_geom)] if len(eig_plot) > len(spikes_geom) else 0.0

        bbp_gap_size = smallest_spike - largest_noise
        total_range = eig_plot[0] - np.min(eig_plot)

        if total_range > 1e-12 and (bbp_gap_size / total_range) > break_threshold:
            do_break = True
            val_top = smallest_spike
            val_bottom = largest_noise

    # =========================================================
    # Annotation logic helper
    # =========================================================
    def _draw_annotation(target_ax):
        if len(spikes_geom) == 0:
            return

        first_spike = spikes_geom.min()
        idx_geom = len(spikes_geom)
        gap = first_spike - λ_plus

        # SOTA Semantics: Explicit anchoring decision
        x_coord = k_eff if annotate_mode == "inferential" else idx_geom

        if annotate_mode == "inferential":
            distance = abs(idx_geom - k_eff)
            # Dynamic curvature (Paper-Grade): scales with distance, caps at 0.3
            rad = -0.3 * min(1.0, distance / 10.0)
            conn_style = f"arc3,rad={rad:.3f}"
        else:
            conn_style = "arc3"

        target_ax.annotate(
            f"BBP Gap\n{gap:.2f}",
            xytext=(x_coord + max(2, int(k * 0.05)), first_spike + eig_plot_safe[0] * 0.05),
            xy=(idx_geom, first_spike),
            arrowprops=dict(arrowstyle="->", color="black", connectionstyle=conn_style),
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
        )

    # =========================================================
    # STRUCTURAL DIAGNOSTICS & SEMANTIC FILTERING
    # =========================================================
    is_robust_mode, diag_msg = extract_robust_diagnostics(result)

    base_title = f"Scree Plot | n={result.n}, p={result.p}, spikes={result.k_effective}"
    if is_robust_mode and show_diagnostics:
        base_title += " [ROBUST FALLBACK]"

    # =========================================================
    # 2. BROKEN-AXIS RENDERING
    # =========================================================
    if do_break:
        if ax is None:
            raise RuntimeError(
                "Invariant violation: 'ax' cannot be None during broken-axis rendering.")

        ax.set_visible(False)

        # Type Narrowing estricto para SubplotSpec
        ss = getattr(ax, "get_subplotspec", lambda: None)()
        if ss is not None:
            gs = GridSpecFromSubplotSpec(
                2, 1, subplot_spec=ss, height_ratios=[1, 3], hspace=0.08
            )
        else:
            gs = fig.add_gridspec(2, 1, height_ratios=[1, 3], hspace=0.08)

        ax_top = fig.add_subplot(gs[0])
        ax_bottom = fig.add_subplot(gs[1])

        for current_ax in (ax_top, ax_bottom):
            current_ax.plot(
                indices, eig_plot_safe, marker="o", markersize=marker_size,
                linewidth=1.5, label="Eigenvalues",
            )
            current_ax.axhline(
                λ_plus, linestyle="--", linewidth=2, color="C1", label=r"MP edge ($\lambda_+$)")
            current_ax.axhline(
                threshold, linestyle="-.", linewidth=2, color="C3", label="Spike threshold")
            current_ax.axvline(k_eff, linestyle=":", color="gray", label="k effective")

        pad_top = float((eig_plot[0] - val_top) * 0.1 if eig_plot[0] > val_top else val_top * 0.05)
        pad_bottom = float((val_bottom - np.min(eig_plot)) * 0.15)

        ax_top.set_ylim(float(val_top - pad_top), float(eig_plot[0] + pad_top))
        ax_bottom.set_ylim(
            float(max(0, np.min(eig_plot) - pad_bottom)), float(val_bottom + pad_bottom * 1.5))

        ax_top.spines["bottom"].set_visible(False)
        ax_bottom.spines["top"].set_visible(False)
        ax_top.xaxis.tick_top()
        ax_top.tick_params(labeltop=False, bottom=False)
        ax_bottom.xaxis.tick_bottom()

        d = 0.015
        ax_top.plot(
            (-d, +d), (-d, +d), transform=ax_top.transAxes, color="k", clip_on=False, linewidth=1.5)

        ax_top.plot(
            (1 - d, 1 + d), (-d, +d), transform=ax_top.transAxes,
              color="k", clip_on=False, linewidth=1.5)

        ax_bottom.plot(
            (-d, +d), (1 - d, 1 + d), transform=ax_bottom.transAxes,
              color="k", clip_on=False, linewidth=1.5)

        ax_bottom.plot(
            (1 - d, 1 + d), (1 - d, 1 + d), transform=ax_bottom.transAxes,
              color="k", clip_on=False, linewidth=1.5)

        _draw_annotation(ax_top)

        ax_bottom.set_xlabel("Component Index")
        ax_bottom.set_ylabel("Eigenvalue Magnitude")
        ax_top.set_title(
            base_title,
            fontsize=11,
            fontweight="bold" if is_robust_mode and show_diagnostics else "normal")

        inject_watermark(ax_bottom, is_robust_mode, diag_msg, show_diagnostics, y_pos=0.01)

        ax_top.legend(loc="upper right")

        return [ax_top, ax_bottom]

    # =========================================================
    # 3. STANDARD RENDERING (Intact Fallback)
    # =========================================================
    else:
        ax.plot(
            indices, eig_plot_safe, marker="o", markersize=marker_size,
            linewidth=1.5, label="Eigenvalues",
        )

        ax.axhline(λ_plus, linestyle="--", linewidth=2, color="C1", label=r"MP edge ($\lambda_+$)")
        ax.axhline(threshold, linestyle="-.", linewidth=2, color="C3", label="Spike threshold")
        ax.axvline(k_eff, linestyle=":", color="gray", label="k effective")

        _draw_annotation(ax)

        if yscale == "log":
            ax.set_yscale("log")
            ax.set_ylabel("Eigenvalue Magnitude (Log Scale)")
        else:
            ax.set_ylabel("Eigenvalue Magnitude")

        ax.set_xlabel("Component Index")
        ax.set_title(base_title, fontsize=11,
                    fontweight="bold" if is_robust_mode and show_diagnostics else "normal")

        inject_watermark(ax, is_robust_mode, diag_msg, show_diagnostics, y_pos=0.01)

        ax.legend()

        return ax
