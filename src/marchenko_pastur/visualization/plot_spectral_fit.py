r"""
Spectral visualization utilities for Random Matrix Theory (RMT) diagnostics.

This module provides high-level visualization tools for comparing empirical
eigenvalue distributions against the theoretical Marchenko-Pastur (MP) law.
It is designed as a core component of the spectral analysis pipeline, enabling
robust graphical validation of noise vs signal separation in high-dimensional data.

Notes
-----
**Key Features**

- Empirical vs theoretical MP comparison (PDF and CDF modes).
- Automatic spike detection visualization (BBP phase transition).
- Bulk vs signal separation with optional dual-view layouts.
- Adaptive broken-axis rendering for extreme signal-to-noise regimes.
- Robust Kernel Density Estimation (KDE) with boundary reflection.
- Numerical stability safeguards (degenerate spectra, zero variance, KDE failures).

The implementation is tightly aligned with Random Matrix Theory assumptions:

- Proper normalization of empirical spectral mass.
- Explicit handling of the Dirac mass at zero when :math:`q > 1`.
- Consistent scaling between empirical histograms and theoretical densities.

**Design Principles**

- API-first: Single entry point (``plot_spectral_fit``) with flexible configuration.
- Numerical robustness: Defensive programming against pathological inputs.
- Visualization clarity: Data-driven layout selection (e.g., broken axes).
- Reproducibility: Deterministic transformations without hidden state.

**Internal Utilities**

- ``_reflected_kde``: Private helper implementing boundary-corrected KDE with
  reflection and exact normalization using trapezoidal integration.

**Dependencies**

- NumPy: numerical operations.
- Matplotlib: plotting backend.
- SciPy: KDE estimation and numerical integration.
- Internal modules:
    - ``mp_density`` (theoretical MP distribution).
    - ``MPResult`` (container for spectral fit results).
    - ``optimal_bins`` (data-driven histogram binning).

**Compatibility & Safety**

- Fully compatible with NumPy 2.0+ (uses ``scipy.integrate.trapezoid``).
- Safe under degenerate spectral conditions (e.g., zero variance, singular KDE).
- Designed for integration into research-grade and production pipelines.
- All LaTeX strings are defined as raw strings to ensure compatibility
      with strict Python parsers (3.12+).
    - Broken axis logic is fully data-driven based on BBP gap detection.

**Eigenvalue Scaling**

This module assumes that empirical eigenvalues are computed consistently
with the normalization used in the MP estimation pipeline. If `MPResult`
was generated using `covariance='pearson'`, the input `eigenvalues` MUST
also be scaled by :`math:1/(n-1)` to maintain visual alignment with theoretical densities.
"""

# ======================================================================
# STANDARD LIBRARY IMPORTS
# ======================================================================
import warnings
from collections import OrderedDict
from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt

# ======================================================================
# THIRD-PARTY IMPORTS
# ======================================================================
import numpy as np
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpecFromSubplotSpec
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.stats import gaussian_kde

# ======================================================================
# LOCAL IMPORTS
# ======================================================================
from marchenko_pastur.core.mp_theory import mp_density
from marchenko_pastur.results.results import MPResult

from .spectral_utils import extract_robust_diagnostics, inject_watermark, optimal_bins

# ======================================================================
# PUBLIC API EXPORTS
# ======================================================================
__all__ = [
    "plot_spectral_fit",
]


# ======================================================================
# PRIVATE HELPER FUNCTIONS
# ======================================================================
# ---------------------------------------------
#  Reflection KDE + Exact Normalization
# ---------------------------------------------
def _reflected_kde(eig_bulk, l_min, l_max):
    r"""
    Constructs a boundary-corrected kernel density estimator (KDE)
    using reflection to mitigate bias near the support edges.

    This implementation includes strong numerical safeguards against
    degenerate inputs (e.g., zero variance), returning a null density
    function when KDE estimation is not feasible.

    Parameters
    ----------
    eig_bulk : np.ndarray
        Eigenvalues within the bulk (noise regime).
    l_min : float
        Lower bound of the theoretical support.
    l_max : float
        Upper bound of the theoretical support.

    Returns
    -------
    Callable
        A density function f(x) that evaluates the reflected KDE.
        Returns zero-valued output if KDE construction fails or
        input variance is numerically negligible.

    Notes
    -----
    - Uses reflection at both boundaries to reduce bias.
    - Normalization is performed using ``scipy.integrate.trapezoid``
      for compatibility with NumPy 2.0+.
    - Includes fallback behavior for singular covariance matrices
      in ``gaussian_kde``.
    """

    lower_ref = 2 * l_min - eig_bulk
    upper_ref = 2 * l_max - eig_bulk

    extended = np.concatenate([eig_bulk, lower_ref, upper_ref])

    # Internal structural hardening against variance collapse
    if np.var(extended) <= 1e-12:
        return lambda x: np.zeros_like(x)

    # Defense in depth against SciPy kernel instability
    try:
        kde = gaussian_kde(extended)
    except np.linalg.LinAlgError:
        return lambda x: np.zeros_like(x)

    def density(x):
        raw = kde(x) + kde(2 * l_min - x) + kde(2 * l_max - x)
        # Exact normalization (SciPy trapezoid for NumPy 2.0+)
        norm = trapezoid(raw, x)
        return raw / norm if norm > 0 else raw

    return density


# ======================================================================
# PUBLIC API FUNCTIONS
# ======================================================================


def plot_spectral_fit(
    eigenvalues: Union[np.ndarray, list],
    result: MPResult,
    bins: Optional[int] = None,
    show_kde: bool = True,
    kde_mode: str = "reflection",
    plot_cdf: bool = False,
    xscale: str = "linear",
    focus_bulk: bool = False,
    dual_view: bool = False,
    ax: Optional[Axes] = None,
    # Params
    auto_break_x: bool = True,
    break_threshold_x: float = 0.4,
    width_ratios_x: Tuple[float, float] = (3, 1),
    show_diagnostics: bool = True,
    **kwargs,
) -> Tuple[Axes, ...]:
    r"""
    Renders advanced spectral diagnostics comparing empirical eigenvalue
    distributions against the theoretical Marchenko-Pastur (MP) law.

    Supports both PDF and CDF visualization, including robust handling
    of degenerate inputs, automatic axis breaking, and multiple views.

    Parameters
    ----------
    eigenvalues : array-like
        Eigenvalues of the sample covariance matrix.
    result : MPResult
        Object containing MP theoretical parameters and spike detection results.
    bins : int, optional
        Number of histogram bins. If None, uses an optimal rule.
    show_kde : bool, default=True
        Whether to overlay a kernel density estimate (KDE).
    kde_mode : {"reflection", "standard", "off"}, default="reflection"
        KDE estimation mode.
    plot_cdf : bool, default=False
        If True, plots the cumulative distribution function instead of PDF.
    xscale : {"linear", "log"}, default="linear"
        Scale of the x-axis.
    focus_bulk : bool, default=False
        If True, restricts visualization to the bulk region.
    dual_view : bool, default=False
        If True, shows both full spectrum and bulk zoom.
    ax : matplotlib.axes.Axes, optional
        Axis to plot on.
    auto_break_x : bool, default=True
        Automatically enables broken x-axis if spike separation is large.
    break_threshold_x : float, default=0.4
        Threshold ratio to trigger axis break.
    width_ratios_x : tuple, default=(3, 1)
        Width ratios for broken axis layout.

    Returns
    -------
    tuple of matplotlib.axes.Axes
        Axes objects used in the visualization.

    Notes
    -----
    - Empirical densities are scaled using ``eig_plot.size`` rather than total
      dimensionality :math:`p` to ensure consistency when :math:`q > 1` and
      zero eigenvalues are excluded.
    - KDE estimation is robust to degenerate inputs:
      if variance is too small or ``gaussian_kde`` fails, KDE is omitted.
    - Reflection KDE includes boundary correction and exact normalization.
    - Numerical integration uses ``scipy.integrate.trapezoid`` for forward
      compatibility with NumPy 2.0+.
    - All LaTeX strings are defined as raw strings to ensure compatibility
      with strict Python parsers (3.12+).
    - Broken axis logic is fully data-driven based on BBP gap detection.
    """

    eig = np.asarray(eigenvalues, dtype=np.float64)
    λ_minus = result.lambda_minus
    λ_plus = result.lambda_plus
    threshold = result.spike_threshold
    sigma2 = result.sigma2_hat
    q = result.q
    k = result.k_effective

    # =========================================================
    # STRUCTURAL DIAGNOSTICS
    # =========================================================
    is_robust_mode, diag_msg = extract_robust_diagnostics(result)
    title_suffix = " [ROBUST FALLBACK]" if (is_robust_mode and show_diagnostics) else ""

    # -------------------------------------------------
    # CDF MODE
    # ------------------------------------------------
    if plot_cdf:
        if ax is None:
            fig, ax_full = plt.subplots(figsize=(8, 5))
        else:
            ax_full = ax

        x_emp = np.sort(eig)
        y_emp = np.arange(1, len(x_emp) + 1) / len(x_emp)

        ax_full.step(x_emp, y_emp, where="post", label="Empirical CDF (Step)")

        # # Extended theoretical support for CDF
        xmax_obs = np.max(x_emp) if x_emp.size > 0 else λ_plus
        x_grid = np.linspace(λ_minus, max(xmax_obs, λ_plus) * 1.05, 1000)
        pdf = mp_density(x_grid, q, sigma2)

        # Using inline CDF
        cdf_cont = cumulative_trapezoid(pdf, x_grid, initial=0)

        continuous_mass = min(1.0, 1.0 / q)
        if cdf_cont[-1] > 0:
            cdf_cont = (cdf_cont / cdf_cont[-1]) * continuous_mass

        # Exact Dirac mass
        jump = max(0.0, 1.0 - (1.0 / q))
        cdf_theory = jump + cdf_cont

        ax_full.plot(
            x_grid,
            cdf_theory,
            linewidth=2.5,
            color="tab:orange",
            label="Theoretical MP CDF (Theory)",
        )
        ax_full.axvline(
            λ_plus, linestyle="--", linewidth=1.5, color="gray", label=r"Noise Edge $\lambda_+$"
        )

        # Raw strings for LaTeX sequences
        ax_full.set_title(rf"CDF Validation | q={q:.3f}, $\sigma^2$={sigma2:.2f}")
        inject_watermark(ax_full, is_robust_mode, diag_msg, show_diagnostics)
        ax_full.set_xlabel(r"Eigenvalue Magnitude ($\lambda$)")
        ax_full.set_ylabel("Cumulative Probability")
        ax_full.legend(loc="best")

        return (ax_full,)

    # ==================================================
    # PDF PREPARATION
    # ==================================================
    # Robust filter for q > 1 (support at zero)
    eig_plot = eig[eig > 1e-10] if q > 1 else eig
    eig_bulk = eig_plot[eig_plot <= threshold]
    spikes_geom = eig_plot[eig_plot > threshold]

    # Use optimal bins if not provided
    if bins is None:
        bins_global = optimal_bins(eig_bulk)
    else:
        bins_global = bins

    # Pure theoretical grid (no scaling)
    x = np.linspace(λ_minus, λ_plus, 500)

    # Pure MP Curve (Area=1). Scaling is handled in Hist Weights.
    mp_pure = mp_density(x, q, sigma2) * max(1.0, q)

    # Truncated empirical mass
    mass_factor = eig_bulk.size / eig_plot.size if eig_plot.size > 0 else 1.0

    # KDE Setup
    kde_density = None
    if show_kde and len(eig_bulk) > 1:
        if kde_mode == "reflection":
            kde_density = _reflected_kde(eig_bulk, λ_minus, λ_plus)
        elif kde_mode == "standard":
            # Inline protection for standard mode
            if np.var(eig_bulk) > 1e-12:
                try:
                    kde_density = gaussian_kde(eig_bulk)
                except np.linalg.LinAlgError:
                    kde_density = None
        elif kde_mode == "off":
            kde_density = None
        else:
            raise ValueError("Invalid kde_mode")

    # --------------------------------------------------
    # Broken-Axis Detection (Data-Driven)
    # --------------------------------------------------
    do_break_x = False
    if auto_break_x and len(spikes_geom) > 0 and not focus_bulk:
        bbp_gap_size = np.min(spikes_geom) - λ_plus
        total_range = np.max(eig_plot) - np.min(eig_plot)
        if total_range > 1e-12 and (bbp_gap_size / total_range) > break_threshold_x:
            do_break_x = True

    # SETUP AXES COMPLEXITY
    ax_left = ax_right = ax_full = ax_bulk = None
    wspace_broken = 0.01

    if dual_view:
        if ax is not None:
            warnings.warn(
                "Dual View does not use the provided axis. Creating a new figure.", UserWarning)
        fig = plt.figure(figsize=(10, 8))
        gs_main = fig.add_gridspec(2, 1, hspace=0.3)
        ax_bulk = fig.add_subplot(gs_main[1])

        if do_break_x:
            gs_top = GridSpecFromSubplotSpec(
                1, 2, subplot_spec=gs_main[0], width_ratios=width_ratios_x, wspace=wspace_broken)
            ax_left = fig.add_subplot(gs_top[0])
            ax_right = fig.add_subplot(gs_top[1], sharey=ax_left)
        else:
            ax_full = fig.add_subplot(gs_main[0])

    elif focus_bulk:
        if ax is None:
            fig, ax_bulk = plt.subplots(figsize=(8, 5))
        else:
            ax_bulk = ax
            fig = ax.figure

    else:  # Full View Only (normal or broken)
        if do_break_x:
            if ax is None:
                fig = plt.figure(figsize=(8, 5))
                gs_main = fig.add_gridspec(1, 1)
                gs = GridSpecFromSubplotSpec(
                    1, 2, subplot_spec=gs_main[0], width_ratios=width_ratios_x, wspace=wspace_broken
                    )
            else:
                fig = ax.figure
                ax.set_visible(False)
                # Strict Type Narrowing for SubplotSpec
                ss = getattr(ax, "get_subplotspec", lambda: None)()
                if ss is not None:
                    gs = GridSpecFromSubplotSpec(
                        1, 2, subplot_spec=ss, width_ratios=width_ratios_x, wspace=wspace_broken)
                else:
                    gs = fig.add_gridspec(1, 2, width_ratios=width_ratios_x, wspace=wspace_broken)

            ax_left = fig.add_subplot(gs[0])
            ax_right = fig.add_subplot(gs[1], sharey=ax_left)
        else:
            if ax is None:
                fig, ax_full = plt.subplots(figsize=(8, 5))
            else:
                ax_full = ax
                fig = ax.figure

    # Pylance Barrier - fig is guaranteed to be bound from this point onward
    if 'fig' not in locals() or fig is None:
        raise RuntimeError("Invariant violation: Figure initialization failed.")

    # -------------------------------------------------------
    # Render Top View (Broken or Full)
    # -------------------------------------------------------
    handles = []
    labels = []

    if not focus_bulk or dual_view:
        if do_break_x:
            # Type Narrowing para Ejes
            if ax_left is None or ax_right is None:
                raise RuntimeError("Invariant violation: Broken axes are None.")

            # --- AX LEFT (ISOLATED BULK) ---
            bins_left = optimal_bins(eig_bulk) if bins is None else bins
            local_bins = np.linspace(λ_minus, λ_plus, bins_left + 1)
            bin_width = local_bins[1] - local_bins[0]

            global_density_weights = np.ones_like(eig_bulk) / (eig_plot.size * bin_width)

            ax_left.hist(
                eig_bulk,
                bins=local_bins.tolist(),  # Bypass Pylance array warning
                weights=global_density_weights,
                color="tab:blue",
                alpha=0.6,
                label="Empirical Noise Density",
            )

            ax_left.plot(
                x, mp_pure * mass_factor, linewidth=2.5,
                color="tab:orange", label="Theoretical Noise PDF (MP)")

            if kde_density is not None:
                ax_left.plot(
                    x, kde_density(x) * mass_factor, linestyle=":", linewidth=2.5,
                    color="black", label="KDE Fit (Reflected)")

            ax_left.axvline(
                λ_minus, linestyle="--", linewidth=1.5, color="tab:orange",
                label=r"Noise Edges $\lambda_\pm$")

            ax_left.axvline(λ_plus, linestyle="--", linewidth=1.5, color="tab:orange")
            ax_left.axvline(
                threshold, linestyle="-.", linewidth=2,
                color="tab:red", label="Spike Inference Threshold")

            ax_left.text(
                0.05, 0.95, f"Mass = {mass_factor:.2f}", transform=ax_left.transAxes,
                verticalalignment="top", fontsize=11, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
            )

            # --- AX RIGHT (ISOLATED SPIKES) ---
            ymax = ax_left.get_ylim()[1]
            ax_right.scatter(
                spikes_geom, np.full_like(spikes_geom, ymax * 0.1),
                marker="x", s=80, color="tab:red", label="Detected Spikes",
            )

            ax_right.text(
                0.05, 0.95, f"Mass = {1 - mass_factor:.2f}", transform=ax_right.transAxes,
                verticalalignment="top", fontsize=11, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
            )

            # --- Broken-Axis Aesthetics ---
            ax_left.spines["right"].set_visible(False)
            ax_right.spines["left"].set_visible(False)
            ax_right.tick_params(labelleft=False, left=False)

            # Explicit assignment to prevent Pylance type collisions
            d_y = 0.015
            d_x_left = d_y * (width_ratios_x[1] / width_ratios_x[0])
            d_x_right = d_y

            ax_left.plot(
                (1 - d_x_left, 1 + d_x_left), (-d_y, +d_y), transform=ax_left.transAxes,
                color="k", clip_on=False, linewidth=1.5)

            ax_left.plot(
                (1 - d_x_left, 1 + d_x_left), (1 - d_y, 1 + d_y), transform=ax_left.transAxes,
                color="k", clip_on=False, linewidth=1.5)

            ax_right.plot(
                (-d_x_right, +d_x_right), (-d_y, +d_y), transform=ax_right.transAxes,
                color="k", clip_on=False, linewidth=1.5)

            ax_right.plot(
                (-d_x_right, +d_x_right), (1 - d_y, 1 + d_y), transform=ax_right.transAxes,
                color="k", clip_on=False, linewidth=1.5)

            # Explicit float coercion for axis limits
            pad_x_noise = float((λ_plus - λ_minus) * 0.05)
            ax_left.set_xlim(float(max(0, λ_minus - pad_x_noise)), float(λ_plus + pad_x_noise))

            sp_range = float(np.max(spikes_geom) - np.min(spikes_geom))
            pad_x_signal = float(sp_range * 0.2 if sp_range > 0 else np.max(spikes_geom) * 0.1)
            ax_right.set_xlim(
                float(np.min(spikes_geom) - pad_x_signal),
                float(np.max(spikes_geom) + pad_x_signal))

            h_left, l_left = ax_left.get_legend_handles_labels()
            h_right, l_right = ax_right.get_legend_handles_labels()
            handles = h_left + h_right
            labels = l_left + l_right

            ax_left.set_ylabel("Density")

            if not dual_view:
                fig.supxlabel(r"Eigenvalue Magnitude ($\lambda$)")
                fig.suptitle(
                  rf"Spectral Decomposition PDF (Broken Axis) | q={q:.3f},spikes={k}{title_suffix}",
                  y=1.02, fontsize=14)
            else:
                ax_left.set_title(
                  rf"Spectral decomposition PDF (Broken Axis) | q={q:.3f},spikes={k}{title_suffix}")

            inject_watermark(ax_right, is_robust_mode, diag_msg, show_diagnostics)

        else:
            # --- FULL VIEW RENDER ---
            # Type Narrowing
            if ax_full is None:
                raise RuntimeError("Invariant violation: ax_full cannot be None.")

            # Safe handling of bins iterable for Pylance
            bins_safe = bins_global.tolist() if isinstance(bins_global, np.ndarray) else bins_global

            hist_vals, _, _ = ax_full.hist(
                eig_plot,
                bins=bins_safe,
                density=True,
                color="tab:blue",
                alpha=0.6,
                label="Empirical spectrum",
            )
            # Avoid .max() method to prevent Pylance issues with List[ndarray]
            ymax = float(np.max(hist_vals)) if len(hist_vals) > 0 else 1.0

            ax_full.plot(
                x, mp_pure * mass_factor, linewidth=2.5, color="tab:orange",
                label="Theoretical Noise PDF (MP)")

            ax_full.plot(
                x,
                mp_pure * mass_factor,
                linewidth=2.5,
                color="tab:orange",
                label="Theoretical Noise PDF (MP)",
            )

            if kde_density is not None:
                ax_full.plot(
                    x,
                    kde_density(x) * mass_factor,
                    linestyle=":",
                    linewidth=2.5,
                    color="black",
                    label="KDE Fit (Reflected)",
                )

            ax_full.axvline(
                λ_minus,
                linestyle="--",
                linewidth=1.5,
                color="tab:orange",
                label=r"Noise Edges $\lambda_\pm$",
            )
            ax_full.axvline(λ_plus, linestyle="--", linewidth=1.5, color="tab:orange")
            ax_full.axvline(
                threshold,
                linestyle="-.",
                linewidth=2,
                color="tab:red",
                label="Spike Inference Threshold",
            )

            if spikes_geom.size > 0:
                ax_full.scatter(
                    spikes_geom,
                    np.full_like(spikes_geom, ymax * 0.9),
                    marker="x",
                    s=80,
                    color="tab:red",
                    label="Detected Spikes",
                )

            if xscale == "log":
                ax_full.set_xscale("log")

            ax_full.set_ylabel("Density")

            if not dual_view:
                ax_full.set_title(
                    rf"Spectral PDF | q={q:.3f}, spikes={k}, $\sigma^2$={sigma2:.2f}",
                    fontsize=14,
                )
                ax_full.set_xlabel(r"Eigenvalue Magnitude ($\lambda$)")
            else:
                ax_full.set_title(
                    rf"Spectral PDF | q={q:.3f}, spikes={k}, $\sigma^2$={sigma2:.2f}"
                )

            inject_watermark(ax_full, is_robust_mode, diag_msg, show_diagnostics)

            h_full, l_full = ax_full.get_legend_handles_labels()
            handles = h_full
            labels = l_full

    # -----------------------------------------------------
    # Render Bulk View (Focus or Bottom Panel of Dual View)
    # -----------------------------------------------------
    # -----------------------------------------------------
    # Render Bulk View (Focus or Bottom Panel of Dual View)
    # -----------------------------------------------------
    if dual_view or focus_bulk:
        target_ax = ax_bulk

        if target_ax is None:
            raise RuntimeError("Invariant violation: target_ax cannot be None in this layout.")

        target_ax.hist(
            eig_bulk,
            bins=bins_global,
            density=True,
            color="tab:blue",
            alpha=0.6,
            label="Empirical Bulk",
        )

        target_ax.plot(
            x, mp_pure, linewidth=2.5, color="tab:orange", label="Theoretical Noise PDF (MP)"
        )

        if kde_density is not None:
            target_ax.plot(
                x, kde_density(x), linestyle=":", linewidth=2.5, color="black", label="KDE Fit"
            )

        target_ax.axvline(λ_minus, linestyle="--", linewidth=1.5, color="tab:orange")
        target_ax.axvline(λ_plus, linestyle="--", linewidth=1.5, color="tab:orange")

        target_ax.set_xlim(max(0, λ_minus * 0.8), λ_plus * 1.1)

        target_ax.set_title(f"Bulk Zoom (Theoretical Support Focus){title_suffix}")
        if not dual_view:
                    inject_watermark(target_ax, is_robust_mode, diag_msg, show_diagnostics)

        target_ax.set_xlabel(r"Eigenvalue Magnitude ($\lambda$)")
        target_ax.set_ylabel("Density")

        h_bulk, l_bulk = target_ax.get_legend_handles_labels()
        if dual_view:
            handles.extend(h_bulk)
            labels.extend(l_bulk)
        elif focus_bulk and not do_break_x:
            handles = h_bulk
            labels = l_bulk

    # ---------------------------------------------------
    # Unified Legend
    # ---------------------------------------------------
    by_label = OrderedDict(zip(labels, handles))

    if do_break_x or dual_view:
        # Ensure figure object exists before calling legend
        if 'fig' not in locals():
            raise RuntimeError("Invariant violation: Figure object not found.")

        fig.legend(
            by_label.values(),
            by_label.keys(),
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            fontsize=10,
            title="Decomposition Key",
            frameon=True,
        )
        fig.subplots_adjust(right=0.82)
    else:
        active_ax = ax_bulk if focus_bulk else ax_full
        if active_ax is None:
            raise RuntimeError("Invariant violation: active_ax cannot be None in this layout.")

        active_ax.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=10)

    # ---------------------------------------------------
    # Consistent Final Return
    # ---------------------------------------------------
    ret_axes = []
    if dual_view:
        if do_break_x:
            ret_axes.extend([ax_left, ax_right])
        else:
            ret_axes.append(ax_full)
        ret_axes.append(ax_bulk)
    elif focus_bulk:
        ret_axes.append(ax_bulk)
    elif do_break_x:
        ret_axes.extend([ax_left, ax_right])
    else:
        ret_axes.append(ax_full)

    # SOTA FIX: Si hay un solo gráfico, devuelve el objeto Axes plano.
    # Si hay múltiples (dual_view o broken axis), devuelve la tupla.
    return ret_axes[0] if len(ret_axes) == 1 else tuple(ret_axes)

