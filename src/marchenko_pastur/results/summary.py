r"""
Summary rendering utilities for Marchenko-Pastur inference results.

This module provides a high-level, state-aware interface for formatting
and interpreting the output of MP-based spectral analysis. It transforms
raw statistical results into structured, human-readable diagnostics in
both text and HTML formats.

Notes
-----
**Role in the Architecture**

- Acts as the presentation layer of the MP pipeline.
- Consumes ``MPResult`` objects and produces interpretative summaries.
- Fully decoupled from computation (no numerical estimation performed).

**Key Features**

- State-aware diagnostics (OK / WARNING / CRITICAL regimes).
- Structured decomposition of spectral properties (bulk, spikes, inference).
- Support for both terminal output and rich HTML (Jupyter-friendly).
- Adaptive truncation of weak factors for readability.

**Design Principles**

- Separation of concerns: computation vs. interpretation.
- Interpretability-first: emphasizes economic meaning of spectral structure.
- Robustness: gracefully handles model breakdown and edge cases.
"""

# ======================================================================
# STANDARD LIBRARY IMPORTS
# ======================================================================
import html
from typing import TYPE_CHECKING, List, Tuple

# ======================================================================
# THIRD-PARTY IMPORTS
# ======================================================================
import numpy as np

from marchenko_pastur.enums.enums import CovarianceMethod, ThresholdMethod

# ======================================================================
# LOCAL IMPORTS
# ======================================================================
from .tables import format_table

if TYPE_CHECKING:
    from .results import MPResult

# ======================================================================
# PUBLIC API EXPORTS
# ======================================================================
__all__ = [
    "Summary",
]


# ======================================================================
# CLASSES
# ======================================================================
class Summary:
    r"""
    High-level formatter for Marchenko-Pastur inference results.

    This class provides a structured and interpretable representation of
    spectral analysis outputs. It organizes model diagnostics into coherent
    sections, including noise structure, signal strength, spike decomposition,
    and statistical inference.

    Parameters
    ----------
    result : MPResult
        Object containing the results of MP spectral analysis.
    max_display : int, default=15
        Maximum number of factors (spikes) to display in the summary output.
        Weak factors beyond this limit may be truncated for readability.

    Notes
    -----
    **Conceptual Role**

    - Bridges the gap between raw statistical output and economic interpretation.
    - Provides a "model summary" analogous to regression outputs in econometrics.

    **State-Aware Diagnostics**

    The summary adapts its interpretation based on an internal collapse state:

    - ``OK``: Clear signal/noise separation.
    - ``WARNING``: Moderate overlap or elevated spike density.
    - ``CRITICAL``: Model breakdown (variance collapse or over-detection).

    **Output Modes**

    - ``as_text()``: Plain-text summary for terminal use.
    - ``as_html()``: Rich HTML representation (Jupyter/IPython).
    - ``_repr_html_()``: Automatic rendering in notebook environments.

    **Performance Consideration**

    All expensive computations (e.g., gap statistics) are precomputed at
    initialization to ensure efficient repeated rendering.
    """

    def __init__(self, result: "MPResult", max_display: int = 15):
        self.result = result
        self._max_display = max_display  # Control centralizado de UI

        # 1. Pre-compute spectral geometry (DRY)
        max_rank = min(result.n, result.p)
        self._spike_ratio = result.k_effective / max_rank if max_rank > 0 else 0.0

        # SOTA FIX: Strict Type Narrowing for Optional array
        if result.k_effective > 0 and result.deltas is not None and len(result.deltas) > 0:
            self._abs_gap = float(np.median(result.deltas))
            self._norm_gap = self._abs_gap / max(result.lambda_plus, 1e-8)
        else:
            self._abs_gap = 0.0
            self._norm_gap = 0.0

        # 2. Centralized diagnostics
        self._collapse_state = self._evaluate_collapse_state()

    # ==================================================
    # STATE PROPAGATION (CORE DIAGNOSTIC)
    # ==================================================
    def _evaluate_collapse_state(self) -> str:
        r"""
        Determine the structural stability of the spectral model.

        Returns
        -------
        str
            One of {"OK", "WARNING", "CRITICAL"} depending on signal density,
            spectral gaps, and backend warnings.

        Notes
        -----
        Combines:
        - Backend warnings (explicit model failures).
        - Spike density heuristics.
        - Gap-based structural diagnostics.
        """

        r = self.result

        # Use structural contract instead of string parsing
        if getattr(r, "robust_fallback_used", False):
            return "CRITICAL"

        # Stabilized structural heuristics
        if self._spike_ratio > 0.30:
            return "CRITICAL"
        elif self._spike_ratio > 0.15:
            # Dual condition: relative AND absolute collapse
            if self._norm_gap < 1.0 and self._abs_gap < 0.5:
                return "CRITICAL"
            elif self._norm_gap < 2.0:
                return "WARNING"

        return "OK"

    # ==================================================
    # PUBLIC API & MAGIC METHODS
    # ==================================================
    def __str__(self) -> str:
        return self.as_text()

    def __repr__(self) -> str:
        return (
            f"<MPResult Summary: {self.result.k_effective} spikes detected. "
            f"Regime: {self.result.regime}>"
        )

    def as_text(self) -> str:
        r"""
        Generate a full text-based summary of the MP analysis.

        Returns
        -------
        str
            A formatted multi-section string containing model configuration,
            noise structure, signal diagnostics, spike decomposition, inference,
            warnings, and execution metadata.
        """

        sep = "\n------------------------------------------------------------------------------\n"

        # NOTE: Conscious UI/Core coupling in Summary (v0.1.0 pragmatism).
        # Branding is intentionally embedded in the text output as part of the
        # presentation contract. This avoids premature API complexity (e.g. flags
        # like include_header). Revisit only if multiple output modes are required.
        branding = (
            "MARCHENKO-PASTUR SPECTRAL ANALYSIS\n"
            "=================================="
        )

        sections = [
            branding,
            self._build_header(),
            self._build_bulk(),
            self._build_spikes(),
            self._build_signal_strength(),
            self._build_inference(),
            self._build_verdict(),
            self._build_warnings(),
            self._build_footer(),
        ]
        return sep.join([s for s in sections if s])

    def as_html(self) -> str:
        r"""
        Generate an HTML-rendered version of the summary.

        Returns
        -------
        str
            HTML string with preformatted text styling suitable for display
            in Jupyter notebooks or web interfaces.
        """

        content = html.escape(self.as_text())
        return (
            "<div style='font-family: monospace; white-space: pre; "
            "background-color: #f8f9fa; padding: 15px; border-radius: 5px; "
            "border: 1px solid #dee2e6;'>"
            f"{content}"
            "</div>"
        )

    def _repr_html_(self) -> str:
        r"""
        Jupyter/IPython rich display hook.

        Returns
        -------
        str
            HTML representation of the summary for automatic rendering in
            notebook environments.
        """

        return self.as_html()

    # ==================================================
    # CORE FORMATTER
    # ==================================================
    def _kv_block(self, title: str, items: List[Tuple[str, str]]) -> str:
        r"""
        Format a block of key-value pairs into aligned text.

        Parameters
        ----------
        title : str
            Section title.
        items : list of (str, str)
            Key-value pairs to display.

        Returns
        -------
        str
            Formatted multi-line string.
        """

        if not items:
            return title
        width = max(len(k) for k, _ in items) + 2
        lines = [title]
        for k, v in items:
            lines.append(f"  {k:<{width}}: {v}")
        return "\n".join(lines)

    # ==================================================
    # HEADER & BULK
    # ==================================================
    def _build_header(self) -> str:
        r"""
        Construct the data and configuration section of the summary.

        Returns
        -------
        str
            Formatted header block with dataset dimensions, model configuration,
            and estimation settings.
        """

        r = self.result
        items = [
            ("Observations (n)", str(r.n)),
            ("Features (p)", str(r.p)),
            ("q = p/n", f"{r.q:.4f}"),
            ("Regime", r.regime.capitalize()),
            ("Zero Mass", f"{r.zero_mass:.4f}"),
            ("Covariance", r.covariance_method.name),
            ("Threshold Method", r.threshold_method.name),
            ("MP Sigma Estimator", r.mp_sigma_estimator.name),
            ("Standardized", "Yes (Correlation)" if r.standardize_data else "No (Covariance)"),
        ]

        if r.covariance_method is CovarianceMethod.SHRINKAGE:
            shrink_name = r.shrinkage_method.name if r.shrinkage_method is not None else "Unknown"
            items.append(("Shrinkage", shrink_name))
        if r.alpha is not None:
            items.append(("Alpha", str(r.alpha)))
        if r.threshold_method is ThresholdMethod.BOOTSTRAP:
            items.append(("Bootstrap Samples (B)", str(r.bootstrap_samples)))
        if r.random_state is not None:
            items.append(("Random State", str(r.random_state)))
        if r.regime == "singular":
            items.append(("Note", "High-dimensional regime (p > n) — zero eigenvalues expected"))

        return self._kv_block("DATA & CONFIGURATION", items)

        return self._kv_block("DATA & CONFIGURATION", items)

    def _build_bulk(self) -> str:
        r"""
        Construct the noise model (bulk) configuration section.

        Returns
        -------
        str
            Formatted block detailing MP theoretical limits and variance.
        """

        r = self.result
        items = [
            ("Lower Edge (λ-)", f"{r.lambda_minus:.6f}"),
            ("Noise Edge (λ+)", f"{r.lambda_plus:.6f}"),
            ("Bulk Width", f"{r.bulk_width:.6f}"),
            ("Variance (σ²)", f"{r.sigma2_hat:.6f}"),
        ]
        return self._kv_block("BULK (NOISE MODEL: MARCHENKO-PASTUR FIT)", items)

    # ==================================================
    # SIGNAL STRENGTH (STATE-AWARE)
    # ==================================================
    def _build_signal_strength(self) -> str:
        r"""
        Construct the signal strength diagnostics section.

        Returns
        -------
        str
            Formatted block with Signal-to-Noise Ratio (SNR) and gap metrics.
        """

        r = self.result

        if self._collapse_state == "CRITICAL":
            snr_val = f"{r.snr:.4f} (invalid under model breakdown)"
            snr_interp = "Unreliable (model breakdown)"
        else:
            snr_val = f"{r.snr:.4f}"
            if r.snr > 2.0:
                snr_interp = "High signal"
            elif r.snr > 1.0:
                snr_interp = "Moderate signal"
            else:
                snr_interp = "Noise-dominated"

        if r.ratio_apex > 100:
            apex_label = "Dominant spike"
        elif r.ratio_apex > 10:
            apex_label = "Strong dominance"
        else:
            apex_label = "Balanced spectrum"

        # Strict Type Narrowing before NumPy operations
        has_signal = r.k_effective > 0 and r.deltas is not None and len(r.deltas) > 0
        mean_gap_str = f"{np.mean(r.deltas):.6f}" if has_signal and r.deltas is not None else "N/A"
        max_gap_str = f"{np.max(r.deltas):.6f}" if has_signal and r.deltas is not None else "N/A"

        items = [
            ("Effective Rank (k)", str(r.k_effective)),
            ("SNR", snr_val),
            ("SNR Regime", snr_interp),
            ("Apex Ratio", f"{r.ratio_apex:.4f} ({apex_label})"),
            ("Mean Gap (Δ)", mean_gap_str),
            ("Max Gap (Δ)", max_gap_str),
        ]

        if r.k_effective == 0:
            items.append(("Interpretation", "No detectable signal"))

        return self._kv_block("SIGNAL STRENGTH & SEPARATION", items)

    # ==================================================
    # SPIKES (STATE-AWARE)
    # ==================================================
    def _build_spikes(self) -> str:
        r"""
        Construct the spike decomposition table.

        Returns
        -------
        str
            Table summarizing detected factors, including sample eigenvalues,
            population estimates, bias, statistical strength, and stability.

        Notes
        -----
        Applies adaptive truncation for weak factors based on ``max_display``.
        """

        r = self.result
        method = r.threshold_method.name
        title = f"SPIKE DECOMPOSITION (BBP + {method} INFERENCE)"

        if r.spikes_sample.size == 0:
            return f"{title}\n  No significant factors detected.\n  Spectrum consistent with noise."

        order = np.argsort(r.spikes_sample)[::-1]
        spikes = r.spikes_sample[order]
        pops = r.population_eigenvalues[order]

        rows = []
        shown = 0

        for i, (s, p) in enumerate(zip(spikes, pops)):
            abs_gap = s - r.spike_threshold
            rel_gap = abs_gap / max(r.spike_threshold, 1e-8)

            # Pure geometric classification (decoupled from global state)
            if rel_gap > 2.0 or abs_gap > 1.0:
                strength = "Strong"
            elif rel_gap > 1.0 or abs_gap > 0.5:
                strength = "Moderate"
            else:
                strength = "Weak"

            # Only legitimate and strong signal bypasses visual truncation
            if strength != "Strong" and shown >= self._max_display:
                continue

            shown += 1
            bias = ((p - s) / s) * 100 if s != 0 else float("nan")

            if r.bootstrap_distribution is not None:
                b_samples = len(r.bootstrap_distribution)
                pval_raw = (np.sum(r.bootstrap_distribution >= s) + 1) / (b_samples + 1)
                pval_str = f"{pval_raw:.4f}"
            else:
                pval_str = "—"

            if rel_gap > 1.0:
                stability = "High"
            elif rel_gap > 0.5:
                stability = "Medium"
            else:
                stability = "Low"

            rows.append(
                [
                    f"F{i + 1}",
                    f"{s:.6f}",
                    f"{p:.6f}",
                    f"{bias:+.2f}%",
                    strength,
                    pval_str,
                    stability,
                ]
            )

        table = format_table(
            headers=[
                "Factor",
                "λ_sample",
                "λ_population",
                "Bias",
                "Strength",
                "p-value*",
                "Stability",
            ],
            rows=rows,
            col_sep=4,
        )

        footer_notes = ["Note: 'Bias' reflects BBP shrinkage correction (sample → population)."]
        if r.bootstrap_distribution is not None:
            footer_notes.append("Note: p-values are empirical (bootstrap λ_max proxy).")

        hidden = r.k_effective - shown
        if hidden > 0:
            footer_notes.append(f"... ({hidden} additional weak factors omitted for clarity)")

        note_str = "\n  " + "\n  ".join(footer_notes) if footer_notes else ""
        return f"{title}\n{table}{note_str}"

    # ==================================================
    # INFERENCE
    # ==================================================
    def _build_inference(self) -> str:
        r"""
        Construct the inference methodology section.

        Returns
        -------
        str
            Formatted block detailing the threshold rule applied (MP, TW, Bootstrap).
        """

        r = self.result
        if r.threshold_method is ThresholdMethod.MP:
            items = [("Rule", "λ > λ+ ⇒ signal"), ("λ+", f"{r.lambda_plus:.6f}")]
            title = "INFERENCE: MARCHENKO-PASTUR"
        elif r.threshold_method is ThresholdMethod.TW:
            items = [("Alpha", str(r.alpha)), ("Threshold", f"{r.spike_threshold:.6f}")]
            title = "INFERENCE: TRACY-WIDOM"
        elif r.threshold_method is ThresholdMethod.BOOTSTRAP:
            items = [
                ("Samples (B)", str(r.bootstrap_samples)),
                ("Alpha", str(r.alpha)),
                ("Threshold", f"{r.spike_threshold:.6f}"),
            ]
            if r.bootstrap_distribution is not None:
                dist = r.bootstrap_distribution
                items.extend(
                    [
                        ("Mean λ_max", f"{np.mean(dist):.6f}"),
                        ("Std λ_max", f"{np.std(dist):.6f}"),
                        ("P95 λ_max", f"{np.percentile(dist, 95):.6f}"),
                        ("P99 λ_max", f"{np.percentile(dist, 99):.6f}"),
                        ("Interpretation", "Empirical null distribution of λ_max under noise"),
                    ]
                )
            title = "INFERENCE: BOOTSTRAP"
        else:
            return ""
        return self._kv_block(title, items)

    # ==================================================
    # VERDICT (STATE-AWARE)
    # ==================================================
    def _build_verdict(self) -> str:
        r"""
        Generate the final interpretative verdict.

        Returns
        -------
        str
            Human-readable conclusion summarizing the presence and reliability
            of detected factors based on the internal state.
        """

        r = self.result
        k_str = f"{r.k_effective} factor(s) detected."

        if r.k_effective > self._max_display:
            k_str = f"{r.k_effective} factor(s) detected (display limit applied)."

        if self._collapse_state == "CRITICAL":
            msg = [
                "Model breakdown (variance collapse).",
                "Results are statistically unreliable.",
                "See WARNINGS for required actions.",
            ]
        elif self._collapse_state == "WARNING":
            msg = [k_str, "Elevated density. Interpret separation with caution."]
        elif r.k_effective == 0:
            msg = ["No significant structure detected.", "Spectrum consistent with noise."]
        else:
            msg = [k_str, "Clear separation from noise."]

        return "SPECTRAL VERDICT\n  " + "\n  ".join(msg)

    # ==================================================
    # WARNINGS (STATE-AWARE & DRY)
    # ==================================================
    def _build_warnings(self) -> str:
        r"""
        Generate warnings associated with the spectral analysis.

        Returns
        -------
        str
            Formatted string of structural or backend warnings, or an empty
            string if no warnings were triggered.
        """

        r = self.result
        lines = []

        # 1. Context injection via structural contract
        if getattr(r, "robust_fallback_used", False):
            lines.append(
                "  ⚠ [WARNING] MP instability detected during estimation "
                "(robust fallback applied)."
            )

        # 2. Drain warnings queue (no 'else' to prevent shadowing)
        if r.warnings:
            for w in r.warnings:
                w_str = str(w)
                # Aesthetic filter to prevent printing the raw native warning if already explained
                if getattr(r, "robust_fallback_used", False) and "breakdown" in w_str.lower():
                    continue
                lines.append(f"  ⚠ [WARNING] {w_str}")

        if self._collapse_state == "CRITICAL":
            if self._spike_ratio > 0.15:
                lines.append(
                    f"  ⚠ [CRITICAL] Severe over-detection ({self._spike_ratio:.1%} of spectrum)."
                )
            else:
                lines.append("  ⚠ [CRITICAL] Base model breakdown detected.")

            lines.append("    -> Variance collapse is mathematically likely.")
            lines.append("    -> ACTION: Use `defactored_mp` (mandatory).")

        elif self._collapse_state == "WARNING":
            lines.append(f"  ⚠ [WARNING] Elevated spike density ({self._spike_ratio:.1%}).")
            lines.append("    -> Signal/noise separation is moderate.")
            lines.append("    -> Results should be interpreted with caution.")

        if not lines:
            return ""
        return "WARNINGS (ATTENTION REQUIRED)\n" + "\n".join(lines)

    # ==================================================
    # FOOTER
    # ==================================================
    def _build_footer(self) -> str:
        r"""
        Construct the execution metadata section.

        Returns
        -------
        str
            Formatted block with timestamp, execution time, and model version.
        """

        r = self.result
        items = [
            ("Date", r.timestamp.strftime("%Y-%m-%d %H:%M:%S")),
            ("Execution Time (s)", f"{r.execution_time_sec:.4f}"),
            ("Model Version", r.model_version),
            ("Config Hash", r.config_hash),
        ]
        return self._kv_block("EXECUTION METADATA", items)
