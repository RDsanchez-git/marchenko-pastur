r"""
Immutable result container for Marchenko-Pastur spectral analysis.

This module defines the core result object returned by the MP inference engine.
The design prioritizes high-performance computing (HPC), reproducibility, and
strict separation between inference and visualization layers.

Notes
-----
**Design Principles**

- **Immutability (C-level enforced)**:
  The result object is frozen and uses NumPy read-only arrays to prevent
  accidental mutation and ensure safe reuse in parallel workflows.

- **HPC-aware memory footprint**:
  The full empirical spectrum is intentionally *not stored* to avoid memory
  overhead in large-scale simulations (e.g., Monte Carlo or bootstrap loops).

- **Reproducibility (MLOps-ready)**:
  A deterministic ``config_hash`` is generated from all model inputs, enabling
  exact experiment tracking and auditability.

- **Strong typing (Enum-driven)**:
  All configuration parameters are represented using enums, preventing invalid
  states and ensuring consistency across the pipeline.

- **Separation of concerns**:
  This module contains no visualization logic. Presentation is delegated to
  higher-level layers (e.g., ``summary``, ``visualization``).

- **Data Ownership (DTO Pattern)**:
  This object acts as a pure Data Transfer Object. All domain logic, metric
  derivations (e.g., eigen-gaps), and threshold distributions are computed
  by the upstream orchestration layer and injected here for safe storage.
"""

# ======================================================================
# STANDARD LIBRARY IMPORTS
# ======================================================================
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Tuple

# ======================================================================
# THIRD-PARTY IMPORTS
# ======================================================================
import numpy as np
from numpy.typing import NDArray

# ======================================================================
# LOCAL IMPORTS
# ======================================================================
from marchenko_pastur.enums.enums import (
    CovarianceMethod,
    MPSigmaEstimator,
    ShrinkageMethod,
    ThresholdMethod,
)
from marchenko_pastur.results.summary import Summary

# ======================================================================
# PUBLIC API EXPORTS
# ======================================================================
__all__ = [
    "MPResult",
]


# ======================================================================
# CLASSES
# ======================================================================
@dataclass(frozen=True, slots=True)
class MPResult:
    r"""
    Immutable container for Marchenko-Pastur spectral inference results.

    This dataclass encapsulates the full output of the MP analysis pipeline,
    including theoretical noise bounds, detected signal components (spikes),
    and diagnostic metrics. It is designed for safe reuse, reproducibility,
    and efficient execution in high-dimensional settings.

    Parameters
    ----------
    n : int
        Number of observations.
    p : int
        Number of features.
    q : float
        Aspect ratio (p / n).
    covariance_method : CovarianceMethod
        Covariance estimator used in the analysis.
    threshold_method : ThresholdMethod
        Method used to detect significant spikes (MP, Tracy-Widom, Bootstrap).
    mp_sigma_estimator : MPSigmaEstimator
        Estimator used for noise variance (σ²).
    shrinkage_method : Optional[ShrinkageMethod]
        Shrinkage method applied if covariance_method is SHRINKAGE.
    alpha : Optional[float]
        Significance level used in inference (if applicable).
    bootstrap_samples : Optional[int]
        Number of bootstrap samples (if bootstrap method is used).
    standardize_data : bool
        Whether the data was standardized before analysis.
    random_state : Optional[int]
        Random seed for reproducibility.
    sigma2_hat : float
        Estimated noise variance.
    lambda_minus : float
        Lower edge of the MP bulk.
    lambda_plus : float
        Upper edge of the MP bulk (noise edge).
    bulk_width : float
        Width of the MP bulk (:math:`\lambda_+ - \lambda_-`).
    spike_threshold : float
        Threshold used to classify eigenvalues as signal.
    spikes_sample : ndarray
        Sample eigenvalues classified as spikes.
    population_eigenvalues : ndarray
        Estimated population eigenvalues (BBP-corrected).
    k_effective : int
        Number of statistically significant factors.
    snr : float
        Signal-to-noise ratio.
    ratio_apex : float
        Ratio between the largest eigenvalue and the noise edge.
    zero_mass : float
        Proportion of zero eigenvalues (relevant in high-dimensional regimes).
    regime : str
        Regime classification (e.g., "classical", "high-dimensional", "singular").
    timestamp : datetime
        Execution timestamp.
    execution_time_sec : float
        Runtime of the MP analysis.
    model_version : str
        Version identifier of the inference engine.
    deltas : ndarray, optional
        Eigen-gaps or signal excesses computed for the detected spikes.
    bootstrap_distribution : ndarray, optional
        Empirical distribution of the null threshold generated via Bootstrap.
    warnings : tuple of str, default=()
        Capture of analytical warnings generated during the inference process.


    Attributes
    ----------
    config_hash : str
        Deterministic hash encoding the full configuration of the model.
        Enables exact reproducibility and experiment tracking.

    Notes
    -----
    **Immutability & Safety**

    - The object is declared with ``frozen=True`` and ``slots=True``.
    - NumPy arrays are explicitly marked as read-only at the C level.
    - Diagnostics (``deltas``, ``bootstrap_distribution``) are strictly injected
      by the API layer, keeping this class as a passive data container.
    - Internal invariants (``config_hash``) are injected via ``object.__setattr__``,
      following the official dataclass pattern for frozen instances.

    **HPC Design Trade-off**

    - The full empirical spectrum is intentionally omitted to reduce memory usage.
    - Downstream components (e.g., plotting) must reconstruct the spectrum if needed.

    **Invariants**

    - ``shrinkage_method`` must be provided when using shrinkage covariance.
    - ``bootstrap_samples`` must be provided for bootstrap inference.

    This class is the canonical output of the MP pipeline and should be treated
    as a read-only artifact.
    """

    # ==================================================
    # Dimentions
    # ==================================================
    n: int
    p: int
    q: float

    # ==================================================
    # Configurations (Enums)
    # ==================================================
    covariance_method: CovarianceMethod
    threshold_method: ThresholdMethod
    mp_sigma_estimator: MPSigmaEstimator
    shrinkage_method: Optional[ShrinkageMethod]

    alpha: Optional[float]
    bootstrap_samples: Optional[int]

    # ==================================================
    # Reproducibility
    # ==================================================
    standardize_data: bool
    random_state: Optional[int]

    # ==================================================
    # MP fit
    # ==================================================
    sigma2_hat: float
    lambda_minus: float
    lambda_plus: float
    bulk_width: float

    # ==================================================
    # Spikes (HPC: solo lo necesario)
    # ==================================================
    spike_threshold: float
    spikes_sample: NDArray[np.float64]
    population_eigenvalues: NDArray[np.float64]

    # ==================================================
    # Metrics
    # ==================================================
    k_effective: int
    snr: float
    ratio_apex: float
    zero_mass: float
    regime: str

    # ==================================================
    # Metadata (MLOps)
    # ==================================================
    timestamp: datetime
    execution_time_sec: float
    model_version: str
    robust_fallback_used: bool = False

    # ==================================================
    # Optional fields injected by the orchestrator
    # ==================================================
    deltas: Optional[NDArray[np.float64]] = None
    bootstrap_distribution: Optional[NDArray[np.float64]] = None
    warnings: Tuple[str, ...] = ()

    # ==================================================
    # DERIVED FIELDS (Auto-calculated, hidden from init)
    # ==================================================
    # Explicit field(init=False) to protect internal invariants.
    config_hash: str = field(default="", init=False)

    # ==================================================
    # POST INIT
    # ==================================================
    def __post_init__(self) -> None:
        # --------------------------------------------------
        # 1. Array sealing (C-level immutability)
        # --------------------------------------------------
        for name in (
            "spikes_sample",
            "population_eigenvalues",
            "deltas",
            "bootstrap_distribution",
        ):
            if hasattr(self, name):
                arr = getattr(self, name)
                if isinstance(arr, np.ndarray):
                    arr.setflags(write=False)

        # --------------------------------------------------
        # 2. Architectural invariants
        # --------------------------------------------------
        if self.covariance_method is CovarianceMethod.SHRINKAGE:
            if self.shrinkage_method is None:
                raise RuntimeError(
                    "Invariant violation: shrinkage_method must not be None "
                    "when covariance_method is SHRINKAGE."
                )

        if self.threshold_method is ThresholdMethod.BOOTSTRAP:
            if self.bootstrap_samples is None:
                raise RuntimeError(
                    "Invariant violation: bootstrap_samples must not be None "
                    "when threshold_method is BOOTSTRAP."
                )

        # --------------------------------------------------
        # 3. Hash configuration (Cryptographic reproducibility in MLOps)
        # --------------------------------------------------
        config_dict = {
            "n": self.n,
            "p": self.p,
            "q": self.q,
            "covariance_method": self.covariance_method.value,
            "threshold_method": self.threshold_method.value,
            "mp_sigma_estimator": self.mp_sigma_estimator.value,
            "shrinkage_method": (self.shrinkage_method.value if self.shrinkage_method else None),
            "alpha": self.alpha,
            "bootstrap_samples": self.bootstrap_samples,
            "standardize_data": self.standardize_data,
            "random_state": self.random_state,
            "model_version": self.model_version,
            "robust_fallback_used": self.robust_fallback_used,
        }

        config_str = json.dumps(config_dict, sort_keys=True)
        hash_val = hashlib.sha256(config_str.encode()).hexdigest()[:12]

        # Note: object.__setattr__ is required due to frozen=True.
        # This is the standard dataclass pattern for derived attributes in __post_init__.
        object.__setattr__(self, "config_hash", hash_val)

    # ==================================================
    # PUBLIC API
    # ==================================================
    def summary(self) -> Summary:
        r"""
        Generate a human-readable summary of the MP analysis.

        Returns
        -------
        Summary
            A formatted summary object providing textual and HTML
            representations of the result, including diagnostics,
            inference details, and warnings.

        Notes
        -----
        This method does not modify the underlying result and serves purely
        as a presentation layer abstraction.
        """

        return Summary(self)

    # ==================================================
    # MAGIC METHODS
    # ==================================================
    def __repr__(self) -> str:
        r"""
        Compact string representation of the result object.

        Returns
        -------
        str
            Summary string including dimensions, effective rank,
            and covariance method.

        Notes
        -----
        Intended for quick inspection in interactive sessions.
        """

        return (
            f"MPResult(n={self.n}, p={self.p}, q={self.q:.4f}, "
            f"k={self.k_effective}, method={self.covariance_method.name})"
        )
