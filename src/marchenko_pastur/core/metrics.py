r"""
Internal spectral diagnostics for Random Matrix Theory analysis.

This module provides utility functions used within the spectral
analysis pipeline to summarize properties of the empirical eigenvalue
spectrum.

The metrics implemented here quantify:

- The number of detected spikes
- The relative strength of dominant eigenvalues
- The signal-to-noise structure of the spectrum

These functions are not part of the public API and are intended
for internal use by higher-level routines such as `run_mp`.

Notes
-----
- All computations assume eigenvalues are precomputed and sorted.
- Metrics are purely descriptive and do not perform statistical inference.
- Interpretation of these metrics depends on the validity of the
  Marchenko-Pastur model assumptions.

See Also
--------
run_mp : Full spectral analysis pipeline.
bbp_population_eigenvalue : Population eigenvalue estimation.
"""

# ======================================================================
# THIRD-PARTY IMPORTS
# ======================================================================

import numpy as np

# ======================================================================
# PRIVATE HELPERS
# ======================================================================


def _compute_metrics(
    eigenvalues: np.ndarray,
    lambda_plus: float,
    spike_threshold: float,
) -> tuple[int, float, float, np.ndarray]:
    r"""
    Compute diagnostic spectral metrics from eigenvalues.

    This internal utility summarizes the empirical eigenvalue spectrum
    by quantifying the number, strength, and relative magnitude of
    detected spikes above a given threshold.

    Parameters
    ----------
    eigenvalues : ndarray
        Eigenvalues of the empirical covariance matrix,
        assumed to be sorted in ascending order.

    lambda_plus : float
        Theoretical upper edge of the Marchenko-Pastur bulk.

    spike_threshold : float
        Threshold used to classify eigenvalues as signal components.

    Returns
    -------
    k_effective : int
        Number of eigenvalues classified as signal (spikes).

    ratio_apex : float
        Ratio between the largest eigenvalue and the theoretical
        Marchenko-Pastur upper edge.

    snr : float
        Spectral signal-to-noise ratio.

    deltas : ndarray
        Excess of each spike above the detection threshold.

    Notes
    -----
    The spectral signal-to-noise ratio (SNR) is defined as:

    .. math::
        \text{SNR} = \frac{\sum_i (\lambda_i - \lambda_{\text{thr}})}{\sum \lambda_{\text{noise}}}

    where:

    - :math:`\lambda_i` are eigenvalues above the threshold
    - :math:`\lambda_{\text{noise}}` are eigenvalues within the bulk

    Edge cases:

    - If no spikes are detected → SNR = 0
    - If noise energy is zero → SNR = 0 (numerical safeguard)
    - If :math:`\lambda_+ = 0` → ratio_apex = 0 (degenerate case)
    """

    # ---------------------------------------------------------------
    # Spike classification
    # ---------------------------------------------------------------

    spike_mask = eigenvalues > spike_threshold
    spikes = eigenvalues[spike_mask]

    k_effective = spikes.size

    # Eigenvalues are assumed to be sorted
    lambda_max = eigenvalues[-1]

    # Degenerate case protection
    ratio_apex = lambda_max / lambda_plus if lambda_plus > 0 else 0.0

    deltas = spikes - spike_threshold

    if k_effective == 0:
        snr = 0.0

    else:
        signal_energy = np.sum(deltas)
        noise_energy = np.sum(eigenvalues[~spike_mask])

        # Numerical safeguard against pure zero-noise bulk
        snr = signal_energy / noise_energy if noise_energy > 0 else 0.0

    return k_effective, ratio_apex, snr, deltas
