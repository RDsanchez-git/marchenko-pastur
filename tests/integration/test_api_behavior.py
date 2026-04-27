import numpy as np
import pytest

from marchenko_pastur.api import run_mp
from marchenko_pastur.enums.enums import ThresholdMethod
from marchenko_pastur.results.results import MPResult


# ==========================================================
# 1. API Output Contracts (Structure & Types)
# ==========================================================
def test_api_returns_mpresult(data_spiked: np.ndarray):
    """The entry point must strictly encapsulate the output in an MPResult."""
    result = run_mp(data_spiked)

    assert isinstance(result, MPResult)
    assert result.k_effective >= 1
    assert result.spikes_sample.size == result.k_effective
    assert result.population_eigenvalues.size == result.k_effective

def test_shapes_consistency(data_spiked: np.ndarray):
    """The dimensions of the spike vectors must match."""
    result = run_mp(data_spiked)
    assert result.spikes_sample.shape == result.population_eigenvalues.shape

# ==========================================================
# 2. Memory Safety (C-Level Immutability)
# ==========================================================
def test_result_immutability(data_spiked: np.ndarray):
    """The result arrays must be locked against writing (read-only)."""
    result = run_mp(data_spiked)

    if result.spikes_sample.size > 0:
        with pytest.raises(ValueError):
            result.spikes_sample[0] = 999.0

# ==========================================================
# 3. Developer Experience (Type Resolution)
# ==========================================================
def test_enum_string_equivalence(data_noise: np.ndarray):
    """The API must treat the string 'tw' exactly the same as ThresholdMethod.TW."""
    res_str = run_mp(data_noise, threshold="tw")
    res_enum = run_mp(data_noise, threshold=ThresholdMethod.TW)

    assert res_str.k_effective == res_enum.k_effective
    assert res_str.spike_threshold == res_enum.spike_threshold
