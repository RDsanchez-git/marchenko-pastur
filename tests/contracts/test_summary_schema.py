import numpy as np

from marchenko_pastur.api import run_mp

# ======================================================
# API PRESENTATION CONTRACTS (Golden Output)
# ======================================================

def test_summary_as_text_contract(rng: np.random.Generator):
    X = rng.standard_normal((300, 100))
    res = run_mp(X, threshold="tw")
    text = res.summary().as_text().lower()

    invariants = [
        "marchenko-pastur spectral analysis",
        "data & configuration",
        "observations (n)",
        "features (p)",
        "q = p/n"
    ]
    for section in invariants:
        assert section in text, f"Regresión UI: Se perdió la sección '{section}'."

def test_summary_bootstrap_section_rendering(rng: np.random.Generator):
    """Ensures dynamic sections (e.g., Bootstrap metadata) render when requested."""
    X = rng.standard_normal((150, 50))
    res = run_mp(X, threshold="bootstrap", bootstrap_samples=10)
    text = res.summary().as_text().lower()

    assert "bootstrap" in text
    assert "samples (b)" in text
