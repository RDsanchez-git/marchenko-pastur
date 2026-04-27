import matplotlib.pyplot as plt
import numpy as np

from marchenko_pastur.api import run_mp
from marchenko_pastur.visualization.mp_pipeline_plot import plot_mp_defactored
from marchenko_pastur.visualization.plot_spectral_fit import plot_spectral_fit


def generate_factor_model(n=1000, p=200, k=3, signal_strength=3.0, seed=42):
    rng = np.random.default_rng(seed)
    F = rng.standard_normal((n, k))
    Lambda = rng.standard_normal((p, k)) * signal_strength
    noise = rng.standard_normal((n, p))
    return F @ Lambda.T + noise

if __name__ == "__main__":
    print("🔬 Renderizando Galería Visual MP...")

    X_noise = np.random.default_rng(42).standard_normal((1000, 200))
    X_factors = generate_factor_model()

    res_noise = run_mp(X_noise, covariance="pearson", standardize_data=True)
    # SOTA FIX: RMT alignment
    eig_noise = np.linalg.eigvalsh(np.cov(X_noise, rowvar=False, ddof=0))

    # 1. PDF Baseline
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    fig1.suptitle("MP Spectral PDF (Pure Noise)")
    plot_spectral_fit(eig_noise, res_noise, ax=ax1)

    # 2. Defactored Scree con Rompe-ejes
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    fig2.suptitle("Comparativa Defactored (Variance Trap vs POET-style)")

    ax_raw = plot_mp_defactored(X_factors, defactor=False, plot_type="scree", ax=axes2[0])
    target_ax_raw = ax_raw[0] if isinstance(ax_raw, list) else ax_raw
    target_ax_raw.set_title("Raw (Varianza colapsada)")

    ax_clean = plot_mp_defactored(X_factors, defactor=True, plot_type="scree", ax=axes2[1])
    target_ax_clean = ax_clean[0] if isinstance(ax_clean, list) else ax_clean
    target_ax_clean.set_title("Defactored (Eje corregido)")

    plt.tight_layout()
    plt.show()
