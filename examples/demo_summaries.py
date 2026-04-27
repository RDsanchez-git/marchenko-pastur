import numpy as np

from marchenko_pastur.api import run_mp
from marchenko_pastur.pipelines.defactored_mp import defactored_mp


def generate_factor_model(
        n: int,
        p: int,
        k: int,
        strength: float = 5.0,
        noise_std: float = 1.0,
        seed: int = 42
        ) -> np.ndarray:

    rng = np.random.default_rng(seed)
    E = rng.standard_normal((n, p)) * noise_std
    if k == 0:
        return E
    F = rng.standard_normal((n, k))
    Lambda = rng.standard_normal((k, p)) * strength
    return F @ Lambda + E

def run_scenario(name: str, X: np.ndarray, use_defactored: bool = False, **kwargs):
    print("\n" + "=" * 80)
    print(f"SCENARIO: {name}")
    print("=" * 80)
    try:
        if use_defactored:
            result = defactored_mp(X, **kwargs)
        else:
            result = run_mp(X, **kwargs)
        print(result.summary().as_text())
    except Exception as e:
        print(f"❌ ERROR ESPERADO/INESPERADO: {str(e)}")

if __name__ == "__main__":
    # Topologías SOTA
    X_low_dim = generate_factor_model(n=1000, p=50, k=3)
    X_mod_dim = generate_factor_model(n=500, p=350, k=2)
    X_singular = generate_factor_model(n=150, p=400, k=2)
    X_noise = generate_factor_model(n=500, p=100, k=0)

    run_scenario(
        "1. BASELINE [Pearson + MP Threshold + Median + Unstandardized]",
        X_low_dim,
        covariance="classical",
        threshold="mp",
        standardize_data=False,
        mp_sigma_estimator="median",
    )

    run_scenario(
        "2. THE ROBUST [Shrinkage LW + Tracy-Widom (a=0.01) + Iterative + Std]",
        X_mod_dim,
        covariance="shrinkage",
        shrinkage_method="lw",
        threshold="tw",
        alpha=0.01,
        standardize_data=True,
        mp_sigma_estimator="iterative",
    )

    run_scenario(
        "3. PURE NOISE [Shrinkage OAS + Bootstrap (B=50) + Auto]",
        X_noise,
        covariance="shrinkage",
        shrinkage_method="oas",
        threshold="bootstrap",
        bootstrap_samples=50,
        alpha=0.05,
        standardize_data=True,
        mp_sigma_estimator="auto",
    )

    run_scenario(
        "4. SINGULAR REGIME [Shrinkage OAS + TW + p > n]",
        X_singular,
        covariance="shrinkage",
        shrinkage_method="oas",
        threshold="tw",
        alpha=0.01,
        standardize_data=True,
        mp_sigma_estimator="auto",
    )

    run_scenario(
        "5. THE CURE [Defactored MP on Moderate Regime]",
        X_mod_dim,
        use_defactored=True,
        threshold="tw",
        alpha=0.01,
        max_factors=10,
        verbose=False,
    )
