# Experimentation Guidelines

This document defines the **official standards** for designing, naming, and executing experiments in the project.

The goal is to ensure:
- Reproducibility
- Traceability
- Scalability
- Zero contamination between research and production code

However, it is natural that each experiment will require its own configuration or selection of sections to display
(especially in .ipynb notebooks), so within this guide, you should select the sections to display that best convey 
the results of the experiments under test.

There may be sections that were not taken into account; we are always open to recommendations.

---

# 🔵 1. ARCHITECTURAL PRINCIPLE

## Separation of Concerns

| Layer         | Responsibility                           |
|--------------|------------------------------------------|
| `src/`       | Core library (production code)           |
| `tests/`     | Deterministic validation (CI/CD)         |
| `experiments/` | Research & exploration (R&D only)     |

### Hard Rule
- `experiments/` **must NOT import from `tests/`**
- `tests/` **must NOT depend on `experiments/`**

---

# 🔵 2. DIRECTORY STRUCTURE

```text
experiments/
├── scripts/
│   └── EXP_XXX_*.py
│
└── notebooks/
    └── EXP_XXX_*_report.ipynb
```

---

# 🔵 3. NAMING CONVENTION (MANDATORY)

## Format

```text
EXP_{ID}_{ENGINE}_{TOPIC}_{DETAIL}
```

## Components

### 1. ID (Sequential)

```text
EXP_001
EXP_014
EXP_032
```
- Unique
- Never reused
- Acts as global anchor

### 2. ENGINE (System component)

| Code | Meaning |
|------|--------|
| MP   | Marchenko-Pastur |
| DEF  | Defactored pipeline |
| TYL  | Tyler estimator |
| PIPE | Full pipeline |
| PLOT | Visualization |

### 3. TOPIC (Phenomenon)

```text
H0
SPECTRUM
SPIKES
SIGMA
THRESHOLD
STABILITY
```

### 4. DETAIL (Optional but recommended)

```text
MONTECARLO
FINITE_SAMPLE
HIGH_DIM
COMPARISON
HEAVY_TAILS
```

## ✅ Valid Examples

```text
EXP_001_MP_H0_MONTECARLO
EXP_014_TYL_SPECTRUM_COMPARISON
EXP_017_DEF_STABILITY_HIGH_DIM
```

## ❌ Invalid Examples

```text
test.py
experiment_final.py
analysis_new.py
```

---

# 🔵 Naming Consistency Rule

Script:
EXP_001_MP_H0_MONTECARLO.py

Notebook:
EXP_001_MP_H0_MONTECARLO_report.ipynb

# 🔵 3.1 FUNCTION NAMING (MANDATORY)

The main function inside each experiment script must match the file name.

## Rule

File:
EXP_001_MP_H0_MONTECARLO.py

Function:
exp_001_mp_h0_montecarlo()

## Constraints

- Must be lowercase
- Must match file name exactly
- Must be the primary entry point of the experiment

---

# 🔵 4. SCRIPT TEMPLATE (MANDATORY)

📁 `experiments/scripts/EXP_XXX_*.py`

## Rules
- Pure computation ONLY
- No plotting
- No UI logic
- Deterministic output
- Structured return

## Standard Contract

```python
{
    "results": {...},
    "meta": {...}
}
```

## Template

```python
import importlib.metadata
import numpy as np
import time
from marchenko_pastur.api import run_mp

"""
EXPERIMENT {ID}: {TITLE}
---------------------------------------------

Objective:
---------
Describe what is being tested.

Methodology:
------------
Explain how the experiment is constructed.

Key Findings:
-------------
Summarize expected or validated results.

Interpretation:
---------------
Explain why the result matters theoretically.
"""

def exp_id_engine_topic_detail(n=500, p=300, M=100, seed=42):
    """
    Executes the experiment pipeline and aggregates the metrics.

    Parameters
    ----------
    n : int
        Number of observations.
    p : int
        Number of features.
    M : int
        Number of Monte Carlo iterations.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Structured output containing:
        - 'results': Computed metrics and statistical aggregations.
        - 'meta': Traceability configuration and model version.
    """
    try:
        model_version = importlib.metadata.version("marchenko_pastur")
    except importlib.metadata.PackageNotFoundError:
        model_version = "unknown"

    rng = np.random.default_rng(seed)
    metrics = []
    
    start_time = time.time()

    for i in range(M):
        X = rng.normal(size=(n, p))
        result = run_mp(X)
        metrics.append(result.k_effective)
    
    elapsed = time.time() - start_time

    return {
        "results": {
            "mean_metric": float(np.mean(metrics)),
            "std_metric": float(np.std(metrics)),
        },
        "meta": {
            "n": n,
            "p": p,
            "M": M,
            "seed": seed,
            "execution_time_seconds": round(elapsed, 2),
            "model_version": model_version,
        }
    }
```

---

# 🔵 5. NOTEBOOK TEMPLATE (MANDATORY)

📁 `experiments/notebooks/EXP_XXX_*_report.ipynb`

## Cell 1 — Environment Setup

```python
# ==========================================
# ENVIRONMENT SETUP
# ==========================================
%load_ext autoreload
%autoreload 2

import sys
from pathlib import Path

def find_project_root(markers=(".git", "pyproject.toml", "src")):
    current = Path.cwd()
    for parent in [current] + list(current.parents):
        if any((parent / marker).exists() for marker in markers):
            return parent
        raise RuntimeError("Project root not found.")

project_root = str(find_project_root())

if project_root not in sys.path:
    sys.path.insert(0, project_root)
```

## Cell 2 — Import Experiment

```python
# ==========================================
# IMPORTS
# ==========================================
from experiments.scripts.EXP_XXX import exp_id_engine_topic_detail
```

## Cell 3 — Configuration

```python
# ==========================================
# CONFIGURATION
# ==========================================
CONFIG = {
    "n": 500,
    "p": 300,
    "M": 200,
    "seed": 42,
}
```

## Cell 4 — Execution

```python
# ==========================================
# RUN EXPERIMENT
# ==========================================
output = exp_id_engine_topic_detail(**CONFIG)

results = output["results"]
meta = output["meta"]
```

## Cell 5 — Results

```python
# ==========================================
# RESULTS
# ==========================================
print("=== RESULTS ===")
for k, v in results.items():
    print(f"{k}: {v:.4f}")
```

## Cell 6 — Metadata

```python
# ==========================================
# METADATA
# ==========================================
print("=== META ===")
for k, v in meta.items():
    print(f"{k}: {v}")
```

## Cell 7 — Interpretation (REQUIRED)

```markdown
Interpretation:
---------------
Provide a high-level scientific synthesis of the empirical findings. Do not merely 
restate the metrics; explain their theoretical and architectural implications. 

Construct a cohesive narrative addressing the following dimensions:
1. Theoretical Alignment: Convergence or divergence from baseline RMT expectations.
2. Finite-Sample Mechanics: Presence of empirical bias, spectral leakage, or structural deformation.
3. Pipeline Justification: How this specific result validates or restricts the current algorithmic design 
   in production.
```

---

## 🔵 5.1 OPTIONAL — TABLE REPRESENTATION

Use this when comparing multiple configurations or regimes.

```python
# ==========================================
# DIAGNOSTICS / TABLE
# ==========================================
import pandas as pd

df = pd.DataFrame([results])
df
```

Use Cases:
- Monte Carlo summaries
- Parameter sweeps (n, p, q)
- Model comparisons

## 🔵 5.2 OPTIONAL — VISUALIZATION

ONLY in notebooks — NEVER in scripts

- Example: Line Plot (e.g., Phase Transition)

```python
# ==========================================
# VISUALIZATION
# ==========================================
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.plot(x_values, y_values, marker="o")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Experiment Visualization")
plt.grid(alpha=0.3)
plt.show()
```

- Example: Histogram (Eigenvalues / Spectrum)

```python
plt.hist(values, bins=30)
plt.title("Spectral Distribution")
plt.show()
```

- When to Use Plots

| Scenario                  | Plot Needed |
|--------------------------|------------|
| Monte Carlo summary      | ❌ Optional |
| Scalar metrics           | ❌ No       |
| Spectrum analysis        | ✅ Yes      |
| Phase transitions (BBP)  | ✅ Critical |
| Model comparison         | ✅ Recommended |

## 🔵 5.3 OPTIONAL — SAVE OUTPUTS

Use only if results need persistence. Leverage `project_root` to ensure paths are absolute and indestructible.

```python
# ==========================================
# SAVE OUTPUTS
# ==========================================
# Create absolute path using the root anchor
output_dir = Path(project_root) / "experiments" / "outputs"
output_dir.mkdir(exist_ok=True) # Failsafe

# Save plot
plt.savefig(output_dir / "EXP_XXX_plot.png", dpi=300)

# Save data
df.to_csv(output_dir / "EXP_XXX_results.csv", index=False)
```

- Output Directory Convention

```text
experiments/
├── scripts/
├── notebooks/
└── outputs/
    └── EXP_XXX_*.png / .csv
```

- Outputs are optional artifacts
- Do NOT commit large outputs to git
- Use `.gitignore` for heavy files

---

# FINAL RULE

> Scripts generate truth.  
> Notebooks explain truth.

- Scripts → numbers  
- Notebooks → insight  
- Outputs → optional artifacts  

---

# 🔵 6. ROOT PATH RESOLUTION (SOTA)

## Problem

`experiments/` is NOT an installed package.

## Solution

Dynamic root detection using `pathlib`.

```python
def find_project_root(markers=(".git", "pyproject.toml", "src")):
    current = Path.cwd()
    for parent in [current] + list(current.parents):
        if any((parent / marker).exists() for marker in markers):
            return parent
        raise RuntimeError("Project root not found.")
```

## ❌ Forbidden

```python
sys.path.append("../../../")
```

---

# 🔵 7. DESIGN PRINCIPLES

## 1. Reproducibility
- Always include `seed`
- Always include `model_version`

## 2. Determinism
- Same inputs → same outputs

## 3. Traceability
- Separate clearly: `results` and `meta`

## 4. No Hidden State
- No global mutable variables
- No implicit randomness

## 5. One Experiment = One Hypothesis
- Avoid mixing multiple objectives in a single script.

---

# 🔵 8. ANTI-PATTERNS (STRICTLY FORBIDDEN)

## ❌ Mixing UI with computation
```python
plt.plot(...)
```

## ❌ Returning raw objects
```python
return result
```

## ❌ Hardcoded globals
```python
N = 500
```

## ❌ Importing from tests
```python
from tests.utils import ...
```

---

# 🔵 9. FUTURE EXTENSIONS

When scaling:
- JSON logging
- CSV aggregation
- Experiment registry
- Dashboard (Streamlit)
- Lightweight tracking system

---

# 🏁 FINAL PRINCIPLE

> Experiments are not scripts — they are scientific assets.

They must be:
- Reproducible
- Auditable
- Comparable
- Stable over time