import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest


@pytest.fixture(autouse=True)
def cleanup_plots():
    """
    Ensures no matplotlib state leaks between tests.
    Prevents memory accumulation and cross-test contamination.
    """
    yield
    plt.close("all")
