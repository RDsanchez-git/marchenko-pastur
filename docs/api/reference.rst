API Reference
=============

Main Interface
--------------
High-level entry points for spectral analysis.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   marchenko_pastur.run_mp

Results & Diagnostics
---------------------
Immutable result containers and summary interfaces.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   marchenko_pastur.results.results.MPResult
   marchenko_pastur.results.summary.Summary

Configuration
-------------
Enums controlling pipeline behavior.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   marchenko_pastur.enums.enums.CovarianceMethod
   marchenko_pastur.enums.enums.ThresholdMethod
   marchenko_pastur.enums.enums.MPSigmaEstimator
   marchenko_pastur.enums.enums.ShrinkageMethod

Visualization
-------------
Plotting utilities for diagnostics.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   marchenko_pastur.visualization.plot_spectral_fit.plot_spectral_fit
   