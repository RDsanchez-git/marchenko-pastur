marchenko_pastur Documentation
==============================

**High-dimensional spectral analysis and signal detection using Random Matrix Theory.**

`marchenko_pastur` is a specialized Python library designed for economists and data scientists 
dealing with high-dimensional data where the number of features (p) is comparable to the 
number of observations (n).

Quickstart
----------

.. code-block:: python

   import numpy as np
   from marchenko_pastur import run_mp

   # Generate signal-plus-noise data
   X = np.random.randn(300, 100)

   # Run the SOTA analysis pipeline
   result = run_mp(X, threshold="tw")

   # Print detailed diagnostics
   print(result.summary().as_text())

Key Features
------------

* **Robust Estimators:** Tyler's M-estimator and Shrinkage (Ledoit-Wolf, OAS) for heavy-tailed data.
* **Signal Detection:** Marchenko-Pastur bulk fitting and Tracy-Widom thresholding.
* **Bias Correction:** BBP transition-based shrinkage for spiked covariance models.
* **Visual Diagnostics:** Advanced plotting for spectral densities and broken-axis scree plots.

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   api/reference

.. toctree::
   :maxdepth: 1
   :caption: Development

   dev/docstring_guide
   dev/experiment_guidelines
   contributing

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`