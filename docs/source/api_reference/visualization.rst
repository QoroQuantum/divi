Visualization
=============

The :mod:`divi.viz` module provides loss-landscape scans, analysis tools, and
plotting utilities for variational programs. See the
:doc:`/user_guide/visualization` guide for usage examples.

Scan Functions
--------------

.. autofunction:: divi.viz.scan_1d

.. autofunction:: divi.viz.scan_2d

.. autofunction:: divi.viz.scan_pca

.. autofunction:: divi.viz.scan_interp_1d

.. autofunction:: divi.viz.scan_interp_2d

Analysis Functions
------------------

.. autofunction:: divi.viz.compute_hessian

.. autofunction:: divi.viz.fourier_analysis_2d

.. autofunction:: divi.viz.run_neb

Enums
-----

.. autoclass:: divi.viz.GradientMethod
   :members:
   :undoc-members:

Utilities
---------

.. autofunction:: divi.viz.periodic_wrap

.. autofunction:: divi.viz.periodic_trajectory_wrap

Result Classes
--------------

.. autoclass:: divi.viz.Scan1DResult
   :members:

.. autoclass:: divi.viz.Scan2DResult
   :members:

.. autoclass:: divi.viz.PCAScanResult
   :members:

.. autoclass:: divi.viz.HessianResult
   :members: top_eigenvectors, bottom_eigenvectors

.. autoclass:: divi.viz.Fourier2DResult
   :members: plot

.. autoclass:: divi.viz.NEBResult
   :members: plot

Convenience Wrapper
-------------------

.. autoclass:: divi.viz.ProgramViz
   :members:
