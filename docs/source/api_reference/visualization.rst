Visualization
=============

The :mod:`divi.viz` module provides loss-landscape scans, analysis tools, and
plotting utilities for variational programs.

Scan functions cover one-dimensional (:func:`~divi.viz.scan_1d`), two-dimensional
(:func:`~divi.viz.scan_2d`), PCA-based (:func:`~divi.viz.scan_pca`), and
interpolation (:func:`~divi.viz.scan_interp_1d`, :func:`~divi.viz.scan_interp_2d`)
slices through the loss landscape. Analysis tools include
:func:`~divi.viz.compute_hessian` for curvature information,
:func:`~divi.viz.fourier_analysis_2d` for frequency decomposition, and
:func:`~divi.viz.run_neb` for nudged elastic band transition paths.
:class:`~divi.viz.ProgramViz` wraps a variational program and exposes all of the
above as convenient methods.

.. automodapi:: divi.viz
   :no-heading:
   :no-inheritance-diagram:
   :no-inherited-members:
   :include-all-objects:
