Visualizing Variational Landscapes
==================================

The :mod:`divi.viz` module provides loss-landscape scans and analysis tools for
variational programs. Use it to inspect the local geometry of an objective
around a chosen parameter vector, compare minima, study curvature, and find
minimum-energy paths between solutions.

Divi's scan API is inspired by `orqviz <https://github.com/zapata-engineering/orqviz>`_
(Zapata Engineering, Apache-2.0; see also `arXiv:2111.04695
<https://arxiv.org/abs/2111.04695>`_), but the implementation is specific to
Divi's program model, batching, and pipeline execution. Beyond orqviz-compatible
scan geometry, ``divi.viz`` adds Hessian eigenvalue analysis, Fourier power
spectra, gradient overlays, 3D rendering, trajectory overlays on PCA scans,
and the Nudged Elastic Band algorithm — all evaluated through Divi's batched
cost pipeline.

Choosing a Scan Type
--------------------

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - I want to...
     - Use
     - Result type
   * - See the landscape around my optimum
     - :func:`~divi.viz.scan_1d` / :func:`~divi.viz.scan_2d`
     - :class:`~divi.viz.Scan1DResult` / :class:`~divi.viz.Scan2DResult`
   * - See the landscape the optimizer explored
     - :func:`~divi.viz.scan_pca`
     - :class:`~divi.viz.PCAScanResult`
   * - Compare two solutions along the connecting line
     - :func:`~divi.viz.scan_interp_1d` / :func:`~divi.viz.scan_interp_2d`
     - :class:`~divi.viz.Scan1DResult` / :class:`~divi.viz.Scan2DResult`
   * - Identify curvature directions at a point
     - :func:`~divi.viz.compute_hessian`
     - :class:`~divi.viz.HessianResult`
   * - Find frequency structure in a landscape
     - :func:`~divi.viz.fourier_analysis_2d`
     - :class:`~divi.viz.Fourier2DResult`
   * - Find the energy barrier between two minima
     - :func:`~divi.viz.run_neb`
     - :class:`~divi.viz.NEBResult`

Every scan function returns a result object with a ``.plot()`` method for
quick visualization. Two-dimensional results also have ``.plot_3d()`` for
surface rendering.

Basic Scans
-----------

The standalone API is useful when you want to make the scan call explicit:

.. code-block:: python

   import numpy as np
   import pennylane as qp

   from divi.backends import MaestroSimulator
   from divi.qprog import GenericLayerAnsatz, VQE
   from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer
   from divi.viz import scan_1d, scan_2d

   hamiltonian = -1.0 * qp.Z(0) + 0.5 * qp.Z(0) @ qp.Z(1)

   vqe = VQE(
       hamiltonian=hamiltonian,
       ansatz=GenericLayerAnsatz([qp.RY, qp.RZ]),
       n_layers=2,
       optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
       max_iterations=12,
       backend=MaestroSimulator(shots=5000),
   )
   vqe.run()

   line = scan_1d(vqe, n_points=31, rng=0)
   plane = scan_2d(vqe, grid_shape=(21, 21), rng=0)

   line.plot(show=True)    # line chart of objective vs offset
   plane.plot(show=True)   # filled contour plot

``line.plot`` opens a matplotlib figure with the scanned offset on the x-axis and
the objective value on the y-axis — useful for spotting barren plateaus or local
minima along a single direction. ``plane.plot`` renders a filled contour plot
over the two scan axes with the optimum at the center, giving a 2-D picture of
the local loss landscape. Both return the underlying :class:`matplotlib.figure.Figure`
and :class:`matplotlib.axes.Axes` so you can customize the rendering.

.. invisible-code-block: python

   program = vqe
   theta_1 = np.asarray(vqe.best_params, dtype=float)
   theta_2 = theta_1 + 0.05
   other_params = theta_2

If ``center`` is omitted, the scan is centered on ``program.best_params`` from a
previous optimization run.

By default, scalar spans use :math:`(-\pi, \pi)` along each scan axis. Omitted
directions are filled with random vectors (in 2D, the second axis is orthogonal
to the first). Pass an integer or :class:`numpy.random.Generator` as ``rng``
for reproducible slices.

Direction vectors are normalized to unit length by default, so span offsets are
in parameter-space Euclidean distance: ``offset=1.0`` moves exactly 1.0 in
parameter space regardless of the direction's original length. Set
``normalize_directions=False`` to use the raw direction vector, where
``offset=1.0`` moves by the full (unnormalized) direction. This matches the
orqviz convention.

Using ``program.viz``
---------------------

Supported variational programs also expose scan and analysis functions through a
convenience wrapper:

.. code-block:: python

   scan = vqe.viz.scan_2d(grid_shape=(25, 25), rng=0)
   fig, ax = scan.plot(show=True)

The wrapper exposes ``scan_1d``, ``scan_2d``, ``scan_pca``, ``scan_interp_1d``,
``scan_interp_2d``, ``compute_hessian``, and ``run_neb``. Utility functions
like :func:`~divi.viz.periodic_trajectory_wrap` and
:func:`~divi.viz.fourier_analysis_2d` operate on results or arrays and should be
imported directly from :mod:`divi.viz`.

PCA Scans
---------

A random-direction scan shows a slice of the landscape that may miss the
structure the optimizer navigated. PCA scans address this: they build scan
directions from the principal components of the optimization trajectory, showing
the landscape along the directions that captured the most variance during
optimization. With few iterations, best-so-far points can lie almost on a line
in parameter space (common with COBYLA early on), so give the run enough steps
that the trajectory spans two independent directions — as in
`tutorials/viz_qaoa_pce_comparison.py <https://github.com/QoroQuantum/divi/blob/main/tutorials/viz_qaoa_pce_comparison.py>`_.

Use :func:`~divi.viz.scan_pca` with parameter vectors from the optimization
history:

.. code-block:: python

   from divi.viz import scan_pca

   samples = np.vstack(vqe.param_history(mode="best_per_iteration"))
   pca_scan = scan_pca(
       vqe,
       samples=samples,
       center=vqe.best_params,
       grid_shape=(21, 21),
       offset=0.2,   # expand grid 0.2 beyond the sample cloud (default is 1.0)
   )
   pca_scan.plot(show=True)   # cell heatmap with sample dots

``samples`` must have shape ``(n_samples, n_params)`` and span at least two
independent directions. After ``run()``, a typical choice is stacking
``program.param_history(mode="best_per_iteration")`` (one best row per callback;
avoids folding an entire population into PCA) or
``program.param_history(mode="all_evaluated")`` when you want every evaluated
member.

Omit ``center`` to anchor the affine plane at the sample mean (orqviz
default). Pass ``center=program.best_params`` to translate the plane so the
same PC directions pass through the optimum.

Periodic Wrapping for PCA
~~~~~~~~~~~~~~~~~~~~~~~~~

Quantum gate parameters are typically :math:`2\pi`-periodic. When an optimization
trajectory crosses the period boundary, PCA can see an artificial jump and
produce distorted landscapes. Use :func:`~divi.viz.periodic_trajectory_wrap`
before feeding samples to :func:`~divi.viz.scan_pca`:

.. code-block:: python

   from divi.viz import periodic_trajectory_wrap, scan_pca

   raw_samples = np.vstack(vqe.param_history(mode="best_per_iteration"))
   samples = periodic_trajectory_wrap(raw_samples)
   pca_scan = scan_pca(vqe, samples=samples, center=vqe.best_params)

The function iterates forward through the rows, wrapping each row relative to
its predecessor so that no single-step jump exceeds half a period.

Trajectory Overlay
~~~~~~~~~~~~~~~~~~

Pass ``show_trajectory=True`` to :meth:`~divi.viz.PCAScanResult.plot` to draw
the optimization path as a connected line on top of the heatmap:

.. code-block:: python

   pca_scan.plot(show=True, show_trajectory=True)

Start and end points are marked with distinct markers. You can combine
``show_trajectory`` with ``show_samples`` (scatter dots, enabled by default).

Interpolation Scans
-------------------

Use :func:`~divi.viz.scan_interp_1d` to evaluate the objective along the
straight line between two parameter vectors — for example, to check if there is
a barrier between two local minima:

.. code-block:: python

   from divi.viz import scan_interp_1d

   result = scan_interp_1d(program, theta_1=vqe.best_params, theta_2=other_params)
   result.plot(show=True)   # line chart with t on the x-axis

The offsets in the result are the interpolation parameter *t* ranging from 0
(``theta_1``) to 1 (``theta_2``).

:func:`~divi.viz.scan_interp_2d` extends this to 2D: the x-direction is the
interpolation vector and the y-direction is orthogonal (random if omitted):

.. code-block:: python

   from divi.viz import scan_interp_2d

   result = scan_interp_2d(program, theta_1, theta_2, grid_shape=(21, 21))
   result.plot(show=True)   # filled contour plot

Default spans are ``(-0.5, 1.5)`` along x (extending the interpolation line)
and ``(-0.5, 0.5)`` along y.

Hessian Analysis
----------------

:func:`~divi.viz.compute_hessian` computes the matrix of second partial
derivatives at a parameter point. Eigenvalues reveal local curvature;
eigenvectors are natural choices as scan directions.

By default, the **parameter-shift rule** is used (exact for standard quantum
gates). Pass ``gradient_method=GradientMethod.FINITE_DIFFERENCE`` for a
finite-difference fallback with configurable ``eps``:

.. code-block:: python

   from divi.viz import GradientMethod

   # Exact (parameter-shift, default):
   hess = program.viz.compute_hessian()

   # Finite-difference alternative:
   hess = program.viz.compute_hessian(gradient_method=GradientMethod.FINITE_DIFFERENCE, eps=1e-4)

Via ``program.viz``, the center defaults to ``best_params``:

.. code-block:: python

   hess = program.viz.compute_hessian()  # at best_params

   # Eigenvalues are sorted ascending (smallest first).
   print("Eigenvalues:", hess.eigenvalues)
   # Inverse condition number: values near 1 indicate isotropic curvature.
   print("Condition ratio:", hess.eigenvalues[0] / hess.eigenvalues[-1])

   # Use steepest-curvature directions for a 2D scan.
   d1, d2 = hess.top_eigenvectors(k=2)
   scan = program.viz.scan_2d(direction_x=d1, direction_y=d2)
   scan.plot(show=True)

The total number of evaluations is :math:`2n^2 + 1` where *n* is the number of
parameters. For a 2-layer QAOA with 4 parameters per layer (*n* = 8), that is
129 evaluations; for 20 parameters, 801.

Fourier Analysis
----------------

:func:`~divi.viz.fourier_analysis_2d` applies a 2D FFT to any
:class:`~divi.viz.Scan2DResult` or :class:`~divi.viz.PCAScanResult` and
returns the power spectrum:

.. code-block:: python

   from divi.viz import fourier_analysis_2d

   spectrum = fourier_analysis_2d(plane)
   spectrum.plot(show=True)   # heatmap of the power spectrum

The power spectrum reveals dominant frequency components in the loss landscape.

Nudged Elastic Band (NEB)
-------------------------

.. warning::
   NEB is **experimental**. Convergence is sensitive to hyperparameters.

Use NEB when you have two candidate solutions and want to characterize the
energy barrier between them — a low barrier suggests the solutions are
connected, while a high barrier suggests distinct basins.

:func:`~divi.viz.run_neb` finds a minimum-energy path by relaxing a chain of
images via gradient descent perpendicular to the chain tangent. Like
:func:`~divi.viz.compute_hessian`, it accepts a ``gradient_method`` parameter
(defaults to :attr:`~divi.viz.GradientMethod.PARAMETER_SHIFT`):

.. code-block:: python

   from divi.viz import run_neb

   result = run_neb(program, theta_1, theta_2, n_pivots=12, n_steps=50)
   result.plot(show=True)   # energy profile along the relaxed path

Set ``convergence_tol`` to stop early when the maximum pivot displacement falls
below a threshold:

.. code-block:: python

   result = run_neb(program, theta_1, theta_2, convergence_tol=1e-4)

3D Surface Plots
----------------

Both :class:`~divi.viz.Scan2DResult` and :class:`~divi.viz.PCAScanResult` have
a ``plot_3d()`` method for matplotlib 3D surface rendering:

.. code-block:: python

   plane.plot_3d(show=True)

Gradient Overlay
----------------

Pass ``show_gradients=True`` to any 2D ``.plot()`` call to overlay a quiver plot
of the numerical gradient (computed via :func:`numpy.gradient` on the grid, no
extra evaluations):

.. code-block:: python

   plane.plot(show=True, show_gradients=True)

Batching Behavior
-----------------

Scans evaluate all grid points in a single batched call through the program's
existing cost pipeline. This means scan evaluations reuse the same backend,
batching, and circuit-compilation behavior as normal optimization. A 41x41 grid
sends 1,681 parameter sets in one batch, not 1,681 separate evaluations.

Supported Programs
------------------

Scans apply to fixed-parameter-space subclasses of
:class:`~divi.qprog.variational_quantum_algorithm.VariationalQuantumAlgorithm`,
including ``VQE``, ``QAOA``, ``PCE``, and ``CustomVQA``.

Limitations:

- ``IterativeQAOA`` is excluded because its parameter space is not fixed across iterations.
- ``PCE`` preserves the same backend rules as ordinary cost evaluation. In particular, hard CVaR mode still requires a sampling backend.
