Visualizing Variational Landscapes
==================================

The :mod:`divi.viz` module provides loss-landscape scans for variational programs.
It is useful when you want to inspect the local geometry of an objective around a
chosen parameter vector instead of only looking at the optimizer trace.

Divi's scan API is inspired by `orqviz <https://github.com/zapata-engineering/orqviz>`_,
but the implementation is specific to Divi's program model, batching, and
pipeline execution. ``orqviz`` is published by Zapata Engineering under
Apache-2.0.

The module provides:

- :func:`divi.viz.scan_1d` for one-dimensional scans
- :func:`divi.viz.scan_2d` for two-dimensional scans
- :func:`divi.viz.scan_pca` for scans in a PCA-derived 2D plane
- :class:`divi.viz.Scan1DResult`, :class:`divi.viz.Scan2DResult`, and
  :class:`divi.viz.PCAScanResult` result objects
- ``program.viz.scan_1d(...)``, ``program.viz.scan_2d(...)``, and
  ``program.viz.scan_pca(...)`` as thin wrappers

Scanning a Program
------------------

The standalone API is useful when you want to make the scan call explicit:

.. code-block:: python

   import numpy as np
   import pennylane as qml

   from divi.backends import MaestroSimulator
   from divi.qprog import GenericLayerAnsatz, VQE
   from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer
   from divi.viz import scan_1d, scan_2d

   hamiltonian = -1.0 * qml.Z(0) + 0.5 * qml.Z(0) @ qml.Z(1)

   vqe = VQE(
       hamiltonian=hamiltonian,
       ansatz=GenericLayerAnsatz([qml.RY, qml.RZ]),
       n_layers=1,
       optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
       max_iterations=5,
       backend=MaestroSimulator(shots=2000),
   )
   vqe.run()

   line = scan_1d(vqe, n_points=31, rng=0)
   plane = scan_2d(vqe, grid_shape=(21, 21), rng=0)

   line.plot(show=True)
   plane.plot(show=True)

If ``center`` is omitted, the scan is centered on ``program.best_params`` from a
previous optimization run. To scan around a different point, pass ``center``
explicitly.

By default, scalar spans follow the usual ``orqviz`` convention
:math:`(-\\pi, \\pi)` along each scan axis. Omitted directions are filled with
random vectors (in 2D, the second axis is orthogonal to the first). Pass an
integer or :class:`numpy.random.Generator` as ``rng`` for reproducible slices.
Set ``normalize_directions=False`` to match ``orqviz`` line/plane scans where
offset coefficients multiply the raw direction vectors (norm affects Euclidean
step size).

Using ``program.viz``
---------------------

Supported variational programs also expose the same functionality through a
convenience wrapper:

.. code-block:: python

   import numpy as np

   from divi.backends import MaestroSimulator
   from divi.qprog import QAOA
   from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer
   from divi.qprog.problems import BinaryOptimizationProblem

   qaoa = QAOA(
       BinaryOptimizationProblem(np.array([[-1.0, 2.0], [0.0, 1.0]])),
       n_layers=1,
       optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
       max_iterations=5,
       backend=MaestroSimulator(shots=2000),
   )
   qaoa.run()

   scan = qaoa.viz.scan_2d(
       grid_shape=(25, 25),
       span_x=(-0.8, 0.8),
       span_y=(-0.8, 0.8),
       rng=0,
   )
   fig, ax = scan.plot(show=True)

The wrapper does not introduce new semantics; it simply binds the program
instance for a more fluent call style.

PCA Scans
---------

Use :func:`divi.viz.scan_pca` when you already have a representative cloud of
parameter vectors. The layout follows the ``orqviz`` PCA workflow: fit
:class:`sklearn.decomposition.PCA` on ``samples``, build a grid in **PCA score
space** on the selected components (default PC0 vs PC1), map each grid point
with ``inverse_transform`` into full parameter space, then evaluate the
objective. By default, score-axis limits are the min/max of the projected
``samples`` on each component, plus ``offset`` (same rule as ``orqviz``; a
scalar ``a`` becomes padding ``(-|a|, |a|)`` on both ends). Pass ``span_x`` and
``span_y`` together to fix score ranges instead.

Omit ``center`` to anchor the affine plane at the **sample mean** (``orqviz``
default). Pass ``center=program.best_params`` (or any vector) to translate the
plane by ``(center - sample_mean)`` after reconstruction.

.. code-block:: python

   import numpy as np

   from divi.viz import scan_pca

   reference_samples = np.array(
       [
           [-0.4, -0.2],
           [-0.2, 0.3],
           [0.2, -0.1],
           [0.4, 0.5],
       ]
   )

   # Data-driven score limits + optional anchor at the optimum
   pca_scan = scan_pca(
       vqe,
       samples=reference_samples,
       center=vqe.best_params,
       grid_shape=(21, 21),
       offset=0.2,
   )

   # Or fixed score ranges (both spans required)
   # pca_scan = scan_pca(..., span_x=(-0.6, 0.6), span_y=(-0.6, 0.6))

   pca_scan.plot(show=True)

``samples`` must have shape ``(n_samples, n_params)`` and span at least two
independent directions. After ``run()``, a typical choice is stacking
``program.param_history(mode="best_per_iteration")`` (one best row per
callback; avoids folding an entire population into PCA) or
``program.param_history(mode="all_evaluated")`` when you want every evaluated
member. Pass ``numpy.vstack(...)`` to obtain a 2D array for ``samples``.

See ``LICENSES/ORQViz-Apache-2.0-acknowledgement.txt`` for how ``divi.viz``
relates to ``orqviz`` (Zapata Engineering, Apache-2.0), including PCA scans
and line/plane conventions.

Batching Behavior
-----------------

``divi.viz`` does not call ``run()`` internally. Instead, it evaluates the
requested parameter sets through ``_evaluate_cost_param_sets(...)``. This means
that scans reuse the same cost pipeline, backend integration, and batching
behavior as the corresponding program.

Supported Programs
------------------

Scans apply to fixed-parameter-space subclasses of
:class:`~divi.qprog.variational_quantum_algorithm.VariationalQuantumAlgorithm`,
including ``VQE``, ``QAOA``, ``PCE``, and ``CustomVQA``.

Limitations:

- ``IterativeQAOA`` is excluded because its parameter space is not fixed across iterations.
- ``PCE`` preserves the same backend rules as ordinary cost evaluation. In particular, hard CVaR mode still requires a sampling backend.
- ``CustomVQA`` is supported only when its objective can be evaluated through the normal variational cost path.
