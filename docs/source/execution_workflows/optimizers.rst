Optimizers
==========

This guide covers behaviour shared across optimizers: where gradients come from,
how to pick a metric estimator, which programs constrain the choice, early
stopping, and reading results. Each optimizer's algorithm, parameters and
references are documented on its class in the
:doc:`optimizer API reference <../api_reference/qprog/optimizers>`. SciPy
methods are selected with :class:`~divi.qprog.optimizers.ScipyMethod`, and
CMA-ES and Differential Evolution with
:class:`~divi.qprog.optimizers.PymooMethod`.

.. _optimizer-gradient-source:

Gradient Source
---------------

Gradient-based optimizers receive the program's gradient from the variational
algorithm.

On a backend that estimates expectation values natively, without a noise model,
and offers adjoint differentiation (:attr:`~divi.backends.CircuitRunner.supports_vjp`),
Divi computes the gradient that way: one submission per gradient, however
many parameters the circuit has. In a :class:`~divi.qprog.ensembles.ProgramEnsemble`
these submissions go straight to the backend rather than into merged batches, and
the progress display shows no batch row for them.

Otherwise Divi uses the parameter-shift rule, which evaluates the function at
shifted parameter values, all in one batch.

.. note::

   :class:`~divi.qprog.algorithms.QAOA` generally has no exact *two-term*
   parameter-shift rule because a shared layer angle drives many Hamiltonian
   terms. For exact trotterization, Divi derives a generalised rule from the
   circuit's gate frequencies [#wierichs2022]_. Adjoint differentiation needs
   no shift rule, so on a Maestro statevector QAOA also gets exact gradients
   for Hamiltonians whose frequencies admit none.
   Stochastic or approximate trotterization still requires SPSA or a
   gradient-free method.

Troubleshooting: on small or barren-plateau circuits the gradient can be
near-zero at the start, so L-BFGS-B may hit its ``gtol`` and stop after a couple
of iterations, leaving a flat, uninformative loss trajectory.

Choosing a Metric Estimator
---------------------------

Optimizers that precondition their gradient with a metric take a
:class:`~divi.qprog.optimizers.metrics.MetricEstimator`, and the estimator
decides which programs they accept. The trade-off is between what the metric
depends on and what it costs. The pullback metric
(:class:`~divi.qprog.optimizers.metrics.PullbackMetricEstimator`) follows the
loss and reuses the gradient measurement pass, but needs a loss built from
expectation values. The Fubini–Study metric
(:class:`~divi.qprog.optimizers.metrics.FubiniStudyMetricEstimator`) depends
only on the ansatz, but needs Pauli-rotation gates with bare parameter angles.
The stochastic-fidelity metric
(:class:`~divi.qprog.optimizers.metrics.StochasticFidelityMetricEstimator`)
also depends only on the ansatz, accepts any invertible ansatz, and costs a
constant number of evaluations per step, at the price of a noisy estimate.
Each optimizer's class documentation names its default.

Program-Specific Constraints
----------------------------

- A parameter in an unsupported gate or a non-linear angle, or with
  incommensurate frequencies, has no parameter-shift rule; its program needs
  adjoint differentiation, SPSA or a gradient-free method. QAOA has a rule only
  under exact trotterization. An optimizer that needs the parameter-shift
  gradient raises :exc:`NotImplementedError` for a program without one.
- A loss computed from raw measurement counts has no gradient, so it needs a
  gradient-free optimizer.
- Checkpointing requires an optimizer whose ``supports_checkpointing`` is
  ``True``. See :doc:`resuming_long_runs`.

.. _rosalin-optimizer:

ROSALIN Compatibility
---------------------

:class:`~divi.qprog.optimizers.RosalinOptimizer` needs an expectation-valued VQA
on a sampling backend, with ``shot_distribution`` unset or ``"weighted_random"``:
either way each gradient component draws its own operator samples.
``grouping_strategy="shadows"`` also works, drawing classical-shadow snapshots
instead (see :ref:`measuring-observables`). Another ``shot_distribution``,
``grouping_strategy="derandomised_shadows"`` or a backend that computes
expectation values itself raises :class:`~divi.pipeline.ContractViolation` from
``run()`` and :meth:`~divi.qprog.QuantumProgram.dry_run`, as does a
data-binding loss other than a plain mean or sum of predictions.

Early Stopping
--------------

Divi's :class:`~divi.qprog.EarlyStopping` controller terminates
the loop automatically based on configurable criteria.

Pass an ``EarlyStopping`` instance to any variational algorithm:

.. invisible-code-block: python

   from pyscf import gto
   from divi.backends import MaestroSimulator
   from divi.qprog.problems import MolecularProblem

   backend = MaestroSimulator(shots=1000)
   problem = MolecularProblem.from_molecule(
       gto.M(atom="H 0 0 -0.6614; H 0 0 0.6614", basis="sto-3g", unit="Bohr")
   )

.. code-block:: python

   from divi.qprog.algorithms import VQE
   from divi.qprog import EarlyStopping
   from divi.qprog.optimizers import ScipyOptimizer, ScipyMethod

   vqe = VQE(
       problem,
       backend=backend,
       optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
       max_iterations=200,
       early_stopping=EarlyStopping(
           patience=10,
           min_delta=1e-5,
       ),
   )

   vqe.run()
   print(f"Stopped at iteration {vqe.current_iteration}")
   print(f"Reason: {vqe.stop_reason}")        # e.g. "patience_exceeded"
   print(f"Converged: {vqe.optimize_result.success}")  # False for early stop

:class:`~divi.qprog.EarlyStopping` documents the three criteria (patience, gradient
norm and cost variance) and their priority order. All three can be enabled
together; the first one that triggers stops the loop.

``grad_norm_threshold`` applies to optimizers that compute the program's
gradient or report one in their callback ``jac``; it has no effect otherwise. It
can stop a converged run before an optimizer with an adaptive sampling budget
spends that budget. The patience and variance criteria read the loss the
optimizer records, which an optimizer's class documentation states and which
need not be the exact cost at the iterate.

After the Run
^^^^^^^^^^^^^

After ``run()`` completes,
:attr:`~divi.qprog.VariationalQuantumAlgorithm.stop_reason` holds the
:class:`~divi.qprog.StopReason` that ended the loop, or ``None`` when it ran to
``max_iterations``. The ``optimize_result`` attribute
is always populated and its ``message`` field includes the stop reason.

Inspecting Optimizer Results
----------------------------

After running a variational algorithm, the loss history is available directly
on the program object via
:attr:`~divi.qprog.VariationalQuantumAlgorithm.min_losses_per_iteration` and
:attr:`~divi.qprog.VariationalQuantumAlgorithm.losses_history` — see
:ref:`reading results <reading-results>` in core concepts for their
semantics and types.

Beyond the loss history, you can inspect the raw result object returned by the
underlying optimizer
via the ``optimize_result`` attribute.
This exposes optimizer-specific diagnostics such as:

- ``nfev`` – number of cost-function evaluations
- ``njev`` – number of Jacobian (gradient) evaluations (gradient-based optimizers)
- ``nit`` – number of iterations completed
- ``success`` – whether the optimizer converged
- ``message`` – convergence or termination message

.. skip: next

.. code-block:: python

   program.run()

   result = program.optimize_result
   if result is not None:
       print(f"Function evaluations: {result.nfev}")
       print(f"Converged: {result.success}")

Next Steps
----------

- `tutorials/ <https://github.com/QoroQuantum/divi/tree/main/tutorials>`_ — runnable examples
- :doc:`../algorithms/ground_state_energy_estimation_vqe` and :doc:`../algorithms/combinatorial_optimization_qaoa_pce` — algorithm-specific guidance
- :doc:`program_ensembles` — optimizers in large-scale sweeps and ensembles

References
----------

.. [#wierichs2022] Wierichs, D., Izaac, J., Wang, C., & Lin, C. Y.-Y. (2022). `General parameter-shift rules for quantum gradients <https://doi.org/10.22331/q-2022-03-30-677>`_. *Quantum*, 6, 677.
