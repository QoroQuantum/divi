Core Concepts
=============

Divi separates what you want to run from how circuits are produced and where
they execute:

.. list-table:: The Divi execution model
   :header-rows: 1
   :widths: 20 48 32

   * - Layer
     - Responsibility
     - Continue with
   * - Program
     - Represents an algorithm or workflow and exposes ``run()`` and results.
     - This page and the algorithm guides
   * - Pipeline
     - Expands specifications into circuit work, executes it, and reduces raw
       results.
     - :doc:`pipelines`
   * - Backend runner
     - Submits circuit batches to a local simulator, cloud simulator, or
       hardware service.
     - :doc:`backends`

This page covers the shared program lifecycle and result model. Specialist
configuration belongs in the linked guides rather than being repeated here.

.. note::
   For complete API documentation of all properties and methods, see :doc:`../api_reference/qprog/index`.

The :class:`~divi.qprog.QuantumProgram` Base Class
--------------------------------------------------

All quantum algorithms in Divi inherit from the abstract base class
:class:`~divi.qprog.QuantumProgram`, which provides the common runtime model for
program execution. In practice, this means coordinating a circuit pipeline
(*expand → execute → reduce*) and handling backend communication through one
consistent interface.

**Core Features:**

- **Pipeline-Oriented Execution** — Structured *expand → execute → reduce* flow
- **Backend Integration** — Unified interface for simulators and hardware
- **Result Handling** — A common structure for aggregating and processing results
- **Error Handling** — Graceful handling of execution failures

**Key Properties:**

- ``total_circuit_count`` - Total circuits executed so far
- ``total_run_time`` - Cumulative execution time in seconds

The :class:`~divi.qprog.VariationalQuantumAlgorithm` Class
----------------------------------------------------------------------------------------

For algorithms that rely on optimising parameters, Divi provides the
:class:`~divi.qprog.VariationalQuantumAlgorithm`
class. This is the base class for algorithms like
:class:`~divi.qprog.algorithms.VQE` and :class:`~divi.qprog.algorithms.QAOA`,
and it extends :class:`~divi.qprog.QuantumProgram` with optimisation logic,
history tracking, and convergence-aware execution on top of the same pipeline
foundation.

Every variational quantum program in Divi follows a consistent lifecycle:

1. **Initialisation** — Set up your problem, ansatz, optimizer, and backend
2. **Expansion** — Generate circuit/evaluation work from the current parameters
3. **Execution** — Run expanded work on the selected backend
4. **Reduction** — Aggregate backend outputs into objective values and metrics
5. **Optimisation Loop** — Update parameters and repeat until stopping criteria are met

.. note::
   Internally, steps 2–5 are orchestrated by a **circuit pipeline** that uses an
   *expand → execute → reduce* pattern. You don't need to interact with the pipeline
   directly when using built-in algorithms, but understanding it enables powerful
   customisation. See :doc:`pipelines` for a deep dive.

``run()`` owns expansion through optimisation. It validates parameter shapes,
drives the optimizer, tracks histories and best/final parameters, applies early
stopping, and optionally checkpoints. Algorithm pages provide complete setup
examples.

.. _reading-results:

**Key Properties:**

The most commonly accessed properties for result analysis:

- ``best_loss`` - The best (lowest) loss value found during optimisation
- ``best_params`` - The parameters that achieved ``best_loss`` (may differ from final parameters)
- ``final_params`` - The parameters from the last optimisation iteration
- :attr:`~divi.qprog.VariationalQuantumAlgorithm.losses_history` - Full loss
  history as ``list[dict]``; each entry maps a parameter-set index to its loss
  for that iteration.  With population-based optimizers
  (:class:`~divi.qprog.optimizers.MonteCarloOptimizer`, CMA-ES) this carries one
  entry per population member per iteration
- :attr:`~divi.qprog.VariationalQuantumAlgorithm.min_losses_per_iteration` -
  Convenience property returning the minimum loss per iteration as
  ``list[float]``; use it to plot the convergence curve

.. note::
   ``best_params`` produced the lowest observed loss; ``final_params`` is the
   last iterate. They differ when the optimizer moves away from its best point.

Reading Results by Algorithm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each algorithm exposes result attributes suited to its output type.  Use this
table to find the right property after calling ``run()``:

.. list-table::
   :header-rows: 1
   :widths: 20 35 45

   * - Algorithm
     - Result attribute(s)
     - Type / notes
   * - :class:`~divi.qprog.algorithms.VQE`
     - ``best_loss`` / ``best_params``
     - ``float`` / ``np.ndarray`` — optimal energy and circuit parameters
   * - :class:`~divi.qprog.algorithms.CustomVQA`
     - ``best_loss`` / ``best_params``
     - ``float`` / ``np.ndarray`` — same as VQE
   * - :class:`~divi.qprog.algorithms.QAOA`
     - ``best_loss`` / ``solution``
     - ``float`` / problem-decoded result (graph partition, QUBO binary vector, etc.)
   * - :class:`~divi.qprog.algorithms.PCE`
     - ``best_loss`` / ``solution``
     - ``float`` / problem-decoded result, as for QAOA
   * - :class:`~divi.qprog.algorithms.QNN`
     - ``best_loss``
     - ``float`` — training loss; use ``.predict(X)`` for inference after training
   * - :class:`~divi.qprog.algorithms.TimeEvolution`
     - ``results``
     - ``dict[str, float]`` (probs) · ``float`` (single observable) · ``list[float]`` (multi-observable)

For QAOA, ``solution`` decodes the highest-probability bitstring via the
problem's decode function (e.g. node indices for graph problems, a NumPy array
for QUBO).  ``solution_bitstring`` gives the raw bitstring.  For multiple
candidates ranked by probability, use
:meth:`~divi.qprog.algorithms.QAOA.get_top_solutions`.

.. _variational-run-controls:

Variational Run Controls and Outputs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For deeper variational workflow details, use these focused guides:

- :doc:`optimizers` for optimizer behaviour and early stopping
- :doc:`resuming_long_runs` for checkpointing and state restore patterns
- :doc:`visualization` for visualising optimisation trajectories, loss
  landscapes, and related diagnostics based on ``losses_history`` and
  ``param_history(...)``
- :doc:`program_ensembles` for multi-run orchestration and sweep-style workflows

**Warm starts.** Pass ``initial_params`` to ``run()``. Match
``get_expected_param_shape()`` when changing optimizer type; population
optimizers require one row per member. Set ``perform_final_computation=False``
when only the trained parameters matter.

.. skip: next

.. code-block:: python

   # One parameter set for a single-set optimizer.
   program.run(
       initial_params=program.best_params.reshape(1, -1),
       perform_final_computation=False,
   )

**Sampling from Pre-Trained Parameters**
   VQE, QAOA, and PCE expose
   :meth:`~divi.qprog.SolutionSamplingMixin.sample_solution`, which runs
   only the final measurement step with a user-supplied parameter set. This is
   the cheapest way to re-sample a circuit when parameters are already known
   (e.g. loaded from a checkpoint or produced by an external training routine).
   Unlike ``run()``, it does not dispatch any expectation-value jobs and does
   not mutate optimizer-side state (``best_params``, ``losses_history``,
   ``current_iteration``).

   Call ``program.sample_solution(program.best_params)`` to skip training and
   perform only the final measurement.

   For the ensemble variant — one call to re-sample every partition of a
   trained :class:`~divi.qprog.workflows.PartitioningProgramEnsemble` —
   see :ref:`ensemble-sample-solution`.

Analysing Solution Distributions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Solution-extracting programs expose
:meth:`~divi.qprog.SolutionSamplingMixin.get_top_solutions`, returning ranked
:class:`~divi.qprog.SolutionEntry` objects with a bitstring, probability, and
optional decoded value. Use ``include_decoded=True`` for problem-domain values
and ``min_prob`` to discard negligible candidates. The QAOA and routing guides
show problem-specific energy, feasibility, and repair workflows.


Circuit Architecture
--------------------

A :class:`~divi.circuits.MetaCircuit` is Divi’s logical circuit representation.
PennyLane, Qiskit, Hamiltonian, and existing-circuit specification stages
produce MetaCircuit batches; the pipeline binds parameters, expands measurement
or mitigation variants, and lowers the final batch to executable OpenQASM. Most
users never construct this representation directly.

Bring an existing PennyLane or Qiskit circuit through
:class:`~divi.qprog.algorithms.CustomVQA`; see :doc:`framework_integration`.
For standalone pipelines, custom specification stages, or other extension work,
see :doc:`pipeline_authoring`.

Backend Abstraction
-------------------

Every execution target implements :class:`~divi.backends.CircuitRunner`.
Programs communicate with that interface through their pipelines, so changing
from a local simulator to a cloud runner does not change algorithm logic. See
:doc:`backends` for backend selection, direct submission, noise, and job
configuration.


Next Steps
----------

- :doc:`../algorithms/ground_state_energy_estimation_vqe` and :doc:`../algorithms/combinatorial_optimization_qaoa_pce` — algorithm-specific guides
- :doc:`framework_integration` — bring your own PennyLane/Qiskit circuit and QNN data binding
- :doc:`backends` — execution environments and results
- :doc:`../api_reference/qprog/index` — custom algorithms and the full API
- `tutorials/ <https://github.com/QoroQuantum/divi/tree/main/tutorials>`_ — runnable walkthroughs
