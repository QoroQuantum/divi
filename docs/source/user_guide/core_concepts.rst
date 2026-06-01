Core Concepts
=============

This guide explains the fundamental concepts and architecture that make Divi work. Understanding these concepts will help you use Divi more effectively and build custom quantum algorithms.

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

For algorithms that rely on optimizing parameters, Divi provides the
:class:`~divi.qprog.VariationalQuantumAlgorithm`
class. This is the base class for algorithms like
:class:`~divi.qprog.algorithms.VQE` and :class:`~divi.qprog.algorithms.QAOA`,
and it extends :class:`~divi.qprog.QuantumProgram` with optimization logic,
history tracking, and convergence-aware execution on top of the same pipeline
foundation.

Every variational quantum program in Divi follows a consistent lifecycle:

1. **Initialization** — Set up your problem, ansatz, optimizer, and backend
2. **Expansion** — Generate circuit/evaluation work from the current parameters
3. **Execution** — Run expanded work on the selected backend
4. **Reduction** — Aggregate backend outputs into objective values and metrics
5. **Optimization Loop** — Update parameters and repeat until stopping criteria are met

.. note::
   Internally, steps 2–5 are orchestrated by a **circuit pipeline** that uses an
   *expand → execute → reduce* pattern. You don't need to interact with the pipeline
   directly when using built-in algorithms, but understanding it enables powerful
   customization. See :doc:`pipelines` for a deep dive.

Here's how a typical :class:`~divi.qprog.algorithms.VQE` program flows through this lifecycle:

.. code-block:: python

   import numpy as np
   import pennylane as qp
   from divi.qprog import VQE, HartreeFockAnsatz
   from divi.backends import MaestroSimulator
   from divi.qprog.optimizers import ScipyOptimizer, ScipyMethod

   # 1. Initialization - Define your quantum problem
   molecule = qp.qchem.Molecule(
       symbols=["H", "H"],
       coordinates=np.array([[0.0, 0.0, -0.6614], [0.0, 0.0, 0.6614]]),
   )
   vqe = VQE(
       molecule=molecule,           # Your molecular system
       ansatz=HartreeFockAnsatz(),  # Quantum circuit template
       n_layers=2,                  # Circuit depth
       backend=MaestroSimulator(),  # Where to run circuits
       optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),  # Choose optimizer
       seed=42                      # For reproducibility
   )

   # 2-5. Expansion, execution, reduction, and parameter update
   # happen inside run() on each optimization iteration.
   vqe.run()

   print(f"Ground state energy: {vqe.best_loss:.6f}")

**Key Features:**

- **Parameter Handling** — Initializes parameter sets and enforces optimizer-specific shapes
- **Optimizer Integration** — Drives :class:`~divi.qprog.optimizers.Optimizer` instances through a consistent callback loop
- **History Surfaces** — Exposes ``losses_history``, ``param_history(...)``, and ``min_losses_per_iteration`` for analysis and visualization
- **Best-vs-Final Tracking** — Separately stores ``best_params`` / ``best_loss`` and ``final_params`` for robust post-run inspection
- **Early-Stopping Controllers** — Accepts :class:`~divi.qprog.early_stopping.EarlyStopping` and reports :class:`~divi.qprog.early_stopping.StopReason`
- **Checkpoint/Resume Support** — Supports :class:`~divi.qprog.checkpointing.CheckpointConfig` in ``run(...)`` and state recovery via ``load_state(...)`` (see :doc:`resuming_long_runs` and :doc:`../api_reference/qprog/checkpointing`)

**Key Properties:**

The most commonly accessed properties for result analysis:

- ``best_loss`` - The best (lowest) loss value found during optimization
- ``best_params`` - The parameters that achieved ``best_loss`` (may differ from final parameters)
- ``final_params`` - The parameters from the last optimization iteration
- ``min_losses_per_iteration`` - Convenience property returning minimum loss per iteration

.. note::
   **Understanding best vs final parameters:**
   Compare
   :attr:`~divi.qprog.VariationalQuantumAlgorithm.best_params`
   and
   :attr:`~divi.qprog.VariationalQuantumAlgorithm.final_params`.
   During optimization, Divi tracks the best loss value found across all
   iterations.
   :attr:`~divi.qprog.VariationalQuantumAlgorithm.best_params` contains the
   parameters that achieved this best loss, while
   :attr:`~divi.qprog.VariationalQuantumAlgorithm.final_params` contains the
   parameters from the final iteration.
   These may differ if the optimizer explores away from the best solution.
   For full property details, see
   :class:`~divi.qprog.VariationalQuantumAlgorithm`.

Variational Run Controls and Outputs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For deeper variational workflow details, use these focused guides:

- :doc:`optimizers` for optimizer behavior, early stopping, and
  ``run(initial_params=...)`` usage
- :doc:`resuming_long_runs` for checkpointing and state restore patterns
- :doc:`visualization` for visualizing optimization trajectories, loss
  landscapes, and related diagnostics based on ``losses_history`` and
  ``param_history(...)``
- :doc:`program_ensembles` for multi-run orchestration and sweep-style workflows

**Inspecting Run State:**

.. code-block:: python

   # Access execution statistics
   print(f"Circuits executed: {vqe.total_circuit_count}")
   print(f"Total runtime: {vqe.total_run_time:.2f}s")

   # Examine optimization history
   for i, best_loss in enumerate(vqe.min_losses_per_iteration):
       print(f"Iteration {i}: {best_loss:.6f}")

   # Get the best parameters found during optimization
   best_params = vqe.best_params

**Warm-Starting and Pre-Training**
   For warm-starting or pre-training routines where you don't need final solution extraction,
   you can skip the final computation step:

   .. invisible-code-block: python

      vqe = VQE(molecule=molecule, ansatz=HartreeFockAnsatz(), n_layers=2,
                backend=MaestroSimulator(), optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA), seed=42)

   .. code-block:: python

      import numpy as np
      from divi.qprog.optimizers import MonteCarloOptimizer

      # Run optimization without final probability computation
      vqe.run(perform_final_computation=False)

      # Extract best parameters for reuse
      best_params = vqe.best_params  # Shape: (n_params,)

      # Reuse parameters in a new run with the same optimizer configuration
      vqe2 = VQE(molecule=molecule, n_layers=2, backend=MaestroSimulator(),
                 optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA))
      vqe2.run(
          initial_params=best_params.reshape(1, -1),
          perform_final_computation=False,
      )

      # If using a different optimizer, adapt to expected shape
      # For example, MonteCarloOptimizer expects (n_param_sets, n_params)
      optimizer = MonteCarloOptimizer(population_size=10)
      vqe3 = VQE(molecule=molecule, optimizer=optimizer, n_layers=2, backend=MaestroSimulator())
      expected_shape = vqe3.get_expected_param_shape()  # (10, n_params)
      # Replicate best_params to match optimizer's n_param_sets
      adapted_params = np.tile(best_params, (expected_shape[0], 1))
      vqe3.run(
          initial_params=adapted_params,
          perform_final_computation=False,
      )

      # When you already have trained parameters and only need to sample the
      # circuit (extract the solution distribution), use ``sample_solution``
      # to skip the optimization loop entirely — no EXPECTATION job is
      # dispatched, only a single EXECUTE/measurement round:
      vqe4 = VQE(molecule=molecule, n_layers=2, backend=MaestroSimulator(),
                 optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA))
      vqe4.sample_solution(best_params)

**Sampling from Pre-Trained Parameters**
   Any :class:`~divi.qprog.VariationalQuantumAlgorithm` exposes
   :meth:`~divi.qprog.VariationalQuantumAlgorithm.sample_solution`, which runs
   only the final measurement step with a user-supplied parameter set. This is
   the cheapest way to re-sample a circuit when parameters are already known
   (e.g. loaded from a checkpoint or produced by an external training routine).
   Unlike ``run()``, it does not dispatch any expectation-value jobs and does
   not mutate optimizer-side state (``best_params``, ``losses_history``,
   ``current_iteration``).

   .. code-block:: python

      # Skip the training loop entirely — just sample with the known-good
      # parameters and decode the result.
      vqe_sample = VQE(molecule=molecule, n_layers=2, backend=MaestroSimulator())
      vqe_sample.sample_solution(best_params)
      print(vqe_sample.eigenstate)

   For the ensemble variant — one call to re-sample every partition of a
   trained :class:`~divi.qprog.workflows.PartitioningProgramEnsemble` —
   see :ref:`ensemble-sample-solution`.

Analyzing Solution Distributions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After running optimization with any variational quantum algorithm, you can
analyze the probability distribution of solutions using the
:meth:`~divi.qprog.VariationalQuantumAlgorithm.get_top_solutions`
method. This is particularly useful for understanding solution quality and
exploring alternative solutions beyond the single best one.

The method returns a list of :class:`~divi.qprog.SolutionEntry` objects, each
containing:
- ``bitstring``: The solution bitstring (raw measurement result)
- ``prob``: The probability of measuring this solution
- ``decoded``: The decoded solution (if ``include_decoded=True``)

Solutions are sorted by probability (descending), with lexicographic
tie-breaking for deterministic ordering.

**Decoding Solutions**

By default, solutions are returned as raw bitstrings. However, many algorithms
provide a ``decode_solution_fn`` parameter that converts bitstrings into
problem-specific formats:

- :class:`~divi.qprog.algorithms.QAOA` with QUBO problems: Bitstrings are automatically decoded to NumPy arrays
- :class:`~divi.qprog.algorithms.QAOA` with graph problems: Bitstrings are decoded to lists of node indices
- :class:`~divi.qprog.algorithms.VQE`: Bitstrings represent eigenstates (typically used as-is)
- **Custom decoders**: You can provide your own decoding function when creating the algorithm

Set ``include_decoded=True`` when calling
:meth:`~divi.qprog.VariationalQuantumAlgorithm.get_top_solutions`
to include decoded solutions in the results.

**Example**

.. code-block:: python

   import dimod
   import numpy as np
   from divi.qprog import QAOA
   from divi.qprog.problems import BinaryOptimizationProblem
   from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer
   from divi.backends import MaestroSimulator

   # Create a QUBO problem
   bqm = dimod.generators.gnp_random_bqm(10, 0.5, vartype="BINARY", random_state=1997)
   qubo_array = bqm.to_numpy_matrix()

   qaoa_problem = QAOA(
       BinaryOptimizationProblem(qubo_array),
       n_layers=2,
       optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
       max_iterations=10,
       backend=MaestroSimulator(shots=5000),
   )

   qaoa_problem.run()

   # Get top 10 solutions by probability
   top_solutions = qaoa_problem.get_top_solutions(n=10)

   print("Top 10 solutions:")
   for i, sol in enumerate(top_solutions, 1):
       # Convert bitstring to numpy array for energy calculation
       solution_array = np.array([int(bit) for bit in sol.bitstring])
       solution_dict = {var: int(val) for var, val in zip(bqm.variables, solution_array)}
       energy = bqm.energy(solution_dict)
       print(f"{i}. {sol.bitstring}: prob={sol.prob:.2%}, energy={energy:.4f}")

   # Filter solutions by minimum probability
   high_prob_solutions = qaoa_problem.get_top_solutions(n=5, min_prob=0.01)
   print(f"\nSolutions with probability >= 1%: {len(high_prob_solutions)}")

   # Get solutions with decoded values (for graph problems, this would be node lists)
   # For QUBO problems, decoded values are NumPy arrays
   decoded_solutions = qaoa_problem.get_top_solutions(n=5, include_decoded=True)
   for sol in decoded_solutions:
       print(f"Bitstring: {sol.bitstring}, Decoded: {sol.decoded}")

Circuit Architecture
--------------------

Divi uses a two-tier circuit system for maximum efficiency:

:class:`~divi.circuits.MetaCircuit`
   Divi's logical circuit IR. A :class:`~divi.circuits.MetaCircuit` holds
   one or more tagged Qiskit :class:`~qiskit.dagcircuit.DAGCircuit` bodies,
   the ordered :class:`~qiskit.circuit.Parameter` objects referenced inside
   them, and optional measurement metadata (a
   :class:`~qiskit.quantum_info.SparsePauliOp` observable for
   expectation-value mode, or a tuple of measured qubit indices for
   probabilities/counts).

   The DAG is the long-lived working IR: gate-level stages such as QEM
   folding, Pauli twirling, and QuEPP path enumeration rewrite DAGs in
   place. OpenQASM 2.0 text is produced lazily — only once per parametric
   body (inside :class:`~divi.pipeline.stages.ParameterBindingStage` when it
   builds a :class:`~divi.circuits.QASMTemplate`) and once at compilation
   time when bound bodies are concatenated with pre-serialized measurement
   QASMs.

   You rarely construct a :class:`~divi.circuits.MetaCircuit` by hand. In
   practice, every pipeline starts with a :class:`~divi.pipeline.SpecStage`
   that produces the batch for you:

   - :class:`~divi.pipeline.stages.PennyLaneSpecStage` — PennyLane
     ``QuantumScript`` / ``QNode`` → MetaCircuit
   - :class:`~divi.pipeline.stages.QiskitSpecStage` — Qiskit
     ``QuantumCircuit`` → MetaCircuit
   - :class:`~divi.pipeline.stages.TrotterSpecStage` — Hamiltonian →
     MetaCircuit batch via a Trotterization strategy
   - :class:`~divi.pipeline.stages.CircuitSpecStage` — pass an existing
     :class:`~divi.circuits.MetaCircuit` (or batch) straight through

   For a runnable walkthrough, see
   `standalone_pipeline.py <https://github.com/QoroQuantum/divi/blob/main/tutorials/advanced/standalone_pipeline.py>`_.
   If you need to assemble a :class:`~divi.circuits.MetaCircuit` directly
   (e.g. to write a custom :class:`~divi.pipeline.SpecStage`), see
   :doc:`pipelines`.

Concrete circuits
   When the pipeline submits work to the backend, each
   :class:`~divi.circuits.MetaCircuit` in the batch is lowered to keyed
   OpenQASM strings (label → QASM). Parameter binding, measurement
   grouping, and optional error-mitigation stages happen inside the
   pipeline; see :doc:`pipelines` for details.

Bringing Your Own Circuit
^^^^^^^^^^^^^^^^^^^^^^^^^

Built-in algorithms generate their circuits for you. When you need a **custom
ansatz or circuit**, author it in PennyLane or Qiskit and wrap it with
:class:`~divi.qprog.algorithms.CustomVQA` — see :doc:`framework_integration`,
which also covers QML-style **data binding**. For a curated quantum-neural-network
workflow (feature map + ansatz composed for you), use the
:class:`~divi.qprog.algorithms.QNN` primitive — see
:doc:`quantum_neural_networks`.

Backend Abstraction
-------------------

Divi's backend system provides a unified interface for different execution environments:

:class:`~divi.backends.CircuitRunner` interface
   All backends implement this common interface:

   .. skip: next

   .. code-block:: python

      from divi.backends import ExecutionResult

      class MyCustomBackend(CircuitRunner):
          def submit_circuits(self, circuits: Mapping[str, str], **kwargs) -> ExecutionResult:
              # Your custom execution logic here
              # Return ExecutionResult(results=...) for sync backends
              # or ExecutionResult(job_id=...) for async backends
              pass

   .. note::
      Built-in programs never call ``submit_circuits`` directly — the
      :doc:`circuit pipeline <pipelines>` handles circuit submission and result
      collection automatically. The :class:`~divi.backends.CircuitRunner` interface is still the
      extension point if you need to add a new execution backend.

   .. note::
      The :class:`~divi.backends.ExecutionResult` class provides a unified return type for all backends.
      See :doc:`backends` for detailed information on working with execution results.

**Available Backends:**

- :class:`~divi.backends.MaestroSimulator` — Local high-performance simulator
- :class:`~divi.backends.QiskitSimulator` — Convenience wrapper around Qiskit Aer with noise modeling and thread-count control
- :class:`~divi.backends.QoroService` — Cloud quantum computing service

**Backend Selection:**

.. skip: next

.. code-block:: python

   from divi.qprog import VQE
   from divi.backends import MaestroSimulator, QoroService

   local_backend = MaestroSimulator(shots=1000)  # Development/testing
   cloud_backend = QoroService(auth_token="your-api-key")  # Production/cloud

   backend = local_backend  # Swap to cloud_backend without changing program code
   vqe = VQE(molecule=molecule, backend=backend)

Next Steps
----------

- :doc:`ground_state_energy_estimation_vqe` and :doc:`combinatorial_optimization_qaoa_pce` — algorithm-specific guides
- :doc:`framework_integration` — bring your own PennyLane/Qiskit circuit and QNN data binding
- :doc:`backends` — execution environments and results
- :doc:`../api_reference/qprog/index` — custom algorithms and the full API
- `tutorials/ <https://github.com/QoroQuantum/divi/tree/main/tutorials>`_ — runnable walkthroughs
