Program Ensembles and Workflows
================================

A :class:`~divi.qprog.ensemble.ProgramEnsemble` runs multiple quantum programs
in parallel — handling scheduling, circuit batching, progress tracking, and
result aggregation.  Typical use cases include parameter sweeps, molecular
dissociation curves, problem decomposition, and algorithm comparison.

Built-in Ensemble Workflows
----------------------------

Divi provides several ready-made ensemble workflows.  Each is covered in
detail on its own page — this section gives a quick overview and links.

**VQE Hyperparameter Sweeps**
   :class:`~divi.qprog.workflows.VQEHyperparameterSweep` runs VQE across multiple
   molecular configurations (bond lengths, ansätze) in parallel.
   See :doc:`ground_state_energy_estimation_vqe` for configuration and examples.

**Time Evolution Trajectories**
   :class:`~divi.qprog.workflows.TimeEvolutionTrajectory` runs one time-evolution
   program per time point and collects expectation values into a trajectory.
   See :doc:`hamiltonian_time_evolution` for full details.

**Problem Decomposition (Graph / QUBO / Matching)**
   :class:`~divi.qprog.workflows.PartitioningProgramEnsemble` decomposes a
   large :class:`~divi.qprog.problems.QAOAProblem` into sub-problems, solves
   each partition with QAOA (or PCE / IterativeQAOA), and stitches results
   via beam search.  Graph partitioning, QUBO partitioning, and matching
   partitioning are all covered in
   :doc:`combinatorial_optimization_qaoa_pce`.

Beam Search Aggregation
-----------------------

When aggregating partition results, each partition has multiple candidate bitstrings ranked by probability. By default, aggregation picks only the **single best** candidate from each partition (greedy). Beam search explores multiple candidates per partition to find better global combinations.

**How it works**

The ``aggregate_results`` method accepts two parameters:

- ``beam_width`` — how many partial solutions are kept after each partition step.
- ``n_partition_candidates`` — how many candidates to extract from each partition (defaults to ``beam_width``).

.. skip: next

.. code-block:: python

   # Greedy (default): single best candidate per partition
   solution, energy = qaoa_partition.aggregate_results(beam_width=1)

   # Beam search: keep top 5 partial solutions, consider 5 candidates per partition
   solution, energy = qaoa_partition.aggregate_results(beam_width=5)

   # Wider candidate pool with narrow beam: consider 10 candidates per partition
   # but only keep the best 3 partial solutions after each step
   solution, energy = qaoa_partition.aggregate_results(beam_width=3, n_partition_candidates=10)

   # Exhaustive: try all candidate combinations (expensive for many partitions)
   solution, energy = qaoa_partition.aggregate_results(beam_width=None)

**When to use beam search**

- **Greedy** (``beam_width=1``): Fast, good for problems with low inter-partition coupling.
- **Bounded beam** (``beam_width=k``): Good trade-off for problems with moderate coupling between partitions. Start with ``beam_width=3`` and increase if solution quality improves.
- **Exhaustive** (``beam_width=None``): Guarantees the global optimum across all candidate combinations, but scales exponentially with the number of partitions.

.. tip::

   Setting ``n_partition_candidates`` higher than ``beam_width`` is useful when you want each partition to propose many alternatives (wider local search) while keeping memory usage controlled (narrow beam).

Top-N Solutions
---------------

:class:`~divi.qprog.workflows.PartitioningProgramEnsemble` exposes a ``get_top_solutions`` method that returns multiple ranked global solutions using beam search.

.. skip: next

.. code-block:: python

   top_solutions = qaoa_partition.get_top_solutions(
       n=5, beam_width=5, n_partition_candidates=10
   )

   # Return type is problem-dependent:
   #   Graph  → list[(node_indices, energy)]
   #   QUBO   → list[(solution_array, energy)]
   for rank, (solution, energy) in enumerate(top_solutions, 1):
       print(f"{rank}. Energy: {energy:.6f}, Solution: {solution}")

This is useful when you want to inspect alternative solutions or post-process candidates with domain-specific constraints. The ``beam_width`` is automatically increased to at least ``n`` so the beam retains enough candidates.


Custom Ensemble Workflows
-------------------------

You can create custom program ensemble workflows by inheriting from :class:`~divi.qprog.ensemble.ProgramEnsemble`:

**Custom Ensemble Implementation**

.. code-block:: python

   from divi.qprog import ProgramEnsemble, VQE
   from divi.backends import CircuitRunner, MaestroSimulator
   import pennylane as qp
   import numpy as np

   class CustomParameterSweep(ProgramEnsemble):
       def __init__(self, backend: CircuitRunner, molecules):
           super().__init__(backend)
           self.molecules = molecules

       def create_programs(self):
           """Generate one VQE program per molecule."""
           super().create_programs()
           for i, mol in enumerate(self.molecules):
               vqe = VQE(
                   molecule=mol,
                   backend=self.backend,
                   max_iterations=10,
               )
               self._programs[f"sweep_{i}"] = vqe

       def aggregate_results(self):
           """Collect and analyze results from all programs"""
           super().aggregate_results()
           results = {}
           for program_id, program in self._programs.items():
               if program.losses_history:  # Check if program completed
                   final_loss = program.best_loss
                   results[program_id] = {
                       'energy': final_loss,
                       'params': program.best_params,
                       'circuits': program.total_circuit_count
                   }
           return results

   # Usage
   mol1 = qp.qchem.Molecule(symbols=["H", "H"], coordinates=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]))
   mol2 = qp.qchem.Molecule(symbols=["Li", "H"], coordinates=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.6]]))
   mol3 = qp.qchem.Molecule(symbols=["H", "F"], coordinates=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.92]]))
   molecules = [mol1, mol2, mol3]

   # Use a local simulator
   local_backend = MaestroSimulator()
   sweep = CustomParameterSweep(local_backend, molecules)
   sweep.create_programs()
   sweep.run(blocking=True)

   results = sweep.aggregate_results()
   print(results)

Progress Monitoring and Control
-------------------------------

Divi provides automatic progress tracking for long-running ensembles. When you
execute an ensemble that contains compatible programs (like :class:`~divi.qprog.algorithms.VQE` or
:class:`~divi.qprog.algorithms.QAOA`), a rich progress display appears in your console showing the
status of each program in real-time.

When circuit batching is active (the default), an additional batch status line
appears below the per-program progress bars. It shows the merged job's polling
status — how many circuits were merged, which programs are part of the current
flush group, and the backend job ID.

**Stopping an Ensemble**

You can gracefully stop a running ensemble at any time by pressing ``Ctrl+C``.
The ``KeyboardInterrupt`` is caught during :meth:`~divi.qprog.ensemble.ProgramEnsemble.join`,
which cancels any in-flight backend jobs and allows currently running programs
to finish their current iteration before shutting down.

Circuit Batching
----------------

By default, :meth:`~divi.qprog.ensemble.ProgramEnsemble.run` merges the circuit
submissions from all programs in the ensemble into **single backend calls**.
This behavior is controlled by :class:`~divi.qprog.ensemble.BatchConfig`.

**How it works**

Each optimization iteration, every program calls ``submit_circuits`` on its
backend. With batching enabled, these calls are intercepted by a coordinator
that:

1. Collects circuit submissions from all active programs (barrier-based flush).
2. Merges them into a single payload with namespaced circuit tags.
3. Submits the merged payload to the real backend in one call.
4. Polls for results once (instead of N times).
5. Demultiplexes the results back to each program by tag prefix.

This happens transparently — programs are unaware they're sharing a backend
call.

**When to use batching**

- **Cloud backends** (:class:`~divi.backends.QoroService`): batching reduces
  the number of API calls, authentication round-trips, and polling loops.
  This is the primary use case.
- **Local simulators** (:class:`~divi.backends.QiskitSimulator`): batching
  adds synchronization overhead for no network benefit. The simulator already
  parallelizes circuits internally.

**Limiting batch size**

By default the coordinator waits for **all** active programs to submit before
merging circuits.  For large ensembles this can produce very large merged jobs.
Use ``max_batch_size`` to cap the number of circuits per flush:

.. skip: next

.. code-block:: python

   from divi.qprog import BatchConfig

   # Flush as soon as 50 circuits are pending (partial flush)
   ensemble.run(blocking=True, batch_config=BatchConfig(max_batch_size=50))

When the pending circuit count reaches ``max_batch_size`` the coordinator
flushes immediately — even if some programs haven't submitted yet.  Those
programs will be included in a later flush.  This reduces per-job size on
the backend and can improve latency for large ensembles.

``max_batch_size`` controls **merging granularity**, not individual payload
size.  A single program that submits more circuits than the limit will still
flush normally.

**Disabling batching**

Pass ``BatchConfig(mode=BatchMode.OFF)`` to disable batching entirely:

.. skip: next

.. code-block:: python

   from divi.qprog import BatchConfig, BatchMode

   # Each program submits circuits independently
   ensemble.run(blocking=True, batch_config=BatchConfig(mode=BatchMode.OFF))

   # Merged submissions (default)
   ensemble.run(blocking=True)

Next Steps
----------

- :doc:`backends` — backend configuration and performance tuning.
- :doc:`resuming_long_runs` — checkpointing and resuming long ensemble runs.
- :doc:`visualization` — result visualization, including :meth:`~divi.qprog.workflows.VQEHyperparameterSweep.visualize_results`.
