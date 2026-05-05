Hamiltonian Time Evolution
==========================

The :class:`~divi.qprog.algorithms.TimeEvolution` program performs Hamiltonian time evolution simulation — simulating real-time quantum dynamics under a given Hamiltonian. Divi supports multiple Trotterization techniques out of the box: :class:`~divi.hamiltonians.ExactTrotterization` (full Trotter-Suzuki decomposition) and :class:`~divi.hamiltonians.QDrift` (randomized term sampling for shallower circuits on large Hamiltonians).

It supports two output modes:

- **Probability mode**: ``te.results`` contains measured basis-state probabilities (``dict[str, float]``).
- **Observable mode**: ``te.results`` contains the estimated expectation value (``float``).

Basic Usage
-----------

Use probability mode when you want a final-state distribution:

.. code-block:: python

   import math
   import pennylane as qp
   from divi.backends import MaestroSimulator
   from divi.qprog import TimeEvolution

   backend = MaestroSimulator(shots=5000)

   te = TimeEvolution(
       hamiltonian=qp.PauliX(0) + qp.PauliX(1),
       time=math.pi / 2,
       backend=backend,
   )
   te.run()

   probs = te.results
   print(probs)
   print(f"Circuits executed: {te.total_circuit_count}")

Observable Mode
---------------

Provide ``observable=...`` to estimate expectation values after evolution:

.. code-block:: python

   import pennylane as qp
   from divi.backends import MaestroSimulator
   from divi.qprog import TimeEvolution

   backend = MaestroSimulator(shots=5000)

   te = TimeEvolution(
       hamiltonian=qp.PauliX(0) + qp.PauliZ(0),
       time=0.6,
       n_steps=8,
       observable=qp.PauliZ(0),
       backend=backend,
   )
   te.run()
   print(f"<Z0> = {te.results:.6f}")

.. tip::

   On sampling backends, pass ``shot_distribution="weighted"`` to focus the
   shot budget on the observable's dominant terms.  See
   :ref:`adaptive-shot-allocation` for the full list of strategies.

QDrift Trotterization
---------------------

For large Hamiltonians, you can use :class:`~divi.hamiltonians.QDrift` to sample terms and average over multiple Hamiltonian samples:

.. code-block:: python

   import pennylane as qp
   from divi.backends import MaestroSimulator
   from divi.hamiltonians import QDrift
   from divi.qprog import SuperpositionState, TimeEvolution

   backend = MaestroSimulator(shots=5000)
   qdrift = QDrift(
       keep_fraction=0.5,
       sampling_budget=2,
       n_hamiltonians_per_iteration=4,
       sampling_strategy="weighted",
       seed=7,
   )

   te = TimeEvolution(
       hamiltonian=0.7 * qp.PauliZ(0) + 0.4 * qp.PauliX(0) + 0.2 * qp.PauliZ(1),
       trotterization_strategy=qdrift,
       time=0.8,
       initial_state=SuperpositionState(),
       backend=backend,
   )
   te.run()
   print(te.results)

.. note::
   Multi-sample QDrift with ``observable`` on an expectation-value backend is currently not supported.
   Use a shot-based backend or run in probability mode for multi-sample averaging.

Time Evolution Trajectory
-------------------------

Use :class:`~divi.qprog.workflows.TimeEvolutionTrajectory` to simulate dynamics at
multiple time points in parallel.  It creates one ``TimeEvolution`` program
per time point, runs them via
:class:`~divi.qprog.ensemble.ProgramEnsemble` (with optional circuit batching), and
collects results into a time-ordered mapping.

.. code-block:: python

   import math
   import numpy as np
   import pennylane as qp
   from divi.backends import MaestroSimulator
   from divi.qprog import TimeEvolutionTrajectory

   backend = MaestroSimulator(shots=5000)

   trajectory = TimeEvolutionTrajectory(
       hamiltonian=qp.PauliX(0),
       time_points=np.linspace(0.01, math.pi, 20).tolist(),
       observable=qp.PauliZ(0),
       backend=backend,
   )
   trajectory.create_programs()
   trajectory.run(blocking=True)

   results = trajectory.aggregate_results()
   # results: {0.01: 0.9996, 0.166: 0.944, ..., 3.14: 0.998}

   trajectory.visualize_results()   # plots ⟨Z⟩ vs time

The trajectory supports all the same options as ``TimeEvolution``
(``trotterization_strategy``, ``n_steps``, ``order``, ``initial_state``).

**Batching across time points**

Each time point becomes its own program in a
:class:`~divi.qprog.ensemble.ProgramEnsemble`, so the trajectory inherits
the ensemble's circuit-batching machinery.  The default ``BatchConfig()``
already merges every program's submission into a single backend call:

.. skip: next

.. code-block:: python

   from divi.qprog import BatchConfig

   # Default — single merged call for all time points
   trajectory.run(blocking=True, batch_config=BatchConfig())

For dense trajectories (e.g. 512 time points) on
:class:`~divi.backends.QoroService`, set ``max_concurrent_programs=-1`` so
all time points run concurrently and the entire ensemble submits as one
cloud job, reducing overhead significantly:

.. skip: next

.. code-block:: python

   trajectory.run(
       blocking=True,
       batch_config=BatchConfig(max_concurrent_programs=-1),
   )

See :ref:`circuit-batching` for the full batching reference.

Next Steps
----------

- Run the full tutorial scripts: `time_evolution.py <https://github.com/QoroQuantum/divi/blob/main/tutorials/time_evolution.py>`_, `time_evolution_trajectory.py <https://github.com/QoroQuantum/divi/blob/main/tutorials/time_evolution_trajectory.py>`_, and `nmr_spectroscopy.py <https://github.com/QoroQuantum/divi/blob/main/tutorials/nmr_spectroscopy.py>`_ — a 34-spin NMR simulation that demonstrates the cloud-merge recipe end-to-end.
- Learn about optimization-based workflows in :doc:`ground_state_energy_estimation_vqe` and :doc:`combinatorial_optimization_qaoa_pce`
- Explore backend choices in :doc:`backends`
