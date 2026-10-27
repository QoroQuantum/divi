Hamiltonian Time Evolution
==========================

The :class:`~divi.qprog.algorithms.TimeEvolution` program performs Hamiltonian time evolution simulation — simulating real-time quantum dynamics under a given Hamiltonian. Divi supports multiple Trotterization techniques out of the box: :class:`~divi.hamiltonians.trotterization.ExactTrotterization` (full Trotter-Suzuki decomposition) and :class:`~divi.hamiltonians.trotterization.QDrift` (randomised term sampling for shallower circuits on large Hamiltonians).

It supports three output modes:

- **Probability mode**: ``te.results`` contains measured basis-state probabilities (``dict[str, float]``).
- **Observable mode**: ``te.results`` contains the estimated expectation value (``float``).
- **Multi-observable mode**: ``te.results`` contains one expectation value per observable (``list[float]``), measured from a single shared evolution.

Basic Usage
-----------

Use probability mode when you want a final-state distribution:

.. code-block:: python

   import math
   from qiskit.quantum_info import SparsePauliOp
   from divi.backends import MaestroSimulator
   from divi.qprog import TimeEvolution

   backend = MaestroSimulator(shots=5000)

   te = TimeEvolution(
       hamiltonian=SparsePauliOp.from_sparse_list(
           [("X", [0], 1.0), ("X", [1], 1.0)], num_qubits=2
       ),
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

   from qiskit.quantum_info import SparsePauliOp
   from divi.backends import MaestroSimulator
   from divi.qprog import TimeEvolution

   backend = MaestroSimulator(shots=5000)

   te = TimeEvolution(
       hamiltonian=SparsePauliOp("X") + SparsePauliOp("Z"),
       time=0.6,
       n_steps=8,
       observable=SparsePauliOp("Z"),
       backend=backend,
   )
   te.run()
   print(f"<Z0> = {te.results:.6f}")

.. tip::

   On sampling backends, pass ``shot_distribution="weighted"`` to focus the
   shot budget on the observable's dominant terms.  See
   :ref:`adaptive-shot-allocation` for the full list of strategies.

.. _time-evolution-multi-observable:

Multi-Observable Mode
---------------------

Pass a list (or tuple) to ``observable=`` to estimate several expectation
values from a single evolution.  ``te.results`` becomes a ``list[float]`` in
the same order as the input observables, and the cost is shared across them:

.. code-block:: python

   import math
   from qiskit.quantum_info import SparsePauliOp
   from divi.backends import MaestroSimulator
   from divi.qprog import TimeEvolution

   backend = MaestroSimulator(shots=5000)

   def two_qubit(*terms):
       return SparsePauliOp.from_sparse_list(list(terms), num_qubits=2)

   te = TimeEvolution(
       hamiltonian=two_qubit(("X", [0], 1.0), ("X", [1], 1.0)),
       time=math.pi / 2,
       observable=[
           two_qubit(("Z", [0], 1.0)),
           two_qubit(("Z", [1], 1.0)),
           two_qubit(("ZZ", [0, 1], 1.0)),
       ],
       backend=backend,
   )
   te.run()

   z0, z1, zz = te.results
   print(f"<Z0>   = {z0:+.4f}")
   print(f"<Z1>   = {z1:+.4f}")
   print(f"<Z0Z1> = {zz:+.4f}")

Two cost-saving mechanisms apply automatically:

- **Shared evolution**: the ``e^{-iHt}`` body is built and executed once per
  measurement group, not once per observable.
- **Cross-observable QWC grouping**:
  :class:`~divi.pipeline.stages.MeasurementStage` unions the Pauli terms of
  every observable and groups commuting terms into shared shot batches.  In
  the example above all three observables are diagonal in the Z basis, so
  they collapse into a single measurement group.

When a quantum error mitigation protocol is set, the savings extend further;
see :ref:`qem-multi-observable`.

Trotter Steps and Order
-----------------------

By default the evolution uses a single first-order Trotter step
(``n_steps=1``, ``order=1``).  Increasing ``n_steps`` or the Suzuki-Trotter
``order`` reduces the Trotter error at the cost of deeper circuits.  ``order``
must be ``1`` or an even integer ``>= 2`` (a non-conforming value raises
``ValueError``); higher even orders converge faster per step but add gates.

.. code-block:: python

   from qiskit.quantum_info import SparsePauliOp
   from divi.backends import MaestroSimulator
   from divi.qprog import TimeEvolution

   backend = MaestroSimulator(shots=5000)
   H = 0.7 * SparsePauliOp("Z") + 0.4 * SparsePauliOp("X")
   Z0 = SparsePauliOp("Z")

   # Second-order Suzuki-Trotter with 8 steps for tighter accuracy.
   te = TimeEvolution(
       hamiltonian=H,
       time=1.0,
       n_steps=8,
       order=2,
       observable=Z0,
       backend=backend,
   )
   te.run()
   print(te.expval())

The default term-selection strategy is
:class:`~divi.hamiltonians.trotterization.ExactTrotterization`, which keeps every Hamiltonian
term.  Pass it explicitly to truncate small-coefficient terms — ``keep_top_n``
keeps the N largest-magnitude terms, ``keep_fraction`` keeps a fraction in
``(0, 1]`` (the two are mutually exclusive):

.. code-block:: python

   from divi.hamiltonians.trotterization import ExactTrotterization

   te = TimeEvolution(
       hamiltonian=H,
       trotterization_strategy=ExactTrotterization(keep_top_n=1),
       time=1.0,
       observable=Z0,
       backend=backend,
   )
   te.run()

QDrift Trotterization
---------------------

For large Hamiltonians, you can use :class:`~divi.hamiltonians.trotterization.QDrift` to sample terms and average over multiple Hamiltonian samples.
Each of the ``n_steps`` Trotter steps draws its own ``sampling_budget`` terms independently.
Sampled evolution is first order, so it requires ``order=1``:

.. code-block:: python

   from qiskit.quantum_info import SparsePauliOp
   from divi.backends import MaestroSimulator
   from divi.hamiltonians.trotterization import QDrift
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
       hamiltonian=SparsePauliOp.from_sparse_list(
           [("Z", [0], 0.7), ("X", [0], 0.4), ("Z", [1], 0.2)], num_qubits=2
       ),
       trotterization_strategy=qdrift,
       time=0.8,
       initial_state=SuperpositionState(),
       backend=backend,
   )
   te.run()
   print(te.results)

**State-aware sampling (StateDRIFT)**

``sampling_strategy="weighted"`` picks terms by coefficient size alone. With
``sampling_strategy="state_drift"``, :class:`~divi.qprog.algorithms.TimeEvolution`
instead picks them by how strongly each term changes the measured observable in
the initial state, following StateDRIFT
(`arXiv:2610.31515 <https://arxiv.org/abs/2610.31515>`_), which reports a
lower bias than weighted sampling at the same circuit depth. Every term keeps at least
a fifth of its weighted probability, and the sampled Hamiltonian still equals
:math:`H` on average.

.. code-block:: python

   from qiskit.quantum_info import SparsePauliOp
   from divi.backends import MaestroSimulator
   from divi.hamiltonians.trotterization import QDrift
   from divi.qprog import CustomPerQubitState, TimeEvolution

   hamiltonian = SparsePauliOp.from_sparse_list(
       [("ZZ", [0, 1], 1.0), ("ZZ", [1, 2], 1.0), ("X", [0], 0.7),
        ("X", [1], 0.7), ("X", [2], 0.7), ("Y", [1], 0.4)],
       num_qubits=3,
   )
   te = TimeEvolution(
       hamiltonian=hamiltonian,
       trotterization_strategy=QDrift(
           sampling_budget=4,
           n_hamiltonians_per_iteration=20,
           sampling_strategy="state_drift",
           seed=7,
       ),
       time=0.05,
       n_steps=2,
       initial_state=CustomPerQubitState("0+1"),
       observable=SparsePauliOp.from_sparse_list([("Z", [0], 1.0)], num_qubits=3),
       backend=MaestroSimulator(),
   )
   te.run()
   print(te.results)

The strategy computes the probabilities when the program runs, from the
initial state, observable and time that ``TimeEvolution`` hands it. It needs:

- exactly one ``observable``;
- an initial state. Product states (``ZerosState``, ``OnesState``,
  ``SuperpositionState``, ``CustomPerQubitState``) are evaluated in closed
  form. Any other state is simulated once with Maestro's matrix-product-state
  simulator, which evaluates in one batch the Pauli strings of the nested
  commutators of :math:`H` with :math:`O` up to fourth order; their number, and
  with it the cost, grows quickly with the number of Hamiltonian terms;
- ``sampling_budget`` set, and no ``keep_fraction`` or ``keep_top_n``.

The probabilities follow the paper's optimiser for the full circuit depth
(Appendices F and J): they minimise the fourth-order mean-squared error of
``n_hamiltonians_per_iteration`` averaged runs, or the bias alone when every
term's first-order effect on the observable vanishes, while keeping both the
bias and the variance within 5% of weighted sampling's. The expansion behind
the optimiser holds only for small rotations, so every sampled rotation angle
:math:`|c_j|\,\mathrm{d}t/q_j` stays within 0.5, where :math:`\mathrm{d}t` is
``time / (n_steps * sampling_budget)``. The strategy warns and samples by
weight when:

- :math:`\lambda\,\mathrm{d}t > 0.5`, with :math:`\lambda = \sum_j |c_j|`, so no
  distribution keeps the angles within 0.5; raising ``n_steps`` or
  ``sampling_budget`` brings them back down;
- no term changes the observable in the initial state;
- the expansion is not reliable for the evolution, because its third- and
  fourth-order corrections are too large next to the result.

The strategy evaluates each initial state and observable once and reuses the
result for every time, also across the copies of it that
:class:`~divi.qprog.workflows.TimeEvolutionTrajectory` makes, so a trajectory
evaluates the state once and only re-solves the probabilities per time point.
QAOA gives the strategy no initial state or observable, so it raises an error
there.

Time Evolution Trajectory
-------------------------

Use :class:`~divi.qprog.workflows.TimeEvolutionTrajectory` to simulate dynamics at
multiple time points in parallel.  It creates one ``TimeEvolution`` program
per time point, runs them via
:class:`~divi.qprog.ensemble.ProgramEnsemble` (with optional circuit batching), and
collects results into a time-ordered mapping.

.. dashboard-example: simulation

.. code-block:: python

   import math
   import numpy as np
   from qiskit.quantum_info import SparsePauliOp
   from divi.backends import MaestroSimulator
   from divi.qprog import TimeEvolutionTrajectory

   backend = MaestroSimulator(shots=5000)

   trajectory = TimeEvolutionTrajectory(
       hamiltonian=SparsePauliOp("X"),
       time_points=np.linspace(0.01, math.pi, 20).tolist(),
       observable=SparsePauliOp("Z"),
       backend=backend,
   )
   trajectory.run()

   results = trajectory.aggregate_results()
   # results: {0.01: 0.9996, 0.166: 0.944, ..., 3.14: 0.998}

   trajectory.visualize_results()   # plots ⟨Z⟩ vs time

The trajectory supports all the same options as ``TimeEvolution``
(``trotterization_strategy``, ``n_steps``, ``order``, ``initial_state``).

**Batching across time points**

Each time point becomes its own program in a
:class:`~divi.qprog.ensemble.ProgramEnsemble`, so the trajectory inherits
the ensemble's circuit-batching machinery — the default ``BatchConfig()``
already merges every program's submission into a single backend call.  For
dense trajectories (e.g. 512 time points) on
:class:`~divi.backends.QoroService`, set ``max_concurrent_programs=-1`` so
all time points run concurrently and the entire ensemble submits as one
cloud job, reducing overhead significantly:

.. skip: next

.. code-block:: python

   from divi.qprog import BatchConfig

   trajectory.run(
       batch_config=BatchConfig(max_concurrent_programs=-1),
   )

See :ref:`circuit-batching` for the full batching reference.

Next Steps
----------

- Run the full tutorial scripts: `time_evolution.py <https://github.com/QoroQuantum/divi/blob/main/tutorials/dynamics/time_evolution.py>`_ — covers single-time-point evolution, trajectories, multi-observable groups, and QDrift — and `nmr_spectroscopy.py <https://github.com/QoroQuantum/divi/blob/main/tutorials/nmr_spectroscopy.py>`_ — a 34-spin NMR simulation that demonstrates the cloud-merge recipe end-to-end.
- Learn about optimisation-based workflows in :doc:`ground_state_energy_estimation_vqe` and :doc:`combinatorial_optimization_qaoa_pce`
- Explore backend choices in :doc:`../execution_workflows/backends`
