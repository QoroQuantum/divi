Combinatorial Optimization with QAOA and PCE
============================================

The Quantum Approximate Optimization Algorithm (QAOA) is designed to solve combinatorial optimization problems on near-term quantum computers.

Divi offers two QAOA modes: single-instance mode for individual problems, and partitioning mode for large, intractable problems.

The :class:`~divi.qprog.problems.QAOAProblem` interface
-------------------------------------------------------

Every problem solved with QAOA in Divi is represented as a
:class:`~divi.qprog.problems.QAOAProblem` subclass.  This base class defines the
contract between domain-specific problem logic and the QAOA algorithm — QAOA
never knows about graphs, QUBOs, or routes directly; it only interacts with the
interface that :class:`~divi.qprog.problems.QAOAProblem` provides.

A subclass must implement four required properties — ``cost_hamiltonian``,
``mixer_hamiltonian``, ``loss_constant``, and ``decode_fn`` — and may
optionally override methods for initial state, feasibility checking, solution
repair, and graph-partitioning decomposition.  See
:class:`~divi.qprog.problems.QAOAProblem` for the full interface specification.

Divi ships several concrete subclasses — graph problems,
:class:`~divi.qprog.problems.BinaryOptimizationProblem` (QUBO/HUBO), routing
problems (:class:`~divi.qprog.problems.TSPProblem`,
:class:`~divi.qprog.problems.CVRPProblem`), and
:class:`~divi.qprog.problems.MaxWeightMatchingProblem` — all described in the
sections below.

Custom Problems
^^^^^^^^^^^^^^^

To solve a problem that doesn't fit the built-in classes, subclass
:class:`~divi.qprog.problems.QAOAProblem` and implement the four required properties.  Here is a
minimal example that encodes a simple 2-qubit Hamiltonian:

.. code-block:: python

   import pennylane as qp
   from divi.qprog import QAOA, ScipyOptimizer, ScipyMethod
   from divi.qprog.problems import QAOAProblem
   from divi.backends import MaestroSimulator

   class MyProblem(QAOAProblem):
       @property
       def cost_hamiltonian(self):
           return -1.0 * qp.Z(0) @ qp.Z(1) + 0.5 * qp.Z(0)

       @property
       def mixer_hamiltonian(self):
           return qp.X(0) + qp.X(1)

       @property
       def loss_constant(self):
           return 0.0

       @property
       def decode_fn(self):
           # Map bitstring to a list of selected qubits
           return lambda bs: [i for i, b in enumerate(bs) if b == "1"]

   qaoa = QAOA(
       MyProblem(),
       n_layers=2,
       optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
       max_iterations=10,
       backend=MaestroSimulator(),
   )
   qaoa.run()
   print(qaoa.solution)

Override ``is_feasible``, ``compute_energy``, or ``repair_infeasible_bitstring``
to enable feasibility-aware post-processing via
:meth:`~divi.qprog.algorithms.QAOA.get_top_solutions`.  Override the
decomposition hooks to enable partitioned solving via
:class:`~divi.qprog.workflows.PartitioningProgramEnsemble`.
See :class:`~divi.qprog.problems.QAOAProblem` for the full interface.

Single-Instance QAOA
--------------------

The :class:`~divi.qprog.algorithms.QAOA` constructor expects a
:class:`~divi.qprog.problems.QAOAProblem` instance that encapsulates the optimization
objective. Two knobs matter in almost every run: the **initial state** and **circuit depth**
(``n_layers``).

Pass an :class:`~divi.qprog.algorithms.InitialState` subclass for ``initial_state``.
Built-in options include :class:`~divi.qprog.algorithms.ZerosState`,
:class:`~divi.qprog.algorithms.OnesState`, :class:`~divi.qprog.algorithms.SuperpositionState`,
:class:`~divi.qprog.algorithms.CustomPerQubitState`\ ``("01+-")``, and
:class:`~divi.qprog.algorithms.WState`\ ``(block_size, n_blocks)`` (one-hot encodings).
When ``initial_state`` is omitted, graph problems use a problem-specific default and
QUBO/HUBO problems default to :class:`~divi.qprog.algorithms.SuperpositionState`.
Using :class:`~divi.qprog.algorithms.WState` selects the XY mixer automatically so the
state stays in the one-hot subspace.

**Initial Parameters**: You can set custom initial parameters for QAOA optimization by passing ``initial_params`` to ``run()``. This is useful for warm-starting from known good parameter regions or continuing from previous runs. For detailed information and examples, see the :doc:`core_concepts` guide on Parameter Management.

Trotterization Strategies
-------------------------

QAOA evolves the cost Hamiltonian in the ansatz. By default, Divi uses
:class:`~divi.hamiltonians.ExactTrotterization`, which applies all Hamiltonian terms in each
circuit. For large Hamiltonians, this can produce deep circuits that are costly or
infeasible on noisy hardware.

:class:`~divi.hamiltonians.QDrift` is a randomized Trotterization strategy that approximates the cost
Hamiltonian by sampling a subset of terms. It yields shallower circuits at the cost
of more circuits per iteration (multiple Hamiltonian samples are averaged). On noisy
hardware, lower depth can improve fidelity despite the higher circuit count.

Key QDrift parameters:

- **keep_fraction**: Deterministically keep the top fraction of terms by coefficient magnitude
- **sampling_budget**: Number of terms to sample from the remaining Hamiltonian
- **n_hamiltonians_per_iteration**: Multiple samples per cost evaluation; losses are averaged
- **sampling_strategy**: ``"uniform"`` or ``"weighted"`` (by coefficient magnitude)

Example: QAOA with QDrift:

.. code-block:: python

   import networkx as nx
   from divi.qprog import QAOA
   from divi.hamiltonians import QDrift
   from divi.qprog.problems import MaxCutProblem
   from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer
   from divi.backends import MaestroSimulator

   G = nx.erdos_renyi_graph(12, 0.3, seed=1997)
   qdrift = QDrift(
       keep_fraction=0.2,
       sampling_budget=5,
       n_hamiltonians_per_iteration=3,
       sampling_strategy="weighted",
       seed=1997,
   )
   qaoa = QAOA(
       MaxCutProblem(G),
       n_layers=2,
       trotterization_strategy=qdrift,
       optimizer=ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
       max_iterations=10,
       backend=MaestroSimulator(),
   )
   qaoa.run()

For a full comparison of Exact Trotterization vs QDrift (including circuit depth and
count), see the `qaoa_qdrift.py
<https://github.com/QoroQuantum/divi/blob/main/tutorials/qaoa_qdrift.py>`_
tutorial.

.. tip::

   On sampling backends, pass ``shot_distribution="weighted"`` to focus the
   cost Hamiltonian's shot budget on its dominant terms.  See
   :ref:`adaptive-shot-allocation` for the full list of strategies.

Graph Problems
--------------

Divi supports several common graph-based optimization problems out of the box,
including Max-Clique, MaxCut, Max Independent Set, Max Weight Cycle, and
Min Vertex Cover.  Each graph problem has a dedicated class:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Problem class
     - Description
   * - :class:`~divi.qprog.problems.MaxCutProblem`\ ``(graph)``
     - Divides a graph into two subsets to maximize the sum of edge weights between them.
   * - :class:`~divi.qprog.problems.MaxCliqueProblem`\ ``(graph)``
     - Finds the largest complete subgraph where every node is connected to every other.
   * - :class:`~divi.qprog.problems.MaxIndependentSetProblem`\ ``(graph)``
     - Finds the largest set of vertices with no edges between them.
   * - :class:`~divi.qprog.problems.MinVertexCoverProblem`\ ``(graph)``
     - Finds the smallest set of vertices such that every edge is incident to at least one selected vertex.
   * - :class:`~divi.qprog.problems.MaxWeightCycleProblem`\ ``(graph)``
     - Identifies a cycle with the maximum total edge weight in a weighted graph.
   * - :class:`~divi.qprog.problems.MaxWeightMatchingProblem`\ ``(graph)``
     - Finds a set of edges with maximum total weight where no two edges share a node.

Example: Finding the max-clique of a graph:

.. code-block:: python

   import networkx as nx
   from divi.qprog import QAOA
   from divi.qprog.problems import MaxCliqueProblem
   from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer
   from divi.backends import MaestroSimulator

   # Create a graph
   G = nx.bull_graph()

   qaoa_problem = QAOA(
       MaxCliqueProblem(G, is_constrained=True),
       n_layers=2,
       optimizer=ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
       max_iterations=10,
       backend=MaestroSimulator(),
   )

   qaoa_problem.run()

   print(f"Quantum Solution: {set(qaoa_problem.solution)}")
   print(f"Total circuits: {qaoa_problem.total_circuit_count}")

   # Get top-N solutions by probability
   top_solutions = qaoa_problem.get_top_solutions(n=5, include_decoded=True)
   print("\nTop 5 solutions by probability:")
   for i, sol in enumerate(top_solutions, 1):
       print(f"{i}. Nodes: {sol.decoded} (probability: {sol.prob:.2%})")

QUBO Problems
-------------

Divi's QAOA solver can also handle Quadratic Unconstrained Binary Optimization (QUBO) problems. Divi currently supports three ways to build a :class:`~divi.qprog.problems.BinaryOptimizationProblem`:

1. **NumPy array** — pass a :class:`numpy.ndarray` or a :mod:`scipy.sparse` matrix directly
2. **Dimod BQM** — use ``dimod`` to construct a :class:`dimod.BinaryQuadraticModel`
3. **Nested list** — pass a Python list (converted to a NumPy array internally)

In contrast to graph-based QAOA instances, the solution format for QUBO-based QAOA instances is a binary :class:`numpy.ndarray` representing the value for each variable in the original QUBO.

NumPy Array-based Input
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import numpy as np
   import dimod
   from divi.qprog import QAOA
   from divi.qprog.problems import BinaryOptimizationProblem
   from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer

   # Generate a random QUBO
   bqm = dimod.generators.randint(5, vartype="BINARY", low=-10, high=10, seed=1997)
   qubo_array = bqm.to_numpy_matrix()

   qaoa_problem = QAOA(
       BinaryOptimizationProblem(qubo_array),
       n_layers=2,
       optimizer=ScipyOptimizer(method=ScipyMethod.L_BFGS_B),
       max_iterations=10,
       backend=MaestroSimulator(),
   )

   qaoa_problem.run()

   print(f"Solution: {qaoa_problem.solution}")
   print(f"Energy: {qaoa_problem.best_loss}")

   # Get top-N solutions by probability
   top_solutions = qaoa_problem.get_top_solutions(n=5)
   print("\nTop 5 solutions by probability:")
   for i, sol in enumerate(top_solutions, 1):
       solution_array = np.array([int(bit) for bit in sol.bitstring])
       energy = bqm.energy({var: int(val) for var, val in zip(bqm.variables, solution_array)})
       print(f"{i}. {sol.bitstring}: {sol.prob:.2%} (energy: {energy:.4f})")

BinaryQuadraticModel Input
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import dimod
   from divi.qprog import QAOA
   from divi.qprog.problems import BinaryOptimizationProblem
   from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer
   from divi.backends import MaestroSimulator

   # Create a BinaryQuadraticModel
   bqm = dimod.BinaryQuadraticModel(
       {"w": 10, "x": -3, "y": 2},
       {("w", "x"): -1, ("x", "y"): 1},
       offset=0.0,
       vartype=dimod.Vartype.BINARY,
   )

   qaoa_problem = QAOA(
       BinaryOptimizationProblem(bqm),
       n_layers=2,
       optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
       max_iterations=10,
       backend=MaestroSimulator(),
   )

   qaoa_problem.run(perform_final_computation=True)

   # BQMs with string variables return a dict solution.
   print(f"Solution: {qaoa_problem.solution}")  # e.g. {"w": 0, "x": 1, "y": 0}
   print(f"Energy: {qaoa_problem.best_loss}")

   # Evaluate energy using the BinaryQuadraticModel directly:
   print(f"BQM Energy: {bqm.energy(qaoa_problem.solution)}")

HUBO Problems
-------------

Divi's QAOA solver supports Higher-Order Binary Optimization (HUBO) problems —
polynomials with cubic or higher-degree interactions.  A HUBO is passed as a
dictionary mapping variable tuples to coefficients:

.. code-block:: python

   hubo = {
       ("a",): -2.0,           # linear
       ("a", "b"): 1.5,        # quadratic
       ("a", "b", "c"): 2.0,   # cubic
   }

Variables can use any hashable labels (strings, integers, etc.).

Hamiltonian Builders
^^^^^^^^^^^^^^^^^^^^

:class:`~divi.qprog.problems.BinaryOptimizationProblem` offers two strategies for converting a HUBO into an Ising Hamiltonian:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Builder
     - Description
   * - ``"native"`` (default)
     - Maps each polynomial term directly to a multi-Z Ising interaction.
       No ancilla qubits are added.
   * - ``"quadratized"``
     - Reduces the polynomial to quadratic form by introducing ancilla qubits
       with a configurable penalty strength (``quadratization_strength``).

Example
^^^^^^^

.. code-block:: python

   from divi.qprog import QAOA
   from divi.qprog.problems import BinaryOptimizationProblem
   from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer
   from divi.backends import MaestroSimulator

   hubo = {
       ("a",): -2.0,
       ("b",): 1.0,
       ("c",): -3.0,
       ("a", "b"): 1.5,
       ("c", "d"): -1.0,
       ("a", "b", "c"): 2.0,
   }

   qaoa = QAOA(
       BinaryOptimizationProblem(hubo, hamiltonian_builder="native"),
       n_layers=2,
       optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
       max_iterations=10,
       backend=MaestroSimulator(shots=5000),
   )

   qaoa.run()

   # HUBO solutions are dictionaries mapping variable names to binary values.
   print(qaoa.solution)   # e.g. {"a": 1, "b": 0, "c": 1, "d": 1}

.. note::

   When variables have non-integer labels, ``.solution`` returns a
   ``dict[variable_name, int]``.  For QUBO matrices (integer-indexed),
   ``.solution`` remains a NumPy array for backwards compatibility.

Matching Problems
-----------------

Divi supports maximum-weight matching via :class:`~divi.qprog.problems.MaxWeightMatchingProblem`.  Given
a weighted graph, it finds a set of edges that maximizes total weight while
ensuring no two selected edges share a node.

For small graphs, use directly with QAOA:

.. code-block:: python

   import networkx as nx
   from divi.qprog import QAOA
   from divi.qprog.problems import MaxWeightMatchingProblem
   from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer
   from divi.backends import MaestroSimulator

   G = nx.Graph()
   G.add_weighted_edges_from([(0, 1, 5.0), (1, 2, 1.0), (2, 3, 5.0)])

   problem = MaxWeightMatchingProblem(G, penalty_scale=10.0)
   qaoa = QAOA(
       problem,
       n_layers=2,
       optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
       max_iterations=10,
       backend=MaestroSimulator(),
   )
   qaoa.run()
   print(f"Matching: {qaoa.solution}")

For large graphs, enable edge-based partitioning with ``max_edges_per_partition``:

.. code-block:: python

   from divi.qprog.workflows import PartitioningProgramEnsemble

   problem = MaxWeightMatchingProblem(
       G,
       penalty_scale=10.0,
       max_edges_per_partition=15,
       partition_algorithm="kernighan_lin",
   )

   ensemble = PartitioningProgramEnsemble(
       problem=problem,
       n_layers=2,
       optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
       max_iterations=10,
       backend=MaestroSimulator(),
   )

   ensemble.create_programs()
   ensemble.run(blocking=True)
   matching, weight = ensemble.aggregate_results()
   print(f"Matching: {matching}, weight: {weight}")

The partitioned workflow splits the graph by edges using Kernighan-Lin or spectral
bisection, solves each partition independently, stitches results via beam search,
and optionally fills unmatched residual nodes using classical
:func:`~networkx.algorithms.matching.max_weight_matching`.

Iterative QAOA
--------------

Standard QAOA uses random initialization at a fixed circuit depth.
:class:`~divi.qprog.algorithms.IterativeQAOA` improves on this by iteratively
increasing the depth from 1 to ``max_depth``, warm-starting each depth with
parameters interpolated from the previous optimum.  This strategy, based on
`arXiv:2504.01694 <https://arxiv.org/abs/2504.01694>`_, often converges to
better solutions with the same per-depth budget.

Three interpolation strategies are available via :class:`~divi.qprog.algorithms.InterpolationStrategy`:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Strategy
     - Description
   * - :attr:`~divi.qprog.algorithms.InterpolationStrategy.INTERP`
     - Linear interpolation (Zhou et al.).  Simple and robust.
   * - :attr:`~divi.qprog.algorithms.InterpolationStrategy.FOURIER`
     - DCT-II Fourier basis.  Fits a smooth frequency representation.
   * - :attr:`~divi.qprog.algorithms.InterpolationStrategy.CHEBYSHEV`
     - Chebyshev polynomial basis at Chebyshev nodes.

Example:

.. code-block:: python

   import networkx as nx
   from divi.qprog import InterpolationStrategy, IterativeQAOA
   from divi.qprog.problems import MaxCutProblem
   from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer
   from divi.backends import MaestroSimulator

   graph = nx.random_regular_graph(3, 16, seed=42)

   iterative = IterativeQAOA(
       MaxCutProblem(graph),
       max_depth=5,
       strategy=InterpolationStrategy.INTERP,
       max_iterations_per_depth=10,
       backend=MaestroSimulator(shots=5000),
       optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
   )
   iterative.run()

   print(f"Best depth: {iterative.best_depth}")
   print(f"Best loss:  {iterative.best_loss:.6f}")
   print(f"Solution:   {iterative.solution}")

   # Per-depth optimization history
   for entry in iterative.depth_history:
       print(f"  p={entry['depth']}  loss={entry['best_loss']:.6f}")

The ``max_iterations_per_depth`` parameter can also be a callable
``(depth) -> int`` for adaptive budgets — for example, allocating more
iterations to deeper circuits:

.. code-block:: python

   iterative = IterativeQAOA(
       MaxCutProblem(graph),
       max_depth=5,
       strategy=InterpolationStrategy.FOURIER,
       max_iterations_per_depth=lambda depth: 10 + 5 * depth,
       convergence_threshold=1e-4,  # stop early if improvement is negligible
       backend=MaestroSimulator(shots=5000),
       optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
   )

Graph Partitioning QAOA
-----------------------

For large graphs that exceed quantum hardware limitations, use
:class:`~divi.qprog.workflows.PartitioningProgramEnsemble`
with a graph problem configured for partitioning via
:class:`~divi.qprog.problems.GraphPartitioningConfig`:

.. code-block:: python

   import networkx as nx
   from divi.qprog.problems import MaxCutProblem, GraphPartitioningConfig
   from divi.qprog.workflows import PartitioningProgramEnsemble
   from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer
   from divi.backends import MaestroSimulator

   # Large graph
   large_graph = nx.erdos_renyi_graph(20, 0.3)

   # Configure partitioning
   config = GraphPartitioningConfig(
       max_n_nodes_per_cluster=8,           # Maximum nodes per quantum partition
       minimum_n_clusters=3,                # Minimum number of partitions (optional)
       partitioning_algorithm="metis"       # Algorithm: "spectral", "metis", or "kernighan_lin"
   )

   # Create the problem with partitioning config
   problem = MaxCutProblem(large_graph, config=config)

   ensemble = PartitioningProgramEnsemble(
       problem=problem,
       n_layers=2,
       optimizer=ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
       max_iterations=10,
       backend=MaestroSimulator(),
   )

   # Execute workflow
   ensemble.create_programs()
   ensemble.run(blocking=True)

   # Aggregate results from all partitions
   quantum_solution, energy = ensemble.aggregate_results()

   print(f"MaxCut value: {energy}")
   print(f"Total circuits executed: {ensemble.total_circuit_count}")

QUBO Partitioning (QAOA or PCE)
-------------------------------

For large QUBO problems, use :class:`~divi.qprog.workflows.PartitioningProgramEnsemble` with a
:class:`~divi.qprog.problems.BinaryOptimizationProblem` configured with D-Wave's hybrid decomposer/composer.
You can choose the per-partition engine via ``quantum_routine``:

- ``quantum_routine="qaoa"`` (default): standard :class:`~divi.qprog.algorithms.QAOA` partitions.
- ``quantum_routine="pce"``: :class:`~divi.qprog.algorithms.PCE` partitions (supports PCE-specific kwargs such as ``encoding_type`` and ``alpha``).
- ``quantum_routine="iterative_qaoa"``: :class:`~divi.qprog.algorithms.IterativeQAOA` partitions with warm-started depth progression.
  Pass ``strategy``, ``max_iterations_per_depth``, and other IterativeQAOA-specific kwargs
  directly; ``n_layers`` is used as ``max_depth``.

One QUBO, one set of imports, and one helper for ``create_programs`` → ``run`` →
``aggregate_results``. Only ``BinaryOptimizationProblem`` and
``PartitioningProgramEnsemble`` change between ``quantum_routine`` choices:

.. code-block:: python

   import dimod
   import hybrid
   import pennylane as qp
   from divi.qprog import InterpolationStrategy
   from divi.qprog.problems import BinaryOptimizationProblem
   from divi.qprog.workflows import PartitioningProgramEnsemble
   from divi.qprog.algorithms import GenericLayerAnsatz
   from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer
   from divi.backends import MaestroSimulator

   def run_partitioned(ensemble):
       ensemble.create_programs()
       ensemble.run()
       return ensemble.aggregate_results()

   large_bqm = dimod.generators.gnp_random_bqm(25, 0.5, vartype="BINARY")
   decomposer = hybrid.EnergyImpactDecomposer(size=5)
   optimizer = ScipyOptimizer(method=ScipyMethod.COBYLA)
   backend = MaestroSimulator()

   # --- QAOA partitions (default ``quantum_routine``): add a composer ---
   problem = BinaryOptimizationProblem(
       large_bqm,
       decomposer=decomposer,
       composer=hybrid.SplatComposer(),
   )
   ensemble = PartitioningProgramEnsemble(
       problem=problem,
       n_layers=2,
       optimizer=optimizer,
       max_iterations=10,
       backend=backend,
   )
   sol_qaoa, energy_qaoa = run_partitioned(ensemble)

   # --- PCE partitions ---
   problem = BinaryOptimizationProblem(large_bqm, decomposer=decomposer)
   ensemble = PartitioningProgramEnsemble(
       problem=problem,
       quantum_routine="pce",
       ansatz=GenericLayerAnsatz([qp.RY, qp.RZ]),
       n_layers=2,
       encoding_type="dense",
       alpha=2.0,
       optimizer=optimizer,
       max_iterations=10,
       backend=backend,
   )
   sol_pce, energy_pce = run_partitioned(ensemble)

   # --- Iterative QAOA partitions ---
   problem = BinaryOptimizationProblem(large_bqm, decomposer=decomposer)
   ensemble = PartitioningProgramEnsemble(
       problem=problem,
       quantum_routine="iterative_qaoa",
       n_layers=2,  # used as max_depth
       strategy=InterpolationStrategy.INTERP,
       max_iterations_per_depth=10,
       optimizer=optimizer,
       backend=backend,
   )
   sol_iter, energy_iter = run_partitioned(ensemble)

The hybrid ``decomposer`` and optional ``composer`` are configured on
:class:`~divi.qprog.problems.BinaryOptimizationProblem` (how the large BQM is split
and, for the default QAOA path, stitched back together). The helper only groups
the usual ensemble calls; for progress output, circuit batching, and Ctrl+C
behavior, see :doc:`program_ensembles`.

Why Partition?
--------------

Quantum hardware is limited in the number of qubits and circuit depth. For large problems:

- Full QAOA is intractable.
- Partitioned QAOA trades global optimality for scalability and parallel execution.
- It enables fast, approximate solutions using many small quantum jobs rather than one large one.

Next Steps
----------

- `tutorials/ <https://github.com/QoroQuantum/divi/tree/main/tutorials>`_ — QAOA/PCE/partitioning examples (``qaoa_*.py``, ``pce_*.py``, ``ce_qaoa_*.py``, ``iterative_qaoa.py``)
- :doc:`routing` — TSP and CVRP with constraint-preserving encodings
- :doc:`optimizers` — optimizer choice and ``run(initial_params=...)``
- :doc:`backends` — simulators and services
- :doc:`../api_reference/qprog/problems` — full :class:`~divi.qprog.problems.QAOAProblem` API
