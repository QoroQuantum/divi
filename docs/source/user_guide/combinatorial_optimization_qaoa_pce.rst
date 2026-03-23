Combinatorial Optimization with QAOA and PCE
============================================

The Quantum Approximate Optimization Algorithm (QAOA) is designed to solve combinatorial optimization problems on near-term quantum computers.

Divi offers two QAOA modes: single-instance mode for individual problems, and graph partitioning mode for large, intractable problems.

Single-Instance QAOA
--------------------

A `QAOA` constructor expects a ``Problem`` instance that encapsulates the optimization objective. However, there are some common arguments that one must pay attention to.

A user has the ability to choose the **initial state** of the quantum system before the optimization by passing an ``InitialState`` instance. Available states include ``ZerosState()``, ``OnesState()``, ``SuperpositionState()``, ``CustomPerQubitState("01+-")``, and ``WState(block_size, n_blocks)`` for one-hot encoded problems. When ``initial_state=None`` (the default), graph problems use a problem-specific recommendation and QUBO/HUBO problems default to ``SuperpositionState()``. When ``WState`` is used, the mixer is automatically set to the XY mixer, which preserves the one-hot subspace. In addition, a user can determine how many **layers** of the QAOA ansatz to apply.

**Initial Parameters**: You can set custom initial parameters for QAOA optimization using the ``initial_params`` constructor argument or the ``curr_params`` property. This is useful for warm-starting from known good parameter regions or continuing from previous runs. For detailed information and examples, see the :doc:`core_concepts` guide on Parameter Management.

Real-World Examples
^^^^^^^^^^^^^^^^^^^

Based on test cases and real applications, here are some proven configurations:

**Bull Graph Max-Clique**:

.. code-block:: python

   import networkx as nx
   import numpy as np
   from divi.qprog import QAOA, MaxCliqueProblem
   from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer
   from divi.backends import MaestroSimulator
   # Tested configuration for bull graph max-clique
   G = nx.bull_graph()

   qaoa_problem = QAOA(
       MaxCliqueProblem(G, is_constrained=True),
       n_layers=1,
       optimizer=ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
       max_iterations=10,
       backend=MaestroSimulator(),
   )

   qaoa_problem.run()
   # Should find the same solution as classical: {0, 1, 2}

**QUBO Optimization**:

.. code-block:: python

   import numpy as np
   from divi.qprog import QAOA, BinaryOptimizationProblem
   from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer
   from divi.backends import MaestroSimulator
   # Tested QUBO matrix that should optimize to [1, 0, 1]
   qubo_matrix = np.array([
       [-3.0, 4.0, 0.0],
       [0.0, 2.0, 0.0],
       [0.0, 0.0, -3.0],
   ])

   qaoa_problem = QAOA(
       BinaryOptimizationProblem(qubo_matrix),
       n_layers=1,
       optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
       max_iterations=12,
       backend=MaestroSimulator(),
   )

   qaoa_problem.run()
   # Should find solution: [1, 0, 1]

Trotterization Strategies
-------------------------

QAOA evolves the cost Hamiltonian in the ansatz. By default, Divi uses
:class:`~divi.qprog.ExactTrotterization`, which applies all Hamiltonian terms in each
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
   from divi.qprog import QAOA, MaxCutProblem, QDrift
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
       n_layers=1,
       trotterization_strategy=qdrift,
       optimizer=ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
       max_iterations=5,
       backend=MaestroSimulator(),
   )
   qaoa.run()

For a full comparison of Exact Trotterization vs QDrift (including circuit depth and
count), see the `qaoa_qdrift_local.py
<https://github.com/QoroQuantum/divi/blob/main/tutorials/qaoa_qdrift_local.py>`_
tutorial.

Graph Problems
--------------

Divi supports several common graph-based optimization problems out of the box,
including Max-Clique, MaxCut, Max Independent Set, Max Weight Cycle, and
Min Vertex Cover.  Each graph problem has a dedicated class:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Problem Class
     - Description
   * - ``MaxCutProblem(graph)``
     - Divides a graph into two subsets to maximize the sum of edge weights between them.
   * - ``MaxCliqueProblem(graph)``
     - Finds the largest complete subgraph where every node is connected to every other.
   * - ``MaxIndependentSetProblem(graph)``
     - Finds the largest set of vertices with no edges between them.
   * - ``MinVertexCoverProblem(graph)``
     - Finds the smallest set of vertices such that every edge is incident to at least one selected vertex.
   * - ``MaxWeightCycleProblem(graph)``
     - Identifies a cycle with the maximum total edge weight in a weighted graph.

Example: Finding the max-clique of a graph:

.. code-block:: python

   import networkx as nx
   from divi.qprog import QAOA, MaxCliqueProblem
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

Divi's QAOA solver can also handle Quadratic Unconstrained Binary Optimization (QUBO) problems. Divi currently supports three methods of formulating the QUBO problem:

1. **NumPy Array Input**: Pass a :class:`numpy.ndarray` or a :class:`scipy.sparse` array directly
2. **BinaryQuadraticModel**: Use the `dimod` library to create :class:`dimod.BinaryQuadraticModel` objects
3. **List Input**: Pass a Python list (converted to NumPy array internally)

In contrast to graph-based QAOA instances, the solution format for QUBO-based QAOA instances is a binary :class:`numpy.ndarray` representing the value for each variable in the original QUBO.

NumPy Array-based Input
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import dimod
   from divi.qprog import QAOA, BinaryOptimizationProblem
   from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer

   # Generate a random QUBO
   bqm = dimod.generators.randint(5, vartype="BINARY", low=-10, high=10, seed=1997)
   qubo_array = bqm.to_numpy_matrix()

   qaoa_problem = QAOA(
       BinaryOptimizationProblem(qubo_array),
       n_layers=2,
       optimizer=ScipyOptimizer(method=ScipyMethod.L_BFGS_B),
       max_iterations=5,
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
   from divi.qprog import QAOA, BinaryOptimizationProblem
   from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer
   from divi.backends import MaestroSimulator

   # Create a BinaryQuadraticModel
   bqm = dimod.BinaryQuadraticModel(
       linear={"w": 10, "x": -3, "y": 2},
       quadratic={("w", "x"): -1, ("x", "y"): 1},
       offset=0.0,
       vartype=dimod.Vartype.BINARY
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

``BinaryOptimizationProblem`` offers two strategies for converting a HUBO into an Ising Hamiltonian:

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

   from divi.qprog import QAOA, BinaryOptimizationProblem
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
       max_iterations=30,
       backend=MaestroSimulator(shots=10000),
   )

   qaoa.run()

   # HUBO solutions are dictionaries mapping variable names to binary values.
   print(qaoa.solution)   # e.g. {"a": 1, "b": 0, "c": 1, "d": 1}

.. note::

   When variables have non-integer labels, ``.solution`` returns a
   ``dict[variable_name, int]``.  For QUBO matrices (integer-indexed),
   ``.solution`` remains a NumPy array for backwards compatibility.

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
   * - ``INTERP``
     - Linear interpolation (Zhou et al.).  Simple and robust.
   * - ``FOURIER``
     - DCT-II Fourier basis.  Fits a smooth frequency representation.
   * - ``CHEBYSHEV``
     - Chebyshev polynomial basis at Chebyshev nodes.

Example:

.. code-block:: python

   import networkx as nx
   from divi.qprog import (
       InterpolationStrategy,
       IterativeQAOA,
       MaxCutProblem,
       ScipyMethod,
       ScipyOptimizer,
   )
   from divi.backends import MaestroSimulator

   graph = nx.random_regular_graph(3, 16, seed=42)

   iterative = IterativeQAOA(
       MaxCutProblem(graph),
       max_depth=5,
       strategy=InterpolationStrategy.INTERP,
       max_iterations_per_depth=15,
       backend=MaestroSimulator(shots=10000),
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
       backend=backend,
       optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
   )

Graph Partitioning QAOA
-----------------------

For large graphs that exceed quantum hardware limitations, use GraphPartitioningQAOA:

.. code-block:: python

   import networkx as nx
   from divi.qprog import GraphPartitioningQAOA, MaxCutProblem, PartitioningConfig
   from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer
   from divi.backends import MaestroSimulator

   # Large graph
   large_graph = nx.erdos_renyi_graph(20, 0.3)

   # Configure partitioning
   config = PartitioningConfig(
       max_n_nodes_per_cluster=8,           # Maximum nodes per quantum partition
       minimum_n_clusters=3,                # Minimum number of partitions (optional)
       partitioning_algorithm="metis"       # Algorithm: "spectral", "metis", or "kernighan_lin"
   )

   qaoa_partition = GraphPartitioningQAOA(
       problem=MaxCutProblem(large_graph),
       n_layers=2,
       partitioning_config=config,
       optimizer=ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
       max_iterations=20,
       backend=MaestroSimulator(),
   )

   # Execute workflow
   qaoa_partition.create_programs()
   qaoa_partition.run(blocking=True)

   # Aggregate results from all partitions
   quantum_solution = qaoa_partition.aggregate_results()

   print(f"Total circuits executed: {qaoa_partition.total_circuit_count}")

QUBO Partitioning (QAOA or PCE)
-------------------------------

For large QUBO problems, use ``QUBOPartitioningQAOA`` with D-Wave's hybrid library.
You can choose the per-partition engine via ``engine``:

- ``engine="qaoa"`` (default): standard QAOA partitions.
- ``engine="pce"``: ``PCE`` partitions (supports ``PCE``-specific kwargs like ``encoding_type`` and ``alpha``).
- ``engine="iterative_qaoa"``: ``IterativeQAOA`` partitions with warm-started depth progression.
  Pass ``strategy``, ``max_iterations_per_depth``, and other ``IterativeQAOA``-specific kwargs
  directly; ``n_layers`` is used as ``max_depth``.

QAOA partitions:

.. code-block:: python

   import dimod
   import hybrid
   from divi.qprog import QUBOPartitioningQAOA
   from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer
   from divi.backends import MaestroSimulator

   # Large QUBO problem
   large_bqm = dimod.generators.gnp_random_bqm(25, 0.5, vartype="BINARY")

   qubo_partition = QUBOPartitioningQAOA(
       qubo=large_bqm,
       decomposer=hybrid.EnergyImpactDecomposer(size=5),
       composer=hybrid.SplatComposer(),
       n_layers=2,
       optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
       max_iterations=10,
       backend=MaestroSimulator(),
   )

   qubo_partition.create_programs()
   qubo_partition.run()

   # Get aggregated solution
   quantum_solution, quantum_energy = qubo_partition.aggregate_results()

   print(f"Quantum solution: {quantum_solution}")
   print(f"Quantum energy: {quantum_energy:.6f}")

PCE partitions:

.. code-block:: python

   import dimod
   import hybrid
   import pennylane as qml
   from divi.qprog import QUBOPartitioningQAOA
   from divi.qprog.algorithms import GenericLayerAnsatz
   from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer
   from divi.backends import MaestroSimulator

   large_bqm = dimod.generators.gnp_random_bqm(25, 0.5, vartype="BINARY")

   qubo_partition = QUBOPartitioningQAOA(
       qubo=large_bqm,
       decomposer=hybrid.EnergyImpactDecomposer(size=5),
       engine="pce",
       ansatz=GenericLayerAnsatz([qml.RY, qml.RZ]),
       n_layers=2,
       encoding_type="dense",
       alpha=2.0,
       optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
       max_iterations=10,
       backend=MaestroSimulator(),
   )

   qubo_partition.create_programs()
   qubo_partition.run()
   quantum_solution, quantum_energy = qubo_partition.aggregate_results()

Iterative QAOA partitions:

.. code-block:: python

   import dimod
   import hybrid
   from divi.qprog import InterpolationStrategy, QUBOPartitioningQAOA
   from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer
   from divi.backends import MaestroSimulator

   large_bqm = dimod.generators.gnp_random_bqm(25, 0.5, vartype="BINARY")

   qubo_partition = QUBOPartitioningQAOA(
       qubo=large_bqm,
       decomposer=hybrid.EnergyImpactDecomposer(size=5),
       engine="iterative_qaoa",
       n_layers=3,  # used as max_depth
       strategy=InterpolationStrategy.INTERP,
       max_iterations_per_depth=10,
       optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
       backend=MaestroSimulator(),
   )

   qubo_partition.create_programs()
   qubo_partition.run()
   quantum_solution, quantum_energy = qubo_partition.aggregate_results()

What's Happening?
^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Step
     - Description
   * - ``decomposer=...``
     - The QUBO is partitioned into smaller subproblems using an energy impact decomposer.
   * - ``create_programs()``
     - Initializes a batch of QAOA programs, each solving a subproblem of the original QUBO.
   * - ``run()``
     - Executes all generated circuits—possibly in parallel across multiple quantum backends.
   * - ``aggregate_results()``
     - The final QUBO solution is formed by combining the results from each subproblem.

Why Partition?
--------------

Quantum hardware is limited in the number of qubits and circuit depth. For large problems:

- Full QAOA is intractable.
- Partitioned QAOA trades global optimality for scalability and parallel execution.
- It enables fast, approximate solutions using many small quantum jobs rather than one large one.

Next Steps
----------

- Try the runnable examples in the `tutorials/ <https://github.com/QoroQuantum/divi/tree/main/tutorials>`_ directory
- Learn about :doc:`optimizers` for optimization strategies
- Explore :doc:`backends` for execution options
