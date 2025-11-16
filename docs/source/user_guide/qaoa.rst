QAOA
====

The Quantum Approximate Optimization Algorithm (QAOA) is designed to solve combinatorial optimization problems on near-term quantum computers.

Divi offers two QAOA modes: single-instance mode for individual problems, and graph partitioning mode for large, intractable problems.

Single-Instance QAOA
--------------------

A `QAOA` constructor expects a problem to be provided. As we will show in the examples, the form of input triggers a different execution pathway under the hood. However, there are some common arguments that one must pay attention to.

A user has the ability to choose the **initial state** of the quantum system before the optimization. By default (when the argument `initial_state = "Recommended"` is passed), a problem-specific initial state would be chosen. Other accepted values are `"Zero"`, `"Ones"`, and `"Superposition"`. In addition, a user can determine how many **layers** of the QAOA ansatz to apply.

**Initial Parameters**: You can set custom initial parameters for QAOA optimization using the ``initial_params`` constructor argument or the ``curr_params`` property. This is useful for warm-starting from known good parameter regions or continuing from previous runs. For detailed information and examples, see the :doc:`core_concepts` guide on Parameter Management.

Real-World Examples
^^^^^^^^^^^^^^^^^^^

Based on test cases and real applications, here are some proven configurations:

**Bull Graph Max-Clique**:

.. code-block:: python

   import networkx as nx
   import numpy as np
   from divi.qprog import QAOA, GraphProblem
   from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer
   from divi.backends import ParallelSimulator
   # Tested configuration for bull graph max-clique
   G = nx.bull_graph()

   qaoa_problem = QAOA(
       problem=G,
       graph_problem=GraphProblem.MAX_CLIQUE,
       n_layers=1,
       optimizer=ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
       max_iterations=10,
       is_constrained=True,
       backend=ParallelSimulator(),
   )

   qaoa_problem.run()
   # Should find the same solution as classical: {0, 1, 2}

**QUBO Optimization**:

.. code-block:: python

   import numpy as np
   from divi.qprog import QAOA
   from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer
   from divi.backends import ParallelSimulator
   # Tested QUBO matrix that should optimize to [1, 0, 1]
   qubo_matrix = np.array([
       [-3.0, 4.0, 0.0],
       [0.0, 2.0, 0.0],
       [0.0, 0.0, -3.0],
   ])

   qaoa_problem = QAOA(
       problem=qubo_matrix,
       n_layers=1,
       optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
       max_iterations=12,
       backend=ParallelSimulator(),
   )

   qaoa_problem.run()
   # Should find solution: [1, 0, 1]

Graph Problems
--------------

Divi includes built-in support for several common graph-based optimization problems:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Problem
     - Description
   * - ``MAX_CLIQUE``
     - Finds the largest complete subgraph where every node is connected to every other.
   * - ``MAX_INDEPENDENT_SET``
     - Finds the largest set of vertices with no edges between them.
   * - ``MAX_WEIGHT_CYCLE``
     - Identifies a cycle with the maximum total edge weight in a weighted graph.
   * - ``MAXCUT``
     - Divides a graph into two subsets to maximize the sum of edge weights between them.
   * - ``MIN_VERTEX_COVER``
     - Finds the smallest set of vertices such that every edge is incident to at least one selected vertex.

Example: Finding the max-clique of a graph:

.. code-block:: python

   import networkx as nx
   from divi.qprog import QAOA, GraphProblem
   from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer
   from divi.backends import ParallelSimulator

   # Create a graph
   G = nx.bull_graph()

   qaoa_problem = QAOA(
       problem=G,
       graph_problem=GraphProblem.MAX_CLIQUE,
       n_layers=2,
       optimizer=ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
       max_iterations=10,
       is_constrained=True,
       backend=ParallelSimulator(),
   )

   qaoa_problem.run()
   qaoa_problem.compute_final_solution()

   print(f"Quantum Solution: {set(qaoa_problem.solution)}")
   print(f"Total circuits: {qaoa_problem.total_circuit_count}")

QUBO Problems
-------------

Divi's QAOA solver can also handle Quadratic Unconstrained Binary Optimization (QUBO) problems. Divi currently supports two methods of formulating the QUBO problem:

1. **NumPy Array Input**: Pass a :class:`numpy.ndarray` or a :class:`scipy.sparse` array directly
2. **Qiskit Quadratic Program**: Use the `qiskit-optimization` library to create :class:`qiskit_optimization.QuadraticProgram` objects

In contrast to graph-based QAOA instances, the solution format for QUBO-based QAOA instances is a binary :class:`numpy.ndarray` representing the value for each variable in the original QUBO.

Numpy Array-based Input
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import dimod
   from divi.qprog import QAOA
   from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer

   # Generate a random QUBO
   bqm = dimod.generators.randint(5, vartype="BINARY", low=-10, high=10, seed=1997)
   qubo_array = bqm.to_numpy_matrix()

   qaoa_problem = QAOA(
       problem=qubo_array,
       n_layers=2,
       optimizer=ScipyOptimizer(method=ScipyMethod.L_BFGS_B),
       max_iterations=5,
       backend=ParallelSimulator(),
   )

   qaoa_problem.run()
   qaoa_problem.compute_final_solution()

   print(f"Solution: {qaoa_problem.solution}")
   print(f"Energy: {qaoa_problem.best_loss}")

Qiskit Quadratic Program Input
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from qiskit_optimization import QuadraticProgram
   from divi.qprog import QAOA
   from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer
   from divi.backends import ParallelSimulator

   qp = QuadraticProgram()
   qp.binary_var("w")
   qp.binary_var("x")
   qp.binary_var("y")
   qp.integer_var(lowerbound=0, upperbound=7, name="z")
   qp.minimize(linear={"x": -3, "y": 2, "z": -1, "w": 10})

   qaoa_problem = QAOA(
       problem=qp,
       n_layers=2,
       optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
       max_iterations=10,
       backend=ParallelSimulator(),
   )

   qaoa_problem.run()
   qaoa_problem.compute_final_solution()

   # The binary mask as is might be useless when importing a QuadraticProgram
   # You can evaluate the energy of the solution sample using:
   print(qaoa_problem.problem.objective.evaluate(qaoa_problem.solution))
   # And you can also translate it to the QuadraticProgram's variables using:
   print(qaoa_problem._qp_converter.interpret(qaoa_problem.solution))

Graph Partitioning QAOA
-----------------------

For large graphs that exceed quantum hardware limitations, use GraphPartitioningQAOA:

.. code-block:: python

   import networkx as nx
   from divi.qprog import GraphPartitioningQAOA, GraphProblem, PartitioningConfig
   from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer
   from divi.backends import ParallelSimulator

   # Large graph
   large_graph = nx.erdos_renyi_graph(20, 0.3)

   # Configure partitioning
   config = PartitioningConfig(
       max_n_nodes_per_cluster=8,           # Maximum nodes per quantum partition
       minimum_n_clusters=3,                # Minimum number of partitions (optional)
       partitioning_algorithm="metis"       # Algorithm: "spectral", "metis", or "kernighan_lin"
   )

   qaoa_partition = GraphPartitioningQAOA(
       graph_problem=GraphProblem.MAXCUT,
       graph=large_graph,
       n_layers=2,
       partitioning_config=config,
       optimizer=ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
       max_iterations=20,
       backend=ParallelSimulator(),
   )

   # Execute workflow
   qaoa_partition.create_programs()
   qaoa_partition.run(blocking=True)

   # Aggregate results from all partitions
   quantum_solution = qaoa_partition.aggregate_results()

   print(f"Total circuits executed: {qaoa_partition.total_circuit_count}")

QUBO Partitioning QAOA
----------------------

For large QUBO problems, use QUBOPartitioningQAOA with D-Wave's hybrid library:

.. code-block:: python

   import dimod
   import hybrid
   from divi.qprog import QUBOPartitioningQAOA
   from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer
   from divi.backends import ParallelSimulator

   # Large QUBO problem
   large_bqm = dimod.generators.gnp_random_bqm(25, 0.5, vartype="BINARY")

   qubo_partition = QUBOPartitioningQAOA(
       qubo=large_bqm,
       decomposer=hybrid.EnergyImpactDecomposer(size=5),
       composer=hybrid.SplatComposer(),
       n_layers=2,
       optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
       max_iterations=10,
       backend=ParallelSimulator(),
   )

   qubo_partition.create_programs()
   qubo_partition.run()

   # Get aggregated solution
   quantum_solution, quantum_energy = qubo_partition.aggregate_results()

   print(f"Quantum solution: {quantum_solution}")
   print(f"Quantum energy: {quantum_energy:.6f}")

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
