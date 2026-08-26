Combinatorial Optimisation with QAOA and PCE
============================================

Divi provides several routes from a graph, QUBO, or HUBO to a binary solution.
Start with the problem representation and scale, then choose the solver:

.. list-table:: Choosing a workflow
   :header-rows: 1
   :widths: 20 35 45

   * - Workflow
     - Use it when
     - Main trade-off
   * - :class:`~divi.qprog.algorithms.QAOA`
     - The problem fits in the available qubits and you want the standard
       cost/mixer construction.
     - Uses roughly one qubit per binary variable.
   * - :class:`~divi.qprog.algorithms.PCE`
     - A QUBO or HUBO has too many variables for one-qubit-per-variable
       encoding.
     - Encodes variables as Pauli correlations, using logarithmic or
       square-root qubit scaling at the cost of a different objective landscape.
   * - :class:`~divi.qprog.algorithms.IterativeQAOA`
     - You want to increase QAOA depth gradually and warm-start each depth.
     - Runs several optimisations, but often starts deeper circuits from better
       parameters.
   * - :class:`~divi.qprog.workflows.PartitioningProgramEnsemble`
     - The full problem should be decomposed into independently executable
       sub-problems.
     - Scales beyond one quantum program, but partition boundaries can reduce
       global solution quality.

For a first run, continue to `Graph Problems`_ or `QUBO Problems`_. The next
section is for users defining a problem type that Divi does not already ship.

QAOAProblem Contract and Hamiltonian Helpers
--------------------------------------------

Every problem solved with QAOA in Divi is represented as a
:class:`~divi.qprog.problems.QAOAProblem` subclass. QAOA depends only on this
interface, not on graphs, QUBOs, or routes directly.

Divi ships several concrete subclasses — graph problems,
:class:`~divi.qprog.problems.BinaryOptimizationProblem` (QUBO/HUBO), routing
problems (:class:`~divi.qprog.problems.TSPProblem`,
:class:`~divi.qprog.problems.CVRPProblem`), and
:class:`~divi.qprog.problems.MaxWeightMatchingProblem` — all described in the
sections below.

Custom problems must provide ``cost_hamiltonian``, ``loss_constant``, and
``decode_fn``: the objective operator, its additive offset, and the function
mapping a measured bitstring to a domain solution. Optional hooks customise the
mixer, initial state, feasibility, repair, and partitioned aggregation. The cost Hamiltonian is a
:class:`~qiskit.quantum_info.SparsePauliOp`; two helpers cover common inputs:

* :func:`~divi.hamiltonians.to_spo` accepts a PennyLane operator, an existing
  sparse Pauli operator, or ``{pauli_string: coefficient}``. The leftmost
  character is qubit 0.
* :func:`~divi.hamiltonians.qubo_to_spo` accepts a QUBO/HUBO mapping, matrix,
  or :class:`dimod.BinaryQuadraticModel`. It folds the offset into an identity
  term; use :func:`~divi.hamiltonians.qubo_to_ising` when you also need decoder
  or encoding metadata.

.. code-block:: python

   from divi.hamiltonians import to_spo, qubo_to_spo

   cost_from_dict = to_spo({"XI": 1.0, "IZ": 0.5})
   cost_from_qubo = qubo_to_spo({(0,): -1.0, (1,): -1.0, (0, 1): 2.0})

The default :func:`~divi.hamiltonians.x_mixer` suits unconstrained binary
problems. Override ``mixer_hamiltonian`` for constrained spaces. Builders include
:func:`~divi.hamiltonians.x_mixer`, :func:`~divi.hamiltonians.xy_mixer`,
:func:`~divi.hamiltonians.bit_flip_mixer`, and
:func:`~divi.hamiltonians.edge_driver`.

Override ``is_feasible``, ``compute_energy``, or
``repair_infeasible_bitstring`` as needed for feasibility-aware
post-processing. See the base-class API for the full contract.

Single-Instance QAOA
--------------------

QAOA takes a :class:`~divi.qprog.problems.QAOAProblem`; its main circuit knobs
are ``initial_state`` and ``n_layers``.

Pass an :class:`~divi.qprog.algorithms.InitialState` subclass for ``initial_state``.
Built-in options include :class:`~divi.qprog.algorithms.ZerosState`,
:class:`~divi.qprog.algorithms.OnesState`, :class:`~divi.qprog.algorithms.SuperpositionState`,
:class:`~divi.qprog.algorithms.CustomPerQubitState`\ ``("01+-")``, and
:class:`~divi.qprog.algorithms.WState`\ ``(block_size, n_blocks)`` (one-hot encodings).
When ``initial_state`` is omitted, graph problems use a problem-specific default and
QUBO/HUBO problems default to :class:`~divi.qprog.algorithms.SuperpositionState`.
Using :class:`~divi.qprog.algorithms.WState` selects the XY mixer automatically so the
state stays in the one-hot subspace.

**Initial parameters:** Pass ``initial_params`` to ``run()`` to warm-start from
known parameters or continue from another run. See
:ref:`variational-run-controls` for the shared run controls.

Trotterization Strategies
-------------------------

The default :class:`~divi.hamiltonians.ExactTrotterization` applies every cost
term. :class:`~divi.hamiltonians.QDrift` samples terms for shallower circuits at
the cost of averaging more circuits.

QDrift controls:

- **keep_fraction**: Deterministically keep the top fraction of terms by coefficient magnitude
- **sampling_budget**: Number of terms to sample from the remaining Hamiltonian
- **n_hamiltonians_per_iteration**: Multiple samples per cost evaluation; losses are averaged
- **sampling_strategy**: ``"uniform"`` or ``"weighted"`` (by coefficient magnitude)

Assuming ``QAOA`` is imported and ``problem`` is defined, pass it as
``trotterization_strategy``:

.. skip: next

.. code-block:: python

   from divi.hamiltonians import QDrift

   qdrift = QDrift(
       keep_fraction=0.2,
       sampling_budget=5,
       n_hamiltonians_per_iteration=3,
       sampling_strategy="weighted",
       seed=1997,
   )
   qaoa = QAOA(problem, trotterization_strategy=qdrift, ...)

.. note::

   Final sampling draws fresh Hamiltonian terms, so ``best_probs`` and
   ``get_top_solutions`` are stochastic. A seed makes runs reproducible, but
   ``run()`` and a later ``sample_solution()`` still use different samples.

For a full comparison of Exact Trotterization vs QDrift (including circuit depth and
count), see the `qaoa_qdrift.py
<https://github.com/QoroQuantum/divi/blob/main/tutorials/optimization/qaoa_qdrift.py>`_
tutorial.

.. tip::

   On sampling backends, pass ``shot_distribution="weighted"`` to focus the
   cost Hamiltonian's shot budget on its dominant terms.  See
   :ref:`adaptive-shot-allocation` for the full list of strategies.

Graph Problems
--------------

Built-in graph problems include:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Problem class
     - Description
   * - :class:`~divi.qprog.problems.MaxCutProblem`\ ``(graph)``
     - Divides a graph into two subsets to maximise the sum of edge weights between them.
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

.. dashboard-example: max-clique

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

Divi's QAOA solver can also handle Quadratic Unconstrained Binary Optimisation (QUBO) problems. Divi currently supports three ways to build a :class:`~divi.qprog.problems.BinaryOptimizationProblem`:

1. **NumPy array** — pass a :class:`numpy.ndarray` or a :mod:`scipy.sparse` matrix directly
2. **Dimod BQM** — use ``dimod`` to construct a :class:`dimod.BinaryQuadraticModel`
3. **Nested list** — pass a Python list (converted to a NumPy array internally)

For matrix and nested-list inputs, the solution is a binary
:class:`numpy.ndarray`. A labelled BQM preserves its labels and returns a mapping.

NumPy Array-based Input
^^^^^^^^^^^^^^^^^^^^^^^

.. dashboard-example: qubo-numpy

.. code-block:: python

   import numpy as np
   import dimod
   from divi.backends import MaestroSimulator
   from divi.qprog import QAOA
   from divi.qprog.problems import BinaryOptimizationProblem
   from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer

   # Generate a random QUBO
   bqm = dimod.generators.randint(5, vartype="BINARY", low=-10, high=10, seed=1997)
   qubo_array = bqm.to_numpy_matrix()

   qaoa_problem = QAOA(
       BinaryOptimizationProblem(qubo_array),
       n_layers=2,
       optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
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

Pass a :class:`dimod.BinaryQuadraticModel` to the same
``BinaryOptimizationProblem`` constructor. Variable labels are preserved, so a
string-labelled BQM returns a mapping such as ``{"w": 0, "x": 1, "y": 0}``
instead of an array. Evaluate it directly with ``bqm.energy(qaoa.solution)``.

.. dashboard-example: bqm

.. code-block:: python

   import dimod

   from divi.backends import MaestroSimulator
   from divi.qprog import QAOA
   from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer
   from divi.qprog.problems import BinaryOptimizationProblem

   bqm = dimod.BinaryQuadraticModel(
       {"w": 10, "x": -3, "y": 2},
       {("w", "x"): -1, ("x", "y"): 1},
       vartype=dimod.Vartype.BINARY,
   )
   qaoa = QAOA(
       BinaryOptimizationProblem(bqm),
       n_layers=2,
       optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
       max_iterations=10,
       backend=MaestroSimulator(),
   )
   qaoa.run()

   print(f"Solution: {qaoa.solution}")
   print(f"BQM energy: {bqm.energy(qaoa.solution)}")

Pauli Correlation Encoding (PCE)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:class:`~divi.qprog.algorithms.PCE` solves the same
:class:`~divi.qprog.problems.BinaryOptimizationProblem` through a compressed
parity encoding. ``encoding_type="dense"`` needs
:math:`\lceil\log_2(N + 1)\rceil` qubits for *N* variables;
``encoding_type="poly"`` chooses the smallest *q* for which
:math:`q(q + 1)/2 \geq N`. Dense encoding minimises qubit count, while the
polynomial encoding uses lower-weight one- and two-qubit correlations.

``alpha`` controls the objective used during optimisation. Values below ``5``
use a smooth parity relaxation; values at or above ``5`` use a discrete,
CVaR-style objective over sampled energies. Start with the default ``2.0``
unless you specifically need the harder discrete objective.

.. dashboard-example: pce

.. code-block:: python

   import numpy as np

   from divi.backends import MaestroSimulator
   from divi.qprog import PCE
   from divi.qprog.optimizers import MonteCarloOptimizer
   from divi.qprog.problems import BinaryOptimizationProblem

   qubo = np.array([[-1.0, 2.0], [0.0, 1.0]])
   pce = PCE(
       problem=BinaryOptimizationProblem(qubo),
       encoding_type="dense",
       alpha=2.0,
       n_layers=2,
       optimizer=MonteCarloOptimizer(population_size=8),
       max_iterations=5,
       backend=MaestroSimulator(shots=2000),
       seed=42,
   )
   pce.run()

   best = pce.get_top_solutions(
       n=1, include_decoded=True, sort_by="energy"
   )[0]
   print(f"Best solution: {best.decoded} (energy={best.energy})")
   print(f"Qubits: {pce.n_qubits} for {pce.n_vars} variables")

HUBO Problems
-------------

Divi's QAOA solver supports Higher-Order Binary Optimisation (HUBO) problems —
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

.. dashboard-example: hubo

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
a weighted graph, it finds a set of edges that maximises total weight while
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

   ensemble.run()
   matching, weight = ensemble.aggregate_results()
   print(f"Matching: {matching}, weight: {weight}")

The partitioned workflow splits the graph by edges using Kernighan-Lin or spectral
bisection, solves each partition independently, stitches results via beam search,
and optionally fills unmatched residual nodes using classical
:func:`~networkx.algorithms.matching.max_weight_matching`.

Iterative QAOA
--------------

Standard QAOA uses random initialisation at a fixed circuit depth.
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

.. dashboard-example: iterative-qaoa

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

.. dashboard-example: optimization

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
   ensemble.run()

   # Aggregate results from all partitions
   quantum_solution, energy = ensemble.aggregate_results()

   print(f"MaxCut value: {energy}")
   print(f"Total circuits executed: {ensemble.total_circuit_count}")

QUBO Partitioning (QAOA or PCE)
-------------------------------

.. note::

   Everything in this section — D-Wave's ``EnergyImpactDecomposer`` and
   ``SplatComposer``, Divi's
   :class:`~divi.qprog.problems.CommunityDecomposer`, and the ``decomposer``
   argument itself — comes from the D-Wave ``hybrid`` package, which ships in
   the ``qubo-decompose`` extra:
   ``pip install "qoro-divi[qubo-decompose]"``. Unpartitioned QUBO and HUBO
   problems work on the core install — see :ref:`optional-extras`.

For large QUBO problems, use :class:`~divi.qprog.workflows.PartitioningProgramEnsemble` with a
:class:`~divi.qprog.problems.BinaryOptimizationProblem` configured with D-Wave's hybrid decomposer/composer.
You can choose the per-partition engine via ``quantum_routine``:

- ``quantum_routine="qaoa"`` (default): standard :class:`~divi.qprog.algorithms.QAOA` partitions.
- ``quantum_routine="pce"``: :class:`~divi.qprog.algorithms.PCE` partitions (supports PCE-specific kwargs such as ``encoding_type`` and ``alpha``).
- ``quantum_routine="iterative_qaoa"``: :class:`~divi.qprog.algorithms.IterativeQAOA` partitions with warm-started depth progression.
  Pass ``strategy``, ``max_iterations_per_depth``, and other IterativeQAOA-specific kwargs
  directly; ``n_layers`` is used as ``max_depth``.

The examples share one BQM and setup; only routine-specific problem and
ensemble arguments change:

.. code-block:: python

   import dimod
   import hybrid
   from qiskit.circuit.library import RYGate, RZGate
   from divi.qprog import InterpolationStrategy
   from divi.qprog.problems import BinaryOptimizationProblem
   from divi.qprog.workflows import PartitioningProgramEnsemble
   from divi.qprog.algorithms import GenericLayerAnsatz
   from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer
   from divi.backends import MaestroSimulator

   def run_partitioned(ensemble):
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
       ansatz=GenericLayerAnsatz([RYGate, RZGate]),
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
behaviour, see :doc:`../execution_workflows/program_ensembles`.

Structure-Aware Partitioning (Community Decomposer)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Instead of the hybrid ``EnergyImpactDecomposer``, pass a
:class:`~divi.qprog.problems.CommunityDecomposer` — a drop-in ``hybrid``
decomposer that splits the QUBO by the *community structure* of its interaction
graph, grouping strongly-coupled variables and cutting weak couplings so little
energy is lost at partition boundaries. This helps most on problems with community
structure; ``EnergyImpactDecomposer`` remains a fine default for featureless (dense,
unstructured) QUBOs.

Key parameters:

- ``max_cluster_size`` — the maximum number of variables in any partition (the
  per-partition qubit budget), analogous to ``EnergyImpactDecomposer(size=...)``.
- ``min_clusters`` — a floor on the number of partitions produced. At least one of
  ``max_cluster_size`` / ``min_clusters`` must be given.
- ``method`` — ``"modularity"`` (default: Louvain community detection, the
  strongest general-purpose choice across structured, dense, and constrained
  QUBOs) or ``"spectral"`` (signed multi-view spectral clustering, which respects
  coupling signs; best on sparse-geometric instances). With the ``local_search``
  polish the two reach comparable quality.

``local_search=True`` is a :class:`~divi.qprog.problems.BinaryOptimizationProblem`
option (not part of the decomposer): it adds a greedy single-bit-flip polish of each
aggregated solution and improves results with **any** decomposer, including
``EnergyImpactDecomposer``.

.. code-block:: python

   from divi.qprog.problems import CommunityDecomposer

   problem = BinaryOptimizationProblem(
       large_bqm,
       decomposer=CommunityDecomposer(max_cluster_size=5),  # method="modularity"
       composer=hybrid.SplatComposer(),
       local_search=True,
   )
   ensemble = PartitioningProgramEnsemble(
       problem=problem,
       n_layers=2,
       optimizer=optimizer,
       max_iterations=10,
       backend=backend,
   )
   sol_community, energy_community = run_partitioned(ensemble)

Why Partition?
--------------

Quantum hardware is limited in the number of qubits and circuit depth. For large problems:

- A monolithic QAOA circuit may exceed the available qubit or depth budget.
- Partitioned QAOA trades global optimality for scalability and parallel execution.
- It enables fast, approximate solutions using many small quantum jobs rather than one large one.

Next Steps
----------

- `tutorials/optimization/ <https://github.com/QoroQuantum/divi/tree/main/tutorials/optimization>`_ — QAOA/PCE/partitioning examples: ``qubo_qaoa_vs_pce.py``, ``qaoa_graph_problems.py``, ``qaoa_partitioning.py``, ``qaoa_hubo.py``, ``qaoa_qdrift.py``, ``iterative_qaoa.py``; routing in `tutorials/routing/ <https://github.com/QoroQuantum/divi/tree/main/tutorials/routing>`_ (``ce_qaoa_routing.py``)
- :doc:`routing` — TSP and CVRP with constraint-preserving encodings
- :doc:`../execution_workflows/optimizers` — optimizer selection and tuning
- :doc:`../execution_workflows/backends` — simulators and services
- :doc:`../tools/qubo_characterization` — diagnose a QUBO and find good
  ``γ`` / ``β`` before running QAOA
- :doc:`../api_reference/qprog/problems` — full :class:`~divi.qprog.problems.QAOAProblem` API
