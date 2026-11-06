Routing Problems (TSP & CVRP)
=============================

Divi provides specialised :class:`~divi.qprog.problems.TSPProblem` and
:class:`~divi.qprog.problems.CVRPProblem` classes for solving routing
problems with QAOA.  Both are :class:`~divi.qprog.problems.QAOAProblem`
subclasses and work with the same ``QAOA`` constructor described in
:doc:`combinatorial_optimization_qaoa_pce`. With the default
``encoding="one_hot"``, they implement the
**Constraint-Enhanced QAOA** (CE-QAOA) protocol [#onah2025]_, which uses
block one-hot encoding with W-state initialisation and an XY mixer, which keep
the state inside the block one-hot subspace.

CE-QAOA Components
------------------

CE-QAOA co-designs the encoding, initial state, and mixer:

- **W-state initialisation**: starts each qubit block in a uniform
  superposition over one-hot basis states (exactly one excitation per block).
- **XY mixer**: all-to-all XY coupling within each block swaps
  excitations without creating or destroying them, preserving the
  one-hot constraint of every block.
- **Reduced phase operator**: the constraint a block enforces is not
  penalised. For TSP a block is one city's qubits across all time steps, so the
  per-step penalty remains. For CVRP a block is one customer's qubits across
  all (vehicle, step) slots, so the penalties on two customers sharing a slot
  and on an occupied step that follows an empty one remain. Capacity is not in
  the phase operator. ``TSPProblem`` uses the reduced operator for QAOA only;
  :class:`~divi.qprog.algorithms.PCE` and ``relaxed_energy`` use the full
  constraint set.

Travelling Salesman Problem (TSP)
---------------------------------

Given a symmetric cost matrix, find the shortest tour visiting every city
exactly once and returning to the start.

.. dashboard-example: tsp

.. code-block:: python

   import numpy as np
   from divi.qprog.algorithms import QAOA
   from divi.qprog.problems import TSPProblem
   from divi.qprog.optimizers import GridSearchOptimizer
   from divi.backends import MaestroSimulator

   cost = np.array([
       [0, 10, 15, 20],
       [10, 0, 35, 25],
       [15, 35, 0, 30],
       [20, 25, 30, 0],
   ])

   problem = TSPProblem(cost, start_city=0)
   backend = MaestroSimulator(shots=5000)

   qaoa = QAOA(
       problem,
       optimizer=GridSearchOptimizer(
           param_ranges=[(0, 3.14), (0, 3.14)],
           grid_points=10,
       ),
       max_iterations=1,
       backend=backend,
   )
   qaoa.run()
   print(qaoa.solution)  # decoded city sequence, returning to start_city

The ``start_city`` is fixed and excluded from the encoding, reducing the
problem from *n²* to *(n−1)²* qubits. ``penalty_weight`` and
``objective_weight`` weight the constraints and the objective. The default
``penalty_weight`` is ``objective_weight`` times the nearest-neighbour tour cost
(twice that for ``encoding="binary"``).

:class:`~divi.qprog.problems.TSPProblem` is a
:class:`~divi.qprog.problems.BinaryOptimizationProblem` whose objective is the
tour length:

- One-hot: ``problem.constraints`` lists the visit and time-step constraints,
  and :class:`~divi.qprog.algorithms.PCE` accepts the problem.
  :func:`~divi.backends.characterization.characterize_and_validate` receives
  all of them, including the per-city penalty that the QAOA cost omits.
- Binary: ``problem.constraints`` is empty, and the Hamiltonian omits the
  constant of each city's visit penalty.

The Hamiltonians are built on first use. The constructor copies
``cost_matrix`` and raises ``ValueError`` for a non-finite cost, fewer than
three cities, an unknown ``encoding`` or an out-of-range ``start_city``, and
``TypeError`` for a non-integer ``start_city``.

.. skip: next

.. code-block:: python

   from divi.qprog.algorithms import PCE

   pce = PCE(
       TSPProblem(cost),
       optimizer=optimizer,
       backend=backend,
   )

Partitioned TSP
^^^^^^^^^^^^^^^

For instances too large for one register, set ``max_cities_per_partition``.
:meth:`~divi.qprog.problems.TSPProblem.decompose` clusters the cities into
groups of at most that many (at least 3). Each group of three or more becomes
its own :class:`~divi.qprog.problems.TSPProblem`; smaller groups need no
qubits. A :class:`~divi.qprog.ensembles.PartitioningProgramEnsemble` runs the
groups and patches their closed tours into one tour, repeatedly merging the
pair whose edge exchange adds the least cost.

- One-hot: a sampled sub-solution that is not a valid tour is repaired
  (Hungarian), for QAOA and PCE alike.
- Binary: there is no repair; such a sub-solution is replaced by its cities in
  index order.
- ``quantum_routine="pce"`` runs :class:`~divi.qprog.algorithms.PCE` on the
  groups instead of QAOA.
- The cost matrix must be finite and symmetric, and at least one group needs
  three cities.
- Each group's default ``penalty_weight`` comes from its own costs; an explicit
  value goes to every group.

.. skip: next

.. code-block:: python

   from divi.qprog.ensembles import PartitioningProgramEnsemble

   problem = TSPProblem(cost, max_cities_per_partition=4)
   ensemble = PartitioningProgramEnsemble(
       problem=problem,
       n_layers=1,
       optimizer=optimizer,
       backend=backend,
   )
   ensemble.run()
   tour, tour_cost = ensemble.aggregate_results()

Capacitated Vehicle Routing (CVRP)
----------------------------------

Route multiple vehicles from a depot to customers, respecting vehicle
capacity constraints.

Continuing with the imports and backend from the TSP example:

.. code-block:: python

   from divi.qprog.problems import CVRPProblem

   cost_matrix = np.array([
       [0, 10, 15, 20],
       [10, 0, 35, 25],
       [15, 35, 0, 30],
       [20, 25, 30, 0],
   ])
   demands = np.array([0, 3, 4, 2])  # depot demand = 0

   problem = CVRPProblem(
       cost_matrix,
       demands=demands,
       capacity=10.0,
       n_vehicles=2,
       depot=0,
   )

   qaoa = QAOA(
       problem,
       optimizer=GridSearchOptimizer(
           param_ranges=[(0, 3.14), (0, 3.14)],
           grid_points=10,
       ),
       max_iterations=1,
       backend=backend,
   )
   qaoa.run()

With the one-hot encoding:

- Variable ``x_{i,v,t}`` (customer *i*, vehicle *v*, step *t*) sits on qubit
  ``i*V*T + v*T + t``, with *V* vehicles and *T* steps, where *T* equals the
  number of customers. The W-state and XY mixer act on each customer's block of
  *V·T* qubits.
- A route fills steps ``0..len-1`` and its return edge is charged at the last
  occupied step. ``penalty_weight`` weights the penalties for two customers
  sharing a slot and for a gap in a route.
- Capacity is enforced when solutions are checked (``is_feasible``), filtered
  and repaired, not by the phase operator.

The default ``penalty_weight`` is ``objective_weight`` times the sum of the
nearest-neighbour tour cost from the depot and ``max(n_customers - 2, 1)`` times
the largest cost entry for the one-hot encoding, and 4.0 for the binary
encoding. ``capacity_penalty_weight`` applies to the binary encoding only.

The constructor copies ``cost_matrix`` and ``demands`` and raises ``ValueError``
for a non-finite cost or demand, a negative demand, a nonzero depot demand, a
demand above ``capacity``, no customer, an unknown ``encoding``, an
out-of-range ``depot``, fewer than one vehicle, or a non-positive ``capacity``
or ``penalty_weight``, and ``TypeError`` for a non-integer ``depot`` or
``n_vehicles``. As with :class:`~divi.qprog.problems.TSPProblem`, the
Hamiltonians are built on first use.

Binary Encoding
^^^^^^^^^^^^^^^

The ``encoding`` argument selects between the two routing encodings:

.. list-table:: One-hot versus binary routing encodings
   :header-rows: 1
   :widths: 18 28 27 27

   * - Encoding
     - Qubit scaling per slot
     - Advantage
     - Cost
   * - ``"one_hot"``
     - :math:`O(N)`
     - W-state initialisation and XY mixing preserve the one-hot constraint.
     - Larger register than binary encoding.
   * - ``"binary"``
     - :math:`O(\log N)`
     - Smaller register than one-hot encoding.
     - Produces a HUBO with higher-order interactions; it does not preserve
       feasibility through the one-hot mixer.

For example, one-hot CVRP can require 1,600 qubits for 20 customers and four
vehicles. Binary encoding reduces the register size:

.. code-block:: python

   problem = CVRPProblem(
       cost_matrix,
       demands=demands,
       capacity=10.0,
       n_vehicles=2,
       encoding="binary",   # compact binary encoding
   )

This produces a HUBO that Divi maps natively to higher-order Z interactions.
See ``compute_binary_block_config`` for qubit-count estimates.

Feasibility, Repair, and Energy
-------------------------------

Both classes provide feasibility and energy checks; for CVRP they also enforce
vehicle capacity. Hungarian repair covers the one-hot encoding; binary routing
repair is unsupported. For CVRP, ``"filter"`` ranking merges bitstrings that
decode to the same routes. CVRP repair moves overflowing customers greedily and,
for at most ten customers, falls back to a search over capacity-feasible
splits. The search keeps the compact assignment agreeing with the most input
bits, then the lowest route cost.

Relevant methods include:

- :meth:`~divi.qprog.problems.TSPProblem.is_feasible` — check
  whether a bitstring represents a valid tour/route.
- :meth:`~divi.qprog.problems.TSPProblem.repair_infeasible_bitstring` — project an
  infeasible bitstring to the nearest valid solution using the
  **Hungarian algorithm** (``scipy.optimize.linear_sum_assignment``).
- :meth:`~divi.qprog.problems.TSPProblem.compute_energy` —
  evaluate the actual travel cost.

These are used by QAOA's ``get_top_solutions`` with the ``feasibility``
parameter:

.. code-block:: python

   # PHQC mode (arXiv:2511.14296, Algorithm 4): keep only feasible
   # solutions, rank by objective energy (not probability).
   solutions = qaoa.get_top_solutions(n=5, feasibility="filter")

   # Repair infeasible solutions before ranking by energy.
   solutions = qaoa.get_top_solutions(n=5, feasibility="repair")

Loading Benchmark Instances
---------------------------

The ``parse_tsplib_file`` utility reads standard TSPLIB/CVRPLIB ``.vrp``
files, as used by benchmarks like QOBLIB:

.. skip: next

.. code-block:: python

   from divi.qprog.problems import parse_tsplib_file, parse_vrp_solution

   inst = parse_tsplib_file("XSH-n20-k4-01.vrp")
   # inst.cost_matrix, inst.demands, inst.capacity, inst.n_vehicles, ...

   opt_routes, opt_cost = parse_vrp_solution("XSH-n20-k4-01.opt.sol")

Qubit Scaling
-------------

.. list-table:: Qubit count by encoding
   :header-rows: 1

   * - Customers
     - Vehicles
     - One-hot
     - Binary (full)
     - Binary (tight)
   * - 3
     - 2
     - 18
     - 12
     - 12
   * - 10
     - 3
     - 300
     - 120
     - 60
   * - 20
     - 4
     - 1,600
     - 400
     - 120
   * - 50
     - 10
     - 25,000
     - 3,000
     - 360

Binary (full) uses the default ``max_steps=n_customers``. Binary (tight) uses
``max_steps ≈ customers / vehicles + 1``; setting it below a vehicle's required
stop count makes the instance infeasible.

Next Steps
----------

- Run the ``tutorials/routing/ce_qaoa_routing.py`` tutorial
- Explore :doc:`../execution_workflows/optimizers`
- See :doc:`combinatorial_optimization_qaoa_pce` for general QAOA usage

References
----------

.. [#onah2025] Onah, C., Firt, R., & Michielsen, K. (2025).
   *Empirical Quantum Advantage in Constrained Optimization from Encoded
   Unitary Designs*. arXiv:2511.14296.
