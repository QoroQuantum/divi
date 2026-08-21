Routing Problems (TSP & CVRP)
=============================

Divi provides specialised :class:`~divi.qprog.problems.TSPProblem` and
:class:`~divi.qprog.problems.CVRPProblem` classes for solving routing
problems with QAOA.  Both are :class:`~divi.qprog.problems.QAOAProblem`
subclasses and work with the same ``QAOA`` constructor described in
:doc:`combinatorial_optimization_qaoa_pce`. With the default
``encoding="one_hot"``, they implement the
**Constraint-Enhanced QAOA** (CE-QAOA) protocol [#onah2025]_, which uses
block one-hot encoding with W-state initialisation and an XY mixer to keep
quantum amplitude concentrated on the feasible (permutation) subspace.

Why CE-QAOA?
------------

For routing problems like TSP and CVRP, standard penalty-QAOA wastes
most of its measurement shots on infeasible bitstrings — the feasible
set (e.g. valid tours, or permutations more generally) is an
exponentially small fraction of the full Hilbert space.  CE-QAOA avoids
this by co-designing the encoding, initial state, and mixer:

- **W-state initialisation**: starts each qubit block in a uniform
  superposition over one-hot basis states (exactly one city per slot).
- **XY mixer**: all-to-all XY coupling within each block swaps
  excitations without creating or destroying them, preserving the
  one-hot constraint with a constant spectral gap.
- **Reduced phase operator**: row constraints are enforced structurally,
  so only column and capacity penalties appear in the cost Hamiltonian.

Travelling Salesman Problem (TSP)
---------------------------------

Given a symmetric cost matrix, find the shortest tour visiting every city
exactly once and returning to the start.

.. code-block:: python

   import numpy as np
   from divi.qprog import QAOA
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
problem from *n²* to *(n−1)²* qubits.  Penalty strengths ``constraint_penalty``
(constraints) and ``objective_weight`` (objective) can be tuned.

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

Binary Encoding
^^^^^^^^^^^^^^^

Choose the encoding from the qubit budget and the importance of staying inside
the feasible subspace:

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
     - Becomes impractical for large routing instances.
   * - ``"binary"``
     - :math:`O(\log N)`
     - Substantially lowers the qubit count.
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
See ``binary_block_config`` for qubit-count estimates.

Feasibility, Repair, and Energy
-------------------------------

Both classes provide feasibility and energy checks. Hungarian repair is
available for one-hot encoding; binary routing repair is unsupported.

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
- Explore :doc:`../execution_workflows/optimizers` — ``GridSearchOptimizer`` is particularly
  suited for the 2-parameter CE-QAOA landscape
- See :doc:`combinatorial_optimization_qaoa_pce` for general QAOA usage

References
----------

.. [#onah2025] Onah, C., Firt, R., & Michielsen, K. (2025).
   *Empirical Quantum Advantage in Constrained Optimization from Encoded
   Unitary Designs*. arXiv:2511.14296.
