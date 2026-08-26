Ground-State Energy Estimation with VQE
=======================================

The Variational Quantum Eigensolver (VQE) estimates a ground-state energy by
optimising a parameterised state against a Hamiltonian. Divi accepts the
problem in the ecosystem where it already lives:

.. list-table:: VQE inputs
   :header-rows: 1
   :widths: 20 38 27 15

   * - Argument
     - Accepted object
     - What Divi does
     - Extra
   * - ``molecule``
     - PennyLane ``qchem.Molecule``
     - Builds the molecular Hamiltonian automatically
     - Default install
   * - ``molecule``
     - PySCF ``gto.Mole`` or restricted mean-field object
     - Runs or reuses RHF, then builds the Hamiltonian
     - ``chem``
   * - ``hamiltonian``
     - PennyLane operator or Qiskit ``SparsePauliOp``
     - Uses the supplied qubit Hamiltonian directly
     - Default install
   * - ``hamiltonian``
     - OpenFermion ``QubitOperator``
     - Converts it to Divi's internal operator form
     - ``chem``

Install chemistry integrations with ``pip install qoro-divi[chem]``.

This page covers single-instance ground-state energy estimation with
:class:`~divi.qprog.algorithms.VQE` and large-scale sweeps with
:class:`~divi.qprog.workflows.VQEHyperparameterSweep`.

On sampling backends, see `Spending Shots Where They Matter`_ before assigning
a large measurement budget.

Basic :class:`~divi.qprog.algorithms.VQE` Usage
-----------------------------------------------

Here's how to set up a basic :class:`~divi.qprog.algorithms.VQE` calculation for the H2 molecule:

.. dashboard-example: vqe-basic

.. code-block:: python

   import numpy as np
   import pennylane as qp
   from divi.qprog import VQE, HartreeFockAnsatz
   from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer
   from divi.backends import MaestroSimulator

   # Create H2 molecule
   mol = qp.qchem.Molecule(
       symbols=["H", "H"],
       coordinates=np.array([[0.0, 0.0, -0.6614], [0.0, 0.0, 0.6614]])
   )

   # Create VQE program
   vqe_problem = VQE(
       molecule=mol,
       ansatz=HartreeFockAnsatz(),
       n_layers=2,
       optimizer=ScipyOptimizer(method=ScipyMethod.L_BFGS_B),
       max_iterations=10,
       backend=MaestroSimulator(),
   )

   # Run optimization
   vqe_problem.run()

   # Get results
   print(f"Ground state energy: {vqe_problem.best_loss:.6f} Ha")
   print(f"Total circuits executed: {vqe_problem.total_circuit_count}")

   # Analyze probability distribution of eigenstates
   top_eigenstates = vqe_problem.get_top_solutions(n=5)
   print("\nTop 5 eigenstates by probability:")
   for i, sol in enumerate(top_eigenstates, 1):
       print(f"{i}. {sol.bitstring}: {sol.prob:.2%}")

PySCF and OpenFermion Inputs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Choose PennyLane, PySCF, or OpenFermion based on where your molecule or
Hamiltonian already lives — Divi does not require converting between
ecosystems before handing an object to :class:`~divi.qprog.algorithms.VQE`.

Pass a PySCF ``gto.Mole`` or restricted mean-field object as ``molecule=``.
Divi runs or reuses RHF and builds the Hamiltonian through
:func:`~divi.hamiltonians.molecular_hamiltonian_from_pyscf`. This requires the
``chem`` extra.

OpenFermion ``QubitOperator`` inputs also require ``chem``. OpenFermion qubit
``q`` maps to Divi circuit qubit ``q``. Convert explicitly when the target
register is wider than the operator support:

.. skip: next

.. code-block:: python

   from openfermion import QubitOperator
   from divi.hamiltonians import qubit_operator_to_spo
   qop = QubitOperator("Z0 Z1", 1.0) + QubitOperator("X0", 0.5)
   ham = qubit_operator_to_spo(qop, n_qubits=4)

Hamiltonian Input
^^^^^^^^^^^^^^^^^

Pass a PennyLane or Qiskit Hamiltonian as ``hamiltonian=``. Chemistry ansätze
such as UCCSD and Hartree–Fock also require ``n_electrons``.

Initial Parameters
^^^^^^^^^^^^^^^^^^

Pass ``initial_params`` to ``run()`` to warm-start from known parameters or
continue a geometry sweep or interrupted optimisation. See
:ref:`variational-run-controls`.

Initial State
^^^^^^^^^^^^^

By default VQE prepares the ``|0...0>`` reference (``ZerosState()``).  Pass a
different :class:`~divi.qprog.algorithms.InitialState` via the ``initial_state``
constructor argument — e.g. ``ZerosState()``, ``OnesState()``,
``SuperpositionState()``, ``WState()``, or ``CustomPerQubitState(...)``.  The
chemistry ansätze (Hartree-Fock, UCCSD, QCC) embed their own reference-state
preparation, so combining them with a non-``ZerosState`` initial state prepends
extra operators and emits a :class:`UserWarning` (it can produce unphysical
circuits); custom or hardware-efficient ansätze are the usual place to set it.

Available Ansätze
-----------------

Use a chemistry ansatz when its reference-state assumptions match the problem;
use a hardware-efficient ansatz when circuit shape matters more than a
chemistry-derived parameterisation. See
:doc:`/api_reference/qprog/algorithms` for constructor details.

.. list-table:: Ansatz selection
   :header-rows: 1
   :widths: 28 72

   * - Ansatz
     - Typical use
   * - :class:`~divi.qprog.algorithms.HartreeFockAnsatz`
     - Minimal chemistry baseline and reference-state preparation.
   * - :class:`~divi.qprog.algorithms.UCCSDAnsatz`
     - Chemistry calculations where excitation-based structure is useful.
   * - :class:`~divi.qprog.algorithms.QCCAnsatz`
     - Qubit-space chemistry with selected entanglers.
   * - :class:`~divi.qprog.algorithms.GenericLayerAnsatz`
     - Custom hardware-efficient gate sequences and connectivity.

Custom Ansätze
^^^^^^^^^^^^^^

A custom :class:`~divi.qprog.algorithms.Ansatz` implements
``n_params_per_layer`` and ``build``; ``build`` returns the PennyLane operations
for all requested layers. Use :doc:`/api_reference/qprog/algorithms` for the
full abstract contract and the built-in implementations as concrete examples.

.. important::

   Gradient-based optimizers differentiate the ansatz with the two-term
   ``±π/2`` parameter-shift rule, which is exact only when each parameter drives
   a single gate whose generator has two distinct eigenvalues. Override
   :meth:`~divi.qprog.algorithms.Ansatz.parameter_frequencies` when that does not
   hold — when a parameter appears in more than one gate, or when a gate's
   generator has a richer spectrum, as excitation gates do. It returns one
   ``(omega, order)`` pair per parameter of a single layer, declaring that the
   energy carries frequencies ``{omega, ..., order * omega}`` in that parameter.
   Declaring more frequencies than the ansatz actually carries is safe (it only
   costs extra circuit evaluations); declaring too few returns a wrong gradient
   with no error.

VQE Hyperparameter Sweep
------------------------

Use :class:`~divi.qprog.workflows.VQEHyperparameterSweep` when the same VQE
study spans several molecular geometries, ansatzes, or optimizer settings. It
builds the Cartesian product of those choices, runs each configuration, and
aggregates the results for comparison. Typical uses include tracing a bond
dissociation curve and testing how robust an ansatz or optimizer remains as the
geometry changes.

Configuring the Molecular Transformations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Divi uses `Z-matrices <https://en.wikipedia.org/wiki/Z-matrix_(chemistry)>`_ to correctly and accurately modify molecules according to the users needs. These modifications can be declared and configured using the :class:`~divi.qprog.workflows.MoleculeTransformer` class, which takes as input the base molecule onto which the transformations are applied. Additionally, these arguments are used to define the specifics of the modifications:

- **atom_connectivity**: The connectivity structure of the molecule, provided as a list of tuples of indices of the atoms that have a bond between them. When not provided, the molecule would be assumed to have a chain structure (i.e. the connectivity would look like ``[(0, 1), (1, 2), ...]``).

- **bonds_to_transform**: A subset of the bonds listed in ``atom_connectivity`` to be modified. If this argument is not provided, all bonds will be affected.

- **bond_modifiers**: A list of actual numeric changes to apply to the chosen bonds. This has two modes: ``scale`` and ``delta``. If the provided list contains only strictly positive values, ``scale`` mode will be activated, where the values represent a multiplier to apply to the original bond length. Otherwise, the ``delta`` mode is enabled, where the provided values act as additives to the original bond length. Include the base molecule in the sweep by providing ``1`` in ``scale`` mode or ``0`` in ``delta`` mode.

- **alignment_atoms**: For debugging purposes, the output molecules can be aligned using `Kabsch algorithm <https://en.wikipedia.org/wiki/Kabsch_algorithm>`_, where users provide a list of indices of reference atoms that act as the "spine" of the whole molecule. An example of such would be the carbon chain of an alkane group.

.. dashboard-example: chemistry

.. code-block:: python

   from divi.qprog import VQEHyperparameterSweep, MoleculeTransformer
   from divi.qprog.optimizers import MonteCarloOptimizer
   import pennylane as qp
   import numpy as np
   from divi.qprog import HartreeFockAnsatz, UCCSDAnsatz
   from divi.backends import MaestroSimulator

   mol = qp.qchem.Molecule(
       symbols=["H", "H"],
       coordinates=np.array([(0, 0, 0), (0, 0, 0.5)])
   )
   # Create molecule transformer for bond length variations
   transformer = MoleculeTransformer(
       base_molecule=mol,
       bond_modifiers=[-0.4, -0.25, 0, 0.25, 0.4]
   )

   # Set up Monte Carlo optimizer
   mc_optimizer = MonteCarloOptimizer(population_size=10, n_best_sets=3)

   # Create hyperparameter sweep
   vqe_sweep = VQEHyperparameterSweep(
       molecule_transformer=transformer,
       ansatze=[HartreeFockAnsatz(), UCCSDAnsatz()],
       optimizer=mc_optimizer,
       max_iterations=10,
       backend=MaestroSimulator(shots=5000),
   )

   # Execute sweep
   vqe_sweep.run()
   vqe_sweep.aggregate_results()

   # Visualize results
   vqe_sweep.visualize_results()

   print(f"Total circuits executed: {vqe_sweep.total_circuit_count}")

A few details worth calling out:

- **Bond modifiers** — with the values above the sweep contracts all bonds by
  -0.4 and -0.25 Bohr, stretches them by 0.25 and 0.4 Bohr, and also runs the
  base molecule unchanged (the ``0`` entry). Deltas are Bohr because that is
  the unit both PennyLane and PySCF store coordinates in.
- **Ansatz comparison** — passing two ansätze runs every bond-modifier point
  under both :class:`~divi.qprog.algorithms.HartreeFockAnsatz` and
  :class:`~divi.qprog.algorithms.UCCSDAnsatz`, so you can compare accuracy
  head-to-head across the full curve.
- **Execution model** — ``run()`` dispatches all VQE programs, potentially in
  parallel, and blocks the script until every one of them finishes before
  returning.

.. tip::

   When using a sampling backend (e.g. ``QiskitSimulator`` or ``QoroService``
   without native expval), pass ``grouping_strategy="qwc"`` (the default) or
   ``"wires"`` to control how multi-term Hamiltonians are split into
   compatible measurement groups.  Backends like ``MaestroSimulator`` compute
   expectation values directly from the state representation, so measurement
   grouping has no effect and is overridden with a warning.

   By default each measurement group only measures the qubits it acts on
   non-trivially — identity positions carry no information for the expectation
   value, so dropping them shrinks the shot histogram and the data returned by
   the backend.  Pass ``measure_all_qubits=True`` to measure the full register
   instead (for example, when you want the raw histograms of every qubit).

Spending Shots Where They Matter
--------------------------------

For chemistry Hamiltonians the coefficient distribution is typically highly
skewed. Under a fixed total budget, uniform allocation can oversample
low-weight groups and undersample dominant ones.

Continuing from the sweep setup above, focus the same total budget on dominant
groups:

.. code-block:: python

   from divi.qprog import VQE
   from divi.backends import QiskitSimulator

   vqe = VQE(
       molecule=mol,
       ansatz=UCCSDAnsatz(),
       optimizer=mc_optimizer,
       backend=QiskitSimulator(force_sampling=True, shots=2000),
       grouping_strategy="qwc",
       shot_distribution="weighted",  # focus shots on dominant terms
   )

The ``"weighted"`` strategy allocates shots proportional to each group's
coefficient L1 norm.  Note that ``shot_distribution`` only takes effect on a
sampling backend (``supports_expval=False``) — on an expval-native backend like
:class:`~divi.backends.MaestroSimulator` the result is computed analytically and
the allocation is ignored (with a :class:`UserWarning`).  The example above uses
:class:`~divi.backends.QiskitSimulator`, which samples.  See the
:ref:`adaptive-shot-allocation` section of the pipelines guide for the full list
of strategies, the ``force_sampling`` recipe, and the bias-vs-budget trade-offs.

Next Steps
----------

- Try the runnable tutorials in the `tutorials/ <https://github.com/QoroQuantum/divi/tree/main/tutorials>`_ directory
- Learn about :doc:`../execution_workflows/optimizers` for optimisation strategies
- Explore :doc:`improving_results_qem` for error mitigation
- Save and resume long runs with :doc:`../execution_workflows/resuming_long_runs`
- Visualise the loss landscape with :doc:`../execution_workflows/visualization`
