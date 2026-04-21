Ground-State Energy Estimation with VQE
=======================================

The Variational Quantum Eigensolver (VQE) is a quantum algorithm for finding the ground state energy of quantum systems, particularly useful for quantum chemistry applications.

For our :class:`~divi.qprog.algorithms.VQE` implementation, we integrate tightly with PennyLane's `qchem <https://docs.pennylane.ai/en/stable/code/qml_qchem.html>`_ module and their Hamiltonian objects. As such, the :class:`~divi.qprog.algorithms.VQE` constructor accepts either a `Molecule <https://docs.pennylane.ai/en/stable/code/api/pennylane.qchem.Molecule.html>`_ object, out of which the molecular Hamiltonian is generated, or the Hamiltonian itself.

This page covers single-instance ground-state energy estimation with
:class:`~divi.qprog.algorithms.VQE` and large-scale sweeps with
:class:`~divi.qprog.workflows.VQEHyperparameterSweep`.

.. tip::

   On sampling backends, pass ``shot_distribution="weighted"`` to focus the
   same shot budget on the Hamiltonian's dominant terms — free variance
   reduction on the skewed coefficient distributions typical of chemistry.
   See `Spending Shots Where They Matter`_ below.

Basic :class:`~divi.qprog.algorithms.VQE` Usage
-----------------------------------------------

Here's how to set up a basic :class:`~divi.qprog.algorithms.VQE` calculation for the H2 molecule:

.. code-block:: python

   import numpy as np
   import pennylane as qml
   from divi.qprog import VQE, HartreeFockAnsatz
   from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer
   from divi.backends import MaestroSimulator

   # Create H2 molecule
   mol = qml.qchem.Molecule(
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

Hamiltonian Input
^^^^^^^^^^^^^^^^^

In the case of a Hamiltonian input, the input would be passed to the constructor as follows:

.. code-block:: python

   import numpy as np
   import pennylane as qml
   from divi.qprog import VQE, HartreeFockAnsatz
   from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer
   from divi.backends import MaestroSimulator

   mol = qml.qchem.Molecule(
       symbols=["H", "H"],
       coordinates=np.array([(0, 0, 0), (0, 0, 0.5)])
   )
   ham, _ = qml.qchem.molecular_hamiltonian(mol)

   vqe_problem = VQE(
       hamiltonian=ham,
       n_electrons=mol.n_electrons,
       ansatz=HartreeFockAnsatz(),
       n_layers=2,
       optimizer=ScipyOptimizer(method=ScipyMethod.L_BFGS_B),
       max_iterations=10,
       backend=MaestroSimulator(),
   )

In the case where the input is a Hamiltonian, the number of electrons present in the given system must be provided when the chosen ansatz is UCCSD or Hartree-Fock.

Initial Parameters
^^^^^^^^^^^^^^^^^^

Setting good initial parameters can significantly improve VQE convergence and prevent getting trapped in local minima. This is particularly useful for:

- **Molecular dissociation curves**: Use optimal parameters from previous bond lengths as starting points
- **Parameter sweeps**: Initialize from known good parameter regions
- **Restarting failed optimizations**: Use parameters from partial convergence

You can set initial parameters by passing ``initial_params`` to ``run()``. For detailed information and examples, see the :doc:`core_concepts` guide on Parameter Management.

Available Ansätze
-----------------

Divi provides several built-in ansätze for VQE calculations. For detailed documentation of each ansatz class, see the :doc:`/api_reference/qprog/algorithms` page.

Custom Ansätze
^^^^^^^^^^^^^^

Implement a custom ansatz by subclassing the abstract :class:`~divi.qprog.algorithms.Ansatz` class and providing two methods — it will then plug directly into Divi's execution routine:

.. skip: next

.. code-block:: python

   class Ansatz(ABC):
       """Abstract base class for all VQE ansätze."""

       @property
       def name(self) -> str:
           """Returns the human-readable name of the ansatz."""
           return self.__class__.__name__

       @staticmethod
       @abstractmethod
       def n_params_per_layer(n_qubits: int, **kwargs) -> int:
           """Returns the number of parameters required by the ansatz for one layer."""
           raise NotImplementedError

       @abstractmethod
       def build(self, params, n_qubits: int, n_layers: int, **kwargs):
           """
           Builds the ansatz circuit.

           Args:
               params (array): The parameters (weights) for the ansatz.
               n_qubits (int): The number of qubits.
               n_layers (int): The number of layers.
               **kwargs: Additional arguments like n_electrons for chemistry ansätze.
           """
           raise NotImplementedError

The ``build`` method must return a list of PennyLane operations
(``list[qml.operation.Operator]``). Refer to the built-in ansätze in the
repository for concrete examples.

VQE Hyperparameter Sweep
------------------------

By sweeping over physical parameters like bond length and varying the ansatz, this mode enables large-scale quantum chemistry simulations — efficiently distributing the workload across cloud or hybrid backends.

This mode is particularly useful for the study **molecular behavior** and **reaction dynamics**. It also allows one to compare **ansatz performance** and **optimizer robustness**. All through a single class!

Configuring the Molecular Transformations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Divi uses `Z-matrices <https://en.wikipedia.org/wiki/Z-matrix_(chemistry)>`_ to correctly and accurately modify molecules according to the users needs. These modifications can be declared and configured using the :class:`~divi.qprog.workflows.MoleculeTransformer` class, which takes as input the base molecule onto which the transformations are applied. Additionally, these arguments are used to define the specifics of the modifications:

- **atom_connectivity**: The connectivity structure of the molecule, provided as a list of tuples of indices of the atoms that have a bond between them. When not provided, the molecule would be assumed to have a chain structure (i.e. the connectivity would look like ``[(0, 1), (1, 2), ...]``).

- **bonds_to_transform**: A subset of the bonds listed in ``atom_connectivity`` to be modified. If this argument is not provided, all bonds will be affected.

- **bond_modifiers**: A list of actual numeric changes to apply to the chosen bonds. This has two modes: ``scale`` and ``delta``. If the provided list contains only strictly positive values, ``scale`` mode will be activated, where the values represent a multiplier to apply to the original bond length. Otherwise, the ``delta`` mode is enabled, where the provided values act as additives to the original bond length. Include the base molecule in the sweep by providing ``1`` in ``scale`` mode or ``0`` in ``delta`` mode.

- **alignment_atoms**: For debugging purposes, the output molecules can be aligned using `Kabsch algorithm <https://en.wikipedia.org/wiki/Kabsch_algorithm>`_, where users provide a list of indices of reference atoms that act as the "spine" of the whole molecule. An example of such would be the carbon chain of an alkane group.

.. code-block:: python

   from divi.qprog import VQEHyperparameterSweep, MoleculeTransformer
   from divi.qprog.optimizers import MonteCarloOptimizer
   import pennylane as qml
   import numpy as np
   from divi.qprog import HartreeFockAnsatz, UCCSDAnsatz
   from divi.backends import MaestroSimulator

   mol = qml.qchem.Molecule(
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
   vqe_sweep.create_programs()
   vqe_sweep.run(blocking=True)
   vqe_sweep.aggregate_results()

   # Visualize results
   vqe_sweep.visualize_results()

   print(f"Total circuits executed: {vqe_sweep.total_circuit_count}")

A few details worth calling out:

- **Bond modifiers** — with the values above the sweep contracts all bonds by
  -0.4 Å and -0.25 Å, stretches them by 0.25 Å and 0.4 Å, and also runs the
  base molecule unchanged (the ``0`` entry).
- **Ansatz comparison** — passing two ansätze runs every bond-modifier point
  under both :class:`~divi.qprog.algorithms.HartreeFockAnsatz` and
  :class:`~divi.qprog.algorithms.UCCSDAnsatz`, so you can compare accuracy
  head-to-head across the full curve.
- **Execution model** — ``run(blocking=True)`` dispatches all VQE programs,
  potentially in parallel, and blocks the script until every one of them
  finishes before returning.

.. tip::

   When using a sampling backend (e.g. ``QiskitSimulator`` or ``QoroService``
   without native expval), pass ``grouping_strategy="qwc"`` (the default) or
   ``"wires"`` to control how multi-term Hamiltonians are split into
   compatible measurement groups.  Backends like ``MaestroSimulator`` compute
   expectation values directly from the state representation, so measurement
   grouping has no effect and is overridden with a warning.

Spending Shots Where They Matter
--------------------------------

For chemistry Hamiltonians the coefficient distribution is typically highly
skewed — a handful of dominant terms account for most of the energy, while
many small terms contribute fractions of a millihartree.  Sampling every
measurement group with the full shot count wastes precision on those small
terms.

Pass ``shot_distribution`` to focus the same total budget on the groups
that matter:

.. code-block:: python

   from divi.qprog import VQE
   from divi.backends import QiskitSimulator

   vqe = VQE(
       molecule=molecule,
       ansatz=UCCSDAnsatz(),
       optimizer=mc_optimizer,
       backend=QiskitSimulator(shots=2000),
       grouping_strategy="qwc",
       shot_distribution="weighted",  # focus shots on dominant terms
   )

The ``"weighted"`` strategy allocates shots proportional to each group's
coefficient L1 norm.  See the :ref:`adaptive-shot-allocation` section of
the pipelines guide for the full list of strategies (``"uniform"``,
``"weighted"``, ``"weighted_random"``, or a custom callable) and their
trade-offs, including the bias-vs-budget implications when
small-coefficient groups end up with zero allocated shots.

This option requires sampling-based execution.  On expval-capable
backends like :class:`~divi.backends.MaestroSimulator`, divi auto-selects
an analytical strategy that ignores shots entirely; pass
``grouping_strategy="qwc"`` (or ``"wires"`` / ``None``) explicitly to opt
into sampling and unlock ``shot_distribution``.

Next Steps
----------

- Try the runnable tutorials in the `tutorials/ <https://github.com/QoroQuantum/divi/tree/main/tutorials>`_ directory
- Learn about :doc:`optimizers` for optimization strategies
- Explore :doc:`improving_results_qem` for error mitigation
- Save and resume long runs with :doc:`resuming_long_runs`
- Visualize the loss landscape with :doc:`visualization`
