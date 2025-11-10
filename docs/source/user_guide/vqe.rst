VQE
===

The Variational Quantum Eigensolver (VQE) is a quantum algorithm for finding the ground state energy of quantum systems, particularly useful for quantum chemistry applications.

For our VQE implementation, we integrate tightly with PennyLane's `qchem <https://docs.pennylane.ai/en/stable/code/qml_qchem.html>`_ module and their Hamiltonian objects. As such, the VQE constructor accepts either a `Molecule <https://docs.pennylane.ai/en/stable/code/api/pennylane.qchem.Molecule.html>`_ object, out of which the molecular Hamiltonian is generated, or the Hamiltonian itself.

Divi offers two VQE modes: standard single-instance ground-state energy estimation, and hyperparameter sweep mode for large-scale simulations.

Basic VQE Usage
---------------

Here's how to set up a basic VQE calculation for the H2 molecule:

.. code-block:: python

   import numpy as np
   import pennylane as qml
   from divi.qprog import VQE, HartreeFockAnsatz
   from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer
   from divi.backends import ParallelSimulator

   # Create H2 molecule
   mol = qml.qchem.Molecule(
       symbols=["H", "H"],
       coordinates=np.array([[0.0, 0.0, -0.6614], [0.0, 0.0, 0.6614]])
   )

   # Create VQE program
   vqe_problem = VQE(
       molecule=mol,
       ansatz=HartreeFockAnsatz(),
       n_layers=1,
       optimizer=ScipyOptimizer(method=ScipyMethod.L_BFGS_B),
       max_iterations=50,
       backend=ParallelSimulator(),
   )

   # Run optimization
   vqe_problem.run()

   # Get results
   print(f"Ground state energy: {vqe_problem.best_loss:.6f} Ha")
   print(f"Total circuits executed: {vqe_problem.total_circuit_count}")

Hamiltonian Input
^^^^^^^^^^^^^^^^^

In the case of a Hamiltonian input, the input would be passed to the constructor as follows:

.. code-block:: python

   import numpy as np
   import pennylane as qml
   from divi.qprog import VQE, HartreeFockAnsatz
   from divi.backends import ParallelSimulator

   mol = qml.qchem.Molecule(
       symbols=["H", "H"],
       coordinates=np.array([(0, 0, 0), (0, 0, 0.5)])
   )
   ham, _ = qml.qchem.molecular_hamiltonian(mol)

   vqe_problem = VQE(
       hamiltonian=ham,
       n_electrons=mol.n_electrons,
       ansatz=HartreeFockAnsatz(),
       n_layers=1,
       optimizer=ScipyOptimizer(method=ScipyMethod.L_BFGS_B),
       max_iterations=50,
       backend=ParallelSimulator(),
   )

In the case where the input is a Hamiltonian, the number of electrons present in the given system must be provided when the chosen ansatz is UCCSD or Hartree-Fock.

Initial Parameters
^^^^^^^^^^^^^^^^^^

Setting good **initial parameters** can significantly improve VQE convergence and prevent getting trapped in local minima. This is particularly important for:

- **Molecular dissociation curves**: Use optimal parameters from previous bond lengths as starting points
- **Parameter sweeps**: Initialize from known good parameter regions
- **Restarting failed optimizations**: Use parameters from partial convergence

.. code-block:: python

   # Example: Using initial parameters for better convergence
   import numpy as np
   import pennylane as qml
   from divi.qprog import VQE, HartreeFockAnsatz
   from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer
   from divi.backends import ParallelSimulator

   mol = qml.qchem.Molecule(
       symbols=["H", "H"],
       coordinates=np.array([(0, 0, 0), (0, 0, 0.5)])
   )
   initial_params = np.random.uniform(-0.1, 0.1, (1, 4))
   vqe_problem = VQE(
       molecule=mol,
       ansatz=HartreeFockAnsatz(),
       n_layers=2,
       optimizer=ScipyOptimizer(method=ScipyMethod.L_BFGS_B),
       max_iterations=50,
       backend=ParallelSimulator(),
       initial_params=initial_params,
   )
   vqe_problem.run()

Available Ansätze
-----------------

Divi provides several built-in ansätze for VQE calculations. For detailed documentation of each ansatz class, see the :ref:`user_guide/vqe:Available Ansätze` section in the API reference.

Custom Ansätze
^^^^^^^^^^^^^^

One can easily implement their own Ansatz that would be immediately compatible with Divi's execution routine by inheriting the abstract `Ansatz` class and implementing two main methods:

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

The `build` function should contain PennyLane quantum operations for it to work properly. Refer to the definition of the other ansätze in our repository whenever in doubt.

VQE Hyperparameter Sweep
-------------------------

By sweeping over physical parameters like bond length and varying the ansatz, this mode enables large-scale quantum chemistry simulations — efficiently distributing the workload across cloud or hybrid backends.

This mode is particularly useful for the study **molecular behavior** and **reaction dynamics**. It also allows one to compare **ansatz performance** and **optimizer robustness**. All through a single class!

Configuring the Molecular Transformations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Divi uses `Z-matrices <https://en.wikipedia.org/wiki/Z-matrix_(chemistry)>`_ to correctly and accurately modify molecules according to the users needs. These modifications can be declared and configured using the `MoleculeTransformer` class, which takes as input the base molecule onto which the transformations are applied. Additionally, these arguments are used to define the specifics of the modifications:

- **atom_connectivity**: The connectivity structure of the molecule, provided as a list of tuples of indices of the atoms that have a bond between them. When not provided, the molecule would be assumed to have a chain structure (i.e. the connectivity would look like `[(0, 1), (1, 2), ...]`).

- **bonds_to_transform**: A subset of the bonds listed in `atom_connectivity` to be modified. If this argument is not provided, all bonds will be affected.

- **bond_modifiers**: A list of actual numeric changes to apply to the chosen bonds. This has two modes: `scale` and `delta`. If the provided list contains only strictly positive values, `scale` mode will be activated, where the values represent a multiplier to apply to the original bond length. Otherwise, the `delta` mode is enabled, where the provided values act as additives to the original bond length. One can trivially provide `1` and `0` for the `scale` and `delta` modes respectively to include the base molecule as an experiment.

- **alignment_atoms**: For debugging purposes, the output molecules can be aligned using `Kabsch algorithm <https://en.wikipedia.org/wiki/Kabsch_algorithm>`_, where users provide a list of indices of reference atoms that act as the "spine" of the whole molecule. An example of such would be the carbon chain of an alkane group.

.. code-block:: python

   from divi.qprog import VQEHyperparameterSweep, MoleculeTransformer
   from divi.qprog.optimizers import MonteCarloOptimizer
   import pennylane as qml
   import numpy as np
   from divi.qprog import HartreeFockAnsatz, UCCSDAnsatz
   from divi.backends import ParallelSimulator

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
   mc_optimizer = MonteCarloOptimizer(n_param_sets=10, n_best_sets=3)

   # Create hyperparameter sweep
   vqe_sweep = VQEHyperparameterSweep(
       molecule_transformer=transformer,
       ansatze=[HartreeFockAnsatz(), UCCSDAnsatz()],
       optimizer=mc_optimizer,
       max_iterations=25,
       backend=ParallelSimulator(shots=2000),
       grouping_strategy="wires"  # PennyLane's wire grouping strategy
   )

   # Execute sweep
   vqe_sweep.create_programs()
   vqe_sweep.run(blocking=True)
   vqe_sweep.aggregate_results()

   # Visualize results
   vqe_sweep.visualize_results()

   print(f"Total circuits executed: {vqe_sweep.total_circuit_count}")

What's Happening?
~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Step
     - Description
   * - ``VQEHyperparameterSweep(...)``
     - Initializes a batch of VQE programs over a range of bond lengths and ansatz strategies.
   * - ``molecule_transformer=...``
     - The transformer declaring the changes to apply to the molecule. In this instance, we are contracting all bonds by -0.4, -0.25 Å and stretching them by 0.25 and 0.4 Å, in addition to the base molecule.
   * - ``ansatze=[HartreeFockAnsatz(), UCCSDAnsatz()]``
     - Runs two different quantum circuit models for comparison.
   * - ``create_programs()``
     - Constructs all circuits for each (bond modifier, ansatz) pair.
   * - ``run(blocking=True)``
     - Executes all VQE circuits — possibly in parallel. Block the script until all programs finish executing.
   * - ``aggregate_results()``
     - Collects and merges the final energy values for plotting.
   * - ``visualize_results()``
     - Displays a graph of energy vs. bond length for each ansatz.

Why Parallelize VQE?
--------------------

- VQE is an iterative algorithm requiring multiple circuit evaluations per step.
- Sweeping over bond lengths and ansätze creates hundreds of circuits.
- Parallelizing execution reduces total compute time and helps saturate available QPU/GPU/CPU resources.

Next Steps
----------

- Try the runnable examples in the `tutorials/ <https://github.com/QoroQuantum/divi/tree/main/tutorials>`_ directory
- Learn about :doc:`optimizers` for optimization strategies
- Explore :doc:`error_mitigation` for improving results
