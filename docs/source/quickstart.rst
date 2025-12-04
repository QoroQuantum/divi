Quick Start Guide
=================

Welcome to Divi! This guide will get you up and running with quantum program execution in minutes.

What is Divi?
-------------

Divi is a powerful Python library that makes quantum computing accessible by automating the complex parts of quantum program development. Whether you're a researcher studying molecular systems or an engineer solving optimization problems, Divi handles the heavy lifting while you focus on your science.

**Core Capabilities:**

* 🚀 **Automated Execution**: Run quantum programs with minimal boilerplate
* 🔄 **Smart Parallelization**: Automatically optimize circuit execution across available resources
* 🎯 **Multiple Backends**: Seamless switching between simulators and real quantum hardware
* 🛡️ **Error Mitigation**: Built-in techniques to improve result accuracy
* 📊 **Progress Tracking**: Real-time feedback during long-running computations

Five-Minute Tutorial
--------------------

Let's solve a quantum chemistry problem - finding the ground state energy of a hydrogen molecule:

.. code-block:: python

   import numpy as np
   import pennylane as qml
   from divi.qprog import VQE, HartreeFockAnsatz
   from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer
   from divi.backends import ParallelSimulator

   # Step 1: Define your molecule
   h2_molecule = qml.qchem.Molecule(
      symbols=["H", "H"], coordinates=np.array([[0.0, 0.0, -0.6614], [0.0, 0.0, 0.6614]])
   )

   # Step 2: Choose your optimizer
   optimizer = ScipyOptimizer(method=ScipyMethod.COBYLA)

   # Step 3: Set up your quantum program
   vqe = VQE(
      molecule=h2_molecule,
      ansatz=HartreeFockAnsatz(),
      n_layers=2,  # Circuit depth
      optimizer=optimizer,
      max_iterations=10,  # Optimization steps
      backend=ParallelSimulator(shots=1000),  # Local simulator
   )

   # Step 4: Run and get results!
   vqe.run()

   # Check your results
   print(f"🎉 Ground state energy: {vqe.best_loss:.6f} Hartree")
   print(f"⚡ Circuits executed: {vqe.total_circuit_count}")

That's it! You just ran a variational quantum algorithm. The energy should be close to -1.137 Hartree (H₂'s true ground state energy).

Choosing the Right Algorithm
-----------------------------

Divi offers specialized algorithms for different problem types:

**VQE - Quantum Chemistry** ⚗️
   Perfect for molecular ground state calculations, dissociation curves, and electronic structure problems.

   .. code-block:: python

      from divi.qprog import VQE, UCCSDAnsatz

      vqe = VQE(
          molecule=h2_molecule,
          ansatz=UCCSDAnsatz(),  # More sophisticated than Hartree-Fock
          n_layers=3,
          backend=ParallelSimulator()
      )

**QAOA - Optimization Problems** 🎯
   Ideal for combinatorial optimization: Max-Cut, Max-Clique, traveling salesman, and similar NP-hard problems.

   .. code-block:: python

      import networkx as nx
      from divi.qprog import QAOA, GraphProblem

      # Create your problem graph
      graph = nx.erdos_renyi_graph(10, 0.5)

      qaoa = QAOA(
          problem=graph,
          graph_problem=GraphProblem.MAXCUT,
          n_layers=3,
          backend=ParallelSimulator()
      )

Backend Options
---------------

**Local Development** 💻
   Use ``ParallelSimulator`` for fast iteration and testing:

   .. code-block:: python

      backend = ParallelSimulator(
          shots=1000,           # Measurement precision
          n_processes=4,        # Parallel execution
          qiskit_backend="auto" # Automatic noisy backend selection
      )

**Cloud & Hardware** ☁️
   Access real quantum computers through ``QoroService`` (contact us for access):

   .. code-block:: python

      from divi.backends import QoroService

      # Initialize cloud service
      service = QoroService()  # Uses QORO_API_KEY from .env file

      qasm_circuit = """OPENQASM 2.0;
      include "qelib1.inc";

      qreg q[2];
      creg c[2];

      h q[0];
      cx q[0], q[1];
      measure q[0] -> c[0];
      measure q[1] -> c[1];"""

      # Submit to quantum hardware
      circuits_dict = {"my_circuit": qasm_circuit}
      execution_result = service.submit_circuits(circuits_dict, qpu_system_name="ibm_one")

Advanced Features
-----------------

**Program Batches** 🔄
   Run multiple quantum programs in parallel for hyperparameter sweeps and large-scale problems:

   .. code-block:: python

      from divi.qprog.workflows import VQEHyperparameterSweep
      from divi.qprog import MoleculeTransformer
      from divi.qprog.optimizers import MonteCarloOptimizer
      from divi.backends import ParallelSimulator
      from divi.qprog import HartreeFockAnsatz, UCCSDAnsatz
      import pennylane as qml
      import numpy as np

      h2_molecule = qml.qchem.Molecule(
         symbols=["H", "H"], coordinates=np.array([[0.0, 0.0, -0.6614], [0.0, 0.0, 0.6614]])
      )

      # Run hyperparameter sweep
      # Create molecule transformer for bond length variations
      transformer = MoleculeTransformer(
         base_molecule=h2_molecule,
         bond_modifiers=[-0.2, 0.0, 0.2],  # Bond length changes in Å
      )

      sweep = VQEHyperparameterSweep(
         molecule_transformer=transformer,
         ansatze=[HartreeFockAnsatz(), UCCSDAnsatz()],
         optimizer=MonteCarloOptimizer(n_param_sets=5, n_best_sets=2),
         max_iterations=10,
         backend=ParallelSimulator(n_processes=4),
      )

      sweep.create_programs()  # Generate all VQE instances
      sweep.run(blocking=True)  # Execute all programs in parallel

      # Get best configuration
      (best_ansatz, best_bond_modifier), best_energy = sweep.aggregate_results()

**Observable Grouping** 🔗
   Optimize measurements by grouping commuting observables using PennyLane's grouping strategies:

   .. code-block:: python

      # Create VQE with observable grouping for efficiency
      vqe = VQE(
          molecule=h2_molecule,
          ansatz=HartreeFockAnsatz(),
          grouping_strategy="qwc",  # PennyLane's qubit-wise commuting strategy
          backend=ParallelSimulator()
      )

      # Commuting measurements are grouped for fewer circuit executions
      vqe.run()

   **Note:** Observable grouping is a PennyLane feature. For detailed information about available strategies (`"qwc"`, `"wires"`, `"default"`), see the `PennyLane grouping documentation <https://docs.pennylane.ai/en/stable/code/api/pennylane.transforms.split_non_commuting.html>`_.

**Error Mitigation** 🛡️
   Improve result accuracy with built-in techniques:

   .. code-block:: python

      from divi.circuits.qem import ZNE
      from mitiq.zne.inference import RichardsonFactory
      from mitiq.zne.scaling import fold_gates_at_random
      from functools import partial

      # Create ZNE protocol
      scale_factors = [1.0, 1.5, 2.0]
      zne_protocol = ZNE(
          scale_factors=scale_factors,
          folding_fn=partial(fold_gates_at_random),
          extrapolation_factory=RichardsonFactory(scale_factors=scale_factors),
      )

      vqe = VQE(
          molecule=h2_molecule,
          qem_protocol=zne_protocol,
          backend=ParallelSimulator(qiskit_backend="auto"),
      )

Next Steps & Getting Help
-------------------------

**Continue Learning:**

* 🎯 **Try More Examples**: Explore the `tutorials/ <https://github.com/QoroQuantum/divi/tree/main/tutorials>`_ directory
* ⚡ **Scale Up**: Learn about :doc:`user_guide/program_batches` for parallel execution
* 🛠️ **Customize**: Create your own algorithms using the :doc:`api_reference/qprog`
* 📊 **Monitor Progress**: Explore :doc:`api_reference/reporting` for advanced progress tracking

**Documentation & Support:**

* 📖 **User Guide**: Complete guides including :doc:`user_guide/core_concepts`, :doc:`user_guide/vqe`, and :doc:`user_guide/qaoa`
* 🔧 **API Reference**: Detailed function documentation in :doc:`api_reference/qprog`, :doc:`api_reference/backends`, and :doc:`api_reference/circuits`
* 🐛 **Issues**: Report bugs on `GitHub <https://github.com/QoroQuantum/divi>`_

**Ready to dive deeper?** Check out the comprehensive guides in the User Guide section!
