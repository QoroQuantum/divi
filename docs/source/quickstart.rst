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
   from divi.backends import MaestroSimulator

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
      backend=MaestroSimulator(shots=1000),  # Local simulator
   )

   # Step 4: Run and get results!
   vqe.run()

   # Check your results
   print(f"🎉 Ground state energy: {vqe.best_loss:.6f} Hartree")
   print(f"⚡ Circuits executed: {vqe.total_circuit_count}")

That's it! You just ran a variational quantum algorithm. The energy should be close to -1.137 Hartree (H₂'s true ground state energy).

.. tip::

   **Stuck?** Try :doc:`divi-ai <tools/divi_ai>` — ask questions, get code examples, and explore
   APIs right in your terminal: ``pip install qoro-divi[ai] && divi-ai``

Choosing the Right Algorithm
-----------------------------

Divi offers specialized algorithms for different problem types:

**VQE – Quantum Chemistry** ⚗️
   Use :class:`VQE` for molecular ground state calculations, dissociation curves, and electronic structure problems.

   .. code-block:: python

      from divi.qprog import VQE, UCCSDAnsatz

      vqe = VQE(
          molecule=h2_molecule,
          ansatz=UCCSDAnsatz(),  # More sophisticated than Hartree-Fock
          n_layers=3,
          backend=MaestroSimulator()
      )

**QAOA – Optimization Problems** 🎯
   Use :class:`QAOA` for combinatorial optimization: Max-Cut, Max-Clique, traveling salesman, QUBO/HUBO, and similar NP-hard problems (graphs or binary polynomial formulations).

   .. code-block:: python

      import networkx as nx
      from divi.qprog import QAOA
      from divi.qprog.problems import MaxCutProblem

      # Create your problem graph
      graph = nx.erdos_renyi_graph(10, 0.5)

      qaoa = QAOA(
          MaxCutProblem(graph),
          n_layers=3,
          backend=MaestroSimulator()
      )

**PCE – QUBO/HUBO with Pauli Correlation Encoding** 📐
   Use :class:`PCE` for QUBO and higher-order (HUBO) binary optimization with parity-based encoding. PCE is a VQE variant that maps each variable to a parity of the measured bitstring, scaling logarithmically in qubits (dense encoding) or as O(√N) (poly encoding).

   .. code-block:: python

      import numpy as np
      from divi.qprog import PCE
      from divi.backends import MaestroSimulator

      qubo_matrix = np.array([[-1.0, 2.0], [0.0, 1.0]])
      pce = PCE(
          problem=qubo_matrix,
          backend=MaestroSimulator(),
      )
      pce.run()

**TimeEvolution – Hamiltonian Dynamics** ⏱️
   Use :class:`TimeEvolution` to simulate real-time quantum dynamics under a Hamiltonian (Trotter-Suzuki or QDrift). Supports probability or observable mode.

   .. code-block:: python

      import math
      import pennylane as qml
      from divi.qprog import TimeEvolution
      from divi.backends import MaestroSimulator

      te = TimeEvolution(
          hamiltonian=qml.PauliX(0) + qml.PauliX(1),
          time=math.pi / 2,
          backend=MaestroSimulator(shots=5000),
      )
      te.run()
      print(te.results)  # basis-state probabilities or expectation value

Backend Options
---------------

**Local Development** 💻
   Use :class:`MaestroSimulator` (shown in all examples above) for fast iteration and testing. For noisy simulation, use :class:`QiskitSimulator` with Qiskit noise models.

**Cloud Simulation & Hardware** ☁️
   Access scalable cloud simulators (statevector, tensor-network, and more) through :class:`QoroService`. Sign up at `dash.qoroquantum.net <https://dash.qoroquantum.net/>`_ to get started with free credits. For real quantum hardware access, `contact us <https://qoroquantum.net>`_:

   .. code-block:: python

      from divi.backends import QoroService, JobConfig

      service = QoroService()  # Uses QORO_API_KEY from .env file
      result = service.submit_circuits(
          {"my_circuit": qasm_string},
          override_job_config=JobConfig(simulator_cluster="qoro_maestro"),
      )

Advanced Features
-----------------

**Program Ensembles** 🔄
   Run multiple quantum programs in parallel for hyperparameter sweeps and large-scale problems:

   .. code-block:: python

      from divi.qprog.workflows import VQEHyperparameterSweep
      from divi.qprog import MoleculeTransformer, HartreeFockAnsatz, UCCSDAnsatz

      # Sweep over bond lengths × ansatze in parallel
      sweep = VQEHyperparameterSweep(
         molecule_transformer=MoleculeTransformer(
            base_molecule=h2_molecule,
            bond_modifiers=[-0.2, 0.0, 0.2],
         ),
         ansatze=[HartreeFockAnsatz(), UCCSDAnsatz()],
         max_iterations=10,
         backend=MaestroSimulator(),
      )
      sweep.create_programs()
      sweep.run(blocking=True)
      (best_ansatz, best_bond_modifier), best_energy = sweep.aggregate_results()

**Observable Grouping** 🔗
   Reduce circuit count by grouping commuting measurements — just add ``grouping_strategy`` to any VQE:

   .. code-block:: python

      vqe = VQE(..., grouping_strategy="qwc")  # qubit-wise commuting groups

   Available strategies: ``"qwc"``, ``"wires"``, ``"default"``. See the `PennyLane grouping documentation <https://docs.pennylane.ai/en/stable/code/api/pennylane.transforms.split_non_commuting.html>`_ for details.

**Error Mitigation** 🛡️
   Improve results on noisy backends with Zero Noise Extrapolation — pass a ``qem_protocol`` and use :class:`QiskitSimulator` with a noise model:

   .. code-block:: python

      from divi.circuits.qem import ZNE

      vqe = VQE(..., qem_protocol=ZNE(...), backend=QiskitSimulator(qiskit_backend="auto"))

   See :doc:`user_guide/improving_results_qem` for a full walkthrough.

Next Steps & Getting Help
-------------------------

**Continue Learning:**

* 🎯 **Try More Examples**: Explore the `tutorials/ <https://github.com/QoroQuantum/divi/tree/main/tutorials>`_ directory
* ⚡ **Scale Up**: Learn about :doc:`user_guide/program_ensembles` for parallel execution
* 🛠️ **Customize**: Create your own algorithms using the :doc:`api_reference/qprog`
* 📊 **Monitor Progress**: Explore :doc:`api_reference/reporting` for advanced progress tracking
* 🔍 **Inspect Landscapes**: Use :doc:`user_guide/visualization` to explore variational loss landscapes with ``divi.viz``

**Documentation & Support:**

* 📖 **User Guide**: Complete guides including :doc:`user_guide/core_concepts`, :doc:`user_guide/visualization`, :doc:`user_guide/ground_state_energy_estimation_vqe`, and :doc:`user_guide/combinatorial_optimization_qaoa_pce`
* 🔧 **API Reference**: Detailed function documentation in :doc:`api_reference/qprog`, :doc:`api_reference/backends`, and :doc:`api_reference/circuits`
* 🐛 **Issues**: Report bugs on `GitHub <https://github.com/QoroQuantum/divi>`_

**Ready to dive deeper?** Check out the comprehensive guides in the User Guide section!
