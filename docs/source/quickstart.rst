Quick Start Guide
=================

Welcome to Divi! This guide will get you up and running with quantum program execution in minutes.

What is Divi?
-------------

Divi is a Python library that automates the orchestration around quantum
programs: circuit generation, batching, error mitigation, parameter optimization,
and result aggregation. Whether you're studying molecular systems or solving
combinatorial optimization problems, Divi handles the plumbing so you can focus
on the problem.

**Core capabilities:**

* **Automated execution** — run quantum programs with minimal boilerplate.
* **Parallel circuit execution** — distribute circuits across available resources automatically.
* **Pluggable backends** — swap between local simulators, noisy simulators, and cloud hardware without changing program code.
* **Integrated error mitigation** — ZNE and QuEPP plug into the variational loop.
* **Progress tracking** — real-time feedback during long-running computations.

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
   print(f"Ground state energy: {vqe.best_loss:.6f} Hartree")
   print(f"Circuits executed: {vqe.total_circuit_count}")

That's it — you just ran a variational quantum algorithm. The energy should be close to -1.137 Hartree (H₂'s true ground state energy).

.. tip::

   **Stuck?** Try :doc:`divi-ai <tools/divi_ai>` — ask questions, get code examples, and explore
   APIs right in your terminal: ``pip install qoro-divi[ai] && divi-ai``

Choosing the Right Algorithm
-----------------------------

Divi offers specialized algorithms for different problem types:

**VQE – Quantum Chemistry**
   Use :class:`~divi.qprog.algorithms.VQE` for molecular ground state calculations, dissociation curves, and electronic structure problems.

   .. code-block:: python

      from divi.qprog import VQE, UCCSDAnsatz

      vqe = VQE(
          molecule=h2_molecule,
          ansatz=UCCSDAnsatz(),  # More sophisticated than Hartree-Fock
          n_layers=2,
          backend=MaestroSimulator()
      )

**QAOA – Optimization Problems**
   Use :class:`~divi.qprog.algorithms.QAOA` for combinatorial optimization: Max-Cut, Max-Clique, traveling salesman, QUBO/HUBO, and similar NP-hard problems (graphs or binary polynomial formulations).

   .. code-block:: python

      import networkx as nx
      from divi.qprog import QAOA
      from divi.qprog.problems import MaxCutProblem

      # Create your problem graph
      graph = nx.erdos_renyi_graph(10, 0.5)

      qaoa = QAOA(
          MaxCutProblem(graph),
          n_layers=2,
          backend=MaestroSimulator()
      )

**PCE – QUBO/HUBO with Pauli Correlation Encoding**
   Use :class:`~divi.qprog.algorithms.PCE` for QUBO and higher-order (HUBO) binary optimization with parity-based encoding. PCE is a VQE variant that maps each variable to a parity of the measured bitstring, scaling logarithmically in qubits (dense encoding) or as O(√N) (poly encoding).

   .. code-block:: python

      import numpy as np
      import pennylane as qml
      from divi.qprog import PCE, GenericLayerAnsatz
      from divi.backends import MaestroSimulator

      qubo_matrix = np.array([[-1.0, 2.0], [0.0, 1.0]])
      pce = PCE(
          problem=qubo_matrix,
          ansatz=GenericLayerAnsatz(
              gate_sequence=[qml.RY, qml.RZ],
              entangler=qml.CNOT,
              entangling_layout="all-to-all",
          ),
          backend=MaestroSimulator(),
      )
      pce.run()

**TimeEvolution – Hamiltonian Dynamics**
   Use :class:`~divi.qprog.algorithms.TimeEvolution` to simulate real-time quantum dynamics under a Hamiltonian (Trotter-Suzuki or QDrift). Supports probability or observable mode.

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

**Local development**
   Use :class:`~divi.backends.MaestroSimulator` (shown in all examples above) for fast iteration and testing. For noisy simulation, use :class:`~divi.backends.QiskitSimulator` with Qiskit noise models.

**Cloud simulation & hardware**
   Access scalable cloud simulators (statevector, tensor-network, and more) through :class:`~divi.backends.QoroService`. Sign up at `dash.qoroquantum.net <https://dash.qoroquantum.net/>`_ to get started with free credits. For real quantum hardware access, `contact us <https://qoroquantum.net>`_:

   .. code-block:: python

      from divi.backends import QoroService, JobConfig

      # Bell pair: H + CNOT, then measure both qubits (OpenQASM 2.0)
      qasm = (
          'OPENQASM 2.0;\n'
          'include "qelib1.inc";\n'
          'qreg q[2];\n'
          'creg c[2];\n'
          'h q[0];\n'
          'cx q[0],q[1];\n'
          'measure q[0] -> c[0];\n'
          'measure q[1] -> c[1];\n'
      )

      service = QoroService()  # Uses QORO_API_KEY from .env file
      result = service.submit_circuits(
          {"my_circuit": qasm},
          override_job_config=JobConfig(simulator_cluster="qoro_maestro"),
      )

What to read next
-----------------

Now that you have a VQE run working, dig into the user guide:

* **Deepen your understanding of the algorithms** — :doc:`user_guide/ground_state_energy_estimation_vqe`, :doc:`user_guide/combinatorial_optimization_qaoa_pce`, :doc:`user_guide/hamiltonian_time_evolution`.
* **Scale up** — run many programs in parallel with :doc:`user_guide/program_ensembles`.
* **Improve noisy results** — mitigate errors with :doc:`user_guide/improving_results_qem`.
* **Tune the optimizer** — see :doc:`user_guide/optimizers`.
* **Inspect and diagnose runs** — :doc:`user_guide/visualization`.
* **Understand how circuits flow through Divi** — the expand/execute/reduce model is in :doc:`user_guide/pipelines`.
* **End-to-end walkthroughs** — the `tutorials/ <https://github.com/QoroQuantum/divi/tree/main/tutorials>`_ directory on GitHub.

Found a bug or want a feature? Open a ticket on `GitHub Issues <https://github.com/QoroQuantum/divi/issues>`_.
