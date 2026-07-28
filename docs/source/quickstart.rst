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
   import pennylane as qp
   from divi.qprog import VQE, HartreeFockAnsatz
   from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer
   from divi.backends import MaestroSimulator

   # Step 1: Define your molecule
   h2_molecule = qp.qchem.Molecule(
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
      from divi.qprog.optimizers import MonteCarloOptimizer

      vqe = VQE(
          molecule=h2_molecule,
          ansatz=UCCSDAnsatz(),  # More sophisticated than Hartree-Fock
          n_layers=2,
          optimizer=MonteCarloOptimizer(),
          backend=MaestroSimulator()
      )

**QAOA – Optimization Problems**
   Use :class:`~divi.qprog.algorithms.QAOA` for combinatorial optimization: Max-Cut, Max-Clique, traveling salesman, QUBO/HUBO, and similar NP-hard problems (graphs or binary polynomial formulations).

   .. code-block:: python

      import networkx as nx
      from divi.qprog import QAOA
      from divi.qprog.problems import MaxCutProblem
      from divi.qprog.optimizers import MonteCarloOptimizer

      # Create your problem graph
      graph = nx.erdos_renyi_graph(10, 0.5, seed=42)

      qaoa = QAOA(
          MaxCutProblem(graph),
          n_layers=2,
          max_iterations=10,
          optimizer=MonteCarloOptimizer(),
          backend=MaestroSimulator()
      )
      qaoa.run()
      print(f"Best loss: {qaoa.best_loss:.4f}")
      print(f"Solution: {qaoa.solution}")

**PCE – QUBO/HUBO with Pauli Correlation Encoding**
   Use :class:`~divi.qprog.algorithms.PCE` for QUBO and higher-order (HUBO) binary optimization with parity-based encoding. PCE is a VQE variant that uses far fewer qubits than standard QAOA for the same problem size — see :doc:`/user_guide/combinatorial_optimization_qaoa_pce` for the encoding details and scaling trade-offs.

   .. code-block:: python

      import numpy as np
      from qiskit.circuit.library import CXGate, RYGate, RZGate
      from divi.qprog import PCE, GenericLayerAnsatz
      from divi.qprog.problems import BinaryOptimizationProblem
      from divi.backends import MaestroSimulator

      from divi.qprog.optimizers import MonteCarloOptimizer

      qubo_matrix = np.array([[-1.0, 2.0], [0.0, 1.0]])
      pce = PCE(
          problem=BinaryOptimizationProblem(qubo_matrix),
          ansatz=GenericLayerAnsatz(
              gate_sequence=[RYGate, RZGate],
              entangler=CXGate,
              entangling_layout="all-to-all",
          ),
          optimizer=MonteCarloOptimizer(),
          backend=MaestroSimulator(),
      )
      pce.run()

**TimeEvolution – Hamiltonian Dynamics**
   Use :class:`~divi.qprog.algorithms.TimeEvolution` to simulate real-time quantum dynamics under a Hamiltonian (Trotter-Suzuki or QDrift). Supports probability or observable mode.

   .. code-block:: python

      import math
      import pennylane as qp
      from divi.qprog import TimeEvolution
      from divi.backends import MaestroSimulator

      te = TimeEvolution(
          hamiltonian=qp.PauliX(0) + qp.PauliX(1),
          time=math.pi / 2,
          backend=MaestroSimulator(shots=5000),
      )
      te.run()
      print(te.results)  # basis-state probabilities or expectation value

**QNN – Quantum Machine Learning**
   Use :class:`~divi.qprog.algorithms.QNN` to train a variational classifier on a
   classical feature batch: a feature map encodes each sample, a trainable ansatz
   supplies the weights, and Divi composes and binds the circuit for you. See
   :doc:`user_guide/quantum_neural_networks`. To bring your own PennyLane or
   Qiskit circuit instead, use :class:`~divi.qprog.algorithms.CustomVQA`
   (:doc:`user_guide/framework_integration`).

   .. code-block:: python

      import numpy as np
      from qiskit.circuit.library import CXGate, RYGate, RZGate
      from divi.qprog import QNN, AngleEmbedding, GenericLayerAnsatz
      from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer
      from divi.backends import MaestroSimulator

      qnn = QNN(
          n_qubits=2,
          feature_map=AngleEmbedding(rotation="Y"),
          ansatz=GenericLayerAnsatz(
              gate_sequence=[RYGate, RZGate],
              entangler=CXGate,
              entangling_layout="linear",
          ),
          feature_batch=np.array([[0.1, 0.2], [2.0, 2.1]]),
          max_iterations=5,
          optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
          backend=MaestroSimulator(),
      )
      qnn.run(perform_final_computation=False)

Check Before You Run
--------------------

Every program can tell you what it *would* submit, without executing anything.
Rebuilding the H₂ VQE from the top of this page:

.. code-block:: python

   from divi.pipeline import format_dry_run

   preview_vqe = VQE(
       molecule=h2_molecule,
       ansatz=HartreeFockAnsatz(),
       n_layers=2,
       optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
       backend=MaestroSimulator(shots=1000),
   )
   format_dry_run(preview_vqe.dry_run())

.. code-block:: text

   cost
   ├── CircuitSpecStage [circuit] → 14
   │   ├── n_qubits: 4
   │   ├── n_gates: 82
   │   └── depth: 47
   ├── MeasurementStage [obs_group] → ÷14
   │   ├── strategy: _backend_expval
   │   └── n_pauli_terms: 14
   ├── Total (per evaluation): 14 ÷ 14 = 1 circuit · 1,000 shots
   └── Summary: avg depth 47, width 4, 36 2q-gates total

   sample
   ├── CircuitSpecStage [circuit] → 1
   ├── MeasurementStage [obs_group] → 1
   ├── Total (once): 1 circuit · 1,000 shots
   └── Summary: avg depth 47, width 4, 36 2q-gates total

One tree per **routine** the program runs — here ``cost``, which the optimizer
re-evaluates, and ``sample``, the single readout VQE performs after training.
Abridged above: each stage also reports more metadata rows, and the real trees
carry two further stages (``PreprocessStage`` and ``ParameterBindingStage``, both
at factor ``1`` here).

Read a tree bottom-up: the ``Total`` line is the number you came for — this program
submits **1 circuit per evaluation**. The rows above show how it got there. The
spec stage's ``14`` is the starting point (this Hamiltonian has 14 Pauli terms),
and each stage below multiplies or divides it: here ``MaestroSimulator`` evaluates
the whole observable at once, so ``÷14`` collapses those 14 terms back into a
single circuit. A different backend, or error mitigation, changes those factors —
and a natural-gradient optimizer drives extra routines of its own, which appear as
further trees.

That makes this the cheapest way to catch a misconfigured program: run it, read
the ``Total``, and see whether the number is what you expected.

One caveat: ``Total`` is per *evaluation*, not per iteration and not per run.
Optimizers evaluate the cost more than once per step — COBYLA, used here, decides
how often as it goes — so ``Total × max_iterations`` is a floor, not a bill. See
:ref:`dry-run` for the full reading guide, and for what a dry run deliberately
does **not** tell you.

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
* **Train a quantum model** — build a quantum neural network with :doc:`user_guide/quantum_neural_networks`, or bring your own PennyLane/Qiskit circuit via :doc:`user_guide/framework_integration`.
* **Scale up** — run many programs in parallel with :doc:`user_guide/program_ensembles`.
* **Improve noisy results** — mitigate errors with :doc:`user_guide/improving_results_qem`.
* **Tune the optimizer** — see :doc:`user_guide/optimizers`.
* **Inspect and diagnose runs** — :doc:`user_guide/visualization`.
* **Understand how circuits flow through Divi** — the expand/execute/reduce model is in :doc:`user_guide/pipelines`, which also covers :ref:`previewing a run before you submit it <dry-run>`.
* **End-to-end walkthroughs** — the `tutorials/ <https://github.com/QoroQuantum/divi/tree/main/tutorials>`_ directory on GitHub.

Found a bug or want a feature? Open a ticket on `GitHub Issues <https://github.com/QoroQuantum/divi/issues>`_.
