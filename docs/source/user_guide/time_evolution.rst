Time Evolution
==============

The :class:`~divi.qprog.TimeEvolution` program simulates real-time quantum dynamics under a Hamiltonian using Trotterized evolution.

It supports two output modes:

- **Probability mode**: ``te.results`` contains measured basis-state probabilities (``dict[str, float]``).
- **Observable mode**: ``te.results`` contains the estimated expectation value (``float``).

Basic Usage
-----------

Use probability mode when you want a final-state distribution:

.. code-block:: python

   import math
   import pennylane as qml
   from divi.backends import ParallelSimulator
   from divi.qprog import TimeEvolution

   backend = ParallelSimulator(shots=5000)

   te = TimeEvolution(
       hamiltonian=qml.PauliX(0) + qml.PauliX(1),
       time=math.pi / 2,
       initial_state="Zeros",
       backend=backend,
   )
   te.run()

   probs = te.results
   print(probs)
   print(f"Circuits executed: {te.total_circuit_count}")

Observable Mode
---------------

Provide ``observable=...`` to estimate expectation values after evolution:

.. code-block:: python

   import pennylane as qml
   from divi.backends import ParallelSimulator
   from divi.qprog import TimeEvolution

   backend = ParallelSimulator(shots=5000)

   te = TimeEvolution(
       hamiltonian=qml.PauliX(0) + qml.PauliZ(0),
       time=0.6,
       n_steps=8,
       initial_state="Zeros",
       observable=qml.PauliZ(0),
       backend=backend,
   )
   te.run()
   print(f"<Z0> = {te.results:.6f}")

QDrift Trotterization
---------------------

For large Hamiltonians, you can use :class:`~divi.qprog.QDrift` to sample terms and average over multiple Hamiltonian samples:

.. code-block:: python

   import pennylane as qml
   from divi.backends import ParallelSimulator
   from divi.qprog import QDrift, TimeEvolution

   backend = ParallelSimulator(shots=5000)
   qdrift = QDrift(
       keep_fraction=0.5,
       sampling_budget=2,
       n_hamiltonians_per_iteration=4,
       sampling_strategy="weighted",
       seed=7,
   )

   te = TimeEvolution(
       hamiltonian=0.7 * qml.PauliZ(0) + 0.4 * qml.PauliX(0) + 0.2 * qml.PauliZ(1),
       trotterization_strategy=qdrift,
       time=0.8,
       initial_state="Superposition",
       backend=backend,
   )
   te.run()
   print(te.results)

.. note::
   Multi-sample QDrift with ``observable`` on an expectation-value backend is currently not supported.
   Use a shot-based backend or run in probability mode for multi-sample averaging.

Next Steps
----------

- Run the full tutorial script: `time_evolution_local.py <https://github.com/QoroQuantum/divi/blob/main/tutorials/time_evolution_local.py>`_
- Learn about optimization-based workflows in :doc:`vqe` and :doc:`qaoa`
- Explore backend choices in :doc:`backends`
