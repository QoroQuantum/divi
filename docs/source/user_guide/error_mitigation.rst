Error Mitigation
================

Divi provides built-in error mitigation to improve results from noisy quantum hardware. The main technique is Zero Noise Extrapolation (ZNE), which runs circuits at different noise levels and extrapolates to the zero-noise limit using :class:`mitiq.zne.inference.RichardsonFactory` and :class:`mitiq.zne.scaling.fold_gates_at_random`.

Basic Usage
-----------

**Simple ZNE Setup:**

.. code-block:: python

   from functools import partial
   from mitiq.zne.inference import RichardsonFactory
   from mitiq.zne.scaling import fold_gates_at_random
   from divi.circuits.qem import ZNE
   from divi.qprog import VQE, HartreeFockAnsatz
   from divi.backends import ParallelSimulator
   import pennylane as qml
   import numpy as np

   # Create ZNE protocol
   scale_factors = [1.0, 1.5, 2.0]
   zne_protocol = ZNE(
       scale_factors=scale_factors,
       folding_fn=partial(fold_gates_at_random),
       extrapolation_factory=RichardsonFactory(scale_factors=scale_factors),
   )

   # Apply to VQE
   h2_molecule = qml.qchem.Molecule(
       symbols=["H", "H"],
       coordinates=np.array([[0.0, 0.0, -0.6614], [0.0, 0.0, 0.6614]])
   )

   vqe = VQE(
       molecule=h2_molecule,
       qem_protocol=zne_protocol,
       backend=ParallelSimulator(qiskit_backend="auto")  # Use noisy simulator
   )

   vqe.run()
   print(f"Mitigated energy: {vqe.best_loss:.6f}")

Configuration Options
---------------------

**Light Mitigation (Faster):**

.. code-block:: python

   light_zne = ZNE(
       scale_factors=[1.0, 1.5],
       folding_fn=partial(fold_gates_at_random),
       extrapolation_factory=RichardsonFactory(scale_factors=[1.0, 1.5]),
   )

**Heavy Mitigation (More Accurate):**

.. code-block:: python

   heavy_zne = ZNE(
       scale_factors=[1.0, 1.5, 2.0, 2.5, 3.0],
       folding_fn=partial(fold_gates_at_random),
       extrapolation_factory=RichardsonFactory(
           scale_factors=[1.0, 1.5, 2.0, 2.5, 3.0]
       ),
   )

Performance Considerations
--------------------------

- **Overhead**: ZNE typically requires 2-5x more circuit evaluations
- **Memory**: Stores results from multiple noise levels
- **Time**: Longer execution due to additional circuits

**Tip**: Use fewer shots (500-1000) with mitigation since results are averaged across noise levels.

Custom Error Mitigation Protocols
---------------------------------

You can implement custom error mitigation strategies by inheriting from
:class:`~divi.circuits.qem.QEMProtocol`.  The protocol operates on **Cirq** circuits
and must implement three members:

.. code-block:: python

   from collections.abc import Sequence
   from cirq.circuits.circuit import Circuit
   from divi.circuits.qem import QEMProtocol

   class WeightedAveraging(QEMProtocol):
       """A simple protocol that runs the circuit twice and averages results."""

       @property
       def name(self) -> str:
           return "weighted_avg"

       def modify_circuit(self, cirq_circuit: Circuit) -> Sequence[Circuit]:
           """Return one or more Cirq circuits to execute.

           For noise-scaling techniques the list contains multiple variants;
           for simple protocols it may return the original circuit unchanged.
           """
           # Run the same circuit twice (e.g. with different readout strategies)
           return [cirq_circuit, cirq_circuit]

       def postprocess_results(self, results: Sequence[float]) -> float:
           """Combine the results of all circuits into a single mitigated value.

           ``results`` contains one expectation value per circuit returned by
           ``modify_circuit``, in the same order.
           """
           return sum(results) / len(results)

   # Pass the custom protocol when constructing any variational program
   vqe = VQE(
       molecule=h2_molecule,
       qem_protocol=WeightedAveraging(),
       backend=ParallelSimulator(),
   )

.. note::
   When a ``qem_protocol`` is provided, the :doc:`circuit pipeline <pipelines>`
   automatically wraps it in a :class:`~divi.pipeline.stages.QEMStage`.
   During execution, ``modify_circuit`` is called in the *expand* pass and
   ``postprocess_results`` is called in the *reduce* pass — you don't need to
   manage pipeline integration yourself.

**Key Members to Implement:**

- ``name`` *(property)* — Unique protocol name used as the pipeline axis identifier
- ``modify_circuit(cirq_circuit)`` — Transform or replicate a Cirq circuit before
  execution. Return a ``Sequence[Circuit]``
- ``postprocess_results(results)`` — Combine a ``Sequence[float]`` of per-circuit
  expectation values into a single ``float``

Next Steps
----------

- 🛠️ **API Reference**: Learn about custom protocols in :doc:`../api_reference/circuits`
- 📊 **Program Batches**: Apply mitigation to large computations in :doc:`program_batches`
- 📈 **Advanced Usage**: Explore :class:`mitiq` documentation for more sophisticated techniques
- 🔧 **Pipelines**: Understand how stages compose in :doc:`pipelines`
