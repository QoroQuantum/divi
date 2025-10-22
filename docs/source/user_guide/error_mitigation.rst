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

You can implement custom error mitigation strategies by inheriting from :class:`QEMProtocol`:

.. code-block:: python

   from divi.circuits.qem import QEMProtocol
   import numpy as np

   class ReadoutErrorMitigation(QEMProtocol):
       """Simple readout error mitigation protocol"""

       def __init__(self, calibration_matrix=None):
           self.calibration_matrix = calibration_matrix
           self.name = "Readout Error Mitigation"

       def modify_circuit(self, circuit):
           """No circuit modification needed for readout mitigation"""
           return [circuit]

       def postprocess_results(self, results):
           """Apply readout error correction to measurement results"""
           if self.calibration_matrix is None:
               return results[0]  # No correction if no calibration

           # Apply matrix correction to measurement probabilities
           corrected_probs = np.dot(results[0], self.calibration_matrix)
           return corrected_probs

   # Usage example
   calibration_matrix = np.array([[0.9, 0.1], [0.1, 0.9]])
   readout_mitigation = ReadoutErrorMitigation(calibration_matrix=calibration_matrix)

   vqe = VQE(
       molecule=h2_molecule,
       qem_protocol=readout_mitigation,
       backend=ParallelSimulator()
   )

**Key Methods to Implement:**

- ``modify_circuit(circuit)`` - Modify circuits before execution (return list of circuits)
- ``postprocess_results(results)`` - Process results after execution
- ``name`` - Protocol name for identification

Next Steps
----------

- 🛠️ **API Reference**: Learn about custom protocols in :doc:`../api_reference/circuits`
- 📊 **Program Batches**: Apply mitigation to large computations in :doc:`program_batches`
- 📈 **Advanced Usage**: Explore :class:`mitiq` documentation for more sophisticated techniques
