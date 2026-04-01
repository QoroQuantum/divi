Improving Results with Error Mitigation
========================================

Divi provides built-in quantum error mitigation (QEM) to improve results from
noisy quantum hardware.  Two protocols are included out of the box:

- **Zero Noise Extrapolation (ZNE)** — runs circuits at artificially increased
  noise levels and extrapolates to the zero-noise limit.
- **Quantum Enhanced Pauli Propagation (QuEPP)** — decomposes the circuit into
  Clifford Pauli paths, simulates them classically, and corrects the noisy
  quantum result with an empirical rescaling factor.

Both protocols plug into any quantum program via the ``qem_protocol``
argument.

Zero Noise Extrapolation (ZNE)
------------------------------

ZNE uses `Mitiq <https://mitiq.readthedocs.io/>`_ to construct noise-scaled
circuits and extrapolate.

**Basic Usage:**

.. code-block:: python

   from functools import partial
   from mitiq.zne.inference import RichardsonFactory
   from mitiq.zne.scaling import fold_gates_at_random
   from divi.circuits.qem import ZNE
   from divi.qprog import VQE, HartreeFockAnsatz
   from divi.backends import QiskitSimulator
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
       backend=QiskitSimulator(qiskit_backend="auto"),
   )

   vqe.run()
   print(f"Mitigated energy: {vqe.best_loss:.6f}")

**Configuration Options:**

.. code-block:: python

   # Light mitigation (faster, 2 scale factors)
   light_zne = ZNE(
       scale_factors=[1.0, 1.5],
       folding_fn=partial(fold_gates_at_random),
       extrapolation_factory=RichardsonFactory(scale_factors=[1.0, 1.5]),
   )

   # Heavy mitigation (more accurate, 5 scale factors)
   heavy_zne = ZNE(
       scale_factors=[1.0, 1.5, 2.0, 2.5, 3.0],
       folding_fn=partial(fold_gates_at_random),
       extrapolation_factory=RichardsonFactory(
           scale_factors=[1.0, 1.5, 2.0, 2.5, 3.0]
       ),
   )

Quantum Enhanced Pauli Propagation (QuEPP)
------------------------------------------

QuEPP is a hybrid classical-quantum protocol based on Clifford Perturbation
Theory (CPT) from `Majumder et al. (2026) <https://arxiv.org/abs/2603.14485>`_.

It works by decomposing the target circuit into a set of Clifford circuits
(Pauli paths) whose expectation values can be computed exactly with a classical
simulator.  The low-order paths capture most of the signal; the residual
higher-order contribution is estimated from the noisy quantum hardware and
corrected with a rescaling factor derived from comparing noisy and ideal values
on the ensemble circuits.

**Basic Usage:**

.. code-block:: python

   from divi.circuits.quepp import QuEPP
   from divi.qprog import VQE, HartreeFockAnsatz
   from divi.backends import QiskitSimulator
   import pennylane as qml
   import numpy as np

   h2_molecule = qml.qchem.Molecule(
       symbols=["H", "H"],
       coordinates=np.array([[0.0, 0.0, -0.6614], [0.0, 0.0, 0.6614]])
   )

   vqe = VQE(
       molecule=h2_molecule,
       qem_protocol=QuEPP(truncation_order=2),
       backend=QiskitSimulator(qiskit_backend="auto"),
   )

   vqe.run()
   print(f"Mitigated energy: {vqe.best_loss:.6f}")

**Parameters:**

- ``truncation_order`` *(int, default 2)* — Maximum CPT expansion order.
  Higher values include more Pauli paths, reducing bias at the cost of
  additional auxiliary circuits.  For circuits with *n* non-Clifford gates,
  the number of paths at order *k* grows as C(n, k).
- ``coefficient_threshold`` *(float, optional)* — Prune paths whose
  weight falls below this threshold during DFS enumeration.  Provides
  early termination for large circuits.
- ``sampling`` — Path selection strategy: ``"montecarlo"`` *(default,*
  *fixed budget via random sampling)* or ``"exhaustive"`` (DFS
  enumeration up to *truncation_order*; deterministic but grows as
  O(n^K_T)).  Use exhaustive for small circuits where you want
  reproducible, exact path sums.
- ``n_samples`` *(int)* — Number of Monte Carlo samples.  Required
  when ``sampling="montecarlo"``.
- ``seed`` *(int, optional)* — RNG seed for Monte Carlo reproducibility.
- ``eta_mode`` — How the rescaling factor η is computed for
  multi-term Hamiltonians:

  - ``"per_group"`` *(default)* — η is computed independently for each
    measurement group (Pauli term).  Allows different terms to have
    different noise sensitivities.  Use for **structured gate noise**.
  - ``"global"`` — η is pooled across all measurement groups into a
    single value.  More statistically robust when noise is **uniform**
    (e.g. readout error, global depolarizing).

  This parameter is a divi extension; the QuEPP paper defines η for a
  single observable and does not discuss multi-term Hamiltonians.

ZNE vs QuEPP
~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Property
     - ZNE
     - QuEPP
   * - Noise model required?
     - No
     - No
   * - Classical pre-computation
     - None
     - Clifford simulation of ensemble
   * - Circuit overhead
     - 1 extra circuit per scale factor
     - 1 + C(n, 1) + ... + C(n, K_T) paths
   * - Best for
     - Coherent gate noise
     - Uniform noise (e.g. readout error)
   * - Observable required?
     - No
     - Yes (used for classical simulation)

Estimating Circuit Cost with Dry Run
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Error mitigation can multiply the number of circuits significantly.  Use
:meth:`~divi.qprog.QuantumProgram.dry_run` to preview the per-stage expansion
before committing to a full run:

.. code-block:: python

   vqe = VQE(
       molecule=h2_molecule,
       qem_protocol=QuEPP(truncation_order=2, n_twirls=10),
       backend=QiskitSimulator(qiskit_backend="auto"),
   )

   # Prints a tree showing the fan-out at each stage
   vqe.dry_run()

The output shows the multiplicative cost of each pipeline stage — including
how many Pauli paths QuEPP generates, the Clifford simulation count, and the
twirl fan-out — so you can tune ``truncation_order``, ``coefficient_threshold``,
and ``n_twirls`` before spending any shots.

See :doc:`pipelines` for full documentation of the dry-run tool.

Signal Destruction and Automatic Fallback
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

QuEPP corrects the noisy quantum result by dividing by the empirical rescaling
factor η.  When noise is so severe that η drops below a safety threshold
(``min_eta=0.1``), the ``1/η`` correction would amplify noise rather than
suppress it.  In this case QuEPP **falls back to the raw noisy value** for that
observable group and emits a summary warning after the evaluation:

.. code-block:: text

   UserWarning: QuEPP: signal destroyed for 3/5 observable group(s) —
   mitigation fell back to noisy values. Consider increasing shots or
   reducing noise.

This is distinct from observable groups whose classical Pauli-path values are
near zero — those carry negligible weight in the Hamiltonian and do not trigger
the warning.

If you see this warning frequently, consider:

- **Increasing the number of shots** to reduce statistical noise in η.
- **Enabling Pauli twirling** (``n_twirls > 0``) to convert coherent noise into
  stochastic noise that QuEPP handles more gracefully.
- **Lowering the noise level** (e.g. using a less noisy backend or reducing
  circuit depth).

Performance Considerations
--------------------------

- **ZNE overhead**: Typically 2-5x more circuit evaluations (one per scale factor).
- **QuEPP overhead**: Scales with the number of non-Clifford gates and truncation
  order.  The classical Clifford simulation is fast (polynomial in qubit count).
- **Tip**: Use fewer shots (500-1000) with mitigation since results are averaged
  across noise levels or ensembles.

Custom Error Mitigation Protocols
---------------------------------

You can implement custom error mitigation strategies by inheriting from
:class:`~divi.circuits.qem.QEMProtocol`.  The protocol operates on **Cirq** circuits
and must implement three members:

.. code-block:: python

   from collections.abc import Sequence
   from cirq.circuits.circuit import Circuit
   from divi.circuits.qem import QEMContext, QEMProtocol

   class WeightedAveraging(QEMProtocol):
       """A simple protocol that runs the circuit twice and averages results."""

       @property
       def name(self) -> str:
           return "weighted_avg"

       def expand(self, cirq_circuit: Circuit, observable=None):
           """Return circuits to execute and a reduce-time context.

           For noise-scaling techniques the tuple contains multiple circuit
           variants; for simple protocols it may return the original circuit
           unchanged.  The optional ``observable`` argument carries the
           PennyLane observable being measured — hybrid protocols like QuEPP
           use it for classical pre-computation.
           """
           # Run the same circuit twice (e.g. with different readout strategies)
           return (cirq_circuit, cirq_circuit), {}

       def reduce(self, quantum_results: Sequence[float], context: QEMContext) -> float:
           """Combine the quantum results into a single mitigated value.

           ``quantum_results`` contains one expectation value per circuit
           returned by ``expand``, in the same order.
           """
           return sum(quantum_results) / len(quantum_results)

   # Pass the custom protocol when constructing any variational program
   vqe = VQE(
       molecule=h2_molecule,
       qem_protocol=WeightedAveraging(),
       backend=MaestroSimulator(),
   )

**Key Members to Implement:**

- ``name`` *(property)* — Unique protocol name used as the pipeline axis identifier
- ``expand(cirq_circuit, observable)`` — Generate one or more Cirq circuits to
  execute on the quantum backend and a :class:`~divi.circuits.qem.QEMContext`
  carrying any classical side-channel data for the reduce phase.
  Return a ``tuple[tuple[Circuit, ...], QEMContext]``.
- ``reduce(quantum_results, context)`` — Combine a ``Sequence[float]`` of
  per-circuit expectation values with the :class:`~divi.circuits.qem.QEMContext`
  into a single ``float``.
- ``post_reduce(contexts)`` *(optional)* — Called once after all per-group
  ``reduce`` calls in an evaluation.  Override to inspect the collected contexts
  and emit summary diagnostics (e.g. QuEPP's signal-destruction warning).
  The default implementation is a no-op.

.. note::
   When a ``qem_protocol`` is provided, the :doc:`circuit pipeline <pipelines>`
   automatically wraps it in a :class:`~divi.pipeline.stages.QEMStage`.
   During execution, ``expand`` is called in the pipeline's *expand* pass and
   ``reduce`` is called in the *reduce* pass — you don't need to manage
   pipeline integration yourself.

Next Steps
----------

- 🛠️ **API Reference**: Learn about protocol classes in :doc:`../api_reference/circuits`
- 📊 **Program Ensembles**: Apply mitigation to large computations in :doc:`program_ensembles`
- 🔧 **Pipelines**: Understand how stages compose in :doc:`pipelines`
