Improving Results with Error Mitigation
========================================

Divi provides built-in quantum error mitigation (QEM) to improve results from
noisy quantum hardware. Two built-in protocols ship with the library:

- **Zero Noise Extrapolation (ZNE)** — runs circuits at artificially increased
  noise levels and extrapolates to the zero-noise limit.
- **Quantum Enhanced Pauli Propagation (QuEPP)** — decomposes the circuit into
  Clifford Pauli paths, simulates them classically, and corrects the noisy
  quantum result with an empirical rescaling factor.

Pass either protocol into variational programs (for example :class:`~divi.qprog.algorithms.VQE`
or :class:`~divi.qprog.algorithms.QAOA`) with the ``qem_protocol`` argument. You can also
subclass :class:`~divi.circuits.qem.QEMProtocol` for custom mitigation; see
`Custom Error Mitigation Protocols`_ below.

Zero Noise Extrapolation (ZNE)
------------------------------

Divi's ZNE runs the target circuit at several amplified noise levels and
extrapolates the per-scale expectation values back to the zero-noise limit.
Folding and extrapolation are both built-in — :class:`~divi.circuits.qem.ZNE`
ships with global-unitary folding by default and uses
:class:`~divi.circuits.qem.RichardsonExtrapolator` unless a custom
extrapolator is provided.

**Basic Usage:**

.. code-block:: python

   from divi.circuits.qem import ZNE, RichardsonExtrapolator
   from divi.qprog import VQE
   from divi.backends import QiskitSimulator
   import pennylane as qml
   import numpy as np

   # Create a ZNE protocol with three noise scale factors.  The default
   # folding function is global unitary folding and requires odd-integer
   # scales; see the ZNE docstring for writing a custom folding callable
   # that accepts arbitrary floats.
   scale_factors = [1, 3, 5]
   zne_protocol = ZNE(
       scale_factors=scale_factors,
       extrapolator=RichardsonExtrapolator(),
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
       max_iterations=10,
   )

   vqe.run()
   print(f"Mitigated energy: {vqe.best_loss:.6f}")

**Configuration Options** (same imports as in **Basic Usage** above):

.. code-block:: python

   # Light mitigation (faster, 2 scale factors)
   light_zne = ZNE(
       scale_factors=[1, 3],
       extrapolator=RichardsonExtrapolator(),
   )

   # Heavy mitigation (more accurate, 5 scale factors)
   heavy_zne = ZNE(
       scale_factors=[1, 3, 5, 7, 9],
       extrapolator=RichardsonExtrapolator(),
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

.. skip: next

.. code-block:: python

   from divi.circuits.quepp import QuEPP
   from divi.qprog import VQE
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
       max_iterations=10,
   )

   vqe.run()
   print(f"Mitigated energy: {vqe.best_loss:.6f}")

**Parameters:**

- ``truncation_order`` *(int, default 2)* — Maximum CPT expansion order *K*.
  For ``sampling="exhaustive"``, higher *K* includes more Pauli paths (cost
  grows combinatorially with the number of non-Clifford gates). For the default
  ``sampling="montecarlo"``, paths are drawn with a fixed budget instead; order
  still affects diagnostics such as the shallow-circuit warning, but path count
  is controlled primarily by ``n_samples``.
- ``coefficient_threshold`` *(float, optional)* — Prune paths whose absolute
  weight falls below this threshold during DFS enumeration (``sampling="exhaustive"``
  only; see the QuEPP class docstring for symbolic-circuit behavior).
- ``sampling`` — ``"montecarlo"`` *(default)* uses ``n_samples`` random paths;
  ``"exhaustive"`` enumerates paths up to ``truncation_order`` (deterministic;
  cost grows with order and circuit size).
- ``n_samples`` *(int, default 200)* — Monte Carlo path budget when
  ``sampling="montecarlo"``.
- ``seed`` *(int, optional)* — RNG seed for Monte Carlo reproducibility.
- ``n_twirls`` *(int, default 10)* — Pauli twirl count; ``0`` disables twirling.
  The parameter ``bind_before_mitigation`` on :class:`~divi.circuits.quepp.QuEPP`
  trades repeated structural work against path count when angles are symbolic.

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

.. skip: next

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

   UserWarning: QuEPP: signal destroyed — η fell below the safety threshold
   and mitigation fell back to the raw noisy value. Consider increasing shots
   or reducing noise.

This is distinct from observable groups whose classical Pauli-path values are
near zero — those carry negligible weight in the Hamiltonian and do not trigger
the warning.

If you see this warning frequently, consider:

- **Increasing the number of shots** to reduce statistical noise in η.
- **Enabling Pauli twirling** (``n_twirls > 0``) to convert coherent noise into
  stochastic noise that QuEPP handles more gracefully.
- **Lowering the noise level** (e.g. using a less noisy backend or reducing
  circuit depth).

Shallow Circuit Warning
~~~~~~~~~~~~~~~~~~~~~~~

QuEPP's correction relies on the CPT expansion being a small perturbation of the
target circuit.  When the truncation order K replaces a large fraction of the
non-Clifford rotations, path circuits differ too much from the target for
reliable η estimation.  QuEPP emits a warning when ``K / n_rotations > 33%``:

.. code-block:: text

   UserWarning: QuEPP: truncation order K=2 replaces a large fraction of the
   4 non-Clifford rotations (50%). Mitigation quality may degrade on shallow
   circuits — consider reducing truncation_order or using a deeper circuit.

This typically occurs on small circuits (< 10 qubits) where the number of
non-Clifford rotations is comparable to K.  The paper validates QuEPP on
49-qubit circuits with hundreds of rotations.

If you see this warning:

- **Reduce truncation_order** to ``K=1`` or use ``sampling="montecarlo"`` which
  does not enumerate all branches.
- **Use a deeper circuit** (more qubits or Trotter steps).
- **Use ZNE instead** for shallow circuits where QuEPP is unreliable.

Performance Considerations
--------------------------

- **ZNE**: Expect roughly one backend evaluation per scale factor per
  unmitigated evaluation (plus extrapolation overhead on the classical side).
- **QuEPP**: Cost grows with path count (Monte Carlo budget or exhaustive
  enumeration), twirls, and circuit size. Classical Clifford simulation of
  paths is comparatively cheap next to quantum shots.
- **Budget**: Mitigation increases total shots or circuit evaluations; use
  :meth:`~divi.qprog.QuantumProgram.dry_run` to preview expansion before a long
  run.

Custom Error Mitigation Protocols
---------------------------------

You can implement custom error mitigation strategies by inheriting from
:class:`~divi.circuits.qem.QEMProtocol`.  The protocol operates on Qiskit
:class:`~qiskit.dagcircuit.DAGCircuit` bodies — the same IR the rest of the
pipeline uses — and must implement three members:

.. code-block:: python

   from collections.abc import Sequence
   from qiskit.dagcircuit import DAGCircuit
   from divi.backends import MaestroSimulator
   from divi.circuits.qem import QEMContext, QEMProtocol

   class WeightedAveraging(QEMProtocol):
       """A simple protocol that runs the circuit twice and averages results."""

       @property
       def name(self) -> str:
           return "weighted_avg"

       def expand(self, dag: DAGCircuit, observable=None):
           """Return circuits to execute and a reduce-time context.

           For noise-scaling techniques the tuple contains multiple circuit
           variants; for simple protocols it may return the original circuit
           unchanged.  The optional ``observable`` argument carries the
           observable being measured (as a Qiskit
           :class:`~qiskit.quantum_info.SparsePauliOp`) — hybrid protocols
           like QuEPP use it for classical pre-computation.
           """
           # Run the same circuit twice (e.g. with different readout strategies)
           return (dag, dag), {}

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
- ``expand(dag, observable)`` — Generate one or more Qiskit
  :class:`~qiskit.dagcircuit.DAGCircuit` bodies to execute on the quantum
  backend and a ``QEMContext`` carrying any classical side-channel data for
  the reduce phase.  Return a ``tuple[tuple[DAGCircuit, ...], QEMContext]``.
- ``reduce(quantum_results, context)`` — Combine a ``Sequence[float]`` of
  per-circuit expectation values with the ``QEMContext``
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

- :doc:`../api_reference/circuits` — ``QEMProtocol``, ``ZNE``, and QuEPP
- :doc:`program_ensembles` — running many mitigated programs together
- :doc:`pipelines` — how QEM fits into the circuit pipeline
