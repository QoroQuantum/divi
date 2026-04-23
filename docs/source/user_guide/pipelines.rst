Pipelines
=========

Every quantum program in Divi executes circuits through a **circuit pipeline**.
The pipeline models the journey from a high-level specification (e.g. a
Hamiltonian or a :class:`~divi.circuits.MetaCircuit`) to final, reduced results as a sequence of
composable **stages**.

This guide explains how the pipeline works, lists the built-in stages shipped
with Divi, and shows two practical examples of extending Divi with custom
algorithms.

.. note::
   If you are using built-in algorithms like :class:`~divi.qprog.algorithms.VQE`, :class:`~divi.qprog.algorithms.QAOA`, or :class:`~divi.qprog.algorithms.TimeEvolution` you
   **don't need to interact with the pipeline directly** — each algorithm
   constructs its own pipeline internally.  This guide is for users who want to
   understand the internals or extend Divi with new algorithms and stages.


How the Pipeline Works
----------------------

A :class:`~divi.pipeline.CircuitPipeline` is an ordered list of stages.
Execution has three phases:

1. **Expand** (forward pass) — Each stage transforms its input into an
   increasingly concrete representation.  The first stage (a
   :class:`~divi.pipeline.SpecStage`) converts the initial specification into a
   keyed batch of :class:`~divi.circuits.MetaCircuit` objects.  Subsequent stages
   (all :class:`~divi.pipeline.BundleStage` instances) transform or fan-out that
   batch — for example, splitting observables into compatible measurement groups,
   binding parameter values, or applying error-mitigation circuit variants.

2. **Execute** — The final batch is compiled to OpenQASM and submitted to the
   configured backend (:class:`~divi.backends.CircuitRunner`).  This step is handled automatically.

3. **Reduce** (backward pass) — Stages are visited in *reverse* order and each
   one collapses or aggregates the raw results using a token it saved during the
   expand pass.  The pipeline returns the fully reduced result to the caller.

.. mermaid::

   flowchart TB
       subgraph row1["Expand (Forward)"]
           direction LR
           A[SpecStage] --> B[BundleStage #1]
           B --> C[BundleStage …]
       end
       subgraph row2["Execute"]
           EXEC[Execute]
       end
       subgraph row3["Reduce (Backward)"]
           direction RL
           R1[Raw results] --> R2[Intermediate result]
           R2 --> R3[Final result]
       end
       row1 --> row2
       row2 --> row3
       style row1 fill:#CC3366,stroke:#e8e8e8
       style row2 fill:#CC3366,stroke:#e8e8e8
       style row3 fill:#CC3366,stroke:#e8e8e8

Pipeline data model
~~~~~~~~~~~~~~~~~~~

Batches and results are keyed by **node keys** so that multi-stage expansion
and reduction stay consistent:

- **NodeKey** (from :mod:`divi.pipeline`): A tuple of ``(axis_name, value)``
  pairs.  A single-circuit batch has a key like ``(("circuit", 0),)``.  As
  stages fan out the batch, axes are appended — e.g.
  ``(("circuit", 0), ("obs_group", 2))`` after measurement grouping.  Keys are
  preserved from the spec stage's ``expand`` through execute and into each
  stage's ``reduce``.

- **MetaCircuitBatch**: A ``dict[NodeKey, MetaCircuit]``. The spec stage produces
  this; bundle stages consume and produce batches (or expansion results) keyed
  by the same or extended keys.

- **Flow**: Spec ``expand`` → one batch of :class:`~divi.circuits.MetaCircuit` →
  bundle stages add axes (e.g. parameter sets, measurement groups) → execute
  compiles to OpenQASM and runs on the backend → **reduce** in reverse order
  collapses results back to the final shape (e.g. a single expectation value or
  a dict of bitstring probabilities per key).

Built-in Stages
---------------

Divi ships with the following built-in stages:

.. list-table::
   :header-rows: 1
   :widths: 25 10 65

   * - Stage
     - Type
     - Description
   * - :class:`~divi.pipeline.stages.CircuitSpecStage`
     - Spec
     - Passes a single :class:`~divi.circuits.MetaCircuit` through as a one-element batch.
       Used by :class:`~divi.qprog.algorithms.VQE`, :class:`~divi.qprog.algorithms.CustomVQA`, and other algorithms that receive a pre-built circuit.
   * - :class:`~divi.pipeline.stages.PennyLaneSpecStage`
     - Spec
     - Converts PennyLane ``QuantumScript`` or ``QNode`` objects into MetaCircuits.
       Supports scalar and array parameters, and ``probs()``, ``expval()``, ``counts()`` measurements.
   * - :class:`~divi.pipeline.stages.QiskitSpecStage`
     - Spec
     - Converts Qiskit ``QuantumCircuit`` objects into MetaCircuits.
       ``ParameterExpression`` objects (e.g. ``2 * theta``) are preserved as sympy expressions.
   * - :class:`~divi.pipeline.stages.TrotterSpecStage`
     - Spec
     - Generates Trotterized circuits from a Hamiltonian for time-evolution and
       :class:`~divi.qprog.algorithms.QAOA` workflows.
   * - :class:`~divi.pipeline.stages.MeasurementStage`
     - Bundle
     - Splits multi-observable Hamiltonians into compatible measurement groups
       (using qubit-wise commutativity or other strategies) and declares the
       result format (counts, probabilities, or expectation values).
   * - :class:`~divi.pipeline.stages.ParameterBindingStage`
     - Bundle
     - Substitutes symbolic parameters with concrete numerical values to produce
       one circuit variant per parameter set.
   * - :class:`~divi.pipeline.stages.QEMStage`
     - Bundle
     - Applies a :class:`~divi.circuits.qem.QEMProtocol` (e.g. ZNE) in the
       expand pass and reduces the scaled results in the reduce pass.
       See :doc:`improving_results_qem` for details.
   * - :class:`~divi.pipeline.stages.PauliTwirlStage`
     - Bundle
     - Generates randomized Pauli-twirl variants of each circuit.
       Used alongside :class:`~divi.pipeline.stages.QEMStage` when the QEM
       protocol requests twirls (e.g. ``QuEPP(n_twirls=10)``).
   * - :class:`~divi.pipeline.stages.PCECostStage`
     - Bundle
     - Computes the custom counts-based objective for ``PCE``-based algorithms.
       In soft mode it evaluates a smooth surrogate from the measured bitstring
       distribution; in hard mode it evaluates a discrete CVaR-style objective
       over sampled energies.


Dry Run
-------

Before executing any circuits you can inspect the pipeline to understand the
total circuit count and how each stage contributes to it.  Call
:meth:`~divi.qprog.QuantumProgram.dry_run` on any quantum program, then pass
the resulting dict to :func:`~divi.pipeline.format_dry_run` for the rich
tree output:

.. skip: next

.. code-block:: python

   from divi.pipeline import format_dry_run

   vqe = VQE(
       molecule=h2_molecule,
       qem_protocol=QuEPP(truncation_order=1, n_twirls=10),
       backend=QiskitSimulator(qiskit_backend="auto"),
   )

   # Runs the forward pass without executing circuits, then pretty-print.
   format_dry_run(vqe.dry_run())

``format_dry_run`` prints a tree for each pipeline showing the fan-out factor
and metadata per stage:

.. code-block:: text

   cost
   ├── CircuitSpecStage [circuit] → 1
   │   ├── n_qubits: 4
   │   ├── n_gates: 4
   │   └── n_2q_gates: 2
   ├── QEMStage [qem_quepp] → ×10
   │   ├── protocol: quepp
   │   ├── n_paths: 9
   │   └── n_clifford_sims: 9
   ├── PauliTwirlStage [twirl] → ×10
   │   └── n_twirls: 10
   ├── MeasurementStage [obs_group] → ×5
   │   ├── strategy: qwc
   │   ├── n_groups: 5
   │   └── n_terms: 14
   ├── ParameterBindingStage [param_set] → 1
   │   └── n_params: 3
   └── Total: 10 × 10 × 5 = 500 circuits

The report shows the multiplicative expansion at each stage.  Use this to
estimate cloud costs, tune ``truncation_order`` or ``n_twirls``, and verify
that measurement grouping behaves as expected — all before spending a single
shot.

``dry_run()`` itself is print-free — it returns a
``dict[str, DryRunReport]`` keyed by pipeline name (e.g. ``"cost"``,
``"measurement"``), so you can inspect the report programmatically instead of
(or in addition to) rendering it:

.. skip: next

.. code-block:: python

   reports = vqe.dry_run()
   print(reports["cost"].total_circuits)   # 500
   print(reports["cost"].stages[3].metadata)  # QEM stage metadata dict

When Dry Run Falls Back
~~~~~~~~~~~~~~~~~~~~~~~

The analytic dry path emits shared DAG references across the branches it
fans out — safe for any downstream stage that either treats those DAGs as
read-only or has its own dry-mode override.  When a downstream stage
instead claims to **consume** DAG bodies (``consumes_dag_bodies=True``) and
provides no ``dry_expand``, the pipeline would risk feeding the same DAG
reference into repeated in-place mutations.  To stay safe, it demotes the
upstream dry-aware stage back to its real ``expand`` for that run and
emits a :class:`~divi.pipeline.DiviPerformanceWarning` naming both
stages.  **The circuit count stays correct**; only the analytic speedup
for the demoted stage is forfeited.

The warning is actionable in two ways: implement ``dry_expand`` on the
downstream stage (the preferred fix — it restores the speedup for every
pipeline that uses it), or, if that stage does not actually mutate body
DAGs in place, declare ``consumes_dag_bodies=False`` on it so the
pipeline no longer sees it as unsafe.


How Existing Algorithms Build Pipelines
---------------------------------------

Every algorithm constructs its pipelines in a ``_build_pipelines`` method.  For
example, :class:`~divi.qprog.algorithms.VQE` builds two pipelines:

.. code-block:: python

   # Simplified from variational_quantum_algorithm.py
   def _build_cost_pipeline(self, spec_stage):
       return CircuitPipeline(stages=[
           spec_stage,              # SpecStage  →  MetaCircuit batch
           QEMStage(protocol=...),  # Apply error mitigation variants
           PauliTwirlStage(...),    # Randomised Pauli twirls (if requested)
           MeasurementStage(...),   # Split observables into groups
           ParameterBindingStage(), # Bind symbolic params → numeric (last!)
       ])

   def _build_measurement_pipeline(self):
       return CircuitPipeline(stages=[
           CircuitSpecStage(),       # Single-circuit spec
           MeasurementStage(),       # Probability measurement
           ParameterBindingStage(),  # Bind best params
       ])

The **cost pipeline** evaluates expectation values during optimization (with
optional error mitigation), while the **measurement pipeline** samples the
probability distribution after optimization to extract the solution.

**Stage ordering affects performance.**  Because each stage in the expand pass
fans out the batch it receives, any work-multiplying stage placed early forces
every downstream stage to repeat its logic across a larger batch.  Conversely,
placing a fan-out stage late keeps the batch small for as long as possible.

The most concrete example is ``ParameterBindingStage``.  By default it runs
last — structural stages process the symbolic circuit once instead of repeating
work per parameter set.  When using
:class:`~divi.circuits.quepp.QuEPP`, this means QuEPP cannot normalize rotation
angles, which may produce more Pauli paths.  If this is a concern (check with
``dry_run()``), set ``QuEPP(bind_before_mitigation=True)`` to bind parameters
first — fewer paths per circuit, but more total mitigation work across parameter
sets.


Example 1: Custom Algorithm with CustomVQA
------------------------------------------

The simplest way to run a custom parameterized circuit through the pipeline is
:class:`~divi.qprog.algorithms.CustomVQA`.  It wraps a **PennyLane QuantumScript** (or a
Qiskit ``QuantumCircuit``) and optimizes its parameters end-to-end, reusing all
the VQA infrastructure.

The following example finds the ground-state energy of a two-qubit transverse-
field Ising model:

.. math::

   H = -Z_0 Z_1 + 0.5\,X_0 + 0.5\,X_1

.. code-block:: python

   import pennylane as qp
   from divi.qprog import CustomVQA
   from divi.qprog.optimizers import ScipyOptimizer, ScipyMethod
   from divi.backends import MaestroSimulator

   # 1. Define the Hamiltonian (observable to minimize)
   H = -1.0 * qp.Z(0) @ qp.Z(1) + 0.5 * qp.X(0) + 0.5 * qp.X(1)

   # 2. Build a parameterized ansatz as a QuantumScript
   ops = [
       qp.RY(0.0, wires=0),
       qp.RY(0.0, wires=1),
       qp.CNOT(wires=[0, 1]),
       qp.RY(0.0, wires=0),
       qp.RY(0.0, wires=1),
   ]
   measurements = [qp.expval(H)]
   qscript = qp.tape.QuantumScript(ops=ops, measurements=measurements)

   # Mark only the gate parameters as trainable (freeze Hamiltonian coefficients)
   qscript.trainable_params = [0, 1, 2, 3]

   # 3. Create the CustomVQA program — it builds a pipeline internally
   program = CustomVQA(
       qscript,
       param_shape=(4,),
       max_iterations=10,
       backend=MaestroSimulator(),
       optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
       seed=42,
   )

   # 4. Run — the pipeline handles circuit compilation, submission, and reduction
   program.run()

   print(f"Ground-state energy: {program.best_loss:.4f}")
   print(f"Optimal parameters: {program.best_params}")

Under the hood, :class:`~divi.qprog.algorithms.CustomVQA` builds a cost pipeline identical to :class:`~divi.qprog.algorithms.VQE`'s:

.. code-block:: text

   CircuitSpecStage → QEMStage → MeasurementStage → ParameterBindingStage

You receive all VQA features (loss history, best parameters, checkpointing)
without writing any pipeline or stage code.


Example 2: Standalone Pipelines with PennyLane and Qiskit
----------------------------------------------------------

You can run PennyLane or Qiskit circuits directly through a pipeline using the
converter spec stages — no ``QuantumProgram`` required.

**PennyLane QuantumScript:**

.. code-block:: python

   import pennylane as qp
   from divi.pipeline import CircuitPipeline, PipelineEnv
   from divi.pipeline.stages import PennyLaneSpecStage, MeasurementStage
   from divi.backends import MaestroSimulator

   qscript = qp.tape.QuantumScript(
       ops=[qp.Hadamard(0), qp.CNOT(wires=[0, 1])],
       measurements=[qp.probs()],
   )

   pipeline = CircuitPipeline(stages=[
       PennyLaneSpecStage(),
       MeasurementStage(),
   ])

   env = PipelineEnv(backend=MaestroSimulator())
   result = pipeline.run(initial_spec=qscript, env=env)
   print(result.value)  # {"00": ~0.5, "11": ~0.5}

**Qiskit QuantumCircuit:**

.. code-block:: python

   from qiskit import QuantumCircuit
   from divi.pipeline import CircuitPipeline, PipelineEnv
   from divi.pipeline.stages import QiskitSpecStage, MeasurementStage
   from divi.backends import MaestroSimulator

   qc = QuantumCircuit(2, 2)
   qc.h(0)
   qc.cx(0, 1)
   qc.measure([0, 1], [0, 1])

   pipeline = CircuitPipeline(stages=[
       QiskitSpecStage(),
       MeasurementStage(),
   ])

   env = PipelineEnv(backend=MaestroSimulator())
   result = pipeline.run(initial_spec=qc, env=env)
   print(result.value)  # {"00": ~0.5, "11": ~0.5}

Both stages accept single circuits, sequences, or mappings as input.


Example 3: Writing a Custom SpecStage
--------------------------------------

For full control you can write a custom :class:`~divi.pipeline.SpecStage` and
construct a :class:`~divi.pipeline.CircuitPipeline` directly.  This is useful
when the built-in spec stages don't cover your circuit-generation logic.

A ``SpecStage`` must implement two methods:

- ``expand(spec, env)`` — Convert an input specification into a keyed batch of
  :class:`~divi.circuits.MetaCircuit` objects and return a token for later use.
- ``reduce(results, env, token)`` — Aggregate the per-key results back into a
  single output using the stored token.

The following example implements a spec stage that creates a simple
Bell-state circuit and measures its probabilities:

.. code-block:: python

   from qiskit import QuantumCircuit
   from qiskit.converters import circuit_to_dag

   from divi.circuits import MetaCircuit
   from divi.pipeline import CircuitPipeline, PipelineEnv, SpecStage
   from divi.pipeline.abc import MetaCircuitBatch
   from divi.pipeline.stages import MeasurementStage
   from divi.backends import MaestroSimulator

   class BellSpecStage(SpecStage):
       """Spec stage that produces a Bell-state circuit."""

       def __init__(self):
           super().__init__(name="bell")

       @property
       def axis_name(self):
           return None          # No fan-out axis

       @property
       def stateful(self):
           return False         # Deterministic — safe to cache

       def expand(self, spec, env):
           # Build the Bell-state circuit as a Qiskit QuantumCircuit and
           # lower it to a DAG — MetaCircuit stores tagged DAGs as its
           # working IR. The empty tuple ``()`` is this body's tag
           # (``QASMTag``); downstream stages extend the tag as they
           # rewrite the body.
           qc = QuantumCircuit(2)
           qc.h(0)
           qc.cx(0, 1)
           meta = MetaCircuit(
               circuit_bodies=(((), circuit_to_dag(qc)),),
               measured_wires=(0, 1),   # probs() over both qubits
           )

           # NodeKey: tuple of (axis_name, value); one entry for a single circuit
           batch: MetaCircuitBatch = {(("bell", 0),): meta}
           return batch, None   # No reduce token needed

       def reduce(self, results, env, token):
           return results       # Pass results through unchanged


   # Build a minimal pipeline
   pipeline = CircuitPipeline(stages=[
       BellSpecStage(),
       MeasurementStage(),   # Declares probability-mode results
   ])

   # Run the pipeline
   backend = MaestroSimulator()
   env = PipelineEnv(backend=backend)
   result = pipeline.run(initial_spec=None, env=env)

   print(result)
   # Result is keyed by NodeKey: result[(("bell", 0),)] ≈ {"00": ~0.5, "11": ~0.5}

This pattern composes naturally — you can insert any ``BundleStage`` between the
spec stage and the measurement stage to add parameter binding, error mitigation,
or any custom transformation.


.. _adaptive-shot-allocation:

Adaptive Shot Allocation
------------------------

By default, every measurement group produced by
:class:`~divi.pipeline.stages.MeasurementStage` is sampled with the
backend's full shot count — even tiny terms with little impact on the
final energy.  Setting the ``shot_distribution`` argument splits the same
total budget across groups according to their importance, reducing
estimator variance without spending more shots:

.. code-block:: python

   from divi.pipeline.stages import MeasurementStage

   # Concentrate shots on dominant Hamiltonian terms
   MeasurementStage(grouping_strategy="qwc", shot_distribution="weighted")

The available strategies (see :data:`~divi.pipeline.ShotDistStrategy`):

- ``"uniform"`` — equal split across groups.
- ``"weighted"`` — proportional to per-group coefficient L1 norm; dominant
  Hamiltonian terms get more shots (largest-remainder rounding preserves
  the total exactly).
- ``"weighted_random"`` — multinomial sample of the same probabilities.
  Reproducible when ``env.rng`` is seeded; may drop more low-weight
  groups than the deterministic ``"weighted"`` for the same budget.
- A callable ``(group_l1_norms, total_shots) -> per_group_shots`` for
  fully custom allocation.

Variational algorithms accept the same option directly as a constructor
keyword (e.g. ``VQE(..., shot_distribution="weighted")``); it is threaded
through to the cost pipeline's measurement stage.  See
:doc:`ground_state_energy_estimation_vqe` for an end-to-end example.

When a group ends up with zero allocated shots its measurement circuit is
skipped and its observables contribute zero to the final estimate.  The
stage emits a :class:`UserWarning` reporting the dropped fraction of the
Hamiltonian's L1 norm so you can quantify the resulting bias.

Adaptive shot allocation only applies to sampling-based execution.
Combining ``shot_distribution`` with the analytical
``grouping_strategy="_backend_expval"`` path (which divi auto-selects on
expval-capable backends like :class:`~divi.backends.MaestroSimulator`)
raises a :class:`ValueError` — pass an explicit
``grouping_strategy="qwc"`` (or ``"wires"`` / ``None``) to opt into
sampling.


Stage Validation
----------------

The pipeline validates stage ordering at construction time.  Built-in stages
declare their own constraints — for example, :class:`~divi.pipeline.stages.QEMStage`
with QuEPP requires a measurement-handling stage after it.

Custom stages can participate in this by overriding the ``validate`` method:

.. skip: next

.. code-block:: python

   from divi.pipeline.abc import ContractViolation

   class MyStage(BundleStage):
       def validate(self, before, after):
           if not any(isinstance(s, MeasurementStage) for s in after):
               raise ContractViolation(
                   "MyStage requires a MeasurementStage after it."
               )

The ``before`` and ``after`` arguments are tuples of stage instances, so you can
inspect any property (``handles_measurement``, ``axis_name``, protocol
attributes, etc.) to decide whether the pipeline is valid.  Violations raise
:class:`~divi.pipeline.abc.ContractViolation` with an actionable error message.

Stages that don't override ``validate`` impose no constraints — the default is a
no-op.


What's Next
-----------

- :doc:`../api_reference/pipeline` — pipeline and stage classes
- :doc:`improving_results_qem` — :class:`~divi.circuits.qem.QEMProtocol` and error mitigation
- :doc:`../api_reference/qprog/algorithms` — :class:`~divi.qprog.algorithms.CustomVQA` and custom circuits
- :doc:`program_ensembles` — parameter sweeps and orchestration
