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

- **Reading single-circuit results**: Use :attr:`~divi.pipeline.PipelineResult.value`
  for the natural shape — a scalar for single-observable expectation values,
  a ``list[float]`` for multi-observable runs, a ``dict`` for probabilities and
  counts.  ``result[()]`` is the canonical key for the pipeline-internal form
  *after* the spec stage strips its own axis; it is not universally available
  — it depends on the spec stage's ``reduce`` collapsing the circuit axis.
  Built-in spec stages (``CircuitSpecStage``, ``PennyLaneSpecStage``,
  ``QiskitSpecStage``) do this automatically for single-circuit batches.
  Custom spec stages must mirror that behavior if you want ``result[()]`` to work.

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


.. _dry-run:

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

``format_dry_run`` prints a tree for each pipeline showing the per-stage
factor (fan-out or reduction) and metadata:

.. code-block:: text

   cost
   ├── CircuitSpecStage [circuit] → 14
   │   ├── n_qubits: 4
   │   ├── n_gates: 4
   │   └── n_2q_gates: 2
   ├── QEMStage [qem_quepp] → ×10
   │   ├── protocol: quepp
   │   ├── n_paths: 9
   │   └── n_clifford_sims: 9
   ├── PauliTwirlStage [twirl] → ×10
   │   └── n_twirls: 10
   ├── MeasurementStage [obs_group] → ÷2.8
   │   ├── strategy: qwc
   │   ├── n_groups: 5
   │   └── n_terms: 14
   ├── ParameterBindingStage [param_set] → 1
   │   └── n_params: 3
   └── Total: 14 × 10 × 10 ÷ 2.8 = 500 circuits

The spec stage's number (here ``14``) is the naive baseline: one circuit per
Pauli term in the observable.  Stages that *fan out* show up as ``×K`` (QEM
path enumeration, Pauli twirling); stages that *reduce* show up as ``÷K``
(observable grouping collapsing commuting Pauli terms into shared
measurement circuits).  Use this to estimate cloud costs, tune
``truncation_order`` or ``n_twirls``, and see at a glance how much grouping
saves — all before spending a single shot.

``dry_run()`` itself is print-free — it returns a
``dict[str, DryRunReport]`` keyed by pipeline name (e.g. ``"cost"``,
``"measurement"``), so you can inspect the report programmatically instead of
(or in addition to) rendering it.  Note that ``dry_run()`` only reports
**pre-registered** named pipelines (those the algorithm builds in its
constructor or via its ``_preprocessors`` hook).  A program that assembles its
pipeline dynamically inside ``run()`` — i.e. never calling ``evaluate()`` with
a named preprocessor — returns ``{}`` from ``dry_run()``.

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

A program declares **how it prepares its state** and **which measurement
protocols** it runs over that state; one verb,
:meth:`~divi.qprog.QuantumProgram.evaluate`, assembles and runs the single
pipeline for any protocol.  State preparation is two hooks —
``_spec_stage`` (the :class:`~divi.pipeline.SpecStage`) and
``_initial_spec`` (its seed).  A
:class:`~divi.pipeline.CircuitPreprocessor` pairs a post-spec ``MetaCircuit``
transform with a :class:`~divi.pipeline.ResultFormat` and an optional terminal
stage.  :class:`~divi.qprog.algorithms.VQE` exposes a ``cost`` protocol;
solution-extracting programs add a ``sample`` protocol from
:class:`~divi.qprog.SolutionSamplingMixin`:

.. code-block:: python

   # Simplified from quantum_program.py / variational_quantum_algorithm.py
   def _spec_stage(self):
       return CircuitSpecStage()                  # SpecStage → MetaCircuit batch

   def _initial_spec(self):
       # VariationalQuantumAlgorithm: returns self.cost_circuit (the cost ansatz).
       # QAOA / TimeEvolution override this to return their Hamiltonian instead.
       return self.cost_circuit

   def cost_preprocessor(self):
       # Public + overridable (PCE returns a counts-based variant).
       return CircuitPreprocessor("cost", cache_key="cost")   # identity, EXPVALS

   # A caller (e.g. an optimizer) measures the prepared state through one verb:
   #   losses = program.evaluate(params, program.cost_preprocessor())

The single pipeline is assembled by ``_assemble_pipeline`` — spec → the
protocol's ``PreprocessStage`` (its post-spec transform) → [error mitigation
(+ Pauli twirls) when the QEM protocol applies to the result format] → terminal
measurement.  :class:`~divi.qprog.VariationalQuantumAlgorithm` appends a
``ParameterBindingStage`` last; the base :class:`~divi.qprog.QuantumProgram`
does **not** — a direct ``QuantumProgram`` subclass with a parameterized seed
must add :class:`~divi.pipeline.stages.ParameterBindingStage` itself (or
subclass ``VariationalQuantumAlgorithm``), or execution raises
:class:`~divi.pipeline.abc.ContractViolation`.

The **cost protocol** evaluates expectation values (or a classical objective)
during optimization; the **sample protocol** samples the probability
distribution afterward to extract the solution.  Whether error mitigation rides
a protocol is decided by the QEM protocol itself
(:meth:`~divi.circuits.qem.QEMProtocol.applies_to`), so extrapolation-style
mitigation rides expectation-value protocols but not the probability-sampling
one. Natural-gradient optimizers measure their metric by passing a dynamic
:class:`~divi.pipeline.CircuitPreprocessor` to
:meth:`~divi.qprog.QuantumProgram.evaluate`, so a metric is never a separate
registered pipeline.

**Stage ordering affects performance.**  Because each stage in the expand pass
fans out the batch it receives, any work-multiplying stage placed early forces
every downstream stage to repeat its logic across a larger batch.  Conversely,
placing a fan-out stage late keeps the batch small for as long as possible.

The most concrete example is ``ParameterBindingStage``.  By default it runs
last — structural stages process the symbolic circuit once instead of repeating
work per parameter set.  When using
:class:`~divi.circuits.quepp.QuEPP`, this means QuEPP cannot normalize rotation
angles, which may produce more Pauli paths. ``QuEPP(sampling="exhaustive")``
binds parameters first — fewer paths per circuit, but more total mitigation work
across parameter sets. ``QuEPP(sampling="montecarlo")`` keeps the cheaper
symbolic ordering.


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

**Passthrough constructor kwargs.** :class:`~divi.qprog.algorithms.CustomVQA`,
:class:`~divi.qprog.algorithms.VQE`, and :class:`~divi.qprog.algorithms.QNN`
all route additional keyword arguments through ``**kwargs`` into the cost
pipeline's :class:`~divi.pipeline.stages.MeasurementStage`.  Because they
aren't explicit parameters in the subclass signature, two useful options are
easy to miss:

- ``grouping_strategy`` (``str``, default ``"qwc"``) — how Hamiltonian terms
  are partitioned into measurement circuits (``"qwc"``, ``"wires"``, or
  ``None``).
- ``shot_distribution`` (``str`` or callable, default ``None``) — how the
  shot budget is split across measurement groups.  See
  `Adaptive Shot Allocation`_ for details and available strategies.

Example::

    vqe = VQE(molecule=mol, ..., shot_distribution="weighted", grouping_strategy="qwc")


Feeding Parameter Values to a Standalone Pipeline
--------------------------------------------------

A standalone :class:`~divi.pipeline.CircuitPipeline` (built by hand, not via a
program) reads parameter values from :class:`~divi.pipeline.PipelineEnv` —
specifically ``env.param_sets``, a 2-D array-like of shape
``(n_param_sets, n_params)``.  There is **no** ``params=`` argument on
:meth:`~divi.pipeline.CircuitPipeline.run`; all per-run data flows through
the env.

The following example evaluates ⟨Z⟩ for two angles of a single-qubit Ry
rotation:

.. code-block:: python

   import numpy as np
   from qiskit import QuantumCircuit
   from qiskit.circuit import Parameter
   from qiskit.converters import circuit_to_dag
   from qiskit.quantum_info import SparsePauliOp

   from divi.circuits import MetaCircuit
   from divi.pipeline import CircuitPipeline, PipelineEnv, extract_param_set_idx
   from divi.pipeline.stages import CircuitSpecStage, MeasurementStage, ParameterBindingStage
   from divi.backends import MaestroSimulator

   # 1. Build a parametric circuit
   ry_theta = Parameter("ry_theta")
   ry_qc = QuantumCircuit(1)
   ry_qc.ry(ry_theta, 0)

   ry_meta = MetaCircuit(
       circuit_bodies=(((), circuit_to_dag(ry_qc)),),
       observable=SparsePauliOp.from_list([("Z", 1.0)]),
       parameters=(ry_theta,),
   )

   # 2. Assemble pipeline — ParameterBindingStage must come after measurement
   ry_pipeline = CircuitPipeline(stages=[
       CircuitSpecStage(),
       MeasurementStage(),
       ParameterBindingStage(),   # reads env.param_sets; placed last
   ])

   # 3. Pass parameter values through PipelineEnv
   ry_env = PipelineEnv(
       backend=MaestroSimulator(),
       param_sets=[[0.0], [np.pi / 2]],   # 2 param sets, 1 param each
   )
   ry_result = ry_pipeline.run(initial_spec=ry_meta, env=ry_env)

   # 4. Read results back by param-set index using extract_param_set_idx
   # ry_result.items() yields (NodeKey, value) pairs; NodeKeys look like
   # (("param_set", 0),) — extract_param_set_idx pulls the int index out.
   # For EXPVALS, each value is list[float] (unsqueezed); [0] gets the scalar.
   by_idx = {extract_param_set_idx(k): v[0] for k, v in ry_result.items()}
   # ⟨Z⟩ for theta=0 (|0⟩) ≈ 1.0; for theta=π/2 (|+y⟩) ≈ 0.0
   assert abs(by_idx[0] - 1.0) < 0.15, f"Expected ~1.0, got {by_idx[0]}"
   assert abs(by_idx[1] - 0.0) < 0.15, f"Expected ~0.0, got {by_idx[1]}"

**Other useful** :class:`~divi.pipeline.PipelineEnv` **fields:**

- ``shots_override`` — overrides ``backend.shots`` for this run without
  mutating the backend (useful when adapting shot counts per evaluation).
- ``collect_variance`` — when ``True``, measurement stages also estimate
  shot-noise variance and record it on the env's own ``env.artifacts``
  dict under ``"cost_variance"`` (keyed by :class:`~divi.pipeline.NodeKey`,
  *not* on the returned :class:`~divi.pipeline.PipelineResult`).  Most callers
  don't set this directly:
  :meth:`~divi.qprog.QuantumProgram.evaluate` with ``return_variance=True``
  flips it on and returns ``(values, per_set_variances)``.
- ``axes_to_preserve`` — tuple of axis names that should not be reduced away
  by downstream stages (advanced use; needed when you want branch-level
  results after normal reductions).
- ``feature_batch`` — classical feature matrix ``(n_samples, n_features)``
  read by :class:`~divi.pipeline.stages.DataBindingStage` (QNN / CustomVQA
  data-binding path; ``None`` for circuits without a data axis).
- ``rng`` — ``numpy.random.Generator`` for stochastic stage decisions (e.g.
  ``"weighted_random"`` shot allocation); when ``None``, stages construct a
  fresh unseeded generator (non-reproducible).


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

.. tip::

   ``result.value`` squeezes results to the natural shape: a scalar
   ``float`` for a single ``qp.expval(...)`` measurement, a ``list[float]``
   for several measurements (or when the user explicitly wrapped a single
   observable in a list), and a ``dict`` for ``qp.probs`` / ``qp.counts``.
   ``result[()]`` gives the pipeline-internal form without squeezing —
   always a ``list[float]`` for expectation values — but only when the
   spec stage's ``reduce`` collapses the circuit axis (which the built-in
   stages do).  ``evaluate(...)`` returns ``{param_set_idx: value}`` where
   the value is **not** squeezed — e.g. ``{0: [1.0]}`` for a single
   expectation value, not ``{0: 1.0}``; use ``result.value`` for the
   squeezed scalar.


Example 3: Writing a Custom SpecStage
--------------------------------------

For full control you can write a custom :class:`~divi.pipeline.SpecStage` and
construct a :class:`~divi.pipeline.CircuitPipeline` directly.  This is useful
when the built-in spec stages don't cover your circuit-generation logic.

A ``SpecStage`` must implement two methods:

- ``expand(spec, env)`` — Convert an input specification into a keyed batch of
  :class:`~divi.circuits.MetaCircuit` objects and return a
  :class:`~divi.pipeline.StageOutput`.
- ``reduce(results, env, token)`` — Aggregate the per-key results back into a
  single output using the stored token.

Each :class:`~divi.pipeline.CircuitPipeline` memoizes its forward pass and
reuses it on identical inputs, so a deterministic stage needs no extra
declaration. Override ``cache_key_extras`` to list any live ``env`` inputs
``expand`` reads beyond its batch — for example ``env.backend.shots`` or
``env.evaluation_counter`` — so the cache invalidates when they change; set
``volatile`` to re-run the stage on every forward pass. Stages that decide the
measurement record that metadata — the result format and any per-group shot
allocation — on each :class:`~divi.circuits.MetaCircuit` they emit.

The following example implements a spec stage that creates a simple
Bell-state circuit and measures its probabilities:

.. code-block:: python

   from qiskit import QuantumCircuit
   from qiskit.converters import circuit_to_dag

   from divi.circuits import MetaCircuit
   from divi.pipeline import (
       CircuitPipeline,
       PipelineEnv,
       SpecStage,
       StageOutput,
       group_by_base_key,
   )
   from divi.pipeline.abc import MetaCircuitBatch
   from divi.pipeline.stages import MeasurementStage
   from divi.backends import MaestroSimulator

   class BellSpecStage(SpecStage):
       """Spec stage that produces a Bell-state circuit."""

       def __init__(self):
           super().__init__(name="bell")

       @property
       def axis_name(self):
           return "bell"

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
           return StageOutput(batch=batch)

       def reduce(self, results, env, token):
           # Strip the "bell" axis — mirrors how CircuitSpecStage.reduce works.
           # Groups child results by their base key (without the "bell" axis)
           # so that a single-circuit batch collapses to key ().
           grouped = group_by_base_key(results, self.axis_name, indexed=False)
           return {
               key: values[0] if len(values) == 1 else values
               for key, values in grouped.items()
           }


   # Build a minimal pipeline
   pipeline = CircuitPipeline(stages=[
       BellSpecStage(),
       MeasurementStage(),   # Declares probability-mode results
   ])

   # Run the pipeline
   env = PipelineEnv(backend=MaestroSimulator())
   result = pipeline.run(initial_spec=None, env=env)

   # BellSpecStage.reduce strips the "bell" axis, so the result collapses
   # to key () — use result.value for the natural dict shape.
   probs = result.value   # ≈ {"00": ~0.5, "11": ~0.5}
   assert isinstance(probs, dict)
   assert set(probs.keys()) == {"00", "11"} or len(probs) >= 1
   # result[()] is equivalent when the spec axis has been stripped.
   assert result[()] == probs

This pattern composes naturally — you can insert any ``BundleStage`` between the
spec stage and the measurement stage to add parameter binding, error mitigation,
or any custom transformation.


Example 4: Writing a Custom BundleStage
-----------------------------------------

A :class:`~divi.pipeline.BundleStage` fans out a :class:`~divi.circuits.MetaCircuit`
batch by appending axis-tagged bodies to ``meta.circuit_bodies`` — it does **not**
extend the ``NodeKey`` in ``expand``.  The axis name (returned by the stage's
``axis_name`` property) appears as a new ``(axis_name, idx)`` pair appended to
each body's ``QASMTag`` tuple.  After execute, ``reduce`` uses
:func:`~divi.pipeline.group_by_base_key` to strip that suffix and collapse
the fan-out back to the parent key.

The canonical reference is :class:`~divi.pipeline.stages.PauliTwirlStage`:
its ``_expand_structural`` method iterates ``meta.circuit_bodies``, computes
twirl variants, and emits one MetaCircuit per parent key with all variants as
separate tagged bodies via ``meta.set_circuit_bodies(tuple(updated_bodies))``.

The following minimal example replicates each circuit body twice along a
``"replica"`` axis and averages the results in ``reduce``:

.. code-block:: python

   from qiskit import QuantumCircuit
   from qiskit.converters import circuit_to_dag
   from qiskit.quantum_info import SparsePauliOp

   from divi.circuits import MetaCircuit
   from divi.pipeline import (
       BundleStage,
       CircuitPipeline,
       PipelineEnv,
       StageOutput,
       group_by_base_key,
       reduce_mean,
   )
   from divi.pipeline.abc import MetaCircuitBatch
   from divi.pipeline.stages import CircuitSpecStage, MeasurementStage
   from divi.backends import MaestroSimulator

   N_REPLICAS = 2

   class ReplicaBundleStage(BundleStage):
       """Fan out each circuit into N identical replicas and average results."""

       def __init__(self, n: int = N_REPLICAS):
           super().__init__(name="replica")
           self._n = n

       @property
       def axis_name(self):
           return "replica"

       def expand(self, batch: MetaCircuitBatch, env: PipelineEnv) -> StageOutput:
           out: MetaCircuitBatch = {}
           for parent_key, meta in batch.items():
               # Fan out: append (axis_name, idx) to each body's QASMTag.
               # Each entry in circuit_bodies is (QASMTag, DAGCircuit).
               new_bodies = []
               for body_tag, dag in meta.circuit_bodies:
                   for i in range(self._n):
                       # Extend the tag tuple with the replica axis label.
                       new_tag = (*body_tag, (self.axis_name, i))
                       new_bodies.append((new_tag, dag))
               # set_circuit_bodies returns a new immutable MetaCircuit copy.
               out[parent_key] = meta.set_circuit_bodies(tuple(new_bodies))
           return StageOutput(batch=out)

       def reduce(self, results, env, token):
           # Strip the "replica" axis and average grouped expectation values.
           grouped = group_by_base_key(results, self.axis_name, indexed=False)
           return reduce_mean(grouped)


   # Build and run a minimal pipeline using the custom bundle stage.
   # CircuitSpecStage wraps the MetaCircuit and assigns the "circuit" axis;
   # ReplicaBundleStage appends a "replica" axis to each body.
   qc = QuantumCircuit(1)
   qc.h(0)
   meta = MetaCircuit(
       circuit_bodies=(((), circuit_to_dag(qc)),),
       observable=SparsePauliOp.from_list([("Z", 1.0)]),
   )

   pipeline = CircuitPipeline(stages=[
       CircuitSpecStage(),
       ReplicaBundleStage(n=N_REPLICAS),
       MeasurementStage(),
   ])

   env = PipelineEnv(backend=MaestroSimulator())
   result = pipeline.run(initial_spec=meta, env=env)
   expval = result.value   # scalar float — averaged over N_REPLICAS replicas
   assert isinstance(expval, float)

The key mechanic: ``set_circuit_bodies`` replaces the body list on an
**immutable** :class:`~divi.circuits.MetaCircuit` (backed by
``dataclasses.replace``), so each stage works on its own copy.  The tag suffix
``(axis_name, idx)`` is the pipeline's bookkeeping token; ``reduce`` uses
:func:`~divi.pipeline.group_by_base_key` to strip that suffix and collapse
values back to the parent key.  Use :func:`~divi.pipeline.reduce_mean` for
EXPVALS, :func:`~divi.pipeline.reduce_merge_histograms` for PROBS/COUNTS.


Example 5: Custom QuantumProgram with evaluate
-----------------------------------------------

For full control over state preparation and measurement, subclass
:class:`~divi.qprog.QuantumProgram` directly and implement ``_spec_stage`` and
``_initial_spec``.  Call :meth:`~divi.qprog.QuantumProgram.evaluate` with a
:class:`~divi.pipeline.CircuitPreprocessor` to measure the prepared state.

:meth:`~divi.qprog.QuantumProgram.run` is ``@abstractmethod`` — every
``QuantumProgram`` subclass must implement it, even as a thin wrapper, or
instantiation raises ``TypeError: Can't instantiate abstract class``.

``_initial_spec`` is only required when your subclass calls ``evaluate``; it
is intentionally not abstract so programs that never call ``evaluate`` (e.g.
those that assemble their own pipeline directly inside ``run``) do not need to
implement it.

A :class:`~divi.circuits.MetaCircuit` for EXPVALS mode is constructed with an
``observable`` keyword (a ``SparsePauliOp``); for PROBS/COUNTS mode use
``measured_wires`` instead.

Direct ``QuantumProgram`` subclasses with parameterized seed circuits must
override ``_assemble_pipeline`` to add :class:`~divi.pipeline.stages.ParameterBindingStage`
themselves — the base class does not append one.  Skipping it raises
:class:`~divi.pipeline.abc.ContractViolation` at execution time.

.. code-block:: python

   import numpy as np
   from qiskit import QuantumCircuit
   from qiskit.circuit import Parameter
   from qiskit.converters import circuit_to_dag
   from qiskit.quantum_info import SparsePauliOp

   from divi.circuits import MetaCircuit
   from divi.backends import MaestroSimulator
   from divi.pipeline import CircuitPipeline, CircuitPreprocessor, ResultFormat
   from divi.pipeline.stages import (
       CircuitSpecStage,
       MeasurementStage,
       ParameterBindingStage,
   )
   from divi.qprog import QuantumProgram

   class SingleQubitRotation(QuantumProgram):
       """Minimal QuantumProgram subclass: parameterized Ry rotation, measures Z."""

       def __init__(self, backend):
           super().__init__(backend=backend)
           theta = Parameter("theta")
           qc = QuantumCircuit(1)
           qc.ry(theta, 0)
           self._meta = MetaCircuit(
               circuit_bodies=(((), circuit_to_dag(qc)),),
               observable=SparsePauliOp.from_list([("Z", 1.0)]),
               parameters=(theta,),
           )
           self._result = None

       def has_results(self) -> bool:
           return self._result is not None

       def _spec_stage(self):
           return CircuitSpecStage()

       def _initial_spec(self):
           return self._meta

       def _assemble_pipeline(self, spec_stage, terminal_stage, *, result_format, extra_stages=()):
           # Direct QuantumProgram subclasses with parameterized circuits must add
           # ParameterBindingStage — the base class does not.
           return CircuitPipeline(stages=[
               spec_stage,
               *extra_stages,
               *self._mitigation_stages(result_format),
               terminal_stage,
               ParameterBindingStage(),   # must come last
           ])

       def run(self):
           preprocessor = CircuitPreprocessor(
               name="cost",
               result_format=ResultFormat.EXPVALS,
               terminal_stage=MeasurementStage(),
           )
           # evaluate() returns {param_set_idx: value} — value is unsqueezed:
           # a list[float] for EXPVALS, not a scalar.
           params = np.array([[0.0]])   # shape (n_param_sets=1, n_params=1)
           raw = self.evaluate(params, preprocessor)
           # raw == {0: [1.0]}  for theta=0 (|0⟩ state, ⟨Z⟩=1.0) — unsqueezed list
           self._result = raw[0][0]    # index the list to get the scalar

   program = SingleQubitRotation(backend=MaestroSimulator())
   program.run()
   # ⟨Z⟩ for theta=0 (|0⟩ state) ≈ 1.0
   assert abs(program._result - 1.0) < 0.15

``evaluate()`` returns ``{param_set_idx: value}`` where ``value`` is the
**unsqueezed** pipeline-internal form — e.g. ``{0: [1.0]}`` for a single
expectation value, not ``{0: 1.0}``.  Use
:attr:`~divi.pipeline.PipelineResult.value` (on the result of ``pipeline.run(...)``)
for the auto-squeezed scalar, or index the list directly as shown above.


Example 6: Injecting a Custom Stage into an Optimizer-Driven Algorithm
-----------------------------------------------------------------------

:class:`~divi.qprog.VariationalQuantumAlgorithm` (the base of
:class:`~divi.qprog.algorithms.CustomVQA`, :class:`~divi.qprog.algorithms.VQE`,
:class:`~divi.qprog.algorithms.QNN`, etc.) assembles its pipeline in
``_assemble_pipeline``.  The ``extra_stages`` keyword is the injection seam:
stages passed there are inserted immediately after the spec stage, before any
mitigation stages and the terminal measurement.  To inject a custom stage into a
VQA-family program, override ``_assemble_pipeline`` and delegate to ``super()``:

.. code-block:: python

   import numpy as np
   import pennylane as qp
   from divi.qprog import CustomVQA
   from divi.qprog.optimizers import ScipyOptimizer, ScipyMethod
   from divi.pipeline import ResultFormat
   from divi.backends import MaestroSimulator

   # Re-using ReplicaBundleStage from Example 4 — it is in scope because
   # code-blocks within one .rst file share a namespace and run in order.

   class ReplicatedCustomVQA(CustomVQA):
       """CustomVQA subclass that replicates every circuit N times."""

       def __init__(self, *args, n_replicas: int = 2, **kwargs):
           super().__init__(*args, **kwargs)
           self._n_replicas = n_replicas

       def _assemble_pipeline(self, spec_stage, terminal_stage, *, result_format, extra_stages=()):
           return super()._assemble_pipeline(
               spec_stage,
               terminal_stage,
               result_format=result_format,
               extra_stages=(*extra_stages, ReplicaBundleStage(n=self._n_replicas)),
           )

   # Build a minimal two-qubit Ising Hamiltonian for the test.
   H = -1.0 * qp.Z(0) @ qp.Z(1) + 0.5 * qp.X(0) + 0.5 * qp.X(1)
   ops = [qp.RY(0.0, wires=0), qp.RY(0.0, wires=1), qp.CNOT(wires=[0, 1])]
   qscript = qp.tape.QuantumScript(ops=ops, measurements=[qp.expval(H)])
   qscript.trainable_params = [0, 1]

   program = ReplicatedCustomVQA(
       qscript,
       param_shape=(2,),
       n_replicas=2,
       max_iterations=3,
       backend=MaestroSimulator(),
       optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
       seed=42,
   )

   # Verify the custom stage is present in the cost pipeline. ``dry_run`` is the
   # public way to introspect the assembled stages (no private access needed).
   # ``StageInfo.name`` is the stage *class* name (``type(stage).__name__``), not
   # the ``name=`` you may pass to a stage's constructor — match the class name.
   stage_names = [s.name for s in program.dry_run()["cost"].stages]
   assert "ReplicaBundleStage" in stage_names, f"Expected ReplicaBundleStage in {stage_names}"

   program.run()
   assert program.best_loss is not None

The ``extra_stages`` tuple is passed through every ``_assemble_pipeline``
override in the MRO, so multiple mixins can each append their own stages
without conflicting.  The canonical example of this pattern is
:class:`~divi.qprog.algorithms.PCE`, which injects its preprocessor stage via
exactly this seam.


Stage-Author Toolkit
--------------------

``divi.pipeline`` exposes the reduction helpers the built-in stages use
internally, so a custom stage can reduce results the same way. A ``reduce``
takes and returns a mapping of result key → value.

**Stage-authoring helpers** (use inside ``reduce``):

- :func:`~divi.pipeline.group_by_base_key` — group child results by stripping one
  axis from each key.  Works with any result format.  Pass ``indexed=True`` to
  produce ``{base_key: {int: value}}`` instead of the default
  ``{base_key: [values]}`` list form — the indexed form is required by
  :func:`~divi.pipeline.reduce_postprocess_ordered`.
- :func:`~divi.pipeline.strip_axis_from_label` — drop a single axis from one key.
  Works with any result format.
- :func:`~divi.pipeline.reduce_mean` — average grouped values (scalars or
  per-observable lists).  **Use for EXPVALS** (expectation-value results). Do
  not use for probability or counts dicts — use ``reduce_merge_histograms`` instead.
- :func:`~divi.pipeline.reduce_merge_histograms` — average grouped probability
  histograms across branches.  **Use for PROBS or COUNTS** result formats.
- :func:`~divi.pipeline.reduce_postprocess_ordered` — sort each group by axis
  index, then apply a postprocessing function.  **Works with any result format**;
  used by QEM and observable grouping stages.

**Quick reference: helper → result format**

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Helper
     - Appropriate result format
   * - ``reduce_mean``
     - EXPVALS (scalar floats or per-observable ``list[float]``)
   * - ``reduce_merge_histograms``
     - PROBS or COUNTS (probability / counts dicts)
   * - ``reduce_postprocess_ordered``
     - Any format (sorts by axis index, applies a postprocessor)
   * - ``group_by_base_key``, ``strip_axis_from_label``
     - Any format (key manipulation only)

Each stage names its own fan-out axis (returned from ``axis_name``).
:func:`~divi.pipeline.extract_param_set_idx` parses the param-set index from a
**result key** — a ``NodeKey`` tuple such as the keys in a raw
:class:`~divi.pipeline.PipelineResult` (e.g. ``pipeline.run(...)``) that still
carry the full axis chain.  It iterates the key as ``(axis, idx)`` pairs and
returns the ``param_set`` index.

Do **not** apply it to the output of ``evaluate()`` — that method already
collapses the pipeline-internal keys and returns ``{param_set_idx: value}``,
where each key is a plain ``int``.  The int *is* the param-set index;
calling ``extract_param_set_idx`` on an int raises ``TypeError`` and is
redundant.

.. code-block:: python

   from divi.pipeline import (
       extract_param_set_idx,
       group_by_base_key,
       reduce_mean,
       reduce_postprocess_ordered,
   )

   # --- Non-indexed path (list form) — use with reduce_mean for EXPVALS ---
   results = {(("circ", 0), ("obs", 0)): 1.0, (("circ", 0), ("obs", 1)): 3.0}
   grouped = group_by_base_key(results, "obs")
   # grouped == {(('circ', 0),): [1.0, 3.0]}
   averaged = reduce_mean(grouped)
   assert averaged == {(("circ", 0),): 2.0}

   # --- Indexed path — use with reduce_postprocess_ordered ---
   # indexed=True produces {base_key: {int: value}} so the values can be
   # ordered by axis index before the postprocess function is applied.
   # This is exactly the input reduce_postprocess_ordered expects.
   results2 = {(("circ", 0), ("qem", 0)): 0.8, (("circ", 0), ("qem", 1)): 1.2}
   grouped_indexed = group_by_base_key(results2, "qem", indexed=True)
   # grouped_indexed == {(('circ', 0),): {0: 0.8, 1: 1.2}}
   # Postprocess: sorted by index [0.8, 1.2], then apply fn
   extrapolated = reduce_postprocess_ordered(grouped_indexed, lambda xs: 2 * xs[-1] - xs[0])
   # 2 * 1.2 - 0.8 = 1.6 (allow for floating-point rounding)
   assert abs(extrapolated[(("circ", 0),)] - 1.6) < 1e-9

   # extract_param_set_idx reads from a NodeKey tuple, not from evaluate() output.
   # The key (("param_set", 2), ("obs", 0)) belongs to param-set index 2.
   key = (("param_set", 2), ("obs", 0))
   assert extract_param_set_idx(key) == 2

**Contributing dry-run metadata.** The per-stage ``metadata`` that
:func:`~divi.pipeline.format_dry_run` renders under each stage comes from the
stage's ``introspect(batch, env, token)`` method.  The base ``Stage`` returns
``{}`` (no metadata); override it to surface stage-specific detail in the
dry-run tree and on :attr:`StageInfo.metadata <divi.pipeline.StageInfo>`.  It is
called after ``expand`` with the post-expand batch, so it can report shapes the
stage just produced:

.. code-block:: python

   from typing import Any
   from divi.pipeline.abc import MetaCircuitBatch, PipelineEnv, StageToken

   def introspect(
       self, batch: MetaCircuitBatch, env: PipelineEnv, token: StageToken
   ) -> dict[str, Any]:
       return {"n_variants": self._n}


.. _adaptive-shot-allocation:

Adaptive Shot Allocation
------------------------

By default, every measurement group produced by
:class:`~divi.pipeline.stages.MeasurementStage` is sampled with the
backend's full shot count — so with ``G`` groups the default spends
``G × shots`` in total, even on tiny terms with little impact on the final
energy.  Setting the ``shot_distribution`` argument instead caps the total at a
single ``shots`` budget split across groups by importance.  At that equal
budget it gives lower estimator variance than a ``"uniform"`` split (the
dominant terms get more samples); compare strategies at the *same* total
budget, since the default's per-group full count is a larger budget, not a
fairer baseline:

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

The equal-budget claim is easy to check without spending a shot: a
:meth:`~divi.qprog.QuantumProgram.dry_run` records the per-group allocation
under ``env_artifacts["per_group_shots"]``, so ``"uniform"`` and ``"weighted"``
can be compared at the *same* total budget directly:

.. code-block:: python

   from divi.backends import QiskitSimulator
   from divi.qprog.algorithms import VQE
   from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer

   # A sampling backend (supports_expval=False) so shot_distribution takes effect.
   sampling_backend = QiskitSimulator(force_sampling=True, shots=1200)

   def group_allocation(strategy):
       vqe = VQE(
           molecule=molecule,
           backend=sampling_backend,
           optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
           grouping_strategy="qwc",
           shot_distribution=strategy,
       )
       # per_group_shots maps each spec key to {group_index: shots}; the cost
       # pipeline has a single spec key, so take its allocation.
       report = vqe.dry_run()["cost"]
       allocation = next(iter(report.env_artifacts["per_group_shots"].values()))
       return list(allocation.values())

   uniform = group_allocation("uniform")
   weighted = group_allocation("weighted")

   # Identical total budget — the comparison is apples-to-apples.
   assert sum(uniform) == sum(weighted) == 1200
   # "weighted" concentrates shots on the dominant terms, so it spreads the
   # budget unevenly while "uniform" stays flat.
   assert (max(weighted) - min(weighted)) > (max(uniform) - min(uniform))

Variational algorithms accept the same option directly as a constructor
keyword (e.g. ``VQE(..., shot_distribution="weighted")``); it is threaded
through to the cost pipeline's measurement stage.  See
:doc:`ground_state_energy_estimation_vqe` for an end-to-end example.

When a group ends up with zero allocated shots its measurement circuit is
skipped and its observables contribute zero to the final estimate.  The
stage emits a :class:`UserWarning` reporting the dropped fraction of the
Hamiltonian's L1 norm so you can quantify the resulting bias.

Adaptive shot allocation only applies to sampling-based execution.
:class:`~divi.pipeline.stages.MeasurementStage` routes single-observable
expectation values to the backend's native expval path whenever
``backend.supports_expval`` is ``True`` — regardless of
``grouping_strategy``.  :class:`~divi.backends.MaestroSimulator` always
reports ``supports_expval=True``, so it takes the analytic path by default.
``grouping_strategy`` controls how observable *terms* are grouped into
measurement circuits; it does **not** by itself force shot-based sampling.

To force genuine shot-based sampling — and unlock ``shot_distribution`` — use
a backend with ``supports_expval=False``:
``QiskitSimulator(force_sampling=True)``, or for
:class:`~divi.backends.QoroService` set
``JobConfig(force_sampling=True)``.  Setting ``shot_distribution`` on an
expval-capable backend is not silent: it emits a :class:`UserWarning` (the
per-group allocation is recorded but cannot change the exact, analytically
computed result), and explicitly pairing it with
``grouping_strategy="_backend_expval"`` raises :class:`ValueError`.


Stage Validation
----------------

The pipeline validates stage ordering at construction time.  Built-in stages
declare their own constraints — for example, :class:`~divi.pipeline.stages.QEMStage`
with QuEPP requires a measurement-handling stage after it.  The pipeline also
validates that at least one stage handles measurement before custom ``validate``
hooks run, so a custom constraint requiring a ``MeasurementStage`` after it is
pre-empted and unreachable.  Pick constraints that the built-in check does **not**
cover, for example ordering relative to another custom stage.

Custom stages can participate in this by overriding the ``validate`` method:

.. code-block:: python

   import pytest
   from divi.pipeline import BundleStage, CircuitPipeline, StageOutput
   from divi.pipeline.abc import ContractViolation, MetaCircuitBatch
   from divi.pipeline.stages import CircuitSpecStage, MeasurementStage

   class PreprocessStage(BundleStage):
       """Pass-through stage that must run before any ReplicaBundleStage.

       This constraint is custom — the built-in pipeline check only validates
       structural rules (one SpecStage first, one measurement stage).
       """

       def __init__(self):
           super().__init__(name="preprocess")

       @property
       def axis_name(self):
           return "preprocess"

       def validate(self, before, after):
           # Ordering constraint: no ReplicaBundleStage may precede this stage.
           if any(type(s).__name__ == "ReplicaBundleStage" for s in before):
               raise ContractViolation(
                   "PreprocessStage must come before any ReplicaBundleStage."
               )

       def expand(self, batch: MetaCircuitBatch, env) -> StageOutput:
           return StageOutput(batch=batch)

       def reduce(self, results, env, token):
           return results

   # Re-using ReplicaBundleStage from Example 4 — it is in scope because
   # code-blocks within one .rst file share a namespace and run in order.
   # Both pipelines below have a SpecStage first and one MeasurementStage,
   # satisfying the built-in structural check; the difference is only ordering.

   # Valid: PreprocessStage before ReplicaBundleStage — no ContractViolation.
   pipeline_ok = CircuitPipeline(stages=[
       CircuitSpecStage(),
       PreprocessStage(),
       ReplicaBundleStage(n=2),
       MeasurementStage(),
   ])

   # Wrong ordering: ReplicaBundleStage before PreprocessStage — constraint fires.
   with pytest.raises(ContractViolation):
       CircuitPipeline(stages=[
           CircuitSpecStage(),
           ReplicaBundleStage(n=2),
           PreprocessStage(),
           MeasurementStage(),
       ])

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
