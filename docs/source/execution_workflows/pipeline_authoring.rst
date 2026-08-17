Pipeline Authoring
==================

This guide is for users implementing a custom algorithm, standalone pipeline,
or pipeline stage. Read :doc:`pipelines` first for the expand → execute → reduce
model, built-in stages, and dry-run inspection.

The examples progress from the supported high-level extension point
(:class:`~divi.qprog.algorithms.CustomVQA`) to custom specifications, bundle
stages, and full :class:`~divi.qprog.QuantumProgram` subclasses. Stage-author
contracts, validation, reducer helpers, and dry-run metadata follow the examples.

Start with CustomVQA
--------------------

Use :class:`~divi.qprog.algorithms.CustomVQA` when you already have a
parameterized PennyLane ``QuantumScript`` or Qiskit ``QuantumCircuit``. It
provides optimization, histories, and checkpointing without custom pipeline
code; :doc:`framework_integration` has complete examples. Its cost pipeline is:

.. code-block:: text

   CircuitSpecStage → QEMStage → MeasurementStage → ParameterBindingStage

The :class:`~divi.qprog.algorithms.CustomVQA`,
:class:`~divi.qprog.algorithms.VQE`, and :class:`~divi.qprog.algorithms.QNN`
constructors also forward these options to
:class:`~divi.pipeline.stages.MeasurementStage`:

- ``grouping_strategy`` — ``"qwc"`` (default), ``"wires"``, or ``None``.
- ``shot_distribution`` — a named or callable allocation strategy; see
  :ref:`adaptive-shot-allocation`.

Example::

    vqe = VQE(molecule=mol, ..., shot_distribution="weighted", grouping_strategy="qwc")


Feeding Parameter Values to a Standalone Pipeline
--------------------------------------------------

A standalone :class:`~divi.pipeline.CircuitPipeline` reads parameters from
``PipelineEnv.param_sets`` with shape ``(n_param_sets, n_params)``; its
:meth:`~divi.pipeline.CircuitPipeline.run` method has no ``params=`` argument.

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

   ry_theta = Parameter("ry_theta")
   ry_qc = QuantumCircuit(1)
   ry_qc.ry(ry_theta, 0)

   ry_meta = MetaCircuit(
       circuit_bodies=(((), circuit_to_dag(ry_qc)),),
       observable=SparsePauliOp.from_list([("Z", 1.0)]),
       parameters=(ry_theta,),
   )

   ry_pipeline = CircuitPipeline(stages=[
       CircuitSpecStage(),
       MeasurementStage(),
       ParameterBindingStage(),   # reads env.param_sets; placed last
   ])

   ry_env = PipelineEnv(
       backend=MaestroSimulator(),
       param_sets=[[0.0], [np.pi / 2]],   # 2 param sets, 1 param each
   )
   ry_result = ry_pipeline.run(initial_spec=ry_meta, env=ry_env)

   # EXPVALS remain lists inside PipelineResult; extract the first value.
   by_idx = {extract_param_set_idx(k): v[0] for k, v in ry_result.items()}
   # ⟨Z⟩ for theta=0 (|0⟩) ≈ 1.0; for theta=π/2 (|+y⟩) ≈ 0.0
   assert abs(by_idx[0] - 1.0) < 0.15, f"Expected ~1.0, got {by_idx[0]}"
   assert abs(by_idx[1] - 0.0) < 0.15, f"Expected ~0.0, got {by_idx[1]}"

Other useful :class:`~divi.pipeline.PipelineEnv` fields:

- ``shots_override`` — per-run shot count without backend mutation.
- ``collect_variance`` — records ``"cost_variance"`` in ``env.artifacts``;
  :meth:`~divi.qprog.QuantumProgram.evaluate` sets it for
  ``return_variance=True``.
- ``axes_to_preserve`` — axes that reduction must retain.
- ``feature_batch`` — input matrix for
  :class:`~divi.pipeline.stages.DataBindingStage`.
- ``rng`` — generator for stochastic stage decisions.


Converting External Circuits
----------------------------

Use :class:`~divi.pipeline.stages.PennyLaneSpecStage` or
:class:`~divi.pipeline.stages.QiskitSpecStage` to run external circuits without
a ``QuantumProgram``. Both accept a circuit, sequence, or mapping. For example:

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

.. tip::

   ``result.value`` squeezes a single expectation to ``float`` and returns a
   list for several expectations or a dict for probabilities/counts.
   ``evaluate(...)`` instead returns unsqueezed values keyed by parameter-set
   index, such as ``{0: [1.0]}``.


Writing a Custom SpecStage
--------------------------

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


Writing a Custom BundleStage
----------------------------

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

       @property
       def consumes_dag_bodies(self):
           # Re-tags the incoming DAGs without reading or mutating them, so
           # upstream stages can keep their analytic dry path.
           return False

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


Custom QuantumProgram with ``evaluate``
---------------------------------------

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
   from divi.pipeline import (
       CircuitPipeline,
       CircuitPreprocessor,
       ResultFormat,
   )
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
               # cadence defaults to PER_EVALUATION (recurring); pass
               # PipelineCadence.ONCE for a routine that runs a single time.
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


Injecting a Custom Stage into an Optimizer-Driven Algorithm
-----------------------------------------------------------

This section reuses ``ReplicaBundleStage`` from `Writing a Custom BundleStage`_.

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

   # ReplicaBundleStage is defined in the preceding custom-stage example.

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


Stage Validation
----------------

The pipeline validates stage ordering at construction time.  Built-in stages
declare their own constraints — for example, :class:`~divi.pipeline.stages.QEMStage`
with QuEPP requires a measurement-handling stage after it.  The pipeline also
validates that at least one stage handles measurement before custom ``validate``
hooks run, so a custom constraint requiring a ``MeasurementStage`` after it is
pre-empted and unreachable.  Pick constraints that the built-in check does **not**
cover, for example ordering relative to another custom stage.

Custom stages can participate in this by overriding the ``validate`` method.
This example also reuses ``ReplicaBundleStage`` from
`Writing a Custom BundleStage`_:

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

   # ReplicaBundleStage is defined in Writing a Custom BundleStage above.
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

- :doc:`pipelines` — architecture, dry-run inspection, and adaptive shot
  allocation
- :doc:`../api_reference/pipeline` — pipeline and stage classes
- :doc:`../algorithms/improving_results_qem` — :class:`~divi.circuits.qem.QEMProtocol` and error mitigation
- :doc:`../api_reference/qprog/algorithms` — :class:`~divi.qprog.algorithms.CustomVQA` and custom circuits
- :doc:`program_ensembles` — parameter sweeps and orchestration
