Pipelines
=========

Every quantum program in Divi executes circuits through a **circuit pipeline**.
The pipeline models the journey from a high-level specification (e.g. a
Hamiltonian or a :class:`~divi.circuits.MetaCircuit`) to final, reduced results as a sequence of
composable **stages**.

This guide explains how the pipeline works, lists the built-in stages shipped
with Divi, and shows how to inspect a program before execution. Advanced
extension examples are collected in :doc:`pipeline_authoring`.

.. note::
   If you are using built-in algorithms like :class:`~divi.qprog.algorithms.VQE`, :class:`~divi.qprog.algorithms.QAOA`, or :class:`~divi.qprog.algorithms.TimeEvolution` you
   **don't need to interact with the pipeline directly** — each algorithm
   constructs its own pipeline internally. Most of this guide is for users who
   want to understand or inspect those internals. For extensions, continue to
   :doc:`pipeline_authoring`.

   One section is for everyone: :ref:`dry-run` previews what any program would
   submit before it executes a single circuit.  You do not need to understand
   anything below to use it.


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
  Custom spec stages must mirror that behaviour if you want ``result[()]`` to work.

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
   * - :class:`~divi.pipeline.stages.DataBindingStage`
     - Bundle
     - Fans each circuit out over a batch of input samples, binding the data
       parameters per sample. Used by :class:`~divi.qprog.algorithms.QNN` and any
       data-bound :class:`~divi.qprog.algorithms.CustomVQA`; adds the
       ``data_sample`` axis.
   * - :class:`~divi.pipeline.stages.QEMStage`
     - Bundle
     - Applies a :class:`~divi.circuits.qem.QEMProtocol` (e.g. ZNE) in the
       expand pass and reduces the scaled results in the reduce pass.
       See :doc:`../algorithms/improving_results_qem` for details.
   * - :class:`~divi.pipeline.stages.PauliTwirlStage`
     - Bundle
     - Generates randomised Pauli-twirl variants of each circuit.
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

A dry run is a **sanity check on the pipeline you just constructed**, performed
before a single circuit executes.  It walks each pipeline's forward pass and
reports what that pipeline *is*: which stages it contains, in what order, what
each one fans out or reduces, and the shape of the circuits that come out the
end.  Use it to confirm the pipeline you built is the pipeline you meant —
that the mitigation you configured actually applies, that observable grouping
is doing what you expect, that an optimizer is driving the auxiliary pipelines
you think it is.

It is deliberately a *back-of-the-envelope* instrument, not a cost model: the
counts it reports describe one pass through the pipeline, and it does not
attempt to predict a full optimisation run (see
:ref:`what-a-dry-run-does-not-tell-you`).  Its strength is **comparison** —
run it twice with different settings and read off what changed.

Call :meth:`~divi.qprog.QuantumProgram.dry_run` on any quantum program, then
pass the resulting dict to :func:`~divi.pipeline.format_dry_run` for the rich
tree output:

.. code-block:: python

   from pyscf import gto

   from divi.backends import QiskitSimulator
   from divi.circuits.quepp import QuEPP
   from divi.pipeline import format_dry_run
   from divi.qprog import VQE
   from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer

   h2_molecule = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", unit="Bohr")

   vqe = VQE(
       molecule=h2_molecule,
       qem_protocol=QuEPP(truncation_order=1, n_twirls=10),
       backend=QiskitSimulator(qiskit_backend="auto"),
       optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
   )

   # Runs the forward pass without executing circuits, then pretty-print.
   format_dry_run(vqe.dry_run())

``format_dry_run`` prints a tree per pipeline showing the per-stage factor
(fan-out or reduction) and metadata.  Here is the ``cost`` tree for the program
above, with some metadata rows trimmed:

.. code-block:: text

   cost
   ├── CircuitSpecStage [circuit] → 14
   │   ├── n_qubits: 4
   │   ├── n_gates: 42
   │   ├── n_2q_gates: 18
   │   └── depth: 24
   ├── PreprocessStage [PreprocessStage] → 1
   ├── QEMStage [qem_quepp] → ×10
   │   ├── protocol: quepp
   │   ├── n_paths: 9
   │   └── n_clifford_sims: 9
   ├── PauliTwirlStage [twirl] → ×10
   │   └── n_twirls: 10
   ├── MeasurementStage [obs_group] → ÷2.8
   │   ├── strategy: qwc
   │   ├── n_groups: 5
   │   ├── n_pauli_terms: 14
   │   └── shots_per_circuit: 5000
   ├── ParameterBindingStage [param_set] → 1
   │   ├── n_param_sets: 1
   │   └── n_bound_params: 3
   ├── Total (per evaluation): 14 × 10 × 10 ÷ 2.8 = 500 circuits · 2,500,000 shots
   └── Summary: avg depth 24, width 4, 9000 2q-gates total

Reading the tree: the spec stage's ``14`` is the naive baseline (one circuit per
Pauli term); ``×K`` stages fan out (QEM paths, twirls), ``÷K`` stages reduce
(observable grouping).  The ``Total`` line is what one pass through this
pipeline submits — **one evaluation** — and the ``Summary`` line describes the
*shape* of those circuits (depth, width, entangling-gate count).

The name in brackets after each stage is the *axis* it fans out over — the
dimension its extra circuits vary along (``[obs_group]`` one per commuting group,
``[twirl]`` one per twirl, ``[param_set]`` one per parameter vector,
``[data_sample]`` one per feature row).  A stage that declares no axis of its own
repeats its class name there, which is why ``PreprocessStage [PreprocessStage]``
looks redundant: it restructures circuits without adding a dimension.

For most optimizers one evaluation is one parameter vector (θ).  A *population*
optimizer binds its whole working set in a single pass, so there the count covers
all of them and the line says so explicitly:
``Total (per evaluation, all 10 parameter sets)``.  Read the label, not the
assumption.

.. important::

   Those shape figures are **logical**: they are measured before transpilation,
   so they model no coupling map and include no routing SWAPs.  On a device whose
   qubits are not all-to-all connected, the submitted circuit is deeper and
   carries more entangling gates than reported — often substantially, for an
   ansatz with long-range interactions.  Treat them as a lower bound on circuit
   shape, and transpile a representative circuit against your target device
   before concluding that it fits.

**The report stops at one evaluation.**  How many evaluations an optimizer spends
per step, and how many steps a run takes, are not modelled here: a gradient-free
method decides as it goes, a line search spends what the landscape demands, and
SPSA-style methods add fixed extras (a resampled gradient, a blocking baseline)
that vary with their own settings.  Reaching a run total means multiplying by
figures only you have — consult the optimizer's own documentation for its call
pattern, or measure one short run and scale.

What the report *does* say about run structure is which routines recur: a
``PER_EVALUATION`` pipeline runs every time the optimizer evaluates, a ``ONCE``
pipeline runs one time (the final readout).  The ensemble roll-up keeps the two
buckets apart for the same reason.

Data-bound programs add a ``data_sample`` axis. Its size multiplies observable
groups, mitigation branches, and population parameter sets. The report lists
data and trainable parameter counts separately.

.. note::

   ``predict()`` builds an unregistered inference pipeline, so it is not
   previewed. Its circuit count is samples × measurement groups.

.. note::

   Preview with the actual backend. ``strategy: _backend_expval`` means the
   backend evaluates the observable directly; ``strategy: qwc`` means sampling
   circuits were grouped. ``QiskitSimulator(force_sampling=True)`` selects the
   latter path, while :class:`~divi.backends.MaestroSimulator` uses expval mode.

``dry_run()`` is print-free — it returns a ``dict[str, DryRunReport]`` keyed by
pipeline name for programmatic use, so you can assert on pipeline structure in a
test or a notebook.

It reports the routines a program *registers*, which is what
``_preprocessors()`` returns — not every pipeline the program happens to run.
Calling :meth:`~divi.qprog.QuantumProgram.evaluate` with an ad-hoc preprocessor
does not register anything, so a custom program that only does that returns
``{}``.  To make a routine previewable, add it to ``_preprocessors()``:

.. skip: next

.. code-block:: python

   def _preprocessors(self):
       return (*super()._preprocessors(), self._my_routine())

.. code-block:: python

   reports = vqe.dry_run()   # `vqe` from the example above
   cost = reports["cost"]

   # What is in this pipeline, and what does each stage do to the batch?
   # `factor` is the ratio of circuits out to circuits in, so a reduction reads as
   # a fraction: the tree's ÷14 is factor == 1 / 14 here.
   print([(s.name, s.factor) for s in cost.stages])
   print(cost.total_circuits)        # 500 — one evaluation (this optimizer: one θ)
   print(cost.cadence)               # PipelineCadence.PER_EVALUATION

When you do want a figure spanning several pipelines, keep the cadences apart:
recurring pipelines are not comparable with one-time ones, and only you know how
many evaluations your optimizer will spend.  A ``ONCE`` routine runs *at most*
once — ``run(perform_final_computation=False)`` skips solution sampling
altogether, so drop it from the total on that path.

Metric-Optimizer Pipelines
~~~~~~~~~~~~~~~~~~~~~~~~~~

A dry run includes optimizer-owned pipelines. QNG and QN-SPSA add ``metric``;
:class:`~divi.qprog.optimizers.FubiniStudyMetricEstimator` adds one
``metric[block N]`` per commuting block. Cadence labels keep these recurring
pipelines separate from the one-time ``sample`` pipeline:

.. code-block:: text

   cost
   ├── …
   ├── Total (per evaluation): 14 ÷ 2.8 = 5 circuits · 25,000 shots
   └── Summary: avg depth 24, width 4, 90 2q-gates total

   sample
   ├── …
   ├── Total (once): 1 circuit · 5,000 shots
   └── Summary: avg depth 24, width 4, 18 2q-gates total

   metric
   ├── …
   ├── Total (per evaluation): 1 circuit · 5,000 shots
   └── Summary: avg depth 48, width 4, 36 2q-gates total

This example exposes QN-SPSA's doubled-depth overlap circuit. The optimizer
guide documents its evaluation budget; a dry run still covers one evaluation.

.. _what-a-dry-run-does-not-tell-you:

What a Dry Run Does Not Tell You
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A dry run describes **pipelines**, not complete runs:

- It does not multiply by ``max_iterations`` or model line searches, gradients,
  baselines, or other optimizer call patterns. Measure a short run when you need
  a total estimate.
- Shot totals reflect configured budgets, not provider billing. Analytic
  backends may report configured shots while consuming none.
- Default QEM/twirling previews preserve circuit counts but may report shape
  before rewriting. Use ``force_circuit_generation=True`` for post-rewrite
  depth and gate counts.
- It checks structure, not numerical quality.

When Dry Run Falls Back
~~~~~~~~~~~~~~~~~~~~~~~

If a consuming stage lacks ``dry_expand``, Divi uses its real ``expand`` to
avoid sharing mutable DAGs and emits
:class:`~divi.pipeline.DiviPerformanceWarning`. Counts remain correct. Stage
authors should implement ``dry_expand`` or declare
``consumes_dag_bodies=False`` when the stage does not mutate DAGs.

Dry Running an Ensemble
~~~~~~~~~~~~~~~~~~~~~~~

A :class:`~divi.qprog.ensemble.ProgramEnsemble` previews all sub-programs after
``create_programs()``. :func:`~divi.pipeline.format_dry_run` selects a layout
by ensemble size, or accepts ``style=``:

- ``"verbose"`` — a full tree per program.
- ``"compact"`` — one line per pipeline and a grand total.
- ``"grouped"`` — one tree per equivalent program structure. Different
  objectives remain separate; other differences appear as ``mixed (…)``.

Failures name the sub-program instead of returning a partial report. Cadence,
width, and depth remain separate in ensemble totals. See
:ref:`ensemble-dry-run` for a runnable example and grouping details.

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
       # identity transform, EXPVALS, recurring per evaluation (the defaults)
       return CircuitPreprocessor("cost", cache_key="cost")

   # A caller (e.g. an optimizer) measures the prepared state through one verb:
   #   losses = program.evaluate(params, program.cost_preprocessor())

The single pipeline is assembled by ``_assemble_pipeline`` — spec → the
protocol's ``PreprocessStage`` (its post-spec transform) → [error mitigation
(+ Pauli twirls) when the QEM protocol applies to the result format] → terminal
measurement.  :class:`~divi.qprog.VariationalQuantumAlgorithm` appends a
``ParameterBindingStage`` last; the base :class:`~divi.qprog.QuantumProgram`
does **not** — a direct ``QuantumProgram`` subclass with a parameterised seed
must add :class:`~divi.pipeline.stages.ParameterBindingStage` itself (or
subclass ``VariationalQuantumAlgorithm``), or execution raises
:class:`~divi.pipeline.abc.ContractViolation`.

The **cost protocol** evaluates expectation values (or a classical objective)
during optimisation; the **sample protocol** samples the probability
distribution afterwards to extract the solution.  Whether error mitigation rides
a protocol is decided by the QEM protocol itself
(:meth:`~divi.circuits.qem.QEMProtocol.applies_to`), so extrapolation-style
mitigation rides expectation-value protocols but not the probability-sampling
one. Natural-gradient optimizers measure their metric by passing a dynamic
:class:`~divi.pipeline.CircuitPreprocessor` to
:meth:`~divi.qprog.QuantumProgram.evaluate` rather than registering it as one of
the program's own routines. They do, however, declare those routines through
:meth:`~divi.qprog.optimizers.Optimizer.preprocessors`, which the program folds
into its own set — so a dry run accounts for their cost (see the metric-optimizer
example under :ref:`dry-run`).

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


Extending Pipelines
-------------------

To build custom algorithms, standalone pipelines, or stages, continue to
:doc:`pipeline_authoring`. That guide owns the extension contracts and complete
authoring examples; this page stays focused on understanding and inspecting
execution.


.. _adaptive-shot-allocation:

Adaptive Shot Allocation
------------------------

By default, each measurement group receives the backend's full shot count, for
``G × shots`` total. ``shot_distribution`` instead divides one ``shots`` budget
among all groups. Compare strategies at equal total budget.

.. code-block:: python

   from divi.pipeline.stages import MeasurementStage

   # Concentrate shots on dominant Hamiltonian terms
   MeasurementStage(grouping_strategy="qwc", shot_distribution="weighted")

Available strategies (see :data:`~divi.pipeline.ShotDistStrategy`):

- ``"uniform"`` — equal split across groups.
- ``"weighted"`` — proportional to coefficient L1 norm, with exact total.
- ``"weighted_random"`` — multinomial allocation, reproducible with a seeded
  ``env.rng``.
- A callable ``(group_l1_norms, total_shots) -> per_group_shots`` for
  fully custom allocation.

A :meth:`~divi.qprog.QuantumProgram.dry_run` exposes allocations under
``env_artifacts["per_group_shots"]``. Variational algorithms accept the same
option directly, for example ``VQE(..., shot_distribution="weighted")``.

Zero-shot groups are skipped and contribute zero. A :class:`UserWarning`
reports their fraction of the Hamiltonian L1 norm.

Allocation requires ``supports_expval=False``; ``grouping_strategy`` alone does
not force sampling. Use ``QiskitSimulator(force_sampling=True)`` or
``JobConfig(force_sampling=True)`` with QoroService. Expval-capable backends
ignore ``shot_distribution`` and warn because they create no measurement groups.


What's Next
-----------

- :doc:`../api_reference/pipeline` — pipeline and stage classes
- :doc:`pipeline_authoring` — custom algorithms, standalone pipelines, and
  stage-author contracts
- :doc:`../algorithms/improving_results_qem` — :class:`~divi.circuits.qem.QEMProtocol` and error mitigation
- :doc:`../api_reference/qprog/algorithms` — :class:`~divi.qprog.algorithms.CustomVQA` and custom circuits
- :doc:`program_ensembles` — parameter sweeps and orchestration
