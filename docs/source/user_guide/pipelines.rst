Pipelines
=========

Every quantum program in Divi executes circuits through a **circuit pipeline**.
The pipeline models the journey from a high-level specification (e.g. a
Hamiltonian or a ``MetaCircuit``) to final, reduced results as a sequence of
composable **stages**.

This guide explains how the pipeline works, lists the built-in stages shipped
with Divi, and shows two practical examples of extending Divi with custom
algorithms.

.. note::
   If you are using built-in algorithms like VQE, QAOA, or TimeEvolution you
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
   configured backend (``CircuitRunner``).  This step is handled automatically.

3. **Reduce** (backward pass) — Stages are visited in *reverse* order and each
   one collapses or aggregates the raw results using a token it saved during the
   expand pass.  The pipeline returns the fully reduced result to the caller.

.. code-block:: text

   ── Expand (forward) ──────────────────────────────────────────────────►

   ┌────────────┐          ┌─────────────────┐          ┌─────────────┐
   │ SpecStage  │ ───────► │  BundleStage #1 │ ───────► │ BundleStage │ ─► …
   └────────────┘          └─────────────────┘          └─────────────┘

                                                                 │
                                                                 ▼
                                                           ╔═══════════╗
                                                           ║  Execute  ║
                                                           ╚═══════════╝
                                                                 │
                                                                 ▼

   ┌────────────┐          ┌─────────────────┐          ┌─────────────┐
   │   final    │ ◄─────── │  intermediate   │ ◄─────── │    raw      │
   │   result   │          │    result       │          │   results   │
   └────────────┘          └─────────────────┘          └─────────────┘

   ◄──────────────────────────────────────────────── Reduce (backward) ──


Built-in Stages
---------------

Divi ships with six stages that cover the most common quantum workflows:

.. list-table::
   :header-rows: 1
   :widths: 25 10 65

   * - Stage
     - Type
     - Description
   * - :class:`~divi.pipeline.stages.CircuitSpecStage`
     - Spec
     - Passes a single ``MetaCircuit`` through as a one-element batch.
       Used by VQE, CustomVQA, and other algorithms that receive a pre-built circuit.
   * - :class:`~divi.pipeline.stages.TrotterSpecStage`
     - Spec
     - Generates Trotterised circuits from a Hamiltonian for time-evolution and
       QAOA workflows.
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
       See :doc:`error_mitigation` for details.
   * - :class:`~divi.pipeline.stages.PCECostStage`
     - Bundle
     - Computes the Pauli-coefficient expectation-value cost for PCE-based
       algorithms.


How Existing Algorithms Build Pipelines
---------------------------------------

Every algorithm constructs its pipelines in a ``_build_pipelines`` method.  For
example, :class:`~divi.qprog.VQE` builds two pipelines:

.. code-block:: python

   # Simplified from variational_quantum_algorithm.py
   def _build_cost_pipeline(self, spec_stage):
       return CircuitPipeline(stages=[
           spec_stage,              # SpecStage  →  MetaCircuit batch
           MeasurementStage(...),   # Split observables into groups
           ParameterBindingStage(), # Bind symbolic params → numeric
           QEMStage(protocol=...),  # Apply error mitigation variants
       ])

   def _build_measurement_pipeline(self):
       return CircuitPipeline(stages=[
           CircuitSpecStage(),       # Single-circuit spec
           MeasurementStage(),       # Probability measurement
           ParameterBindingStage(),  # Bind best params
       ])

The **cost pipeline** evaluates expectation values during optimisation (with
optional error mitigation), while the **measurement pipeline** samples the
probability distribution after optimisation to extract the solution.


Example 1: Custom Algorithm with CustomVQA
------------------------------------------

The simplest way to run a custom parameterised circuit through the pipeline is
:class:`~divi.qprog.CustomVQA`.  It wraps a **PennyLane QuantumScript** (or a
Qiskit ``QuantumCircuit``) and optimises its parameters end-to-end, reusing all
the VQA infrastructure.

The following example finds the ground-state energy of a two-qubit transverse-
field Ising model:

.. math::

   H = -Z_0 Z_1 + 0.5\,X_0 + 0.5\,X_1

.. code-block:: python

   import pennylane as qml
   from divi.qprog import CustomVQA
   from divi.qprog.optimizers import ScipyOptimizer, ScipyMethod
   from divi.backends import ParallelSimulator

   # 1. Define the Hamiltonian (observable to minimise)
   H = -1.0 * qml.Z(0) @ qml.Z(1) + 0.5 * qml.X(0) + 0.5 * qml.X(1)

   # 2. Build a parameterised ansatz as a QuantumScript
   ops = [
       qml.RY(0.0, wires=0),
       qml.RY(0.0, wires=1),
       qml.CNOT(wires=[0, 1]),
       qml.RY(0.0, wires=0),
       qml.RY(0.0, wires=1),
   ]
   measurements = [qml.expval(H)]
   qscript = qml.tape.QuantumScript(ops=ops, measurements=measurements)

   # Mark only the gate parameters as trainable (freeze Hamiltonian coefficients)
   qscript.trainable_params = [0, 1, 2, 3]

   # 3. Create the CustomVQA program — it builds a pipeline internally
   program = CustomVQA(
       qscript,
       param_shape=(4,),
       max_iterations=30,
       backend=ParallelSimulator(),
       optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
       seed=42,
   )

   # 4. Run — the pipeline handles circuit compilation, submission, and reduction
   program.run()

   print(f"Ground-state energy: {program.best_loss:.4f}")
   print(f"Optimal parameters: {program.best_params}")

Under the hood, ``CustomVQA`` builds a cost pipeline identical to VQE's:

.. code-block:: text

   CircuitSpecStage → MeasurementStage → ParameterBindingStage → QEMStage

You receive all VQA features (loss history, best parameters, checkpointing)
without writing any pipeline or stage code.


Example 2: Writing a Custom SpecStage
--------------------------------------

For full control you can write a custom :class:`~divi.pipeline.SpecStage` and
construct a :class:`~divi.pipeline.CircuitPipeline` directly.  This is useful
when the built-in spec stages don't cover your circuit-generation logic.

A ``SpecStage`` must implement two methods:

- ``expand(spec, env)`` — Convert an input specification into a keyed batch of
  ``MetaCircuit`` objects and return a token for later use.
- ``reduce(results, env, token)`` — Aggregate the per-key results back into a
  single output using the stored token.

The following example implements a spec stage that creates a simple
Bell-state circuit and measures its probabilities:

.. code-block:: python

   import pennylane as qml
   from divi.circuits import MetaCircuit
   from divi.pipeline import CircuitPipeline, PipelineEnv, SpecStage
   from divi.pipeline.abc import (
       ChildResults,
       MetaCircuitBatch,
       StageToken,
   )
   from divi.pipeline.stages import MeasurementStage
   from divi.backends import ParallelSimulator

   class BellSpecStage(SpecStage):
       """Spec stage that produces a Bell-state circuit."""

       @property
       def axis_name(self):
           return None          # No fan-out axis

       @property
       def stateful(self):
           return False         # Deterministic — safe to cache

       def expand(self, spec, env):
           qscript = qml.tape.QuantumScript(
               ops=[qml.Hadamard(0), qml.CNOT(wires=[0, 1])],
               measurements=[qml.probs()],
           )
           meta = MetaCircuit(source_circuit=qscript, symbols=[])

           # Return a single-element batch keyed by "bell"
           batch: MetaCircuitBatch = {"bell": meta}
           return batch, None   # No reduce token needed

       def reduce(self, results, env, token):
           return results       # Pass results through unchanged


   # Build a minimal pipeline
   pipeline = CircuitPipeline(stages=[
       BellSpecStage(),
       MeasurementStage(),   # Declares probability-mode results
   ])

   # Run the pipeline
   backend = ParallelSimulator()
   env = PipelineEnv(backend=backend)
   result = pipeline.run(initial_spec=None, env=env)

   print(result)
   # Expected: {"bell": {"00": ~0.5, "11": ~0.5}}

This pattern composes naturally — you can insert any ``BundleStage`` between the
spec stage and the measurement stage to add parameter binding, error mitigation,
or any custom transformation.


What's Next
-----------

- 📕 **API Reference**: Full class documentation in :doc:`../api_reference/pipeline`
- 🎛️ **Error Mitigation**: Add a :class:`~divi.circuits.qem.QEMProtocol` to your pipeline in :doc:`error_mitigation`
- ⚡ **Custom Circuits**: Wrap any ``QuantumScript`` with :class:`~divi.qprog.CustomVQA` from :doc:`../api_reference/qprog`
- 📊 **Program Batches**: Scale pipelines across parameter sweeps in :doc:`program_batches`
