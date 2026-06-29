# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Circuit preprocessors: post-spec MetaCircuit transforms + the readout they target.

A :class:`CircuitPreprocessor` is a named transform of the seed circuit emitted
by the spec stage — ``MetaCircuit -> MetaCircuit`` — paired with the readout that
transform prepares for (a :class:`~divi.pipeline.ResultFormat` and an optional
custom terminal stage). :meth:`~divi.qprog.QuantumProgram.evaluate` applies the
preprocessor (via :class:`~divi.pipeline.stages.PreprocessStage`) and runs the
program's one pipeline, so callers (optimizers, metric estimators) select a
routine by passing a preprocessor instead of assembling pipelines themselves.

These default preprocessors are factory *functions* (``cost_preprocessor``,
``sample_preprocessor``); each program surfaces one through a method that
returns it — the public, overridable ``VariationalQuantumAlgorithm.cost_preprocessor``
(PCE overrides it with its counts-based variant) and the internal
``_sample_preprocessor``. The metric/overlap preprocessors live in
``divi.qprog._metrics`` as factory functions too, parameterized by
per-evaluation state (the overlap closure, the Fubini-Study block id) rather
than overridden.
"""

from collections.abc import Callable, Hashable
from dataclasses import dataclass, replace

from divi.circuits import MetaCircuit
from divi.pipeline.abc import ResultFormat, Stage


def _identity(meta: MetaCircuit) -> MetaCircuit:
    return meta


def _clear_observable(meta: MetaCircuit) -> MetaCircuit:
    """Drop the observable and measure every wire — turns an expval seed into a
    computational-basis sampling circuit."""
    return replace(
        meta,
        observable=None,
        measured_wires=tuple(range(meta.n_qubits)),
        measurement_qasms=(),
        measurement_groups=(),
    )


@dataclass(frozen=True)
class CircuitPreprocessor:
    """A named seed transform and the readout it targets.

    Attributes:
        name: Identifier for the routine (``"cost"``, ``"sample"``, ...).
        preprocess: Transform applied to each post-spec ``MetaCircuit`` before
            mitigation and the terminal measurement. Defaults to identity.
        result_format: Format the raw backend results convert into; also drives
            error-mitigation applicability when the pipeline is assembled.
            Defaults to expectation values.
        terminal_stage: Optional custom ``handles_measurement`` terminal. ``None``
            means the program supplies its default
            :class:`~divi.pipeline.stages.MeasurementStage` (configured with the
            program's grouping / shot strategy); PCE supplies its own
            :class:`~divi.pipeline.stages.PCECostStage`.
        consumes_dag_bodies: Whether ``preprocess`` reads or replaces circuit
            DAG bodies. Metadata-only transforms leave this ``False`` so dry
            runs can keep analytic shortcuts.
        cache_key: Identity under which
            :meth:`~divi.qprog.QuantumProgram._build_preprocessor_pipeline`
            memoizes the assembled pipeline (so its forward-pass cache survives
            across optimizer iterations). ``None`` (the default) means *do not
            cache* — the pipeline is rebuilt per call and discarded. Every
            built-in routine (cost/sample/evolution/overlap and the metric
            estimators) passes a constant key, because each one's ``preprocess``
            is a pure transform; per-iteration freshness where it is needed (the
            QDrift stochastic resampling) comes from the spec stage's
            ``cache_key_extras`` invalidating the forward-pass cache, not from
            leaving this ``None``.
    """

    name: str
    preprocess: Callable[[MetaCircuit], MetaCircuit] = _identity
    result_format: ResultFormat = ResultFormat.EXPVALS
    terminal_stage: Stage | None = None
    consumes_dag_bodies: bool = False
    cache_key: Hashable | None = None


def cost_preprocessor() -> CircuitPreprocessor:
    """Measure the seed's cost observable as expectation values (identity transform)."""
    return CircuitPreprocessor("cost", cache_key="cost")


def sample_preprocessor() -> CircuitPreprocessor:
    """Sample the prepared state in the computational basis (clears the observable)."""
    return CircuitPreprocessor(
        "sample",
        preprocess=_clear_observable,
        result_format=ResultFormat.PROBS,
        cache_key="sample",
    )
