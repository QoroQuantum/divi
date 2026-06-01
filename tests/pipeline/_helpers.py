# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Shared test helpers for pipeline tests (stages, execute fn, meta circuit factory)."""

import re
from typing import cast

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import SparsePauliOp

from divi.backends import CircuitRunner, ExecutionResult
from divi.circuits import MetaCircuit
from divi.pipeline import CircuitPipeline, PipelineEnv, PipelineTrace
from divi.pipeline._compilation import _compile_batch
from divi.pipeline.abc import (
    BundleStage,
    ChildResults,
    ExpansionResult,
    MetaCircuitBatch,
    SpecStage,
    Stage,
    StageToken,
)
from divi.pipeline.stages import MeasurementStage, ParameterBindingStage


class DummySpecStage(SpecStage[str]):
    """Simple spec stage that emits a single logical circuit."""

    def __init__(self, meta: MetaCircuit | None = None) -> None:
        super().__init__(name=type(self).__name__)
        self._meta = meta or cast(MetaCircuit, object())

    def expand(
        self, items: str, env: PipelineEnv
    ) -> tuple[MetaCircuitBatch, StageToken]:
        return {(("spec", "circ"),): self._meta}, None

    def reduce(
        self, results: ChildResults, env: PipelineEnv, token: StageToken
    ) -> ChildResults:
        return results


class FanoutAndSumStage(BundleStage):
    """Fan-out stage used to validate expansion lineage and reduce fan-in."""

    def __init__(self, branch_prefix: str, n_children: int) -> None:
        super().__init__(name=f"{type(self).__name__}:{branch_prefix}")
        self._branch_prefix = branch_prefix
        self._n_children = n_children

    def expand(
        self, batch: MetaCircuitBatch, env: PipelineEnv
    ) -> tuple[ExpansionResult, StageToken]:
        out: MetaCircuitBatch = {}
        for parent_key, meta in batch.items():
            for idx in range(self._n_children):
                child_key = parent_key + ((self._branch_prefix, idx),)
                out[child_key] = meta
        return ExpansionResult(batch=out), None

    def reduce(
        self, results: ChildResults, env: PipelineEnv, token: StageToken
    ) -> ChildResults:
        rolled_up: ChildResults = {}
        for child_key, value in results.items():
            # Strip this stage's axis (last axis we added in expand)
            parent_key = tuple(
                e
                for e in child_key
                if not (isinstance(e, tuple) and e[0] == self._branch_prefix)
            )
            rolled_up[parent_key] = rolled_up.get(parent_key, 0) + value
        return rolled_up


class StatefulFanoutStage(FanoutAndSumStage):
    """Like FanoutAndSumStage but stateful=True to exercise run_forward_pass cache/partial rerun."""

    @property
    def stateful(self) -> bool:
        return True


def run_binding_pipeline(
    meta: MetaCircuit,
    *,
    backend,
    param_sets,
    input_key: str = "x",
) -> PipelineTrace:
    """Drive ``meta`` through the canonical spec → measure → param-bind forward
    pass and return the trace. Covers the dominant per-test pipeline shape;
    tests that vary the stage order (path-selection) build their own list."""
    pipeline = CircuitPipeline(
        stages=[
            DummySpecStage(meta=meta),
            MeasurementStage(),
            ParameterBindingStage(),
        ]
    )
    env = PipelineEnv(backend=backend, param_sets=param_sets)
    return pipeline.run_forward_pass(input_key, env)


def two_group_meta() -> MetaCircuit:
    """MetaCircuit with 0.9*Z + 0.4*X for MeasurementStage to produce 2 groups."""
    qc = QuantumCircuit(1)
    qc.h(0)
    observable = SparsePauliOp.from_list([("Z", 0.9), ("X", 0.4)])
    return MetaCircuit(
        circuit_bodies=(((), circuit_to_dag(qc)),),
        observable=observable,
    )


def two_group_pipeline_stages(
    meta: MetaCircuit | None = None,
    fanout: tuple[str, int] | None = None,
) -> list[Stage]:
    """Stages: DummySpecStage(two_group_meta) -> MeasurementStage -> optional FanoutAndSum."""
    stages: list[Stage] = [
        DummySpecStage(meta=meta if meta is not None else two_group_meta()),
        MeasurementStage(),
    ]
    if fanout is not None:
        prefix, n = fanout
        stages.append(FanoutAndSumStage(prefix, n))
    return stages


def ones_execute_fn(
    trace: PipelineTrace,
    env: PipelineEnv,
) -> ChildResults:
    """Return 1 for each branch key so reduce stages get correct key structure (BranchKeys)."""
    try:
        _, lineage_by_label = _compile_batch(trace.final_batch)
        return {branch_key: 1 for branch_key in lineage_by_label.values()}
    except (ValueError, AttributeError):
        return {key: 1 for key in trace.final_batch}


def build_pipeline_with_shots(
    meta: MetaCircuit,
    distribution: str | None,
    backend: CircuitRunner,
    **stage_kw,
) -> tuple[CircuitPipeline, PipelineEnv]:
    """Build a pipeline with optional shot distribution."""
    env = PipelineEnv(backend=backend)
    pipeline = CircuitPipeline(
        stages=[
            DummySpecStage(meta=meta),
            MeasurementStage(shot_distribution=distribution, **stage_kw),
        ],
    )
    return pipeline, env


class ExpvalBackendSpy(CircuitRunner):
    """Backend that records kwargs and returns per-Pauli expectation values."""

    def __init__(self, shots=100):
        super().__init__(shots=shots)
        self.last_ham_ops: str | None = None

    @property
    def is_async(self):
        return False

    @property
    def supports_expval(self):
        return True

    def submit_circuits(self, circuits, **kwargs):
        self.last_ham_ops = kwargs.get("ham_ops")
        results = []
        if self.last_ham_ops is not None:
            terms = self.last_ham_ops.split(";")
            for label in circuits:
                pauli_dict = {term: 0.1 * (i + 1) for i, term in enumerate(terms)}
                results.append({"label": label, "results": pauli_dict})
        return ExecutionResult(results=results)


class ShotsBackendSpy(CircuitRunner):
    """Shots-based backend (supports_expval=False) for probs tests."""

    @property
    def is_async(self):
        return False

    @property
    def supports_expval(self):
        return False

    def submit_circuits(self, circuits, **kwargs):
        results = []
        for label, qasm in circuits.items():
            match = re.search(r"qreg q\[(\d+)\]", qasm)
            n_qubits = int(match.group(1))
            results.append(
                {
                    "label": label,
                    "results": {"0" * n_qubits: 80, "1" * n_qubits: 20},
                }
            )
        return ExecutionResult(results=results)


class RecordingBackend(CircuitRunner):
    """Captures kwargs passed to ``submit_circuits`` by ``_default_execute_fn``."""

    def __init__(self, shots: int = 1000) -> None:
        super().__init__(shots=shots)
        self.last_circuits: dict[str, str] | None = None
        self.last_kwargs: dict = {}

    @property
    def is_async(self) -> bool:
        return False

    @property
    def supports_expval(self) -> bool:
        return False

    def submit_circuits(self, circuits, **kwargs):
        self.last_circuits = dict(circuits)
        self.last_kwargs = dict(kwargs)
        results = [
            {"label": label, "results": {"0": kwargs.get("shots_for_label", 100)}}
            for label in circuits
        ]
        return ExecutionResult(results=results)
