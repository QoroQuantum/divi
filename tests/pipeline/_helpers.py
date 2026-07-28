# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Shared test helpers for pipeline tests (stages, execute fn, meta circuit factory)."""

import re
from typing import cast

import numpy as np
import pennylane as qp
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import RYGate, RZGate
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import SparsePauliOp

from divi.backends import CircuitRunner, ExecutionResult
from divi.circuits import MetaCircuit
from divi.pipeline import (
    CircuitPipeline,
    PipelineCadence,
    PipelineEnv,
    PipelineTrace,
    StageOutput,
    dry_run_pipeline,
)
from divi.pipeline._compilation import _compile_batch
from divi.pipeline.abc import (
    BundleStage,
    ChildResults,
    MetaCircuitBatch,
    SpecStage,
    Stage,
    StageToken,
)
from divi.pipeline.stages import MeasurementStage, ParameterBindingStage
from divi.qprog import VQE, HartreeFockAnsatz
from divi.qprog.algorithms import GenericLayerAnsatz


class DummySpecStage(SpecStage[str]):
    """Simple spec stage that emits a single logical circuit."""

    def __init__(self, meta: MetaCircuit | None = None) -> None:
        super().__init__(name=type(self).__name__)
        self._meta = meta or cast(MetaCircuit, object())

    def expand(self, items: str, env: PipelineEnv) -> StageOutput[MetaCircuitBatch]:
        return StageOutput(batch={(("spec", "circ"),): self._meta})

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
    ) -> StageOutput[MetaCircuitBatch]:
        out: MetaCircuitBatch = {}
        for parent_key, meta in batch.items():
            for idx in range(self._n_children):
                child_key = parent_key + ((self._branch_prefix, idx),)
                out[child_key] = meta
        return StageOutput(batch=out)

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
    """Fan-out stage whose output is keyed to one evaluation.

    Folds ``env.evaluation_counter`` into the cache key so its output is
    reused across forward passes of one evaluation, then recomputed once the
    counter advances — mirroring QDrift's per-evaluation resampling."""

    def cache_key_extras(self, env):
        return (env.evaluation_counter,)


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


def measured_qubits(qasm: str) -> set[int]:
    """Qubit indices measured by a QASM body/string (``measure q[i] -> c[i];``)."""
    return {
        int(line.split("[")[1].split("]")[0])
        for line in qasm.splitlines()
        if line.startswith("measure")
    }


def two_group_meta() -> MetaCircuit:
    """MetaCircuit with 0.9*Z + 0.4*X for MeasurementStage to produce 2 groups."""
    qc = QuantumCircuit(1)
    qc.h(0)
    observable = SparsePauliOp.from_list([("Z", 0.9), ("X", 0.4)])
    return MetaCircuit(
        circuit_bodies=(((), circuit_to_dag(qc)),),
        observable=observable,
    )


def meta_with_observable(observable: SparsePauliOp) -> MetaCircuit:
    """MetaCircuit over ``observable``'s register, carrying it as the observable."""
    qc = QuantumCircuit(observable.num_qubits)
    qc.h(range(observable.num_qubits))
    return MetaCircuit(
        circuit_bodies=(((), circuit_to_dag(qc)),),
        observable=observable,
    )


def h2_vqe(backend, optimizer, **kwargs):
    """An H₂ HartreeFock VQE — the molecule boilerplate the dry-run tests share."""
    return VQE(
        molecule=qp.qchem.Molecule(
            symbols=["H", "H"],
            coordinates=np.array([(0.0, 0.0, 0.0), (0.0, 0.0, 0.5)]),
        ),
        ansatz=HartreeFockAnsatz(),
        n_layers=1,
        backend=backend,
        optimizer=optimizer,
        **kwargs,
    )


def metric_compatible_vqe(backend, optimizer, n_layers: int = 1):
    """A VQE whose GenericLayerAnsatz is compatible with all three metric
    estimators (expval cost, invertible, FS-supported RY/RZ gates)."""
    return VQE(
        molecule=qp.qchem.Molecule(
            symbols=["H", "H"],
            coordinates=np.array([(0.0, 0.0, 0.0), (0.0, 0.0, 0.74)]),
        ),
        ansatz=GenericLayerAnsatz([RYGate, RZGate]),
        n_layers=n_layers,
        backend=backend,
        optimizer=optimizer,
    )


def parametric_twirlable_meta() -> MetaCircuit:
    """MetaCircuit with CX gates (twirlable) and free parameters (bindable)."""
    theta = Parameter("theta")
    phi = Parameter("phi")
    qc = QuantumCircuit(2)
    qc.rx(theta, 0)
    qc.cx(0, 1)
    qc.ry(phi, 1)
    qc.cx(1, 0)
    return MetaCircuit(
        circuit_bodies=(((), circuit_to_dag(qc)),),
        parameters=(theta, phi),
        observable=SparsePauliOp.from_list([("ZZ", 0.9), ("XX", 0.4)]),
    )


def dry_run_stages(
    stages,
    env,
    *,
    dry=True,
    name="test",
    cadence=PipelineCadence.PER_EVALUATION,
    **pipeline_kwargs,
):
    """Build a pipeline over ``stages``, run one forward pass, and return
    ``(trace, report)`` for that pass.

    These are bare pipelines with no routine behind them, so ``cadence`` is the
    caller's choice rather than a declared one.

    A fresh ``CircuitPipeline`` is built per call, so a real and a dry pass over
    equivalent ``stages`` never share a forward-pass cache.
    """
    pipeline = CircuitPipeline(stages=stages, **pipeline_kwargs)
    trace = pipeline.run_forward_pass("ignored", env, dry=dry)
    report = dry_run_pipeline(name, trace, pipeline.stages, env, cadence)
    return trace, report


def assert_same_fanout(report_a, report_b):
    """Two reports agree on total circuit count and per-stage fan-out factor."""
    assert report_a.total_circuits == report_b.total_circuits
    for stage_a, stage_b in zip(report_a.stages, report_b.stages):
        assert (
            stage_a.factor == stage_b.factor
        ), f"{stage_a.name}: {stage_a.factor} != {stage_b.factor}"


def assert_report_dicts_match(reports_a, reports_b):
    """Per-pipeline report dicts (``QuantumProgram.dry_run`` output) agree
    pipeline-by-pipeline on totals and per-stage fan-out."""
    assert set(reports_a) == set(reports_b)
    for name in reports_a:
        assert_same_fanout(reports_a[name], reports_b[name])


class MisTaggedFanoutStage(BundleStage):
    """Test double: the defining bug of a hand-written fan-out stage.

    Duplicates each body without extending its tag, so execution — which keys
    circuits by tag — would collapse them back into one submission.
    """

    def __init__(self, n_copies: int = 3):
        super().__init__(name="mis-tagged")
        self._n_copies = n_copies

    @property
    def axis_name(self) -> str:
        return "replica"

    @property
    def consumes_dag_bodies(self) -> bool:
        return False

    def expand(self, batch, env):
        return StageOutput(
            batch={
                key: mc.set_circuit_bodies(
                    tuple(
                        (tag, dag)
                        for tag, dag in mc.circuit_bodies
                        for _ in range(self._n_copies)
                    )
                )
                for key, mc in batch.items()
            }
        )

    def reduce(self, results, env, token):
        return dict(results)


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
