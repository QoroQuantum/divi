# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for divi.pipeline.stages._trotter_spec_stage."""

import pennylane as qml
import pytest

from divi.circuits import MetaCircuit, qscript_to_meta
from divi.hamiltonians import ExactTrotterization
from divi.pipeline import CircuitPipeline, PipelineEnv, PipelineTrace
from divi.pipeline._compilation import _compile_batch
from divi.pipeline.abc import ChildResults
from divi.pipeline.stages import MeasurementStage, TrotterSpecStage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _DummyStrategy:
    """Trotterization strategy that returns the Hamiltonian unchanged N times."""

    def __init__(self, n: int = 3, stateful: bool = False):
        self.n_hamiltonians_per_iteration = n
        self.stateful = stateful

    def process_hamiltonian(self, hamiltonian):
        return hamiltonian


def _meta_factory(hamiltonian, ham_id):
    qscript = qml.tape.QuantumScript(
        ops=[qml.Hadamard(0)], measurements=[qml.expval(hamiltonian)]
    )
    return qscript_to_meta(qscript)


# ---------------------------------------------------------------------------
# expand
# ---------------------------------------------------------------------------


class TestExpand:
    def test_produces_n_samples_keyed_by_ham_id(self, dummy_expval_backend):
        """expand produces one MetaCircuit per strategy sample, keyed by (ham, i)."""
        stage = TrotterSpecStage(
            _DummyStrategy(n=3), meta_circuit_factory=_meta_factory
        )
        env = PipelineEnv(backend=dummy_expval_backend)
        batch, token = stage.expand(qml.Z(0), env)

        assert len(batch) == 3
        for i in range(3):
            key = (("ham", i),)
            assert key in batch
            assert isinstance(batch[key], MetaCircuit)

    def test_exact_trotterization_produces_single_circuit(self, dummy_expval_backend):
        """ExactTrotterization has n_samples=1 → single entry."""
        stage = TrotterSpecStage(
            ExactTrotterization(), meta_circuit_factory=_meta_factory
        )
        env = PipelineEnv(backend=dummy_expval_backend)
        batch, _ = stage.expand(qml.Z(0) + qml.Z(1), env)
        assert len(batch) == 1

    def test_raises_empty_hamiltonian(self, dummy_expval_backend):
        stage = TrotterSpecStage(
            ExactTrotterization(), meta_circuit_factory=_meta_factory
        )
        env = PipelineEnv(backend=dummy_expval_backend)
        with pytest.raises(ValueError, match="only constant terms|empty"):
            stage.expand(qml.I(0), env)

    def test_raises_non_operator_input(self, dummy_expval_backend):
        stage = TrotterSpecStage(_DummyStrategy(), meta_circuit_factory=_meta_factory)
        env = PipelineEnv(backend=dummy_expval_backend)
        with pytest.raises(TypeError, match="Operator"):
            stage.expand("not a hamiltonian", env)


# ---------------------------------------------------------------------------
# introspect
# ---------------------------------------------------------------------------


class TestIntrospect:
    def test_returns_token_info(self, dummy_expval_backend):
        stage = TrotterSpecStage(
            _DummyStrategy(n=3), meta_circuit_factory=_meta_factory
        )
        env = PipelineEnv(backend=dummy_expval_backend)
        batch, token = stage.expand(qml.Z(0), env)

        info = stage.introspect(batch, env, token)
        assert info["strategy"] == "_DummyStrategy"
        assert info["n_samples"] == 3
        assert info["n_qubits"] == 1
        assert info["n_terms"] >= 1


# ---------------------------------------------------------------------------
# reduce
# ---------------------------------------------------------------------------


class TestReduce:
    def test_averages_scalar_results(self, dummy_expval_backend):
        """Scalar results from multiple ham samples are averaged."""
        env = PipelineEnv(backend=dummy_expval_backend)
        pipeline = CircuitPipeline(
            stages=[
                TrotterSpecStage(
                    _DummyStrategy(n=3), meta_circuit_factory=_meta_factory
                ),
                MeasurementStage(),
            ]
        )

        def _execute_fn(trace: PipelineTrace, env: PipelineEnv) -> ChildResults:
            _, lineage_by_label = _compile_batch(trace.final_batch)
            branch_keys = sorted(lineage_by_label.values(), key=str)
            return {bk: 1.0 + i for i, bk in enumerate(branch_keys)}

        reduced = pipeline.run(initial_spec=qml.Z(0), env=env, execute_fn=_execute_fn)
        assert len(reduced) >= 1
        assert any(v == pytest.approx(2.0) for v in reduced.values())

    def test_merges_histogram_results(self, dummy_expval_backend):
        """Dict (histogram) results from multiple ham samples are averaged."""
        stage = TrotterSpecStage(
            _DummyStrategy(n=2), meta_circuit_factory=_meta_factory
        )
        # Simulate histogram results keyed by ham axis
        results = {
            (("ham", 0),): {"00": 50, "11": 50},
            (("ham", 1),): {"00": 30, "11": 70},
        }
        env = PipelineEnv(backend=dummy_expval_backend)
        reduced = stage.reduce(results, env, token=None)
        assert len(reduced) == 1
        merged = next(iter(reduced.values()))
        assert isinstance(merged, dict)
        assert merged["00"] == pytest.approx(40.0)
        assert merged["11"] == pytest.approx(60.0)


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestProperties:
    def test_axis_name(self):
        stage = TrotterSpecStage(_DummyStrategy(), meta_circuit_factory=_meta_factory)
        assert stage.axis_name == "ham"

    def test_stateful_delegates_to_strategy(self):
        stage_stateless = TrotterSpecStage(
            _DummyStrategy(stateful=False), meta_circuit_factory=_meta_factory
        )
        stage_stateful = TrotterSpecStage(
            _DummyStrategy(stateful=True), meta_circuit_factory=_meta_factory
        )
        assert stage_stateless.stateful is False
        assert stage_stateful.stateful is True


# ---------------------------------------------------------------------------
# Dry-run / forward pass
# ---------------------------------------------------------------------------


class TestDryRun:
    def test_forward_pass_produces_ham_keyed_trace(self, dummy_expval_backend):
        """Forward pass produces a trace with ham-keyed MetaCircuits."""
        pipeline = CircuitPipeline(
            stages=[
                TrotterSpecStage(
                    _DummyStrategy(n=2), meta_circuit_factory=_meta_factory
                ),
                MeasurementStage(),
            ]
        )
        env = PipelineEnv(backend=dummy_expval_backend)
        trace = pipeline.run_forward_pass(initial_spec=qml.Z(0), env=env)

        assert len(trace.final_batch) >= 2
        # All keys should contain the ham axis
        for key in trace.final_batch:
            ham_axes = [axis for axis in key if axis[0] == "ham"]
            assert len(ham_axes) == 1
