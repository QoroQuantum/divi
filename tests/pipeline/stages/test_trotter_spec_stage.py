# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for divi.pipeline.stages._trotter_spec_stage."""

import pennylane as qp
import pytest
from qiskit.quantum_info import SparsePauliOp

from divi.circuits import MetaCircuit, qscript_to_meta
from divi.hamiltonians import (
    ExactTrotterization,
    QDrift,
    TrotterizationResult,
    TrotterizationStrategy,
)
from divi.pipeline import CircuitPipeline, PipelineEnv, PipelineTrace
from divi.pipeline._compilation import _compile_batch
from divi.pipeline.abc import ChildResults
from divi.pipeline.stages import MeasurementStage, TrotterSpecStage

_Z0 = SparsePauliOp.from_list([("Z", 1.0)])
_Z0_Z1 = SparsePauliOp.from_list([("ZI", 1.0), ("IZ", 1.0)])
_I0 = SparsePauliOp.from_list([("I", 1.0)])


class _DummyStrategy(TrotterizationStrategy):
    """Trotterization strategy that returns the SPO unchanged N times."""

    def __init__(self, n: int = 3):
        self.n_hamiltonians_per_iteration = n

    def process_hamiltonian(self, hamiltonian, *, rng=None):
        return TrotterizationResult(hamiltonian)


def _meta_factory(result, ham_id):
    """Test factory: build a PL observable matching the SPO's single ``Z(0)`` term
    so ``qscript_to_meta`` produces the same MetaCircuit shape that
    TrotterSpecStage would otherwise emit on its own."""
    qscript = qp.tape.QuantumScript(
        ops=[qp.Hadamard(0)], measurements=[qp.expval(qp.Z(0))]
    )
    return qscript_to_meta(qscript)


class TestExpand:
    def test_produces_n_samples_keyed_by_ham_id(self, dummy_expval_backend):
        """expand produces one MetaCircuit per strategy sample, keyed by (ham, i)."""
        stage = TrotterSpecStage(
            _DummyStrategy(n=3), meta_circuit_factory=_meta_factory
        )
        env = PipelineEnv(backend=dummy_expval_backend)
        batch, token = stage.expand(_Z0, env)

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
        batch, _ = stage.expand(_Z0_Z1, env)
        assert len(batch) == 1

    def test_raises_empty_hamiltonian(self, dummy_expval_backend):
        stage = TrotterSpecStage(
            ExactTrotterization(), meta_circuit_factory=_meta_factory
        )
        env = PipelineEnv(backend=dummy_expval_backend)
        with pytest.raises(ValueError, match="only constant terms|empty"):
            stage.expand(_I0, env)

    def test_raises_non_operator_input(self, dummy_expval_backend):
        stage = TrotterSpecStage(_DummyStrategy(), meta_circuit_factory=_meta_factory)
        env = PipelineEnv(backend=dummy_expval_backend)
        with pytest.raises(TypeError, match="SparsePauliOp"):
            stage.expand("not a hamiltonian", env)

    def test_accepts_sparse_pauli_op_input(self, dummy_expval_backend):
        stage = TrotterSpecStage(
            _DummyStrategy(n=1), meta_circuit_factory=_meta_factory
        )
        env = PipelineEnv(backend=dummy_expval_backend)
        batch, token = stage.expand(SparsePauliOp.from_list([("Z", 1.0)]), env)

        assert len(batch) == 1
        assert token["n_terms"] == 1

    def test_rejects_non_hermitian_sparse_pauli_op(self, dummy_expval_backend):
        stage = TrotterSpecStage(
            _DummyStrategy(n=1), meta_circuit_factory=_meta_factory
        )
        env = PipelineEnv(backend=dummy_expval_backend)
        with pytest.raises(ValueError, match="Hermitian"):
            stage.expand(SparsePauliOp.from_list([("Z", 1.0j)]), env)


class TestIntrospect:
    def test_returns_token_info(self, dummy_expval_backend):
        stage = TrotterSpecStage(
            _DummyStrategy(n=3), meta_circuit_factory=_meta_factory
        )
        env = PipelineEnv(backend=dummy_expval_backend)
        batch, token = stage.expand(_Z0, env)

        info = stage.introspect(batch, env, token)
        assert info["strategy"] == "_DummyStrategy"
        assert info["n_samples"] == 3
        assert info["n_qubits"] == 1
        assert info["n_terms"] >= 1

    def test_exact_trotterization_omits_n_samples(self, dummy_expval_backend):
        # ExactTrotterization is deterministic — its n_samples is
        # structurally always 1, so introspect drops the line. Other fields
        # stay; only n_samples is filtered.
        stage = TrotterSpecStage(
            ExactTrotterization(), meta_circuit_factory=_meta_factory
        )
        env = PipelineEnv(backend=dummy_expval_backend)
        batch, token = stage.expand(_Z0, env)

        info = stage.introspect(batch, env, token)
        assert info["strategy"] == "ExactTrotterization"
        assert "n_samples" not in info
        assert info["n_qubits"] == 1
        assert info["n_terms"] >= 1


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

        reduced = pipeline.run(initial_spec=_Z0, env=env, execute_fn=_execute_fn)
        assert len(reduced) >= 1
        assert any(v == pytest.approx([2.0]) for v in reduced.values())

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


class TestProperties:
    def test_axis_name(self):
        stage = TrotterSpecStage(_DummyStrategy(), meta_circuit_factory=_meta_factory)
        assert stage.axis_name == "ham"

    def test_qdrift_keys_cache_per_evaluation(self):
        deterministic = TrotterSpecStage(
            ExactTrotterization(), meta_circuit_factory=_meta_factory
        )
        stochastic = TrotterSpecStage(
            QDrift(sampling_budget=1), meta_circuit_factory=_meta_factory
        )
        env = PipelineEnv(backend=None, evaluation_counter=3)
        assert deterministic.cache_key_extras(env) == ()
        assert stochastic.cache_key_extras(env) == (3,)

    def test_unseeded_rng_is_stable_within_evaluation(self):
        """Unseeded QDrift derives its RNG from (base_seed, evaluation_counter),
        so repeated expansions in one evaluation draw the SAME cohort (cost and
        metric agree) and advancing the counter resamples — without consuming the
        shared, mutable env.rng."""
        strategy = QDrift(sampling_budget=2, seed=None)
        env = PipelineEnv(backend=None, base_seed=12345, evaluation_counter=3)

        first = TrotterSpecStage._rng_for_evaluation(strategy, env).integers(
            0, 10**6, 8
        )
        again = TrotterSpecStage._rng_for_evaluation(strategy, env).integers(
            0, 10**6, 8
        )
        assert first.tolist() == again.tolist()

        next_eval = PipelineEnv(backend=None, base_seed=12345, evaluation_counter=4)
        advanced = TrotterSpecStage._rng_for_evaluation(strategy, next_eval).integers(
            0, 10**6, 8
        )
        assert first.tolist() != advanced.tolist()


def test_forward_pass_produces_ham_keyed_trace(dummy_expval_backend):
    """Forward pass produces a trace with ham-keyed MetaCircuits."""
    pipeline = CircuitPipeline(
        stages=[
            TrotterSpecStage(_DummyStrategy(n=2), meta_circuit_factory=_meta_factory),
            MeasurementStage(),
        ]
    )
    env = PipelineEnv(backend=dummy_expval_backend)
    trace = pipeline.run_forward_pass(initial_spec=_Z0, env=env)

    assert len(trace.final_batch) >= 2
    # All keys should contain the ham axis
    for key in trace.final_batch:
        ham_axes = [axis for axis in key if axis[0] == "ham"]
        assert len(ham_axes) == 1
