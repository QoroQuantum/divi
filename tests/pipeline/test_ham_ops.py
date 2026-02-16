# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for measurement pipeline support.

Validates that:
- MeasurementStage auto-detects expval backend and sets ham_ops in env.artifacts
- _default_execute_fn passes ham_ops to submit_circuits
- MeasurementStage.reduce unpacks {pauli: value} results correctly
- MeasurementStage with probs() converts counts → probs automatically
"""

import re

import numpy as np
import pennylane as qml
import pytest

from divi.backends import CircuitRunner, ExecutionResult
from divi.circuits import MetaCircuit
from divi.pipeline import CircuitPipeline, PipelineEnv
from divi.pipeline.stages import MeasurementStage

from .helpers import DummySpecStage

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


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
                # Return deterministic values: 0.1 * (index + 1) for each Pauli term.
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


@pytest.fixture
def expval_spy():
    return ExpvalBackendSpy(shots=100)


@pytest.fixture
def shots_spy():
    return ShotsBackendSpy(shots=100)


def _make_expval_meta():
    """MetaCircuit with observable: 0.5*Z(0) + 0.3*Z(1) on 2 qubits."""
    qscript = qml.tape.QuantumScript(
        ops=[qml.Hadamard(0)],
        measurements=[qml.expval(0.5 * qml.Z(0) + 0.3 * qml.Z(1))],
    )
    return MetaCircuit(source_circuit=qscript, symbols=np.array([], dtype=object))


def _make_probs_meta():
    """MetaCircuit with probs() on 2 qubits."""
    qscript = qml.tape.QuantumScript(
        ops=[qml.Hadamard(0)],
        measurements=[qml.probs()],
    )
    return MetaCircuit(source_circuit=qscript, symbols=np.array([], dtype=object))


# --------------------------------------------------------------------------- #
# Tests: MeasurementStage + expval backend
# --------------------------------------------------------------------------- #


class TestHamOpsExpvalBackend:
    """MeasurementStage with expval backend sets ham_ops and unpacks results."""

    def test_expand_sets_env_artifacts_ham_ops(self, expval_spy):
        """Expand with expval backend populates env.artifacts['ham_ops'] from observable."""
        env = PipelineEnv(backend=expval_spy)
        pipeline = CircuitPipeline(
            stages=[
                DummySpecStage(meta=_make_expval_meta()),
                MeasurementStage(),
            ]
        )
        trace = pipeline.run_forward_pass(initial_spec="x", env=env)
        assert "ham_ops" in env.artifacts
        terms = env.artifacts["ham_ops"].split(";")
        assert len(terms) == 2  # Z(0) and Z(1) → "ZI" and "IZ"
        assert "ZI" in terms
        assert "IZ" in terms

    def test_execute_fn_passes_ham_ops_to_backend(self, expval_spy):
        """_default_execute_fn passes env.artifacts['ham_ops'] to submit_circuits."""
        env = PipelineEnv(backend=expval_spy)
        pipeline = CircuitPipeline(
            stages=[
                DummySpecStage(meta=_make_expval_meta()),
                MeasurementStage(),
            ]
        )
        pipeline.run(initial_spec="x", env=env)
        assert expval_spy.last_ham_ops is not None
        assert "ZI" in expval_spy.last_ham_ops
        assert "IZ" in expval_spy.last_ham_ops

    def test_reduce_unpacks_pauli_dict_to_scalar(self, expval_spy):
        """Pipeline with expval backend reduces {pauli: value} → scalar expval."""
        env = PipelineEnv(backend=expval_spy)
        pipeline = CircuitPipeline(
            stages=[
                DummySpecStage(meta=_make_expval_meta()),
                MeasurementStage(),
            ]
        )
        result = pipeline.run(initial_spec="x", env=env)
        assert len(result) == 1
        value = next(iter(result.values()))
        assert isinstance(value, (int, float))

    def test_ham_ops_not_set_for_shots_backend(self, shots_spy):
        """Shots backends don't populate env.artifacts['ham_ops']."""
        env = PipelineEnv(backend=shots_spy)
        pipeline = CircuitPipeline(
            stages=[
                DummySpecStage(meta=_make_expval_meta()),
                MeasurementStage(),
            ]
        )
        pipeline.run_forward_pass(initial_spec="x", env=env)
        assert "ham_ops" not in env.artifacts


# --------------------------------------------------------------------------- #
# Tests: MeasurementStage counts→probs
# --------------------------------------------------------------------------- #


class TestProbsMeasurementReduce:
    """MeasurementStage with probs() converts counts → probability dicts."""

    def test_probs_pipeline_returns_probability_dict(self, shots_spy):
        """Pipeline with probs MeasurementStage returns {bitstring: prob}."""
        env = PipelineEnv(backend=shots_spy)
        pipeline = CircuitPipeline(
            stages=[
                DummySpecStage(meta=_make_probs_meta()),
                MeasurementStage(),
            ]
        )
        result = pipeline.run(initial_spec="x", env=env)
        assert len(result) == 1
        probs = next(iter(result.values()))
        assert isinstance(probs, dict)
        # Sum of probabilities should be 1.0
        assert abs(sum(probs.values()) - 1.0) < 1e-9
        # Bitstrings should be reversed (MSB-first convention)
        for bitstring in probs:
            assert len(bitstring) == 1  # 1-qubit circuit

    def test_probs_are_normalised_by_shots(self, shots_spy):
        """Probabilities equal count / total_shots, not raw count values."""
        env = PipelineEnv(backend=shots_spy)
        pipeline = CircuitPipeline(
            stages=[
                DummySpecStage(meta=_make_probs_meta()),
                MeasurementStage(),
            ]
        )
        result = pipeline.run(initial_spec="x", env=env)
        probs = next(iter(result.values()))
        # ShotsBackend returns "0": 80, "1": 20 with 100 shots
        assert probs.get("0") == 0.8
        assert probs.get("1") == 0.2
