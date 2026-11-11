# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

from divi.backends import CircuitRunner, ExecutionResult, normalise_circuit_batch
from divi.circuits._payloads import CircuitPayload
from divi.exceptions import IncrementalEvolutionUnavailableError, VJPUnavailableError
from tests._helpers import exact_match
from tests.backends._helpers import parametric_payload


class ConcreteCircuitRunner(CircuitRunner):
    """Concrete implementation of CircuitRunner for testing."""

    @property
    def supports_expval(self) -> bool:
        return False

    def submit_circuits(self, payloads: Sequence[CircuitPayload], **kwargs):
        return ExecutionResult(results=[])


class TestCircuitRunner:
    """Tests for CircuitRunner abstract base class."""

    def test_init_with_valid_shots(self):
        """Test initialization with valid shots."""
        runner = ConcreteCircuitRunner(shots=1000)
        assert runner.shots == 1000

    @pytest.mark.parametrize("shots", [0, -1])
    def test_init_with_non_positive_shots_raises(self, shots):
        """Test that ValueError is raised when shots is not positive."""
        with pytest.raises(
            ValueError,
            match=exact_match(f"Shots must be a positive integer. Got {shots}."),
        ):
            ConcreteCircuitRunner(shots=shots)

    def test_vjp_is_unavailable_by_default(self):
        runner = ConcreteCircuitRunner(shots=100)

        assert runner.supports_vjp is False
        with pytest.raises(
            VJPUnavailableError,
            match=exact_match(
                "ConcreteCircuitRunner cannot compute vector-Jacobian products."
            ),
        ):
            runner.submit_vjp([], ham_ops="Z", cotangents={})

    def test_incremental_evolution_is_unavailable_by_default(self):
        runner = ConcreteCircuitRunner(shots=100)

        assert runner.supports_incremental_evolution is False
        with pytest.raises(
            IncrementalEvolutionUnavailableError,
            match=exact_match(
                "ConcreteCircuitRunner cannot evolve a state incrementally."
            ),
        ):
            runner.submit_incremental_evolution("", "", steps=[1], ham_ops="Z")

    def test_payload_depths_follow_payload_then_row_order(self):
        deep = parametric_payload(
            "ry(w_0) q[0];\nrz(w_1) q[0];\nrx(w_0) q[0];\n",
            rows=((0.1, 0.2), (0.3, 0.4)),
            parameters=("w_0", "w_1"),
            n_qubits=2,
        )
        shallow = parametric_payload(
            "ry(w_0) q[0];\nry(w_1) q[1];\n",
            rows=((0.1, 0.2), (0.3, 0.4), (0.5, 0.6)),
            parameters=("w_0", "w_1"),
            n_qubits=2,
        )
        runner = ConcreteCircuitRunner(shots=100, track_depth=True)

        runner._record_payload_depths([deep, shallow])

        assert runner.depth_history == [[4, 4, 2, 2, 2]]
        untracked = ConcreteCircuitRunner(shots=100)
        untracked._record_payload_depths([deep, shallow])
        assert untracked.depth_history == []


@pytest.mark.parametrize("shots", [1000, np.int64(1000), np.uint16(1000)])
def test_shots_are_held_as_a_plain_int(shots):
    runner = ConcreteCircuitRunner(shots=shots)

    assert runner.shots == 1000
    assert type(runner.shots) is int


@pytest.mark.parametrize("shots", [100.0, 2.5, np.float32(10), True, "100", None])
def test_shots_that_are_not_integers_are_rejected(shots):
    with pytest.raises(
        ValueError,
        match=exact_match(f"Shots must be a positive integer. Got {shots}."),
    ):
        ConcreteCircuitRunner(shots=shots)


QASM_MINIMAL = (
    'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\ncreg c[1];\n'
    "h q[0];\nmeasure q[0] -> c[0];\n"
)


def _bell() -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return qc


def test_normalise_passes_qasm_mapping_through():
    assert normalise_circuit_batch({"c1": QASM_MINIMAL}) == {"c1": QASM_MINIMAL}


def test_normalise_labels_sequence_by_position():
    normalised = normalise_circuit_batch([_bell(), _bell()])

    assert list(normalised) == ["0", "1"]
    assert all(qasm.startswith("OPENQASM 2.0;") for qasm in normalised.values())


def test_normalise_exports_qiskit_circuits_in_mapping():
    normalised = normalise_circuit_batch({"bell": _bell()})

    assert list(normalised) == ["bell"]
    assert "cx q[0],q[1];" in normalised["bell"]


def test_normalise_accepts_mixed_mapping_values():
    normalised = normalise_circuit_batch({"raw": QASM_MINIMAL, "built": _bell()})

    assert normalised["raw"] == QASM_MINIMAL
    assert normalised["built"].startswith("OPENQASM 2.0;")


def test_normalise_empty_inputs():
    assert normalise_circuit_batch({}) == {}
    assert normalise_circuit_batch([]) == {}


@pytest.mark.parametrize("single", [QASM_MINIMAL, _bell()])
def test_normalise_rejects_single_circuit(single):
    with pytest.raises(TypeError, match="wrap a single circuit in a list"):
        normalise_circuit_batch(single)


def test_normalise_rejects_unsupported_element_type():
    with pytest.raises(TypeError, match="must be an OpenQASM string"):
        normalise_circuit_batch([42])


def test_normalise_reports_unexportable_circuit_by_label():
    qc = QuantumCircuit(1)
    qc.rx(Parameter("theta"), 0)

    with pytest.raises(ValueError, match="Circuit 'unbound' cannot be exported"):
        normalise_circuit_batch({"unbound": qc})
