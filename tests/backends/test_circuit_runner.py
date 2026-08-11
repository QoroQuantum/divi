# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence

import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

from divi.backends import CircuitRunner, ExecutionResult, normalise_circuit_batch
from divi.circuits._payloads import CircuitPayload


class ConcreteCircuitRunner(CircuitRunner):
    """Concrete implementation of CircuitRunner for testing."""

    @property
    def supports_expval(self) -> bool:
        return False

    @property
    def is_async(self) -> bool:
        return False

    def submit_circuits(self, payloads: Sequence[CircuitPayload], **kwargs):
        return ExecutionResult(results=[])


class TestCircuitRunner:
    """Tests for CircuitRunner abstract base class."""

    def test_init_with_valid_shots(self):
        """Test initialization with valid shots."""
        runner = ConcreteCircuitRunner(shots=1000)
        assert runner.shots == 1000

    def test_init_with_zero_shots_raises(self):
        """Test that ValueError is raised when shots is 0 (line 15)."""
        with pytest.raises(ValueError, match="Shots must be a positive integer"):
            ConcreteCircuitRunner(shots=0)

    def test_init_with_negative_shots_raises(self):
        """Test that ValueError is raised when shots is negative (line 15)."""
        with pytest.raises(ValueError, match="Shots must be a positive integer"):
            ConcreteCircuitRunner(shots=-1)

    def test_shots_property(self):
        """Test shots property getter."""
        runner = ConcreteCircuitRunner(shots=5000)
        assert runner.shots == 5000

    def test_concrete_implementation(self):
        """Test that a concrete implementation works correctly."""
        runner = ConcreteCircuitRunner(shots=100)
        assert runner.shots == 100
        assert runner.supports_expval is False
        assert runner.is_async is False
        result = runner.submit_circuits([])
        assert isinstance(result, ExecutionResult)
        assert result.results == []


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
