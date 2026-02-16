# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import cirq
import pennylane as qml
import pytest
import sympy as sp
from sympy import Symbol

from divi.circuits._qasm_conversion import _ops_to_qasm, to_openqasm
from divi.circuits.qem import _NoMitigation, apply_protocol_to_qasm


class TestOpsToQasm:
    """Test the _ops_to_qasm function."""

    def _create_circuit_and_extract(self, ops, measurements=None):
        """Helper to create a QuantumScript and extract operations and wires."""
        if measurements is None:
            measurements = [qml.expval(qml.PauliZ(0))]
        qscript = qml.tape.QuantumScript(ops=ops, measurements=measurements)
        return qscript.operations, qscript.wires

    def test_unsupported_operation_raises_error(self):
        """Test that unsupported operations raise ValueError."""
        # Create a mock operation with an unsupported name
        mock_op = type(
            "MockOp",
            (),
            {
                "name": "UnsupportedGate",
                "wires": qml.wires.Wires([0]),
                "num_params": 0,
                "parameters": [],
            },
        )()

        operations = [mock_op]
        wires = qml.wires.Wires([0])

        with pytest.raises(
            ValueError,
            match="Operation UnsupportedGate not supported by the QASM serializer",
        ):
            _ops_to_qasm(operations, precision=None, wires=wires)

    def test_precision_parameter_formatting(self):
        """Test precision parameter formatting."""

        # Create a simple circuit with parameterized gates
        ops = [
            qml.RX(0.123456789, wires=0),
            qml.RY(0.987654321, wires=1),
        ]
        operations, wires = self._create_circuit_and_extract(ops)

        # Test with precision=3
        result = _ops_to_qasm(operations, precision=3, wires=wires)

        # Check that parameters are formatted with 3 decimal places
        assert "rx(0.123)" in result
        assert "ry(0.988)" in result  # Rounded to 3 decimal places

        # Test with precision=5
        result = _ops_to_qasm(operations, precision=5, wires=wires)
        assert "rx(0.12346)" in result
        assert "ry(0.98765)" in result

    def test_no_precision_uses_default_formatting(self):
        """Test that None precision uses default string formatting."""

        ops = [qml.RX(0.123456789, wires=0)]
        operations, wires = self._create_circuit_and_extract(ops)

        result = _ops_to_qasm(operations, precision=None, wires=wires)

        # Should use default Python string formatting (no rounding)
        assert "rx(0.123456789)" in result


class TestToOpenqasm:
    """Test the to_openqasm function."""

    @pytest.fixture
    def simple_circuit(self):
        """Fixture for a simple single-qubit circuit with RX(0.5) and expval(PauliZ(0))."""
        ops = [qml.RX(0.5, wires=0)]
        return qml.tape.QuantumScript(ops=ops, measurements=[qml.expval(qml.PauliZ(0))])

    @pytest.fixture
    def default_measurement_group(self):
        """Fixture for the default measurement group with expval(PauliZ(0))."""
        return [[qml.expval(qml.PauliZ(0))]]

    def _assert_result_structure(self, result, expected_length=1):
        """Helper to assert the structure of a result list with tuples."""
        assert isinstance(result, list)
        assert len(result) == expected_length
        circuit, measurements = result[0]
        assert isinstance(circuit, str)
        assert isinstance(measurements, str)
        return circuit, measurements

    def _assert_qasm_headers(self, qasm_str, n_qubits=None):
        """Helper to assert QASM headers are present."""
        assert "OPENQASM 2.0;" in qasm_str
        assert 'include "qelib1.inc";' in qasm_str
        if n_qubits is not None:
            assert f"qreg q[{n_qubits}];" in qasm_str
            assert f"creg c[{n_qubits}];" in qasm_str

    def _assert_measurement(self, measurements_str, wire, should_exist=True):
        """Helper to assert measurement presence/absence for a wire."""
        measurement = f"measure q[{wire}] -> c[{wire}];"
        if should_exist:
            assert measurement in measurements_str
        else:
            assert measurement not in measurements_str

    def test_apply_protocol_to_qasm_without_symbols_on_symbolic_qasm_raises_error(
        self, default_measurement_group
    ):
        """Symbolic QASM should fail parsing if symbols are not declared."""
        theta = Symbol("theta")
        symbolic_circuit = qml.tape.QuantumScript(
            ops=[qml.RX(theta, wires=0)], measurements=[qml.expval(qml.PauliZ(0))]
        )
        circuits, _ = to_openqasm(
            symbolic_circuit,
            default_measurement_group,
            return_measurements_separately=True,
        )
        qasm_body = circuits[0]

        with pytest.raises(Exception):
            apply_protocol_to_qasm(qasm_body, _NoMitigation())

    def test_empty_circuit_returns_header_only(self):
        """Test empty circuit returns only QASM headers."""

        # Create a circuit with zero wires to trigger early return
        qscript = qml.tape.QuantumScript(ops=[], measurements=[])
        # Force num_wires to be 0 to trigger the early return
        qscript._wires = qml.wires.Wires([])
        measurement_groups = []

        with pytest.warns(
            UserWarning,
            match="No measurement groups provided",
        ):
            result = to_openqasm(qscript, measurement_groups)

        # Should return just the QASM headers as a string (early return)
        assert isinstance(result, str)
        assert "OPENQASM 2.0;" in result
        assert 'include "qelib1.inc";' in result
        # Should not have any quantum or classical registers
        assert "qreg" not in result
        assert "creg" not in result

    def test_measure_all_false_only_measured_wires(self):
        """Test measure_all=False only measures wires from the current measurement group."""

        # The key insight: measure_all=False uses measurement groups, not original measurements
        ops = [
            qml.RX(0.5, wires=0),
            qml.RY(0.5, wires=1),
            qml.RZ(0.5, wires=2),
        ]
        # Original circuit measures wires 0 and 2
        qscript = qml.tape.QuantumScript(
            ops=ops, measurements=[qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliX(2))]
        )
        # Create measurement groups that only care about wire 1
        measurement_groups = [[qml.expval(qml.PauliY(1))]]  # Only wire 1

        result = to_openqasm(qscript, measurement_groups, measure_all=False)

        # Should have measurements for each group
        circuit, measurements = self._assert_result_structure(result)

        # Circuit should contain the operations
        assert "rx(0.5) q[0];" in circuit
        assert "ry(0.5) q[1];" in circuit
        assert "rz(0.5) q[2];" in circuit

        # CORRECT BEHAVIOR: When measure_all=False, it should only measure wires from the current measurement group
        # So it should measure wire 1 (from the measurement group) and NOT wires 0 and 2
        self._assert_measurement(
            measurements, 1, should_exist=True
        )  # From measurement group
        self._assert_measurement(
            measurements, 0, should_exist=False
        )  # Not in measurement group
        self._assert_measurement(
            measurements, 2, should_exist=False
        )  # Not in measurement group

    def test_measure_all_false_with_no_original_measurements(self):
        """Test measure_all=False when main_qscript has no measurements."""

        # Now that the implementation is fixed, it should measure wires from the measurement group
        ops = [
            qml.RX(0.5, wires=0),
            qml.RY(0.5, wires=1),
        ]
        # No measurements in the original circuit
        qscript = qml.tape.QuantumScript(ops=ops, measurements=[qml.probs()])
        measurement_groups = [[qml.expval(qml.PauliZ(0))]]

        result = to_openqasm(qscript, measurement_groups, measure_all=False)

        # Should have measurements for each group
        circuit, measurements = self._assert_result_structure(result)

        # Now it measures wires from the measurement group (wire 0)
        # even though main_qscript.measurements is empty
        self._assert_measurement(
            measurements, 0, should_exist=True
        )  # From measurement group
        self._assert_measurement(
            measurements, 1, should_exist=False
        )  # Not in measurement group

    def test_empty_measurement_groups_warning(self, simple_circuit):
        """Test empty measurement groups triggers warning."""

        measurement_groups = []  # Empty measurement groups

        with pytest.warns(
            UserWarning,
            match="No measurement groups provided. Returning the QASM of the circuit operations only.",
        ):
            result = to_openqasm(simple_circuit, measurement_groups)

        # Should return just the circuit operations without measurements
        assert isinstance(result, list) and len(result) == 1
        self._assert_qasm_headers(result[0], n_qubits=1)
        assert "rx(0.5) q[0];" in result[0]
        # Should not have any measurements
        assert "measure" not in result[0]

    def test_apply_protocol_to_qasm_with_symbols(self, default_measurement_group):
        """Test apply_protocol_to_qasm works on symbolic circuits when symbols are provided."""
        theta = Symbol("theta")
        symbolic_circuit = qml.tape.QuantumScript(
            ops=[qml.RX(theta, wires=0)], measurements=[qml.expval(qml.PauliZ(0))]
        )
        circuits, _ = to_openqasm(
            symbolic_circuit,
            default_measurement_group,
            return_measurements_separately=True,
        )
        qasm_body = circuits[0]

        result = apply_protocol_to_qasm(
            qasm_body,
            _NoMitigation(),
            symbols=[theta],
        )

        assert isinstance(result, tuple) and len(result) == 1
        assert result[0][0] == (("qem", 0),)  # single (tag, body) pair
        assert "OPENQASM 2.0;" in result[0][1]

    def test_apply_protocol_to_qasm_with_empty_symbols_list(
        self, simple_circuit, default_measurement_group
    ):
        """Empty symbols list should be fine for non-symbolic QASM."""
        circuits, _ = to_openqasm(
            simple_circuit,
            default_measurement_group,
            return_measurements_separately=True,
        )
        qasm_body = circuits[0]

        result = apply_protocol_to_qasm(
            qasm_body,
            _NoMitigation(),
            symbols=[],
        )

        assert isinstance(result, tuple) and len(result) == 1
        assert isinstance(result[0], tuple) and len(result[0]) == 2
        assert isinstance(result[0][1], str)

    def test_apply_protocol_to_qasm_with_sympy_array_symbols(
        self, default_measurement_group
    ):
        """Test apply_protocol_to_qasm with numpy array of sympy symbols (from sp.symarray).

        This test verifies that when symbols contains numpy arrays (from sp.symarray),
        they are properly flattened and declared individually in QASM 3.0 format.
        Without this fix, the QASM parser would fail with QasmException due to
        invalid array notation like "[theta_0 theta_1 theta_2]".
        """
        # Create a circuit with both individual symbols and arrays of symbols
        beta = sp.Symbol("beta")
        thetas = sp.symarray(
            "theta", 3
        )  # Returns numpy array: [theta_0, theta_1, theta_2]

        ops = [
            qml.RX(beta, wires=0),
            qml.U3(*thetas, wires=1),  # Unpack array for U3 gate
        ]
        circuit = qml.tape.QuantumScript(
            ops=ops, measurements=[qml.expval(qml.PauliZ(0))]
        )

        # Symbols list contains both individual symbol and array
        symbols = [beta, thetas]

        # This should work without error - array symbols should be flattened
        circuits, _ = to_openqasm(
            circuit,
            default_measurement_group,
            return_measurements_separately=True,
        )
        qasm_body = circuits[0]

        result = apply_protocol_to_qasm(
            qasm_body,
            _NoMitigation(),
            symbols=symbols,
        )

        assert isinstance(result, tuple) and len(result) == 1
        assert isinstance(result[0][1], str)

    def test_apply_protocol_to_qasm_multiple_modified_circuits(
        self, simple_circuit, default_measurement_group
    ):
        """Test apply_protocol_to_qasm returns one QASM per modified cirq circuit."""

        # Use a mock QEM protocol that returns multiple circuits
        class MockQEMProtocol:
            def modify_circuit(self, cirq_circuit):
                # Return multiple circuits to test the loop
                return [cirq_circuit, cirq_circuit]

        circuits, _ = to_openqasm(
            simple_circuit,
            default_measurement_group,
            return_measurements_separately=True,
        )
        qasm_body = circuits[0]

        result = apply_protocol_to_qasm(
            qasm_body,
            MockQEMProtocol(),
            symbols=[Symbol("theta")],
        )

        # Should return multiple (tag, body) pairs (one per QEM circuit)
        assert isinstance(result, tuple) and len(result) == 2
        for tag, qasm_str in result:
            assert isinstance(tag, tuple)
            assert isinstance(qasm_str, str)
            self._assert_qasm_headers(qasm_str)

    def test_diagonalizing_gates_edge_case(
        self, simple_circuit, default_measurement_group
    ):
        """Test edge case with diagonalizing gates."""

        # Test with measurement group that has no diagonalizing gates
        # PauliZ doesn't need diagonalizing
        result = to_openqasm(simple_circuit, default_measurement_group)

        # Should work without error
        circuit, measurements = self._assert_result_structure(result)

        # Should have measurements but no diagonalizing gates
        self._assert_measurement(measurements, 0, should_exist=True)
        # Should not have Hadamard gates (which would be diagonalizing gates)
        assert "h q[0];" not in measurements

    def test_return_measurements_separately(
        self, simple_circuit, default_measurement_group
    ):
        """Test return_measurements_separately=True returns tuple."""

        result = to_openqasm(
            simple_circuit,
            default_measurement_group,
            return_measurements_separately=True,
        )

        # Should return tuple of (circuits, measurements)
        assert isinstance(result, tuple)
        assert len(result) == 2

        circuits, measurements = result
        assert isinstance(circuits, list)
        assert isinstance(measurements, list)
        assert len(circuits) == 1
        assert len(measurements) == 1

        # Circuit should not contain measurements
        assert "measure" not in circuits[0]
        # Measurements should contain measurement instructions
        self._assert_measurement(measurements[0], 0, should_exist=True)

    def test_multiple_measurement_groups(self):
        """Test multiple measurement groups are handled correctly."""

        ops = [
            qml.RX(0.5, wires=0),
            qml.RY(0.5, wires=1),
        ]
        qscript = qml.tape.QuantumScript(
            ops=ops, measurements=[qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliX(1))]
        )
        measurement_groups = [[qml.expval(qml.PauliZ(0))], [qml.expval(qml.PauliX(1))]]

        result = to_openqasm(qscript, measurement_groups)

        # Should return one circuit for each measurement group
        assert isinstance(result, list) and len(result) == 2

        # Each result should be a tuple of (circuit, measurements)
        for i, (circuit, measurements) in enumerate(result):
            assert isinstance(circuit, str)
            assert isinstance(measurements, str)
            self._assert_qasm_headers(circuit, n_qubits=2)
            assert "rx(0.5) q[0];" in circuit
            assert "ry(0.5) q[1];" in circuit
            self._assert_measurement(measurements, 0, should_exist=True)
            self._assert_measurement(measurements, 1, should_exist=True)

    def test_measure_all_true_measures_all_wires(self, default_measurement_group):
        """Test that measure_all=True measures all circuit wires."""

        ops = [
            qml.Hadamard(wires=0),
            qml.CNOT(wires=[0, 1]),
            qml.RZ(0.1, wires=2),
        ]
        # Circuit has 3 wires, but measurement group only specifies one.
        qscript = qml.tape.QuantumScript(
            ops=ops, measurements=[qml.expval(qml.PauliZ(0))]
        )
        # Measurement group only involves wire 0

        # Call with measure_all=True (the default)
        result = to_openqasm(qscript, default_measurement_group, measure_all=True)

        # Result is a list of (circuit, measurement) products
        _circuit_body, measurement_qasm = self._assert_result_structure(result)

        # Assert that all three wires are measured, despite the measurement group
        self._assert_measurement(measurement_qasm, 0, should_exist=True)
        self._assert_measurement(measurement_qasm, 1, should_exist=True)
        self._assert_measurement(measurement_qasm, 2, should_exist=True)

    def test_apply_protocol_to_qasm_cleanup_works(
        self, monkeypatch, simple_circuit, default_measurement_group
    ):
        """Test that QASM from cirq is correctly cleaned up by apply_protocol_to_qasm."""

        # A "dirty" QASM string with features that should be cleaned.
        dirty_qasm_from_cirq = """
// Some header comment from Cirq

OPENQASM 2.0;
include "qelib1.inc";

qreg q[1];

// A gate
rx(0.5) q[0];


"""
        # 1. Mock the cirq.qasm function to return our dirty string
        monkeypatch.setattr(cirq, "qasm", lambda circuit: dirty_qasm_from_cirq)

        # 2. Define a mock QEM protocol that will trigger the conversion
        class MockQEMProtocol:
            def modify_circuit(self, cirq_circuit):
                # This doesn't need to be a real circuit, as cirq.qasm is mocked
                return [None]

        circuits, _ = to_openqasm(
            simple_circuit,
            default_measurement_group,
            return_measurements_separately=True,
        )
        qasm_body = circuits[0]

        # 3. Run the function with the mock protocol
        result = apply_protocol_to_qasm(
            qasm_body,
            MockQEMProtocol(),
            symbols=[Symbol("theta")],
        )

        assert isinstance(result, tuple) and len(result) == 1
        cleaned_qasm = result[0][1]  # (tag, body) -> body

        # 4. Assert the cleanup was successful
        assert "//" not in cleaned_qasm  # Comments should be removed
        assert "\n\n" not in cleaned_qasm  # Extra newlines should be removed
        assert "creg c[1];" in cleaned_qasm  # Classical register should be added

    def test_circuit_with_sympy_parameter_is_handled(self, default_measurement_group):
        """Tests that a circuit with a sympy Symbol is processed correctly."""

        theta = Symbol("theta")

        ops = [
            qml.RX(theta, wires=0),
            qml.RY(0.4, wires=1),
        ]
        qscript = qml.tape.QuantumScript(
            ops=ops, measurements=[qml.expval(qml.PauliZ(0))]
        )

        # The main goal is to ensure this runs without error.
        # The internal handling of sympy objects is critical for decomposition.
        try:
            result = to_openqasm(
                qscript, default_measurement_group, return_measurements_separately=True
            )
            circuits, _ = result
            circuit_qasm = circuits[0]

            # The string representation of the symbol should be in the output
            assert f"rx({str(theta)}) q[0];" in circuit_qasm
            assert "ry(0.4)" in circuit_qasm  # The numeric param should also be there

        except Exception as e:
            pytest.fail(f"to_openqasm failed with a sympy parameter: {e}")
