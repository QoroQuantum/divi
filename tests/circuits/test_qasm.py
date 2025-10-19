# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import cirq
import pennylane as qml
import pytest
from sympy import Symbol

from divi.circuits.qasm import _ops_to_qasm, to_openqasm
from divi.circuits.qem import _NoMitigation


class TestOpsToQasm:
    """Test the _ops_to_qasm function."""

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
        def test_circuit():
            qml.RX(0.123456789, wires=0)
            qml.RY(0.987654321, wires=1)
            return qml.expval(qml.PauliZ(0))

        qscript = qml.tape.make_qscript(test_circuit)()
        operations = qscript.operations
        wires = qscript.wires

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

        def test_circuit():
            qml.RX(0.123456789, wires=0)
            return qml.expval(qml.PauliZ(0))

        qscript = qml.tape.make_qscript(test_circuit)()
        operations = qscript.operations
        wires = qscript.wires

        result = _ops_to_qasm(operations, precision=None, wires=wires)

        # Should use default Python string formatting (no rounding)
        assert "rx(0.123456789)" in result


class TestToOpenqasm:
    """Test the to_openqasm function."""

    def test_qem_protocol_without_symbols_raises_error(self):
        """Test that QEM protocol without symbols raises ValueError."""

        def test_circuit():
            qml.RX(0.5, wires=0)
            return qml.expval(qml.PauliZ(0))

        qscript = qml.tape.make_qscript(test_circuit)()
        measurement_groups = [[qml.expval(qml.PauliZ(0))]]

        # Use a QEM protocol but don't provide symbols
        qem_protocol = _NoMitigation()

        with pytest.raises(
            ValueError,
            match="When passing a QEMProtocol instance, the Sympy symbols in the circuit should be provided",
        ):
            to_openqasm(qscript, measurement_groups, qem_protocol=qem_protocol)

    def test_empty_circuit_returns_header_only(self):
        """Test empty circuit returns only QASM headers."""

        # Create a circuit with zero wires to trigger early return
        def empty_circuit():
            pass

        qscript = qml.tape.make_qscript(empty_circuit)()
        # Force num_wires to be 0 to trigger the early return
        qscript._wires = qml.wires.Wires([])
        measurement_groups = []

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
        def test_circuit():
            qml.RX(0.5, wires=0)
            qml.RY(0.5, wires=1)
            qml.RZ(0.5, wires=2)
            # Original circuit measures wires 0 and 2
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliX(2))

        qscript = qml.tape.make_qscript(test_circuit)()
        # Create measurement groups that only care about wire 1
        measurement_groups = [[qml.expval(qml.PauliY(1))]]  # Only wire 1

        result = to_openqasm(qscript, measurement_groups, measure_all=False)

        # Should have measurements for each group
        assert len(result) == 1

        # Each result should be a tuple of (circuit, measurements)
        circuit, measurements = result[0]
        assert isinstance(circuit, str)
        assert isinstance(measurements, str)

        # Circuit should contain the operations
        assert "rx(0.5) q[0];" in circuit
        assert "ry(0.5) q[1];" in circuit
        assert "rz(0.5) q[2];" in circuit

        # CORRECT BEHAVIOR: When measure_all=False, it should only measure wires from the current measurement group
        # So it should measure wire 1 (from the measurement group) and NOT wires 0 and 2
        assert "measure q[1] -> c[1];" in measurements  # From measurement group
        assert "measure q[0] -> c[0];" not in measurements  # Not in measurement group
        assert "measure q[2] -> c[2];" not in measurements  # Not in measurement group

    def test_measure_all_false_with_no_original_measurements(self):
        """Test measure_all=False when main_qscript has no measurements."""

        # Now that the implementation is fixed, it should measure wires from the measurement group
        def test_circuit():
            qml.RX(0.5, wires=0)
            qml.RY(0.5, wires=1)
            # No measurements in the original circuit
            return (
                qml.probs()
            )  # This doesn't create measurements in main_qscript.measurements

        qscript = qml.tape.make_qscript(test_circuit)()
        measurement_groups = [[qml.expval(qml.PauliZ(0))]]

        result = to_openqasm(qscript, measurement_groups, measure_all=False)

        # Should have measurements for each group
        assert len(result) == 1

        circuit, measurements = result[0]
        assert isinstance(circuit, str)
        assert isinstance(measurements, str)

        # Now it measures wires from the measurement group (wire 0)
        # even though main_qscript.measurements is empty
        assert "measure q[0] -> c[0];" in measurements  # From measurement group
        assert "measure q[1] -> c[1];" not in measurements  # Not in measurement group

    def test_empty_measurement_groups_warning(self):
        """Test empty measurement groups triggers warning."""

        def test_circuit():
            qml.RX(0.5, wires=0)
            return qml.expval(qml.PauliZ(0))

        qscript = qml.tape.make_qscript(test_circuit)()
        measurement_groups = []  # Empty measurement groups

        with pytest.warns(
            UserWarning,
            match="No measurement groups provided. Returning the QASM of the circuit operations only.",
        ):
            result = to_openqasm(qscript, measurement_groups)

        # Should return just the circuit operations without measurements
        assert isinstance(result, list)
        assert len(result) == 1
        assert "OPENQASM 2.0;" in result[0]
        assert 'include "qelib1.inc";' in result[0]
        assert "qreg q[1];" in result[0]
        assert "creg c[1];" in result[0]
        assert "rx(0.5) q[0];" in result[0]
        # Should not have any measurements
        assert "measure" not in result[0]

    def test_qem_protocol_with_symbols(self):
        """Test QEM protocol with symbols works correctly."""

        def test_circuit():
            qml.RX(0.5, wires=0)
            return qml.expval(qml.PauliZ(0))

        qscript = qml.tape.make_qscript(test_circuit)()
        measurement_groups = [[qml.expval(qml.PauliZ(0))]]
        symbols = [Symbol("theta")]
        qem_protocol = _NoMitigation()

        result = to_openqasm(
            qscript, measurement_groups, symbols=symbols, qem_protocol=qem_protocol
        )

        # Should work without error and return tuples
        assert isinstance(result, list)
        assert len(result) == 1

        circuit, measurements = result[0]
        assert isinstance(circuit, str)
        assert isinstance(measurements, str)
        # QEM protocol processing converts back to OpenQASM 2.0
        assert "OPENQASM 2.0;" in circuit

    def test_qem_protocol_with_empty_symbols_list(self):
        """Test QEM protocol with empty symbols list (potential edge case)."""

        def test_circuit():
            qml.RX(0.5, wires=0)
            return qml.expval(qml.PauliZ(0))

        qscript = qml.tape.make_qscript(test_circuit)()
        measurement_groups = [[qml.expval(qml.PauliZ(0))]]
        symbols = []  # Empty symbols list
        qem_protocol = _NoMitigation()

        # This should work without error even with empty symbols
        result = to_openqasm(
            qscript, measurement_groups, symbols=symbols, qem_protocol=qem_protocol
        )

        # Should work without error
        assert isinstance(result, list)
        assert len(result) == 1

        circuit, measurements = result[0]
        assert isinstance(circuit, str)
        assert isinstance(measurements, str)

    def test_qem_protocol_cirq_conversion_bug(self):
        """Test QEM protocol cirq conversion with multiple circuits."""

        def test_circuit():
            qml.RX(0.5, wires=0)
            return qml.expval(qml.PauliZ(0))

        qscript = qml.tape.make_qscript(test_circuit)()
        measurement_groups = [[qml.expval(qml.PauliZ(0))]]
        symbols = [Symbol("theta")]

        # Use a mock QEM protocol that returns multiple circuits
        class MockQEMProtocol:
            def modify_circuit(self, cirq_circuit):
                # Return multiple circuits to test the loop
                return [cirq_circuit, cirq_circuit]

        qem_protocol = MockQEMProtocol()

        result = to_openqasm(
            qscript, measurement_groups, symbols=symbols, qem_protocol=qem_protocol
        )

        # Should return multiple circuits (one for each QEM circuit)
        assert isinstance(result, list)
        assert len(result) == 2  # Two circuits from QEM protocol

        # Each result should be a tuple
        for circuit, measurements in result:
            assert isinstance(circuit, str)
            assert isinstance(measurements, str)
            # QEM protocol processing converts back to OpenQASM 2.0
            assert "OPENQASM 2.0;" in circuit

    def test_diagonalizing_gates_edge_case(self):
        """Test edge case with diagonalizing gates."""

        def test_circuit():
            qml.RX(0.5, wires=0)
            return qml.expval(qml.PauliZ(0))

        qscript = qml.tape.make_qscript(test_circuit)()

        # Test with measurement group that has no diagonalizing gates
        measurement_groups = [
            [qml.expval(qml.PauliZ(0))]
        ]  # PauliZ doesn't need diagonalizing

        result = to_openqasm(qscript, measurement_groups)

        # Should work without error
        assert isinstance(result, list)
        assert len(result) == 1

        circuit, measurements = result[0]
        assert isinstance(circuit, str)
        assert isinstance(measurements, str)

        # Should have measurements but no diagonalizing gates
        assert "measure q[0] -> c[0];" in measurements
        # Should not have Hadamard gates (which would be diagonalizing gates)
        assert "h q[0];" not in measurements

    def test_return_measurements_separately(self):
        """Test return_measurements_separately=True returns tuple."""

        def test_circuit():
            qml.RX(0.5, wires=0)
            return qml.expval(qml.PauliZ(0))

        qscript = qml.tape.make_qscript(test_circuit)()
        measurement_groups = [[qml.expval(qml.PauliZ(0))]]

        result = to_openqasm(
            qscript, measurement_groups, return_measurements_separately=True
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
        assert "measure q[0] -> c[0];" in measurements[0]

    def test_multiple_measurement_groups(self):
        """Test multiple measurement groups are handled correctly."""

        def test_circuit():
            qml.RX(0.5, wires=0)
            qml.RY(0.5, wires=1)
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliX(1))

        qscript = qml.tape.make_qscript(test_circuit)()
        measurement_groups = [[qml.expval(qml.PauliZ(0))], [qml.expval(qml.PauliX(1))]]

        result = to_openqasm(qscript, measurement_groups)

        # Should return one circuit for each measurement group
        assert isinstance(result, list)
        assert len(result) == 2

        # Each result should be a tuple of (circuit, measurements)
        for i, (circuit, measurements) in enumerate(result):
            assert isinstance(circuit, str)
            assert isinstance(measurements, str)
            assert "OPENQASM 2.0;" in circuit
            assert 'include "qelib1.inc";' in circuit
            assert "qreg q[2];" in circuit
            assert "creg c[2];" in circuit
            assert "rx(0.5) q[0];" in circuit
            assert "ry(0.5) q[1];" in circuit
            assert "measure q[0] -> c[0];" in measurements
            assert "measure q[1] -> c[1];" in measurements

    def test_measure_all_true_measures_all_wires(self):
        """Test that measure_all=True measures all circuit wires."""

        def test_circuit():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.RZ(0.1, wires=2)
            # Circuit has 3 wires, but measurement group only specifies one.
            return qml.expval(qml.PauliZ(0))

        qscript = qml.tape.make_qscript(test_circuit)()
        # Measurement group only involves wire 0
        measurement_groups = [[qml.expval(qml.PauliZ(0))]]

        # Call with measure_all=True (the default)
        result = to_openqasm(qscript, measurement_groups, measure_all=True)

        # Result is a list of (circuit, measurement) products
        assert len(result) == 1
        _circuit_body, measurement_qasm = result[0]

        # Assert that all three wires are measured, despite the measurement group
        assert "measure q[0] -> c[0];" in measurement_qasm
        assert "measure q[1] -> c[1];" in measurement_qasm
        assert "measure q[2] -> c[2];" in measurement_qasm

    def test_qem_protocol_qasm_cleanup_works(self, monkeypatch):
        """Test that QASM from cirq is correctly cleaned up."""

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

        def test_circuit():
            qml.RX(0.5, wires=0)
            return qml.expval(qml.PauliZ(0))

        qscript = qml.tape.make_qscript(test_circuit)()
        measurement_groups = [[qml.expval(qml.PauliZ(0))]]

        # 3. Run the function with the mock protocol
        result = to_openqasm(
            qscript,
            measurement_groups,
            qem_protocol=MockQEMProtocol(),
            symbols=[Symbol("theta")],  # Must be provided
            return_measurements_separately=True,
        )

        circuits, _measurements = result
        cleaned_qasm = circuits[0]

        # 4. Assert the cleanup was successful
        assert "//" not in cleaned_qasm  # Comments should be removed
        assert "\n\n" not in cleaned_qasm  # Extra newlines should be removed
        assert "creg c[1];" in cleaned_qasm  # Classical register should be added

    def test_circuit_with_sympy_parameter_is_handled(self):
        """Tests that a circuit with a sympy Symbol is processed correctly."""

        theta = Symbol("theta")

        def test_circuit():
            qml.RX(theta, wires=0)
            qml.RY(0.4, wires=1)
            return qml.expval(qml.PauliZ(0))

        qscript = qml.tape.make_qscript(test_circuit)()
        measurement_groups = [[qml.expval(qml.PauliZ(0))]]

        # The main goal is to ensure this runs without error.
        # The internal handling of sympy objects is critical for decomposition.
        try:
            result = to_openqasm(
                qscript, measurement_groups, return_measurements_separately=True
            )
            circuits, _ = result
            circuit_qasm = circuits[0]

            # The string representation of the symbol should be in the output
            assert f"rx({str(theta)}) q[0];" in circuit_qasm
            assert "ry(0.4)" in circuit_qasm  # The numeric param should also be there

        except Exception as e:
            pytest.fail(f"to_openqasm failed with a sympy parameter: {e}")
