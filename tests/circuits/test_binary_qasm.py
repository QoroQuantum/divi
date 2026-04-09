# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Round-trip correctness tests for the binary QASM encoder/decoder."""

import math
import struct

import pytest

from divi.circuits._binary_qasm import (
    CircuitIR,
    Instruction,
    Measurement,
    decode,
    decode_columnar,
    decode_delta_columnar,
    decode_to_ir,
    encode,
    encode_columnar,
    encode_delta_columnar,
    encode_ir,
    ir_to_qasm,
    parse_qasm,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assert_ir_equal(a: CircuitIR, b: CircuitIR, param_tol: float = 1e-6):
    """Assert two CircuitIR objects are semantically equal."""
    assert a.n_qubits == b.n_qubits
    assert a.n_clbits == b.n_clbits
    assert len(a.instructions) == len(b.instructions)
    for ia, ib in zip(a.instructions, b.instructions):
        assert ia.gate == ib.gate
        assert ia.qubits == ib.qubits
        assert len(ia.params) == len(ib.params)
        for pa, pb in zip(ia.params, ib.params):
            assert abs(pa - pb) < param_tol, f"param mismatch: {pa} vs {pb}"
    assert len(a.measurements) == len(b.measurements)
    for ma, mb in zip(a.measurements, b.measurements):
        assert ma.qubit == mb.qubit
        assert ma.clbit == mb.clbit


# ---------------------------------------------------------------------------
# Test QASM parsing
# ---------------------------------------------------------------------------

class TestParseQasm:
    def test_simple_circuit(self):
        qasm = (
            'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
            "qreg q[2];\ncreg c[2];\n"
            "h q[0];\ncx q[0],q[1];\n"
            "measure q[0] -> c[0];\nmeasure q[1] -> c[1];\n"
        )
        ir = parse_qasm(qasm)
        assert ir.n_qubits == 2
        assert ir.n_clbits == 2
        assert len(ir.instructions) == 2
        assert ir.instructions[0].gate == "h"
        assert ir.instructions[0].qubits == [0]
        assert ir.instructions[1].gate == "cx"
        assert ir.instructions[1].qubits == [0, 1]
        assert len(ir.measurements) == 2

    def test_parameterized_gates(self):
        qasm = (
            'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
            "qreg q[1];\ncreg c[1];\n"
            "ry(1.23456789) q[0];\nrz(3.14159265) q[0];\n"
        )
        ir = parse_qasm(qasm)
        assert len(ir.instructions) == 2
        assert ir.instructions[0].gate == "ry"
        assert abs(ir.instructions[0].params[0] - 1.23456789) < 1e-7
        assert ir.instructions[1].gate == "rz"

    def test_no_measurements(self):
        qasm = (
            'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
            "qreg q[2];\ncreg c[2];\n"
            "h q[0];\n"
        )
        ir = parse_qasm(qasm)
        assert len(ir.measurements) == 0

    def test_pi_parameter(self):
        qasm = (
            'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
            "qreg q[1];\ncreg c[1];\n"
            "rx(pi) q[0];\n"
        )
        ir = parse_qasm(qasm)
        assert abs(ir.instructions[0].params[0] - math.pi) < 1e-6

    def test_multi_param_gate(self):
        qasm = (
            'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
            "qreg q[1];\ncreg c[1];\n"
            "u3(1.0,2.0,3.0) q[0];\n"
        )
        ir = parse_qasm(qasm)
        assert ir.instructions[0].gate == "u3"
        assert len(ir.instructions[0].params) == 3
        assert abs(ir.instructions[0].params[0] - 1.0) < 1e-6
        assert abs(ir.instructions[0].params[1] - 2.0) < 1e-6
        assert abs(ir.instructions[0].params[2] - 3.0) < 1e-6


# ---------------------------------------------------------------------------
# Test binary round-trip (encode → decode)
# ---------------------------------------------------------------------------

class TestBinaryRoundTrip:
    def test_bell_state(self):
        qasm = (
            'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
            "qreg q[2];\ncreg c[2];\n"
            "h q[0];\ncx q[0],q[1];\n"
            "measure q[0] -> c[0];\nmeasure q[1] -> c[1];\n"
        )
        ir_original = parse_qasm(qasm)
        binary = encode(qasm)
        ir_decoded = decode_to_ir(binary)
        _assert_ir_equal(ir_original, ir_decoded)

    def test_parameterized_circuit(self):
        qasm = (
            'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
            "qreg q[4];\ncreg c[4];\n"
            "ry(0.12345678) q[0];\nrz(2.71828183) q[1];\n"
            "cx q[0],q[1];\nrx(1.57079633) q[2];\n"
            "u3(1.0,2.0,3.0) q[3];\n"
            "measure q[0] -> c[0];\nmeasure q[1] -> c[1];\n"
            "measure q[2] -> c[2];\nmeasure q[3] -> c[3];\n"
        )
        ir_original = parse_qasm(qasm)
        binary = encode(qasm)
        ir_decoded = decode_to_ir(binary)
        _assert_ir_equal(ir_original, ir_decoded)

    def test_no_measurements(self):
        qasm = (
            'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
            "qreg q[3];\ncreg c[3];\n"
            "h q[0];\nh q[1];\nh q[2];\n"
        )
        ir_original = parse_qasm(qasm)
        binary = encode(qasm)
        ir_decoded = decode_to_ir(binary)
        _assert_ir_equal(ir_original, ir_decoded)
        assert len(ir_decoded.measurements) == 0

    def test_measure_all(self):
        """When all qubits are measured to matching classical bits, measure_all flag is set."""
        qasm = (
            'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
            "qreg q[3];\ncreg c[3];\n"
            "h q[0];\n"
            "measure q[0] -> c[0];\n"
            "measure q[1] -> c[1];\n"
            "measure q[2] -> c[2];\n"
        )
        binary = encode(qasm)
        ir_decoded = decode_to_ir(binary)
        assert len(ir_decoded.measurements) == 3
        for i in range(3):
            assert ir_decoded.measurements[i].qubit == i
            assert ir_decoded.measurements[i].clbit == i

    def test_partial_measurement(self):
        """Partial measurements should be preserved exactly."""
        qasm = (
            'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
            "qreg q[4];\ncreg c[4];\n"
            "h q[0];\n"
            "measure q[0] -> c[2];\n"
            "measure q[3] -> c[1];\n"
        )
        ir_original = parse_qasm(qasm)
        binary = encode(qasm)
        ir_decoded = decode_to_ir(binary)
        _assert_ir_equal(ir_original, ir_decoded)

    def test_all_single_qubit_gates(self):
        """Test all supported 1-qubit, 0-param gates."""
        gates = ["id", "x", "y", "z", "h", "s", "sdg", "t", "tdg"]
        lines = ['OPENQASM 2.0;', 'include "qelib1.inc";', "qreg q[1];", "creg c[1];"]
        for g in gates:
            lines.append(f"{g} q[0];")
        qasm = "\n".join(lines) + "\n"
        ir_original = parse_qasm(qasm)
        binary = encode(qasm)
        ir_decoded = decode_to_ir(binary)
        _assert_ir_equal(ir_original, ir_decoded)
        assert len(ir_decoded.instructions) == len(gates)

    def test_two_qubit_gates(self):
        gates_no_param = ["cx", "cz", "swap"]
        gates_1_param = ["crx", "cry", "crz"]
        lines = ['OPENQASM 2.0;', 'include "qelib1.inc";', "qreg q[2];", "creg c[2];"]
        for g in gates_no_param:
            lines.append(f"{g} q[0],q[1];")
        for g in gates_1_param:
            lines.append(f"{g}(1.5) q[0],q[1];")
        qasm = "\n".join(lines) + "\n"
        ir_original = parse_qasm(qasm)
        binary = encode(qasm)
        ir_decoded = decode_to_ir(binary)
        _assert_ir_equal(ir_original, ir_decoded)

    def test_three_qubit_gates(self):
        lines = [
            'OPENQASM 2.0;', 'include "qelib1.inc";',
            "qreg q[3];", "creg c[3];",
            "ccx q[0],q[1],q[2];",
            "cswap q[0],q[1],q[2];",
        ]
        qasm = "\n".join(lines) + "\n"
        ir_original = parse_qasm(qasm)
        binary = encode(qasm)
        ir_decoded = decode_to_ir(binary)
        _assert_ir_equal(ir_original, ir_decoded)

    def test_empty_circuit(self):
        qasm = (
            'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
            "qreg q[2];\ncreg c[2];\n"
        )
        ir_original = parse_qasm(qasm)
        binary = encode(qasm)
        ir_decoded = decode_to_ir(binary)
        _assert_ir_equal(ir_original, ir_decoded)
        assert len(ir_decoded.instructions) == 0

    def test_wide_indices(self):
        """Circuits with >255 qubits should use 2-byte indices."""
        ir = CircuitIR(
            n_qubits=300,
            n_clbits=300,
            instructions=[
                Instruction("h", [0]),
                Instruction("cx", [0, 299]),
            ],
            measurements=[Measurement(0, 0), Measurement(299, 299)],
        )
        binary = encode_ir(ir)
        ir_decoded = decode_to_ir(binary)
        _assert_ir_equal(ir, ir_decoded)

    def test_qasm_text_round_trip(self):
        """encode → decode should produce valid, re-parseable QASM."""
        qasm = (
            'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
            "qreg q[4];\ncreg c[4];\n"
            "x q[0];\nx q[1];\n"
            "ry(0.00000000) q[2];\n"
            "cx q[2],q[3];\ncx q[2],q[0];\ncx q[3],q[1];\n"
            "measure q[0] -> c[0];\nmeasure q[1] -> c[1];\n"
            "measure q[2] -> c[2];\nmeasure q[3] -> c[3];\n"
        )
        binary = encode(qasm)
        reconstructed = decode(binary)
        # Re-parse the reconstructed QASM
        ir_from_reconstructed = parse_qasm(reconstructed)
        ir_from_original = parse_qasm(qasm)
        _assert_ir_equal(ir_from_original, ir_from_reconstructed)


# ---------------------------------------------------------------------------
# Test binary format properties
# ---------------------------------------------------------------------------

class TestBinaryFormat:
    def test_header_magic(self):
        qasm = (
            'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
            "qreg q[1];\ncreg c[1];\nh q[0];\n"
        )
        binary = encode(qasm)
        assert binary[:2] == b"BQ"

    def test_compact_size(self):
        """Binary should be smaller than the raw QASM text."""
        qasm = (
            'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
            "qreg q[4];\ncreg c[4];\n"
            "x q[0];\nx q[1];\n"
            "ry(0.12345678) q[2];\n"
            "cx q[2],q[3];\ncx q[2],q[0];\ncx q[3],q[1];\n"
            "measure q[0] -> c[0];\nmeasure q[1] -> c[1];\n"
            "measure q[2] -> c[2];\nmeasure q[3] -> c[3];\n"
        )
        binary = encode(qasm)
        assert len(binary) < len(qasm.encode("utf-8"))

    def test_invalid_magic_raises(self):
        with pytest.raises(ValueError, match="Invalid magic"):
            decode(b"XX\x01\x00\x01\x00\x01\x00\x00\x00\x01\x00")

    def test_unsupported_gate_raises(self):
        qasm = (
            'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
            "qreg q[1];\ncreg c[1];\n"
            "fakegate q[0];\n"
        )
        with pytest.raises(ValueError, match="Unsupported gate"):
            encode(qasm)

    def test_unknown_opcode_raises(self):
        # Craft binary with unknown opcode 0xBB
        header = struct.pack(">2sBHHIB", b"BQ", 0x01, 1, 1, 1, 0)
        bad_instruction = struct.pack(">BB", 0xBB, 0)
        with pytest.raises(ValueError, match="Unknown opcode"):
            decode(header + bad_instruction)


# ---------------------------------------------------------------------------
# Test columnar encoder round-trip
# ---------------------------------------------------------------------------

class TestColumnarRoundTrip:
    """Test the columnar (v2) encoder with byte-shuffled floats."""

    def _round_trip(self, ir: CircuitIR, shuffle: bool = True):
        binary = encode_columnar(ir, shuffle_floats=shuffle)
        ir_back = decode_columnar(binary)
        _assert_ir_equal(ir, ir_back)

    def test_bell_state(self):
        qasm = (
            'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
            "qreg q[2];\ncreg c[2];\n"
            "h q[0];\ncx q[0],q[1];\n"
            "measure q[0] -> c[0];\nmeasure q[1] -> c[1];\n"
        )
        self._round_trip(parse_qasm(qasm))

    def test_parameterized_circuit(self):
        qasm = (
            'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
            "qreg q[4];\ncreg c[4];\n"
            "ry(0.12345678) q[0];\nrz(2.71828183) q[1];\n"
            "cx q[0],q[1];\nrx(1.57079633) q[2];\n"
            "u3(1.0,2.0,3.0) q[3];\n"
            "measure q[0] -> c[0];\nmeasure q[1] -> c[1];\n"
            "measure q[2] -> c[2];\nmeasure q[3] -> c[3];\n"
        )
        self._round_trip(parse_qasm(qasm))

    def test_no_shuffle(self):
        qasm = (
            'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
            "qreg q[2];\ncreg c[2];\n"
            "ry(1.5) q[0];\ncx q[0],q[1];\n"
        )
        self._round_trip(parse_qasm(qasm), shuffle=False)

    def test_empty_circuit(self):
        ir = CircuitIR(2, 2, [], [])
        self._round_trip(ir)

    def test_wide_indices(self):
        ir = CircuitIR(
            n_qubits=300, n_clbits=300,
            instructions=[Instruction("h", [0]), Instruction("cx", [0, 299])],
            measurements=[Measurement(0, 0), Measurement(299, 299)],
        )
        self._round_trip(ir)

    def test_partial_measurement(self):
        ir = CircuitIR(
            n_qubits=4, n_clbits=4,
            instructions=[Instruction("h", [0])],
            measurements=[Measurement(0, 2), Measurement(3, 1)],
        )
        self._round_trip(ir)

    def test_all_gate_types(self):
        ir = CircuitIR(
            n_qubits=3, n_clbits=3,
            instructions=[
                Instruction("h", [0]),
                Instruction("rx", [0], [1.5]),
                Instruction("u2", [1], [0.5, 1.0]),
                Instruction("u3", [2], [1.0, 2.0, 3.0]),
                Instruction("cx", [0, 1]),
                Instruction("crx", [1, 2], [0.7]),
                Instruction("cu3", [0, 1], [1.0, 2.0, 3.0]),
                Instruction("ccx", [0, 1, 2]),
            ],
            measurements=[Measurement(i, i) for i in range(3)],
        )
        self._round_trip(ir)


class TestDeltaColumnarRoundTrip:
    """Test the delta-columnar (v3) encoder."""

    def _round_trip(self, ir: CircuitIR, shuffle: bool = True):
        binary = encode_delta_columnar(ir, shuffle_floats=shuffle)
        ir_back = decode_delta_columnar(binary)
        _assert_ir_equal(ir, ir_back)

    def test_bell_state(self):
        qasm = (
            'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
            "qreg q[2];\ncreg c[2];\n"
            "h q[0];\ncx q[0],q[1];\n"
            "measure q[0] -> c[0];\nmeasure q[1] -> c[1];\n"
        )
        self._round_trip(parse_qasm(qasm))

    def test_parameterized_large(self):
        """Test with a larger circuit to exercise delta encoding."""
        instructions = []
        for i in range(50):
            instructions.append(Instruction("ry", [i % 10], [float(i) * 0.1]))
            if i > 0:
                instructions.append(Instruction("cx", [i % 10, (i + 1) % 10]))
        ir = CircuitIR(
            n_qubits=10, n_clbits=10,
            instructions=instructions,
            measurements=[Measurement(i, i) for i in range(10)],
        )
        self._round_trip(ir)

    def test_wide_indices(self):
        ir = CircuitIR(
            n_qubits=300, n_clbits=300,
            instructions=[Instruction("h", [0]), Instruction("cx", [0, 299])],
            measurements=[Measurement(0, 0), Measurement(299, 299)],
        )
        self._round_trip(ir)

    def test_empty(self):
        ir = CircuitIR(2, 2, [], [])
        self._round_trip(ir)

    def test_no_shuffle(self):
        ir = CircuitIR(
            n_qubits=2, n_clbits=2,
            instructions=[Instruction("ry", [0], [1.5]), Instruction("cx", [0, 1])],
            measurements=[],
        )
        self._round_trip(ir, shuffle=False)
