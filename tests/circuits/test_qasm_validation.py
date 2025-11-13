# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from divi.circuits import is_valid_qasm, validate_qasm, validate_qasm_count_qubits

VALID_QASM = {
    "argvalues": [
        """OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    creg c[2];
    h q[0];
    """,
        """OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[1];
    gate my_gate(a, b, c) q {
        U(a, b, c) q;
    }
    my_gate(pi/2, 0, 0) q[0];
    """,
        """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    creg c[2];
    x q[0];
    barrier q;
    measure q[0] -> c[0];
    if(c==1) h q[1];
    """,
        """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg a[2];
    qreg b[2];
    cx a, b;
    """,
        """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[3];
    barrier q[0], q[2];
    """,
        """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[1];
    ry(cos(pi)) q[0];
    """,
        """
    OPENQASM 2.0;
    qubit[2] q;
    bit[2] c;
    measure q[0] -> c[0];
    """,
        """
    OPENQASM 2.0;
    qreg q[1];
    reset q[0];
    """,
        """
    OPENQASM 2.0;
    gate my_gate q {
        h q;
    }
    qreg q[1];
    my_gate q[0];
    """,
        """
    OPENQASM 2.0;
    qreg q[1];
    ry((2*pi)/3) q[0];
    """,
        """
    OPENQASM 2.0;
    // line comment
    qreg q[1];
    /* block
       comment */
    h q[0];
    """,
        """
    OPENQASM 2.0;
    qubit q;
    bit c;
    measure q -> c;
    """,
        """
    OPENQASM 2.0;
    qreg r[1];
    gate g1 q {}
    gate g2 q { g1 q; }
    g2 r[0];
    """,
        """
    OPENQASM 2.0;
    qreg q[1];
    ry(-2*((-pi+1)/3)^4) q[0];
    """,
    ],
    "ids": [
        "Simple",
        "GateDef",
        "BarrierConditional",
        "Broadcast",
        "ParamBarrier",
        "Math",
        "QubitBitDecl",
        "Reset",
        "GateDefNoParams",
        "ComplexMath",
        "WithComments",
        "QubitBitNoSize",
        "UserGateInBody",
        "ComplexExpr",
    ],
}

INVALID_QASM = {
    "argvalues": [
        # Missing header
        """qreg q[2];
    creg c[2];
    h q[0];""",
        # Incorrect register name
        """OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    h my_qreg[0];""",
        # Missing semicolon
        """OPENQASM 2.0
    include "qelib1.inc";
    qreg q[2];
    h q[0];""",
        # Gate with an invalid parameter count
        """OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    cx q[0];""",
        # Invalid conditional statement (if on qreg)
        """OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    creg c[2];
    measure q[0] -> c[0];
    if(q==1) h q[1];""",
        # Invalid broadcast (mismatched register sizes)
        """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg a[2];
    qreg b[3];
    cx a, b;
    """,
        # Invalid redefinition of register
        """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    creg q[2];
    """,
        # Out of bounds index
        """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    h q[2];
    """,
        # Invalid parameter usage (undefined parameter)
        """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[1];
    gate my_rx(theta) q {
    rx(theta) q;
    }
    rx(theta) q[0];
    """,
        # Using a classical register in a quantum operation
        """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[1];
    creg c[1];
    h c[0];
    """,
        # Redefining a built-in gate
        """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[1];
    gate h q { U(pi/2, 0, pi/2) q; }
    h q[0];
    """,
        # Illegal character
        """OPENQASM 2.0;
    qreg q[1];
    $ q[0];
    """,
        # Unsupported QASM version
        """OPENQASM 1.0;
    qreg q[1];
    h q[0];
    """,
        # Statement not ending in semicolon
        """OPENQASM 2.0;
    qreg q[2]
    """,
        # Float where integer is expected
        """OPENQASM 2.0;
    qreg q[2.5];
    """,
        # If value too large for creg
        """OPENQASM 2.0;
    creg c[2];
    if(c==4) h q[0];
    """,
        # Unknown symbol in expression
        """OPENQASM 2.0;
    qreg q[1];
    rx(alpha) q[0];
    """,
        # Invalid token in gate body
        """OPENQASM 2.0;
    gate bad_gate q {
        qreg r[1];
    }
    """,
        # Gate call in body with wrong qubit count
        """OPENQASM 2.0;
    gate my_gate q {
        cx q, q;
    }
    """,
        # --- NEW INVALID CASES ---
        # Duplicate qreg
        "OPENQASM 2.0; qreg q[1]; qreg q[1];",
        # Duplicate qubit
        "OPENQASM 2.0; qubit q; qubit q;",
        # Duplicate bit
        "OPENQASM 2.0; bit c; bit c;",
        # Duplicate gate
        "OPENQASM 2.0; gate g q {} gate g q {}",
        # Unknown gate top-level
        "OPENQASM 2.0; qreg q[1]; bad_gate q[0];",
        # Wrong param count top-level
        "OPENQASM 2.0; qreg q[1]; rx q[0];",
        # Unknown gate in body
        "OPENQASM 2.0; qreg q[1]; gate g q { bad_gate q; }",
        # Wrong param count in body
        "OPENQASM 2.0; qreg q[1]; gate g q { rx q; }",
        # Wrong qubit count in body
        "OPENQASM 2.0; qreg q[1]; gate g q { h q, q; }",
        # Unknown qubit in body
        "OPENQASM 2.0; qreg q[1]; gate g q { h r; }",
        # Measure size mismatch
        "OPENQASM 2.0; qreg q[2]; creg c[3]; measure q -> c;",
        # Measure unknown qreg
        "OPENQASM 2.0; creg c[1]; measure q -> c;",
        # Measure qubit out of bounds
        "OPENQASM 2.0; qreg q[1]; creg c[1]; measure q[1] -> c[0];",
        # Measure unknown creg
        "OPENQASM 2.0; qreg q[1]; measure q -> c;",
        # Measure bit out of bounds
        "OPENQASM 2.0; qreg q[1]; creg c[1]; measure q[0] -> c[1];",
        # Reset unknown qreg
        "OPENQASM 2.0; reset q;",
        # Reset out of bounds
        "OPENQASM 2.0; qreg q[1]; reset q[1];",
        # Unexpected token
        "OPENQASM 2.0; -> q[0];",
        # Invalid expression
        "OPENQASM 2.0; qreg q[1]; ry((pi) q[0];",
    ],
    "ids": [
        "MissingHeader",
        "IncorrectRegName",
        "MissingSemicolon",
        "InvalidGateParamCount",
        "InvalidConditional",
        "InvalidBroadcast",
        "RegRedefinition",
        "OutOfBoundsIndex",
        "UndefinedParam",
        "ClassicalInQuantumOp",
        "RedefineBuiltinGate",
        "IllegalChar",
        "UnsupportedVersion",
        "MissingStmtSemicolon",
        "FloatAsInt",
        "IfValTooLarge",
        "UnknownSymbolInExpr",
        "InvalidTokenInGateBody",
        "WrongQubitCountInGateBody",
        "DuplicateQreg",
        "DuplicateQubit",
        "DuplicateBit",
        "DuplicateGate",
        "UnknownGateTop",
        "WrongParamCountTop",
        "UnknownGateInBody",
        "WrongParamCountInBody",
        "WrongQubitCountInBody2",
        "UnknownQubitInBody",
        "MeasureSizeMismatch",
        "MeasureUnknownQreg",
        "MeasureQubitOOB",
        "MeasureUnknownCreg",
        "MeasureBitOOB",
        "ResetUnknownQreg",
        "ResetOOB",
        "UnexpectedToken",
        "InvalidExpr",
    ],
}


@pytest.mark.parametrize("qasm", **VALID_QASM)
def test_validate_qasm_valid(qasm):
    """Test that valid QASM strings pass validation."""
    assert is_valid_qasm(qasm) is True
    # Also verify we can get qubit count
    qubit_count = validate_qasm_count_qubits(qasm)
    assert isinstance(qubit_count, int)
    assert qubit_count > 0


@pytest.mark.parametrize("qasm", **INVALID_QASM)
def test_validate_qasm_invalid(qasm):
    """Test that invalid QASM strings return False."""
    assert is_valid_qasm(qasm) is False
    # Also verify it raises when using validate_qasm
    with pytest.raises(SyntaxError):
        validate_qasm(qasm)
