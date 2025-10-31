# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from divi.extern.cirq import is_valid_qasm

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
    ],
    "ids": [
        "Simple",
        "GateDef",
        "BarrierConditional",
        "Broadcast",
        "ParamBarrier",
        "Math",
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
    ],
}


@pytest.mark.parametrize("qasm", **VALID_QASM)
def test_validate_qasm_valid(qasm):
    """Test that valid QASM strings pass validation."""
    result = is_valid_qasm(qasm)
    assert isinstance(result, int)
    assert result > 0


@pytest.mark.parametrize("qasm", **INVALID_QASM)
def test_validate_qasm_invalid(qasm):
    """Test that invalid QASM strings returns an error string."""
    assert isinstance(is_valid_qasm(qasm), str)
