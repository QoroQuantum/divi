# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from divi.exp.cirq import validate_qasm

VALID_QASM = [
    # Simple circuit
    """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    creg c[2];
    h q[0];
    """,
    # Circuit with a gate definition
    """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[1];
    gate my_gate(a, b, c) q {
        U(a, b, c) q;
    }
    my_gate(pi/2, 0, 0) q[0];
    """,
    # Circuit with a barrier and a conditional
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
    # Broadcast gate application
    """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg a[2];
    qreg b[2];
    cx a, b;
    """,
    # Parameterized Barrier
    """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[3];
    barrier q[0], q[2];
    """,
    # Math
    """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[1];
    ry(cos(pi)) q[0];
    """,
]

INVALID_QASM = [
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
    # Invalid conditional statement
    """OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    creg c[2];
    measure q[0] -> c[0];
    if(q==1) h q[1];""",
    # Invalid broadcast
    """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg a[2];
    qreg b[3];
    cx a, b;
    """,
    # Invalid redefinition of reg
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
    # Invalid parameter usage
    """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[1];
    gate my_rx(theta) q {
    rx(theta) q;
    }
    rx(theta) q[0];
    """,
]


@pytest.mark.parametrize("qasm", VALID_QASM)
def test_validate_qasm_valid(qasm):
    """Test that valid QASM strings pass validation."""
    assert validate_qasm(qasm)


@pytest.mark.parametrize("qasm", INVALID_QASM)
def test_validate_qasm_invalid(qasm):
    """Test that invalid QASM strings raise a SyntaxError."""
    assert not validate_qasm(qasm)
