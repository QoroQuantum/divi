# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Payload helpers around :class:`~divi.circuits.CircuitPayload`."""

import pytest
from qiskit.circuit import Parameter

from divi.circuits import CircuitPayload
from divi.circuits._payloads import (
    as_payloads,
    bound_circuits,
    bound_payloads,
    is_bound,
)

QASM_BELL = (
    'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncreg c[2];\n'
    "h q[0];\ncx q[0],q[1];\nmeasure q[0] -> c[0];\nmeasure q[1] -> c[1];\n"
)
QASM_H = (
    'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\ncreg c[1];\n'
    "h q[0];\nmeasure q[0] -> c[0];\n"
)


def test_bound_payloads_makes_one_single_row_payload_per_circuit():
    payloads = bound_payloads({"a": QASM_H, "b": QASM_BELL})

    assert [payload.circuit for payload in payloads] == [QASM_H, QASM_BELL]
    assert [payload.parameter_sets for payload in payloads] == [
        (("a", ()),),
        (("b", ()),),
    ]
    assert all(payload.parameters == () for payload in payloads)


def test_bound_circuits_inverts_bound_payloads():
    circuits = {"a": QASM_H, "b": QASM_BELL}

    assert bound_circuits(bound_payloads(circuits)) == circuits


def test_is_bound_distinguishes_resolved_from_parametric():
    parametric = CircuitPayload(
        circuit=QASM_H,
        parameters=(Parameter("theta"),),
        parameter_sets=(("row", (0.5,)),),
    )

    assert is_bound(bound_payloads({"a": QASM_H})) is True
    assert is_bound([parametric]) is False


def test_bound_circuits_rejects_a_parametric_payload():
    """Every row of a parametric CircuitPayload shares one template, so flattening would
    silently map all labels to the same unresolved circuit."""
    parametric = CircuitPayload(
        circuit=QASM_H,
        parameters=(Parameter("theta"),),
        parameter_sets=(("row_0", (0.1,)), ("row_1", (0.2,))),
    )

    with pytest.raises(ValueError, match="still carries free parameters"):
        bound_circuits([parametric])


def test_bound_circuits_accepts_the_mapping_shorthand():
    """Backends need no separate normalization step, so the contract cannot
    be skipped by forgetting one."""
    circuits = {"a": QASM_H, "b": QASM_BELL}

    assert bound_circuits(circuits) == circuits


def test_as_payloads_converts_the_mapping_shorthand():
    payloads = as_payloads({"a": QASM_H})

    assert bound_circuits(payloads) == {"a": QASM_H}


def test_as_payloads_passes_a_payload_sequence_through_untouched():
    payloads = bound_payloads({"a": QASM_H})

    assert as_payloads(payloads) is payloads


@pytest.mark.parametrize("payload", [QASM_H, QASM_H.encode()], ids=["str", "bytes"])
def test_as_payloads_rejects_a_bare_circuit(payload):
    """A bare string is a Sequence, so it must be refused explicitly rather
    than silently iterated character by character."""
    with pytest.raises(TypeError, match="bare circuit string"):
        as_payloads(payload)


def test_backend_accepts_the_mapping_shorthand(default_test_simulator):
    """The documented one-off call submits without building payloads by hand."""
    result = default_test_simulator.submit_circuits({"bell": QASM_BELL})

    assert [row["label"] for row in result.results] == ["bell"]
    assert set(result.results[0]["results"]) <= {"00", "01", "10", "11"}
