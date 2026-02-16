# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from functools import partial
from itertools import product
from warnings import warn

import cirq
import numpy as np
import pennylane as qml
from pennylane.tape import QuantumScript
from pennylane.wires import Wires
from sympy import Expr

from ._cirq import ExtendedQasmParser as QasmParser

OPENQASM_GATES = {
    "CNOT": "cx",
    "CZ": "cz",
    "U3": "u3",
    "U2": "u2",
    "U1": "u1",
    "Identity": "id",
    "PauliX": "x",
    "PauliY": "y",
    "PauliZ": "z",
    "Hadamard": "h",
    "S": "s",
    "Adjoint(S)": "sdg",
    "T": "t",
    "Adjoint(T)": "tdg",
    "RX": "rx",
    "RY": "ry",
    "RZ": "rz",
    "CRX": "crx",
    "CRY": "cry",
    "CRZ": "crz",
    "SWAP": "swap",
    "Toffoli": "ccx",
    "CSWAP": "cswap",
    "PhaseShift": "u1",
}


def _cirq_circuit_from_qasm(qasm: str) -> cirq.Circuit:
    """Parses an OpenQASM string to `cirq.Circuit`.

    Args:
        qasm: The OpenQASM string

    Returns:
        The parsed circuit
    """

    return QasmParser().parse(qasm).circuit


def _ops_to_qasm(operations, precision, wires):
    """
    Convert PennyLane operations to OpenQASM instruction strings.

    Translates a sequence of PennyLane quantum operations into their OpenQASM
    2.0 equivalent representations. Each operation is mapped to its corresponding
    QASM gate with appropriate parameters and wire labels.

    Args:
        operations: Sequence of PennyLane operation objects to convert.
        precision (int | None): Number of decimal places for parameter values.
            If None, uses default Python string formatting.
        wires: Wire labels used in the circuit for indexing.

    Returns:
        str: OpenQASM instruction string with each operation on a new line.

    Raises:
        ValueError: If an operation is not supported by the QASM serializer.
    """
    # create the QASM code representing the operations
    qasm_str = ""

    for op in operations:
        try:
            gate = OPENQASM_GATES[op.name]
        except KeyError as e:
            raise ValueError(
                f"Operation {op.name} not supported by the QASM serializer"
            ) from e

        wire_labels = ",".join([f"q[{wires.index(w)}]" for w in op.wires.tolist()])
        params = ""

        if op.num_params > 0:
            # If the operation takes parameters, construct a string
            # with parameter values.
            if precision is not None:
                # Format parameters with precision, but use str() for sympy expressions
                param_strs = []
                for p in op.parameters:
                    if isinstance(p, Expr):
                        # Sympy expressions (Symbol, Mul, Add, etc.) should be kept as-is
                        # (will be replaced later during parameter substitution)
                        param_strs.append(str(p))
                    else:
                        # Numeric parameters can be formatted with precision
                        param_strs.append(f"{p:.{precision}}")
                params = "(" + ",".join(param_strs) + ")"
            else:
                # use default precision
                params = "(" + ",".join([str(p) for p in op.parameters]) + ")"

        qasm_str += f"{gate}{params} {wire_labels};\n"

    return qasm_str


def circuit_body_to_qasm(
    main_qscript,
    precision: int | None = None,
) -> str:
    """
    Convert the circuit body (operations only) to OpenQASM 2.0.

    Returns headers, qreg/creg, and gate operations. No measurement instructions.
    The body is stable until QEM (e.g. folding) modifies it.

    Args:
        main_qscript: The quantum circuit to convert.
        precision: Decimal digits for parameter values. None for default formatting.

    Returns:
        OpenQASM 2.0 string (headers + registers + operations).
    """
    main_qasm_str = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'

    if main_qscript.num_wires == 0:
        return main_qasm_str

    wires = main_qscript.wires
    main_qasm_str += f"qreg q[{len(wires)}];\ncreg c[{len(wires)}];\n"

    # Wrapping Sympy Symbols in a numpy object to bypass Pennylane's sanitization
    for op in main_qscript.operations:
        if qml.math.get_interface(*op.data) == "sympy":
            op.data = np.array(op.data)

    [transformed_tape], _ = qml.transforms.convert_to_numpy_parameters(main_qscript)
    operations = transformed_tape.operations

    just_ops = QuantumScript(operations)
    [decomposed_tape], _ = qml.transforms.decompose(
        just_ops, gate_set=lambda obj: obj.name in OPENQASM_GATES
    )

    _to_qasm = partial(_ops_to_qasm, precision=precision, wires=wires)
    main_qasm_str += _to_qasm(decomposed_tape.operations)

    return main_qasm_str


def measurements_to_qasm(
    main_qscript,
    measurement_groups: list[list[qml.measurements.ExpectationMP]],
    measure_all: bool = True,
    precision: int | None = None,
) -> list[str]:
    """
    Convert measurement groups to OpenQASM 2.0 measurement instructions.

    For each group: diagonalizing gates (if any) + measure instructions.

    Args:
        main_qscript: The quantum circuit (for wires) to convert.
        measurement_groups: List of commuting observable groups.
        measure_all: If True, measure all qubits; else only those in the group.
        precision: Decimal digits for parameter values. None for default.

    Returns:
        List of measurement QASM strings, one per group.
    """
    wires = main_qscript.wires
    _to_qasm = partial(_ops_to_qasm, precision=precision, wires=wires)
    measurement_qasms = []

    for meas_group in measurement_groups:
        wrapped_group = [
            m if isinstance(m, qml.measurements.MeasurementProcess) else qml.expval(m)
            for m in meas_group
        ]

        curr_diag_qasm_str = (
            _to_qasm(diag_ops)
            if (
                diag_ops := QuantumScript(
                    measurements=wrapped_group
                ).diagonalizing_gates
            )
            else ""
        )

        measure_qasm_str = ""
        if measure_all:
            for wire in range(len(wires)):
                measure_qasm_str += f"measure q[{wire}] -> c[{wire}];\n"
        else:
            measured_wires = Wires.all_wires([m.wires for m in meas_group])
            for w in measured_wires:
                wire_indx = main_qscript.wires.index(w)
                measure_qasm_str += f"measure q[{wire_indx}] -> c[{wire_indx}];\n"

        measurement_qasms.append(curr_diag_qasm_str + measure_qasm_str)

    return measurement_qasms


def to_openqasm(
    main_qscript,
    measurement_groups: list[list[qml.measurements.ExpectationMP]],
    measure_all: bool = True,
    precision: int | None = None,
    return_measurements_separately: bool = False,
) -> list[str] | list[tuple[str, str]] | tuple[list[str], list[str]]:
    """
    Serialize the circuit as an OpenQASM 2.0 program.

    A modified version of PennyLane's function that is more compatible with having
    several measurements and incorporates modifications introduced by splitting transforms,
    as well as error mitigation through folding.

    The measurement outputs can be restricted to only those specified in the script by
    setting ``measure_all=False``.

    .. note::

        The serialized OpenQASM program assumes that gate definitions
        in ``qelib1.inc`` are available.

    Args:
        main_qscript (QuantumScript): the quantum circuit to be converted, as a QuantumScript/QuantumTape object.
        measurement_groups (list[list]): A list of list of commuting observables, generated by the grouping Pennylane transformation.
        measure_all (bool): whether to perform a computational basis measurement on all qubits
            or just those specified in the script
        precision (int): decimal digits to display for parameters
        return_measurements_separately (bool): whether to not append the measurement instructions
            and their diagonalizations to the main circuit QASM code and return separately.

    Returns:
        list[str] or list[tuple[str, str]] or tuple[list[str], list[str]]: OpenQASM serialization of the circuit
    """
    body = circuit_body_to_qasm(main_qscript, precision=precision)

    if len(measurement_groups) == 0:
        warn(
            "No measurement groups provided. Returning the QASM of the circuit operations only."
        )
        # Empty circuit returns str; otherwise list of body strings
        if main_qscript.num_wires == 0:
            return body
        return [body]

    measurement_qasms = measurements_to_qasm(
        main_qscript,
        measurement_groups=measurement_groups,
        measure_all=measure_all,
        precision=precision,
    )

    if return_measurements_separately:
        return ([body], measurement_qasms)

    return list(product([body], measurement_qasms))
