from functools import partial
from typing import Optional

import pennylane as qml
from pennylane.tape import QuantumScript
from pennylane.wires import Wires

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


def _ops_to_qasm(operations, precision, wires):
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
                params = (
                    "(" + ",".join([f"{p:.{precision}}" for p in op.parameters]) + ")"
                )
            else:
                # use default precision
                params = "(" + ",".join([str(p) for p in op.parameters]) + ")"

        qasm_str += f"{gate}{params} {wire_labels};\n"

    return qasm_str


def to_openqasm(
    qscript,
    wires: Optional[Wires] = None,
    measure_all: bool = True,
    precision: Optional[int] = None,
) -> str:
    """
    A modified version of PennyLane's function that is more compatible with having several measurements.
    Serialize the circuit as an OpenQASM 2.0 program.

    The measurement outputs can be restricted to only those specified in the script by
    setting ``measure_all=False``.

    .. note::

        The serialized OpenQASM program assumes that gate definitions
        in ``qelib1.inc`` are available.

    Args:
        wires (Wires or None): the wires to use when serializing the circuit
        measure_all (bool): whether to perform a computational basis measurement on all qubits
            or just those specified in the script
        precision (int): decimal digits to display for parameters

    Returns:
        str: OpenQASM serialization of the circuit
    """
    wires = wires or qscript.wires
    _to_qasm = partial(_ops_to_qasm, precision=precision, wires=wires)

    # add the QASM headers
    main_qasm_str = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'

    if qscript.num_wires == 0:
        # empty circuit
        return main_qasm_str

    # create the quantum and classical registers
    main_qasm_str += f"qreg q[{len(wires)}];\n"
    main_qasm_str += f"creg c[{len(wires)}];\n"

    # get the user applied circuit operations without interface information
    [transformed_tape], _ = qml.transforms.convert_to_numpy_parameters(qscript)
    operations = transformed_tape.operations

    # decompose the queue
    # pylint: disable=no-member
    just_ops = QuantumScript(operations)
    operations = just_ops.expand(
        depth=10, stop_at=lambda obj: obj.name in OPENQASM_GATES
    ).operations

    main_qasm_str += _to_qasm(operations)

    qasm_circuits = []

    # Create a copy of the program for every measurement that we have
    for meas in qscript.measurements:
        if diag_op := meas.diagonalizing_gates():
            diag_qasm_str = _to_qasm(
                QuantumScript(diag_op)
                .expand(depth=10, stop_at=lambda obj: obj.name in OPENQASM_GATES)
                .operations
            )
        else:
            diag_qasm_str = ""

        measure_qasm_str = ""
        if measure_all:
            for wire in range(len(wires)):
                measure_qasm_str += f"measure q[{wire}] -> c[{wire}];\n"
        else:
            measured_wires = Wires.all_wires([m.wires for m in qscript.measurements])

            for w in measured_wires:
                wire_indx = qscript.wires.index(w)
                measure_qasm_str += f"measure q[{wire_indx}] -> c[{wire_indx}];\n"

        qasm_circuits.append(main_qasm_str + diag_qasm_str + measure_qasm_str)

    return qasm_circuits
