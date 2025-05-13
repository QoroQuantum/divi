from functools import partial, reduce
from typing import Optional
from warnings import warn

import numpy as np
import pennylane as qml
import scipy.sparse as sps
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
    main_qscript,
    measurement_groups: list[list[qml.measurements.ExpectationMP]],
    wires: Optional[Wires] = None,
    measure_all: bool = True,
    precision: Optional[int] = None,
    return_measurements_separately: bool = False,
) -> str | tuple[str, list[str]]:
    """
    A modified version of PennyLane's function that is more compatible with having
    several measurements and incorporates modifications introduced by splitting transforms.
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
        return_measurements_separately (bool): whether to not append the measurement instructions
            and their diagonalizations to the main circuit QASM code and return separately.

    Returns:
        str or tuple[str, list[str]]: OpenQASM serialization of the circuit
    """

    wires = wires or main_qscript.wires
    _to_qasm = partial(_ops_to_qasm, precision=precision, wires=wires)

    # add the QASM headers
    main_qasm_str = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'

    if main_qscript.num_wires == 0:
        # empty circuit
        return main_qasm_str
    # create the quantum and classical registers
    main_qasm_str += f"qreg q[{len(wires)}];\n"
    main_qasm_str += f"creg c[{len(wires)}];\n"

    # Wrapping Sympy Symbols in a numpy object to bypass
    # Pennylane's sanitization
    for op in main_qscript.operations:
        if qml.math.get_interface(*op.data) == "sympy":
            op.data = np.array(op.data)

    [transformed_tape], _ = qml.transforms.convert_to_numpy_parameters(main_qscript)
    operations = transformed_tape.operations

    # decompose the queue
    # pylint: disable=no-member
    just_ops = QuantumScript(operations)
    operations = just_ops.expand(
        depth=10, stop_at=lambda obj: obj.name in OPENQASM_GATES
    ).operations

    main_qasm_str += _to_qasm(operations)

    qasm_circuits = []
    measurement_qasms = []

    if len(measurement_groups) == 0:
        warn(
            "No measurement groups provided. Returning the QASM of the circuit operations only."
        )
        return [main_qasm_str]

    # Create a copy of the program for every measurement that we have
    for meas_group in measurement_groups:
        curr_diag_qasm_str = (
            _to_qasm(diag_ops)
            if (
                diag_ops := qml.tape.QuantumScript(
                    measurements=meas_group
                ).diagonalizing_gates
            )
            else ""
        )

        measure_qasm_str = ""
        if measure_all:
            for wire in range(len(wires)):
                measure_qasm_str += f"measure q[{wire}] -> c[{wire}];\n"
        else:
            measured_wires = Wires.all_wires(
                [m.wires for m in main_qscript.measurements]
            )

            for w in measured_wires:
                wire_indx = main_qscript.wires.index(w)
                measure_qasm_str += f"measure q[{wire_indx}] -> c[{wire_indx}];\n"

        if return_measurements_separately:
            measurement_qasms.append(curr_diag_qasm_str + measure_qasm_str)
        else:
            qasm_circuits.append(main_qasm_str + curr_diag_qasm_str + measure_qasm_str)

    return qasm_circuits or (main_qasm_str, measurement_qasms)


def _is_sanitized(
    qubo_matrix: np.ndarray | sps.spmatrix,
) -> np.ndarray | sps.spmatrix:
    # Sanitize the QUBO matrix to ensure it is either symmetric or upper triangular.

    is_sparse = sps.issparse(qubo_matrix)

    return (
        (
            ((qubo_matrix != qubo_matrix.T).nnz == 0)
            or ((qubo_matrix != sps.triu(qubo_matrix)).nnz == 0)
        )
        if is_sparse
        else (
            np.allclose(qubo_matrix, qubo_matrix.T)
            or np.allclose(qubo_matrix, np.triu(qubo_matrix))
        )
    )


def convert_qubo_matrix_to_pennylane_ising(
    qubo_matrix: np.ndarray | sps.spmatrix,
) -> tuple[qml.operation.Operator, float]:
    """Convert QUBO matrix to Ising Hamiltonian in Pennylane.

    The conversion follows the mapping:
    - QUBO variables x_i ∈ {0,1} map to Ising variables s_i ∈ {-1,1} via s_i = 2x_i - 1
    - This transforms a QUBO problem into an equivalent Ising problem

    Args:
        qubo_matrix: The QUBO matrix Q where the objective is to minimize x^T Q x

    Returns:
        A tuple of (Ising Hamiltonian as a PennyLane operator, constant term)
    """
    # Ensure the matrix is symmetric
    is_sparse = sps.issparse(qubo_matrix)
    backend = sps if is_sparse else np

    if not _is_sanitized(qubo_matrix):
        warn(
            "The QUBO matrix is neither symmetric nor upper triangular."
            " Symmetrizing it for the Ising Hamiltonian creation."
        )
        qubo_matrix = (qubo_matrix + qubo_matrix.T) / 2

    # Gather non-zero indices in the upper triangle of the matrix
    triu_matrix = backend.triu(
        qubo_matrix,
        **(
            {"format": qubo_matrix.format if qubo_matrix.format != "coo" else "csc"}
            if is_sparse
            else {}
        ),
    )
    rows, cols = triu_matrix.nonzero()
    values = triu_matrix[rows, cols].A1 if is_sparse else triu_matrix[rows, cols]

    n = qubo_matrix.shape[0]
    linear_terms = np.zeros(n)
    constant_term = 0.0
    ising_terms = []
    ising_weights = []

    for i, j, weight in zip(rows, cols, values):
        weight = float(weight)
        i, j = int(i), int(j)

        if i == j:
            # Diagonal elements
            linear_terms[i] -= weight / 2
            constant_term += weight / 2
        else:
            # Off-diagonal elements (i < j since we're using triu)
            ising_terms.append([i, j])
            ising_weights.append(weight / 4)

            # Update linear terms
            linear_terms[i] -= weight / 4
            linear_terms[j] -= weight / 4

            # Update constant term
            constant_term += weight / 4

    # Add the linear terms (Z operators)
    for i, curr_lin_term in filter(lambda x: x[1] != 0, enumerate(linear_terms)):
        ising_terms.append([i])
        ising_weights.append(float(curr_lin_term))

    # Construct the Ising Hamiltonian as a PennyLane operator
    pauli_string = qml.Identity(0) * 0
    for term, weight in zip(ising_terms, ising_weights):
        if len(term) == 1:
            # Single-qubit term (Z operator)
            curr_term = qml.Z(term[0]) * weight
        else:
            # Two-qubit term (ZZ interaction)
            curr_term = (
                reduce(lambda x, y: x @ y, map(lambda x: qml.Z(x), term)) * weight
            )

        pauli_string += curr_term

    return pauli_string.simplify(), constant_term
