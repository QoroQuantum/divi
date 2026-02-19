# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tutorial: CustomVQA with QuantumScript and Qiskit inputs."""

import pennylane as qml
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

from divi.qprog import CustomVQA
from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer
from tutorials._backend import get_backend


def build_quantum_script() -> qml.tape.QuantumScript:
    """Create a parameterized QuantumScript with a single expval measurement."""
    ops = [
        qml.RX(0.0, wires=0),
        qml.RZ(0.0, wires=0),
    ]
    measurements = [qml.expval(qml.Z(0))]
    return qml.tape.QuantumScript(ops=ops, measurements=measurements)


def build_qiskit_quantum_circuit() -> QuantumCircuit:
    """Create a Qiskit QuantumCircuit with measurements."""
    theta = Parameter("theta")
    phi = Parameter("phi")
    qc = QuantumCircuit(1, 1)
    qc.rx(theta, 0)
    qc.rz(phi, 0)
    qc.measure(0, 0)
    return qc


if __name__ == "__main__":
    optimizer = ScipyOptimizer(method=ScipyMethod.COBYLA)

    # Example 1: PennyLane QuantumScript
    print("Example 1: PennyLane QuantumScript")
    qscript = build_quantum_script()
    program1 = CustomVQA(
        qscript=qscript,
        param_shape=(2,),
        optimizer=optimizer,
        max_iterations=5,
        backend=get_backend(),
    )
    program1.run(perform_final_computation=False)
    print(f"Best loss: {program1.best_loss:.4f}")
    print(f"Best params: {program1.best_params}")
    print(f"Total circuits: {program1.total_circuit_count}\n")

    # Example 2: Qiskit QuantumCircuit
    print("Example 2: Qiskit QuantumCircuit")
    qc = build_qiskit_quantum_circuit()
    program2 = CustomVQA(
        qscript=qc,
        param_shape=(2,),
        optimizer=optimizer,
        max_iterations=5,
        backend=get_backend(),
    )
    program2.run(perform_final_computation=False)
    print(f"Best loss: {program2.best_loss:.4f}")
    print(f"Best params: {program2.best_params}")
    print(f"Total circuits: {program2.total_circuit_count}")
