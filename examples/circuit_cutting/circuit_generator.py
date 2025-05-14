from qiskit import QuantumCircuit
from qiskit.circuit.library import EfficientSU2
from qiskit import qasm2
import numpy as np
import random

random.seed(32)


class CircuitGenerator:
    def __init__(self, num_qubits=4):
        self.num_qubits = num_qubits
        self.qc = QuantumCircuit(num_qubits)

    def ghz_state(self):
        """Create a GHZ state"""
        self.qc.h(0)
        for i in range(self.num_qubits - 1):
            self.qc.cx(i, i + 1)
        self.qc.measure_all()

        # Remove barriers from the circuit
        self.qc.data = [instr for instr in self.qc.data if instr[0].name != "barrier"]

        ghz_qasm_str = qasm2.dumps(self.qc)
        return ghz_qasm_str

    def w_state(self):
        """Create a W state"""

        for i in range(self.num_qubits - 1):
            theta = 2 * np.arccos(np.sqrt(1 / (self.num_qubits - i)))
            if i == 0:
                self.qc.ry(theta, 0)
            else:
                self.qc.cry(theta, i - 1, i)
        for i in reversed(range(1, self.num_qubits)):
            self.qc.cx(i - 1, i)

        self.qc.x(0)
        self.qc.measure_all()

        # Remove barriers from the circuit
        self.qc.data = [instr for instr in self.qc.data if instr[0].name != "barrier"]

        w_qasm_str = qasm2.dumps(self.qc)
        return w_qasm_str

    def hea_ansatz(self, depth=1, entanglement="linear"):
        """Create a Hardware Efficient Ansatz state"""
        self.qc = EfficientSU2(
            num_qubits=self.num_qubits, entanglement=entanglement, reps=depth
        )
        self.qc.measure_all()

        # parameter values s.t. -> clifford gates
        angles = [
            0,
            -np.pi / 2,
            np.pi / 2,
            -np.pi,
            np.pi,
            -(3 * np.pi) / 2,
            (3 * np.pi) / 2,
            -2 * np.pi,
            2 * np.pi,
        ]

        parameter_values = random.choices(angles, k=2 * self.num_qubits * (1 + depth))
        # parameter_values = [0] * (2 * self.num_qubits * (1 + depth))
        bound_circuit = self.qc.assign_parameters(parameter_values)
        self.qc = bound_circuit.decompose()

        # Remove barriers from the circuit
        self.qc.data = [instr for instr in self.qc.data if instr[0].name != "barrier"]

        cnot_gates = dict(self.qc.count_ops())["cx"]
        cnot_gates_per_layer = int(cnot_gates / depth)

        if cnot_gates_per_layer < self.num_qubits:
            hea_qasm_str = qasm2.dumps(self.qc)
            return hea_qasm_str
        else:
            print(f"Number of cuts not feasible")
            return None
