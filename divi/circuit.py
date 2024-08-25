import qiskit
import qiskit.qasm2


class Circuit:
    _id_counter = 0

    def __init__(self, circuit, tag="", circuit_type="pennylane", circuit_generator=None):
        self.circuit = circuit
        self.tag = tag
        self.circuit_type = circuit_type
        self.qasm_circuit = self.convert_to_qasm()
        self.circuit_id = Circuit._id_counter
        self.circuit_generator = None
        Circuit._id_counter += 1

    def __str__(self):
        return f"Circuit: self.circuit_id"

    def convert_to_qasm(self):
        if self.circuit_type == "pennylane":
            return self._convert_pennylane_to_qasm()
        elif self.circuit_type == "qiskit":
            return self._convert_qiskit_to_qasm()
        else:
            raise ValueError(
                f"Invalid circuit type. Circuit type {self.circuit_type} not currently supported.")

    def _convert_pennylane_to_qasm(self):
        try:
            return qiskit.qasm2.dumps(self.circuit._circuit)
        except Exception as e:
            raise ValueError(f"Error converting Pennylane circuit to QASM: {e}")

    def _convert_qiskit_to_qasm(self):
        pass
