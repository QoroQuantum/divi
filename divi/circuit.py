from divi.qprog.utils import to_openqasm


class Circuit:
    _id_counter = 0

    def __init__(
        self,
        main_circuit,
        tag_prefix: str = "",
        circuit_type="pennylane",
        circuit_generator=None,
    ):
        self.main_circuit = main_circuit
        self.tag_prefix = tag_prefix
        self.circuit_type = circuit_type
        self.convert_to_qasm()

        self.circuit_id = Circuit._id_counter
        self.circuit_generator = circuit_generator
        Circuit._id_counter += 1

    def __str__(self):
        return f"Circuit: {self.circuit_id}"

    def convert_to_qasm(self):
        if self.circuit_type == "pennylane":
            return self._convert_pennylane_to_qasm()
        elif self.circuit_type == "qiskit":
            return self._convert_qiskit_to_qasm()
        else:
            raise ValueError(
                f"Invalid circuit type. Circuit type {self.circuit_type} not currently supported."
            )

    def _convert_pennylane_to_qasm(self):
        try:
            self.qasm_circuits = to_openqasm(self.main_circuit)

            processed_tag_prefix = (
                f"{self.tag_prefix}_" if len(self.tag_prefix) > 0 else ""
            )
            self.tags = [
                f"{processed_tag_prefix}{i}" for i in range(len(self.qasm_circuits))
            ]
            return
        except Exception as e:
            raise ValueError(f"Error converting Pennylane circuit to QASM: {e}")

    def _convert_qiskit_to_qasm(self):
        pass
