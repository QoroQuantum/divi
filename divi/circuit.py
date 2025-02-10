from qiskit.qasm2 import dumps

from divi.qprog.utils import to_openqasm


class Circuit:
    _id_counter = 0

    def __init__(
        self,
        main_circuit,
        tags: list[str],
        qasm_circuits: list[str] = None,
    ):
        self.main_circuit = main_circuit
        self.circuit_type = main_circuit.__module__.split(".")[0]
        self.tags = tags

        self.qasm_circuits = qasm_circuits

        if self.qasm_circuits is None:
            self.convert_to_qasm()

        self.circuit_id = Circuit._id_counter
        Circuit._id_counter += 1

    def __str__(self):
        return f"Circuit: {self.circuit_id}"

    def convert_to_qasm(self):
        if self.circuit_type == "pennylane":
            self.qasm_circuits = to_openqasm(self.main_circuit)

        elif self.circuit_type == "qiskit":
            self.qasm_circuits = [dumps(self.main_circuit)]

        else:
            raise ValueError(
                f"Invalid circuit type. Circuit type {self.circuit_type} not currently supported."
            )


class MetaCircuit:
    def __init__(self, main_circuit, symbols):
        self.main_circuit = main_circuit
        self.symbols = symbols

        self.compiled_circuit, self.measurements = to_openqasm(
            main_circuit, return_measurements_separately=True
        )

    def initialize_circuit_from_params(
        self, param_list, tag_prefix: str = "", precision: int = 8
    ) -> Circuit:
        final_qasm_str = self.compiled_circuit
        for param, symbol in zip(param_list, self.symbols):
            final_qasm_str = final_qasm_str.replace(
                str(symbol), f"{param:.{precision}f}"
            )

        processed_tag_prefix = f"{tag_prefix}_" if len(tag_prefix) > 0 else ""
        tags = []
        qasm_circuits = []
        for i, meas_str in enumerate(self.measurements):
            qasm_circuits.append(final_qasm_str + meas_str)
            tags.append(f"{processed_tag_prefix}{i}")

        return Circuit(self.main_circuit, qasm_circuits=qasm_circuits, tags=tags)
