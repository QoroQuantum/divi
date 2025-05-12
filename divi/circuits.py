from typing import Literal, Optional

import dill
import pennylane as qml
from pennylane.transforms.core.transform_program import TransformProgram
from qiskit.qasm2 import dumps

from divi.utils import to_openqasm

TRANSFORM_PROGRAM = TransformProgram()
TRANSFORM_PROGRAM.add_transform(qml.transforms.split_to_single_terms)
TRANSFORM_PROGRAM.add_transform(qml.transforms.split_non_commuting)


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
            self.qasm_circuits = to_openqasm(
                self.main_circuit,
                measurement_groups=[self.main_circuit.measurements],
                return_measurements_separately=False,
            )

        elif self.circuit_type == "qiskit":
            self.qasm_circuits = [dumps(self.main_circuit)]

        else:
            raise ValueError(
                f"Invalid circuit type. Circuit type {self.circuit_type} not currently supported."
            )


class MetaCircuit:
    def __init__(
        self,
        main_circuit,
        symbols,
        grouping_strategy: Optional[Literal["wires", "default", "qwc"]] = None,
    ):
        self.main_circuit = main_circuit
        self.symbols = symbols

        TRANSFORM_PROGRAM[1].kwargs["grouping_strategy"] = grouping_strategy

        qscripts, self.postprocessing_fn = TRANSFORM_PROGRAM((main_circuit,))

        self.compiled_circuit, self.measurements = to_openqasm(
            main_circuit,
            measurement_groups=[qsc.measurements for qsc in qscripts],
            return_measurements_separately=True,
        )

        # Need to store the measurement groups for computing
        # expectation values later on, stripped of the `qml.expval` wrapper
        self.measurement_groups = [
            [meas.obs for meas in qsc.measurements] for qsc in qscripts
        ]

    def __getstate__(self):
        state = self.__dict__.copy()
        state["postprocessing_fn"] = dill.dumps(self.postprocessing_fn)
        return state

    def __setstate__(self, state):
        state["postprocessing_fn"] = dill.loads(state["postprocessing_fn"])

        self.__dict__.update(state)

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

        # Note: The main circuit's parameters are still in symbol form.
        # Not sure if it is necessary for any useful application to parameterize them.
        return Circuit(self.main_circuit, qasm_circuits=qasm_circuits, tags=tags)
