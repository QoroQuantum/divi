# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import re
from copy import deepcopy
from itertools import product
from typing import Literal

import dill
import pennylane as qml
from pennylane.transforms.core.transform_program import TransformProgram

from divi.circuits.qasm import to_openqasm
from divi.circuits.qem import QEMProtocol

TRANSFORM_PROGRAM = TransformProgram()
TRANSFORM_PROGRAM.add_transform(qml.transforms.split_to_single_terms)
TRANSFORM_PROGRAM.add_transform(qml.transforms.split_non_commuting)


class Circuit:
    """
    Represents a quantum circuit with its QASM representation and metadata.

    This class encapsulates a PennyLane quantum circuit along with its OpenQASM
    serialization and associated tags for identification. Each circuit instance
    is assigned a unique ID for tracking purposes.

    Attributes:
        main_circuit: The PennyLane quantum circuit/tape object.
        tags (list[str]): List of string tags for circuit identification.
        qasm_circuits (list[str]): List of OpenQASM string representations.
        circuit_id (int): Unique identifier for this circuit instance.
    """

    _id_counter = 0

    def __init__(
        self,
        main_circuit,
        tags: list[str],
        qasm_circuits: list[str] = None,
    ):
        """
        Initialize a Circuit instance.

        Args:
            main_circuit: A PennyLane quantum circuit or tape object to be wrapped.
            tags (list[str]): List of string tags for identifying this circuit.
            qasm_circuits (list[str], optional): Pre-computed OpenQASM string
                representations. If None, they will be generated from main_circuit.
                Defaults to None.
        """
        self.main_circuit = main_circuit
        self.tags = tags

        self.qasm_circuits = qasm_circuits

        if self.qasm_circuits is None:
            self.qasm_circuits = to_openqasm(
                self.main_circuit,
                measurement_groups=[self.main_circuit.measurements],
                return_measurements_separately=False,
            )

        self.circuit_id = Circuit._id_counter
        Circuit._id_counter += 1

    def __str__(self):
        """
        Return a string representation of the circuit.

        Returns:
            str: String in format "Circuit: {circuit_id}".
        """
        return f"Circuit: {self.circuit_id}"


class MetaCircuit:
    """
    A parameterized quantum circuit template for batch circuit generation.

    MetaCircuit represents a symbolic quantum circuit that can be instantiated
    multiple times with different parameter values. It handles circuit compilation,
    observable grouping, and measurement decomposition for efficient execution.

    Attributes:
        main_circuit: The PennyLane quantum circuit with symbolic parameters.
        symbols: Array of sympy symbols used as circuit parameters.
        qem_protocol (QEMProtocol): Quantum error mitigation protocol to apply.
        compiled_circuits_bodies (list[str]): QASM bodies without measurements.
        measurements (list[str]): QASM measurement strings.
        measurement_groups (list[list]): Grouped observables for each circuit variant.
        postprocessing_fn: Function to combine measurement results.
    """

    def __init__(
        self,
        main_circuit,
        symbols,
        grouping_strategy: Literal["wires", "default", "qwc"] | None = None,
        qem_protocol: QEMProtocol | None = None,
    ):
        """
        Initialize a MetaCircuit with symbolic parameters.

        Args:
            main_circuit: A PennyLane quantum circuit/tape with symbolic parameters.
            symbols: Array of sympy Symbol objects representing circuit parameters.
            grouping_strategy (str, optional): Strategy for grouping commuting
                observables. Options are "wires", "default", or "qwc" (qubit-wise
                commuting). Defaults to None.
            qem_protocol (QEMProtocol, optional): Quantum error mitigation protocol
                to apply to the circuits. Defaults to None.
        """
        self.main_circuit = main_circuit
        self.symbols = symbols
        self.qem_protocol = qem_protocol

        transform_program = deepcopy(TRANSFORM_PROGRAM)
        transform_program[1].kwargs["grouping_strategy"] = grouping_strategy

        qscripts, self.postprocessing_fn = transform_program((main_circuit,))

        self.compiled_circuits_bodies, self.measurements = to_openqasm(
            main_circuit,
            measurement_groups=[qsc.measurements for qsc in qscripts],
            return_measurements_separately=True,
            # TODO: optimize later
            measure_all=True,
            symbols=self.symbols,
            qem_protocol=qem_protocol,
        )

        # Need to store the measurement groups for computing
        # expectation values later on, stripped of the `qml.expval` wrapper
        self.measurement_groups = [
            [meas.obs for meas in qsc.measurements] for qsc in qscripts
        ]

    def __getstate__(self):
        """
        Prepare the MetaCircuit for pickling.

        Serializes the postprocessing function using dill since regular pickle
        cannot handle certain PennyLane function objects.

        Returns:
            dict: State dictionary with serialized postprocessing function.
        """
        state = self.__dict__.copy()
        state["postprocessing_fn"] = dill.dumps(self.postprocessing_fn)
        return state

    def __setstate__(self, state):
        """
        Restore the MetaCircuit from a pickled state.

        Deserializes the postprocessing function that was serialized with dill
        during pickling.

        Args:
            state (dict): State dictionary from pickling with serialized
                postprocessing function.
        """
        state["postprocessing_fn"] = dill.loads(state["postprocessing_fn"])

        self.__dict__.update(state)

    def initialize_circuit_from_params(
        self, param_list, tag_prefix: str = "", precision: int = 8
    ) -> Circuit:
        """
        Instantiate a concrete Circuit by substituting symbolic parameters with values.

        Takes a list of parameter values and creates a fully instantiated Circuit
        by replacing all symbolic parameters in the QASM representations with their
        concrete numerical values.

        Args:
            param_list: Array of numerical parameter values to substitute for symbols.
                Must match the length and order of self.symbols.
            tag_prefix (str, optional): Prefix to prepend to circuit tags for
                identification. Defaults to "".
            precision (int, optional): Number of decimal places for parameter values
                in the QASM output. Defaults to 8.

        Returns:
            Circuit: A new Circuit instance with parameters substituted and proper
                tags for identification.

        Note:
            The main_circuit attribute in the returned Circuit still contains
            symbolic parameters. Only the QASM representations have concrete values.
        """
        mapping = dict(
            zip(
                map(lambda x: re.escape(str(x)), self.symbols),
                map(lambda x: f"{x:.{precision}f}", param_list),
            )
        )
        pattern = re.compile("|".join(k for k in mapping.keys()))

        final_qasm_strs = []
        for circuit_body in self.compiled_circuits_bodies:
            final_qasm_strs.append(
                pattern.sub(lambda match: mapping[match.group(0)], circuit_body)
            )

        tags = []
        qasm_circuits = []
        for (i, body_str), (j, meas_str) in product(
            enumerate(final_qasm_strs), enumerate(self.measurements)
        ):
            qasm_circuits.append(body_str + meas_str)

            nonempty_subtags = filter(
                None,
                [tag_prefix, f"{self.qem_protocol.name}:{i}", str(j)],
            )
            tags.append("_".join(nonempty_subtags))

        # Note: The main circuit's parameters are still in symbol form.
        # Not sure if it is necessary for any useful application to parameterize them.
        return Circuit(self.main_circuit, qasm_circuits=qasm_circuits, tags=tags)
