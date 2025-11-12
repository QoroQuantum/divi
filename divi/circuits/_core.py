# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import re
from dataclasses import dataclass, field
from itertools import product
from typing import Callable, Literal

import dill
import numpy as np
import pennylane as qml
from pennylane.transforms.core.transform_program import TransformProgram

from divi.circuits import to_openqasm
from divi.circuits.qem import QEMProtocol

TRANSFORM_PROGRAM = TransformProgram()
TRANSFORM_PROGRAM.add_transform(qml.transforms.split_to_single_terms)


def _wire_grouping(measurements: list[qml.measurements.MeasurementProcess]):
    """
    Groups a list of PennyLane MeasurementProcess objects by mutually non-overlapping wires.

    Each group contains measurements whose wires do not overlap with those of any other
    measurement in the same group. This enables parallel measurement of compatible observables,
    e.g., for grouped execution or more efficient sampling.

    Returns:
        partition_indices (list[list[int]]): Indices of the original measurements in each group.
        mp_groups (list[list[MeasurementProcess]]): Grouped MeasurementProcess objects.
    """
    mp_groups = []
    wires_for_each_group = []
    group_mapping = {}  # original_index -> (group_idx, pos_in_group)

    for i, mp in enumerate(measurements):
        added = False
        for group_idx, wires in enumerate(wires_for_each_group):
            if not qml.wires.Wires.shared_wires([wires, mp.wires]):
                mp_groups[group_idx].append(mp)
                wires_for_each_group[group_idx] += mp.wires
                group_mapping[i] = (group_idx, len(mp_groups[group_idx]) - 1)
                added = True
                break
        if not added:
            mp_groups.append([mp])
            wires_for_each_group.append(mp.wires)
            group_mapping[i] = (len(mp_groups) - 1, 0)

    partition_indices = [[] for _ in range(len(mp_groups))]
    for original_idx, (group_idx, _) in group_mapping.items():
        partition_indices[group_idx].append(original_idx)

    return partition_indices, mp_groups


def _create_final_postprocessing_fn(coefficients, partition_indices, num_total_obs):
    """Create a wrapper fn that reconstructs the flat results list and computes the final energy."""
    reverse_map = [None] * num_total_obs
    for group_idx, indices_in_group in enumerate(partition_indices):
        for idx_within_group, original_flat_idx in enumerate(indices_in_group):
            reverse_map[original_flat_idx] = (group_idx, idx_within_group)

    missing_indices = [i for i, v in enumerate(reverse_map) if v is None]
    if missing_indices:
        raise RuntimeError(
            f"partition_indices does not cover all observable indices. Missing indices: {missing_indices}"
        )

    def final_postprocessing_fn(grouped_results):
        """
        Takes grouped results, flattens them to the original order,
        multiplies by coefficients, and sums to get the final energy.
        """
        if len(grouped_results) != len(partition_indices):
            raise RuntimeError(
                f"Expected {len(partition_indices)} grouped results, but got {len(grouped_results)}."
            )
        flat_results = np.zeros(num_total_obs, dtype=np.float64)
        for original_flat_idx in range(num_total_obs):
            group_idx, idx_within_group = reverse_map[original_flat_idx]

            group_result = grouped_results[group_idx]
            # When a group has one measurement, the result is a scalar.
            if len(partition_indices[group_idx]) == 1:
                flat_results[original_flat_idx] = group_result
            else:
                flat_results[original_flat_idx] = group_result[idx_within_group]

        # Perform the final summation using the efficient dot product method.
        return np.dot(coefficients, flat_results)

    return final_postprocessing_fn


@dataclass(frozen=True)
class ExecutableQASMCircuit:
    """Represents a single, executable QASM circuit with its associated tag."""

    tag: str
    qasm: str


@dataclass(frozen=True)
class CircuitBundle:
    """
    Represents a bundle of logically related quantum circuits.

    A CircuitBundle is typically generated from a single `MetaCircuit` by
    instantiating it with concrete parameters. It may contain multiple
    executable circuits due to measurement grouping or error mitigation
    protocols. Each executable circuit has a QASM representation and a
    unique tag for identification.
    """

    executables: tuple[ExecutableQASMCircuit, ...]
    """Tuple of executable circuits."""

    def __str__(self):
        """
        Return a string representation of the circuit bundle.

        Returns:
            str: String in format "CircuitBundle ({num_executables} executables)".
        """
        return f"CircuitBundle ({len(self.executables)} executables)"

    @property
    def tags(self) -> list[str]:
        """A list of tags for all executables in the bundle."""
        return [e.tag for e in self.executables]

    @property
    def qasm_circuits(self) -> list[str]:
        """A list of QASM strings for all executables in the bundle."""
        return [e.qasm for e in self.executables]


@dataclass(frozen=True)
class MetaCircuit:
    """
    A parameterized quantum circuit template for batch circuit generation.

    MetaCircuit represents a symbolic quantum circuit that can be instantiated
    multiple times with different parameter values. It handles circuit compilation,
    observable grouping, and measurement decomposition for efficient execution.
    """

    source_circuit: qml.tape.QuantumScript
    """The PennyLane quantum circuit with symbolic parameters."""
    symbols: np.ndarray
    """Array of sympy symbols used as circuit parameters."""
    grouping_strategy: Literal["wires", "default", "qwc", "_backend_expval"] | None = (
        None
    )
    """Strategy for grouping commuting observables."""
    qem_protocol: QEMProtocol | None = None
    """Quantum error mitigation protocol to apply."""

    # --- Compiled artifacts ---
    _compiled_circuit_bodies: tuple[str, ...] = field(init=False)
    _measurements: tuple[str, ...] = field(init=False)
    measurement_groups: tuple[tuple[qml.operation.Operator, ...], ...] = field(
        init=False
    )
    postprocessing_fn: Callable = field(init=False)

    def __post_init__(self):
        """
        Compiles the circuit template after initialization.

        This method performs several steps:
        1. Decomposes the source circuit's measurement into single-term observables.
        2. Groups commuting observables according to the specified strategy.
        3. Generates a post-processing function to correctly combine measurement results.
        4. Compiles the circuit body and measurement instructions into QASM strings.
        """
        # Validate that the circuit has exactly one valid observable measurement.
        if len(self.source_circuit.measurements) != 1:
            raise ValueError(
                f"MetaCircuit requires a circuit with exactly one measurement, "
                f"but {len(self.source_circuit.measurements)} were found."
            )

        measurement = self.source_circuit.measurements[0]
        # If the measurement is not an expectation value, we assume it is for sampling
        # and does not require special post-processing.
        if not hasattr(measurement, "obs") or measurement.obs is None:
            postprocessing_fn = lambda x: x
            measurement_groups = ((),)
            (
                compiled_circuit_bodies,
                measurements,
            ) = to_openqasm(
                self.source_circuit,
                measurement_groups=measurement_groups,
                return_measurements_separately=True,
                symbols=self.symbols,
                qem_protocol=self.qem_protocol,
            )
            # Use object.__setattr__ because the class is frozen
            object.__setattr__(self, "postprocessing_fn", postprocessing_fn)
            object.__setattr__(self, "measurement_groups", measurement_groups)
            object.__setattr__(
                self, "_compiled_circuit_bodies", tuple(compiled_circuit_bodies)
            )
            object.__setattr__(self, "_measurements", tuple(measurements))

            return

        # Step 1: Use split_to_single_terms to get a flat list of measurement
        # processes. We no longer need its post-processing function.
        measurements_only_tape = qml.tape.QuantumScript(
            measurements=self.source_circuit.measurements
        )
        s_tapes, _ = TRANSFORM_PROGRAM((measurements_only_tape,))
        single_term_mps = s_tapes[0].measurements

        # Extract the coefficients, which we will now use in our own post-processing.
        obs = self.source_circuit.measurements[0].obs
        if isinstance(obs, (qml.Hamiltonian, qml.ops.Sum)):
            coeffs, _ = obs.terms()
        else:
            # For single observables, the coefficient is implicitly 1.0
            coeffs = [1.0]

        # Step 2: Manually group the flat list of measurements based on the strategy.
        if self.grouping_strategy in ("qwc", "default"):
            obs_list = [m.obs for m in single_term_mps]
            # This computes the grouping indices for the flat list of observables
            partition_indices = qml.pauli.compute_partition_indices(obs_list)
            measurement_groups = tuple(
                tuple(single_term_mps[i].obs for i in group)
                for group in partition_indices
            )
        elif self.grouping_strategy == "wires":
            partition_indices, grouped_mps = _wire_grouping(single_term_mps)
            measurement_groups = tuple(
                tuple(m.obs for m in group) for group in grouped_mps
            )
        elif self.grouping_strategy is None:
            # Each measurement is its own group
            measurement_groups = tuple(tuple([m.obs]) for m in single_term_mps)
            partition_indices = [[i] for i in range(len(single_term_mps))]
        elif self.grouping_strategy == "_backend_expval":
            measurement_groups = ((),)
            # For backends that compute expectation values directly, no explicit
            # measurement basis rotations (diagonalizing gates) are needed in the QASM.
            # The `to_openqasm` function interprets an empty measurement group `()`
            # as a signal to skip adding these gates.
            # All observables are still tracked in a single group for post-processing.
            partition_indices = [list(range(len(single_term_mps)))]
        else:
            raise ValueError(f"Unknown grouping strategy: {self.grouping_strategy}")

        # Step 3: Create our own post-processing function that handles the final summation.
        postprocessing_fn = _create_final_postprocessing_fn(
            coeffs, partition_indices, len(single_term_mps)
        )

        compiled_circuit_bodies, measurements = to_openqasm(
            self.source_circuit,
            measurement_groups=measurement_groups,
            return_measurements_separately=True,
            # TODO: optimize later
            measure_all=True,
            symbols=self.symbols,
            qem_protocol=self.qem_protocol,
        )
        # Use object.__setattr__ because the class is frozen
        object.__setattr__(self, "postprocessing_fn", postprocessing_fn)
        object.__setattr__(self, "measurement_groups", measurement_groups)
        object.__setattr__(
            self, "_compiled_circuit_bodies", tuple(compiled_circuit_bodies)
        )
        object.__setattr__(self, "_measurements", tuple(measurements))

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
    ) -> CircuitBundle:
        """
        Instantiate a concrete CircuitBundle by substituting symbolic parameters with values.

        Takes a list of parameter values and creates a fully instantiated CircuitBundle
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
            CircuitBundle: A new CircuitBundle instance with parameters substituted and proper
                tags for identification.

        Note:
            The main circuit's parameters are still in symbol form.
            Not sure if it is necessary for any useful application to parameterize them.
        """
        mapping = dict(
            zip(
                map(lambda x: re.escape(str(x)), self.symbols),
                map(lambda x: f"{x:.{precision}f}", param_list),
            )
        )
        pattern = re.compile("|".join(k for k in mapping.keys()))

        final_qasm_bodies = [
            pattern.sub(lambda match: mapping[match.group(0)], body)
            for body in self._compiled_circuit_bodies
        ]

        executables = []
        for (i, body_str), (j, meas_str) in product(
            enumerate(final_qasm_bodies), enumerate(self._measurements)
        ):
            qasm_circuit = body_str + meas_str
            tag_parts = [tag_prefix]
            if self.qem_protocol:
                tag_parts.append(f"{self.qem_protocol.name}:{i}")
            tag_parts.append(str(j))

            tag = "_".join(filter(None, tag_parts))
            executables.append(ExecutableQASMCircuit(tag=tag, qasm=qasm_circuit))

        return CircuitBundle(executables=tuple(executables))
