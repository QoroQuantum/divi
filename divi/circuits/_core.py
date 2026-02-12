# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import product
from typing import Literal, NamedTuple

import dill
import numpy as np
import numpy.typing as npt
import pennylane as qml

from divi.circuits import to_openqasm
from divi.circuits._grouping import compute_measurement_groups
from divi.circuits.qem import QEMProtocol


class CircuitTag(NamedTuple):
    """Structured tag for identifying circuit executions."""

    param_id: int
    qem_name: str
    qem_id: int
    meas_id: int
    hamiltonian_id: int = 0
    """Hamiltonian sample index for multi-sample QDrift; 0 for single-sample (default)."""


def format_circuit_tag(tag: CircuitTag) -> str:
    """Format a CircuitTag into its wire-safe string representation."""
    return f"{tag.param_id}_{tag.qem_name}:{tag.qem_id}_ham:{tag.hamiltonian_id}_{tag.meas_id}"


@dataclass(frozen=True)
class ExecutableQASMCircuit:
    """Represents a single, executable QASM circuit with its associated tag."""

    tag: CircuitTag
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
    def tags(self) -> list[CircuitTag]:
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
    symbols: npt.NDArray[np.object_]
    """Array of sympy symbols used as circuit parameters."""
    grouping_strategy: Literal["wires", "default", "qwc", "_backend_expval"] | None = (
        None
    )
    """Strategy for grouping commuting observables."""
    qem_protocol: QEMProtocol | None = None
    """Quantum error mitigation protocol to apply."""
    precision: int = 8
    """Number of decimal places for parameter values in QASM conversion."""
    measurement_groups_override: (
        tuple[tuple[qml.operation.Operator, ...], ...] | None
    ) = None
    """Pre-computed measurement groups. When provided with postprocessing_fn_override, skips grouping."""
    postprocessing_fn_override: Callable | None = None
    """Pre-computed postprocessing function. When provided with measurement_groups_override, skips grouping."""

    # --- Compiled artifacts ---
    _compiled_circuit_bodies: tuple[str, ...] = field(init=False)
    _measurements: tuple[str, ...] = field(init=False)
    _measurement_groups: tuple[tuple[qml.operation.Operator, ...], ...] = field(
        init=False
    )
    _postprocessing_fn: Callable = field(init=False)

    def _set_compiled_state(
        self,
        postprocessing_fn: Callable,
        measurement_groups: tuple[tuple[qml.operation.Operator, ...], ...],
        compiled_circuit_bodies: tuple[str, ...] | list[str],
        measurements: tuple[str, ...] | list[str],
    ) -> None:
        # Use object.__setattr__ because the class is frozen
        object.__setattr__(self, "_postprocessing_fn", postprocessing_fn)
        object.__setattr__(self, "_measurement_groups", measurement_groups)
        object.__setattr__(
            self, "_compiled_circuit_bodies", tuple(compiled_circuit_bodies)
        )
        object.__setattr__(self, "_measurements", tuple(measurements))

    @property
    def postprocessing_fn(self) -> Callable:
        """Postprocessing function to combine grouped results."""
        return self._postprocessing_fn

    @property
    def measurement_groups(self) -> tuple[tuple[qml.operation.Operator, ...], ...]:
        """Measurement groups for circuit compilation."""
        return self._measurement_groups

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

        # When both overrides are provided, use them directly.
        if (
            self.measurement_groups_override is not None
            and self.postprocessing_fn_override is not None
        ):
            postprocessing_fn = self.postprocessing_fn_override
            measurement_groups = self.measurement_groups_override
        else:
            # Compute from measurement and grouping_strategy.
            measurement_groups, _, postprocessing_fn = compute_measurement_groups(
                measurement, self.grouping_strategy
            )

        compiled_circuit_bodies, measurements = self._compile_qasm(
            measurement_groups=measurement_groups,
            measure_all=True,
        )

        self._set_compiled_state(
            postprocessing_fn,
            measurement_groups,
            compiled_circuit_bodies,
            measurements,
        )

    def _compile_qasm(
        self,
        measurement_groups: tuple[tuple[qml.operation.Operator, ...], ...],
        *,
        measure_all: bool = False,
    ):
        return to_openqasm(
            self.source_circuit,
            measurement_groups=measurement_groups,
            return_measurements_separately=True,
            # TODO: optimize later
            measure_all=measure_all,
            symbols=self.symbols,
            qem_protocol=self.qem_protocol,
            precision=self.precision,
        )

    def __getstate__(self):
        """
        Prepare the MetaCircuit for pickling.

        Serializes the postprocessing function using dill since regular pickle
        cannot handle certain PennyLane function objects.

        Returns:
            dict: State dictionary with serialized postprocessing function.
        """
        state = self.__dict__.copy()
        state["postprocessing_fn"] = dill.dumps(self._postprocessing_fn)
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
        state["_postprocessing_fn"] = dill.loads(state["postprocessing_fn"])
        del state["postprocessing_fn"]

        self.__dict__.update(state)

    def initialize_circuit_from_params(
        self,
        param_list: npt.NDArray[np.floating] | list[float],
        param_idx: int = 0,
        precision: int | None = None,
        hamiltonian_id: int = 0,
    ) -> CircuitBundle:
        """
        Instantiate a concrete CircuitBundle by substituting symbolic parameters with values.

        Takes a list of parameter values and creates a fully instantiated CircuitBundle
        by replacing all symbolic parameters in the QASM representations with their
        concrete numerical values.

        Args:
            param_list (npt.NDArray[np.floating] | list[float]): Array of numerical
                parameter values to substitute for symbols.
                Must match the length and order of self.symbols.
            param_idx (int, optional): Parameter set index used for structured tags.
                Defaults to 0.
            precision (int | None, optional): Number of decimal places for parameter values
                in the QASM output. If None, uses the precision set on this MetaCircuit instance.
                Defaults to None.
            hamiltonian_id (int, optional): Hamiltonian sample index for multi-sample QDrift.
                Use 0 for single-sample (default).

        Returns:
            CircuitBundle: A new CircuitBundle instance with parameters substituted and proper
                tags for identification.

        Note:
            The main circuit's parameters are still in symbol form.
            Not sure if it is necessary for any useful application to parameterize them.
        """
        if precision is None:
            precision = self.precision

        # Parameter-free circuits: skip substitution when symbols is empty.
        if len(self.symbols) == 0:
            final_qasm_bodies = list(self._compiled_circuit_bodies)
        else:
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
        param_id = param_idx
        for (i, body_str), (j, meas_str) in product(
            enumerate(final_qasm_bodies), enumerate(self._measurements)
        ):
            qasm_circuit = body_str + meas_str
            tag = CircuitTag(
                param_id=param_id,
                qem_name=(
                    self.qem_protocol.name if self.qem_protocol else "NoMitigation"
                ),
                qem_id=i,
                hamiltonian_id=hamiltonian_id,
                meas_id=j,
            )
            executables.append(ExecutableQASMCircuit(tag=tag, qasm=qasm_circuit))

        return CircuitBundle(executables=tuple(executables))
