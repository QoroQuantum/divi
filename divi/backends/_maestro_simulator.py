# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import logging
import re
from collections.abc import Mapping

from qiskit import QuantumCircuit

try:
    import maestro
except ImportError:
    maestro = None

from divi.backends._circuit_runner import CircuitRunner
from divi.backends._execution_result import ExecutionResult

logger = logging.getLogger(__name__)


class MaestroSimulator(CircuitRunner):
    """A CircuitRunner backend powered by qoro-maestro, Qoro's C++ quantum simulator.

    Supports multiple simulation methods (Statevector, MPS, Stabilizer, TensorNetwork,
    PauliPropagator), intelligent auto-routing, GPU acceleration, and native observable
    estimation.

    Available on Linux and macOS only (not Windows).

    Args:
        shots: Number of measurement shots. Defaults to 5000.
        simulator_type: Maestro simulator type, e.g. ``"QCSim"``, ``"Gpu"``.
            ``None`` enables auto-routing.
        simulation_type: Simulation method, e.g. ``"Statevector"``, ``"MPS"``.
            ``None`` enables auto-selection.
        max_bond_dimension: Maximum bond dimension for MPS simulation.
        singular_value_threshold: SVD truncation threshold for MPS simulation.
        use_double_precision: Use double-precision floating point. Defaults to False.
        track_depth: Record circuit depth per submission. Defaults to False.
    """

    def __init__(
        self,
        shots: int = 5000,
        simulator_type: str | None = None,
        simulation_type: str | None = None,
        max_bond_dimension: int | None = None,
        singular_value_threshold: float | None = None,
        use_double_precision: bool = False,
        track_depth: bool = False,
    ):
        if maestro is None:
            raise ImportError(
                "qoro-maestro is required for MaestroSimulator but could not be imported. "
                "Install it with: pip install qoro-maestro  "
                "(available on Linux and macOS only, not Windows)."
            )

        super().__init__(shots=shots, track_depth=track_depth)

        self.simulator_type = simulator_type
        self.simulation_type = simulation_type
        self.max_bond_dimension = max_bond_dimension
        self.singular_value_threshold = singular_value_threshold
        self.use_double_precision = use_double_precision

    @property
    def supports_expval(self) -> bool:
        """Maestro supports native observable estimation."""
        return True

    @property
    def is_async(self) -> bool:
        """Maestro executes circuits synchronously."""
        return False

    def _build_config(self) -> dict:
        """Build kwargs dict from non-None configuration options."""
        config = {}

        if self.simulator_type is not None:
            config["simulator_type"] = maestro.SimulatorType[self.simulator_type]

        if self.simulation_type is not None:
            config["simulation_type"] = maestro.SimulationType[self.simulation_type]

        if self.max_bond_dimension is not None:
            config["max_bond_dimension"] = self.max_bond_dimension

        if self.singular_value_threshold is not None:
            config["singular_value_threshold"] = self.singular_value_threshold

        if self.use_double_precision:
            config["use_double_precision"] = True

        return config

    @staticmethod
    def _strip_measurements(qasm: str) -> str:
        """Remove measurement instructions from QASM.

        Measurement gates collapse the statevector, which corrupts
        expectation-value estimation.  They must be stripped before
        passing circuits to ``simple_estimate``.
        """
        return re.sub(r"measure\s+q\[\d+\]\s*->\s*c\[\d+\]\s*;\n?", "", qasm)

    def _get_ham_ops_for_circuit(
        self,
        circuit_index: int,
        ham_ops: str,
        circuit_ham_map: list[list[int]] | None,
    ) -> str:
        """Resolve which observable string applies to a given circuit index.

        Args:
            circuit_index: Index of the circuit in the batch.
            ham_ops: Semicolon-separated Pauli string, optionally with ``|``-delimited
                groups when ``circuit_ham_map`` is provided.
            circuit_ham_map: Each entry is ``[start, end)`` mapping a ``|``-group
                to a contiguous slice of circuits.  ``None`` means all circuits
                share the same observables.

        Returns:
            Semicolon-separated Pauli string for this circuit.
        """
        if circuit_ham_map is None:
            return ham_ops

        groups = ham_ops.split("|")
        for group_index, (start, end) in enumerate(circuit_ham_map):
            if start <= circuit_index < end:
                return groups[group_index]

        return ham_ops

    def submit_circuits(
        self,
        circuits: Mapping[str, str],
        ham_ops: str | None = None,
        circuit_ham_map: list[list[int]] | None = None,
        **kwargs,
    ) -> ExecutionResult:
        """Submit quantum circuits for execution on the maestro simulator.

        Args:
            circuits: Dictionary mapping circuit labels to OpenQASM string representations.
            ham_ops: Semicolon-separated Pauli string for expectation value estimation,
                e.g. ``"ZI;IZ;XX"``. If None, runs in sampling mode.
            circuit_ham_map: Maps circuit index ranges to observable groups for
                heterogeneous batches. Each inner list contains circuit indices
                belonging to that observable group.
            **kwargs: Additional parameters (unused, accepted for interface compatibility).

        Returns:
            ExecutionResult containing either counts (sampling) or expectation values.
        """
        circuit_labels = list(circuits.keys())
        qasm_strings = list(circuits.values())

        if self.track_depth:
            depths = [
                QuantumCircuit.from_qasm_str(qasm).depth() for qasm in qasm_strings
            ]
            self._depth_history.append(depths)

        config = self._build_config()
        results = []

        if ham_ops is None:
            # Sampling mode — reverse bitstrings from maestro's big-endian
            # (q[0] leftmost) to Qiskit's little-endian (q[0] rightmost).
            for label, qasm in zip(circuit_labels, qasm_strings):
                raw = maestro.simple_execute(qasm, shots=self.shots, **config)
                counts = {bs[::-1]: n for bs, n in raw["counts"].items()}
                results.append({"label": label, "results": counts})
        else:
            # Expectation value mode — strip measurement gates so they don't
            # collapse the statevector before expectation values are computed.
            for i, (label, qasm) in enumerate(zip(circuit_labels, qasm_strings)):
                pauli_string = self._get_ham_ops_for_circuit(
                    i, ham_ops, circuit_ham_map
                )
                raw = maestro.simple_estimate(
                    self._strip_measurements(qasm),
                    observables=pauli_string,
                    **config,
                )
                ops = pauli_string.split(";")
                expvals = dict(zip(ops, raw["expectation_values"]))
                results.append({"label": label, "results": expvals})

        return ExecutionResult(results=results)
