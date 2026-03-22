# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import logging
import re
from collections.abc import Mapping

from qiskit import QuantumCircuit

try:
    import maestro

    _maestro_import_error = None
except ImportError as _err:
    maestro = None
    _maestro_import_error = _err

from divi.backends._circuit_runner import CircuitRunner
from divi.backends._execution_result import ExecutionResult

logger = logging.getLogger(__name__)


def _strip_id_gates(qasm: str) -> str:
    """Remove ``id`` (identity) gates from QASM.

    Maestro's QASM parser does not recognise the ``id`` gate.
    Since identity gates are no-ops, stripping them is safe.
    """
    return re.sub(r"id\s+q\[\d+\]\s*;\n?", "", qasm)


def _strip_measurements(qasm: str) -> str:
    """Remove measurement instructions from QASM.

    Measurement gates collapse the statevector, which corrupts
    expectation-value estimation.  They must be stripped before
    passing circuits to ``simple_estimate``.
    """
    return re.sub(r"measure\s+q\[\d+\]\s*->\s*c\[\d+\]\s*;\n?", "", qasm)


class MaestroSimulator(CircuitRunner):
    """A CircuitRunner backend powered by qoro-maestro, Qoro's C++ quantum simulator.

    Supports multiple simulation methods (Statevector, MPS, Stabilizer, TensorNetwork,
    PauliPropagator), intelligent auto-routing, GPU acceleration, and native observable
    estimation.

    When ``simulation_type`` is left as ``None``, the simulator automatically
    switches from Statevector to MPS for circuits exceeding
    ``mps_qubit_threshold`` qubits (default 22).

    Args:
        shots: Number of measurement shots. Defaults to 5000.
        simulator_type: Maestro simulator type, e.g. ``"QCSim"``, ``"Gpu"``.
            ``None`` enables auto-routing.
        simulation_type: Simulation method, e.g. ``"Statevector"``,
            ``"MatrixProductState"``. ``None`` enables automatic selection
            based on qubit count.
        max_bond_dimension: Maximum bond dimension for MPS simulation.
        singular_value_threshold: SVD truncation threshold for MPS simulation.
        use_double_precision: Use double-precision floating point. Defaults to False.
        track_depth: Record circuit depth per submission. Defaults to False.
        mps_qubit_threshold: Qubit count above which automatic MPS selection
            kicks in when ``simulation_type`` is ``None``. Defaults to 22.
    """

    _MPS_QUBIT_THRESHOLD_DEFAULT = 22
    _MPS_AUTO_BOND_DIMENSION = 64

    def __init__(
        self,
        shots: int = 5000,
        simulator_type: str | None = None,
        simulation_type: str | None = None,
        max_bond_dimension: int | None = None,
        singular_value_threshold: float | None = None,
        use_double_precision: bool = False,
        track_depth: bool = False,
        mps_qubit_threshold: int | None = None,
    ):
        if maestro is None:
            raise ImportError(
                "qoro-maestro is required for MaestroSimulator but could not be imported."
            ) from _maestro_import_error

        super().__init__(shots=shots, track_depth=track_depth)

        self.simulator_type = simulator_type
        self.simulation_type = simulation_type
        self.max_bond_dimension = max_bond_dimension
        self.singular_value_threshold = singular_value_threshold
        self.use_double_precision = use_double_precision
        self.mps_qubit_threshold = (
            mps_qubit_threshold
            if mps_qubit_threshold is not None
            else self._MPS_QUBIT_THRESHOLD_DEFAULT
        )

    @property
    def supports_expval(self) -> bool:
        """Maestro supports native observable estimation."""
        return True

    @property
    def is_async(self) -> bool:
        """Maestro executes circuits synchronously."""
        return False

    def _resolve_simulation_type(self, n_qubits: int) -> str | None:
        """Choose simulation type based on qubit count when not explicitly set.

        Returns the user's explicit choice if set, otherwise switches to MPS
        for circuits exceeding :pyattr:`mps_qubit_threshold`.
        """
        if self.simulation_type is not None:
            return self.simulation_type
        if n_qubits > self.mps_qubit_threshold:
            logger.info(
                "Circuit has %d qubits (> %d threshold), using MPS simulation.",
                n_qubits,
                self.mps_qubit_threshold,
            )
            return "MatrixProductState"
        return None

    def _build_config(self, n_qubits: int = 0) -> dict:
        """Build kwargs dict from non-None configuration options.

        Args:
            n_qubits: Maximum qubit count in the batch, used for automatic
                simulation type selection.
        """
        config = {}

        if self.simulator_type is not None:
            config["simulator_type"] = maestro.SimulatorType[self.simulator_type]

        auto_mps = False
        simulation_type = self._resolve_simulation_type(n_qubits)
        if simulation_type is not None:
            config["simulation_type"] = maestro.SimulationType[simulation_type]
            auto_mps = (
                self.simulation_type is None and simulation_type == "MatrixProductState"
            )

        if self.max_bond_dimension is not None:
            config["max_bond_dimension"] = self.max_bond_dimension
        elif auto_mps:
            config["max_bond_dimension"] = self._MPS_AUTO_BOND_DIMENSION

        if self.singular_value_threshold is not None:
            config["singular_value_threshold"] = self.singular_value_threshold

        if self.use_double_precision:
            config["use_double_precision"] = True

        return config

    def set_seed(self, seed: int) -> None:  # noqa: ARG002
        """No-op — maestro does not yet expose seeding from C++."""

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

        # Determine max qubit count for automatic simulation type selection.
        max_qubits = max(
            int(m.group(1))
            for q in qasm_strings
            if (m := re.search(r"qreg\s+q\[(\d+)\]", q))
        )
        config = self._build_config(n_qubits=max_qubits)
        results = []

        # Pre-process: strip id gates (not supported by maestro's QASM parser).
        qasm_strings = [_strip_id_gates(q) for q in qasm_strings]

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
                    _strip_measurements(qasm),
                    observables=pauli_string,
                    **config,
                )
                ops = pauli_string.split(";")
                expvals = dict(zip(ops, raw["expectation_values"]))
                results.append({"label": label, "results": expvals})

        return ExecutionResult(results=results)
