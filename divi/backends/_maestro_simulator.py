# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass, fields

try:
    import maestro

    _maestro_import_error = None
except ImportError as _err:
    maestro = None
    _maestro_import_error = _err

from qiskit import QuantumCircuit

from divi.backends import CircuitRunner, ExecutionResult

logger = logging.getLogger(__name__)

_MPS_AUTO_BOND_DIMENSION = 64


def _strip_id_gates(qasm: str) -> str:
    """Remove ``id`` (identity) gates from QASM.

    Maestro's QASM parser does not recognize the ``id`` gate.
    Since identity gates are no-ops, stripping them is safe.
    """
    return re.sub(r"id\s+q\[\d+\]\s*;\n?", "", qasm)


def _strip_measurements(qasm: str) -> str:
    """Remove measurement instructions from QASM.

    Measurement gates collapse the statevector, which corrupts
    expectation-value estimation.  They must be stripped before
    passing circuits to ``simple_estimate``.
    """
    return re.sub(r"measure\s+q\[\d+\]\s*->\s*\w+\[\d+\]\s*;\n?", "", qasm)


@dataclass(frozen=True)
class MaestroConfig:
    """Configuration object for :class:`MaestroSimulator`.

    Each field maps directly to an identically-named field on
    ``maestro.SimulatorConfig``; see the `maestro Python bindings guide
    <https://qoroquantum.github.io/maestro/d7/d01/python_guide.html#py_config>`_
    for the underlying semantics of each knob.  :attr:`mps_qubit_threshold`
    is Divi-specific and drives automatic Statevector → MatrixProductState
    selection.

    ``simulator_type`` and ``simulation_type`` accept the string names of the
    corresponding maestro enum members, e.g. ``"QCSim"``, ``"Gpu"``,
    ``"Statevector"``, ``"MatrixProductState"``.  ``None`` means "use maestro's
    default".

    Every field is explicit — unknown options raise ``TypeError`` at
    construction time instead of being silently dropped by a ``**kwargs``
    passthrough, so upstream changes to ``maestro.SimulatorConfig`` surface
    as loud failures.
    """

    simulator_type: str | None = None
    """Maestro simulator type, e.g. ``"QCSim"`` or ``"Gpu"``.  ``None`` uses
    maestro's default (``"QCSim"``)."""

    simulation_type: str | None = None
    """Simulation method, e.g. ``"Statevector"`` or ``"MatrixProductState"``.
    ``None`` enables automatic selection based on qubit count."""

    max_bond_dimension: int | None = None
    """Maximum bond dimension for MPS simulation.  ``None`` uses maestro's
    default, except when auto-MPS is triggered (in which case 64 is used)."""

    singular_value_threshold: float | None = None
    """SVD truncation threshold for MPS simulation.  ``None`` uses maestro's
    default."""

    use_double_precision: bool = False
    """Use double-precision floating point."""

    disable_optimized_swapping: bool = False
    """Disable MPS swap-cost optimization."""

    lookahead_depth: int = -1
    """Lookahead depth for the MPS swap optimizer.  ``-1`` is maestro's default."""

    mps_measure_no_collapse: bool = True
    """If ``True``, use the non-collapsing MPS measurement algorithm; if
    ``False``, use the collapsing one."""

    mps_qubit_threshold: int = 22
    """Qubit count above which automatic MPS selection kicks in.  Only active
    when :attr:`simulation_type` is ``None``; has no effect when
    ``simulation_type`` is set explicitly.  Divi-specific; not forwarded to
    ``maestro.SimulatorConfig``."""

    def override(self, other: "MaestroConfig") -> "MaestroConfig":
        """Return a new config overriding fields with non-default values from ``other``.

        "Non-default" here means a field whose value differs from the class default.
        This keeps the override semantics consistent with
        :class:`~divi.backends.ExecutionConfig`.
        """
        defaults = {f.name: f.default for f in fields(MaestroConfig)}
        merged = {f.name: getattr(self, f.name) for f in fields(MaestroConfig)}

        for f in fields(MaestroConfig):
            other_value = getattr(other, f.name)
            if other_value != defaults[f.name]:
                merged[f.name] = other_value

        return MaestroConfig(**merged)

    def _resolve_simulation_type(self, n_qubits: int) -> str | None:
        """Choose simulation type based on qubit count when not explicitly set."""
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

    def _to_maestro_config(self, n_qubits: int) -> "maestro.SimulatorConfig":
        """Build a ``maestro.SimulatorConfig`` for a batch of ``n_qubits`` circuits.

        Internal — the per-submission ``n_qubits`` drives auto-MPS selection.
        """
        kwargs: dict = {}

        if self.simulator_type is not None:
            kwargs["simulator_type"] = maestro.SimulatorType[self.simulator_type]

        resolved_sim_type = self._resolve_simulation_type(n_qubits)
        auto_mps = (
            self.simulation_type is None and resolved_sim_type == "MatrixProductState"
        )
        if resolved_sim_type is not None:
            kwargs["simulation_type"] = maestro.SimulationType[resolved_sim_type]

        if self.max_bond_dimension is not None:
            kwargs["max_bond_dimension"] = self.max_bond_dimension
        elif auto_mps:
            kwargs["max_bond_dimension"] = _MPS_AUTO_BOND_DIMENSION

        if self.singular_value_threshold is not None:
            kwargs["singular_value_threshold"] = self.singular_value_threshold

        if self.use_double_precision:
            kwargs["use_double_precision"] = True

        if self.disable_optimized_swapping:
            kwargs["disable_optimized_swapping"] = True

        if self.lookahead_depth != -1:
            kwargs["lookahead_depth"] = self.lookahead_depth

        if not self.mps_measure_no_collapse:
            kwargs["mps_measure_no_collapse"] = False

        return maestro.SimulatorConfig(**kwargs)


class MaestroSimulator(CircuitRunner):
    """A CircuitRunner backend powered by qoro-maestro, Qoro's C++ quantum simulator.

    Supports multiple simulation methods (Statevector, MPS, Stabilizer, TensorNetwork,
    PauliPropagator), intelligent auto-routing, GPU acceleration, and native observable
    estimation.

    All maestro-level configuration is carried in a :class:`MaestroConfig` object
    rather than as loose keyword arguments, matching the
    :class:`~divi.backends.ExecutionConfig` / :class:`~divi.backends.QoroService`
    pattern.

    .. note::

        Maestro's C++ extension must be loaded before other C++ libraries
        (Qiskit, PennyLane) to avoid initialization order conflicts.  This
        is handled automatically by ``divi/__init__.py``.

    Args:
        shots: Number of measurement shots. Defaults to 5000.
        config: :class:`MaestroConfig` controlling simulator backend, simulation
            method, bond dimension, and related knobs.  Defaults to
            ``MaestroConfig()``.
        track_depth: Record circuit depth per submission. Defaults to False.
    """

    def __init__(
        self,
        shots: int = 5000,
        config: MaestroConfig | None = None,
        track_depth: bool = False,
    ):
        if maestro is None:
            raise ImportError(
                "qoro-maestro is required for MaestroSimulator but could not be imported."
            ) from _maestro_import_error

        super().__init__(shots=shots, track_depth=track_depth)
        self.config: MaestroConfig = config if config is not None else MaestroConfig()

    @property
    def supports_expval(self) -> bool:
        """Maestro supports native observable estimation."""
        return True

    @property
    def is_async(self) -> bool:
        """Maestro executes circuits synchronously."""
        return False

    def set_seed(self, seed: int) -> None:  # noqa: ARG002
        """No-op — maestro does not yet expose seeding from C++."""

    def _get_ham_ops_for_circuit(
        self,
        circuit_index: int,
        ham_ops: str,
        circuit_ham_map: list[list[int]] | None,
    ) -> str:
        """Resolve which observable string applies to a given circuit index."""
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
        **kwargs,  # noqa: ARG002 — accepted for CircuitRunner interface compatibility
    ) -> ExecutionResult:
        """Submit quantum circuits for execution on the maestro simulator.

        Args:
            circuits: Dictionary mapping circuit labels to OpenQASM string representations.
            ham_ops: Semicolon-separated Pauli string for expectation value estimation,
                e.g. ``"ZI;IZ;XX"``. If None, runs in sampling mode.
            circuit_ham_map: Maps circuit index ranges to observable groups for
                heterogeneous batches. Each inner list contains circuit indices
                belonging to that observable group.
            **kwargs: Ignored — accepted so callers using the generic
                :class:`~divi.backends.CircuitRunner` interface can forward
                unrelated options without breaking.

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

        # Pre-process: strip id gates (not supported by maestro's QASM parser).
        qasm_strings = [_strip_id_gates(q) for q in qasm_strings]

        sim_config = self.config._to_maestro_config(n_qubits=max_qubits)
        results = []

        if ham_ops is None:
            # Sampling mode — reverse bitstrings from maestro's big-endian
            # (q[0] leftmost) to Qiskit's little-endian (q[0] rightmost).
            for label, qasm in zip(circuit_labels, qasm_strings):
                raw = maestro.simple_execute(qasm, config=sim_config, shots=self.shots)
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
                    config=sim_config,
                )
                ops = pauli_string.split(";")
                expvals = dict(zip(ops, raw["expectation_values"]))
                results.append({"label": label, "results": expvals})

        return ExecutionResult(results=results)
