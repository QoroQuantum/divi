# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, replace

from qiskit.circuit import Parameter
from qiskit.dagcircuit import DAGCircuit
from qiskit.quantum_info import SparsePauliOp

from divi.typing import QASMTag


@dataclass(frozen=True)
class MetaCircuit:
    """Logical circuit IR.

    Stores one or more tagged :class:`~qiskit.dagcircuit.DAGCircuit`
    bodies together with the parameters that appear in them, optional
    observable / measured-wire metadata, and the (already-serialised)
    measurement QASM strings produced by
    :class:`~divi.pipeline.stages.MeasurementStage`.

    The DAGCircuit bodies are the long-lived working IR for all stages that
    mutate circuits at the gate level (QEM folding, Pauli twirling, QuEPP
    path enumeration).  QASM2 text is produced only once per parametric
    body — inside :class:`~divi.pipeline.stages.ParameterBindingStage` when
    it builds a :class:`~divi.circuits.QASMTemplate` — and once at
    compilation time when bound bodies are concatenated with the
    pre-serialised measurement QASMs.
    """

    circuit_bodies: tuple[tuple[QASMTag, DAGCircuit], ...]
    """Tagged parametric DAGs. Every body shares the same logical qubit layout."""

    parameters: tuple[Parameter, ...] = ()
    """Ordered Qiskit Parameter objects referenced by the DAGs.
    Order matches the flat parameter-values array fed by
    :class:`~divi.pipeline.stages.ParameterBindingStage`."""

    observable: SparsePauliOp | None = None
    """Observable for expectation-value measurements. ``None`` for probs/counts."""

    measured_wires: tuple[int, ...] | None = None
    """Qubit indices to measure for probs/counts measurements. ``None`` for expval."""

    measurement_qasms: tuple[tuple[QASMTag, str], ...] = ()
    """Pre-serialised, non-parametric OpenQASM 2.0 measurement strings
    (diagonalising gates + ``measure`` instructions), one per commuting
    observable group. Populated by ``MeasurementStage.set_measurement_bodies``."""

    bound_circuit_bodies: tuple[tuple[QASMTag, str], ...] = ()
    """Bound (parameter-free) OpenQASM 2.0 body strings produced by
    :class:`~divi.pipeline.stages.ParameterBindingStage`.  When non-empty,
    the pipeline's compilation pass consumes these instead of re-serialising
    ``circuit_bodies`` per submission."""

    measurement_groups: tuple[tuple[object, ...], ...] = ()
    """Cached grouped observables set by
    :class:`~divi.pipeline.stages.MeasurementStage`."""

    precision: int = 8
    """Number of decimal places for numeric parameter values in QASM conversion."""

    def __post_init__(self):
        """Minimal shape validation — caller owns correctness of the DAGs."""
        if not self.circuit_bodies:
            raise ValueError("MetaCircuit requires at least one circuit body.")

    @property
    def n_qubits(self) -> int:
        """Number of qubits in the circuit (from the first body DAG)."""
        _, dag = self.circuit_bodies[0]
        return dag.num_qubits()

    def set_circuit_bodies(
        self, bodies: tuple[tuple[QASMTag, DAGCircuit], ...]
    ) -> MetaCircuit:
        """Return a new MetaCircuit with updated circuit-body DAGs."""
        return replace(self, circuit_bodies=bodies)

    def set_measurement_bodies(
        self, bodies: tuple[tuple[QASMTag, str], ...]
    ) -> MetaCircuit:
        """Return a new MetaCircuit with updated measurement QASMs."""
        return replace(self, measurement_qasms=bodies)

    def set_bound_bodies(self, bodies: tuple[tuple[QASMTag, str], ...]) -> MetaCircuit:
        """Return a new MetaCircuit with updated bound body QASMs."""
        return replace(self, bound_circuit_bodies=bodies)

    def set_measurement_groups(
        self, measurement_groups: tuple[tuple[object, ...], ...]
    ) -> MetaCircuit:
        """Return a new MetaCircuit with updated measurement groups."""
        return replace(self, measurement_groups=measurement_groups)
