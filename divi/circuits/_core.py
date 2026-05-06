# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from qiskit.circuit import Parameter
from qiskit.dagcircuit import DAGCircuit
from qiskit.quantum_info import SparsePauliOp

from divi.circuits._types import QASMTag


def flatten_observable_tuple(
    observable: tuple[SparsePauliOp, ...],
) -> tuple[SparsePauliOp, list[list[int]]]:
    """Flatten a tuple of observables into a single union ``SparsePauliOp``.

    Pauli labels that appear in multiple observables (or repeat within one)
    collapse to a single union slot keyed on their little-endian label.
    The union's coefficient on a slot is the sum of every owning
    observable's *absolute* coefficient on that Pauli.

    Args:
        observable: Non-empty tuple of Hermitian ``SparsePauliOp``.

    Returns:
        ``(union, per_obs_term_indices)``:

        * ``union`` — a ``SparsePauliOp`` whose Pauli terms are the unique
          Paulis across all observables, with absolute-summed coefficients.
        * ``per_obs_term_indices`` — for each observable in tuple-order, a
          list whose ``k``-th entry is the union slot index of that
          observable's ``k``-th Pauli term.
    """
    if not observable:
        raise ValueError("flatten_observable_tuple requires at least one observable.")

    label_to_slot: dict[str, int] = {}
    union_labels: list[str] = []
    union_coeffs_real: list[float] = []
    per_obs_term_indices: list[list[int]] = []

    for obs in observable:
        coeffs = np.abs(np.real(obs.coeffs)).astype(np.float64)
        indices: list[int] = []
        for label, c in zip(obs.paulis.to_labels(), coeffs):
            slot = label_to_slot.get(label)
            if slot is None:
                slot = len(union_labels)
                label_to_slot[label] = slot
                union_labels.append(label)
                union_coeffs_real.append(float(c))
            else:
                union_coeffs_real[slot] += float(c)
            indices.append(slot)
        per_obs_term_indices.append(indices)

    if not union_labels:
        raise ValueError(
            "flatten_observable_tuple: every observable in the tuple is empty."
        )

    union = SparsePauliOp.from_list(list(zip(union_labels, union_coeffs_real)))
    return union, per_obs_term_indices


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

    observable: SparsePauliOp | tuple[SparsePauliOp, ...] | None = None
    """Observable(s) for expectation-value measurements.

    * ``None`` — probs/counts measurement (uses :attr:`measured_wires` instead).
    * ``SparsePauliOp`` — accepted as input; ``__post_init__`` wraps it
      in a length-1 tuple.
    * ``tuple[SparsePauliOp, ...]`` — canonical stored form; one mitigated
      expectation value per entry."""

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

    _was_multi_obs: bool = False
    """Caller-set flag: ``True`` when the user explicitly opted into the
    multi-observable API (e.g. ``observable=[O]`` or
    ``observable=(O1, O2)``).  Drives result-shape squeeze policy at the
    :class:`~divi.pipeline.PipelineResult` boundary — ``False`` allows
    a length-1 expval list to be unwrapped to a scalar."""

    def __post_init__(self):
        """Minimal shape validation — caller owns correctness of the DAGs."""
        if not self.circuit_bodies:
            raise ValueError("MetaCircuit requires at least one circuit body.")

        # Wrap a bare SparsePauliOp in a 1-tuple to match the canonical shape.
        if isinstance(self.observable, SparsePauliOp):
            object.__setattr__(self, "observable", (self.observable,))

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
