# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from qiskit.circuit import Parameter
from qiskit.dagcircuit import DAGCircuit
from qiskit.quantum_info import PauliList, SparsePauliOp

from divi.circuits import QASMTag
from divi.hamiltonians._term_ops import _assert_hermitian_spo


def flatten_observable_tuple(
    observable: tuple[SparsePauliOp, ...],
) -> tuple[SparsePauliOp, list[list[int]]]:
    """Flatten a tuple of observables into a single union ``SparsePauliOp``.

    Pauli terms that appear in multiple observables (or repeat within one)
    collapse to a single union slot. The union's coefficient on a slot is
    the sum of every owning observable's *absolute* coefficient on that
    Pauli (the weights used by shot allocation).

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

    slot_by_key: dict[bytes, int] = {}
    union_z_rows: list[np.ndarray] = []
    union_x_rows: list[np.ndarray] = []
    union_coeffs: list[float] = []
    per_obs_term_indices: list[list[int]] = []

    for obs in observable:
        x_arr = obs.paulis.x  # bool[N_terms, n_qubits]
        z_arr = obs.paulis.z
        coeffs = np.abs(obs.coeffs.real)
        indices: list[int] = []
        for i in range(x_arr.shape[0]):
            key = z_arr[i].tobytes() + x_arr[i].tobytes()
            slot = slot_by_key.get(key)
            if slot is None:
                slot = len(union_z_rows)
                slot_by_key[key] = slot
                union_z_rows.append(z_arr[i])
                union_x_rows.append(x_arr[i])
                union_coeffs.append(float(coeffs[i]))
            else:
                union_coeffs[slot] += float(coeffs[i])
            indices.append(slot)
        per_obs_term_indices.append(indices)

    if not union_z_rows:
        raise ValueError(
            "flatten_observable_tuple: every observable in the tuple is empty."
        )

    union_z = np.stack(union_z_rows)
    union_x = np.stack(union_x_rows)
    union = SparsePauliOp(
        PauliList.from_symplectic(union_z, union_x),
        coeffs=np.array(union_coeffs, dtype=complex),
    )
    return union, per_obs_term_indices


# Canonical decimal-precision for QASM parameter rendering, threaded
# through every MetaCircuit-producing path. Users override per-program
# via ``QuantumProgram(precision=...)``.
DEFAULT_PRECISION = 8


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

    template_circuit_bodies: tuple[tuple[QASMTag, str], ...] = ()
    """Parametric OpenQASM 2.0 body strings carrying named-symbol placeholders.

    Populated by :class:`~divi.pipeline.stages.ParameterBindingStage` when
    the active backend implements
    :class:`~divi.backends.SupportsCircuitTemplates` and no downstream
    stage requires bound DAGs. When non-empty, the pipeline's compilation
    pass emits a list of :class:`~divi.circuits.TemplateEntry` rows rather
    than fully bound circuits, deferring parameter substitution to the
    backend and drastically reducing wire payload for variational loops.
    """

    measurement_groups: tuple[tuple[object, ...], ...] = ()
    """Cached grouped observables set by
    :class:`~divi.pipeline.stages.MeasurementStage`."""

    precision: int = DEFAULT_PRECISION
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

        if self.observable is not None:
            for obs in self.observable:
                if not isinstance(obs, SparsePauliOp):
                    raise TypeError(
                        "MetaCircuit.observable must be a SparsePauliOp or a "
                        "tuple of SparsePauliOp instances."
                    )
                _assert_hermitian_spo(obs)

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

    def set_template_bodies(
        self, bodies: tuple[tuple[QASMTag, str], ...]
    ) -> MetaCircuit:
        """Return a new MetaCircuit with updated parametric template QASMs."""
        return replace(self, template_circuit_bodies=bodies)

    def set_measurement_groups(
        self, measurement_groups: tuple[tuple[object, ...], ...]
    ) -> MetaCircuit:
        """Return a new MetaCircuit with updated measurement groups."""
        return replace(self, measurement_groups=measurement_groups)
