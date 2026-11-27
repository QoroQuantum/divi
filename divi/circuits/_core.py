# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from fractions import Fraction
from functools import cached_property
from math import gcd, lcm

import numpy as np
from qiskit.circuit import Parameter, ParameterExpression
from qiskit.dagcircuit import DAGCircuit
from qiskit.quantum_info import PauliList, SparsePauliOp

from divi.circuits._types import QASMTag, ResultFormat
from divi.hamiltonians._term_ops import _assert_hermitian_spo


def flatten_observable_tuple(
    observable: Sequence[SparsePauliOp],
) -> tuple[SparsePauliOp, list[list[int]]]:
    """Flatten a tuple of observables into a single union ``SparsePauliOp``.

    Pauli terms that appear in multiple observables (or repeat within one)
    collapse to a single union slot, in order of first appearance. The union's
    coefficient on a slot is the sum of every owning observable's *absolute*
    coefficient on that Pauli (the weights used by shot allocation).

    Args:
        observable: Non-empty tuple of Hermitian ``SparsePauliOp`` of one width.

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

    z = np.concatenate([obs.paulis.z for obs in observable])
    x = np.concatenate([obs.paulis.x for obs in observable])
    if not len(z):
        raise ValueError(
            "flatten_observable_tuple: every observable in the tuple is empty."
        )

    packed = np.packbits(np.concatenate([z, x], axis=1), axis=1)
    keys = packed.view(np.dtype((np.void, packed.shape[1]))).ravel()
    _, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
    by_appearance = np.argsort(first)
    slots = np.argsort(by_appearance)[inverse]

    coeffs = np.bincount(
        slots,
        weights=np.concatenate([np.abs(obs.coeffs.real) for obs in observable]),
        minlength=len(first),
    )
    rows = first[by_appearance]
    union = SparsePauliOp(
        PauliList.from_symplectic(z[rows], x[rows]),
        coeffs=coeffs.astype(complex),
    )
    bounds = np.cumsum([len(obs) for obs in observable])[:-1]
    return union, [part.tolist() for part in np.split(slots, bounds)]


# Canonical precision for QASM parameter rendering, threaded
# through every MetaCircuit-producing path. Users override per-program
# via ``QuantumProgram(precision=...)``.
DEFAULT_PRECISION = 8


#: Frequency gaps each parametric gate contributes per unit of its angle.
_GATE_FREQUENCY_GAPS: dict[str, tuple[float, ...]] = {
    "rx": (1.0,),
    "ry": (1.0,),
    "rz": (1.0,),
    "rxx": (1.0,),
    "ryy": (1.0,),
    "rzz": (1.0,),
    "rzx": (1.0,),
    "p": (1.0,),
    "cp": (1.0,),
    "u": (1.0,),
    "u2": (1.0,),
    "crx": (0.5, 1.0),
    "cry": (0.5, 1.0),
    "crz": (0.5, 1.0),
}


def _combine_occurrence_frequencies(
    occurrences_per_body: Sequence[Sequence[Sequence[float]]],
) -> tuple[float, int] | None:
    """One parameter's ``(omega, order)`` from its gate occurrences in each body.

    ``omega`` divides every occurrence's frequencies; ``order`` is the largest
    body's sum of per-occurrence maxima, in units of ``omega``. A parameter
    that moves no gate gets ``(1, 1)``.
    """
    values = [
        frequency
        for body in occurrences_per_body
        for occurrence in body
        for frequency in occurrence
        if frequency > 1e-12
    ]
    if not values:
        return 1.0, 1
    omega = _compute_common_frequency(values)
    if omega is None:
        return None
    order = max(
        round(sum(max(occurrence) for occurrence in body) / omega)
        for body in occurrences_per_body
    )
    return omega, order


def _compute_common_frequency(values: Sequence[float]) -> float | None:
    """The largest frequency every value is an integer multiple of, or ``None``
    when their ratios need a denominator above 10 000."""
    scale = max(values)
    ratios = [float(value) / scale for value in values]
    rational = [Fraction(ratio).limit_denominator(10_000) for ratio in ratios]
    if any(
        not np.isclose(float(fraction), ratio, rtol=1e-10, atol=1e-12)
        for fraction, ratio in zip(rational, ratios)
    ):
        return None
    denominator = lcm(*(fraction.denominator for fraction in rational))
    numerators = (
        fraction.numerator * (denominator // fraction.denominator)
        for fraction in rational
    )
    return scale * gcd(*numerators) / denominator


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

    qasm_bodies: tuple[tuple[QASMTag, str], ...] = ()
    """OpenQASM 2.0 body strings at the current binding level.

    One slot carries a body through progressive binding by the parameter-binding
    stages: :class:`~divi.pipeline.stages.DataBindingStage` parks per-sample
    *partial* bodies (data substituted, weight placeholders kept), which
    :class:`~divi.pipeline.stages.ParameterBindingStage` consults by ``body_tag``
    prefix before falling back to serialising ``circuit_bodies`` via
    :func:`~divi.circuits.dag_to_qasm_body`; that stage then renders either
    *fully bound* bodies (no placeholders) or *template* bodies (placeholders
    preserved for backend substitution). The compilation pass reads this slot
    when non-empty, else serialises ``circuit_bodies`` on the fly.
    """

    measurement_groups: tuple[tuple[object, ...], ...] = ()
    """Cached grouped observables set by
    :class:`~divi.pipeline.stages.MeasurementStage`."""

    result_format: ResultFormat | None = None
    """Canonical format the raw backend results should be converted into.
    Set by a measurement stage; read by ``pipeline.run()``."""

    group_shots: Mapping[int, int] | None = None
    """Per-observable-group shot allocation (``group_index -> shots``) when a
    ``shot_distribution`` is active; ``None`` otherwise."""

    param_group_shots: Mapping[int, Mapping[int, int]] | None = None
    """Per-parameter-set observable-group shots for estimator-sample batches."""

    backend_ham_ops: str | None = None
    """Observable string handed to a backend that computes expectation values
    itself; ``None`` for counts-based paths."""

    precision: int = DEFAULT_PRECISION
    """Precision of numeric parameter values in QASM. Magnitudes of 0.1 and
    above render to this many decimal places and smaller nonzero ones to this
    many significant figures (exponent notation where needed), so tiny angles
    never round to zero."""

    _was_multi_obs: bool = False
    """Caller-set flag: ``True`` when the user explicitly opted into the
    multi-observable API (e.g. ``observable=[O]`` or
    ``observable=(O1, O2)``).  Drives result-shape squeeze policy at the
    :class:`~divi.pipeline.PipelineResult` boundary — ``False`` allows
    a length-1 expval list to be unwrapped to a scalar."""

    _validated_observable: Sequence[SparsePauliOp] | None = field(
        default=None, compare=False, repr=False
    )
    """The observable already checked, so copies that keep it skip the check."""

    def __post_init__(self):
        """Minimal shape validation — caller owns correctness of the DAGs."""
        if not self.circuit_bodies:
            raise ValueError("MetaCircuit requires at least one circuit body.")

        # Wrap a bare SparsePauliOp in a 1-tuple to match the canonical shape.
        if isinstance(self.observable, SparsePauliOp):
            object.__setattr__(self, "observable", (self.observable,))

        if self.observable is not None and (
            self.observable is not self._validated_observable
        ):
            for obs in self.observable:
                if not isinstance(obs, SparsePauliOp):
                    raise TypeError(
                        "MetaCircuit.observable must be a SparsePauliOp or a "
                        "tuple of SparsePauliOp instances."
                    )
                _assert_hermitian_spo(obs)
            object.__setattr__(self, "_validated_observable", self.observable)

    @property
    def n_qubits(self) -> int:
        """Number of qubits in the circuit (from the first body DAG)."""
        _, dag = self.circuit_bodies[0]
        return dag.num_qubits()

    @property
    def is_unmeasured(self) -> bool:
        """Whether every expectation value is an exact constant, so no circuit runs."""
        return (
            self.result_format is ResultFormat.EXPVALS
            and isinstance(self.observable, tuple)
            and len(self.observable) > 0
            and all(obs.size == 0 for obs in self.observable)
        )

    @cached_property
    def parameter_frequencies(self) -> tuple[tuple[float, int] | None, ...]:
        """Each parameter's ``(omega, order)`` across all bodies, derived from
        the gates it drives.

        A parameter that drives no gate gets ``(1, 1)``. An entry is ``None``
        for a gate outside the supported rotations, a non-linear or
        symbol-dependent angle, or incommensurate frequencies.
        """
        indices = {parameter: index for index, parameter in enumerate(self.parameters)}
        occurrences = [[[] for _ in self.parameters] for _ in self.circuit_bodies]
        unsupported: set[int] = set()
        for body, (_, dag) in zip(occurrences, self.circuit_bodies):
            for node in dag.op_nodes():
                gaps = _GATE_FREQUENCY_GAPS.get(node.op.name)
                for angle in node.op.params:
                    if not isinstance(angle, ParameterExpression):
                        continue
                    for symbol in angle.parameters:
                        index = indices.get(symbol)
                        if index is None:
                            continue
                        slope = angle.gradient(symbol)
                        if gaps is None or isinstance(slope, ParameterExpression):
                            unsupported.add(index)
                            continue
                        body[index].append([abs(float(slope)) * gap for gap in gaps])
        return tuple(
            None
            if index in unsupported
            else _combine_occurrence_frequencies([body[index] for body in occurrences])
            for index in range(len(self.parameters))
        )

    def set_circuit_bodies(
        self, bodies: Sequence[tuple[QASMTag, DAGCircuit]]
    ) -> MetaCircuit:
        """Return a new MetaCircuit with updated circuit-body DAGs."""
        return replace(self, circuit_bodies=bodies)

    def set_observable(
        self,
        observable: SparsePauliOp | tuple[SparsePauliOp, ...],
        *,
        validated: bool = False,
    ) -> MetaCircuit:
        """Return a new MetaCircuit measuring *observable* instead.

        Lets a stage refine what is measured downstream — for example by
        splitting a multi-term observable into its individual Pauli terms,
        so the measurement stage reports each separately. ``validated=True``
        skips the type and Hermitian checks, for an observable derived from
        one that passed them, such as a subset of its terms.
        """
        if isinstance(observable, SparsePauliOp):
            observable = (observable,)
        return replace(
            self,
            observable=observable,
            _validated_observable=observable if validated else None,
        )

    def set_measurement_bodies(
        self, bodies: Sequence[tuple[QASMTag, str]]
    ) -> MetaCircuit:
        """Return a new MetaCircuit with updated measurement QASMs."""
        return replace(self, measurement_qasms=bodies)

    def set_qasm_bodies(self, bodies: Sequence[tuple[QASMTag, str]]) -> MetaCircuit:
        """Return a new MetaCircuit with updated QASM body strings."""
        return replace(self, qasm_bodies=bodies)

    def set_measurement_groups(
        self, measurement_groups: Sequence[tuple[object, ...]]
    ) -> MetaCircuit:
        """Return a new MetaCircuit with updated measurement groups."""
        return replace(self, measurement_groups=measurement_groups)

    def set_result_format(self, result_format: ResultFormat) -> MetaCircuit:
        """Return a new MetaCircuit with the result format set."""
        return replace(self, result_format=result_format)

    def set_group_shots(self, group_shots: Mapping[int, int]) -> MetaCircuit:
        """Return a new MetaCircuit with per-group shot allocation set."""
        return replace(self, group_shots=group_shots)

    def set_param_group_shots(
        self, param_group_shots: Mapping[int, Mapping[int, int]]
    ) -> MetaCircuit:
        """Return a new MetaCircuit with per-parameter-set group allocations."""
        return replace(self, param_group_shots=param_group_shots)

    def set_backend_ham_ops(self, ham_ops: str) -> MetaCircuit:
        """Return a new MetaCircuit with the backend observable string set."""
        return replace(self, backend_ham_ops=ham_ops)
