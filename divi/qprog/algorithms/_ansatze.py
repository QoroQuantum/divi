# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Built-in QAOA / VQE ansätze.

Every ``Ansatz.build`` creates and returns a :class:`~qiskit.circuit.QuantumCircuit`.
``UCCSDAnsatz`` sources its excitations and Hartree-Fock reference from
``qiskit_nature`` (the ``chem`` extra), remapping that library's blocked spin
ordering onto Divi's interleaved one. The remaining chemistry ansätze
(``HartreeFockAnsatz``, ``QCCAnsatz``) enumerate their own excitations and
Hartree-Fock reference against the interleaved ordering.
"""

import inspect
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from itertools import combinations, pairwise, product
from numbers import Integral
from typing import Any, Literal
from warnings import warn

import numpy as np
import numpy.typing as npt
from qiskit.circuit import Gate, QuantumCircuit
from qiskit.circuit.library import (
    RXGate,
    RXXGate,
    RYGate,
    RYYGate,
    RZGate,
    RZXGate,
    RZZGate,
    XXPlusYYGate,
)
from scipy.linalg import expm
from scipy.optimize import least_squares

from divi._optional import import_optional
from divi.hamiltonians._term_ops import _HALF_PI


def _require_trainable_params(n_params: int, ansatz_name: str) -> int:
    if n_params <= 0:
        raise ValueError(
            f"{ansatz_name} must define at least one trainable parameter. "
            "Parameter-free circuits are not supported."
        )
    return n_params


@dataclass(frozen=True)
class ElectronCounts:
    """Electron counts of the reference determinant a chemistry ansatz builds on.

    A :class:`~divi.qprog.algorithms.VQE` derives them from its
    :class:`~divi.qprog.problems.HamiltonianProblem` and passes them to the
    ansatz as ``reference``.

    Args:
        n_electrons: Total electron count.
        n_alpha: Spin-up electron count. Set together with ``n_beta``; with both
            unset, ``n_electrons`` splits evenly.
        n_beta: Spin-down electron count.
    """

    n_electrons: int
    n_alpha: int | None = None
    n_beta: int | None = None


def _require_reference(
    reference: ElectronCounts | None, ansatz_name: str
) -> ElectronCounts:
    """Reject a missing reference by name.

    A chemistry ansatz cannot prepare its reference state or enumerate
    excitations without electron counts, so reject the missing setting at the
    boundary that can name the ansatz which needs it.
    """
    if reference is None:
        raise ValueError(
            f"{ansatz_name} requires n_electrons: it builds excitations from a "
            "reference state, which needs the electron count. Set n_electrons "
            "on the program's HamiltonianProblem (MolecularProblem "
            "supplies it automatically)."
        )
    return reference


def _require_balanced_spin(reference: ElectronCounts, ansatz_name: str) -> None:
    """Reject a spin-polarised reference by name."""
    if reference.n_alpha != reference.n_beta or reference.n_electrons % 2:
        raise ValueError(
            f"{ansatz_name} prepares a closed-shell reference, so it cannot "
            f"honour n_electrons={reference.n_electrons}, "
            f"n_alpha={reference.n_alpha}, n_beta={reference.n_beta}. Use "
            "UCCSDAnsatz or LUCJAnsatz for a spin-polarised reference."
        )


class Ansatz(ABC):
    """Abstract base class for all VQE ansätze."""

    @property
    def name(self) -> str:
        """Returns the human-readable name of the ansatz."""
        return self.__class__.__name__

    def __repr__(self) -> str:
        options = ", ".join(
            f"{key}={value!r}"
            for key, value in vars(self).items()
            if not key.startswith("_")
        )
        return f"{type(self).__name__}({options})"

    @abstractmethod
    def n_params_per_layer(
        self, n_qubits: int, reference: ElectronCounts | None = None
    ) -> int:
        """Returns the number of parameters required by the ansatz for one layer.

        Args:
            n_qubits: Number of qubits in the circuit.
            reference: Electron counts of the reference determinant. Only the
                chemistry ansätze read it.
        """
        raise NotImplementedError

    def parameter_frequencies(
        self, n_qubits: int, reference: ElectronCounts | None = None
    ) -> Sequence[tuple[float, int]] | None:
        """One layer's per-parameter ``(omega, order)``, or ``None`` for ``(1, 1)``.

        The energy carries frequencies ``{omega, ..., order * omega}`` in that
        parameter, which sets its parameter-shift rule. Override when a gate's
        generator has more than two distinct eigenvalues, or when gates share a
        parameter; both raise the frequency content, and the default two-term
        rule then returns a wrong gradient. A superset is safe.
        """
        return None

    @abstractmethod
    def build(
        self,
        params,
        n_qubits: int,
        n_layers: int,
        reference: ElectronCounts | None = None,
    ) -> QuantumCircuit:
        """
        Builds the ansatz circuit and returns a list of operations.

        Args:
            params: Parameter array for the ansatz.
            n_qubits (int): Number of qubits in the circuit.
            n_layers (int): Number of ansatz layers.
            reference: Electron counts of the reference determinant. Only the
                chemistry ansätze read it.

        Returns:
            QuantumCircuit: The ansatz circuit on ``n_qubits`` qubits.
        """
        raise NotImplementedError


class AmplitudeSeedError(ValueError):
    """The coupled-cluster amplitudes cannot be mapped onto the ansatz's parameters."""


class AmplitudeSeededAnsatz(Ansatz):
    """An ansatz whose parameters correspond to coupled-cluster amplitudes, so a
    CCSD solution can seed them."""

    @abstractmethod
    def params_from_amplitudes(
        self,
        t1: tuple[np.ndarray, np.ndarray],
        t2: tuple[np.ndarray, np.ndarray, np.ndarray],
        n_qubits: int,
        reference: ElectronCounts,
    ) -> np.ndarray:
        """One layer's parameters realising unrestricted coupled-cluster amplitudes.

        The correspondence is positional, so the amplitudes must come from a
        solution in the orbitals the ansatz acts on. Later layers left at zero
        leave the state unchanged.

        Args:
            t1: ``(t1_alpha, t1_beta)`` in PySCF's UCCSD layout.
            t2: ``(t2_alpha_alpha, t2_alpha_beta, t2_beta_beta)`` in PySCF's UCCSD
                layout.
            n_qubits: Number of qubits in the circuit.
            reference: Electron counts of the reference determinant.

        Returns:
            The parameters of one layer.

        Raises:
            AmplitudeSeedError: If the amplitudes cannot be mapped onto this
                ansatz's configuration.
        """
        raise NotImplementedError


# --- Template Ansätze ---


def _gate_n_params(gate_cls: type[Gate]) -> int:
    """Number of free parameters a Qiskit ``Gate`` class takes — i.e. the count
    of required positional args of its ``__init__`` (Qiskit encodes rotation
    angles as required positionals; see e.g. :class:`RXGate`, :class:`UGate`).
    """
    return sum(
        1
        for name, p in inspect.signature(gate_cls.__init__).parameters.items()
        if name != "self"
        and p.default is inspect.Parameter.empty
        and p.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    )


def _gate_n_qubits(gate_cls: type[Gate]) -> int:
    """Qubit arity of a Qiskit ``Gate`` subclass via a zero-parameter probe."""
    probe = gate_cls(*([0.0] * _gate_n_params(gate_cls)))  # type: ignore[bad-argument-type]
    return probe.num_qubits


def _validate_gate_cls(
    cls,
    *,
    expected_qubits: int,
    role: str,
    example: str,
    expected_params: int | None = None,
) -> None:
    """Reject anything that is not a Qiskit ``Gate`` subclass of the right arity.

    If ``expected_params`` is provided, also reject gate classes whose
    ``__init__`` requires a different number of positional parameters.
    """
    if not (isinstance(cls, type) and issubclass(cls, Gate)):
        raise TypeError(
            f"{role} must be a Qiskit Gate subclass ({example}), got {cls!r}."
        )
    n_q = _gate_n_qubits(cls)
    if n_q != expected_qubits:
        raise ValueError(
            f"{role} must be a {expected_qubits}-qubit gate; "
            f"{cls.__name__} acts on {n_q} qubits."
        )
    if expected_params is not None:
        n_p = _gate_n_params(cls)
        if n_p != expected_params:
            raise ValueError(
                f"{role} must take {expected_params} parameters; "
                f"{cls.__name__} takes {n_p}."
            )


class GenericLayerAnsatz(Ansatz):
    """
    A flexible ansatz alternating single-qubit gates with optional entanglers.
    """

    _layout_fn: Callable[[int], Iterable[tuple[int, int]]]

    def __init__(
        self,
        gate_sequence: Sequence[type[Gate]],
        entangler: type[Gate] | None = None,
        entangling_layout: (
            Literal["linear", "brick", "circular", "all-to-all"]
            | Sequence[tuple[int, int]]
            | npt.NDArray[np.integer]
            | None
        ) = None,
    ):
        """
        Args:
            gate_sequence: Sequence of one-qubit Qiskit ``Gate`` subclasses
                (e.g., ``RYGate``, ``RZGate``).
            entangler: Two-qubit Qiskit ``Gate`` subclass (e.g., ``CXGate``,
                ``CZGate``). If None, no entanglement is applied.
            entangling_layout (str): Layout for entangling layer ("linear", "brick",
                "circular", "all-to-all", or a sequence of qubit pairs).
        """
        for cls in gate_sequence:
            _validate_gate_cls(
                cls,
                expected_qubits=1,
                role="gate_sequence entries",
                example="e.g. RYGate, RZGate",
            )
        if entangler is not None:
            _validate_gate_cls(
                entangler,
                expected_qubits=2,
                role="entangler",
                example="e.g. CXGate, CZGate",
                expected_params=0,
            )
        self.gate_sequence = list(gate_sequence)
        self._gate_param_counts = [_gate_n_params(g) for g in self.gate_sequence]
        self.entangler = entangler

        if isinstance(entangling_layout, np.ndarray):
            if entangling_layout.ndim != 2 or entangling_layout.shape[1] != 2:
                raise ValueError(
                    "entangling_layout must be 'linear', 'brick', 'circular', "
                    "'all-to-all', or a Sequence of tuples of integers."
                )
            entangling_layout = list(map(tuple, entangling_layout.tolist()))

        self.entangling_layout = entangling_layout
        if entangler is None and entangling_layout is not None:
            warn("`entangling_layout` provided but `entangler` is None.")
        match entangling_layout:
            case None | "linear":
                self.entangling_layout = "linear"
                self._layout_fn = lambda n_qubits: pairwise(range(n_qubits))
            case "brick":
                self._layout_fn = lambda n_qubits: [
                    (i, i + 1) for r in range(2) for i in range(r, n_qubits - 1, 2)
                ]
            case "circular":
                # On 2 qubits the wrap pair (1, 0) would repeat (0, 1).
                self._layout_fn = lambda n_qubits: zip(
                    range(n_qubits if n_qubits > 2 else n_qubits - 1),
                    [(i + 1) % n_qubits for i in range(n_qubits)],
                )
            case "all-to-all":
                self._layout_fn = lambda n_qubits: combinations(range(n_qubits), 2)
            case list() | tuple() as custom_layout:
                if not all(
                    isinstance(ent, tuple)
                    and len(ent) == 2
                    and all(
                        isinstance(qubit, Integral) and not isinstance(qubit, bool)
                        for qubit in ent
                    )
                    for ent in custom_layout
                ):
                    raise ValueError(
                        "entangling_layout must be 'linear', 'brick', 'circular', "
                        "'all-to-all', or a Sequence of tuples of integers."
                    )
                pairs = [(int(first), int(second)) for first, second in custom_layout]
                self._layout_fn = lambda _: list(pairs)
            case _:
                raise ValueError(
                    f"Unknown entangling_layout: {entangling_layout!r}. "
                    "Must be 'linear', 'brick', 'circular', 'all-to-all', or "
                    "a Sequence of (int, int) tuples."
                )

    def n_params_per_layer(
        self, n_qubits: int, reference: ElectronCounts | None = None
    ) -> int:
        """``sum(_gate_n_params(g) for g in gate_sequence) * n_qubits``."""
        per_qubit = sum(self._gate_param_counts)
        return _require_trainable_params(per_qubit * n_qubits, self.name)

    def build(
        self,
        params,
        n_qubits: int,
        n_layers: int,
        reference: ElectronCounts | None = None,
    ) -> QuantumCircuit:
        qc = QuantumCircuit(n_qubits)
        gate_param_counts = self._gate_param_counts
        per_qubit = sum(gate_param_counts)

        params = np.asarray(params, dtype=object).reshape(n_layers, n_qubits, per_qubit)
        layout = list(self._layout_fn(n_qubits))

        for layer_idx in range(n_layers):
            layer_params = params[layer_idx]
            for q, qubit_params in enumerate(layer_params):
                idx = 0
                for gate_cls, n_p in zip(self.gate_sequence, gate_param_counts):
                    args = list(qubit_params[idx : idx + n_p])
                    qc.append(gate_cls(*args), [q])
                    idx += n_p

            if self.entangler is not None:
                for wire_a, wire_b in layout:
                    qc.append(self.entangler(), [wire_a, wire_b])  # type: ignore[call-arg]

        return qc


_BRICKWORK_ENTANGLERS = (RXXGate, RYYGate, RZZGate, RZXGate)


class BrickworkAnsatz(Ansatz):
    """The brickwork circuit of Sciorilli et al. (2025), used with PCE.

    Each layer applies an ``RX`` rotation to every qubit followed by the
    entangler on the pairs ``(0, 1), (2, 3), ...``; an ``RY`` rotation to
    every qubit followed by the entangler on ``(1, 2), (3, 4), ...``; and an
    ``RZ`` rotation to every qubit followed by the entangler on the first
    pairs again. Every gate has its own angle, so a layer on ``n`` qubits
    with ``e`` first and ``o`` second pairs has ``3n + 2e + o`` parameters.

    Args:
        entangler: Two-qubit Pauli rotation between neighbours: ``RXXGate``
            (the default), ``RYYGate``, ``RZZGate`` or ``RZXGate``.

    Raises:
        TypeError: If ``entangler`` is not one of those gates.
    """

    def __init__(self, entangler: type[Gate] = RXXGate):
        if entangler not in _BRICKWORK_ENTANGLERS:
            names = ", ".join(gate.__name__ for gate in _BRICKWORK_ENTANGLERS)
            raise TypeError(f"entangler must be one of {names}, got {entangler!r}.")
        self.entangler = entangler

    @staticmethod
    def _bricks(n_qubits: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
        """The first and second brick sublayers' qubit pairs."""
        first = [(q, q + 1) for q in range(0, n_qubits - 1, 2)]
        second = [(q, q + 1) for q in range(1, n_qubits - 1, 2)]
        return first, second

    def n_params_per_layer(
        self, n_qubits: int, reference: ElectronCounts | None = None
    ) -> int:
        """``3 * n_qubits`` rotations plus one angle per brick."""
        first, second = self._bricks(n_qubits)
        return _require_trainable_params(
            3 * n_qubits + 2 * len(first) + len(second), self.name
        )

    def build(
        self,
        params,
        n_qubits: int,
        n_layers: int,
        reference: ElectronCounts | None = None,
    ) -> QuantumCircuit:
        first, second = self._bricks(n_qubits)
        sublayers = (
            (RXGate, first),
            (RYGate, second),
            (RZGate, first),
        )
        params = np.asarray(params, dtype=object).reshape(n_layers, -1)
        qc = QuantumCircuit(n_qubits)
        for layer_params in params:
            angles = iter(layer_params)
            for rotation, pairs in sublayers:
                for q in range(n_qubits):
                    qc.append(rotation(next(angles)), [q])
                for pair in pairs:
                    qc.append(self.entangler(next(angles)), list(pair))
        return qc


def _emit_rx(qc: QuantumCircuit, theta, q: int) -> None:
    qc.rx(theta, q)


def _emit_ry(qc: QuantumCircuit, theta, q: int) -> None:
    qc.ry(theta, q)


def _emit_rz(qc: QuantumCircuit, theta, q: int) -> None:
    qc.rz(theta, q)


_QAOA_LOCAL_FIELDS: Mapping[
    type[Gate], Callable[[QuantumCircuit, object, int], None]
] = {
    RXGate: _emit_rx,
    RYGate: _emit_ry,
    RZGate: _emit_rz,
}


class QAOAAnsatz(Ansatz):
    """QAOA-style ansatz inspired by Killoran et al. (2020).

    Each of the ``L`` layers consists of a Hadamard encoding layer followed
    by a weight Hamiltonian:

    * for ``n_qubits == 1`` — a single local-field rotation;
    * for ``n_qubits == 2`` — one ``RZZ`` on the pair, then one local field
      per qubit (no wrap-around);
    * for ``n_qubits >= 3`` — ``RZZ`` gates on a closed ring (``i ↔ (i+1) %
      n``), then one local field per qubit.

    A trailing Hadamard layer is applied after the ``L``-th weight
    Hamiltonian. The default local field is ``RYGate``.

    Args:
        local_field: Single-qubit rotation used as the local field. Must be
            one of ``RXGate``, ``RYGate``, ``RZGate``. Defaults to ``RYGate``.
    """

    def __init__(self, local_field: type[Gate] = RYGate) -> None:
        if local_field not in _QAOA_LOCAL_FIELDS:
            raise ValueError(
                f"local_field must be one of RXGate, RYGate, RZGate; "
                f"got {local_field!r}."
            )
        self.local_field = local_field
        self._emit_local_field = _QAOA_LOCAL_FIELDS[local_field]

    def n_params_per_layer(
        self, n_qubits: int, reference: ElectronCounts | None = None
    ) -> int:
        """Per-layer parameter count.

        * ``n_qubits == 1`` → ``1`` (single local-field rotation)
        * ``n_qubits == 2`` → ``3`` (``RZZ`` + one local field per qubit)
        * ``n_qubits >= 3`` → ``2 * n_qubits`` (ring of ``RZZ`` + per-qubit local field)
        """
        if n_qubits == 1:
            n_params = 1
        elif n_qubits == 2:
            n_params = 3
        else:
            n_params = 2 * n_qubits
        return _require_trainable_params(n_params, QAOAAnsatz.__name__)

    def build(
        self,
        params,
        n_qubits: int,
        n_layers: int,
        reference: ElectronCounts | None = None,
    ) -> QuantumCircuit:
        """Build the QAOA ansatz circuit.

        Args:
            params: Flat parameter array of length
                ``n_layers * n_params_per_layer(n_qubits)``.
            n_qubits: Number of qubits.
            n_layers: Number of QAOA layers.
            reference: Unused.

        Returns:
            QuantumCircuit: Qiskit circuit implementing the QAOA ansatz.
        """
        per_layer = self.n_params_per_layer(n_qubits)
        layered = np.asarray(params, dtype=object).reshape(n_layers, per_layer)

        qc = QuantumCircuit(n_qubits)
        for layer in range(n_layers):
            # Encoding Hamiltonian: Hadamard on every qubit.
            for q in range(n_qubits):
                qc.h(q)
            # Weight Hamiltonian.
            weights = layered[layer]
            if n_qubits == 1:
                self._emit_local_field(qc, weights[0], 0)
            elif n_qubits == 2:
                _emit_two_qubit_pauli_rot(qc, "ZZ", weights[0], 0, 1)
                self._emit_local_field(qc, weights[1], 0)
                self._emit_local_field(qc, weights[2], 1)
            else:
                for q in range(n_qubits):
                    _emit_two_qubit_pauli_rot(
                        qc, "ZZ", weights[q], q, (q + 1) % n_qubits
                    )
                for q in range(n_qubits):
                    self._emit_local_field(qc, weights[n_qubits + q], q)

        # Trailing encoding layer.
        for q in range(n_qubits):
            qc.h(q)

        return qc


# --- Chemistry Ansätze ---


def _resolve_spin_counts(
    n_qubits: int,
    n_electrons: int,
    n_alpha: int | None,
    n_beta: int | None,
    ansatz_name: str,
) -> tuple[int, tuple[int, int]]:
    """Resolve ``(n_spatial_orbitals, (n_alpha, n_beta))`` for a reference state.

    With neither spin count given, the closed-shell split
    ``n_alpha = n_beta = n_electrons // 2`` is used, which requires an even
    ``n_electrons``.
    """
    if n_qubits % 2:
        raise ValueError(
            f"{ansatz_name} needs an even qubit count (two spin-orbitals per "
            f"spatial orbital); got n_qubits={n_qubits}."
        )
    n_spatial = n_qubits // 2

    if (n_alpha is None) != (n_beta is None):
        raise ValueError(
            f"{ansatz_name} needs n_alpha and n_beta together; got "
            f"n_alpha={n_alpha}, n_beta={n_beta}."
        )

    if n_alpha is None or n_beta is None:
        if n_electrons % 2:
            raise ValueError(
                f"{ansatz_name} cannot split {n_electrons} electrons into equal "
                "alpha/beta counts. Pass n_alpha and n_beta explicitly for a "
                "spin-imbalanced reference."
            )
        n_alpha = n_beta = n_electrons // 2
    elif n_alpha + n_beta != n_electrons:
        raise ValueError(
            f"{ansatz_name} got n_alpha={n_alpha} and n_beta={n_beta}, which sum "
            f"to {n_alpha + n_beta}, not n_electrons={n_electrons}."
        )

    for name, count in (("n_alpha", n_alpha), ("n_beta", n_beta)):
        if not 0 <= count <= n_spatial:
            raise ValueError(
                f"{ansatz_name} got {name}={count}, outside the range "
                f"[0, {n_spatial}] set by {n_qubits} qubits."
            )
    return n_spatial, (n_alpha, n_beta)


@cache
def _uccsd_template(
    n_spatial: int, n_particles: tuple[int, int]
) -> tuple[QuantumCircuit, QuantumCircuit, tuple[tuple[tuple[int, ...], ...], ...]]:
    """``(reference_state, ansatz, excitations)`` from qiskit-nature, in
    interleaved spin order.

    Both circuits are built eagerly and copied into plain ``QuantumCircuit``\\ s:
    qiskit-nature's ``BlueprintCircuit`` marks itself built before it finishes
    appending gates, so one left unbuilt can be read half-constructed from
    another worker thread. The results are cached and shared, so callers must
    compose or bind without mutating them.
    """
    mappers = import_optional(
        "qiskit_nature.second_q.mappers", extra="chem", capability="UCCSDAnsatz"
    )
    library = import_optional(
        "qiskit_nature.second_q.circuit.library", extra="chem", capability="UCCSDAnsatz"
    )

    # Interleaving must happen in the mapper: Jordan-Wigner parity strings are
    # built over the mapper's mode order, so permuting qubits afterwards leaves
    # each excitation's Z-string covering the wrong modes.
    mapper = mappers.InterleavedQubitMapper(mappers.JordanWignerMapper())

    blueprint = library.UCCSD(n_spatial, n_particles, mapper)
    blueprint.num_parameters  # force the lazy build
    excitations = tuple(blueprint.excitation_list or ())

    ansatz = QuantumCircuit(2 * n_spatial)
    ansatz.compose(blueprint, inplace=True)

    reference = QuantumCircuit(2 * n_spatial)
    reference.compose(library.HartreeFock(n_spatial, n_particles, mapper), inplace=True)
    return reference, ansatz, excitations


def uccsd_excitations(
    n_spatial: int, n_particles: tuple[int, int]
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """UCCSD excitation list, as ``(occupied, unoccupied)`` blocked indices.

    Positionally aligned with the ansatz's parameter vector.
    """
    return _uccsd_template(n_spatial, n_particles)[2]


class UCCSDAnsatz(AmplitudeSeededAnsatz):
    """
    Unitary Coupled Cluster Singles and Doubles (UCCSD) ansatz.

    This ansatz is specifically designed for quantum chemistry calculations,
    implementing the UCCSD approximation which includes all single and double
    electron excitations from a reference state.

    Excitations and the Hartree-Fock reference come from ``qiskit_nature``,
    which requires the ``chem`` extra. Spin-imbalanced references are
    supported through the ``n_alpha`` and ``n_beta`` of the
    :class:`ElectronCounts`; without them the closed-shell split
    ``n_alpha = n_beta = n_electrons // 2`` is used.
    """

    def n_params_per_layer(
        self, n_qubits: int, reference: ElectronCounts | None = None
    ) -> int:
        """Number of UCCSD excitation amplitudes for the given reference.

        Requires ``reference``.
        """
        counts = _require_reference(reference, "UCCSDAnsatz")
        n_spatial, n_particles = _resolve_spin_counts(
            n_qubits,
            counts.n_electrons,
            counts.n_alpha,
            counts.n_beta,
            "UCCSDAnsatz",
        )
        _, template, _ = _uccsd_template(n_spatial, n_particles)
        return _require_trainable_params(template.num_parameters, UCCSDAnsatz.__name__)

    def parameter_frequencies(
        self, n_qubits: int, reference: ElectronCounts | None = None
    ):
        """``{1, 2}`` per amplitude: each excitation exponentiates a generator
        whose eigenvalues span two distinct gaps."""
        return [(1.0, 2)] * self.n_params_per_layer(n_qubits, reference)

    def build(
        self,
        params,
        n_qubits: int,
        n_layers: int,
        reference: ElectronCounts | None = None,
    ) -> QuantumCircuit:
        counts = _require_reference(reference, "UCCSDAnsatz")
        n_spatial, n_particles = _resolve_spin_counts(
            n_qubits,
            counts.n_electrons,
            counts.n_alpha,
            counts.n_beta,
            "UCCSDAnsatz",
        )
        hartree_fock, template, _ = _uccsd_template(n_spatial, n_particles)
        params = np.asarray(params, dtype=object).reshape(n_layers, -1)

        qc = QuantumCircuit(n_qubits)
        qc.compose(hartree_fock, inplace=True)
        for layer in params:
            qc.compose(
                template.assign_parameters(list(layer), inplace=False), inplace=True
            )
        return qc

    def params_from_amplitudes(
        self,
        t1: tuple[np.ndarray, np.ndarray],
        t2: tuple[np.ndarray, np.ndarray, np.ndarray],
        n_qubits: int,
        reference: ElectronCounts,
    ) -> np.ndarray:
        """Each excitation takes its own amplitude: ``-t1`` for singles, ``+t2``
        for doubles.

        The amplitudes become antisymmetric spin-orbital tensors, occupied spin
        orbitals before virtual ones, which are read back through the blocked
        register (alpha orbitals first, then beta).
        """
        counts = _require_reference(reference, "UCCSDAnsatz")
        n_spatial, n_particles = _resolve_spin_counts(
            n_qubits,
            counts.n_electrons,
            counts.n_alpha,
            counts.n_beta,
            "UCCSDAnsatz",
        )
        n_alpha, n_beta = n_particles
        cc_addons = import_optional(
            "pyscf.cc.addons", extra="chem", capability="UCCSDAnsatz"
        )
        n_virtual = (n_spatial - n_alpha, n_spatial - n_beta)
        orbspin = np.repeat([0, 1, 0, 1], (n_alpha, n_beta, *n_virtual))
        t1_spin = cc_addons.spatial2spin(t1, orbspin)
        t2_spin = cc_addons.spatial2spin(t2, orbspin)

        def occupied_index(blocked: int) -> int:
            spin, spatial = divmod(blocked, n_spatial)
            return spatial + spin * n_alpha

        def virtual_index(blocked: int) -> int:
            spin, spatial = divmod(blocked, n_spatial)
            return spatial - n_particles[spin] + spin * n_virtual[0]

        amplitudes = []
        for occupied, unoccupied in uccsd_excitations(n_spatial, n_particles):
            occupied_indices = [occupied_index(index) for index in occupied]
            virtual_indices = [virtual_index(index) for index in unoccupied]
            if len(occupied) == 1:
                amplitudes.append(-t1_spin[occupied_indices[0], virtual_indices[0]])
            else:
                amplitudes.append(t2_spin[tuple(occupied_indices + virtual_indices)])
        return np.array(amplitudes)


class HartreeFockAnsatz(Ansatz):
    """
    Hartree-Fock-based ansatz for quantum chemistry.

    This ansatz prepares the Hartree-Fock reference state and applies
    parameterised single and double excitation gates. It's a simplified
    alternative to UCCSD, often used as a starting point for VQE calculations.
    The reference is closed-shell: an odd ``n_electrons`` or unequal
    ``n_alpha`` and ``n_beta`` raise ``ValueError``.
    """

    def n_params_per_layer(
        self, n_qubits: int, reference: ElectronCounts | None = None
    ) -> int:
        """``len(singles) + len(doubles)`` spin-conserving excitations out of the
        reference set by ``reference`` (required)."""
        counts = _require_reference(reference, "HartreeFockAnsatz")
        singles, doubles = _spin_conserving_excitations(counts.n_electrons, n_qubits)
        _require_balanced_spin(counts, "HartreeFockAnsatz")
        n_params = len(singles) + len(doubles)
        return _require_trainable_params(n_params, HartreeFockAnsatz.__name__)

    def parameter_frequencies(
        self, n_qubits: int, reference: ElectronCounts | None = None
    ):
        """``{1/2, 1}`` per amplitude: both ``SingleExcitation`` and
        ``DoubleExcitation`` have generator eigenvalues ``{0, 0, +-1/2}``."""
        return [(0.5, 2)] * self.n_params_per_layer(n_qubits, reference)

    def build(
        self,
        params,
        n_qubits: int,
        n_layers: int,
        reference: ElectronCounts | None = None,
    ) -> QuantumCircuit:
        counts = _require_reference(reference, "HartreeFockAnsatz")
        singles, doubles = _spin_conserving_excitations(counts.n_electrons, n_qubits)
        _require_balanced_spin(counts, "HartreeFockAnsatz")
        params = np.asarray(params, dtype=object).reshape(n_layers, -1)

        qc = QuantumCircuit(n_qubits)
        for qubit, occupied in enumerate(_hf_occupation(counts.n_electrons, n_qubits)):
            if occupied:
                qc.x(qubit)

        # Amplitudes are ordered singles-then-doubles, but the doubles are
        # applied first.
        for layer_params in params:
            for index, wires in enumerate(doubles):
                _emit_double_excitation(qc, layer_params[len(singles) + index], *wires)
            for index, wires in enumerate(singles):
                _emit_single_excitation(qc, layer_params[index], *wires)
        return qc


class QCCAnsatz(Ansatz):
    """Qubit Coupled Cluster ansatz.

    Hartree-Fock ``X`` flips on occupied orbitals, then per-layer single-qubit
    ``RY`` rotations followed by Pauli-word exponentials (``XX``, ``YY``,
    ``ZZ``) on adjacent qubit pairs.
    """

    def n_params_per_layer(
        self, n_qubits: int, reference: ElectronCounts | None = None
    ) -> int:
        """``n_qubits`` single-qubit ``RY`` rotations plus ``3 * (n_qubits - 1)``
        entangler parameters (one ``XX``, ``YY``, ``ZZ`` per adjacent pair)."""
        n_params = n_qubits + 3 * (n_qubits - 1)
        return _require_trainable_params(n_params, QCCAnsatz.__name__)

    def build(
        self,
        params,
        n_qubits: int,
        n_layers: int,
        reference: ElectronCounts | None = None,
    ) -> QuantumCircuit:
        counts = _require_reference(reference, "QCCAnsatz")
        occupation = _hf_occupation(counts.n_electrons, n_qubits)
        params = np.asarray(params, dtype=object).reshape(n_layers, -1)

        qc = QuantumCircuit(n_qubits)
        for qubit, occupied in enumerate(occupation):
            if occupied:
                qc.x(qubit)

        n_singles = n_qubits
        for layer_params in params:
            for q in range(n_qubits):
                qc.ry(layer_params[q], q)
            ent_params = layer_params[n_singles:]
            ent_idx = 0
            for q in range(n_qubits - 1):
                # exp(-i theta/2 * P) on qubits (q, q+1) for P in {XX, YY, ZZ}.
                for pauli in ("XX", "YY", "ZZ"):
                    theta = ent_params[ent_idx]
                    _emit_two_qubit_pauli_rot(qc, pauli, theta, q, q + 1)
                    ent_idx += 1

        return qc


def _rotation_schedule(n_orb: int, depth: int | None = None) -> list[int]:
    """Lower orbital of each Givens rotation in an orbital rotation.

    The Clements brick-wall schedule: alternating half-layers of disjoint
    adjacent pairs. At the full ``depth`` of ``n_orb`` half-layers that is
    ``n_orb * (n_orb - 1) / 2`` gates, each carrying two parameters, which
    reaches any unitary; see :func:`n_rotation_params`. A smaller ``depth``
    truncates the network to a shallower, less expressive one. A larger one is
    rejected rather than capped, since it only adds redundant parameters.
    """
    if depth is None:
        depth = n_orb
    if not 0 <= depth <= n_orb:
        raise ValueError(
            f"Rotation depth must be between 0 and n_orb ({n_orb}), beyond which "
            f"the network is redundant; got {depth}."
        )
    return [
        p for half_layer in range(depth) for p in range(half_layer % 2, n_orb - 1, 2)
    ]


def n_rotation_params(
    n_orb: int, *, orbital_phases: bool, depth: int | None = None
) -> int:
    """Parameters one spin sector's orbital rotation takes: an angle and a phase
    per scheduled Givens pair, plus one phase per orbital when
    ``orbital_phases``.

    The gate phases reach the unitary group rather than only its orthogonal
    subgroup, which ``exp(iJ)`` needs -- under a real rotation the first-order
    energy correction is imaginary and cancels. A rotation that gets inverted
    around the Jastrow needs no per-orbital phases: for diagonal ``D``,
    ``(D G)^-1 J (D G) = G^-1 J G``.

    Zero for a single orbital, whose only rotation would be a global phase.
    """
    if n_orb < 2:
        return 0
    n_gates = len(_rotation_schedule(n_orb, depth))
    return 2 * n_gates + (n_orb if orbital_phases else 0)


def _rotation_one_particle(
    params: npt.ArrayLike, n_orb: int, depth: int | None = None
) -> np.ndarray:
    """One-particle matrix one spin sector's rotation block realizes.

    Gates apply in schedule order so their matrices compose right to left, and
    the per-orbital phases apply last.
    """
    values = np.asarray(params, dtype=float)
    schedule = _rotation_schedule(n_orb, depth)
    n_gates = len(schedule)
    # Zero-padded, so the sandwiched layout -- which carries no orbital phases --
    # evaluates rather than raising on an empty diagonal.
    orbital_phases = np.zeros(n_orb)
    supplied = values[2 * n_gates :]
    orbital_phases[: len(supplied)] = supplied
    angles = values[0 : 2 * n_gates : 2]
    gate_phases = values[1 : 2 * n_gates : 2]
    cosines, sines = np.cos(angles / 2), np.sin(angles / 2)
    upper = -1j * sines * np.exp(-1j * gate_phases)
    lower = -1j * sines * np.exp(1j * gate_phases)
    matrix = np.eye(n_orb, dtype=complex)
    for index, orbital in enumerate(schedule):
        row_a, row_b = matrix[orbital].copy(), matrix[orbital + 1].copy()
        matrix[orbital] = cosines[index] * row_a + upper[index] * row_b
        matrix[orbital + 1] = lower[index] * row_a + cosines[index] * row_b
    return np.exp(1j * orbital_phases)[:, None] * matrix


def fit_rotation_angles(
    target: np.ndarray, *, tol: float = 1e-7, depth: int | None = None
) -> np.ndarray | None:
    """Parameters whose rotation block realizes ``target``, or ``None``.

    ``target`` is a unitary over spatial orbitals. A full-``depth`` block spans
    that group, but not in closed form here: every gate in a given orbital pair
    contributes the same generator, so the map's derivative at zero has rank
    ``n_orb - 1`` rather than full rank. A small rotation between distant
    orbitals therefore needs order-one angles that partly cancel, and the
    parameters are recovered by a least-squares solve rather than read off.

    Returns ``None`` if no start converges below ``tol``, leaving the caller to
    fall back rather than seed from a wrong rotation. A truncated ``depth`` may
    not reach ``target`` at all.
    """
    n_orb = target.shape[0]
    n_params = n_rotation_params(n_orb, orbital_phases=True, depth=depth)
    if n_params == 0:
        return np.zeros(0)

    def residual(params: np.ndarray) -> np.ndarray:
        difference = _rotation_one_particle(params, n_orb, depth) - target
        return np.concatenate([difference.real.ravel(), difference.imag.ravel()])

    # A near-identity target is the hard case, not the easy one: zero parameters
    # realize the identity exactly but sit at the rank-deficient point, so the
    # solve can walk away from it. Return it directly when it already fits.
    if np.max(np.abs(np.eye(n_orb) - target)) < tol:
        return np.zeros(n_params)

    # Fixed starts, so a seeded run is reproducible.
    rng = np.random.default_rng(0)
    starts = [np.zeros(n_params), np.full(n_params, 0.1)] + [
        rng.uniform(-np.pi, np.pi, n_params) for _ in range(12)
    ]
    for start in starts:
        solution = least_squares(residual, start)
        if np.max(np.abs(residual(solution.x))) < tol:
            return np.asarray(solution.x, dtype=float)
    return None


def _emit_givens_rotation(
    circuit: QuantumCircuit, angle, phase, orbital: int, spin: int
) -> None:
    """Emit a Givens rotation between adjacent same-spin orbitals.

    ``angle`` mixes the two orbitals, ``phase`` sets the relative phase of the
    mixing; at ``phase = -pi / 2`` the gate reduces to the real rotation
    ``exp(angle / 2 * (a+_j a_l - a+_l a_j))``.

    The Jordan-Wigner ``Z`` on the intervening opposite-spin qubit is not
    emitted; :meth:`LUCJAnsatz.build` applies those signs once, at the end.
    """
    lower = 2 * orbital + spin
    circuit.append(XXPlusYYGate(angle, phase), [lower, lower + 2])


def _emit_rotation_block(
    circuit: QuantumCircuit,
    params,
    n_orb: int,
    spin: int,
    inverse: bool = False,
    depth: int | None = None,
) -> None:
    """Emit one spin sector's orbital rotation, or its inverse.

    ``params`` interleaves each gate's ``(angle, phase)``, then any per-orbital
    phases -- omitting those leaves the trailing slice empty. Inverting reverses
    the gates and negates the angles, keeping each gate's phase.
    """
    schedule = _rotation_schedule(n_orb, depth)
    n_gates = len(schedule)
    angles = [params[2 * index] for index in range(n_gates)]
    phases = [params[2 * index + 1] for index in range(n_gates)]
    orbital_phases = params[2 * n_gates :]

    if inverse:
        for orbital, orbital_phase in enumerate(orbital_phases):
            circuit.p(-orbital_phase, 2 * orbital + spin)
        for angle, phase, orbital in zip(
            reversed(angles), reversed(phases), reversed(schedule)
        ):
            _emit_givens_rotation(circuit, -angle, phase, orbital, spin)
        return

    for angle, phase, orbital in zip(angles, phases, schedule):
        _emit_givens_rotation(circuit, angle, phase, orbital, spin)
    for orbital, orbital_phase in enumerate(orbital_phases):
        circuit.p(orbital_phase, 2 * orbital + spin)


def lucj_jastrow_pairs(
    n_orb: int,
    same_spin_pairs: Sequence[Sequence[int]] | None = None,
    opposite_spin_pairs: Sequence[Sequence[int]] | None = None,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Orbital pairs the diagonal Coulomb operator acts on: same spin, then
    opposite spin.

    Either argument defaults to LUCJ's local pattern -- same-spin nearest
    neighbours and on-site opposite-spin pairs, the connectivity a heavy-hex
    device supports without routing. An explicit list replaces that default:
    ``[]`` drops the channel entirely, and every pair recovers the unrestricted
    diagonal Coulomb operator. A same-spin pair must name two distinct orbitals;
    an opposite-spin pair need not.
    """

    def validated(
        pairs: Sequence[Sequence[int]], label: str, distinct: bool
    ) -> list[tuple[int, int]]:
        checked = []
        for pair in pairs:
            p, q = (int(index) for index in pair)
            if not (0 <= p < n_orb and 0 <= q < n_orb):
                raise ValueError(
                    f"{label} pair {(p, q)} is outside the {n_orb} orbitals."
                )
            if distinct and p == q:
                raise ValueError(
                    f"{label} pair {(p, q)} repeats an orbital, which is a "
                    "one-body term rather than a Coulomb interaction."
                )
            checked.append((p, q))
        return checked

    same = (
        [(p, p + 1) for p in range(n_orb - 1)]
        if same_spin_pairs is None
        else validated(same_spin_pairs, "Same-spin", distinct=True)
    )
    opposite = (
        [(p, p) for p in range(n_orb)]
        if opposite_spin_pairs is None
        else validated(opposite_spin_pairs, "Opposite-spin", distinct=False)
    )
    return same, opposite


def _lucj_orbital_count(n_qubits: int) -> int:
    if n_qubits % 2:
        raise ValueError(
            f"LUCJAnsatz needs an even qubit count (two spin-orbitals per "
            f"spatial orbital); got {n_qubits}."
        )
    return n_qubits // 2


def _emit_jordan_wigner_signs(circuit: QuantumCircuit, n_orb: int) -> None:
    """Apply ``(-1)^(sum of n_beta[q] * n_alpha[p] over q < p)``.

    A CNOT ladder keeps the parity of the beta occupations below orbital ``p`` on
    beta qubit ``p - 1``, which a ``CZ`` then pairs with alpha qubit ``p``.
    """
    for p in range(1, n_orb):
        if p > 1:
            circuit.cx(2 * p - 3, 2 * p - 1)
        circuit.cz(2 * p - 1, 2 * p)
    for p in range(n_orb - 1, 1, -1):
        circuit.cx(2 * p - 3, 2 * p - 1)


def _layered(
    couplings: Sequence[tuple[Any, tuple[int, int]]],
) -> list[tuple[Any, tuple[int, int]]]:
    """Reorder mutually commuting two-qubit gates so each layer touches every
    qubit at most once. A gate takes the first layer free on both its qubits."""
    taken: defaultdict[int, set[int]] = defaultdict(set)
    layers: list[list[tuple[Any, tuple[int, int]]]] = []
    for angle, wires in couplings:
        busy = taken[wires[0]] | taken[wires[1]]
        index = next(i for i in range(len(layers) + 1) if i not in busy)
        if index == len(layers):
            layers.append([])
        layers[index].append((angle, wires))
        taken[wires[0]].add(index)
        taken[wires[1]].add(index)
    return [coupling for layer in layers for coupling in layer]


def _one_body_from_excitations(
    block: np.ndarray, n_orb: int
) -> tuple[np.ndarray, np.ndarray]:
    """Diagonalise the one-body operator an occupied-virtual block defines.

    Returns ``(eigenvectors, eigenvalues)``.
    """
    n_occupied, n_virtual = block.shape
    one_body = np.zeros((n_orb, n_orb))
    one_body[:n_occupied, n_occupied : n_occupied + n_virtual] = block
    one_body[n_occupied : n_occupied + n_virtual, :n_occupied] = block.T
    eigenvalues, eigenvectors = np.linalg.eigh(one_body)
    return eigenvectors, eigenvalues


class LUCJAnsatz(AmplitudeSeededAnsatz):
    """Local unitary cluster Jastrow ansatz.

    Each layer applies ``exp(K) exp(iJ) exp(-K)``, where ``K`` is a general
    orbital rotation -- a brick-wall network of Givens rotations, independent
    per spin sector -- and ``J`` is a diagonal Coulomb operator restricted to
    same-orbital opposite-spin pairs plus same-spin neighbours. That restriction
    on ``J`` alone is what makes the ansatz *local*; the rotation is
    unrestricted. Both factors conserve particle number and Sz, which
    sample-based diagonalisation requires -- its symmetry filter checks alpha
    and beta populations separately.

    That default is the flavour the literature uses. The arguments select others.

    Args:
        trailing_rotation: Close each layer with an independent orbital
            rotation rather than the inverse of that layer's own. At
            ``n_layers=1`` that is the truncated LUCJ circuit
            ``exp(K2) exp(-K1) exp(iJ1) exp(K1)`` of arXiv:2405.05068 and
            arXiv:2512.14936; without it, the same circuit lacking ``exp(K2)``.
        shared_spin_params: Drive both spin sectors from one set of rotation and
            same-spin Coulomb parameters, halving the count and imposing spin
            symmetry.
        rotation_depth: Truncate each rotation's brick-wall network to this many
            half-layers. ``None`` keeps the full ``n_orb`` half-layers.
        same_spin_pairs: Replace the Jastrow's default same-spin nearest
            neighbours with explicit ``(orbital, orbital)`` pairs, from ``[]``
            to every pair.
        opposite_spin_pairs: Replace the Jastrow's default on-site opposite-spin
            pairs with explicit ``(orbital, orbital)`` pairs.

    Assumes the interleaved Jordan-Wigner ordering that
    ``divi.hamiltonians._chem._spo_from_integrals`` produces: qubit ``2p`` is
    the alpha spin-orbital of spatial orbital ``p`` and ``2p + 1`` is its beta
    partner. The Hartree-Fock reference is embedded, so no separate initial
    state is needed.

    The Jordan-Wigner signs of every hop are applied once, as a single parity
    ladder after the last layer, instead of around each Givens rotation.

    ``n_electrons`` must be at most ``n_qubits``. Spin-imbalanced references
    are supported through the ``n_alpha`` and ``n_beta`` of the
    :class:`ElectronCounts`; without them the closed-shell split
    ``n_alpha = n_beta = n_electrons // 2`` is used, which requires an even
    ``n_electrons``. A single spatial orbital offers no variational freedom:
    only the on-site diagonal term survives, and its value cannot change the
    state.
    """

    def __init__(
        self,
        *,
        trailing_rotation: bool = False,
        shared_spin_params: bool = False,
        rotation_depth: int | None = None,
        same_spin_pairs: Sequence[Sequence[int]] | None = None,
        opposite_spin_pairs: Sequence[Sequence[int]] | None = None,
    ) -> None:
        self.trailing_rotation = trailing_rotation
        self.shared_spin_params = shared_spin_params
        self.rotation_depth = rotation_depth
        self.same_spin_pairs = same_spin_pairs
        self.opposite_spin_pairs = opposite_spin_pairs

    def _layout(
        self, n_orb: int
    ) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]], int | None]:
        """A layer's structure: independent spin sectors, Jastrow pairs, and the
        rotation depth."""
        same, opposite = lucj_jastrow_pairs(
            n_orb, self.same_spin_pairs, self.opposite_spin_pairs
        )
        n_sectors = 1 if self.shared_spin_params else 2
        return n_sectors, same, opposite, self.rotation_depth

    def n_params_per_layer(
        self, n_qubits: int, reference: ElectronCounts | None = None
    ) -> int:
        """Per-layer parameter count, where ``n_orb = n_qubits // 2``.

        A full-depth rotation takes two parameters per Givens pair -- an angle
        and a phase -- so ``n_orb * (n_orb - 1)`` per spin sector, and the
        default Jastrow adds ``n_orb`` on-site Coulomb terms plus one per
        same-spin neighbour pair. Both are counted once per independent spin
        sector: twice by default, once under ``shared_spin_params``. That leaves
        ``1`` for a single spatial orbital (``n_qubits == 2``), where only the
        on-site term survives.

        ``trailing_rotation=True`` adds another rotation per sector, each also
        carrying one phase per orbital. ``rotation_depth`` and the Jastrow pair
        arguments shrink the count further; see :meth:`build`.
        """
        n_orb = _lucj_orbital_count(n_qubits)
        n_sectors, same_pairs, opposite_pairs, depth = self._layout(n_orb)
        n_params = n_sectors * n_rotation_params(
            n_orb, orbital_phases=False, depth=depth
        )
        n_params += len(opposite_pairs) + n_sectors * len(same_pairs)
        if self.trailing_rotation:
            n_params += n_sectors * n_rotation_params(
                n_orb, orbital_phases=True, depth=depth
            )
        return _require_trainable_params(n_params, LUCJAnsatz.__name__)

    def parameter_frequencies(
        self, n_qubits: int, reference: ElectronCounts | None = None
    ):
        """Per-parameter frequency families, which differ by role.

        A gate's mixing angle sets eigenphases ``+-theta / 2``, so one occurrence
        carries ``{1/2, 1}``; a sandwiched gate appears twice, compounding to
        ``{1/2, ..., 2}``. A gate's phase enters as ``exp(+-i beta)`` instead, so
        two occurrences reach degree two in the amplitude and the energy squares
        that -- measured up to ``4``. The ``RZZ`` Jastrow angles and the
        per-orbital phases are plain Pauli rotations.

        Trailing parameters appear once rather than twice, halving each family;
        ``shared_spin_params`` parameters drive both sectors, doubling it.
        """
        n_orb = n_qubits // 2
        n_sectors, same_pairs, opposite_pairs, depth = self._layout(n_orb)
        n_gates = len(_rotation_schedule(n_orb, depth))
        sandwiched = 4 // n_sectors
        trailing = 2 // n_sectors

        frequencies: list[tuple[float, int]] = []
        for _sector in range(n_sectors):
            frequencies += [(0.5, 2 * sandwiched), (1.0, 2 * sandwiched)] * n_gates
        frequencies += [(1.0, 1)] * len(opposite_pairs)
        frequencies += [(1.0, trailing)] * (n_sectors * len(same_pairs))
        if self.trailing_rotation:
            for _sector in range(n_sectors):
                frequencies += [(0.5, 2 * trailing), (1.0, 2 * trailing)] * n_gates
                frequencies += [(1.0, trailing)] * n_orb
        return frequencies

    def build(
        self,
        params,
        n_qubits: int,
        n_layers: int,
        reference: ElectronCounts | None = None,
    ) -> QuantumCircuit:
        """Build the LUCJ ansatz circuit.

        Args:
            params: Flat parameter array of length
                ``n_layers * n_params_per_layer(n_qubits)``.
            n_qubits: Number of qubits. Must be even.
            n_layers: Number of LUCJ layers.
            reference: Electron counts of the reference determinant (required).
                ``n_electrons`` is at most ``n_qubits``, and ``n_alpha`` /
                ``n_beta`` select a spin-imbalanced one.

        Parameters are consumed in the order: each sandwiched rotation (alpha
        sector then beta) as an ``(angle, phase)`` pair per Givens gate in
        brick-wall order, then the opposite-spin Coulomb terms, then the
        same-spin ones (alpha sector then beta), then each trailing rotation --
        its own ``(angle, phase)`` pairs followed by one phase per orbital. Under
        ``shared_spin_params`` only the alpha sector's parameters appear, and drive
        both.

        Returns:
            QuantumCircuit: Qiskit circuit implementing the LUCJ ansatz.

        Raises:
            ValueError: If ``n_qubits`` is odd, if ``n_electrons`` is missing or
                exceeds ``n_qubits``, if ``n_electrons`` is odd without
                ``n_alpha``/``n_beta``, or if ``params`` is shorter than
                ``n_layers * n_params_per_layer(n_qubits)``.
        """
        counts = _require_reference(reference, "LUCJAnsatz")
        if counts.n_electrons > n_qubits:
            raise ValueError(
                f"n_electrons ({counts.n_electrons}) cannot exceed n_qubits "
                f"({n_qubits})."
            )
        n_orb, (n_alpha, n_beta) = _resolve_spin_counts(
            n_qubits,
            counts.n_electrons,
            counts.n_alpha,
            counts.n_beta,
            "LUCJAnsatz",
        )

        per_layer = self.n_params_per_layer(n_qubits)
        flat = np.asarray(params, dtype=object).flatten()
        n_required = n_layers * per_layer
        if flat.size < n_required:
            raise ValueError(
                f"LUCJAnsatz expected {n_required} parameters "
                f"({n_layers} layers x {per_layer} per layer); got {flat.size}."
            )

        circuit = QuantumCircuit(n_qubits)
        for p in range(n_alpha):
            circuit.x(2 * p)
        for p in range(n_beta):
            circuit.x(2 * p + 1)

        n_sectors, same_pairs, opposite_pairs, depth = self._layout(n_orb)
        sandwiched_size = n_rotation_params(n_orb, orbital_phases=False, depth=depth)
        trailing_size = n_rotation_params(n_orb, orbital_phases=True, depth=depth)

        for layer in range(n_layers):
            cursor = layer * per_layer

            rotation_params = []
            for _sector in range(n_sectors):
                rotation_params.append(flat[cursor : cursor + sandwiched_size])
                cursor += sandwiched_size
            for spin in (0, 1):
                _emit_rotation_block(
                    circuit, rotation_params[spin % n_sectors], n_orb, spin, depth=depth
                )

            opposite_start = cursor
            same_start = opposite_start + len(opposite_pairs)
            couplings = [
                (
                    flat[same_start + (spin % n_sectors) * len(same_pairs) + index],
                    (2 * p + spin, 2 * q + spin),
                )
                for spin in (0, 1)
                for index, (p, q) in enumerate(same_pairs)
            ] + [
                (flat[opposite_start + index], (2 * p, 2 * q + 1))
                for index, (p, q) in enumerate(opposite_pairs)
            ]
            for angle, wires in _layered(couplings):
                circuit.append(RZZGate(angle), wires)
            cursor = same_start + n_sectors * len(same_pairs)

            for spin in (1, 0):
                _emit_rotation_block(
                    circuit,
                    rotation_params[spin % n_sectors],
                    n_orb,
                    spin,
                    inverse=True,
                    depth=depth,
                )

            if self.trailing_rotation:
                trailing_params = []
                for _sector in range(n_sectors):
                    trailing_params.append(flat[cursor : cursor + trailing_size])
                    cursor += trailing_size
                for spin in (0, 1):
                    _emit_rotation_block(
                        circuit,
                        trailing_params[spin % n_sectors],
                        n_orb,
                        spin,
                        depth=depth,
                    )

        _emit_jordan_wigner_signs(circuit, n_orb)
        return circuit

    def params_from_amplitudes(
        self,
        t1: tuple[np.ndarray, np.ndarray],
        t2: tuple[np.ndarray, np.ndarray, np.ndarray],
        n_qubits: int,
        reference: ElectronCounts,
    ) -> np.ndarray:
        """Map the amplitudes onto the parameters through a double factorisation.

        The leading singular triplet of the opposite-spin doubles, reshaped over
        ``(occupied, virtual)`` pairs, gives one one-body operator per spin. Each
        spin's eigenbasis is its rotation, the eigenvalues give
        ``J_pq = sigma * d_p * d_q``, and ``t1`` supplies the trailing rotation.
        Opposite-spin doubles are used because a spin-polarised reference with
        one electron of a spin has no same-spin doubles; same-spin Coulomb
        weights stay at zero.

        The result is a starting point, not an encoding of the amplitudes: one
        Jastrow layer holds only the leading factorisation term, ``J`` is
        projected onto the layer's pair pattern, and the ``RZZ`` gates omit the
        one-body remainder of ``exp(i J n_p n_q)``.

        Raises:
            AmplitudeSeedError: If ``shared_spin_params`` is set, if the
                opposite-spin doubles block is empty, or if a spin sector's
                rotation cannot be realised.
        """
        if self.shared_spin_params:
            raise AmplitudeSeedError(
                "the double factorisation gives each spin sector its own "
                "rotation, which shared_spin_params cannot hold."
            )

        n_orb = _lucj_orbital_count(n_qubits)
        _, same_pairs, opposite_pairs, depth = self._layout(n_orb)
        t1_alpha, t1_beta = t1
        t2_opposite = np.asarray(t2[1])

        n_occupied_alpha, n_occupied_beta, n_virtual_alpha, n_virtual_beta = (
            t2_opposite.shape
        )
        if min(t2_opposite.shape) == 0:
            raise AmplitudeSeedError(
                "the opposite-spin doubles block is empty, which leaves all-zero "
                "parameters, a stationary point a gradient optimiser cannot leave."
            )

        # Rectangular when the spin sectors differ in size.
        matrix = t2_opposite.transpose(0, 2, 1, 3).reshape(
            n_occupied_alpha * n_virtual_alpha, n_occupied_beta * n_virtual_beta
        )
        left, singular_values, right = np.linalg.svd(matrix)
        scale = float(singular_values[0])

        rotation_alpha, diagonal_alpha = _one_body_from_excitations(
            left[:, 0].reshape(n_occupied_alpha, n_virtual_alpha), n_orb
        )
        rotation_beta, diagonal_beta = _one_body_from_excitations(
            right[0, :].reshape(n_occupied_beta, n_virtual_beta), n_orb
        )
        # The beta rotation takes -i on its virtual orbitals to absorb the Jastrow's factor of i.
        phase = np.ones(n_orb, dtype=complex)
        phase[n_occupied_beta:] = -1j
        rotation_beta = rotation_beta.astype(complex) * phase[:, None]

        seed = np.zeros(self.n_params_per_layer(n_qubits, reference))
        cursor = 0
        sandwiched = n_rotation_params(n_orb, orbital_phases=False, depth=depth)
        for rotation in (rotation_alpha, rotation_beta):
            # The block runs first, in exp(-K), so it realises the inverse; the
            # conjugate transpose, not the transpose, pairs with the -i above.
            angles = fit_rotation_angles(rotation.conj().T, depth=depth)
            if angles is None:
                raise AmplitudeSeedError(
                    "a spin sector's orbital rotation cannot be realised by the "
                    "rotation network."
                )
            # The fit's per-orbital phases are dropped; the sandwich cancels them.
            seed[cursor : cursor + sandwiched] = angles[:sandwiched]
            cursor += sandwiched

        # exp(i J n_p n_q) contributes exp(i J Z_p Z_q / 4), which is RZZ(-J / 2).
        for p, q in opposite_pairs:
            seed[cursor] = -0.5 * scale * diagonal_alpha[p] * diagonal_beta[q]
            cursor += 1

        # Same-spin Coulomb weights stay at zero.
        cursor += 2 * len(same_pairs)

        if self.trailing_rotation:
            for t1_block in (t1_alpha, t1_beta):
                generator = np.zeros((n_orb, n_orb))
                n_occupied, n_virtual = t1_block.shape
                generator[:n_occupied, n_occupied : n_occupied + n_virtual] = -t1_block
                generator -= generator.T
                angles = fit_rotation_angles(expm(generator), depth=depth)
                if angles is None:
                    # Only this block is lost; the rest of the seed stays useful.
                    warn(
                        "LUCJAnsatz could not realise a trailing rotation; "
                        "seeding the rest and leaving it at the identity.",
                        UserWarning,
                        stacklevel=2,
                    )
                    angles = np.zeros(
                        n_rotation_params(n_orb, orbital_phases=True, depth=depth)
                    )
                seed[cursor : cursor + len(angles)] = angles
                cursor += len(angles)

        return seed


def _validate_reference_counts(
    n_electrons: int, n_qubits: int, *, require_virtual: bool
) -> None:
    """Validate the electron count for a spin-orbital reference register."""
    if isinstance(n_electrons, bool) or not isinstance(n_electrons, Integral):
        raise ValueError(
            f"n_electrons must be a positive integer; got {n_electrons!r}."
        )
    if n_electrons <= 0:
        raise ValueError(
            f"n_electrons must be a positive integer; got {n_electrons!r}."
        )
    if n_electrons > n_qubits:
        raise ValueError(
            f"n_electrons ({n_electrons}) cannot exceed n_qubits ({n_qubits})."
        )
    if require_virtual and n_electrons == n_qubits:
        raise ValueError(
            "An excitation ansatz requires at least one virtual spin-orbital; "
            f"got n_electrons == n_qubits == {n_qubits}."
        )


def _hf_occupation(n_electrons: int, n_qubits: int) -> tuple[int, ...]:
    """Occupation-number vector of the Hartree-Fock determinant."""
    _validate_reference_counts(n_electrons, n_qubits, require_virtual=False)
    return tuple(1 if i < n_electrons else 0 for i in range(n_qubits))


def _spin_conserving_excitations(
    n_electrons: int, n_qubits: int
) -> tuple[list[tuple[int, int]], list[tuple[int, int, int, int]]]:
    """Singles and doubles out of the Hartree-Fock reference.

    Even qubits carry spin-up and odd qubits spin-down, so an excitation is kept
    only when it leaves the total spin projection unchanged.
    """
    _validate_reference_counts(n_electrons, n_qubits, require_virtual=True)
    spin_up = [qubit % 2 == 0 for qubit in range(n_qubits)]
    occupied = range(n_electrons)
    virtual = range(n_electrons, n_qubits)
    singles = [
        (r, p) for r, p in product(occupied, virtual) if spin_up[p] == spin_up[r]
    ]
    doubles = [
        (s, r, q, p)
        for s, r in combinations(occupied, 2)
        for q, p in combinations(virtual, 2)
        if spin_up[p] + spin_up[q] == spin_up[r] + spin_up[s]
    ]
    return singles, doubles


def _emit_single_excitation(qc: QuantumCircuit, theta, w0: int, w1: int) -> None:
    """Emit a Givens rotation mixing ``|01>`` and ``|10>`` on ``(w0, w1)``.

    Two CNOTs and two ``RY(-theta/2)``, from figure 2 of Anselmetti et al.,
    *New J. Phys.* **23** 113010 (2021), arXiv:2104.05695.
    """
    half = theta / 2
    qc.h(w0)
    qc.cx(w0, w1)
    qc.ry(-half, w0)
    qc.ry(-half, w1)
    qc.cx(w0, w1)
    qc.h(w0)


def _emit_double_excitation(
    qc: QuantumCircuit, theta, w0: int, w1: int, w2: int, w3: int
) -> None:
    """Emit a Givens rotation mixing ``|0011>`` and ``|1100>`` on ``(w0..w3)``.

    Fourteen CNOTs and eight ``RY(±theta/8)``, from page 17 of Anselmetti et al.,
    *New J. Phys.* **23** 113010 (2021), arXiv:2104.05695.
    """
    eighth = theta / 8
    qc.cx(w2, w3)
    qc.cx(w0, w2)
    qc.h(w3)
    qc.h(w0)
    qc.cx(w2, w3)
    qc.cx(w0, w1)
    qc.ry(eighth, w1)
    qc.ry(-eighth, w0)
    qc.cx(w0, w3)
    qc.h(w3)
    qc.cx(w3, w1)
    qc.ry(eighth, w1)
    qc.ry(-eighth, w0)
    qc.cx(w2, w1)
    qc.cx(w2, w0)
    qc.ry(-eighth, w1)
    qc.ry(eighth, w0)
    qc.cx(w3, w1)
    qc.h(w3)
    qc.cx(w0, w3)
    qc.ry(-eighth, w1)
    qc.ry(eighth, w0)
    qc.cx(w0, w1)
    qc.cx(w2, w0)
    qc.h(w0)
    qc.h(w3)
    qc.cx(w0, w2)
    qc.cx(w2, w3)


def _emit_two_qubit_pauli_rot(
    qc: QuantumCircuit, pauli: str, theta, q1: int, q2: int
) -> None:
    """Emit ``exp(-i theta/2 * P)`` for ``P ∈ {XX, YY, ZZ}`` onto ``qc`` as a
    ``H``/``RX(±π/2)`` basis change plus a CX-RZ-CX ladder.
    """
    if pauli == "XX":
        qc.h(q1)
        qc.h(q2)
        qc.cx(q1, q2)
        qc.rz(theta, q2)
        qc.cx(q1, q2)
        qc.h(q1)
        qc.h(q2)
    elif pauli == "YY":
        qc.rx(_HALF_PI, q1)
        qc.rx(_HALF_PI, q2)
        qc.cx(q1, q2)
        qc.rz(theta, q2)
        qc.cx(q1, q2)
        qc.rx(-_HALF_PI, q1)
        qc.rx(-_HALF_PI, q2)
    elif pauli == "ZZ":
        qc.cx(q1, q2)
        qc.rz(theta, q2)
        qc.cx(q1, q2)
    else:
        raise ValueError(f"Unsupported two-qubit Pauli {pauli!r}; expected XX/YY/ZZ.")
