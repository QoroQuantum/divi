# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Built-in QAOA / VQE ansätze.

Every ``Ansatz.build`` creates and returns a :class:`~qiskit.circuit.QuantumCircuit`.
``UCCSDAnsatz`` sources its excitations and Hartree-Fock reference from
``qiskit_nature`` (the ``chem`` extra), remapping that library's blocked spin
ordering onto Divi's interleaved one. The remaining chemistry ansätze
(``HartreeFockAnsatz``, ``QCCAnsatz``) source excitation / Hartree-Fock data
from ``pennylane.qchem`` and route the PL gates through the local PL → Qiskit
converter; consumers always see Qiskit instructions.
"""

import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import cache
from typing import Literal
from warnings import warn

import numpy as np
import numpy.typing as npt
import pennylane as qp
from qiskit.circuit import Gate, QuantumCircuit
from qiskit.circuit.library import RXGate, RYGate, RZGate, RZZGate, XXPlusYYGate
from scipy.optimize import least_squares

from divi.circuits._conversions import _qscript_to_dag
from divi.hamiltonians._chem import requires_chem_extra
from divi.hamiltonians._term_ops import _HALF_PI


def _require_trainable_params(n_params: int, ansatz_name: str) -> int:
    if n_params <= 0:
        raise ValueError(
            f"{ansatz_name} must define at least one trainable parameter. "
            "Parameter-free circuits are not supported."
        )
    return n_params


def _require_n_electrons(kwargs: dict, ansatz_name: str) -> int:
    """Pop ``n_electrons``, rejecting a missing one by name.

    A chemistry ansatz cannot enumerate excitations without it, and passing
    ``None`` through surfaces as a comparison against ``NoneType`` from inside
    PennyLane, naming neither the setting nor the ansatz.
    """
    n_electrons = kwargs.pop("n_electrons", None)
    if n_electrons is None:
        raise ValueError(
            f"{ansatz_name} requires n_electrons: it builds excitations from a "
            "reference state, which needs the electron count. Pass "
            "n_electrons=... to the program (a molecule input supplies it "
            "automatically; a raw Hamiltonian does not)."
        )
    return n_electrons


def _pl_ops_to_qc(pl_ops: Sequence, n_qubits: int) -> QuantumCircuit:
    """Translate ``pl_ops`` to Qiskit gates and return a circuit on ``n_qubits`` qubits."""
    qc = QuantumCircuit(n_qubits)
    if not pl_ops:
        return qc
    script = qp.tape.QuantumScript(list(pl_ops))
    dag, _params, _wire_map = _qscript_to_dag(script)
    for node in dag.topological_op_nodes():
        qubit_indices = [dag.qubits.index(q) for q in node.qargs]
        qc.append(node.op, [qc.qubits[i] for i in qubit_indices])
    return qc


class Ansatz(ABC):
    """Abstract base class for all VQE ansätze."""

    @property
    def name(self) -> str:
        """Returns the human-readable name of the ansatz."""
        return self.__class__.__name__

    @staticmethod
    @abstractmethod
    def n_params_per_layer(n_qubits: int, **kwargs) -> int:
        """Returns the number of parameters required by the ansatz for one layer."""
        raise NotImplementedError

    def parameter_frequencies(
        self, n_qubits: int, **kwargs
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
    def build(self, params, n_qubits: int, n_layers: int, **kwargs) -> QuantumCircuit:
        """
        Builds the ansatz circuit and returns a list of operations.

        Args:
            params: Parameter array for the ansatz.
            n_qubits (int): Number of qubits in the circuit.
            n_layers (int): Number of ansatz layers.
            **kwargs: Additional arguments specific to the ansatz.

        Returns:
            QuantumCircuit: The ansatz circuit on ``n_qubits`` qubits.
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
            | None
        ) = None,
    ):
        """
        Args:
            gate_sequence: Sequence of one-qubit Qiskit ``Gate`` subclasses
                (e.g., ``RYGate``, ``RZGate``).
            entangler: Two-qubit Qiskit ``Gate`` subclass (e.g., ``CXGate``,
                ``CZGate``). If None, no entanglement is applied.
            entangling_layout (str): Layout for entangling layer ("linear", "all-to-all", etc.).
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

        self.entangling_layout = entangling_layout
        if entangler is None and entangling_layout is not None:
            warn("`entangling_layout` provided but `entangler` is None.")
        match entangling_layout:
            case None | "linear":
                self.entangling_layout = "linear"
                self._layout_fn = lambda n_qubits: zip(
                    range(n_qubits), range(1, n_qubits)
                )
            case "brick":
                self._layout_fn = lambda n_qubits: [
                    (i, i + 1) for r in range(2) for i in range(r, n_qubits - 1, 2)
                ]
            case "circular":
                self._layout_fn = lambda n_qubits: zip(
                    range(n_qubits), [(i + 1) % n_qubits for i in range(n_qubits)]
                )
            case "all-to-all":
                self._layout_fn = lambda n_qubits: (
                    (i, j) for i in range(n_qubits) for j in range(i + 1, n_qubits)
                )
            case list() | tuple() as custom_layout:
                if not all(
                    isinstance(ent, tuple)
                    and len(ent) == 2
                    and isinstance(ent[0], int)
                    and isinstance(ent[1], int)
                    for ent in custom_layout
                ):
                    raise ValueError(
                        "entangling_layout must be 'linear', 'circular', "
                        "'all-to-all', or a Sequence of tuples of integers."
                    )
                self._layout_fn = lambda _: list(custom_layout)
            case _:
                raise ValueError(
                    f"Unknown entangling_layout: {entangling_layout!r}. "
                    "Must be 'linear', 'circular', 'all-to-all', or "
                    "a Sequence of (int, int) tuples."
                )

    def n_params_per_layer(self, n_qubits: int, **kwargs) -> int:
        """``sum(_gate_n_params(g) for g in gate_sequence) * n_qubits``."""
        per_qubit = sum(self._gate_param_counts)
        return _require_trainable_params(per_qubit * n_qubits, self.name)

    def build(self, params, n_qubits: int, n_layers: int, **kwargs) -> QuantumCircuit:
        qc = QuantumCircuit(n_qubits)
        gate_param_counts = self._gate_param_counts
        per_qubit = sum(gate_param_counts)

        params = np.asarray(params, dtype=object).reshape(n_layers, n_qubits, per_qubit)
        layout = list(self._layout_fn(n_qubits))

        for layer_idx in range(n_layers):
            layer_params = params[layer_idx]
            for q, qubit_params in zip(range(n_qubits), layer_params):
                idx = 0
                for gate_cls, n_p in zip(self.gate_sequence, gate_param_counts):
                    args = list(qubit_params[idx : idx + n_p])
                    qc.append(gate_cls(*args), [q])
                    idx += n_p

            if self.entangler is not None:
                for wire_a, wire_b in layout:
                    qc.append(self.entangler(), [wire_a, wire_b])  # type: ignore[call-arg]

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

    @staticmethod
    def n_params_per_layer(n_qubits: int, **kwargs) -> int:
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

    def build(self, params, n_qubits: int, n_layers: int, **kwargs) -> QuantumCircuit:
        """Build the QAOA ansatz circuit.

        Args:
            params: Flat parameter array of length
                ``n_layers * n_params_per_layer(n_qubits)``.
            n_qubits: Number of qubits.
            n_layers: Number of QAOA layers.
            **kwargs: Additional unused arguments.

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
    with requires_chem_extra("UCCSDAnsatz"):
        # pyrefly: ignore[missing-import]
        from qiskit_nature.second_q import mappers

        # pyrefly: ignore[missing-import]
        from qiskit_nature.second_q.circuit import library

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


def _uccsd_excitations(
    n_spatial: int, n_particles: tuple[int, int]
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """UCCSD excitation list, as ``(occupied, unoccupied)`` blocked indices.

    Positionally aligned with the ansatz's parameter vector.
    """
    return _uccsd_template(n_spatial, n_particles)[2]


class UCCSDAnsatz(Ansatz):
    """
    Unitary Coupled Cluster Singles and Doubles (UCCSD) ansatz.

    This ansatz is specifically designed for quantum chemistry calculations,
    implementing the UCCSD approximation which includes all single and double
    electron excitations from a reference state.

    Excitations and the Hartree-Fock reference come from ``qiskit_nature``,
    which requires the ``chem`` extra. Spin-imbalanced references are
    supported by passing ``n_alpha`` and ``n_beta``; without them the
    closed-shell split ``n_alpha = n_beta = n_electrons // 2`` is used.
    """

    @staticmethod
    def n_params_per_layer(n_qubits: int, **kwargs) -> int:
        """Number of UCCSD excitation amplitudes for the given reference.

        Requires ``n_electrons``; optionally accepts ``n_alpha`` / ``n_beta``.
        """
        n_electrons = _require_n_electrons(kwargs, "UCCSDAnsatz")
        n_spatial, n_particles = _resolve_spin_counts(
            n_qubits,
            n_electrons,
            kwargs.get("n_alpha"),
            kwargs.get("n_beta"),
            "UCCSDAnsatz",
        )
        _, template, _ = _uccsd_template(n_spatial, n_particles)
        return _require_trainable_params(template.num_parameters, UCCSDAnsatz.__name__)

    def parameter_frequencies(self, n_qubits: int, **kwargs):
        """``{1, 2}`` per amplitude: each excitation exponentiates a generator
        whose eigenvalues span two distinct gaps."""
        return [(1.0, 2)] * UCCSDAnsatz.n_params_per_layer(n_qubits, **kwargs)

    def build(self, params, n_qubits: int, n_layers: int, **kwargs) -> QuantumCircuit:
        n_electrons = _require_n_electrons(kwargs, "UCCSDAnsatz")
        n_spatial, n_particles = _resolve_spin_counts(
            n_qubits,
            n_electrons,
            kwargs.get("n_alpha"),
            kwargs.get("n_beta"),
            "UCCSDAnsatz",
        )
        reference, template, _ = _uccsd_template(n_spatial, n_particles)
        params = np.asarray(params, dtype=object).reshape(n_layers, -1)

        qc = QuantumCircuit(n_qubits)
        qc.compose(reference, inplace=True)
        for layer in params:
            qc.compose(
                template.assign_parameters(list(layer), inplace=False), inplace=True
            )
        return qc


class HartreeFockAnsatz(Ansatz):
    """
    Hartree-Fock-based ansatz for quantum chemistry.

    This ansatz prepares the Hartree-Fock reference state and applies
    parameterised single and double excitation gates. It's a simplified
    alternative to UCCSD, often used as a starting point for VQE calculations.
    """

    @staticmethod
    def n_params_per_layer(n_qubits: int, **kwargs) -> int:
        """``len(singles) + len(doubles)`` from ``qp.qchem.excitations`` for
        the given ``n_electrons`` (required kwarg)."""
        n_electrons = _require_n_electrons(kwargs, "HartreeFockAnsatz")
        singles, doubles = qp.qchem.excitations(n_electrons, n_qubits)
        n_params = len(singles) + len(doubles)
        return _require_trainable_params(n_params, HartreeFockAnsatz.__name__)

    def parameter_frequencies(self, n_qubits: int, **kwargs):
        """``{1/2, 1}`` per amplitude: both ``SingleExcitation`` and
        ``DoubleExcitation`` have generator eigenvalues ``{0, 0, +-1/2}``."""
        return [(0.5, 2)] * HartreeFockAnsatz.n_params_per_layer(n_qubits, **kwargs)

    def build(self, params, n_qubits: int, n_layers: int, **kwargs) -> QuantumCircuit:
        n_electrons = _require_n_electrons(kwargs, "HartreeFockAnsatz")
        singles, doubles = qp.qchem.excitations(n_electrons, n_qubits)
        hf_state = qp.qchem.hf_state(n_electrons, n_qubits)
        params = np.asarray(params, dtype=object).reshape(n_layers, -1)

        pl_ops: list = []
        for layer_idx, layer_params in enumerate(params):
            layer_ops = list(
                qp.AllSinglesDoubles.compute_decomposition(
                    layer_params,
                    wires=range(n_qubits),
                    hf_state=hf_state,
                    singles=singles,
                    doubles=doubles,
                )
            )
            # Only the first layer should prepare the Hartree-Fock state; reset
            # the basis-state init for subsequent layers.
            if layer_idx > 0:
                layer_ops = [op for op in layer_ops if op.name != "BasisState"]
            pl_ops.extend(layer_ops)
        return _pl_ops_to_qc(pl_ops, n_qubits)


class QCCAnsatz(Ansatz):
    """Qubit Coupled Cluster ansatz.

    Hartree-Fock ``X`` flips on occupied orbitals, then per-layer single-qubit
    ``RY`` rotations followed by Pauli-word exponentials (``XX``, ``YY``,
    ``ZZ``) on adjacent qubit pairs.
    """

    @staticmethod
    def n_params_per_layer(n_qubits: int, **kwargs) -> int:
        """``n_qubits`` single-qubit ``RY`` rotations plus ``3 * (n_qubits - 1)``
        entangler parameters (one ``XX``, ``YY``, ``ZZ`` per adjacent pair)."""
        n_params = n_qubits + 3 * (n_qubits - 1)
        return _require_trainable_params(n_params, QCCAnsatz.__name__)

    def build(self, params, n_qubits: int, n_layers: int, **kwargs) -> QuantumCircuit:
        n_electrons = _require_n_electrons(kwargs, "QCCAnsatz")
        hf_state = qp.qchem.hf_state(n_electrons, n_qubits)
        params = np.asarray(params, dtype=object).reshape(n_layers, -1)

        qc = QuantumCircuit(n_qubits)
        # Hartree-Fock prep: ``hf_state`` is a 0/1 vector of length n_qubits.
        for q, bit in enumerate(hf_state):
            if bit:
                qc.x(q)

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
    matrix = np.eye(n_orb, dtype=complex)
    for index, orbital in enumerate(schedule):
        angle, phase = values[2 * index], values[2 * index + 1]
        block = np.eye(n_orb, dtype=complex)
        cosine, sine = np.cos(angle / 2), np.sin(angle / 2)
        block[orbital, orbital] = cosine
        block[orbital, orbital + 1] = -1j * sine * np.exp(-1j * phase)
        block[orbital + 1, orbital] = -1j * sine * np.exp(1j * phase)
        block[orbital + 1, orbital + 1] = cosine
        matrix = block @ matrix
    return np.diag(np.exp(1j * orbital_phases)) @ matrix


def rotation_angles(
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

    Interleaved Jordan-Wigner leaves the opposite-spin partner between the two
    hopped qubits, so the generator carries a ``Z`` there. ``CZ`` conjugation
    supplies it, mapping ``X_l -> Z_k X_l`` and ``Y_l -> Z_k Y_l``; without it
    the gate is a qubit XY exchange, not a fermionic rotation.
    """
    lower = 2 * orbital + spin
    middle = lower + 1
    upper = 2 * (orbital + 1) + spin
    circuit.cz(middle, upper)
    circuit.append(XXPlusYYGate(angle, phase), [lower, upper])
    circuit.cz(middle, upper)


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


def _lucj_layout(
    n_orb: int, kwargs: dict
) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]], int | None]:
    """A layer's structure: independent spin sectors, Jastrow pairs, and the
    rotation depth."""
    same, opposite = lucj_jastrow_pairs(
        n_orb,
        kwargs.get("same_spin_pairs"),
        kwargs.get("opposite_spin_pairs"),
    )
    n_sectors = 1 if kwargs.get("shared_spin_params") else 2
    return n_sectors, same, opposite, kwargs.get("rotation_depth")


class LUCJAnsatz(Ansatz):
    """Local unitary cluster Jastrow ansatz.

    Each layer applies ``exp(K) exp(iJ) exp(-K)``, where ``K`` is a general
    orbital rotation -- a brick-wall network of Givens rotations, independent
    per spin sector -- and ``J`` is a diagonal Coulomb operator restricted to
    same-orbital opposite-spin pairs plus same-spin neighbours. That restriction
    on ``J`` alone is what makes the ansatz *local*; the rotation is
    unrestricted. Both factors conserve particle number and Sz, which
    sample-based diagonalisation requires -- its symmetry filter checks alpha
    and beta populations separately.

    That default is the flavour the literature uses. Keywords select others:
    ``trailing_rotation`` adds a closing orbital rotation per layer,
    ``shared_spin_params`` ties the two spin sectors together, ``rotation_depth``
    shortens each rotation network, and ``same_spin_pairs`` /
    ``opposite_spin_pairs`` reshape ``J``. See :meth:`build`.

    Assumes the interleaved Jordan-Wigner ordering that
    ``divi.hamiltonians._chem._spo_from_integrals`` produces: qubit ``2p`` is
    the alpha spin-orbital of spatial orbital ``p`` and ``2p + 1`` is its beta
    partner. The Hartree-Fock reference is embedded, so no separate initial
    state is needed.

    ``n_electrons`` must be at most ``n_qubits``. Spin-imbalanced references
    are supported by passing ``n_alpha`` and ``n_beta``; without them the
    closed-shell split ``n_alpha = n_beta = n_electrons // 2`` is used, which
    requires an even ``n_electrons``. A single spatial orbital offers no
    variational freedom: only the on-site diagonal term survives, and its value
    cannot change the state.
    """

    @staticmethod
    def n_params_per_layer(n_qubits: int, **kwargs) -> int:
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
        if n_qubits % 2:
            raise ValueError(
                f"LUCJAnsatz needs an even qubit count (two spin-orbitals per "
                f"spatial orbital); got {n_qubits}."
            )
        n_orb = n_qubits // 2
        n_sectors, same_pairs, opposite_pairs, depth = _lucj_layout(n_orb, kwargs)
        n_params = n_sectors * n_rotation_params(
            n_orb, orbital_phases=False, depth=depth
        )
        n_params += len(opposite_pairs) + n_sectors * len(same_pairs)
        if kwargs.get("trailing_rotation"):
            n_params += n_sectors * n_rotation_params(
                n_orb, orbital_phases=True, depth=depth
            )
        return _require_trainable_params(n_params, LUCJAnsatz.__name__)

    def parameter_frequencies(self, n_qubits: int, **kwargs):
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
        n_sectors, same_pairs, opposite_pairs, depth = _lucj_layout(n_orb, kwargs)
        n_gates = len(_rotation_schedule(n_orb, depth))
        sandwiched = 4 // n_sectors
        trailing = 2 // n_sectors

        frequencies: list[tuple[float, int]] = []
        for _sector in range(n_sectors):
            frequencies += [(0.5, 2 * sandwiched), (1.0, 2 * sandwiched)] * n_gates
        frequencies += [(1.0, 1)] * len(opposite_pairs)
        frequencies += [(1.0, trailing)] * (n_sectors * len(same_pairs))
        if kwargs.get("trailing_rotation"):
            for _sector in range(n_sectors):
                frequencies += [(0.5, 2 * trailing), (1.0, 2 * trailing)] * n_gates
                frequencies += [(1.0, trailing)] * n_orb
        return frequencies

    def build(self, params, n_qubits: int, n_layers: int, **kwargs) -> QuantumCircuit:
        """Build the LUCJ ansatz circuit.

        Args:
            params: Flat parameter array of length
                ``n_layers * n_params_per_layer(n_qubits)``.
            n_qubits: Number of qubits. Must be even.
            n_layers: Number of LUCJ layers.
            **kwargs: Must include ``n_electrons`` (at most ``n_qubits``).
                Optionally accepts ``n_alpha`` / ``n_beta`` to select a
                spin-imbalanced reference determinant, and
                ``trailing_rotation`` (default ``False``) to close each layer
                with an independent orbital rotation rather than the inverse of
                that layer's own. At ``n_layers=1`` that is the truncated LUCJ
                circuit ``exp(K2) exp(-K1) exp(iJ1) exp(K1)`` of
                arXiv:2405.05068 and arXiv:2512.14936; without it, the same
                circuit lacking ``exp(K2)``.

                Three further keywords trade expressiveness for parameters.
                ``shared_spin_params`` (default ``False``) drives both spin sectors
                from one set of rotation and same-spin Coulomb parameters,
                halving the count and imposing spin symmetry.
                ``rotation_depth`` (default ``None``, the full ``n_orb``
                half-layers) truncates each rotation's brick-wall network.
                ``same_spin_pairs`` and ``opposite_spin_pairs`` replace the
                Jastrow's default local pattern -- same-spin nearest neighbours
                and on-site opposite-spin pairs -- with explicit
                ``(orbital, orbital)`` pairs, from ``[]`` to every pair.

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
        n_electrons = _require_n_electrons(kwargs, "LUCJAnsatz")
        if n_electrons > n_qubits:
            raise ValueError(
                f"n_electrons ({n_electrons}) cannot exceed n_qubits ({n_qubits})."
            )
        n_orb, (n_alpha, n_beta) = _resolve_spin_counts(
            n_qubits,
            n_electrons,
            kwargs.get("n_alpha"),
            kwargs.get("n_beta"),
            "LUCJAnsatz",
        )

        trailing_rotation = bool(kwargs.get("trailing_rotation", False))
        per_layer = LUCJAnsatz.n_params_per_layer(n_qubits, **kwargs)
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

        n_sectors, same_pairs, opposite_pairs, depth = _lucj_layout(n_orb, kwargs)
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

            for p, q in opposite_pairs:
                circuit.append(RZZGate(flat[cursor]), [2 * p, 2 * q + 1])
                cursor += 1

            same_start = cursor
            for spin in (0, 1):
                offset = same_start + (spin % n_sectors) * len(same_pairs)
                for index, (p, q) in enumerate(same_pairs):
                    circuit.append(
                        RZZGate(flat[offset + index]), [2 * p + spin, 2 * q + spin]
                    )
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

            if trailing_rotation:
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

        return circuit


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
