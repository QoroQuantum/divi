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
import pennylane as qp
from qiskit.circuit import Gate, QuantumCircuit
from qiskit.circuit.library import RXGate, RYGate, RZGate, RZZGate, XXPlusYYGate

from divi.circuits._conversions import _qscript_to_dag
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
    try:
        # pyrefly: ignore[missing-import]  # optional ``chem`` extra
        from qiskit_nature.second_q import mappers

        # pyrefly: ignore[missing-import]  # optional ``chem`` extra
        from qiskit_nature.second_q.circuit import library
    except ImportError as exc:
        raise ImportError(
            "UCCSDAnsatz requires the 'chem' extra; "
            "install it with `pip install qoro-divi[chem]`."
        ) from exc

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
    parameterized single and double excitation gates. It's a simplified
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


class LUCJAnsatz(Ansatz):
    """Local unitary cluster Jastrow ansatz.

    Each layer applies the LUCJ structure ``exp(K) exp(iJ) exp(-K)`` with
    locality restrictions: ``K`` is a nearest-neighbor Givens chain confined to
    each spin sector, and ``J`` is a diagonal Coulomb operator on same-orbital
    opposite-spin pairs plus same-spin neighbors. Both factors conserve
    particle number and Sz, which sample-based diagonalization requires — its
    symmetry filter checks alpha and beta populations separately.

    Assumes the interleaved Jordan-Wigner ordering that
    ``divi.hamiltonians._chem._spo_from_integrals`` produces: qubit ``2p`` is
    the alpha spin-orbital of spatial orbital ``p`` and ``2p + 1`` is its beta
    partner. The Hartree-Fock reference is embedded, so no separate initial
    state is needed.

    ``n_electrons`` must be at most ``n_qubits``. Spin-imbalanced references
    are supported by passing ``n_alpha`` and ``n_beta``; without them the
    closed-shell split ``n_alpha = n_beta = n_electrons // 2`` is used, which
    requires an even ``n_electrons``. A two-qubit register (one spatial
    orbital) has a single diagonal on-site term and no hopping, so its output
    is fixed regardless of the parameter value — it offers no variational
    freedom.

    Measured limitation: on a minimal two-orbital, one-alpha-one-beta
    fragment, exact optimization of this ansatz's parameters (global search,
    no shot noise) plateaus about 20 mHa above the exact ground state.
    Additional layers do not lift the plateau: 1, 2, and 3 layers converge to
    the same energy to 8+ significant figures.
    """

    @staticmethod
    def n_params_per_layer(n_qubits: int, **kwargs) -> int:
        """Per-layer parameter count.

        ``2 * (n_orb - 1)`` hopping angles, ``n_orb`` on-site Coulomb terms,
        and ``2 * (n_orb - 1)`` same-spin-neighbor Coulomb terms, where
        ``n_orb = n_qubits // 2``. Collapses to ``1`` for a single spatial
        orbital (``n_qubits == 2``), where only the on-site term applies.
        """
        if n_qubits % 2:
            raise ValueError(
                f"LUCJAnsatz needs an even qubit count (two spin-orbitals per "
                f"spatial orbital); got {n_qubits}."
            )
        n_orb = n_qubits // 2
        if n_orb == 1:
            # One spatial orbital: no hopping or same-spin neighbors, only the
            # on-site opposite-spin Coulomb term.
            return _require_trainable_params(1, LUCJAnsatz.__name__)
        n_params = 2 * (n_orb - 1) + n_orb + 2 * (n_orb - 1)
        return _require_trainable_params(n_params, LUCJAnsatz.__name__)

    def build(self, params, n_qubits: int, n_layers: int, **kwargs) -> QuantumCircuit:
        """Build the LUCJ ansatz circuit.

        Args:
            params: Flat parameter array of length
                ``n_layers * n_params_per_layer(n_qubits)``.
            n_qubits: Number of qubits. Must be even.
            n_layers: Number of LUCJ layers.
            **kwargs: Must include ``n_electrons`` (at most ``n_qubits``).
                Optionally accepts ``n_alpha`` / ``n_beta`` to select a
                spin-imbalanced reference determinant.

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

        per_layer = LUCJAnsatz.n_params_per_layer(n_qubits)
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

        hop_pairs = [
            (2 * p + spin, 2 * (p + 1) + spin)
            for spin in (0, 1)
            for p in range(n_orb - 1)
        ]

        for layer in range(n_layers):
            offset = layer * per_layer
            cursor = offset

            hop_angles = []
            for lower, upper in hop_pairs:
                angle = flat[cursor]
                cursor += 1
                hop_angles.append(angle)
                circuit.append(XXPlusYYGate(angle, 0.0), [lower, upper])

            for p in range(n_orb):
                circuit.append(RZZGate(flat[cursor]), [2 * p, 2 * p + 1])
                cursor += 1

            for lower, upper in hop_pairs:
                circuit.append(RZZGate(flat[cursor]), [lower, upper])
                cursor += 1

            for (lower, upper), angle in zip(reversed(hop_pairs), reversed(hop_angles)):
                circuit.append(XXPlusYYGate(-angle, 0.0), [lower, upper])

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
