# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Built-in QAOA / VQE ansätze.

Every ``Ansatz.build`` creates and returns a :class:`~qiskit.circuit.QuantumCircuit`.
Chemistry ansätze (``UCCSDAnsatz``, ``HartreeFockAnsatz``,
``QCCAnsatz``) source excitation / Hartree-Fock data from ``pennylane.qchem``
and route the PL gates through the local PL → Qiskit converter; consumers
always see Qiskit instructions.
"""

import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence
from typing import Literal
from warnings import warn

import numpy as np
import pennylane as qp
from qiskit.circuit import Gate, QuantumCircuit

from divi.circuits._conversions import _qscript_to_dag
from divi.hamiltonians._term_ops import _HALF_PI


def _require_trainable_params(n_params: int, ansatz_name: str) -> int:
    if n_params <= 0:
        raise ValueError(
            f"{ansatz_name} must define at least one trainable parameter. "
            "Parameter-free circuits are not supported."
        )
    return n_params


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
        for cls in (*gate_sequence, *([entangler] if entangler is not None else ())):
            if not (isinstance(cls, type) and issubclass(cls, Gate)):
                raise TypeError(
                    f"Expected a Qiskit Gate subclass (e.g. RXGate, CXGate), got {cls!r}."
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


class QAOAAnsatz(Ansatz):
    """
    QAOA-style ansatz using PennyLane's QAOAEmbedding.

    Implements a parameterized ansatz based on the Quantum Approximate Optimization
    Algorithm structure, alternating between problem and mixer Hamiltonians.
    """

    @staticmethod
    def n_params_per_layer(n_qubits: int, **kwargs) -> int:
        """``2 * n_qubits`` — one ``γ`` and one ``β`` per qubit per layer."""
        return _require_trainable_params(2 * n_qubits, QAOAAnsatz.__name__)

    def build(self, params, n_qubits: int, n_layers: int, **kwargs) -> QuantumCircuit:
        """
        Build the QAOA ansatz circuit.

        Args:
            params: Parameter array to use for the ansatz.
            n_qubits (int): Number of qubits.
            n_layers (int): Number of QAOA layers.
            **kwargs: Additional unused arguments.

        Returns:
            QuantumCircuit: Qiskit circuit implementing the QAOA ansatz.
        """
        qc = QuantumCircuit(n_qubits)
        # Initial superposition.
        for q in range(n_qubits):
            qc.h(q)

        params = np.asarray(params, dtype=object).reshape(n_layers, 2 * n_qubits)
        for layer in range(n_layers):
            layer_params = params[layer]
            gammas = layer_params[:n_qubits]
            betas = layer_params[n_qubits:]
            # Cost: ZZ on adjacent pairs + RZ per qubit.
            for q in range(n_qubits - 1):
                _emit_two_qubit_pauli_rot(qc, "ZZ", gammas[q], q, q + 1)
            for q in range(n_qubits):
                qc.rz(gammas[q], q)
            # Mixer.
            for q in range(n_qubits):
                qc.ry(betas[q], q)

        return qc


# --- Chemistry Ansätze ---


class UCCSDAnsatz(Ansatz):
    """
    Unitary Coupled Cluster Singles and Doubles (UCCSD) ansatz.

    This ansatz is specifically designed for quantum chemistry calculations,
    implementing the UCCSD approximation which includes all single and double
    electron excitations from a reference state.
    """

    @staticmethod
    def n_params_per_layer(n_qubits: int, **kwargs) -> int:
        """``len(s_wires) + len(d_wires)`` from ``qp.qchem.excitations`` for
        the given ``n_electrons`` (required kwarg)."""
        n_electrons = kwargs.pop("n_electrons")
        singles, doubles = qp.qchem.excitations(n_electrons, n_qubits)
        s_wires, d_wires = qp.qchem.excitations_to_wires(singles, doubles)
        n_params = len(s_wires) + len(d_wires)
        return _require_trainable_params(n_params, UCCSDAnsatz.__name__)

    def build(self, params, n_qubits: int, n_layers: int, **kwargs) -> QuantumCircuit:
        n_electrons = kwargs.pop("n_electrons")
        singles, doubles = qp.qchem.excitations(n_electrons, n_qubits)
        s_wires, d_wires = qp.qchem.excitations_to_wires(singles, doubles)
        hf_state = qp.qchem.hf_state(n_electrons, n_qubits)
        params = np.asarray(params, dtype=object).reshape(n_layers, -1)

        pl_ops = qp.UCCSD.compute_decomposition(
            params,
            wires=range(n_qubits),
            s_wires=s_wires,
            d_wires=d_wires,
            init_state=hf_state,
            n_repeats=n_layers,
        )
        return _pl_ops_to_qc(pl_ops, n_qubits)


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
        n_electrons = kwargs.pop("n_electrons")
        singles, doubles = qp.qchem.excitations(n_electrons, n_qubits)
        n_params = len(singles) + len(doubles)
        return _require_trainable_params(n_params, HartreeFockAnsatz.__name__)

    def build(self, params, n_qubits: int, n_layers: int, **kwargs) -> QuantumCircuit:
        n_electrons = kwargs.pop("n_electrons")
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
        n_electrons = kwargs.pop("n_electrons")
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
