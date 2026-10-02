# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Initial-state preparation and block-mixer utilities.

Provides an :class:`InitialState` base class and concrete implementations
consumed by QAOA, VQE and TimeEvolution, and recommended by problems through
``recommended_initial_state``.

Class-based API (preferred)::

    state = WState(block_size=3, n_blocks=4)
    sub_qc = state.build(wires=range(12))

Pass instances directly to algorithm constructors (e.g. ``initial_state=WState(3, 4)``).
"""

from abc import ABC, abstractmethod
from typing import Literal, Sequence

import networkx as nx
import numpy as np
from qiskit.circuit import QuantumCircuit

__all__ = [
    "CustomPerQubitState",
    "DickeState",
    "InitialState",
    "OnesState",
    "SuperpositionState",
    "WState",
    "ZerosState",
]

# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class InitialState(ABC):
    """Abstract base class for initial quantum state preparation.

    Subclasses implement :meth:`build` to return a :class:`~qiskit.circuit.QuantumCircuit`
    of size ``len(wires)`` that prepares the desired state. Qubit ``i`` of
    the returned circuit corresponds positionally to ``wires[i]`` — the
    ``wires`` argument exists purely to let callers communicate domain-level
    labels (e.g. graph node names) that subclasses may need for length /
    shape validation.
    """

    @abstractmethod
    def build(self, wires: Sequence) -> QuantumCircuit:
        """Return a state-preparation circuit on ``len(wires)`` qubits.

        Args:
            wires: Ordered sequence of wire labels (qubit ``i`` ↔ ``wires[i]``).
                May contain non-integer labels (e.g. graph node names);
                only the *length* and ordering matter for circuit emission.

        Returns:
            A :class:`~qiskit.circuit.QuantumCircuit` with ``len(wires)`` qubits.
        """

    @property
    def name(self) -> str:
        """Human-readable name of the initial state."""
        return self.__class__.__name__


# ---------------------------------------------------------------------------
# Concrete implementations
# ---------------------------------------------------------------------------


class ZerosState(InitialState):
    r"""Computational basis state \|00…0⟩ (no gates needed)."""

    def build(self, wires: Sequence) -> QuantumCircuit:
        return QuantumCircuit(len(wires))


class OnesState(InitialState):
    r"""All-ones state \|11…1⟩ via PauliX on every qubit."""

    def build(self, wires: Sequence) -> QuantumCircuit:
        qc = QuantumCircuit(len(wires))
        for q in range(len(wires)):
            qc.x(q)
        return qc


class SuperpositionState(InitialState):
    """Equal superposition via Hadamard on every qubit."""

    def build(self, wires: Sequence) -> QuantumCircuit:
        qc = QuantumCircuit(len(wires))
        for q in range(len(wires)):
            qc.h(q)
        return qc


class CustomPerQubitState(InitialState):
    """Per-qubit state from a string of ``'0'``, ``'1'``, ``'+'``, ``'-'``.

    Args:
        state_string: One character per qubit.
            ``'0'`` → nothing, ``'1'`` → PauliX,
            ``'+'`` → Hadamard, ``'-'`` → PauliX then Hadamard.
    """

    _VALID_CHARS = frozenset("01+-")

    def __init__(self, state_string: str):
        if not state_string or not all(c in self._VALID_CHARS for c in state_string):
            raise ValueError(
                f"state_string must be non-empty and contain only '0', '1', '+', '-', "
                f"got {state_string!r}"
            )
        self.state_string = state_string

    def build(self, wires: Sequence) -> QuantumCircuit:
        n_wires = len(wires)
        if n_wires != len(self.state_string):
            raise ValueError(
                f"state_string length ({len(self.state_string)}) "
                f"must match wire count ({n_wires})."
            )
        qc = QuantumCircuit(n_wires)
        for qubit, char in enumerate(self.state_string):
            if char == "1":
                qc.x(qubit)
            elif char == "+":
                qc.h(qubit)
            elif char == "-":
                qc.x(qubit)
                qc.h(qubit)
        return qc


class WState(InitialState):
    r"""Product of W-states on contiguous qubit blocks.

    Prepares a uniform superposition over one-hot basis states within
    each block::

        |s₀⟩ = |W_{block_size}⟩^{⊗ n_blocks}

    where \|W_n⟩ = (\|10…0⟩ + \|01…0⟩ + … + \|00…1⟩) / √n.

    Useful as the initial state for any one-hot encoded problem
    (routing, assignment, scheduling, graph colouring, etc.).

    Args:
        block_size: Number of qubits per block (≥ 1).
        n_blocks: Number of blocks (≥ 1).
    """

    def __init__(self, block_size: int, n_blocks: int):
        if block_size < 1:
            raise ValueError(f"block_size must be ≥ 1, got {block_size}.")
        if n_blocks < 1:
            raise ValueError(f"n_blocks must be ≥ 1, got {n_blocks}.")
        self.block_size = block_size
        self.n_blocks = n_blocks

    def build(self, wires: Sequence) -> QuantumCircuit:
        """Prepare W-states on each block of qubits.

        Args:
            wires: Must have length ``block_size * n_blocks``.

        Returns:
            A :class:`~qiskit.circuit.QuantumCircuit` with ``block_size * n_blocks`` qubits.
        """
        n_wires = len(wires)
        expected = self.block_size * self.n_blocks
        if n_wires != expected:
            raise ValueError(
                f"Expected {expected} wires ({self.block_size} × {self.n_blocks}), "
                f"got {n_wires}."
            )
        qc = QuantumCircuit(n_wires)
        for b in range(self.n_blocks):
            start = b * self.block_size
            self._w_state(qc, list(range(start, start + self.block_size)))
        return qc

    @staticmethod
    def _w_state(qc: QuantumCircuit, qubits: list[int]) -> None:
        """CRY + CNOT ladder for a single W-state on the given qubits."""
        n = len(qubits)
        qc.x(qubits[0])
        for k in range(n - 1):
            angle = 2 * np.arccos(np.sqrt(1.0 / (n - k)))
            qc.cry(angle, qubits[k], qubits[k + 1])
            qc.cx(qubits[k + 1], qubits[k])


class DickeState(InitialState):
    r"""Dicke state: uniform superposition over bitstrings of fixed Hamming weight.

    Prepares :math:`|D^n_k\rangle` deterministically with the construction of
    Bärtschi and Eidenbenz (`arXiv:1904.07358 <https://arxiv.org/abs/1904.07358>`_),
    using :math:`O(nk)` gates. An XY mixer conserves Hamming weight, so QAOA
    started from this state stays in the weight-:math:`k` subspace of those
    wires.

    Args:
        hamming_weight: Number of ones :math:`k` in every basis state.
        n_qubits: Number of leading wires that carry the Dicke state. Any
            remaining wires are put in equal superposition, the starting state
            QAOA uses for wires driven by an X mixer. Defaults to all wires.

    Raises:
        ValueError: If ``hamming_weight`` is negative, or at build time if it
            exceeds ``n_qubits`` or ``n_qubits`` exceeds the wire count.
    """

    def __init__(self, hamming_weight: int, n_qubits: int | None = None):
        if hamming_weight < 0:
            raise ValueError(f"hamming_weight must be ≥ 0, got {hamming_weight}.")
        if n_qubits is not None and n_qubits < 1:
            raise ValueError(f"n_qubits must be ≥ 1, got {n_qubits}.")
        self.hamming_weight = hamming_weight
        self.n_qubits = n_qubits

    def build(self, wires: Sequence) -> QuantumCircuit:
        n_wires = len(wires)
        n = n_wires if self.n_qubits is None else self.n_qubits
        k = self.hamming_weight
        if n > n_wires:
            raise ValueError(f"n_qubits ({n}) exceeds the wire count ({n_wires}).")
        if k > n:
            raise ValueError(f"hamming_weight ({k}) exceeds n_qubits ({n}).")

        qc = QuantumCircuit(n_wires)
        for q in range(n - k, n):
            qc.x(q)
        if 0 < k < n:
            for m in range(n, k, -1):
                self._split_and_cyclic_shift(qc, m, k)
            for m in range(k, 1, -1):
                self._split_and_cyclic_shift(qc, m, m - 1)
        for q in range(n, n_wires):
            qc.h(q)
        return qc

    @staticmethod
    def _split_and_cyclic_shift(qc: QuantumCircuit, m: int, length: int) -> None:
        """SCS_{m,length} on qubits ``m-length-1 … m-1``."""
        theta = 2 * np.arccos(np.sqrt(1 / m))
        qc.cx(m - 2, m - 1)
        qc.cry(theta, m - 1, m - 2)
        qc.cx(m - 2, m - 1)
        for i in range(2, length + 1):
            a, b, t = m - i - 1, m - i, m - 1
            theta = 2 * np.arccos(np.sqrt(i / m))
            qc.cx(a, t)
            # Doubly-controlled RY on ``a`` (controls ``t``, ``b``) from CRY and CX.
            qc.cry(theta / 2, b, a)
            qc.cx(t, b)
            qc.cry(-theta / 2, b, a)
            qc.cx(t, b)
            qc.cry(theta / 2, t, a)
            qc.cx(a, t)


# ---------------------------------------------------------------------------
# Block-XY mixer graph (for use with ``xy_mixer``)
# ---------------------------------------------------------------------------


def build_block_xy_mixer_graph(
    block_size: int,
    n_blocks: int,
    wires: Sequence[int],
    connectivity: Literal["complete", "path", "ring"] = "complete",
) -> nx.Graph:
    """Build the connectivity graph for a block-XY mixer.

    Returns a ``networkx.Graph`` whose edges define the XY coupling
    terms within each qubit block.  Pass the result to
    :func:`~divi.hamiltonians.xy_mixer` to obtain the mixer
    Hamiltonian as a :class:`~qiskit.quantum_info.SparsePauliOp`.

    Args:
        block_size: Qubits per block (≥ 2 for mixing to occur).
        n_blocks: Number of blocks.
        wires: Must have length ``block_size * n_blocks``.
        connectivity: Intra-block coupling pattern.

            * ``"complete"`` (default) — all-to-all edges within each
              block.  Matches the CE-QAOA mixer from
              `arXiv:2511.14296 <https://arxiv.org/abs/2511.14296>`_
              and provides a constant spectral gap on the
              one-excitation sector.
            * ``"path"`` — nearest-neighbour (linear chain) edges
              within each block.  Uses O(n) terms instead of O(n²),
              which may be preferable on hardware with limited
              connectivity, at the cost of a weaker spectral gap.
            * ``"ring"`` — the path closed into a cycle (a path for blocks of
              two qubits). Uses O(n) terms, like ``"path"``, while connecting
              the ends of each block. Every connectivity conserves Hamming
              weight within each block; ``"complete"`` has every Dicke state
              (:class:`DickeState`) as an eigenstate, ``"path"`` and ``"ring"``
              in general do not.

    Returns:
        ``networkx.Graph`` for :func:`~divi.hamiltonians.xy_mixer`.
    """
    wires = list(wires)
    expected = block_size * n_blocks
    if len(wires) != expected:
        raise ValueError(
            f"Expected {expected} wires ({block_size} × {n_blocks}), "
            f"got {len(wires)}."
        )

    g = nx.Graph()
    g.add_nodes_from(wires)
    for b in range(n_blocks):
        start = b * block_size
        block_wires = wires[start : start + block_size]
        if connectivity == "complete":
            g.update(nx.complete_graph(block_wires))
        elif connectivity == "ring" and len(block_wires) > 2:
            g.update(nx.cycle_graph(block_wires))
        else:
            g.update(nx.path_graph(block_wires))
    return g
