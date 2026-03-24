# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Initial-state preparation and block-mixer utilities.

Provides an :class:`InitialState` base class and concrete implementations
consumed by QAOA, VQE, TimeEvolution, and any future algorithm that
prepends an initial-state layer to its circuit.

Class-based API (preferred)::

    state = WState(block_size=3, n_blocks=4)
    ops = state.build(wires=range(12))

Pass instances directly to algorithm constructors (e.g. ``initial_state=WState(3, 4)``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal, Sequence

import networkx as nx
import numpy as np
import pennylane as qml

# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class InitialState(ABC):
    """Abstract base class for initial quantum state preparation.

    Subclasses implement :meth:`build` to return a list of PennyLane
    operations that prepare the desired state on the given wires.
    """

    @abstractmethod
    def build(self, wires: Sequence[int]) -> list[qml.operation.Operator]:
        """Return gate operations that prepare this state on *wires*.

        Args:
            wires: Ordered sequence of wire labels.

        Returns:
            List of PennyLane operations.
        """

    @property
    def name(self) -> str:
        """Human-readable name of the initial state."""
        return self.__class__.__name__


# ---------------------------------------------------------------------------
# Concrete implementations
# ---------------------------------------------------------------------------


class ZerosState(InitialState):
    """Computational basis state |00…0⟩ (no gates needed)."""

    def build(self, wires: Sequence[int]) -> list[qml.operation.Operator]:
        return []


class OnesState(InitialState):
    """All-ones state |11…1⟩ via PauliX on every qubit."""

    def build(self, wires: Sequence[int]) -> list[qml.operation.Operator]:
        return [qml.PauliX(wires=w) for w in wires]


class SuperpositionState(InitialState):
    """Equal superposition via Hadamard on every qubit."""

    def build(self, wires: Sequence[int]) -> list[qml.operation.Operator]:
        return [qml.Hadamard(wires=w) for w in wires]


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

    def build(self, wires: Sequence[int]) -> list[qml.operation.Operator]:
        wires = list(wires)
        if len(wires) != len(self.state_string):
            raise ValueError(
                f"state_string length ({len(self.state_string)}) "
                f"must match wire count ({len(wires)})."
            )
        ops: list[qml.operation.Operator] = []
        for w, char in zip(wires, self.state_string):
            if char == "1":
                ops.append(qml.PauliX(wires=w))
            elif char == "+":
                ops.append(qml.Hadamard(wires=w))
            elif char == "-":
                ops.append(qml.PauliX(wires=w))
                ops.append(qml.Hadamard(wires=w))
        return ops


class WState(InitialState):
    """Product of W-states on contiguous qubit blocks.

    Prepares a uniform superposition over one-hot basis states within
    each block::

        |s₀⟩ = |W_{block_size}⟩^{⊗ n_blocks}

    where |W_n⟩ = (|10…0⟩ + |01…0⟩ + … + |00…1⟩) / √n.

    Useful as the initial state for any one-hot encoded problem
    (routing, assignment, scheduling, graph coloring, etc.).

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

    def build(self, wires: Sequence[int]) -> list[qml.operation.Operator]:
        """Prepare W-states on each block of *wires*.

        Args:
            wires: Must have length ``block_size * n_blocks``.
        """
        wires = list(wires)
        expected = self.block_size * self.n_blocks
        if len(wires) != expected:
            raise ValueError(
                f"Expected {expected} wires ({self.block_size} × {self.n_blocks}), "
                f"got {len(wires)}."
            )
        ops: list[qml.operation.Operator] = []
        for b in range(self.n_blocks):
            start = b * self.block_size
            ops.extend(self._w_state(wires[start : start + self.block_size]))
        return ops

    @staticmethod
    def _w_state(wires: list[int]) -> list[qml.operation.Operator]:
        """CRY + CNOT ladder for a single W-state on *wires*."""
        n = len(wires)
        ops: list[qml.operation.Operator] = [qml.PauliX(wires=wires[0])]
        for k in range(n - 1):
            angle = 2 * np.arccos(np.sqrt(1.0 / (n - k)))
            ops.append(qml.CRY(phi=angle, wires=[wires[k], wires[k + 1]]))
            ops.append(qml.CNOT(wires=[wires[k + 1], wires[k]]))
        return ops


# ---------------------------------------------------------------------------
# Block-XY mixer graph (for use with pennylane.qaoa.xy_mixer)
# ---------------------------------------------------------------------------


def build_block_xy_mixer_graph(
    block_size: int,
    n_blocks: int,
    wires: Sequence[int],
    connectivity: Literal["complete", "path"] = "complete",
) -> nx.Graph:
    """Build the connectivity graph for a block-XY mixer.

    Returns a ``networkx.Graph`` whose edges define the XY coupling
    terms within each qubit block.  Pass the result to
    ``pennylane.qaoa.xy_mixer()`` to obtain the mixer Hamiltonian.

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

    Returns:
        ``networkx.Graph`` for ``pennylane.qaoa.xy_mixer()``.
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
        else:
            g.update(nx.path_graph(block_wires))
    return g
