# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Graph problem classes for QAOA."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pennylane as qml
import pennylane.qaoa as pqaoa

from divi.hamiltonians import _clean_hamiltonian, _is_empty_hamiltonian
from divi.qprog.algorithms._initial_state import (
    InitialState,
    OnesState,
    SuperpositionState,
    ZerosState,
)
from divi.qprog.problems._base import QAOAProblem
from divi.typing import GraphProblemTypes


class _GraphProblemBase(QAOAProblem):
    """Shared logic for PennyLane-backed graph problems.

    Subclasses only need to set ``_pl_func_name`` and the two
    ``_*_state_cls`` class attributes, then call ``super().__init__``.
    """

    _pl_func_name: str
    _constrained_state_cls: type[InitialState]
    _unconstrained_state_cls: type[InitialState]

    def __init__(self, graph: GraphProblemTypes, *, is_constrained: bool = True):
        self._graph = graph
        self._is_constrained = is_constrained

        cost_ham, self._mixer_hamiltonian, *self._metadata = self._resolve(
            graph, is_constrained
        )

        cleaned, ham_constant = _clean_hamiltonian(cost_ham)
        if _is_empty_hamiltonian(cleaned):
            raise ValueError("Hamiltonian contains only constant terms.")

        self._cost_hamiltonian = cleaned
        self._loss_constant = ham_constant
        self._wire_labels = tuple(cleaned.wires)
        self._initial_state = (
            self._constrained_state_cls
            if is_constrained
            else self._unconstrained_state_cls
        )()

    @classmethod
    def _resolve(cls, graph, is_constrained):
        """Call the PennyLane QAOA function for this problem type."""
        pl_fn = getattr(pqaoa, cls._pl_func_name)
        try:
            return pl_fn(graph, constrained=is_constrained)
        except TypeError:
            return pl_fn(graph)

    @property
    def graph(self) -> GraphProblemTypes:
        """The underlying graph."""
        return self._graph

    @property
    def cost_hamiltonian(self) -> qml.operation.Operator:
        return self._cost_hamiltonian

    @property
    def mixer_hamiltonian(self) -> qml.operation.Operator:
        return self._mixer_hamiltonian

    @property
    def loss_constant(self) -> float:
        return self._loss_constant

    @property
    def recommended_initial_state(self) -> InitialState:
        return self._initial_state

    @property
    def decode_fn(self) -> Callable[[str], Any]:
        wires = self._wire_labels

        def _decode(bitstring: str) -> list:
            return [
                wires[idx]
                for idx, bit in enumerate(bitstring)
                if bit == "1" and idx < len(wires)
            ]

        return _decode

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata[0] if self._metadata else {}


class MaxCutProblem(_GraphProblemBase):
    """MaxCut problem on a graph.

    Args:
        graph: NetworkX or RustworkX graph.
    """

    _pl_func_name = "maxcut"
    _constrained_state_cls = SuperpositionState
    _unconstrained_state_cls = SuperpositionState


class MaxCliqueProblem(_GraphProblemBase):
    """Max clique problem on a graph.

    Args:
        graph: NetworkX or RustworkX graph.
        is_constrained: Use constrained mixer. Defaults to True.
    """

    _pl_func_name = "max_clique"
    _constrained_state_cls = ZerosState
    _unconstrained_state_cls = SuperpositionState


class MaxIndependentSetProblem(_GraphProblemBase):
    """Max independent set problem on a graph.

    Args:
        graph: NetworkX or RustworkX graph.
        is_constrained: Use constrained mixer. Defaults to True.
    """

    _pl_func_name = "max_independent_set"
    _constrained_state_cls = ZerosState
    _unconstrained_state_cls = SuperpositionState


class MinVertexCoverProblem(_GraphProblemBase):
    """Min vertex cover problem on a graph.

    Args:
        graph: NetworkX or RustworkX graph.
        is_constrained: Use constrained mixer. Defaults to True.
    """

    _pl_func_name = "min_vertex_cover"
    _constrained_state_cls = OnesState
    _unconstrained_state_cls = SuperpositionState


class MaxWeightCycleProblem(_GraphProblemBase):
    """Max weight cycle problem on a graph.

    Args:
        graph: NetworkX or RustworkX graph.
        is_constrained: Use constrained mixer. Defaults to True.
    """

    _pl_func_name = "max_weight_cycle"
    _constrained_state_cls = SuperpositionState
    _unconstrained_state_cls = SuperpositionState


class EdgePartitioningProblem(_GraphProblemBase):
    """Placeholder for edge-partitioning problems.

    Edge partitioning operates on directed graphs and uses weak connectivity.
    The Hamiltonian construction is not yet implemented — this class exists
    so that :class:`GraphPartitioningQAOA` can detect the partitioning mode
    from the Problem type.

    .. note:: This is incomplete.  Passing an ``EdgePartitioningProblem``
       to :class:`QAOA` directly will raise ``AttributeError`` because
       ``_pl_func_name`` is not set.
    """

    _constrained_state_cls = SuperpositionState
    _unconstrained_state_cls = SuperpositionState

    #: Signals that the graph is directed and needs weak-connectivity checks.
    is_edge_problem: bool = True

    def __init__(self, graph: GraphProblemTypes):
        # Skip _GraphProblemBase.__init__ — no PennyLane function to call.
        self._graph = graph
        self._is_constrained = True

    @property
    def cost_hamiltonian(self):
        raise NotImplementedError(
            "EdgePartitioningProblem does not yet support Hamiltonian construction."
        )

    @property
    def mixer_hamiltonian(self):
        raise NotImplementedError(
            "EdgePartitioningProblem does not yet support Hamiltonian construction."
        )

    @property
    def loss_constant(self):
        raise NotImplementedError(
            "EdgePartitioningProblem does not yet support Hamiltonian construction."
        )

    @property
    def decode_fn(self):
        raise NotImplementedError(
            "EdgePartitioningProblem does not yet support decoding."
        )
