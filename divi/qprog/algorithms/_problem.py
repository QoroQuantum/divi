# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Problem protocol and concrete problem classes for QAOA.

Defines the :class:`Problem` protocol that all QAOA-compatible problems
must implement, along with concrete classes for common graph and binary
optimization problems.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Literal

import matplotlib.pyplot as plt
import networkx as nx
import pennylane as qml
import pennylane.qaoa as pqaoa

from divi.hamiltonians import (
    _clean_hamiltonian,
    _is_empty_hamiltonian,
    normalize_binary_polynomial_problem,
    qubo_to_ising,
)
from divi.qprog.algorithms._initial_state import (
    InitialState,
    OnesState,
    SuperpositionState,
    ZerosState,
)
from divi.typing import (
    GraphProblemTypes,
    HUBOProblemTypes,
    QUBOProblemTypes,
)

# ---------------------------------------------------------------------------
# Problem protocol
# ---------------------------------------------------------------------------


class Problem(ABC):
    """Base class for all QAOA-compatible problems.

    Subclasses must implement the four abstract properties that provide
    the ingredients QAOA needs.  Default implementations of
    ``recommended_initial_state``, ``is_feasible``, ``repair``, and
    ``compute_energy`` are provided and can be overridden.
    """

    @property
    @abstractmethod
    def cost_hamiltonian(self) -> qml.operation.Operator:
        """The cost Hamiltonian encoding the optimisation objective."""

    @property
    @abstractmethod
    def mixer_hamiltonian(self) -> qml.operation.Operator:
        """The mixer Hamiltonian for exploring the solution space."""

    @property
    @abstractmethod
    def loss_constant(self) -> float:
        """Constant offset added back to the expectation value."""

    @property
    @abstractmethod
    def decode_fn(self) -> Callable[[str], Any]:
        """Map a measurement bitstring to a domain-level solution."""

    @property
    def recommended_initial_state(self) -> InitialState:
        """Recommended initial quantum state for this problem.

        Defaults to :class:`SuperpositionState` (ground state of the
        standard X mixer).
        """
        return SuperpositionState()

    def is_feasible(self, bitstring: str) -> bool:
        """Check whether a bitstring represents a feasible solution.

        Defaults to ``True`` (unconstrained).
        """
        return True

    def repair_infeasible_bitstring(
        self, bitstring: str
    ) -> tuple[str, Any, float | None]:
        """Repair an infeasible bitstring into a feasible one.

        Returns:
            A three-element tuple ``(repaired_bitstring, decoded, energy)``:

            - **repaired_bitstring**: The feasible bitstring after repair.
            - **decoded**: Domain-level solution (e.g. tour list, routes),
              or ``None`` if unavailable.
            - **energy**: Objective value of the repaired solution,
              or ``None`` if unknown.

        The default implementation returns the bitstring unchanged.
        """
        return bitstring, None, None

    def compute_energy(self, bitstring: str) -> float | None:
        """Evaluate the objective energy for a bitstring.

        Defaults to ``None`` (unknown).
        """
        return None


# ---------------------------------------------------------------------------
# Graph problem classes
# ---------------------------------------------------------------------------


class _GraphProblemBase(Problem):
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


def draw_graph_solution_nodes(main_graph: nx.Graph, partition_nodes):
    """Visualize a graph with solution nodes highlighted.

    Draws the graph with nodes colored to distinguish solution nodes (red) from
    other nodes (light blue).

    Args:
        main_graph (nx.Graph): NetworkX graph to visualize.
        partition_nodes: Collection of node indices that are part of the solution.
    """
    node_colors = [
        "red" if node in partition_nodes else "lightblue" for node in main_graph.nodes()
    ]

    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(main_graph)
    nx.draw_networkx_nodes(main_graph, pos, node_color=node_colors, node_size=500)
    nx.draw_networkx_edges(main_graph, pos)
    nx.draw_networkx_labels(main_graph, pos, font_size=10, font_weight="bold")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Binary optimisation (QUBO / HUBO)
# ---------------------------------------------------------------------------


class BinaryOptimizationProblem(Problem):
    """Generic QUBO or HUBO problem.

    Normalises the input, converts to an Ising Hamiltonian, and provides
    a standard X-mixer with equal superposition initial state.

    Args:
        problem: QUBO matrix, BinaryQuadraticModel, HUBO dict, or
            BinaryPolynomial.
        hamiltonian_builder: Ising conversion backend (``"native"`` or
            ``"quadratized"``).
        quadratization_strength: Penalty strength for quadratization.
    """

    def __init__(
        self,
        problem: QUBOProblemTypes | HUBOProblemTypes,
        *,
        hamiltonian_builder: Literal["native", "quadratized"] = "native",
        quadratization_strength: float = 10.0,
    ):
        self._canonical_problem = normalize_binary_polynomial_problem(problem)
        self._ising = qubo_to_ising(
            problem,
            hamiltonian_builder=hamiltonian_builder,
            quadratization_strength=quadratization_strength,
        )
        self._mixer_hamiltonian = pqaoa.x_mixer(range(self._ising.n_qubits))

    @property
    def cost_hamiltonian(self) -> qml.operation.Operator:
        return self._ising.cost_hamiltonian

    @property
    def mixer_hamiltonian(self) -> qml.operation.Operator:
        return self._mixer_hamiltonian

    @property
    def loss_constant(self) -> float:
        return self._ising.loss_constant

    @property
    def decode_fn(self) -> Callable[[str], Any]:
        base_decode = self._ising.encoding.decode_fn
        vo = self._canonical_problem.variable_order

        if vo != tuple(range(self._canonical_problem.n_vars)):

            def _decode_with_names(bitstring: str) -> dict | None:
                decoded = base_decode(bitstring)
                if decoded is None:
                    return None
                return dict(zip(vo, decoded))

            return _decode_with_names
        return base_decode

    @property
    def metadata(self) -> dict[str, Any]:
        return self._ising.encoding.metadata or {}

    @property
    def canonical_problem(self):
        """The normalised ``BinaryPolynomialProblem``."""
        return self._canonical_problem
