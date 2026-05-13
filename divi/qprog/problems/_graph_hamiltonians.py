# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Graph-problem cost and mixer Hamiltonians as ``SparsePauliOp``.

Mirrors the formulations in :mod:`pennylane.qaoa.cost` and
:mod:`pennylane.qaoa.cycle` while emitting Qiskit ``SparsePauliOp`` objects.
"""

from typing import TypeAlias

import networkx as nx
import numpy as np
import rustworkx as rx
from qiskit.quantum_info import SparsePauliOp

from divi.hamiltonians import (
    bit_driver_spo,
    bit_flip_mixer_spo,
    edge_driver_spo,
    x_mixer_spo,
)
from divi.hamiltonians._mixers import multi_pauli_label, single_pauli_label

GraphLike: TypeAlias = nx.Graph | rx.PyGraph
DiGraphLike: TypeAlias = nx.DiGraph | rx.PyDiGraph


def _to_nx_graph(graph: GraphLike) -> nx.Graph:
    """Return a ``networkx`` undirected graph view of ``graph``.

    rustworkx ``PyGraph`` instances are converted by remapping their internal
    integer indices to the user-facing node values in ``graph.nodes()``.
    """
    if isinstance(graph, nx.Graph) and not isinstance(graph, nx.DiGraph):
        return graph
    if isinstance(graph, rx.PyGraph):
        nodes = list(graph.nodes())
        out = nx.Graph()
        out.add_nodes_from(nodes)
        for left, right in graph.edge_list():
            out.add_edge(nodes[left], nodes[right])
        return out
    raise TypeError(
        f"Expected an undirected graph (nx.Graph or rx.PyGraph), got "
        f"{type(graph).__name__}."
    )


def _to_nx_digraph(graph: DiGraphLike) -> nx.DiGraph:
    """Return a ``networkx`` directed graph view of ``graph``.

    Edge weights stored under the ``"weight"`` attribute are preserved.
    """
    if isinstance(graph, nx.DiGraph):
        return graph
    if isinstance(graph, rx.PyDiGraph):
        nodes = list(graph.nodes())
        out = nx.DiGraph()
        out.add_nodes_from(nodes)
        for left, right, payload in graph.weighted_edge_list():
            kwargs: dict = {}
            if isinstance(payload, dict):
                kwargs.update(payload)
            elif payload is not None:
                kwargs["weight"] = payload
            out.add_edge(nodes[left], nodes[right], **kwargs)
        return out
    raise TypeError(
        f"Expected a directed graph (nx.DiGraph or rx.PyDiGraph), got "
        f"{type(graph).__name__}."
    )


def _node_to_qubit(graph: nx.Graph) -> dict:
    """Map graph nodes to dense 0-indexed qubit positions in node-iteration order."""
    return {node: i for i, node in enumerate(graph.nodes())}


def _node_to_qubit_undirected(graph: GraphLike) -> tuple[nx.Graph, dict]:
    """Convert ``graph`` to an int-qubit-indexed nx.Graph plus the node→qubit map."""
    nx_graph = _to_nx_graph(graph)
    mapping = _node_to_qubit(nx_graph)
    relabelled = nx.relabel_nodes(nx_graph, mapping, copy=True)
    return relabelled, mapping


# ---------------------------------------------------------------------------
# Cost-Hamiltonian builders for graph problems.
# ---------------------------------------------------------------------------


def maxcut_hamiltonians(graph: GraphLike) -> tuple[SparsePauliOp, SparsePauliOp]:
    """Cost and mixer for MaxCut.

    .. math::
        H_C = \\frac{1}{2} \\sum_{(i,j) \\in E} (Z_i Z_j - I), \\quad
        H_M = \\sum_v X_v
    """
    relabelled, _ = _node_to_qubit_undirected(graph)
    n_qubits = relabelled.number_of_nodes()
    cost = edge_driver_spo(
        relabelled, ["10", "01"], n_qubits=n_qubits
    ) + SparsePauliOp.from_list([("I" * n_qubits, -0.5 * relabelled.number_of_edges())])
    mixer = x_mixer_spo(n_qubits)
    return cost, mixer


def max_independent_set_hamiltonians(
    graph: GraphLike, *, constrained: bool = True
) -> tuple[SparsePauliOp, SparsePauliOp]:
    """Cost and mixer for Maximum Independent Set."""
    relabelled, _ = _node_to_qubit_undirected(graph)
    n_qubits = relabelled.number_of_nodes()
    if constrained:
        cost = bit_driver_spo(n_qubits, b=1)
        mixer = bit_flip_mixer_spo(relabelled, b=0)
        return cost, mixer
    cost = 3.0 * edge_driver_spo(
        relabelled, ["10", "01", "00"], n_qubits=n_qubits
    ) + bit_driver_spo(n_qubits, b=1)
    return cost, x_mixer_spo(n_qubits)


def min_vertex_cover_hamiltonians(
    graph: GraphLike, *, constrained: bool = True
) -> tuple[SparsePauliOp, SparsePauliOp]:
    """Cost and mixer for Minimum Vertex Cover."""
    relabelled, _ = _node_to_qubit_undirected(graph)
    n_qubits = relabelled.number_of_nodes()
    if constrained:
        cost = bit_driver_spo(n_qubits, b=0)
        mixer = bit_flip_mixer_spo(relabelled, b=1)
        return cost, mixer
    cost = 3.0 * edge_driver_spo(
        relabelled, ["11", "10", "01"], n_qubits=n_qubits
    ) + bit_driver_spo(n_qubits, b=0)
    return cost, x_mixer_spo(n_qubits)


def max_clique_hamiltonians(
    graph: GraphLike, *, constrained: bool = True
) -> tuple[SparsePauliOp, SparsePauliOp]:
    """Cost and mixer for Maximum Clique. The mixer acts on the complement graph."""
    relabelled, _ = _node_to_qubit_undirected(graph)
    n_qubits = relabelled.number_of_nodes()
    complement = nx.complement(relabelled)
    if constrained:
        cost = bit_driver_spo(n_qubits, b=1)
        mixer = bit_flip_mixer_spo(complement, b=0)
        return cost, mixer
    cost = 3.0 * edge_driver_spo(
        complement, ["10", "01", "00"], n_qubits=n_qubits
    ) + bit_driver_spo(n_qubits, b=1)
    return cost, x_mixer_spo(n_qubits)


# ---------------------------------------------------------------------------
# Maximum-weighted cycle (directed graph, edge variables).
# ---------------------------------------------------------------------------


def edges_to_wires(graph: DiGraphLike | GraphLike) -> dict[tuple, int]:
    """Map graph edges to dense 0-indexed wire positions.

    Mirrors :func:`pennylane.qaoa.cycle.edges_to_wires` for both ``nx`` and
    ``rx`` graphs. Rustworkx inputs are routed through their networkx
    equivalent so that every wire assignment in this module — including
    the mappings returned by :func:`max_weight_cycle_hamiltonians` — is
    derived from a single canonical edge order. Endpoints are reported
    as the user-facing node values.
    """
    # nx.DiGraph is a subclass of nx.Graph; this catches both.
    if isinstance(graph, nx.Graph):
        return {edge: i for i, edge in enumerate(graph.edges())}
    if isinstance(graph, rx.PyDiGraph):
        return edges_to_wires(_to_nx_digraph(graph))
    if isinstance(graph, rx.PyGraph):
        return edges_to_wires(_to_nx_graph(graph))
    raise TypeError(
        f"Expected nx.Graph / nx.DiGraph / rx.PyGraph / rx.PyDiGraph, got "
        f"{type(graph).__name__}."
    )


def wires_to_edges(graph: DiGraphLike | GraphLike) -> dict[int, tuple]:
    """Inverse of :func:`edges_to_wires`."""
    return {wire: edge for edge, wire in edges_to_wires(graph).items()}


def loss_hamiltonian_spo(graph: DiGraphLike | GraphLike) -> SparsePauliOp:
    """Loss Hamiltonian ``sum_(i,j) log(c_ij) * Z_(i,j)`` over weighted edges."""
    n_qubits = (
        len(graph.edge_list())
        if isinstance(graph, (rx.PyGraph, rx.PyDiGraph))
        else graph.number_of_edges()
    )
    if n_qubits == 0:
        return SparsePauliOp.from_list([("", 0.0)])

    edge_to_wire = edges_to_wires(graph)
    if isinstance(graph, (rx.PyGraph, rx.PyDiGraph)):
        nodes = list(graph.nodes())
        weighted = [
            ((nodes[u], nodes[v]), payload)
            for u, v, payload in graph.weighted_edge_list()
        ]
        weighted.sort(key=lambda item: item[0])
    else:
        weighted = [((u, v), data) for u, v, data in graph.edges(data=True)]

    terms: list[tuple[str, float]] = []
    for edge, payload in weighted:
        if edge[0] == edge[1]:
            raise ValueError("Graph contains self-loops")
        try:
            weight = payload["weight"] if isinstance(payload, dict) else payload
        except KeyError as e:
            raise KeyError(f"Edge {edge} does not contain weight data") from e
        if weight is None:
            raise KeyError(f"Edge {edge} does not contain weight data")
        wire = edge_to_wire[edge]
        terms.append((single_pauli_label(n_qubits, wire, "Z"), float(np.log(weight))))
    return SparsePauliOp.from_list(terms)


def cycle_mixer_spo(graph: DiGraphLike) -> SparsePauliOp:
    """Cycle-mixer Hamiltonian for the maximum-weighted cycle problem.

    For each edge ``(i,j)`` with intermediate node ``k`` such that ``(i,k)``
    and ``(k,j)`` are edges, contributes
    ``0.25 * (X_ij X_ik X_kj + Y_ij Y_ik X_kj + Y_ij X_ik Y_kj - X_ij Y_ik Y_kj)``.
    """
    nx_dg = _to_nx_digraph(graph)
    edge_to_wire = edges_to_wires(nx_dg)
    n_qubits = nx_dg.number_of_edges()
    if n_qubits == 0:
        return SparsePauliOp.from_list([("", 0.0)])

    nodes = list(nx_dg.nodes())
    edges = set(nx_dg.edges())
    terms: list[tuple[str, float]] = []
    for edge in nx_dg.edges():
        i, j = edge
        wire_ij = edge_to_wire[edge]
        for k in nodes:
            if k == i or k == j:
                continue
            out_edge = (i, k)
            in_edge = (k, j)
            if out_edge not in edges or in_edge not in edges:
                continue
            wire_ik = edge_to_wire[out_edge]
            wire_kj = edge_to_wire[in_edge]
            terms.extend(
                [
                    (
                        multi_pauli_label(
                            n_qubits,
                            [(wire_ij, "X"), (wire_ik, "X"), (wire_kj, "X")],
                        ),
                        0.25,
                    ),
                    (
                        multi_pauli_label(
                            n_qubits,
                            [(wire_ij, "Y"), (wire_ik, "Y"), (wire_kj, "X")],
                        ),
                        0.25,
                    ),
                    (
                        multi_pauli_label(
                            n_qubits,
                            [(wire_ij, "Y"), (wire_ik, "X"), (wire_kj, "Y")],
                        ),
                        0.25,
                    ),
                    (
                        multi_pauli_label(
                            n_qubits,
                            [(wire_ij, "X"), (wire_ik, "Y"), (wire_kj, "Y")],
                        ),
                        -0.25,
                    ),
                ]
            )

    if not terms:
        return SparsePauliOp.from_list([("I" * n_qubits, 0.0)])
    return SparsePauliOp.from_list(terms)


def out_flow_constraint_spo(graph: DiGraphLike) -> SparsePauliOp:
    """Out-flow constraint Hamiltonian (squared sum of out-edge Z's per node)."""
    nx_dg = _to_nx_digraph(graph)
    edge_to_wire = edges_to_wires(nx_dg)
    n_qubits = nx_dg.number_of_edges()
    if n_qubits == 0:
        return SparsePauliOp.from_list([("", 0.0)])

    identity = SparsePauliOp.from_list([("I" * n_qubits, 1.0)])
    h = SparsePauliOp.from_list([("I" * n_qubits, 0.0)])
    for node in nx_dg.nodes():
        out_wires = [edge_to_wire[edge] for edge in nx_dg.out_edges(node)]
        d = len(out_wires)
        if d == 0:
            continue
        z_sum = SparsePauliOp.from_list(
            [(single_pauli_label(n_qubits, w, "Z"), 1.0) for w in out_wires]
        )
        h = h + ((d * (d - 2)) * identity + (-2 * (d - 1)) * z_sum + (z_sum @ z_sum))
    return h.simplify()


def net_flow_constraint_spo(graph: DiGraphLike) -> SparsePauliOp:
    """Net-flow constraint Hamiltonian (squared sum of in/out Z deltas per node)."""
    nx_dg = _to_nx_digraph(graph)
    edge_to_wire = edges_to_wires(nx_dg)
    n_qubits = nx_dg.number_of_edges()
    if n_qubits == 0:
        return SparsePauliOp.from_list([("", 0.0)])

    h = SparsePauliOp.from_list([("I" * n_qubits, 0.0)])
    for node in nx_dg.nodes():
        out_wires = [edge_to_wire[edge] for edge in nx_dg.out_edges(node)]
        in_wires = [edge_to_wire[edge] for edge in nx_dg.in_edges(node)]
        delta = len(out_wires) - len(in_wires)
        terms: list[tuple[str, float]] = [("I" * n_qubits, float(delta))]
        terms.extend((single_pauli_label(n_qubits, w, "Z"), -1.0) for w in out_wires)
        terms.extend((single_pauli_label(n_qubits, w, "Z"), 1.0) for w in in_wires)
        inner = SparsePauliOp.from_list(terms)
        h = h + (inner @ inner)
    return h.simplify()


def max_weight_cycle_hamiltonians(
    graph: DiGraphLike, *, constrained: bool = True
) -> tuple[SparsePauliOp, SparsePauliOp, dict[int, tuple]]:
    """Cost, mixer, and wire→edge mapping for max-weight cycle.

    The mapping uses node values as recorded by :func:`wires_to_edges`. For
    rustworkx inputs, node indices are remapped back to user-facing values.
    """
    nx_dg = _to_nx_digraph(graph)
    mapping = wires_to_edges(nx_dg)
    if constrained:
        cost = loss_hamiltonian_spo(nx_dg)
        mixer = cycle_mixer_spo(nx_dg)
    else:
        cost = (
            loss_hamiltonian_spo(nx_dg)
            + 3.0 * net_flow_constraint_spo(nx_dg)
            + 3.0 * out_flow_constraint_spo(nx_dg)
        )
        mixer = x_mixer_spo(nx_dg.number_of_edges())
    cost = cost.simplify(atol=1e-12)
    mixer = mixer.simplify(atol=1e-12)
    return cost, mixer, mapping


__all__ = [
    "edges_to_wires",
    "wires_to_edges",
    "loss_hamiltonian_spo",
    "cycle_mixer_spo",
    "out_flow_constraint_spo",
    "net_flow_constraint_spo",
    "maxcut_hamiltonians",
    "max_independent_set_hamiltonians",
    "min_vertex_cover_hamiltonians",
    "max_clique_hamiltonians",
    "max_weight_cycle_hamiltonians",
]
