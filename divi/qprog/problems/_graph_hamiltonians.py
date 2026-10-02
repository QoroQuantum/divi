# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Graph-problem cost and mixer Hamiltonians as ``SparsePauliOp``.

Mirrors the formulations in :mod:`pennylane.qaoa.cost` and
:mod:`pennylane.qaoa.cycle` while emitting Qiskit ``SparsePauliOp`` objects.

Every builder works on qubit positions: node ``i`` of a graph is the ``i``-th
entry of :func:`_node_ids`, its node labels for networkx and its node indices
for rustworkx.
"""

from collections import defaultdict
from numbers import Real
from typing import Any, TypeAlias

import networkx as nx
import numpy as np
import rustworkx as rx
from qiskit.quantum_info import SparsePauliOp

from divi.hamiltonians import (
    bit_driver,
    bit_flip_mixer,
    edge_driver,
    x_mixer,
)
from divi.hamiltonians._mixers import multi_pauli_label, single_pauli_label

GraphLike: TypeAlias = nx.Graph | rx.PyGraph
DiGraphLike: TypeAlias = nx.DiGraph | rx.PyDiGraph


def _node_ids(graph: GraphLike | DiGraphLike) -> list:
    """Node identifiers in qubit order: node indices for rustworkx, labels for networkx."""
    if isinstance(graph, (rx.PyGraph, rx.PyDiGraph)):
        return list(graph.node_indexes())
    return list(graph.nodes())


def _edge_weight(payload: Any) -> float | None:
    """Weight of an edge from a networkx attribute dict or a rustworkx payload.

    A ``dict`` contributes its ``"weight"`` entry and a real number is the weight
    itself; anything else, ``None`` included, carries no weight.
    """
    if isinstance(payload, dict):
        payload = payload.get("weight")
    if isinstance(payload, Real):
        return float(payload)
    return None


def _qubit_edges(graph: GraphLike) -> tuple[int, list[tuple[int, int]]]:
    """Qubit count and the edges of an undirected graph as qubit pairs."""
    if isinstance(graph, (nx.DiGraph, rx.PyDiGraph)) or not isinstance(
        graph, (nx.Graph, rx.PyGraph)
    ):
        raise TypeError(
            f"Expected an undirected graph (nx.Graph or rx.PyGraph), got "
            f"{type(graph).__name__}."
        )
    ids, edges = _wire_edges(graph)
    return len(ids), [(source, target) for source, target, _ in edges]


def _qubit_graph(n_qubits: int, edges: list[tuple[int, int]]) -> nx.Graph:
    """``networkx`` graph on nodes ``0..n_qubits-1`` holding ``edges``."""
    graph = nx.Graph()
    graph.add_nodes_from(range(n_qubits))
    graph.add_edges_from(edges)
    return graph


def _wire_edges(
    graph: GraphLike | DiGraphLike,
) -> tuple[list, list[tuple[int, int, Any]]]:
    """Node ids and the edges in wire order as ``(source, target, payload)``.

    Endpoints are positions into the node ids. networkx graphs keep their own
    edge order; rustworkx edges are ordered by source node, then by insertion,
    which is the order networkx reports for the same graph built in index order.
    Parallel edges, from a rustworkx graph or a networkx multigraph, collapse
    into the first, keeping the last payload, as repeated ``add_edge`` calls do
    on a ``networkx.Graph``.
    """
    ids = _node_ids(graph)
    position = {node: i for i, node in enumerate(ids)}
    if isinstance(graph, nx.Graph):
        raw = graph.edges(data=True)
        directed = graph.is_directed()
    elif isinstance(graph, (rx.PyGraph, rx.PyDiGraph)):
        raw = sorted(graph.weighted_edge_list(), key=lambda edge: edge[0])
        directed = isinstance(graph, rx.PyDiGraph)
    else:
        raise TypeError(
            f"Expected nx.Graph / nx.DiGraph / rx.PyGraph / rx.PyDiGraph, got "
            f"{type(graph).__name__}."
        )
    merged: dict[tuple[int, int], tuple[int, int, Any]] = {}
    for left, right, payload in raw:
        source, target = position[left], position[right]
        key = (
            (source, target) if directed else (min(source, target), max(source, target))
        )
        first = merged.get(key, (source, target, None))
        merged[key] = (first[0], first[1], payload)
    return ids, list(merged.values())


def _directed_wire_edges(
    graph: DiGraphLike,
) -> tuple[list, list[tuple[int, int, Any]]]:
    """:func:`_wire_edges` restricted to directed graphs."""
    if not isinstance(graph, (nx.DiGraph, rx.PyDiGraph)):
        raise TypeError(
            f"Expected a directed graph (nx.DiGraph or rx.PyDiGraph), got "
            f"{type(graph).__name__}."
        )
    return _wire_edges(graph)


# ---------------------------------------------------------------------------
# Cost-Hamiltonian builders for graph problems.
# ---------------------------------------------------------------------------


def maxcut_hamiltonians(graph: GraphLike) -> tuple[SparsePauliOp, SparsePauliOp]:
    """Cost and mixer for MaxCut.

    .. math::
        H_C = \\frac{1}{2} \\sum_{(i,j) \\in E} (Z_i Z_j - I), \\quad
        H_M = \\sum_v X_v
    """
    n_qubits, edges = _qubit_edges(graph)
    cost = edge_driver(
        edges, ["10", "01"], n_qubits=n_qubits
    ) + SparsePauliOp.from_list([("I" * n_qubits, -0.5 * len(edges))])
    return cost, x_mixer(n_qubits)


def max_independent_set_hamiltonians(
    graph: GraphLike, *, constrained: bool = True
) -> tuple[SparsePauliOp, SparsePauliOp]:
    """Cost and mixer for Maximum Independent Set."""
    n_qubits, edges = _qubit_edges(graph)
    if constrained:
        cost = bit_driver(n_qubits, b=1)
        mixer = bit_flip_mixer(_qubit_graph(n_qubits, edges), b=0)
        return cost, mixer
    cost = 3.0 * edge_driver(edges, ["10", "01", "00"], n_qubits=n_qubits) + bit_driver(
        n_qubits, b=1
    )
    return cost, x_mixer(n_qubits)


def min_vertex_cover_hamiltonians(
    graph: GraphLike, *, constrained: bool = True
) -> tuple[SparsePauliOp, SparsePauliOp]:
    """Cost and mixer for Minimum Vertex Cover."""
    n_qubits, edges = _qubit_edges(graph)
    if constrained:
        cost = bit_driver(n_qubits, b=0)
        mixer = bit_flip_mixer(_qubit_graph(n_qubits, edges), b=1)
        return cost, mixer
    cost = 3.0 * edge_driver(edges, ["11", "10", "01"], n_qubits=n_qubits) + bit_driver(
        n_qubits, b=0
    )
    return cost, x_mixer(n_qubits)


def max_clique_hamiltonians(
    graph: GraphLike, *, constrained: bool = True
) -> tuple[SparsePauliOp, SparsePauliOp]:
    """Cost and mixer for Maximum Clique. The mixer acts on the complement graph."""
    n_qubits, edges = _qubit_edges(graph)
    complement = nx.complement(_qubit_graph(n_qubits, edges))
    if constrained:
        cost = bit_driver(n_qubits, b=1)
        mixer = bit_flip_mixer(complement, b=0)
        return cost, mixer
    cost = 3.0 * edge_driver(
        complement, ["10", "01", "00"], n_qubits=n_qubits
    ) + bit_driver(n_qubits, b=1)
    return cost, x_mixer(n_qubits)


# ---------------------------------------------------------------------------
# Maximum-weighted cycle (directed graph, edge variables).
# ---------------------------------------------------------------------------


def edges_to_wires(graph: DiGraphLike | GraphLike) -> dict[tuple, int]:
    """Map graph edges to dense 0-indexed wire positions.

    Mirrors :func:`pennylane.qaoa.cycle.edges_to_wires` for both ``nx`` and
    ``rx`` graphs. Endpoints are node labels for networkx graphs and node
    indices for rustworkx graphs.
    """
    ids, edges = _wire_edges(graph)
    return {
        (ids[source], ids[target]): wire
        for wire, (source, target, _) in enumerate(edges)
    }


def wires_to_edges(graph: DiGraphLike | GraphLike) -> dict[int, tuple]:
    """Inverse of :func:`edges_to_wires`."""
    return {wire: edge for edge, wire in edges_to_wires(graph).items()}


def loss_hamiltonian_spo(graph: DiGraphLike | GraphLike) -> SparsePauliOp:
    """Loss Hamiltonian ``sum_(i,j) log(c_ij) * Z_(i,j)`` over weighted edges."""
    ids, edges = _wire_edges(graph)
    n_qubits = len(edges)
    if n_qubits == 0:
        return SparsePauliOp.from_list([("", 0.0)])

    terms: list[tuple[str, float]] = []
    for wire, (source, target, payload) in enumerate(edges):
        edge = (ids[source], ids[target])
        if source == target:
            raise ValueError("Graph contains self-loops")
        weight = _edge_weight(payload)
        if weight is None:
            raise KeyError(f"Edge {edge} does not contain weight data")
        terms.append((single_pauli_label(n_qubits, wire, "Z"), float(np.log(weight))))
    return SparsePauliOp.from_list(terms)


_CYCLE_MIXER_TERMS = (("XXX", 0.25), ("YYX", 0.25), ("YXY", 0.25), ("XYY", -0.25))


def cycle_mixer_spo(graph: DiGraphLike) -> SparsePauliOp:
    """Cycle-mixer Hamiltonian for the maximum-weighted cycle problem.

    For each edge ``(i,j)`` with intermediate node ``k`` such that ``(i,k)``
    and ``(k,j)`` are edges, contributes
    ``0.25 * (X_ij X_ik X_kj + Y_ij Y_ik X_kj + Y_ij X_ik Y_kj - X_ij Y_ik Y_kj)``.
    """
    _, edges = _directed_wire_edges(graph)
    n_qubits = len(edges)
    if n_qubits == 0:
        return SparsePauliOp.from_list([("", 0.0)])

    edge_to_wire = {
        (source, target): wire for wire, (source, target, _) in enumerate(edges)
    }
    successors: defaultdict[int, set[int]] = defaultdict(set)
    predecessors: defaultdict[int, set[int]] = defaultdict(set)
    for source, target in edge_to_wire:
        successors[source].add(target)
        predecessors[target].add(source)

    terms: list[tuple[str, float]] = []
    for (i, j), wire_ij in edge_to_wire.items():
        for k in sorted(successors[i] & predecessors[j]):
            if k == i or k == j:
                continue
            wire_ik = edge_to_wire[(i, k)]
            wire_kj = edge_to_wire[(k, j)]
            wires = (wire_ij, wire_ik, wire_kj)
            terms.extend(
                (multi_pauli_label(n_qubits, list(zip(wires, paulis))), coeff)
                for paulis, coeff in _CYCLE_MIXER_TERMS
            )

    if not terms:
        return SparsePauliOp.from_list([("I" * n_qubits, 0.0)])
    return SparsePauliOp.from_list(terms)


def _out_wires(n_nodes: int, edges: list[tuple[int, int, Any]]) -> list[list[int]]:
    """Wires of the edges leaving each node, in wire order."""
    wires: list[list[int]] = [[] for _ in range(n_nodes)]
    for wire, (source, _target, _payload) in enumerate(edges):
        wires[source].append(wire)
    return wires


def _in_wires(
    graph: DiGraphLike, ids: list, edges: list[tuple[int, int, Any]]
) -> list[list[int]]:
    """Wires of the edges entering each node, in edge insertion order.

    That is networkx predecessor order, and rustworkx edge-index order, which
    agree for a networkx graph built by inserting the rustworkx edges in order.
    """
    wire = {(source, target): w for w, (source, target, _) in enumerate(edges)}
    position = {node: i for i, node in enumerate(ids)}
    if isinstance(graph, nx.DiGraph):
        return [
            [wire[(position[pred], i)] for pred in graph.pred[node]]
            for i, node in enumerate(ids)
        ]
    wires: list[list[int]] = [[] for _ in ids]
    seen: set[tuple[int, int]] = set()
    for left, right in graph.edge_list():
        key = (position[left], position[right])
        if key not in seen:
            seen.add(key)
            wires[key[1]].append(wire[key])
    return wires


def out_flow_constraint_spo(graph: DiGraphLike) -> SparsePauliOp:
    """Out-flow constraint Hamiltonian (squared sum of out-edge Z's per node)."""
    ids, edges = _directed_wire_edges(graph)
    n_qubits = len(edges)
    if n_qubits == 0:
        return SparsePauliOp.from_list([("", 0.0)])

    identity = SparsePauliOp.from_list([("I" * n_qubits, 1.0)])
    blocks = [SparsePauliOp.from_list([("I" * n_qubits, 0.0)])]
    for out_wires in _out_wires(len(ids), edges):
        d = len(out_wires)
        if d == 0:
            continue
        z_sum = SparsePauliOp.from_list(
            [(single_pauli_label(n_qubits, w, "Z"), 1.0) for w in out_wires]
        )
        blocks.append(
            (d * (d - 2)) * identity + (-2 * (d - 1)) * z_sum + (z_sum @ z_sum)
        )
    return SparsePauliOp.sum(blocks).simplify()


def net_flow_constraint_spo(graph: DiGraphLike) -> SparsePauliOp:
    """Net-flow constraint Hamiltonian (squared sum of in/out Z deltas per node)."""
    ids, edges = _directed_wire_edges(graph)
    n_qubits = len(edges)
    if n_qubits == 0:
        return SparsePauliOp.from_list([("", 0.0)])

    blocks = [SparsePauliOp.from_list([("I" * n_qubits, 0.0)])]
    for out_wires, in_wires in zip(
        _out_wires(len(ids), edges), _in_wires(graph, ids, edges)
    ):
        delta = len(out_wires) - len(in_wires)
        terms: list[tuple[str, float]] = [("I" * n_qubits, float(delta))]
        terms.extend((single_pauli_label(n_qubits, w, "Z"), -1.0) for w in out_wires)
        terms.extend((single_pauli_label(n_qubits, w, "Z"), 1.0) for w in in_wires)
        inner = SparsePauliOp.from_list(terms)
        blocks.append(inner @ inner)
    return SparsePauliOp.sum(blocks).simplify()


def max_weight_cycle_hamiltonians(
    graph: DiGraphLike, *, constrained: bool = True
) -> tuple[SparsePauliOp, SparsePauliOp, dict[int, tuple]]:
    """Cost, mixer, and wire→edge mapping for max-weight cycle.

    The mapping is :func:`wires_to_edges`: endpoints are node labels for
    networkx graphs and node indices for rustworkx graphs.
    """
    ids, edges = _directed_wire_edges(graph)
    mapping = {
        wire: (ids[source], ids[target])
        for wire, (source, target, _) in enumerate(edges)
    }
    if constrained:
        cost = loss_hamiltonian_spo(graph)
        mixer = cycle_mixer_spo(graph)
    else:
        cost = (
            loss_hamiltonian_spo(graph)
            + 3.0 * net_flow_constraint_spo(graph)
            + 3.0 * out_flow_constraint_spo(graph)
        )
        mixer = x_mixer(len(mapping))
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
