# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Parity tests for SparsePauliOp-native graph QAOA Hamiltonian builders.

Asserts each resolver in :mod:`divi.qprog.problems._graph_hamiltonians`
produces a ``SparsePauliOp`` numerically equal to the equivalent PennyLane
``pennylane.qaoa.cost`` output, on both constrained and unconstrained
formulations.
"""

import networkx as nx
import numpy as np
import pennylane as qp
import pennylane.qaoa as pqaoa
import pytest
import rustworkx as rx
from qiskit.quantum_info import SparsePauliOp

from divi.qprog.problems._graph_hamiltonians import (
    cycle_mixer_spo,
    edges_to_wires,
    loss_hamiltonian_spo,
    max_clique_hamiltonians,
    max_independent_set_hamiltonians,
    max_weight_cycle_hamiltonians,
    maxcut_hamiltonians,
    min_vertex_cover_hamiltonians,
    net_flow_constraint_spo,
    out_flow_constraint_spo,
    wires_to_edges,
)


def _assert_matches_pennylane(actual: SparsePauliOp, expected_pl) -> None:
    """Matrix-equality check against a PennyLane operator (see
    ``tests/hamiltonians/test_mixers.py`` for the reversed wire_order)."""
    n = actual.num_qubits
    expected_mat = qp.matrix(expected_pl, wire_order=list(reversed(range(n))))
    np.testing.assert_allclose(actual.to_matrix(), expected_mat, atol=1e-10)


# Graphs chosen to exercise: small (bull, triangle), regular (cycle),
# degree-asymmetric (star), and non-trivial complement (bull).
_GRAPHS = [
    ("bull", nx.bull_graph),
    ("triangle", lambda: nx.cycle_graph(3)),
    ("cycle5", lambda: nx.cycle_graph(5)),
    ("star5", lambda: nx.star_graph(5)),
    ("path4", lambda: nx.path_graph(4)),
]
_GRAPH_IDS = [name for name, _ in _GRAPHS]


@pytest.mark.parametrize("name,factory", _GRAPHS, ids=_GRAPH_IDS)
def test_maxcut_matches_pennylane(name, factory):
    graph = factory()
    cost, mixer = maxcut_hamiltonians(graph)
    pl_cost, pl_mixer = pqaoa.cost.maxcut(graph)

    _assert_matches_pennylane(cost, pl_cost)
    _assert_matches_pennylane(mixer, pl_mixer)


@pytest.mark.parametrize("constrained", [True, False])
@pytest.mark.parametrize("name,factory", _GRAPHS, ids=_GRAPH_IDS)
def test_max_independent_set_matches_pennylane(name, factory, constrained):
    graph = factory()
    cost, mixer = max_independent_set_hamiltonians(graph, constrained=constrained)
    pl_cost, pl_mixer = pqaoa.cost.max_independent_set(graph, constrained=constrained)

    _assert_matches_pennylane(cost, pl_cost)
    _assert_matches_pennylane(mixer, pl_mixer)


@pytest.mark.parametrize("constrained", [True, False])
@pytest.mark.parametrize("name,factory", _GRAPHS, ids=_GRAPH_IDS)
def test_min_vertex_cover_matches_pennylane(name, factory, constrained):
    graph = factory()
    cost, mixer = min_vertex_cover_hamiltonians(graph, constrained=constrained)
    pl_cost, pl_mixer = pqaoa.cost.min_vertex_cover(graph, constrained=constrained)

    _assert_matches_pennylane(cost, pl_cost)
    _assert_matches_pennylane(mixer, pl_mixer)


@pytest.mark.parametrize("constrained", [True, False])
@pytest.mark.parametrize("name,factory", _GRAPHS, ids=_GRAPH_IDS)
def test_max_clique_matches_pennylane(name, factory, constrained):
    graph = factory()
    cost, mixer = max_clique_hamiltonians(graph, constrained=constrained)
    pl_cost, pl_mixer = pqaoa.cost.max_clique(graph, constrained=constrained)

    _assert_matches_pennylane(cost, pl_cost)
    _assert_matches_pennylane(mixer, pl_mixer)


# ---------------------------------------------------------------------------
# Maximum-weighted cycle (DiGraph)
# ---------------------------------------------------------------------------


def _three_node_tournament() -> nx.DiGraph:
    """Weighted 3-node tournament (all ordered pairs)."""
    g = nx.DiGraph()
    g.add_weighted_edges_from(
        [(0, 1, 1.5), (1, 0, 0.5), (0, 2, 2.0), (2, 0, 1.0), (1, 2, 1.2), (2, 1, 0.8)]
    )
    return g


def _four_node_partial_digraph() -> nx.DiGraph:
    """Weighted 4-node DiGraph with asymmetric in/out degrees per node."""
    g = nx.DiGraph()
    g.add_weighted_edges_from(
        [(0, 1, 1.0), (1, 2, 1.5), (2, 3, 0.7), (3, 0, 0.9), (0, 2, 1.1), (2, 1, 0.6)]
    )
    return g


_DIGRAPHS = [
    ("3-tournament", _three_node_tournament),
    ("4-partial", _four_node_partial_digraph),
]
_DIGRAPH_IDS = [name for name, _ in _DIGRAPHS]


def test_edges_to_wires_inverse_of_wires_to_edges():
    g = _three_node_tournament()
    forward = edges_to_wires(g)
    backward = wires_to_edges(g)
    assert {edge: wire for edge, wire in forward.items()} == {
        edge: wire for wire, edge in backward.items()
    }


@pytest.mark.parametrize("name,factory", _DIGRAPHS, ids=_DIGRAPH_IDS)
def test_loss_hamiltonian_matches_pennylane(name, factory):
    g = factory()
    _assert_matches_pennylane(loss_hamiltonian_spo(g), pqaoa.cycle.loss_hamiltonian(g))


@pytest.mark.parametrize("name,factory", _DIGRAPHS, ids=_DIGRAPH_IDS)
def test_cycle_mixer_matches_pennylane(name, factory):
    g = factory()
    _assert_matches_pennylane(cycle_mixer_spo(g), pqaoa.cycle.cycle_mixer(g))


@pytest.mark.parametrize("name,factory", _DIGRAPHS, ids=_DIGRAPH_IDS)
def test_out_flow_constraint_matches_pennylane(name, factory):
    g = factory()
    _assert_matches_pennylane(
        out_flow_constraint_spo(g), pqaoa.cycle.out_flow_constraint(g)
    )


@pytest.mark.parametrize("name,factory", _DIGRAPHS, ids=_DIGRAPH_IDS)
def test_net_flow_constraint_matches_pennylane(name, factory):
    g = factory()
    _assert_matches_pennylane(
        net_flow_constraint_spo(g), pqaoa.cycle.net_flow_constraint(g)
    )


# ---------------------------------------------------------------------------
# rustworkx ingress
# ---------------------------------------------------------------------------


def _spo_equal(a: SparsePauliOp, b: SparsePauliOp) -> bool:
    """Numerical equality for SPOs over the same canonical qubit order."""
    if a.num_qubits != b.num_qubits:
        return False
    diff = (a - b).simplify(atol=1e-12)
    return diff.size == 0 or np.allclose(diff.coeffs, 0, atol=1e-12)


def test_rustworkx_pygraph_matches_nx_for_maxcut():
    """``rx.PyGraph`` input produces the same Hamiltonians as the equivalent
    ``nx.Graph`` input."""
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (1, 3)]
    nx_graph = nx.Graph(edges)
    rx_graph = rx.PyGraph()
    rx_graph.add_nodes_from(list(nx_graph.nodes()))
    rx_graph.add_edges_from([(u, v, None) for u, v in edges])

    nx_cost, nx_mixer = maxcut_hamiltonians(nx_graph)
    rx_cost, rx_mixer = maxcut_hamiltonians(rx_graph)

    assert _spo_equal(rx_cost, nx_cost), "cost SPO mismatch: rx.PyGraph vs nx.Graph"
    assert _spo_equal(rx_mixer, nx_mixer), "mixer SPO mismatch: rx.PyGraph vs nx.Graph"


def test_wires_to_edges_consistent_with_max_weight_cycle_for_unsorted_rx():
    """Wire-ordering consistency: ``wires_to_edges(rx_dg)`` must reproduce the
    mapping returned by ``max_weight_cycle_hamiltonians(rx_dg)`` even when
    ``rx_dg.edge_list()`` is not in lexicographic order."""
    # Insert edges in an order whose lexicographic sort differs from
    # insertion order — (1, 0) precedes (0, 1) in the rx graph but sorts
    # after it.
    unsorted_edges = [(1, 0, 0.5), (0, 1, 1.5), (2, 0, 1.0), (0, 2, 2.0)]
    rx_graph = rx.PyDiGraph()
    rx_graph.add_nodes_from([0, 1, 2])
    rx_graph.add_edges_from(unsorted_edges)

    _, _, mapping = max_weight_cycle_hamiltonians(rx_graph, constrained=True)
    assert mapping == wires_to_edges(rx_graph)


def test_rustworkx_pydigraph_matches_nx_for_max_weight_cycle():
    """``rx.PyDiGraph`` (weighted) input produces the same cost, mixer, and
    wire→edge mapping as the equivalent ``nx.DiGraph`` input."""
    nx_graph = _three_node_tournament()
    rx_graph = rx.PyDiGraph()
    rx_graph.add_nodes_from(list(nx_graph.nodes()))
    rx_graph.add_edges_from([(u, v, w) for u, v, w in nx_graph.edges(data="weight")])

    for constrained in (True, False):
        nx_cost, nx_mixer, nx_map = max_weight_cycle_hamiltonians(
            nx_graph, constrained=constrained
        )
        rx_cost, rx_mixer, rx_map = max_weight_cycle_hamiltonians(
            rx_graph, constrained=constrained
        )
        assert _spo_equal(
            rx_cost, nx_cost
        ), f"cost SPO mismatch constrained={constrained}"
        assert _spo_equal(
            rx_mixer, nx_mixer
        ), f"mixer SPO mismatch constrained={constrained}"
        assert rx_map == nx_map, f"wire→edge map mismatch constrained={constrained}"


@pytest.mark.parametrize("constrained", [True, False])
@pytest.mark.parametrize("name,factory", _DIGRAPHS, ids=_DIGRAPH_IDS)
def test_max_weight_cycle_composite(name, factory, constrained):
    """Composite check: the cost and mixer SPOs returned by
    :func:`max_weight_cycle_hamiltonians` reduce to the documented building
    blocks (``loss + 3·net_flow + 3·out_flow`` for the unconstrained variant,
    bare ``loss`` for the constrained variant)."""
    g = factory()
    cost, mixer, mapping = max_weight_cycle_hamiltonians(g, constrained=constrained)

    assert mapping == wires_to_edges(g)

    pl_loss = pqaoa.cycle.loss_hamiltonian(g)
    if constrained:
        _assert_matches_pennylane(cost, pl_loss)
        _assert_matches_pennylane(mixer, pqaoa.cycle.cycle_mixer(g))
    else:
        pl_cost = (
            pl_loss
            + 3.0 * pqaoa.cycle.net_flow_constraint(g)
            + 3.0 * pqaoa.cycle.out_flow_constraint(g)
        )
        _assert_matches_pennylane(cost, pl_cost)
        # Unconstrained variant uses an X mixer over all wires.
        _assert_matches_pennylane(mixer, pqaoa.x_mixer(range(g.number_of_edges())))
