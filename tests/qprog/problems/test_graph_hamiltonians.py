# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for SparsePauliOp-native graph QAOA Hamiltonian builders.

Each resolver in :mod:`divi.qprog.problems._graph_hamiltonians` is pinned to
the combinatorics it encodes rather than to a second implementation: a cost
Hamiltonian by the objective its diagonal scores over every basis state, a
mixer by the basis-state moves it permits.
"""

import networkx as nx
import numpy as np
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
from tests.hamiltonians._helpers import (
    add_transition,
    assert_diagonal_scores,
    assert_transitions,
    basis_bits,
    bit_flip_transitions,
    single_flip_transitions,
)

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


def _selected(bits) -> int:
    return sum(bits)


def _both_endpoints_selected(graph: nx.Graph, bits) -> int:
    return sum(bits[left] and bits[right] for left, right in graph.edges())


def _neighbourhoods(graph: nx.Graph) -> dict[int, list[int]]:
    return {vertex: list(graph.neighbors(vertex)) for vertex in graph.nodes()}


@pytest.mark.parametrize("name,factory", _GRAPHS, ids=_GRAPH_IDS)
def test_maxcut_cost_counts_cut_edges(name, factory):
    """``H_C = 0.5 * sum (Z_i Z_j - I)`` has eigenvalue ``-|cut|`` exactly, so
    minimising it maximises the cut."""
    graph = factory()
    cost, mixer = maxcut_hamiltonians(graph)

    assert_diagonal_scores(
        cost,
        lambda bits: -sum(bits[left] != bits[right] for left, right in graph.edges()),
        offset=0.0,
    )
    assert_transitions(mixer, single_flip_transitions(graph.number_of_nodes()))


@pytest.mark.parametrize("name,factory", _GRAPHS, ids=_GRAPH_IDS)
def test_max_independent_set_constrained_scores_set_size(name, factory):
    """The constrained formulation leaves independence to the mixer: the cost
    only counts the chosen vertices, and the bit-flip mixer refuses to select a
    vertex whose neighbours are not all unselected."""
    graph = factory()
    cost, mixer = max_independent_set_hamiltonians(graph, constrained=True)

    assert_diagonal_scores(
        cost, lambda bits: sum(1 - 2 * bit for bit in bits), offset=0.0
    )
    assert_transitions(
        mixer,
        bit_flip_transitions(_neighbourhoods(graph), graph.number_of_nodes(), 0),
    )


@pytest.mark.parametrize("name,factory", _GRAPHS, ids=_GRAPH_IDS)
def test_max_independent_set_unconstrained_penalises_adjacent_selections(name, factory):
    """Dropping the constraint moves independence into the cost: every edge with
    both endpoints selected costs 3, against a reward of 2 per selected vertex."""
    graph = factory()
    cost, mixer = max_independent_set_hamiltonians(graph, constrained=False)

    assert_diagonal_scores(
        cost,
        lambda bits: 3 * _both_endpoints_selected(graph, bits) - 2 * _selected(bits),
    )
    assert_transitions(mixer, single_flip_transitions(graph.number_of_nodes()))


@pytest.mark.parametrize("name,factory", _GRAPHS, ids=_GRAPH_IDS)
def test_min_vertex_cover_constrained_scores_cover_size(name, factory):
    """Cover validity is the mixer's job: a vertex may leave the cover only when
    all of its neighbours are already in it."""
    graph = factory()
    cost, mixer = min_vertex_cover_hamiltonians(graph, constrained=True)

    assert_diagonal_scores(
        cost, lambda bits: -sum(1 - 2 * bit for bit in bits), offset=0.0
    )
    assert_transitions(
        mixer,
        bit_flip_transitions(_neighbourhoods(graph), graph.number_of_nodes(), 1),
    )


@pytest.mark.parametrize("name,factory", _GRAPHS, ids=_GRAPH_IDS)
def test_min_vertex_cover_unconstrained_penalises_uncovered_edges(name, factory):
    """Every edge with neither endpoint selected costs 3, against a cost of 2
    per selected vertex — so the minimum is the smallest valid cover."""
    graph = factory()
    cost, mixer = min_vertex_cover_hamiltonians(graph, constrained=False)

    assert_diagonal_scores(
        cost,
        lambda bits: 3
        * sum(not bits[left] and not bits[right] for left, right in graph.edges())
        + 2 * _selected(bits),
    )
    assert_transitions(mixer, single_flip_transitions(graph.number_of_nodes()))


@pytest.mark.parametrize("name,factory", _GRAPHS, ids=_GRAPH_IDS)
def test_max_clique_constrained_scores_clique_size(name, factory):
    """A clique in ``graph`` is an independent set in its complement, so the
    mixer runs the bit-flip rule over the complement's neighbourhoods."""
    graph = factory()
    cost, mixer = max_clique_hamiltonians(graph, constrained=True)

    assert_diagonal_scores(
        cost, lambda bits: sum(1 - 2 * bit for bit in bits), offset=0.0
    )
    assert_transitions(
        mixer,
        bit_flip_transitions(
            _neighbourhoods(nx.complement(graph)), graph.number_of_nodes(), 0
        ),
    )


@pytest.mark.parametrize("name,factory", _GRAPHS, ids=_GRAPH_IDS)
def test_max_clique_unconstrained_penalises_non_adjacent_selections(name, factory):
    """Selecting two vertices that are *not* adjacent in ``graph`` costs 3."""
    graph = factory()
    complement = nx.complement(graph)
    cost, mixer = max_clique_hamiltonians(graph, constrained=False)

    assert_diagonal_scores(
        cost,
        lambda bits: 3 * _both_endpoints_selected(complement, bits)
        - 2 * _selected(bits),
    )
    assert_transitions(mixer, single_flip_transitions(graph.number_of_nodes()))


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


def _selected_edges(graph: nx.DiGraph, bits, edges) -> int:
    wire = edges_to_wires(graph)
    return sum(bits[wire[edge]] for edge in edges)


def test_edges_to_wires_inverse_of_wires_to_edges():
    g = _three_node_tournament()
    forward = edges_to_wires(g)
    backward = wires_to_edges(g)
    assert {edge: wire for edge, wire in forward.items()} == {
        edge: wire for wire, edge in backward.items()
    }


@pytest.mark.parametrize("name,factory", _DIGRAPHS, ids=_DIGRAPH_IDS)
def test_loss_hamiltonian_scores_the_log_weight_of_each_selected_edge(name, factory):
    g = factory()
    wire = edges_to_wires(g)

    def score(bits):
        return sum(
            np.log(data["weight"]) * (1 - 2 * bits[wire[(left, right)]])
            for left, right, data in g.edges(data=True)
        )

    assert_diagonal_scores(loss_hamiltonian_spo(g), score, offset=0.0)


@pytest.mark.parametrize("name,factory", _DIGRAPHS, ids=_DIGRAPH_IDS)
def test_cycle_mixer_swaps_an_edge_for_its_two_hop_detour(name, factory):
    """The mixer moves between equal-flow configurations: it exchanges a chosen
    edge ``(i,j)`` for the pair ``(i,k), (k,j)`` whenever that detour exists,
    leaving every other wire untouched."""
    g = factory()
    wire = edges_to_wires(g)
    n_qubits = g.number_of_edges()
    edges = set(g.edges())

    expected: dict[tuple[int, int], float] = {}
    for left, right in g.edges():
        for middle in g.nodes():
            if middle in (left, right):
                continue
            if (left, middle) not in edges or (middle, right) not in edges:
                continue
            direct = wire[(left, right)]
            first, second = wire[(left, middle)], wire[(middle, right)]
            for state in range(2**n_qubits):
                bits = basis_bits(state, n_qubits)
                if bits[direct] == 1 and bits[first] == 0 and bits[second] == 0:
                    detour = state ^ (1 << direct) ^ (1 << first) ^ (1 << second)
                    add_transition(expected, detour, state)
                    add_transition(expected, state, detour)

    assert_transitions(cycle_mixer_spo(g), expected)


@pytest.mark.parametrize("name,factory", _DIGRAPHS, ids=_DIGRAPH_IDS)
def test_out_flow_constraint_penalises_nodes_with_several_out_edges(name, factory):
    """Zero cost when every node emits at most one edge; otherwise ``4k(k-1)``
    for a node emitting ``k``."""
    g = factory()

    def score(bits):
        return sum(
            4
            * _selected_edges(g, bits, g.out_edges(node))
            * (_selected_edges(g, bits, g.out_edges(node)) - 1)
            for node in g.nodes()
        )

    assert_diagonal_scores(out_flow_constraint_spo(g), score, offset=0.0)


@pytest.mark.parametrize("name,factory", _DIGRAPHS, ids=_DIGRAPH_IDS)
def test_net_flow_constraint_penalises_unbalanced_nodes(name, factory):
    """Zero cost exactly when in-flow equals out-flow at every node."""
    g = factory()

    def score(bits):
        return sum(
            4
            * (
                _selected_edges(g, bits, g.out_edges(node))
                - _selected_edges(g, bits, g.in_edges(node))
            )
            ** 2
            for node in g.nodes()
        )

    assert_diagonal_scores(net_flow_constraint_spo(g), score, offset=0.0)


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

    loss = loss_hamiltonian_spo(g)
    if constrained:
        assert _spo_equal(cost, loss)
        assert _spo_equal(mixer, cycle_mixer_spo(g))
    else:
        expected_cost = (
            loss + 3.0 * net_flow_constraint_spo(g) + 3.0 * out_flow_constraint_spo(g)
        )
        assert _spo_equal(cost, expected_cost)
        # Unconstrained variant uses an X mixer over all wires.
        assert_transitions(mixer, single_flip_transitions(g.number_of_edges()))
