# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for SparsePauliOp-native QAOA mixer builders.

Each builder is pinned to the combinatorics it encodes: a driver by the
objective its diagonal scores, a mixer by the basis-state moves it permits.
"""

import networkx as nx
import numpy as np
import pytest
from qiskit.quantum_info import SparsePauliOp

from divi.hamiltonians import (
    bit_driver,
    bit_flip_mixer,
    edge_driver,
    x_mixer,
    xy_mixer,
)
from tests.hamiltonians._helpers import (
    add_transition,
    assert_diagonal_scores,
    assert_transitions,
    basis_bits,
    bit_flip_transitions,
    single_flip_transitions,
)


def _assert_spo_equivalent(actual: SparsePauliOp, expected: SparsePauliOp) -> None:
    # ``simplify(atol=...)`` does not always shrink ``size`` to 0 — Qiskit can
    # leave a residual identity row with ``coeff == 0`` after cancellation, so
    # we also accept any non-empty result whose coefficients are all near zero.
    diff = (actual - expected).simplify(atol=1e-12)
    assert diff.size == 0 or np.allclose(diff.coeffs, 0, atol=1e-12)


def test_x_mixer_flips_each_qubit_independently():
    assert_transitions(x_mixer(4), single_flip_transitions(4))


def test_xy_mixer_swaps_the_endpoints_of_each_edge():
    """``0.5 * (XX + YY)`` exchanges ``|01>`` and ``|10>`` across an edge and
    annihilates endpoints that already agree, so the mixer conserves the number
    of set bits."""
    graph = nx.Graph([(0, 1), (1, 2)])
    n_qubits = 3
    expected: dict[tuple[int, int], float] = {}
    for state in range(2**n_qubits):
        bits = basis_bits(state, n_qubits)
        for left, right in graph.edges():
            if bits[left] != bits[right]:
                add_transition(expected, state ^ (1 << left) ^ (1 << right), state)
    assert_transitions(xy_mixer(graph), expected)


def test_xy_mixer_preserves_trailing_isolated_qubits():
    actual = xy_mixer(nx.Graph([(0, 1)]), n_qubits=4)

    assert actual.num_qubits == 4
    _assert_spo_equivalent(
        actual,
        SparsePauliOp.from_list([("IIXX", 0.5), ("IIYY", 0.5)]),
    )


def _graph_with_string_isolated_node() -> nx.Graph:
    g = nx.Graph()
    g.add_node("a")
    return g


@pytest.mark.parametrize(
    "graph_factory",
    [
        lambda: nx.Graph([("a", "b")]),
        _graph_with_string_isolated_node,
    ],
    ids=["string_edge", "string_isolated_node"],
)
def test_xy_mixer_requires_integer_nodes(graph_factory):
    with pytest.raises(TypeError, match="integer"):
        xy_mixer(graph_factory())


@pytest.mark.parametrize("b", [0, 1])
def test_bit_driver_rewards_qubits_sitting_at_b(b):
    """``b=1`` scores ``+sum_i Z_i`` and ``b=0`` its negation, so the diagonal
    is an affine function of how many qubits are set."""
    sign = 1.0 if b == 1 else -1.0
    assert_diagonal_scores(
        bit_driver(n_qubits=5, b=b),
        lambda bits: sign * sum(1 - 2 * bit for bit in bits),
        offset=0.0,
    )


def test_bit_driver_rejects_invalid_b():
    with pytest.raises(ValueError, match="b"):
        bit_driver(n_qubits=3, b=2)


@pytest.mark.parametrize(
    "reward",
    [
        ["10", "01"],
        ["00"],
        ["11"],
        ["10", "01", "00"],
        ["10", "01", "11"],
        ["00", "01", "10", "11"],
    ],
)
def test_edge_driver_penalises_each_unrewarded_edge_by_one(reward):
    """The documented contract: rewarded and penalised endpoint patterns are
    separated by exactly ``1`` in energy, whatever the reward set."""
    graph = nx.Graph([(0, 1), (1, 2), (2, 3)])
    assert_diagonal_scores(
        edge_driver(graph, reward),
        lambda bits: sum(
            f"{bits[left]}{bits[right]}" not in reward for left, right in graph.edges()
        ),
    )


def test_edge_driver_rejects_unpaired_directed_bits():
    with pytest.raises(ValueError, match="01"):
        edge_driver(nx.Graph([(0, 1)]), ["10"])


def _isolated_nodes() -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from([0, 1, 2])  # degree 0 everywhere
    return graph


@pytest.mark.parametrize("b", [0, 1])
@pytest.mark.parametrize(
    "graph_factory",
    [
        lambda: nx.path_graph(4),
        lambda: nx.cycle_graph(5),
        lambda: nx.star_graph(5),  # degree-5 hub exercises 2^d expansion sign bugs
        _isolated_nodes,  # empty neighbourhood: the flip is unconditional
    ],
    ids=["path4", "cycle5", "star5", "isolated"],
)
def test_bit_flip_mixer_flips_a_vertex_only_when_its_neighbours_sit_at_b(
    graph_factory, b
):
    graph = graph_factory()
    n_qubits = max(graph.nodes()) + 1
    neighbourhoods = {vertex: list(graph.neighbors(vertex)) for vertex in graph.nodes()}
    assert_transitions(
        bit_flip_mixer(graph, b=b),
        bit_flip_transitions(neighbourhoods, n_qubits, b),
    )


def test_bit_flip_mixer_rejects_invalid_b():
    with pytest.raises(ValueError, match="b"):
        bit_flip_mixer(nx.path_graph(3), b=2)


def test_bit_flip_mixer_rejects_non_nx_graph():
    with pytest.raises(TypeError, match="networkx"):
        bit_flip_mixer([(0, 1)], b=0)


def test_x_mixer_zero_qubits_returns_zero_operator():
    spo = x_mixer(0)
    assert spo.num_qubits == 0
    np.testing.assert_allclose(spo.coeffs, [0.0])


def test_x_mixer_rejects_negative_qubits():
    with pytest.raises(ValueError, match="non-negative"):
        x_mixer(-1)


def test_graph_builders_reject_negative_qubit_nodes():
    """``xy_mixer``, ``edge_driver``, and ``bit_flip_mixer`` share
    the ``_validate_int_nodes`` helper — exercise it via one representative
    entry point."""
    with pytest.raises(ValueError, match="non-negative"):
        bit_flip_mixer(nx.Graph([(0, -1)]), b=0)
