# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for SparsePauliOp-native QAOA mixer builders."""

import networkx as nx
import numpy as np
import pennylane.qaoa as pqaoa
import pytest
from qiskit.quantum_info import SparsePauliOp

from divi.hamiltonians import (
    bit_driver,
    bit_flip_mixer,
    edge_driver,
    x_mixer,
    xy_mixer,
)
from tests.hamiltonians._helpers import assert_matches_pennylane


def _assert_spo_equivalent(actual: SparsePauliOp, expected: SparsePauliOp) -> None:
    # ``simplify(atol=...)`` does not always shrink ``size`` to 0 — Qiskit can
    # leave a residual identity row with ``coeff == 0`` after cancellation, so
    # we also accept any non-empty result whose coefficients are all near zero.
    diff = (actual - expected).simplify(atol=1e-12)
    assert diff.size == 0 or np.allclose(diff.coeffs, 0, atol=1e-12)


def test_x_mixer_matches_pennylane_qaoa_x_mixer():
    actual = x_mixer(4)
    assert_matches_pennylane(actual, pqaoa.x_mixer(range(4)))


def test_xy_mixer_matches_pennylane_qaoa_xy_mixer():
    graph = nx.Graph([(0, 1), (1, 2)])
    actual = xy_mixer(graph)
    assert_matches_pennylane(actual, pqaoa.xy_mixer(graph))


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
def test_bit_driver_matches_pennylane(b):
    actual = bit_driver(n_qubits=5, b=b)
    assert_matches_pennylane(actual, pqaoa.bit_driver(range(5), b=b))


def test_bit_driver_rejects_invalid_b():
    with pytest.raises(ValueError, match="b"):
        bit_driver(n_qubits=3, b=2)


@pytest.mark.parametrize(
    "reward",
    [["10", "01"], ["00"], ["11"], ["10", "01", "00"], ["10", "01", "11"]],
)
def test_edge_driver_matches_pennylane(reward):
    graph = nx.Graph([(0, 1), (1, 2), (2, 3)])
    actual = edge_driver(graph, reward)
    assert_matches_pennylane(actual, pqaoa.edge_driver(graph, reward))


def test_edge_driver_constant_reward_set_matches_pennylane():
    graph = nx.Graph([(0, 1), (1, 2)])
    actual = edge_driver(graph, ["00", "01", "10", "11"])
    assert_matches_pennylane(actual, pqaoa.edge_driver(graph, ["00", "01", "10", "11"]))


def test_edge_driver_rejects_unpaired_directed_bits():
    with pytest.raises(ValueError, match="01"):
        edge_driver(nx.Graph([(0, 1)]), ["10"])


@pytest.mark.parametrize("b", [0, 1])
@pytest.mark.parametrize(
    "graph_factory",
    [
        lambda: nx.path_graph(4),
        lambda: nx.cycle_graph(5),
        lambda: nx.star_graph(5),  # degree-5 hub exercises 2^d expansion sign bugs
    ],
)
def test_bit_flip_mixer_matches_pennylane(graph_factory, b):
    graph = graph_factory()
    actual = bit_flip_mixer(graph, b=b)
    assert_matches_pennylane(actual, pqaoa.bit_flip_mixer(graph, b=b))


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


def test_bit_flip_mixer_isolated_node_matches_pennylane():
    graph = nx.Graph()
    graph.add_nodes_from([0, 1, 2])  # all isolated, degree 0 everywhere
    actual = bit_flip_mixer(graph, b=0)
    assert_matches_pennylane(actual, pqaoa.bit_flip_mixer(graph, 0))
