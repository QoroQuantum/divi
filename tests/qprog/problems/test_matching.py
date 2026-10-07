# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import itertools
import math
import random

import networkx as nx
import numpy as np
import pytest
import rustworkx as rx

from divi.qprog import QAOA
from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer
from divi.qprog.problems import (
    BinaryOptimizationProblem,
    MaxWeightMatchingProblem,
)
from divi.qprog.problems import _matching as _matching_module
from divi.qprog.problems import check_matching_matrix, is_valid_matching
from divi.qprog.problems._matching import (
    _bitstring_to_matching,
    _classical_cleanup,
    _construct_matching_qubo,
    _count_conflicts,
    _edge_index_graph,
    _integer_weights,
    _kl_bisect,
    _matching_edges,
    _partition_graph_by_edges,
    _repair_matching,
    _spectral_bisect,
)
from divi.qprog.workflows import PartitioningProgramEnsemble
from tests.qprog.problems._helpers import nx_twin


@pytest.fixture
def diamond_graph():
    """Diamond graph: 4 nodes, 5 edges, varied weights.

    ::

        0 --- 1
        |  X  |
        3 --- 2

    Edges: (0,1)=3, (0,2)=1, (0,3)=4, (1,2)=5, (2,3)=2
    """
    G = nx.Graph()
    G.add_weighted_edges_from(
        [
            (0, 1, 3.0),
            (0, 2, 1.0),
            (0, 3, 4.0),
            (1, 2, 5.0),
            (2, 3, 2.0),
        ]
    )
    return G


@pytest.fixture
def path_graph():
    """Path graph: 0--1--2--3 with unit weights."""
    G = nx.path_graph(4)
    for u, v in G.edges():
        G[u][v]["weight"] = 1.0
    return G


@pytest.fixture
def triangle_graph():
    """Triangle: 3 nodes, 3 edges, distinct weights."""
    G = nx.Graph()
    G.add_weighted_edges_from([(0, 1, 3.0), (1, 2, 5.0), (0, 2, 4.0)])
    return G


def _graph_qubo(graph, penalty_weight=10.0):
    _, edges, weights = _matching_edges(graph)
    return edges, weights, _construct_matching_qubo(edges, weights, penalty_weight)


def _dense_matching_qubo(edges, weights, penalty_weight):
    """Reference symmetric QUBO matrix: ``-w`` diagonal, half the penalty per incident pair entry."""
    qubo = np.diag(-np.asarray(weights, dtype=float))
    penalty = penalty_weight * sum(weights)
    for i, j in itertools.combinations(range(len(edges)), 2):
        if set(edges[i]) & set(edges[j]):
            qubo[i, j] = qubo[j, i] = penalty / 2
    return qubo


def _term_energy(terms, bits) -> float:
    return math.fsum(
        coeff * math.prod(bits[k] for k in key) for key, coeff in terms.items()
    )


def _weighted_gnm(n_nodes, n_edges, seed, draw=lambda rng: rng.uniform(0.1, 10.0)):
    graph = nx.gnm_random_graph(n_nodes, n_edges, seed=seed)
    rng = np.random.default_rng(seed)
    for u, v in graph.edges():
        graph[u][v]["weight"] = float(draw(rng))
    return graph


_QUBO_GRAPHS = [
    pytest.param(lambda: nx.path_graph(4), id="unweighted_path"),
    pytest.param(lambda: nx.complete_graph(5), id="unweighted_k5"),
    pytest.param(lambda: _weighted_gnm(7, 11, seed=0), id="gnm_7_11"),
    pytest.param(lambda: _weighted_gnm(8, 6, seed=3), id="gnm_8_6_isolated"),
    pytest.param(
        lambda: nx.star_graph(5).subgraph(range(1, 6)).copy(), id="edgeless_no_terms"
    ),
]


class TestConstructMatchingQubo:
    def test_linear_terms_are_negative_weights(self, triangle_graph):
        edges, weights, terms = _graph_qubo(triangle_graph, penalty_weight=1.0)
        assert edges == [(0, 1), (0, 2), (1, 2)]
        assert [terms[(i,)] for i in range(3)] == [-3.0, -4.0, -5.0]

    def test_penalty_on_every_incident_pair(self, triangle_graph):
        """Every pair of triangle edges shares a node."""
        _, _, terms = _graph_qubo(triangle_graph, penalty_weight=2.0)
        assert {key: c for key, c in terms.items() if len(key) == 2} == {
            (0, 1): 24.0,
            (0, 2): 24.0,
            (1, 2): 24.0,
        }

    def test_no_penalty_for_independent_edges(self):
        graph = nx.Graph()
        graph.add_weighted_edges_from([(0, 1, 1.0), (2, 3, 1.0)])
        _, _, terms = _graph_qubo(graph)
        assert terms == {(0,): -1.0, (1,): -1.0}

    @pytest.mark.parametrize("penalty_weight", [0.5, 10.0])
    @pytest.mark.parametrize("make_graph", _QUBO_GRAPHS)
    def test_terms_match_dense_reference_on_every_bitstring(
        self, make_graph, penalty_weight
    ):
        edges, weights, terms = _graph_qubo(make_graph(), penalty_weight)
        dense = _dense_matching_qubo(edges, weights, penalty_weight)
        for bits in itertools.product((0, 1), repeat=len(edges)):
            x = np.array(bits, dtype=float)
            assert _term_energy(terms, bits) == pytest.approx(x @ dense @ x)

    @pytest.mark.parametrize(
        "make_graph", [p for p in _QUBO_GRAPHS if p.id != "edgeless_no_terms"]
    )
    def test_hamiltonian_equals_dense_reference(self, make_graph):
        edges, weights, terms = _graph_qubo(make_graph())
        sparse = BinaryOptimizationProblem(terms)
        dense = BinaryOptimizationProblem(_dense_matching_qubo(edges, weights, 10.0))
        assert sparse.cost_hamiltonian.equiv(dense.cost_hamiltonian)
        assert sparse.loss_constant == pytest.approx(dense.loss_constant)

    def test_problem_hamiltonian_equals_dense_reference_on_a_larger_graph(self):
        graph = _weighted_gnm(30, 80, seed=5)
        problem = MaxWeightMatchingProblem(graph, penalty_weight=3.0)
        dense = BinaryOptimizationProblem(
            _dense_matching_qubo(problem._edges, problem._weights, 3.0)
        )
        assert problem.cost_hamiltonian.equiv(dense.cost_hamiltonian)
        assert problem.loss_constant == pytest.approx(dense.loss_constant)


@pytest.mark.parametrize(
    "matching, expected",
    [
        pytest.param([(0, 1), (2, 3)], True, id="valid"),
        pytest.param([(0, 1), (1, 2)], False, id="invalid_shared_node"),
        pytest.param([(0, 1), (0, 2)], False, id="invalid_shared_first_node"),
        pytest.param([], True, id="empty_is_valid"),
        pytest.param([(0, 1)], True, id="single_edge_valid"),
    ],
)
def test_is_valid_matching(matching, expected):
    assert is_valid_matching(matching) is expected


@pytest.mark.parametrize(
    "bitstring, expected",
    [
        pytest.param("10", [(0, 1)], id="qubit_0_is_leftmost_bit"),
        pytest.param("11", [(0, 1), (2, 3)], id="all_ones"),
        pytest.param("00", [], id="all_zeros"),
    ],
)
def test_bitstring_to_matching(bitstring, expected):
    """Qubit 0 = leftmost bit (left-to-right ordering)."""
    assert _bitstring_to_matching(bitstring, [(0, 1), (2, 3)]) == expected


_TRIANGLE_ADJACENCY = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]


@pytest.mark.parametrize(
    "matching_matrix, adjacency, expected",
    [
        pytest.param(
            [[0, 1, 0], [1, 0, 0], [0, 0, 0]],
            _TRIANGLE_ADJACENCY,
            True,
            id="valid_matching",
        ),
        pytest.param([[0, 1], [1, 0]], [[0, 0], [0, 0]], False, id="edge_not_in_graph"),
        pytest.param(
            [[0, 1, 1], [1, 0, 0], [1, 0, 0]],
            _TRIANGLE_ADJACENCY,
            False,
            id="node_used_twice",
        ),
    ],
)
def test_check_matching_matrix(matching_matrix, adjacency, expected):
    assert (
        check_matching_matrix(np.array(matching_matrix), np.array(adjacency))
        is expected
    )


@pytest.mark.parametrize(
    "matching_matrix, adjacency, message",
    [
        pytest.param(
            [[0, 1, 0], [0, 0, 0], [0, 0, 0]],
            _TRIANGLE_ADJACENCY,
            "M must be symmetric",
            id="asymmetric_matching",
        ),
        pytest.param(
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 1, 1], [0, 0, 1], [1, 1, 0]],
            "A must be symmetric",
            id="asymmetric_adjacency",
        ),
        pytest.param(
            [[0, 1, 0], [1, 0, 0]],
            _TRIANGLE_ADJACENCY,
            "M must be a square matrix",
            id="non_square_matching",
        ),
        pytest.param(
            [[0, 1], [1, 0]],
            _TRIANGLE_ADJACENCY,
            "same shape",
            id="shape_mismatch",
        ),
    ],
)
def test_check_matching_matrix_rejects_malformed_input(
    matching_matrix, adjacency, message
):
    with pytest.raises(ValueError, match=message):
        check_matching_matrix(np.array(matching_matrix), np.array(adjacency))


_PARTITION_ALGORITHMS = ["kernighan_lin", "spectral"]


def _triangle_with_isolated_nodes() -> nx.Graph:
    graph = nx.complete_graph(3)
    graph.add_nodes_from([3, 4, 5])
    return graph


def _edge_set(edges) -> list[frozenset]:
    return [frozenset(edge) for edge in edges]


def _partition(graph: nx.Graph, max_edges: int, algorithm="kernighan_lin"):
    """:func:`_partition_graph_by_edges` on ``graph``, each part as an edge subgraph."""
    ids, edges, weights = _matching_edges(graph)
    parts = _partition_graph_by_edges(edges, weights, max_edges, algorithm=algorithm)
    return [
        graph.edge_subgraph((ids[edges[i][0]], ids[edges[i][1]]) for i in part)
        for part in parts
    ]


def _bisection_input(graph: nx.Graph):
    """The :func:`_edge_index_graph` of ``graph`` and its edge weights."""
    _, edges, weights = _matching_edges(graph)
    return _edge_index_graph(edges), weights


def _assert_partitions_within_budget(graph, parts, max_edges):
    assert all(1 <= part.size() <= max_edges for part in parts)
    partition_edges = _edge_set(edge for part in parts for edge in part.edges())
    assert len(partition_edges) == len(set(partition_edges))
    assert set(partition_edges) <= set(_edge_set(graph.edges()))


class TestPartitionGraphByEdges:
    def test_small_graph_no_split(self, path_graph):
        parts = _partition(path_graph, max_edges=10)
        assert len(parts) == 1
        assert parts[0].size() == path_graph.size()

    def test_splits_when_exceeding_max(self, diamond_graph):
        parts = _partition(diamond_graph, max_edges=2)
        assert all(sg.size() <= 2 for sg in parts)
        # Total edges across partitions <= original (some cut edges lost)
        assert sum(sg.size() for sg in parts) <= diamond_graph.size()

    def test_spectral_algorithm(self, diamond_graph):
        parts = _partition(diamond_graph, max_edges=2, algorithm="spectral")
        assert all(sg.size() <= 2 for sg in parts)

    def test_spectral_partition_is_independent_of_global_rng(self):
        graph = nx.cycle_graph(8)
        random_state = np.random.get_state()
        try:
            np.random.seed(1)
            first = _partition(graph, 2, algorithm="spectral")
            np.random.seed(2)
            second = _partition(graph, 2, algorithm="spectral")
        finally:
            np.random.set_state(random_state)

        assert [set(part) for part in first] == [set(part) for part in second]

    def test_partition_order_does_not_depend_on_bisection_order(
        self, diamond_graph, mocker
    ):
        split = mocker.patch.object(
            _matching_module,
            "_kl_bisect",
            side_effect=[({2, 3}, {0, 1}), ({0, 1}, {2, 3})],
        )

        first = _partition(diamond_graph, 2)
        second = _partition(diamond_graph, 2)

        assert split.call_count == 2
        assert [set(part) for part in first] == [set(part) for part in second]

    @pytest.mark.parametrize("seed", range(3))
    def test_kernighan_lin_keeps_heavy_edges_inside_partitions(self, seed):
        graph = nx.Graph()
        graph.add_weighted_edges_from(
            [(0, 1, 10.0), (1, 2, 1.0), (2, 3, 10.0), (3, 0, 1.0)]
        )
        part_a, part_b = _kl_bisect(*_bisection_input(graph), seed=seed)
        assert {frozenset(part_a), frozenset(part_b)} == {
            frozenset({0, 1}),
            frozenset({2, 3}),
        }

    def test_invalid_algorithm_raises(self, path_graph):
        with pytest.raises(ValueError, match="Unsupported"):
            _partition(path_graph, max_edges=2, algorithm="bogus")

    @pytest.mark.parametrize("algorithm", _PARTITION_ALGORITHMS)
    def test_triangle_with_isolated_nodes_stays_within_budget(self, algorithm):
        graph = _triangle_with_isolated_nodes()
        parts = _partition(graph, max_edges=2, algorithm=algorithm)
        _assert_partitions_within_budget(graph, parts, 2)
        assert all(nx.is_connected(part) for part in parts)

    @pytest.mark.parametrize("algorithm", _PARTITION_ALGORITHMS)
    @pytest.mark.parametrize("max_edges", [1, 2, 3, 5])
    @pytest.mark.parametrize("n_nodes, n_edges", [(10, 9), (12, 20)])
    @pytest.mark.parametrize("seed", range(8))
    def test_random_graphs_stay_within_budget(
        self, seed, n_nodes, n_edges, max_edges, algorithm
    ):
        graph = nx.gnm_random_graph(n_nodes, n_edges, seed=seed)
        parts = _partition(graph, max_edges, algorithm=algorithm)
        _assert_partitions_within_budget(graph, parts, max_edges)

    @pytest.mark.parametrize("algorithm", _PARTITION_ALGORITHMS)
    def test_components_within_budget_keep_every_edge_once(self, algorithm):
        graph = nx.disjoint_union_all([nx.path_graph(3), nx.cycle_graph(3)])
        graph.add_node(99)
        parts = _partition(graph, max_edges=3, algorithm=algorithm)
        assert sorted(map(sorted, _edge_set(e for p in parts for e in p.edges()))) == (
            sorted(map(sorted, _edge_set(graph.edges())))
        )

    def test_edgeless_graph_has_no_partitions(self):
        assert _partition(nx.empty_graph(4), max_edges=1) == []

    def test_spectral_bisection_splits_tied_fiedler_entries(self):
        part_a, part_b = _spectral_bisect(*_bisection_input(nx.complete_graph(3)))
        assert sorted([len(part_a), len(part_b)]) == [1, 2]
        assert part_a | part_b == {0, 1, 2}

    def test_bisection_with_an_empty_side_raises(self, path_graph, mocker):
        mocker.patch.object(
            _matching_module, "_kl_bisect", return_value=(set(path_graph), set())
        )
        with pytest.raises(ValueError, match="max_edges=2"):
            _partition(path_graph, max_edges=2)


@pytest.mark.parametrize(
    "solution, edges, expected",
    [
        pytest.param([1, 1], [(0, 1), (2, 3)], 0, id="no_conflicts"),
        pytest.param([1, 1], [(0, 1), (1, 2)], 1, id="one_conflict"),
        pytest.param([1, 1], [(0, 2), (1, 2)], 1, id="conflict_on_second_node"),
        pytest.param([0, 0], [(0, 1), (1, 2)], 0, id="all_zeros"),
    ],
)
def test_count_conflicts(solution, edges, expected):
    assert _count_conflicts(solution, edges) == expected


@pytest.mark.parametrize(
    "weighted_edges, expected",
    [
        pytest.param([(0, 1, 3.0), (1, 2, 5.0)], [(1, 2)], id="keeps_highest_weight"),
        pytest.param(
            [(0, 1, 1.0), (1, 2, 5.0), (2, 3, 1.0)],
            [(1, 2)],
            id="heavy_middle_edge_beats_two_light",
        ),
        pytest.param(
            [(0, 1, 1.0), (2, 3, 1.0)],
            [(0, 1), (2, 3)],
            id="valid_matching_unchanged",
        ),
    ],
)
def test_repair_matching(weighted_edges, expected):
    graph = nx.Graph()
    graph.add_weighted_edges_from(weighted_edges)
    _, edges, weights = _matching_edges(graph)
    repaired = _repair_matching(range(len(edges)), edges, weights)
    assert [edges[i] for i in repaired] == expected


def _cleanup(graph: nx.Graph, solution: list[int]) -> list[int]:
    _, edges, weights = _matching_edges(graph)
    return _classical_cleanup(solution, edges, _integer_weights(weights))


@pytest.mark.parametrize(
    "weighted_edges, solution, expected",
    [
        pytest.param(
            [(0, 1, 3.0), (2, 3, 5.0)], [1, 0], [1, 1], id="fills_residual_nodes"
        ),
        pytest.param([(0, 1, 3.0), (2, 3, 5.0)], [1, 1], [1, 1], id="no_residual"),
        pytest.param(
            [(0, 1, 1.0), (1, 2, 5.0), (2, 3, 1.0)],
            [0, 0, 0],
            [0, 1, 0],
            id="prefers_weight_over_cardinality",
        ),
        pytest.param(
            [(0, 1, -1.0), (2, 3, 0.0), (4, 5, 2.0)],
            [0, 0, 0],
            [0, 0, 1],
            id="skips_non_positive_edges",
        ),
        pytest.param(
            [(0, 1, 1e-30), (1, 2, 1e30)], [0, 0], [0, 1], id="wide_weight_range"
        ),
    ],
)
def test_classical_cleanup(weighted_edges, solution, expected):
    graph = nx.Graph()
    graph.add_weighted_edges_from(weighted_edges)
    assert _cleanup(graph, solution) == expected


@pytest.mark.parametrize(
    "weights, expected",
    [
        pytest.param([1.0, 0.5, 3.0], [2**98, 2**97, 3 * 2**98], id="dyadic_exact"),
        pytest.param([0.0, -2.0], [0, 0], id="no_positive_weight"),
        pytest.param([1.0, 2.0**-60], [2**99, 2**39], id="light_weight_exact"),
        pytest.param([1.0, 2.0**-101], [2**99, 0], id="below_step_rounds_to_zero"),
    ],
)
def test_integer_weights(weights, expected):
    assert _integer_weights(weights) == expected


def test_integer_weights_keep_order_of_float_weights():
    weights = np.random.default_rng(0).uniform(0.0, 1.0, 200).tolist()
    scaled = _integer_weights(weights)
    assert np.array_equal(
        np.argsort(weights, kind="stable"), np.argsort(scaled, kind="stable")
    )
    assert max(scaled).bit_length() == 100


_CLEANUP_WEIGHT_DRAWS = [
    pytest.param(lambda rng: rng.uniform(0.1, 10.0), id="float"),
    pytest.param(lambda rng: rng.integers(1, 4), id="integer_ties"),
    pytest.param(lambda rng: rng.choice([0.1, 0.2, 0.3]), id="decimal_ties"),
    pytest.param(lambda rng: rng.normal(), id="signed"),
    pytest.param(lambda rng: 10.0 ** rng.uniform(-6, 6), id="wide_range"),
]


def _matched_weight(graph: nx.Graph, matching) -> float:
    return math.fsum(graph[u][v]["weight"] for u, v in matching)


@pytest.mark.parametrize("draw", _CLEANUP_WEIGHT_DRAWS)
@pytest.mark.parametrize("n_nodes, n_edges", [(10, 15), (16, 40), (30, 60)])
@pytest.mark.parametrize("seed", range(6))
def test_cleanup_weight_matches_networkx_matching(seed, n_nodes, n_edges, draw):
    """From an empty solution and from a partial one, the filled-in weight is optimal."""
    graph = _weighted_gnm(n_nodes, n_edges, seed, draw)
    _, edges, _ = _matching_edges(graph)
    partial = [int(i == 0) for i in range(len(edges))]
    for solution in ([0] * len(edges), partial):
        result = _cleanup(graph, solution)
        kept = [edges[i] for i, bit in enumerate(solution) if bit]
        added = [
            edges[i] for i, (old, new) in enumerate(zip(solution, result)) if new > old
        ]
        assert all(result[i] for i, bit in enumerate(solution) if bit)
        assert is_valid_matching(kept + added)
        matched = {node for edge in kept for node in edge}
        residual = graph.subgraph(node for node in graph if node not in matched)
        expected = nx.max_weight_matching(residual, maxcardinality=False)
        assert _matched_weight(graph, added) == pytest.approx(
            _matched_weight(graph, expected), rel=1e-12, abs=1e-12
        )


class TestMaxWeightMatchingProblem:
    def test_cost_hamiltonian_ground_energy_is_heaviest_single_edge(
        self, triangle_graph
    ):
        """Every pair of triangle edges shares a node, so the best matching is the
        single heaviest edge (1, 2) of weight 5, at QUBO energy -5."""
        p = MaxWeightMatchingProblem(triangle_graph)
        diagonal = np.real(np.diag(p.cost_hamiltonian.to_matrix()))
        assert diagonal.min() + p.loss_constant == pytest.approx(-5.0)

    def test_mixer_is_x_on_every_edge_qubit(self, triangle_graph):
        p = MaxWeightMatchingProblem(triangle_graph)
        assert set(p.mixer_hamiltonian.paulis.to_labels()) == {"IIX", "IXI", "XII"}

    def test_decode_fn(self, path_graph):
        """Qubit 2 is edge (2, 3) of the path 0--1--2--3."""
        p = MaxWeightMatchingProblem(path_graph)
        assert p.decode_fn("001") == [(2, 3)]

    def test_initial_solution_size(self, diamond_graph):
        p = MaxWeightMatchingProblem(diamond_graph)
        assert p.initial_solution_size() == diamond_graph.size()

    def test_evaluate_valid_matching(self, path_graph):
        """Path 0--1--2--3: select edges (0,1) and (2,3) → valid, weight 2."""
        p = MaxWeightMatchingProblem(path_graph)
        edges = p._edges
        sol = [0] * len(edges)
        for i, (u, v) in enumerate(edges):
            if (u, v) in [(0, 1), (2, 3)] or (v, u) in [(0, 1), (2, 3)]:
                sol[i] = 1

        score = p.evaluate_global_solution(sol)
        assert score == pytest.approx(-2.0)  # weight=2, 0 conflicts

    def test_evaluate_conflicting_matching(self):
        G = nx.Graph()
        G.add_weighted_edges_from([(0, 1, 1.0), (1, 2, 1.0)])
        p = MaxWeightMatchingProblem(G)
        # Both edges selected → node 1 used twice
        score = p.evaluate_global_solution([1, 1])
        assert score > -2.0  # penalized, so less negative than -2

    def test_decompose_raises_without_config(self, diamond_graph):
        p = MaxWeightMatchingProblem(diamond_graph)
        with pytest.raises(ValueError, match="max_edges_per_partition"):
            p.decompose()

    def test_decompose_returns_sub_problems(self, diamond_graph):
        p = MaxWeightMatchingProblem(diamond_graph, max_edges_per_partition=2)
        subs = p.decompose()
        # One bisection of the 4 nodes leaves two 2-node parts of at most 1 edge.
        assert len(subs) == 2
        for sub in subs.values():
            assert isinstance(sub, BinaryOptimizationProblem)

    @pytest.mark.parametrize("algorithm", _PARTITION_ALGORITHMS)
    @pytest.mark.parametrize(
        "graph, max_edges",
        [
            pytest.param(_triangle_with_isolated_nodes(), 2, id="triangle_isolated"),
            pytest.param(nx.gnm_random_graph(10, 9, seed=3), 2, id="gnm_10_9"),
            pytest.param(nx.gnm_random_graph(12, 20, seed=1), 4, id="gnm_12_20"),
        ],
    )
    def test_decompose_sub_qubos_stay_within_budget(self, graph, max_edges, algorithm):
        p = MaxWeightMatchingProblem(
            graph, max_edges_per_partition=max_edges, partition_algorithm=algorithm
        )
        subs = p.decompose()
        assert subs
        for prog_id, sub in subs.items():
            assert 1 <= len(p._edge_index_maps[prog_id]) <= max_edges
            assert sub.cost_hamiltonian.num_qubits <= max_edges

    @pytest.mark.parametrize("max_edges", [0, -1])
    def test_rejects_non_positive_partition_budget(self, diamond_graph, max_edges):
        with pytest.raises(
            ValueError, match="max_edges_per_partition must be at least 1"
        ):
            MaxWeightMatchingProblem(diamond_graph, max_edges_per_partition=max_edges)

    def test_rejects_self_loops(self, triangle_graph):
        triangle_graph.add_edge(0, 0, weight=1.0)
        with pytest.raises(ValueError, match="self-loops"):
            MaxWeightMatchingProblem(triangle_graph)

    def test_default_partitioning_is_independent_of_global_rng(self):
        graph = nx.cycle_graph(8)
        random_state = random.getstate()
        try:
            random.seed(1)
            first = MaxWeightMatchingProblem(graph, max_edges_per_partition=2)
            first.decompose()
            random.seed(2)
            second = MaxWeightMatchingProblem(graph, max_edges_per_partition=2)
            second.decompose()
        finally:
            random.setstate(random_state)

        assert first._edge_index_maps == second._edge_index_maps

    def test_extend_solution(self, diamond_graph):
        p = MaxWeightMatchingProblem(diamond_graph, max_edges_per_partition=2)
        p.decompose()

        initial = [0] * p.initial_solution_size()
        # Only a partition whose global indices differ from its local ones can
        # distinguish the mapping from the identity, so pick one of those.
        shifted = [
            (prog_id, index_map)
            for prog_id, index_map in p._edge_index_maps.items()
            if list(index_map) != list(range(len(index_map)))
        ]
        assert shifted, "no partition maps local indices away from the identity"
        prog_id, index_map = shifted[0]

        result = p.extend_solution(initial, prog_id, [1] * len(index_map))

        assert len(result) == len(initial)
        # The partition's own global positions carry the set bits; a wrong
        # local-to-global mapping would set the right count in the wrong places.
        selected = {index for index, bit in enumerate(result) if bit}
        assert selected == set(index_map)

    def test_finalize_repairs_and_cleans(self):
        G = nx.Graph()
        G.add_weighted_edges_from([(0, 1, 3.0), (1, 2, 5.0), (2, 3, 2.0), (0, 3, 4.0)])
        p = MaxWeightMatchingProblem(G)

        # Conflicting: both edges at node 1
        edges_map = {e: i for i, e in enumerate(p._edges)}
        sol = [0] * len(p._edges)
        sol[edges_map[(0, 1)]] = 1
        sol[edges_map[(1, 2)]] = 1

        with pytest.warns(UserWarning, match="was not a valid matching"):
            matching, weight = p.postprocess_candidates([(-4.5, sol)])[0]
        # Should be repaired and cleaned up
        assert is_valid_matching(matching)
        assert weight > 0

    def test_compute_energy_uses_edge_weights(self, triangle_graph):
        p = MaxWeightMatchingProblem(triangle_graph)
        assert p.compute_energy(
            "".join("1" if edge == (1, 2) else "0" for edge in p._edges)
        ) == pytest.approx(-5.0)

    def test_decompose_honours_penalty_weight(self):
        """Selecting both edges of a 2-edge path costs ``-2 + penalty_weight * 2``."""
        p = MaxWeightMatchingProblem(
            nx.path_graph(6), penalty_weight=3.0, max_edges_per_partition=2
        )
        subs = p.decompose()
        assert [sub.compute_energy("11") for sub in subs.values()] == pytest.approx(
            [4.0, 4.0]
        )

    def test_unweighted_edges_default_to_unit_weight(self):
        graph = nx.path_graph(4)
        p = MaxWeightMatchingProblem(graph)

        _, _, terms = _graph_qubo(graph)
        assert [terms[(i,)] for i in range(3)] == [-1.0, -1.0, -1.0]
        assert p.compute_energy("101") == pytest.approx(-2.0)
        assert p.evaluate_global_solution([1, 0, 1]) == pytest.approx(-2.0)
        assert p.evaluate_global_solution([1, 1, 0]) == pytest.approx(-1.0)
        assert _repair_matching([0, 1], p._edges, p._weights) == [0]
        with pytest.warns(UserWarning, match="was not a valid matching"):
            (matching, weight), *_ = p.postprocess_candidates([(0.0, [1, 1, 0])])
        assert matching == [(0, 1), (2, 3)]
        assert weight == pytest.approx(2.0)

    @pytest.mark.filterwarnings("error")
    def test_postprocess_sorts_valid_candidates_by_weight_without_warning(
        self, diamond_graph
    ):
        p = MaxWeightMatchingProblem(diamond_graph, use_classical_cleanup=False)
        light = _select_edges(p, [(0, 2)])
        heavy = _select_edges(p, [(1, 2), (0, 3)])

        out = p.postprocess_candidates([(0.0, light), (0.0, heavy)])

        assert out == [([(0, 3), (1, 2)], 9.0), ([(0, 2)], 1.0)]

    def test_finalize_returns_tuple(self, path_graph):
        p = MaxWeightMatchingProblem(path_graph)
        result = p.postprocess_candidates([(-1.0, [1, 0, 0])])[0]
        assert isinstance(result, tuple)
        assert len(result) == 2
        edges, weight = result
        assert isinstance(edges, list)
        assert isinstance(weight, float)

    def test_qubits_follow_sorted_node_positions(self):
        """Edge qubits are ordered by endpoint positions, not networkx edge order."""
        graph = nx.Graph()
        graph.add_nodes_from([0, 1, 2, 3])
        graph.add_weighted_edges_from([(2, 3, 1.0), (0, 3, 2.0), (1, 2, 3.0)])
        p = MaxWeightMatchingProblem(graph)
        assert list(graph.edges()) == [(0, 3), (1, 2), (2, 3)]
        assert p.decode_fn("100") == [(0, 3)]
        assert p._weights == [2.0, 3.0, 1.0]

    def test_matchings_report_node_labels(self):
        graph = nx.Graph()
        graph.add_nodes_from(["d", "a", ("t", 1), "b"])
        graph.add_weighted_edges_from(
            [("d", "a", 2.0), ("a", ("t", 1), 3.0), (("t", 1), "b", 2.0)]
        )
        p = MaxWeightMatchingProblem(graph)
        assert p._labelled_edges == [("d", "a"), ("a", ("t", 1)), (("t", 1), "b")]
        assert p.decode_fn("101") == [("d", "a"), (("t", 1), "b")]
        assert p.compute_energy("101") == pytest.approx(-4.0)
        assert p.postprocess_candidates([(0.0, [0, 0, 0])]) == [
            ([("d", "a"), (("t", 1), "b")], 4.0)
        ]


def _select_edges(problem: MaxWeightMatchingProblem, edges_to_select: list[tuple]):
    """Return a bit vector with the given labelled edges flagged."""
    wanted = {frozenset(edge) for edge in edges_to_select}
    return [int(frozenset(edge) in wanted) for edge in problem._labelled_edges]


class TestMaxWeightMatchingProblemStrict:
    def test_strict_valid_skips_repair_and_cleanup(self, mocker, path_graph):
        p = MaxWeightMatchingProblem(path_graph, use_classical_cleanup=True)
        repair_spy = mocker.spy(_matching_module, "_repair_matching")
        cleanup_spy = mocker.spy(_matching_module, "_classical_cleanup")

        sol = _select_edges(p, [(0, 1), (2, 3)])
        edges, weight = p.postprocess_candidates([(-99.0, sol)], strict=True)[0]

        assert is_valid_matching(edges)
        assert weight == pytest.approx(2.0)
        assert repair_spy.call_count == 0
        assert cleanup_spy.call_count == 0

    def test_strict_invalid_returns_empty_without_repair_or_cleanup(
        self, mocker, path_graph
    ):
        p = MaxWeightMatchingProblem(path_graph, use_classical_cleanup=True)
        repair_spy = mocker.spy(_matching_module, "_repair_matching")
        cleanup_spy = mocker.spy(_matching_module, "_classical_cleanup")
        sol = _select_edges(p, [(0, 1), (1, 2)])

        with pytest.warns(UserWarning, match="No valid matching candidates"):
            out = p.postprocess_candidates([(-99.0, sol)], strict=True)

        assert out == []
        assert repair_spy.call_count == 0
        assert cleanup_spy.call_count == 0

    def test_strict_empty_matching(self, path_graph):
        p = MaxWeightMatchingProblem(path_graph)
        edges, weight = p.postprocess_candidates(
            [(0.0, [0] * len(p._edges))], strict=True
        )[0]
        assert edges == []
        assert weight == 0.0

    def test_strict_raw_weight_handles_signed_weights(self):
        G = nx.Graph()
        G.add_weighted_edges_from([(0, 1, 1e16), (2, 3, -1e16), (4, 5, 1.0)])
        p = MaxWeightMatchingProblem(G)
        sol = _select_edges(p, [(0, 1), (2, 3), (4, 5)])
        edges, weight = p.postprocess_candidates([(0.0, sol)], strict=True)[0]
        assert len(edges) == 3
        assert weight == pytest.approx(1.0, abs=0.0)

    def test_format_top_strict_filters_invalid_without_repair_or_cleanup(
        self, mocker, path_graph
    ):
        p = MaxWeightMatchingProblem(path_graph, use_classical_cleanup=True)
        repair_spy = mocker.spy(_matching_module, "_repair_matching")
        cleanup_spy = mocker.spy(_matching_module, "_classical_cleanup")

        invalid = _select_edges(p, [(0, 1), (1, 2)])
        valid = _select_edges(p, [(0, 1), (2, 3)])
        out = p.postprocess_candidates([(-99.0, invalid), (-2.0, valid)], strict=True)

        assert repair_spy.call_count == 0
        assert cleanup_spy.call_count == 0
        assert out == [([(0, 1), (2, 3)], 2.0)]

    def test_format_top_strict_all_valid(self, path_graph):
        p = MaxWeightMatchingProblem(path_graph)
        a = _select_edges(p, [(0, 1)])
        b = _select_edges(p, [(2, 3)])
        out = p.postprocess_candidates([(-1.0, a), (-1.0, b)], strict=True)
        assert len(out) == 2
        assert all(is_valid_matching(edges) for edges, _ in out)

    def test_format_top_strict_all_invalid_returns_empty_with_warning(self, path_graph):
        p = MaxWeightMatchingProblem(path_graph)
        invalid = _select_edges(p, [(0, 1), (1, 2)])

        with pytest.warns(UserWarning, match="No valid matching candidates"):
            out = p.postprocess_candidates([(-99.0, invalid)], strict=True)

        assert out == []

    def test_format_top_strict_dedupes(self, path_graph):
        p = MaxWeightMatchingProblem(path_graph)
        sol = _select_edges(p, [(0, 1), (2, 3)])
        out = p.postprocess_candidates([(-2.0, sol), (-2.0, sol[:])], strict=True)
        assert len(out) == 1

    def test_format_top_strict_does_not_backfill_beyond_given_results(self, path_graph):
        p = MaxWeightMatchingProblem(path_graph)
        invalid = _select_edges(p, [(0, 1), (1, 2)])
        later_valid = _select_edges(p, [(0, 1), (2, 3)])

        with pytest.warns(UserWarning, match="No valid matching candidates"):
            out = p.postprocess_candidates([(-99.0, invalid)], strict=True)

        assert out == []
        assert p.postprocess_candidates(
            [(-99.0, invalid), (-2.0, later_valid)], strict=True
        ) == [([(0, 1), (2, 3)], 2.0)]


def _rx_none_payloads() -> rx.PyGraph:
    graph = rx.generators.cycle_graph(6)
    graph.add_edge(0, 3, None)
    return graph


def _rx_removed_node() -> rx.PyGraph:
    graph = rx.PyGraph()
    graph.add_nodes_from(range(7))
    graph.add_edges_from(
        [(0, 1, 2.0), (1, 2, 1.0), (2, 3, 4.0), (3, 4, 1.5), (4, 5, 3.0)]
        + [(5, 6, 1.0), (6, 0, 2.5), (1, 4, 0.5)]
    )
    graph.remove_node(2)
    return graph


def _rx_weighted(wrap) -> rx.PyGraph:
    graph = rx.PyGraph()
    graph.add_nodes_from(["same"] * 6)
    graph.add_edges_from(
        [
            (u, v, wrap(w))
            for u, v, w in [(4, 3, 2.0), (3, 0, 3.0), (0, 1, 1.0), (2, 4, 1.5)]
            + [(5, 1, 2.5), (2, 5, 0.5), (1, 3, 4.0)]
        ]
    )
    return graph


_RX_MATCHING_GRAPHS = [
    pytest.param(_rx_none_payloads, id="none-payloads"),
    pytest.param(_rx_removed_node, id="removed-node"),
    pytest.param(lambda: _rx_weighted(lambda w: w), id="numeric-weights"),
    pytest.param(
        lambda: _rx_weighted(lambda w: {"weight": w, "colour": "red"}),
        id="dict-weights",
    ),
]


def _twin_problems(graph, **kwargs):
    return (
        MaxWeightMatchingProblem(graph, **kwargs),
        MaxWeightMatchingProblem(nx_twin(graph), **kwargs),
    )


def _all_bitstrings(n: int) -> list[str]:
    return ["".join(bits) for bits in itertools.product("01", repeat=n)]


@pytest.mark.parametrize("make_graph", _RX_MATCHING_GRAPHS)
def test_rx_matching_qubo_and_energies_match_nx_twin(make_graph):
    graph = make_graph()
    rx_problem, nx_problem = _twin_problems(graph)

    assert rx_problem.graph is graph
    assert rx_problem._edges == nx_problem._edges
    assert rx_problem._labelled_edges == nx_problem._labelled_edges
    assert set(rx_problem._labelled_edges) == {
        tuple(sorted(e)) for e in graph.edge_list()
    }
    assert rx_problem.cost_hamiltonian == nx_problem.cost_hamiltonian
    assert rx_problem.loss_constant == pytest.approx(nx_problem.loss_constant)
    for bitstring in _all_bitstrings(len(rx_problem._edges)):
        assert rx_problem.decode_fn(bitstring) == nx_problem.decode_fn(bitstring)
        assert rx_problem.compute_energy(bitstring) == nx_problem.compute_energy(
            bitstring
        )
        solution = [int(bit) for bit in bitstring]
        assert rx_problem.evaluate_global_solution(
            solution
        ) == nx_problem.evaluate_global_solution(solution)


@pytest.mark.parametrize("algorithm", _PARTITION_ALGORITHMS)
@pytest.mark.parametrize("make_graph", _RX_MATCHING_GRAPHS)
def test_rx_matching_decomposes_like_nx_twin(make_graph, algorithm):
    rx_problem, nx_problem = _twin_problems(
        make_graph(), max_edges_per_partition=2, partition_algorithm=algorithm
    )

    rx_subs, nx_subs = rx_problem.decompose(), nx_problem.decompose()

    assert list(rx_subs) == list(nx_subs)
    assert rx_problem._edge_index_maps == nx_problem._edge_index_maps
    for prog_id, sub in rx_subs.items():
        assert sub.cost_hamiltonian == nx_subs[prog_id].cost_hamiltonian


@pytest.mark.filterwarnings("ignore:At least one partition aggregate")
@pytest.mark.filterwarnings("ignore:No valid matching candidates")
@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("use_classical_cleanup", [True, False])
@pytest.mark.parametrize("make_graph", _RX_MATCHING_GRAPHS)
def test_rx_matching_postprocesses_like_nx_twin(
    make_graph, use_classical_cleanup, strict
):
    rx_problem, nx_problem = _twin_problems(
        make_graph(), use_classical_cleanup=use_classical_cleanup
    )
    rng = np.random.default_rng(0)
    candidates = [
        (0.0, rng.integers(0, 2, len(rx_problem._edges)).tolist()) for _ in range(8)
    ]

    assert rx_problem.postprocess_candidates(
        candidates, strict=strict
    ) == nx_problem.postprocess_candidates(candidates, strict=strict)


def _with_self_loop(graph):
    if isinstance(graph, rx.PyGraph):
        graph.add_edge(1, 1, 1.0)
    else:
        graph.add_edge(1, 1, weight=1.0)
    return graph


@pytest.mark.parametrize("backend", ["rx", "nx"])
def test_matching_rejects_self_loops_alike(backend):
    graph = _rx_none_payloads()
    graph = _with_self_loop(graph if backend == "rx" else nx_twin(graph))
    with pytest.raises(ValueError, match="self-loops"):
        MaxWeightMatchingProblem(graph)


@pytest.mark.parametrize("max_edges", [0, -1])
@pytest.mark.parametrize("backend", ["rx", "nx"])
def test_matching_rejects_non_positive_budget_alike(backend, max_edges):
    graph = _rx_none_payloads()
    with pytest.raises(ValueError, match="max_edges_per_partition must be at least 1"):
        MaxWeightMatchingProblem(
            graph if backend == "rx" else nx_twin(graph),
            max_edges_per_partition=max_edges,
        )


@pytest.mark.parametrize(
    "graph",
    [
        pytest.param(rx.generators.directed_path_graph(3), id="rx"),
        pytest.param(nx.path_graph(3, create_using=nx.DiGraph), id="nx"),
    ],
)
def test_matching_rejects_directed_graphs(graph):
    with pytest.raises(TypeError, match="Expected an undirected graph"):
        MaxWeightMatchingProblem(graph)


@pytest.mark.e2e
class TestMaxWeightMatchingProblemE2E:
    def test_qaoa_small_graph(self, default_test_simulator):
        """E2E: QAOA finds the optimal matching on a small graph.

        Graph: 0--1--2--3 with weights 5, 1, 5.
        Optimal matching: {(0,1), (2,3)} with weight 10.
        """
        G = nx.Graph()
        G.add_weighted_edges_from([(0, 1, 5.0), (1, 2, 1.0), (2, 3, 5.0)])

        problem = MaxWeightMatchingProblem(G, penalty_weight=10.0)
        default_test_simulator.set_seed(42)
        qaoa = QAOA(
            problem,
            n_layers=2,
            optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
            max_iterations=15,
            backend=default_test_simulator,
        )
        qaoa.run()

        # Use feasibility-filtered ranking (PHQC post-processing) to find
        # the best valid matching by objective energy, rather than relying
        # on the single highest-probability bitstring.
        top = qaoa.get_top_solutions(n=1, feasibility="filter", include_decoded=True)
        assert len(top) >= 1
        decoded = top[0].decoded
        assert is_valid_matching(decoded)

        classical = nx.max_weight_matching(G, maxcardinality=False)
        classical_weight = sum(G[u][v]["weight"] for u, v in classical)
        quantum_weight = sum(G[u][v]["weight"] for u, v in decoded)
        assert quantum_weight >= 0.5 * classical_weight

    @pytest.mark.filterwarnings(
        "ignore:At least one partition aggregate was not a valid matching"
    )
    def test_partitioned_e2e(self, default_test_simulator):
        """E2E: Partitioned matching produces a valid, positive-weight result.

        ``max_edges_per_partition`` is set so one bisection keeps most edges
        inside a partition (11 of 15). Under this seed the stitched aggregate
        conflicts across the cut and is repaired;
        ``TestMaxWeightMatchingProblemStrict`` covers the repair path explicitly.
        """
        G = nx.gnm_random_graph(10, 15, seed=42)
        for u, v in G.edges():
            G[u][v]["weight"] = float(u + v)

        problem = MaxWeightMatchingProblem(
            G,
            penalty_weight=10.0,
            max_edges_per_partition=8,
            partition_algorithm="kernighan_lin",
            seed=42,
        )

        default_test_simulator.set_seed(42)
        ensemble = PartitioningProgramEnsemble(
            problem=problem,
            n_layers=2,
            backend=default_test_simulator,
            optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
            max_iterations=5,
            seed=42,
        )

        ensemble.run()

        matching, weight = ensemble.aggregate_results()
        assert is_valid_matching(matching)
        assert len(matching) > 0

        # All returned edges must exist in the original graph
        for u, v in matching:
            assert G.has_edge(u, v)

        # Weight must be consistent with the edges
        expected_weight = sum(G[u][v]["weight"] for u, v in matching)
        assert weight == pytest.approx(expected_weight)

        # Should achieve at least some fraction of classical optimal
        classical = nx.max_weight_matching(G, maxcardinality=False)
        classical_weight = sum(G[u][v]["weight"] for u, v in classical)
        assert weight > 0
        assert weight <= classical_weight  # can't beat optimal
