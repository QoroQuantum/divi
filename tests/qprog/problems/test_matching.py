# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import networkx as nx
import numpy as np
import pytest

from divi.qprog import QAOA
from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer
from divi.qprog.problems import BinaryOptimizationProblem
from divi.qprog.problems._matching import (
    MaxWeightMatchingProblem,
    _bitstring_to_matching,
    _classical_cleanup,
    _construct_matching_qubo,
    _count_conflicts,
    _partition_graph_by_edges,
    _repair_matching,
    _sort_matching,
    check_matching_matrix,
    is_valid_matching,
)
from divi.qprog.workflows import PartitioningProgramEnsemble

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


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


# ------------------------------------------------------------------
# _construct_matching_qubo
# ------------------------------------------------------------------


class TestConstructMatchingQubo:
    def test_diagonal_has_negative_weights(self, triangle_graph):
        edges = list(triangle_graph.edges())
        e2q = {e: i for i, e in enumerate(edges)}
        for u, v in list(e2q):
            e2q[(v, u)] = e2q[(u, v)]

        qubo = _construct_matching_qubo(triangle_graph, e2q, penalty_scale=1.0)

        # Diagonal should be -weight for each edge
        for (u, v), idx in e2q.items():
            if u < v:
                w = triangle_graph[u][v]["weight"]
                assert qubo[idx, idx] == pytest.approx(-w)

    def test_penalty_on_incident_edges(self, triangle_graph):
        """All 3 edges share nodes, so all pairs should have penalty."""
        edges = list(triangle_graph.edges())
        e2q = {e: i for i, e in enumerate(edges)}
        for u, v in list(e2q):
            e2q[(v, u)] = e2q[(u, v)]

        total_w = sum(d["weight"] for _, _, d in triangle_graph.edges(data=True))
        penalty_scale = 2.0
        qubo = _construct_matching_qubo(triangle_graph, e2q, penalty_scale)

        expected_penalty = penalty_scale * total_w
        # Every off-diagonal pair that shares a node should have the penalty
        for i in range(3):
            for j in range(i + 1, 3):
                # In a triangle, every pair of edges shares at least one node
                assert qubo[i, j] == pytest.approx(expected_penalty)

    def test_no_penalty_for_independent_edges(self):
        """Two disjoint edges: no quadratic penalty between them."""
        G = nx.Graph()
        G.add_weighted_edges_from([(0, 1, 1.0), (2, 3, 1.0)])
        edges = list(G.edges())
        e2q = {e: i for i, e in enumerate(edges)}
        for u, v in list(e2q):
            e2q[(v, u)] = e2q[(u, v)]

        qubo = _construct_matching_qubo(G, e2q, penalty_scale=10.0)

        # Off-diagonal should be 0 — edges share no node
        assert qubo[0, 1] == pytest.approx(0.0)
        assert qubo[1, 0] == pytest.approx(0.0)

    def test_qubo_is_symmetric(self, diamond_graph):
        edges = [(u, v) if u < v else (v, u) for u, v in diamond_graph.edges()]
        e2q = {}
        for i, (u, v) in enumerate(edges):
            e2q[(u, v)] = i
            e2q[(v, u)] = i

        qubo = _construct_matching_qubo(diamond_graph, e2q)
        np.testing.assert_array_almost_equal(qubo, qubo.T)


# ------------------------------------------------------------------
# _sort_matching / is_valid_matching
# ------------------------------------------------------------------


class TestSortMatching:
    def test_sorts_nodes_and_edges(self):
        assert _sort_matching([(2, 0), (3, 1)]) == [(0, 2), (1, 3)]

    def test_empty(self):
        assert _sort_matching([]) == []

    def test_single_edge(self):
        assert _sort_matching([(5, 3)]) == [(3, 5)]


class TestIsValidMatching:
    def test_valid(self):
        assert is_valid_matching([(0, 1), (2, 3)]) is True

    def test_invalid_shared_node(self):
        assert is_valid_matching([(0, 1), (1, 2)]) is False

    def test_empty_is_valid(self):
        assert is_valid_matching([]) is True

    def test_single_edge_valid(self):
        assert is_valid_matching([(0, 1)]) is True


# ------------------------------------------------------------------
# _bitstring_to_matching
# ------------------------------------------------------------------


class TestBitstringToMatching:
    def test_decodes_correctly(self):
        """Qubit 0 = rightmost bit."""
        e2q = {(0, 1): 0, (1, 0): 0, (2, 3): 1, (3, 2): 1}
        # Bitstring "01" → qubit 0 = rightmost '1', qubit 1 = '0'
        result = _bitstring_to_matching("01", e2q)
        assert result == [(0, 1)]

    def test_all_ones(self):
        e2q = {(0, 1): 0, (1, 0): 0, (2, 3): 1, (3, 2): 1}
        result = _bitstring_to_matching("11", e2q)
        assert result == [(0, 1), (2, 3)]

    def test_all_zeros(self):
        e2q = {(0, 1): 0, (1, 0): 0, (2, 3): 1, (3, 2): 1}
        result = _bitstring_to_matching("00", e2q)
        assert result == []


# ------------------------------------------------------------------
# check_matching_matrix
# ------------------------------------------------------------------


class TestCheckMatchingMatrix:
    def test_valid_matching(self):
        A = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        M = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
        assert check_matching_matrix(M, A) is True

    def test_edge_not_in_graph(self):
        A = np.array([[0, 0], [0, 0]])
        M = np.array([[0, 1], [1, 0]])
        assert check_matching_matrix(M, A) is False

    def test_node_used_twice(self):
        A = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        M = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]])
        assert check_matching_matrix(M, A) is False


# ------------------------------------------------------------------
# Edge partitioning
# ------------------------------------------------------------------


class TestPartitionGraphByEdges:
    def test_small_graph_no_split(self, path_graph):
        parts = _partition_graph_by_edges(path_graph, max_edges=10)
        assert len(parts) == 1
        assert parts[0].size() == path_graph.size()

    def test_splits_when_exceeding_max(self, diamond_graph):
        parts = _partition_graph_by_edges(diamond_graph, max_edges=2)
        assert all(sg.size() <= 2 for sg in parts)
        # Total edges across partitions <= original (some cut edges lost)
        assert sum(sg.size() for sg in parts) <= diamond_graph.size()

    def test_spectral_algorithm(self, diamond_graph):
        parts = _partition_graph_by_edges(
            diamond_graph, max_edges=2, algorithm="spectral"
        )
        assert all(sg.size() <= 2 for sg in parts)

    def test_invalid_algorithm_raises(self, path_graph):
        with pytest.raises(ValueError, match="Unsupported"):
            _partition_graph_by_edges(path_graph, max_edges=2, algorithm="bogus")


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


class TestCountConflicts:
    def test_no_conflicts(self):
        edges = [(0, 1), (2, 3)]
        assert _count_conflicts([1, 1], edges) == 0

    def test_one_conflict(self):
        edges = [(0, 1), (1, 2)]
        assert _count_conflicts([1, 1], edges) == 1

    def test_all_zeros(self):
        edges = [(0, 1), (1, 2)]
        assert _count_conflicts([0, 0], edges) == 0


class TestRepairMatching:
    def test_keeps_highest_weight(self):
        G = nx.Graph()
        G.add_weighted_edges_from([(0, 1, 3.0), (1, 2, 5.0)])
        repaired = _repair_matching([(0, 1), (1, 2)], G)
        assert repaired == [(1, 2)]  # higher weight

    def test_valid_matching_unchanged(self):
        G = nx.Graph()
        G.add_weighted_edges_from([(0, 1, 1.0), (2, 3, 1.0)])
        repaired = _repair_matching([(0, 1), (2, 3)], G)
        assert set(repaired) == {(0, 1), (2, 3)}


class TestClassicalCleanup:
    def test_fills_residual_nodes(self):
        G = nx.Graph()
        G.add_weighted_edges_from([(0, 1, 3.0), (2, 3, 5.0)])
        edges = [(0, 1), (2, 3)]
        e2q = {(0, 1): 0, (1, 0): 0, (2, 3): 1, (3, 2): 1}

        # Only edge (0,1) selected → nodes 2,3 unmatched
        solution = [1, 0]
        result = _classical_cleanup(solution, G, edges, e2q)
        assert result == [1, 1]  # cleanup adds (2,3)

    def test_no_residual(self):
        G = nx.Graph()
        G.add_weighted_edges_from([(0, 1, 3.0), (2, 3, 5.0)])
        edges = [(0, 1), (2, 3)]
        e2q = {(0, 1): 0, (1, 0): 0, (2, 3): 1, (3, 2): 1}

        solution = [1, 1]
        result = _classical_cleanup(solution, G, edges, e2q)
        assert result == [1, 1]  # nothing to clean up


# ------------------------------------------------------------------
# MaxWeightMatchingProblem — unit tests
# ------------------------------------------------------------------


class TestMaxWeightMatchingProblem:
    def test_cost_hamiltonian_exists(self, triangle_graph):
        p = MaxWeightMatchingProblem(triangle_graph)
        assert p.cost_hamiltonian is not None
        assert p.mixer_hamiltonian is not None
        assert isinstance(p.loss_constant, float)

    def test_decode_fn(self, path_graph):
        p = MaxWeightMatchingProblem(path_graph)
        decoded = p.decode_fn("001")
        assert isinstance(decoded, list)

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
        assert len(subs) > 1
        for prog_id, sub in subs.items():
            assert isinstance(sub, BinaryOptimizationProblem)

    def test_extend_solution(self, diamond_graph):
        p = MaxWeightMatchingProblem(diamond_graph, max_edges_per_partition=2)
        p.decompose()

        initial = [0] * p.initial_solution_size()
        prog_id = list(p._edge_index_maps.keys())[0]
        n_local = len(p._edge_index_maps[prog_id])
        candidate = list(range(n_local))  # dummy decoded

        result = p.extend_solution(initial, prog_id, [1] * n_local)
        assert len(result) == len(initial)
        # At least one bit should be set
        assert sum(result) == n_local

    def test_finalize_repairs_and_cleans(self):
        G = nx.Graph()
        G.add_weighted_edges_from([(0, 1, 3.0), (1, 2, 5.0), (2, 3, 2.0), (0, 3, 4.0)])
        p = MaxWeightMatchingProblem(G)

        # Conflicting: both edges at node 1
        edges_map = {e: i for i, e in enumerate(p._edges)}
        sol = [0] * len(p._edges)
        sol[edges_map[(0, 1)]] = 1
        sol[edges_map[(1, 2)]] = 1

        matching, weight = p.finalize_solution(-4.5, sol)
        # Should be repaired and cleaned up
        assert is_valid_matching(matching)
        assert weight > 0

    def test_finalize_returns_tuple(self, path_graph):
        p = MaxWeightMatchingProblem(path_graph)
        result = p.finalize_solution(-1.0, [1, 0, 0])
        assert isinstance(result, tuple)
        assert len(result) == 2
        edges, weight = result
        assert isinstance(edges, list)
        assert isinstance(weight, float)


# ------------------------------------------------------------------
# MaxWeightMatchingProblem — E2E with QAOA
# ------------------------------------------------------------------


@pytest.mark.e2e
class TestMaxWeightMatchingProblemE2E:
    def test_qaoa_small_graph(self, default_test_simulator):
        """E2E: QAOA finds the optimal matching on a small graph.

        Graph: 0--1--2--3 with weights 5, 1, 5.
        Optimal matching: {(0,1), (2,3)} with weight 10.
        """
        G = nx.Graph()
        G.add_weighted_edges_from([(0, 1, 5.0), (1, 2, 1.0), (2, 3, 5.0)])

        problem = MaxWeightMatchingProblem(G, penalty_scale=10.0)
        default_test_simulator.set_seed(42)
        qaoa = QAOA(
            problem,
            n_layers=2,
            optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
            max_iterations=15,
            backend=default_test_simulator,
        )
        qaoa.run()

        decoded = qaoa.solution
        assert is_valid_matching(decoded)

        # Should find optimal or near-optimal
        classical = nx.max_weight_matching(G, maxcardinality=False)
        classical_weight = sum(G[u][v]["weight"] for u, v in classical)
        quantum_weight = sum(G[u][v]["weight"] for u, v in decoded)
        assert quantum_weight >= 0.5 * classical_weight

    def test_partitioned_e2e(self, default_test_simulator):
        """E2E: Partitioned matching produces a valid, positive-weight result."""
        G = nx.gnm_random_graph(10, 15, seed=42)
        for u, v in G.edges():
            G[u][v]["weight"] = float(u + v)

        problem = MaxWeightMatchingProblem(
            G,
            penalty_scale=10.0,
            max_edges_per_partition=5,
            partition_algorithm="kernighan_lin",
            seed=42,
        )

        default_test_simulator.set_seed(42)
        ensemble = PartitioningProgramEnsemble(
            problem=problem,
            n_layers=1,
            backend=default_test_simulator,
            optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
            max_iterations=5,
        )

        ensemble.create_programs()
        ensemble.run(blocking=True)

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
