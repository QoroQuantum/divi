# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import sys
from collections.abc import Sequence

try:
    import pymetis
except ImportError:
    pymetis = None

PYMETIS_AVAILABLE = pymetis is not None

import networkx as nx
import pytest
import rustworkx as rx

_skip_no_pymetis = pytest.mark.skipif(
    sys.platform == "win32" and not PYMETIS_AVAILABLE,
    reason="pymetis not available (install via conda: conda install -c conda-forge pymetis)",
)

from divi.qprog.problems import GraphPartitioningConfig, _graph_partitioning_utils
from divi.qprog.problems._graph_partitioning_utils import (
    _apply_split_with_relabel,
    _bisect_with_predicate,
    _node_partition_graph,
    _pygraph_to_nx,
    _split_graph,
    draw_partitions,
)
from divi.qprog.problems._graphs import MaxCutProblem

# ---------------------------------------------------------------------------
# Shared helpers and graph-type parametrization fixtures
# ---------------------------------------------------------------------------


def _make_pygraph_cycle(n: int) -> rx.PyGraph:
    g: rx.PyGraph = rx.PyGraph()
    for i in range(n):
        g.add_node(i)
    for i in range(n):
        g.add_edge(i, (i + 1) % n, None)
    return g


def _make_pygraph_complete(n: int) -> rx.PyGraph:
    g: rx.PyGraph = rx.PyGraph()
    for i in range(n):
        g.add_node(i)
    for i in range(n):
        for j in range(i + 1, n):
            g.add_edge(i, j, None)
    return g


def _make_pygraph_path(n: int) -> rx.PyGraph:
    g: rx.PyGraph = rx.PyGraph()
    for i in range(n):
        g.add_node(i)
    for i in range(n - 1):
        g.add_edge(i, i + 1, None)
    return g


# Parametrization fixtures: each entry is (cycle, complete, path) factory tuple.
GRAPH_FACTORIES = [
    pytest.param(
        (nx.cycle_graph, nx.complete_graph, nx.path_graph),
        id="nx",
    ),
    pytest.param(
        (_make_pygraph_cycle, _make_pygraph_complete, _make_pygraph_path),
        id="rx",
    ),
]


def _node_set(graph) -> set:
    if isinstance(graph, rx.PyGraph):
        return set(graph.node_indexes())
    return set(graph.nodes)


def _num_nodes(graph) -> int:
    if isinstance(graph, rx.PyGraph):
        return graph.num_nodes()
    return graph.number_of_nodes()


def _assert_partitions_correct(
    original_graph,
    clusters,
    expected_n_clusters: int | None = None,
):
    """Validate that ``clusters`` is a correct partition of ``original_graph``.

    ``clusters`` is a sequence of ``(relabeled_subgraph, cluster_ids)`` pairs.
    Each subgraph must be locally indexed ``0..M-1`` (the partitioning util's
    contract), and the union of ``cluster_ids`` must equal the original
    graph's node set.  Works for both ``nx.Graph`` and ``rx.PyGraph``.
    """
    if expected_n_clusters is not None:
        assert len(clusters) == expected_n_clusters

    expected_subgraph_type = (
        rx.PyGraph if isinstance(original_graph, rx.PyGraph) else nx.Graph
    )
    all_ids: set = set()
    for sub, cluster_ids in clusters:
        assert isinstance(sub, expected_subgraph_type)
        sub_size = _num_nodes(sub)
        assert sub_size == len(cluster_ids)
        # Subgraphs are uniformly relabeled to 0..M-1.
        assert _node_set(sub) == set(range(sub_size))
        all_ids |= set(cluster_ids)
    assert _node_set(original_graph) == all_ids

    # Pairwise disjoint.
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            assert set(clusters[i][1]).isdisjoint(set(clusters[j][1]))


# ---------------------------------------------------------------------------
# GraphPartitioningConfig
# ---------------------------------------------------------------------------


@pytest.fixture
def _raise_qubit_ceiling(mocker):
    mocker.patch.object(_graph_partitioning_utils, "_MAXIMUM_AVAILABLE_QUBITS", 10_000)


class TestGraphPartitioningConfig:
    def test_valid_max_nodes_only(self):
        config = GraphPartitioningConfig(max_n_nodes_per_cluster=10)
        assert config.max_n_nodes_per_cluster == 10
        assert config.minimum_n_clusters is None

    def test_valid_min_clusters_only(self):
        config = GraphPartitioningConfig(minimum_n_clusters=2)
        assert config.minimum_n_clusters == 2
        assert config.max_n_nodes_per_cluster is None

    def test_valid_both_constraints(self):
        config = GraphPartitioningConfig(
            max_n_nodes_per_cluster=5,
            minimum_n_clusters=3,
            partitioning_algorithm="metis",
        )
        assert config.max_n_nodes_per_cluster == 5
        assert config.minimum_n_clusters == 3
        assert config.partitioning_algorithm == "metis"

    def test_default_algorithm(self):
        # Introspect the default value from the dataclass field
        field_info = GraphPartitioningConfig.__dataclass_fields__[
            "partitioning_algorithm"
        ]
        default_value = field_info.default

        config = GraphPartitioningConfig(max_n_nodes_per_cluster=1)
        assert config.partitioning_algorithm == default_value

    def test_invalid_no_constraints(self):
        with pytest.raises(
            ValueError, match="At least one constraint must be specified."
        ):
            GraphPartitioningConfig()

    def test_invalid_min_clusters_zero(self):
        with pytest.raises(
            ValueError, match="'minimum_n_clusters' must be a positive integer."
        ):
            GraphPartitioningConfig(minimum_n_clusters=0)

    def test_invalid_max_nodes_zero(self):
        with pytest.raises(
            ValueError, match="'max_n_nodes_per_cluster' must be a positive number."
        ):
            GraphPartitioningConfig(max_n_nodes_per_cluster=0)

    def test_invalid_algorithm(self):
        with pytest.raises(ValueError, match="Unsupported partitioning algorithm:.*"):
            GraphPartitioningConfig(
                max_n_nodes_per_cluster=3, partitioning_algorithm="louvain"
            )

    def test_negative_values(self):
        with pytest.raises(ValueError):
            GraphPartitioningConfig(minimum_n_clusters=-5)

        with pytest.raises(ValueError):
            GraphPartitioningConfig(max_n_nodes_per_cluster=-1)

    def test_valid_algorithm_variants(self):
        for algo in ["spectral", "metis", "kernighan_lin"]:
            config = GraphPartitioningConfig(
                minimum_n_clusters=1, partitioning_algorithm=algo
            )
            assert config.partitioning_algorithm == algo

    @pytest.mark.parametrize("graph_factories", GRAPH_FACTORIES)
    @pytest.mark.parametrize(
        "algorithm",
        [
            "spectral",
            pytest.param("metis", marks=_skip_no_pymetis),
        ],
    )
    def test_apply_split_with_relabel_mocked(self, algorithm, graph_factories, mocker):
        cycle_factory, _, path_factory = graph_factories
        # Cycle for spectral (symmetric), path for metis (mirrors prior fixtures).
        G = cycle_factory(6) if algorithm == "spectral" else path_factory(6)

        if algorithm == "spectral":
            mock_cls = mocker.patch(
                f"{_graph_partitioning_utils.__name__}.SpectralClustering"
            )
            mock_cls.return_value.fit_predict.return_value = [0, 0, 0, 1, 1, 1]
        else:
            mock_part_graph = mocker.patch("pymetis.part_graph")
            mock_part_graph.return_value = (None, [0, 0, 0, 1, 1, 1])

        clusters = _apply_split_with_relabel(G, algorithm=algorithm, n_clusters=2)

        _assert_partitions_correct(G, clusters, expected_n_clusters=2)
        if algorithm == "spectral":
            mock_cls.return_value.fit_predict.assert_called_once()
        else:
            mock_part_graph.assert_called_once()

    def test_apply_split_with_relabel_invalid_algorithm_raises(self):
        with pytest.raises(RuntimeError, match="Relabeling only needed"):
            _apply_split_with_relabel(
                nx.path_graph(4), algorithm="kernighan_lin", n_clusters=2
            )

    def test_apply_split_with_relabel_drops_empty_clusters_with_warning(self, mocker):
        G = nx.cycle_graph(6)
        mock_cls = mocker.patch(
            f"{_graph_partitioning_utils.__name__}.SpectralClustering"
        )
        # All 6 nodes assigned to cluster 0; cluster 1 is empty.
        mock_cls.return_value.fit_predict.return_value = [0, 0, 0, 0, 0, 0]

        with pytest.warns(UserWarning, match="empty clusters were dropped"):
            result = _apply_split_with_relabel(G, algorithm="spectral", n_clusters=2)

        assert len(result) == 1
        sub, ids = result[0]
        assert _num_nodes(sub) == 6
        assert set(ids) == set(G.nodes())

    @pytest.mark.parametrize("algorithm", ["metis", "spectral"])
    def test_split_graph(self, algorithm, mocker):
        """_split_graph routes to _apply_split_with_relabel with the correct args.

        Note: this is a delegation test. _split_graph is a thin router; the
        real partitioning logic is tested in test_apply_split_with_relabel_*.
        """
        G = nx.path_graph(9)
        config = GraphPartitioningConfig(
            minimum_n_clusters=3, partitioning_algorithm=algorithm
        )

        mock_split = mocker.patch(
            f"{_apply_split_with_relabel.__module__}.{_apply_split_with_relabel.__name__}"
        )
        mock_split.return_value = (
            (G.subgraph([0, 1, 2]).copy(), [0, 1, 2]),
            (G.subgraph([3, 4, 5]).copy(), [3, 4, 5]),
            (G.subgraph([6, 7, 8]).copy(), [6, 7, 8]),
        )

        result = _split_graph(G, config)

        assert len(result) == 3
        mock_split.assert_called_once_with(G, algorithm, 3)

    def test_split_graph_kernighan_lin(self):
        G = nx.path_graph(6)
        config = GraphPartitioningConfig(
            minimum_n_clusters=10, partitioning_algorithm="kernighan_lin"
        )

        result = _split_graph(G, config)

        assert isinstance(result, Sequence)
        assert len(result) == 2
        assert set(result[0][1]) | set(result[1][1]) == set(G.nodes)

    def test_split_graph_kernighan_lin_returns_copies(self):
        """Subgraphs returned by _split_graph are independent copies, not views."""
        G = nx.path_graph(6)
        config = GraphPartitioningConfig(
            minimum_n_clusters=2, partitioning_algorithm="kernighan_lin"
        )

        (sg1, _), (sg2, _) = _split_graph(G, config)
        original_node_count = G.number_of_nodes()

        # Mutating a partition must not affect the parent graph
        sg1.add_node(999)
        assert G.number_of_nodes() == original_node_count

    def test_split_graph_kernighan_lin_preserves_pygraph_type(self):
        """KL round-trips PyGraph input through nx and back to PyGraph."""
        G = _make_pygraph_path(8)
        config = GraphPartitioningConfig(
            minimum_n_clusters=2, partitioning_algorithm="kernighan_lin"
        )

        result = _split_graph(G, config)

        assert len(result) == 2
        for sub, ids in result:
            assert isinstance(sub, rx.PyGraph)
            assert set(sub.node_indexes()) == set(range(sub.num_nodes()))
        assert set(result[0][1]) | set(result[1][1]) == set(G.node_indexes())
        assert set(result[0][1]).isdisjoint(set(result[1][1]))

    @pytest.mark.parametrize("graph_factories", GRAPH_FACTORIES)
    def test_apply_split_with_relabel_returns_copies(self, graph_factories, mocker):
        """Subgraphs returned by _apply_split_with_relabel are independent copies."""
        cycle_factory, _, _ = graph_factories
        G = cycle_factory(6)

        mock_spectral_cls = mocker.patch(
            f"{_graph_partitioning_utils.__name__}.SpectralClustering"
        )
        mock_spectral_cls.return_value.fit_predict.return_value = [0, 0, 0, 1, 1, 1]

        clusters = _apply_split_with_relabel(G, algorithm="spectral", n_clusters=2)
        original_node_count = _num_nodes(G)

        # Mutating the subgraph copy must not affect the parent graph.
        sub, _ids = clusters[0]
        sub.add_node(999)
        assert _num_nodes(G) == original_node_count

    def test_split_graph_unsupported_algorithm_raises(self):
        """_split_graph raises ValueError for unsupported algorithms."""
        G = nx.path_graph(4)
        config = GraphPartitioningConfig(minimum_n_clusters=2)
        # Bypass __post_init__ validation by setting the attribute directly
        object.__setattr__(config, "partitioning_algorithm", "bogus")

        with pytest.raises(ValueError, match="Unsupported partitioning algorithm"):
            _split_graph(G, config)

    def test_predicate_receives_correct_arguments(self):
        G1 = nx.path_graph(4)
        G2 = nx.path_graph(3)
        initial = [
            (-G1.number_of_nodes(), 0, G1, list(G1.nodes())),
            (-G2.number_of_nodes(), 1, G2, list(G2.nodes())),
        ]
        called_args = []

        def predicate(subgraph, others):
            called_args.append((subgraph, list(others)))
            return False

        _bisect_with_predicate(initial, predicate, partitioning_config=None)

        # Ensure predicate was called at least once with a non-empty others list.
        assert len(called_args) == 2
        assert any(others for _, others in called_args)

        # Check the types and contents
        for subgraph, others in called_args:
            assert isinstance(subgraph, nx.Graph)
            assert isinstance(others, list)
            # HeapEntry = (int, int, GraphProblemTypes, list)
            for entry in others:
                assert isinstance(entry, tuple)
                assert len(entry) == 4
                assert isinstance(entry[0], int)
                assert isinstance(entry[1], int)
                assert isinstance(entry[2], nx.Graph)
                assert isinstance(entry[3], list)

    def test_no_split_predicate(self):
        G = nx.path_graph(4)
        initial = [(-G.number_of_nodes(), 0, G, list(G.nodes()))]

        # Predicate always False, so no splitting
        predicate = lambda _, __: False

        result = _bisect_with_predicate(initial, predicate, partitioning_config=None)

        assert len(result) == 1
        _assert_partitions_correct(G, [(result[0][2], result[0][3])], 1)

    def test_single_split_predicate(self, mocker):
        G = nx.path_graph(4)
        initial = [(-G.number_of_nodes(), 0, G, list(G.nodes()))]

        split_called = False

        def predicate(subgraph, others):
            # Split only the initial graph, not the new ones
            nonlocal split_called

            if not split_called:
                split_called = True
                return True
            return False

        # We'll mock _split_graph to split G into two halves with cluster IDs.
        # Each child subgraph is relabeled to 0..M-1 (the partitioner contract).
        def fake_split(graph, config):
            sub_lower = nx.relabel_nodes(graph.subgraph([0, 1]).copy(), {0: 0, 1: 1})
            sub_upper = nx.relabel_nodes(graph.subgraph([2, 3]).copy(), {2: 0, 3: 1})
            return ((sub_lower, [0, 1]), (sub_upper, [2, 3]))

        mocker.patch(
            f"{_split_graph.__module__}.{_split_graph.__name__}", side_effect=fake_split
        )

        result = _bisect_with_predicate(initial, predicate, partitioning_config=None)

        # After one split, we expect 2 subgraphs
        assert len(result) == 2
        _assert_partitions_correct(G, [(r[2], r[3]) for r in result], 2)

    def test_multiple_splits_until_predicate_false(self, mocker):
        G = nx.path_graph(8)
        initial = [(-G.number_of_nodes(), 0, G, list(G.nodes()))]

        # We'll split any graph with more than 2 nodes into halves
        def predicate(subgraph, others):
            return subgraph.number_of_nodes() > 2

        def fake_split(graph, config):
            nodes = list(graph.nodes)
            half = len(nodes) // 2
            cluster_1, cluster_2 = nodes[:half], nodes[half:]
            # Each child subgraph is relabeled to 0..M-1; the cluster lists
            # carry the parent-frame labels for local nodes.
            sub1 = nx.relabel_nodes(
                graph.subgraph(cluster_1).copy(),
                {n: i for i, n in enumerate(cluster_1)},
            )
            sub2 = nx.relabel_nodes(
                graph.subgraph(cluster_2).copy(),
                {n: i for i, n in enumerate(cluster_2)},
            )
            return ((sub1, cluster_1), (sub2, cluster_2))

        mocker.patch(
            f"{_split_graph.__module__}.{_split_graph.__name__}", side_effect=fake_split
        )

        result = _bisect_with_predicate(initial, predicate, partitioning_config=None)

        # Now all subgraphs must have <= 2 nodes
        for _, _, sg, _ in result:
            assert sg.number_of_nodes() <= 2

        _assert_partitions_correct(G, [(r[2], r[3]) for r in result], 4)

    def test_node_partition_raises_if_min_clusters_too_high(self):
        G = nx.path_graph(5)
        config = GraphPartitioningConfig(minimum_n_clusters=6)

        with pytest.raises(ValueError, match="Number of requested clusters"):
            _node_partition_graph(G, config)

    def test_partition_warns_for_oversized_clusters(self, mocker):
        # Mock the maximum available qubits to a smaller number for the test
        mocker.patch.object(_graph_partitioning_utils, "_MAXIMUM_AVAILABLE_QUBITS", 20)

        graph = nx.complete_graph(40)
        config = GraphPartitioningConfig(minimum_n_clusters=1)  # No splitting

        with pytest.warns(UserWarning, match="At least one cluster has more nodes"):
            partitions = _node_partition_graph(graph, config)

        # Even with the warning, the result should be a single partition
        assert len(partitions) == 1
        assert partitions[0][0].number_of_nodes() == 40
        _assert_partitions_correct(graph, partitions)

    def test_min_clusters_enforced(self, mocker):
        graph = nx.cycle_graph(6)
        # Mocked bisection output: each subgraph already relabeled to 0..M-1
        # per the partitioner contract; cluster_ids carry parent-frame labels.
        mock_bisect = mocker.patch(
            f"{_bisect_with_predicate.__module__}.{_bisect_with_predicate.__name__}",
            return_value=[
                (0, 0, nx.Graph([(0, 1), (1, 2)]), [0, 1, 2]),
                (0, 1, nx.Graph([(0, 1), (1, 2)]), [3, 4, 5]),
            ],
        )

        config = GraphPartitioningConfig(minimum_n_clusters=2)
        result = _node_partition_graph(graph, config)

        _assert_partitions_correct(graph, result, expected_n_clusters=2)
        assert mock_bisect.call_count >= 1

    @pytest.mark.usefixtures("_raise_qubit_ceiling")
    @pytest.mark.parametrize("graph_factories", GRAPH_FACTORIES)
    @pytest.mark.parametrize(
        "algorithm",
        [
            "spectral",
            pytest.param("metis", marks=_skip_no_pymetis),
            "kernighan_lin",
        ],
    )
    def test_partition_with_min_clusters(self, algorithm, graph_factories):
        _, complete_factory, _ = graph_factories
        G = complete_factory(100)
        n_clusters = 6
        config = GraphPartitioningConfig(
            minimum_n_clusters=n_clusters, partitioning_algorithm=algorithm
        )
        partitions = _node_partition_graph(G, config)
        _assert_partitions_correct(G, partitions, expected_n_clusters=n_clusters)

    @pytest.mark.usefixtures("_raise_qubit_ceiling")
    @pytest.mark.parametrize("graph_factories", GRAPH_FACTORIES)
    @pytest.mark.parametrize(
        "algorithm",
        [
            "spectral",
            pytest.param("metis", marks=_skip_no_pymetis),
            "kernighan_lin",
        ],
    )
    def test_partition_with_max_nodes(self, algorithm, graph_factories):
        _, complete_factory, _ = graph_factories
        G = complete_factory(100)
        max_nodes = 20
        config = GraphPartitioningConfig(
            max_n_nodes_per_cluster=max_nodes, partitioning_algorithm=algorithm
        )

        partitions = _node_partition_graph(G, config)
        _assert_partitions_correct(G, partitions)

        for sub, _ in partitions:
            assert _num_nodes(sub) <= max_nodes

    @pytest.mark.usefixtures("_raise_qubit_ceiling")
    @pytest.mark.parametrize("graph_factories", GRAPH_FACTORIES)
    @pytest.mark.parametrize(
        "algorithm",
        [
            "spectral",
            pytest.param("metis", marks=_skip_no_pymetis),
            "kernighan_lin",
        ],
    )
    def test_partition_with_both_constraints(self, algorithm, graph_factories):
        _, complete_factory, _ = graph_factories
        G = complete_factory(100)

        min_clusters = 3
        max_nodes = 15
        config = GraphPartitioningConfig(
            minimum_n_clusters=min_clusters,
            max_n_nodes_per_cluster=max_nodes,
            partitioning_algorithm=algorithm,
        )
        partitions = _node_partition_graph(G, config)

        # The final number of partitions should be at least min_clusters
        assert len(partitions) >= min_clusters

        # All partitions must be smaller than max_nodes
        for sub, _ in partitions:
            assert _num_nodes(sub) <= max_nodes

        _assert_partitions_correct(G, partitions)


# ---------------------------------------------------------------------------
# rx.PyGraph end-to-end (decomposition path)
# ---------------------------------------------------------------------------


class TestPyGraphPartitioning:
    """Coverage for the rustworkx ``PyGraph`` end-to-end decomposition path.

    Algorithm-level coverage (apply_split / partition_with_*) is parametrized
    across graph types in ``TestGraphPartitioningConfig`` above; this class
    holds only the rx-specific end-to-end test that has no nx counterpart.
    """

    @pytest.mark.usefixtures("_raise_qubit_ceiling")
    def test_decompose_with_pygraph_input(self):
        """End-to-end: ``_GraphProblemBase.decompose`` honours rx.PyGraph input."""
        g = _make_pygraph_cycle(6)
        problem = MaxCutProblem(
            g,
            config=GraphPartitioningConfig(
                minimum_n_clusters=2, partitioning_algorithm="spectral"
            ),
        )

        sub_problems = problem.decompose()

        assert len(sub_problems) >= 2
        # Each sub-problem must be backed by a rx.PyGraph relabeled to 0..M-1.
        all_orig_ids: set = set()
        for prog_id, sub in sub_problems.items():
            assert isinstance(sub.graph, rx.PyGraph)
            local_to_global = problem._reverse_index_maps[prog_id]
            assert set(local_to_global) == set(range(sub.graph.num_nodes()))
            all_orig_ids |= set(local_to_global.values())
        # Reverse maps collectively cover the original graph's node indices.
        assert all_orig_ids == set(g.node_indexes())


# ---------------------------------------------------------------------------
# _pygraph_to_nx — payload handling
# ---------------------------------------------------------------------------


class TestPyGraphToNxConversion:
    def test_dict_payloads_pass_through(self):
        g: rx.PyGraph = rx.PyGraph()
        a, b, c = g.add_node("a"), g.add_node("b"), g.add_node("c")
        g.add_edge(a, b, {"weight": 1.5, "label": "ab"})
        g.add_edge(b, c, {"weight": 2.5})

        nx_g = _pygraph_to_nx(g)

        assert nx_g.number_of_nodes() == 3
        assert nx_g[a][b]["weight"] == 1.5
        assert nx_g[a][b]["label"] == "ab"
        assert nx_g[b][c]["weight"] == 2.5

    @pytest.mark.parametrize(
        "payload, expected_weight",
        [
            pytest.param(3.14, 3.14, id="float"),
            pytest.param(5, 5.0, id="int"),
        ],
    )
    def test_numeric_payloads_become_weight_attribute(self, payload, expected_weight):
        g: rx.PyGraph = rx.PyGraph()
        a, b = g.add_node("a"), g.add_node("b")
        g.add_edge(a, b, payload)

        nx_g = _pygraph_to_nx(g)

        assert nx_g[a][b]["weight"] == pytest.approx(expected_weight)

    def test_none_payload_unweighted(self):
        g: rx.PyGraph = rx.PyGraph()
        a, b = g.add_node("a"), g.add_node("b")
        g.add_edge(a, b, None)

        nx_g = _pygraph_to_nx(g)

        assert nx_g.has_edge(a, b)
        assert "weight" not in nx_g[a][b]

    def test_opaque_payload_warns_and_drops(self):
        g: rx.PyGraph = rx.PyGraph()
        a, b = g.add_node("a"), g.add_node("b")
        g.add_edge(a, b, ("custom", "tuple"))

        with pytest.warns(UserWarning, match="dropped non-dict, non-numeric"):
            nx_g = _pygraph_to_nx(g)

        assert nx_g.has_edge(a, b)
        assert "weight" not in nx_g[a][b]

    def test_no_kwarg_collision_with_reserved_keys(self):
        """nx.Graph.add_edge has reserved kwarg names like ``u_of_edge``;
        ``add_edges_from`` must not collide with them."""
        g: rx.PyGraph = rx.PyGraph()
        a, b = g.add_node("a"), g.add_node("b")
        g.add_edge(a, b, {"u_of_edge": "stored", "v_of_edge": "stored"})

        nx_g = _pygraph_to_nx(g)

        assert nx_g[a][b]["u_of_edge"] == "stored"
        assert nx_g[a][b]["v_of_edge"] == "stored"


# ---------------------------------------------------------------------------
# Recursive composition correctness for non-contiguous cluster ids
# ---------------------------------------------------------------------------


class TestBisectCompositionCorrectness:
    """Exercises ``[parent_ids[i] for i in child_local_ids]`` under non-trivial
    input shapes — the existing tests use contiguous halves where the
    composition formula is degenerate."""

    def test_composes_non_contiguous_cluster_ids_across_two_depths(self, mocker):
        G = nx.path_graph(16)

        # Top-level split: even/odd indices (non-contiguous in the original frame).
        # Each child relabels its 8 selected nodes to local 0..7.
        even_ids = list(range(0, 16, 2))  # [0, 2, 4, 6, 8, 10, 12, 14]
        odd_ids = list(range(1, 16, 2))  # [1, 3, 5, 7, 9, 11, 13, 15]
        even_sub = nx.relabel_nodes(
            G.subgraph(even_ids).copy(), {n: i for i, n in enumerate(even_ids)}
        )
        odd_sub = nx.relabel_nodes(
            G.subgraph(odd_ids).copy(), {n: i for i, n in enumerate(odd_ids)}
        )

        # Recursive split of the even subgraph: pick parent-frame indices
        # [0, 4, 6] and [1, 2, 3, 5, 7] (non-contiguous, out-of-order in the
        # second cluster).
        even_lower_local = [0, 4, 6]
        even_upper_local = [1, 2, 3, 5, 7]
        even_lower = nx.relabel_nodes(
            even_sub.subgraph(even_lower_local).copy(),
            {n: i for i, n in enumerate(even_lower_local)},
        )
        even_upper = nx.relabel_nodes(
            even_sub.subgraph(even_upper_local).copy(),
            {n: i for i, n in enumerate(even_upper_local)},
        )

        call_count = {"n": 0}

        def fake_split(graph, config):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return ((even_sub, even_ids), (odd_sub, odd_ids))
            if call_count["n"] == 2:
                return (
                    (even_lower, even_lower_local),
                    (even_upper, even_upper_local),
                )
            return ()

        seen_top = {"v": False}
        seen_even = {"v": False}

        def predicate(subgraph, _):
            if not seen_top["v"]:
                seen_top["v"] = True
                return True
            if not seen_even["v"] and subgraph is even_sub:
                seen_even["v"] = True
                return True
            return False

        mocker.patch(
            f"{_split_graph.__module__}.{_split_graph.__name__}", side_effect=fake_split
        )

        config = GraphPartitioningConfig(minimum_n_clusters=2)
        initial = [(-G.number_of_nodes(), 0, G, list(G.nodes()))]
        result = _bisect_with_predicate(initial, predicate, config)

        leaf_ids = [r[3] for r in result]
        all_ids: set = set()
        for ids in leaf_ids:
            all_ids |= set(ids)
        # Coverage and disjointness across the original graph.
        assert all_ids == set(range(16))
        for i in range(len(leaf_ids)):
            for j in range(i + 1, len(leaf_ids)):
                assert set(leaf_ids[i]).isdisjoint(set(leaf_ids[j]))

        # The lower-even leaf carries parent_ids[0]=0, parent_ids[4]=8, parent_ids[6]=12.
        even_lower_leaf = next(ids for ids in leaf_ids if 0 in ids)
        assert sorted(even_lower_leaf) == [0, 8, 12]
        # The upper-even leaf carries parent_ids[1,2,3,5,7] = [2, 4, 6, 10, 14].
        even_upper_leaf = next(ids for ids in leaf_ids if set(ids) == {2, 4, 6, 10, 14})
        assert sorted(even_upper_leaf) == [2, 4, 6, 10, 14]


# ---------------------------------------------------------------------------
# draw_partitions
# ---------------------------------------------------------------------------


class TestDrawPartitions:
    def test_draw_partitions_calls_plt_show(self, mocker):
        mock_show = mocker.patch("matplotlib.pyplot.show")
        mocker.patch("networkx.draw")

        graph = nx.cycle_graph(6)
        reverse_index_maps = {
            ("A", 3): {0: 0, 1: 1, 2: 2},
            ("B", 3): {0: 3, 1: 4, 2: 5},
        }

        draw_partitions(graph, reverse_index_maps)

        mock_show.assert_called_once()

    def test_draw_partitions_raises_if_no_maps(self):
        graph = nx.cycle_graph(6)

        with pytest.raises(RuntimeError, match="no partitions to draw"):
            draw_partitions(graph, {})
