# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import re
import sys
import warnings
from collections.abc import Sequence

try:
    import pymetis
except ImportError:
    pymetis = None

PYMETIS_AVAILABLE = pymetis is not None

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pytest
import rustworkx as rx
from matplotlib.colors import to_rgba

_skip_no_pymetis = pytest.mark.skipif(
    sys.platform == "win32" and not PYMETIS_AVAILABLE,
    reason="pymetis not available (install via conda: conda install -c conda-forge pymetis)",
)

from divi.qprog import GraphProblemTypes
from divi.qprog.problems import (
    GraphPartitioningConfig,
    MaxCutProblem,
    _graph_partitioning_utils,
    draw_partitions,
)
from divi.qprog.problems._graph_hamiltonians import _node_ids
from divi.qprog.problems._graph_partitioning_utils import (
    _apply_split_with_relabel,
    _bisect_with_predicate,
    _canonical_edges,
    _merge_edgeless_clusters,
    _node_partition_graph,
    _relabeled_subgraph_with_ids,
    _split_graph,
)
from tests._helpers import exact_match


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


def _sorted_id_groups(clusters) -> list[list]:
    return sorted(sorted(ids) for _sub, ids in clusters)


def _assert_balanced_cycle_arcs(cycle: nx.Graph, clusters, arc_size: int):
    """Clusters partition ``cycle`` into connected arcs of ``arc_size`` nodes, cutting
    one edge per arc."""
    id_groups = [list(ids) for _sub, ids in clusters]
    assert sorted(i for ids in id_groups for i in ids) == list(cycle.nodes)
    assert [len(ids) for ids in id_groups] == [arc_size] * len(id_groups)
    assert all(nx.is_connected(cycle.subgraph(ids)) for ids in id_groups)
    internal = sum(cycle.subgraph(ids).number_of_edges() for ids in id_groups)
    assert cycle.number_of_edges() - internal == len(id_groups)


def _split_nodes_at(graph, cut: int):
    """Split ``graph`` into its first ``cut`` nodes and the rest, relabeled ``0..M-1``."""
    nodes = list(graph.nodes)
    halves = (nodes[:cut], nodes[cut:])
    return tuple(
        (
            nx.relabel_nodes(
                graph.subgraph(part).copy(), {n: i for i, n in enumerate(part)}
            ),
            part,
        )
        for part in halves
    )


def _split_halves(graph, _config):
    return _split_nodes_at(graph, len(graph) // 2)


def _patch_split_graph(mocker, side_effect):
    return mocker.patch(
        f"{_split_graph.__module__}.{_split_graph.__name__}", side_effect=side_effect
    )


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


@pytest.fixture
def _raise_qubit_ceiling(mocker):
    mocker.patch.object(_graph_partitioning_utils, "_MAXIMUM_AVAILABLE_QUBITS", 10_000)


class TestGraphPartitioningConfig:
    def test_invalid_no_constraints(self):
        with pytest.raises(
            ValueError, match="At least one constraint must be specified."
        ):
            GraphPartitioningConfig()

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            (
                {"minimum_n_clusters": 0},
                "'minimum_n_clusters' must be a positive integer.",
            ),
            (
                {"minimum_n_clusters": -5},
                "'minimum_n_clusters' must be a positive integer.",
            ),
            (
                {"max_n_nodes_per_cluster": 0},
                "'max_n_nodes_per_cluster' must be a positive number.",
            ),
            (
                {"max_n_nodes_per_cluster": -1},
                "'max_n_nodes_per_cluster' must be a positive number.",
            ),
        ],
        ids=[
            "min-clusters-zero",
            "min-clusters-negative",
            "max-nodes-zero",
            "max-nodes-negative",
        ],
    )
    def test_invalid_non_positive_constraint(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            GraphPartitioningConfig(**kwargs)

    def test_invalid_algorithm(self):
        with pytest.raises(ValueError, match="Unsupported partitioning algorithm:.*"):
            GraphPartitioningConfig(
                max_n_nodes_per_cluster=3, partitioning_algorithm="louvain"
            )

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
        with pytest.raises(
            RuntimeError,
            match=exact_match("Relabeling only needed for `spectral` and `metis`."),
        ):
            _apply_split_with_relabel(
                nx.path_graph(4), algorithm="kernighan_lin", n_clusters=2
            )

    @pytest.mark.parametrize(
        "pymetis_present",
        [
            pytest.param(
                True,
                marks=pytest.mark.skipif(not PYMETIS_AVAILABLE, reason="needs pymetis"),
                id="present",
            ),
            pytest.param(False, id="absent"),
        ],
    )
    def test_metis_windows_guard(self, pymetis_present, mocker):
        mocker.patch.object(_graph_partitioning_utils.sys, "platform", "win32")
        probe = mocker.patch.object(
            _graph_partitioning_utils,
            "optional_module",
            return_value=object() if pymetis_present else None,
        )
        G = nx.path_graph(6)

        if pymetis_present:
            mocker.patch("pymetis.part_graph", return_value=(None, [0, 0, 0, 1, 1, 1]))
            clusters = _apply_split_with_relabel(G, algorithm="metis", n_clusters=2)
            _assert_partitions_correct(G, clusters, expected_n_clusters=2)
        else:
            with pytest.raises(
                ImportError,
                match=exact_match(
                    "The 'metis' partitioning algorithm needs pymetis, which is not "
                    "installed on Windows by default; install it via conda: "
                    "conda install -c conda-forge pymetis. Otherwise use 'spectral' "
                    "or 'kernighan_lin' instead."
                ),
            ):
                _apply_split_with_relabel(G, algorithm="metis", n_clusters=2)

        probe.assert_called_once_with("pymetis")

    def test_spectral_split_of_cycle_is_contiguous_arcs(self):
        cycle = nx.cycle_graph(12)

        clusters = _apply_split_with_relabel(cycle, "spectral", 3)

        _assert_balanced_cycle_arcs(cycle, clusters, arc_size=4)

    def test_apply_split_with_relabel_drops_empty_clusters_with_warning(self, mocker):
        G = nx.cycle_graph(6)
        mock_cls = mocker.patch(
            f"{_graph_partitioning_utils.__name__}.SpectralClustering"
        )
        # All 6 nodes assigned to cluster 0; cluster 1 is empty.
        mock_cls.return_value.fit_predict.return_value = [0, 0, 0, 0, 0, 0]

        with pytest.warns(
            UserWarning,
            match=exact_match(
                "_apply_split_with_relabel: 'spectral' requested 2 clusters but "
                "produced 1 non-empty cluster(s); empty clusters were dropped from "
                "the result."
            ),
        ):
            result = _apply_split_with_relabel(G, algorithm="spectral", n_clusters=2)

        assert len(result) == 1
        sub, ids = result[0]
        assert _num_nodes(sub) == 6
        assert set(ids) == set(G.nodes())

    @pytest.mark.parametrize("algorithm", ["metis", "spectral"])
    @pytest.mark.parametrize(
        ("constraint", "expected_n_clusters"),
        [
            pytest.param({"minimum_n_clusters": 3}, 3, id="minimum"),
            pytest.param({"max_n_nodes_per_cluster": 4}, 2, id="bisection-default"),
        ],
    )
    def test_split_graph(self, algorithm, constraint, expected_n_clusters, mocker):
        """_split_graph routes to _apply_split_with_relabel with the correct args."""
        G = nx.path_graph(9)
        config = GraphPartitioningConfig(**constraint, partitioning_algorithm=algorithm)

        mock_split = mocker.patch(
            f"{_apply_split_with_relabel.__module__}.{_apply_split_with_relabel.__name__}"
        )

        result = _split_graph(G, config)

        assert result is mock_split.return_value
        mock_split.assert_called_once_with(G, algorithm, expected_n_clusters)

    def test_split_graph_kernighan_lin(self, mocker):
        G = nx.path_graph(6)
        config = GraphPartitioningConfig(
            minimum_n_clusters=10, partitioning_algorithm="kernighan_lin"
        )
        split = mocker.spy(nx.algorithms.community, "kernighan_lin_bisection")

        result = _split_graph(G, config)

        assert isinstance(result, Sequence)
        assert len(result) == 2
        assert set(result[0][1]) | set(result[1][1]) == set(G.nodes)
        split.assert_called_once()
        (kl_graph,), kwargs = split.call_args
        assert kwargs == {"seed": 0}
        assert nx.utils.graphs_equal(kl_graph, G)

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
        """KL on PyGraph input returns rustworkx subgraphs."""
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

    def test_multiple_splits_until_predicate_false(self, mocker):
        G = nx.path_graph(8)
        initial = [(-G.number_of_nodes(), 0, G, list(G.nodes()))]

        def predicate(subgraph, others):
            return subgraph.number_of_nodes() > 2

        _patch_split_graph(mocker, _split_halves)

        result = _bisect_with_predicate(initial, predicate, partitioning_config=None)

        for _, _, sg, _ in result:
            assert sg.number_of_nodes() <= 2

        _assert_partitions_correct(G, [(r[2], r[3]) for r in result], 4)

    def test_predicate_context_holds_processed_and_pending_entries(self, mocker):
        G = nx.path_graph(6)
        initial = [(-6, 0, G, list(G.nodes()))]
        _patch_split_graph(
            mocker,
            lambda graph, _config: _split_nodes_at(
                graph, 2 if len(graph) > 2 else len(graph) // 2
            ),
        )

        result = _bisect_with_predicate(
            initial, lambda _, others: len(others) < 2, partitioning_config=None
        )

        assert sorted(len(r[2]) for r in result) == [2, 2, 2]
        _assert_partitions_correct(G, [(r[2], r[3]) for r in result], 3)

    def test_bisect_with_no_partitions_returns_empty(self):
        assert _bisect_with_predicate([], lambda _, __: True, None) == []

    def test_child_counters_do_not_collide_with_initial_entries(self, mocker):
        big, small = nx.path_graph(4), nx.path_graph(2)
        initial = [(-4, 0, big, [0, 1, 2, 3]), (-2, 1, small, [4, 5])]
        _patch_split_graph(mocker, _split_halves)

        result = _bisect_with_predicate(
            initial, lambda graph, _: len(graph) == 4, partitioning_config=None
        )

        assert sorted(sorted(r[3]) for r in result) == [[0, 1], [2, 3], [4, 5]]

    @pytest.mark.parametrize("minimum_n_clusters", [5, 6])
    def test_node_partition_raises_if_min_clusters_too_high(self, minimum_n_clusters):
        G = nx.path_graph(5)
        config = GraphPartitioningConfig(minimum_n_clusters=minimum_n_clusters)

        with pytest.raises(
            ValueError,
            match=exact_match(
                "Number of requested clusters must be smaller than the size of "
                "the graph, since every cluster needs at least one edge."
            ),
        ):
            _node_partition_graph(G, config)

    def test_node_partition_of_pygraph_uses_node_indices(self):
        g: rx.PyGraph = rx.PyGraph()
        g.add_nodes_from(list("abcdef"))
        for i in range(6):
            g.add_edge(i, (i + 1) % 6, None)
        config = GraphPartitioningConfig(
            minimum_n_clusters=2, partitioning_algorithm="kernighan_lin"
        )

        clusters = _node_partition_graph(g, config)

        _assert_balanced_cycle_arcs(nx.cycle_graph(6), clusters, arc_size=3)

    def test_kernighan_lin_max_nodes_bisects_path_once(self):
        config = GraphPartitioningConfig(
            max_n_nodes_per_cluster=4, partitioning_algorithm="kernighan_lin"
        )

        clusters = _node_partition_graph(nx.path_graph(8), config)

        assert _sorted_id_groups(clusters) == [[0, 1, 2, 3], [4, 5, 6, 7]]

    def test_cluster_at_qubit_ceiling_does_not_warn(self, mocker):
        mocker.patch.object(_graph_partitioning_utils, "_MAXIMUM_AVAILABLE_QUBITS", 4)
        config = GraphPartitioningConfig(minimum_n_clusters=1)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            clusters = _node_partition_graph(nx.path_graph(4), config)

        assert len(clusters) == 1

    def test_partition_warns_for_oversized_clusters(self, mocker):
        mocker.patch.object(_graph_partitioning_utils, "_MAXIMUM_AVAILABLE_QUBITS", 20)

        graph = nx.complete_graph(40)
        config = GraphPartitioningConfig(minimum_n_clusters=1)

        with pytest.warns(
            UserWarning,
            match=exact_match(
                "At least one cluster has more nodes than what can be executed on "
                "the available backends: 20 qubits."
            ),
        ):
            partitions = _node_partition_graph(graph, config)

        assert len(partitions) == 1
        assert partitions[0][0].number_of_nodes() == 40
        _assert_partitions_correct(graph, partitions)

    def test_min_clusters_undershot_by_edgeless_merge_warns(self, mocker):
        graph = nx.cycle_graph(6)
        mocker.patch(
            f"{_bisect_with_predicate.__module__}.{_bisect_with_predicate.__name__}",
            return_value=[
                (0, 0, nx.Graph([(0, 1), (1, 2)]), [0, 1, 2]),
                (0, 1, nx.Graph([(0, 1)]), [3, 4]),
                (0, 2, nx.empty_graph(1), [5]),
            ],
        )

        config = GraphPartitioningConfig(minimum_n_clusters=3)

        with pytest.warns(
            UserWarning, match="below the requested 'minimum_n_clusters'"
        ):
            result = _node_partition_graph(graph, config)

        _assert_partitions_correct(graph, result, expected_n_clusters=2)

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
        """The minimum is best-effort: undershooting it must be announced."""
        _, complete_factory, _ = graph_factories
        G = complete_factory(100)
        n_clusters = 6
        config = GraphPartitioningConfig(
            minimum_n_clusters=n_clusters, partitioning_algorithm=algorithm
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            partitions = _node_partition_graph(G, config)

        _assert_partitions_correct(G, partitions)
        # Which nodes a bisection peels off a complete graph is eigensolver-dependent.
        if len(partitions) < n_clusters:
            assert any(
                "below the requested 'minimum_n_clusters'" in str(entry.message)
                for entry in caught
            )

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


@pytest.mark.usefixtures("_raise_qubit_ceiling")
def test_decompose_with_pygraph_input():
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


def _clusters_from_ids(graph, id_groups):
    """Build the partitioner's ``(relabeled_subgraph, cluster_ids)`` shape."""
    return [_relabeled_subgraph_with_ids(graph, ids) for ids in id_groups]


def _ids_containing(clusters, node):
    """The ``cluster_ids`` of the cluster holding ``node``."""
    return next(sorted(ids) for _sub, ids in clusters if node in ids)


def test_merge_edgeless_picks_most_connected_target():
    # Node 4 is edgeless on its own; it has 2 edges into [2, 3] and 1 into [0, 1].
    graph = nx.Graph([(0, 1), (2, 3), (4, 2), (4, 3), (4, 0)])
    clusters = _clusters_from_ids(graph, [[0, 1], [2, 3], [4]])
    config = GraphPartitioningConfig(minimum_n_clusters=1)

    merged = _merge_edgeless_clusters(graph, clusters, config)

    assert len(merged) == 2
    assert _ids_containing(merged, 4) == [2, 3, 4]
    _assert_partitions_correct(graph, merged)


def test_merge_edgeless_respects_max_nodes_over_connectivity():
    # Node 5 has 3 edges into [2, 3, 4] and 1 into [0, 1], but the better-connected
    # cluster is already at the cap, so the orphan must go to the one with room.
    graph = nx.Graph([(0, 1), (2, 3), (3, 4), (5, 2), (5, 3), (5, 4), (5, 0)])
    clusters = _clusters_from_ids(graph, [[0, 1], [2, 3, 4], [5]])
    config = GraphPartitioningConfig(max_n_nodes_per_cluster=3)

    merged = _merge_edgeless_clusters(graph, clusters, config)

    assert len(merged) == 2
    assert _ids_containing(merged, 5) == [0, 1, 5]
    for sub, _ids in merged:
        assert _num_nodes(sub) <= 3


def test_merge_edgeless_raises_when_no_cluster_has_room():
    graph = nx.Graph([(0, 1), (2, 3), (4, 0), (4, 2)])
    clusters = _clusters_from_ids(graph, [[0, 1], [2, 3], [4]])
    config = GraphPartitioningConfig(max_n_nodes_per_cluster=2)

    with pytest.raises(
        ValueError,
        match=exact_match(
            "Cannot merge edgeless cluster [4]: no cluster has room under "
            "'max_n_nodes_per_cluster'=2. The graph is too dense to split this "
            "small — raise 'max_n_nodes_per_cluster' or lower 'minimum_n_clusters'."
        ),
    ):
        _merge_edgeless_clusters(graph, clusters, config)


def test_merge_edgeless_isolated_vertex_goes_to_smallest_cluster():
    # Node 5 has no edges anywhere, so every target scores zero and the
    # tie-break by size decides.
    graph = nx.Graph([(0, 1), (1, 2), (3, 4)])
    graph.add_node(5)
    clusters = _clusters_from_ids(graph, [[0, 1, 2], [3, 4], [5]])
    config = GraphPartitioningConfig(minimum_n_clusters=1)

    merged = _merge_edgeless_clusters(graph, clusters, config)

    assert len(merged) == 2
    assert _ids_containing(merged, 5) == [3, 4, 5]


def test_merge_edgeless_warns_when_dropping_below_min_clusters():
    graph = nx.Graph([(0, 1), (2, 3), (4, 0)])
    clusters = _clusters_from_ids(graph, [[0, 1], [2, 3], [4]])
    config = GraphPartitioningConfig(minimum_n_clusters=3)

    with pytest.warns(
        UserWarning,
        match=exact_match(
            "Merging edgeless clusters left 2 cluster(s), below the requested "
            "'minimum_n_clusters'=3."
        ),
    ):
        merged = _merge_edgeless_clusters(graph, clusters, config)

    assert len(merged) == 2


def test_merge_edgeless_reaching_min_clusters_does_not_warn():
    graph = nx.Graph([(0, 1), (2, 3), (4, 0)])
    clusters = _clusters_from_ids(graph, [[0, 1], [2, 3], [4]])
    config = GraphPartitioningConfig(minimum_n_clusters=2)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        merged = _merge_edgeless_clusters(graph, clusters, config)

    assert len(merged) == 2


def test_merge_edgeless_raises_when_every_cluster_is_edgeless():
    graph = nx.Graph()
    graph.add_nodes_from(range(3))
    clusters = _clusters_from_ids(graph, [[0], [1], [2]])
    config = GraphPartitioningConfig(minimum_n_clusters=1)

    with pytest.raises(
        ValueError,
        match=exact_match(
            "Cannot repair the partition: every cluster is edgeless, so there is "
            "no cluster with internal edges to merge into. The graph has too few "
            "edges for the requested partitioning."
        ),
    ):
        _merge_edgeless_clusters(graph, clusters, config)


def test_merge_edgeless_is_noop_when_all_clusters_have_edges():
    graph = nx.Graph([(0, 1), (2, 3), (1, 2)])
    clusters = _clusters_from_ids(graph, [[0, 1], [2, 3]])
    config = GraphPartitioningConfig(max_n_nodes_per_cluster=2)

    merged = _merge_edgeless_clusters(graph, clusters, config)

    assert [ids for _sub, ids in merged] == [ids for _sub, ids in clusters]


def test_merge_edgeless_preserves_pygraph_type():
    graph = _make_pygraph_path(5)
    graph.add_node(5)  # isolated, so it forms an edgeless cluster
    clusters = _clusters_from_ids(graph, [[0, 1, 2], [3, 4], [5]])
    config = GraphPartitioningConfig(minimum_n_clusters=1)

    merged = _merge_edgeless_clusters(graph, clusters, config)

    assert len(merged) == 2
    for sub, _ids in merged:
        assert isinstance(sub, rx.PyGraph)
    _assert_partitions_correct(graph, merged)


@pytest.mark.usefixtures("_raise_qubit_ceiling")
@pytest.mark.parametrize("seed", [100, 101, 102, 103, 104])
def test_decompose_survives_dense_graphs(seed):
    """Regression: dense graphs used to shed singleton clusters, whose
    constant-only MaxCut Hamiltonian made ``decompose()`` raise."""
    graph = nx.gnp_random_graph(12, 0.85, seed=seed)
    problem = MaxCutProblem(
        graph, config=GraphPartitioningConfig(max_n_nodes_per_cluster=4)
    )

    sub_problems = problem.decompose()

    assert sub_problems
    for prog_id, sub in sub_problems.items():
        # Every sub-problem built, so no cluster was left without internal edges.
        assert sub.cost_hamiltonian.size > 0
        assert len(problem._reverse_index_maps[prog_id]) == len(sub.graph)


def _one_edge_graphs(*payloads) -> tuple[rx.PyGraph, nx.Graph]:
    """Two-node rx graph with one edge per payload, and its nx twin."""
    rx_graph: rx.PyGraph = rx.PyGraph()
    rx_graph.add_nodes_from(["a", "b"])
    nx_graph = nx.Graph()
    nx_graph.add_nodes_from([0, 1])
    for payload in payloads:
        rx_graph.add_edge(1, 0, payload)
        if isinstance(payload, dict):
            nx_graph.add_edges_from([(1, 0, payload)])
        elif payload is not None:
            nx_graph.add_edge(1, 0, weight=payload)
        else:
            nx_graph.add_edge(1, 0)
    return rx_graph, nx_graph


@pytest.mark.parametrize(
    "payloads, expected_weight",
    [
        pytest.param(({"weight": 1.5, "label": "ab"},), 1.5, id="dict"),
        pytest.param(({"label": "ab"},), None, id="dict-without-weight"),
        pytest.param(({"u_of_edge": "x", "weight": 2.0},), 2.0, id="reserved-key"),
        pytest.param((3.14,), 3.14, id="float"),
        pytest.param((5,), 5.0, id="int"),
        pytest.param((np.int64(4),), 4.0, id="numpy-int"),
        pytest.param((None,), None, id="none"),
        pytest.param((1.0, 2.5), 2.5, id="parallel-last-wins"),
    ],
)
@pytest.mark.parametrize("backend", ["rx", "nx"])
def test_canonical_edges_read_weights_alike_for_both_backends(
    backend, payloads, expected_weight
):
    graph = dict(zip(("rx", "nx"), _one_edge_graphs(*payloads)))[backend]

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ids, edges = _canonical_edges(graph)

    assert ids == [0, 1]
    assert edges == [(0, 1, expected_weight)]


@pytest.mark.parametrize("backend", ["rx", "nx"])
def test_canonical_edges_warn_on_non_numeric_weight(backend):
    graph = dict(zip(("rx", "nx"), _one_edge_graphs(("custom", "tuple"))))[backend]

    with pytest.warns(
        UserWarning,
        match=exact_match(
            "Edge weights of type(s) ['tuple'] are not real numbers; graph "
            "partitioning treats those edges as unweighted."
        ),
    ):
        _ids, edges = _canonical_edges(graph)

    assert edges == [(0, 1, None)]


def test_composes_non_contiguous_cluster_ids_across_two_depths(mocker):
    """Exercises ``[parent_ids[i] for i in child_local_ids]`` under non-trivial
    input shapes — the existing tests use contiguous halves where the
    composition formula is degenerate."""
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


def _weighted_gnp_edges(seed: int, n_nodes: int) -> list[tuple[int, int, float]]:
    return [
        (u, v, float(u + v))
        for u, v in rx.undirected_gnp_random_graph(n_nodes, 0.5, seed=seed).edge_list()
    ]


def _rx_generator_graph(seed: int) -> rx.PyGraph:
    return rx.undirected_gnp_random_graph(10, 0.5, seed=seed)


def _nx_edge_order_graph(seed: int) -> nx.Graph:
    graph = nx.Graph()
    graph.add_weighted_edges_from(_weighted_gnp_edges(seed, 10))
    graph.add_nodes_from(range(10))
    return graph


def _nx_string_graph(seed: int) -> nx.Graph:
    return nx.relabel_nodes(_nx_edge_order_graph(seed), lambda node: f"n{node}")


def _rx_removed_node_graph(seed: int) -> rx.PyGraph:
    graph: rx.PyGraph = rx.PyGraph()
    graph.add_nodes_from(range(11))
    graph.add_edges_from(_weighted_gnp_edges(seed, 11))
    graph.remove_node(4)
    return graph


def _edges_with_data(graph) -> list[tuple]:
    if isinstance(graph, rx.PyGraph):
        return list(graph.weighted_edge_list())
    return list(graph.edges(data=True))


def _assert_clusters_map_to_original_edges(graph, clusters):
    """Cluster ids cover every node once and each subgraph is the induced subgraph."""
    all_ids = [node for _sub, ids in clusters for node in ids]
    assert sorted(map(str, all_ids)) == sorted(map(str, _node_set(graph)))
    assert len(all_ids) == len(set(all_ids))
    for sub, ids in clusters:
        mapped = {frozenset((ids[u], ids[v])) for u, v, _data in _edges_with_data(sub)}
        induced = {
            frozenset((u, v))
            for u, v, _data in _edges_with_data(graph)
            if u in ids and v in ids
        }
        assert mapped == induced
        if isinstance(graph, nx.Graph):
            for u, v, data in _edges_with_data(sub):
                assert data == graph.edges[ids[u], ids[v]]


_GRAPH_KINDS = {
    "rx-generator": _rx_generator_graph,
    "nx-edge-order": _nx_edge_order_graph,
    "nx-string-labels": _nx_string_graph,
    "rx-removed-node": _rx_removed_node_graph,
}
_SEEDS = (2, 5, 11, 22, 26, 38)
_ALGORITHMS = [
    "spectral",
    pytest.param("metis", marks=_skip_no_pymetis),
    "kernighan_lin",
]
# A cap of 4 splits every kind and seed under every algorithm; rx-generator-38 at
# a cap of 3 drives the edgeless-cluster merge under spectral.
_GRAPH_CASES = [
    pytest.param(make_graph, seed, 4, id=f"{kind}-{seed}")
    for kind, make_graph in _GRAPH_KINDS.items()
    for seed in _SEEDS
] + [pytest.param(_rx_generator_graph, 38, 3, id="rx-generator-38-cap3")]


@pytest.mark.usefixtures("_raise_qubit_ceiling")
@pytest.mark.parametrize(("make_graph", "seed", "cap"), _GRAPH_CASES)
@pytest.mark.parametrize("algorithm", _ALGORITHMS)
def test_partition_subgraph_edges_map_to_original_edges(
    algorithm, make_graph, seed, cap
):
    graph = make_graph(seed)
    config = GraphPartitioningConfig(
        max_n_nodes_per_cluster=cap, partitioning_algorithm=algorithm
    )

    clusters = _node_partition_graph(graph, config)

    _assert_clusters_map_to_original_edges(graph, clusters)


def _twin(graph) -> tuple[GraphProblemTypes, list]:
    """The same graph in the other library, and the twin's node ids by position.

    A rustworkx graph becomes a networkx graph labelled by its node indices; a
    networkx graph becomes a rustworkx graph with one node per label, in
    iteration order. Edge weights carry over, and ``None`` stays unweighted.
    """
    if isinstance(graph, rx.PyGraph):
        twin = nx.Graph()
        twin.add_nodes_from(graph.node_indexes())
        for u, v, weight in graph.weighted_edge_list():
            twin.add_edge(u, v, **({} if weight is None else {"weight": weight}))
        return twin, list(twin.nodes())
    labels = list(graph.nodes())
    position = {label: i for i, label in enumerate(labels)}
    rx_twin: rx.PyGraph = rx.PyGraph()
    rx_twin.add_nodes_from(labels)
    rx_twin.add_edges_from(
        [(position[u], position[v], w) for u, v, w in graph.edges(data="weight")]
    )
    return rx_twin, list(rx_twin.node_indexes())


def _partition_outcome(graph, config, node_ids) -> list[list] | str:
    """Cluster ids in ``graph``'s positional frame, or the raised error message."""
    position = {node: i for i, node in enumerate(node_ids)}
    try:
        clusters = _node_partition_graph(graph, config)
    except ValueError as exc:
        # The message names the offending nodes by their own library's ids.
        return re.sub(r"\[.*?\]", "[...]", str(exc))
    return [[position[node] for node in ids] for _sub, ids in clusters]


@pytest.mark.usefixtures("_raise_qubit_ceiling")
@pytest.mark.parametrize("cap", [3, 4])
@pytest.mark.parametrize("seed", _SEEDS)
@pytest.mark.parametrize(
    "make_graph", list(_GRAPH_KINDS.values()), ids=list(_GRAPH_KINDS)
)
@pytest.mark.parametrize("algorithm", _ALGORITHMS)
def test_partition_matches_other_library_twin(algorithm, make_graph, seed, cap):
    """The same graph partitions identically, or fails identically, in both libraries."""
    graph = make_graph(seed)
    twin, twin_ids = _twin(graph)
    config = GraphPartitioningConfig(
        max_n_nodes_per_cluster=cap, partitioning_algorithm=algorithm
    )

    assert _partition_outcome(graph, config, _node_ids(graph)) == _partition_outcome(
        twin, config, twin_ids
    )


def _weighted_twin_pair(seed: int, *, weighted: bool, shuffled: bool):
    graph = rx.undirected_gnp_random_graph(12, 0.45, seed=seed)
    if weighted:
        for index, (u, v) in zip(graph.edge_indices(), graph.edge_list()):
            graph.update_edge_by_index(index, float(1 + (u * 7 + v * 3) % 5))
    graph.remove_node(4)
    twin, _ids = _twin(graph)
    if shuffled:
        reordered = nx.Graph()
        reordered.add_nodes_from(twin.nodes())
        reordered.add_edges_from(reversed(list(twin.edges(data=True))))
        twin = reordered
    return graph, twin


@pytest.mark.usefixtures("_raise_qubit_ceiling")
@pytest.mark.parametrize(
    "shuffled", [False, True], ids=["same-order", "reversed-edges"]
)
@pytest.mark.parametrize("weighted", [False, True], ids=["unweighted", "weighted"])
@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("algorithm", _ALGORITHMS)
def test_rx_and_nx_twins_give_identical_partitions(algorithm, seed, weighted, shuffled):
    """Same ids per cluster, whatever order the networkx edges were inserted in."""
    graph, twin = _weighted_twin_pair(seed, weighted=weighted, shuffled=shuffled)
    config = GraphPartitioningConfig(
        max_n_nodes_per_cluster=5, partitioning_algorithm=algorithm
    )

    rx_clusters = [ids for _sub, ids in _node_partition_graph(graph, config)]
    nx_clusters = [ids for _sub, ids in _node_partition_graph(twin, config)]

    assert rx_clusters == nx_clusters


@pytest.mark.parametrize("make_graph", [_make_pygraph_path, nx.path_graph])
def test_kernighan_lin_split_ids_follow_local_indices(make_graph, mocker):
    graph = make_graph(6)
    mocker.patch.object(
        nx.algorithms.community,
        "kernighan_lin_bisection",
        return_value=([2, 0, 1], [5, 3, 4]),
    )
    config = GraphPartitioningConfig(
        minimum_n_clusters=2, partitioning_algorithm="kernighan_lin"
    )

    clusters = _split_graph(graph, config)

    _assert_clusters_map_to_original_edges(graph, clusters)


@pytest.mark.parametrize(
    "cluster",
    [pytest.param([5, 1, 4], id="unsorted"), pytest.param([1, 4, 5], id="sorted")],
)
def test_relabeled_pygraph_subgraph_ids_follow_local_indices(cluster):
    graph: rx.PyGraph = rx.PyGraph()
    graph.add_nodes_from(list("abcdef"))
    graph.add_edges_from([(0, 1, 1.0), (4, 5, 2.0), (1, 4, 3.0)])

    sub, ids = _relabeled_subgraph_with_ids(graph, cluster)

    assert [graph[node] for node in ids] == sub.nodes()
    _assert_clusters_map_to_original_edges(
        graph.subgraph(sorted(cluster)), [(sub, list(range(len(cluster))))]
    )


@pytest.mark.usefixtures("_raise_qubit_ceiling")
def test_decompose_string_labelled_graph_extends_by_label():
    labels = "abcdef"
    graph = nx.relabel_nodes(nx.cycle_graph(6), dict(enumerate(labels)))
    problem = MaxCutProblem(
        graph, config=GraphPartitioningConfig(max_n_nodes_per_cluster=3)
    )

    sub_problems = problem.decompose()

    solution = [0] * problem.initial_solution_size()
    for prog_id, sub in sub_problems.items():
        reverse = problem._reverse_index_maps[prog_id]
        assert all(graph.has_edge(reverse[u], reverse[v]) for u, v in sub.graph.edges())
        solution = problem.extend_solution(solution, prog_id, [0])
    selected = {problem._reverse_index_maps[prog_id][0] for prog_id in sub_problems}
    assert solution == [int(label in selected) for label in labels]


def test_spectral_decomposition_is_reproducible():
    config = GraphPartitioningConfig(
        max_n_nodes_per_cluster=6, partitioning_algorithm="spectral"
    )

    maps = []
    for _ in range(2):
        problem = MaxCutProblem(nx.cycle_graph(12), config=config)
        problem.decompose()
        maps.append(problem._reverse_index_maps)

    assert maps[0] == maps[1]


_THREE_PARTITIONS = {
    ("A", 2): {0: 0, 1: 1},
    ("B", 2): {0: 2, 1: 3},
    ("C", 2): {0: 4, 1: 5},
}


class TestDrawPartitions:
    @pytest.fixture
    def mock_show(self, mocker):
        show = mocker.patch("matplotlib.pyplot.show")
        yield show
        plt.close("all")

    @pytest.fixture
    def draw_spy(self, mocker):
        return mocker.spy(nx, "draw")

    def test_draw_partitions_calls_plt_show(self, mock_show, draw_spy):
        graph = nx.cycle_graph(6)
        reverse_index_maps = {
            ("A", 3): {0: 0, 1: 1, 2: 2},
            ("B", 3): {0: 3, 1: 4, 2: 5},
        }

        draw_partitions(graph, reverse_index_maps)

        mock_show.assert_called_once()
        draw_spy.assert_called_once()

    def test_each_partition_gets_a_distinct_colour(self, mock_show, draw_spy):
        draw_partitions(nx.cycle_graph(6), _THREE_PARTITIONS)

        node_colors = draw_spy.call_args.kwargs["node_color"]
        assert len({tuple(c) for c in node_colors}) == 3
        for first, second in ((0, 1), (2, 3), (4, 5)):
            np.testing.assert_array_equal(node_colors[first], node_colors[second])

    def test_node_labels_are_drawn(self, mock_show, draw_spy):
        draw_partitions(nx.cycle_graph(6), _THREE_PARTITIONS)

        assert sorted(t.get_text() for t in plt.gca().texts) == list("012345")

    def test_explicit_positions_are_used(self, mock_show, draw_spy):
        graph = nx.cycle_graph(6)
        pos = {node: (float(node), 0.0) for node in graph}

        draw_partitions(graph, _THREE_PARTITIONS, pos=pos)

        assert draw_spy.call_args.args[1] is pos

    def test_default_layout_is_seeded_spring_layout(self, mock_show, draw_spy):
        graph = nx.cycle_graph(6)
        expected = nx.spring_layout(graph, seed=42)

        draw_partitions(graph, _THREE_PARTITIONS)

        pos = draw_spy.call_args.args[1]
        assert pos.keys() == expected.keys()
        for node, xy in expected.items():
            np.testing.assert_allclose(pos[node], xy)

    def test_figure_has_title_legend_and_no_axis(self, mock_show, draw_spy):
        draw_partitions(nx.cycle_graph(6), _THREE_PARTITIONS)

        ax = plt.gca()
        assert ax.get_title() == "Graph Partitions Visualization"
        assert not ax.axison
        legend = ax.get_legend()
        assert [t.get_text() for t in legend.get_texts()] == [
            "Partition A",
            "Partition B",
            "Partition C",
        ]
        node_colors = draw_spy.call_args.kwargs["node_color"]
        for handle, node in zip(legend.legend_handles, (0, 2, 4)):
            np.testing.assert_array_equal(
                handle.get_markerfacecolor(), node_colors[node]
            )

    def test_unassigned_nodes_get_their_own_colour_and_legend_entry(
        self, mock_show, draw_spy
    ):
        draw_partitions(nx.cycle_graph(7), _THREE_PARTITIONS)

        node_colors = draw_spy.call_args.kwargs["node_color"]
        assert node_colors[6] == to_rgba("dimgray")
        assert all(tuple(c) != node_colors[6] for c in node_colors[:6])
        legend = plt.gca().get_legend()
        assert legend.get_texts()[-1].get_text() == "Unassigned"
        assert legend.legend_handles[-1].get_markerfacecolor() == node_colors[6]

    def test_rx_graph_is_drawn_by_node_index(self, mock_show, draw_spy):
        graph = rx.generators.cycle_graph(6)
        for index in graph.node_indexes():
            graph[index] = f"payload{index}"

        draw_partitions(graph, _THREE_PARTITIONS)
        rx_colours = draw_spy.call_args.kwargs["node_color"]
        rx_labels = sorted(t.get_text() for t in plt.gca().texts)
        draw_partitions(nx.cycle_graph(6), _THREE_PARTITIONS)

        assert rx_labels == list("012345")
        np.testing.assert_array_equal(
            rx_colours, draw_spy.call_args.kwargs["node_color"]
        )

    def test_draw_partitions_raises_if_no_maps(self):
        graph = nx.cycle_graph(6)

        with pytest.raises(
            RuntimeError,
            match=exact_match(
                "There are no partitions to draw. Did you call decompose()?"
            ),
        ):
            draw_partitions(graph, {})
