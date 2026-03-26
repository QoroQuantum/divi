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

_skip_no_pymetis = pytest.mark.skipif(
    sys.platform == "win32" and not PYMETIS_AVAILABLE,
    reason="pymetis not available (install via conda: conda install -c conda-forge pymetis)",
)

from divi.qprog.problems import GraphPartitioningConfig, _graph_partitioning_utils
from divi.qprog.problems._graph_partitioning_utils import (
    _apply_split_with_relabel,
    _bisect_with_predicate,
    _node_partition_graph,
    _split_graph,
    dominance_aggregation,
    draw_partitions,
    linear_aggregation,
)

# ---------------------------------------------------------------------------
# GraphPartitioningConfig
# ---------------------------------------------------------------------------


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

    def _assert_partitions_correct(
        self,
        original_graph: nx.Graph,
        clusters: Sequence[nx.Graph],
        expected_n_clusters: int | None = None,
    ):
        # Ensure expected number of clusters
        if expected_n_clusters is not None:
            assert len(clusters) == expected_n_clusters

        # Collect nodes and ensure cover graph
        all_nodes = set()
        for g in clusters:
            assert isinstance(g, nx.Graph)
            all_nodes |= set(g.nodes)
        assert set(original_graph.nodes) == all_nodes

        # Ensure all partitions are disjoint
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                assert set(clusters[i].nodes).isdisjoint(set(clusters[j].nodes))

    def test_apply_split_with_relabel_spectral(self, mocker):
        G = nx.cycle_graph(6)  # nice, symmetric, and simple

        mock_spectral_cls = mocker.patch(
            f"{_graph_partitioning_utils.__name__}.SpectralClustering"
        )
        instance = mock_spectral_cls.return_value
        # Fake prediction: 0,0,0,1,1,1
        instance.fit_predict.return_value = [0, 0, 0, 1, 1, 1]

        clusters = _apply_split_with_relabel(G, algorithm="spectral", n_clusters=2)
        self._assert_partitions_correct(G, clusters, expected_n_clusters=2)
        mock_spectral_cls.assert_called_once()
        instance.fit_predict.assert_called_once()

    @_skip_no_pymetis
    def test_apply_split_with_relabel_metis(self, mocker):
        G = nx.path_graph(6)

        mock_part_graph = mocker.patch("pymetis.part_graph")
        mock_part_graph.return_value = (None, [0, 0, 0, 1, 1, 1])

        clusters = _apply_split_with_relabel(G, algorithm="metis", n_clusters=2)
        mock_part_graph.assert_called_once()

        self._assert_partitions_correct(G, clusters, expected_n_clusters=2)

    def test_apply_split_with_relabel_invalid_algorithm_raises(self):
        with pytest.raises(RuntimeError, match="Relabeling only needed"):
            _apply_split_with_relabel(
                nx.path_graph(4), algorithm="kernighan_lin", n_clusters=2
            )

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
            G.subgraph([0, 1, 2]),
            G.subgraph([3, 4, 5]),
            G.subgraph([6, 7, 8]),
        )

        result = _split_graph(G, config)

        assert isinstance(result, Sequence)
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
        assert set(result[0].nodes) | set(result[1].nodes) == set(G.nodes)

    def test_predicate_receives_correct_arguments(self):
        G = nx.path_graph(4)
        initial = [(-G.number_of_nodes(), 0, G)]
        called_args = []

        def predicate(subgraph, others):
            called_args.append((subgraph, others))
            return False

        _bisect_with_predicate(initial, predicate, partitioning_config=None)

        # Ensure predicate was called at least once
        assert len(called_args) > 0

        # Check the types and contents
        for subgraph, others in called_args:
            assert isinstance(subgraph, nx.Graph)
            assert isinstance(others, list)
            assert all(isinstance(o, tuple) for o in others)

    def test_no_split_predicate(self):
        G = nx.path_graph(4)
        initial = [(-G.number_of_nodes(), 0, G)]

        # Predicate always False, so no splitting
        predicate = lambda _, __: False

        result = _bisect_with_predicate(initial, predicate, partitioning_config=None)

        assert len(result) == 1
        self._assert_partitions_correct(G, [result[0][-1]], 1)

    def test_single_split_predicate(self, mocker):
        G = nx.path_graph(4)
        initial = [(-G.number_of_nodes(), 0, G)]

        split_called = False

        def predicate(subgraph, others):
            # Split only the initial graph, not the new ones
            nonlocal split_called

            if not split_called:
                split_called = True
                return True
            return False

        # We'll mock _split_graph to split G into two halves:
        # We'll patch _split_graph inside the test method

        def fake_split(graph, config):
            # Split G into two graphs with nodes [0,1] and [2,3]
            sg1 = graph.subgraph([0, 1]).copy()
            sg2 = graph.subgraph([2, 3]).copy()
            return (sg1, sg2)

        mocker.patch(
            f"{_split_graph.__module__}.{_split_graph.__name__}", side_effect=fake_split
        )

        result = _bisect_with_predicate(initial, predicate, partitioning_config=None)

        # After one split, we expect 2 subgraphs
        assert len(result) == 2
        self._assert_partitions_correct(G, [rslt[-1] for rslt in result], 2)

    def test_multiple_splits_until_predicate_false(self, mocker):
        G = nx.path_graph(8)
        initial = [(-G.number_of_nodes(), 0, G)]

        # We'll split any graph with more than 2 nodes into halves
        def predicate(subgraph, others):
            return subgraph.number_of_nodes() > 2

        def fake_split(graph, config):
            nodes = list(graph.nodes)
            half = len(nodes) // 2
            sg1 = graph.subgraph(nodes[:half]).copy()
            sg2 = graph.subgraph(nodes[half:]).copy()
            return (sg1, sg2)

        mocker.patch(
            f"{_split_graph.__module__}.{_split_graph.__name__}", side_effect=fake_split
        )

        result = _bisect_with_predicate(initial, predicate, partitioning_config=None)

        # Now all subgraphs must have <= 2 nodes
        for _, _, sg in result:
            assert sg.number_of_nodes() <= 2

        self._assert_partitions_correct(G, [rslt[-1] for rslt in result], 4)

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
        assert partitions[0].number_of_nodes() == 40
        self._assert_partitions_correct(graph, partitions)

    def test_min_clusters_enforced(self, mocker):
        graph = nx.cycle_graph(6)
        mock_bisect = mocker.patch(
            f"{_bisect_with_predicate.__module__}.{_bisect_with_predicate.__name__}",
            return_value=[
                (0, 0, nx.Graph([(0, 1), (1, 2)])),
                (0, 1, nx.Graph([(3, 4), (4, 5)])),
            ],
        )

        config = GraphPartitioningConfig(minimum_n_clusters=2)
        result = _node_partition_graph(graph, config)

        self._assert_partitions_correct(graph, result, expected_n_clusters=2)
        assert mock_bisect.call_count >= 1

    @pytest.mark.filterwarnings("ignore::UserWarning")
    @pytest.mark.parametrize(
        "algorithm",
        [
            "spectral",
            pytest.param("metis", marks=_skip_no_pymetis),
            "kernighan_lin",
        ],
    )
    def test_partition_with_min_clusters(self, algorithm):
        G = nx.complete_graph(100)
        n_clusters = 6
        config = GraphPartitioningConfig(
            minimum_n_clusters=n_clusters, partitioning_algorithm=algorithm
        )
        partitions = _node_partition_graph(G, config)
        self._assert_partitions_correct(G, partitions, expected_n_clusters=n_clusters)

    @pytest.mark.parametrize(
        "algorithm",
        [
            "spectral",
            pytest.param("metis", marks=_skip_no_pymetis),
            "kernighan_lin",
        ],
    )
    def test_partition_with_max_nodes(self, algorithm):
        G = nx.complete_graph(100)
        max_nodes = 20
        config = GraphPartitioningConfig(
            max_n_nodes_per_cluster=max_nodes, partitioning_algorithm=algorithm
        )

        partitions = _node_partition_graph(G, config)
        self._assert_partitions_correct(G, partitions)

        for partition in partitions:
            assert partition.number_of_nodes() <= max_nodes

    @pytest.mark.parametrize(
        "algorithm",
        [
            "spectral",
            pytest.param("metis", marks=_skip_no_pymetis),
            "kernighan_lin",
        ],
    )
    def test_partition_with_both_constraints(self, algorithm):
        G = nx.complete_graph(100)

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
        for partition in partitions:
            assert partition.number_of_nodes() <= max_nodes

        self._assert_partitions_correct(G, partitions)


# ---------------------------------------------------------------------------
# Aggregation functions
# ---------------------------------------------------------------------------


class TestAggregationFunctions:
    def test_linear_aggregation(self):
        # Initial solution is all zeros
        main_solution = [0, 0, 0, 0, 0]
        # Subproblem solution identifies nodes 1 and 3 (in its own context)
        subproblem_solution = {1, 3}
        # Map subproblem nodes back to original main graph indices (1->2, 3->4)
        reverse_map = {0: 0, 1: 2, 2: 1, 3: 4, 4: 3}

        # Expected result: nodes at original indices 2 and 4 should be set to 1
        expected = [0, 0, 1, 0, 1]

        result = linear_aggregation(main_solution, subproblem_solution, reverse_map)
        assert result == expected

    def test_dominance_aggregation(self):
        """
        Tests dominance aggregation.
        Note: The current implementation of dominance_aggregation has a potential
        bug where the counts of 0s and 1s are re-calculated inside the loop.
        This makes the function's output dependent on the iteration order of the
        subproblem_solution set, which is non-deterministic. This test assumes
        a specific iteration order to pass but highlights this fragility.
        """
        # 0s are dominant (3 vs 2)
        main_solution = [0, 1, 0, 0, 1]
        # Subproblem solution for original nodes 0 and 3
        # We use a list to control iteration order for this test
        subproblem_solution = [0, 3]
        reverse_map = {i: i for i in range(5)}

        expected = [1, 1, 0, 0, 1]

        result = dominance_aggregation(main_solution, subproblem_solution, reverse_map)
        assert result == expected


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
