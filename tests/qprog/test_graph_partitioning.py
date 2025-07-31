# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence

import networkx as nx
import pytest
from qprog_contracts import verify_basic_program_batch_behaviour

from divi.parallel_simulator import ParallelSimulator
from divi.qprog import (
    GraphPartitioningQAOA,
    GraphProblem,
    Optimizers,
    PartitioningConfig,
)
from divi.qprog._graph_partitioning import (
    _apply_split_with_relabel,
    _bisect_with_predicate,
    _node_partition_graph,
    _split_graph,
)

problem_args = {
    "graph": nx.erdos_renyi_graph(15, 0.2, seed=1997),
    "graph_problem": GraphProblem.MAXCUT,
    "n_layers": 1,
    "partitioning_config": PartitioningConfig(
        minimum_n_clusters=2, partitioning_algorithm="spectral"
    ),
    "optimizer": Optimizers.NELDER_MEAD,
    "max_iterations": 10,
    "backend": ParallelSimulator(shots=5000),
}


class TestPartitioningConfig:
    def test_valid_max_nodes_only(self):
        config = PartitioningConfig(max_n_nodes_per_cluster=10)
        assert config.max_n_nodes_per_cluster == 10
        assert config.minimum_n_clusters is None

    def test_valid_min_clusters_only(self):
        config = PartitioningConfig(minimum_n_clusters=2)
        assert config.minimum_n_clusters == 2
        assert config.max_n_nodes_per_cluster is None

    def test_valid_both_constraints(self):
        config = PartitioningConfig(
            max_n_nodes_per_cluster=5,
            minimum_n_clusters=3,
            partitioning_algorithm="metis",
        )
        assert config.max_n_nodes_per_cluster == 5
        assert config.minimum_n_clusters == 3
        assert config.partitioning_algorithm == "metis"

    def test_default_algorithm(self):
        # Introspect the default value from the dataclass field
        field_info = PartitioningConfig.__dataclass_fields__["partitioning_algorithm"]
        default_value = field_info.default

        config = PartitioningConfig(max_n_nodes_per_cluster=1)
        assert config.partitioning_algorithm == default_value

    def test_invalid_no_constraints(self):
        with pytest.raises(
            ValueError, match="At least one constraint must be specified."
        ):
            PartitioningConfig()

    def test_invalid_min_clusters_zero(self):
        with pytest.raises(
            ValueError, match="'minimum_n_clusters' must be a positive integer."
        ):
            PartitioningConfig(minimum_n_clusters=0)

    def test_invalid_max_nodes_zero(self):
        with pytest.raises(
            ValueError, match="'max_n_nodes_per_cluster' must be a positive number."
        ):
            PartitioningConfig(max_n_nodes_per_cluster=0)

    def test_invalid_algorithm(self):
        with pytest.raises(ValueError, match="Unsupported partitioning algorithm:.*"):
            PartitioningConfig(
                max_n_nodes_per_cluster=3, partitioning_algorithm="louvain"
            )

    def test_negative_values(self):
        with pytest.raises(ValueError):
            PartitioningConfig(minimum_n_clusters=-5)

        with pytest.raises(ValueError):
            PartitioningConfig(max_n_nodes_per_cluster=-1)

    def test_valid_algorithm_variants(self):
        for algo in ["spectral", "metis", "kernighan_lin"]:
            config = PartitioningConfig(
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

        # Fake prediction: 0,0,0,1,1,1
        mock_spectral_cls = mocker.patch(
            "divi.qprog._graph_partitioning.SpectralClustering"
        )
        instance = mock_spectral_cls.return_value
        instance.fit_predict.return_value = [0, 0, 0, 1, 1, 1]

        clusters = _apply_split_with_relabel(G, algorithm="spectral", n_clusters=2)
        self._assert_partitions_correct(G, clusters, expected_n_clusters=2)
        mock_spectral_cls.assert_called_once()
        instance.fit_predict.assert_called_once()

    def test_apply_split_with_relabel_metis(self, mocker):
        G = nx.path_graph(6)

        mock_part_graph = mocker.patch("divi.qprog._graph_partitioning.part_graph")
        mock_part_graph.return_value = (None, [0, 0, 0, 1, 1, 1])

        clusters = _apply_split_with_relabel(G, algorithm="metis", n_clusters=2)

        self._assert_partitions_correct(G, clusters, expected_n_clusters=2)
        mock_part_graph.assert_called_once()

    def test_apply_split_with_relabel_invalid_algorithm_raises(self):
        with pytest.raises(RuntimeError, match="Relabeling only needed"):
            _apply_split_with_relabel(
                nx.path_graph(4), algorithm="kernighan_lin", n_clusters=2
            )

    @pytest.mark.parametrize("algorithm", ["metis", "spectral"])
    def test_split_graph(self, algorithm, mocker):
        G = nx.path_graph(9)
        config = PartitioningConfig(
            minimum_n_clusters=3, partitioning_algorithm=algorithm
        )

        mock_split = mocker.patch(
            "divi.qprog._graph_partitioning._apply_split_with_relabel"
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
        config = PartitioningConfig(
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
            "divi.qprog._graph_partitioning._split_graph", side_effect=fake_split
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
            "divi.qprog._graph_partitioning._split_graph", side_effect=fake_split
        )

        result = _bisect_with_predicate(initial, predicate, partitioning_config=None)

        # Now all subgraphs must have <= 2 nodes
        for _, _, sg in result:
            assert sg.number_of_nodes() <= 2

        self._assert_partitions_correct(G, [rslt[-1] for rslt in result], 4)

    def test_node_partition_raises_if_min_clusters_too_high(self):
        G = nx.path_graph(5)
        config = PartitioningConfig(minimum_n_clusters=6)

        with pytest.raises(ValueError, match="Number of requested clusters"):
            _node_partition_graph(G, config)

    def test_partition_warns_for_oversized_clusters(self, mocker):
        # Mock the maximum available qubits to a smaller number for the test
        mocker.patch("divi.qprog._graph_partitioning._MAXIMUM_AVAILABLE_QUBITS", 20)

        graph = nx.complete_graph(40)
        config = PartitioningConfig(minimum_n_clusters=1)  # No splitting

        with pytest.warns(UserWarning, match="At least one cluster has more nodes"):
            partitions = _node_partition_graph(graph, config)

        # Even with the warning, the result should be a single partition
        assert len(partitions) == 1
        assert partitions[0].number_of_nodes() == 40
        self._assert_partitions_correct(graph, partitions)

    def test_min_clusters_enforced(self, mocker):
        graph = nx.cycle_graph(6)
        mock_bisect = mocker.patch(
            "divi.qprog._graph_partitioning._bisect_with_predicate",
            return_value=[
                (0, 0, nx.Graph([(0, 1), (1, 2)])),
                (0, 1, nx.Graph([(3, 4), (4, 5)])),
            ],
        )

        config = PartitioningConfig(minimum_n_clusters=2)
        result = _node_partition_graph(graph, config)

        self._assert_partitions_correct(graph, result, expected_n_clusters=2)
        assert mock_bisect.call_count >= 1

    @pytest.mark.filterwarnings("ignore::UserWarning")
    @pytest.mark.parametrize("algorithm", ["spectral", "metis", "kernighan_lin"])
    def test_partition_with_min_clusters(self, algorithm):
        G = nx.complete_graph(100)
        n_clusters = 6
        config = PartitioningConfig(
            minimum_n_clusters=n_clusters, partitioning_algorithm=algorithm
        )
        partitions = _node_partition_graph(G, config)
        self._assert_partitions_correct(G, partitions, expected_n_clusters=n_clusters)

    @pytest.mark.parametrize("algorithm", ["spectral", "metis", "kernighan_lin"])
    def test_partition_with_max_nodes(self, algorithm):
        G = nx.complete_graph(100)
        max_nodes = 20
        config = PartitioningConfig(
            max_n_nodes_per_cluster=max_nodes, partitioning_algorithm=algorithm
        )

        partitions = _node_partition_graph(G, config)
        self._assert_partitions_correct(G, partitions)

        for partition in partitions:
            assert partition.number_of_nodes() <= max_nodes

    @pytest.mark.parametrize("algorithm", ["spectral", "metis", "kernighan_lin"])
    def test_partition_with_both_constraints(self, algorithm):
        G = nx.complete_graph(100)

        min_clusters = 3
        max_nodes = 15
        config = PartitioningConfig(
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


@pytest.fixture
def node_partitioning_qaoa():
    return GraphPartitioningQAOA(**problem_args)


def test_verify_basic_behaviour(mocker, node_partitioning_qaoa):
    verify_basic_program_batch_behaviour(mocker, node_partitioning_qaoa)

    mock_program = mocker.MagicMock()
    mock_program.losses = [{0: -1.0}]

    node_partitioning_qaoa.programs = {"dummy": mock_program}

    with pytest.raises(RuntimeError, match="Not all final probabilities"):
        node_partitioning_qaoa.aggregate_results()


def test_correct_initialization(node_partitioning_qaoa):
    assert node_partitioning_qaoa.main_graph == problem_args["graph"]
    assert node_partitioning_qaoa.is_edge_problem == False
    assert (
        node_partitioning_qaoa.partitioning_config
        == problem_args["partitioning_config"]
    )


def test_correct_number_of_programs_created(mocker, node_partitioning_qaoa):
    mocker.patch("divi.qprog.QAOA")

    node_partitioning_qaoa.create_programs()

    assert (
        len(node_partitioning_qaoa.programs)
        >= problem_args["partitioning_config"].minimum_n_clusters
    )

    # Assert common values propagated to all programs
    for program in node_partitioning_qaoa.programs.values():
        assert program.optimizer == Optimizers.NELDER_MEAD
        assert program.max_iterations == 10
        assert isinstance(program.backend, ParallelSimulator)
        assert program.backend.shots == 5000

    # Need to clean up at the end of the test
    node_partitioning_qaoa._progress_bar.stop()


def test_results_aggregated_correctly(node_partitioning_qaoa):
    node_partitioning_qaoa.create_programs()

    prog_1_key, prog_2_key = tuple(node_partitioning_qaoa.programs.keys())

    mock_program_1_nodes = node_partitioning_qaoa.programs[
        prog_1_key
    ].problem.number_of_nodes()
    node_partitioning_qaoa.programs[prog_1_key].losses = [{0: -1.0}]
    node_partitioning_qaoa.programs[prog_1_key].probs = {
        "0_0": {"0" * mock_program_1_nodes: 0.9, "1" * mock_program_1_nodes: 0.1}
    }

    mock_program_2_nodes = node_partitioning_qaoa.programs[
        prog_2_key
    ].problem.number_of_nodes()
    node_partitioning_qaoa.programs[prog_2_key].losses = [{0: -2.0, 1: -3.0}]
    node_partitioning_qaoa.programs[prog_2_key].probs = {
        "0_0": {"0" * mock_program_2_nodes: 0.9, "1" * mock_program_2_nodes: 0.1},
        "1_0": {"0" * mock_program_2_nodes: 0.2, "1" * mock_program_2_nodes: 0.8},
    }

    solution = node_partitioning_qaoa.aggregate_results()

    assert len(solution) == mock_program_2_nodes

    # Need to clean up at the end of the test
    node_partitioning_qaoa._progress_bar.stop()
