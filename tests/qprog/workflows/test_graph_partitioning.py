# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import sys
import warnings
from collections.abc import Sequence

try:
    import pymetis
except ImportError:
    pymetis = None

PYMETIS_AVAILABLE = pymetis is not None
import networkx as nx
import pytest

from divi.qprog import ScipyMethod, ScipyOptimizer
from divi.qprog.problems import (
    GraphPartitioningConfig,
    MaxCliqueProblem,
    MaxCutProblem,
    MaxIndependentSetProblem,
    MinVertexCoverProblem,
    _graph_partitioning_utils,
)
from divi.qprog.problems._graph_partitioning_utils import (
    _apply_split_with_relabel,
    _bisect_with_predicate,
    _node_partition_graph,
    _split_graph,
    dominance_aggregation,
    draw_partitions,
    linear_aggregation,
)
from divi.qprog.workflows import PartitioningProgramEnsemble
from tests.qprog.qprog_contracts import verify_basic_program_ensemble_behaviour

_GRAPH = nx.erdos_renyi_graph(15, 0.2, seed=1997)
_PARTITIONING_CONFIG = GraphPartitioningConfig(
    minimum_n_clusters=2, partitioning_algorithm="spectral"
)
_PROBLEM = MaxCutProblem(_GRAPH, config=_PARTITIONING_CONFIG)
_ENSEMBLE_ARGS = {
    "problem": _PROBLEM,
    "n_layers": 1,
    "optimizer": ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
    "max_iterations": 10,
}


@pytest.fixture
def ensemble_args(dummy_simulator):
    return {**_ENSEMBLE_ARGS, "backend": dummy_simulator}


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

    @pytest.mark.skipif(
        sys.platform == "win32" and not PYMETIS_AVAILABLE,
        reason="pymetis not available (install via conda: conda install -c conda-forge pymetis)",
    )
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
            pytest.param(
                "metis",
                marks=pytest.mark.skipif(
                    sys.platform == "win32" and not PYMETIS_AVAILABLE,
                    reason="pymetis not available (install via conda: conda install -c conda-forge pymetis)",
                ),
            ),
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
            pytest.param(
                "metis",
                marks=pytest.mark.skipif(
                    sys.platform == "win32" and not PYMETIS_AVAILABLE,
                    reason="pymetis not available (install via conda: conda install -c conda-forge pymetis)",
                ),
            ),
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
            pytest.param(
                "metis",
                marks=pytest.mark.skipif(
                    sys.platform == "win32" and not PYMETIS_AVAILABLE,
                    reason="pymetis not available (install via conda: conda install -c conda-forge pymetis)",
                ),
            ),
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


class TestEvaluateSolution:
    """Tests for problem.evaluate_global_solution on graph problems.

    Uses small graphs with analytically-known MaxCut energies so we can
    verify the Hamiltonian evaluation is correct.
    """

    @staticmethod
    def _make_problem(graph, problem_cls=MaxCutProblem):
        """Create a MaxCutProblem with partitioning config for unit-testing."""
        return problem_cls(graph, config=GraphPartitioningConfig(minimum_n_clusters=1))

    def test_perfect_maxcut_on_4_cycle(self):
        """A bipartite 4-cycle has a perfect cut of 4 edges."""
        graph = nx.cycle_graph(4)
        problem = self._make_problem(graph)
        # [1,0,1,0] cuts all 4 edges
        energy = problem.evaluate_global_solution([1, 0, 1, 0])
        assert energy == pytest.approx(-4.0)

    def test_no_cut_all_zeros(self):
        """All-zero assignment cuts nothing."""
        graph = nx.cycle_graph(4)
        problem = self._make_problem(graph)
        energy = problem.evaluate_global_solution([0, 0, 0, 0])
        assert energy == pytest.approx(0.0)

    def test_no_cut_all_ones(self):
        """All-one assignment cuts nothing."""
        graph = nx.cycle_graph(4)
        problem = self._make_problem(graph)
        energy = problem.evaluate_global_solution([1, 1, 1, 1])
        assert energy == pytest.approx(0.0)

    def test_partial_cut_on_4_cycle(self):
        """[1,1,0,0] on a 4-cycle cuts edges (0,3) and (1,2) = 2 cut edges."""
        graph = nx.cycle_graph(4)
        problem = self._make_problem(graph)
        energy = problem.evaluate_global_solution([1, 1, 0, 0])
        assert energy == pytest.approx(-2.0)

    def test_weighted_graph(self):
        """Weighted edges: MaxCut Hamiltonian counts cut edges."""
        graph = nx.Graph()
        graph.add_edge(0, 1, weight=3.0)
        graph.add_edge(1, 2, weight=5.0)
        problem = self._make_problem(graph)
        # [1,0,1] cuts both edges
        energy = problem.evaluate_global_solution([1, 0, 1])
        assert energy == pytest.approx(-2.0)

    def test_triangle_graph(self):
        """A triangle (K3): best cut has 2 edges, e.g. [1,0,0]."""
        graph = nx.complete_graph(3)
        problem = self._make_problem(graph)
        # [1,0,0] cuts edges (0,1) and (0,2) = 2 cut edges
        energy = problem.evaluate_global_solution([1, 0, 0])
        assert energy == pytest.approx(-2.0)

    def test_lower_energy_means_more_cuts(self):
        """Verify that more cuts produce lower (more negative) energy."""
        graph = nx.cycle_graph(4)
        problem = self._make_problem(graph)
        no_cut = problem.evaluate_global_solution([0, 0, 0, 0])
        partial_cut = problem.evaluate_global_solution([1, 1, 0, 0])
        perfect_cut = problem.evaluate_global_solution([1, 0, 1, 0])
        assert no_cut > partial_cut > perfect_cut


@pytest.fixture
def graph_ensemble(ensemble_args):
    return PartitioningProgramEnsemble(**ensemble_args)


class TestGraphPartitioningEnsemble:
    def test_verify_basic_behaviour(self, mocker, graph_ensemble):
        verify_basic_program_ensemble_behaviour(mocker, graph_ensemble)

    def test_correct_number_of_programs_created(self, mocker, graph_ensemble):
        mocker.patch("divi.qprog.QAOA")

        graph_ensemble.create_programs()

        assert len(graph_ensemble.programs) >= _PARTITIONING_CONFIG.minimum_n_clusters

    def test_results_aggregated_correctly(self, graph_ensemble):
        # Create programs and partitions
        graph_ensemble.create_programs()

        # Identify two sub-programs to mock
        prog_keys = list(graph_ensemble.programs.keys())
        prog_1_key, prog_2_key = prog_keys[0], prog_keys[1]

        # Get the number of nodes (qubits) in each subgraph
        n_qubits_1 = graph_ensemble.programs[prog_1_key].n_qubits
        n_qubits_2 = graph_ensemble.programs[prog_2_key].n_qubits

        # Mock _best_probs: program 1 has all-zeros (empty solution), program 2 has all-ones
        all_zeros_bitstring = "0" * n_qubits_1
        all_ones_bitstring = "1" * n_qubits_2

        graph_ensemble.programs[prog_1_key]._best_probs = {
            "tag": {all_zeros_bitstring: 1.0}
        }
        graph_ensemble.programs[prog_2_key]._best_probs = {
            "tag": {all_ones_bitstring: 1.0}
        }

        # For any other programs, mock as all-zeros
        for key in prog_keys[2:]:
            n_qubits = graph_ensemble.programs[key].n_qubits
            graph_ensemble.programs[key]._best_probs = {"tag": {"0" * n_qubits: 1.0}}

        # Ensure all programs appear to have been run
        for program in graph_ensemble.programs.values():
            program._losses_history = [{"dummy_loss": 0.0}]

        # The expected global solution should contain only the original nodes from the second program
        # The problem object stores the reverse_index_maps
        problem = graph_ensemble._problem
        expected_nodes = set(problem._reverse_index_maps[prog_2_key].values())

        # Aggregate the results
        solution, energy = graph_ensemble.aggregate_results()

        # Verify that the aggregated solution matches the expected nodes
        assert set(solution) == expected_nodes
        assert isinstance(energy, float)

    def test_aggregate_results_raises_if_not_run(self, graph_ensemble):
        """
        Tests that a RuntimeError is raised if aggregate_results is called before
        the programs have been run.
        """
        graph_ensemble.create_programs()
        # Do not run the programs
        with pytest.raises(RuntimeError, match="Some/All programs have empty losses"):
            graph_ensemble.aggregate_results()

    def test_get_top_solutions_numerical_correctness(self, graph_ensemble):
        """Verify that get_top_solutions returns correctly ranked, distinct solutions.

        Uses the standard 15-node fixture. Each partition is given two candidates
        ("all-zeros" = no nodes selected, "all-ones" = all nodes selected).
        The beam search should produce distinct global solutions sorted by
        MaxCut energy (lower is better).
        """
        graph_ensemble.create_programs()
        prog_keys = list(graph_ensemble.programs.keys())
        n_nodes = _GRAPH.number_of_nodes()

        # Mock two candidates per partition
        for key in prog_keys:
            n_qubits = graph_ensemble.programs[key].n_qubits
            graph_ensemble.programs[key]._best_probs = {
                "tag": {"0" * n_qubits: 0.3, "1" * n_qubits: 0.7}
            }
            graph_ensemble.programs[key]._losses_history = [{"dummy_loss": 0.0}]

        n_partitions = len(prog_keys)
        # 2 candidates^n_partitions combinations
        n_expected = min(2**n_partitions, 2**n_partitions)

        results = graph_ensemble.get_top_solutions(
            n=n_expected, beam_width=n_expected, n_partition_candidates=n_expected
        )

        assert len(results) == n_expected
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)

        # All returned nodes must be valid graph nodes
        graph_nodes = set(_GRAPH.nodes())
        for nodes, energy in results:
            assert isinstance(nodes, list)
            assert isinstance(energy, float)
            assert set(nodes).issubset(graph_nodes)

        # Results should be sorted by energy (lower = more cut edges = better)
        energies = [energy for _nodes, energy in results]
        assert energies == sorted(energies)

        # Solutions should be distinct
        solution_tuples = [tuple(sorted(nodes)) for nodes, _energy in results]
        assert len(set(solution_tuples)) == len(results)

    def test_get_top_solutions_matches_aggregate_results(self, graph_ensemble):
        """The best solution from get_top_solutions matches aggregate_results."""
        graph_ensemble.create_programs()

        prog_keys = list(graph_ensemble.programs.keys())
        prog_1_key, prog_2_key = prog_keys[0], prog_keys[1]

        n_qubits_1 = graph_ensemble.programs[prog_1_key].n_qubits
        n_qubits_2 = graph_ensemble.programs[prog_2_key].n_qubits

        # Program 1: all-zeros (no nodes selected), Program 2: all-ones (all nodes selected)
        graph_ensemble.programs[prog_1_key]._best_probs = {
            "tag": {"0" * n_qubits_1: 1.0}
        }
        graph_ensemble.programs[prog_2_key]._best_probs = {
            "tag": {"1" * n_qubits_2: 1.0}
        }
        for key in prog_keys[2:]:
            n_qubits = graph_ensemble.programs[key].n_qubits
            graph_ensemble.programs[key]._best_probs = {"tag": {"0" * n_qubits: 1.0}}
        for program in graph_ensemble.programs.values():
            program._losses_history = [{"dummy_loss": 0.0}]

        problem = graph_ensemble._problem
        expected_nodes = set(problem._reverse_index_maps[prog_2_key].values())

        results = graph_ensemble.get_top_solutions(n=1, beam_width=1)
        assert len(results) == 1
        solution, energy = results[0]
        assert set(solution) == expected_nodes
        assert isinstance(energy, float)

    def test_get_top_solutions_raises_if_not_run(self, graph_ensemble):
        graph_ensemble.create_programs()
        with pytest.raises(RuntimeError, match="Some/All programs have empty losses"):
            graph_ensemble.get_top_solutions()

    def test_get_top_solutions_raises_on_invalid_n(self, graph_ensemble):
        with pytest.raises(ValueError, match="n must be >= 1"):
            graph_ensemble.get_top_solutions(n=0)


class TestExtendSolutionGraph:
    """Tests for problem.extend_solution on graph problems."""

    def test_sets_selected_nodes_and_zeroes_others(self, dummy_simulator):
        """Candidate's decoded nodes are set to 1; other partition nodes reset to 0."""
        graph = nx.path_graph(6)  # nodes 0-5
        problem = MaxCutProblem(
            graph,
            config=GraphPartitioningConfig(
                minimum_n_clusters=2, partitioning_algorithm="spectral"
            ),
        )
        ensemble = PartitioningProgramEnsemble(
            problem=problem,
            n_layers=1,
            optimizer=ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
            backend=dummy_simulator,
        )
        ensemble.create_programs()

        # Pick the first partition
        prog_id = list(problem._reverse_index_maps.keys())[0]
        reverse_map = problem._reverse_index_maps[prog_id]

        # Create a candidate that selects only the first local node
        first_local_node = list(reverse_map.keys())[0]

        initial = [0] * graph.number_of_nodes()
        result = problem.extend_solution(initial, prog_id, [first_local_node])

        # The global index for the selected node should be 1
        assert result[reverse_map[first_local_node]] == 1
        # All other partition positions should be 0
        for local_node, global_idx in reverse_map.items():
            if local_node != first_local_node:
                assert result[global_idx] == 0

    def test_resets_partition_positions_before_applying(self, dummy_simulator):
        """Pre-existing 1s in the partition's positions are cleared first."""
        graph = nx.path_graph(6)
        problem = MaxCutProblem(
            graph,
            config=GraphPartitioningConfig(
                minimum_n_clusters=2, partitioning_algorithm="spectral"
            ),
        )
        ensemble = PartitioningProgramEnsemble(
            problem=problem,
            n_layers=1,
            optimizer=ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
            backend=dummy_simulator,
        )
        ensemble.create_programs()

        prog_id = list(problem._reverse_index_maps.keys())[0]
        reverse_map = problem._reverse_index_maps[prog_id]

        # Start with all 1s, apply empty decoded -> all partition positions become 0
        result = problem.extend_solution([1] * graph.number_of_nodes(), prog_id, [])

        # All positions for this partition should be 0
        for global_idx in reverse_map.values():
            assert result[global_idx] == 0

    def test_does_not_mutate_input(self, dummy_simulator):
        """extend_solution returns a new list, not a mutation of the input."""
        graph = nx.path_graph(4)
        problem = MaxCutProblem(
            graph,
            config=GraphPartitioningConfig(
                minimum_n_clusters=2, partitioning_algorithm="spectral"
            ),
        )
        ensemble = PartitioningProgramEnsemble(
            problem=problem,
            n_layers=1,
            optimizer=ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
            backend=dummy_simulator,
        )
        ensemble.create_programs()

        prog_id = list(problem._reverse_index_maps.keys())[0]
        original = [0] * graph.number_of_nodes()

        result = problem.extend_solution(original, prog_id, [])

        assert result is not original
        assert original == [0] * graph.number_of_nodes()


class TestDecomposeGraph:
    def test_decompose_returns_correct_number_of_subproblems(self):
        graph = nx.cycle_graph(10)
        problem = MaxCutProblem(
            graph,
            config=GraphPartitioningConfig(
                minimum_n_clusters=2, partitioning_algorithm="kernighan_lin"
            ),
        )

        sub_problems = problem.decompose()

        assert len(sub_problems) >= 2
        for prog_id, sub_problem in sub_problems.items():
            assert isinstance(sub_problem, MaxCutProblem)

    def test_decompose_populates_reverse_index_maps(self):
        graph = nx.cycle_graph(10)
        problem = MaxCutProblem(
            graph,
            config=GraphPartitioningConfig(
                minimum_n_clusters=2, partitioning_algorithm="kernighan_lin"
            ),
        )

        sub_problems = problem.decompose()

        assert len(problem._reverse_index_maps) == len(sub_problems)
        all_original_nodes = set()
        for prog_id in sub_problems:
            assert prog_id in problem._reverse_index_maps
            all_original_nodes |= set(problem._reverse_index_maps[prog_id].values())
        assert all_original_nodes == set(graph.nodes())

    def test_decompose_raises_without_partitioning_config(self):
        graph = nx.complete_graph(6)
        problem = MaxCutProblem(graph)

        with pytest.raises(ValueError, match="Cannot decompose"):
            problem.decompose()

    def test_decompose_sub_problems_are_same_type(self):
        graph = nx.cycle_graph(10)
        problem = MaxCutProblem(
            graph,
            config=GraphPartitioningConfig(
                minimum_n_clusters=2, partitioning_algorithm="kernighan_lin"
            ),
        )

        sub_problems = problem.decompose()

        for sub_problem in sub_problems.values():
            assert type(sub_problem) == MaxCutProblem


class TestPartitioningWarnings:
    def test_warns_for_max_clique(self):
        graph = nx.cycle_graph(6)
        problem = MaxCliqueProblem(
            graph,
            config=GraphPartitioningConfig(minimum_n_clusters=2),
        )

        with pytest.warns(UserWarning, match="Heuristic-risk"):
            problem.decompose()

    def test_warns_for_max_independent_set(self):
        graph = nx.cycle_graph(6)
        problem = MaxIndependentSetProblem(
            graph,
            config=GraphPartitioningConfig(minimum_n_clusters=2),
        )

        with pytest.warns(UserWarning, match="Heuristic-risk"):
            problem.decompose()

    def test_warns_for_min_vertex_cover(self):
        graph = nx.cycle_graph(6)
        problem = MinVertexCoverProblem(
            graph,
            config=GraphPartitioningConfig(minimum_n_clusters=2),
        )

        with pytest.warns(UserWarning, match="Heuristic-risk"):
            problem.decompose()

    def test_no_warning_for_maxcut(self):
        graph = nx.cycle_graph(6)
        problem = MaxCutProblem(
            graph,
            config=GraphPartitioningConfig(minimum_n_clusters=2),
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            problem.decompose()


class TestInitialSolutionSizeGraph:
    def test_returns_number_of_nodes(self):
        graph = nx.cycle_graph(7)
        problem = MaxCutProblem(graph)

        assert problem.initial_solution_size() == graph.number_of_nodes()


class TestFinalizeSolutionGraph:
    def test_returns_node_indices_and_energy(self):
        graph = nx.path_graph(4)
        problem = MaxCutProblem(graph)

        nodes, energy = problem.finalize_solution(-5.0, [1, 0, 1, 0])

        assert nodes == [0, 2]
        assert energy == -5.0

    def test_all_zeros_returns_empty(self):
        graph = nx.path_graph(3)
        problem = MaxCutProblem(graph)

        nodes, energy = problem.finalize_solution(0.0, [0, 0, 0])

        assert nodes == []
        assert energy == 0.0


class TestFormatTopSolutionsGraph:
    def test_formats_multiple_results(self):
        graph = nx.path_graph(4)
        problem = MaxCutProblem(graph)

        results = [(-5.0, [1, 0, 1, 0]), (-3.0, [0, 1, 0, 1])]
        formatted = problem.format_top_solutions(results)

        assert formatted == [([0, 2], -5.0), ([1, 3], -3.0)]


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
