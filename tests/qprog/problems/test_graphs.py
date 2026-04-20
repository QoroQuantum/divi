# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import warnings

import networkx as nx
import numpy as np
import pytest

from divi.backends import CircuitRunner
from divi.qprog import (
    QAOA,
    ScipyMethod,
    ScipyOptimizer,
    SuperpositionState,
    ZerosState,
)
from divi.qprog.checkpointing import CheckpointConfig
from divi.qprog.problems import (
    GraphPartitioningConfig,
    MaxCliqueProblem,
    MaxCutProblem,
    MaxIndependentSetProblem,
    MinVertexCoverProblem,
)
from divi.qprog.workflows import PartitioningProgramEnsemble
from tests.qprog.problems._helpers import make_bull_graph, make_string_node_graph
from tests.qprog.qprog_contracts import (
    CHECKPOINTING_OPTIMIZERS,
    OPTIMIZERS_TO_TEST,
    verify_metacircuit_dict,
)

# ---------------------------------------------------------------------------
# evaluate_global_solution
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# extend_solution
# ---------------------------------------------------------------------------


class TestExtendSolutionGraph:
    """Tests for problem.extend_solution on graph problems."""

    def test_sets_selected_nodes_and_zeroes_others(self):
        """Candidate's decoded nodes are set to 1; other partition nodes reset to 0."""
        graph = nx.path_graph(6)  # nodes 0-5
        problem = MaxCutProblem(
            graph,
            config=GraphPartitioningConfig(
                minimum_n_clusters=2, partitioning_algorithm="spectral"
            ),
        )
        problem.decompose()

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

    def test_resets_partition_positions_before_applying(self):
        """Pre-existing 1s in the partition's positions are cleared first."""
        graph = nx.path_graph(6)
        problem = MaxCutProblem(
            graph,
            config=GraphPartitioningConfig(
                minimum_n_clusters=2, partitioning_algorithm="spectral"
            ),
        )
        problem.decompose()

        prog_id = list(problem._reverse_index_maps.keys())[0]
        reverse_map = problem._reverse_index_maps[prog_id]

        # Start with all 1s, apply empty decoded -> all partition positions become 0
        result = problem.extend_solution([1] * graph.number_of_nodes(), prog_id, [])

        # All positions for this partition should be 0
        for global_idx in reverse_map.values():
            assert result[global_idx] == 0

    def test_does_not_mutate_input(self):
        """extend_solution returns a new list, not a mutation of the input."""
        graph = nx.path_graph(4)
        problem = MaxCutProblem(
            graph,
            config=GraphPartitioningConfig(
                minimum_n_clusters=2, partitioning_algorithm="spectral"
            ),
        )
        problem.decompose()

        prog_id = list(problem._reverse_index_maps.keys())[0]
        original = [0] * graph.number_of_nodes()

        result = problem.extend_solution(original, prog_id, [])

        assert result is not original
        assert original == [0] * graph.number_of_nodes()


# ---------------------------------------------------------------------------
# decompose
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Partitioning warnings
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# initial_solution_size
# ---------------------------------------------------------------------------


class TestInitialSolutionSizeGraph:
    def test_returns_number_of_nodes(self):
        graph = nx.cycle_graph(7)
        problem = MaxCutProblem(graph)

        assert problem.initial_solution_size() == graph.number_of_nodes()


# ---------------------------------------------------------------------------
# finalize_solution
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# format_top_solutions
# ---------------------------------------------------------------------------


class TestFormatTopSolutionsGraph:
    def test_formats_multiple_results(self):
        graph = nx.path_graph(4)
        problem = MaxCutProblem(graph)

        results = [(-5.0, [1, 0, 1, 0]), (-3.0, [0, 1, 0, 1])]
        formatted = problem.format_top_solutions(results)

        assert formatted == [([0, 2], -5.0), ([1, 3], -3.0)]


# ---------------------------------------------------------------------------
# QAOA + Graph Problem integration tests
# ---------------------------------------------------------------------------


class TestGraphInput:
    def test_graph_basic_initialization(self, default_test_simulator):
        G = make_bull_graph()

        qaoa_problem = QAOA(
            MaxCliqueProblem(G, is_constrained=True),
            n_layers=1,
            optimizer=ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
            max_iterations=10,
            backend=default_test_simulator,
        )

        assert isinstance(qaoa_problem.backend, CircuitRunner)
        assert qaoa_problem.backend.shots == 5000
        assert isinstance(qaoa_problem.optimizer, ScipyOptimizer)
        assert qaoa_problem.optimizer.method == ScipyMethod.NELDER_MEAD
        assert qaoa_problem.max_iterations == 10
        assert isinstance(qaoa_problem.problem, MaxCliqueProblem)
        assert qaoa_problem.problem.graph == G
        assert qaoa_problem.n_layers == 1

        verify_metacircuit_dict(qaoa_problem, ["cost_circuit", "meas_circuit"])

    def test_graph_unsuppported_initial_state(self, dummy_simulator):
        with pytest.raises(TypeError):
            QAOA(
                MaxCliqueProblem(nx.bull_graph(), is_constrained=True),
                initial_state="Bell",
                backend=dummy_simulator,
            )

    def test_constant_only_hamiltonian_raises(self, dummy_simulator):
        """QAOA rejects constant-only cost Hamiltonian (e.g. empty graph) at init."""
        with pytest.raises(
            ValueError, match="Hamiltonian contains only constant terms"
        ):
            QAOA(
                MaxCutProblem(nx.empty_graph(1)),
                backend=dummy_simulator,
            )

    def test_graph_initial_state_recommended(self, dummy_simulator):
        qaoa_problem = QAOA(
            MaxCliqueProblem(nx.bull_graph(), is_constrained=True),
            backend=dummy_simulator,
        )

        assert isinstance(qaoa_problem.initial_state, ZerosState)

    def test_graph_initial_state_superposition(self, dummy_simulator):
        qaoa_problem = QAOA(
            MaxCliqueProblem(nx.bull_graph(), is_constrained=True),
            initial_state=SuperpositionState(),
            backend=dummy_simulator,
        )

        assert isinstance(qaoa_problem.initial_state, SuperpositionState)
        # SuperpositionState seeds each qubit with a Hadamard; verify via DAG.
        _, dag = qaoa_problem.meta_circuit_factories["cost_circuit"].circuit_bodies[0]
        n_hadamards = sum(1 for n in dag.op_nodes() if n.op.name == "h")
        assert n_hadamards >= nx.bull_graph().number_of_nodes()

    def test_perform_final_computation_extracts_correct_solution(
        self, mocker, dummy_simulator
    ):
        G = make_bull_graph()
        qaoa_problem = QAOA(
            MaxCliqueProblem(G, is_constrained=True),
            n_layers=1,
            optimizer=ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
            max_iterations=1,
            backend=dummy_simulator,
        )

        # Simulate measurement results
        qaoa_problem._best_probs = {
            "0_NoMitigation:0_0": {"11001": 0.1444, "00101": 0.0526}
        }

        # Patch measurement to do nothing (since we set probs manually)
        mocker.patch.object(qaoa_problem, "_run_solution_measurement_for")

        qaoa_problem._perform_final_computation()

        # Should extract bitstring "11001"
        assert qaoa_problem._decoded_solution == [0, 1, 4]
        assert qaoa_problem.solution == [0, 1, 4]

    @pytest.mark.e2e
    @pytest.mark.parametrize("optimizer", **OPTIMIZERS_TO_TEST)
    def test_graph_qaoa_e2e_solution(self, optimizer, default_test_simulator):
        optimizer = optimizer()  # Create fresh instance

        G = nx.bull_graph()

        # L-BFGS-B needs more layers for sufficient circuit expressibility
        # to solve MAX_CLIQUE — 1 layer converges to a local optimum.
        n_layers = (
            2
            if isinstance(optimizer, ScipyOptimizer)
            and optimizer.method == ScipyMethod.L_BFGS_B
            else 1
        )

        default_test_simulator.set_seed(1997)

        qaoa_problem = QAOA(
            MaxCliqueProblem(G, is_constrained=True),
            n_layers=n_layers,
            optimizer=optimizer,
            max_iterations=10,
            backend=default_test_simulator,
            seed=1997,
        )

        qaoa_problem.run()

        assert all(
            len(bitstring) == G.number_of_nodes()
            for probs_dict in qaoa_problem.best_probs.values()
            for bitstring in probs_dict.keys()
        )

        assert set(qaoa_problem.solution) == nx.algorithms.approximation.max_clique(G)

    @pytest.mark.e2e
    @pytest.mark.parametrize("optimizer", **CHECKPOINTING_OPTIMIZERS)
    def test_graph_qaoa_e2e_checkpointing_resume(
        self, optimizer, default_test_simulator, tmp_path
    ):
        """Test QAOA e2e with checkpointing and multiple resume cycles.

        Tests checkpoint infrastructure (multiple save/load cycles) with all checkpointing-capable
        optimizers to verify their nuanced checkpoint handling (CMAES generator reinit, DE pop handling).
        """
        optimizer = optimizer()  # Create fresh instance

        G = nx.bull_graph()
        checkpoint_dir = tmp_path / "checkpoint_test"
        default_test_simulator.set_seed(1997)

        max_clique_problem = MaxCliqueProblem(G, is_constrained=True)

        # First run: iterations 1-3
        qaoa_problem1 = QAOA(
            max_clique_problem,
            n_layers=1,
            optimizer=optimizer,
            max_iterations=3,
            backend=default_test_simulator,
            seed=1997,
        )
        qaoa_problem1.run(
            checkpoint_config=CheckpointConfig(checkpoint_dir=checkpoint_dir)
        )
        assert qaoa_problem1.current_iteration == 3

        # Verify checkpoint was created
        checkpoint_path = checkpoint_dir / "checkpoint_003"
        assert checkpoint_path.exists()
        assert (checkpoint_path / "program_state.json").exists()

        # Store state from first run for comparison
        first_run_iteration = qaoa_problem1.current_iteration
        first_run_losses_count = len(qaoa_problem1.losses_history)

        # Second run: resume and run iterations 4-6
        qaoa_problem2 = QAOA.load_state(
            checkpoint_dir,
            backend=default_test_simulator,
            problem=max_clique_problem,
            n_layers=1,
        )

        # Verify loaded state matches first run
        assert qaoa_problem2.current_iteration == first_run_iteration
        assert len(qaoa_problem2.losses_history) == first_run_losses_count

        qaoa_problem2.max_iterations = 6
        qaoa_problem2.run(
            checkpoint_config=CheckpointConfig(checkpoint_dir=checkpoint_dir)
        )
        assert qaoa_problem2.current_iteration == 6
        assert (checkpoint_dir / "checkpoint_006").exists()

        # Third run: resume and run iterations 7-10
        qaoa_problem3 = QAOA.load_state(
            checkpoint_dir,
            backend=default_test_simulator,
            problem=max_clique_problem,
            n_layers=1,
        )
        assert qaoa_problem3.current_iteration == 6
        qaoa_problem3.max_iterations = 10
        qaoa_problem3.run()
        assert qaoa_problem3.current_iteration == 10

        # Verify final results are correct
        assert all(
            len(bitstring) == G.number_of_nodes()
            for probs_dict in qaoa_problem3.best_probs.values()
            for bitstring in probs_dict.keys()
        )
        assert set(qaoa_problem3.solution) == nx.algorithms.approximation.max_clique(G)

    def test_string_node_labels_bitstring_length(self, mocker, dummy_simulator):
        """Test that graphs with string node labels produce correct bitstring lengths."""
        G = nx.Graph()
        G.add_nodes_from(["0", "1", "2", "3"])
        G.add_edges_from([("0", "1"), ("1", "2"), ("2", "3")])

        qaoa_problem = QAOA(
            MaxCutProblem(G),
            n_layers=1,
            optimizer=ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
            max_iterations=1,
            backend=dummy_simulator,
        )

        # Verify circuit_wires are correctly set up with string node labels
        assert len(qaoa_problem._circuit_wires) == G.number_of_nodes()
        assert all(wire in G.nodes() for wire in qaoa_problem._circuit_wires)

        # Mock optimizer and measurement to skip expensive circuit execution
        mock_result = mocker.MagicMock()
        mock_result.x, mock_result.fun, mock_result.nfev, mock_result.njev = (
            np.array([[0.1, 0.2]]),
            0.5,
            1,
            0,
        )
        mocker.patch.object(
            qaoa_problem.optimizer, "optimize", return_value=mock_result
        )

        # Mock best_probs with bitstrings of correct length (4 bits for 4 nodes)
        n_nodes = G.number_of_nodes()
        mock_probs = {"0_0": {f"{i:0{n_nodes}b}": 0.25 for i in range(4)}}
        mocker.patch.object(
            qaoa_problem,
            "_run_solution_measurement_for",
            side_effect=lambda _param_sets: setattr(
                qaoa_problem, "_best_probs", mock_probs
            ),
        )

        qaoa_problem.run()

        # Verify all bitstrings have the correct length
        assert all(
            len(bitstring) == n_nodes
            for probs_dict in qaoa_problem.best_probs.values()
            for bitstring in probs_dict.keys()
        )

    def test_string_node_labels_solution_mapping(self, mocker, dummy_simulator):
        """Test that solution correctly maps to string node labels."""
        G = nx.Graph()
        G.add_nodes_from(["a", "b", "c"])
        G.add_edges_from([("a", "b"), ("b", "c")])

        qaoa_problem = QAOA(
            MaxCutProblem(G),
            n_layers=1,
            optimizer=ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
            max_iterations=1,
            backend=dummy_simulator,
        )

        qaoa_problem._best_probs = {"0_NoMitigation:0_0": {"101": 0.6, "010": 0.4}}
        mocker.patch.object(qaoa_problem, "_run_solution_measurement_for")

        qaoa_problem._perform_final_computation()

        assert all(isinstance(node, str) for node in qaoa_problem.solution)
        assert len(qaoa_problem.solution) == 2
        assert all(node in G.nodes() for node in qaoa_problem.solution)

    def test_string_node_labels_circuit_wires(self, dummy_simulator):
        """Test that circuit_wires correctly uses Hamiltonian wire labels."""
        G = nx.Graph()
        G.add_nodes_from(["x", "y", "z"])
        G.add_edges_from([("x", "y"), ("y", "z")])

        qaoa_problem = QAOA(
            MaxCutProblem(G),
            n_layers=1,
            optimizer=ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
            max_iterations=1,
            backend=dummy_simulator,
        )

        assert isinstance(qaoa_problem._circuit_wires, tuple)
        assert len(qaoa_problem._circuit_wires) == G.number_of_nodes()
        assert all(wire in G.nodes() for wire in qaoa_problem._circuit_wires)
        assert all(isinstance(wire, str) for wire in qaoa_problem._circuit_wires)

    def test_mixed_type_node_labels(self, mocker, dummy_simulator):
        """Test that graphs with mixed type node labels work correctly."""
        G = nx.Graph()
        G.add_nodes_from([0, "1", 2, "3"])
        G.add_edges_from([(0, "1"), ("1", 2), (2, "3")])

        qaoa_problem = QAOA(
            MaxCutProblem(G),
            n_layers=1,
            optimizer=ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
            max_iterations=1,
            backend=dummy_simulator,
        )

        assert len(qaoa_problem._circuit_wires) == G.number_of_nodes()
        assert all(wire in G.nodes() for wire in qaoa_problem._circuit_wires)

        qaoa_problem._best_probs = {"0_NoMitigation:0_0": {"1010": 0.5, "0101": 0.5}}
        mocker.patch.object(qaoa_problem, "_run_solution_measurement_for")

        qaoa_problem._perform_final_computation()

        assert len(qaoa_problem.solution) == 2
        assert all(node in G.nodes() for node in qaoa_problem.solution)

    @pytest.mark.e2e
    def test_string_node_labels_e2e(self, default_test_simulator):
        """End-to-end test with string node labels."""
        G = make_string_node_graph()

        default_test_simulator.set_seed(1997)

        qaoa_problem = QAOA(
            MaxCutProblem(G),
            n_layers=1,
            optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
            max_iterations=5,
            backend=default_test_simulator,
            seed=1997,
        )

        qaoa_problem.run()

        assert all(
            len(bitstring) == G.number_of_nodes()
            for probs_dict in qaoa_problem.best_probs.values()
            for bitstring in probs_dict.keys()
        )
        assert all(isinstance(node, str) for node in qaoa_problem.solution)
        assert all(node in G.nodes() for node in qaoa_problem.solution)


# ---------------------------------------------------------------------------
# Graph PartitioningProgramEnsemble integration tests
# ---------------------------------------------------------------------------

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


@pytest.fixture
def graph_ensemble(ensemble_args):
    return PartitioningProgramEnsemble(**ensemble_args)


class TestGraphPartitioningEnsemble:
    def test_correct_number_of_programs_created(self, mocker, graph_ensemble):
        mocker.patch("divi.qprog.QAOA")

        graph_ensemble.create_programs()

        assert len(graph_ensemble.programs) >= _PARTITIONING_CONFIG.minimum_n_clusters

    def test_results_aggregated_correctly(self, graph_ensemble):
        graph_ensemble.create_programs()

        prog_keys = list(graph_ensemble.programs.keys())
        prog_1_key, prog_2_key = prog_keys[0], prog_keys[1]

        n_qubits_1 = graph_ensemble.programs[prog_1_key].n_qubits
        n_qubits_2 = graph_ensemble.programs[prog_2_key].n_qubits

        all_zeros_bitstring = "0" * n_qubits_1
        all_ones_bitstring = "1" * n_qubits_2

        graph_ensemble.programs[prog_1_key]._best_probs = {
            "tag": {all_zeros_bitstring: 1.0}
        }
        graph_ensemble.programs[prog_2_key]._best_probs = {
            "tag": {all_ones_bitstring: 1.0}
        }

        for key in prog_keys[2:]:
            n_qubits = graph_ensemble.programs[key].n_qubits
            graph_ensemble.programs[key]._best_probs = {"tag": {"0" * n_qubits: 1.0}}

        for program in graph_ensemble.programs.values():
            program._losses_history = [{"dummy_loss": 0.0}]

        problem = graph_ensemble._problem
        expected_nodes = set(problem._reverse_index_maps[prog_2_key].values())

        solution, energy = graph_ensemble.aggregate_results()

        assert set(solution) == expected_nodes
        assert isinstance(energy, float)

    def test_get_top_solutions_numerical_correctness(self, graph_ensemble):
        """Verify that get_top_solutions returns correctly ranked, distinct solutions."""
        graph_ensemble.create_programs()
        prog_keys = list(graph_ensemble.programs.keys())
        _GRAPH.number_of_nodes()

        for key in prog_keys:
            n_qubits = graph_ensemble.programs[key].n_qubits
            graph_ensemble.programs[key]._best_probs = {
                "tag": {"0" * n_qubits: 0.3, "1" * n_qubits: 0.7}
            }
            graph_ensemble.programs[key]._losses_history = [{"dummy_loss": 0.0}]

        n_partitions = len(prog_keys)
        n_expected = min(2**n_partitions, 2**n_partitions)

        results = graph_ensemble.get_top_solutions(
            n=n_expected, beam_width=n_expected, n_partition_candidates=n_expected
        )

        assert len(results) == n_expected
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)

        graph_nodes = set(_GRAPH.nodes())
        for nodes, energy in results:
            assert isinstance(nodes, list)
            assert isinstance(energy, float)
            assert set(nodes).issubset(graph_nodes)

        energies = [energy for _nodes, energy in results]
        assert energies == sorted(energies)

        solution_tuples = [tuple(sorted(nodes)) for nodes, _energy in results]
        assert len(set(solution_tuples)) == len(results)

    def test_get_top_solutions_matches_aggregate_results(self, graph_ensemble):
        """The best solution from get_top_solutions matches aggregate_results."""
        graph_ensemble.create_programs()

        prog_keys = list(graph_ensemble.programs.keys())
        prog_1_key, prog_2_key = prog_keys[0], prog_keys[1]

        n_qubits_1 = graph_ensemble.programs[prog_1_key].n_qubits
        n_qubits_2 = graph_ensemble.programs[prog_2_key].n_qubits

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
