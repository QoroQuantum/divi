# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from functools import partial

import dimod
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pennylane as qml
import pytest
import scipy.sparse as sps
from mitiq.zne.inference import LinearFactory
from mitiq.zne.scaling import fold_global

from divi.backends import CircuitRunner
from divi.circuits.qem import ZNE
from divi.hamiltonians import ExactTrotterization, QDrift
from divi.pipeline.stages import TrotterSpecStage
from divi.qprog import (
    QAOA,
    GraphProblem,
    ScipyMethod,
    ScipyOptimizer,
)
from divi.qprog.algorithms import _qaoa
from divi.qprog.checkpointing import CheckpointConfig
from tests.qprog.qprog_contracts import (
    CHECKPOINTING_OPTIMIZERS,
    OPTIMIZERS_TO_TEST,
    verify_correct_circuit_count,
    verify_metacircuit_dict,
)

pytestmark = pytest.mark.algo


class TestGeneralQAOA:
    @pytest.mark.parametrize("optimizer", **OPTIMIZERS_TO_TEST)
    def test_qaoa_optimization_runs_cost_pipeline(
        self, mocker, optimizer, default_test_simulator
    ):
        """
        Verifies that the cost pipeline is used during the optimization loop.
        """
        optimizer = optimizer()  # Create fresh instance
        qaoa_problem = QAOA(
            problem=nx.bull_graph(),
            graph_problem=GraphProblem.MAX_CLIQUE,
            n_layers=1,
            optimizer=optimizer,
            max_iterations=1,
            backend=default_test_simulator,
            is_constrained=True,
        )

        # Spy on the cost pipeline's run method
        spy = mocker.spy(qaoa_problem._cost_pipeline, "run")

        # Mock final computation to isolate optimization phase
        mocker.patch.object(qaoa_problem, "_perform_final_computation")

        qaoa_problem.run()

        # Cost pipeline should be called once per iteration
        assert spy.call_count >= 1

    @pytest.mark.parametrize("optimizer", **OPTIMIZERS_TO_TEST)
    def test_qaoa_final_computation_runs_measurement_pipeline(
        self, mocker, optimizer, default_test_simulator
    ):
        """
        Verifies that the measurement pipeline is used during final computation.
        """
        optimizer = optimizer()  # Create fresh instance
        qaoa_problem = QAOA(
            problem=nx.bull_graph(),
            graph_problem=GraphProblem.MAX_CLIQUE,
            n_layers=1,
            optimizer=optimizer,
            max_iterations=1,
            backend=default_test_simulator,
            is_constrained=True,
        )
        # Set preconditions
        qaoa_problem._final_params = np.array([[0.1, 0.2]])
        qaoa_problem._best_params = np.array([[0.1, 0.2]])

        # Spy on measurement pipeline
        spy = mocker.spy(qaoa_problem._measurement_pipeline, "run")

        qaoa_problem._perform_final_computation()

        # Measurement pipeline should be called once
        assert spy.call_count == 1

    @pytest.mark.parametrize("optimizer", **OPTIMIZERS_TO_TEST)
    def test_graph_correct_circuits_count_and_energies(
        self, optimizer, dummy_simulator
    ):
        optimizer = optimizer()  # Create fresh instance
        qaoa_problem = QAOA(
            problem=nx.bull_graph(),
            graph_problem=GraphProblem.MAX_CLIQUE,
            n_layers=1,
            optimizer=optimizer,
            max_iterations=1,
            is_constrained=True,
            backend=dummy_simulator,
        )

        qaoa_problem.run()

        verify_correct_circuit_count(qaoa_problem)


class TestGraphInput:
    def test_graph_basic_initialization(self, default_test_simulator):
        G = nx.bull_graph()

        qaoa_problem = QAOA(
            problem=G,
            graph_problem=GraphProblem.MAX_CLIQUE,
            n_layers=1,
            optimizer=ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
            max_iterations=10,
            backend=default_test_simulator,
            is_constrained=True,
        )

        assert isinstance(qaoa_problem.backend, CircuitRunner)
        assert qaoa_problem.backend.shots == 5000
        assert isinstance(qaoa_problem.optimizer, ScipyOptimizer)
        assert qaoa_problem.optimizer.method == ScipyMethod.NELDER_MEAD
        assert qaoa_problem.max_iterations == 10
        assert qaoa_problem.graph_problem == GraphProblem.MAX_CLIQUE
        assert qaoa_problem.problem == G
        assert qaoa_problem.n_layers == 1

        verify_metacircuit_dict(qaoa_problem, ["cost_circuit", "meas_circuit"])

    def test_graph_unsuppported_problem(self):
        with pytest.raises(ValueError, match="travelling_salesman"):
            QAOA(
                problem=nx.bull_graph(),
                graph_problem="travelling_salesman",
                backend=None,
            )

    def test_graph_unsuppported_initial_state(self, dummy_simulator):
        with pytest.raises(ValueError, match="Bell"):
            QAOA(
                problem=nx.bull_graph(),
                graph_problem=GraphProblem.MAX_CLIQUE,
                initial_state="Bell",
                backend=dummy_simulator,
            )

    def test_constant_only_hamiltonian_raises(self, dummy_simulator):
        """QAOA rejects constant-only cost Hamiltonian (e.g. empty graph) at init."""
        with pytest.raises(
            ValueError, match="Hamiltonian contains only constant terms"
        ):
            QAOA(
                problem=nx.empty_graph(1),
                graph_problem=GraphProblem.MAXCUT,
                backend=dummy_simulator,
            )

    def test_graph_initial_state_recommended(self, dummy_simulator):
        qaoa_problem = QAOA(
            problem=nx.bull_graph(),
            graph_problem=GraphProblem.MAX_CLIQUE,
            initial_state="Recommended",
            is_constrained=True,
            backend=dummy_simulator,
        )

        assert qaoa_problem.initial_state == "Zeros"

    def test_graph_initial_state_superposition(self, dummy_simulator):
        qaoa_problem = QAOA(
            problem=nx.bull_graph(),
            graph_problem=GraphProblem.MAX_CLIQUE,
            initial_state="Superposition",
            backend=dummy_simulator,
        )

        assert qaoa_problem.initial_state == "Superposition"
        assert (
            sum(
                isinstance(op, qml.Hadamard)
                for op in qaoa_problem.meta_circuit_factories[
                    "cost_circuit"
                ].source_circuit.operations
            )
            == nx.bull_graph().number_of_nodes()
        )

    def test_perform_final_computation_extracts_correct_solution(
        self, mocker, dummy_simulator
    ):
        G = nx.bull_graph()
        qaoa_problem = QAOA(
            graph_problem=GraphProblem.MAX_CLIQUE,
            problem=G,
            n_layers=1,
            optimizer=ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
            max_iterations=1,
            is_constrained=True,
            backend=dummy_simulator,
        )

        # Simulate measurement results
        qaoa_problem._best_probs = {
            "0_NoMitigation:0_0": {"11001": 0.1444, "00101": 0.0526}
        }

        # Patch _run_solution_measurement to do nothing (since we set probs manually)
        mocker.patch.object(qaoa_problem, "_run_solution_measurement")

        qaoa_problem._perform_final_computation()

        # Should extract bitstring "11001"
        assert qaoa_problem._solution_nodes == [0, 1, 4]
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
            graph_problem=GraphProblem.MAX_CLIQUE,
            problem=G,
            n_layers=n_layers,
            optimizer=optimizer,
            max_iterations=10,
            is_constrained=True,
            backend=default_test_simulator,
            seed=1997,
        )

        qaoa_problem.run()

        assert all(
            len(bitstring) == G.number_of_nodes()
            for probs_dict in qaoa_problem.best_probs.values()
            for bitstring in probs_dict.keys()
        )

        assert set(
            qaoa_problem._solution_nodes
        ) == nx.algorithms.approximation.max_clique(G)

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

        # First run: iterations 1-3
        qaoa_problem1 = QAOA(
            graph_problem=GraphProblem.MAX_CLIQUE,
            problem=G,
            n_layers=1,
            optimizer=optimizer,
            max_iterations=3,
            is_constrained=True,
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
            problem=G,
            graph_problem=GraphProblem.MAX_CLIQUE,
            n_layers=1,
            is_constrained=True,
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
            problem=G,
            graph_problem=GraphProblem.MAX_CLIQUE,
            n_layers=1,
            is_constrained=True,
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
        assert set(
            qaoa_problem3._solution_nodes
        ) == nx.algorithms.approximation.max_clique(G)

    def test_draw_solution_returns_graph_with_expected_properties(
        self, mocker, dummy_simulator
    ):
        G = nx.bull_graph()

        qaoa_problem = QAOA(
            graph_problem=GraphProblem.MAX_CLIQUE,
            problem=G,
            n_layers=1,
            optimizer=ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
            max_iterations=2,
            is_constrained=True,
            backend=dummy_simulator,
        )

        qaoa_problem._solution_nodes = [0, 1, 2]

        # 1. Mock the entire 'nx' alias within the _qaoa module in one line
        mock_nx = mocker.patch.object(_qaoa, "nx")

        # 2. Mock the pyplot 'show' function
        mocker.patch("matplotlib.pyplot.show")

        # 3. Call the function you want to test
        qaoa_problem.draw_solution()

        # 4. Verify that the methods on your single mock were called
        mock_nx.draw_networkx_nodes.assert_called_once()
        mock_nx.draw_networkx_edges.assert_called_once()
        mock_nx.draw_networkx_labels.assert_called_once()

        # Get the node_color argument that was passed to draw_networkx_nodes
        node_colors = mock_nx.draw_networkx_nodes.call_args[1]["node_color"]

        # Verify that solution nodes are red and non-solution nodes are lightblue
        expected_colors = [
            "red" if node in qaoa_problem._solution_nodes else "lightblue"
            for node in G.nodes()
        ]
        assert node_colors == expected_colors

        # Verify node size
        assert mock_nx.draw_networkx_nodes.call_args[1]["node_size"] == 500

        # Clean up the plot
        plt.close()

    def test_string_node_labels_bitstring_length(self, mocker, dummy_simulator):
        """Test that graphs with string node labels produce correct bitstring lengths.

        This test verifies that bitstrings have the correct length (matching number of nodes)
        when using string node labels. The test mocks circuit execution for speed while
        still validating the bitstring length logic. For full integration testing, see
        test_string_node_labels_e2e.
        """
        G = nx.Graph()
        G.add_nodes_from(["0", "1", "2", "3"])
        G.add_edges_from([("0", "1"), ("1", "2"), ("2", "3")])

        qaoa_problem = QAOA(
            problem=G,
            graph_problem=GraphProblem.MAXCUT,
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
            "_run_solution_measurement",
            side_effect=lambda: setattr(qaoa_problem, "_best_probs", mock_probs),
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
        # Create a graph with string node labels
        G = nx.Graph()
        G.add_nodes_from(["a", "b", "c"])
        G.add_edges_from([("a", "b"), ("b", "c")])

        qaoa_problem = QAOA(
            problem=G,
            graph_problem=GraphProblem.MAXCUT,
            n_layers=1,
            optimizer=ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
            max_iterations=1,
            is_constrained=True,
            backend=dummy_simulator,
        )

        # Simulate measurement results with bitstring "101" (nodes 'a' and 'c' selected)
        # The bitstring order corresponds to the Hamiltonian's wire order
        qaoa_problem._best_probs = {"0_NoMitigation:0_0": {"101": 0.6, "010": 0.4}}

        # Patch _run_solution_measurement to do nothing (since we set probs manually)
        mocker.patch.object(qaoa_problem, "_run_solution_measurement")

        qaoa_problem._perform_final_computation()

        # Verify solution contains the correct node labels (not integer indices)
        # The solution should be a list of the actual graph node labels
        assert all(
            isinstance(node, str) for node in qaoa_problem.solution
        ), "Solution should contain string node labels, not integers"
        assert len(qaoa_problem.solution) == 2, "Bitstring '101' should map to 2 nodes"
        # Verify the solution nodes are valid graph nodes
        assert all(
            node in G.nodes() for node in qaoa_problem.solution
        ), "All solution nodes should be valid graph nodes"

    def test_string_node_labels_circuit_wires(self, dummy_simulator):
        """Test that circuit_wires correctly uses Hamiltonian wire labels."""
        # Create a graph with string node labels
        G = nx.Graph()
        G.add_nodes_from(["x", "y", "z"])
        G.add_edges_from([("x", "y"), ("y", "z")])

        qaoa_problem = QAOA(
            problem=G,
            graph_problem=GraphProblem.MAXCUT,
            n_layers=1,
            optimizer=ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
            max_iterations=1,
            backend=dummy_simulator,
        )

        # Verify circuit_wires uses the Hamiltonian's wire labels (strings)
        assert isinstance(qaoa_problem._circuit_wires, tuple)
        assert len(qaoa_problem._circuit_wires) == G.number_of_nodes()
        assert all(
            wire in G.nodes() for wire in qaoa_problem._circuit_wires
        ), "Circuit wires should match graph node labels"
        assert all(
            isinstance(wire, str) for wire in qaoa_problem._circuit_wires
        ), "Circuit wires should be strings for string-labeled graphs"

    def test_mixed_type_node_labels(self, mocker, dummy_simulator):
        """Test that graphs with mixed type node labels work correctly."""
        # Create a graph with mixed type node labels (integers and strings)
        G = nx.Graph()
        G.add_nodes_from([0, "1", 2, "3"])
        G.add_edges_from([(0, "1"), ("1", 2), (2, "3")])

        qaoa_problem = QAOA(
            problem=G,
            graph_problem=GraphProblem.MAXCUT,
            n_layers=1,
            optimizer=ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
            max_iterations=1,
            backend=dummy_simulator,
        )

        # Verify circuit_wires contains both types
        assert len(qaoa_problem._circuit_wires) == G.number_of_nodes()
        assert all(
            wire in G.nodes() for wire in qaoa_problem._circuit_wires
        ), "Circuit wires should match graph node labels"

        # Simulate measurement results
        qaoa_problem._best_probs = {"0_NoMitigation:0_0": {"1010": 0.5, "0101": 0.5}}

        # Patch _run_solution_measurement
        mocker.patch.object(qaoa_problem, "_run_solution_measurement")

        qaoa_problem._perform_final_computation()

        # Verify solution contains the correct node labels (mixed types)
        assert len(qaoa_problem.solution) == 2, "Bitstring '1010' should map to 2 nodes"
        assert all(
            node in G.nodes() for node in qaoa_problem.solution
        ), "All solution nodes should be valid graph nodes"

    @pytest.mark.e2e
    def test_string_node_labels_e2e(self, default_test_simulator):
        """End-to-end test with string node labels."""
        # Create a simple graph with string labels
        G = nx.Graph()
        G.add_nodes_from(["node0", "node1", "node2"])
        G.add_edges_from([("node0", "node1"), ("node1", "node2")])

        default_test_simulator.set_seed(1997)

        qaoa_problem = QAOA(
            problem=G,
            graph_problem=GraphProblem.MAXCUT,
            n_layers=1,
            optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
            max_iterations=5,
            backend=default_test_simulator,
            seed=1997,
        )

        qaoa_problem.run()

        # Verify bitstring lengths are correct
        assert all(
            len(bitstring) == G.number_of_nodes()
            for probs_dict in qaoa_problem.best_probs.values()
            for bitstring in probs_dict.keys()
        ), "All bitstrings should have length equal to number of nodes"

        # Verify solution contains valid string node labels
        assert all(
            isinstance(node, str) for node in qaoa_problem.solution
        ), "Solution should contain string node labels"
        assert all(
            node in G.nodes() for node in qaoa_problem.solution
        ), "All solution nodes should be valid graph nodes"


QUBO_MATRIX_LIST = [
    [-3.0, 4.0, 0.0],
    [0.0, 2.0, 0.0],
    [0.0, 0.0, -3.0],
]
QUBO_MATRIX_NP = np.array(QUBO_MATRIX_LIST)

QUBO_FORMATS_TO_TEST = {
    "argvalues": [
        QUBO_MATRIX_LIST,
        QUBO_MATRIX_NP,
        sps.csc_matrix(QUBO_MATRIX_NP),
        sps.csr_matrix(QUBO_MATRIX_NP),
        sps.coo_matrix(QUBO_MATRIX_NP),
        sps.lil_matrix(QUBO_MATRIX_NP),
    ],
    "ids": ["List", "Numpy", "CSC", "CSR", "COO", "LIL"],
}


class TestQUBOInput:
    """Test suite for QUBO problem inputs in QAOA."""

    @pytest.mark.parametrize("input_qubo", **QUBO_FORMATS_TO_TEST)
    def test_qubo_basic_initialization(self, input_qubo, default_test_simulator):
        qaoa_problem = QAOA(
            problem=input_qubo,
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
        assert qaoa_problem.graph_problem == None
        assert qaoa_problem.n_layers == 1
        if isinstance(input_qubo, sps.spmatrix):
            np.testing.assert_equal(
                qaoa_problem.problem.toarray(), input_qubo.toarray()
            )
        else:
            np.testing.assert_equal(qaoa_problem.problem, input_qubo)

        assert len(qaoa_problem.cost_hamiltonian) == 4
        assert all(
            isinstance(op, (qml.Z, qml.ops.Prod))
            for op in qaoa_problem.cost_hamiltonian.terms()[1]
        )
        assert len(qaoa_problem.mixer_hamiltonian) == 3
        assert all(
            isinstance(op, qml.X) for op in qaoa_problem.mixer_hamiltonian.terms()[1]
        )

        verify_metacircuit_dict(qaoa_problem, ["cost_circuit", "meas_circuit"])

    def test_redundant_graph_problem_raises_warning(self, dummy_simulator):
        with pytest.warns(
            UserWarning,
            match="Ignoring the 'problem' argument as it is not applicable to QUBO.",
        ):
            QAOA(
                problem=QUBO_MATRIX_LIST,
                graph_problem="max_clique",
                n_layers=1,
                optimizer=ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
                max_iterations=10,
                backend=dummy_simulator,
            )

    def test_non_square_qubo_fails(self, dummy_simulator):
        with pytest.raises(
            ValueError,
            match=r"Invalid QUBO matrix\. Got array of shape \(3, 2\)\. Must be a square matrix\.",
        ):
            QAOA(
                problem=np.array([[1, 2], [3, 4], [5, 6]]),
                n_layers=1,
                optimizer=ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
                max_iterations=10,
                backend=dummy_simulator,
            )

    def test_non_symmetrical_qubo_raises_warning(self, dummy_simulator):
        test_array = np.array([[1, 2], [3, 4]])
        test_array_sp = sps.csc_matrix(test_array)

        with pytest.warns(
            UserWarning,
            match=r"The QUBO matrix is neither symmetric nor upper triangular\. Symmetrizing it for the Ising Hamiltonian creation\.",
        ):
            qaoa_problem = QAOA(
                problem=test_array,
                n_layers=1,
                optimizer=ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
                max_iterations=10,
                backend=dummy_simulator,
            )

        # Ensure the problem matrix was untouched
        np.testing.assert_equal(
            qaoa_problem.problem,
            np.array([[1, 2], [3, 4]]),
        )

        # Test again for sparse matrix
        with pytest.warns(
            UserWarning,
            match=r"The QUBO matrix is neither symmetric nor upper triangular\. Symmetrizing it for the Ising Hamiltonian creation\.",
        ):
            QAOA(
                problem=test_array_sp,
                n_layers=1,
                optimizer=ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
                max_iterations=10,
                backend=dummy_simulator,
            )

    def test_qubo_fails_when_drawing_solution(self, dummy_simulator):
        qaoa_problem = QAOA(
            problem=QUBO_MATRIX_LIST,
            n_layers=1,
            optimizer=ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
            max_iterations=10,
            backend=dummy_simulator,
        )

        with pytest.raises(
            RuntimeError,
            match="The problem is not a graph problem. Cannot draw solution.",
        ):
            qaoa_problem.draw_solution()

    @pytest.mark.e2e
    def test_qubo_returns_correct_solution(self, default_test_simulator):
        default_test_simulator.set_seed(1997)

        qaoa_problem = QAOA(
            problem=QUBO_MATRIX_NP,
            n_layers=1,
            optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
            max_iterations=12,
            backend=default_test_simulator,
            seed=1997,
        )

        qaoa_problem.run()

        np.testing.assert_equal(qaoa_problem.solution, [1, 0, 1])

    @pytest.mark.e2e
    def test_qubo_e2e_checkpointing_resume(self, default_test_simulator, tmp_path):
        """Test QAOA QUBO e2e with checkpointing and resume functionality.

        Tests QUBO problem type handling with checkpointing, not optimizer-specific behavior.
        Full optimizer coverage is tested with Graph problems. Uses MonteCarloOptimizer as representative.
        """
        from divi.qprog import MonteCarloOptimizer

        optimizer = MonteCarloOptimizer(population_size=10, n_best_sets=3)

        checkpoint_dir = tmp_path / "checkpoint_test"
        default_test_simulator.set_seed(1997)

        # Run first half with checkpointing
        qaoa_problem1 = QAOA(
            problem=QUBO_MATRIX_NP,
            n_layers=1,
            optimizer=optimizer,
            max_iterations=6,  # First half
            backend=default_test_simulator,
            seed=1997,
        )

        qaoa_problem1.run(
            checkpoint_config=CheckpointConfig(checkpoint_dir=checkpoint_dir)
        )

        # Verify checkpoint was created
        assert checkpoint_dir.exists()
        checkpoint_path = checkpoint_dir / "checkpoint_006"
        assert checkpoint_path.exists()
        assert (checkpoint_path / "program_state.json").exists()

        # Store state from first run for comparison
        first_run_iteration = qaoa_problem1.current_iteration
        first_run_losses_count = len(qaoa_problem1.losses_history)

        # Load and resume - configuration must be provided by the caller
        qaoa_problem2 = QAOA.load_state(
            checkpoint_dir,
            backend=default_test_simulator,
            problem=QUBO_MATRIX_NP,
            n_layers=1,
        )

        # Verify loaded state matches first run
        assert qaoa_problem2.current_iteration == first_run_iteration
        assert len(qaoa_problem2.losses_history) == first_run_losses_count

        # Continue running to complete the full run
        qaoa_problem2.max_iterations = 12  # Total should be 12
        qaoa_problem2.run()

        # Verify final results are correct
        np.testing.assert_equal(qaoa_problem2.solution, [1, 0, 1])

        # Verify we completed the full run
        assert qaoa_problem2.current_iteration == 12

    @pytest.fixture
    def binary_quadratic_model(self):
        bqm = dimod.BinaryQuadraticModel(
            {"x": -1, "y": -2, "z": 3}, {}, 0.0, dimod.Vartype.BINARY
        )
        return bqm

    @pytest.fixture
    def bqm_minimize(self):
        """BQM for minimization test: x=1, y=-2, z=3, w=-1"""
        return dimod.BinaryQuadraticModel(
            {"x": 1, "y": -2, "z": 3, "w": -1}, {}, 0.0, dimod.Vartype.BINARY
        )

    @pytest.fixture
    def bqm_maximize(self):
        """BQM for maximization test (negated for minimization): x=-1, y=2, z=-3, w=1"""
        return dimod.BinaryQuadraticModel(
            {"x": -1, "y": 2, "z": -3, "w": 1}, {}, 0.0, dimod.Vartype.BINARY
        )

    def test_binary_quadratic_model_initialization(
        self, binary_quadratic_model, default_test_simulator
    ):
        qaoa_problem = QAOA(
            problem=binary_quadratic_model,
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
        assert qaoa_problem.graph_problem == None
        assert qaoa_problem.n_layers == 1

        assert len(qaoa_problem.cost_hamiltonian) == 3
        assert all(
            isinstance(op, qml.Z) for op in qaoa_problem.cost_hamiltonian.terms()[1]
        )
        assert len(qaoa_problem.mixer_hamiltonian) == 3
        assert all(
            isinstance(op, qml.X) for op in qaoa_problem.mixer_hamiltonian.terms()[1]
        )

        verify_metacircuit_dict(qaoa_problem, ["cost_circuit", "meas_circuit"])

    def test_binary_quadratic_model_with_spin_raises_error(self, dummy_simulator):
        # Create a BQM with SPIN vartype (non-binary)
        bqm = dimod.BinaryQuadraticModel(
            {"x": -1, "y": -2, "z": 3}, {}, 0.0, dimod.Vartype.SPIN
        )

        with pytest.raises(
            ValueError,
            match=r"BinaryQuadraticModel must have vartype='BINARY', got Vartype\.SPIN",
        ):
            QAOA(
                problem=bqm,
                n_layers=1,
                optimizer=ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
                max_iterations=10,
                backend=dummy_simulator,
            )

    @pytest.mark.e2e
    def test_binary_quadratic_model_minimize_correct(
        self, bqm_minimize, default_test_simulator
    ):
        default_test_simulator.set_seed(1997)

        qaoa_problem = QAOA(
            problem=bqm_minimize,
            n_layers=2,
            optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
            max_iterations=15,
            backend=default_test_simulator,
            seed=1997,
        )

        qaoa_problem.run()

        # The optimal solution for minimize with x=1, y=-2, z=3, w=-1 is [0, 1, 0, 1]
        # (y=1, w=1 gives energy -2-1=-3)
        expected_solution = [0, 1, 0, 1]
        np.testing.assert_equal(qaoa_problem.solution, expected_solution)

    @pytest.mark.e2e
    def test_binary_quadratic_model_maximize_correct(
        self, bqm_maximize, default_test_simulator
    ):
        # For maximize, we negate the BQM (since QAOA minimizes)
        # Original: maximize x=1, y=-2, z=3, w=-1
        # This is equivalent to minimize: x=-1, y=2, z=-3, w=1
        default_test_simulator.set_seed(1997)

        qaoa_problem = QAOA(
            problem=bqm_maximize,
            n_layers=2,
            optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
            max_iterations=15,
            backend=default_test_simulator,
            seed=1997,
        )

        qaoa_problem.run()

        # For maximize (minimize negated), the optimal solution is [1, 0, 1, 0]
        # (x=1, z=1 gives energy -1-3+1=-3 in negated form, which maximizes original)
        expected_solution = [1, 0, 1, 0]
        np.testing.assert_equal(qaoa_problem.solution, expected_solution)

    @pytest.mark.e2e
    def test_binary_quadratic_model_e2e_checkpointing_resume(
        self, bqm_minimize, default_test_simulator, tmp_path
    ):
        """Test QAOA BinaryQuadraticModel e2e with checkpointing and resume functionality.

        Tests BinaryQuadraticModel problem type handling with checkpointing, not optimizer-specific behavior.
        Full optimizer coverage is tested with Graph problems. Uses MonteCarloOptimizer as representative.
        """
        from divi.qprog import MonteCarloOptimizer

        optimizer = MonteCarloOptimizer(population_size=10, n_best_sets=3)

        checkpoint_dir = tmp_path / "checkpoint_test"
        default_test_simulator.set_seed(1997)

        # Run first half with checkpointing
        qaoa_problem1 = QAOA(
            problem=bqm_minimize,
            n_layers=2,
            optimizer=optimizer,
            max_iterations=7,  # First half
            backend=default_test_simulator,
            seed=1997,
        )

        qaoa_problem1.run(
            checkpoint_config=CheckpointConfig(checkpoint_dir=checkpoint_dir)
        )

        # Verify checkpoint was created
        assert checkpoint_dir.exists()
        checkpoint_path = checkpoint_dir / "checkpoint_007"
        assert checkpoint_path.exists()
        assert (checkpoint_path / "program_state.json").exists()

        # Store state from first run for comparison
        first_run_iteration = qaoa_problem1.current_iteration
        first_run_losses_count = len(qaoa_problem1.losses_history)

        # Load and resume - configuration must be provided by the caller
        qaoa_problem2 = QAOA.load_state(
            checkpoint_dir,
            backend=default_test_simulator,
            problem=bqm_minimize,
            n_layers=2,
        )

        # Verify loaded state matches first run
        assert qaoa_problem2.current_iteration == first_run_iteration
        assert len(qaoa_problem2.losses_history) == first_run_losses_count

        # Continue running to complete the full run
        qaoa_problem2.max_iterations = 15  # Total should be 15
        qaoa_problem2.run()

        # Verify final results are correct
        # The optimal solution for minimize with x=1, y=-2, z=3, w=-1 is [0, 1, 0, 1]
        expected_solution = [0, 1, 0, 1]
        np.testing.assert_equal(qaoa_problem2.solution, expected_solution)

        # Verify we completed the full run
        assert qaoa_problem2.current_iteration == 15


class TestQAOAQDriftMultiSample:
    """Tests for QAOA with multi-sample QDrift (n_hamiltonians_per_iteration > 1)."""

    def test_exact_trotterization_uses_single_hamiltonian_sample(
        self, mocker, default_test_simulator
    ):
        """With ExactTrotterization, TrotterSpecStage.expand produces a single ham sample."""
        strategy = ExactTrotterization(keep_top_n=3)
        qaoa = QAOA(
            problem=nx.bull_graph(),
            graph_problem=GraphProblem.MAXCUT,
            n_layers=1,
            trotterization_strategy=strategy,
            max_iterations=1,
            backend=default_test_simulator,
            optimizer=ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
        )

        trotter_stage = None
        for stage in qaoa._cost_pipeline._stages:
            if isinstance(stage, TrotterSpecStage):
                trotter_stage = stage
                break
        assert trotter_stage is not None

        # Spy on the TrotterSpecStage.expand to inspect how many ham samples are produced
        spy = mocker.spy(trotter_stage, "expand")
        mocker.patch.object(qaoa, "_perform_final_computation")
        qaoa.run()

        # Check that expand produced only 1 ham sample (ham_id=0)
        batch, _token = spy.spy_return
        ham_ids = {
            key[0][1] for key in batch
        }  # Extract ham_id from (("ham", id),) keys
        assert ham_ids == {0}

    def test_multi_sample_generates_circuits_with_hamiltonian_id(
        self, mocker, default_test_simulator
    ):
        """TrotterSpecStage.expand with multi-sample QDrift produces multiple ham IDs."""
        strategy = QDrift(
            keep_fraction=0.3,
            sampling_budget=5,
            n_hamiltonians_per_iteration=3,
            seed=42,
        )
        qaoa = QAOA(
            problem=nx.bull_graph(),
            graph_problem=GraphProblem.MAXCUT,
            n_layers=1,
            trotterization_strategy=strategy,
            max_iterations=1,
            backend=default_test_simulator,
            optimizer=ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
        )

        trotter_stage = None
        for stage in qaoa._cost_pipeline._stages:
            if isinstance(stage, TrotterSpecStage):
                trotter_stage = stage
                break
        assert trotter_stage is not None

        spy = mocker.spy(trotter_stage, "expand")
        mocker.patch.object(qaoa, "_perform_final_computation")
        qaoa.run()

        # Check that expand produced 3 ham samples
        batch, _token = spy.spy_return
        ham_ids = {key[0][1] for key in batch}
        assert ham_ids == {0, 1, 2}

    @pytest.mark.e2e
    def test_multi_sample_qaoa_e2e_solution(self, default_test_simulator):
        """QAOA with multi-sample QDrift runs to completion and finds correct MAXCUT."""
        G = nx.bull_graph()
        default_test_simulator.set_seed(1997)

        strategy = QDrift(
            keep_fraction=0.5,
            sampling_budget=6,
            n_hamiltonians_per_iteration=2,
            seed=123,
        )
        qaoa = QAOA(
            problem=G,
            graph_problem=GraphProblem.MAXCUT,
            n_layers=1,
            trotterization_strategy=strategy,
            max_iterations=20,
            backend=default_test_simulator,
            optimizer=ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
            seed=1997,
        )
        count, runtime = qaoa.run()

        assert count > 0
        assert runtime >= 0
        assert len(qaoa.losses_history) == 20
        assert qaoa.best_loss < float("inf")

        # At least one of the top solutions achieves the optimal cut (more stable than single best)
        max_cut_val, _ = nx.algorithms.approximation.maxcut.one_exchange(G)

        def cut_value(partition1):
            partition0 = set(G.nodes()) - set(partition1)
            return sum(
                1 for u, v in G.edges() if (u in partition0) != (v in partition0)
            )

        top_solutions = qaoa.get_top_solutions(n=5, include_decoded=True)
        optimal_solutions = [
            sol
            for sol in top_solutions
            if sol.decoded is not None and cut_value(sol.decoded) == max_cut_val
        ]
        assert len(optimal_solutions) >= 1

        # Verify nodes in the optimal cut are valid graph nodes
        for sol in optimal_solutions:
            assert all(node in G.nodes() for node in sol.decoded)

    def test_multi_sample_trotter_stage_is_stateful_for_qdrift(
        self, default_test_simulator
    ):
        """TrotterSpecStage is correctly marked stateful for QDrift (ensures cache invalidation)."""
        strategy = QDrift(
            keep_fraction=0.5,
            sampling_budget=4,
            n_hamiltonians_per_iteration=3,
            seed=42,
        )
        qaoa = QAOA(
            problem=nx.bull_graph(),
            graph_problem=GraphProblem.MAXCUT,
            n_layers=1,
            trotterization_strategy=strategy,
            max_iterations=1,
            backend=default_test_simulator,
            optimizer=ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
        )

        trotter_stage = None
        for stage in qaoa._cost_pipeline._stages:
            if isinstance(stage, TrotterSpecStage):
                trotter_stage = stage
                break
        assert trotter_stage is not None
        assert trotter_stage.stateful is True

    def test_multi_sample_final_computation_merges_histograms(
        self, default_test_simulator
    ):
        """Final computation with multi-sample QDrift samples Hamiltonians and merges histograms."""
        strategy = QDrift(
            keep_fraction=0.5,
            sampling_budget=4,
            n_hamiltonians_per_iteration=3,
            seed=456,
        )
        qaoa = QAOA(
            problem=nx.bull_graph(),
            graph_problem=GraphProblem.MAXCUT,
            n_layers=1,
            trotterization_strategy=strategy,
            max_iterations=2,
            backend=default_test_simulator,
            optimizer=ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
        )
        qaoa.run()
        # best_probs should contain merged distribution (one entry per param set)
        assert len(qaoa.best_probs) >= 1
        probs = next(iter(qaoa.best_probs.values()))
        assert isinstance(probs, dict)
        assert all(
            isinstance(k, str) and isinstance(v, (int, float)) for k, v in probs.items()
        )
        assert np.isclose(sum(probs.values()), 1.0)

    @pytest.mark.e2e
    def test_qdrift_zne_with_shot_based_backend(self, default_test_simulator):
        """ZNE works with shot-based backends via per-observable postprocessing."""
        G = nx.bull_graph()
        default_test_simulator.set_seed(1997)

        scale_factors = [1.0, 2.0]
        zne_protocol = ZNE(
            folding_fn=partial(fold_global),
            scale_factors=scale_factors,
            extrapolation_factory=LinearFactory(scale_factors=scale_factors),
        )
        strategy = QDrift(
            keep_fraction=0.5,
            sampling_budget=4,
            n_hamiltonians_per_iteration=2,
            seed=123,
        )
        qaoa = QAOA(
            problem=G,
            graph_problem=GraphProblem.MAXCUT,
            n_layers=1,
            trotterization_strategy=strategy,
            qem_protocol=zne_protocol,
            max_iterations=5,
            backend=default_test_simulator,
            optimizer=ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
            seed=1997,
        )

        count, runtime = qaoa.run()

        assert count > 0
        assert runtime >= 0
        assert len(qaoa.losses_history) == 5
        assert qaoa.best_loss < float("inf")
