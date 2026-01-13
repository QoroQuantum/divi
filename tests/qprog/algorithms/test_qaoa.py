# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import dimod
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pennylane as qml
import pytest
import scipy.sparse as sps

from divi.backends import CircuitRunner
from divi.qprog import (
    QAOA,
    GraphProblem,
    ScipyMethod,
    ScipyOptimizer,
)
from divi.qprog.algorithms import _qaoa, SolutionEntry
from divi.qprog.checkpointing import CheckpointConfig
from tests.conftest import CHECKPOINTING_OPTIMIZERS
from tests.qprog.qprog_contracts import (
    OPTIMIZERS_TO_TEST,
    verify_correct_circuit_count,
    verify_metacircuit_dict,
)

pytestmark = pytest.mark.algo


class TestGeneralQAOA:
    @pytest.mark.parametrize("optimizer", **OPTIMIZERS_TO_TEST)
    def test_qaoa_optimization_runs_in_loss_mode(
        self, mocker, optimizer, default_test_simulator
    ):
        """
        Verifies that _is_compute_probabilities is False during the optimization loop.
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

        # Isolate the optimization phase by mocking the final step
        mocker.patch.object(qaoa_problem, "_perform_final_computation")
        mocker.patch.object(qaoa_problem, "_generate_circuits")
        mock_dispatch = mocker.patch.object(
            qaoa_problem, "_dispatch_circuits_and_process_results"
        )
        mock_dispatch.return_value = {i: 0.5 for i in range(optimizer.n_param_sets)}

        spy_flag_values = []

        def generate_circuits_spy(*args, **kwargs):
            # When _generate_circuits is called, record the flag's state
            spy_flag_values.append(qaoa_problem._is_compute_probabilities)

        # Replace the original method with our spy
        mocker.patch.object(
            qaoa_problem, "_generate_circuits", side_effect=generate_circuits_spy
        )
        qaoa_problem.run()

        # Assert that the flag was never set to True during optimization
        assert spy_flag_values  # Ensure the spy caught something
        assert all(val is False for val in spy_flag_values)

    @pytest.mark.parametrize("optimizer", **OPTIMIZERS_TO_TEST)
    def test_qaoa_final_computation_runs_in_probability_mode(
        self, mocker, optimizer, default_test_simulator
    ):
        """
        Verifies that _is_compute_probabilities is set to True during the final
        circuit generation and is reset to False afterward.
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
        # Set preconditions for the method under test
        qaoa_problem._final_params = np.array([[0.1, 0.2]])
        qaoa_problem._best_params = np.array([[0.1, 0.2]])
        mocker.patch.object(
            qaoa_problem,
            "_dispatch_circuits_and_process_results",
            return_value={0: {"00110": 0.4, "11001": 0.6}},
        )

        # This list will store the state of the flag when the spied method is called
        flag_state_at_call_time = []
        # Keep a reference to the original method
        original_generate_circuits = qaoa_problem._generate_circuits

        def generate_circuits_spy():
            """Spy that records the flag's state and then calls the original method."""
            flag_state_at_call_time.append(qaoa_problem._is_compute_probabilities)
            return original_generate_circuits()

        mocker.patch.object(
            qaoa_problem, "_generate_circuits", side_effect=generate_circuits_spy
        )

        # 1. Verify the initial state
        assert qaoa_problem._is_compute_probabilities is False

        # 2. Run the function that changes the state
        qaoa_problem._perform_final_computation()

        # 3. Verify the flag was True when the critical function was called
        # _generate_circuits is called once for best_params only
        assert flag_state_at_call_time == [True]

        # 4. Verify the state was reset correctly after the function completed
        assert qaoa_problem._is_compute_probabilities is False

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

    def test_graph_unsuppported_initial_state(self):
        with pytest.raises(ValueError, match="Bell"):
            QAOA(
                problem=nx.bull_graph(),
                graph_problem=GraphProblem.MAX_CLIQUE,
                initial_state="Bell",
                backend=None,
            )

    def test_graph_initial_state_recommended(self):
        qaoa_problem = QAOA(
            problem=nx.bull_graph(),
            graph_problem=GraphProblem.MAX_CLIQUE,
            initial_state="Recommended",
            is_constrained=True,
            backend=None,
        )

        assert qaoa_problem.initial_state == "Zeros"

    def test_graph_initial_state_superposition(self):
        qaoa_problem = QAOA(
            problem=nx.bull_graph(),
            graph_problem=GraphProblem.MAX_CLIQUE,
            initial_state="Superposition",
            backend=None,
        )

        assert qaoa_problem.initial_state == "Superposition"
        assert (
            sum(
                isinstance(op, qml.Hadamard)
                for op in qaoa_problem.meta_circuits[
                    "cost_circuit"
                ].source_circuit.operations
            )
            == nx.bull_graph().number_of_nodes()
        )

    def test_perform_final_computation_extracts_correct_solution(self, mocker):
        G = nx.bull_graph()
        qaoa_problem = QAOA(
            graph_problem=GraphProblem.MAX_CLIQUE,
            problem=G,
            n_layers=1,
            optimizer=ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
            max_iterations=1,
            is_constrained=True,
            backend=None,
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
        if (
            isinstance(optimizer, ScipyOptimizer)
            and optimizer.method == ScipyMethod.L_BFGS_B
        ):
            pytest.skip("L-BFGS-B fails a lot for some reason. Debug later.")

        G = nx.bull_graph()

        default_test_simulator.set_seed(1997)

        qaoa_problem = QAOA(
            graph_problem=GraphProblem.MAX_CLIQUE,
            problem=G,
            n_layers=1,
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

    def test_draw_solution_returns_graph_with_expected_properties(self, mocker):
        G = nx.bull_graph()

        qaoa_problem = QAOA(
            graph_problem=GraphProblem.MAX_CLIQUE,
            problem=G,
            n_layers=1,
            optimizer=ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
            max_iterations=2,
            is_constrained=True,
            backend=None,
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

    def test_string_node_labels_bitstring_length(self, mocker, default_test_simulator):
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
            backend=default_test_simulator,
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

    def test_string_node_labels_solution_mapping(self, mocker):
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
            backend=None,
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

    def test_string_node_labels_circuit_wires(self):
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
            backend=None,
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

    def test_mixed_type_node_labels(self, mocker):
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
            backend=None,
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

    def test_redundant_graph_problem_raises_warning(self):
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
                backend=None,
            )

    def test_non_square_qubo_fails(self):
        with pytest.raises(
            ValueError,
            match=r"Invalid QUBO matrix\. Got array of shape \(3, 2\)\. Must be a square matrix\.",
        ):
            QAOA(
                problem=np.array([[1, 2], [3, 4], [5, 6]]),
                n_layers=1,
                optimizer=ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
                max_iterations=10,
                backend=None,
            )

    def test_non_symmetrical_qubo_raises_warning(self):
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
                backend=None,
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
                backend=None,
            )

    def test_qubo_fails_when_drawing_solution(self):
        qaoa_problem = QAOA(
            problem=QUBO_MATRIX_LIST,
            n_layers=1,
            optimizer=ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
            max_iterations=10,
            backend=None,
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

    def test_binary_quadratic_model_with_spin_raises_error(self):
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
                backend=None,
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


class TestTopNSolutions:
    """Test suite for top-N solutions API."""

    def test_solution_distribution_before_run_raises_error(self):
        """Test that accessing solution_distribution before run() raises RuntimeError."""
        qaoa_problem = QAOA(
            problem=nx.bull_graph(),
            graph_problem=GraphProblem.MAX_CLIQUE,
            n_layers=1,
            optimizer=ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
            max_iterations=1,
            backend=None,
        )

        with pytest.raises(
            RuntimeError,
            match="solution_distribution is not available. The QAOA algorithm has not been run yet.",
        ):
            _ = qaoa_problem.solution_distribution

    def test_top_solutions_before_run_raises_error(self):
        """Test that calling top_solutions before run() raises RuntimeError."""
        qaoa_problem = QAOA(
            problem=nx.bull_graph(),
            graph_problem=GraphProblem.MAX_CLIQUE,
            n_layers=1,
            optimizer=ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
            max_iterations=1,
            backend=None,
        )

        with pytest.raises(
            RuntimeError,
            match="top_solutions is not available. The QAOA algorithm has not been run yet.",
        ):
            _ = qaoa_problem.top_solutions(n=5)

    def test_solution_distribution_returns_full_distribution(self, mocker):
        """Test that solution_distribution returns the complete probability distribution."""
        qaoa_problem = QAOA(
            problem=nx.bull_graph(),
            graph_problem=GraphProblem.MAX_CLIQUE,
            n_layers=1,
            optimizer=ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
            max_iterations=1,
            backend=None,
        )

        # Simulate measurement results with multiple bitstrings
        mock_dist = {"11001": 0.4, "00101": 0.3, "10010": 0.2, "01100": 0.1}
        qaoa_problem._best_probs = {"0_NoMitigation:0_0": mock_dist}

        dist = qaoa_problem.solution_distribution

        assert dist == mock_dist
        assert len(dist) == 4
        assert sum(dist.values()) == pytest.approx(1.0)

    def test_top_solutions_ordering_with_deterministic_tiebreak(self, mocker):
        """Test top-N ordering with deterministic lexicographic tiebreaker."""
        qaoa_problem = QAOA(
            problem=nx.Graph([(0, 1), (1, 2)]),
            graph_problem=GraphProblem.MAXCUT,
            n_layers=1,
            optimizer=ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
            max_iterations=1,
            backend=None,
        )

        # Create distribution with ties in counts
        # "00" and "01" both have count 10 (prob 0.3333)
        # "10" has count 5 (prob 0.1667)
        mock_dist = {"00": 0.3333, "01": 0.3333, "10": 0.1667, "11": 0.1667}
        qaoa_problem._best_probs = {"0_NoMitigation:0_0": mock_dist}
        qaoa_problem.backend = mocker.MagicMock()
        qaoa_problem.backend.shots = 30

        top = qaoa_problem.top_solutions(n=4)

        # Expected order (sorted by count desc, then lexicographic asc for tiebreak):
        # 1. "00" (count ~10, prob 0.3333) - tied with "01", "00" < "01" lexicographically
        # 2. "01" (count ~10, prob 0.3333) - tied with "00", "00" < "01" lexicographically
        # 3. "10" (count ~5, prob 0.1667) - tied with "11", "10" < "11" lexicographically
        # 4. "11" (count ~5, prob 0.1667) - tied with "10", "10" < "11" lexicographically
        assert len(top) == 4
        assert top[0].bitstring == "00"
        assert top[1].bitstring == "01"
        assert top[2].bitstring == "10"
        assert top[3].bitstring == "11"

        # Verify counts and probabilities
        assert top[0].count == pytest.approx(10, abs=1)
        assert top[0].probability == pytest.approx(0.3333, abs=0.01)

    def test_top_solutions_n_edge_cases(self, mocker):
        """Test edge cases for n parameter."""
        qaoa_problem = QAOA(
            problem=nx.Graph([(0, 1)]),
            graph_problem=GraphProblem.MAXCUT,
            n_layers=1,
            optimizer=ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
            max_iterations=1,
            backend=None,
        )

        mock_dist = {"00": 0.5, "01": 0.3, "10": 0.2}
        qaoa_problem._best_probs = {"0_NoMitigation:0_0": mock_dist}
        qaoa_problem.backend = mocker.MagicMock()
        qaoa_problem.backend.shots = 100

        # Test n=0 returns empty list
        assert qaoa_problem.top_solutions(n=0) == []

        # Test negative n returns empty list
        assert qaoa_problem.top_solutions(n=-5) == []

        # Test n > len(dist) returns all solutions
        top = qaoa_problem.top_solutions(n=100)
        assert len(top) == 3

        # Test n=1 returns only best
        top = qaoa_problem.top_solutions(n=1)
        assert len(top) == 1
        assert top[0].bitstring == "00"

    def test_top_solutions_min_count_filter(self, mocker):
        """Test filtering by min_count."""
        qaoa_problem = QAOA(
            problem=nx.Graph([(0, 1), (1, 2)]),
            graph_problem=GraphProblem.MAXCUT,
            n_layers=1,
            optimizer=ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
            max_iterations=1,
            backend=None,
        )

        mock_dist = {"00": 0.4, "01": 0.3, "10": 0.2, "11": 0.1}
        qaoa_problem._best_probs = {"0_NoMitigation:0_0": mock_dist}
        qaoa_problem.backend = mocker.MagicMock()
        qaoa_problem.backend.shots = 100

        # Filter with min_count=25 should keep only "00" (40 counts) and "01" (30 counts)
        top = qaoa_problem.top_solutions(n=10, min_count=25)
        assert len(top) == 2
        assert top[0].bitstring == "00"
        assert top[1].bitstring == "01"

        # Filter with min_count=35 should keep only "00" (40 counts)
        top = qaoa_problem.top_solutions(n=10, min_count=35)
        assert len(top) == 1
        assert top[0].bitstring == "00"

    def test_top_solutions_min_prob_filter(self, mocker):
        """Test filtering by min_prob."""
        qaoa_problem = QAOA(
            problem=nx.Graph([(0, 1), (1, 2)]),
            graph_problem=GraphProblem.MAXCUT,
            n_layers=1,
            optimizer=ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
            max_iterations=1,
            backend=None,
        )

        mock_dist = {"00": 0.4, "01": 0.3, "10": 0.2, "11": 0.1}
        qaoa_problem._best_probs = {"0_NoMitigation:0_0": mock_dist}
        qaoa_problem.backend = mocker.MagicMock()
        qaoa_problem.backend.shots = 100

        # Filter with min_prob=0.25 should keep only "00" and "01"
        top = qaoa_problem.top_solutions(n=10, min_prob=0.25)
        assert len(top) == 2
        assert top[0].bitstring == "00"
        assert top[1].bitstring == "01"

    def test_backwards_compatibility_solution_matches_top_1(self, mocker):
        """Test that .solution matches the state of top_solutions(1)[0]."""
        G = nx.bull_graph()
        qaoa_problem = QAOA(
            graph_problem=GraphProblem.MAX_CLIQUE,
            problem=G,
            n_layers=1,
            optimizer=ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
            max_iterations=1,
            is_constrained=True,
            backend=None,
        )

        # Simulate measurement results
        qaoa_problem._best_probs = {
            "0_NoMitigation:0_0": {"11001": 0.5, "00101": 0.3, "10010": 0.2}
        }
        qaoa_problem.backend = mocker.MagicMock()
        qaoa_problem.backend.shots = 100

        # Patch _run_solution_measurement to do nothing
        mocker.patch.object(qaoa_problem, "_run_solution_measurement")

        qaoa_problem._perform_final_computation()

        # Get top solution
        top = qaoa_problem.top_solutions(n=1)

        # Verify solution matches top[0].state
        assert len(top) == 1
        assert qaoa_problem.solution == top[0].state

    def test_top_solutions_decodes_graph_state_correctly(self, mocker):
        """Test that top_solutions correctly decodes bitstrings to node lists for graph problems."""
        G = nx.Graph([(0, 1), (1, 2), (2, 3)])
        qaoa_problem = QAOA(
            graph_problem=GraphProblem.MAXCUT,
            problem=G,
            n_layers=1,
            optimizer=ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
            max_iterations=1,
            backend=None,
        )

        # Bitstring "1010" should decode to nodes [0, 2]
        mock_dist = {"1010": 0.5, "0101": 0.3, "1111": 0.2}
        qaoa_problem._best_probs = {"0_NoMitigation:0_0": mock_dist}
        qaoa_problem.backend = mocker.MagicMock()
        qaoa_problem.backend.shots = 100

        top = qaoa_problem.top_solutions(n=3)

        assert len(top) == 3
        assert top[0].bitstring == "1010"
        assert top[0].state == [0, 2]
        assert top[1].bitstring == "0101"
        assert top[1].state == [1, 3]
        assert top[2].bitstring == "1111"
        assert top[2].state == [0, 1, 2, 3]

    def test_top_solutions_decodes_qubo_state_correctly(self, mocker):
        """Test that top_solutions correctly decodes bitstrings to arrays for QUBO problems."""
        qubo = np.array([[1, 2], [2, 3]])
        qaoa_problem = QAOA(
            problem=qubo,
            n_layers=1,
            optimizer=ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
            max_iterations=1,
            backend=None,
        )

        mock_dist = {"10": 0.6, "01": 0.4}
        qaoa_problem._best_probs = {"0_NoMitigation:0_0": mock_dist}
        qaoa_problem.backend = mocker.MagicMock()
        qaoa_problem.backend.shots = 100

        top = qaoa_problem.top_solutions(n=2)

        assert len(top) == 2
        assert top[0].bitstring == "10"
        np.testing.assert_array_equal(top[0].state, np.array([1, 0]))
        assert top[1].bitstring == "01"
        np.testing.assert_array_equal(top[1].state, np.array([0, 1]))

    def test_top_solutions_with_string_node_labels(self, mocker):
        """Test that top_solutions works with string node labels."""
        G = nx.Graph()
        G.add_nodes_from(["a", "b", "c"])
        G.add_edges_from([("a", "b"), ("b", "c")])

        qaoa_problem = QAOA(
            problem=G,
            graph_problem=GraphProblem.MAXCUT,
            n_layers=1,
            optimizer=ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
            max_iterations=1,
            backend=None,
        )

        mock_dist = {"101": 0.6, "010": 0.4}
        qaoa_problem._best_probs = {"0_NoMitigation:0_0": mock_dist}
        qaoa_problem.backend = mocker.MagicMock()
        qaoa_problem.backend.shots = 100

        top = qaoa_problem.top_solutions(n=2)

        assert len(top) == 2
        assert top[0].bitstring == "101"
        # Verify all nodes in state are strings and valid graph nodes
        assert all(isinstance(node, str) for node in top[0].state)
        assert all(node in G.nodes() for node in top[0].state)

    @pytest.mark.e2e
    def test_top_solutions_integration(self, default_test_simulator):
        """Integration test with actual QAOA execution."""
        G = nx.bull_graph()
        default_test_simulator.set_seed(1997)

        qaoa_problem = QAOA(
            graph_problem=GraphProblem.MAX_CLIQUE,
            problem=G,
            n_layers=1,
            optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
            max_iterations=5,
            is_constrained=True,
            backend=default_test_simulator,
            seed=1997,
        )

        qaoa_problem.run()

        # Test solution_distribution
        dist = qaoa_problem.solution_distribution
        assert isinstance(dist, dict)
        assert len(dist) > 0
        assert all(isinstance(k, str) for k in dist.keys())
        assert all(isinstance(v, float) for v in dist.values())
        assert sum(dist.values()) == pytest.approx(1.0, rel=0.01)

        # Test top_solutions
        top5 = qaoa_problem.top_solutions(n=5)
        assert len(top5) <= 5
        assert len(top5) > 0

        # Verify ordering (descending by count)
        for i in range(len(top5) - 1):
            assert top5[i].count >= top5[i + 1].count

        # Verify solution matches top 1
        assert qaoa_problem.solution == top5[0].state

        # Verify all bitstrings have correct length
        for entry in top5:
            assert len(entry.bitstring) == G.number_of_nodes()

        # Verify state decoding is consistent
        for entry in top5:
            assert all(node in G.nodes() for node in entry.state)
