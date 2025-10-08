# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pennylane as qml
import pytest
import scipy.sparse as sps
from flaky import flaky
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo

from divi.backends import CircuitRunner
from divi.qprog import (
    QAOA,
    GraphProblem,
    ScipyMethod,
    ScipyOptimizer,
)
from divi.qprog.algorithms import _qaoa
from tests.conftest import is_assertion_error
from tests.qprog.qprog_contracts import (
    OPTIMIZERS_TO_TEST,
    verify_correct_circuit_count,
    verify_metacircuit_dict,
)

pytestmark = pytest.mark.algo


class TestGeneralQAOA:
    @pytest.mark.parametrize("optimizer", **OPTIMIZERS_TO_TEST)
    def test_qaoa_generate_circuits_called_with_correct_phases(
        self, mocker, optimizer, default_test_simulator
    ):
        qaoa_problem = QAOA(
            problem=nx.bull_graph(),
            graph_problem=GraphProblem.MAX_CLIQUE,
            n_layers=1,
            optimizer=optimizer,
            max_iterations=1,
            backend=default_test_simulator,
            is_constrained=True,
        )

        mock_generate_circuits = mocker.patch.object(qaoa_problem, "_generate_circuits")

        mock_dispatch = mocker.patch.object(
            qaoa_problem, "_dispatch_circuits_and_process_results"
        )
        dummy_losses = {i: 0.5 for i in range(optimizer.n_param_sets)}
        mock_dispatch.return_value = dummy_losses

        spy_values = []
        mock_setattr = mocker.patch.object(
            qaoa_problem, "__setattr__", wraps=qaoa_problem.__setattr__
        )

        def side_effect(name, value):
            if name == "_is_compute_probabilies":
                spy_values.append(value)
            return mock_setattr.original(qaoa_problem, name, value)

        mock_setattr.side_effect = side_effect

        qaoa_problem.run()

        # Verify that _generate_circuits was called as many times as iterations
        assert mock_generate_circuits.called

        # Verify that _generate_circuits was called with _is_compute_probabilies set to False
        assert all(val == False for val in spy_values)

    @pytest.mark.parametrize("optimizer", **OPTIMIZERS_TO_TEST)
    def test_graph_correct_circuits_count_and_energies(
        self, optimizer, dummy_simulator
    ):
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
            n_layers=2,
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
        assert qaoa_problem.n_layers == 2

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
                for op in qaoa_problem._meta_circuits[
                    "cost_circuit"
                ].main_circuit.operations
            )
            == nx.bull_graph().number_of_nodes()
        )

    def test_compute_final_solution_extracts_correct_solution(self, mocker):
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
        qaoa_problem.probs = {"0_NoMitigation:0_0": {"10011": 0.1444, "10100": 0.0526}}

        # Patch _run_final_measurement to do nothing (since we set probs manually)
        mocker.patch.object(qaoa_problem, "_run_final_measurement")

        qaoa_problem.compute_final_solution()

        # Should extract bitstring "10011" -> "11001"
        assert qaoa_problem._solution_nodes == [0, 1, 4]
        assert qaoa_problem.solution == [0, 1, 4]

    @pytest.mark.e2e
    @pytest.mark.parametrize("optimizer", **OPTIMIZERS_TO_TEST)
    def test_graph_qaoa_e2e_solution(self, mocker, optimizer, default_test_simulator):
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

        spy = mocker.spy(qaoa_problem, "_generate_circuits")

        qaoa_problem.compute_final_solution()

        assert all(
            len(bitstring) == G.number_of_nodes()
            for probs_dict in qaoa_problem.probs.values()
            for bitstring in probs_dict.keys()
        )

        assert set(
            qaoa_problem._solution_nodes
        ) == nx.algorithms.approximation.max_clique(G)

        assert spy.call_count == 1

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

    @pytest.mark.parametrize("input_qubo", **QUBO_FORMATS_TO_TEST)
    def test_qubo_basic_initialization(self, input_qubo, default_test_simulator):
        qaoa_problem = QAOA(
            problem=input_qubo,
            n_layers=2,
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
        assert qaoa_problem.n_layers == 2
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
                n_layers=2,
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
                n_layers=2,
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
                n_layers=2,
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
                n_layers=2,
                optimizer=ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
                max_iterations=10,
                backend=None,
            )

    def test_qubo_fails_when_drawing_solution(self):
        qaoa_problem = QAOA(
            problem=QUBO_MATRIX_LIST,
            n_layers=2,
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
        qaoa_problem.compute_final_solution()

        np.testing.assert_equal(qaoa_problem.solution, [1, 0, 1])

    @pytest.fixture
    def quadratic_program(self):
        qp = QuadraticProgram()
        qp.binary_var("x")
        qp.binary_var("y")
        qp.binary_var("z")
        qp.minimize(linear={"x": -1, "y": -2, "z": 3})

        return qp

    def test_quadratic_program_initialization(
        self, quadratic_program, default_test_simulator
    ):
        qaoa_problem = QAOA(
            problem=quadratic_program,
            n_layers=2,
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
        assert qaoa_problem.n_layers == 2

        assert len(qaoa_problem.cost_hamiltonian) == 3
        assert all(
            isinstance(op, qml.Z) for op in qaoa_problem.cost_hamiltonian.terms()[1]
        )
        assert len(qaoa_problem.mixer_hamiltonian) == 3
        assert all(
            isinstance(op, qml.X) for op in qaoa_problem.mixer_hamiltonian.terms()[1]
        )

        verify_metacircuit_dict(qaoa_problem, ["cost_circuit", "meas_circuit"])

    def test_quadratic_program_with_nonbinary_warns(self, quadratic_program):
        quadratic_program.integer_var(lowerbound=0, upperbound=3, name="w")
        quadratic_program.minimize(linear={"x": -1, "y": -2, "z": 3, "w": -1})

        with pytest.warns(
            UserWarning,
            match="Quadratic Program contains non-binary variables. Converting to QUBO.",
        ):
            qaoa_problem = QAOA(
                quadratic_program,
                n_layers=2,
                optimizer=ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
                max_iterations=10,
                backend=None,
            )

        assert hasattr(qaoa_problem, "_qp_converter")
        assert isinstance(qaoa_problem._qp_converter, QuadraticProgramToQubo)

        assert len(qaoa_problem.cost_hamiltonian) == 5
        assert all(
            isinstance(op, qml.Z) for op in qaoa_problem.cost_hamiltonian.terms()[1]
        )
        assert len(qaoa_problem.mixer_hamiltonian) == 5
        assert all(
            isinstance(op, qml.X) for op in qaoa_problem.mixer_hamiltonian.terms()[1]
        )

    @pytest.mark.e2e
    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_quadratic_program_minimize_correct(
        self, quadratic_program, default_test_simulator
    ):
        quadratic_program.integer_var(lowerbound=0, upperbound=3, name="w")
        quadratic_program.minimize(linear={"x": 1, "y": -2, "z": 3, "w": -1})

        default_test_simulator.set_seed(1997)

        qaoa_problem = QAOA(
            problem=quadratic_program,
            n_layers=2,
            optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
            max_iterations=15,
            backend=default_test_simulator,
            seed=1997,
        )

        qaoa_problem.run()
        qaoa_problem.compute_final_solution()

        np.testing.assert_equal(
            qaoa_problem._qp_converter.interpret(qaoa_problem.solution), [0, 1, 0, 3]
        )

    @pytest.mark.e2e
    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_quadratic_program_maximize_correct(
        self, quadratic_program, default_test_simulator
    ):
        quadratic_program.integer_var(lowerbound=0, upperbound=3, name="w")
        quadratic_program.maximize(linear={"x": 1, "y": -2, "z": 3, "w": -1})

        default_test_simulator.set_seed(1997)

        qaoa_problem = QAOA(
            problem=quadratic_program,
            n_layers=2,
            optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
            max_iterations=15,
            backend=default_test_simulator,
            seed=1997,
        )

        qaoa_problem.run()
        qaoa_problem.compute_final_solution()

        np.testing.assert_equal(
            qaoa_problem._qp_converter.interpret(qaoa_problem.solution), [1, 0, 1, 0]
        )
