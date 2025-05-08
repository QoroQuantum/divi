import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pennylane as qml
import pytest
import scipy.sparse as sps
from flaky import flaky
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qprog_contracts import (
    verify_correct_circuit_count,
    verify_metacircuit_dict,
)

from divi.qprog import QAOA
from divi.qprog.optimizers import Optimizers

pytestmark = pytest.mark.algo


class TestGeneralQAOA:
    @pytest.mark.parametrize("optimizer", list(Optimizers))
    def test_qaoa_generate_circuits_called_with_correct_phases(self, mocker, optimizer):
        qaoa_problem = QAOA(
            problem=nx.bull_graph(),
            graph_problem="max_clique",
            n_layers=1,
            optimizer=optimizer,
            max_iterations=1,
            is_constrained=True,
            qoro_service=None,
        )

        mock_generate_circuits = mocker.patch.object(qaoa_problem, "_generate_circuits")
        mock_pl_postprocessing_fn = mocker.patch.object(
            qaoa_problem._meta_circuits["cost_circuit"], "postprocessing_fn"
        )
        mock_pl_postprocessing_fn.return_value = (np.array(0.0),)

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

        # Verify that the stored iteration count is correct
        assert qaoa_problem.current_iteration == 1

        # Verify that losses is of expected length
        assert len(qaoa_problem.losses) == 1

        # Verify that _generate_circuits was called with _is_compute_probabilies set to False
        assert all(val == False for val in spy_values)

    @pytest.mark.parametrize("optimizer", list(Optimizers))
    def test_graph_correct_circuits_count_and_energies(self, optimizer):
        qaoa_problem = QAOA(
            problem=nx.bull_graph(),
            graph_problem="max_clique",
            n_layers=1,
            optimizer=optimizer,
            max_iterations=1,
            is_constrained=True,
            qoro_service=None,
        )

        verify_correct_circuit_count(qaoa_problem)


class TestGraphInput:
    def test_graph_basic_initialization(self):
        G = nx.bull_graph()

        qaoa_problem = QAOA(
            problem=G,
            graph_problem="max_clique",
            n_layers=2,
            optimizer=Optimizers.NELDER_MEAD,
            max_iterations=10,
            shots=6000,
            is_constrained=True,
            qoro_service=None,
        )

        assert qaoa_problem.shots == 6000
        assert qaoa_problem.qoro_service is None
        assert qaoa_problem.optimizer == Optimizers.NELDER_MEAD
        assert qaoa_problem.max_iterations == 10
        assert qaoa_problem.graph_problem == "max_clique"
        assert qaoa_problem.problem == G
        assert qaoa_problem.n_layers == 2

        verify_metacircuit_dict(qaoa_problem, ["cost_circuit", "meas_circuit"])

    def test_graph_unsuppported_problem(self):
        with pytest.raises(ValueError, match="travelling_salesman"):
            QAOA(
                problem=nx.bull_graph(),
                graph_problem="travelling_salesman",
                qoro_service=None,
            )

    def test_graph_unsuppported_initial_state(self):
        with pytest.raises(ValueError, match="Bell"):
            QAOA(
                problem=nx.bull_graph(),
                graph_problem="max_clique",
                initial_state="Bell",
                qoro_service=None,
            )

    def test_graph_initial_state_recommended(self):
        qaoa_problem = QAOA(
            problem=nx.bull_graph(),
            graph_problem="max_clique",
            initial_state="Recommended",
            is_constrained=True,
            qoro_service=None,
        )

        assert qaoa_problem.initial_state == "Zeros"

    def test_graph_initial_state_superposition(self):
        qaoa_problem = QAOA(
            problem=nx.bull_graph(),
            graph_problem="max_clique",
            initial_state="Superposition",
            qoro_service=None,
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

    @flaky(max_runs=3, min_passes=1)
    @pytest.mark.parametrize("optimizer", list(Optimizers))
    def test_graph_compute_final_solution(self, mocker, optimizer):
        G = nx.bull_graph()

        if optimizer == Optimizers.MONTE_CARLO:
            # Use smaller number of samples for faster testing
            mocker.patch.object(
                Optimizers.MONTE_CARLO.__class__,
                "n_samples",
                new_callable=mocker.PropertyMock,
                return_value=3,
            )

        qaoa_problem = QAOA(
            graph_problem="max_clique",
            problem=G,
            n_layers=1,
            optimizer=optimizer,
            max_iterations=8 if optimizer != Optimizers.MONTE_CARLO else 2,
            is_constrained=True,
            qoro_service=None,
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
            graph_problem="max_clique",
            problem=G,
            n_layers=1,
            optimizer=Optimizers.NELDER_MEAD,
            max_iterations=2,
            is_constrained=True,
            qoro_service=None,
        )

        qaoa_problem._solution_nodes = [0, 1, 2]

        # Mock networkx draw functions to capture their arguments
        mock_draw_nodes = mocker.patch("divi.qprog._qaoa.nx.draw_networkx_nodes")
        mock_draw_edges = mocker.patch("divi.qprog._qaoa.nx.draw_networkx_edges")
        mock_draw_labels = mocker.patch("divi.qprog._qaoa.nx.draw_networkx_labels")
        mocker.patch("matplotlib.pyplot.show")

        qaoa_problem.draw_solution()

        # Verify that all drawing functions were called
        mock_draw_nodes.assert_called_once()
        mock_draw_edges.assert_called_once()
        mock_draw_labels.assert_called_once()

        # Get the node_color argument that was passed to draw_networkx_nodes
        node_colors = mock_draw_nodes.call_args[1]["node_color"]

        # Verify that solution nodes are red and non-solution nodes are lightblue
        expected_colors = [
            "red" if node in qaoa_problem._solution_nodes else "lightblue"
            for node in G.nodes()
        ]
        assert node_colors == expected_colors

        # Verify node size
        assert mock_draw_nodes.call_args[1]["node_size"] == 500

        # Clean up the plot
        plt.close()


qubo_matrix_list = [
    [-1, 0, -4.5],
    [0, 1, 0],
    [-4.5, 0, -1],
]

qubo_matrix_np = np.array(qubo_matrix_list)
qubo_matrix_sp = sps.csc_matrix(qubo_matrix_np)


class TestQUBOInput:

    @pytest.mark.parametrize(
        "input_qubo", [qubo_matrix_list, qubo_matrix_np, qubo_matrix_sp]
    )
    def test_qubo_basic_initialization(self, input_qubo):
        qaoa_problem = QAOA(
            problem=input_qubo,
            n_layers=2,
            optimizer=Optimizers.NELDER_MEAD,
            max_iterations=10,
            shots=6000,
            qoro_service=None,
        )

        assert qaoa_problem.shots == 6000
        assert qaoa_problem.qoro_service is None
        assert qaoa_problem.optimizer == Optimizers.NELDER_MEAD
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
                problem=qubo_matrix_list,
                graph_problem="max_clique",
                n_layers=2,
                optimizer=Optimizers.NELDER_MEAD,
                max_iterations=10,
                qoro_service=None,
            )

    def test_non_square_qubo_fails(self):
        with pytest.raises(
            ValueError,
            match=r"Invalid QUBO matrix\. Got array of shape \(3, 2\)\. Must be a square matrix\.",
        ):
            QAOA(
                problem=np.array([[1, 2], [3, 4], [5, 6]]),
                n_layers=2,
                optimizer=Optimizers.NELDER_MEAD,
                max_iterations=10,
                qoro_service=None,
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
                optimizer=Optimizers.NELDER_MEAD,
                max_iterations=10,
                qoro_service=None,
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
                optimizer=Optimizers.NELDER_MEAD,
                max_iterations=10,
                qoro_service=None,
            )

    def test_qubo_fails_when_drawing_solution(self):
        qaoa_problem = QAOA(
            problem=qubo_matrix_list,
            n_layers=2,
            optimizer=Optimizers.NELDER_MEAD,
            max_iterations=10,
            qoro_service=None,
        )

        with pytest.raises(
            RuntimeError,
            match="The problem is not a graph problem. Cannot draw solution.",
        ):
            qaoa_problem.draw_solution()

    @flaky(max_runs=3, min_passes=1)
    def test_qubo_returns_correct_solution(self):
        qaoa_problem = QAOA(
            problem=qubo_matrix_np,
            n_layers=1,
            optimizer=Optimizers.NELDER_MEAD,
            max_iterations=10,
            qoro_service=None,
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

    def test_quadratic_program_initialization(self, quadratic_program):
        qaoa_problem = QAOA(
            problem=quadratic_program,
            n_layers=2,
            optimizer=Optimizers.NELDER_MEAD,
            max_iterations=10,
            shots=6000,
            qoro_service=None,
        )

        assert qaoa_problem.shots == 6000
        assert qaoa_problem.qoro_service is None
        assert qaoa_problem.optimizer == Optimizers.NELDER_MEAD
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
                optimizer=Optimizers.NELDER_MEAD,
                max_iterations=10,
                shots=10000,
                qoro_service=None,
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

    @flaky(max_runs=3, min_passes=1)
    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_quadratic_program_minimize_correct(self, quadratic_program):
        quadratic_program.integer_var(lowerbound=0, upperbound=3, name="w")
        quadratic_program.minimize(linear={"x": -1, "y": -2, "z": 3, "w": -1})

        qaoa_problem = QAOA(
            problem=quadratic_program,
            n_layers=2,
            optimizer=Optimizers.NELDER_MEAD,
            max_iterations=20,
            shots=10000,
            qoro_service=None,
        )

        qaoa_problem.run()
        qaoa_problem.compute_final_solution()

        np.testing.assert_equal(
            qaoa_problem._qp_converter.interpret(qaoa_problem.solution), [1, 1, 0, 3]
        )

    @flaky(max_runs=3, min_passes=1)
    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_quadratic_program_maximize_correct(self, quadratic_program):
        quadratic_program.integer_var(lowerbound=0, upperbound=3, name="w")
        quadratic_program.maximize(linear={"x": -1, "y": -2, "z": 3, "w": -1})

        qaoa_problem = QAOA(
            problem=quadratic_program,
            n_layers=1,
            optimizer=Optimizers.NELDER_MEAD,
            max_iterations=20,
            shots=6000,
            qoro_service=None,
        )

        qaoa_problem.run()
        qaoa_problem.compute_final_solution()

        np.testing.assert_equal(
            qaoa_problem._qp_converter.interpret(qaoa_problem.solution), [0, 0, 1, 0]
        )
