import matplotlib.pyplot as plt
import networkx as nx
import pennylane as qml
import pytest
from flaky import flaky
from qprog_contracts import (
    verify_correct_circuit_count,
    verify_hamiltonian_metadata,
    verify_metacircuit_dict,
)

from divi.qprog import QAOA
from divi.qprog.optimizers import Optimizers

pytestmark = pytest.mark.algo


def test_qaoa_basic_initialization():
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

    verify_hamiltonian_metadata(qaoa_problem)

    verify_metacircuit_dict(qaoa_problem, ["cost_circuit", "meas_circuit"])


def test_qaoa_unsuppported_problem():
    with pytest.raises(ValueError, match="travelling_salesman"):
        QAOA(
            problem=nx.bull_graph(),
            graph_problem="travelling_salesman",
            qoro_service=None,
        )


def test_qaoa_unsuppported_initial_state():
    with pytest.raises(ValueError, match="Bell"):
        QAOA(
            problem=nx.bull_graph(),
            graph_problem="max_clique",
            initial_state="Bell",
            qoro_service=None,
        )


def test_qaoa_initial_state_recommended():
    qaoa_problem = QAOA(
        problem=nx.bull_graph(),
        graph_problem="max_clique",
        initial_state="Recommended",
        is_constrained=True,
        qoro_service=None,
    )

    assert qaoa_problem.initial_state == "Zeros"


def test_qaoa_initial_state_superposition():
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


@pytest.mark.parametrize("optimizer", list(Optimizers))
def test_qaoa_generate_circuits_called_with_correct_phases(mocker, optimizer):
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
def test_qaoa_correct_circuits_count_and_energies(optimizer):
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


@flaky(max_runs=3, min_passes=1)
@pytest.mark.parametrize("optimizer", list(Optimizers))
def test_qaoa_compute_final_solution(mocker, optimizer):
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

    assert set(qaoa_problem._solution_nodes) == nx.algorithms.approximation.max_clique(
        G
    )

    assert spy.call_count == 1


def test_draw_solution_returns_graph_with_expected_properties(mocker):
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
