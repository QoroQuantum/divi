import matplotlib.pyplot as plt
import networkx as nx
import pennylane as qml
import pytest
from flaky import flaky
from qprog_contracts import verify_hamiltonian_metadata, verify_metacircuit_dict

from divi.qprog import QAOA
from divi.qprog.optimizers import Optimizers

pytestmark = pytest.mark.algo


def test_qaoa_basic_initialization():
    G = nx.bull_graph()

    qaoa_problem = QAOA(
        "max_clique",
        G,
        n_layers=2,
        optimizer=Optimizers.NELDER_MEAD,
        max_iterations=10,
        shots=6000,
        is_constrained=True,
        qoro_service=None,
    )

    assert qaoa_problem.shots == 6000
    assert qaoa_problem.qoro_service is None

    assert qaoa_problem.problem == "max_clique"
    assert qaoa_problem.graph == G
    assert qaoa_problem.n_layers == 2

    verify_hamiltonian_metadata(qaoa_problem)

    verify_metacircuit_dict(qaoa_problem, ["cost_circuit", "meas_circuit"])


def test_qaoa_unsuppported_problem():
    with pytest.raises(ValueError, match="travelling_salesman"):
        QAOA(
            "travelling_salesman",
            nx.bull_graph(),
            qoro_service=None,
        )


def test_qaoa_unsuppported_initial_state():
    with pytest.raises(ValueError, match="Bell"):
        QAOA(
            "max_clique",
            nx.bull_graph(),
            initial_state="Bell",
            qoro_service=None,
        )


def test_qaoa_initial_state_recommended():
    qaoa_problem = QAOA(
        "max_clique",
        nx.bull_graph(),
        initial_state="Recommended",
        is_constrained=True,
        qoro_service=None,
    )

    assert qaoa_problem.initial_state == "Zeros"


def test_qaoa_initial_state_superposition():
    qaoa_problem = QAOA(
        "max_clique",
        nx.bull_graph(),
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


@pytest.mark.parametrize("optimizer", [Optimizers.NELDER_MEAD, Optimizers.MONTE_CARLO])
def test_qaoa_generate_circuits_called_with_correct_phases(mocker, optimizer):
    qaoa_problem = QAOA(
        "max_clique",
        nx.bull_graph(),
        n_layers=1,
        optimizer=optimizer,
        max_iterations=1,
        is_constrained=True,
        qoro_service=None,
    )

    mock_generate_circuits = mocker.patch.object(qaoa_problem, "_generate_circuits")

    qaoa_problem.run()

    # Verify that _generate_circuits was called twice per iteration
    assert mock_generate_circuits.call_count % 2 == 0

    # Verify that _generate_circuits was called with measurement_phase=False first and then with measurement_phase=True
    for i in range(mock_generate_circuits.call_count, 2):
        assert not mock_generate_circuits.call_args_list[i][1]["measurement_phase"]
        assert mock_generate_circuits.call_args_list[i + 1][1]["measurement_phase"]


@pytest.mark.parametrize("optimizer", list(Optimizers))
def test_qaoa_correct_circuits_count_and_energies(optimizer):
    qaoa_problem = QAOA(
        "max_clique",
        nx.bull_graph(),
        n_layers=1,
        optimizer=optimizer,
        max_iterations=1,
        is_constrained=True,
        qoro_service=None,
    )

    qaoa_problem.run()

    assert qaoa_problem.current_iteration == 1

    # Need to add one here for the measurement phase
    if optimizer == Optimizers.MONTE_CARLO:
        assert len(qaoa_problem.losses) == 1
        assert (
            qaoa_problem.total_circuit_count
            == qaoa_problem.optimizer.n_param_sets
            * (len(qaoa_problem.cost_hamiltonian) + 1)
        )
    elif optimizer == Optimizers.NELDER_MEAD:
        assert len(qaoa_problem.losses) == qaoa_problem._minimize_res.nfev
        assert qaoa_problem.total_circuit_count == qaoa_problem._minimize_res.nfev * (
            len(qaoa_problem.cost_hamiltonian) + 1
        )


@flaky(max_runs=3, min_passes=1)
def test_qaoa_compute_final_solution():
    G = nx.bull_graph()

    qaoa_problem = QAOA(
        "max_clique",
        G,
        n_layers=1,
        optimizer=Optimizers.NELDER_MEAD,
        max_iterations=5,
        is_constrained=True,
        qoro_service=None,
    )

    qaoa_problem.run()

    assert set(
        qaoa_problem.compute_final_solution()
    ) == nx.algorithms.approximation.max_clique(G)


def test_draw_solution_returns_graph_with_expected_properties(mocker):
    G = nx.bull_graph()

    qaoa_problem = QAOA(
        "max_clique",
        G,
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
