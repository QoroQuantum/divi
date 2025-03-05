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

    verify_metacircuit_dict(qaoa_problem, ["opt_circuit", "meas_circuit"])


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
            for op in qaoa_problem._meta_circuits["opt_circuit"].main_circuit.operations
        )
        == nx.bull_graph().number_of_nodes()
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
