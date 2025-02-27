import networkx as nx
from divi.qprog import QAOA
from divi.qprog.optimizers import Optimizers
import pytest
from qprog_contracts import verify_hamiltonian_metadata, verify_metacircuit_dict

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
            n_layers=2,
            optimizer=Optimizers.NELDER_MEAD,
            max_iterations=10,
            is_constrained=True,
            qoro_service=None,
        )


def test_qaoa_unsuppported_initial_state():
    with pytest.raises(ValueError, match="Bell"):
        QAOA(
            "max_clique",
            nx.bull_graph(),
            n_layers=2,
            optimizer=Optimizers.NELDER_MEAD,
            max_iterations=10,
            initial_state="Bell",
            is_constrained=True,
            qoro_service=None,
        )
