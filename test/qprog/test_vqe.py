import pennylane as qml
import pytest
from qprog_contracts import verify_hamiltonian_metadata, verify_metacircuit_dict

from divi.qprog import VQE, VQEAnsatze

pytestmark = pytest.mark.algo


def test_vqe_basic_initialization():
    vqe_problem = VQE(
        symbols=["H", "H"],
        bond_length=0.5,
        coordinate_structure=[(1, 0, 0), (0, -1, 0)],
        n_layers=2,
        shots=2000,
        qoro_service=None,
    )

    assert vqe_problem.shots == 2000
    assert vqe_problem.qoro_service is None

    assert vqe_problem.symbols == ["H", "H"]
    assert vqe_problem.bond_length == 0.5
    assert vqe_problem.coordinate_structure == [(1, 0, 0), (0, -1, 0)]
    assert vqe_problem.n_layers == 2

    # Check Hamiltonian type
    assert (
        isinstance(vqe_problem.hamiltonian, qml.operation.Operator) == 1
    ), "Expected a pennylane Operator object for the hamiltonian"

    # Check Hamiltonian Meta-data exists in expected format
    verify_hamiltonian_metadata(vqe_problem)

    # Check meta-circuits
    verify_metacircuit_dict(vqe_problem, ["circuit"])


def test_vqe_symbol_coordinates_mismatch():
    with pytest.raises(
        ValueError,
        match="The number of symbols must match the number of coordinates",
    ):
        VQE(
            symbols=["H", "H", "H"],
            bond_length=0.5,
            coordinate_structure=[(1, 0, 0), (0, -1, 0)],
        )


def test_vqe_fail_with_hw_efficient_ansatz():
    with pytest.raises(
        NotImplementedError,
    ):
        VQE(
            symbols=["H", "H"],
            bond_length=0.5,
            coordinate_structure=[(1, 0, 0), (0, -1, 0)],
            ansatz=VQEAnsatze.HW_EFFICIENT,
        )
