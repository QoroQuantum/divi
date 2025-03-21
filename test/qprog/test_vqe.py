import re

import pennylane as qml
import pytest
from qprog_contracts import (
    verify_correct_circuit_count,
    verify_hamiltonian_metadata,
    verify_metacircuit_dict,
)

from divi.qprog import VQE, Optimizers, VQEAnsatze

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
        isinstance(vqe_problem.cost_hamiltonian, qml.operation.Operator) == 1
    ), "Expected a pennylane Operator object for the hamiltonian"

    # Check Hamiltonian Meta-data exists in expected format
    verify_hamiltonian_metadata(vqe_problem)

    # Check meta-circuits
    verify_metacircuit_dict(vqe_problem, ["cost_circuit"])


@pytest.mark.parametrize("ansatz", list(VQEAnsatze))
@pytest.mark.parametrize("n_layers", [1, 2])
def test_meta_circuit_qasm(ansatz, n_layers):
    if ansatz == VQEAnsatze.HW_EFFICIENT:
        pytest.skip("Skipping HW_EFFICIENT ansatz")

    vqe_problem = VQE(
        symbols=["H", "H"],
        bond_length=0.5,
        coordinate_structure=[(1, 0, 0), (0, -1, 0)],
        n_layers=n_layers,
        ansatz=ansatz,
        qoro_service=None,
    )

    meta_circuit_obj = vqe_problem._meta_circuits["cost_circuit"]
    meta_circuit_qasm = meta_circuit_obj.compiled_circuit

    pattern = r"w_(\d+)_(\d+)"
    matches = re.findall(pattern, meta_circuit_qasm)

    # Check that we have the correct number of unique parameters
    assert len(set(matches)) == n_layers * vqe_problem.n_params
    assert len(set(matches)) // n_layers == vqe_problem.n_params


@pytest.mark.parametrize("optimizer", list(Optimizers))
def test_vqe_symbol_coordinates_mismatch(optimizer):
    with pytest.raises(
        ValueError,
        match="The number of symbols must match the number of coordinates",
    ):
        VQE(
            symbols=["H", "H", "H"],
            bond_length=0.5,
            coordinate_structure=[(1, 0, 0), (0, -1, 0)],
            optimizer=optimizer,
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


@pytest.mark.parametrize("optimizer", list(Optimizers))
def test_vqe_correct_circuits_count_and_energies(optimizer):
    vqe_problem = VQE(
        symbols=["H", "H"],
        bond_length=0.5,
        coordinate_structure=[(1, 0, 0), (0, -1, 0)],
        n_layers=1,
        ansatz=VQEAnsatze.HARTREE_FOCK,
        optimizer=optimizer,
        max_iterations=1,
        qoro_service=None,
    )

    verify_correct_circuit_count(vqe_problem)
