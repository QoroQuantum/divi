# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import re

import pennylane as qml
import pytest
from qprog_contracts import (
    verify_correct_circuit_count,
    verify_metacircuit_dict,
)

from divi.qprog import VQE, Optimizer, VQEAnsatz

pytestmark = pytest.mark.algo


def test_vqe_basic_initialization(default_test_simulator):
    vqe_problem = VQE(
        symbols=["H", "H"],
        bond_length=0.5,
        coordinate_structure=[(1, 0, 0), (0, -1, 0)],
        n_layers=2,
        backend=default_test_simulator,
    )

    assert vqe_problem.backend.shots == 5000

    assert vqe_problem.symbols == ["H", "H"]
    assert vqe_problem.bond_length == 0.5
    assert vqe_problem.coordinate_structure == [(1, 0, 0), (0, -1, 0)]
    assert vqe_problem.n_layers == 2

    # Check Hamiltonian type
    assert (
        isinstance(vqe_problem.cost_hamiltonian, qml.operation.Operator) == 1
    ), "Expected a pennylane Operator object for the hamiltonian"

    # Check meta-circuits
    verify_metacircuit_dict(vqe_problem, ["cost_circuit"])


@pytest.mark.parametrize("ansatz", list(VQEAnsatz))
@pytest.mark.parametrize("n_layers", [1, 2])
def test_meta_circuit_qasm(ansatz, n_layers):
    if ansatz == VQEAnsatz.HW_EFFICIENT:
        pytest.skip("Skipping HW_EFFICIENT ansatz")

    vqe_problem = VQE(
        symbols=["H", "H"],
        bond_length=0.5,
        coordinate_structure=[(1, 0, 0), (0, -1, 0)],
        n_layers=n_layers,
        ansatz=ansatz,
        backend=None,
    )

    meta_circuit_obj = vqe_problem._meta_circuits["cost_circuit"]
    meta_circuit_qasm = meta_circuit_obj.compiled_circuits_bodies[0]

    pattern = r"w_(\d+)_(\d+)"
    matches = re.findall(pattern, meta_circuit_qasm)

    # Check that we have the correct number of unique parameters
    assert len(set(matches)) == n_layers * vqe_problem.n_params
    assert len(set(matches)) // n_layers == vqe_problem.n_params


@pytest.mark.parametrize("optimizer", list(Optimizer))
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
            ansatz=VQEAnsatz.HW_EFFICIENT,
        )


@pytest.mark.parametrize("optimizer", list(Optimizer))
def test_vqe_correct_circuits_count_and_energies(optimizer, dummy_simulator):
    vqe_problem = VQE(
        symbols=["H", "H"],
        bond_length=0.5,
        coordinate_structure=[(1, 0, 0), (0, -1, 0)],
        n_layers=1,
        ansatz=VQEAnsatz.HARTREE_FOCK,
        optimizer=optimizer,
        max_iterations=1,
        backend=dummy_simulator,
    )

    vqe_problem.run()

    verify_correct_circuit_count(vqe_problem)
