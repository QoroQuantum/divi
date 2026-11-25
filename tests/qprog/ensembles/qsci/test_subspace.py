# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from qiskit.quantum_info import SparsePauliOp

from divi.qprog.ensembles.qsci import _subspace
from divi.qprog.ensembles.qsci._subspace import (
    ConfigurationSubspace,
    configs_to_limbs,
    limbs_to_configs,
)
from tests._helpers import exact_match
from tests.qprog.ensembles.qsci._helpers import (
    H_SMALL,
    N_QUBITS,
    build_random_hamiltonian,
    compute_basis_index,
    compute_lowest_penalised_state,
    list_all_configs,
    solve_subspace,
)

_NON_HERMITIAN = SparsePauliOp.from_list([("XZ", 1.0 + 0.5j)])
_EMPTY = SparsePauliOp.from_list([("XZ", 0.0)])


@pytest.fixture(params=["dense", "sparse"])
def solver_path(request, mocker):
    """Diagonalise densely, or with the sparse solver at every size."""
    if request.param == "sparse":
        mocker.patch.object(_subspace, "_DENSE_SOLVE_THRESHOLD", 0)


@pytest.mark.parametrize("odd_y", [False, True], ids=["real", "complex"])
def test_ground_state_of_full_space_matches_exact(solver_path, odd_y):
    hamiltonian = build_random_hamiltonian(N_QUBITS, 11, odd_y=odd_y)

    subspace = solve_subspace(hamiltonian, list_all_configs(N_QUBITS))

    assert np.iscomplexobj(subspace.eigenvector) == odd_y
    assert subspace.energy == pytest.approx(
        np.linalg.eigvalsh(hamiltonian.to_matrix())[0], abs=1e-9
    )


@pytest.mark.parametrize("odd_y", [False, True], ids=["real", "complex"])
def test_projected_matrix_matches_dense_reference(odd_y):
    hamiltonian = build_random_hamiltonian(N_QUBITS, 5, odd_y=odd_y)
    configs = list_all_configs(N_QUBITS)[::3]
    subspace = ConfigurationSubspace(hamiltonian)
    subspace.add_configurations(configs)

    subspace.extend_hamiltonian_matrix()

    index = [compute_basis_index(c) for c in configs]
    expected = hamiltonian.to_matrix()[np.ix_(index, index)]
    np.testing.assert_allclose(
        subspace.hamiltonian_matrix.toarray(), expected, atol=1e-12
    )


def test_extending_matrix_equals_building_it_whole():
    hamiltonian = build_random_hamiltonian(N_QUBITS, 8, odd_y=True)
    configs = list_all_configs(N_QUBITS)[:12]
    grown = ConfigurationSubspace(hamiltonian)
    for chunk in (configs[:5], configs[5:]):
        grown.add_configurations(chunk)
        grown.extend_hamiltonian_matrix()

    whole = ConfigurationSubspace(hamiltonian)
    whole.add_configurations(configs)
    whole.extend_hamiltonian_matrix()

    np.testing.assert_allclose(
        grown.hamiltonian_matrix.toarray(),
        whole.hamiltonian_matrix.toarray(),
        atol=1e-12,
    )


@pytest.mark.parametrize("odd_y", [False, True], ids=["real", "complex"])
def test_penalised_state_is_the_lowest_of_the_penalised_hamiltonian(solver_path, odd_y):
    hamiltonian = build_random_hamiltonian(N_QUBITS, 11, odd_y=odd_y)
    penalty = build_random_hamiltonian(N_QUBITS, 4, odd_y=False)

    subspace = solve_subspace(
        hamiltonian, list_all_configs(N_QUBITS), penalty=penalty, lambda_penalty=0.7
    )

    expected, energy = compute_lowest_penalised_state(hamiltonian, penalty, 0.7)
    state = np.zeros(2**N_QUBITS, dtype=complex)
    state[[compute_basis_index(c) for c in subspace.configs]] = subspace.eigenvector
    assert abs(np.vdot(expected, state)) == pytest.approx(1.0, abs=1e-8)
    assert subspace.energy == pytest.approx(energy, abs=1e-9)


def test_penalty_follows_extension():
    hamiltonian = build_random_hamiltonian(N_QUBITS, 8, odd_y=False)
    penalty = build_random_hamiltonian(N_QUBITS, 2, odd_y=False)
    configs = list_all_configs(N_QUBITS)[:20]
    grown = ConfigurationSubspace(hamiltonian, penalty=penalty)
    for chunk in (configs[:7], configs[7:]):
        grown.add_configurations(chunk)
        grown.extend_hamiltonian_matrix()

    grown.solve_eigensystem()

    whole = solve_subspace(hamiltonian, configs, penalty=penalty)
    assert grown.energy == pytest.approx(whole.energy, abs=1e-10)


def test_penalty_follows_pruning():
    hamiltonian = build_random_hamiltonian(N_QUBITS, 8, odd_y=False)
    penalty = build_random_hamiltonian(N_QUBITS, 2, odd_y=False)
    subspace = solve_subspace(hamiltonian, list_all_configs(N_QUBITS), penalty=penalty)

    subspace.prune(float(np.sort(np.abs(subspace.eigenvector))[5]))
    subspace.solve_eigensystem()

    rebuilt = solve_subspace(hamiltonian, subspace.configs, penalty=penalty)
    assert subspace.energy == pytest.approx(rebuilt.energy, abs=1e-10)


def test_penalty_on_another_register_is_rejected():
    with pytest.raises(
        ValueError,
        match=exact_match("The penalty acts on 5 qubits; the Hamiltonian on 4."),
    ):
        ConfigurationSubspace(
            H_SMALL, penalty=build_random_hamiltonian(5, 1, odd_y=False)
        )


def test_wide_register_matches_four_qubit_reference():
    terms = [("XYZI", 0.6), ("YYII", -0.3), ("ZZXX", 0.4), ("IIIZ", 0.2), ("XIIY", 0.5)]
    reference = SparsePauliOp.from_sparse_list(
        [(label, [0, 1, 2, 3], c) for label, c in terms], num_qubits=4
    )
    wide = SparsePauliOp.from_sparse_list(
        [(label, [0, 1, 68, 69], c) for label, c in terms], num_qubits=70
    )
    patterns = list_all_configs(4)
    configs = [p[:2] + "0" * 66 + p[2:] for p in patterns]

    subspace = solve_subspace(wide, configs)

    index = [compute_basis_index(p) for p in patterns]
    np.testing.assert_allclose(
        subspace.hamiltonian_matrix.toarray(),
        reference.to_matrix()[np.ix_(index, index)],
        atol=1e-12,
    )
    assert subspace.energy == pytest.approx(
        np.linalg.eigvalsh(reference.to_matrix())[0]
    )


@pytest.mark.parametrize("odd_y", [False, True], ids=["real", "complex"])
def test_external_contributions_match_dense_second_order(odd_y):
    hamiltonian = build_random_hamiltonian(N_QUBITS, 21, odd_y=odd_y)
    configs = list_all_configs(N_QUBITS)[1::5]
    subspace = solve_subspace(hamiltonian, configs)

    dets, contributions = subspace.external_contributions(amplitude_cutoff=0)

    dense = hamiltonian.to_matrix()
    psi = np.zeros(2**N_QUBITS, dtype=complex)
    for config, amplitude in zip(configs, subspace.eigenvector):
        psi[compute_basis_index(config)] = amplitude
    projected = dense @ psi
    outside = [
        i
        for i in range(2**N_QUBITS)
        if i not in {compute_basis_index(c) for c in configs}
    ]
    expected = {
        i: abs(projected[i]) ** 2 / (subspace.energy - dense[i, i].real)
        for i in outside
        if abs(projected[i]) > 1e-12
    }
    found = {
        compute_basis_index(c): v
        for c, v in zip(limbs_to_configs(dets, N_QUBITS), contributions)
        if abs(v) > 1e-14
    }
    assert found.keys() == expected.keys()
    for key, value in expected.items():
        assert found[key] == pytest.approx(value)


def test_a_determinant_degenerate_with_the_energy_contributes_without_bound():
    subspace = solve_subspace(SparsePauliOp("X"), ["0"])

    dets, contributions = subspace.external_contributions()

    assert limbs_to_configs(dets, 1) == ["1"]
    assert contributions.tolist() == [-np.inf]


def test_prune_drops_small_amplitudes_and_matrix_rows():
    subspace = solve_subspace(H_SMALL, list_all_configs(4))
    threshold = float(np.sort(np.abs(subspace.eigenvector))[4])
    kept = [
        c for c, v in zip(subspace.configs, subspace.eigenvector) if abs(v) > threshold
    ]

    removed = subspace.prune(threshold)

    assert removed == 16 - len(kept)
    assert subspace.configs == kept
    assert subspace.hamiltonian_matrix.shape == (len(kept),) * 2
    assert subspace.eigenvector.size == len(kept)


def test_prune_needs_a_solved_eigensystem():
    subspace = ConfigurationSubspace(H_SMALL)
    subspace.add_configurations(["0000"])

    with pytest.raises(ValueError, match="Solve the eigensystem"):
        subspace.prune(0.1)


@pytest.mark.parametrize(
    "configs, message",
    [(["000"], "4 characters wide"), (["00a0"], "only '0' and '1'")],
)
def test_malformed_configurations_are_rejected(configs, message):
    with pytest.raises(ValueError, match=message):
        ConfigurationSubspace(H_SMALL).add_configurations(configs)


@pytest.mark.parametrize(
    "operators, message",
    [
        ({"hamiltonian": _NON_HERMITIAN}, "Hamiltonian is not Hermitian"),
        ({"hamiltonian": _EMPTY}, "Hamiltonian has no non-zero terms"),
        ({"penalty": _NON_HERMITIAN}, "Penalty is not Hermitian"),
        ({"penalty": _EMPTY}, "Penalty has no non-zero terms"),
    ],
)
def test_invalid_operators_are_named_when_rejected(operators, message):
    with pytest.raises(ValueError, match=message):
        ConfigurationSubspace(
            **{"hamiltonian": SparsePauliOp.from_list([("ZI", 1.0)]), **operators}
        )


@pytest.mark.parametrize("lambda_penalty", [-0.1, float("inf"), float("nan")])
def test_invalid_lambda_penalty_is_rejected(lambda_penalty):
    with pytest.raises(
        ValueError,
        match=exact_match(
            f"lambda_penalty must be finite and non-negative; got {lambda_penalty!r}."
        ),
    ):
        ConfigurationSubspace(H_SMALL, lambda_penalty=lambda_penalty)


def test_configurations_round_trip_through_limbs():
    configs = ["1" + "0" * 68 + "1", "0" * 70, "10" * 35]

    assert limbs_to_configs(configs_to_limbs(configs, 70), 70) == configs
