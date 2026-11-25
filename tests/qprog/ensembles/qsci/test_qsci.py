# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from qiskit.quantum_info import SparsePauliOp

from divi.qprog.algorithms import VQE, TimeEvolution
from divi.qprog.ensembles.qsci import (
    QSCI,
    ExpansionRecord,
    QSCIState,
    _subspace,
    extrapolate_pt2_to_zero,
)
from divi.qprog.ensembles.qsci._subspace import (
    ConfigurationSubspace,
    configs_to_limbs,
    limbs_to_configs,
)
from tests._helpers import exact_match

_N_QUBITS = 5

_H_SMALL = SparsePauliOp.from_sparse_list(
    [
        ("ZZ", [0, 1], 0.7),
        ("ZZ", [1, 2], -0.4),
        ("X", [0], 0.5),
        ("X", [3], 0.3),
        ("XX", [2, 3], 0.2),
        ("Z", [3], 0.1),
    ],
    num_qubits=4,
)


def _random_hamiltonian(n_qubits, seed, *, odd_y):
    rng = np.random.default_rng(seed)
    terms = {}
    while len(terms) < 14:
        label = "".join(rng.choice(list("IXYZ"), n_qubits))
        if odd_y or label.count("Y") % 2 == 0:
            terms[label] = rng.normal()
    return SparsePauliOp.from_list(list(terms.items()))


def _basis_index(config):
    return sum(1 << q for q, bit in enumerate(config) if bit == "1")


def _all_configs(n_qubits):
    return [format(i, f"0{n_qubits}b")[::-1] for i in range(2**n_qubits)]


def _subspace_of(hamiltonian, configs, **penalty):
    subspace = ConfigurationSubspace(hamiltonian, **penalty)
    subspace.add_configurations(configs)
    subspace.extend_hamiltonian_matrix()
    subspace.solve_eigensystem()
    return subspace


def _build_probability_program(mocker, distribution):
    program = mocker.Mock()
    program.probabilities.return_value = distribution
    return program


def _build_vqe(mocker, best_probs):
    vqe = mocker.Mock(spec=VQE)
    vqe.best_probs = best_probs
    return vqe


def _build_qsci(dummy_simulator, programs, hamiltonian=None, **kwargs):
    """A QSCI whose sampling programs are ``programs``, already run."""
    qsci = QSCI(
        hamiltonian or _random_hamiltonian(_N_QUBITS, 3, odd_y=False),
        lambda backend: {},
        backend=dummy_simulator,
        **kwargs,
    )
    qsci.programs = programs
    return qsci


def _qsci_on(mocker, dummy_simulator, distribution, hamiltonian=None, **kwargs):
    return _build_qsci(
        dummy_simulator,
        {"sampler": _build_probability_program(mocker, distribution)},
        hamiltonian,
        **kwargs,
    )


def _lowest_penalised_state(hamiltonian, penalty, lambda_penalty):
    """The lowest eigenvector of ``H + lambda * D @ D`` over the whole space,
    and its ``<H>``."""
    dense_hamiltonian = hamiltonian.to_matrix()
    dense_penalty = penalty.to_matrix()
    _, vectors = np.linalg.eigh(
        dense_hamiltonian + lambda_penalty * dense_penalty @ dense_penalty
    )
    expected = vectors[:, 0]
    return expected, np.vdot(expected, dense_hamiltonian @ expected).real


@pytest.fixture(params=["dense", "sparse"])
def solver_path(request, mocker):
    """Diagonalise densely, or with the sparse solver at every size."""
    if request.param == "sparse":
        mocker.patch.object(_subspace, "_DENSE_SOLVE_THRESHOLD", 0)


@pytest.mark.parametrize("odd_y", [False, True], ids=["real", "complex"])
def test_ground_state_of_full_space_matches_exact(solver_path, odd_y):
    hamiltonian = _random_hamiltonian(_N_QUBITS, 11, odd_y=odd_y)

    subspace = _subspace_of(hamiltonian, _all_configs(_N_QUBITS))

    assert np.iscomplexobj(subspace.eigenvector) == odd_y
    assert subspace.energy == pytest.approx(
        np.linalg.eigvalsh(hamiltonian.to_matrix())[0], abs=1e-9
    )


@pytest.mark.parametrize("odd_y", [False, True], ids=["real", "complex"])
def test_projected_matrix_matches_dense_reference(odd_y):
    hamiltonian = _random_hamiltonian(_N_QUBITS, 5, odd_y=odd_y)
    configs = _all_configs(_N_QUBITS)[::3]
    subspace = ConfigurationSubspace(hamiltonian)
    subspace.add_configurations(configs)

    subspace.extend_hamiltonian_matrix()

    index = [_basis_index(c) for c in configs]
    expected = hamiltonian.to_matrix()[np.ix_(index, index)]
    np.testing.assert_allclose(
        subspace.hamiltonian_matrix.toarray(), expected, atol=1e-12
    )


def test_extending_matrix_equals_building_it_whole():
    hamiltonian = _random_hamiltonian(_N_QUBITS, 8, odd_y=True)
    configs = _all_configs(_N_QUBITS)[:12]
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
    hamiltonian = _random_hamiltonian(_N_QUBITS, 11, odd_y=odd_y)
    penalty = _random_hamiltonian(_N_QUBITS, 4, odd_y=False)

    subspace = _subspace_of(
        hamiltonian, _all_configs(_N_QUBITS), penalty=penalty, lambda_penalty=0.7
    )

    expected, energy = _lowest_penalised_state(hamiltonian, penalty, 0.7)
    state = np.zeros(2**_N_QUBITS, dtype=complex)
    state[[_basis_index(c) for c in subspace.configs]] = subspace.eigenvector
    assert abs(np.vdot(expected, state)) == pytest.approx(1.0, abs=1e-8)
    assert subspace.energy == pytest.approx(energy, abs=1e-9)


def test_penalty_follows_extension():
    hamiltonian = _random_hamiltonian(_N_QUBITS, 8, odd_y=False)
    penalty = _random_hamiltonian(_N_QUBITS, 2, odd_y=False)
    configs = _all_configs(_N_QUBITS)[:20]
    grown = ConfigurationSubspace(hamiltonian, penalty=penalty)
    for chunk in (configs[:7], configs[7:]):
        grown.add_configurations(chunk)
        grown.extend_hamiltonian_matrix()

    grown.solve_eigensystem()

    whole = _subspace_of(hamiltonian, configs, penalty=penalty)
    assert grown.energy == pytest.approx(whole.energy, abs=1e-10)


def test_penalty_follows_pruning():
    hamiltonian = _random_hamiltonian(_N_QUBITS, 8, odd_y=False)
    penalty = _random_hamiltonian(_N_QUBITS, 2, odd_y=False)
    subspace = _subspace_of(hamiltonian, _all_configs(_N_QUBITS), penalty=penalty)

    subspace.prune(float(np.sort(np.abs(subspace.eigenvector))[5]))
    subspace.solve_eigensystem()

    rebuilt = _subspace_of(hamiltonian, subspace.configs, penalty=penalty)
    assert subspace.energy == pytest.approx(rebuilt.energy, abs=1e-10)


def test_penalty_on_another_register_is_rejected():
    with pytest.raises(
        ValueError,
        match=exact_match("The penalty acts on 5 qubits; the Hamiltonian on 4."),
    ):
        ConfigurationSubspace(_H_SMALL, penalty=_random_hamiltonian(5, 1, odd_y=False))


def test_wide_register_matches_four_qubit_reference():
    terms = [("XYZI", 0.6), ("YYII", -0.3), ("ZZXX", 0.4), ("IIIZ", 0.2), ("XIIY", 0.5)]
    reference = SparsePauliOp.from_sparse_list(
        [(label, [0, 1, 2, 3], c) for label, c in terms], num_qubits=4
    )
    wide = SparsePauliOp.from_sparse_list(
        [(label, [0, 1, 68, 69], c) for label, c in terms], num_qubits=70
    )
    patterns = _all_configs(4)
    configs = [p[:2] + "0" * 66 + p[2:] for p in patterns]

    subspace = _subspace_of(wide, configs)

    index = [_basis_index(p) for p in patterns]
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
    hamiltonian = _random_hamiltonian(_N_QUBITS, 21, odd_y=odd_y)
    configs = _all_configs(_N_QUBITS)[1::5]
    subspace = _subspace_of(hamiltonian, configs)

    dets, contributions = subspace.external_contributions(amplitude_cutoff=0)

    dense = hamiltonian.to_matrix()
    psi = np.zeros(2**_N_QUBITS, dtype=complex)
    for config, amplitude in zip(configs, subspace.eigenvector):
        psi[_basis_index(config)] = amplitude
    projected = dense @ psi
    outside = [
        i for i in range(2**_N_QUBITS) if i not in {_basis_index(c) for c in configs}
    ]
    expected = {
        i: abs(projected[i]) ** 2 / (subspace.energy - dense[i, i].real)
        for i in outside
        if abs(projected[i]) > 1e-12
    }
    found = {
        _basis_index(c): v
        for c, v in zip(limbs_to_configs(dets, _N_QUBITS), contributions)
        if abs(v) > 1e-14
    }
    assert found.keys() == expected.keys()
    for key, value in expected.items():
        assert found[key] == pytest.approx(value)


def test_prune_drops_small_amplitudes_and_matrix_rows():
    subspace = _subspace_of(_H_SMALL, _all_configs(4))
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
    subspace = ConfigurationSubspace(_H_SMALL)
    subspace.add_configurations(["0000"])

    with pytest.raises(ValueError, match="Solve the eigensystem"):
        subspace.prune(0.1)


@pytest.mark.parametrize(
    "configs, message",
    [(["000"], "4 characters wide"), (["00a0"], "only '0' and '1'")],
)
def test_malformed_configurations_are_rejected(configs, message):
    with pytest.raises(ValueError, match=message):
        ConfigurationSubspace(_H_SMALL).add_configurations(configs)


_NON_HERMITIAN = SparsePauliOp.from_list([("XZ", 1.0 + 0.5j)])
_EMPTY = SparsePauliOp.from_list([("XZ", 0.0)])


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


def test_configurations_round_trip_through_limbs():
    configs = ["1" + "0" * 68 + "1", "0" * 70, "10" * 35]

    assert limbs_to_configs(configs_to_limbs(configs, 70), 70) == configs


def test_qsci_energy_reaches_exact_ground_state_from_sampled_seed(
    default_test_simulator,
):
    qsci = QSCI(
        _H_SMALL,
        lambda backend: {
            "evolved": TimeEvolution(hamiltonian=_H_SMALL, time=1.0, backend=backend)
        },
        backend=default_test_simulator,
    )

    state = qsci.run().aggregate_results()

    assert isinstance(state, QSCIState)
    assert state.energy == pytest.approx(
        np.linalg.eigvalsh(_H_SMALL.to_matrix())[0], abs=1e-6
    )
    assert state.records[-1].pt2_correction == pytest.approx(0.0, abs=1e-8)


def test_seed_is_filtered_ranked_and_capped(mocker, dummy_simulator):
    qsci = _qsci_on(
        mocker,
        dummy_simulator,
        {"00011": 0.2, "00101": 0.5, "11111": 0.9, "01001": 0.3},
        config_filter=lambda config: config.count("1") == 2,
        max_seed_configs=2,
        n_expansions=0,
    )

    state = qsci.update_state(qsci.initial_state())

    assert set(state.configs) == {"00101", "01001"}


@pytest.mark.parametrize(
    "best_probs",
    [
        {0: {"00011": 0.3}, 1: {"00011": 0.3, "00101": 0.5}},
        {0: {"00011": 0.3, "00101": 0.5}, 1: {"00011": 0.3}},
    ],
    ids=["runner-up-last", "runner-up-first"],
)
def test_seed_sums_every_parameter_set_of_a_solution_sampling_program(
    mocker, dummy_simulator, best_probs
):
    qsci = _build_qsci(
        dummy_simulator,
        {"vqe": _build_vqe(mocker, best_probs)},
        max_seed_configs=1,
        n_expansions=0,
    )

    state = qsci.update_state(qsci.initial_state())

    assert state.configs == ("00011",)


def test_seed_pools_solution_sampling_and_probability_programs(mocker, dummy_simulator):
    qsci = _build_qsci(
        dummy_simulator,
        {
            "sampler": _build_probability_program(
                mocker, {"00101": 0.25, "00110": 0.3}
            ),
            "vqe": _build_vqe(mocker, {0: {"00011": 0.3, "00101": 0.2}}),
        },
        max_seed_configs=1,
        n_expansions=0,
    )

    state = qsci.update_state(qsci.initial_state())

    assert state.configs == ("00101",)


def test_penalty_applies_at_the_default_weight(mocker, dummy_simulator):
    hamiltonian = _random_hamiltonian(_N_QUBITS, 11, odd_y=False)
    penalty = _random_hamiltonian(_N_QUBITS, 4, odd_y=False)
    every_config = dict.fromkeys(_all_configs(_N_QUBITS), 1 / 2**_N_QUBITS)
    qsci = _qsci_on(
        mocker,
        dummy_simulator,
        every_config,
        hamiltonian,
        penalty=penalty,
        n_expansions=0,
    )

    state = qsci.update_state(qsci.initial_state())

    _, energy = _lowest_penalised_state(hamiltonian, penalty, 0.2)
    assert state.energy == pytest.approx(energy, abs=1e-9)


def test_expansion_keeps_the_penalty(mocker, dummy_simulator):
    hamiltonian = _random_hamiltonian(_N_QUBITS, 11, odd_y=False)
    penalty = _random_hamiltonian(_N_QUBITS, 4, odd_y=False)
    qsci = _qsci_on(
        mocker,
        dummy_simulator,
        {"00000": 0.6, "11000": 0.4},
        hamiltonian,
        penalty=penalty,
        expansion_width=4,
        n_expansions=3,
        delta_conv=-1.0,
        epsilon_wf=1e-3,
    )

    state = qsci.update_state(qsci.initial_state())

    rebuilt = _subspace_of(hamiltonian, state.configs, penalty=penalty)
    assert state.energy == pytest.approx(rebuilt.energy, abs=1e-10)


def test_every_sample_filtered_out_raises(mocker, dummy_simulator):
    qsci = _qsci_on(
        mocker, dummy_simulator, {"00011": 1.0}, config_filter=lambda c: False
    )

    with pytest.raises(ValueError, match="No sampled configuration passed"):
        qsci.update_state(qsci.initial_state())


def test_expansion_adds_width_configurations_per_step(mocker, dummy_simulator):
    qsci = _qsci_on(
        mocker,
        dummy_simulator,
        {"00000": 0.6, "11000": 0.4},
        expansion_width=2,
        n_expansions=3,
        delta_conv=-1.0,
        epsilon_wf=0.0,
    )

    state = qsci.update_state(qsci.initial_state())

    energies = [r.energy for r in state.records]
    assert [r.subspace_size for r in state.records] == [2, 4, 6, 8]
    assert energies == sorted(energies, reverse=True)


def test_expansion_stops_at_max_subspace_size(mocker, dummy_simulator):
    qsci = _qsci_on(
        mocker,
        dummy_simulator,
        {"00000": 0.6, "11000": 0.4},
        expansion_width=2,
        max_subspace_size=5,
        delta_conv=-1.0,
        epsilon_wf=0.0,
    )

    state = qsci.update_state(qsci.initial_state())

    assert [r.subspace_size for r in state.records] == [2, 4, 5]


def test_expansion_stops_once_a_step_gains_less_than_delta_conv(
    mocker, dummy_simulator
):
    qsci = _qsci_on(mocker, dummy_simulator, {"00000": 1.0}, delta_conv=1e9)

    state = qsci.update_state(qsci.initial_state())

    assert len(state.records) == 2


def test_plain_qsci_records_only_the_seed(mocker, dummy_simulator):
    qsci = _qsci_on(
        mocker, dummy_simulator, {"00000": 0.6, "10000": 0.4}, n_expansions=0
    )

    state = qsci.update_state(qsci.initial_state())

    assert [r.subspace_size for r in state.records] == [2]


def test_update_state_is_repeatable(mocker, dummy_simulator):
    qsci = _qsci_on(
        mocker, dummy_simulator, {"00000": 0.6, "10000": 0.4}, n_expansions=2
    )

    first = qsci.update_state(qsci.initial_state())
    second = qsci.update_state(first)

    assert second.records == first.records


def test_workflow_completes_once_a_state_has_an_eigenvector(mocker, dummy_simulator):
    qsci = _qsci_on(mocker, dummy_simulator, {"00000": 1.0}, n_expansions=0)

    state = qsci.update_state(qsci.initial_state())

    assert not qsci.is_complete(qsci.initial_state())
    assert qsci.is_complete(state)


def test_pt2_energy_adds_the_final_correction():
    state = QSCIState(
        energy=-1.0,
        records=(ExpansionRecord(3, -0.9, -0.2), ExpansionRecord(5, -1.0, -0.05)),
    )

    assert state.pt2_energy == pytest.approx(-1.05)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_seed_configs": 0},
        {"n_expansions": -1},
        {"expansion_width": True},
        {"max_subspace_size": 1.5},
    ],
)
def test_invalid_counts_are_rejected(dummy_simulator, kwargs):
    name = next(iter(kwargs))

    with pytest.raises(ValueError, match=f"{name} must be an integer"):
        QSCI(_H_SMALL, lambda backend: {}, backend=dummy_simulator, **kwargs)


@pytest.mark.parametrize("lambda_penalty", [-0.1, float("inf"), float("nan")])
def test_invalid_lambda_penalty_is_rejected(lambda_penalty):
    with pytest.raises(
        ValueError,
        match=exact_match(
            f"lambda_penalty must be finite and non-negative; got {lambda_penalty!r}."
        ),
    ):
        ConfigurationSubspace(_H_SMALL, lambda_penalty=lambda_penalty)


def test_extrapolation_recovers_the_intercept_of_a_linear_trend():
    records = [
        ExpansionRecord(n, -2.0 + 3.0 * pt2, pt2)
        for n, pt2 in enumerate([-0.4, -0.2, -0.1, -0.05])
    ]

    energy, slope, r_squared = extrapolate_pt2_to_zero(records)

    assert (energy, slope, r_squared) == pytest.approx((-2.0, 3.0, 1.0))


def test_extrapolation_window_drops_early_records():
    records = [ExpansionRecord(0, 5.0, -3.0)] + [
        ExpansionRecord(n, -2.0 + 3.0 * pt2, pt2)
        for n, pt2 in enumerate([-0.2, -0.1, -0.05], 1)
    ]

    energy, _, _ = extrapolate_pt2_to_zero(records, max_correction=0.5)

    assert energy == pytest.approx(-2.0)


def test_extrapolation_of_a_converged_subspace_is_its_energy():
    records = [ExpansionRecord(4, -1.5, 0.0), ExpansionRecord(6, -1.5, 0.0)]

    assert extrapolate_pt2_to_zero(records) == (-1.5, 0.0, 1.0)


def test_extrapolation_needs_two_records():
    with pytest.raises(
        ValueError,
        match=exact_match(
            "Need at least two records to extrapolate, got 1; widen n_points or max_correction, or allow more expansions."
        ),
    ):
        extrapolate_pt2_to_zero([ExpansionRecord(1, -1.0, -0.1)])
