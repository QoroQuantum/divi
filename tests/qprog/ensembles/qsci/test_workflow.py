# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from qiskit.quantum_info import SparsePauliOp

from divi.qprog.algorithms import VQE, TimeEvolution
from divi.qprog.checkpointing import CheckpointConfig
from divi.qprog.ensembles import ReportingLevel
from divi.qprog.ensembles.qsci import (
    QSCI,
    ExpansionRecord,
    QSCIState,
    extrapolate_pt2_to_zero,
)
from tests._helpers import exact_match
from tests.qprog.ensembles.qsci._helpers import (
    H_SMALL,
    N_QUBITS,
    build_random_hamiltonian,
    compute_lowest_penalised_state,
    list_all_configs,
    solve_subspace,
)


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
        hamiltonian or build_random_hamiltonian(N_QUBITS, 3, odd_y=False),
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


def _build_evolving_qsci(backend):
    return QSCI(
        H_SMALL,
        lambda backend: {
            "evolved": TimeEvolution(hamiltonian=H_SMALL, time=1.0, backend=backend)
        },
        backend=backend,
        reporting_level=ReportingLevel.OFF,
    )


def test_qsci_energy_reaches_exact_ground_state_from_sampled_seed(
    default_test_simulator,
):
    state = _build_evolving_qsci(default_test_simulator).run().aggregate_results()

    assert isinstance(state, QSCIState)
    assert state.energy == pytest.approx(
        np.linalg.eigvalsh(H_SMALL.to_matrix())[0], abs=1e-6
    )
    assert state.records[-1].pt2_correction == pytest.approx(0.0, abs=1e-8)


def test_a_checkpointed_run_restores_its_final_state(tmp_path, default_test_simulator):
    state = (
        _build_evolving_qsci(default_test_simulator)
        .run(checkpoint_config=CheckpointConfig(checkpoint_dir=tmp_path))
        .aggregate_results()
    )

    restored = _build_evolving_qsci(default_test_simulator).restore_state(tmp_path)

    loaded = restored.workflow_state
    assert (loaded.energy, loaded.configs, loaded.records) == (
        state.energy,
        state.configs,
        state.records,
    )
    np.testing.assert_array_equal(loaded.eigenvector, state.eigenvector)


def test_expansion_reaches_a_determinant_degenerate_with_the_seed(
    mocker, dummy_simulator
):
    qsci = _qsci_on(mocker, dummy_simulator, {"0": 1.0}, SparsePauliOp("X"))

    state = qsci.update_state(qsci.initial_state())

    assert set(state.configs) == {"0", "1"}
    assert state.energy == pytest.approx(-1.0)


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


def test_seed_never_exceeds_max_subspace_size(mocker, dummy_simulator):
    qsci = _qsci_on(
        mocker,
        dummy_simulator,
        {"00011": 0.2, "00101": 0.5, "11111": 0.9, "01001": 0.3},
        max_subspace_size=2,
    )

    state = qsci.update_state(qsci.initial_state())

    assert set(state.configs) == {"11111", "00101"}


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
    hamiltonian = build_random_hamiltonian(N_QUBITS, 11, odd_y=False)
    penalty = build_random_hamiltonian(N_QUBITS, 4, odd_y=False)
    every_config = dict.fromkeys(list_all_configs(N_QUBITS), 1 / 2**N_QUBITS)
    qsci = _qsci_on(
        mocker,
        dummy_simulator,
        every_config,
        hamiltonian,
        penalty=penalty,
        n_expansions=0,
    )

    state = qsci.update_state(qsci.initial_state())

    _, energy = compute_lowest_penalised_state(hamiltonian, penalty, 0.2)
    assert state.energy == pytest.approx(energy, abs=1e-9)


def test_expansion_keeps_the_penalty(mocker, dummy_simulator):
    hamiltonian = build_random_hamiltonian(N_QUBITS, 11, odd_y=False)
    penalty = build_random_hamiltonian(N_QUBITS, 4, odd_y=False)
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

    rebuilt = solve_subspace(hamiltonian, state.configs, penalty=penalty)
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
        QSCI(H_SMALL, lambda backend: {}, backend=dummy_simulator, **kwargs)


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
            "Need at least two records to extrapolate, got 1; widen n_points or "
            "max_correction, or allow more expansions."
        ),
    ):
        extrapolate_pt2_to_zero([ExpansionRecord(1, -1.0, -0.1)])
