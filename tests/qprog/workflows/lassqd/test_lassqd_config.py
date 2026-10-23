# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the LASSQD configuration objects' validation and normalisation."""

import re
from functools import partial

import numpy as np
import pytest
from pydantic import ValidationError

from divi.qprog.algorithms import UCCSDAnsatz
from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer
from divi.qprog.workflows.lassqd._config import (
    FragmentationConfig,
    LBFGSOrbitalUpdate,
    SecondOrderOrbitalUpdate,
    StateAveragedOrbitalUpdate,
    VQEPreparation,
)
from divi.qprog.workflows.lassqd._sqd import SQDConfig
from divi.qprog.workflows.lassqd._state import FragmentSpec

_SPECS = (
    FragmentSpec(orbitals=(0, 1), n_alpha=1, n_beta=1),
    FragmentSpec(orbitals=(2, 3), n_alpha=1, n_beta=1),
)


class _ConfiguredUCCSD(UCCSDAnsatz):
    def __init__(self, setting):
        self.setting = setting


@pytest.mark.parametrize(
    "kwargs, normalised",
    [
        ({"active_spaces": list(_SPECS)}, {"active_spaces": _SPECS}),
        (
            {"active_orbitals": [3, 1, 2]},
            {"active_orbitals": (3, 1, 2)},
        ),
        (
            {
                "n_active_orbitals": 4,
                "fragment_atoms": [[0], [1]],
                "local_spins": [2, -2],
            },
            {"fragment_atoms": ((0,), (1,)), "local_spins": (2, -2)},
        ),
    ],
    ids=["active-spaces", "active-orbitals", "fragment-atoms-and-spins"],
)
def test_fragmentation_config_stores_sequences_as_hashable_tuples(kwargs, normalised):
    config = FragmentationConfig(**kwargs)

    assert {name: getattr(config, name) for name in normalised} == normalised
    assert hash(config) == hash(FragmentationConfig(**kwargs))


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (
            {},
            "Pass exactly one of active_spaces (explicit fragment layout), "
            "n_active_orbitals (frontier selection), or active_orbitals "
            "(explicit MO indices).",
        ),
        (
            {"active_spaces": _SPECS, "local_spins": [0, 0]},
            "local_spins applies to automatic fragmentation only; active_spaces "
            "already fixes the fragment layout.",
        ),
        (
            {"active_spaces": _SPECS, "fragment_atoms": [[0], [1]]},
            "fragment_atoms applies to automatic fragmentation only; active_spaces "
            "already fixes the fragment layout.",
        ),
        (
            {"n_active_orbitals": 4, "local_spins": [0, 0]},
            "local_spins requires fragment_atoms: coupling-graph fragment order "
            "is not stable, so a positional spin list would not name a fragment.",
        ),
        (
            {"n_active_orbitals": 4, "fragment_atoms": [[0], [1]], "local_spins": [2]},
            "local_spins has 1 entries but fragment_atoms names 2 fragments.",
        ),
        (
            {"n_active_orbitals": 1},
            "n_active_orbitals\n  Input should be greater than or equal to 2",
        ),
        (
            {"n_active_orbitals": 4, "max_orbitals_per_fragment": 1},
            "max_orbitals_per_fragment\n  Input should be greater than or equal to 2",
        ),
        (
            {"n_active_orbitals": 4, "max_orbitals_per_fragment": float("inf")},
            "max_orbitals_per_fragment\n  Input should be a finite number",
        ),
        (
            {"n_active_orbitals": 4, "coupling_threshold": -0.1},
            "coupling_threshold\n  Input should be greater than or equal to 0",
        ),
        (
            {"n_active_orbitals": 4, "coupling_threshold": float("nan")},
            "coupling_threshold\n  Input should be a finite number",
        ),
        (
            {"n_active_orbitals": 4, "coupling_threshold": 1.5},
            "coupling_threshold\n  Input should be less than or equal to 1",
        ),
        (
            {"active_orbitals": (0.9, 1.9)},
            "active_orbitals.0\n  Input should be a valid integer, got a number "
            "with a fractional part",
        ),
        (
            {"active_spaces": ["0,1"]},
            "active_spaces.0\n  Input should be a dictionary or an instance of "
            "FragmentSpec",
        ),
        (
            {"active_spaces": _SPECS, "max_orbitals_per_fragment": 4},
            "max_orbitals_per_fragment shapes coupling-graph fragments only; "
            "active_spaces already fixes the fragment layout.",
        ),
        (
            {
                "n_active_orbitals": 4,
                "fragment_atoms": [[0], [1]],
                "coupling_threshold": 1e-3,
            },
            "coupling_threshold shapes coupling-graph fragments only; "
            "fragment_atoms already fixes the fragment layout.",
        ),
    ],
    ids=[
        "no-selector",
        "spins-with-layout",
        "atoms-with-layout",
        "spins-without-atoms",
        "spins-length",
        "too-few-active-orbitals",
        "too-small-fragments",
        "infinite-fragment-cap",
        "negative-threshold",
        "nan-threshold",
        "threshold-above-one",
        "fractional-orbital-index",
        "non-spec-active-space",
        "fragment-cap-with-layout",
        "threshold-with-atoms",
    ],
)
def test_fragmentation_config_rejects_invalid_settings(kwargs, message):
    with pytest.raises(ValidationError, match=re.escape(message)):
        FragmentationConfig(**kwargs)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"n_batches": 0}, "n_batches\n  Input should be greater than or equal to 1"),
        (
            {"batch_size": 0},
            "batch_size\n  Input should be greater than or equal to 1",
        ),
        (
            {"batch_size": 1.5},
            "batch_size\n  Input should be a valid integer, got a number with a "
            "fractional part",
        ),
        (
            {"n_recovery_iterations": 0},
            "n_recovery_iterations\n  Input should be greater than or equal to 1",
        ),
        (
            {"lambda_penalty": -1.0},
            "lambda_penalty\n  Input should be greater than or equal to 0",
        ),
        (
            {"lambda_penalty": float("nan")},
            "lambda_penalty\n  Input should be a finite number",
        ),
        (
            {"lambda_penalty": float("inf")},
            "lambda_penalty\n  Input should be a finite number",
        ),
        (
            {"carryover_cutoff": 0.0},
            "carryover_cutoff\n  Input should be greater than 0",
        ),
        ({"carryover_cutoff": 1.0}, "carryover_cutoff\n  Input should be less than 1"),
        (
            {"carryover_mapping": "nearest"},
            "carryover_mapping\n  Input should be 'assignment' or 'argmax'",
        ),
        (
            {"max_carryover": 5, "carryover_cutoff": None},
            "max_carryover needs carryover_cutoff to be set.",
        ),
        (
            {"max_carryover": 0},
            "max_carryover\n  Input should be greater than or equal to 1",
        ),
        (
            {"max_dim": (1, 2, 3)},
            "Tuple should have at most 2 items after validation, not 3",
        ),
        ({"max_dim": 0}, "max_dim must be at least 1; got 0."),
        (
            {"recovery_energy_tol": -1.0},
            "recovery_energy_tol\n  Input should be greater than or equal to 0",
        ),
        (
            {"recovery_occupancies_tol": -1.0},
            "recovery_occupancies_tol\n  Input should be greater than or equal to 0",
        ),
        (
            {"recovery_energy_tol": float("nan")},
            "recovery_energy_tol\n  Input should be a finite number",
        ),
        ({"states": {}}, "states must request at least one state."),
        (
            {"states": {0: 1}},
            "states multiplicities and root counts must be at least 1; got {0: 1}.",
        ),
        (
            {"states": {1: 0}},
            "states multiplicities and root counts must be at least 1; got {1: 0}.",
        ),
        (
            {"states": {1: 1.5}},
            "states.1\n  Input should be a valid integer, got a number with a "
            "fractional part",
        ),
        ({"states": [(1, 2)]}, "states\n  Input should be a valid dictionary"),
        (
            {"extension": "triples"},
            "extension\n  Input should be 'auto', 'singles' or 'doubles'",
        ),
        ({"extension_cutoff": 1.0}, "extension_cutoff\n  Input should be less than 1"),
        (
            {"carryover_cutoff": None, "carryover_mapping": "argmax"},
            "carryover_mapping needs carryover_cutoff to be set.",
        ),
        (
            {"extension": None, "extension_cutoff": 1e-2},
            "extension_cutoff needs extension to be set.",
        ),
        (
            {"states": {1: 2}, "symmetrize_spin": False},
            "symmetrize_spin=False contradicts states, which always pools the "
            "alpha and beta halves. Leave symmetrize_spin unset.",
        ),
    ],
    ids=[
        "n-batches",
        "batch-size",
        "fractional-batch-size",
        "recovery-iterations",
        "lambda-penalty",
        "nan-lambda-penalty",
        "infinite-lambda-penalty",
        "cutoff-zero",
        "cutoff-one",
        "mapping",
        "cap-without-cutoff",
        "cap-zero",
        "max-dim-length",
        "max-dim-zero",
        "energy-tol",
        "occupancies-tol",
        "nan-energy-tol",
        "states-empty",
        "states-multiplicity-zero",
        "states-count-zero",
        "states-fractional-count",
        "states-not-a-mapping",
        "extension-unknown",
        "extension-cutoff",
        "mapping-without-carryover",
        "extension-cutoff-without-extension",
        "unsymmetrized-states",
    ],
)
def test_sqd_config_rejects_invalid_settings(kwargs, message):
    with pytest.raises(ValidationError, match=re.escape(message)):
        SQDConfig(**kwargs)


def test_fragment_spec_rejects_a_set_of_orbitals():
    """A set's iteration order would decide which orbitals count as occupied."""
    with pytest.raises(
        ValidationError,
        match=re.escape(
            "orbitals must be an ordered sequence; a set loses the occupied-first "
            "order."
        ),
    ):
        FragmentSpec(orbitals={0, 1}, n_alpha=1, n_beta=1)


def test_sqd_config_stores_a_max_dim_pair_as_an_int_tuple():
    max_dim = SQDConfig(max_dim=[np.int64(4), 5]).max_dim

    assert max_dim == (4, 5)
    assert all(type(dim) is int for dim in max_dim)


def test_numpy_integers_keep_every_digit():
    """pydantic's own int validation rounds a large numpy integer through a
    float."""
    assert SQDConfig(batch_size=np.int64(2**62 + 1)).batch_size == 2**62 + 1


def test_sqd_config_resolves_unset_options_to_their_defaults():
    config = SQDConfig()

    assert config.resolved_carryover_mapping == "assignment"
    assert config.resolved_extension_cutoff == 1e-3
    assert SQDConfig(carryover_mapping="argmax").resolved_carryover_mapping == "argmax"
    assert SQDConfig(extension_cutoff=1e-2).resolved_extension_cutoff == 1e-2


def test_sqd_config_stores_states_sorted_by_multiplicity():
    assert list(SQDConfig(states={3: 1, 1: 2}).states.items()) == [(1, 2), (3, 1)]


@pytest.mark.parametrize(
    "build",
    [
        partial(VQEPreparation, ScipyOptimizer(ScipyMethod.COBYLA)),
        LBFGSOrbitalUpdate,
        SecondOrderOrbitalUpdate,
        StateAveragedOrbitalUpdate,
    ],
    ids=[
        "vqe-preparation",
        "full-orbital-solve",
        "second-order-orbital-solve",
        "state-averaged-orbital-solve",
    ],
)
def test_iteration_caps_must_be_positive(build):
    assert build(max_iterations=1).max_iterations == 1

    with pytest.raises(
        ValidationError,
        match=re.escape("max_iterations\n  Input should be greater than or equal to 1"),
    ):
        build(max_iterations=0)


@pytest.mark.parametrize(
    "weights, message",
    [
        (
            {(1, 0): 1.5, (1, 1): -0.5},
            "weights must be finite and non-negative; got {(1, 0): 1.5, (1, 1): -0.5}.",
        ),
        (
            {(1, 0): float("inf")},
            "weights must be finite and non-negative; got {(1, 0): inf}.",
        ),
        ({(1, 0): 0.5, (1, 1): 0.4}, "weights must sum to 1; got 0.9."),
    ],
    ids=["negative", "infinite", "not-normalised"],
)
def test_state_averaged_solve_rejects_invalid_weights(weights, message):
    with pytest.raises(ValidationError, match=re.escape(message)):
        StateAveragedOrbitalUpdate(weights=weights)


def test_state_averaged_solve_accepts_weights_rounded_off_their_sum():
    weights = {(1, 0): 0.4999999999, (1, 1): 0.5}

    assert StateAveragedOrbitalUpdate(weights=weights).weights == weights


def test_state_averaged_solve_records_weights_independently_of_key_order():
    """The checkpoint compares settings by their repr, so the same weights given
    in another order must not read as a different configuration."""
    forward = StateAveragedOrbitalUpdate(weights={(1, 0): 0.5, (3, 0): 0.5})
    backward = StateAveragedOrbitalUpdate(weights={(3, 0): 0.5, (1, 0): 0.5})

    assert repr(forward) == repr(backward)


def test_state_averaged_solve_weights_states_equally_by_default():
    assert StateAveragedOrbitalUpdate()._weights_for({1: 2, 3: 1}) == {
        (1, 0): pytest.approx(1 / 3),
        (1, 1): pytest.approx(1 / 3),
        (3, 0): pytest.approx(1 / 3),
    }


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (
            {"optimizer": ScipyOptimizer(ScipyMethod.COBYLA), "ansatz": "uccsd"},
            "ansatz\n  Input should be an instance of Ansatz",
        ),
        (
            {"optimizer": ScipyOptimizer},
            "optimizer\n  Input should be an instance of Optimizer",
        ),
    ],
    ids=["ansatz", "optimizer-class"],
)
def test_vqe_preparation_rejects_components_of_the_wrong_kind(kwargs, message):
    with pytest.raises(ValidationError, match=re.escape(message)):
        VQEPreparation(**kwargs)


def test_vqe_preparation_checkpoint_record_names_its_component_classes():
    preparation = VQEPreparation(
        ScipyOptimizer(ScipyMethod.COBYLA), ansatz=UCCSDAnsatz(), max_iterations=3
    )

    record = preparation._checkpoint_record()

    assert record[0] == "VQEPreparation"
    assert record[1] == "ScipyOptimizer"
    assert record[3] == "UCCSDAnsatz"
    assert record[-1] == 3


def test_vqe_checkpoint_record_tracks_optimizer_and_ansatz_settings():
    def record(method, setting):
        return VQEPreparation(
            ScipyOptimizer(method), ansatz=_ConfiguredUCCSD(setting)
        )._checkpoint_record()

    assert record(ScipyMethod.COBYLA, 1) == record(ScipyMethod.COBYLA, 1)
    assert record(ScipyMethod.COBYLA, 1) != record(ScipyMethod.L_BFGS_B, 1)
    assert record(ScipyMethod.COBYLA, 1) != record(ScipyMethod.COBYLA, 2)
