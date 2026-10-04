# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the LASSQD configuration objects' validation and normalisation."""

from functools import partial

import pytest

from divi.qprog.algorithms import UCCSDAnsatz
from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer
from divi.qprog.workflows._lassqd._config import (
    FragmentationConfig,
    FullOrbitalSolve,
    SecondOrderOrbitalSolve,
    VQEPreparation,
)
from divi.qprog.workflows._lassqd._sqd import SQDConfig
from divi.qprog.workflows._lassqd._state import FragmentSpec
from tests._helpers import exact_match

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
            "depends on max_orbitals_per_fragment, coupling_threshold and the "
            "localisation RNG, so a positional spin list would not name a stable "
            "fragment.",
        ),
        (
            {"n_active_orbitals": 4, "fragment_atoms": [[0], [1]], "local_spins": [2]},
            "local_spins has 1 entries but fragment_atoms names 2 fragments.",
        ),
        (
            {"n_active_orbitals": 1},
            "n_active_orbitals must be at least 2, one occupied and one virtual; "
            "got 1.",
        ),
        (
            {"n_active_orbitals": 4, "max_orbitals_per_fragment": 1},
            "max_orbitals_per_fragment must be at least 2, since a fragment needs "
            "an occupied and a virtual orbital; got 1.",
        ),
        (
            {"n_active_orbitals": 4, "coupling_threshold": -0.1},
            "coupling_threshold must be non-negative; got -0.1.",
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
        "negative-threshold",
    ],
)
def test_fragmentation_config_rejects_invalid_settings(kwargs, message):
    with pytest.raises(ValueError, match=exact_match(message)):
        FragmentationConfig(**kwargs)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"n_batches": 0}, "n_batches must be at least 1; got 0."),
        ({"batch_size": 0}, "batch_size must be at least 1; got 0."),
        (
            {"n_recovery_iterations": 0},
            "n_recovery_iterations must be at least 1; got 0.",
        ),
        ({"lambda_penalty": -1.0}, "lambda_penalty must be non-negative; got -1.0."),
        ({"carryover_cutoff": 0.0}, "carryover_cutoff must be positive; got 0.0."),
        (
            {"carryover_cutoff": 1.0},
            "carryover_cutoff must be below 1: it is a fraction of the largest "
            "coefficient, which none exceeds, so 1.0 would retain nothing. Use "
            "None to turn carryover off.",
        ),
        (
            {"carryover_mapping": "nearest"},
            "carryover_mapping must be 'assignment' or 'argmax'; got 'nearest'.",
        ),
        (
            {"max_carryover": 5, "carryover_cutoff": None},
            "max_carryover caps what carryover retains, so it needs "
            "carryover_cutoff to be set.",
        ),
        ({"max_carryover": 0}, "max_carryover must be at least 1; got 0."),
        (
            {"max_dim": (1, 2, 3)},
            "max_dim takes one integer or an (alpha, beta) pair; got 3 entries.",
        ),
        ({"max_dim": 0}, "max_dim entries must be at least 1; got 0."),
        (
            {"recovery_energy_tol": -1.0},
            "recovery_energy_tol must be non-negative; got -1.0.",
        ),
        (
            {"recovery_occupancies_tol": -1.0},
            "recovery_occupancies_tol must be non-negative; got -1.0.",
        ),
    ],
    ids=[
        "n-batches",
        "batch-size",
        "recovery-iterations",
        "lambda-penalty",
        "cutoff-zero",
        "cutoff-one",
        "mapping",
        "cap-without-cutoff",
        "cap-zero",
        "max-dim-length",
        "max-dim-zero",
        "energy-tol",
        "occupancies-tol",
    ],
)
def test_sqd_config_rejects_invalid_settings(kwargs, message):
    with pytest.raises(ValueError, match=exact_match(message)):
        SQDConfig(**kwargs)


@pytest.mark.parametrize(
    "build",
    [
        partial(VQEPreparation, ScipyOptimizer(ScipyMethod.COBYLA)),
        FullOrbitalSolve,
        SecondOrderOrbitalSolve,
    ],
    ids=["vqe-preparation", "full-orbital-solve", "second-order-orbital-solve"],
)
def test_iteration_caps_accept_one_and_reject_zero(build):
    assert build(max_iterations=1).max_iterations == 1

    with pytest.raises(
        ValueError, match=exact_match("max_iterations must be at least 1; got 0.")
    ):
        build(max_iterations=0)


def test_vqe_preparation_names_a_non_ansatz_by_type():
    with pytest.raises(
        TypeError,
        match=exact_match("ansatz must be an Ansatz instance; got str."),
    ):
        VQEPreparation(ScipyOptimizer(ScipyMethod.COBYLA), ansatz="uccsd")


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
