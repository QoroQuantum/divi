# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the LASSQD state objects and fragment validation."""

import numpy as np
import pytest

from divi.qprog.workflows._lassqd._state import (
    FragmentSpec,
    FragmentState,
    validate_fragment_specs,
)
from tests._helpers import exact_match

_PAIR = FragmentSpec(orbitals=(0, 1), n_alpha=1, n_beta=1)


def _pair_state(**kwargs):
    """A ``FragmentState`` on ``_PAIR`` with zero RDMs."""
    return FragmentState(
        spec=_PAIR, rdm1=np.zeros((2, 2)), rdm2=np.zeros((2,) * 4), **kwargs
    )


def test_fragment_state_stores_carried_strings_as_tuples():
    state = _pair_state(
        carried_alpha=["01", "10"], carried_beta=["10"], sampled_orbitals=np.eye(2)
    )

    assert state.carried_alpha == ("01", "10")
    assert state.carried_beta == ("10",)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (
            {"rdm1_alpha": np.zeros((2, 2))},
            "rdm1_alpha and rdm1_beta must be given together or not at all.",
        ),
        (
            {"carried_alpha": ["01"]},
            "Carried strings need sampled_orbitals, the basis they refer to.",
        ),
        (
            {"carried_beta": ["01"]},
            "Carried strings need sampled_orbitals, the basis they refer to.",
        ),
    ],
    ids=["one-spin-rdm", "alpha-strings-only", "beta-strings-only"],
)
def test_fragment_state_rejects_inconsistent_fields(kwargs, message):
    with pytest.raises(ValueError, match=exact_match(message)):
        _pair_state(**kwargs)


@pytest.mark.parametrize(
    "specs, n_orbitals_total, n_occupied, message",
    [
        ([], 4, 2, "At least one fragment is required."),
        (
            [FragmentSpec(orbitals=(0, 1), n_alpha=2, n_beta=2)],
            4,
            2,
            "Fragment 0 (orbitals (0, 1)) has no excitation available: n_alpha=2, "
            "n_beta=2 leave every spin channel of its 2 orbitals either empty or "
            "full, so there is no correlation for this fragment to capture.",
        ),
        (
            [FragmentSpec(orbitals=(0, 5), n_alpha=1, n_beta=1)],
            4,
            2,
            "Fragment 0 orbital 5 is out of range for a molecule with 4 orbitals.",
        ),
        (
            [_PAIR, FragmentSpec(orbitals=(1, 2), n_alpha=1, n_beta=1)],
            4,
            2,
            "Fragments 0 and 1 overlap on orbital 1. Fragments must be disjoint.",
        ),
        (
            [FragmentSpec(orbitals=(2, 0), n_alpha=1, n_beta=1)],
            4,
            2,
            "Fragment 0 lists virtual orbital 2 before occupied orbital 0. List "
            "each fragment's occupied orbitals first: its reference determinant "
            "fills them in order.",
        ),
        (
            [
                FragmentSpec(orbitals=(0, 2), n_alpha=1, n_beta=1),
                FragmentSpec(orbitals=(3, 4), n_alpha=1, n_beta=1),
            ],
            5,
            2,
            "Fragments declare 4 electrons but cover occupied orbitals holding 2.",
        ),
        (
            [FragmentSpec(orbitals=(0, 1, 2, 3), n_alpha=3, n_beta=1)],
            4,
            2,
            "Fragments declare 3 alpha and 1 beta electrons, a total Sz of 1.0, "
            "but the molecule's is 0.0. Individual fragments may be polarised; "
            "their total must match.",
        ),
    ],
    ids=[
        "empty",
        "no-excitation",
        "out-of-range",
        "overlap",
        "virtual-first",
        "electron-count",
        "net-spin",
    ],
)
def test_validate_fragment_specs_rejects_invalid_layouts(
    specs, n_orbitals_total, n_occupied, message
):
    with pytest.raises(ValueError, match=exact_match(message)):
        validate_fragment_specs(specs, n_orbitals_total, n_occupied)


@pytest.mark.parametrize(
    "specs, message",
    [
        (
            [FragmentSpec(orbitals=(0, 2), n_alpha=1, n_beta=1)],
            "Singly occupied orbitals [1] are in no fragment. The core must be "
            "closed-shell, so every singly occupied orbital has to be active.",
        ),
        (
            [FragmentSpec(orbitals=(0, 1, 2), n_alpha=1, n_beta=2)],
            "Fragments declare 1 alpha and 2 beta electrons, a total Sz of -0.5, "
            "but the molecule's is 0.5. Individual fragments may be polarised; "
            "their total must match.",
        ),
    ],
    ids=["singly-occupied-left-out", "wrong-net-spin"],
)
def test_an_open_shell_layout_must_hold_every_unpaired_electron(specs, message):
    """Two occupied orbitals with spin 1: orbital 0 doubly, orbital 1 singly
    occupied."""
    with pytest.raises(ValueError, match=exact_match(message)):
        validate_fragment_specs(specs, 4, 2, spin=1)


def test_an_open_shell_layout_counts_one_electron_per_singly_occupied_orbital():
    validate_fragment_specs(
        [FragmentSpec(orbitals=(0, 1, 2), n_alpha=2, n_beta=1)], 4, 2, spin=1
    )
