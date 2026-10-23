# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for LASSQD's subspace CI state tracking across rounds."""

import numpy as np
import pytest
from pyscf.fci import cistring

from divi.qprog.workflows.lassqd._integrals import AOIntegrals
from divi.qprog.workflows.lassqd._state import _SampledSubspace
from divi.qprog.workflows.lassqd._subspace import _root_overlaps, track_states
from tests.qprog.workflows.lassqd._helpers import h4_chain_mean_field  # noqa: F401


def _h4_full_subspace(mo_coeff, labels=None):
    """H4's whole 36-determinant space over ``mo_coeff``, three states."""
    strings = tuple(
        format(string, "04b")[::-1] for string in cistring.make_strings(range(4), 2)
    )
    return _SampledSubspace(
        mo_coeff, (strings, strings), {1: 2, 3: 1}, 1.0, labels=labels
    )


@pytest.mark.parametrize(
    "rotation",
    [np.eye(4), np.eye(4)[[2, 0, 3, 1]] * np.array([1.0, -1.0, -1.0, 1.0])],
    ids=["same-orbitals", "permuted-and-sign-flipped"],
)
def test_tracking_keeps_each_state_its_previous_label(h4_chain_mean_field, rotation):
    """A previous round that labelled its two singlets the other way round
    passes those labels on, however the active orbitals were reordered or
    rephased in between."""
    mo_coeff = np.asarray(h4_chain_mean_field.mo_coeff)
    swapped = {(1, 0): (1, 1), (1, 1): (1, 0), (3, 0): (3, 0)}

    labels = track_states(
        AOIntegrals.of(h4_chain_mean_field.mol),
        _h4_full_subspace(mo_coeff, swapped),
        _h4_full_subspace(mo_coeff @ rotation),
        0,
    )

    assert labels == swapped


def test_carried_ci_vectors_overlap_their_own_states_fully(h4_chain_mean_field):
    """Reordering and rephasing the active orbitals changes nothing physical,
    so each state overlaps itself with modulus 1 and the others not at all,
    which needs each carried determinant's sign right."""
    mo_coeff = np.asarray(h4_chain_mean_field.mo_coeff)
    rotation = np.eye(4)[[2, 0, 3, 1]] * np.array([1.0, -1.0, -1.0, 1.0])

    for _, _, now, overlaps in _root_overlaps(
        AOIntegrals.of(h4_chain_mean_field.mol),
        _h4_full_subspace(mo_coeff),
        _h4_full_subspace(mo_coeff @ rotation),
        0,
    ):
        np.testing.assert_allclose(overlaps, np.eye(len(now)), atol=1e-8)
