# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the divi <-> SQD bitstring convention boundary.

Divi solution bitstrings index ``bs[k] == qubit k`` (verified empirically: a
3-qubit ``GenericLayerAnsatz([RYGate])`` with parameter 0 set to pi samples
``'100'``). Jordan-Wigner via OpenFermion interleaves spin-orbitals, so qubit
``2p`` is alpha of spatial orbital ``p`` and ``2p + 1`` is beta. SQD's solver
expects blocked ``alpha_bits + beta_bits`` (``sqd_core.py:228-229``).
"""

import numpy as np
import pytest

from divi.qprog.workflows._lassqd._sqd import (
    deinterleave_spin_bitstring,
    interleave_spin_bitstring,
    probs_to_sqd_bitstrings,
)


def test_deinterleave_hand_checked_case():
    """qubit0=alpha0=1, qubit1=beta0=0, qubit2=alpha1=0, qubit3=beta1=1."""
    assert deinterleave_spin_bitstring("1001", n_orb=2) == "1001"
    # Distinguish the identity coincidence above with an asymmetric case.
    # qubits: a0=1 b0=1 a1=0 b1=0  ->  alpha "10", beta "10"
    assert deinterleave_spin_bitstring("1100", n_orb=2) == "1010"


def test_deinterleave_three_orbitals():
    # qubits a0 b0 a1 b1 a2 b2 = 1 0 0 1 1 1
    # alpha = q0 q2 q4 = "101";  beta = q1 q3 q5 = "011"
    assert deinterleave_spin_bitstring("100111", n_orb=3) == "101011"


def test_interleave_is_the_inverse():
    rng = np.random.default_rng(0)
    for n_orb in (1, 2, 3, 5):
        for _ in range(20):
            bits = "".join(rng.choice(["0", "1"]) for _ in range(2 * n_orb))
            assert (
                interleave_spin_bitstring(
                    deinterleave_spin_bitstring(bits, n_orb), n_orb
                )
                == bits
            )


def test_deinterleave_rejects_wrong_width():
    with pytest.raises(ValueError, match="width"):
        deinterleave_spin_bitstring("101", n_orb=2)


def test_probs_conversion_preserves_total_probability():
    probs = {"1100": 0.7, "0011": 0.3}
    converted = probs_to_sqd_bitstrings(probs, n_orb=2)
    assert sum(converted.values()) == pytest.approx(1.0)
    assert converted == {"1010": 0.7, "0101": 0.3}


def test_probs_conversion_preserves_particle_counts():
    """Checks the exact blocked mapping, not just popcount: a popcount-only
    assertion is invariant under any permutation bug this conversion could
    have, including one that mixes up which alpha/beta bits land where."""
    probs = {"1100": 0.5, "1001": 0.5}
    assert probs_to_sqd_bitstrings(probs, n_orb=2) == {"1010": 0.5, "1001": 0.5}
