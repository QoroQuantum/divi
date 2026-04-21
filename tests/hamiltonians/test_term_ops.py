# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for term-manipulation primitives in divi.hamiltonians._term_ops."""

import numpy as np
import pennylane as qml
import pytest

from divi.hamiltonians import (
    _clean_hamiltonian,
    _hamiltonian_term_count,
    _sort_hamiltonian_terms,
)


@pytest.fixture
def simple_hamiltonian():
    """Three-term Hamiltonian (coefficients 1.0, 2.0, 3.0)."""
    return (1.0 * qml.Z(0) + 2.0 * qml.Z(1) + 3.0 * (qml.Z(0) @ qml.Z(1))).simplify()


class TestHamiltonianTermCount:
    """Tests for _hamiltonian_term_count helper."""

    def test_multi_term_sum(self, simple_hamiltonian):
        """Multi-term Sum returns correct count."""
        assert _hamiltonian_term_count(simple_hamiltonian) == 3

    def test_single_term_sprod(self):
        """Single-term SProd (e.g. 0.5*Z(0)) returns 1."""
        single_sprod = 0.5 * qml.Z(0)
        assert _hamiltonian_term_count(single_sprod) == 1

    def test_single_term_bare_pauli(self):
        """Single bare Pauli operator returns 1."""
        assert _hamiltonian_term_count(qml.Z(0)) == 1

    def test_empty_hamiltonian(self):
        """Empty Hamiltonian returns 0."""
        empty = qml.Hamiltonian([], [])
        assert _hamiltonian_term_count(empty) == 0


class TestCleanHamiltonian:
    """Tests for the _clean_hamiltonian helper."""

    @pytest.mark.parametrize(
        "hamiltonian, expected_op, expected_constant",
        [
            # Test case 1: Hamiltonian with no constant part
            (
                qml.sum(qml.s_prod(2, qml.PauliX(0)), qml.PauliZ(1)),
                qml.sum(qml.s_prod(2, qml.PauliX(0)), qml.PauliZ(1)),
                0.0,
            ),
            # Test case 2: Hamiltonian with only constant parts
            (
                qml.sum(
                    qml.s_prod(2.5, qml.Identity(0)), qml.s_prod(1.5, qml.Identity(1))
                ),
                qml.Hamiltonian([], []),
                4.0,
            ),
            # Test case 3: Mixed Hamiltonian
            (
                qml.sum(
                    qml.s_prod(2, qml.PauliX(0)),
                    qml.s_prod(3, qml.Identity(0)),
                    qml.PauliZ(1),
                ),
                qml.sum(qml.s_prod(2, qml.PauliX(0)), qml.PauliZ(1)),
                3.0,
            ),
            # Test case 4: Single Identity operator
            (qml.Identity(0), qml.Hamiltonian([], []), 1.0),
            # Test case 5: Single scaled Identity operator
            (qml.s_prod(5.0, qml.Identity(0)), qml.Hamiltonian([], []), 5.0),
            # Test case 6: Product of Identities
            (qml.prod(qml.Identity(0), qml.Identity(1)), qml.Hamiltonian([], []), 1.0),
            # Test case 7: Single non-constant operator
            (qml.PauliZ(0), qml.PauliZ(0), 0.0),
            # Test case 8: Empty Hamiltonian
            (qml.Hamiltonian([], []), qml.Hamiltonian([], []), 0.0),
        ],
    )
    def test_clean_hamiltonian_various_cases(
        self, hamiltonian, expected_op, expected_constant
    ):
        """
        Tests the clean_hamiltonian function with various scenarios.
        """
        new_hamiltonian, constant = _clean_hamiltonian(hamiltonian)

        assert np.isclose(constant, expected_constant)

        try:
            # qml.equal can be sensitive, so we compare terms for robustness
            new_coeffs, new_ops = new_hamiltonian.terms()
            exp_coeffs, exp_ops = expected_op.terms()

            assert len(new_coeffs) == len(exp_coeffs)

            if len(new_coeffs) > 0:
                # Create dictionaries for easy lookup and comparison
                new_map = {op.hash: coeff for coeff, op in zip(new_coeffs, new_ops)}
                exp_map = {op.hash: coeff for coeff, op in zip(exp_coeffs, exp_ops)}

                assert new_map.keys() == exp_map.keys()
                for key in new_map:
                    assert np.isclose(new_map[key], exp_map[key])
        except qml.operation.TermsUndefinedError:
            # Fallback for operators that don't have .terms()
            assert qml.equal(new_hamiltonian, expected_op)


class TestSortHamiltonianTerms:
    """Tests for _sort_hamiltonian_terms."""

    def test_absolute_order_sorts_by_literal_coefficient(self):
        """'absolute' sorts by the literal coefficient value (ascending)."""
        ham = qml.Hamiltonian([0.5, -0.3, 0.1], [qml.Z(0), qml.Z(1), qml.Z(2)])
        result = _sort_hamiltonian_terms(ham, order="absolute")
        coeffs, _ = result.terms()
        assert list(coeffs) == pytest.approx([-0.3, 0.1, 0.5])

    def test_magnitude_order_sorts_by_absolute_value(self):
        """'magnitude' sorts by abs(coefficient) (ascending)."""
        ham = qml.Hamiltonian([0.5, -0.3, 0.1], [qml.Z(0), qml.Z(1), qml.Z(2)])
        result = _sort_hamiltonian_terms(ham, order="magnitude")
        coeffs, _ = result.terms()
        assert [abs(c) for c in coeffs] == pytest.approx([0.1, 0.3, 0.5])

    def test_single_term_operator_passes_through(self):
        """A single PauliZ (not a Sum/Hamiltonian) is returned unchanged."""
        op = qml.PauliZ(0)
        result = _sort_hamiltonian_terms(op)
        assert qml.equal(result, op)

    def test_negative_coefficients_preserved(self):
        """Negative coefficients are preserved after sorting, not lost."""
        ham = qml.Hamiltonian([-2.0, 1.0], [qml.Z(0), qml.Z(1)])
        result = _sort_hamiltonian_terms(ham, order="absolute")
        coeffs, _ = result.terms()
        assert -2.0 in [float(c) for c in coeffs]
        assert 1.0 in [float(c) for c in coeffs]
