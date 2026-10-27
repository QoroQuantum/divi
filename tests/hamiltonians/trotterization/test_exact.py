# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for ExactTrotterization."""

import pytest
from qiskit.quantum_info import SparsePauliOp

from divi.hamiltonians.trotterization import ExactTrotterization, TrotterSpec
from tests._helpers import exact_match
from tests.hamiltonians.trotterization._helpers import (
    KEEP_TOP_N_NO_OP,
    SIMPLE_HAMILTONIAN,
    single_term_hamiltonians,
)

_KEEP_FRACTION_NO_OP = exact_match(
    "keep_fraction is 1.0 (no truncation); returning the full Hamiltonian."
)


class TestExactTrotterization:
    """Tests for ExactTrotterization strategy (public API and specified behavior)."""

    def test_both_keep_fraction_and_keep_top_n_raises(self):
        """At most one of keep_fraction or keep_top_n may be provided."""
        with pytest.raises(
            ValueError,
            match=exact_match(
                "At most one of keep_fraction or keep_top_n may be provided."
            ),
        ):
            ExactTrotterization(keep_fraction=0.5, keep_top_n=2)

    @pytest.mark.parametrize("keep_fraction", [-0.1, 0, 1.5])
    def test_keep_fraction_out_of_range_raises(self, keep_fraction):
        """keep_fraction must be in (0, 1]."""
        with pytest.raises(
            ValueError,
            match=exact_match(f"keep_fraction must be in (0, 1], got {keep_fraction}"),
        ):
            ExactTrotterization(keep_fraction=keep_fraction)

    @pytest.mark.parametrize("keep_top_n", [-1, 0, 0.5])
    def test_keep_top_n_invalid_raises(self, keep_top_n):
        """keep_top_n must be a positive integer (>= 1)."""
        with pytest.raises(
            ValueError,
            match=exact_match(
                f"keep_top_n must be a positive integer (>= 1), got {keep_top_n}"
            ),
        ):
            ExactTrotterization(keep_top_n=keep_top_n)

    def test_no_truncation_returns_simplified_hamiltonian(self, simple_hamiltonian):
        """When both keep_fraction and keep_top_n are None, returns Hamiltonian unchanged."""
        result = ExactTrotterization().process_hamiltonian(
            TrotterSpec(simple_hamiltonian)
        )
        assert result.effective_hamiltonian.simplify() == simple_hamiltonian.simplify()

    @pytest.mark.parametrize(
        "strategy_kwargs, warn_match, hamiltonian",
        [
            ({"keep_fraction": 1.0}, _KEEP_FRACTION_NO_OP, SIMPLE_HAMILTONIAN),
            ({"keep_top_n": 10}, KEEP_TOP_N_NO_OP, SIMPLE_HAMILTONIAN),
            ({"keep_top_n": 1}, KEEP_TOP_N_NO_OP, 0.5 * SparsePauliOp("Z")),
            ({"keep_top_n": 1}, KEEP_TOP_N_NO_OP, SparsePauliOp("Z")),
        ],
        ids=[
            "keep_fraction_one",
            "keep_top_n_exceeds_terms",
            "single_term_scaled",
            "single_term_unit",
        ],
    )
    def test_early_return_returns_full_and_warns(
        self, strategy_kwargs, warn_match, hamiltonian
    ):
        """When keep_fraction=1.0 or keep_top_n >= terms, returns full Hamiltonian and warns."""
        with pytest.warns(UserWarning, match=warn_match):
            result = ExactTrotterization(**strategy_kwargs).process_hamiltonian(
                TrotterSpec(hamiltonian)
            )
        assert result.effective_hamiltonian.simplify() == hamiltonian.simplify()

    @single_term_hamiltonians
    def test_single_term_hamiltonian_with_keep_fraction(self, single_term_spo):
        """Single-term operators work with keep_fraction; returns full operator."""
        strategy = ExactTrotterization(keep_fraction=0.5)
        result = strategy.process_hamiltonian(TrotterSpec(single_term_spo))
        assert result.effective_hamiltonian.simplify() == single_term_spo.simplify()

    @pytest.mark.parametrize(
        "hamiltonian, keep_top_n, expected_terms",
        [
            # Qiskit big-endian labels: rightmost char is qubit 0, so
            # Z(1) -> "ZI" and Z(0)@Z(1) -> "ZZ".
            (SIMPLE_HAMILTONIAN, 1, [("ZZ", 3.0)]),
            (SIMPLE_HAMILTONIAN, 2, [("ZI", 2.0), ("ZZ", 3.0)]),
            (
                SparsePauliOp.from_list([("IZ", 1.0), ("ZI", -5.0), ("ZZ", 2.0)]),
                1,
                [("ZI", -5.0)],
            ),
        ],
        ids=["keep_top_n_1", "keep_top_n_2", "ranks_by_magnitude_not_sign"],
    )
    def test_keep_top_n_keeps_largest_terms(
        self, hamiltonian, keep_top_n, expected_terms
    ):
        """keep_top_n keeps that many largest-magnitude terms (plus constant)."""
        result = ExactTrotterization(keep_top_n=keep_top_n).process_hamiltonian(
            TrotterSpec(hamiltonian)
        )
        expected = SparsePauliOp.from_list(expected_terms)
        assert result.effective_hamiltonian.size == keep_top_n
        assert result.effective_hamiltonian.simplify() == expected.simplify()

    @pytest.mark.parametrize(
        "keep_fraction, expected_terms",
        [(0.5, [("ZZ", 3.0)]), (0.8, [("ZI", 2.0), ("ZZ", 3.0)])],
        ids=["largest_term_only", "several_terms"],
    )
    def test_keep_fraction_keeps_the_largest_terms_covering_the_fraction(
        self, simple_hamiltonian, keep_fraction, expected_terms
    ):
        result = ExactTrotterization(keep_fraction=keep_fraction).process_hamiltonian(
            TrotterSpec(simple_hamiltonian)
        )
        assert (
            result.effective_hamiltonian.simplify()
            == SparsePauliOp.from_list(expected_terms).simplify()
        )

    def test_truncation_keeps_the_constant(self, simple_hamiltonian):
        hamiltonian = simple_hamiltonian + SparsePauliOp("II", 1.0)
        result = ExactTrotterization(keep_top_n=1).process_hamiltonian(
            TrotterSpec(hamiltonian)
        )
        assert result.effective_hamiltonian.equiv(
            SparsePauliOp.from_list([("ZZ", 3.0), ("II", 1.0)])
        )

    def test_exact_trotterization_has_no_mutable_call_state(self, simple_hamiltonian):
        strategy = ExactTrotterization(keep_top_n=1)
        ham2 = SparsePauliOp.from_list([("IZ", 4.0), ("ZI", 5.0)])

        result1 = strategy.process_hamiltonian(TrotterSpec(simple_hamiltonian))
        result2 = strategy.process_hamiltonian(TrotterSpec(ham2))
        result3 = strategy.process_hamiltonian(TrotterSpec(simple_hamiltonian))

        assert result1 is not result3
        assert (
            result1.effective_hamiltonian.simplify()
            != result2.effective_hamiltonian.simplify()
        )
        assert (
            result1.effective_hamiltonian.simplify()
            == SparsePauliOp.from_list([("ZZ", 3.0)]).simplify()
        )
        assert (
            result2.effective_hamiltonian.simplify()
            == SparsePauliOp.from_list([("ZI", 5.0)]).simplify()
        )
