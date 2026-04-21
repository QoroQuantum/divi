# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for TrotterizationStrategy implementations (ExactTrotterization, QDrift)."""

import warnings

import numpy as np
import pennylane as qml
import pytest

from divi import hamiltonians
from divi.hamiltonians import ExactTrotterization, QDrift


@pytest.fixture
def simple_hamiltonian():
    """Hamiltonian with three terms (coefficients 1.0, 2.0, 3.0) for Trotterization tests."""
    return (1.0 * qml.Z(0) + 2.0 * qml.Z(1) + 3.0 * (qml.Z(0) @ qml.Z(1))).simplify()


class TestExactTrotterization:
    """Tests for ExactTrotterization strategy (public API and specified behavior)."""

    def test_both_keep_fraction_and_keep_top_n_raises(self):
        """At most one of keep_fraction or keep_top_n may be provided."""
        with pytest.raises(
            ValueError, match="At most one of keep_fraction or keep_top_n"
        ):
            ExactTrotterization(keep_fraction=0.5, keep_top_n=2)

    @pytest.mark.parametrize("keep_fraction", [-0.1, 0, 1.5])
    def test_keep_fraction_out_of_range_raises(self, keep_fraction):
        """keep_fraction must be in (0, 1]."""
        with pytest.raises(ValueError, match="keep_fraction must be in \\(0, 1\\]"):
            ExactTrotterization(keep_fraction=keep_fraction)

    @pytest.mark.parametrize("keep_top_n", [-1, 0, 0.5])
    def test_keep_top_n_invalid_raises(self, keep_top_n):
        """keep_top_n must be a positive integer (>= 1)."""
        with pytest.raises(ValueError, match="keep_top_n must be a positive integer"):
            ExactTrotterization(keep_top_n=keep_top_n)

    def test_no_truncation_returns_simplified_hamiltonian(self, simple_hamiltonian):
        """When both keep_fraction and keep_top_n are None, returns Hamiltonian unchanged."""
        result = ExactTrotterization().process_hamiltonian(simple_hamiltonian)
        assert qml.equal(result, simple_hamiltonian.simplify())

    @pytest.mark.parametrize(
        "strategy_kwargs,warn_match",
        [
            ({"keep_fraction": 1.0}, "keep_fraction is 1.0.*no truncation"),
            ({"keep_top_n": 10}, "keep_top_n is greater than or equal"),
        ],
        ids=["keep_fraction_one", "keep_top_n_exceeds_terms"],
    )
    def test_early_return_returns_full_and_warns(
        self, simple_hamiltonian, strategy_kwargs, warn_match
    ):
        """When keep_fraction=1.0 or keep_top_n >= terms, returns full Hamiltonian and warns."""
        with pytest.warns(UserWarning, match=warn_match):
            result = ExactTrotterization(**strategy_kwargs).process_hamiltonian(
                simple_hamiltonian
            )
        assert qml.equal(result, simple_hamiltonian.simplify())

    @pytest.mark.parametrize(
        "single_term_hamiltonian",
        [0.5 * qml.Z(0), qml.Z(0)],
        ids=["sprod", "bare_pauli"],
    )
    def test_single_term_hamiltonian_with_keep_top_n(self, single_term_hamiltonian):
        """Single-term operators (SProd, bare Pauli) work with keep_top_n; no len() error."""
        strategy = ExactTrotterization(keep_top_n=1)
        with pytest.warns(UserWarning, match="keep_top_n is greater than or equal"):
            result = strategy.process_hamiltonian(single_term_hamiltonian)
        assert qml.equal(result, single_term_hamiltonian.simplify())

    @pytest.mark.parametrize(
        "single_term_hamiltonian",
        [0.5 * qml.Z(0), qml.Z(0)],
        ids=["sprod", "bare_pauli"],
    )
    def test_single_term_hamiltonian_with_keep_fraction(self, single_term_hamiltonian):
        """Single-term operators work with keep_fraction; returns full operator."""
        strategy = ExactTrotterization(keep_fraction=0.5)
        result = strategy.process_hamiltonian(single_term_hamiltonian)
        assert qml.equal(result, single_term_hamiltonian.simplify())

    def test_constant_only_hamiltonian_raises(self):
        """Constant-only Hamiltonian raises ValueError; rejected at boundary."""
        constant_only = qml.Identity(0) * 5.0
        strategy = ExactTrotterization(keep_fraction=0.5)
        with pytest.raises(
            ValueError, match="Hamiltonian contains only constant terms"
        ):
            strategy.process_hamiltonian(constant_only)

    @pytest.mark.parametrize(
        "keep_top_n,expected",
        [
            (
                1,
                (3.0 * (qml.Z(0) @ qml.Z(1))).simplify(),
            ),
            (
                2,
                (2.0 * qml.Z(1) + 3.0 * (qml.Z(0) @ qml.Z(1))).simplify(),
            ),
        ],
        ids=["keep_top_n_1", "keep_top_n_2"],
    )
    def test_keep_top_n_keeps_largest_terms(
        self, simple_hamiltonian, keep_top_n, expected
    ):
        """keep_top_n keeps that many largest-magnitude terms (plus constant)."""
        result = ExactTrotterization(keep_top_n=keep_top_n).process_hamiltonian(
            simple_hamiltonian
        )
        coeffs, _ = result.terms()
        assert len(coeffs) == keep_top_n
        assert qml.equal(result, expected)

    def test_keep_fraction_reduces_term_count(self, simple_hamiltonian):
        """keep_fraction < 1 yields fewer terms; kept terms have total |coeff| >= fraction of full."""
        result = ExactTrotterization(keep_fraction=0.5).process_hamiltonian(
            simple_hamiltonian
        )
        full_coeffs, _ = simple_hamiltonian.terms()
        result_coeffs, _ = result.terms()
        assert len(result_coeffs) <= len(full_coeffs)
        # For keep_fraction=0.5, total |coeff| in simple_hamiltonian is 6; we keep terms summing to >= 3
        full_sum_abs = np.sum(np.abs(full_coeffs))
        result_sum_abs = np.sum(np.abs(result_coeffs))
        assert result_sum_abs >= 0.5 * full_sum_abs
        # With 0.5 we keep exactly the largest term (3.0*Z(0)@Z(1))
        expected = (3.0 * (qml.Z(0) @ qml.Z(1))).simplify()
        assert qml.equal(result, expected)

    def test_exact_trotterization_stateful_is_false(self):
        """ExactTrotterization reports stateful=False (cache is memoization only, not state)."""
        assert ExactTrotterization(keep_top_n=2).stateful is False

    @pytest.mark.parametrize(
        "strategy_kwargs",
        [{"keep_top_n": 2}, {"keep_fraction": 0.5}],
        ids=["keep_top_n", "keep_fraction"],
    )
    def test_exact_trotterization_caches_result(
        self, mocker, simple_hamiltonian, strategy_kwargs
    ):
        """Repeated process_hamiltonian with same Hamiltonian uses cache; returns same object."""
        spy = mocker.spy(hamiltonians._trotterization, "_sort_hamiltonian_terms")
        strategy = ExactTrotterization(**strategy_kwargs)
        result1 = strategy.process_hamiltonian(simple_hamiltonian)
        result2 = strategy.process_hamiltonian(simple_hamiltonian)
        assert result1 is result2
        assert spy.call_count == 1

    def test_exact_trotterization_different_hamiltonians_separate_cache(
        self, simple_hamiltonian
    ):
        """Different Hamiltonians get separate cache entries; each returns correct result."""
        strategy = ExactTrotterization(keep_top_n=1)
        ham2 = (4.0 * qml.Z(0) + 5.0 * qml.Z(1)).simplify()

        result1 = strategy.process_hamiltonian(simple_hamiltonian)
        result2 = strategy.process_hamiltonian(ham2)
        result3 = strategy.process_hamiltonian(simple_hamiltonian)

        # Cached: result1 and result3 are same object
        assert result1 is result3
        # Different Hamiltonians yield different results
        assert not qml.equal(result1, result2)
        # Correct truncation: simple_hamiltonian -> 3.0*Z(0)@Z(1); ham2 -> 5.0*Z(1)
        expected1 = (3.0 * (qml.Z(0) @ qml.Z(1))).simplify()
        expected2 = (5.0 * qml.Z(1)).simplify()
        assert qml.equal(result1, expected1)
        assert qml.equal(result2, expected2)

    @pytest.mark.parametrize(
        "strategy_kwargs,warn_match",
        [
            ({}, None),
            ({"keep_fraction": 1.0}, "keep_fraction is 1.0"),
            ({"keep_top_n": 10}, "keep_top_n is greater than or equal"),
        ],
        ids=["no_truncation", "keep_fraction_one", "keep_top_n_exceeds_terms"],
    )
    def test_exact_trotterization_no_cache_on_early_return(
        self, mocker, simple_hamiltonian, strategy_kwargs, warn_match
    ):
        """Early-return paths do not use cache; _sort_hamiltonian_terms never called."""
        spy = mocker.spy(hamiltonians._trotterization, "_sort_hamiltonian_terms")
        strategy = ExactTrotterization(**strategy_kwargs)
        for _ in range(2):
            if warn_match is not None:
                with pytest.warns(UserWarning, match=warn_match):
                    strategy.process_hamiltonian(simple_hamiltonian)
            else:
                strategy.process_hamiltonian(simple_hamiltonian)
        assert spy.call_count == 0


class TestQDrift:
    """Tests for QDrift strategy (public API and specified behavior)."""

    def test_invalid_sampling_strategy_raises(self):
        """sampling_strategy must be 'uniform' or 'weighted'."""
        with pytest.raises(ValueError, match="Invalid sampling_strategy"):
            QDrift(sampling_budget=5, sampling_strategy="invalid")

    def test_seed_non_int_raises(self):
        """seed must be an integer when provided."""
        with pytest.raises(ValueError, match="seed must be an integer"):
            QDrift(sampling_budget=5, seed=1.5)

    def test_all_none_returns_unchanged_and_warns(self, simple_hamiltonian):
        """When keep_fraction, keep_top_n and sampling_budget are all None, returns Hamiltonian unchanged."""
        with pytest.warns(
            UserWarning, match="Neither keep_fraction, keep_top_n, nor sampling_budget"
        ):
            result = QDrift().process_hamiltonian(simple_hamiltonian)
        assert qml.equal(result, simple_hamiltonian.simplify())

    def test_sample_budget_only_returns_valid_hamiltonian(self, simple_hamiltonian):
        """When only sample_budget is set (no keep_*), result is a valid operator with terms."""
        result = QDrift(sampling_budget=4, seed=42).process_hamiltonian(
            simple_hamiltonian
        )
        coeffs, ops = result.terms()
        assert len(coeffs) >= 1
        assert len(coeffs) == len(ops)

    def test_seed_gives_reproducible_result(self, simple_hamiltonian):
        """Same seed yields identical first sample across fresh instances."""
        s1 = QDrift(sampling_budget=5, seed=123, sampling_strategy="uniform")
        s2 = QDrift(sampling_budget=5, seed=123, sampling_strategy="uniform")
        r1 = s1.process_hamiltonian(simple_hamiltonian)
        r2 = s2.process_hamiltonian(simple_hamiltonian)
        assert qml.equal(r1, r2)

    def test_with_keep_fraction_and_sample_budget(self, simple_hamiltonian):
        """QDrift with keep_fraction and sample_budget returns keep terms + sampled terms."""
        result = QDrift(
            keep_fraction=0.5, sampling_budget=3, seed=0, sampling_strategy="weighted"
        ).process_hamiltonian(simple_hamiltonian)
        coeffs, _ = result.terms()
        # Kept terms (from 0.5 fraction) + 3 sampled from the rest
        assert len(coeffs) >= 3

    def test_keep_fraction_one_and_sample_budget_warns_no_terms_to_sample(
        self, simple_hamiltonian
    ):
        """When keep_fraction=1.0 and sample_budget set, all terms kept so no sampling; warns."""
        with pytest.warns(UserWarning) as record:
            result = QDrift(
                keep_fraction=1.0, sampling_budget=5, seed=0
            ).process_hamiltonian(simple_hamiltonian)
        # ExactTrotterization may warn "keep_fraction is 1.0..."; QDrift warns "no terms left to sample"
        messages = [str(w.message) for w in record]
        assert any("no terms left to sample" in m for m in messages)
        assert qml.equal(result, simple_hamiltonian)

    def test_sample_budget_none_equivalent_to_exact_trotterization(
        self, simple_hamiltonian
    ):
        """When sample_budget is None but keep_fraction set, result equals ExactTrotterization."""
        with pytest.warns(UserWarning, match="sampling_budget is not set"):
            qdrift = QDrift(keep_fraction=0.5)
        exact = ExactTrotterization(keep_fraction=0.5)
        qdrift_result = qdrift.process_hamiltonian(simple_hamiltonian)
        exact_result = exact.process_hamiltonian(simple_hamiltonian)
        assert qml.equal(qdrift_result, exact_result)

    def test_qdrift_caches_keep_hamiltonian(self, mocker, simple_hamiltonian):
        """Repeated process_hamiltonian with same Hamiltonian uses cache; ExactTrotterization called once."""
        spy = mocker.spy(ExactTrotterization, "process_hamiltonian")
        strategy = QDrift(
            keep_fraction=0.5, sampling_budget=3, seed=42, sampling_strategy="weighted"
        )
        strategy.process_hamiltonian(simple_hamiltonian)
        strategy.process_hamiltonian(simple_hamiltonian)
        assert spy.call_count == 1

    def test_keep_fraction_one_warns_only_on_first_call(self, simple_hamiltonian):
        """'No terms left to sample' warning is emitted only on first call, not on cached calls."""
        strategy = QDrift(keep_fraction=1.0, sampling_budget=5, seed=0)
        with pytest.warns(UserWarning) as first_record:
            first_result = strategy.process_hamiltonian(simple_hamiltonian)
        first_messages = [str(w.message) for w in first_record]
        assert any("no terms left to sample" in m for m in first_messages)
        assert qml.equal(first_result, simple_hamiltonian)

        with warnings.catch_warnings(record=True) as second_record:
            warnings.simplefilter("always")
            second_result = strategy.process_hamiltonian(simple_hamiltonian)
        second_messages = [str(w.message) for w in second_record]
        assert not any("no terms left to sample" in m for m in second_messages)
        assert qml.equal(second_result, simple_hamiltonian)

    @pytest.mark.parametrize(
        "single_term_hamiltonian",
        [0.5 * qml.Z(0), qml.Z(0)],
        ids=["sprod", "bare_pauli"],
    )
    def test_single_term_hamiltonian_with_keep_top_n(self, single_term_hamiltonian):
        """Single-term operators work with QDrift keep_top_n; no len() error."""
        with pytest.warns(
            UserWarning,
            match="keep_top_n is greater than or equal|All terms were kept",
        ):
            result = QDrift(
                keep_top_n=1, sampling_budget=2, seed=42
            ).process_hamiltonian(single_term_hamiltonian)
        assert qml.equal(result, single_term_hamiltonian.simplify())

    @pytest.mark.parametrize(
        "single_term_hamiltonian",
        [0.5 * qml.Z(0), qml.Z(0)],
        ids=["sprod", "bare_pauli"],
    )
    def test_single_term_hamiltonian_with_sampling_budget_only(
        self, single_term_hamiltonian
    ):
        """Single-term operators work with QDrift sampling_budget only; returns term unchanged."""
        result = QDrift(sampling_budget=3, seed=42).process_hamiltonian(
            single_term_hamiltonian
        )
        assert qml.equal(result.simplify(), single_term_hamiltonian.simplify())

    def test_empty_hamiltonian_warns_and_returns_kept(self):
        """Empty to_sample_hamiltonian (no terms) warns and returns empty Hamiltonian."""
        empty = qml.Hamiltonian([], [])
        with pytest.warns(UserWarning, match="No terms to sample"):
            result = QDrift(sampling_budget=3, seed=42).process_hamiltonian(empty)
        assert qml.equal(result, empty)

    def test_qdrift_stateful_is_true(self):
        """QDrift reports stateful=True (caches intermediate results)."""
        assert QDrift(sampling_budget=5, seed=0).stateful is True

    def test_n_hamiltonians_per_iteration_less_than_one_raises(self):
        """n_hamiltonians_per_iteration must be >= 1."""
        with pytest.raises(
            ValueError, match="n_hamiltonians_per_iteration must be >= 1"
        ):
            QDrift(sampling_budget=5, n_hamiltonians_per_iteration=0)

    def test_rng_produces_different_samples_on_repeated_calls(self, simple_hamiltonian):
        """Instance RNG produces different Hamiltonian samples on repeated calls."""
        strategy = QDrift(sampling_budget=4, seed=42, sampling_strategy="uniform")
        r0 = strategy.process_hamiltonian(simple_hamiltonian)
        r1 = strategy.process_hamiltonian(simple_hamiltonian)
        r2 = strategy.process_hamiltonian(simple_hamiltonian)
        # With 3 terms and sample_budget=4, sampling with replacement can produce
        # different orderings; at least two of the three should differ
        results = [r0, r1, r2]
        assert not all(qml.equal(results[0], r) for r in results)
