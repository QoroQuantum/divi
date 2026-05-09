# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for TrotterizationStrategy implementations (ExactTrotterization, QDrift)."""

import warnings

import numpy as np
import pennylane as qp
import pytest
from qiskit.quantum_info import SparsePauliOp

from divi import hamiltonians
from divi.hamiltonians import ExactTrotterization, QDrift
from divi.hamiltonians._term_ops import _to_spo
from divi.hamiltonians._trotterization import _STRATEGY_CACHE_MAXSIZE


def _spo(operator: qp.operation.Operator) -> SparsePauliOp:
    """Build the SPO of a PennyLane operator with canonical 0..N-1 wires."""
    return _to_spo(operator)


def _spo_eq(
    actual: SparsePauliOp, expected: qp.operation.Operator | SparsePauliOp
) -> bool:
    """Equality up to ``simplify()`` for either SPO or PL-expressed expected."""
    expected_spo = (
        expected if isinstance(expected, SparsePauliOp) else _to_spo(expected)
    )
    return actual.simplify() == expected_spo.simplify()


@pytest.fixture
def simple_hamiltonian() -> SparsePauliOp:
    """Three-term SPO over 2 qubits (coeffs 1.0, 2.0, 3.0)."""
    # Qiskit big-endian: rightmost char is qubit 0.
    return SparsePauliOp.from_list([("IZ", 1.0), ("ZI", 2.0), ("ZZ", 3.0)])


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
        assert _spo_eq(result, simple_hamiltonian)

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
        assert _spo_eq(result, simple_hamiltonian)

    @pytest.mark.parametrize(
        "single_term_pl",
        [0.5 * qp.Z(0), qp.Z(0)],
        ids=["sprod", "bare_pauli"],
    )
    def test_single_term_hamiltonian_with_keep_top_n(self, single_term_pl):
        """Single-term operators work with keep_top_n; no len() error."""
        single_term_spo = _spo(single_term_pl)
        strategy = ExactTrotterization(keep_top_n=1)
        with pytest.warns(UserWarning, match="keep_top_n is greater than or equal"):
            result = strategy.process_hamiltonian(single_term_spo)
        assert _spo_eq(result, single_term_spo)

    @pytest.mark.parametrize(
        "single_term_pl",
        [0.5 * qp.Z(0), qp.Z(0)],
        ids=["sprod", "bare_pauli"],
    )
    def test_single_term_hamiltonian_with_keep_fraction(self, single_term_pl):
        """Single-term operators work with keep_fraction; returns full operator."""
        single_term_spo = _spo(single_term_pl)
        strategy = ExactTrotterization(keep_fraction=0.5)
        result = strategy.process_hamiltonian(single_term_spo)
        assert _spo_eq(result, single_term_spo)

    def test_constant_only_hamiltonian_raises(self):
        """Constant-only Hamiltonian raises ValueError; rejected at boundary."""
        constant_only = _spo(qp.Identity(0) * 5.0)
        strategy = ExactTrotterization(keep_fraction=0.5)
        with pytest.raises(
            ValueError, match="Hamiltonian contains only constant terms"
        ):
            strategy.process_hamiltonian(constant_only)

    @pytest.mark.parametrize(
        "keep_top_n,expected_terms",
        [
            # Qiskit big-endian labels: rightmost char is qubit 0, so
            # Z(1) -> "ZI" and Z(0)@Z(1) -> "ZZ".
            (1, [("ZZ", 3.0)]),
            (2, [("ZI", 2.0), ("ZZ", 3.0)]),
        ],
        ids=["keep_top_n_1", "keep_top_n_2"],
    )
    def test_keep_top_n_keeps_largest_terms(
        self, simple_hamiltonian, keep_top_n, expected_terms
    ):
        """keep_top_n keeps that many largest-magnitude terms (plus constant)."""
        result = ExactTrotterization(keep_top_n=keep_top_n).process_hamiltonian(
            simple_hamiltonian
        )
        expected = SparsePauliOp.from_list(expected_terms)
        assert result.size == keep_top_n
        assert _spo_eq(result, expected)

    def test_keep_fraction_reduces_term_count(self, simple_hamiltonian):
        """keep_fraction < 1 yields fewer terms; kept terms have total |coeff| >= fraction of full."""
        result = ExactTrotterization(keep_fraction=0.5).process_hamiltonian(
            simple_hamiltonian
        )
        full_sum_abs = float(np.sum(np.abs(simple_hamiltonian.coeffs.real)))
        result_sum_abs = float(np.sum(np.abs(result.coeffs.real)))
        assert result.size <= simple_hamiltonian.size
        assert result_sum_abs >= 0.5 * full_sum_abs
        # With 0.5 we keep exactly the largest term (3.0*Z(0)@Z(1))
        assert _spo_eq(result, SparsePauliOp.from_list([("ZZ", 3.0)]))

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
        spy = mocker.spy(hamiltonians._trotterization, "_sort_hamiltonian_terms_spo")
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
        ham2 = _spo((4.0 * qp.Z(0) + 5.0 * qp.Z(1)).simplify())

        result1 = strategy.process_hamiltonian(simple_hamiltonian)
        result2 = strategy.process_hamiltonian(ham2)
        result3 = strategy.process_hamiltonian(simple_hamiltonian)

        # Cached: result1 and result3 are same object
        assert result1 is result3
        # Different Hamiltonians yield different results
        assert result1.simplify() != result2.simplify()
        # Correct truncation
        assert _spo_eq(result1, SparsePauliOp.from_list([("ZZ", 3.0)]))
        assert _spo_eq(result2, SparsePauliOp.from_list([("ZI", 5.0)]))

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
        """Early-return paths do not use cache; the sort helper is never called."""
        spy = mocker.spy(hamiltonians._trotterization, "_sort_hamiltonian_terms_spo")
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
        assert _spo_eq(result, simple_hamiltonian)

    def test_sample_budget_only_returns_valid_hamiltonian(self, simple_hamiltonian):
        """When only sample_budget is set (no keep_*), result is a valid SPO with at least one term."""
        result = QDrift(sampling_budget=4, seed=42).process_hamiltonian(
            simple_hamiltonian
        )
        assert result.size >= 1

    def test_seed_gives_reproducible_result(self, simple_hamiltonian):
        """Same seed yields identical first sample across fresh instances."""
        s1 = QDrift(sampling_budget=5, seed=123, sampling_strategy="uniform")
        s2 = QDrift(sampling_budget=5, seed=123, sampling_strategy="uniform")
        r1 = s1.process_hamiltonian(simple_hamiltonian)
        r2 = s2.process_hamiltonian(simple_hamiltonian)
        assert r1.simplify() == r2.simplify()

    def test_with_keep_fraction_and_sample_budget(self, simple_hamiltonian):
        """QDrift with keep_fraction and sample_budget returns keep terms + sampled terms."""
        result = QDrift(
            keep_fraction=0.5, sampling_budget=3, seed=0, sampling_strategy="weighted"
        ).process_hamiltonian(simple_hamiltonian)
        # Kept terms (from 0.5 fraction) + 3 sampled from the rest
        assert result.size >= 3

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
        assert _spo_eq(result, simple_hamiltonian)

    def test_sample_budget_none_equivalent_to_exact_trotterization(
        self, simple_hamiltonian
    ):
        """When sample_budget is None but keep_fraction set, result equals ExactTrotterization."""
        with pytest.warns(UserWarning, match="sampling_budget is not set"):
            qdrift = QDrift(keep_fraction=0.5)
        exact = ExactTrotterization(keep_fraction=0.5)
        qdrift_result = qdrift.process_hamiltonian(simple_hamiltonian)
        exact_result = exact.process_hamiltonian(simple_hamiltonian)
        assert qdrift_result.simplify() == exact_result.simplify()

    def test_qdrift_caches_keep_hamiltonian(self, mocker, simple_hamiltonian):
        """Repeated process_hamiltonian with same Hamiltonian uses cache; truncation runs once."""
        spy = mocker.spy(ExactTrotterization, "_truncate_spo")
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
        assert _spo_eq(first_result, simple_hamiltonian)

        with warnings.catch_warnings(record=True) as second_record:
            warnings.simplefilter("always")
            second_result = strategy.process_hamiltonian(simple_hamiltonian)
        second_messages = [str(w.message) for w in second_record]
        assert not any("no terms left to sample" in m for m in second_messages)
        assert _spo_eq(second_result, simple_hamiltonian)

    @pytest.mark.parametrize(
        "single_term_pl",
        [0.5 * qp.Z(0), qp.Z(0)],
        ids=["sprod", "bare_pauli"],
    )
    def test_single_term_hamiltonian_with_keep_top_n(self, single_term_pl):
        """Single-term operators work with QDrift keep_top_n; no len() error."""
        single_term_spo = _spo(single_term_pl)
        with pytest.warns(
            UserWarning,
            match="keep_top_n is greater than or equal|All terms were kept",
        ):
            result = QDrift(
                keep_top_n=1, sampling_budget=2, seed=42
            ).process_hamiltonian(single_term_spo)
        assert _spo_eq(result, single_term_spo)

    @pytest.mark.parametrize(
        "single_term_pl",
        [0.5 * qp.Z(0), qp.Z(0)],
        ids=["sprod", "bare_pauli"],
    )
    def test_single_term_hamiltonian_with_sampling_budget_only(self, single_term_pl):
        """Single-term operators work with QDrift sampling_budget only; returns term unchanged."""
        single_term_spo = _spo(single_term_pl)
        result = QDrift(sampling_budget=3, seed=42).process_hamiltonian(single_term_spo)
        assert _spo_eq(result, single_term_spo)

    def test_empty_hamiltonian_warns_and_returns_kept(self):
        """Empty to_sample_hamiltonian (no terms) warns and returns empty Hamiltonian."""
        empty_spo = SparsePauliOp(["I"], coeffs=[0])[np.zeros(0, dtype=int)]
        with pytest.warns(UserWarning, match="No terms to sample"):
            result = QDrift(sampling_budget=3, seed=42).process_hamiltonian(empty_spo)
        assert result.size == 0

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
        results = [r0.simplify(), r1.simplify(), r2.simplify()]
        assert not all(r == results[0] for r in results)

    @pytest.mark.parametrize("strategy", ["uniform", "weighted"])
    def test_qdrift_expected_value_matches_input_hamiltonian(self, strategy):
        """E[L · sampled-row-product / channel] → H. Concretely: average the
        rescaled SPO over many seeds and confirm it converges to the input.
        """
        spo = SparsePauliOp.from_list([("X", 0.4), ("Z", 0.6)])
        budget = 20
        n_seeds = 500
        sum_coeffs = np.zeros(2, dtype=complex)  # [X, Z]
        for seed in range(n_seeds):
            sampled = (
                QDrift(sampling_budget=budget, seed=seed, sampling_strategy=strategy)
                .process_hamiltonian(spo)
                .simplify()
            )
            label_to_coeff = dict(zip(sampled.paulis.to_labels(), sampled.coeffs.real))
            sum_coeffs[0] += label_to_coeff.get("X", 0.0)
            sum_coeffs[1] += label_to_coeff.get("Z", 0.0)
        # E[sampled SPO] = H. Within Monte Carlo error of a few %.
        assert sum_coeffs.real[0] / n_seeds == pytest.approx(0.4, abs=0.05)
        assert sum_coeffs.real[1] / n_seeds == pytest.approx(0.6, abs=0.05)

    @pytest.mark.parametrize("strategy", ["uniform", "weighted"])
    def test_qdrift_wide_spo_narrow_truncation_no_index_error(self, strategy):
        """A 5-qubit SPO truncated to 3 terms (one qubit becomes identity-only)
        must not raise ``IndexError`` downstream — regression guard for the
        wire-permutation issue surfaced during the SPO migration.
        """
        spo = SparsePauliOp.from_list(
            [
                ("ZZIII", 1.0),
                ("IIZZI", 2.0),
                ("IIIZZ", 3.0),
                ("ZIIZI", 0.5),
                ("IZIIZ", 0.4),
            ]
        )
        result = QDrift(
            keep_top_n=3, sampling_budget=2, seed=0, sampling_strategy=strategy
        ).process_hamiltonian(spo)
        assert result.size >= 1
        assert result.num_qubits == 5

    @pytest.mark.parametrize(
        "strategy_factory",
        [
            lambda: ExactTrotterization(keep_top_n=2),
            lambda: QDrift(
                keep_fraction=0.5,
                sampling_budget=2,
                seed=0,
                sampling_strategy="weighted",
            ),
        ],
        ids=["exact", "qdrift"],
    )
    def test_strategy_cache_is_bounded_by_lru_maxsize(self, strategy_factory):
        """Processing many distinct SPOs through the same strategy instance
        must not let the cache grow without bound."""
        strategy = strategy_factory()
        n_distinct = 100
        last_spo = None
        for i in range(n_distinct):
            last_spo = SparsePauliOp.from_list(
                [("ZZ", float(i + 1)), ("IZ", float(i + 2)), ("ZI", float(i + 3))]
            )
            strategy.process_hamiltonian(last_spo)

        assert len(strategy._cache) <= _STRATEGY_CACHE_MAXSIZE
        # Most-recent SPO is still cached (LRU correctness).
        assert id(last_spo) in strategy._cache
