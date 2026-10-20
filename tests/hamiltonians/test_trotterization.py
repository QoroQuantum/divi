# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for TrotterizationStrategy implementations (ExactTrotterization, QDrift)."""

import itertools

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from scipy.linalg import expm

from divi.circuits._conversions import _QISKIT_TO_QASM2
from divi.hamiltonians import ExactTrotterization, QDrift
from divi.hamiltonians._term_ops import generate_empty_spo
from tests._helpers import equal_up_to_phase, exact_match, unitary

_KEEP_FRACTION_NO_OP = exact_match(
    "keep_fraction is 1.0 (no truncation); returning the full Hamiltonian."
)
_KEEP_TOP_N_NO_OP = exact_match(
    "keep_top_n is greater than or equal to the number of terms; "
    "returning the full Hamiltonian."
)
_SAMPLING_BUDGET_NOT_SET = exact_match(
    "sampling_budget is not set; only the kept terms will be applied, "
    "equivalent to ExactTrotterization."
)


# Qiskit big-endian: rightmost char is qubit 0.
_SIMPLE_HAMILTONIAN = SparsePauliOp.from_list([("IZ", 1.0), ("ZI", 2.0), ("ZZ", 3.0)])


@pytest.fixture
def simple_hamiltonian() -> SparsePauliOp:
    """Three-term SPO over 2 qubits (coeffs 1.0, 2.0, 3.0)."""
    return _SIMPLE_HAMILTONIAN


def _only_step(result) -> SparsePauliOp:
    """The draw of a single-step sampling result."""
    assert result.sampled_terms is not None
    (step_terms,) = result.sampled_terms
    return step_terms


single_term_hamiltonians = pytest.mark.parametrize(
    "single_term_spo",
    [
        0.5 * SparsePauliOp("Z"),
        SparsePauliOp("Z"),
    ],
    ids=["scaled", "unit"],
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
        result = ExactTrotterization().process_hamiltonian(simple_hamiltonian)
        assert result.effective_hamiltonian.simplify() == simple_hamiltonian.simplify()

    @pytest.mark.parametrize(
        "strategy_kwargs, warn_match, hamiltonian",
        [
            ({"keep_fraction": 1.0}, _KEEP_FRACTION_NO_OP, _SIMPLE_HAMILTONIAN),
            ({"keep_top_n": 10}, _KEEP_TOP_N_NO_OP, _SIMPLE_HAMILTONIAN),
            ({"keep_top_n": 1}, _KEEP_TOP_N_NO_OP, 0.5 * SparsePauliOp("Z")),
            ({"keep_top_n": 1}, _KEEP_TOP_N_NO_OP, SparsePauliOp("Z")),
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
                hamiltonian
            )
        assert result.effective_hamiltonian.simplify() == hamiltonian.simplify()

    @single_term_hamiltonians
    def test_single_term_hamiltonian_with_keep_fraction(self, single_term_spo):
        """Single-term operators work with keep_fraction; returns full operator."""
        strategy = ExactTrotterization(keep_fraction=0.5)
        result = strategy.process_hamiltonian(single_term_spo)
        assert result.effective_hamiltonian.simplify() == single_term_spo.simplify()

    @pytest.mark.parametrize(
        "hamiltonian, keep_top_n, expected_terms",
        [
            # Qiskit big-endian labels: rightmost char is qubit 0, so
            # Z(1) -> "ZI" and Z(0)@Z(1) -> "ZZ".
            (_SIMPLE_HAMILTONIAN, 1, [("ZZ", 3.0)]),
            (_SIMPLE_HAMILTONIAN, 2, [("ZI", 2.0), ("ZZ", 3.0)]),
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
            hamiltonian
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
            simple_hamiltonian
        )
        assert (
            result.effective_hamiltonian.simplify()
            == SparsePauliOp.from_list(expected_terms).simplify()
        )

    def test_truncation_keeps_the_constant(self, simple_hamiltonian):
        hamiltonian = simple_hamiltonian + SparsePauliOp("II", 1.0)
        result = ExactTrotterization(keep_top_n=1).process_hamiltonian(hamiltonian)
        assert result.effective_hamiltonian.equiv(
            SparsePauliOp.from_list([("ZZ", 3.0), ("II", 1.0)])
        )

    def test_exact_trotterization_has_no_mutable_call_state(self, simple_hamiltonian):
        strategy = ExactTrotterization(keep_top_n=1)
        ham2 = SparsePauliOp.from_list([("IZ", 4.0), ("ZI", 5.0)])

        result1 = strategy.process_hamiltonian(simple_hamiltonian)
        result2 = strategy.process_hamiltonian(ham2)
        result3 = strategy.process_hamiltonian(simple_hamiltonian)

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


@pytest.mark.parametrize(
    "strategy",
    [
        ExactTrotterization(keep_fraction=0.5),
        ExactTrotterization(keep_fraction=1.0),
        ExactTrotterization(keep_top_n=1),
        QDrift(keep_fraction=0.5, sampling_budget=2, seed=0),
        QDrift(keep_fraction=1.0, sampling_budget=2, seed=0),
        QDrift(keep_top_n=1, sampling_budget=2, seed=0),
    ],
    ids=[
        "exact_partial_fraction",
        "exact_full_fraction",
        "exact_top_n_covers_all",
        "qdrift_partial_fraction",
        "qdrift_full_fraction",
        "qdrift_top_n_covers_all",
    ],
)
def test_truncating_a_constant_only_hamiltonian_raises(strategy):
    """Every truncation setting rejects a constant-only Hamiltonian."""
    with pytest.raises(
        ValueError, match=exact_match("Hamiltonian contains only constant terms.")
    ):
        strategy.process_hamiltonian(5.0 * SparsePauliOp("I"))


@pytest.mark.parametrize(
    "build, field, value",
    [
        (lambda v: ExactTrotterization(keep_top_n=v), "keep_top_n", np.int64(3)),
        (
            lambda v: ExactTrotterization(keep_fraction=v),
            "keep_fraction",
            np.float32(0.5),
        ),
        (
            lambda v: QDrift(sampling_budget=v, seed=0),
            "sampling_budget",
            np.int64(2),
        ),
        (lambda v: QDrift(sampling_budget=2, seed=v), "seed", np.uint64(2**63 + 5)),
        (
            lambda v: QDrift(sampling_budget=2, seed=0, n_hamiltonians_per_iteration=v),
            "n_hamiltonians_per_iteration",
            np.int64(4),
        ),
    ],
    ids=[
        "keep_top_n",
        "keep_fraction",
        "sampling_budget",
        "seed",
        "n_hamiltonians_per_iteration",
    ],
)
def test_strategies_accept_numpy_scalars_and_store_plain_python(build, field, value):
    stored = getattr(build(value), field)

    assert stored == value
    assert type(stored) is (float if field == "keep_fraction" else int)


@pytest.mark.parametrize(
    "build, message",
    [
        (
            lambda: ExactTrotterization(keep_top_n=True),
            "keep_top_n must be a positive integer (>= 1), got True",
        ),
        (
            lambda: ExactTrotterization(keep_fraction=float("nan")),
            "keep_fraction must be in (0, 1], got nan",
        ),
        (
            lambda: QDrift(sampling_budget=True),
            "sampling_budget must be a positive integer (>= 1), got True",
        ),
        (
            lambda: QDrift(sampling_budget=2, seed=True),
            "seed must be an integer, got True",
        ),
        (
            lambda: QDrift(sampling_budget=2, seed=-1),
            "seed must be non-negative, got -1",
        ),
        (
            lambda: QDrift(sampling_budget=2, n_hamiltonians_per_iteration=True),
            "n_hamiltonians_per_iteration must be an integer, got True",
        ),
        (
            lambda: QDrift(sampling_budget=2, n_hamiltonians_per_iteration=2.5),
            "n_hamiltonians_per_iteration must be an integer, got 2.5",
        ),
    ],
    ids=[
        "keep_top_n_bool",
        "keep_fraction_nan",
        "sampling_budget_bool",
        "seed_bool",
        "seed_negative",
        "n_hamiltonians_bool",
        "n_hamiltonians_float",
    ],
)
def test_strategies_reject_invalid_scalars(build, message):
    with pytest.raises(ValueError, match=exact_match(message)):
        build()


class TestQDrift:
    """Tests for QDrift strategy (public API and specified behavior)."""

    def test_invalid_sampling_strategy_raises(self):
        """sampling_strategy must be 'uniform' or 'weighted'."""
        with pytest.raises(
            ValueError,
            match=exact_match(
                "Invalid sampling_strategy: invalid. Must be 'uniform' or 'weighted'."
            ),
        ):
            QDrift(sampling_budget=5, sampling_strategy="invalid")

    def test_seed_non_int_raises(self):
        """seed must be an integer when provided."""
        with pytest.raises(
            ValueError, match=exact_match("seed must be an integer, got 1.5")
        ):
            QDrift(sampling_budget=5, seed=1.5)

    def test_all_none_returns_unchanged_and_warns(self, simple_hamiltonian):
        """When keep_fraction, keep_top_n and sampling_budget are all None, returns Hamiltonian unchanged."""
        with pytest.warns(
            UserWarning,
            match=exact_match(
                "Neither keep_fraction, keep_top_n, nor sampling_budget is set; "
                "the Hamiltonian will be returned unchanged."
            ),
        ):
            result = QDrift().process_hamiltonian(simple_hamiltonian)
        assert result.effective_hamiltonian.simplify() == simple_hamiltonian.simplify()

    def test_sample_budget_only_returns_valid_hamiltonian(self, simple_hamiltonian):
        """When only sample_budget is set (no keep_*), exactly ``sampling_budget``
        terms are drawn, all from the input Hamiltonian."""
        step_terms = _only_step(
            QDrift(sampling_budget=4, seed=42).process_hamiltonian(simple_hamiltonian)
        )
        assert step_terms.size == 4
        assert set(step_terms.paulis.to_labels()) <= set(
            simple_hamiltonian.paulis.to_labels()
        )

    def test_seed_gives_reproducible_result(self, simple_hamiltonian):
        """Same seed yields identical first sample across fresh instances."""
        s1 = QDrift(sampling_budget=5, seed=123, sampling_strategy="uniform")
        s2 = QDrift(sampling_budget=5, seed=123, sampling_strategy="uniform")
        r1 = s1.process_hamiltonian(simple_hamiltonian)
        r2 = s2.process_hamiltonian(simple_hamiltonian)
        assert (
            r1.effective_hamiltonian.simplify() == r2.effective_hamiltonian.simplify()
        )

    def test_with_keep_fraction_and_sample_budget(self, simple_hamiltonian):
        """QDrift with keep_fraction and sample_budget returns keep terms + sampled terms."""
        result = QDrift(
            keep_fraction=0.5, sampling_budget=3, seed=0, sampling_strategy="weighted"
        ).process_hamiltonian(simple_hamiltonian)
        effective = dict(
            zip(
                result.effective_hamiltonian.paulis.to_labels(),
                result.effective_hamiltonian.coeffs.real,
            )
        )
        assert effective.pop("ZZ") == 3.0
        assert set(effective) and set(effective) <= {"IZ", "ZI"}
        assert _only_step(result).size == 4

    @pytest.mark.parametrize(
        "keep", [{"keep_fraction": 0.5}, {"keep_top_n": 1}], ids=["fraction", "top_n"]
    )
    @pytest.mark.parametrize("batched", [False, True], ids=["single", "batch"])
    def test_sample_budget_none_equivalent_to_exact_trotterization(
        self, simple_hamiltonian, keep, batched
    ):
        """When sample_budget is None but a keep option is set, every result equals ExactTrotterization."""
        with pytest.warns(UserWarning, match=_SAMPLING_BUDGET_NOT_SET):
            qdrift = QDrift(**keep)
        exact_result = ExactTrotterization(**keep).process_hamiltonian(
            simple_hamiltonian
        )
        results = (
            qdrift.process_hamiltonian_batch(simple_hamiltonian, n_samples=3)
            if batched
            else [qdrift.process_hamiltonian(simple_hamiltonian)]
        )
        for result in results:
            assert (
                result.effective_hamiltonian.simplify()
                == exact_result.effective_hamiltonian.simplify()
            )

    def test_qdrift_has_no_mutable_call_state(self, simple_hamiltonian):
        strategy = QDrift(
            keep_fraction=0.5, sampling_budget=3, seed=42, sampling_strategy="weighted"
        )
        first = strategy.process_hamiltonian(simple_hamiltonian)
        second = strategy.process_hamiltonian(simple_hamiltonian)
        assert first.effective_hamiltonian == second.effective_hamiltonian
        assert first.sampled_terms == second.sampled_terms

    def test_keep_fraction_one_warns_on_each_stateless_call(self, simple_hamiltonian):
        strategy = QDrift(keep_fraction=1.0, sampling_budget=5, seed=0)
        with pytest.warns(UserWarning) as first_record:
            first_result = strategy.process_hamiltonian(simple_hamiltonian)
        first_messages = [str(w.message) for w in first_record]
        assert any("no terms left to sample" in m for m in first_messages)
        assert (
            first_result.effective_hamiltonian.simplify()
            == simple_hamiltonian.simplify()
        )

        with pytest.warns(UserWarning) as second_record:
            second_result = strategy.process_hamiltonian(simple_hamiltonian)
        second_messages = [str(w.message) for w in second_record]
        assert any("no terms left to sample" in m for m in second_messages)
        assert (
            second_result.effective_hamiltonian.simplify()
            == simple_hamiltonian.simplify()
        )

    @single_term_hamiltonians
    def test_single_term_hamiltonian_with_keep_top_n(self, single_term_spo):
        """Single-term operators work with QDrift keep_top_n; no len() error."""
        with (
            pytest.warns(UserWarning, match=_KEEP_TOP_N_NO_OP),
            pytest.warns(
                UserWarning,
                match=exact_match(
                    "All terms were kept; there are no terms left to sample. "
                    "Returning the full Hamiltonian."
                ),
            ),
        ):
            result = QDrift(
                keep_top_n=1, sampling_budget=2, seed=42
            ).process_hamiltonian(single_term_spo)
        assert result.effective_hamiltonian.simplify() == single_term_spo.simplify()

    @single_term_hamiltonians
    def test_single_term_hamiltonian_with_sampling_budget_only(self, single_term_spo):
        """Single-term sampling preserves replacement multiplicity explicitly."""
        result = QDrift(sampling_budget=3, seed=42).process_hamiltonian(single_term_spo)
        assert result.effective_hamiltonian.simplify() == single_term_spo.simplify()
        step_terms = _only_step(result)
        assert step_terms.size == 3
        np.testing.assert_allclose(
            step_terms.coeffs,
            np.repeat(single_term_spo.coeffs / 3, 3),
        )

    @pytest.mark.parametrize("sampling_budget", [0, -1, 1.5])
    def test_sampling_budget_must_be_positive_integer(self, sampling_budget):
        with pytest.raises(
            ValueError,
            match=exact_match(
                f"sampling_budget must be a positive integer (>= 1), got {sampling_budget}"
            ),
        ):
            QDrift(sampling_budget=sampling_budget)

    def test_empty_hamiltonian_warns_and_returns_kept(self):
        """Empty to_sample_hamiltonian (no terms) warns and returns empty Hamiltonian."""
        empty_spo = SparsePauliOp(["I"], coeffs=[0])[np.zeros(0, dtype=int)]
        with pytest.warns(
            UserWarning,
            match=exact_match("No terms to sample; returning the kept Hamiltonian."),
        ):
            result = QDrift(sampling_budget=3, seed=42).process_hamiltonian(empty_spo)
        assert result.effective_hamiltonian.size == 0

    def test_n_hamiltonians_per_iteration_less_than_one_raises(self):
        """n_hamiltonians_per_iteration must be >= 1."""
        with pytest.raises(
            ValueError,
            match=exact_match("n_hamiltonians_per_iteration must be >= 1, got 0"),
        ):
            QDrift(sampling_budget=5, n_hamiltonians_per_iteration=0)

    def test_rng_produces_different_samples_on_repeated_calls(self, simple_hamiltonian):
        """Instance RNG produces different Hamiltonian samples on repeated calls."""
        strategy = QDrift(sampling_budget=4, seed=42, sampling_strategy="uniform")
        rng = np.random.default_rng(42)
        r0 = strategy.process_hamiltonian(simple_hamiltonian, rng=rng)
        r1 = strategy.process_hamiltonian(simple_hamiltonian, rng=rng)
        r2 = strategy.process_hamiltonian(simple_hamiltonian, rng=rng)
        # With 3 terms and sample_budget=4, sampling with replacement can produce
        # different orderings; at least two of the three should differ
        results = [
            r0.effective_hamiltonian.simplify(),
            r1.effective_hamiltonian.simplify(),
            r2.effective_hamiltonian.simplify(),
        ]
        assert not all(r == results[0] for r in results)

    def test_process_hamiltonian_batch_draws_are_independent(self, simple_hamiltonian):
        """A multi-sample batch advances the RNG between draws (it is not reset
        per draw), so the n sampled Hamiltonians are not all identical, while
        the seed alone reproduces the whole batch on a fresh instance."""
        batches = [
            QDrift(
                sampling_budget=2, seed=42, n_hamiltonians_per_iteration=3
            ).process_hamiltonian_batch(simple_hamiltonian, n_samples=3)
            for _ in range(2)
        ]
        sampled = [r.effective_hamiltonian.simplify() for r in batches[0]]
        assert len(sampled) == 3
        assert not all(s == sampled[0] for s in sampled)
        assert [r.sampled_terms for r in batches[0]] == [
            r.sampled_terms for r in batches[1]
        ]

    def test_sampled_terms_include_the_kept_terms(self, simple_hamiltonian):
        step_terms = _only_step(
            QDrift(keep_top_n=1, sampling_budget=2, seed=0).process_hamiltonian(
                simple_hamiltonian
            )
        )
        assert step_terms.size == 3
        kept = [
            coeff.real
            for label, coeff in zip(step_terms.paulis.to_labels(), step_terms.coeffs)
            if label == "ZZ"
        ]
        assert kept == [3.0]

    def test_tiny_terms_stay_in_the_sampling_pool(self):
        hamiltonian = SparsePauliOp.from_list([("ZZ", 3.0), ("IZ", 1e-9)])
        step_terms = _only_step(
            QDrift(keep_top_n=1, sampling_budget=2, seed=0).process_hamiltonian(
                hamiltonian
            )
        )
        assert "IZ" in step_terms.paulis.to_labels()

    def test_each_trotter_step_draws_its_own_terms_with_the_kept_terms(self):
        hamiltonian = SparsePauliOp.from_list(
            [("ZZ", 3.0), ("IZ", 1.0), ("ZI", 1.0), ("XX", 1.0), ("YY", 1.0)]
        )
        result = QDrift(keep_top_n=1, sampling_budget=3, seed=0).process_hamiltonian(
            hamiltonian, n_steps=6
        )
        assert result.sampled_terms is not None
        assert len(result.sampled_terms) == 6
        labels = [tuple(step.paulis.to_labels()) for step in result.sampled_terms]
        assert all(len(step) == 4 and step[-1] == "ZZ" for step in labels)
        assert len(set(labels)) > 1
        assert result.effective_hamiltonian == result.sampled_terms[0].simplify()

    def test_single_step_draw_is_unchanged_by_the_step_axis(self, simple_hamiltonian):
        """``n_steps=1`` consumes the RNG exactly as one ``rng.choice`` draw."""
        strategy = QDrift(sampling_budget=4, sampling_strategy="uniform")
        step_terms = _only_step(
            strategy.process_hamiltonian(
                simple_hamiltonian, rng=np.random.default_rng(5)
            )
        )
        indices = np.random.default_rng(5).choice(3, size=4, replace=True)
        assert step_terms.paulis == simple_hamiltonian.paulis[indices]

    @pytest.mark.parametrize(
        "strategy, hamiltonian, expected",
        [
            (
                QDrift(sampling_budget=3, seed=1),
                SparsePauliOp("Z", 0.0),
                generate_empty_spo(1),
            ),
            (
                QDrift(keep_fraction=0.9, sampling_budget=3, seed=1),
                SparsePauliOp.from_list([("IZ", 1.0), ("ZI", 1.0)]),
                SparsePauliOp.from_list([("IZ", 1.0), ("ZI", 1.0)]),
            ),
        ],
        ids=["nothing_kept", "everything_kept_by_fraction"],
    )
    def test_zero_weight_sampling_pool_returns_the_kept_terms(
        self, strategy, hamiltonian, expected
    ):
        with pytest.warns(
            UserWarning,
            match=exact_match(
                "All term coefficients are zero; returning the kept Hamiltonian."
            ),
        ):
            result = strategy.process_hamiltonian(hamiltonian)
        assert result.effective_hamiltonian.num_qubits == expected.num_qubits
        assert result.effective_hamiltonian == expected

    @pytest.mark.parametrize("n_steps", [1, 3])
    def test_process_hamiltonian_batch_matches_per_sample_calls(
        self, simple_hamiltonian, n_steps
    ):
        """The batch path is identical to calling process_hamiltonian n times on
        the same RNG (partition-once is an optimisation, not a behaviour change)."""
        budget, n = 3, 4
        batched = QDrift(
            sampling_budget=budget, seed=7, n_hamiltonians_per_iteration=n
        ).process_hamiltonian_batch(
            simple_hamiltonian,
            n_samples=n,
            n_steps=n_steps,
            rng=np.random.default_rng(7),
        )
        loose_strategy = QDrift(sampling_budget=budget, seed=7)
        loose_rng = np.random.default_rng(7)
        per_sample = [
            loose_strategy.process_hamiltonian(
                simple_hamiltonian, n_steps=n_steps, rng=loose_rng
            )
            for _ in range(n)
        ]
        assert [b.sampled_terms for b in batched] == [
            p.sampled_terms for p in per_sample
        ]

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
                .effective_hamiltonian.simplify()
            )
            label_to_coeff = dict(zip(sampled.paulis.to_labels(), sampled.coeffs.real))
            sum_coeffs[0] += label_to_coeff.get("X", 0.0)
            sum_coeffs[1] += label_to_coeff.get("Z", 0.0)
        # Standard error of the 500-seed mean is at most 0.006, so this is ~4σ.
        assert sum_coeffs.real[0] / n_seeds == pytest.approx(0.4, abs=0.025)
        assert sum_coeffs.real[1] / n_seeds == pytest.approx(0.6, abs=0.025)

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
        assert result.effective_hamiltonian.size >= 1
        assert result.effective_hamiltonian.num_qubits == 5


# Y-carrying terms make the sign of t observable; YY and ZZ commute, so every
# product formula is exact and the circuit must equal exp(-i t H) exactly.
_COMMUTING_YY_ZZ = SparsePauliOp(["YY", "ZZ"], [0.8, 0.3])
_SYNTHESIS_TIME = 0.7


def _synthesize(
    result, *, n_steps: int, order: int, n_qubits: int, time: float = _SYNTHESIS_TIME
) -> QuantumCircuit:
    return result.synthesize_evolution(
        QuantumCircuit(n_qubits),
        time=time,
        n_steps=n_steps,
        order=order,
        qubits=list(range(n_qubits)),
        basis_gates=list(_QISKIT_TO_QASM2),
    )


def _evolution(generator: SparsePauliOp, time: float = _SYNTHESIS_TIME) -> np.ndarray:
    return expm(-1j * time * generator.to_matrix())


@pytest.mark.parametrize(
    "hamiltonian",
    [SparsePauliOp("Y", 0.8), _COMMUTING_YY_ZZ],
    ids=["single-term", "multi-term"],
)
@pytest.mark.parametrize("order", [1, 2])
def test_synthesized_evolution_is_exp_minus_i_t_h(
    default_test_simulator, hamiltonian, order
):
    result = ExactTrotterization().process_hamiltonian(hamiltonian, n_steps=2)
    qc = _synthesize(result, n_steps=2, order=order, n_qubits=hamiltonian.num_qubits)
    assert equal_up_to_phase(
        unitary(default_test_simulator, qc), _evolution(hamiltonian), atol=1e-8
    )


def test_sampled_synthesis_applies_each_step_draw(default_test_simulator):
    """Seed 3 draws different terms per step, so replaying one draw would fail."""
    n_steps = 2
    result = QDrift(sampling_budget=4, seed=3).process_hamiltonian(
        _COMMUTING_YY_ZZ, n_steps=n_steps
    )
    assert result.sampled_terms is not None
    assert result.sampled_terms[0] != result.sampled_terms[1]
    qc = _synthesize(result, n_steps=n_steps, order=1, n_qubits=2)
    assert equal_up_to_phase(
        unitary(default_test_simulator, qc),
        _evolution(sum(result.sampled_terms) / n_steps),
        atol=1e-8,
    )


def test_qdrift_without_sampling_budget_synthesizes_deterministically(
    default_test_simulator,
):
    with pytest.warns(UserWarning, match=_SAMPLING_BUDGET_NOT_SET):
        strategy = QDrift(keep_top_n=1)
    result = strategy.process_hamiltonian(_COMMUTING_YY_ZZ, n_steps=2)
    assert result.sampled_terms is None
    qc = _synthesize(result, n_steps=2, order=2, n_qubits=2)
    assert equal_up_to_phase(
        unitary(default_test_simulator, qc),
        _evolution(SparsePauliOp("YY", 0.8)),
        atol=1e-8,
    )


def test_sampled_synthesis_rejects_higher_order():
    result = QDrift(sampling_budget=2, seed=0).process_hamiltonian(
        _SIMPLE_HAMILTONIAN, n_steps=3
    )
    with pytest.raises(
        ValueError,
        match=exact_match("Sampled Trotterization supports only order=1, got order=2."),
    ):
        _synthesize(result, n_steps=3, order=2, n_qubits=2)


@pytest.mark.parametrize(
    "pauli",
    ["".join(p) for k in (1, 2, 3, 4) for p in itertools.product("XYZ", repeat=k)],
)
def test_every_pauli_synthesizes_into_the_qasm_basis(default_test_simulator, pauli):
    hamiltonian = SparsePauliOp(pauli, 0.4)
    qc = _synthesize(
        ExactTrotterization().process_hamiltonian(hamiltonian),
        n_steps=1,
        order=1,
        n_qubits=len(pauli),
    )
    assert {inst.operation.name for inst in qc.data} <= set(_QISKIT_TO_QASM2)
    assert equal_up_to_phase(
        unitary(default_test_simulator, qc), _evolution(hamiltonian), atol=1e-10
    )


_NON_COMMUTING = SparsePauliOp(["IX", "ZZ", "XI", "YY"], [0.7, 0.4, -0.5, 0.3])
_NON_COMMUTING_TERMS = list(
    zip(_NON_COMMUTING.paulis.to_labels(), _NON_COMMUTING.coeffs.real)
)
_EVOLUTION_TIME = 0.9


def _synthesized(simulator, order: int, n_steps: int) -> np.ndarray:
    return unitary(
        simulator,
        _synthesize(
            ExactTrotterization().process_hamiltonian(_NON_COMMUTING),
            n_steps=n_steps,
            order=order,
            n_qubits=2,
            time=_EVOLUTION_TIME,
        ),
    )


def _ordered_product(sequence, n_steps: int) -> np.ndarray:
    """``n_steps`` repetitions of ``exp(-i f dt c P)`` applied in ``sequence`` order."""
    dt = _EVOLUTION_TIME / n_steps
    step = np.eye(4, dtype=complex)
    for label, coeff, fraction in sequence:
        generator = coeff * SparsePauliOp(label).to_matrix()
        step = expm(-1j * fraction * dt * generator) @ step
    return np.linalg.matrix_power(step, n_steps)


@pytest.mark.parametrize(
    "order, sequence",
    [
        (1, [(label, coeff, 1.0) for label, coeff in _NON_COMMUTING_TERMS]),
        (
            2,
            [(label, coeff, 0.5) for label, coeff in _NON_COMMUTING_TERMS]
            + [(label, coeff, 0.5) for label, coeff in _NON_COMMUTING_TERMS[::-1]],
        ),
    ],
    ids=["lie_trotter", "suzuki_second_order"],
)
def test_product_formula_applies_terms_in_input_order(
    default_test_simulator, order, sequence
):
    assert equal_up_to_phase(
        _synthesized(default_test_simulator, order, n_steps=3),
        _ordered_product(sequence, n_steps=3),
        atol=1e-8,
    )


def test_fourth_order_formula_is_more_accurate_than_second_order(
    default_test_simulator,
):
    exact = _evolution(_NON_COMMUTING, _EVOLUTION_TIME)
    errors = {
        order: np.linalg.norm(
            _synthesized(default_test_simulator, order, n_steps=3) - exact
        )
        for order in (2, 4)
    }
    assert errors[4] < errors[2] / 10
