# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for QDrift."""

import copy
import itertools
from dataclasses import replace

import numpy as np
import pytest
from qiskit.quantum_info import SparsePauliOp

from divi.hamiltonians._term_ops import generate_empty_spo
from divi.hamiltonians.trotterization import ExactTrotterization, QDrift, TrotterSpec
from divi.hamiltonians.trotterization import _qdrift as qdrift_module
from divi.hamiltonians.trotterization._allocation import state_drift_distribution
from divi.hamiltonians.trotterization._state_drift import StateDriftExpansion
from divi.qprog.initial_states import CustomPerQubitState, DickeState, ZerosState
from tests._helpers import exact_match
from tests.hamiltonians.trotterization._helpers import (
    KEEP_TOP_N_NO_OP,
    SAMPLING_BUDGET_NOT_SET,
    SIMPLE_HAMILTONIAN,
    single_term_hamiltonians,
)

# ``simplify()`` merges the two IZ entries and reorders the terms.
_MERGING_HAMILTONIAN = SparsePauliOp.from_list(
    [("XX", 0.5), ("ZZ", 3.0), ("IZ", 0.25), ("ZI", -2.0), ("IZ", 0.75)]
)
_MERGING_SPEC = TrotterSpec(_MERGING_HAMILTONIAN)
_STATE_DRIFT_HAMILTONIAN = SparsePauliOp.from_list(
    [("ZZI", 1.0), ("IXX", -0.6), ("YIZ", 0.4), ("XII", 0.8)]
)


def _state_drift_spec(
    initial_state, observable=SparsePauliOp("IZI"), time=0.1
) -> TrotterSpec:
    """Spec of an evolution of ``_STATE_DRIFT_HAMILTONIAN``."""
    return TrotterSpec(
        _STATE_DRIFT_HAMILTONIAN,
        initial_state=initial_state,
        observable=observable,
        time=time,
    )


def _only_step(result) -> SparsePauliOp:
    """The draw of a single-step sampling result."""
    assert result.sampled_terms is not None
    (step_terms,) = result.sampled_terms
    return step_terms


def _scripted_rng(mocker, draw):
    """Generator mock whose ``choice`` returns ``draw``."""
    rng = mocker.Mock(spec=np.random.Generator)
    rng.choice.return_value = np.array(draw)
    return rng


def _exact_expected_step(strategy, spec, mocker) -> SparsePauliOp:
    """E[one step], enumerating every index tuple ``rng.choice`` can return,
    weighted by the probabilities the strategy passes to it."""
    budget = strategy.sampling_budget
    probe = _scripted_rng(mocker, [0] * budget)
    strategy.process_hamiltonian(spec, rng=probe)
    (n_terms,), kwargs = probe.choice.call_args
    probs = np.full(n_terms, 1 / n_terms) if kwargs["p"] is None else kwargs["p"]
    return SparsePauliOp.sum(
        [
            np.prod(probs[list(draw)])
            * _only_step(
                strategy.process_hamiltonian(spec, rng=_scripted_rng(mocker, draw))
            )
            for draw in itertools.product(range(n_terms), repeat=budget)
        ]
    ).simplify()


class TestQDrift:
    """Tests for QDrift strategy (public API and specified behavior)."""

    def test_invalid_sampling_strategy_raises(self):
        with pytest.raises(
            ValueError,
            match=exact_match(
                "Invalid sampling_strategy: invalid. Must be 'uniform', 'weighted', "
                "'state_drift' or a mapping."
            ),
        ):
            QDrift(sampling_budget=5, sampling_strategy="invalid")

    @pytest.mark.parametrize(
        "kwargs, message",
        [
            (
                {"keep_top_n": 1, "sampling_budget": 2},
                "sampling_strategy='state_drift' samples every term; "
                "keep_fraction and keep_top_n are not supported.",
            ),
            (
                {"keep_fraction": 0.5, "sampling_budget": 2},
                "sampling_strategy='state_drift' samples every term; "
                "keep_fraction and keep_top_n are not supported.",
            ),
            ({}, "sampling_strategy='state_drift' requires sampling_budget."),
        ],
        ids=["keep_top_n", "keep_fraction", "no_budget"],
    )
    def test_state_drift_rejects_unsupported_settings(self, kwargs, message):
        with pytest.raises(ValueError, match=exact_match(message)):
            QDrift(**kwargs, sampling_strategy="state_drift")

    @pytest.mark.parametrize(
        "spec_fields, error, message",
        [
            (
                {"observable": SIMPLE_HAMILTONIAN, "time": 0.1},
                ValueError,
                "sampling_strategy='state_drift' needs the initial state and "
                "time of a time evolution; only TimeEvolution supplies them.",
            ),
            (
                {"initial_state": ZerosState(), "observable": SIMPLE_HAMILTONIAN},
                ValueError,
                "sampling_strategy='state_drift' needs the initial state and "
                "time of a time evolution; only TimeEvolution supplies them.",
            ),
            (
                {"initial_state": ZerosState(), "time": 0.1},
                ValueError,
                "sampling_strategy='state_drift' needs exactly one observable, "
                "got none.",
            ),
            (
                {
                    "initial_state": ZerosState(),
                    "observable": (SIMPLE_HAMILTONIAN, SIMPLE_HAMILTONIAN),
                    "time": 0.1,
                },
                ValueError,
                "sampling_strategy='state_drift' needs exactly one observable, got 2.",
            ),
            (
                {"initial_state": ZerosState(), "observable": (), "time": 0.1},
                ValueError,
                "sampling_strategy='state_drift' needs exactly one observable, got 0.",
            ),
            (
                {"initial_state": ZerosState(), "observable": "ZZ", "time": 0.1},
                TypeError,
                "sampling_strategy='state_drift' needs a SparsePauliOp observable, "
                "got str.",
            ),
            (
                {
                    "initial_state": ZerosState(),
                    "observable": SIMPLE_HAMILTONIAN,
                    "time": "0.1",
                },
                TypeError,
                "sampling_strategy='state_drift' needs a numeric time, got str.",
            ),
            (
                {
                    "initial_state": ZerosState(),
                    "observable": SIMPLE_HAMILTONIAN,
                    "time": True,
                },
                TypeError,
                "sampling_strategy='state_drift' needs a numeric time, got bool.",
            ),
        ],
        ids=[
            "no_initial_state",
            "no_time",
            "no_observable",
            "several_observables",
            "empty_observables",
            "not_an_operator",
            "non_numeric_time",
            "bool_time",
        ],
    )
    def test_state_drift_validates_the_evolution(
        self, simple_hamiltonian, spec_fields, error, message
    ):
        strategy = QDrift(sampling_budget=2, sampling_strategy="state_drift")
        with pytest.raises(error, match=exact_match(message)):
            strategy.process_hamiltonian(TrotterSpec(simple_hamiltonian, **spec_fields))

    @pytest.mark.parametrize(
        "initial_state",
        [CustomPerQubitState("+0-"), DickeState(1)],
        ids=["product", "dicke"],
    )
    def test_state_drift_expected_value_matches_input_hamiltonian(
        self, mocker, initial_state
    ):
        """StateDRIFT reweights terms but keeps E[step] = H."""
        expected = _exact_expected_step(
            QDrift(sampling_budget=2, sampling_strategy="state_drift"),
            _state_drift_spec(
                initial_state,
                SparsePauliOp.from_list([("IIZ", 1.0), ("XIX", 0.5)]),
                time=0.05,
            ),
            mocker,
        )
        np.testing.assert_allclose(
            expected.to_matrix(), _STATE_DRIFT_HAMILTONIAN.to_matrix(), atol=1e-12
        )

    @pytest.mark.parametrize("n_steps", [1, 3])
    def test_state_drift_samples_by_the_distribution_for_its_time(
        self, mocker, n_steps
    ):
        """The strategy draws with the StateDRIFT q of the time and step count
        it is given, over ``n_steps · sampling_budget`` rotations."""
        observable = SparsePauliOp("IZI")
        state = CustomPerQubitState("+0-")
        strategy = QDrift(
            sampling_budget=2,
            n_hamiltonians_per_iteration=5,
            sampling_strategy="state_drift",
        )
        rng = _scripted_rng(mocker, [0, 0])
        strategy.process_hamiltonian(
            _state_drift_spec(state, observable, time=0.2), n_steps=n_steps, rng=rng
        )
        expected = state_drift_distribution(
            StateDriftExpansion.compute(_STATE_DRIFT_HAMILTONIAN, observable, state),
            time=0.2,
            n_samples=2 * n_steps,
            n_trajectories=5,
        )
        weighted = np.abs(_STATE_DRIFT_HAMILTONIAN.coeffs)
        assert not np.allclose(list(expected.values()), weighted / weighted.sum())
        np.testing.assert_allclose(
            rng.choice.call_args.kwargs["p"],
            [expected[label] for label in _STATE_DRIFT_HAMILTONIAN.paulis.to_labels()],
        )

    def test_copies_share_the_state_drift_densities(self, mocker):
        """Per-program copies of one strategy evaluate the state only once."""
        state = CustomPerQubitState("+0-")
        spy = mocker.spy(StateDriftExpansion, "compute")
        strategy = QDrift(sampling_budget=2, seed=1, sampling_strategy="state_drift")
        for copy_ in (strategy, replace(strategy, seed=2), copy.deepcopy(strategy)):
            copy_.process_hamiltonian(_state_drift_spec(state))
        assert spy.call_count == 1

    def test_state_drift_takes_one_observable_given_as_a_sequence(self):
        strategy = QDrift(sampling_budget=2, seed=3, sampling_strategy="state_drift")
        state = CustomPerQubitState("+0-")
        observable = SparsePauliOp("IZI")
        bare = strategy.process_hamiltonian(_state_drift_spec(state, observable))
        wrapped = strategy.process_hamiltonian(_state_drift_spec(state, (observable,)))
        assert wrapped.sampled_terms == bare.sampled_terms

    def test_state_drift_batch_draws_from_one_distribution(self, mocker):
        spy = mocker.spy(qdrift_module, "state_drift_distribution")
        results = QDrift(
            sampling_budget=2, seed=3, sampling_strategy="state_drift"
        ).process_hamiltonian_batch(
            _state_drift_spec(CustomPerQubitState("+0-")), n_samples=4
        )
        assert len(results) == 4
        assert spy.call_count == 1

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
            result = QDrift().process_hamiltonian(TrotterSpec(simple_hamiltonian))
        assert result.effective_hamiltonian.simplify() == simple_hamiltonian.simplify()

    def test_sample_budget_only_returns_valid_hamiltonian(self, simple_hamiltonian):
        """When only sample_budget is set (no keep_*), exactly ``sampling_budget``
        terms are drawn, all from the input Hamiltonian."""
        step_terms = _only_step(
            QDrift(sampling_budget=4, seed=42).process_hamiltonian(
                TrotterSpec(simple_hamiltonian)
            )
        )
        assert step_terms.size == 4
        assert set(step_terms.paulis.to_labels()) <= set(
            simple_hamiltonian.paulis.to_labels()
        )

    def test_seed_gives_reproducible_result(self, simple_hamiltonian):
        """Same seed yields identical first sample across fresh instances."""
        s1 = QDrift(sampling_budget=5, seed=123, sampling_strategy="uniform")
        s2 = QDrift(sampling_budget=5, seed=123, sampling_strategy="uniform")
        r1 = s1.process_hamiltonian(TrotterSpec(simple_hamiltonian))
        r2 = s2.process_hamiltonian(TrotterSpec(simple_hamiltonian))
        assert (
            r1.effective_hamiltonian.simplify() == r2.effective_hamiltonian.simplify()
        )

    def test_with_keep_fraction_and_sample_budget(self, simple_hamiltonian):
        """QDrift with keep_fraction and sample_budget returns keep terms + sampled terms."""
        result = QDrift(
            keep_fraction=0.5, sampling_budget=3, seed=0, sampling_strategy="weighted"
        ).process_hamiltonian(TrotterSpec(simple_hamiltonian))
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
        with pytest.warns(UserWarning, match=SAMPLING_BUDGET_NOT_SET):
            qdrift = QDrift(**keep)
        exact_result = ExactTrotterization(**keep).process_hamiltonian(
            TrotterSpec(simple_hamiltonian)
        )
        results = (
            qdrift.process_hamiltonian_batch(
                TrotterSpec(simple_hamiltonian), n_samples=3
            )
            if batched
            else [qdrift.process_hamiltonian(TrotterSpec(simple_hamiltonian))]
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
        first = strategy.process_hamiltonian(TrotterSpec(simple_hamiltonian))
        second = strategy.process_hamiltonian(TrotterSpec(simple_hamiltonian))
        assert first.effective_hamiltonian == second.effective_hamiltonian
        assert first.sampled_terms == second.sampled_terms

    def test_keep_fraction_one_warns_on_each_stateless_call(self, simple_hamiltonian):
        strategy = QDrift(keep_fraction=1.0, sampling_budget=5, seed=0)
        with pytest.warns(UserWarning) as first_record:
            first_result = strategy.process_hamiltonian(TrotterSpec(simple_hamiltonian))
        first_messages = [str(w.message) for w in first_record]
        assert any("no terms left to sample" in m for m in first_messages)
        assert (
            first_result.effective_hamiltonian.simplify()
            == simple_hamiltonian.simplify()
        )

        with pytest.warns(UserWarning) as second_record:
            second_result = strategy.process_hamiltonian(
                TrotterSpec(simple_hamiltonian)
            )
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
            pytest.warns(UserWarning, match=KEEP_TOP_N_NO_OP),
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
            ).process_hamiltonian(TrotterSpec(single_term_spo))
        assert result.effective_hamiltonian.simplify() == single_term_spo.simplify()

    @single_term_hamiltonians
    def test_single_term_hamiltonian_with_sampling_budget_only(self, single_term_spo):
        """Single-term sampling preserves replacement multiplicity explicitly."""
        result = QDrift(sampling_budget=3, seed=42).process_hamiltonian(
            TrotterSpec(single_term_spo)
        )
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
            result = QDrift(sampling_budget=3, seed=42).process_hamiltonian(
                TrotterSpec(empty_spo)
            )
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
        r0 = strategy.process_hamiltonian(TrotterSpec(simple_hamiltonian), rng=rng)
        r1 = strategy.process_hamiltonian(TrotterSpec(simple_hamiltonian), rng=rng)
        r2 = strategy.process_hamiltonian(TrotterSpec(simple_hamiltonian), rng=rng)
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
            ).process_hamiltonian_batch(TrotterSpec(simple_hamiltonian), n_samples=3)
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
                TrotterSpec(simple_hamiltonian)
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
                TrotterSpec(hamiltonian)
            )
        )
        assert "IZ" in step_terms.paulis.to_labels()

    def test_each_trotter_step_draws_its_own_terms_with_the_kept_terms(self):
        hamiltonian = SparsePauliOp.from_list(
            [("ZZ", 3.0), ("IZ", 1.0), ("ZI", 1.0), ("XX", 1.0), ("YY", 1.0)]
        )
        result = QDrift(keep_top_n=1, sampling_budget=3, seed=0).process_hamiltonian(
            TrotterSpec(hamiltonian), n_steps=6
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
                TrotterSpec(simple_hamiltonian), rng=np.random.default_rng(5)
            )
        )
        indices = np.random.default_rng(5).choice(3, size=4, replace=True)
        assert step_terms.paulis == simple_hamiltonian.paulis[indices]

    @pytest.mark.parametrize(
        "strategy, spec, expected",
        [
            (
                QDrift(sampling_budget=3, seed=1),
                TrotterSpec(SparsePauliOp("Z", 0.0)),
                generate_empty_spo(1),
            ),
            (
                QDrift(keep_fraction=0.9, sampling_budget=3, seed=1),
                TrotterSpec(SparsePauliOp.from_list([("IZ", 1.0), ("ZI", 1.0)])),
                SparsePauliOp.from_list([("IZ", 1.0), ("ZI", 1.0)]),
            ),
            (
                QDrift(sampling_budget=3, seed=1, sampling_strategy="state_drift"),
                TrotterSpec(
                    SparsePauliOp("ZZ", 0.0),
                    initial_state=ZerosState(),
                    observable=SparsePauliOp("XI"),
                    time=0.1,
                ),
                generate_empty_spo(2),
            ),
        ],
        ids=["nothing_kept", "everything_kept_by_fraction", "state_drift"],
    )
    def test_zero_weight_sampling_pool_returns_the_kept_terms(
        self, strategy, spec, expected
    ):
        with pytest.warns(
            UserWarning,
            match=exact_match(
                "All term coefficients are zero; returning the kept Hamiltonian."
            ),
        ) as caught:
            result = strategy.process_hamiltonian(spec)
        assert len(caught) == 1
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
            TrotterSpec(simple_hamiltonian),
            n_samples=n,
            n_steps=n_steps,
            rng=np.random.default_rng(7),
        )
        loose_strategy = QDrift(sampling_budget=budget, seed=7)
        loose_rng = np.random.default_rng(7)
        per_sample = [
            loose_strategy.process_hamiltonian(
                TrotterSpec(simple_hamiltonian), n_steps=n_steps, rng=loose_rng
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
                .process_hamiltonian(TrotterSpec(spo))
                .effective_hamiltonian.simplify()
            )
            label_to_coeff = dict(zip(sampled.paulis.to_labels(), sampled.coeffs.real))
            sum_coeffs[0] += label_to_coeff.get("X", 0.0)
            sum_coeffs[1] += label_to_coeff.get("Z", 0.0)
        # Standard error of the 500-seed mean is at most 0.006, so this is ~4σ.
        assert sum_coeffs.real[0] / n_seeds == pytest.approx(0.4, abs=0.025)
        assert sum_coeffs.real[1] / n_seeds == pytest.approx(0.6, abs=0.025)

    @pytest.mark.parametrize(
        "hamiltonian, keep, distribution",
        [
            (SIMPLE_HAMILTONIAN, {}, {"ZZ": 0.1, "IZ": 0.6, "ZI": 0.3}),
            (
                _MERGING_HAMILTONIAN,
                {},
                {"ZI": 0.4, "IZ": 0.1, "ZZ": 0.3, "XX": 0.2},
            ),
            (
                _MERGING_HAMILTONIAN,
                {"keep_top_n": 1},
                {"ZI": 0.5, "IZ": 0.2, "XX": 0.3},
            ),
        ],
        ids=["distinct_terms", "merged_by_simplify", "after_keep_top_n"],
    )
    def test_mapping_distribution_expected_value_matches_input_hamiltonian(
        self, mocker, hamiltonian, keep, distribution
    ):
        """Any label-keyed distribution with q_j > 0 gives E[step] = H exactly,
        however ``simplify()`` and the keep split reorder or merge terms."""
        strategy = QDrift(**keep, sampling_budget=2, sampling_strategy=distribution)
        expected = _exact_expected_step(strategy, TrotterSpec(hamiltonian), mocker)
        np.testing.assert_allclose(
            expected.to_matrix(), hamiltonian.to_matrix(), atol=1e-12
        )

    def test_weighted_distribution_mapping_matches_weighted_strategy(self, mocker):
        simplified = _MERGING_HAMILTONIAN.simplify()
        absolute = np.abs(simplified.coeffs)
        distribution = dict(
            zip(simplified.paulis.to_labels(), (absolute / absolute.sum()).tolist())
        )
        mapped = QDrift(sampling_budget=2, sampling_strategy=distribution)
        weighted = QDrift(sampling_budget=2, sampling_strategy="weighted")
        for draw in itertools.product(range(simplified.size), repeat=2):
            mapped_step, weighted_step = (
                _only_step(
                    strategy.process_hamiltonian(
                        TrotterSpec(_MERGING_HAMILTONIAN),
                        rng=_scripted_rng(mocker, draw),
                    )
                )
                for strategy in (mapped, weighted)
            )
            assert mapped_step.paulis == weighted_step.paulis
            np.testing.assert_allclose(
                mapped_step.coeffs, weighted_step.coeffs, rtol=1e-12
            )
        np.testing.assert_allclose(
            _exact_expected_step(mapped, _MERGING_SPEC, mocker).to_matrix(),
            _exact_expected_step(weighted, _MERGING_SPEC, mocker).to_matrix(),
            atol=1e-12,
        )

    def test_mapping_strategy_is_copied_and_hashable(self):
        distribution = {"IZ": 0.5, "ZI": 0.25, "ZZ": 0.25}
        strategy = QDrift(sampling_budget=2, sampling_strategy=distribution)
        distribution["IZ"] = 0.0
        assert strategy.sampling_strategy == {"IZ": 0.5, "ZI": 0.25, "ZZ": 0.25}
        assert hash(strategy) == hash(
            QDrift(
                sampling_budget=2, sampling_strategy=dict(strategy.sampling_strategy)
            )
        )

    @pytest.mark.parametrize(
        "distribution, message",
        [
            ({1: 1.0}, "sampling_strategy keys must be Pauli label strings, got [1]"),
            (
                {"Z": "a"},
                "sampling_strategy probabilities must be real numbers, got ['a']",
            ),
            (
                {"Z": 0.0, "X": 1.0},
                "sampling_strategy probabilities must be finite and > 0, "
                "got {'Z': 0.0, 'X': 1.0}",
            ),
            (
                {"Z": -0.5, "X": 1.5},
                "sampling_strategy probabilities must be finite and > 0, "
                "got {'Z': -0.5, 'X': 1.5}",
            ),
            (
                {"Z": float("nan"), "X": 1.0},
                "sampling_strategy probabilities must be finite and > 0, "
                "got {'Z': nan, 'X': 1.0}",
            ),
            (
                {"Z": float("inf")},
                "sampling_strategy probabilities must be finite and > 0, "
                "got {'Z': inf}",
            ),
            (
                {"Z": 0.5, "X": 0.4},
                "sampling_strategy probabilities must sum to 1, got 0.9",
            ),
            ({}, "sampling_strategy probabilities must sum to 1, got 0.0"),
        ],
        ids=[
            "non_string_label",
            "non_numeric",
            "zero",
            "negative",
            "nan",
            "inf",
            "unnormalised",
            "empty",
        ],
    )
    def test_invalid_mapping_strategy_raises(self, distribution, message):
        with pytest.raises(ValueError, match=exact_match(message)):
            QDrift(sampling_budget=2, sampling_strategy=distribution)

    @pytest.mark.parametrize(
        "keep, distribution, message",
        [
            (
                {},
                {"ZZ": 0.5, "IZ": 0.5},
                "sampling_strategy has no probability for the sampled terms ['ZI']",
            ),
            (
                {"keep_top_n": 1},
                {"ZZ": 0.2, "IZ": 0.4, "ZI": 0.4},
                "sampling_strategy has probabilities for ['ZZ'], "
                "which are not among the sampled terms ['IZ', 'ZI']",
            ),
        ],
        ids=["missing_label", "kept_label"],
    )
    def test_mapping_must_cover_exactly_the_sampled_terms(
        self, simple_hamiltonian, keep, distribution, message
    ):
        strategy = QDrift(**keep, sampling_budget=2, sampling_strategy=distribution)
        with pytest.raises(ValueError, match=exact_match(message)):
            strategy.process_hamiltonian(TrotterSpec(simple_hamiltonian))

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
        ).process_hamiltonian(TrotterSpec(spo))
        assert result.effective_hamiltonian.size >= 1
        assert result.effective_hamiltonian.num_qubits == 5
