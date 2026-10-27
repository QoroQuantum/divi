# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import itertools
from threading import Event, Thread

import numpy as np
import pytest
from qiskit.quantum_info import SparsePauliOp
from scipy.linalg import expm

from divi.hamiltonians.trotterization import _pauli_bits as _bits
from divi.hamiltonians.trotterization import _state_drift
from divi.hamiltonians.trotterization._allocation import state_drift_distribution
from divi.hamiltonians.trotterization._state_drift import (
    StateDriftCache,
    StateDriftDensities,
    StateDriftExpansion,
)
from divi.qprog.initial_states import (
    CustomPerQubitState,
    DickeState,
    OnesState,
    SuperpositionState,
    WState,
    ZerosState,
)
from tests._helpers import exact_match, initial_state_amplitudes

# Odd-Y terms give non-zero rotation slopes in real-amplitude states.
_HAMILTONIAN = SparsePauliOp.from_list(
    [
        ("IIZZ", 1.0),
        ("IZZI", 1.0),
        ("ZZII", 1.0),
        ("IIIX", 0.7),
        ("IIXI", 0.7),
        ("IXII", 0.7),
        ("XIII", 0.7),
        ("IIYI", 0.4),
        ("YIII", 0.3),
        ("XXYY", -0.25),
        ("IYXI", 0.5),
    ]
)
_OBSERVABLE = SparsePauliOp.from_list(
    [("IIIZ", 1.0), ("IYXI", 0.3), ("ZIXI", -0.6), ("IIII", 2.0)]
)


def _commutator(a, b):
    return a @ b - b @ a


def _dense_densities(hamiltonian, observable, initial_state):
    """``(γ, β, φ_HH)`` of ``initial_state``, from dense matrices."""
    psi = initial_state_amplitudes(initial_state, hamiltonian.num_qubits)
    obs = observable.to_matrix()
    ham = hamiltonian.to_matrix()

    def expval(matrix):
        return psi.conj() @ matrix @ psi

    shifts, slopes = [], []
    for pauli in hamiltonian.paulis:
        term = pauli.to_matrix()
        shifts.append((expval(term @ obs @ term) - expval(obs)).real)
        slopes.append((-1j * expval(_commutator(obs, term))).real)
    curvature = expval(_commutator(ham, _commutator(obs, ham))).real
    return np.array(shifts), np.array(slopes), curvature


def _dense_expansion_inputs(hamiltonian, observable, initial_state):
    """The per-term inputs of :class:`StateDriftExpansion` by field name, the
    matrix ``φ_{a,b}``, then ``<L_H³ O>`` and ``<L_H⁴ O>``, from dense
    matrices."""
    psi = initial_state_amplitudes(initial_state, hamiltonian.num_qubits)
    obs = observable.to_matrix()
    ham = hamiltonian.to_matrix()

    def expval(matrix):
        return (psi.conj() @ matrix @ psi).real

    def ladder(x):
        return 1j * _commutator(ham, x)

    def shifted(term, x):
        return term @ x @ term - x

    first, second = ladder(obs), ladder(ladder(obs))
    terms = [pauli.to_matrix() for pauli in hamiltonian.simplify().paulis]
    per_term = {
        "inner_curvatures": [
            expval(_commutator(ham, _commutator(obs, t))) for t in terms
        ],
        "outer_curvatures": [
            expval(_commutator(t, _commutator(obs, ham))) for t in terms
        ],
        "shift_slopes": [expval(ladder(shifted(t, obs))) for t in terms],
        "shift_curvatures": [expval(ladder(ladder(shifted(t, obs)))) for t in terms],
        "slope_shifts": [expval(shifted(t, first)) for t in terms],
        "slope_shift_slopes": [expval(ladder(shifted(t, first))) for t in terms],
        "curvature_shifts": [expval(shifted(t, second)) for t in terms],
        "curvature_slopes": [expval(-1j * _commutator(second, t)) for t in terms],
        "rotated_slope_slopes": [
            expval(ladder(1j * _commutator(t, first))) for t in terms
        ],
        "rotated_curvatures": [
            expval(ladder(ladder(1j * _commutator(t, obs)))) for t in terms
        ],
    }
    nested = np.array(
        [[expval(_commutator(a, _commutator(obs, b))) for b in terms] for a in terms]
    )
    return (
        per_term,
        nested,
        expval(ladder(second)),
        expval(ladder(ladder(second))),
    )


def _expansion_fields(expansion):
    return {
        "inner_curvatures": expansion.inner_curvatures,
        "outer_curvatures": expansion.outer_curvatures,
        "shift_slopes": expansion.shift_slopes,
        "shift_curvatures": expansion.shift_curvatures,
        "slope_shifts": expansion.slope_shifts,
        "slope_shift_slopes": expansion.slope_shift_slopes,
        "curvature_shifts": expansion.curvature_shifts,
        "curvature_slopes": expansion.curvature_slopes,
        "rotated_slope_slopes": expansion.rotated_slope_slopes,
        "rotated_curvatures": expansion.rotated_curvatures,
    }


def _nested_matrix(expansion):
    a, b, values = expansion.nested_curvatures
    size = len(expansion.densities.terms)
    matrix = np.zeros((size, size))
    matrix[a, b] = values
    return matrix


def _assert_expansion_matches_dense(hamiltonian, observable, initial_state):
    expansion = StateDriftExpansion.compute(hamiltonian, observable, initial_state)
    per_term, nested, third, fourth = _dense_expansion_inputs(
        hamiltonian, observable, initial_state
    )
    found = _expansion_fields(expansion)
    for name, expected in per_term.items():
        np.testing.assert_allclose(found[name], expected, atol=1e-10, err_msg=name)
    np.testing.assert_allclose(_nested_matrix(expansion), nested, atol=1e-10)
    assert expansion.third_derivative == pytest.approx(third, abs=1e-10)
    assert expansion.fourth_derivative == pytest.approx(fourth, abs=1e-9)


def _exact_variance(hamiltonian, observable, initial_state, probs, time, n_samples):
    """Exact variance of one run of ``n_samples`` QDrift rotations over
    ``time`` drawn from ``probs``, over every sequence of drawn terms."""
    psi = initial_state_amplitudes(initial_state, hamiltonian.num_qubits)
    obs = observable.to_matrix()
    dt = time / n_samples
    rotations = [
        expm(-1j * coeff * dt / prob * pauli.to_matrix())
        for pauli, coeff, prob in zip(
            hamiltonian.paulis, hamiltonian.coeffs.real, probs
        )
    ]
    mean = mean_square = 0.0
    for drawn in itertools.product(range(len(probs)), repeat=n_samples):
        state = psi.astype(complex)
        for term in drawn:
            state = rotations[term] @ state
        value = (state.conj() @ obs @ state).real
        weight = np.prod(probs[list(drawn)])
        mean += weight * value
        mean_square += weight * value**2
    return mean_square - mean**2


def _exact_bias(hamiltonian, observable, initial_state, probs, time, n_samples=1):
    """Exact bias of ``n_samples`` QDrift rotations over ``time`` drawn from
    ``probs`` (ordered as ``hamiltonian.paulis``), from the averaged channel."""
    psi = initial_state_amplitudes(initial_state, hamiltonian.num_qubits)
    obs = observable.to_matrix()
    dt = time / n_samples
    rotations = [
        expm(-1j * coeff * dt / prob * pauli.to_matrix())
        for pauli, coeff, prob in zip(
            hamiltonian.paulis, hamiltonian.coeffs.real, probs
        )
    ]
    rho = np.outer(psi, psi.conj())
    for _ in range(n_samples):
        rho = sum(prob * u @ rho @ u.conj().T for prob, u in zip(probs, rotations))
    evolved = expm(-1j * time * hamiltonian.to_matrix()) @ psi
    return np.trace(obs @ rho).real - (evolved.conj() @ obs @ evolved).real


# Upper bound on any wait; reached only if the code under test misbehaves.
_WAIT = 10.0


def _hold_evaluation(mocker, state, error=None):
    """Make the first ``state.pauli_expectations`` call signal ``entered`` and
    wait for ``release``, then evaluate, or raise ``error`` when given; later
    calls evaluate straight away."""
    entered, release = Event(), Event()
    evaluate = state.pauli_expectations

    def held(*args):
        if not entered.is_set():
            entered.set()
            release.wait(_WAIT)
            if error is not None:
                raise error
        return evaluate(*args)

    mocker.patch.object(state, "pauli_expectations", side_effect=held)
    return entered, release


def _signal_matches(mocker):
    """Event set once the cache matches a request to a stored evolution."""
    matched = Event()
    same = _state_drift._same_evolution

    def spy(stored, requested):
        if same(stored, requested):
            matched.set()
            return True
        return False

    mocker.patch.object(_state_drift, "_same_evolution", side_effect=spy)
    return matched


def _start(call):
    """Run ``call`` on a thread; the returned list receives its result or error."""
    outcome = []

    def run():
        try:
            outcome.append(call())
        except Exception as error:
            outcome.append(error)

    thread = Thread(target=run)
    thread.start()
    return thread, outcome


def _ordered_probs(distribution):
    return np.array([distribution[label] for label in _HAMILTONIAN.paulis.to_labels()])


def _weighted_probs():
    vanilla = np.abs(_HAMILTONIAN.coeffs.real)
    return vanilla / vanilla.sum()


class TestStateDriftDensities:
    @pytest.mark.parametrize(
        "initial_state",
        [
            ZerosState(),
            OnesState(),
            SuperpositionState(),
            CustomPerQubitState("0+-1"),
            DickeState(2),
            WState(2, 2),
        ],
        ids=["zeros", "ones", "superposition", "per_qubit", "dicke", "w"],
    )
    def test_densities_match_dense_matrices(self, initial_state):
        densities = StateDriftDensities.compute(
            _HAMILTONIAN, _OBSERVABLE, initial_state
        )
        dense_shifts, dense_slopes, dense_curvature = _dense_densities(
            _HAMILTONIAN, _OBSERVABLE, initial_state
        )
        assert densities.terms == _HAMILTONIAN.simplify()
        np.testing.assert_allclose(
            densities.conjugation_shifts, dense_shifts, atol=1e-10
        )
        np.testing.assert_allclose(densities.rotation_slopes, dense_slopes, atol=1e-10)
        assert densities.curvature == pytest.approx(dense_curvature, abs=1e-10)

    def test_curvature_of_a_small_hamiltonian_scales_with_it(self):
        state = CustomPerQubitState("0+-1")
        base = StateDriftDensities.compute(_HAMILTONIAN, _OBSERVABLE, state)
        small = StateDriftDensities.compute(1e-5 * _HAMILTONIAN, _OBSERVABLE, state)
        assert small.curvature == pytest.approx(1e-10 * base.curvature, rel=1e-9)

    def test_densities_are_nontrivial_for_the_test_problem(self):
        densities = StateDriftDensities.compute(
            _HAMILTONIAN, _OBSERVABLE, CustomPerQubitState("0+-1")
        )
        assert np.count_nonzero(np.round(densities.conjugation_shifts, 12)) >= 2
        assert np.count_nonzero(np.round(densities.rotation_slopes, 12)) >= 1
        assert densities.curvature != pytest.approx(0.0)


_EXPANSION_STATES = pytest.mark.parametrize(
    "initial_state",
    [
        CustomPerQubitState("0+-1"),
        OnesState(),
        CustomPerQubitState("+-+-"),
        DickeState(2),
        WState(2, 2),
    ],
    ids=["mixed_axes", "ones", "x_axes", "dicke", "w"],
)
_CONVERGENCE_STATES = pytest.mark.parametrize(
    "initial_state",
    [CustomPerQubitState("0+-1"), DickeState(2), WState(2, 2)],
    ids=["product", "dicke", "w"],
)


def _bounded_probs(expansion):
    """Half the weighted distribution, half a random one: every rotation angle
    stays small as the time shrinks."""
    coeffs = expansion.densities.terms.coeffs.real
    weighted = np.abs(coeffs) / np.abs(coeffs).sum()
    return 0.5 * weighted + 0.5 * np.random.default_rng(1).dirichlet(
        np.ones(len(coeffs))
    )


class TestStateDriftExpansion:
    @_EXPANSION_STATES
    def test_inputs_match_dense_matrices(self, initial_state):
        _assert_expansion_matches_dense(_HAMILTONIAN, _OBSERVABLE, initial_state)

    def test_inputs_match_dense_matrices_without_nested_terms(self):
        _assert_expansion_matches_dense(
            SparsePauliOp.from_list([("ZZ", -1.0), ("IX", -0.7), ("XI", -0.7)]),
            SparsePauliOp("IY"),
            ZerosState(),
        )

    @pytest.mark.parametrize(
        "initial_state",
        [CustomPerQubitState("0+-1"), DickeState(2)],
        ids=["product", "dicke"],
    )
    def test_inputs_are_the_same_when_evaluated_in_blocks(self, mocker, initial_state):
        whole = StateDriftExpansion.compute(_HAMILTONIAN, _OBSERVABLE, initial_state)
        mocker.patch.object(_bits, "PAIR_BUDGET", 1)
        blocked = StateDriftExpansion.compute(_HAMILTONIAN, _OBSERVABLE, initial_state)
        for name, values in _expansion_fields(whole).items():
            np.testing.assert_allclose(
                _expansion_fields(blocked)[name], values, atol=1e-12, err_msg=name
            )
        np.testing.assert_allclose(
            _nested_matrix(blocked), _nested_matrix(whole), atol=1e-12
        )
        assert blocked.densities.curvature == pytest.approx(
            whole.densities.curvature, abs=1e-12
        )
        assert blocked.fourth_derivative == pytest.approx(
            whole.fourth_derivative, abs=1e-12
        )

    def test_fourth_derivative_of_a_wide_register_skips_no_term(self, mocker):
        """The key joins on a 70-qubit product state find what evaluating
        every string does."""
        n = 70
        hamiltonian = SparsePauliOp.from_sparse_list(
            [("ZZ", [i, (i + 1) % n], 1.0) for i in range(n)]
            + [("X", [i], 0.7) for i in range(n)]
            + [("Y", [i], 0.3) for i in range(0, n, 7)],
            num_qubits=n,
        )
        observable = SparsePauliOp.from_sparse_list(
            [("Z", [0], 1.0), ("XX", [30, 69], 0.5)], num_qubits=n
        )
        state = CustomPerQubitState("0+" * 35)
        joined = StateDriftExpansion.compute(hamiltonian, observable, state)
        mocker.patch.object(_bits._ProductMasks, "of", return_value=None)
        evaluated = StateDriftExpansion.compute(hamiltonian, observable, state)
        assert joined.fourth_derivative != pytest.approx(0.0)
        assert joined.fourth_derivative == pytest.approx(
            evaluated.fourth_derivative, rel=1e-12
        )
        for name, values in _expansion_fields(evaluated).items():
            np.testing.assert_allclose(
                _expansion_fields(joined)[name], values, atol=1e-10, err_msg=name
            )

    @_CONVERGENCE_STATES
    def test_truncated_bias_is_exact_to_fourth_order(self, initial_state):
        """Halving the time divides the error left by ``B_N^(4)`` by about 2⁵."""
        expansion = StateDriftExpansion.compute(
            _HAMILTONIAN, _OBSERVABLE, initial_state
        )
        probs = _bounded_probs(expansion)
        errors = []
        for time in (0.1, 0.05):
            exact = _exact_bias(
                expansion.densities.terms, _OBSERVABLE, initial_state, probs, time, 2
            )
            *_, fourth_order = expansion.truncated_bias(probs, dt=time / 2, n_samples=2)
            errors.append(abs(exact - fourth_order))

        assert errors[0] / errors[1] > 20

    @_CONVERGENCE_STATES
    def test_truncated_variance_is_exact_to_fourth_order(self, initial_state):
        """Halving the time divides the error left by ``V_N^(4)`` by at least
        about 2⁵, at three rotations so every ``N`` group of Eq. C94 enters."""
        expansion = StateDriftExpansion.compute(
            _HAMILTONIAN, _OBSERVABLE, initial_state
        )
        probs = _bounded_probs(expansion)
        errors = []
        for time in (0.1, 0.05):
            exact = _exact_variance(
                expansion.densities.terms, _OBSERVABLE, initial_state, probs, time, 3
            )
            *_, fourth_order = expansion.truncated_variance(
                probs, dt=time / 3, n_samples=3
            )
            errors.append(abs(exact - fourth_order))

        assert errors[0] / errors[1] > 20


class TestStateDriftCache:
    def test_each_evolution_is_evaluated_once(self, mocker):
        state = DickeState(2)
        spy = mocker.spy(StateDriftExpansion, "compute")
        cache = StateDriftCache()
        first = cache.expansion(_HAMILTONIAN, _OBSERVABLE, state)
        again = cache.expansion(_HAMILTONIAN.copy(), _OBSERVABLE.copy(), DickeState(2))
        assert again is first
        assert spy.call_count == 1

    @pytest.mark.parametrize(
        "hamiltonian, observable, state",
        [
            (_HAMILTONIAN, SparsePauliOp("IIIZ"), DickeState(2)),
            (_HAMILTONIAN, _OBSERVABLE, WState(2, 2)),
            (_HAMILTONIAN, _OBSERVABLE, DickeState(1)),
            (2 * _HAMILTONIAN, _OBSERVABLE, DickeState(2)),
        ],
        ids=["observable", "state_type", "state_parameter", "hamiltonian"],
    )
    def test_a_different_evolution_gets_its_own_expansion(
        self, hamiltonian, observable, state
    ):
        cache = StateDriftCache()
        first = cache.expansion(_HAMILTONIAN, _OBSERVABLE, DickeState(2))
        other = cache.expansion(hamiltonian, observable, state).densities
        assert other is not first.densities
        expected = StateDriftDensities.compute(hamiltonian, observable, state)
        np.testing.assert_allclose(
            other.conjugation_shifts, expected.conjugation_shifts
        )
        np.testing.assert_allclose(other.rotation_slopes, expected.rotation_slopes)

    def test_a_slow_evolution_does_not_hold_up_another(self, mocker):
        cache = StateDriftCache()
        slow = DickeState(2)
        entered, release = _hold_evaluation(mocker, slow)
        worker, _ = _start(lambda: cache.expansion(_HAMILTONIAN, _OBSERVABLE, slow))
        try:
            assert entered.wait(_WAIT)
            other = cache.expansion(_HAMILTONIAN, _OBSERVABLE, ZerosState())
            slow_still_running = worker.is_alive()
        finally:
            release.set()
            worker.join(_WAIT)

        assert slow_still_running
        np.testing.assert_allclose(
            other.densities.conjugation_shifts,
            StateDriftDensities.compute(
                _HAMILTONIAN, _OBSERVABLE, ZerosState()
            ).conjugation_shifts,
        )

    def test_a_request_during_an_evaluation_waits_for_its_result(self, mocker):
        spy = mocker.spy(StateDriftExpansion, "compute")
        cache = StateDriftCache()
        first = DickeState(2)
        entered, release = _hold_evaluation(mocker, first)
        matched = _signal_matches(mocker)
        owner, owned = _start(lambda: cache.expansion(_HAMILTONIAN, _OBSERVABLE, first))
        try:
            assert entered.wait(_WAIT)
            waiter, waited = _start(
                lambda: cache.expansion(_HAMILTONIAN, _OBSERVABLE, DickeState(2))
            )
            assert matched.wait(_WAIT)
        finally:
            release.set()
            owner.join(_WAIT)
        waiter.join(_WAIT)

        assert waited[0] is owned[0]
        assert spy.call_count == 1

    def test_a_failed_evaluation_reaches_its_waiters_and_is_retried(self, mocker):
        state = DickeState(2)
        failure = RuntimeError("evaluation failed")
        entered, release = _hold_evaluation(mocker, state, error=failure)
        matched = _signal_matches(mocker)
        spy = mocker.spy(StateDriftExpansion, "compute")
        cache = StateDriftCache()
        owner, owned = _start(lambda: cache.expansion(_HAMILTONIAN, _OBSERVABLE, state))
        try:
            assert entered.wait(_WAIT)
            waiter, waited = _start(
                lambda: cache.expansion(_HAMILTONIAN, _OBSERVABLE, DickeState(2))
            )
            assert matched.wait(_WAIT)
        finally:
            release.set()
            owner.join(_WAIT)
        waiter.join(_WAIT)

        assert owned == waited == [failure]
        retried = cache.expansion(_HAMILTONIAN, _OBSERVABLE, state)
        assert retried.densities.terms == _HAMILTONIAN.simplify()
        assert spy.call_count == 2

    def test_nearly_equal_hamiltonians_get_their_own_expansion(self):
        cache = StateDriftCache()
        first = cache.expansion(_HAMILTONIAN, _OBSERVABLE, ZerosState())
        nudged = cache.expansion((1 + 1e-7) * _HAMILTONIAN, _OBSERVABLE, ZerosState())
        assert nudged is not first


_DEPTH_OBSERVABLE = SparsePauliOp.from_list([("IIIZ", 1.0), ("ZIXI", -0.6)])
# _HAMILTONIAN without its odd-Y terms: every β_j is zero in a real state.
_REAL_HAMILTONIAN = SparsePauliOp.from_list(
    [
        ("IIZZ", 1.0),
        ("IZZI", 1.0),
        ("ZZII", 1.0),
        ("IIIX", 0.7),
        ("IIXI", 0.7),
        ("IXII", 0.7),
        ("XIII", 0.7),
        ("XXYY", -0.25),
    ]
)


def _distribution_probs(expansion, **kwargs):
    distribution = state_drift_distribution(expansion, **kwargs)
    return np.array(
        [distribution[label] for label in expansion.densities.terms.paulis.to_labels()]
    )


class TestStateDriftDistribution:
    @pytest.mark.parametrize("time", [0.05, 0.1])
    def test_bias_at_depth_is_below_weighted_qdrift(self, time):
        state = CustomPerQubitState("0+-1")
        expansion = StateDriftExpansion.compute(_HAMILTONIAN, _DEPTH_OBSERVABLE, state)
        terms = expansion.densities.terms
        probs = _distribution_probs(
            expansion, time=time, n_samples=8, n_trajectories=100
        )
        weighted = np.abs(terms.coeffs.real) / np.abs(terms.coeffs.real).sum()
        state_drift_bias = _exact_bias(terms, _DEPTH_OBSERVABLE, state, probs, time, 8)
        weighted_bias = _exact_bias(terms, _DEPTH_OBSERVABLE, state, weighted, time, 8)
        assert abs(state_drift_bias) < 0.8 * abs(weighted_bias)

    @pytest.mark.parametrize(
        "hamiltonian, time, n_samples",
        [
            (_HAMILTONIAN, 0.05, 8),
            (_REAL_HAMILTONIAN, 0.2, 8),
            (SparsePauliOp(["ZIII", "IIIY"], [1.0, 0.05]), 0.45 / 1.05, 1),
        ],
        ids=["complex", "real", "tight_angles"],
    )
    def test_distribution_keeps_every_bound(self, hamiltonian, time, n_samples):
        """Every term keeps a fifth of its weighted probability and an angle
        within 0.5, and the bias and variance stay within 5% of weighted
        sampling's."""
        expansion = StateDriftExpansion.compute(
            hamiltonian,
            SparsePauliOp.from_list([("IIIX", 1.0), ("XIII", 1.0), ("IIIZ", 0.5)]),
            CustomPerQubitState("0+-1"),
        )
        coeffs = np.abs(expansion.densities.terms.coeffs.real)
        weighted = coeffs / coeffs.sum()
        probs = _distribution_probs(
            expansion, time=time, n_samples=n_samples, n_trajectories=100
        )
        dt = time / n_samples

        def at(p):
            return (
                expansion.truncated_bias(p, dt=dt, n_samples=n_samples)[-1],
                expansion.truncated_variance(p, dt=dt, n_samples=n_samples)[-1],
            )

        (bias, variance), (weighted_bias, weighted_variance) = at(probs), at(weighted)
        assert probs.sum() == pytest.approx(1.0, abs=1e-12)
        assert np.all(probs >= 0.2 * weighted - 1e-15)
        assert np.max(coeffs * dt / probs) <= 0.5 + 1e-12
        assert abs(bias) <= 1.05 * abs(weighted_bias) + 1e-15
        assert 0 <= variance <= 1.05 * weighted_variance + 1e-15

    @pytest.mark.parametrize(
        "hamiltonian, observable, state, time, reason",
        [
            (
                SparsePauliOp.from_list([("ZI", 1.0), ("IZ", 3.0)]),
                SparsePauliOp("ZZ"),
                ZerosState(),
                0.1,
                "no Hamiltonian term changes the observable in the initial state",
            ),
            (
                SparsePauliOp.from_list([("XI", 1.0), ("IX", 3.0)]),
                SparsePauliOp("ZZ"),
                ZerosState(),
                1.0,
                "no distribution keeps every rotation angle |c_j|·dt/q_j within "
                "0.5 (dt = time / (n_steps · sampling_budget)); increase n_steps "
                "or sampling_budget",
            ),
            (
                _HAMILTONIAN,
                _DEPTH_OBSERVABLE,
                DickeState(2),
                0.05,
                "its fourth-order expansion is not reliable for this evolution "
                "(R_val = 1 ≤ 2)",
            ),
        ],
        ids=["unaffected_observable", "angles_too_large", "unreliable_expansion"],
    )
    def test_falls_back_to_weighted(self, hamiltonian, observable, state, time, reason):
        expansion = StateDriftExpansion.compute(hamiltonian, observable, state)
        with pytest.warns(
            UserWarning,
            match=exact_match(f"StateDRIFT falls back to weighted sampling: {reason}."),
        ):
            probs = _distribution_probs(
                expansion, time=time, n_samples=4, n_trajectories=10
            )
        coeffs = np.abs(expansion.densities.terms.coeffs.real)
        np.testing.assert_allclose(probs, coeffs / coeffs.sum())
