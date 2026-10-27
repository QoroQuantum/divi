# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from threading import Event, Thread

import numpy as np
import pytest
from qiskit.quantum_info import SparsePauliOp
from scipy.linalg import expm

from divi.hamiltonians.trotterization import _state_drift
from divi.hamiltonians.trotterization._state_drift import (
    StateDriftCache,
    StateDriftDensities,
    state_drift_distribution,
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
        slopes.append((-1j * expval(obs @ term - term @ obs)).real)
    commutator = obs @ ham - ham @ obs
    curvature = expval(ham @ commutator - commutator @ ham).real
    return np.array(shifts), np.array(slopes), curvature


def _single_sample_bias(hamiltonian, observable, initial_state, probs, time):
    """Exact bias of one QDrift rotation at ``time`` sampled from ``probs``."""
    psi = initial_state_amplitudes(initial_state, hamiltonian.num_qubits)
    obs = observable.to_matrix()

    def evolved(unitary):
        state = unitary @ psi
        return (state.conj() @ obs @ state).real

    exact = evolved(expm(-1j * time * hamiltonian.to_matrix()))
    sampled = sum(
        prob * evolved(expm(-1j * coeff * time / prob * pauli.to_matrix()))
        for pauli, coeff, prob in zip(
            hamiltonian.paulis, hamiltonian.coeffs.real, probs
        )
    )
    return sampled - exact


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

    def test_curvature_is_the_same_when_evaluated_in_blocks(self, mocker):
        state = CustomPerQubitState("0+-1")
        whole = StateDriftDensities.compute(_HAMILTONIAN, _OBSERVABLE, state)
        mocker.patch.object(_state_drift, "_PAIR_BUDGET", 1)
        blocked = StateDriftDensities.compute(_HAMILTONIAN, _OBSERVABLE, state)
        assert blocked.curvature == pytest.approx(whole.curvature, abs=1e-12)

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


class TestStateDriftCache:
    def test_each_evolution_is_evaluated_once(self, mocker):
        state = DickeState(2)
        spy = mocker.spy(StateDriftDensities, "compute")
        cache = StateDriftCache()
        first = cache.densities(_HAMILTONIAN, _OBSERVABLE, state)
        again = cache.densities(_HAMILTONIAN.copy(), _OBSERVABLE.copy(), DickeState(2))
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
    def test_a_different_evolution_gets_its_own_densities(
        self, hamiltonian, observable, state
    ):
        cache = StateDriftCache()
        first = cache.densities(_HAMILTONIAN, _OBSERVABLE, DickeState(2))
        other = cache.densities(hamiltonian, observable, state)
        assert other is not first
        expected = StateDriftDensities.compute(hamiltonian, observable, state)
        np.testing.assert_allclose(
            other.conjugation_shifts, expected.conjugation_shifts
        )
        np.testing.assert_allclose(other.rotation_slopes, expected.rotation_slopes)

    def test_a_slow_evolution_does_not_hold_up_another(self, mocker):
        cache = StateDriftCache()
        slow = DickeState(2)
        entered, release = _hold_evaluation(mocker, slow)
        worker, _ = _start(lambda: cache.densities(_HAMILTONIAN, _OBSERVABLE, slow))
        try:
            assert entered.wait(_WAIT)
            other = cache.densities(_HAMILTONIAN, _OBSERVABLE, ZerosState())
            slow_still_running = worker.is_alive()
        finally:
            release.set()
            worker.join(_WAIT)

        assert slow_still_running
        np.testing.assert_allclose(
            other.conjugation_shifts,
            StateDriftDensities.compute(
                _HAMILTONIAN, _OBSERVABLE, ZerosState()
            ).conjugation_shifts,
        )

    def test_a_request_during_an_evaluation_waits_for_its_result(self, mocker):
        spy = mocker.spy(StateDriftDensities, "compute")
        cache = StateDriftCache()
        first = DickeState(2)
        entered, release = _hold_evaluation(mocker, first)
        matched = _signal_matches(mocker)
        owner, owned = _start(lambda: cache.densities(_HAMILTONIAN, _OBSERVABLE, first))
        try:
            assert entered.wait(_WAIT)
            waiter, waited = _start(
                lambda: cache.densities(_HAMILTONIAN, _OBSERVABLE, DickeState(2))
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
        spy = mocker.spy(StateDriftDensities, "compute")
        cache = StateDriftCache()
        owner, owned = _start(lambda: cache.densities(_HAMILTONIAN, _OBSERVABLE, state))
        try:
            assert entered.wait(_WAIT)
            waiter, waited = _start(
                lambda: cache.densities(_HAMILTONIAN, _OBSERVABLE, DickeState(2))
            )
            assert matched.wait(_WAIT)
        finally:
            release.set()
            owner.join(_WAIT)
        waiter.join(_WAIT)

        assert owned == waited == [failure]
        retried = cache.densities(_HAMILTONIAN, _OBSERVABLE, state)
        assert retried.terms == _HAMILTONIAN.simplify()
        assert spy.call_count == 2

    def test_nearly_equal_hamiltonians_get_their_own_densities(self):
        cache = StateDriftCache()
        first = cache.densities(_HAMILTONIAN, _OBSERVABLE, ZerosState())
        nudged = cache.densities((1 + 1e-7) * _HAMILTONIAN, _OBSERVABLE, ZerosState())
        assert nudged is not first


class TestStateDriftDistribution:
    @pytest.mark.parametrize(
        "initial_state, observable",
        [
            (CustomPerQubitState("0+1-"), SparsePauliOp("IIIZ")),
            (WState(2, 2), SparsePauliOp("IIZZ")),
            (DickeState(2), SparsePauliOp("IZII")),
        ],
        ids=["product", "w", "dicke"],
    )
    def test_single_sample_bias_is_below_weighted_qdrift(
        self, initial_state, observable
    ):
        time = 0.02
        distribution = state_drift_distribution(
            StateDriftDensities.compute(_HAMILTONIAN, observable, initial_state),
            time=time,
            n_samples=1,
            n_trajectories=100,
        )
        state_drift_bias = _single_sample_bias(
            _HAMILTONIAN, observable, initial_state, _ordered_probs(distribution), time
        )
        weighted_bias = _single_sample_bias(
            _HAMILTONIAN, observable, initial_state, _weighted_probs(), time
        )
        assert abs(state_drift_bias) < 0.5 * abs(weighted_bias)

    def test_every_term_keeps_a_fifth_of_its_weighted_probability(self):
        distribution = state_drift_distribution(
            StateDriftDensities.compute(
                _HAMILTONIAN, _OBSERVABLE, CustomPerQubitState("0+1-")
            ),
            time=0.02,
            n_samples=1,
            n_trajectories=10,
        )
        assert sum(distribution.values()) == pytest.approx(1.0, abs=1e-12)
        assert np.all(_ordered_probs(distribution) >= 0.2 * _weighted_probs() - 1e-15)

    @pytest.mark.parametrize(
        "hamiltonian, observable, time, reason",
        [
            (
                SparsePauliOp.from_list([("ZI", 1.0), ("IZ", 3.0)]),
                SparsePauliOp("ZZ"),
                0.1,
                "no Hamiltonian term changes the observable in the initial state",
            ),
            (
                SparsePauliOp.from_list([("XI", 1.0), ("IX", 3.0)]),
                SparsePauliOp("ZZ"),
                1.0,
                "no distribution keeps every rotation angle |c_j|·dt/q_j within "
                "0.5 (dt = time / (n_steps · sampling_budget)); increase n_steps "
                "or sampling_budget",
            ),
        ],
        ids=["unaffected_observable", "angles_too_large"],
    )
    def test_falls_back_to_weighted(self, hamiltonian, observable, time, reason):
        with pytest.warns(
            UserWarning,
            match=exact_match(f"StateDRIFT falls back to weighted sampling: {reason}."),
        ):
            distribution = state_drift_distribution(
                StateDriftDensities.compute(hamiltonian, observable, ZerosState()),
                time=time,
                n_samples=4,
                n_trajectories=10,
            )
        labels = hamiltonian.paulis.to_labels()
        assert distribution == pytest.approx({labels[0]: 0.25, labels[1]: 0.75})

    def test_every_rotation_angle_stays_within_the_bound(self):
        """At λ·dt = 0.45 the best-scoring distributions would rotate the small
        term by more than 0.5; the chosen one keeps every angle within it."""
        hamiltonian = SparsePauliOp(["ZI", "IY"], [1.0, 0.05])
        time = 0.45 / 1.05
        distribution = state_drift_distribution(
            StateDriftDensities.compute(
                hamiltonian, SparsePauliOp(["IX", "XI"]), CustomPerQubitState("0+")
            ),
            time=time,
            n_samples=1,
            n_trajectories=1,
        )
        probs = np.array(
            [distribution[label] for label in hamiltonian.paulis.to_labels()]
        )
        assert np.max(np.abs(hamiltonian.coeffs.real) * time / probs) <= 0.5
