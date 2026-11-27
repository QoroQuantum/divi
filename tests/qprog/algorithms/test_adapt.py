# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for ADAPT-VQE, its operator pools and its growing ansatz."""

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations_with_replacement, count

import numpy as np
import pytest
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from scipy.linalg import expm
from scipy.optimize import OptimizeResult
from scipy.stats import chi2, norm

from divi.backends import MaestroConfig
from divi.circuits.zne import ZNE
from divi.qprog import EarlyStopping, StopReason
from divi.qprog.algorithms import AdaptAnsatz, AdaptVQE, ElectronCounts
from divi.qprog.algorithms.adapt import (
    CEOPool,
    CoupledExchange,
    OperatorPool,
    PauliPool,
    PauliRotation,
    PoolOperator,
    QubitExcitation,
    QubitExcitationPool,
    _build_landscape,
    _detect_cycle,
    _distribute_with_floor,
    _evaluate_landscape,
    _find_landscape_angles,
)
from divi.qprog.checkpointing import CheckpointConfig
from divi.qprog.optimizers import RotosolveOptimizer, ScipyMethod, ScipyOptimizer
from divi.qprog.problems import MolecularProblem
from tests._helpers import (
    H2_ATOM,
    equal_up_to_phase,
    exact_match,
    run_mcscf,
    run_mean_field,
    unitary,
)

LIH_ATOM = "Li 0 0 0; H 0 0 1.6"
H4_ATOM = "; ".join(f"H 0 0 {3.0 * i}" for i in range(4))

_SINGLE = QubitExcitation((0,), (2,))
_DOUBLE = QubitExcitation((0, 1), (2, 3))
_CROSSED_DOUBLE = QubitExcitation((0, 3), (1, 2))
_PAULI = PauliRotation((0, 1, 3), "XZY")
_COUPLED_PAIR = CoupledExchange((_DOUBLE, _CROSSED_DOUBLE))
_SAME_SPIN = (
    QubitExcitation((0, 2), (4, 6)),
    QubitExcitation((0, 4), (2, 6)),
    QubitExcitation((0, 6), (2, 4)),
)

_OPERATORS = {
    "single excitation": (_SINGLE, (0.7,)),
    "double excitation": (_DOUBLE, (0.7,)),
    "pauli rotation": (_PAULI, (0.7,)),
    "coupled sum": (CoupledExchange((_DOUBLE, _CROSSED_DOUBLE), (1, 1)), (0.7,)),
    "coupled difference": (
        CoupledExchange((_DOUBLE, _CROSSED_DOUBLE), (1, -1)),
        (0.7,),
    ),
    "coupled pair": (_COUPLED_PAIR, (0.7, -0.4)),
    "coupled triple": (CoupledExchange(_SAME_SPIN), (0.7, -0.4, 0.2)),
    **{
        f"same-spin {first}{'+' if sign > 0 else '-'}{second}": (
            CoupledExchange((_SAME_SPIN[first], _SAME_SPIN[second]), (1, sign)),
            (0.7,),
        )
        for first, second in ((0, 1), (0, 2), (1, 2))
        for sign in (1, -1)
    },
}
_EACH_OPERATOR = pytest.mark.parametrize(
    ("operator", "params"), _OPERATORS.values(), ids=_OPERATORS.keys()
)

#: Amplitude pairs of a one-parameter landscape: 1, cos(theta/2) - 1, sin(theta/2).
_ONE_ANGLE = tuple(combinations_with_replacement(range(3), 2))
_TWO_ANGLES = tuple(combinations_with_replacement(range(5), 2))

_NOTHING_APPENDED = (
    "No pool gradient at the reference state counts as nonzero, so no operator "
    "was appended."
)

_PREPROCESSOR_KEYS = count()


def _build_molecule(atom, n_orbitals, n_electrons):
    """``(problem, CASCI energy, Hartree-Fock energy)`` for an STO-3G active space."""
    mean_field = run_mean_field(atom)
    mcscf = pytest.importorskip("pyscf.mcscf")
    exact = run_mcscf(mcscf.CASCI, mean_field, n_orbitals, n_electrons).e_tot
    problem = MolecularProblem.from_casci(mean_field, n_orbitals, n_electrons)
    return problem, exact, mean_field.e_tot


@pytest.fixture(scope="module")
def lih():
    return _build_molecule(LIH_ATOM, 4, 4)


@pytest.fixture(scope="module")
def h2():
    return _build_molecule(H2_ATOM, 2, 2)


@pytest.fixture(scope="module")
def h4():
    """Linear H4 at 3 Å, CAS(4,4)."""
    return _build_molecule(H4_ATOM, 4, 4)


@pytest.fixture
def sampling_backend(make_maestro_simulator):
    return make_maestro_simulator(
        shots=2000, force_sampling=True, maestro_config=MaestroConfig(seed=7)
    )


@dataclass(frozen=True)
class _WideRotation(PoolOperator):
    """``exp(-i theta (Z_0 + Z_1))``, whose generator has eigenvalues ``0, +-1``."""

    @property
    def qubits(self) -> tuple[int, ...]:
        return (0, 1)

    def build_generators(self, n_qubits: int) -> tuple[SparsePauliOp, ...]:
        return (
            SparsePauliOp.from_sparse_list([("Z", [0], 1), ("Z", [1], 1)], n_qubits),
        )

    def emit(self, qc: QuantumCircuit, params: Sequence) -> None:
        (theta,) = params
        qc.rz(2 * theta, 0)
        qc.rz(2 * theta, 1)


class _TwoParameterPool(OperatorPool):
    """Offers one coupled exchange with a parameter per excitation."""

    def build_operators(self, n_qubits, occupied):
        return (_COUPLED_PAIR,)


def _build_adapt(problem, backend, **kwargs):
    return AdaptVQE(
        problem,
        backend=backend,
        optimizer=ScipyOptimizer(ScipyMethod.BFGS),
        **kwargs,
    )


def _run_adapt(problem, backend, **kwargs):
    return _build_adapt(problem, backend, **kwargs).run(perform_final_computation=False)


def _build_sampled(problem, backend, **kwargs):
    return AdaptVQE(
        problem,
        backend=backend,
        **{
            "optimizer": RotosolveOptimizer(),
            "max_iterations_per_step": 2,
            "seed": 11,
            **kwargs,
        },
    )


def _build_grown(problem, operators, backend):
    """An :class:`AdaptVQE` already holding ``operators``."""
    adapt = _build_adapt(problem, backend)
    adapt._set_operators(tuple(operators))
    return adapt


def _measure(adapt, observables, params, **kwargs):
    """``adapt``'s expectations of ``observables`` at ``params``."""
    preprocessor = adapt._build_tuple_preprocessor(
        tuple(observables), f"test_observables_{next(_PREPROCESSOR_KEYS)}"
    )
    return adapt.evaluate(np.atleast_2d(params), preprocessor, **kwargs)


def _place(pairs, **values):
    """Landscape values with ``values[f"v{m}{n}"]`` at pair ``(m, n)``."""
    out = np.zeros(len(pairs))
    for name, value in values.items():
        out[pairs.index((int(name[1]), int(name[2])))] = value
    return out


@_EACH_OPERATOR
def test_emitted_circuit_is_the_exponential_of_its_generators(
    operator, params, default_test_simulator
):
    n_qubits = max(operator.qubits) + 1
    circuit = QuantumCircuit(n_qubits)
    operator.emit(circuit, params)
    generator = sum(
        theta * g.to_matrix()
        for theta, g in zip(params, operator.build_generators(n_qubits), strict=True)
    )

    assert equal_up_to_phase(
        unitary(default_test_simulator, circuit), expm(-1j * generator)
    )


@_EACH_OPERATOR
def test_declared_frequencies_cover_every_generator_gap(operator, params):
    n_qubits = max(operator.qubits) + 1
    for generator, (omega, order) in zip(
        operator.build_generators(n_qubits), operator.frequencies, strict=True
    ):
        eigenvalues = np.linalg.eigvalsh(generator.to_matrix())
        gaps = np.abs(np.subtract.outer(eigenvalues, eigenvalues))
        allowed = omega * np.arange(order + 1)
        assert np.all(np.min(np.abs(gaps[..., None] - allowed), axis=-1) < 1e-9)


@pytest.mark.parametrize(
    ("operator", "n_cnots"),
    [
        (CoupledExchange((_DOUBLE, _CROSSED_DOUBLE), (1, -1)), 9),
        (CoupledExchange(_SAME_SPIN[1:], (1, 1)), 9),
        (_COUPLED_PAIR, 13),
        (CoupledExchange(_SAME_SPIN), 13),
    ],
    ids=["opposite-spin-sum", "same-spin-sum", "pair", "triple"],
)
def test_coupled_exchanges_take_the_papers_cnot_counts(operator, n_cnots):
    circuit = QuantumCircuit(max(operator.qubits) + 1)
    operator.emit(circuit, [0.3] * operator.n_params)

    assert circuit.count_ops()["cx"] == n_cnots


def test_a_coupled_exchange_needs_doubles_on_four_qubits():
    message = "A coupled exchange couples double excitations on one set of four qubits."
    with pytest.raises(ValueError, match=exact_match(message)):
        CoupledExchange((_DOUBLE, QubitExcitation((0, 1), (4, 5))))


def test_a_coupled_exchange_takes_each_pairing_once():
    message = "A coupled exchange takes each pairing of its four qubits at most once."
    with pytest.raises(ValueError, match=exact_match(message)):
        CoupledExchange((_DOUBLE, QubitExcitation((1, 0), (2, 3))))


def test_a_qubit_excitation_moves_one_or_two_qubits():
    message = (
        "A qubit excitation moves one or two qubits onto as many; got "
        "source=(0, 1, 2), target=(3, 4, 5)."
    )
    with pytest.raises(ValueError, match=exact_match(message)):
        QubitExcitation((0, 1, 2), (3, 4, 5))


class TestLandscape:
    @_EACH_OPERATOR
    def test_gives_the_energy_after_appending(
        self, operator, params, lih, default_test_simulator
    ):
        problem = lih[0]
        base, base_params = (_SINGLE, _DOUBLE), np.array([0.4, -0.3])
        hamiltonian = problem.hamiltonian
        observables, pairs = _build_landscape(operator, hamiltonian, 8)
        (values,) = _measure(
            _build_grown(problem, base, default_test_simulator),
            observables,
            base_params,
        ).values()
        grown = _build_grown(problem, (*base, operator), default_test_simulator)

        for thetas in np.random.default_rng(7).uniform(-6, 6, (3, operator.n_params)):
            ((energy,),) = _measure(
                grown, (hamiltonian,), np.concatenate([base_params, thetas])
            ).values()
            (predicted,) = _evaluate_landscape(pairs, np.asarray(values), thetas)
            assert predicted == pytest.approx(energy, abs=1e-7)

    def test_rejects_generators_outside_its_algebra(self, lih):
        with pytest.raises(ValueError, match="needs generators G with"):
            _build_landscape(_WideRotation(), lih[0].hamiltonian, 8)

    def test_evaluates_one_set_of_values_per_leading_row(self):
        rows = np.random.default_rng(3).normal(size=(2, len(_ONE_ANGLE)))
        thetas = np.array([[0.1], [1.2], [-2.0]])

        energies = _evaluate_landscape(_ONE_ANGLE, rows, thetas)

        assert energies.shape == (2, 3)
        for row, expected in zip(rows, energies, strict=True):
            np.testing.assert_allclose(
                _evaluate_landscape(_ONE_ANGLE, row, thetas), expected
            )

    def test_finds_the_exact_minimum_without_noise(self):
        # E = -1 + 0.3 sin(theta/2): lowest at theta = -pi, 0.3 below E(0).
        values = _place(_ONE_ANGLE, v00=-1.0, v02=0.3)

        angles, gain, gain_se = _find_landscape_angles(
            _ONE_ANGLE, values, np.zeros((6, 6)), 1, 1.96, np.random.default_rng(0)
        )

        np.testing.assert_allclose(angles, [-np.pi], atol=1e-5)
        assert gain == pytest.approx(0.3)
        assert gain_se == 0.0

    def test_prefers_the_minimum_nearer_zero_among_equals(self):
        # E = sin(theta) = 2 a_2 + 2 a_1 a_2: equal minima at -pi/2 and 3 pi/2.
        values = _place(_ONE_ANGLE, v02=2.0, v12=2.0)

        angles, gain, gain_se = _find_landscape_angles(
            _ONE_ANGLE,
            values,
            1e-6 * np.eye(len(_ONE_ANGLE)),
            1,
            1.96,
            np.random.default_rng(0),
        )

        np.testing.assert_allclose(angles, [-np.pi / 2], atol=1e-4)
        assert gain == pytest.approx(1.0, abs=1e-6)
        assert gain_se > 0.0

    @pytest.mark.parametrize(
        ("pairs", "n_params", "sd"),
        [(_TWO_ANGLES, 2, 0.0), (_ONE_ANGLE, 1, 1e-6)],
        ids=["noiseless-pair", "tiny-noise-single"],
    )
    def test_descends_below_every_grid_point_without_failing(self, pairs, n_params, sd):
        values = np.random.default_rng(5).normal(size=len(pairs))

        angles, gain, gain_se = _find_landscape_angles(
            pairs,
            values,
            sd**2 * np.eye(len(pairs)),
            n_params,
            1.96,
            np.random.default_rng(0),
        )

        assert angles.shape == (n_params,)
        assert np.isfinite(gain) and gain >= 0.0
        assert np.isfinite(gain_se)


@pytest.mark.parametrize(
    ("picks", "expected"),
    [
        (["a", "b", "c"], False),
        (["a", "b", "b"], True),
        (["d", "a", "b", "a", "b"], True),
        (["a", "b", "a", "c"], False),
    ],
)
def test_a_cycle_is_a_repeated_block_at_the_end(picks, expected):
    assert _detect_cycle(picks) is expected


def test_every_group_keeps_a_shot_floor():
    shots = _distribute_with_floor([5.0, 0.0, 1.0], 1000)

    assert sum(shots) == 1000
    assert min(shots) >= 100
    assert shots[0] > shots[2]


def test_counts_covariance_matches_the_spread_of_seeded_repeats(
    h2, make_maestro_simulator
):
    problem = h2[0]
    adapt = _build_grown(
        problem,
        (_DOUBLE,),
        make_maestro_simulator(shots=2000, force_sampling=True),
    )
    preprocessor = adapt._build_tuple_preprocessor(
        (
            problem.hamiltonian,
            SparsePauliOp.from_list([("XXYY", 0.5), ("ZIZI", 0.3)]),
        ),
        "calibration",
    )
    n_repeats = 40
    estimates, variances = [], []
    for seed in range(n_repeats):
        backend = make_maestro_simulator(
            shots=2000, force_sampling=True, maestro_config=MaestroConfig(seed=seed)
        )
        values, covariances = adapt.evaluate(
            np.array([[0.3]]), preprocessor, backend=backend, return_variance=True
        )
        estimates.append(values[0])
        variances.append(np.diag(covariances[0]))

    # The sample variance over the estimated one follows chi2(n - 1) / (n - 1).
    low, high = chi2.ppf([0.0005, 0.9995], n_repeats - 1) / (n_repeats - 1)
    ratio = np.var(estimates, axis=0, ddof=1) / np.mean(variances, axis=0)
    assert np.all((ratio > low) & (ratio < high))


class TestPools:
    def test_qubit_excitation_pool_holds_the_spin_conserving_excitations(self):
        assert QubitExcitationPool().build_operators(4, (0, 1)) == (
            _SINGLE,
            QubitExcitation((1,), (3,)),
            _DOUBLE,
        )

    def test_pauli_pool_holds_each_excitation_word_once(self):
        operators = PauliPool().build_operators(4, (0, 1))
        words_per_single, words_per_double = 2, 8

        assert len(operators) == 2 * words_per_single + words_per_double
        assert len(set(operators)) == len(operators)
        assert all(op.letters.count("Y") % 2 == 1 for op in operators)

    def test_ceo_pool_couples_the_excitations_sharing_four_qubits(self):
        pool = CEOPool()
        operators = pool.build_operators(4, (0, 1))

        assert operators == (
            _SINGLE,
            QubitExcitation((1,), (3,)),
            CoupledExchange((_DOUBLE, _CROSSED_DOUBLE), (1, 1)),
            CoupledExchange((_DOUBLE, _CROSSED_DOUBLE), (1, -1)),
            _DOUBLE,
            _CROSSED_DOUBLE,
        )
        assert pool.mark_candidates(operators).tolist() == [True] * 4 + [False] * 2

    @pytest.mark.parametrize(
        ("excitations_active", "n_params"),
        [((True, False), 1), ((True, True), 2)],
        ids=["one-active", "both-active"],
    )
    def test_ceo_pool_gives_each_active_excitation_its_own_parameter(
        self, excitations_active, n_params
    ):
        pool = CEOPool()
        operators = pool.build_operators(4, (0, 1))
        active = np.array([False, False, True, True, *excitations_active])

        assert pool.resolve(2, operators, active).n_params == n_params


class TestAdaptAnsatz:
    def test_fixed_parameters_are_bound_ahead_of_the_trainable_ones(
        self, default_test_simulator
    ):
        reference = ElectronCounts(2)
        operators = [_SINGLE, _DOUBLE]
        partly_fixed = AdaptAnsatz(operators, fixed_params=[0.3])

        assert partly_fixed.n_params_per_layer(4, reference) == 1
        assert equal_up_to_phase(
            unitary(
                default_test_simulator,
                partly_fixed.build([0.8], 4, 1, reference),
            ),
            unitary(
                default_test_simulator,
                AdaptAnsatz(operators).build([0.3, 0.8], 4, 1, reference),
            ),
        )

    def test_frequencies_follow_the_trainable_parameters(self):
        ansatz = AdaptAnsatz([_PAULI, _SINGLE, _COUPLED_PAIR], fixed_params=[0.1])

        assert (
            list(ansatz.parameter_frequencies(4, ElectronCounts(2))) == [(0.5, 2)] * 3
        )

    def test_each_layer_repeats_the_operators(self, default_test_simulator):
        reference = ElectronCounts(2)
        params = [0.3, 0.8, -0.4, 0.6]

        assert equal_up_to_phase(
            unitary(
                default_test_simulator,
                AdaptAnsatz([_SINGLE, _DOUBLE]).build(params, 4, 2, reference),
            ),
            unitary(
                default_test_simulator,
                AdaptAnsatz([_SINGLE, _DOUBLE, _SINGLE, _DOUBLE]).build(
                    params, 4, 1, reference
                ),
            ),
        )

    def test_fixed_parameters_need_a_single_layer(self):
        message = "An AdaptAnsatz with fixed parameters takes one layer; got 2."

        with pytest.raises(ValueError, match=exact_match(message)):
            AdaptAnsatz([_SINGLE, _DOUBLE], fixed_params=[0.3]).build(
                [0.8, 0.1], 4, 2, ElectronCounts(2)
            )


class TestAdaptVQE:
    @pytest.mark.e2e
    @pytest.mark.parametrize(
        "pool",
        [QubitExcitationPool(), PauliPool(), CEOPool()],
        ids=["qubit-excitation", "pauli", "ceo"],
    )
    @pytest.mark.parametrize("growth", ["single", "tetris"])
    def test_reaches_the_exact_energy(self, lih, pool, growth, default_test_simulator):
        problem, exact, _ = lih
        adapt = _run_adapt(problem, default_test_simulator, pool=pool, growth=growth)

        assert adapt.stop_reason is StopReason.CONVERGED
        assert adapt.best_loss == pytest.approx(exact, abs=1e-5)
        assert len(adapt.best_params) == sum(op.n_params for op in adapt.operators)

    @pytest.mark.e2e
    def test_first_step_takes_the_largest_reference_gradient(
        self, h2, default_test_simulator
    ):
        problem = h2[0]
        hamiltonian = problem.hamiltonian.to_matrix()
        reference = np.zeros(16)
        reference[0b0011] = 1.0

        def evaluate_energy(generator, theta):
            state = expm(-1j * theta * generator) @ reference
            return np.vdot(state, hamiltonian @ state).real

        step = 1e-5
        generators = [
            op.build_generators(4)[0].to_matrix()
            for op in QubitExcitationPool().build_operators(4, (0, 1))
        ]
        gradients = [
            (evaluate_energy(g, step) - evaluate_energy(g, -step)) / (2 * step)
            for g in generators
        ]

        adapt = _run_adapt(problem, default_test_simulator, max_steps=1)

        assert adapt.step_history[0]["max_gradient"] == pytest.approx(
            max(np.abs(gradients)), abs=1e-6
        )
        assert adapt.step_history[0]["operators"] == (_DOUBLE,)

    @pytest.mark.e2e
    def test_converges_on_the_last_allowed_step(self, h2, default_test_simulator):
        adapt = _run_adapt(h2[0], default_test_simulator, max_steps=1)

        assert len(adapt.step_history) == 1
        assert adapt.stop_reason is StopReason.CONVERGED

    @pytest.mark.e2e
    def test_samples_the_final_state(self, h2, default_test_simulator):
        adapt = _build_adapt(h2[0], default_test_simulator).run()

        assert adapt.operators
        assert adapt.best_probs

    @pytest.mark.e2e
    def test_tetris_appends_operators_on_disjoint_qubits(
        self, lih, default_test_simulator
    ):
        adapt = _run_adapt(lih[0], default_test_simulator, growth="tetris")

        first = adapt.step_history[0]["operators"]
        assert len(first) > 1
        qubits = [q for op in first for q in op.qubits]
        assert len(qubits) == len(set(qubits))

    @pytest.mark.e2e
    def test_reoptimising_only_new_parameters_returns_every_parameter(
        self, lih, default_test_simulator
    ):
        problem, exact, _ = lih
        adapt = _run_adapt(problem, default_test_simulator, reoptimize="new")

        assert adapt.best_loss == pytest.approx(exact, abs=1e-5)
        assert len(adapt.best_params) == adapt.step_history[-1]["n_params"]
        assert adapt.ansatz.fixed_params == ()

    @pytest.mark.e2e
    def test_reoptimising_only_new_parameters_stops_on_a_cycle(
        self, h4, default_test_simulator
    ):
        adapt = _run_adapt(h4[0], default_test_simulator, reoptimize="new")

        assert adapt.stop_reason is StopReason.CYCLE
        assert len(adapt.step_history) < adapt.max_steps

    @pytest.mark.e2e
    def test_bfgs_starts_each_step_from_the_padded_inverse_hessian(
        self, lih, default_test_simulator, mocker
    ):
        adapt = _build_adapt(lih[0], default_test_simulator)
        spy = mocker.spy(adapt.optimizer, "optimize")

        adapt.run(perform_final_computation=False)

        later_calls = spy.call_args_list[1:]
        assert later_calls
        for call, previous, step in zip(
            later_calls,
            spy.spy_return_list[:-1],
            adapt.step_history[1:],
            strict=True,
        ):
            start = call.kwargs["initial_inverse_hessian"]
            n_old = len(previous.hess_inv)
            assert start.shape == (step["n_params"], step["n_params"])
            np.testing.assert_allclose(
                start[:n_old, :n_old], previous.hess_inv, atol=1e-12
            )
            np.testing.assert_array_equal(
                start[n_old:, n_old:], np.eye(step["n_params"] - n_old)
            )

    @pytest.mark.parametrize(
        ("kwargs", "hess_inv", "n_previous"),
        [
            ({"reoptimize": "new"}, np.eye(2), 2),
            ({"optimizer": ScipyOptimizer(ScipyMethod.L_BFGS_B)}, np.eye(2), 2),
            ({}, np.eye(2), 3),
            ({}, -np.eye(2), 2),
        ],
        ids=["new-only", "not-bfgs", "shape-mismatch", "indefinite"],
    )
    def test_bfgs_warm_start_falls_back_to_a_cold_start(
        self, h2, default_test_simulator, kwargs, hess_inv, n_previous
    ):
        adapt = AdaptVQE(
            h2[0],
            backend=default_test_simulator,
            **{"optimizer": ScipyOptimizer(ScipyMethod.BFGS), **kwargs},
        )
        adapt.optimize_result = OptimizeResult(hess_inv=hess_inv)

        assert adapt._build_warm_start(n_previous, 1) == {}

    @pytest.mark.e2e
    def test_early_stopping_ends_the_growth(self, lih, default_test_simulator):
        adapt = _run_adapt(
            lih[0],
            default_test_simulator,
            early_stopping=EarlyStopping(patience=1, min_delta=1.0),
        )

        assert adapt.stop_reason is StopReason.PATIENCE_EXCEEDED
        assert len(adapt.step_history) == 2

    @pytest.mark.e2e
    def test_warns_when_the_step_budget_runs_out(self, lih, default_test_simulator):
        message = (
            "AdaptVQE reached max_steps=1 before its gradients fell below "
            "gradient_threshold."
        )

        with pytest.warns(UserWarning, match=exact_match(message)):
            adapt = _run_adapt(lih[0], default_test_simulator, max_steps=1)

        assert len(adapt.step_history) == 1
        assert adapt.stop_reason is StopReason.MAX_STEPS

    @pytest.mark.e2e
    @pytest.mark.parametrize(
        ("admission_threshold", "n_appended"),
        [(1e-5, 2), (1.0, 1)],
        ids=["low", "high"],
    )
    def test_tetris_admits_operators_by_the_admission_threshold(
        self, lih, admission_threshold, n_appended, default_test_simulator
    ):
        adapt = _run_adapt(
            lih[0],
            default_test_simulator,
            growth="tetris",
            admission_threshold=admission_threshold,
            early_stopping=EarlyStopping(patience=1, min_delta=1.0),
        )

        assert len(adapt.step_history[0]["operators"]) == n_appended

    @pytest.mark.e2e
    @pytest.mark.parametrize(
        "pool",
        [QubitExcitationPool(True), PauliPool(True), CEOPool(True)],
        ids=["qubit-excitation", "pauli", "ceo"],
    )
    def test_generalised_pools_reach_the_exact_energy(
        self, lih, pool, default_test_simulator
    ):
        problem, exact, _ = lih
        adapt = _run_adapt(problem, default_test_simulator, pool=pool)

        assert adapt.best_loss == pytest.approx(exact, abs=1e-5)

    @pytest.mark.e2e
    def test_a_converged_reference_appends_nothing(self, lih, default_test_simulator):
        adapt = _build_adapt(lih[0], default_test_simulator, gradient_threshold=1e6)

        with pytest.warns(UserWarning, match=exact_match(_NOTHING_APPENDED)):
            adapt.run()

        assert adapt.stop_reason is StopReason.CONVERGED
        assert adapt.operators == ()
        assert adapt.step_history == []
        assert adapt.has_results()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert np.isfinite(adapt.best_loss)

    def test_a_pool_operator_with_several_parameters_is_rejected(
        self, h2, default_test_simulator
    ):
        adapt = _build_adapt(h2[0], default_test_simulator, pool=_TwoParameterPool())
        message = "Pool operators whose gradient is measured need one parameter each."

        with pytest.raises(ValueError, match=exact_match(message)):
            adapt.run()

    @pytest.mark.parametrize(
        ("name", "replacement"),
        [
            ("ansatz", "pool"),
            ("n_layers", "max_steps"),
            ("max_iterations", "max_iterations_per_step"),
        ],
    )
    def test_rejects_arguments_it_sets_itself(
        self, lih, name, replacement, dummy_simulator
    ):
        message = (
            f"AdaptVQE grows its own ansatz; pass {replacement} instead of {name}."
        )
        with pytest.raises(TypeError, match=exact_match(message)):
            AdaptVQE(lih[0], backend=dummy_simulator, **{name: 1})

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"growth": "wide"}, "growth must be 'single' or 'tetris'; got 'wide'."),
            ({"reoptimize": "some"}, "reoptimize must be 'all' or 'new'; got 'some'."),
            ({"max_steps": 0}, "max_steps must be at least 1; got 0."),
            (
                {"significance": 1.0},
                "significance must lie between 0 and 1; got 1.0.",
            ),
            ({"shortlist_size": 0}, "shortlist_size must be at least 1; got 0."),
            ({"max_shot_scale": 3}, "max_shot_scale must be a power of two; got 3."),
            ({"max_shot_scale": 0}, "max_shot_scale must be a power of two; got 0."),
        ],
        ids=[
            "growth",
            "reoptimize",
            "max-steps",
            "significance",
            "shortlist",
            "shot-scale",
            "shot-scale-below-one",
        ],
    )
    def test_rejects_invalid_options(self, lih, kwargs, message, dummy_simulator):
        with pytest.raises(ValueError, match=exact_match(message)):
            AdaptVQE(lih[0], backend=dummy_simulator, **kwargs)

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            (
                {"initial_params": np.zeros(1)},
                "AdaptVQE starts from the reference state; initial_params is not "
                "supported.",
            ),
            (
                {"checkpoint_config": CheckpointConfig(checkpoint_dir="adapt")},
                "AdaptVQE does not support checkpointing.",
            ),
        ],
        ids=["initial-params", "checkpoint"],
    )
    def test_run_rejects_unsupported_options(
        self, lih, kwargs, message, dummy_simulator
    ):
        adapt = _build_adapt(lih[0], dummy_simulator)

        with pytest.raises(ValueError, match=exact_match(message)):
            adapt.run(**kwargs)

    def test_checkpoint_state_cannot_be_saved_or_loaded(
        self, lih, dummy_simulator, tmp_path
    ):
        adapt = _build_adapt(lih[0], dummy_simulator)
        message = "AdaptVQE does not support checkpointing."

        with pytest.raises(NotImplementedError, match=exact_match(message)):
            adapt.save_state(CheckpointConfig(checkpoint_dir=tmp_path))
        with pytest.raises(NotImplementedError, match=exact_match(message)):
            AdaptVQE.load_state(tmp_path, dummy_simulator)

    @pytest.mark.e2e
    def test_a_second_run_warns_and_leaves_the_result(
        self, lih, default_test_simulator
    ):
        adapt = _run_adapt(
            lih[0],
            default_test_simulator,
            early_stopping=EarlyStopping(patience=1, min_delta=1.0),
        )
        history = adapt.step_history
        message = (
            "This AdaptVQE has already run, so run() neither grows nor finalises "
            "it. Build a new program to run again."
        )

        with pytest.warns(UserWarning, match=exact_match(message)):
            adapt.run(perform_final_computation=False)

        assert adapt.step_history == history


class TestAdaptVQEOnASamplingBackend:
    @pytest.mark.e2e
    @pytest.mark.parametrize(
        ("pool", "growth", "first_kind"),
        [
            (QubitExcitationPool(), "single", QubitExcitation),
            (CEOPool(), "tetris", CoupledExchange),
        ],
        ids=["qubit-excitation", "ceo-tetris"],
    )
    def test_grows_to_the_exact_energy(
        self, h2, sampling_backend, pool, growth, first_kind
    ):
        problem, exact, hartree_fock = h2
        adapt = _build_sampled(
            problem, sampling_backend, pool=pool, growth=growth, significance=1e-3
        )
        adapt.run(perform_final_computation=False)

        assert adapt.stop_reason is StopReason.RESOLUTION_LIMITED
        assert adapt.best_loss == pytest.approx(exact, abs=5e-3)
        first = adapt.step_history[0]
        assert isinstance(first["operators"][0], first_kind)
        assert first["gains"][0] == pytest.approx(hartree_fock - exact, abs=5e-3)
        assert not any(
            isinstance(op, QubitExcitation) and len(op.source) == 2
            for op in adapt.operators
            if isinstance(pool, CEOPool)
        )

    @pytest.mark.e2e
    def test_measures_afresh_with_doubled_shots_until_resolution_runs_out(
        self, h2, sampling_backend, mocker
    ):
        adapt = _build_sampled(
            h2[0], sampling_backend, significance=1e-300, max_shot_scale=4
        )
        evaluate = mocker.spy(adapt, "evaluate")

        with pytest.warns(UserWarning, match=exact_match(_NOTHING_APPENDED)):
            adapt.run(perform_final_computation=False)

        unit = (
            adapt._count_groups(adapt._screening_observables) * sampling_backend.shots
        )
        budgets = [
            call.kwargs["shots"]
            for call in evaluate.call_args_list
            if call.args[1] is adapt._screening
        ]
        assert adapt.stop_reason is StopReason.RESOLUTION_LIMITED
        assert adapt.operators == ()
        assert budgets == [unit, 2 * unit, 4 * unit]

    @pytest.mark.e2e
    def test_without_early_stopping_spends_its_step_budget(self, h2, sampling_backend):
        adapt = _build_sampled(h2[0], sampling_backend, max_steps=1)
        message = (
            "AdaptVQE reached max_steps=1 while a pool gradient was still significant."
        )

        with pytest.warns(UserWarning, match=exact_match(message)):
            adapt.run(perform_final_computation=False)

        assert adapt.stop_reason is StopReason.MAX_STEPS
        assert len(adapt.step_history) == 1

    @pytest.mark.e2e
    def test_early_stopping_ends_the_growth(self, lih, sampling_backend):
        adapt = _build_sampled(
            lih[0],
            sampling_backend,
            max_iterations_per_step=1,
            early_stopping=EarlyStopping(patience=1, min_delta=1.0),
        )
        adapt.run(perform_final_computation=False)

        assert adapt.stop_reason is StopReason.PATIENCE_EXCEEDED
        assert len(adapt.step_history) == 2

    @pytest.mark.e2e
    def test_measures_step_and_final_energies_at_the_largest_shot_scale(
        self, h2, sampling_backend, mocker
    ):
        adapt = _build_sampled(h2[0], sampling_backend, max_shot_scale=8)
        evaluate = mocker.spy(adapt, "evaluate")
        cost_key = adapt.cost_preprocessor().cache_key
        adapt.run(perform_final_computation=False)

        budgets = [
            call.kwargs["shots"]
            for call in evaluate.call_args_list
            if call.args[1].cache_key == cost_key and call.kwargs["shots"] is not None
        ]
        assert budgets == [sampling_backend.shots * 8] * (len(adapt.step_history) + 1)

    @pytest.mark.e2e
    def test_reoptimising_only_new_parameters_runs_the_optimiser(
        self, h2, sampling_backend
    ):
        adapt = _build_sampled(
            h2[0], sampling_backend, reoptimize="new", significance=1e-3
        )
        adapt.run(perform_final_computation=False)

        assert adapt.step_history[0]["n_iterations"] >= 1
        assert len(adapt.best_params) == adapt.step_history[-1]["n_params"]
        assert adapt.stop_reason in (StopReason.RESOLUTION_LIMITED, StopReason.CYCLE)

    @pytest.mark.e2e
    def test_tetris_appends_operators_on_disjoint_qubits(self, h4, sampling_backend):
        adapt = _build_sampled(h4[0], sampling_backend, growth="tetris", max_steps=1)

        with pytest.warns(UserWarning, match="max_steps=1"):
            adapt.run(perform_final_computation=False)

        (step,) = adapt.step_history
        assert len(step["operators"]) == len(step["gains"]) > 1
        qubits = [q for op in step["operators"] for q in op.qubits]
        assert len(qubits) == len(set(qubits))

    def test_choosing_the_last_operator_again_turns_it_further(
        self, h2, sampling_backend, mocker
    ):
        adapt = _build_sampled(h2[0], sampling_backend, reoptimize="new")
        adapt._set_operators((_SINGLE, _DOUBLE))
        optimise = mocker.patch.object(
            adapt,
            "_optimise_step",
            side_effect=lambda fixed, trainable, added: np.concatenate(
                [fixed, trainable]
            ),
        )

        params, _ = adapt._extend(
            np.array([0.1, 0.3]), [(_DOUBLE, np.array([0.2]), 0.05)]
        )

        np.testing.assert_allclose(params, [0.1, 0.5])
        fixed, trainable, added = optimise.call_args.args
        np.testing.assert_allclose(fixed, [0.1])
        np.testing.assert_allclose(trainable, [0.5])
        assert added == ()

    @pytest.mark.parametrize(
        ("wider_gain", "n_params"), [(2.0, 2), (1.1, 1)], ids=["accept", "reject"]
    )
    def test_a_coupled_exchange_widens_only_when_that_gains_significantly(
        self, h2, sampling_backend, mocker, wider_gain, n_params
    ):
        adapt = _build_sampled(h2[0], sampling_backend, pool=CEOPool())

        def judge(operator, params):
            if operator.n_params == 2:
                return np.zeros(2), wider_gain, 0.1
            return np.zeros(1), 1.0, 0.1

        mocker.patch.object(adapt, "_judge", side_effect=judge)
        operators = adapt._pool_operators
        coupled_sum = operators.index(
            CoupledExchange((_DOUBLE, _CROSSED_DOUBLE), (1, 1))
        )

        ((chosen, _, _),) = adapt._choose_sampled(
            [coupled_sum], np.ones(len(operators), dtype=bool), np.zeros(0)
        )

        assert chosen.n_params == n_params

    def test_judges_only_the_shortlist(self, h2, sampling_backend, mocker):
        adapt = _build_sampled(h2[0], sampling_backend, shortlist_size=2)
        operators = adapt._pool_operators
        gains = {_DOUBLE: 0.5, _SINGLE: 0.9, QubitExcitation((1,), (3,)): 2.0}
        judge = mocker.patch.object(
            adapt,
            "_judge",
            side_effect=lambda operator, params: (np.zeros(1), gains[operator], 0.0),
        )
        order = [operators.index(op) for op in gains]

        ((chosen, _, gain),) = adapt._choose_sampled(
            order, np.ones(len(operators), dtype=bool), np.zeros(0)
        )

        assert [call.args[0] for call in judge.call_args_list] == [_DOUBLE, _SINGLE]
        assert (chosen, gain) == (_SINGLE, 0.9)

    def test_the_gate_holds_the_family_wise_significance(self, h2, sampling_backend):
        adapt = _build_sampled(h2[0], sampling_backend, significance=0.05)
        n_candidates, n_looks = 3, 5

        expected = norm.isf((1 - (1 - 0.05 / n_looks) ** (1 / n_candidates)) / 2)

        assert adapt._build_gate() == pytest.approx(expected)

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            (
                {"shot_distribution": "weighted_random"},
                "AdaptVQE on a sampling backend needs fixed shots per measurement "
                "group; shot_distribution='weighted_random' draws them at random.",
            ),
            (
                {"grouping_strategy": "shadows"},
                "AdaptVQE on a sampling backend measures by commuting groups; "
                "grouping_strategy='shadows' is not supported.",
            ),
            (
                {"qem_protocol": ZNE(scale_factors=[1.0, 3.0])},
                "AdaptVQE on a sampling backend takes its uncertainties from the "
                "measurement counts, which error mitigation does not carry; pass "
                "no qem_protocol.",
            ),
        ],
        ids=["weighted-random", "shadows", "zne"],
    )
    def test_rejects_estimates_without_count_uncertainties(
        self, h2, dummy_simulator, kwargs, message
    ):
        with pytest.raises(ValueError, match=exact_match(message)):
            _build_adapt(h2[0], dummy_simulator, **kwargs)
