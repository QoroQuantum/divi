# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Rotosolve optimizer."""

import numpy as np
import pytest

from divi.pipeline import ContractViolation
from divi.qprog import ParameterSpace
from divi.qprog.algorithms import VQE, UCCSDAnsatz
from divi.qprog.checkpointing import CheckpointConfig
from divi.qprog.optimizers import RotosolveOptimizer
from divi.qprog.problems import HamiltonianProblem, MolecularProblem
from tests._helpers import H2_ATOM, exact_match, run_mean_field
from tests.qprog.algorithms._helpers import needs_qiskit_nature
from tests.qprog.optimizers._checkpointing_contracts import (
    verify_save_creates_directory_if_needed,
)
from tests.qprog.optimizers._helpers import periodic_cost_fn_batch_aware, small_vqe

_INITIAL = np.array([0.4, 0.8, 1.2])


def _compute_separable_cosine_cost(params):
    """Minimum ``-n`` at ``x = pi + 0.1 * arange(n)``."""
    params = np.atleast_2d(params)
    return np.sum(np.cos(params - 0.1 * np.arange(params.shape[1])), axis=1)


def _compute_coupled_cost(params):
    x = np.atleast_2d(params)
    return (
        np.cos(x[:, 0]) * np.cos(x[:, 1])
        + 0.3 * np.sin(x[:, 0])
        + 0.2 * np.cos(x[:, 1] - x[:, 2])
    )


def _build_trigonometric_cost(omega, order, seed):
    """A random degree-``order`` trigonometric polynomial in ``omega * x[0]``."""
    cosines, sines = np.random.default_rng(seed).normal(size=(2, order))
    harmonics = np.arange(1, order + 1)

    def cost_fn(params):
        angles = omega * np.outer(np.atleast_2d(params)[:, 0], harmonics)
        return np.cos(angles) @ cosines + np.sin(angles) @ sines

    return cost_fn


def _record_batches(cost_fn):
    """``(calls, wrapped)``: ``wrapped`` appends a copy of every batch to ``calls``."""
    calls = []

    def wrapped(params):
        calls.append(np.array(params, dtype=np.float64))
        return cost_fn(params)

    return calls, wrapped


def _find_varied_columns(calls):
    """The column each batch with more than one row shifts."""
    return [
        int(np.flatnonzero(np.ptp(batch, axis=0))[0])
        for batch in calls
        if len(batch) > 1
    ]


@pytest.mark.parametrize("shuffle", [False, True], ids=["ordered", "shuffled"])
def test_optimizer_contract(optimizer_contract, shuffle):
    optimizer_contract(
        RotosolveOptimizer(shuffle=shuffle), periodic_cost_fn_batch_aware
    )


@pytest.mark.parametrize(("omega", "order"), [(1.0, 1), (1.0, 2), (0.5, 3), (2.0, 4)])
def test_one_sweep_reaches_the_global_minimum_of_one_parameter(omega, order):
    cost_fn = _build_trigonometric_cost(omega, order, seed=order)
    grid = np.linspace(-np.pi / omega, np.pi / omega, 400_001)
    values = cost_fn(grid[:, None])

    result = RotosolveOptimizer().optimize(
        cost_fn,
        np.array([0.4]),
        parameter_space=lambda: ParameterSpace(((omega, order),)),
        max_iterations=1,
    )

    assert result.fun[0] == pytest.approx(values.min(), abs=1e-8)
    phase = np.angle(np.exp(1j * omega * (result.x[0] - grid[np.argmin(values)])))
    assert phase == pytest.approx(0.0, abs=1e-3)


def test_each_step_batches_equidistant_shifts_of_one_parameter():
    calls, cost_fn = _record_batches(_compute_separable_cosine_cost)
    initial = np.array([0.2, 0.4, 0.6])

    RotosolveOptimizer().optimize(
        cost_fn,
        initial,
        parameter_space=lambda: ParameterSpace(((0.5, 2),) * 3),
        max_iterations=1,
    )

    assert [batch.shape for batch in calls] == [(5, 3)] * 3 + [(1, 3)]
    np.testing.assert_allclose(
        calls[0][:, 0] - initial[0], 4 * np.pi * np.arange(5) / 5
    )
    np.testing.assert_array_equal(calls[0][:, 1:], np.tile(initial[1:], (5, 1)))
    assert _find_varied_columns(calls) == [0, 1, 2]


def test_defaults_to_unit_frequencies_without_a_parameter_space():
    calls, cost_fn = _record_batches(_compute_separable_cosine_cost)

    result = RotosolveOptimizer().optimize(cost_fn, np.zeros(3), max_iterations=1)

    assert [batch.shape for batch in calls] == [(3, 3)] * 3 + [(1, 3)]
    assert result.fun[0] == pytest.approx(-3.0, abs=1e-12)


def test_reported_cost_is_measured_at_the_reported_parameters():
    trace = []

    result = RotosolveOptimizer().optimize(
        _compute_coupled_cost, _INITIAL, trace.append, max_iterations=3
    )

    for step in trace:
        np.testing.assert_allclose(step.fun, _compute_coupled_cost(step.x))
    np.testing.assert_allclose(result.fun, _compute_coupled_cost(result.x))
    assert [step.nit for step in trace] == [1, 2, 3]


def test_sweeps_never_raise_the_cost():
    trace = []

    RotosolveOptimizer().optimize(
        _compute_coupled_cost, _INITIAL, trace.append, max_iterations=5
    )

    costs = [step.fun[0] for step in trace]
    assert np.all(np.diff(costs) <= 1e-12)


def test_returns_the_best_measured_sweep():
    measured = 0

    def cost_fn(params):
        nonlocal measured
        values = _compute_coupled_cost(params)
        if len(values) == 1:
            measured += 1
            if measured > 1:
                values = values + 10.0
        return values

    trace = []
    result = RotosolveOptimizer().optimize(
        cost_fn, _INITIAL, trace.append, max_iterations=3
    )

    np.testing.assert_array_equal(result.x, trace[0].x[0])
    np.testing.assert_array_equal(result.fun, trace[0].fun)


def test_a_flat_direction_keeps_its_parameter():
    noise = np.random.default_rng(0).normal(scale=1e-16, size=7)

    result = RotosolveOptimizer().optimize(
        lambda params: 1.0 + noise[: len(np.atleast_2d(params))],
        np.array([0.4]),
        parameter_space=lambda: ParameterSpace(((1.0, 3),)),
        max_iterations=1,
    )

    np.testing.assert_array_equal(result.x, [0.4])


def test_shuffle_draws_a_fresh_reproducible_order_each_sweep():
    def sweep_order(rng, shuffle=True):
        calls, cost_fn = _record_batches(_compute_separable_cosine_cost)
        RotosolveOptimizer(shuffle=shuffle).optimize(
            cost_fn, np.zeros(6), max_iterations=2, rng=rng
        )
        return _find_varied_columns(calls)

    shuffled = sweep_order(np.random.default_rng(5))

    assert shuffled == sweep_order(np.random.default_rng(5))
    assert sorted(shuffled[:6]) == sorted(shuffled[6:]) == list(range(6))
    assert shuffled[:6] != shuffled[6:]
    assert sweep_order(np.random.default_rng(5), shuffle=False) == [*range(6)] * 2
    assert sorted(sweep_order(None)[:6]) == list(range(6))


def test_a_non_finite_cost_leaves_its_parameter_unchanged():
    calls = 0

    def cost_fn(params):
        nonlocal calls
        values = _compute_separable_cosine_cost(params)
        if calls == 0:
            values[1] = np.nan
        calls += 1
        return values

    result = RotosolveOptimizer().optimize(
        cost_fn, np.array([0.4, 0.8]), max_iterations=1
    )

    assert result.x[0] == 0.4
    assert result.x[1] == pytest.approx(np.pi + 0.1)


def test_reports_failure_when_no_measurement_is_finite():
    def cost_fn(params):
        values = _compute_coupled_cost(params)
        return np.full_like(values, np.nan) if len(values) == 1 else values

    result = RotosolveOptimizer().optimize(cost_fn, _INITIAL, max_iterations=2)

    assert not result.success


def test_binds_the_program_parameter_space(mocker):
    space = ParameterSpace(((1.0, 2),))
    program = mocker.Mock(parameter_space=space)

    evaluators = RotosolveOptimizer().build_evaluators(program)

    assert evaluators["parameter_space"]() is space


def test_vqe_passes_its_parameter_space_to_the_optimizer(
    default_test_simulator, mocker
):
    vqe = small_vqe(default_test_simulator, RotosolveOptimizer())
    vqe.max_iterations = 1
    optimize = mocker.spy(RotosolveOptimizer, "optimize")

    vqe.run(perform_final_computation=False)

    assert optimize.call_args.kwargs["parameter_space"]() is vqe.parameter_space


def test_rejects_parameters_with_unknown_frequencies(mocker):
    program = mocker.Mock(parameter_space=ParameterSpace(((1.0, 1), None, None)))

    with pytest.raises(
        ContractViolation,
        match=exact_match(
            "RotosolveOptimizer needs the frequencies of every parameter, but "
            "those of parameters [1, 2] are unknown. Use a gradient-free "
            "optimizer or SPSA."
        ),
    ):
        RotosolveOptimizer().validate_program(program)


@pytest.mark.parametrize(
    ("initial", "message"),
    [
        (None, "RotosolveOptimizer requires initial_params."),
        (np.zeros((2, 2)), "initial_params must contain exactly one parameter set."),
        (np.zeros((2, 1)), "initial_params must contain exactly one parameter set."),
        (np.array([]), "initial_params must contain at least one parameter."),
    ],
    ids=["no-initial-params", "two-sets", "two-one-parameter-sets", "no-parameters"],
)
def test_rejects_unusable_initial_params(initial, message):
    with pytest.raises(ValueError, match=exact_match(message)):
        RotosolveOptimizer().optimize(_compute_separable_cosine_cost, initial)


@pytest.mark.parametrize("resumed", [False, True], ids=["fresh", "resumed"])
def test_rejects_a_parameter_space_of_the_wrong_length(resumed, tmp_path):
    optimizer = RotosolveOptimizer()
    initial = _INITIAL
    if resumed:
        optimizer.optimize(_compute_coupled_cost, _INITIAL, max_iterations=1)
        optimizer.save_state(tmp_path)
        optimizer = RotosolveOptimizer.load_state(tmp_path)
        initial = None

    with pytest.raises(
        ValueError,
        match=exact_match(
            "The parameter space describes 2 parameters, but the iterate has 3."
        ),
    ):
        optimizer.optimize(
            _compute_coupled_cost,
            initial,
            parameter_space=lambda: ParameterSpace(((1.0, 1),) * 2),
        )


def test_rejects_a_batch_of_the_wrong_length():
    with pytest.raises(
        ValueError,
        match=exact_match("The cost function returned 1 values for 3 parameter sets."),
    ):
        RotosolveOptimizer().optimize(
            lambda params: np.zeros(1), _INITIAL, max_iterations=1
        )


def test_holds_resume_state_until_reset():
    optimizer = RotosolveOptimizer()
    assert not optimizer.has_resume_state

    optimizer.optimize(_compute_coupled_cost, _INITIAL, max_iterations=1)
    assert optimizer.has_resume_state

    optimizer.reset()
    assert not optimizer.has_resume_state


def test_config_and_copy_preserve_the_shuffle_setting():
    optimizer = RotosolveOptimizer(shuffle=True)
    optimizer.optimize(_compute_coupled_cost, _INITIAL, max_iterations=1)

    copied = optimizer.copy()

    assert optimizer.get_config() == {"type": "RotosolveOptimizer", "shuffle": True}
    assert copied.get_config() == optimizer.get_config()
    assert not copied.has_resume_state


def test_checkpoint_resume_matches_uninterrupted_run(tmp_path):
    expected = RotosolveOptimizer().optimize(
        _compute_coupled_cost, _INITIAL, max_iterations=3
    )

    interrupted = RotosolveOptimizer()
    interrupted.optimize(_compute_coupled_cost, _INITIAL, max_iterations=1)
    interrupted.save_state(tmp_path)
    resumed = RotosolveOptimizer.load_state(tmp_path)
    actual = resumed.optimize(_compute_coupled_cost, np.full(3, 99.0), max_iterations=2)

    assert actual.nit == expected.nit == 3
    np.testing.assert_allclose(actual.x, expected.x)
    np.testing.assert_allclose(actual.fun, expected.fun)


def test_rejects_checkpoint_before_first_iteration(tmp_path):
    with pytest.raises(
        RuntimeError,
        match=exact_match(
            "Cannot save checkpoint: Rotosolve optimisation has not been run."
        ),
    ):
        RotosolveOptimizer().save_state(tmp_path)


def test_save_state_creates_missing_directories(tmp_path):
    verify_save_creates_directory_if_needed(
        RotosolveOptimizer(),
        _INITIAL,
        _compute_coupled_cost,
        np.random.default_rng(0),
        tmp_path,
    )


def test_vqe_checkpoint_loads_and_resumes(default_test_simulator, tmp_path):
    vqe = small_vqe(default_test_simulator, RotosolveOptimizer())
    vqe.max_iterations = 2
    vqe.run(
        checkpoint_config=CheckpointConfig(checkpoint_dir=tmp_path),
        perform_final_computation=False,
    )

    resumed = VQE.load_state(
        tmp_path,
        backend=default_test_simulator,
        problem=HamiltonianProblem(vqe.cost_hamiltonian),
        ansatz=vqe.ansatz,
        n_layers=1,
    )
    resumed.max_iterations = 3
    resumed.run(perform_final_computation=False)

    assert isinstance(resumed.optimizer, RotosolveOptimizer)
    assert resumed.current_iteration == resumed.optimizer._nit == 3
    assert resumed.best_loss <= vqe.best_loss


@needs_qiskit_nature
@pytest.mark.e2e
def test_uccsd_h2_reaches_the_exact_ground_state_in_two_sweeps(
    default_test_simulator,
):
    problem = MolecularProblem.from_casci(run_mean_field(H2_ATOM), 2, 2)
    matrix = problem.hamiltonian.to_matrix()
    two_electrons = [
        index for index in range(len(matrix)) if bin(index).count("1") == 2
    ]
    exact = np.linalg.eigvalsh(matrix[np.ix_(two_electrons, two_electrons)])[0]
    vqe = VQE(
        problem,
        ansatz=UCCSDAnsatz(),
        optimizer=RotosolveOptimizer(),
        max_iterations=2,
        backend=default_test_simulator,
    )

    vqe.run(perform_final_computation=False)

    assert vqe.best_loss == pytest.approx(exact, abs=1e-8)
