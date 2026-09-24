# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ROSALIN optimizer."""

from collections.abc import Sequence

import numpy as np
import pytest
from qiskit.circuit.library import RYGate, RZGate
from qiskit.quantum_info import SparsePauliOp

from divi.qprog.algorithms import VQE, GenericLayerAnsatz
from divi.qprog.checkpointing import CheckpointConfig
from divi.qprog.optimizers import RosalinOptimizer
from divi.qprog.optimizers._rosalin import _gcans_shots, _icans_shots_and_gains
from divi.qprog.problems import HamiltonianProblem
from divi.qprog.variational_quantum_algorithm import _compute_parameter_shift_rule


def _parameter_shift_cost(
    params: np.ndarray,
    *,
    estimator_samples: Sequence[int],
    return_variance: bool,
):
    values = np.sum(np.cos(np.atleast_2d(params)), axis=1)
    if not return_variance:
        return values
    samples = np.asarray(estimator_samples, dtype=np.float64)
    return values, 0.4 / samples


setattr(_parameter_shift_cost, "supports_variance", True)


def _multi_frequency_cost(
    params: np.ndarray,
    *,
    estimator_samples: Sequence[int],
    return_variance: bool,
):
    params = np.atleast_2d(params)
    values = np.cos(params[:, 0]) + 0.25 * np.cos(2 * params[:, 0])
    values += np.cos(params[:, 1])
    if not return_variance:
        return values
    samples = np.asarray(estimator_samples, dtype=np.float64)
    return values, 0.4 / samples


setattr(_multi_frequency_cost, "supports_variance", True)


def _controlled_gradient_cost(
    params: np.ndarray,
    *,
    estimator_samples: Sequence[int],
    return_variance: bool,
):
    """Return fixed gradients whose iCANS proposals are five and nine shots."""
    del params
    values = np.array([0.0, 0.1, -0.1, 0.1, -0.1])
    if not return_variance:
        return values

    samples = np.asarray(estimator_samples, dtype=np.float64)
    single_shot_variances = np.array([0.4, 0.8])
    variances = np.zeros(5, dtype=np.float64)
    variances[1:] = np.repeat(2 * single_shot_variances, 2) / samples[1:]
    return values, variances


setattr(_controlled_gradient_cost, "supports_variance", True)


def test_optimizer_contract(optimizer_contract):
    optimizer = RosalinOptimizer(
        learning_rate=0.05,
        total_shots=1000,
        lipschitz=10.0,
    )

    optimizer_contract(optimizer)


def test_icans_allocation_matches_the_paper_formula():
    shots, gains = _icans_shots_and_gains(
        gradient_ema=np.array([2.0, 1.0]),
        variance_ema=np.array([4.0, 1.0]),
        learning_rate=0.2,
        lipschitz=2.0,
        bias_term=0.25,
        min_shots=2,
    )

    factor = 2 * 2.0 * 0.2 / (2 - 2.0 * 0.2)
    expected = np.ceil(factor * np.array([4.0, 1.0]) / (np.array([4.0, 1.0]) + 0.25))
    expected = np.maximum(expected, 2).astype(int)
    expected_gains = (
        (0.2 - 2.0 * 0.2**2 / 2) * np.array([4.0, 1.0])
        - 2.0 * 0.2**2 * np.array([4.0, 1.0]) / (2 * expected)
    ) / expected

    np.testing.assert_array_equal(shots, expected)
    np.testing.assert_allclose(gains, expected_gains)


@pytest.mark.parametrize("min_shots", [2, 10, 50, 200])
def test_icans_caps_low_gain_flat_direction_at_best_gain_allocation(min_shots):
    learning_rate = 0.1
    lipschitz = 1.0
    bias = 1e-12
    gradients = np.array([0.2, 1e-4])
    raw_proposals = np.array([1.5, 4400.5])
    factor = 2 * lipschitz * learning_rate / (2 - lipschitz * learning_rate)
    variances = raw_proposals * (np.square(gradients) + bias) / factor

    shots, gains = _icans_shots_and_gains(
        gradients,
        variances,
        learning_rate=learning_rate,
        lipschitz=lipschitz,
        bias_term=bias,
        min_shots=min_shots,
        evaluation_counts=np.array([2, 2]),
    )

    np.testing.assert_array_equal(np.ceil(raw_proposals).astype(int), [2, 4401])
    assert np.argmax(gains / 2) == 0
    np.testing.assert_array_equal(shots, [min_shots, min_shots])


def test_gcans_allocation_matches_the_paper_formula():
    gradients = np.array([2.0, 1.0])
    variances = np.array([400.0, 100.0])

    shots = _gcans_shots(
        gradients,
        variances,
        learning_rate=0.2,
        lipschitz=2.0,
        bias_term=0.25,
        min_shots=2,
    )

    factor = 2 * 2.0 * 0.2 / (2 - 2.0 * 0.2)
    standard_deviations = np.sqrt(variances)
    expected = np.ceil(
        factor
        * standard_deviations
        * np.sum(standard_deviations)
        / (gradients @ gradients + 0.25)
    )
    np.testing.assert_array_equal(shots, expected.astype(int))


def test_gcans_accounts_for_generalized_shift_evaluation_costs():
    shots = _gcans_shots(
        np.array([0.2, 0.2]),
        np.array([1.0, 1.0]),
        learning_rate=0.1,
        lipschitz=1.0,
        bias_term=1e-12,
        min_shots=2,
        evaluation_counts=np.array([2, 8]),
    )

    assert shots[0] > shots[1]


def test_rosalin_allocation_can_grow_across_iterations():
    callbacks = []
    optimizer = RosalinOptimizer(
        learning_rate=0.1,
        total_shots=100,
        lipschitz=1.0,
        min_shots=2,
        ema_decay=0.0,
        bias=1e-12,
    )

    result = optimizer.optimize(
        _controlled_gradient_cost,
        np.zeros(2),
        callback_fn=callbacks.append,
        max_iterations=2,
    )

    np.testing.assert_array_equal(callbacks[0].shots_per_parameter, [2, 2])
    np.testing.assert_array_equal(callbacks[1].shots_per_parameter, [5, 5])
    assert result.shots_used == 32
    assert result.shots_used < optimizer.total_shots
    assert result.message == "Optimisation terminated: reached max_iterations."


def test_rosalin_can_use_gcans_allocation():
    callbacks = []
    optimizer = RosalinOptimizer(
        learning_rate=0.1,
        total_shots=100,
        lipschitz=1.0,
        min_shots=2,
        ema_decay=0.0,
        bias=1e-12,
        allocation="gcans",
    )

    result = optimizer.optimize(
        _controlled_gradient_cost,
        np.zeros(2),
        callback_fn=callbacks.append,
        max_iterations=2,
    )

    np.testing.assert_array_equal(callbacks[0].shots_per_parameter, [2, 2])
    np.testing.assert_array_equal(callbacks[1].shots_per_parameter, [6, 8])
    assert result.shots_used == 40


def test_rosalin_batches_current_loss_and_all_parameter_shifts():
    calls = []

    def cost_fn(params, *, estimator_samples, return_variance):
        calls.append((np.asarray(params).copy(), tuple(estimator_samples)))
        return _parameter_shift_cost(
            params,
            estimator_samples=estimator_samples,
            return_variance=return_variance,
        )

    setattr(cost_fn, "supports_variance", True)

    initial = np.array([0.2, 0.4, 0.6])
    optimizer = RosalinOptimizer(
        learning_rate=0.1,
        total_shots=14,
        lipschitz=3.0,
        min_shots=2,
    )

    result = optimizer.optimize(cost_fn, initial, max_iterations=10)

    assert len(calls) == 1
    params, samples = calls[0]
    assert params.shape == (7, 3)
    np.testing.assert_allclose(params[0], initial)
    np.testing.assert_allclose(params[1::2] - initial, np.eye(3) * (np.pi / 2))
    np.testing.assert_allclose(params[2::2] - initial, -np.eye(3) * (np.pi / 2))
    assert samples == (2,) * 7
    assert result.nit == 1
    assert result.shots_used == 14


def test_rosalin_recovers_gradient_and_single_shot_variance():
    callbacks = []
    initial = np.array([0.2, 0.4])
    optimizer = RosalinOptimizer(
        learning_rate=0.1,
        total_shots=10,
        lipschitz=2.0,
        min_shots=2,
    )

    optimizer.optimize(
        _parameter_shift_cost,
        initial,
        callback_fn=callbacks.append,
        max_iterations=1,
    )

    intermediate = callbacks[0]
    np.testing.assert_allclose(intermediate.jac[0], -np.sin(initial))
    np.testing.assert_allclose(intermediate.gradient_variance, np.full(2, 0.2))


def test_rosalin_uses_a_generalized_parameter_shift_rule():
    shifts, weights = _compute_parameter_shift_rule([(1.0, 2), (1.0, 1)])
    callbacks = []
    optimizer = RosalinOptimizer(
        learning_rate=0.1,
        total_shots=14,
        lipschitz=2.0,
        min_shots=2,
    )

    optimizer.optimize(
        _multi_frequency_cost,
        np.array([0.4, 0.8]),
        callback_fn=callbacks.append,
        shift_rule=lambda: (shifts, weights),
        max_iterations=1,
    )

    expected_gradient = np.array([-np.sin(0.4) - 0.5 * np.sin(0.8), -np.sin(0.8)])
    expected_variance = 0.4 * np.sum(np.square(weights), axis=1)
    np.testing.assert_allclose(callbacks[0].jac[0], expected_gradient)
    np.testing.assert_allclose(callbacks[0].gradient_variance, expected_variance)


def test_rosalin_binds_the_program_shift_rule(mocker):
    rule = _compute_parameter_shift_rule([(1.0, 2)])
    optimizer = RosalinOptimizer(learning_rate=0.1, total_shots=10, lipschitz=2.0)
    program = mocker.Mock(_grad_shift_rule=rule)

    evaluators = optimizer.build_evaluators(program)

    assert evaluators["shift_rule"]() is rule


def test_rosalin_never_overspends_the_total_shot_budget():
    requested = []

    def cost_fn(params, *, estimator_samples, return_variance):
        requested.extend(estimator_samples)
        return _parameter_shift_cost(
            params,
            estimator_samples=estimator_samples,
            return_variance=return_variance,
        )

    setattr(cost_fn, "supports_variance", True)

    optimizer = RosalinOptimizer(
        learning_rate=0.1,
        total_shots=39,
        lipschitz=2.0,
        min_shots=2,
    )
    result = optimizer.optimize(cost_fn, np.array([0.4, 0.8]), max_iterations=20)

    assert sum(requested) == result.shots_used
    assert result.shots_used <= optimizer.total_shots


def test_rosalin_scales_allocation_to_use_the_remaining_budget():
    optimizer = RosalinOptimizer(
        learning_rate=0.1,
        total_shots=39,
        lipschitz=2.0,
        min_shots=2,
    )

    result = optimizer.optimize(
        _parameter_shift_cost,
        np.zeros(2),
        max_iterations=20,
    )

    assert result.shots_used == 38
    assert result.nit == 2


def test_rosalin_converges_on_parameter_shift_objective():
    initial = np.full(3, 1.0)
    optimizer = RosalinOptimizer(
        learning_rate=0.15,
        total_shots=5000,
        lipschitz=3.0,
        min_shots=2,
        ema_decay=0.9,
    )

    result = optimizer.optimize(_parameter_shift_cost, initial, max_iterations=150)

    assert result.fun[0] < np.sum(np.cos(initial))
    assert np.sum(np.cos(result.x)) < -2.8


def test_rosalin_returns_last_evaluated_iterate_instead_of_noisy_minimum():
    calls = 0

    def cost_fn(params, *, estimator_samples, return_variance):
        nonlocal calls
        values, variances = _parameter_shift_cost(
            params,
            estimator_samples=estimator_samples,
            return_variance=return_variance,
        )
        values[0] = -100.0 if calls == 0 else 100.0
        calls += 1
        return values, variances

    setattr(cost_fn, "supports_variance", True)
    callbacks = []
    optimizer = RosalinOptimizer(
        learning_rate=0.1,
        total_shots=20,
        lipschitz=2.0,
        min_shots=2,
    )

    result = optimizer.optimize(
        cost_fn,
        np.array([0.4, 0.8]),
        callback_fn=callbacks.append,
        max_iterations=2,
    )

    np.testing.assert_allclose(result.x, callbacks[-1].x[0])
    assert result.fun == pytest.approx([100.0])
    assert callbacks[-1].track_best is False


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"learning_rate": 0.0}, "learning_rate"),
        ({"total_shots": 0}, "total_shots"),
        ({"lipschitz": 0.0}, "lipschitz"),
        ({"learning_rate": 1.0, "lipschitz": 2.0}, "2 / lipschitz"),
        ({"min_shots": 1}, "min_shots"),
        ({"ema_decay": 1.0}, "ema_decay"),
        ({"bias": 0.0}, "bias"),
        ({"allocation": "other"}, "allocation"),
    ],
)
def test_rosalin_constructor_validation(kwargs, match):
    defaults = {"learning_rate": 0.1, "total_shots": 100, "lipschitz": 2.0}
    defaults.update(kwargs)
    with pytest.raises(ValueError, match=match):
        RosalinOptimizer(**defaults)


def test_rosalin_requires_enough_budget_for_one_iteration():
    optimizer = RosalinOptimizer(
        learning_rate=0.1,
        total_shots=9,
        lipschitz=2.0,
        min_shots=2,
    )

    with pytest.raises(ValueError, match="one ROSALIN iteration"):
        optimizer.optimize(_parameter_shift_cost, np.array([0.2, 0.4]))


def test_rosalin_requires_variance_aware_cost_function():
    optimizer = RosalinOptimizer(
        learning_rate=0.1,
        total_shots=10,
        lipschitz=2.0,
        min_shots=2,
    )

    with pytest.raises(TypeError, match="variance-aware"):
        optimizer.optimize(lambda x: np.sum(np.asarray(x) ** 2), np.ones(2))


def test_rosalin_propagates_type_error_from_variance_aware_cost_function():
    def cost_fn(*args, **kwargs):
        raise TypeError("pipeline implementation failure")

    setattr(cost_fn, "supports_variance", True)
    optimizer = RosalinOptimizer(
        learning_rate=0.1,
        total_shots=10,
        lipschitz=2.0,
        min_shots=2,
    )

    with pytest.raises(TypeError, match="pipeline implementation failure"):
        optimizer.optimize(cost_fn, np.ones(2))


def test_rosalin_reports_budget_exhaustion_when_limits_coincide():
    optimizer = RosalinOptimizer(
        learning_rate=0.1,
        total_shots=10,
        lipschitz=2.0,
        min_shots=2,
    )

    result = optimizer.optimize(
        _parameter_shift_cost,
        np.ones(2),
        max_iterations=1,
    )

    assert result.message == "Optimisation terminated: total_shots budget exhausted."


def test_rosalin_requires_weighted_random_sampling(mocker):
    optimizer = RosalinOptimizer(
        learning_rate=0.1,
        total_shots=10,
        lipschitz=2.0,
    )

    with pytest.raises(ValueError, match="weighted_random"):
        optimizer.validate_program(mocker.Mock(_shot_distribution="weighted"))


@pytest.mark.parametrize("allocation", ["icans", "gcans"])
def test_rosalin_runs_through_weighted_random_vqe(sampling_test_simulator, allocation):
    optimizer = RosalinOptimizer(
        learning_rate=0.2,
        total_shots=120,
        lipschitz=1.0,
        allocation=allocation,
    )
    vqe = VQE(
        HamiltonianProblem(
            SparsePauliOp.from_list([("ZI", 0.5), ("IZ", -0.3), ("XX", 0.2)])
        ),
        ansatz=GenericLayerAnsatz([RYGate, RZGate]),
        n_layers=1,
        backend=sampling_test_simulator,
        optimizer=optimizer,
        grouping_strategy="qwc",
        shot_distribution="weighted_random",
        seed=1997,
    )
    vqe.max_iterations = 3

    vqe.run(perform_final_computation=False)

    assert 1 <= len(vqe.losses_history) <= 3
    assert np.isfinite(vqe.best_loss)
    assert vqe.best_loss == pytest.approx(vqe.optimize_result.fun[0])
    assert vqe.optimize_result.shots_used <= optimizer.total_shots
    assert vqe.optimize_result.device_shots_used == vqe.optimize_result.shots_used
    assert vqe.optimize_result.circuits_used > 0
    assert vqe.optimize_result.backend_jobs_used > 0


def test_rosalin_config_and_copy_preserve_constructor_settings():
    optimizer = RosalinOptimizer(
        learning_rate=0.1,
        total_shots=100,
        lipschitz=2.0,
        min_shots=3,
        ema_decay=0.8,
        bias=1e-5,
        allocation="gcans",
    )

    config = optimizer.get_config()
    copied = optimizer.copy()

    assert config == {
        "type": "RosalinOptimizer",
        "learning_rate": 0.1,
        "total_shots": 100,
        "lipschitz": 2.0,
        "min_shots": 3,
        "ema_decay": 0.8,
        "bias": 1e-5,
        "allocation": "gcans",
    }
    assert copied is not optimizer
    assert copied.get_config() == config


def test_rosalin_checkpoint_resume_matches_uninterrupted_run(tmp_path):
    settings = {
        "learning_rate": 0.1,
        "total_shots": 500,
        "lipschitz": 2.0,
        "ema_decay": 0.8,
        "allocation": "gcans",
    }
    initial = np.array([0.4, 0.8])
    uninterrupted = RosalinOptimizer(**settings)
    expected = uninterrupted.optimize(
        _parameter_shift_cost,
        initial,
        max_iterations=5,
    )

    interrupted = RosalinOptimizer(**settings)
    interrupted.optimize(_parameter_shift_cost, initial, max_iterations=2)
    interrupted.save_state(tmp_path)
    resumed = RosalinOptimizer.load_state(tmp_path)
    actual = resumed.optimize(
        _parameter_shift_cost,
        np.full_like(initial, 99.0),
        max_iterations=3,
    )

    assert resumed.supports_checkpointing is True
    assert resumed.get_config() == interrupted.get_config()
    assert actual.nit == expected.nit == 5
    assert actual.shots_used == expected.shots_used
    np.testing.assert_allclose(actual.x, expected.x)
    np.testing.assert_allclose(actual.fun, expected.fun)


def test_rosalin_rejects_checkpoint_before_first_iteration(tmp_path):
    optimizer = RosalinOptimizer(
        learning_rate=0.1,
        total_shots=100,
        lipschitz=2.0,
    )

    with pytest.raises(RuntimeError, match="has not been run"):
        optimizer.save_state(tmp_path)


def test_rosalin_vqe_checkpoint_loads_and_resumes(
    sampling_test_simulator,
    tmp_path,
):
    problem = HamiltonianProblem(
        SparsePauliOp.from_list([("ZI", 0.5), ("IZ", -0.3), ("XX", 0.2)])
    )
    ansatz = GenericLayerAnsatz([RYGate, RZGate])
    vqe = VQE(
        problem,
        ansatz=ansatz,
        n_layers=1,
        backend=sampling_test_simulator,
        optimizer=RosalinOptimizer(
            learning_rate=0.2,
            total_shots=200,
            lipschitz=1.0,
        ),
        grouping_strategy="qwc",
        shot_distribution="weighted_random",
        seed=1997,
    )
    vqe.max_iterations = 2
    vqe.run(
        checkpoint_config=CheckpointConfig(
            checkpoint_dir=tmp_path,
            checkpoint_interval=1,
        ),
        perform_final_computation=False,
    )

    resumed = VQE.load_state(
        tmp_path,
        backend=sampling_test_simulator,
        problem=problem,
        ansatz=ansatz,
        n_layers=1,
    )
    resumed.max_iterations = 3
    resumed.run(perform_final_computation=False)

    assert isinstance(resumed.optimizer, RosalinOptimizer)
    assert resumed.current_iteration == 3
    assert resumed.optimizer._nit == 3
    assert resumed.optimize_result.shots_used <= resumed.optimizer.total_shots
