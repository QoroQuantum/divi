# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Reusable behavioural checks for optimizer-specific test modules."""

import warnings
from collections.abc import Callable, Mapping
from copy import deepcopy
from inspect import Parameter, signature
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

from divi.qprog.optimizers import Optimizer, ScipyMethod, ScipyOptimizer
from tests.qprog.optimizers._helpers import (
    BOWL_CURVATURE,
    noisy_bowl,
    periodic_cost_fn_batch_aware,
    periodic_cost_fn_population,
    periodic_cost_fn_single,
    sphere_cost_fn_batch_aware,
)


def _initial_params(optimizer: Optimizer, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((optimizer.n_param_sets, 4)) * 2 * np.pi


def _variance_aware_periodic_cost(
    params: np.ndarray,
    *,
    estimator_samples=None,
    return_variance: bool = False,
):
    values = periodic_cost_fn_batch_aware(params)
    if not return_variance:
        return values
    return values, np.zeros(len(np.atleast_2d(params)), dtype=np.float64)


setattr(_variance_aware_periodic_cost, "supports_variance", True)


CostFn = Callable[[np.ndarray], float | np.ndarray]


def _resolve_cost_fn(optimizer: Optimizer, cost_fn: CostFn | None) -> CostFn:
    """``cost_fn``, or the periodic cost shaped for ``optimizer``."""
    if cost_fn is not None:
        return cost_fn
    if getattr(optimizer, "requires_variance", False):
        return _variance_aware_periodic_cost
    if optimizer.n_param_sets == 1:
        return periodic_cost_fn_single
    return periodic_cost_fn_population


def verify_final_result_is_consistent(
    optimizer: Optimizer, cost_fn: CostFn | None = None
) -> None:
    initial_params = _initial_params(optimizer)
    result = optimizer.optimize(
        _resolve_cost_fn(optimizer, cost_fn), initial_params, max_iterations=5
    )

    assert isinstance(result, OptimizeResult)
    assert result.x.shape == (4,)
    assert np.isclose(result.fun, periodic_cost_fn_single(result.x))


def verify_optimizer_descends_from_initial(
    optimizer: Optimizer, cost_fn: CostFn | None = None
) -> None:
    initial_params = _initial_params(optimizer)
    cost_fn = _resolve_cost_fn(optimizer, cost_fn)
    initial_cost = float(np.min(np.atleast_1d(cost_fn(initial_params))))

    result = optimizer.optimize(
        cost_fn, initial_params, max_iterations=20, rng=np.random.default_rng(7)
    )

    assert float(np.atleast_1d(result.fun)[0]) < initial_cost


def verify_checkpointing_capability_matches_config(
    optimizer: Optimizer, _cost_fn: CostFn | None = None
) -> None:
    if optimizer.supports_checkpointing:
        assert isinstance(optimizer.get_config(), dict)
    else:
        with pytest.raises(NotImplementedError):
            optimizer.get_config()


def verify_callback_results_are_consistent(
    optimizer: Optimizer,
    cost_fn: CostFn | None = None,
    *,
    require_matching_fun: bool | None = None,
) -> None:
    initial_params = _initial_params(optimizer)
    cost_fn = _resolve_cost_fn(optimizer, cost_fn)
    callback_results: list[OptimizeResult] = []

    optimizer.optimize(
        cost_fn,
        initial_params,
        callback_fn=callback_results.append,
        max_iterations=5,
    )

    if require_matching_fun is None:
        require_matching_fun = not (
            isinstance(optimizer, ScipyOptimizer)
            and optimizer.method == ScipyMethod.L_BFGS_B
        )

    assert callback_results
    for result in callback_results:
        assert isinstance(result, OptimizeResult)
        assert result.x.shape == (optimizer.n_param_sets, 4)
        assert result.fun.shape == (optimizer.n_param_sets,)
        if require_matching_fun:
            assert np.allclose(result.fun, cost_fn(result.x))
        else:
            assert np.isfinite(result.fun).all()


def verify_reset_allows_reuse(
    optimizer: Optimizer, cost_fn: CostFn | None = None
) -> None:
    cost_fn = _resolve_cost_fn(optimizer, cost_fn)
    first = optimizer.optimize(cost_fn, _initial_params(optimizer), max_iterations=3)

    optimizer.reset()
    second = optimizer.optimize(
        cost_fn, _initial_params(optimizer, seed=43), max_iterations=3
    )

    for result in (first, second):
        assert isinstance(result, OptimizeResult)
        assert result.x.shape == (4,)
        assert np.isfinite(result.fun)


def verify_reset_preserves_configuration(
    optimizer: Optimizer, cost_fn: CostFn | None = None
) -> None:
    """``reset()`` clears run state without touching constructor configuration.

    Compares configuration attributes with ``==``, so it applies only to
    optimizers whose constructor arguments are comparable scalars. An array
    (``GridSearchOptimizer.param_grid``) raises on the ambiguous truth value, and
    an object without ``__eq__`` (``QNSPSAOptimizer.metric_estimator``) compares
    unequal to its own deep copy.
    """
    constructor_parameters = signature(type(optimizer).__init__).parameters.values()
    attribute_names = [
        parameter.name
        for parameter in constructor_parameters
        if parameter.name != "self"
        and parameter.kind not in {Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD}
        and hasattr(optimizer, parameter.name)
    ]
    original = {name: deepcopy(getattr(optimizer, name)) for name in attribute_names}
    optimizer.optimize(
        _resolve_cost_fn(optimizer, cost_fn),
        _initial_params(optimizer),
        max_iterations=2,
    )

    optimizer.reset()

    assert {name: getattr(optimizer, name) for name in attribute_names} == original


def verify_reset_is_idempotent(
    optimizer: Optimizer, cost_fn: CostFn | None = None
) -> None:
    cost_fn = _resolve_cost_fn(optimizer, cost_fn)
    initial_params = _initial_params(optimizer)
    optimizer.optimize(cost_fn, initial_params, max_iterations=2)

    optimizer.reset()
    optimizer.reset()
    optimizer.reset()
    assert not optimizer.has_resume_state

    result = optimizer.optimize(cost_fn, initial_params, max_iterations=2)
    assert isinstance(result, OptimizeResult)
    assert result.x.shape == (4,)


def verify_gradient_callback_shapes(
    optimizer: Optimizer, optimize_kwargs: Mapping[str, Any]
) -> None:
    captured: list[tuple[np.ndarray, np.ndarray]] = []
    optimizer.optimize(
        sphere_cost_fn_batch_aware,
        initial_params=np.full(3, 0.5),
        callback_fn=lambda result: captured.append((result.x, result.fun)),
        max_iterations=4,
        rng=np.random.default_rng(0),
        **optimize_kwargs,
    )

    assert len(captured) == 4
    for params, loss in captured:
        assert params.shape == (1, 3)
        assert loss.shape == (1,)


def verify_gradient_callback_stop_iteration_propagates(
    optimizer: Optimizer, optimize_kwargs: Mapping[str, Any]
) -> None:
    def stop_at_second_step(result: OptimizeResult) -> None:
        if result.nit == 2:
            raise StopIteration

    with pytest.raises(StopIteration):
        optimizer.optimize(
            sphere_cost_fn_batch_aware,
            initial_params=np.full(3, 0.5),
            callback_fn=stop_at_second_step,
            max_iterations=10,
            rng=np.random.default_rng(0),
            **optimize_kwargs,
        )


def verify_gradient_final_result_shape(
    optimizer: Optimizer, optimize_kwargs: Mapping[str, Any]
) -> None:
    result = optimizer.optimize(
        sphere_cost_fn_batch_aware,
        initial_params=np.full(3, 0.5),
        max_iterations=5,
        rng=np.random.default_rng(0),
        **optimize_kwargs,
    )

    assert result.x.shape == (3,)
    assert np.isfinite(result.fun[0])


def verify_gradient_reset_allows_reuse(
    optimizer: Optimizer, optimize_kwargs: Mapping[str, Any]
) -> None:
    initial_params = np.full(3, 0.5)
    optimizer.optimize(
        sphere_cost_fn_batch_aware,
        initial_params=initial_params,
        max_iterations=3,
        rng=np.random.default_rng(0),
        **optimize_kwargs,
    )
    optimizer.reset()
    result = optimizer.optimize(
        sphere_cost_fn_batch_aware,
        initial_params=initial_params,
        max_iterations=3,
        rng=np.random.default_rng(1),
        **optimize_kwargs,
    )

    assert np.isfinite(result.fun[0])


def verify_no_checkpointing(
    optimizer: Optimizer, _optimize_kwargs: Mapping[str, Any]
) -> None:
    assert optimizer.supports_checkpointing is False
    with pytest.raises(NotImplementedError):
        optimizer.get_config()
    with TemporaryDirectory() as checkpoint_dir:
        with pytest.raises(NotImplementedError):
            optimizer.save_state(checkpoint_dir)
        with pytest.raises(NotImplementedError):
            type(optimizer).load_state(checkpoint_dir)


def _median_descent_ratio(
    factory: Callable[[], Optimizer],
    optimize_kwargs: Mapping[str, Any],
    make_cost: Callable[[np.random.Generator], tuple[Callable, Callable]],
    seeds: int = 12,
) -> float:
    ratios = []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        cost_fn, true_cost = make_cost(rng)
        start = rng.uniform(-1.0, 1.0, size=BOWL_CURVATURE.shape)
        result = factory().optimize(
            cost_fn,
            start,
            max_iterations=40,
            rng=rng,
            **optimize_kwargs,
        )

        assert result.success
        assert np.all(np.isfinite(np.atleast_1d(result.x)))
        ratios.append(float(true_cost(np.ravel(result.x))) / float(true_cost(start)))
    return float(np.median(ratios))


def verify_descends_under_shot_noise(
    factory: Callable[[], Optimizer],
    optimize_kwargs: Mapping[str, Any],
    ceiling: float,
) -> None:
    """Shot noise alone must not provoke the divergence warning, so no filter."""
    ratio = _median_descent_ratio(
        factory, optimize_kwargs, lambda rng: noisy_bowl(rng, sigma=1e-2)
    )
    assert ratio < ceiling


def verify_descends_despite_failed_evaluations(
    factory: Callable[[], Optimizer],
    optimize_kwargs: Mapping[str, Any],
    ceiling: float,
) -> None:
    """A nan loss legitimately reports divergence, so that warning is expected
    here and only here; the descent ratio is what the run is judged on."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=".*appears to be diverging.*", category=UserWarning
        )
        ratio = _median_descent_ratio(
            factory, optimize_kwargs, lambda rng: noisy_bowl(rng, dropout=0.05)
        )
    assert ratio < ceiling


OPTIMIZER_CONTRACTS = (
    verify_final_result_is_consistent,
    verify_optimizer_descends_from_initial,
    verify_checkpointing_capability_matches_config,
    verify_callback_results_are_consistent,
    verify_reset_allows_reuse,
    verify_reset_preserves_configuration,
    verify_reset_is_idempotent,
)

GRADIENT_OPTIMIZER_CONTRACTS = (
    verify_gradient_callback_shapes,
    verify_gradient_callback_stop_iteration_propagates,
    verify_gradient_final_result_shape,
    verify_gradient_reset_allows_reuse,
    verify_no_checkpointing,
)

NOISY_OPTIMIZER_CONTRACTS = (
    verify_descends_under_shot_noise,
    verify_descends_despite_failed_evaluations,
)
