# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from scipy.optimize import OptimizeResult
from threadpoolctl import threadpool_info, threadpool_limits

from divi.qprog.optimizers import (
    ScipyMethod,
    ScipyOptimizer,
)
from tests._helpers import exact_match
from tests.qprog.optimizers._helpers import (
    sphere_cost_fn_single,
)

_SCIPY_STATE = (
    "Scipy's optimisation methods do not provide access to internal optimizer "
    "state during minimisation. Please use MonteCarloOptimizer or PymooOptimizer "
    "for checkpointing support."
)


def _blas_threads():
    return {
        lib["num_threads"] for lib in threadpool_info() if lib["user_api"] == "blas"
    }


def _minimise(cost_fn, method=ScipyMethod.NELDER_MEAD, max_iterations=2):
    return ScipyOptimizer(method=method).optimize(
        cost_fn, np.array([0.5, -0.5]), max_iterations=max_iterations
    )


def _failing_cost(x):
    raise RuntimeError("cost failed")


def _plainly(cost_fn, method):
    _minimise(cost_fn, method)


def _nested_in_another(cost_fn, method):
    def outer_cost(x):
        _minimise(sphere_cost_fn_single, ScipyMethod.COBYLA)
        return cost_fn(x)

    _minimise(outer_cost, method)


def _after_a_failed_run(cost_fn, method):
    with pytest.raises(RuntimeError, match="cost failed"):
        _minimise(_failing_cost, method)
    _minimise(cost_fn, method)


@pytest.fixture(
    params=[
        ScipyMethod.L_BFGS_B,
        ScipyMethod.SLSQP,
        ScipyMethod.COBYLA,
        ScipyMethod.NELDER_MEAD,
    ],
    ids=lambda method: method.value,
)
def scipy_contract_optimizer(request):
    return ScipyOptimizer(method=request.param)


def test_optimizer_contract(scipy_contract_optimizer, optimizer_contract):
    optimizer_contract(scipy_contract_optimizer)


class TestScipyOptimizer:
    """Specific tests for ScipyOptimizer features."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.n_params = 4
        self.rng = np.random.default_rng(42)

    @pytest.mark.parametrize(
        ("call", "message"),
        [
            (
                lambda tmp_path: ScipyOptimizer(ScipyMethod.L_BFGS_B).save_state(
                    tmp_path / "checkpoint"
                ),
                f"ScipyOptimizer does not support state saving. {_SCIPY_STATE}",
            ),
            (
                lambda tmp_path: ScipyOptimizer.load_state(tmp_path / "checkpoint"),
                f"ScipyOptimizer does not support state loading. {_SCIPY_STATE}",
            ),
            (
                lambda tmp_path: ScipyOptimizer(ScipyMethod.L_BFGS_B).get_config(),
                "ScipyOptimizer does not support checkpointing. Please use "
                "MonteCarloOptimizer or PymooOptimizer for checkpointing support.",
            ),
        ],
        ids=["save_state", "load_state", "get_config"],
    )
    def test_checkpointing_entry_points_raise_not_implemented(
        self, call, message, tmp_path
    ):
        with pytest.raises(NotImplementedError, match=exact_match(message)):
            call(tmp_path)

    def test_optimize_without_initial_params_raises(self):
        """ScipyOptimizer cannot resume, so initial_params is always required."""
        optimizer = ScipyOptimizer(method=ScipyMethod.L_BFGS_B)

        with pytest.raises(
            ValueError, match=exact_match("ScipyOptimizer requires initial_params.")
        ):
            optimizer.optimize(sphere_cost_fn_single, max_iterations=3)

    def test_optimize_without_max_iterations_runs_to_convergence(self):
        result = ScipyOptimizer(method=ScipyMethod.NELDER_MEAD).optimize(
            sphere_cost_fn_single, np.array([0.5, -0.5])
        )

        assert result.success
        np.testing.assert_allclose(result.x, [0.0, 0.0], atol=1e-3)

    @pytest.mark.parametrize(
        ("method", "expects_jac"),
        [
            (ScipyMethod.L_BFGS_B, True),
            (ScipyMethod.SLSQP, True),
            (ScipyMethod.NELDER_MEAD, False),
            (ScipyMethod.COBYLA, False),
        ],
    )
    def test_optimize_passes_jac_only_to_gradient_methods(
        self, method: ScipyMethod, expects_jac: bool, monkeypatch
    ):
        """Ensure jacobian callbacks are only forwarded when supported."""
        recorded = {}

        def fake_minimize(cost_fn, x0, **kwargs):
            recorded["jac"] = kwargs.get("jac")
            recorded["method"] = kwargs.get("method")
            recorded["callback"] = kwargs.get("callback")
            return OptimizeResult(x=x0, fun=0.0)

        monkeypatch.setattr("divi.qprog.optimizers._scipy.minimize", fake_minimize)

        optimizer = ScipyOptimizer(method=method)

        def jac_fn(x):
            return x

        initial_params = self.rng.random((1, self.n_params)) * 0.1

        optimizer.optimize(
            sphere_cost_fn_single,
            initial_params,
            jac=jac_fn,
            max_iterations=5,
        )

        if expects_jac:
            assert recorded["jac"] is jac_fn
        else:
            assert recorded["jac"] is None
        assert recorded["method"] == method.value

    def test_callback_shapes_are_normalized(self, monkeypatch):
        """Callbacks should always observe 2D parameters and 1D losses."""
        captured_shapes: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

        def fake_minimize(cost_fn, x0, **kwargs):
            callback = kwargs.get("callback")
            assert callback is not None
            callback(OptimizeResult(x=np.ones(self.n_params), fun=1.23))
            return OptimizeResult(x=x0, fun=0.0)

        monkeypatch.setattr("divi.qprog.optimizers._scipy.minimize", fake_minimize)

        optimizer = ScipyOptimizer(method=ScipyMethod.NELDER_MEAD)
        initial_params = self.rng.random((1, self.n_params)) * 0.1

        def tracking_callback(result: OptimizeResult):
            captured_shapes.append((result.x.shape, result.fun.shape))

        optimizer.optimize(
            sphere_cost_fn_single,
            initial_params,
            max_iterations=3,
            callback_fn=tracking_callback,
        )

        assert captured_shapes == [((1, self.n_params), (1,))]

    @pytest.mark.parametrize(
        "run",
        [_plainly, _nested_in_another, _after_a_failed_run],
        ids=["plain", "nested", "after-a-failure"],
    )
    @pytest.mark.parametrize("method", list(ScipyMethod), ids=lambda m: m.value)
    def test_blas_is_single_threaded_only_while_minimising(self, method, run):
        """A thread per core in every process would starve concurrent runs."""
        observed = []

        def cost_fn(x):
            observed.append(_blas_threads())
            return sphere_cost_fn_single(x)

        with threadpool_limits(limits=2, user_api="blas"):
            before = _blas_threads()
            run(cost_fn, method)
            assert _blas_threads() == before

        assert 2 in before
        assert observed and all(threads == {1} for threads in observed)

    def test_copy_returns_fresh_optimizer(self):
        optimizer = ScipyOptimizer(method=ScipyMethod.COBYLA)
        copied = optimizer.copy()

        assert isinstance(copied, ScipyOptimizer)
        assert copied.method == optimizer.method
