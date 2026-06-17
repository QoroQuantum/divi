# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import pytest
from scipy.optimize import OptimizeResult

from divi.qprog.optimizers import (
    MonteCarloOptimizer,
    Optimizer,
    PymooOptimizer,
    QNGOptimizer,
    QNSPSAOptimizer,
    ScipyMethod,
    ScipyOptimizer,
    SPSAOptimizer,
)
from tests.qprog.optimizers._contracts import (
    CONTRACT_VARIANTS,
    sphere_cost_fn_batch_aware,
    sphere_cost_fn_population,
    sphere_cost_fn_single,
)


@pytest.fixture(params=CONTRACT_VARIANTS, ids=lambda param: param[0])
def contract_optimizer(request):
    """Optimizer-contract variants (the shared base set plus monte-carlo-keep-best)."""
    _, factory = request.param
    return factory()


class TestOptimizerContract:
    """A contract of tests that every optimizer should pass."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.n_params = 4
        self.rng = np.random.default_rng(42)

    def _get_initial_params(self, optimizer: Optimizer) -> np.ndarray:
        """Generate initial parameters with the correct shape for the optimizer."""
        shape = (optimizer.n_param_sets, self.n_params)
        return self.rng.random(shape) * 2 * np.pi

    def test_final_result_is_consistent(self, contract_optimizer: Optimizer):
        """Verify that the final returned parameters match the final cost."""
        initial_params = self._get_initial_params(contract_optimizer)
        cost_fn = (
            sphere_cost_fn_single
            if contract_optimizer.n_param_sets == 1
            else sphere_cost_fn_population
        )

        result = contract_optimizer.optimize(cost_fn, initial_params, max_iterations=5)

        assert isinstance(result, OptimizeResult)
        assert result.x.shape == (self.n_params,)
        recalculated_fun = sphere_cost_fn_single(result.x)
        assert np.isclose(
            result.fun, recalculated_fun
        ), "Final cost value should correspond to the final parameters."

    def test_supports_checkpointing_matches_get_config(
        self, contract_optimizer: Optimizer
    ):
        """``supports_checkpointing`` must agree with whether the optimizer can
        actually serialize its config (get_config raises for those that cannot)."""
        if contract_optimizer.supports_checkpointing:
            assert isinstance(contract_optimizer.get_config(), dict)
        else:
            with pytest.raises(NotImplementedError):
                contract_optimizer.get_config()

    def test_callback_provides_consistent_results(self, contract_optimizer: Optimizer):
        """Verify that parameters and costs are consistent in each callback call."""
        initial_params = self._get_initial_params(contract_optimizer)
        cost_fn = (
            sphere_cost_fn_single
            if contract_optimizer.n_param_sets == 1
            else sphere_cost_fn_population
        )

        callback_results = []

        def callback(intermediate_result: OptimizeResult):
            callback_results.append(intermediate_result)

        contract_optimizer.optimize(
            cost_fn, initial_params, callback_fn=callback, max_iterations=5
        )

        assert len(callback_results) > 0
        for res in callback_results:
            assert isinstance(res, OptimizeResult)
            assert res.x.shape == (contract_optimizer.n_param_sets, self.n_params)
            assert res.fun.shape == (contract_optimizer.n_param_sets,)

            if contract_optimizer.n_param_sets == 1:
                recalculated_fun = sphere_cost_fn_single(res.x)
            else:
                recalculated_fun = sphere_cost_fn_population(res.x)

            # L-BFGS-B can provide a stale loss value in its callback due to its
            # line-search mechanism. For this optimizer only, we relax the test
            # to confirm the loss is a valid number, but not that it's perfectly
            # in sync with the parameters, as that would require re-computation.
            if (
                isinstance(contract_optimizer, ScipyOptimizer)
                and contract_optimizer.method == ScipyMethod.L_BFGS_B
            ):
                assert np.isfinite(res.fun).all()
            else:
                assert np.allclose(res.fun, recalculated_fun)

    def test_reset_allows_reusing_optimizer(self, contract_optimizer: Optimizer):
        """Verify that reset() allows reusing an optimizer for fresh optimization runs."""
        initial_params = self._get_initial_params(contract_optimizer)
        cost_fn = (
            sphere_cost_fn_single
            if contract_optimizer.n_param_sets == 1
            else sphere_cost_fn_population
        )

        # Run first optimization
        result1 = contract_optimizer.optimize(cost_fn, initial_params, max_iterations=3)
        assert isinstance(result1, OptimizeResult)

        # Reset and run second optimization with different initial params
        contract_optimizer.reset()
        different_initial_params = self._get_initial_params(contract_optimizer)
        result2 = contract_optimizer.optimize(
            cost_fn, different_initial_params, max_iterations=3
        )
        assert isinstance(result2, OptimizeResult)

        # Both results should be valid
        assert result1.x.shape == (self.n_params,)
        assert result2.x.shape == (self.n_params,)
        assert np.isfinite(result1.fun)
        assert np.isfinite(result2.fun)

    def test_reset_preserves_optimizer_configuration(
        self, contract_optimizer: Optimizer
    ):
        """Verify that reset() does not affect optimizer configuration."""
        # Store original configuration
        if isinstance(contract_optimizer, ScipyOptimizer):
            original_method = contract_optimizer.method
        elif isinstance(contract_optimizer, PymooOptimizer):
            original_method = contract_optimizer.method
            original_pop_size = contract_optimizer.population_size
        elif isinstance(contract_optimizer, MonteCarloOptimizer):
            original_pop_size = contract_optimizer.population_size
            original_n_best = contract_optimizer.n_best_sets
            original_keep_best = contract_optimizer.keep_best_params
        else:
            pytest.fail(
                f"Unhandled optimizer type: {type(contract_optimizer).__name__}"
            )

        # Run optimization to set state
        initial_params = self._get_initial_params(contract_optimizer)
        cost_fn = (
            sphere_cost_fn_single
            if contract_optimizer.n_param_sets == 1
            else sphere_cost_fn_population
        )
        contract_optimizer.optimize(cost_fn, initial_params, max_iterations=2)

        # Reset and verify configuration is unchanged
        contract_optimizer.reset()

        if isinstance(contract_optimizer, ScipyOptimizer):
            assert contract_optimizer.method == original_method
        elif isinstance(contract_optimizer, PymooOptimizer):
            assert contract_optimizer.method == original_method
            assert contract_optimizer.population_size == original_pop_size
        elif isinstance(contract_optimizer, MonteCarloOptimizer):
            assert contract_optimizer.population_size == original_pop_size
            assert contract_optimizer.n_best_sets == original_n_best
            assert contract_optimizer.keep_best_params == original_keep_best
        else:
            pytest.fail(
                f"Unhandled optimizer type: {type(contract_optimizer).__name__}"
            )

    def test_reset_can_be_called_multiple_times(self, contract_optimizer: Optimizer):
        """Verify that reset() can be called multiple times safely."""
        initial_params = self._get_initial_params(contract_optimizer)
        cost_fn = (
            sphere_cost_fn_single
            if contract_optimizer.n_param_sets == 1
            else sphere_cost_fn_population
        )

        # Run optimization
        contract_optimizer.optimize(cost_fn, initial_params, max_iterations=2)

        # Call reset multiple times
        contract_optimizer.reset()
        contract_optimizer.reset()
        contract_optimizer.reset()

        # Should still be able to optimize after multiple resets
        result = contract_optimizer.optimize(cost_fn, initial_params, max_iterations=2)
        assert isinstance(result, OptimizeResult)
        assert result.x.shape == (self.n_params,)


# --------------------------------------------------------------------------- #
# Tier-2 contract: gradient / metric optimizers
# --------------------------------------------------------------------------- #


# (id, optimizer factory, optimize kwargs supplying the evaluators each needs).
_METRIC_GRADIENT_VARIANTS = [
    ("spsa", lambda: SPSAOptimizer(learning_rate=0.2, c=0.1), {}),
    (
        "qn-spsa",
        lambda: QNSPSAOptimizer(learning_rate=0.2, c=0.1),
        {"metric_fn": lambda x: np.eye(np.asarray(x).reshape(-1).shape[0])},
    ),
    (
        "qng",
        lambda: QNGOptimizer(step_size=0.2),
        {"jac": lambda x: 2 * x, "metric_fn": lambda x: np.eye(len(x))},
    ),
]


@pytest.fixture(params=_METRIC_GRADIENT_VARIANTS, ids=lambda param: param[0])
def metric_gradient_variant(request):
    """A gradient/metric optimizer bundled with the evaluators its ``optimize`` needs."""
    return request.param


class TestMetricGradientOptimizerContract:
    """Shared behavioural contract for the gradient/metric optimizers — QNG,
    QN-SPSA, SPSA.

    These cannot join :class:`TestOptimizerContract`: QNG/QN-SPSA require ``jac`` /
    ``metric_fn`` (or ``fidelity_fn``) evaluators rather than a bare cost function,
    and SPSA evaluates its perturbations as a single 2-row batch and reports a
    perturbation-average loss proxy — so ``result.fun == cost_fn(result.x)`` and the
    single-argument sphere cost from tier 1 do not apply. This tier supplies the
    evaluators and a batch-aware cost, and asserts the portion of the contract that
    *does* hold for all of them.
    """

    n_params = 3

    def _initial_params(self) -> np.ndarray:
        return np.full(self.n_params, 0.5)

    def test_callback_x_is_2d_fun_is_1d(self, metric_gradient_variant):
        _, factory, kwargs = metric_gradient_variant
        captured = []
        factory().optimize(
            sphere_cost_fn_batch_aware,
            initial_params=self._initial_params(),
            callback_fn=lambda res: captured.append((res.x, res.fun)),
            max_iterations=4,
            rng=np.random.default_rng(0),
            **kwargs,
        )
        assert len(captured) > 0
        for x, fun in captured:
            assert x.shape == (1, self.n_params)
            assert fun.shape == (1,)

    def test_final_result_x_is_1d(self, metric_gradient_variant):
        _, factory, kwargs = metric_gradient_variant
        result = factory().optimize(
            sphere_cost_fn_batch_aware,
            initial_params=self._initial_params(),
            max_iterations=5,
            rng=np.random.default_rng(0),
            **kwargs,
        )
        assert result.x.shape == (self.n_params,)
        assert np.isfinite(result.fun[0])

    def test_reset_allows_reuse(self, metric_gradient_variant):
        _, factory, kwargs = metric_gradient_variant
        opt = factory()
        opt.optimize(
            sphere_cost_fn_batch_aware,
            initial_params=self._initial_params(),
            max_iterations=3,
            rng=np.random.default_rng(0),
            **kwargs,
        )
        opt.reset()
        result = opt.optimize(
            sphere_cost_fn_batch_aware,
            initial_params=self._initial_params(),
            max_iterations=3,
            rng=np.random.default_rng(1),
            **kwargs,
        )
        assert np.isfinite(result.fun[0])

    def test_no_checkpointing(self, metric_gradient_variant):
        _, factory, _kwargs = metric_gradient_variant
        opt = factory()
        assert opt.supports_checkpointing is False
        with pytest.raises(NotImplementedError):
            opt.get_config()
