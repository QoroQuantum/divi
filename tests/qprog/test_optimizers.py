# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

from divi.qprog.optimizers import (
    MonteCarloOptimizer,
    Optimizer,
    PymooMethod,
    PymooOptimizer,
    ScipyMethod,
    ScipyOptimizer,
)


def sphere_cost_fn_population(params: np.ndarray) -> np.ndarray:
    """Sphere cost function (sum of squares) for a population of parameter sets."""
    if params.ndim != 2:
        raise ValueError(
            "Input params for population cost function must be a 2D array."
        )
    return np.sum(params**2, axis=1)


def sphere_cost_fn_single(params: np.ndarray) -> float:
    """Sphere cost function (sum of squares) for a single parameter set."""
    if params.ndim != 1:
        # Allow single-row 2D array for convenience with some optimizers
        if params.ndim == 2 and params.shape[0] == 1:
            params = params.squeeze(0)
        else:
            raise ValueError(
                "Input params for single cost function must be a 1D array."
            )
    return float(np.sum(params**2))


@pytest.fixture(
    params=[
        ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
        ScipyOptimizer(method=ScipyMethod.COBYLA),
        ScipyOptimizer(method=ScipyMethod.L_BFGS_B),
        MonteCarloOptimizer(population_size=10, n_best_sets=3),
        PymooOptimizer(method=PymooMethod.CMAES, population_size=10),
        PymooOptimizer(method=PymooMethod.DE, population_size=10),
    ],
    ids=[
        "scipy-nelder-mead",
        "scipy-cobyla",
        "scipy-l-bfgs-b",
        "monte-carlo",
        "pymoo-cmaes",
        "pymoo-de",
    ],
)
def optimizer(request):
    """Provides various optimizer instances to be tested against the contract."""
    return request.param


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

    def test_final_result_is_consistent(self, optimizer: Optimizer):
        """Verify that the final returned parameters match the final cost."""
        initial_params = self._get_initial_params(optimizer)
        cost_fn = (
            sphere_cost_fn_single
            if optimizer.n_param_sets == 1
            else sphere_cost_fn_population
        )

        result = optimizer.optimize(cost_fn, initial_params, maxiter=5)

        assert isinstance(result, OptimizeResult)
        assert result.x.shape == (self.n_params,)
        recalculated_fun = sphere_cost_fn_single(result.x)
        assert np.isclose(
            result.fun, recalculated_fun
        ), "Final cost value should correspond to the final parameters."

    def test_callback_provides_consistent_results(self, optimizer: Optimizer):
        """Verify that parameters and costs are consistent in each callback call."""
        initial_params = self._get_initial_params(optimizer)
        cost_fn = (
            sphere_cost_fn_single
            if optimizer.n_param_sets == 1
            else sphere_cost_fn_population
        )

        callback_results = []

        def callback(intermediate_result: OptimizeResult):
            callback_results.append(intermediate_result)

        optimizer.optimize(cost_fn, initial_params, callback_fn=callback, maxiter=5)

        assert len(callback_results) > 0
        for res in callback_results:
            assert isinstance(res, OptimizeResult)
            assert res.x.shape == (optimizer.n_param_sets, self.n_params)
            assert res.fun.shape == (optimizer.n_param_sets,)

            if optimizer.n_param_sets == 1:
                recalculated_fun = sphere_cost_fn_single(res.x)
            else:
                recalculated_fun = sphere_cost_fn_population(res.x)

            # L-BFGS-B can provide a stale loss value in its callback due to its
            # line-search mechanism. For this optimizer only, we relax the test
            # to confirm the loss is a valid number, but not that it's perfectly
            # in sync with the parameters, as that would require re-computation.
            if (
                isinstance(optimizer, ScipyOptimizer)
                and optimizer.method == ScipyMethod.L_BFGS_B
            ):
                assert np.isfinite(res.fun).all()
            else:
                assert np.allclose(res.fun, recalculated_fun)
