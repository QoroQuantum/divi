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
        MonteCarloOptimizer(population_size=10, n_best_sets=3, keep_best_params=True),
        PymooOptimizer(method=PymooMethod.CMAES, population_size=10),
        PymooOptimizer(method=PymooMethod.DE, population_size=10),
    ],
    ids=[
        "scipy-nelder-mead",
        "scipy-cobyla",
        "scipy-l-bfgs-b",
        "monte-carlo",
        "monte-carlo-keep-best",
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


class TestMonteCarloOptimizer:
    """Specific tests for MonteCarloOptimizer features."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.n_params = 4
        self.rng = np.random.default_rng(42)
        self.population_size = 10
        self.n_best_sets = 3

    def _create_optimizer(self, keep_best_params: bool) -> MonteCarloOptimizer:
        """Helper to create a MonteCarloOptimizer with standard test parameters."""
        return MonteCarloOptimizer(
            population_size=self.population_size,
            n_best_sets=self.n_best_sets,
            keep_best_params=keep_best_params,
        )

    def _create_initial_params(self) -> np.ndarray:
        """Helper to create initial parameters with correct shape."""
        return self.rng.random((self.population_size, self.n_params)) * 2 * np.pi

    def _get_best_params(self, population: np.ndarray, n_best: int) -> np.ndarray:
        """Extract the best n_best parameter sets from a population."""
        losses = sphere_cost_fn_population(population)
        best_indices = np.argpartition(losses, n_best - 1)[:n_best]
        return population[best_indices]

    def _run_optimization_with_callback(
        self,
        optimizer: MonteCarloOptimizer,
        initial_params: np.ndarray,
        maxiter: int = 3,
    ) -> list[np.ndarray]:
        """Run optimization and collect all populations via callback."""
        all_populations = []

        def callback(intermediate_result: OptimizeResult):
            all_populations.append(intermediate_result.x.copy())

        optimizer.optimize(
            sphere_cost_fn_population,
            initial_params,
            callback_fn=callback,
            maxiter=maxiter,
            rng=self.rng,
        )
        return all_populations

    def test_keep_best_params_validation_and_property(self):
        """Test validation errors and property access."""
        # Test validation error
        with pytest.raises(
            ValueError,
            match="If keep_best_params is True, n_best_sets must be less than population_size.",
        ):
            MonteCarloOptimizer(
                population_size=10, n_best_sets=10, keep_best_params=True
            )

        # Test no error when keep_best_params=False
        optimizer_no_error = MonteCarloOptimizer(
            population_size=10, n_best_sets=10, keep_best_params=False
        )
        assert optimizer_no_error.keep_best_params is False

        # Test property returns correct values
        optimizer_false = self._create_optimizer(keep_best_params=False)
        optimizer_true = self._create_optimizer(keep_best_params=True)
        assert optimizer_false.keep_best_params is False
        assert optimizer_true.keep_best_params is True

    @pytest.mark.parametrize("keep_best_params", [True, False])
    def test_keep_best_params_behavior(self, keep_best_params: bool):
        """Test that keep_best_params correctly controls whether best parameters are kept."""
        optimizer = self._create_optimizer(keep_best_params=keep_best_params)
        initial_params = self._create_initial_params()

        all_populations = self._run_optimization_with_callback(
            optimizer, initial_params, maxiter=3
        )

        assert len(all_populations) >= 2

        # Get best params from initial population
        best_params_first = self._get_best_params(
            all_populations[0], optimizer.n_best_sets
        )

        # Check second population
        second_population = all_populations[1]
        assert second_population.shape == (optimizer.population_size, self.n_params)
        first_n_best_in_second = second_population[: optimizer.n_best_sets]

        # Check if each of the best parameters from the previous generation is present
        # in the best parameters of the new generation.
        is_present = [
            np.any(np.all(np.isclose(first_n_best_in_second, p), axis=1))
            for p in best_params_first
        ]

        if keep_best_params:
            # If we keep the best params, all of them should be present.
            assert all(
                is_present
            ), "Not all best parameters were kept when keep_best_params=True"
        else:
            # If we don't, none of them should be present as exact copies.
            assert not any(
                is_present
            ), "A best parameter was kept as an exact copy when keep_best_params=False"
