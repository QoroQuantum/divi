# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import json

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

from divi.qprog.checkpointing import OPTIMIZER_STATE_FILE
from divi.qprog.optimizers import (
    MonteCarloOptimizer,
    Optimizer,
)
from tests._helpers import exact_match
from tests.qprog.optimizers._checkpointing_contracts import (
    verify_load_state_raises_file_not_found,
    verify_load_state_rejects_corrupted_state,
    verify_save_creates_checkpoint_file,
    verify_save_creates_directory_if_needed,
    verify_save_load_round_trip,
)
from tests.qprog.optimizers._helpers import sphere_cost_fn_population

#: Four two-parameter sets, row ``i`` holding ``[2i, 2i + 1]``.
_POPULATION = np.arange(8.0).reshape(4, 2)


def _small_optimizer() -> MonteCarloOptimizer:
    return MonteCarloOptimizer(population_size=4, n_best_sets=2)


def _constant_losses(*losses):
    return lambda population: np.array(losses, dtype=np.float64)


@pytest.fixture(params=[False, True], ids=["default", "keep-best"])
def monte_carlo_contract_optimizer(request):
    return MonteCarloOptimizer(
        population_size=10, n_best_sets=3, keep_best_params=request.param
    )


def test_optimizer_contract(monte_carlo_contract_optimizer, optimizer_contract):
    optimizer_contract(monte_carlo_contract_optimizer)


class _CustomMonteCarloOptimizer(MonteCarloOptimizer):
    pass


def test_copy_preserves_subclass_type():
    optimizer = _CustomMonteCarloOptimizer(population_size=5, n_best_sets=2)
    assert type(optimizer.copy()) is _CustomMonteCarloOptimizer


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
        max_iterations: int = 3,
    ) -> list[np.ndarray]:
        """Run optimization and collect all populations via callback."""
        all_populations = []

        def callback(intermediate_result: OptimizeResult):
            all_populations.append(intermediate_result.x.copy())

        optimizer.optimize(
            sphere_cost_fn_population,
            initial_params,
            callback_fn=callback,
            max_iterations=max_iterations,
            rng=self.rng,
        )
        return all_populations

    def test_keep_best_params_validation_and_property(self):
        with pytest.raises(
            ValueError,
            match=exact_match(
                "If keep_best_params is True, n_best_sets must be less than "
                "population_size."
            ),
        ):
            MonteCarloOptimizer(
                population_size=10, n_best_sets=10, keep_best_params=True
            )

        optimizer_no_error = MonteCarloOptimizer(
            population_size=10, n_best_sets=10, keep_best_params=False
        )
        assert optimizer_no_error.keep_best_params is False

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
            optimizer, initial_params, max_iterations=3
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

    def test_save_state_creates_checkpoint_file(self, tmp_path):
        """Test that save_state() creates the expected checkpoint file."""
        optimizer = self._create_optimizer(keep_best_params=False)
        initial_params = self._create_initial_params()
        verify_save_creates_checkpoint_file(
            optimizer,
            initial_params,
            sphere_cost_fn_population,
            self.rng,
            tmp_path,
        )

    def test_save_state_preserves_configuration(self, tmp_path):
        """Test that save_state() saves optimizer configuration correctly."""
        optimizer = MonteCarloOptimizer(
            population_size=6, n_best_sets=2, keep_best_params=True
        )
        initial_params = self.rng.random((6, self.n_params)) * 2 * np.pi

        # Run optimization to set state
        optimizer.optimize(
            sphere_cost_fn_population, initial_params, max_iterations=2, rng=self.rng
        )

        checkpoint_dir = str(tmp_path / "checkpoint")
        optimizer.save_state(checkpoint_dir)

        # Load and verify configuration
        loaded_optimizer = MonteCarloOptimizer.load_state(checkpoint_dir)
        assert loaded_optimizer.population_size == optimizer.population_size
        assert loaded_optimizer.n_best_sets == optimizer.n_best_sets
        assert loaded_optimizer.keep_best_params == optimizer.keep_best_params

    def test_load_state_restores_optimization_state(self, tmp_path):
        """Test that load_state() correctly restores all optimization state."""
        optimizer = self._create_optimizer(keep_best_params=False)
        initial_params = self._create_initial_params()

        # Run optimization to set state
        optimizer.optimize(
            sphere_cost_fn_population, initial_params, max_iterations=3, rng=self.rng
        )

        # Capture state before saving
        original_population = optimizer._curr_population.copy()
        original_evaluated = optimizer._curr_evaluated_population.copy()
        original_losses = optimizer._curr_losses.copy()
        original_iteration = optimizer._curr_iteration

        checkpoint_dir = str(tmp_path / "checkpoint")
        optimizer.save_state(checkpoint_dir)

        # Load state
        loaded_optimizer = MonteCarloOptimizer.load_state(checkpoint_dir)

        # Verify all state is restored
        np.testing.assert_array_equal(
            loaded_optimizer._curr_population, original_population
        )
        np.testing.assert_array_equal(
            loaded_optimizer._curr_evaluated_population, original_evaluated
        )
        np.testing.assert_array_equal(loaded_optimizer._curr_losses, original_losses)
        assert loaded_optimizer._curr_iteration == original_iteration
        assert loaded_optimizer._curr_rng_state is not None
        assert loaded_optimizer._curr_best_loss == optimizer._curr_best_loss
        np.testing.assert_array_equal(
            loaded_optimizer._curr_best_params, optimizer._curr_best_params
        )

    def test_checkpoint_with_a_failed_evaluation_reloads_and_resumes(self, tmp_path):
        def cost_with_one_failure(population):
            losses = sphere_cost_fn_population(population)
            losses[0] = np.nan
            return losses

        optimizer = self._create_optimizer(keep_best_params=False)
        optimizer.optimize(
            cost_with_one_failure,
            self._create_initial_params(),
            max_iterations=1,
            rng=self.rng,
        )
        checkpoint_dir = tmp_path / "checkpoint"
        optimizer.save_state(checkpoint_dir)

        loaded = MonteCarloOptimizer.load_state(checkpoint_dir)
        np.testing.assert_array_equal(loaded._curr_losses, optimizer._curr_losses)
        result = loaded.optimize(cost_with_one_failure, None, max_iterations=1)
        assert np.isfinite(result.fun)

    def test_save_load_round_trip(self, tmp_path):
        """Test that saving and loading state preserves optimizer functionality."""
        optimizer = self._create_optimizer(keep_best_params=False)
        initial_params = self._create_initial_params()
        verify_save_load_round_trip(
            optimizer,
            initial_params,
            sphere_cost_fn_population,
            self.rng,
            tmp_path,
            MonteCarloOptimizer.load_state,
            self.n_params,
        )

    def test_load_state_raises_file_not_found(self, tmp_path):
        """Test that load_state() raises CheckpointNotFoundError for missing checkpoint."""
        verify_load_state_raises_file_not_found(
            MonteCarloOptimizer.load_state, tmp_path
        )

    def test_save_state_creates_directory_if_needed(self, tmp_path):
        """Test that save_state() creates the checkpoint directory if it doesn't exist."""
        optimizer = self._create_optimizer(keep_best_params=False)
        initial_params = self._create_initial_params()
        verify_save_creates_directory_if_needed(
            optimizer, initial_params, sphere_cost_fn_population, self.rng, tmp_path
        )

    @pytest.mark.parametrize("keep_best_params", [True, False])
    def test_compute_new_parameters_preserves_population_size(
        self, keep_best_params: bool
    ):
        """Low level test to ensure sampling logic preserves shapes and bounds."""
        optimizer = self._create_optimizer(keep_best_params=keep_best_params)
        population = self._create_initial_params()
        losses = sphere_cost_fn_population(population)
        best_indices = np.argsort(losses)[: optimizer.n_best_sets]

        rng = np.random.default_rng(7)
        new_population = optimizer._compute_new_parameters(
            population, curr_iteration=1, best_indices=best_indices, rng=rng
        )

        assert new_population.shape == (optimizer.population_size, self.n_params)
        assert np.all(new_population >= 0.0)
        assert np.all(new_population < 2 * np.pi)

        if keep_best_params:
            np.testing.assert_allclose(
                new_population[: optimizer.n_best_sets], population[best_indices]
            )

    def test_resume_runs_the_iterations_it_is_given(self):
        """``max_iterations`` is what to run *now*, not a total to reach.

        Only a program knows how many iterations it has already spent, and the
        stateless optimizers cannot know it at all, so the caller subtracts and
        every optimizer runs exactly what it is handed —
        :class:`~divi.qprog.VariationalQuantumAlgorithm` owns the total.
        """
        optimizer = self._create_optimizer(keep_best_params=False)
        initial_params = self._create_initial_params()

        eval_counter = {"calls": 0}

        def counting_cost_fn(population: np.ndarray) -> np.ndarray:
            eval_counter["calls"] += 1
            return sphere_cost_fn_population(population)

        optimizer.optimize(
            counting_cost_fn, initial_params, max_iterations=3, rng=self.rng
        )
        assert eval_counter["calls"] == 3

        result = optimizer.optimize(
            counting_cost_fn, initial_params, max_iterations=1, rng=self.rng
        )

        assert eval_counter["calls"] == 4, "the resumed call runs the one it was given"
        assert isinstance(result, OptimizeResult)
        assert result.x.shape == (self.n_params,)

    def test_save_state_without_prior_run_raises(self, tmp_path):
        """Calling save_state before running optimize should fail clearly."""
        optimizer = self._create_optimizer(keep_best_params=False)
        checkpoint_dir = tmp_path / "checkpoint"

        with pytest.raises(
            RuntimeError,
            match=exact_match(
                "Cannot save checkpoint: optimisation has not been run. At least "
                "one iteration must complete before saving optimizer state."
            ),
        ):
            optimizer.save_state(str(checkpoint_dir))

    def test_fresh_run_without_initial_params_raises(self):
        """A fresh run must receive initial_params."""
        optimizer = self._create_optimizer(keep_best_params=False)

        with pytest.raises(
            ValueError,
            match=exact_match(
                "initial_params is required for a fresh MonteCarloOptimizer run."
            ),
        ):
            optimizer.optimize(sphere_cost_fn_population, max_iterations=3)

    def test_fresh_run_zero_iterations_raises_and_leaves_the_optimizer_reusable(
        self,
    ):
        """Fresh run with max_iterations=0 never evaluates the cost, so no state
        exists to return; the failure leaves the optimizer usable."""
        optimizer = self._create_optimizer(keep_best_params=False)
        initial_params = self._create_initial_params()

        with pytest.raises(
            RuntimeError,
            match=exact_match(
                "MonteCarloOptimizer.optimize produced no evaluated population; "
                "nothing to return."
            ),
        ):
            optimizer.optimize(
                sphere_cost_fn_population,
                initial_params,
                max_iterations=0,
                rng=self.rng,
            )

        result = optimizer.optimize(
            sphere_cost_fn_population, initial_params, max_iterations=1, rng=self.rng
        )
        assert result.nit == 1

    def test_copy_preserves_config_and_resets_state(self):
        optimizer = MonteCarloOptimizer(
            population_size=8, n_best_sets=2, keep_best_params=True
        )
        initial_params = (
            self.rng.random((optimizer.population_size, self.n_params)) * 2 * np.pi
        )
        optimizer.optimize(
            sphere_cost_fn_population, initial_params, max_iterations=1, rng=self.rng
        )
        assert optimizer._curr_population is not None

        copied = optimizer.copy()

        assert isinstance(copied, MonteCarloOptimizer)
        assert copied.population_size == optimizer.population_size
        assert copied.keep_best_params == optimizer.keep_best_params
        assert copied.n_best_sets == optimizer.n_best_sets
        assert copied._curr_population is None


class TestPopulationOptimizerCheckpointing:
    """Shared checkpoint/resume behaviours for population-based optimizers."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.n_params = 4
        self.rng = np.random.default_rng(1337)

    def _initial_params(self, optimizer: Optimizer) -> np.ndarray:
        """Create initial parameters that respect the optimizer contract."""
        shape = (optimizer.n_param_sets, self.n_params)
        return self.rng.random(shape) * 2 * np.pi

    def test_resume_with_max_iterations_less_than_completed(
        self, tmp_path, checkpointing_optimizer
    ):
        """A resumed optimizer runs the iterations it is handed on top of those it
        restored, even when that is fewer than it already completed."""
        load_state_func = type(checkpointing_optimizer).load_state

        initial_params = self._initial_params(checkpointing_optimizer)

        checkpointing_optimizer.optimize(
            sphere_cost_fn_population, initial_params, max_iterations=5, rng=self.rng
        )

        checkpoint_dir = tmp_path / "checkpoint"
        checkpointing_optimizer.save_state(str(checkpoint_dir))

        loaded_optimizer = load_state_func(str(checkpoint_dir))
        result = loaded_optimizer.optimize(
            sphere_cost_fn_population, initial_params, max_iterations=3, rng=self.rng
        )

        assert isinstance(result, OptimizeResult)
        assert result.nit == 5 + 3

    def test_resume_with_different_initial_params(
        self, tmp_path, checkpointing_optimizer
    ):
        """Checkpoints ignore newly provided initial parameters when resuming."""
        load_state_func = type(checkpointing_optimizer).load_state
        initial_params = self._initial_params(checkpointing_optimizer)

        checkpointing_optimizer.optimize(
            sphere_cost_fn_population, initial_params, max_iterations=2, rng=self.rng
        )

        checkpoint_dir = str(tmp_path / "checkpoint")
        checkpointing_optimizer.save_state(checkpoint_dir)

        def resume(params):
            return load_state_func(checkpoint_dir).optimize(
                sphere_cost_fn_population,
                params,
                max_iterations=5,
                rng=np.random.default_rng(0),
            )

        with_original = resume(initial_params)
        with_different = resume(self._initial_params(checkpointing_optimizer))

        np.testing.assert_array_equal(with_different.x, with_original.x)
        np.testing.assert_array_equal(with_different.fun, with_original.fun)


def test_default_configuration():
    assert MonteCarloOptimizer().get_config() == {
        "type": "MonteCarloOptimizer",
        "population_size": 10,
        "n_best_sets": 3,
        "keep_best_params": False,
    }


def test_rejects_more_best_sets_than_the_population():
    with pytest.raises(
        ValueError,
        match=exact_match("n_best_sets must be less than or equal to population_size."),
    ):
        MonteCarloOptimizer(population_size=3, n_best_sets=4)


@pytest.mark.parametrize(
    ("curr_iteration", "spread"), [(0, 0.5), (1, 0.25), (3, 0.125)]
)
def test_new_population_samples_around_the_best_with_shrinking_spread_and_wraps(
    curr_iteration, spread
):
    optimizer = MonteCarloOptimizer(population_size=3, n_best_sets=1)

    new_population = optimizer._compute_new_parameters(
        np.zeros((3, 2)),
        curr_iteration=curr_iteration,
        best_indices=np.array([0]),
        rng=np.random.default_rng(0),
    )

    expected = np.random.default_rng(0).normal(0.0, spread, size=(3, 2)) % (2 * np.pi)
    np.testing.assert_allclose(new_population, expected)


def test_non_finite_loss_is_never_selected_as_best():
    result = _small_optimizer().optimize(
        _constant_losses(np.nan, 3.0, 1.0, 2.0),
        _POPULATION,
        max_iterations=1,
        rng=np.random.default_rng(0),
    )

    assert result.fun == 1.0
    np.testing.assert_array_equal(result.x, [4.0, 5.0])


def test_tied_losses_keep_the_first_best_iterate():
    result = _small_optimizer().optimize(
        _constant_losses(1.0, 1.0, 1.0, 1.0),
        _POPULATION,
        max_iterations=3,
        rng=np.random.default_rng(0),
    )

    np.testing.assert_array_equal(result.x, _POPULATION[0])


def test_same_seed_reproduces_the_run():
    def run():
        return _small_optimizer().optimize(
            sphere_cost_fn_population,
            np.ones((4, 2)),
            max_iterations=2,
            rng=np.random.default_rng(5),
        )

    np.testing.assert_array_equal(run().x, run().x)


def test_runs_five_iterations_by_default():
    result = _small_optimizer().optimize(
        sphere_cost_fn_population, np.ones((4, 2)), rng=np.random.default_rng(0)
    )

    assert result.nit == 5


def test_reset_forgets_the_previous_run():
    optimizer = _small_optimizer()
    optimizer.optimize(
        _constant_losses(-5.0, -5.0, -5.0, -5.0),
        _POPULATION,
        max_iterations=1,
        rng=np.random.default_rng(0),
    )

    optimizer.reset()

    with pytest.raises(
        ValueError,
        match=exact_match(
            "initial_params is required for a fresh MonteCarloOptimizer run."
        ),
    ):
        optimizer.optimize(_constant_losses(1.0, 1.0, 1.0, 1.0), max_iterations=1)
    result = optimizer.optimize(
        _constant_losses(1.0, 1.0, 1.0, 1.0),
        _POPULATION,
        max_iterations=1,
        rng=np.random.default_rng(0),
    )
    assert result.fun == 1.0
    assert result.nit == 1


def test_checkpoint_predating_best_ever_tracking_seeds_it_from_the_last_evaluation(
    tmp_path,
):
    optimizer = _small_optimizer()
    optimizer.optimize(
        _constant_losses(-5.0, 1.0, 1.0, 1.0),
        _POPULATION,
        max_iterations=1,
        rng=np.random.default_rng(0),
    )
    optimizer.save_state(tmp_path)
    state_file = tmp_path / OPTIMIZER_STATE_FILE
    state = json.loads(state_file.read_text())
    del state["best_params"], state["best_loss"]
    state_file.write_text(json.dumps(state))

    result = MonteCarloOptimizer.load_state(tmp_path).optimize(
        _constant_losses(1.0, 1.0, 1.0, 1.0),
        None,
        max_iterations=1,
        rng=np.random.default_rng(1),
    )

    assert result.fun == -5.0
    np.testing.assert_array_equal(result.x, _POPULATION[0])


@pytest.mark.parametrize(
    ("state", "message_prefix"),
    [
        (
            {"population_size": 4, "curr_iteration": 0},
            "Checkpoint file is missing required fields: ",
        ),
        (
            {"population_size": "four", "curr_iteration": 0, "rng_state_b64": ""},
            "Failed to validate Monte Carlo optimizer checkpoint state: ",
        ),
    ],
    ids=["missing-field", "invalid-field"],
)
def test_load_state_rejects_a_corrupted_checkpoint(tmp_path, state, message_prefix):
    verify_load_state_rejects_corrupted_state(
        MonteCarloOptimizer.load_state, tmp_path, state, message_prefix
    )
