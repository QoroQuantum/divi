# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import pytest
from scipy.optimize import OptimizeResult

from divi.qprog.optimizers import (
    PymooMethod,
    PymooOptimizer,
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

#: Four two-parameter sets with distinct rows.
_POPULATION = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])


def _run(optimizer, cost_fn=sphere_cost_fn_population, initial=_POPULATION, **kwargs):
    """Run ``optimizer`` seeded with ``default_rng(0)``; return the result and the
    population each callback received."""
    populations = []
    kwargs.setdefault("rng", np.random.default_rng(0))
    result = optimizer.optimize(
        cost_fn,
        initial,
        callback_fn=lambda r: populations.append(r.x.copy()),
        **kwargs,
    )
    return result, populations


@pytest.fixture(
    params=[(PymooMethod.CMAES, 10), (PymooMethod.DE, 5)],
    ids=["cmaes", "de"],
)
def pymoo_contract_optimizer(request):
    method, population_size = request.param
    return PymooOptimizer(method=method, population_size=population_size)


def test_optimizer_contract(pymoo_contract_optimizer, optimizer_contract):
    optimizer_contract(pymoo_contract_optimizer)


class _CustomPymooOptimizer(PymooOptimizer):
    pass


def test_copy_preserves_subclass_type():
    optimizer = _CustomPymooOptimizer(method=PymooMethod.CMAES, population_size=5)
    assert type(optimizer.copy()) is _CustomPymooOptimizer


class TestPymooOptimizer:
    """Specific tests for PymooOptimizer features."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.n_params = 4
        self.rng = np.random.default_rng(42)

    def test_reset_clears_pymoo_state(self):
        """Test that reset() clears all internal state variables for PymooOptimizer."""
        optimizer = PymooOptimizer(method=PymooMethod.CMAES, population_size=10)
        initial_params = self.rng.random((10, self.n_params)) * 2 * np.pi

        # Run optimization to set state
        optimizer.optimize(
            sphere_cost_fn_population, initial_params, max_iterations=3, rng=self.rng
        )

        # Verify state is set
        assert optimizer._curr_algorithm_obj is not None

        # Reset and verify state is cleared
        optimizer.reset()
        assert optimizer._curr_algorithm_obj is None

    def test_save_state_creates_checkpoint_file(self, tmp_path):
        """Test that save_state() creates the expected checkpoint file."""
        optimizer = PymooOptimizer(method=PymooMethod.CMAES, population_size=10)
        initial_params = self.rng.random((10, self.n_params)) * 2 * np.pi
        verify_save_creates_checkpoint_file(
            optimizer,
            initial_params,
            sphere_cost_fn_population,
            self.rng,
            tmp_path,
        )

    def test_save_state_without_prior_run_raises(self, tmp_path):
        """Saving without having run optimize should raise a clear error."""
        optimizer = PymooOptimizer(method=PymooMethod.CMAES, population_size=5)
        with pytest.raises(
            RuntimeError,
            match=exact_match(
                "Cannot save checkpoint: optimisation has not been run. At least "
                "one iteration must complete before saving optimizer state."
            ),
        ):
            optimizer.save_state(str(tmp_path / "checkpoint"))

    def test_save_state_preserves_configuration(self, tmp_path):
        """Test that save_state() saves optimizer configuration correctly."""
        optimizer = PymooOptimizer(method=PymooMethod.DE, population_size=15)
        initial_params = self.rng.random((15, self.n_params)) * 2 * np.pi

        # Run optimization to set state
        optimizer.optimize(
            sphere_cost_fn_population, initial_params, max_iterations=2, rng=self.rng
        )

        checkpoint_dir = str(tmp_path / "checkpoint")
        optimizer.save_state(checkpoint_dir)

        # Load and verify configuration
        loaded_optimizer = PymooOptimizer.load_state(checkpoint_dir)
        assert loaded_optimizer.method == optimizer.method
        assert loaded_optimizer.population_size == optimizer.population_size

    def test_save_load_round_trip(self, tmp_path):
        """Test that saving and loading state preserves optimizer functionality."""
        optimizer = PymooOptimizer(method=PymooMethod.CMAES, population_size=10)
        initial_params = self.rng.random((10, self.n_params)) * 2 * np.pi
        verify_save_load_round_trip(
            optimizer,
            initial_params,
            sphere_cost_fn_population,
            self.rng,
            tmp_path,
            PymooOptimizer.load_state,
            self.n_params,
        )

    def test_load_state_raises_file_not_found(self, tmp_path):
        """Test that load_state() raises CheckpointNotFoundError for missing checkpoint."""
        verify_load_state_raises_file_not_found(PymooOptimizer.load_state, tmp_path)

    def test_save_state_creates_directory_if_needed(self, tmp_path):
        """Test that save_state() creates the checkpoint directory if it doesn't exist."""
        optimizer = PymooOptimizer(method=PymooMethod.CMAES, population_size=10)
        initial_params = self.rng.random((10, self.n_params)) * 2 * np.pi
        verify_save_creates_directory_if_needed(
            optimizer, initial_params, sphere_cost_fn_population, self.rng, tmp_path
        )

    @pytest.mark.parametrize(
        ("method", "population_size"),
        [(PymooMethod.CMAES, 6), (PymooMethod.DE, 10)],
        ids=["cmaes", "de"],
    )
    def test_resume_runs_the_iterations_it_is_given(
        self, tmp_path, method, population_size
    ):
        """``max_iterations`` is what to run *now*: the caller has already
        subtracted whatever a previous run spent, so a resumed optimizer adds that
        many on top of its restored generation count."""
        optimizer = PymooOptimizer(method=method, population_size=population_size)
        initial_params = (
            self.rng.random((optimizer.n_param_sets, self.n_params)) * 2 * np.pi
        )

        optimizer.optimize(
            sphere_cost_fn_population, initial_params, max_iterations=2, rng=self.rng
        )
        checkpoint_dir = str(tmp_path / "checkpoint")
        optimizer.save_state(checkpoint_dir)

        loaded_optimizer = PymooOptimizer.load_state(checkpoint_dir)
        result = loaded_optimizer.optimize(
            sphere_cost_fn_population, max_iterations=4, rng=self.rng
        )

        assert isinstance(result, OptimizeResult)
        assert result.x.shape == (self.n_params,)
        assert np.isfinite(result.fun)
        assert result.nit == 6, "2 restored generations plus the 4 it was given"

    def test_n_param_sets_respects_popsize_kwarg(self):
        """n_param_sets should honor CMAES popsize overrides."""
        optimizer_with_kwarg = PymooOptimizer(
            method=PymooMethod.CMAES, population_size=6, popsize=4
        )
        assert optimizer_with_kwarg.n_param_sets == 4

        optimizer_default_cmaes = PymooOptimizer(
            method=PymooMethod.CMAES, population_size=8
        )
        assert optimizer_default_cmaes.n_param_sets == 8

        optimizer_de = PymooOptimizer(method=PymooMethod.DE, population_size=11)
        assert optimizer_de.n_param_sets == 11

    def test_de_population_preserved_in_checkpoint(self, tmp_path):
        """DE's population survives a save/load round trip unchanged."""
        optimizer = PymooOptimizer(method=PymooMethod.DE, population_size=8)
        initial_params = self.rng.random((8, self.n_params)) * 2 * np.pi

        optimizer.optimize(
            sphere_cost_fn_population, initial_params, max_iterations=3, rng=self.rng
        )
        saved_population = optimizer._curr_algorithm_obj.pop.get("X")

        checkpoint_dir = str(tmp_path / "checkpoint")
        optimizer.save_state(checkpoint_dir)

        loaded_optimizer = PymooOptimizer.load_state(checkpoint_dir)
        np.testing.assert_array_equal(
            loaded_optimizer._curr_algorithm_obj.pop.get("X"), saved_population
        )

    def test_cmaes_custom_kwargs_preserved_through_save_load(self, tmp_path):
        """Test that custom CMAES kwargs survive save/load cycle."""
        # Create optimizer with custom kwargs
        optimizer = PymooOptimizer(
            method=PymooMethod.CMAES, population_size=10, popsize=8, sigma0=0.3
        )
        initial_params = self.rng.random((8, self.n_params)) * 2 * np.pi

        # Run optimization
        optimizer.optimize(
            sphere_cost_fn_population, initial_params, max_iterations=2, rng=self.rng
        )

        checkpoint_dir = str(tmp_path / "checkpoint")
        optimizer.save_state(checkpoint_dir)

        # Load and verify kwargs are preserved
        loaded_optimizer = PymooOptimizer.load_state(checkpoint_dir)
        assert loaded_optimizer.population_size == 10
        assert loaded_optimizer.algorithm_kwargs.get("popsize") == 8
        assert loaded_optimizer.algorithm_kwargs.get("sigma0") == 0.3

        # Verify it can continue optimization
        result = loaded_optimizer.optimize(
            sphere_cost_fn_population, max_iterations=4, rng=self.rng
        )
        assert isinstance(result, OptimizeResult)
        assert result.nit == 6  # 2 restored iterations plus the 4 it was given

    def test_optimize_with_zero_max_iterations(self):
        """Test that optimize with max_iterations=0 returns immediately."""
        optimizer = PymooOptimizer(method=PymooMethod.CMAES, population_size=5)
        initial_params = self.rng.random((5, self.n_params)) * 2 * np.pi

        # First run some iterations
        optimizer.optimize(
            sphere_cost_fn_population, initial_params, max_iterations=3, rng=self.rng
        )

        # Now try to resume with max_iterations=0
        result = optimizer.optimize(
            sphere_cost_fn_population, max_iterations=0, rng=self.rng
        )

        # Should return current state without running more iterations
        assert isinstance(result, OptimizeResult)
        assert result.nit == 3  # Should still be at iteration 3

    def test_resume_adds_to_what_it_already_ran(self):
        """The optimizer does not cap itself against a total — it runs what it is
        handed, and the program subtracts what it already spent."""
        optimizer = PymooOptimizer(method=PymooMethod.CMAES, population_size=5)
        initial_params = self.rng.random((5, self.n_params)) * 2 * np.pi

        # Run 5 iterations
        result1 = optimizer.optimize(
            sphere_cost_fn_population, initial_params, max_iterations=5, rng=self.rng
        )
        assert result1.nit == 5

        result2 = optimizer.optimize(
            sphere_cost_fn_population, max_iterations=5, rng=self.rng
        )
        assert result2.nit == 10  # the 5 it already ran plus the 5 it was given

    def test_fresh_run_without_initial_params_raises(self):
        """A fresh run must receive initial_params."""
        optimizer = PymooOptimizer(method=PymooMethod.CMAES, population_size=5)

        with pytest.raises(
            ValueError,
            match=exact_match(
                "initial_params is required for a fresh PymooOptimizer run."
            ),
        ):
            optimizer.optimize(sphere_cost_fn_population, max_iterations=3)

    def test_copy_preserves_kwargs(self):
        optimizer = PymooOptimizer(
            method=PymooMethod.CMAES, population_size=5, popsize=4, sigma=0.5
        )
        initial_params = (
            self.rng.random((optimizer.n_param_sets, self.n_params)) * 2 * np.pi
        )
        optimizer.optimize(
            sphere_cost_fn_population, initial_params, max_iterations=1, rng=self.rng
        )
        assert optimizer._curr_algorithm_obj is not None

        copied = optimizer.copy()

        assert isinstance(copied, PymooOptimizer)
        assert copied.method == optimizer.method
        assert copied.population_size == optimizer.population_size
        assert copied.algorithm_kwargs["popsize"] == 4
        assert copied.algorithm_kwargs["sigma"] == 0.5
        assert copied._curr_algorithm_obj is None


def test_default_population_iterations_and_config():
    optimizer = PymooOptimizer(method=PymooMethod.DE)
    assert optimizer.population_size == 50

    result, _ = _run(PymooOptimizer(method=PymooMethod.DE, population_size=4))
    assert result.nit == 5

    assert PymooOptimizer(
        method=PymooMethod.CMAES, population_size=4, sigma0=0.3
    ).get_config() == {
        "type": "PymooOptimizer",
        "method": "CMAES",
        "population_size": 4,
        "sigma0": 0.3,
    }


def test_cmaes_centres_on_the_first_parameter_set():
    _, populations = _run(
        PymooOptimizer(method=PymooMethod.CMAES, population_size=4, sigma0=1e-9),
        max_iterations=1,
    )

    np.testing.assert_allclose(
        populations[0], np.tile(_POPULATION[0], (4, 1)), atol=1e-6
    )


@pytest.mark.parametrize(
    ("kwargs", "sigma0"),
    [({}, 0.1), ({"sigma": 0.3}, 0.3), ({"sigma0": 0.2, "sigma": 0.3}, 0.2)],
    ids=["default", "sigma-alias", "sigma0-wins"],
)
def test_cmaes_initial_step_size(kwargs, sigma0):
    optimizer = PymooOptimizer(method=PymooMethod.CMAES, population_size=4, **kwargs)
    _run(optimizer, max_iterations=1)

    assert optimizer._curr_algorithm_obj.sigma0 == sigma0


@pytest.mark.parametrize("method", list(PymooMethod), ids=lambda m: m.value)
def test_same_seed_reproduces_the_run(method):
    def run():
        result, _ = _run(
            PymooOptimizer(method=method, population_size=4),
            max_iterations=3,
            rng=np.random.default_rng(7),
        )
        return result

    np.testing.assert_array_equal(run().x, run().x)


@pytest.mark.parametrize("method", list(PymooMethod), ids=lambda m: m.value)
def test_runs_silently(method, capsys):
    _run(PymooOptimizer(method=method, population_size=4), max_iterations=2)

    assert capsys.readouterr().out == ""


def test_de_forwards_algorithm_kwargs():
    optimizer = PymooOptimizer(method=PymooMethod.DE, population_size=4, CR=0.9)
    _run(optimizer, max_iterations=1)

    assert optimizer._curr_algorithm_obj.mating.CR.value == 0.9


def test_de_offspring_stay_within_one_period():
    _, populations = _run(
        PymooOptimizer(method=PymooMethod.DE, population_size=6),
        cost_fn=lambda X: -np.sum(X, axis=1),
        initial=np.full((6, 2), 6.2) + np.arange(6)[:, None] * 0.01,
        max_iterations=6,
    )

    evaluated = np.vstack(populations)
    assert np.all(evaluated >= 0.0)
    assert np.all(evaluated <= 2 * np.pi)


def test_cmaes_survives_a_non_finite_loss():
    def first_row_nan(X):
        losses = np.sum(X**2, axis=1)
        losses[0] = np.nan
        return losses

    result, _ = _run(
        PymooOptimizer(method=PymooMethod.CMAES, population_size=4),
        cost_fn=first_row_nan,
        max_iterations=2,
    )

    assert np.isfinite(result.fun)


def test_cmaes_popsize_override_sets_the_evaluated_population():
    optimizer = PymooOptimizer(method=PymooMethod.CMAES, population_size=6, popsize=4)
    _, populations = _run(
        optimizer, initial=np.ones((optimizer.n_param_sets, 3)), max_iterations=2
    )

    assert [p.shape for p in populations] == [(4, 3), (4, 3)]


@pytest.mark.parametrize("method", list(PymooMethod), ids=lambda m: m.value)
def test_fresh_run_with_zero_iterations_raises(method):
    with pytest.raises(
        ValueError,
        match=exact_match(
            "A fresh PymooOptimizer run needs max_iterations >= 1, got 0."
        ),
    ):
        _run(PymooOptimizer(method=method, population_size=4), max_iterations=0)


@pytest.mark.parametrize(
    ("state", "message_prefix"),
    [
        ({"method_value": "DE"}, "Checkpoint file is missing required fields: "),
        (
            {
                "method_value": "DE",
                "population_size": "four",
                "algorithm_kwargs": {},
                "algorithm_obj_b64": "",
            },
            "Failed to validate Pymoo optimizer checkpoint state: ",
        ),
    ],
    ids=["missing-field", "invalid-field"],
)
def test_load_state_rejects_a_corrupted_checkpoint(tmp_path, state, message_prefix):
    verify_load_state_rejects_corrupted_state(
        PymooOptimizer.load_state, tmp_path, state, message_prefix
    )
