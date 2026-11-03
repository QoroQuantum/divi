# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import shutil

import numpy as np
import pytest

from divi.qprog.algorithms import (
    BrickworkAnsatz,
    MultiBasisEncoding,
    ProgressivePCE,
    SoftObjective,
)
from divi.qprog.algorithms._progressive_pce import _next_alpha
from divi.qprog.checkpointing import CheckpointConfig, list_checkpoints
from divi.qprog.checkpointing.names import PROGRAM_COMPLETION_FILE
from divi.qprog.optimizers import MonteCarloOptimizer, ScipyMethod, ScipyOptimizer
from divi.qprog.problems import BinaryOptimizationProblem, LinearConstraint
from tests._helpers import exact_match
from tests.qprog._program_contracts import verify_load_state_rejects_missing_state_key
from tests.qprog.problems._helpers import PCE_QUBO_MATRIX

#: Three variables, which binarise to 0.995 at the first step.
_QUBO_3 = np.array([[1.0, -2.0, 0.5], [0.0, 3.0, 1.5], [0.0, 0.0, -1.0]])
#: Nine variables whose relaxation binarises within a few steps.
_BINARISING_QUBO = np.triu(np.random.default_rng(3).uniform(-1, 1, (9, 9)))


def _progressive(backend, optimizer=None, problem=_QUBO_3, **kwargs):
    if not isinstance(problem, BinaryOptimizationProblem):
        problem = BinaryOptimizationProblem(problem)
    return ProgressivePCE(
        problem,
        optimizer=optimizer or ScipyOptimizer(method=ScipyMethod.L_BFGS_B),
        backend=backend,
        **{"seed": 1997, **kwargs},
    )


def _budget_qubo(n, c, seed):
    """A random upper-triangular QUBO on n variables that must set exactly c."""
    qubo = np.triu(np.random.default_rng(seed).uniform(-1, 1, (n, n)))
    return BinaryOptimizationProblem(
        qubo,
        constraints=[LinearConstraint([1] * n, "==", c)],
        penalty_weight=float(np.abs(qubo).sum()),
    )


def _paper_progressive(backend, problem, seed, **kwargs):
    """The PB-PCE settings: SLSQP, one brickwork layer, uncapped alpha."""
    return _progressive(
        backend,
        optimizer=ScipyOptimizer(method=ScipyMethod.SLSQP),
        problem=problem,
        seed=seed,
        ansatz=BrickworkAnsatz(),
        max_steps=200,
        max_alpha=1e30,
        max_iterations_per_step=1000,
        **kwargs,
    )


class TestNextAlpha:
    def test_scales_the_closest_unbinarised_variable_onto_the_threshold(self):
        """Eq. (34): 0.95 is already binarised; 0.7 is closer to 0.9 than 0.5."""
        alpha = _next_alpha(3.0, np.array([0.95, 0.5, 0.7]), threshold=0.9)
        assert alpha == pytest.approx(3.0 * np.arctanh(0.9) / np.arctanh(0.7))

    def test_falls_back_to_eq_35_near_the_threshold(self):
        """Eq. (34) would grow alpha by under 1.01 at 0.899."""
        alpha = _next_alpha(3.0, np.array([0.899]), threshold=0.9)
        assert alpha == pytest.approx(3.0 * np.arctanh(0.9) / 0.899)

    def test_keeps_alpha_once_every_variable_is_binarised(self):
        assert _next_alpha(3.0, np.array([0.95, 0.91]), threshold=0.9) == 3.0

    def test_rejects_variables_that_no_alpha_can_binarise(self):
        message = (
            "Every unbinarised variable has an expectation of exactly 0, which "
            "no binarisation parameter can push to the threshold."
        )
        with pytest.raises(ValueError, match=exact_match(message)):
            _next_alpha(3.0, np.array([0.0, 0.95]), threshold=0.9)


class TestProgressivePCE:
    @pytest.mark.parametrize(
        "kwarg, replacement",
        [("objective", "initial_alpha"), ("max_iterations", "max_iterations_per_step")],
    )
    def test_rejects_settings_each_step_owns(
        self, make_maestro_simulator, kwarg, replacement
    ):
        message = (
            f"ProgressivePCE sets {kwarg} for each step; pass {replacement} instead."
        )
        with pytest.raises(TypeError, match=exact_match(message)):
            _progressive(make_maestro_simulator(), **{kwarg: None})

    @pytest.mark.parametrize(
        "kwargs, message",
        [
            ({"threshold": 1.0}, "threshold must be in (0, 1), got 1.0"),
            ({"threshold": 0.0}, "threshold must be in (0, 1), got 0.0"),
            ({"max_steps": 0}, "max_steps must be at least 1, got 0"),
            (
                {"max_alpha": 2.0},
                "max_alpha must be at least initial_alpha=3.0, got 2.0",
            ),
            ({"max_restarts": -1}, "max_restarts must be non-negative, got -1"),
        ],
        ids=[
            "threshold-one",
            "threshold-zero",
            "no-steps",
            "alpha-cap-below-start",
            "negative-restarts",
        ],
    )
    def test_rejects_out_of_range_settings(
        self, make_maestro_simulator, kwargs, message
    ):
        with pytest.raises(ValueError, match=exact_match(message)):
            _progressive(make_maestro_simulator(), **kwargs)

    def test_defaults_to_the_paper_s_encoding_and_alpha(self, make_maestro_simulator):
        program = _progressive(make_maestro_simulator())
        assert program.encoding == MultiBasisEncoding(order=2)
        assert program.objective == SoftObjective(alpha=3.0)

    def test_raises_alpha_until_every_variable_is_binarised(
        self, make_maestro_simulator
    ):
        program = _progressive(
            make_maestro_simulator(),
            problem=_BINARISING_QUBO,
            n_layers=2,
            max_iterations_per_step=20,
        )

        program.run(perform_final_computation=False)

        history = program.step_history
        alphas = [entry["alpha"] for entry in history]
        assert len(history) < 20
        assert alphas == sorted(alphas) and len(set(alphas)) == len(alphas)
        assert history[-1]["min_binarisation"] >= 0.9
        assert history[-1]["binarised_fraction"] == 1.0
        assert all(entry["min_binarisation"] < 0.9 for entry in history[:-1])

    def test_stops_at_max_steps_and_keeps_the_last_step(self, make_maestro_simulator):
        program = _progressive(
            make_maestro_simulator(),
            problem=_BINARISING_QUBO,
            threshold=0.9999,
            max_steps=2,
        )

        program.run(perform_final_computation=False)

        history = program.step_history
        assert [entry["step"] for entry in history] == [1, 2]
        assert history[1]["alpha"] > history[0]["alpha"]
        assert program.alpha == history[-1]["alpha"]
        assert program.objective == SoftObjective(alpha=history[-1]["alpha"])
        assert program.best_loss == history[-1]["best_loss"]
        with pytest.warns(UserWarning, match="already run all its steps"):
            program.run()

    def test_each_step_starts_from_the_previous_best_parameters(
        self, make_maestro_simulator, mocker
    ):
        program = _progressive(make_maestro_simulator(), threshold=0.9999, max_steps=2)
        seed = np.full(program.n_params, 0.3)
        optimize = mocker.spy(program, "_optimize")

        program.run(initial_params=seed, perform_final_computation=False)

        first, second = (call.args[0] for call in optimize.call_args_list)
        np.testing.assert_array_equal(first, seed)
        np.testing.assert_array_equal(
            second,
            np.tile(
                program.step_history[0]["best_params"],
                (program.optimizer.n_param_sets, 1),
            ),
        )

    def test_gives_each_step_its_own_iteration_budget(self, make_maestro_simulator):
        program = _progressive(
            make_maestro_simulator(),
            optimizer=MonteCarloOptimizer(population_size=3, n_best_sets=1),
            threshold=0.9999,
            max_steps=3,
            max_iterations_per_step=lambda step: step,
        )

        program.run(perform_final_computation=False)

        assert [entry["n_iterations"] for entry in program.step_history] == [1, 2, 3]

    def test_reads_out_the_last_step_s_solution(self, make_maestro_simulator):
        program = _progressive(
            make_maestro_simulator(),
            problem=_BINARISING_QUBO,
            n_layers=2,
            max_iterations_per_step=20,
        )

        program.run()

        [entry] = program.get_top_solutions(sort_by="energy")
        assert entry.prob == 1.0
        assert len(entry.bitstring) == 9

    def test_stops_with_a_warning_when_alpha_would_pass_max_alpha(
        self, make_maestro_simulator
    ):
        program = _progressive(
            make_maestro_simulator(),
            threshold=0.9999,
            max_alpha=4.0,
            max_iterations_per_step=20,
        )

        with pytest.warns(UserWarning, match="exceeds max_alpha=4") as records:
            program.run(perform_final_computation=False)

        assert len(records) == 1
        history = program.step_history
        assert all(entry["alpha"] <= 4.0 for entry in history)
        assert history[-1]["next_alpha"] > 4.0
        assert history[-1]["min_binarisation"] < 0.9999
        with pytest.warns(UserWarning, match="reached max_alpha=4"):
            program.run()

    def test_every_step_of_an_unconstrained_problem_is_feasible(
        self, make_maestro_simulator
    ):
        program = _progressive(make_maestro_simulator(), threshold=0.9999, max_steps=2)

        program.run(perform_final_computation=False)

        assert all(entry["feasible"] for entry in program.step_history)
        assert program.result_step == program.step_history[-1]["step"]

    def test_reads_out_the_best_feasible_step_when_the_last_breaks_a_constraint(
        self, make_maestro_simulator
    ):
        """Two variables park at complementary fractional values, satisfying
        sum(x) = 2, until the final alpha jump snaps both to 0."""
        program = _paper_progressive(make_maestro_simulator(), _budget_qubo(9, 2, 1), 1)

        program.run()

        history = program.step_history
        assert not history[-1]["feasible"]
        result = next(e for e in history if e["step"] == program.result_step)
        assert result["feasible"]
        assert result["energy"] == min(e["energy"] for e in history if e["feasible"])
        assert program.alpha == result["alpha"]
        [entry] = program.get_top_solutions()
        assert entry.bitstring.count("1") == 2

    def test_returns_the_last_step_when_no_step_meets_the_constraints(
        self, make_maestro_simulator
    ):
        program = _paper_progressive(make_maestro_simulator(), _budget_qubo(9, 4, 2), 2)

        program.run(perform_final_computation=False)

        history = program.step_history
        assert not any(entry["feasible"] for entry in history)
        assert {entry["attempt"] for entry in history} == {0}
        assert program.result_step == history[-1]["step"]

    def test_restarts_until_a_step_meets_the_constraints(self, make_maestro_simulator):
        program = _paper_progressive(
            make_maestro_simulator(), _budget_qubo(9, 4, 2), 2, max_restarts=3
        )

        program.run()

        history = program.step_history
        first = [entry for entry in history if entry["attempt"] == 0]
        assert first and not any(entry["feasible"] for entry in first)
        later = [entry for entry in history if entry["attempt"] > 0]
        assert later[0]["alpha"] == 3.0
        assert [entry["step"] for entry in history] == list(range(1, len(history) + 1))
        assert any(entry["feasible"] for entry in later)
        [entry] = program.get_top_solutions()
        assert entry.bitstring.count("1") == 4


class TestProgressivePCECheckpointing:
    MAX_STEPS = 3
    ITERS_PER_STEP = 2

    def _build(self, backend, **kwargs):
        return dict(
            problem=BinaryOptimizationProblem(PCE_QUBO_MATRIX),
            threshold=0.9999,
            max_steps=self.MAX_STEPS,
            max_iterations_per_step=self.ITERS_PER_STEP,
            **kwargs,
        )

    def _run_unsampled(self, backend, checkpoint_dir):
        program = ProgressivePCE(
            **self._build(backend),
            backend=backend,
            optimizer=MonteCarloOptimizer(population_size=4, n_best_sets=2),
            seed=1997,
        )
        program.run(
            checkpoint_config=CheckpointConfig(checkpoint_dir=checkpoint_dir),
            perform_final_computation=False,
        )
        return program

    def _run_without_completion(self, backend, checkpoint_dir):
        """A finished run's checkpoints without its completion file: what a
        crash between the last step and the completion write leaves."""
        program = self._run_unsampled(backend, checkpoint_dir)
        (checkpoint_dir / PROGRAM_COMPLETION_FILE).unlink()
        return program

    def _load(self, backend, checkpoint_dir, **kwargs):
        return ProgressivePCE.load_state(
            checkpoint_dir, backend=backend, **self._build(backend), **kwargs
        )

    def test_each_step_keeps_its_own_checkpoints(
        self, make_maestro_simulator, tmp_path
    ):
        self._run_without_completion(make_maestro_simulator(), tmp_path)

        for step in range(1, self.MAX_STEPS + 1):
            iterations = [
                info.iteration
                for info in list_checkpoints(tmp_path / f"step_{step:02d}")
            ]
            assert iterations == list(range(1, self.ITERS_PER_STEP + 1))

    def test_load_resolves_the_latest_step_and_its_alpha(
        self, make_maestro_simulator, tmp_path
    ):
        """The checkpoint of the last step is written before that step joins the
        history, and carries that step's alpha, not the initial one."""
        program = self._run_without_completion(make_maestro_simulator(), tmp_path)

        loaded = self._load(make_maestro_simulator(), tmp_path)

        assert [entry["step"] for entry in loaded.step_history] == [1, 2]
        assert loaded.alpha == program.step_history[-1]["alpha"]
        assert loaded.alpha != 3.0

    def test_load_a_given_iteration_names_its_step(
        self, make_maestro_simulator, tmp_path
    ):
        program = self._run_without_completion(make_maestro_simulator(), tmp_path)

        loaded = self._load(
            make_maestro_simulator(), tmp_path, subdirectory="step_02/checkpoint_001"
        )

        assert loaded.current_iteration == 1
        assert loaded.alpha == program.step_history[1]["alpha"]

    def test_resume_finishes_a_partially_optimised_step(
        self, make_maestro_simulator, tmp_path
    ):
        """A step interrupted with budget left is continued, not restarted."""
        self._run_without_completion(make_maestro_simulator(), tmp_path)
        last = tmp_path / f"step_{self.MAX_STEPS:02d}"
        for info in list_checkpoints(last):
            if info.iteration > 1:
                shutil.rmtree(info.path)

        loaded = self._load(make_maestro_simulator(), tmp_path)
        assert loaded.current_iteration == 1
        assert loaded._completed_iterations == (
            (self.MAX_STEPS - 1) * self.ITERS_PER_STEP + 1
        )

        loaded.run(perform_final_computation=False)

        last_entry = loaded.step_history[-1]
        assert last_entry["step"] == self.MAX_STEPS
        assert last_entry["n_iterations"] == self.ITERS_PER_STEP

    def test_resume_continues_after_the_completed_steps(
        self, make_maestro_simulator, tmp_path
    ):
        self._run_without_completion(make_maestro_simulator(), tmp_path)
        shutil.rmtree(tmp_path / f"step_{self.MAX_STEPS:02d}")

        loaded = self._load(make_maestro_simulator(), tmp_path)
        loaded.run(perform_final_computation=False)

        assert [entry["step"] for entry in loaded.step_history] == list(
            range(1, self.MAX_STEPS + 1)
        )

    def test_restored_finished_run_does_nothing(
        self, make_maestro_simulator, tmp_path, mocker
    ):
        self._run_unsampled(make_maestro_simulator(), tmp_path)
        loaded = self._load(make_maestro_simulator(), tmp_path)
        sample = mocker.spy(loaded, "sample_solution")

        with pytest.warns(UserWarning, match="call sample_solution"):
            loaded.run()

        sample.assert_not_called()

    def test_load_rejects_a_state_missing_step_keys(
        self, make_maestro_simulator, tmp_path
    ):
        self._run_without_completion(make_maestro_simulator(), tmp_path)

        verify_load_state_rejects_missing_state_key(
            tmp_path, "step", lambda: self._load(make_maestro_simulator(), tmp_path)
        )

    def test_load_rejects_another_initial_alpha(self, make_maestro_simulator, tmp_path):
        self._run_unsampled(make_maestro_simulator(), tmp_path)

        with pytest.raises(ValueError, match="different cost Hamiltonian"):
            self._load(make_maestro_simulator(), tmp_path, initial_alpha=5.0)

    def test_fresh_run_rejects_a_used_directory(self, make_maestro_simulator, tmp_path):
        self._run_unsampled(make_maestro_simulator(), tmp_path)

        with pytest.raises(ValueError, match="already holds checkpoints"):
            self._run_unsampled(make_maestro_simulator(), tmp_path)
        assert (tmp_path / PROGRAM_COMPLETION_FILE).is_file()
