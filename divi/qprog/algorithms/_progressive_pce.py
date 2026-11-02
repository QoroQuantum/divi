# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Progressive-binarisation PCE (PB-PCE).

Implements Algorithm 1 of Padín-Martínez et al., `arXiv:2602.17479
<https://arxiv.org/abs/2602.17479>`_: PCE is optimised at a binarisation
parameter alpha, alpha is raised until the least-binarised variable just
reaches the threshold, and the optimisation restarts from the previous
parameters, until every variable is binarised.
"""

from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, ClassVar
from warnings import warn

import numpy as np
import numpy.typing as npt

from divi.qprog.algorithms._pce import (
    PCE,
    MultiBasisEncoding,
    PCEEncoding,
    SoftObjective,
)
from divi.qprog.checkpointing import (
    PROGRAM_COMPLETION_FILE,
    CheckpointConfig,
    CheckpointNotFoundError,
    _find_latest_checkpoint_subdir,
    _write_program_completion,
)
from divi.qprog.problems import BinaryOptimizationProblem
from divi.qprog.quantum_program import reject_unclaimed_run_kwargs
from divi.reporting._events import ProgressEvent, TerminalStatus

STEP_SUBDIR_PREFIX = "step_"

#: An Eq. (34) growth factor below this falls back to Eq. (35).
_MIN_ALPHA_GROWTH = 1.01


def _step_of(path: Path) -> int | None:
    """Step encoded in a ``step_NN`` subdirectory name, or None if not one."""
    if not path.is_dir() or not path.name.startswith(STEP_SUBDIR_PREFIX):
        return None
    suffix = path.name[len(STEP_SUBDIR_PREFIX) :]
    return int(suffix) if suffix.isdigit() else None


def _next_alpha(
    alpha: float, binarisation: npt.NDArray[np.float64], threshold: float
) -> float:
    """The binarisation parameter for the next step.

    Picks the unbinarised variable closest to ``threshold`` and scales
    ``alpha`` by Eq. (34) of the PB-PCE paper,
    ``arctanh(M) / arctanh(b)``, so that variable lands on the threshold.
    When that factor is below 1.01, which stalls the run near the threshold,
    Eq. (35), ``arctanh(M) / b``, is used instead.

    Args:
        alpha: The current binarisation parameter.
        binarisation: Each variable's ``|tanh(alpha * z_i)|``.
        threshold: The binarisation threshold ``M``, in ``(0, 1)``.

    Returns:
        The next alpha, or ``alpha`` itself when every variable is binarised.

    Raises:
        ValueError: If every unbinarised variable reads exactly zero, which no
            alpha can binarise.
    """
    unbinarised = binarisation[binarisation < threshold]
    if unbinarised.size == 0:
        return alpha
    closest = float(unbinarised.max())
    if closest == 0.0:
        raise ValueError(
            "Every unbinarised variable has an expectation of exactly 0, which "
            "no binarisation parameter can push to the threshold."
        )
    target = float(np.arctanh(threshold))
    factor = target / float(np.arctanh(closest))
    if factor < _MIN_ALPHA_GROWTH:
        factor = target / closest
    return alpha * factor


class ProgressivePCE(PCE):
    """PCE whose binarisation parameter grows until every variable is binary.

    Each step optimises PCE with :class:`~divi.qprog.algorithms.SoftObjective`
    at the step's ``alpha``, warm-started from the previous step's best
    parameters. After a step, each variable's binarisation
    ``|tanh(alpha * <P_i>)|`` is measured at the best parameters, and
    ``alpha`` is raised so the unbinarised variable closest to ``threshold``
    lands on it: by Eq. (34) of the PB-PCE paper,
    ``arctanh(M) / arctanh(b)``, or by Eq. (35), ``arctanh(M) / b``, when the
    Eq. (34) factor is below 1.01 and would stall the run. The run stops once
    every variable is at or above ``threshold``, after ``max_steps``, or, with
    a warning, when the next ``alpha`` would exceed ``max_alpha``: that
    happens when the relaxation's optimum is fractional, so the optimiser
    shrinks the expectations as fast as ``alpha`` grows.

    The result is the last step. If that step's sign-rounded assignment
    breaks one of the problem's ``constraints``, it is instead the earlier
    step with the lowest-energy feasible assignment. With ``max_restarts``, a
    run with no feasible step restarts from ``initial_alpha`` and fresh
    parameters. Neither rule is in the paper's Algorithm 1.
    :attr:`step_history` records every step.

    Checkpoints are written per step, so the ``subdirectory`` passed to
    ``load_state()`` names the step as well, e.g. ``"step_03/checkpoint_005"``.
    """

    _supports_fixed_param_scans: ClassVar[bool] = False

    def __init__(
        self,
        problem: BinaryOptimizationProblem,
        *,
        encoding: PCEEncoding | None = None,
        initial_alpha: float = 3.0,
        threshold: float = 0.9,
        max_steps: int = 20,
        max_alpha: float = 1e4,
        max_iterations_per_step: int | Callable[[int], int] = 10,
        max_restarts: int = 0,
        **kwargs,
    ):
        """
        Args:
            problem: The binary problem to minimise.
            encoding: How variables map onto qubits. Defaults to
                ``MultiBasisEncoding(order=2)``; Z-parity encodings tend to
                binarise at the first step, where growing ``alpha`` does
                nothing.
            initial_alpha: ``alpha`` of the first step. Defaults to 3.0.
            threshold: Binarisation threshold ``M`` in ``(0, 1)``. Defaults
                to 0.9.
            max_steps: Most steps per attempt, at least 1. Defaults to 20.
            max_alpha: Largest ``alpha`` a step may use, at least
                ``initial_alpha``. Defaults to 1e4.
            max_iterations_per_step: Optimisation iterations per step, an
                integer or a callable ``(step) -> int`` of the step's position
                within its attempt. Defaults to 10.
            max_restarts: Restarts allowed for a run with no feasible step.
                Defaults to 0.
            **kwargs: The remaining :class:`~divi.qprog.algorithms.PCE`
                arguments except ``objective`` and ``max_iterations``, which
                each step sets.

        Raises:
            TypeError: If ``objective`` or ``max_iterations`` is passed.
            ValueError: If ``threshold`` is outside ``(0, 1)``, ``max_steps``
                is below 1, ``max_alpha`` is below ``initial_alpha``, or
                ``max_restarts`` is negative.
        """
        for fixed, replacement in (
            ("objective", "initial_alpha"),
            ("max_iterations", "max_iterations_per_step"),
        ):
            if fixed in kwargs:
                raise TypeError(
                    f"ProgressivePCE sets {fixed} for each step; "
                    f"pass {replacement} instead."
                )
        if not 0.0 < threshold < 1.0:
            raise ValueError(f"threshold must be in (0, 1), got {threshold}")
        if max_steps < 1:
            raise ValueError(f"max_steps must be at least 1, got {max_steps}")
        if max_alpha < initial_alpha:
            raise ValueError(
                f"max_alpha must be at least initial_alpha={initial_alpha}, "
                f"got {max_alpha}"
            )
        if max_restarts < 0:
            raise ValueError(f"max_restarts must be non-negative, got {max_restarts}")
        self._initial_alpha = initial_alpha
        self._alpha = initial_alpha
        self._threshold = threshold
        self._max_steps = max_steps
        self._max_alpha = max_alpha
        self._max_iterations_per_step = max_iterations_per_step
        self._max_restarts = max_restarts
        self._step = 1
        self._attempt = 0
        self._step_history: list[dict[str, Any]] = []

        super().__init__(
            problem,
            encoding=MultiBasisEncoding(order=2) if encoding is None else encoding,
            objective=SoftObjective(alpha=initial_alpha),
            max_iterations=self._step_max_iterations(1),
            **kwargs,
        )

    def _set_alpha(self, alpha: float) -> None:
        self._alpha = alpha
        self.objective = SoftObjective(alpha=alpha)

    def _step_max_iterations(self, step: int) -> int:
        if callable(self._max_iterations_per_step):
            return self._max_iterations_per_step(step)
        return self._max_iterations_per_step

    def _objective_fingerprint(self) -> str:
        """The run's settings, not the current step's alpha, so a checkpoint
        of any step resumes into a freshly built program."""
        return (
            f"ProgressivePCE(initial_alpha={self._initial_alpha!r}, "
            f"threshold={self._threshold!r})"
        )

    @property
    def _expected_total_iterations(self) -> int:
        """Iteration budget of every step of every attempt, for progress display."""
        per_attempt = sum(
            self._step_max_iterations(step) for step in range(1, self._max_steps + 1)
        )
        return per_attempt * (self._max_restarts + 1)

    @property
    def _completed_iterations(self) -> int:
        """Iterations already run across all steps, for progress display."""
        done = sum(entry["n_iterations"] for entry in self._step_history)
        if self._step == len(self._step_history) + 1:
            done += self.current_iteration
        return done

    def _save_subclass_state(self) -> dict[str, Any]:
        """Save PCE state plus the current step and attempt and the steps run so far."""
        state = super()._save_subclass_state()
        state.update(
            {
                "step": self._step,
                "attempt": self._attempt,
                "alpha": self._alpha,
                "step_history": [
                    {**entry, "best_params": np.asarray(entry["best_params"]).tolist()}
                    for entry in self._step_history
                ],
            }
        )
        return state

    def _load_subclass_state(self, state: Mapping[str, Any]) -> None:
        """Restore the current step and attempt, its alpha and the steps run so far."""
        super()._load_subclass_state(state)
        missing = [
            key
            for key in ("step", "attempt", "alpha", "step_history")
            if key not in state
        ]
        if missing:
            raise KeyError(
                f"Corrupted checkpoint: missing required state keys: {missing}"
            )
        self._step = state["step"]
        self._attempt = state["attempt"]
        self._set_alpha(state["alpha"])
        self._step_history = [
            {**entry, "best_params": np.asarray(entry["best_params"], dtype=np.float64)}
            for entry in state["step_history"]
        ]

    @classmethod
    def _resolve_checkpoint_path(
        cls,
        checkpoint_dir: Path | str,
        subdirectory: str | None = None,
    ) -> Path:
        """Resolve the latest step holding a complete checkpoint before its
        iteration."""
        main_dir = Path(checkpoint_dir)
        if subdirectory is None and main_dir.is_dir():
            steps = {
                path: step
                for path in main_dir.iterdir()
                if (step := _step_of(path)) is not None
            }
            for step_dir in sorted(steps, key=steps.__getitem__, reverse=True):
                try:
                    _find_latest_checkpoint_subdir(step_dir)
                except CheckpointNotFoundError:
                    continue
                main_dir = step_dir
                break
        return super()._resolve_checkpoint_path(main_dir, subdirectory)

    @staticmethod
    def _step_checkpoint_config(
        checkpoint_config: CheckpointConfig, step: int
    ) -> CheckpointConfig:
        """Point ``checkpoint_config`` at this step's own subdirectory."""
        if checkpoint_config.checkpoint_dir is None:
            return checkpoint_config
        return replace(
            checkpoint_config,
            checkpoint_dir=Path(checkpoint_config.checkpoint_dir)
            / f"{STEP_SUBDIR_PREFIX}{step:02d}",
        )

    def run(
        self,
        initial_params=None,
        perform_final_computation=True,
        checkpoint_config=None,
        **kwargs,
    ):
        """Run PCE at a growing ``alpha`` until every variable is binarised.

        A program that has run all its steps, or binarised every variable,
        warns and returns without training or finalising; build a new
        program to run it again.

        Args:
            initial_params: Starting parameters of the first step; later
                steps start from the previous step's best parameters.
            perform_final_computation: Whether to read out the solution at
                the last step's best parameters. Defaults to True.
            checkpoint_config: Each step is checkpointed under its own
                ``step_NN`` subdirectory of ``checkpoint_dir``, so
                :meth:`~divi.qprog.VariationalQuantumAlgorithm.load_state`
                can resume at the step where the run stopped.
            **kwargs: ``max_iterations`` overrides every step's iteration
                budget; any other keyword raises ``TypeError``.

        Returns:
            ProgressivePCE: ``self``, for method chaining.
        """
        with self._ensure_progress_session(
            label="Progressive PCE", total=self._expected_total_iterations
        ):
            result = self._run_steps(
                initial_params, perform_final_computation, checkpoint_config, **kwargs
            )
            self._progress_emitter(
                ProgressEvent.finish(
                    self._progress_key,
                    TerminalStatus.SUCCESS,
                    detail="Finished successfully!",
                )
            )
            return result

    def _run_steps(
        self, initial_params, perform_final_computation, checkpoint_config, **kwargs
    ):
        """The body of :meth:`run`, inside its progress session."""
        if checkpoint_config is not None:
            self._checkpoint_config = checkpoint_config
        checkpoint_config = self._checkpoint_config or CheckpointConfig()
        max_iterations = kwargs.pop("max_iterations", None)
        reject_unclaimed_run_kwargs(self, kwargs)

        if self._training_finished:
            last = self._step_history[-1]
            if self._converged:
                ended = f"binarised every variable at step {last['step']}"
            elif self._alpha_capped:
                ended = f"reached max_alpha={self._max_alpha:g} at step {last['step']}"
            else:
                ended = f"already run all its steps (max_steps={self._max_steps})"
            warn(
                f"This ProgressivePCE has {ended}, so run() neither trains nor "
                "finalises. To read out the result step's solution, call "
                "sample_solution().",
                UserWarning,
                stacklevel=3,
            )
            return self

        step = len(self._step_history) + 1
        resumes_mid_step = self._step == step and self.current_iteration > 0
        if checkpoint_config.checkpoint_dir is not None:
            self._reject_used_checkpoint_dir(
                self._step_checkpoint_config(checkpoint_config, step).checkpoint_dir,
                self.current_iteration if resumes_mid_step else 0,
            )
            (Path(checkpoint_config.checkpoint_dir) / PROGRAM_COMPLETION_FILE).unlink(
                missing_ok=True
            )
        warm_start = None
        if self._step_history and self._attempt_ended:
            self._begin_attempt(self._attempt + 1)
        elif self._step_history:
            warm_start = self._step_history[-1]["best_params"].copy()

        while True:
            alpha = self._alpha
            position = self._attempt_steps + 1
            attempt = f", attempt {self._attempt + 1}" if self._max_restarts else ""
            self._progress_emitter(
                ProgressEvent.show(
                    self._progress_key,
                    f"Step {position}/{self._max_steps}{attempt} (alpha={alpha:.3g})",
                )
            )
            step_initial_params = None
            if not resumes_mid_step:
                self._step = step
                self._reset_optimization_state()
                if warm_start is not None:
                    step_initial_params = np.tile(
                        warm_start, (self.optimizer.n_param_sets, 1)
                    )
                elif self._attempt == 0 and initial_params is not None:
                    step_initial_params = initial_params
            resumes_mid_step = False

            self.max_iterations = (
                self._step_max_iterations(position)
                if max_iterations is None
                else max_iterations
            )
            if self.current_iteration < self.max_iterations:
                self._optimize(
                    step_initial_params,
                    self._step_checkpoint_config(checkpoint_config, step),
                )

            z = self._variable_expectations(np.atleast_2d(self._best_params))[0]
            binarisation = np.abs(np.tanh(alpha * z))
            following = _next_alpha(alpha, binarisation, self._threshold)
            assignment = (z > 0).astype(np.float64)
            self._step_history.append(
                {
                    "step": step,
                    "attempt": self._attempt,
                    "alpha": alpha,
                    "next_alpha": following,
                    "best_loss": self._best_loss,
                    "best_params": self._best_params.copy(),
                    "n_iterations": self.current_iteration,
                    "binarised_fraction": float(
                        np.mean(binarisation >= self._threshold)
                    ),
                    "min_binarisation": float(binarisation.min()),
                    "feasible": self._satisfies_constraints(assignment),
                    "energy": float(self.problem.relaxed_energy(assignment)),
                }
            )
            step += 1
            if not self._attempt_ended:
                self._set_alpha(following)
                warm_start = self._best_params.copy()
                continue
            if self._training_finished:
                break
            self._begin_attempt(self._attempt + 1)
            warm_start = None

        if self._alpha_capped:
            last = self._step_history[-1]
            warn(
                f"ProgressivePCE stopped at step {last['step']}: the next alpha, "
                f"{last['next_alpha']:.3g}, exceeds max_alpha={self._max_alpha:g}, "
                f"and {1 - last['binarised_fraction']:.0%} of the variables are "
                "still below the threshold. The relaxation's optimum is likely "
                "fractional; try another encoding, ansatz or initial_alpha.",
                UserWarning,
                stacklevel=3,
            )
        result = self._result_entry
        self._set_alpha(result["alpha"])
        self._best_params = result["best_params"].copy()
        self._best_loss = result["best_loss"]
        self._finalize(perform_final_computation)
        _write_program_completion(self, checkpoint_config.checkpoint_dir)
        return self

    def _begin_attempt(self, attempt: int) -> None:
        """Start ``attempt`` from ``initial_alpha``."""
        self._attempt = attempt
        self._set_alpha(self._initial_alpha)

    def _satisfies_constraints(self, assignment: npt.NDArray[np.float64]) -> bool:
        """Whether a 0/1 assignment of the canonical variables meets every
        constraint of the problem."""
        values = dict(zip(self._polynomial.variable_order, assignment.tolist()))
        return all(c.is_satisfied(values) for c in self.problem.constraints)

    @property
    def _attempt_steps(self) -> int:
        """Steps the current attempt has finished."""
        return sum(entry["attempt"] == self._attempt for entry in self._step_history)

    @property
    def _converged(self) -> bool:
        """Whether the last step binarised every variable."""
        return (
            bool(self._step_history)
            and self._step_history[-1]["min_binarisation"] >= self._threshold
        )

    @property
    def _alpha_capped(self) -> bool:
        """Whether the step after the last one would exceed ``max_alpha``."""
        return (
            bool(self._step_history)
            and self._step_history[-1]["next_alpha"] > self._max_alpha
        )

    @property
    def _attempt_ended(self) -> bool:
        # A step's iteration limit does not end an attempt; only its last
        # step, full binarisation or the alpha cap does.
        return bool(self._step_history) and (
            self._attempt_steps == self._max_steps
            or self._converged
            or self._alpha_capped
        )

    @property
    def _training_finished(self) -> bool:
        """Whether the last attempt ended with a feasible step found or no
        restart left."""
        return self._attempt_ended and (
            any(entry["feasible"] for entry in self._step_history)
            or self._attempt >= self._max_restarts
        )

    @property
    def _result_entry(self) -> dict[str, Any]:
        """The last step, or the lowest-energy feasible step when the last
        step's assignment violates a constraint."""
        last = self._step_history[-1]
        feasible = [entry for entry in self._step_history if entry["feasible"]]
        if last["feasible"] or not feasible:
            return last
        return min(feasible, key=lambda entry: entry["energy"])

    @property
    def alpha(self) -> float:
        """The binarisation parameter of the current, or result, step."""
        return self._alpha

    @property
    def result_step(self) -> int | None:
        """The step the solution is read out from, or ``None`` before a run."""
        return self._result_entry["step"] if self._step_history else None

    @property
    def step_history(self) -> list[dict[str, Any]]:
        """Per-step results, across every attempt.

        Each entry has the keys ``step``, ``attempt`` (0 for the first run,
        counting restarts), ``alpha``, ``next_alpha`` (the alpha the following
        step uses, or would use), ``best_loss``, ``best_params``,
        ``n_iterations``, ``binarised_fraction`` (share of variables at or
        above ``threshold``), ``min_binarisation`` (smallest
        ``|tanh(alpha * <P_i>)|``), ``feasible`` (whether the sign-rounded
        assignment meets every constraint) and ``energy`` (its penalised
        energy).
        """
        return self._step_history.copy()
