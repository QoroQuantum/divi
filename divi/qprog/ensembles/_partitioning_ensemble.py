# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import copy
import warnings
from collections.abc import Callable
from functools import partial

from divi.backends import CircuitRunner
from divi.qprog.mixins import SolutionSamplingMixin
from divi.qprog.optimizers import Optimizer
from divi.qprog.problems import QAOAProblem

from ._aggregation import AggregationStrategy, BeamSearchStrategy
from ._base import ProgramEnsemble, ReportingLevel, _child_seed


class PartitioningProgramEnsemble(ProgramEnsemble):
    """Generic orchestrator for partition-solve-aggregate quantum optimisation.

    Delegates all domain-specific logic to the :class:`~divi.qprog.problems.QAOAProblem` instance:
    decomposition, solution extension, evaluation, and result post-processing.
    The ensemble handles program creation, execution, and result aggregation.

    Args:
        problem: A :class:`~divi.qprog.problems.QAOAProblem` configured for decomposition
            (e.g. ``MaxCutProblem(graph, config=...)``).
        backend: Backend for circuit execution.
        optimizer: Optimizer for each sub-program; every sub-program gets its
            own copy.
        program_factory: Program class or callable returning a new
            solution-sampling program for each sub-problem. It must build the
            same program for the same sub-problem every time, decode
            candidates in the sub-problem's own variable order, and accept
            ``feasibility`` in ``get_top_solutions`` when the problem sets
            ``candidate_feasibility``.
        **kwargs: If ``early_stopping`` is present it is extracted and
            deep-copied per sub-program.  ``reporting_level`` accepts a
            :class:`~divi.qprog.ensembles.ReportingLevel` controlling how much live
            progress is shown.  ``sampling_backend`` optionally selects a
            separate backend for final solution sampling.  ``seed`` seeds
            each sub-program with its own seed derived from it and the
            sub-problem's position in the decomposition.  Remaining
            kwargs are passed to ``program_factory`` for every sub-program.
    """

    def __init__(
        self,
        problem: QAOAProblem,
        *,
        backend: CircuitRunner,
        optimizer: Optimizer,
        program_factory: Callable[..., SolutionSamplingMixin],
        **kwargs,
    ):
        super().__init__(
            backend=backend,
            sampling_backend=kwargs.pop("sampling_backend", None),
            reporting_level=kwargs.pop("reporting_level", ReportingLevel.COMPACT),
        )
        self._problem = problem
        self._optimizer_template = optimizer
        self._early_stopping_template = kwargs.pop("early_stopping", None)
        self._seed = kwargs.pop("seed", None)
        self._constructor = partial(program_factory, backend=backend, **kwargs)

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _check_best_probs_available(self):
        """Validate that all programs have computed final probabilities."""
        if any(len(program.best_probs) == 0 for program in self.programs.values()):
            raise RuntimeError(
                "Not all final probabilities computed yet. Please call `run()` first."
            )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def create_programs(self, state=None):
        """Decompose the problem and create quantum programs for each sub-problem.

        Raises:
            TypeError: If a sub-program does not sample solutions.
            ValueError: If a sub-program is the same object as an earlier one.
        """
        super().create_programs()
        programs = {}
        sub_problems = self._problem.decompose()
        for slot, (prog_id, sub_problem) in enumerate(sub_problems.items()):
            program = self._constructor(
                problem=sub_problem,
                seed=_child_seed(self._seed, slot),
                optimizer=self._optimizer_template.copy(),
                early_stopping=copy.deepcopy(self._early_stopping_template),
            )
            if not isinstance(program, SolutionSamplingMixin):
                raise TypeError(
                    "Partition sub-programs must sample solutions "
                    f"(SolutionSamplingMixin); program {prog_id!r} is "
                    f"{type(program).__name__}."
                )
            if any(program is other for other in programs.values()):
                raise ValueError(
                    f"Program {prog_id!r} is the same object as an earlier "
                    "partition's; build a new program for every partition."
                )
            programs[prog_id] = program
        self._programs.update(programs)

    def _child_recovery_states(self):
        if not self._problem._has_reproducible_decomposition():
            raise ValueError(
                "Ensemble checkpointing requires a deterministic decomposer for "
                f"{type(self._problem).__name__}."
            )
        return super()._child_recovery_states()

    def _aggregate(self, strategy, top_n):
        def _extend_fn(current, prog_id, candidate):
            return self._problem.extend_solution(current, prog_id, candidate.decoded)

        feasibility = self._problem.candidate_feasibility
        candidate_kwargs = (
            {} if feasibility == "ignore" else {"feasibility": feasibility}
        )
        order = self._problem.order_partitions(list(self._programs))
        return strategy.aggregate(
            programs={prog_id: self._programs[prog_id] for prog_id in order},
            initial_solution=[0] * self._problem.initial_solution_size(),
            extend_fn=_extend_fn,
            evaluate_fn=self._problem.evaluate_global_solution,
            top_n=top_n,
            **candidate_kwargs,
        )

    def aggregate_results(self, strategy: AggregationStrategy = BeamSearchStrategy()):
        """Aggregate partition results into a global solution.

        Args:
            strategy: An :class:`~divi.qprog.ensembles.AggregationStrategy` controlling how
                per-partition candidates are combined. Defaults to
                :class:`~divi.qprog.ensembles.BeamSearchStrategy`.

        Returns:
            Problem-specific post-processed result (see
            ``QAOAProblem.postprocess_candidates``), or ``None`` if
            post-processing rejects all candidates.
        """
        results = self.get_top_solutions(1, strategy=strategy)
        if not results:
            warnings.warn(
                "aggregate_results produced no valid post-processed solution "
                f"with {type(strategy).__name__}. "
                "Try wider parameters or "
                "get_top_solutions(..., strict=True).",
                UserWarning,
                stacklevel=2,
            )
            return None

        return results[0]

    def get_top_solutions(
        self,
        n=10,
        *,
        strategy: AggregationStrategy = BeamSearchStrategy(),
        strict: bool = False,
    ):
        """Get the top-N global solutions from partition aggregation.

        Args:
            n: Number of top solutions to return (>= 1).
            strategy: An :class:`~divi.qprog.ensembles.AggregationStrategy` controlling how
                per-partition candidates are combined. Defaults to
                :class:`~divi.qprog.ensembles.BeamSearchStrategy`.
            strict: Ask problem-specific post-processing to reject invalid raw
                constrained solutions rather than repair them. The returned
                list may contain fewer than *n* entries for constrained
                problems.

        Returns:
            Problem-specific post-processed results (see
            ``QAOAProblem.postprocess_candidates``).
        """
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")

        self._check_ready_for_aggregation()
        self._check_best_probs_available()

        top_results = self._aggregate(strategy, top_n=n)

        return self._problem.postprocess_candidates(top_results, strict=strict)
