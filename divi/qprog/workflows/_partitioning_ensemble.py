# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import copy
import warnings
from functools import partial
from typing import Literal

from divi.backends import CircuitRunner
from divi.qprog.aggregation import AggregationStrategy, BeamSearchStrategy
from divi.qprog.algorithms import PCE, QAOA, IterativeQAOA
from divi.qprog.ensemble import ProgramEnsemble
from divi.qprog.optimizers import Optimizer
from divi.qprog.problems import BinaryOptimizationProblem, QAOAProblem


class PartitioningProgramEnsemble(ProgramEnsemble):
    """Generic orchestrator for partition-solve-aggregate quantum optimization.

    Delegates all domain-specific logic to the :class:`~divi.qprog.problems.QAOAProblem` instance:
    decomposition, solution extension, evaluation, and result post-processing.
    The ensemble handles program creation, execution, and result aggregation.

    Args:
        problem: A :class:`~divi.qprog.problems.QAOAProblem` configured for decomposition
            (e.g. ``MaxCutProblem(graph, partitioning_config=...)``).
        n_layers: Number of ansatz layers per sub-program.
        backend: Backend for circuit execution.
        optimizer: Optimizer for each sub-program.
        quantum_routine: Per-partition quantum algorithm.
            ``"qaoa"`` (default), ``"pce"``, or ``"iterative_qaoa"``.
        max_iterations: Max optimization iterations per sub-program.
        **kwargs: If ``early_stopping`` is present it is extracted and
            deep-copied per sub-program.  Remaining kwargs are forwarded
            to the engine constructor.
    """

    def __init__(
        self,
        problem: QAOAProblem,
        n_layers: int,
        backend: CircuitRunner,
        optimizer: Optimizer,
        quantum_routine: Literal["qaoa", "pce", "iterative_qaoa"] = "qaoa",
        max_iterations: int = 10,
        **kwargs,
    ):
        super().__init__(backend=backend)
        self._problem = problem
        self.quantum_routine = quantum_routine
        self.max_iterations = max_iterations
        self._optimizer_template = optimizer
        self._early_stopping_template = kwargs.pop("early_stopping", None)
        self._engine_kwargs = kwargs

        # Build the engine constructor partial
        _ENGINE_MAP = {
            "qaoa": (QAOA, dict(max_iterations=max_iterations, n_layers=n_layers)),
            "pce": (PCE, dict(max_iterations=max_iterations, n_layers=n_layers)),
            "iterative_qaoa": (IterativeQAOA, dict(max_depth=n_layers)),
        }

        routine = quantum_routine.lower()
        if routine not in _ENGINE_MAP:
            raise ValueError(
                f"Unsupported quantum_routine: {quantum_routine!r}. "
                f"Supported values are {', '.join(map(repr, _ENGINE_MAP))}."
            )

        self._engine_cls, engine_args = _ENGINE_MAP[routine]
        self._constructor = partial(
            self._engine_cls, backend=backend, **engine_args, **self._engine_kwargs
        )

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _make_program_args(self, prog_id) -> dict:
        """Return common kwargs for instantiating a sub-program."""
        return dict(
            program_id=prog_id,
            optimizer=self._optimizer_template.copy(),
            early_stopping=copy.deepcopy(self._early_stopping_template),
            progress_queue=self._queue,
        )

    def _check_best_probs_available(self):
        """Validate that all programs have computed final probabilities."""
        if any(len(program.best_probs) == 0 for program in self.programs.values()):
            raise RuntimeError(
                "Not all final probabilities computed yet. "
                "Please call `run()` first."
            )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def create_programs(self):
        """Decompose the problem and create quantum programs for each sub-problem."""
        super().create_programs()
        sub_problems = self._problem.decompose()
        for prog_id, sub_problem in sub_problems.items():
            # QAOA/IterativeQAOA need Problem objects;
            # PCE takes raw QUBO matrices directly.
            if self._engine_cls == PCE and isinstance(
                sub_problem, BinaryOptimizationProblem
            ):
                problem_arg = sub_problem.raw_problem
            else:
                problem_arg = sub_problem

            self._programs[prog_id] = self._constructor(
                problem=problem_arg,
                **self._make_program_args(prog_id),
            )

    def _aggregate(self, strategy, top_n):
        def _extend_fn(current, prog_id, candidate):
            return self._problem.extend_solution(current, prog_id, candidate.decoded)

        return strategy.aggregate(
            programs=self._programs,
            initial_solution=[0] * self._problem.initial_solution_size(),
            extend_fn=_extend_fn,
            evaluate_fn=self._problem.evaluate_global_solution,
            top_n=top_n,
        )

    def aggregate_results(self, strategy: AggregationStrategy | None = None):
        """Aggregate partition results into a global solution.

        Args:
            strategy: An :class:`~divi.qprog.AggregationStrategy` controlling how
                per-partition candidates are combined. Defaults to
                :class:`~divi.qprog.BeamSearchStrategy`.

        Returns:
            Problem-specific post-processed result (see
            ``QAOAProblem.postprocess_candidates``), or ``None`` if
            post-processing rejects all candidates.
        """
        super().aggregate_results()
        self._check_best_probs_available()

        if strategy is None:
            strategy = BeamSearchStrategy()

        candidates = self._aggregate(strategy, top_n=1)
        results = self._problem.postprocess_candidates(candidates)
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
        strategy: AggregationStrategy | None = None,
        strict: bool = False,
    ):
        """Get the top-N global solutions from partition aggregation.

        Args:
            n: Number of top solutions to return (>= 1).
            strategy: An :class:`~divi.qprog.AggregationStrategy` controlling how
                per-partition candidates are combined. Defaults to
                :class:`~divi.qprog.BeamSearchStrategy`.
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

        if strategy is None:
            strategy = BeamSearchStrategy()

        top_results = self._aggregate(strategy, top_n=n)

        return self._problem.postprocess_candidates(top_results, strict=strict)
