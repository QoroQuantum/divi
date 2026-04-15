# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import copy
from functools import partial
from typing import Literal

from divi.backends import CircuitRunner
from divi.qprog.algorithms import PCE, QAOA, IterativeQAOA
from divi.qprog.ensemble import ProgramEnsemble, _beam_search_aggregate_top_n
from divi.qprog.optimizers import Optimizer, copy_optimizer
from divi.qprog.problems import BinaryOptimizationProblem, QAOAProblem


class PartitioningProgramEnsemble(ProgramEnsemble):
    """Generic orchestrator for partition-solve-aggregate quantum optimization.

    Delegates all domain-specific logic to the :class:`~divi.qprog.problems.QAOAProblem` instance:
    decomposition, solution extension, evaluation, and result formatting.
    The ensemble handles program creation, execution, and beam search.

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
            optimizer=copy_optimizer(self._optimizer_template),
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

    def aggregate_results(self, beam_width=1, n_partition_candidates=None):
        """Aggregate partition results into a global solution via beam search.

        Args:
            beam_width: Width of the beam search. ``1`` is
                greedy, ``None`` is exhaustive.
            n_partition_candidates: Candidates to fetch per
                partition. Defaults to *beam_width*.

        Returns:
            Problem-specific format (see ``QAOAProblem.finalize_solution``).
        """
        super().aggregate_results()
        self._check_best_probs_available()

        def _extend_fn(current, prog_id, candidate):
            return self._problem.extend_solution(current, prog_id, candidate.decoded)

        score, best_solution = _beam_search_aggregate_top_n(
            programs=self._programs,
            initial_solution=[0] * self._problem.initial_solution_size(),
            extend_fn=_extend_fn,
            evaluate_fn=self._problem.evaluate_global_solution,
            beam_width=beam_width,
            n_partition_candidates=n_partition_candidates,
        )[0]

        return self._problem.finalize_solution(score, best_solution)

    def get_top_solutions(self, n=10, *, beam_width=1, n_partition_candidates=None):
        """Get the top-N global solutions from beam search aggregation.

        Args:
            n (int): Number of top solutions to return. Must be >= 1.
            beam_width: Beam search width.
            n_partition_candidates: Candidates per partition.

        Returns:
            Problem-specific format (see ``QAOAProblem.format_top_solutions``).
        """
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")

        self._check_ready_for_aggregation()
        self._check_best_probs_available()

        def _extend_fn(current, prog_id, candidate):
            return self._problem.extend_solution(current, prog_id, candidate.decoded)

        top_results = _beam_search_aggregate_top_n(
            programs=self._programs,
            initial_solution=[0] * self._problem.initial_solution_size(),
            extend_fn=_extend_fn,
            evaluate_fn=self._problem.evaluate_global_solution,
            beam_width=beam_width,
            n_partition_candidates=n_partition_candidates,
            top_n=n,
        )

        return self._problem.format_top_solutions(top_results)
