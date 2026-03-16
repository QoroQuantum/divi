# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import copy
from abc import abstractmethod
from typing import Any

from divi.backends import CircuitRunner
from divi.qprog.ensemble import ProgramEnsemble, _beam_search_aggregate_top_n
from divi.qprog.optimizers import Optimizer, copy_optimizer
from divi.qprog.variational_quantum_algorithm import SolutionEntry


class PartitioningProgramEnsemble(ProgramEnsemble):
    """Abstract base for partition-solve-aggregate quantum optimization workflows.

    Provides concrete implementations of :meth:`aggregate_results` and
    :meth:`get_top_solutions` using beam search.  Subclasses implement five
    hooks that supply domain-specific logic — they never override the
    lifecycle methods directly.

    Args:
        backend: Backend responsible for running quantum circuits.
        quantum_routine: Which quantum algorithm to use per partition.
            Defaults to ``"qaoa"``.
        optimizer: Optimizer for each sub-program.
        max_iterations: Maximum optimization iterations per sub-program.
        **kwargs: If ``early_stopping`` is present it is extracted and
            deep-copied per sub-program.  Remaining kwargs are stored in
            ``_engine_kwargs`` for subclass use.

    Hooks to implement:

    * :meth:`_initial_solution` — zero-vector sized to the full problem.
    * :meth:`_extend_solution` — map a partition candidate into the global
      solution vector.
    * :meth:`_evaluate_global_solution` — score a complete global solution.
    * :meth:`_finalize_best` — store state and return the formatted result
      for :meth:`aggregate_results`.
    * :meth:`_format_top_results` — format beam search output for
      :meth:`get_top_solutions`.
    """

    def __init__(
        self,
        backend: CircuitRunner,
        quantum_routine: str = "qaoa",
        *,
        optimizer: Optimizer,
        max_iterations: int = 10,
        **kwargs,
    ):
        super().__init__(backend=backend)
        self.quantum_routine = quantum_routine
        self.max_iterations = max_iterations
        self._optimizer_template = optimizer
        self._early_stopping_template = kwargs.pop("early_stopping", None)
        self._engine_kwargs = kwargs

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _make_program_args(self, prog_id) -> dict:
        """Return common kwargs for instantiating a sub-program.

        Copies the optimizer and early-stopping templates so that each
        sub-program gets independent instances.
        """
        return dict(
            program_id=prog_id,
            optimizer=copy_optimizer(self._optimizer_template),
            early_stopping=copy.deepcopy(self._early_stopping_template),
            progress_queue=self._queue,
        )

    # ------------------------------------------------------------------
    # Abstract hooks
    # ------------------------------------------------------------------

    @abstractmethod
    def _initial_solution(self) -> list[int]:
        """Return the initial (all-zeros) solution vector for beam search.

        The length must equal the full problem size (e.g. number of graph
        nodes or QUBO variables).  Beam search starts from this vector and
        progressively fills it via :meth:`_extend_solution`.

        Example — a 6-node graph problem::

            def _initial_solution(self):
                return [0] * self.main_graph.number_of_nodes()
        """

    @abstractmethod
    def _extend_solution(
        self,
        current_solution: list[int],
        prog_id: Any,
        candidate: SolutionEntry,
    ) -> list[int]:
        """Merge a partition candidate into the global solution vector.

        Called once per partition during beam search.  Must return a **new**
        list (do not mutate *current_solution*) with the candidate's bits
        written into the positions that belong to *prog_id*.

        Example — a QUBO workflow with ``_variable_maps``::

            def _extend_solution(self, current_solution, prog_id, candidate):
                extended = list(current_solution)
                for local_idx, global_idx in enumerate(self._variable_maps[prog_id]):
                    extended[global_idx] = int(candidate.decoded[local_idx])
                return extended
        """

    @abstractmethod
    def _evaluate_global_solution(self, solution: list[int]) -> float:
        """Score a complete global solution.  Lower is better."""

    @abstractmethod
    def _finalize_best(self, score: float, solution: list[int]) -> Any:
        """Store solution state and return the formatted result.

        Called by :meth:`aggregate_results` with the single best beam search
        result.  Subclasses should persist whatever attributes they expose
        (e.g. ``self.solution``) and return the public result.
        """

    @abstractmethod
    def _format_top_results(self, results: list[tuple[float, list[int]]]) -> Any:
        """Format beam search output for :meth:`get_top_solutions`."""

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_best_probs_available(self):
        """Validate that all programs have computed final probabilities."""
        if any(len(program.best_probs) == 0 for program in self.programs.values()):
            raise RuntimeError(
                "Not all final probabilities computed yet. "
                "Please call `run()` first."
            )

    # ------------------------------------------------------------------
    # Concrete lifecycle (template methods)
    # ------------------------------------------------------------------

    def aggregate_results(self, beam_width=1, n_partition_candidates=None):
        """Aggregate partition results into a global solution via beam search.

        This is a concrete template method.  Subclasses customise behavior
        through the five abstract hooks rather than overriding this method.

        Args:
            beam_width (int | None): Width of the beam search.  ``1`` is
                greedy, ``None`` is exhaustive.
            n_partition_candidates (int | None): Candidates to fetch per
                partition.  Defaults to *beam_width*.

        Returns:
            Subclass-specific format (see :meth:`_finalize_best`).

        Raises:
            RuntimeError: If programs haven't been created or run.
        """
        super().aggregate_results()
        self._check_best_probs_available()

        score, best_solution = _beam_search_aggregate_top_n(
            programs=self._programs,
            initial_solution=self._initial_solution(),
            extend_fn=self._extend_solution,
            evaluate_fn=self._evaluate_global_solution,
            beam_width=beam_width,
            n_partition_candidates=n_partition_candidates,
        )[0]

        return self._finalize_best(score, best_solution)

    def get_top_solutions(self, n=10, *, beam_width=1, n_partition_candidates=None):
        """Get the top-N global solutions from beam search aggregation.

        This is a concrete template method.  Subclasses customise the return
        format through :meth:`_format_top_results`.

        Args:
            n (int): Number of top solutions to return.  Must be >= 1.
            beam_width (int | None): Beam search width.
            n_partition_candidates (int | None): Candidates per partition.

        Returns:
            Subclass-specific format (see :meth:`_format_top_results`).

        Raises:
            ValueError: If *n* < 1.
            RuntimeError: If programs haven't been created or run.
        """
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")

        self._check_ready_for_aggregation()
        self._check_best_probs_available()

        top_results = _beam_search_aggregate_top_n(
            programs=self._programs,
            initial_solution=self._initial_solution(),
            extend_fn=self._extend_solution,
            evaluate_fn=self._evaluate_global_solution,
            beam_width=beam_width,
            n_partition_candidates=n_partition_candidates,
            top_n=n,
        )

        return self._format_top_results(top_results)
