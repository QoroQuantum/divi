# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import atexit
import traceback
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from queue import Queue
from threading import Event, Lock, Thread
from typing import Any
from warnings import warn

from rich.console import Console
from rich.progress import TaskID

from divi.backends import CircuitRunner
from divi.qprog._batch_coordinator import (
    BatchConfig,
    BatchMode,
    _BatchCoordinator,
    _ProxyBackend,
)
from divi.qprog.quantum_program import QuantumProgram
from divi.qprog.variational_quantum_algorithm import (
    SolutionEntry,
    VariationalQuantumAlgorithm,
)
from divi.reporting import (
    disable_logging,
    make_batch_display,
    make_progress_display,
    queue_listener,
)


def _default_task_function(program: QuantumProgram):
    return program.run()


class ProgramEnsemble(ABC):
    """This abstract class provides the basic scaffolding for higher-order
    computations that require more than one quantum program to achieve its goal.

    Each implementation of this class has to have an implementation of two functions:
        1. `create_programs`: This function generates the independent programs that
            are needed to achieve the objective of the job. The creation of those
            programs can utilize the instance variables of the class to initialize
            their parameters. The programs should be stored in a key-value store
            where the keys represent the identifier of the program, whether random
            or identificatory.

        2. `aggregate_results`: This function aggregates the results of the programs
            after they are done executing. This function should be aware of the different
            formats the programs might have (counts dictionary, expectation value, etc) and
            handle such cases accordingly.
    """

    def __init__(self, backend: CircuitRunner):
        super().__init__()

        self.backend = backend
        self._executor = None
        self._task_fn = _default_task_function
        self._programs = {}
        self._coordinator: _BatchCoordinator | None = None
        self._program_key_map: dict[QuantumProgram, str] = {}

        self._total_circuit_count = 0
        self._total_run_time = 0.0

        self._is_jupyter = Console().is_jupyter

        # Disable logging since we already have the bars to track progress
        disable_logging()

    @property
    def total_circuit_count(self):
        """
        Get the total number of circuits executed across all programs in the batch.

        Returns:
            int: Cumulative count of circuits submitted by all programs.
        """
        return self._total_circuit_count

    @property
    def total_run_time(self):
        """
        Get the total runtime across all programs in the batch.

        Returns:
            float: Cumulative execution time in seconds across all programs.
        """
        return self._total_run_time

    @property
    def programs(self) -> dict:
        """
        Get a copy of the programs dictionary.

        Returns:
            dict: Copy of the programs dictionary mapping program IDs to
                QuantumProgram instances. Modifications to this dict will not
                affect the internal state.
        """
        return self._programs.copy()

    @programs.setter
    def programs(self, value: dict):
        """Set the programs dictionary."""
        self._programs = value

    @abstractmethod
    def create_programs(self):
        """Generate and populate the programs dictionary for batch execution.

        This method must be implemented by subclasses to create the quantum programs
        that will be executed as part of the batch. The method operates via side effects:
        it populates `self._programs` (or `self.programs`) with a dictionary mapping
        program identifiers to `QuantumProgram` instances.

        Implementation Notes:
            - Subclasses should call `super().create_programs()` first to initialize
              internal state (queue, events, etc.) and validate that no programs
              already exist.
            - After calling super(), subclasses should populate `self.programs` or
              `self._programs` with their program instances.
            - Program identifiers can be any hashable type (e.g., strings, tuples).
              Common patterns include strings like "program_1", "program_2" or tuples like
              ('A', 5) for partitioned problems.

        Side Effects:
            - Populates `self._programs` with program instances.
            - Initializes `self._queue` for progress reporting.
            - Initializes `self._done_event` if `max_iterations` attribute exists.

        Raises:
            RuntimeError: If programs already exist (should call `reset()` first).

        Example:
            >>> def create_programs(self):
            ...     super().create_programs()
            ...     self.programs = {
            ...         "prog1": QAOA(...),
            ...         "prog2": QAOA(...),
            ...     }
        """
        if len(self._programs) > 0:
            raise RuntimeError(
                "Some programs already exist. "
                "Clear the program dictionary before creating new ones by using batch.reset()."
            )

        self._queue = Queue()
        self._done_event = Event()

    def reset(self):
        """
        Reset the batch to its initial state.

        Clears all programs, stops any running executors, terminates listener threads,
        and stops progress bars. This allows the batch to be reused for a new set of
        programs.

        Note:
            Any running programs will be forcefully stopped. Results from incomplete
            programs will be lost.
        """
        # Shutdown coordinator before clearing programs
        if self._coordinator is not None:
            self._coordinator.shutdown()
            self._coordinator = None
        self._program_key_map.clear()

        self._programs.clear()

        # Stop any active executor
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None
            self.futures = None

        # Signal and wait for listener thread to stop
        if hasattr(self, "_done_event") and self._done_event is not None:
            self._done_event.set()
            self._done_event = None

        if getattr(self, "_listener_thread", None) is not None:
            self._listener_thread.join(timeout=1)
            if self._listener_thread.is_alive():
                warn("Listener thread did not terminate within timeout.")
            self._listener_thread = None

        # Stop the live display if it's still active
        if getattr(self, "_live_display", None) is not None:
            try:
                self._live_display.stop()
            except Exception:
                pass  # Already stopped or not running
            self._live_display = None
            self._progress_bar = None
            self._batch_progress = None
            self._pb_task_map.clear()

    def _atexit_cleanup_hook(self):
        # This hook is only registered for non-blocking runs.
        if self._executor is not None:
            warn(
                "A non-blocking ProgramEnsemble run was not explicitly closed with "
                "'join()'. The batch was cleaned up automatically on exit.",
                UserWarning,
            )
            self.reset()

    def _add_program_to_executor(self, program: QuantumProgram) -> Future:
        """
        Add a quantum program to the thread pool executor for execution.

        Sets up the program with cancellation support and progress tracking, then
        submits it for execution in a separate thread.  The program is
        automatically deregistered from the batch coordinator when it finishes.

        Args:
            program (QuantumProgram): The quantum program to execute.

        Returns:
            Future: A Future object representing the program's execution.
        """
        if hasattr(program, "_set_cancellation_event"):
            program._set_cancellation_event(self._cancellation_event)

        if self._progress_bar is not None:
            with self._pb_lock:
                total = getattr(
                    program,
                    "_expected_total_iterations",
                    getattr(self, "max_iterations", 1),
                )
                task_id = self._progress_bar.add_task(
                    "",
                    job_name=f"Program {program.program_id}",
                    total=total,
                    completed=0,
                    message="",
                    batch_color="",
                )
                self._pb_task_map[program.program_id] = task_id

                # Link program_key to this progress bar task for batch coloring
                prog_key = self._program_key_map.get(program)
                if prog_key is not None:
                    self._program_key_to_task_ids.setdefault(prog_key, []).append(
                        task_id
                    )

        coordinator = self._coordinator
        program_key = self._program_key_map.get(program)
        task_fn = self._task_fn

        def _coordinated_task(prog):
            try:
                return task_fn(prog)
            finally:
                if coordinator is not None and program_key is not None:
                    coordinator.deregister_program(program_key)

        return self._executor.submit(_coordinated_task, program)

    def run(
        self,
        blocking: bool = False,
        *,
        batch_config: BatchConfig = BatchConfig(),
    ):
        """
        Execute all programs in the batch.

        Starts all quantum programs in parallel using a thread pool. Can run in
        blocking or non-blocking mode.

        Args:
            blocking (bool, optional): If True, waits for all programs to complete
                before returning. If False, returns immediately and programs run in
                the background. Defaults to False.
            batch_config (BatchConfig): Configuration for circuit batching.
                Controls whether submissions are merged and how.  Defaults to
                ``BatchConfig()`` which merges all submissions via a wait-for-all
                barrier.  Use ``BatchConfig(mode=BatchMode.OFF)`` to let each
                program submit independently, or
                ``BatchConfig(max_batch_size=50)`` to cap the number of circuits
                per merged backend call.

        Returns:
            ProgramEnsemble: Returns self for method chaining.

        Raises:
            RuntimeError: If a batch is already running or if no programs have been
                created.

        Note:
            In non-blocking mode, call `join()` later to wait for completion and
            collect results.
        """
        if self._executor is not None:
            raise RuntimeError("A batch is already being run.")

        if len(self._programs) == 0:
            raise RuntimeError("No programs to run.")

        self._progress_bar = None
        self._batch_progress = None
        self._live_display = None
        batching_enabled = batch_config.mode is BatchMode.MERGED
        _wants_progress = hasattr(self, "max_iterations") or getattr(
            self, "_show_progress", False
        )
        if _wants_progress:
            if batching_enabled:
                self._batch_progress, self._progress_bar, self._live_display = (
                    make_batch_display(is_jupyter=self._is_jupyter)
                )
            else:
                self._progress_bar, self._live_display = make_progress_display(
                    is_jupyter=self._is_jupyter
                )

        # Validate that all program instances are unique to prevent thread-safety issues
        program_instances = list(self._programs.values())
        if len(set(program_instances)) != len(program_instances):
            raise RuntimeError(
                "Duplicate program instances detected in batch. "
                "QuantumProgram instances are stateful and NOT thread-safe. "
                "You must provide a unique instance for each program ID."
            )

        self._executor = ThreadPoolExecutor()
        self._cancellation_event = Event()
        self.futures = []
        self._future_to_program = {}
        self._pb_task_map = {}
        self._pb_lock = Lock()
        self._program_key_to_task_ids: dict[str, list[TaskID]] = {}

        # Set up batch coordinator to merge circuit submissions.
        if batching_enabled:
            progress_queue = getattr(self, "_queue", None)
            self._coordinator = _BatchCoordinator(
                self.backend,
                progress_queue=progress_queue,
                batch_config=batch_config,
            )
            self._program_key_map = {}
            for idx, (prog_id, program) in enumerate(self._programs.items()):
                program_key = str(idx)
                self._program_key_map[program] = program_key
                self._coordinator.register_program(program_key)
                program.backend = _ProxyBackend(
                    self.backend, self._coordinator, program_key
                )

        if self._progress_bar is not None:
            self._live_display.start()

            listener_kwargs = {
                "live_display": self._live_display,
                "is_jupyter": self._is_jupyter,
            }

            if batching_enabled and self._batch_progress is not None:
                listener_kwargs.update(
                    batch_progress=self._batch_progress,
                    batch_task_ids={},
                    program_key_to_task_ids=self._program_key_to_task_ids,
                )

            self._listener_thread = Thread(
                target=queue_listener,
                args=(
                    self._queue,
                    self._progress_bar,
                    self._pb_task_map,
                    self._done_event,
                    self._pb_lock,
                ),
                kwargs=listener_kwargs,
                daemon=True,
            )
            self._listener_thread.start()

        for program in self._programs.values():
            future = self._add_program_to_executor(program)
            self.futures.append(future)
            self._future_to_program[future] = program

        if not blocking:
            # Arm safety net
            atexit.register(self._atexit_cleanup_hook)
        else:
            self.join()

        return self

    def check_all_done(self) -> bool:
        """
        Check if all programs in the batch have completed execution.

        Returns:
            bool: True if all programs are finished (successfully or with errors),
                False if any are still running.
        """
        return all(future.done() for future in self.futures)

    def _collect_completed_results(self, completed_futures: list):
        """
        Collects results from any futures that have completed successfully.
        Appends (circuit_count, run_time) tuples to the completed_futures list.

        Args:
            completed_futures: List to append results to
        """
        for future in self.futures:
            if future.done() and not future.cancelled():
                try:
                    completed_futures.append(future.result())
                except Exception:
                    pass  # Skip failed futures

    def _handle_cancellation(self):
        """Handle cancellation gracefully with accurate progress feedback.

        With the batch coordinator active, cancellation works as follows:
        1. ``coordinator.cancel()`` sets the cancelled flag, cancels any
           in-flight backend jobs, and resolves pending futures with
           ``_CancelledError``.
        2. The program threads see the ``_CancelledError`` (or the
           ``_cancellation_event``) and exit.
        3. We wait for all still-running futures and mark them in the
           progress bar.

        Without the coordinator the legacy path applies: we try
        ``future.cancel()`` for pending tasks and ``cancel_unfinished_job()``
        for running ones.
        """
        self._cancellation_event.set()

        # Cancel the coordinator first — it unblocks all programs waiting
        # on the barrier and cancels real backend jobs.
        if self._coordinator is not None:
            self._coordinator.cancel()

        successfully_cancelled = []
        unstoppable_futures = []

        for future, program in self._future_to_program.items():
            if future.done():
                continue

            cancel_result = future.cancel()
            if cancel_result:
                successfully_cancelled.append(program)
            else:
                # Already running — cancel the backend job directly only
                # when there is no coordinator (the coordinator already
                # cancelled real backend jobs above; the proxy has no job_id).
                if self._coordinator is None:
                    program.cancel_unfinished_job()
                unstoppable_futures.append(future)
                task_id = self._pb_task_map.get(program.program_id)
                if self._progress_bar and task_id is not None:
                    self._progress_bar.update(task_id, message="Finishing... ⏳")

        # Immediately mark successfully cancelled tasks
        for program in successfully_cancelled:
            task_id = self._pb_task_map.get(program.program_id)
            if self._progress_bar and task_id is not None:
                self._progress_bar.update(
                    task_id,
                    final_status="Cancelled",
                    message="Cancelled by user",
                )

        # Wait for running tasks to finish
        for future in as_completed(unstoppable_futures):
            program = self._future_to_program[future]
            task_id = self._pb_task_map.get(program.program_id)
            if self._progress_bar and task_id is not None:
                self._progress_bar.update(
                    task_id,
                    final_status="Aborted",
                    message="Completed during cancellation",
                )

    def join(self):
        """
        Wait for all programs in the batch to complete and collect results.

        Blocks until all programs finish execution, aggregating their circuit counts
        and run times. Handles keyboard interrupts gracefully by attempting to cancel
        remaining programs.

        Returns:
            bool or None: Returns False if interrupted by KeyboardInterrupt, None otherwise.

        Raises:
            RuntimeError: If any program fails with an exception, after cancelling
                remaining programs.

        Note:
            This method should be called after `run(blocking=False)` to wait for
            completion. It's automatically called when using `run(blocking=True)`.
        """
        if self._executor is None:
            return

        completed_futures = []
        try:
            # The as_completed iterator will yield futures as they finish.
            # If a task fails, future.result() will raise the exception immediately.
            for future in as_completed(self.futures):
                completed_futures.append(future.result())

        except KeyboardInterrupt:
            if self._progress_bar is not None:
                self._progress_bar.console.print(
                    "[bold yellow]Shutdown signal received, waiting for programs "
                    "to finish current iteration...[/bold yellow]"
                )
            self._handle_cancellation()

            # Re-collect all completed results from scratch to avoid duplicates
            # from the as_completed loop above.
            completed_futures.clear()
            self._collect_completed_results(completed_futures)

            return False

        except Exception as e:
            # A task has failed. Cancel the coordinator and remaining futures.
            traceback.print_exception(type(e), e, e.__traceback__)

            if self._coordinator is not None:
                self._coordinator.cancel()
            for f in self.futures:
                f.cancel()

            # Re-collect all completed results from scratch to avoid duplicates
            # from the as_completed loop above.
            completed_futures.clear()
            self._collect_completed_results(completed_futures)

            # Re-raise a new error to indicate the batch failed.
            raise RuntimeError("Batch execution failed and was cancelled.") from e

        finally:
            # Aggregate results from completed futures
            if completed_futures:
                self._total_circuit_count += sum(
                    result[0] for result in completed_futures
                )
                # For async backends the individual programs don't track runtime
                # (the proxy returns sync results). Use the coordinator's total
                # which is captured from the real backend's poll responses.
                if (
                    self._coordinator is not None
                    and self._coordinator.total_runtime > 0
                ):
                    self._total_run_time += self._coordinator.total_runtime
                else:
                    self._total_run_time += sum(
                        result[1] for result in completed_futures
                    )
                self.futures.clear()

            # Shutdown coordinator
            if self._coordinator is not None:
                self._coordinator.shutdown()
                self._coordinator = None

            # Shutdown executor and wait for all threads to complete
            # This is critical for Python 3.12 to prevent process hangs
            if self._executor is not None:
                self._executor.shutdown(wait=True)
                self._executor = None

            if self._progress_bar is not None:
                self._queue.join()
                self._done_event.set()
                self._listener_thread.join()
                self._live_display.stop()

        # After successful cleanup, try to unregister the hook.
        try:
            atexit.unregister(self._atexit_cleanup_hook)
        except TypeError:
            pass

    def _check_ready_for_aggregation(self):
        """Validate that programs exist, are complete, and results are ready."""
        if len(self._programs) == 0:
            raise RuntimeError("No programs to aggregate. Run create_programs() first.")

        if self._executor is not None:
            self.join()

        for program in self._programs.values():
            if isinstance(program, VariationalQuantumAlgorithm):
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        category=UserWarning,
                        message=".*losses_history is empty.*",
                    )
                    if len(program.losses_history) == 0:
                        raise RuntimeError(
                            "Some/All programs have empty losses. Did you call run()?"
                        )
            else:
                # Non-VQA programs (QuantumProgram subclasses, mocks, etc.)
                # Check results first; fall back to losses_history for
                # backward compatibility with test helpers that set it.
                results = getattr(program, "results", None)
                if results is not None:
                    continue
                losses = getattr(program, "losses_history", None)
                if losses is not None:
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore",
                            category=UserWarning,
                            message=".*losses_history is empty.*",
                        )
                        if len(losses) == 0:
                            raise RuntimeError(
                                "Some/All programs have empty losses. "
                                "Did you call run()?"
                            )
                    continue
                raise RuntimeError(
                    "Some/All programs have no results. Did you call run()?"
                )

    @abstractmethod
    def aggregate_results(self):
        """
        Aggregate results from all programs in the batch after execution.

        This is an abstract method that must be implemented by subclasses. The base
        implementation performs validation checks:
        - Ensures programs have been created
        - Waits for any running programs to complete (calls join() if needed)
        - Verifies that all programs have completed execution (non-empty losses_history)

        Subclasses should call super().aggregate_results() first, then implement
        their own aggregation logic to combine results from all programs. The
        aggregation should handle different result formats (counts dictionary,
        expectation values, etc.) as appropriate for the specific use case.

        Returns:
            The aggregated result, format depends on the subclass implementation.

        Raises:
            RuntimeError: If no programs exist, or if programs haven't completed
                execution (empty losses_history).
        """
        self._check_ready_for_aggregation()

    def get_top_solutions(self, n=10, *, beam_width=1, n_partition_candidates=None):
        """Get the top-N global solutions from beam search aggregation.

        Available on subclasses that use beam-search-based aggregation
        (e.g., ``PartitioningProgramEnsemble``).

        Args:
            n (int): Number of top solutions to return. Must be >= 1.
            beam_width (int | None): Beam search width. Internally bumped
                to at least ``n`` so the beam retains enough candidates.
            n_partition_candidates (int | None): Candidates per partition.
                Defaults to ``beam_width``.

        Returns:
            Subclass-specific format. See subclass documentation.

        Raises:
            NotImplementedError: If the subclass does not override this method.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support get_top_solutions."
        )


def _beam_search_aggregate_top_n(
    programs: dict[Any, VariationalQuantumAlgorithm],
    initial_solution: Sequence[int],
    extend_fn: Callable[[list[int], Any, SolutionEntry], list[int]],
    evaluate_fn: Callable[[list[int]], float],
    beam_width: int | None = None,
    n_partition_candidates: int | None = None,
    top_n: int = 1,
) -> list[tuple[float, list[int]]]:
    """Core beam search returning the top-N ``(score, solution)`` pairs.

    Args:
        programs: Dictionary mapping program IDs to executed
            ``VariationalQuantumAlgorithm`` instances.
        initial_solution: Starting solution vector.
        extend_fn: ``(current_solution, prog_id, candidate) -> extended_solution``.
        evaluate_fn: ``(solution) -> float``.  Lower is better.
        beam_width: Maximum candidates to retain per partition step.
            ``None`` means keep all (exhaustive).
        n_partition_candidates: Candidates to fetch per partition.
            Defaults to ``beam_width`` (or all when exhaustive).
        top_n: Number of top solutions to return.

    Returns:
        List of ``(score, solution)`` tuples sorted ascending by score
        (best first), with at most ``top_n`` entries.
    """
    if beam_width is not None and beam_width < 1:
        raise ValueError(f"beam_width must be >= 1 or None, got {beam_width}")

    if n_partition_candidates is not None and n_partition_candidates < 1:
        raise ValueError(
            f"n_partition_candidates must be >= 1 or None, got {n_partition_candidates}"
        )

    # Ensure the beam retains enough candidates for top_n
    if beam_width is not None and beam_width < top_n:
        beam_width = top_n

    if (
        beam_width is not None
        and n_partition_candidates is not None
        and n_partition_candidates < beam_width
    ):
        raise ValueError(
            f"n_partition_candidates ({n_partition_candidates}) must be >= "
            f"beam_width ({beam_width}). Extracting fewer candidates than the "
            f"beam width wastes beam capacity."
        )

    # Resolve the number of candidates to fetch per partition
    if n_partition_candidates is not None:
        n_fetch = n_partition_candidates
    elif beam_width is not None:
        n_fetch = beam_width
    else:
        n_fetch = 2**20  # exhaustive

    beam: list[tuple[float, list[int]]] = [
        (evaluate_fn(initial_solution), list(initial_solution))
    ]

    for prog_id, program in programs.items():
        candidates = program.get_top_solutions(n=n_fetch, include_decoded=True)
        if not candidates:
            continue

        new_beam: list[tuple[float, list[int]]] = []
        for _, partial_solution in beam:
            for candidate in candidates:
                extended = extend_fn(partial_solution, prog_id, candidate)
                new_beam.append((evaluate_fn(extended), extended))

        new_beam.sort(key=lambda entry: entry[0])
        beam = new_beam[:beam_width] if beam_width is not None else new_beam

    beam.sort(key=lambda entry: entry[0])
    return beam[:top_n]
