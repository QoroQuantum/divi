# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import atexit
import os
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from queue import Queue
from threading import Event, Lock, Thread
from typing import Any
from warnings import warn

from rich.console import Console
from rich.panel import Panel

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
    TerminalStatus,
    disable_logging,
    make_progress_display,
    queue_listener,
)
from divi.reporting._pbar import _drain_queue_quietly

__all__ = ["BatchConfig", "BatchMode", "ProgramEnsemble"]

#: Largest ensemble size for which :meth:`ProgramEnsemble.run` will allocate
#: one executor thread per program under the default wait-for-all barrier.
_BARRIER_PROGRAM_LIMIT = 256

#: Above this program count, per-program progress rows are created
#: hidden so the display isn't flooded by hundreds of rows.  Failed,
#: cancelled, and aborted programs are revealed on terminal status so
#: users can still diagnose what went wrong.  In hide mode the
#: ensemble also adds a "Submitting circuits" prep row that ticks up
#: as workers finish circuit construction and call ``submit``.
_HIDE_PROGRAM_ROWS_THRESHOLD = 64

#: Above this many ``max_concurrent_programs``, ``run`` warns to flag a
#: likely misuse of the knob (e.g. the user wanted ``max_batch_size``).
_CONCURRENT_PROGRAMS_SOFT_CAP = 1024


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
        self.futures: list[Future] = []

        self._total_circuit_count = 0
        self._total_run_time = 0.0

        self._progress_bar = None
        self._listener_thread = None
        self._hide_program_rows = False
        self._prep_task_id = None

        self._is_jupyter = Console().is_jupyter

        # Disable logging since we already have the bars to track progress
        disable_logging()

    @property
    def total_circuit_count(self):
        """
        Get the total number of circuits executed across all programs in the ensemble.

        Returns:
            int: Cumulative count of circuits submitted by all programs.
        """
        return self._total_circuit_count

    @property
    def total_run_time(self):
        """
        Get the total runtime across all programs in the ensemble.

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
        """Generate and populate the programs dictionary for ensemble execution.

        This method must be implemented by subclasses to create the quantum programs
        that will be executed as part of the ensemble. The method operates via side effects:
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
                "Clear the program dictionary before creating new ones by using ensemble.reset()."
            )

        self._queue = Queue()
        self._done_event: Event | None = Event()

    def reset(self):
        """
        Reset the ensemble to its initial state.

        Clears all programs, stops any running executors, terminates listener threads,
        and stops progress bars. This allows the ensemble to be reused for a new set of
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
            self.futures.clear()

        # Signal and wait for listener thread to stop
        if hasattr(self, "_done_event") and self._done_event is not None:
            self._done_event.set()
            self._done_event = None

        if (listener_thread := getattr(self, "_listener_thread", None)) is not None:
            listener_thread.join(timeout=1)
            if listener_thread.is_alive():
                warn("Listener thread did not terminate within timeout.")
            self._listener_thread = None

        # Stop the live display if it's still active
        if (live_display := getattr(self, "_live_display", None)) is not None:
            try:
                live_display.stop()
            except Exception:
                pass  # Already stopped or not running
            self._live_display = None
            self._progress_bar = None
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
                    row_kind="program",
                    program_key=self._program_key_map.get(program),
                    final_status=None,
                    visible=not self._hide_program_rows,
                )
                self._pb_task_map[program.program_id] = task_id

        coordinator = self._coordinator
        program_key = self._program_key_map.get(program)
        task_fn = self._task_fn

        def _coordinated_task(prog):
            try:
                return task_fn(prog)
            finally:
                if coordinator is not None and program_key is not None:
                    coordinator.deregister_program(program_key)

        if self._executor is None:
            raise RuntimeError(
                "Cannot submit program: executor is not initialized. Call run() first."
            )
        return self._executor.submit(_coordinated_task, program)

    def run(
        self,
        blocking: bool = False,
        *,
        batch_config: BatchConfig = BatchConfig(),
    ):
        """
        Execute all programs in the ensemble.

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

                The default barrier requires one executor thread per program,
                so it is capped at ``256`` programs.  Larger ensembles must
                opt into ``max_batch_size`` (bounded pool, smaller merges),
                ``max_concurrent_programs`` (explicit pool size, ideal for
                cloud submission of one large merged job), or
                ``BatchMode.OFF``.

        Returns:
            ProgramEnsemble: Returns self for method chaining.

        Raises:
            RuntimeError: If an ensemble is already running, if no programs
                have been created, or if the ensemble exceeds 256 programs
                without an explicit batching strategy.

        Note:
            In non-blocking mode, call `join()` later to wait for completion and
            collect results.
        """
        if self._executor is not None:
            raise RuntimeError("An ensemble is already being run.")

        if len(self._programs) == 0:
            raise RuntimeError("No programs to run.")

        self._progress_bar = None
        self._live_display = None
        self._prep_task_id = None
        batching_enabled = batch_config.mode is BatchMode.MERGED
        self._hide_program_rows = (
            batching_enabled and len(self._programs) > _HIDE_PROGRAM_ROWS_THRESHOLD
        )
        _wants_progress = hasattr(self, "max_iterations") or getattr(
            self, "_show_progress", False
        )
        if _wants_progress:
            self._progress_bar, self._live_display = make_progress_display(
                is_jupyter=self._is_jupyter
            )
            if self._progress_bar is not None and self._hide_program_rows:
                # In hide mode, add a single "Submitting circuits" row up
                # front so the user sees activity during the prep window
                # (workers building circuits in Python, no batch flush
                # yet).  The coordinator emits a ``prep_advance`` message
                # per program on its first ``submit`` call, ticking this
                # row up until it reaches len(programs) — at which point
                # the merged backend submission fires.
                self._prep_task_id = self._progress_bar.add_task(
                    "",
                    job_name="Submitting circuits",
                    total=len(self._programs),
                    completed=0,
                    message="",
                    batch_color="",
                    row_kind="program",
                    program_key=None,
                    final_status=None,
                )

        # Validate that all program instances are unique to prevent thread-safety issues
        program_instances = list(self._programs.values())
        if len(set(program_instances)) != len(program_instances):
            raise RuntimeError(
                "Duplicate program instances detected in ensemble. "
                "QuantumProgram instances are stateful and NOT thread-safe. "
                "You must provide a unique instance for each program ID."
            )

        # In barrier mode the pool is sized to fit every program so the
        # merge is maximal.
        default_workers = (os.cpu_count() or 1) + 4
        if batching_enabled:
            if batch_config.max_concurrent_programs is not None:
                if batch_config.max_concurrent_programs == -1:
                    n_workers = len(self._programs)
                else:
                    n_workers = batch_config.max_concurrent_programs
                    if n_workers > _CONCURRENT_PROGRAMS_SOFT_CAP:
                        warn(
                            f"max_concurrent_programs={n_workers} spawns that many "
                            f"executor threads; if you meant to merge submissions, "
                            f"set max_batch_size on BatchConfig instead.",
                            UserWarning,
                            stacklevel=2,
                        )
            elif batch_config.max_batch_size is not None:
                n_workers = default_workers
            elif len(self._programs) <= _BARRIER_PROGRAM_LIMIT:
                n_workers = max(len(self._programs), default_workers)
            else:
                raise RuntimeError(
                    f"Ensemble has {len(self._programs)} programs, exceeding the "
                    f"wait-for-all barrier limit ({_BARRIER_PROGRAM_LIMIT}). Set "
                    f"BatchConfig(max_batch_size=N) for early-flush, "
                    f"BatchConfig(max_concurrent_programs=N) to bypass the cap, or "
                    f"BatchConfig(mode=BatchMode.OFF)."
                )
        else:
            n_workers = default_workers

        self._executor = ThreadPoolExecutor(max_workers=n_workers)
        self._cancellation_event = Event()
        self.futures.clear()
        self._future_to_program = {}
        self._pb_task_map = {}
        # Guards ``_pb_task_map`` mutations and the snapshot taken when
        # iterating ``progress_bar._tasks`` for batch coloring.  Any
        # cross-task field lookup that walks the Progress's task dict
        # must take this lock around the snapshot.
        self._pb_lock = Lock()
        self._listener_thread = None

        # Setup → registration → executor.submit happen sequentially on the
        # main thread.  If any of them raises after partial work is done
        # (e.g. some programs registered with the coordinator but the
        # remainder didn't reach _add_program_to_executor), the barrier
        # invariant ``len(_pending) >= len(_active_programs)`` would never
        # hold for survivors and they'd hang forever.  Tear everything
        # down on failure so the caller can retry cleanly.
        try:
            # Set up batch coordinator to merge circuit submissions.
            if batching_enabled:
                progress_queue = getattr(self, "_queue", None)
                self._coordinator = _BatchCoordinator(
                    self.backend,
                    progress_queue=progress_queue,
                    batch_config=batch_config,
                    n_workers=n_workers,
                    cancellation_event=self._cancellation_event,
                )
                self._program_key_map = {}
                for idx, (prog_id, program) in enumerate(self._programs.items()):
                    program_key = str(idx)
                    self._program_key_map[program] = program_key
                    self._coordinator.register_program(program_key)
                    program.backend = _ProxyBackend(
                        self.backend, self._coordinator, program_key
                    )

            if self._progress_bar is not None and self._live_display is not None:
                self._live_display.start()

                # Co-allocated with ``_progress_bar`` in the setup path, so a
                # progress-active branch implies ``_done_event`` is live.
                assert self._done_event is not None
                queue_obj = self._queue
                progress_bar = self._progress_bar
                pb_task_map = self._pb_task_map
                done_event = self._done_event
                pb_lock = self._pb_lock
                hide_program_rows = self._hide_program_rows
                prep_task_id = self._prep_task_id

                def _listener_runner():
                    # Spawn-side guard: if ``queue_listener`` dies for any
                    # reason — including signature errors before its body
                    # runs — drain the queue so any ``queue.join()`` caller
                    # doesn't hang waiting for ``task_done()`` calls that
                    # will never come.
                    try:
                        queue_listener(
                            queue_obj,
                            progress_bar,
                            pb_task_map,
                            done_event,
                            pb_lock,
                            hide_program_rows=hide_program_rows,
                            prep_task_id=prep_task_id,
                        )
                    except BaseException:
                        _drain_queue_quietly(queue_obj)
                        raise

                self._listener_thread = Thread(target=_listener_runner, daemon=True)
                self._listener_thread.start()

            for program in self._programs.values():
                future = self._add_program_to_executor(program)
                self.futures.append(future)
                self._future_to_program[future] = program

        except BaseException:
            # Tear down: shut down coordinator (clears state under lock),
            # signal/join the listener if it was started, stop live display,
            # and shut down the executor.  Any half-submitted futures will
            # resolve with ExecutionCancelledError via coordinator.cancel().
            if self._coordinator is not None:
                self._coordinator.shutdown()
                self._coordinator = None
            self._program_key_map.clear()

            if self._done_event is not None:
                self._done_event.set()
            if self._listener_thread is not None:
                self._listener_thread.join(timeout=5)
                self._listener_thread = None

            if self._live_display is not None:
                try:
                    self._live_display.stop()
                except Exception:
                    pass
                self._live_display = None
                self._progress_bar = None

            if self._executor is not None:
                self._executor.shutdown(wait=False)
                self._executor = None
            self.futures.clear()
            self._future_to_program = {}
            raise

        if not blocking:
            # Arm safety net
            atexit.register(self._atexit_cleanup_hook)
        else:
            self.join()

        return self

    def check_all_done(self) -> bool:
        """
        Check if all programs in the ensemble have completed execution.

        Returns:
            bool: True if all programs are finished (successfully or with errors),
                False if any are still running.
        """
        if not self.futures:
            warn(
                "check_all_done called with no active futures — run() has "
                "not been invoked (or the ensemble has been reset).",
                UserWarning,
                stacklevel=2,
            )
        return all(future.done() for future in self.futures)

    def _collect_completed_results(self, completed_futures: list):
        """Collect completed program instances from futures.

        Args:
            completed_futures: List to append program instances to.
        """
        for future in self.futures:
            if future.done() and not future.cancelled():
                try:
                    completed_futures.append(future.result())
                except Exception:
                    pass  # Skip failed futures

    def _wait_for_listener_drain(self, timeout: float = 30.0) -> None:
        """Wait for the listener to drain the queue, with a watchdog.

        ``queue.join()`` blocks until ``task_done()`` count matches
        ``put()`` count.  This helper polls for drain progress and bails
        out on either of two failure modes so ``ensemble.join()`` never
        hangs:

        - **Listener dead**: the loop returns immediately with a warning.
        - **Listener alive but stuck**: after ``timeout`` seconds with
          no drain progress the loop returns with a warning.  Display
          will be incomplete but the program exits.
        """
        if self._listener_thread is None:
            return
        deadline = time.monotonic() + timeout
        while self._queue.unfinished_tasks > 0:
            if not self._listener_thread.is_alive():
                warn(
                    "Progress listener thread died before draining the queue; "
                    "some terminal-status updates may be missing from the "
                    "displayed progress.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return
            if time.monotonic() >= deadline:
                warn(
                    f"Progress listener did not drain within {timeout:.0f}s; "
                    "some terminal-status updates may be missing.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return
            time.sleep(0.05)

    def _emit_progress_message(
        self,
        prog_id: str | None,
        *,
        final_status: TerminalStatus | None = None,
        message: str | None = None,
    ) -> None:
        """Put a synthetic progress message on the listener queue.

        Used by failure / cancellation paths so every progress mutation
        flows through the same single-writer queue → listener pipeline,
        keeping ordering coherent with the worker-emitted messages.

        No-op when there is no progress display: without a listener
        nothing would consume the message.
        """
        if prog_id is None or self._queue is None or self._progress_bar is None:
            return
        payload: dict[str, Any] = {"job_id": prog_id, "progress": 0}
        if final_status is not None:
            payload["final_status"] = final_status
        if message is not None:
            payload["message"] = message
        self._queue.put(payload)

    def _stop_remaining_programs(
        self,
        *,
        pending_status: TerminalStatus,
        pending_message: str,
        running_status: TerminalStatus,
        running_message: str,
        failed_status: TerminalStatus = TerminalStatus.FAILED,
        failed_message: str = "Job failed",
    ) -> None:
        """Signal all remaining programs to stop and wait for them to finish.

        Shared mechanical core used by both the cancellation path
        (``KeyboardInterrupt``) and the failure path (program exception).

        Args:
            pending_status: Progress bar ``final_status`` for futures that
                were successfully cancelled before they started.
            pending_message: Progress bar ``message`` for those futures.
            running_status: Progress bar ``final_status`` for futures that
                were already running and had to be waited on.
            running_message: Progress bar ``message`` for those futures.
            failed_status: Progress bar ``final_status`` for futures that
                already finished with an exception (e.g. batch coordinator
                failed all waiting programs).
            failed_message: Progress bar ``message`` for those futures.
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
                # Mark already-failed futures so their progress bars don't
                # freeze.  Futures that completed successfully are fine.
                if not future.cancelled():
                    try:
                        exc = future.exception()
                    except Exception:
                        exc = True  # defensive; treat as failed
                    if exc is not None:
                        self._emit_progress_message(
                            program.program_id,
                            final_status=failed_status,
                            message=failed_message,
                        )
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
                self._emit_progress_message(
                    program.program_id,
                    message="Finishing... ⏳",
                )

        # Immediately mark successfully cancelled tasks
        for program in successfully_cancelled:
            self._emit_progress_message(
                program.program_id,
                final_status=pending_status,
                message=pending_message,
            )

        # Wait for running tasks to finish
        for future in as_completed(unstoppable_futures):
            program = self._future_to_program[future]
            self._emit_progress_message(
                program.program_id,
                final_status=running_status,
                message=running_message,
            )

    def _handle_cancellation(self):
        """Handle cancellation gracefully with accurate progress feedback.

        With the batch coordinator active, cancellation works as follows:
        1. ``coordinator.cancel()`` sets the cancelled flag, cancels any
           in-flight backend jobs, and resolves pending futures with
           ``ExecutionCancelledError``.
        2. The program threads see the ``ExecutionCancelledError`` (or the
           ``_cancellation_event``) and exit.
        3. We wait for all still-running futures and mark them in the
           progress bar.

        Without the coordinator the legacy path applies: we try
        ``future.cancel()`` for pending tasks and ``cancel_unfinished_job()``
        for running ones.
        """
        self._stop_remaining_programs(
            pending_status=TerminalStatus.CANCELLED,
            pending_message="Cancelled by user",
            running_status=TerminalStatus.ABORTED,
            running_message="Stopped after current iteration",
        )

    def _handle_failure(self, failed_future: Future | None) -> None:
        """Handle a program failure by stopping remaining programs.

        Marks the failed program's progress bar as failed, then stops
        all other running programs using the same mechanism as
        cancellation.

        Args:
            failed_future: The future that raised the exception, or
                ``None`` if it could not be identified.
        """
        if failed_future is not None:
            failed_program = self._future_to_program.get(failed_future)
            if failed_program is not None:
                self._emit_progress_message(
                    failed_program.program_id,
                    final_status=TerminalStatus.FAILED,
                    message="Job failed",
                )

        self._stop_remaining_programs(
            pending_status=TerminalStatus.CANCELLED,
            pending_message="Cancelled due to failure",
            running_status=TerminalStatus.ABORTED,
            running_message="Aborted due to failure",
        )

    def join(self):
        """
        Wait for all programs in the ensemble to complete and collect results.

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
            # A task has failed. Identify the culprit and stop everything.
            failed_future = None
            # Count programs that finished successfully *before* we stop
            # anything — programs interrupted by the cancellation event
            # should not count as "completed".
            n_already_done = 0
            for f in self.futures:
                if f.done() and not f.cancelled():
                    try:
                        exc = f.exception()
                    except Exception:
                        exc = True
                    if exc is not None:
                        if failed_future is None:
                            failed_future = f
                    else:
                        n_already_done += 1

            # Show a condensed error in a Rich panel so the user gets
            # immediate feedback.  The full traceback is preserved in the
            # chained RuntimeError that propagates afterwards.
            failed_program_label = ""
            if failed_future is not None:
                fp = self._future_to_program.get(failed_future)
                if fp is not None:
                    failed_program_label = f" (Program {fp.program_id})"

            console = (
                self._progress_bar.console
                if self._progress_bar is not None
                else Console(stderr=True)
            )
            console.print(
                Panel(
                    f"[bold]{type(e).__name__}[/bold]: {e}",
                    title=f"[bold red]Program Failure{failed_program_label}[/bold red]",
                    subtitle="[dim]Full traceback follows below[/dim]",
                    border_style="red",
                )
            )

            self._handle_failure(failed_future)

            # Re-collect all completed results from scratch to avoid duplicates
            # from the as_completed loop above.
            completed_futures.clear()
            self._collect_completed_results(completed_futures)

            n_total = len(self._programs)
            raise RuntimeError(
                f"Ensemble execution failed: {n_already_done}/{n_total} programs "
                f"completed before failure."
            ) from e

        finally:
            # Aggregate results from completed program instances.
            # run() returns self, so completed_futures contains programs.
            if completed_futures:
                self._total_circuit_count += sum(
                    p._total_circuit_count for p in completed_futures
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
                        p._total_run_time for p in completed_futures
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

            if (
                self._progress_bar is not None
                and self._done_event is not None
                and self._listener_thread is not None
                and self._live_display is not None
            ):
                self._wait_for_listener_drain()
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
            if not program.has_results():
                raise RuntimeError(
                    "Some/All programs have no results. Did you call run()?"
                )

    @abstractmethod
    def aggregate_results(self) -> Any:
        """
        Aggregate results from all programs in the ensemble after execution.

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
            beam_width: Beam search width. Internally bumped
                to at least ``n`` so the beam retains enough candidates.
            n_partition_candidates: Candidates per partition.
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

    initial_list = list(initial_solution)
    beam: list[tuple[float, list[int]]] = [(evaluate_fn(initial_list), initial_list)]

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
