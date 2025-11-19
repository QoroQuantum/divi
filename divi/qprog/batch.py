# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import atexit
import traceback
from abc import ABC, abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from queue import Empty, Queue
from threading import Event, Lock, Thread
from typing import Any
from warnings import warn

from rich.console import Console
from rich.progress import Progress, TaskID

from divi.backends import CircuitRunner, ParallelSimulator
from divi.qprog.quantum_program import QuantumProgram
from divi.reporting import disable_logging, make_progress_bar


def _queue_listener(
    queue: Queue,
    progress_bar: Progress,
    pb_task_map: dict[QuantumProgram, TaskID],
    done_event: Event,
    is_jupyter: bool,
    lock: Lock,
):
    while not done_event.is_set():
        try:
            msg: dict[str, Any] = queue.get(timeout=0.1)
        except Empty:
            continue
        except Exception as e:
            progress_bar.console.log(f"[queue_listener] Unexpected exception: {e}")
            continue

        with lock:
            task_id = pb_task_map[msg["job_id"]]

        # Prepare update arguments, starting with progress.
        update_args = {"advance": msg["progress"]}

        if "poll_attempt" in msg:
            update_args["poll_attempt"] = msg.get("poll_attempt", 0)
        if "max_retries" in msg:
            update_args["max_retries"] = msg.get("max_retries")
        if "service_job_id" in msg:
            update_args["service_job_id"] = msg.get("service_job_id")
        if "job_status" in msg:
            update_args["job_status"] = msg.get("job_status")
        if msg.get("message"):
            update_args["message"] = msg.get("message")
        if "final_status" in msg:
            update_args["final_status"] = msg.get("final_status", "")

        update_args["refresh"] = is_jupyter

        progress_bar.update(task_id, **update_args)
        queue.task_done()


def _default_task_function(program: QuantumProgram):
    return program.run()


class ProgramBatch(ABC):
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

        self._total_circuit_count = 0
        self._total_run_time = 0.0

        self._is_local = isinstance(backend, ParallelSimulator)
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

        if hasattr(self, "max_iterations"):
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

        # Stop the progress bar if it's still active
        if getattr(self, "_progress_bar", None) is not None:
            try:
                self._progress_bar.stop()
            except Exception:
                pass  # Already stopped or not running
            self._progress_bar = None
            self._pb_task_map.clear()

    def _atexit_cleanup_hook(self):
        # This hook is only registered for non-blocking runs.
        if self._executor is not None:
            warn(
                "A non-blocking ProgramBatch run was not explicitly closed with "
                "'join()'. The batch was cleaned up automatically on exit.",
                UserWarning,
            )
            self.reset()

    def _add_program_to_executor(self, program: QuantumProgram) -> Future:
        """
        Add a quantum program to the thread pool executor for execution.

        Sets up the program with cancellation support and progress tracking, then
        submits it for execution in a separate thread.

        Args:
            program (QuantumProgram): The quantum program to execute.

        Returns:
            Future: A Future object representing the program's execution.
        """
        if hasattr(program, "_set_cancellation_event"):
            program._set_cancellation_event(self._cancellation_event)

        if self._progress_bar is not None:
            with self._pb_lock:
                self._pb_task_map[program.job_id] = self._progress_bar.add_task(
                    "",
                    job_name=f"Job {program.job_id}",
                    total=self.max_iterations,
                    completed=0,
                    message="",
                    mode=("simulation" if self._is_local else "network"),
                )

        return self._executor.submit(self._task_fn, program)

    def run(self, blocking: bool = False):
        """
        Execute all programs in the batch.

        Starts all quantum programs in parallel using a thread pool. Can run in
        blocking or non-blocking mode.

        Args:
            blocking (bool, optional): If True, waits for all programs to complete
                before returning. If False, returns immediately and programs run in
                the background. Defaults to False.

        Returns:
            ProgramBatch: Returns self for method chaining.

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

        self._progress_bar = (
            make_progress_bar(is_jupyter=self._is_jupyter)
            if hasattr(self, "max_iterations")
            else None
        )

        self._executor = ThreadPoolExecutor()
        self._cancellation_event = Event()
        self.futures = []
        self._future_to_program = {}
        self._pb_task_map = {}
        self._pb_lock = Lock()

        if self._progress_bar is not None:
            self._progress_bar.start()
            self._listener_thread = Thread(
                target=_queue_listener,
                args=(
                    self._queue,
                    self._progress_bar,
                    self._pb_task_map,
                    self._done_event,
                    self._is_jupyter,
                    self._pb_lock,
                ),
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
        """
        Handles cancellation gracefully, providing accurate feedback by checking
        the result of future.cancel().
        """
        self._cancellation_event.set()

        successfully_cancelled = []
        unstoppable_futures = []

        # --- Phase 1: Attempt to cancel all non-finished tasks ---
        for future, program in self._future_to_program.items():
            if future.done():
                continue

            task_id = self._pb_task_map.get(program.job_id)
            if self._progress_bar and task_id is not None:
                cancel_result = future.cancel()
                if cancel_result:
                    # The task was pending and was successfully cancelled.
                    successfully_cancelled.append(program)
                else:
                    # The task is already running and cannot be stopped.
                    unstoppable_futures.append(future)
                    self._progress_bar.update(
                        task_id,
                        message="Finishing... ⏳",
                        refresh=self._is_jupyter,
                    )

        # --- Phase 2: Immediately mark the successfully cancelled tasks ---
        for program in successfully_cancelled:
            task_id = self._pb_task_map.get(program.job_id)
            if self._progress_bar and task_id is not None:
                self._progress_bar.update(
                    task_id,
                    final_status="Cancelled",
                    message="Cancelled by user",
                    refresh=self._is_jupyter,
                )

        # --- Phase 3: Wait for the unstoppable tasks to finish ---
        if unstoppable_futures:
            for future in as_completed(unstoppable_futures):
                program = self._future_to_program[future]
                task_id = self._pb_task_map.get(program.job_id)
                if self._progress_bar and task_id is not None:
                    self._progress_bar.update(
                        task_id,
                        final_status="Aborted",
                        message="Completed during cancellation",
                        refresh=self._is_jupyter,
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
                    "[bold yellow]Shutdown signal received, waiting for programs to finish current iteration...[/bold yellow]"
                )
                self._handle_cancellation()

            # Collect results from any futures that completed before/during cancellation
            self._collect_completed_results(completed_futures)

            return False

        except Exception as e:
            # A task has failed. Print the error and cancel the rest.
            print(f"A task failed with an exception. Cancelling remaining tasks...")
            traceback.print_exception(type(e), e, e.__traceback__)

            # Collect results from any futures that completed before the failure
            self._collect_completed_results(completed_futures)

            # Cancel all other futures that have not yet completed.
            for f in self.futures:
                f.cancel()

            # Re-raise a new error to indicate the batch failed.
            raise RuntimeError("Batch execution failed and was cancelled.") from e

        finally:
            # Aggregate results from completed futures
            if completed_futures:
                self._total_circuit_count += sum(
                    result[0] for result in completed_futures
                )
                self._total_run_time += sum(result[1] for result in completed_futures)
                self.futures.clear()

            self._executor.shutdown(wait=False)
            self._executor = None

            if self._progress_bar is not None:
                self._queue.join()
                self._done_event.set()
                self._listener_thread.join()
                self._progress_bar.stop()

        # After successful cleanup, try to unregister the hook.
        try:
            atexit.unregister(self._atexit_cleanup_hook)
        except TypeError:
            pass

    @abstractmethod
    def aggregate_results(self):
        if len(self._programs) == 0:
            raise RuntimeError("No programs to aggregate. Run create_programs() first.")

        if self._executor is not None:
            self.join()

        if any(len(program.losses_history) == 0 for program in self._programs.values()):
            raise RuntimeError(
                "Some/All programs have empty losses. Did you call run()?"
            )
