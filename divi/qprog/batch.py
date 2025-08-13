# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import atexit
import traceback
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Event, Manager
from multiprocessing.synchronize import Event as EventClass
from queue import Empty, Queue
from threading import Lock, Thread
from warnings import warn

from rich.console import Console
from rich.progress import Progress, TaskID

from divi._pbar import make_progress_bar
from divi.interfaces import CircuitRunner
from divi.parallel_simulator import ParallelSimulator
from divi.qlogger import disable_logging
from divi.qprog.quantum_program import QuantumProgram


def queue_listener(
    queue: Queue,
    progress_bar: Progress,
    pb_task_map: dict[QuantumProgram, TaskID],
    done_event: EventClass,
    is_jupyter: bool,
    lock: Lock,
):
    while not done_event.is_set():
        try:
            msg = queue.get(timeout=0.1)
        except Empty:
            continue
        except Exception as e:
            progress_bar.console.log(f"[queue_listener] Unexpected exception: {e}")
            continue

        with lock:
            task_id = pb_task_map[msg["job_id"]]

        progress_bar.update(
            task_id,
            advance=msg["progress"],
            poll_attempt=msg.get("poll_attempt", 0),
            message=msg.get("message", ""),
            final_status=msg.get("final_status", ""),
            refresh=is_jupyter,
        )


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
        self.programs = {}

        self._total_circuit_count = 0
        self._total_run_time = 0.0

        self._is_local = isinstance(backend, ParallelSimulator)
        self._is_jupyter = Console().is_jupyter

        # Disable logging since we already have the bars to track progress
        disable_logging()

    @property
    def total_circuit_count(self):
        return self._total_circuit_count

    @property
    def total_run_time(self):
        return self._total_run_time

    @abstractmethod
    def create_programs(self):
        if len(self.programs) > 0:
            raise RuntimeError(
                "Some programs already exist. "
                "Clear the program dictionary before creating new ones by using batch.reset()."
            )

        self._manager = Manager()
        self._queue = self._manager.Queue()

        if hasattr(self, "max_iterations"):
            self._done_event = Event()

    def reset(self):
        self.programs.clear()

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

        # Shut down the manager process, which handles the queue cleanup.
        if hasattr(self, "_manager") and self._manager is not None:
            self._manager.shutdown()
            self._manager = None
            self._queue = None

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

    def add_program_to_executor(self, program):
        self.futures.append(self._executor.submit(self._task_fn, program))

        if self._progress_bar is not None:
            with self._pb_lock:
                self._pb_task_map[program.job_id] = self._progress_bar.add_task(
                    "",
                    job_name=f"Job {program.job_id}",
                    total=self.max_iterations,
                    completed=0,
                    poll_attempt=0,
                    message="",
                    final_status="",
                    mode=("simulation" if self._is_local else "network"),
                )

    def run(self, blocking: bool = False):
        if self._executor is not None:
            raise RuntimeError("A batch is already being run.")

        if len(self.programs) == 0:
            raise RuntimeError("No programs to run.")

        self._progress_bar = (
            make_progress_bar(
                max_retries=None if self._is_local else self.backend.max_retries,
                is_jupyter=self._is_jupyter,
            )
            if hasattr(self, "max_iterations")
            else None
        )

        self._executor = ProcessPoolExecutor()
        self.futures = []
        self._pb_task_map = {}
        self._pb_lock = Lock()

        if self._progress_bar is not None:
            self._progress_bar.start()
            self._listener_thread = Thread(
                target=queue_listener,
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

        for program in self.programs.values():
            self.add_program_to_executor(program)

        if not blocking:
            # Arm safety net
            atexit.register(self._atexit_cleanup_hook)

        if blocking:
            self.join()

        return self

    def check_all_done(self):
        return all(future.done() for future in self.futures)

    def join(self):
        if self._executor is None:
            return

        exceptions = []
        try:
            # Ensure all futures are completed and handle exceptions.
            for future in as_completed(self.futures):
                try:
                    future.result()  # Raises an exception if the task failed.
                except Exception as e:
                    exceptions.append(e)

        finally:
            self._executor.shutdown(wait=True, cancel_futures=False)
            self._executor = None

            if self._progress_bar is not None:
                self._done_event.set()
                self._listener_thread.join()

        if exceptions:
            for i, exc in enumerate(exceptions, 1):
                print(f"Task {i} failed with exception:")
                traceback.print_exception(type(exc), exc, exc.__traceback__)
            raise RuntimeError("One or more tasks failed. Check logs for details.")

        if self._progress_bar is not None:
            self._progress_bar.stop()

        self._total_circuit_count += sum(future.result()[0] for future in self.futures)
        self._total_run_time += sum(future.result()[1] for future in self.futures)
        self.futures.clear()

        # After successful cleanup, try to unregister the hook.
        # This will only succeed if it was a non-blocking run.
        try:
            atexit.unregister(self._atexit_cleanup_hook)
        except TypeError:
            # This is expected for blocking runs where the hook was never registered.
            pass

    @abstractmethod
    def aggregate_results(self):
        if len(self.programs) == 0:
            raise RuntimeError("No programs to aggregate. Run create_programs() first.")

        if self._executor is not None:
            self.join()

        if any(len(program.losses) == 0 for program in self.programs.values()):
            raise RuntimeError(
                "Some/All programs have empty losses. Did you call run()?"
            )
