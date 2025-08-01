# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import traceback
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Event, Manager
from multiprocessing.synchronize import Event as EventClass
from queue import Queue
from threading import Thread

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
):
    while not done_event.is_set():
        try:
            msg = queue.get(timeout=0.1)
        except:
            continue

        progress_bar.update(
            pb_task_map[msg["job_id"]],
            advance=msg["progress"],
            poll_attempt=msg.get("poll_attempt", 0),
            message=msg.get("message", ""),
            final_status=msg.get("final_status", ""),
            refresh=is_jupyter,
        )


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
        self.programs = {}

        self._total_circuit_count = 0
        self._total_run_time = 0.0

        self._is_local = isinstance(backend, ParallelSimulator)
        self._is_jupyter = Console().is_jupyter
        self._progress_bar = make_progress_bar(
            max_retries=None if self._is_local else self.backend.max_retries,
            is_jupyter=self._is_jupyter,
        )

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
        self._manager = Manager()
        self._queue = self._manager.Queue()

        if hasattr(self, "max_iterations"):
            self._done_event = Event()

    def reset(self):
        self.programs.clear()
        self._executor = None
        self._queue = None

    def run(self):
        if self._executor is not None:
            raise RuntimeError("A batch is already being run.")

        if len(self.programs) == 0:
            raise RuntimeError("No programs to run.")

        self._populate_progress_bars()

        self._executor = ProcessPoolExecutor()

        # Only generate progress bars for iteration-based programs
        if hasattr(self, "max_iterations"):
            self._listener_thread = Thread(
                target=queue_listener,
                args=(
                    self._queue,
                    self._progress_bar,
                    self._pb_task_map,
                    self._done_event,
                    self._is_jupyter,
                ),
                daemon=True,
            )
            self._listener_thread.start()

        self.futures = [
            self._executor.submit(program.run) for program in self.programs.values()
        ]

    def check_all_done(self):
        return all(future.done() for future in self.futures)

    def wait_for_all(self):
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

            if hasattr(self, "max_iterations"):
                self._done_event.set()
                self._listener_thread.join()

        if exceptions:
            for i, exc in enumerate(exceptions, 1):
                print(f"Task {i} failed with exception:")
                traceback.print_exception(type(exc), exc, exc.__traceback__)
            raise RuntimeError("One or more tasks failed. Check logs for details.")

        if hasattr(self, "max_iterations"):
            self._progress_bar.stop()

        self._total_circuit_count += sum(future.result()[0] for future in self.futures)
        self._total_run_time += sum(future.result()[1] for future in self.futures)
        self.futures = []

    def _populate_progress_bars(self):
        if not hasattr(self, "max_iterations"):
            raise RuntimeError(
                "Can not generate progress bars for tasks without `max_iterations` attribute."
            )

        self._progress_bar.start()
        self._pb_task_map = {}
        for program in self.programs:
            pb_id = self._progress_bar.add_task(
                "",
                job_name=f"Job {program}",
                total=self.max_iterations,
                completed=0,
                poll_attempt=0,
                message="",
                final_status="",
                mode=("simulation" if self._is_local else "network"),
            )
            self._pb_task_map[program] = pb_id

    @abstractmethod
    def aggregate_results(self):
        raise NotImplementedError
