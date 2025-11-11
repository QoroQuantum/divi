# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import pickle
from abc import ABC, abstractmethod
from queue import Queue
from threading import Event
from typing import Any

from divi.backends import CircuitRunner, JobStatus
from divi.circuits import CircuitBundle


class QuantumProgram(ABC):
    """Abstract base class for quantum programs.

    This class defines the interface and provides common functionality for quantum algorithms.
    It handles circuit execution, result processing, and data persistence.

    Subclasses must implement:
        - run(): Execute the quantum algorithm
        - _generate_circuits(): Generate quantum circuits for execution
        - _post_process_results(): Process execution results

    Attributes:
        backend (CircuitRunner): The quantum circuit execution backend.
        _seed (int | None): Random seed for reproducible results.
        _progress_queue (Queue | None): Queue for progress reporting.
        _circuits (list): List of circuits to be executed.
        _curr_service_job_id: Current service job ID for QoroService backends.
    """

    def __init__(
        self,
        backend: CircuitRunner,
        seed: int | None = None,
        progress_queue: Queue | None = None,
        **kwargs,
    ):
        """Initialize the QuantumProgram.

        Args:
            backend (CircuitRunner): Quantum circuit execution backend.
            seed (int | None): Random seed for reproducible results. Defaults to None.
            progress_queue (Queue | None): Queue for progress reporting. Defaults to None.
            **kwargs: Additional keyword arguments for subclasses.
        """
        self.backend = backend
        self._seed = seed
        self._progress_queue = progress_queue
        self._total_circuit_count = 0
        self._total_run_time = 0.0
        self._curr_circuits = []
        self._curr_service_job_id = None

    @abstractmethod
    def run(self, data_file: str | None = None, **kwargs) -> tuple[int, float]:
        """Execute the quantum algorithm.

        Args:
            data_file (str | None): The file to store the data in. If None, no data is stored. Defaults to None.
            **kwargs: Additional keyword arguments for subclasses.

        Returns:
            tuple[int, float]: A tuple containing:
                - int: Total number of circuits executed
                - float: Total runtime in seconds
        """
        pass

    @abstractmethod
    def _generate_circuits(self, **kwargs) -> list[CircuitBundle]:
        """Generate quantum circuits for execution.

        This method should generate and return a list of CircuitBundle objects based on
        the current algorithm state. The circuits will be executed by the backend.

        Args:
            **kwargs: Additional keyword arguments for circuit generation.

        Returns:
            list[CircuitBundle]: List of CircuitBundle objects to be executed.
        """
        pass

    @abstractmethod
    def _post_process_results(self, results: dict, **kwargs) -> Any:
        """Process execution results.

        Args:
            results (dict): Raw results from circuit execution.

        Returns:
            Any: Processed results specific to the algorithm.
        """
        pass

    def _set_cancellation_event(self, event: Event):
        """Set a cancellation event for graceful program termination.

        This method is called by batch runners to provide a mechanism
        for stopping the optimization loop cleanly when requested.

        Args:
            event (Event): Threading Event object that signals cancellation when set.
        """
        self._cancellation_event = event

    @property
    def total_circuit_count(self) -> int:
        """Get the total number of circuits executed.

        Returns:
            int: Cumulative count of circuits submitted for execution.
        """
        return self._total_circuit_count

    @property
    def total_run_time(self) -> float:
        """Get the total runtime across all circuit executions.

        Returns:
            float: Cumulative execution time in seconds.
        """
        return self._total_run_time

    def _prepare_and_send_circuits(self, **kwargs):
        """Prepare circuits for execution and submit them to the backend.

        Returns:
            Backend output from circuit submission.
        """
        job_circuits = {}

        for bundle in self._curr_circuits:
            for executable in bundle.executables:
                job_circuits[executable.tag] = executable.qasm

        self._total_circuit_count += len(job_circuits)

        backend_output = self.backend.submit_circuits(job_circuits, **kwargs)

        if self.backend.is_async:
            self._curr_service_job_id = backend_output

        return backend_output

    def _track_runtime(self, response):
        """Extract and track runtime from a backend response.

        Args:
            response: Backend response containing runtime information.
                Can be a dict or a list of responses.
        """
        if isinstance(response, dict):
            self._total_run_time += float(response["run_time"])
        elif isinstance(response, list):
            self._total_run_time += sum(float(r.json()["run_time"]) for r in response)

    def _wait_for_qoro_job_completion(self, job_id: str) -> list[dict]:
        """Wait for a QoroService job to complete and return results.

        Args:
            job_id: The QoroService job identifier.

        Returns:
            list[dict]: The job results from the backend.

        Raises:
            Exception: If job fails or doesn't complete.
        """
        # Build the poll callback if reporter is available
        if hasattr(self, "reporter"):
            update_function = lambda n_polls, status: self.reporter.info(
                message="",
                poll_attempt=n_polls,
                max_retries=self.backend.max_retries,
                service_job_id=job_id,
                job_status=status,
            )
        else:
            update_function = None

        # Poll until complete
        status = self.backend.poll_job_status(
            job_id,
            loop_until_complete=True,
            on_complete=self._track_runtime,
            verbose=False,  # Disable the default logger in QoroService
            poll_callback=update_function,
        )

        if status != JobStatus.COMPLETED:
            raise Exception("Job has not completed yet, cannot post-process results")
        return self.backend.get_job_results(job_id)

    def _dispatch_circuits_and_process_results(
        self, data_file: str | None = None, **kwargs
    ):
        """Run an iteration of the program.

        The outputs are stored in the Program object.
        Optionally, the data can be stored in a file.

        Args:
            data_file (str | None): The file to store the data in. If None, no data is stored. Defaults to None.
            **kwargs: Additional keyword arguments for circuit submission and result processing.

        Returns:
            Any: Processed results from _post_process_results.
        """
        results = self._prepare_and_send_circuits(**kwargs)

        if self.backend.is_async:
            results = self._wait_for_qoro_job_completion(self._curr_service_job_id)

        results = {r["label"]: r["results"] for r in results}

        result = self._post_process_results(results, **kwargs)

        if data_file is not None:
            self.save_iteration(data_file)

        return result

    def save_iteration(self, data_file: str):
        """Save the current state of the quantum program to a file.

        Serializes the entire QuantumProgram instance including parameters,
        losses, and circuit history using pickle.

        Args:
            data_file (str): Path to the file where the program state will be saved.

        Note:
            The file is written in binary mode and can be restored using
            `import_iteration()`.
        """
        with open(data_file, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def import_iteration(data_file: str):
        """Load a previously saved quantum program state from a file.

        Deserializes a QuantumProgram instance that was saved using `save_iteration()`.

        Args:
            data_file (str): Path to the file containing the saved program state.

        Returns:
            QuantumProgram: The restored QuantumProgram instance with all its state,
                including parameters, losses, and circuit history.
        """
        with open(data_file, "rb") as f:
            return pickle.load(f)
