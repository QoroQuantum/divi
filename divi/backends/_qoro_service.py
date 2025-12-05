# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import base64
import gzip
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, fields, replace
from enum import Enum
from http import HTTPStatus

import requests
from dotenv import dotenv_values
from requests.adapters import HTTPAdapter, Retry
from rich.console import Console

from divi.backends import CircuitRunner
from divi.backends._execution_result import ExecutionResult
from divi.backends._qpu_system import (
    QPUSystem,
    get_qpu_system,
    parse_qpu_systems,
    update_qpu_systems_cache,
)
from divi.backends._results_processing import _decode_qh1_b64
from divi.circuits import is_valid_qasm, validate_qasm

API_URL = "https://app.qoroquantum.net/api"
_MAX_PAYLOAD_SIZE_MB = 0.95

session = requests.Session()
retry_configuration = Retry(
    total=5,
    backoff_factor=0.1,
    status_forcelist=[502],
    allowed_methods=["GET", "POST", "DELETE"],
)

session.mount("http://", HTTPAdapter(max_retries=retry_configuration))
session.mount("https://", HTTPAdapter(max_retries=retry_configuration))

logger = logging.getLogger(__name__)


def _raise_with_details(resp: requests.Response):
    try:
        data = resp.json()
        body = json.dumps(data, ensure_ascii=False)
    except ValueError:
        body = resp.text
    msg = f"{resp.status_code} {resp.reason}: {body}"
    raise requests.HTTPError(msg, response=resp)


class JobStatus(Enum):
    """Status of a job on the Qoro Service."""

    PENDING = "PENDING"
    """Job is queued and waiting to be processed."""

    RUNNING = "RUNNING"
    """Job is currently being executed."""

    COMPLETED = "COMPLETED"
    """Job has finished successfully."""

    FAILED = "FAILED"
    """Job execution encountered an error."""

    CANCELLED = "CANCELLED"
    """Job was cancelled before completion."""


class JobType(Enum):
    """Type of job to execute on the Qoro Service."""

    EXECUTE = "EXECUTE"
    """Execute circuits on real quantum hardware (sampling mode only)."""

    SIMULATE = "SIMULATE"
    """Simulate circuits using cloud-based simulation services (sampling mode)."""

    EXPECTATION = "EXPECTATION"
    """Compute expectation values for Hamiltonian operators (simulation only)."""

    CIRCUIT_CUT = "CIRCUIT_CUT"
    """Automatically decompose large circuits that wouldn't fit on a QPU."""


@dataclass(frozen=True)
class JobConfig:
    """Configuration for a Qoro Service job."""

    shots: int | None = None
    """Number of shots for the job."""

    qpu_system: QPUSystem | str | None = None
    """The QPU system to use, can be a string or a QPUSystem object."""

    use_circuit_packing: bool | None = None
    """Whether to use circuit packing optimization."""

    tag: str = "default"
    """Tag to associate with the job for identification."""

    force_sampling: bool = False
    """Whether to force sampling instead of expectation value measurements."""

    def override(self, other: "JobConfig") -> "JobConfig":
        """Creates a new config by overriding attributes with non-None values.

        This method ensures immutability by always returning a new `JobConfig` object
        and leaving the original instance unmodified.

        Args:
            other: Another JobConfig instance to take values from. Only non-None
                   attributes from this instance will be used for the override.

        Returns:
            A new JobConfig instance with the merged configurations.
        """
        current_attrs = {f.name: getattr(self, f.name) for f in fields(self)}

        for f in fields(other):
            other_value = getattr(other, f.name)
            if other_value is not None:
                current_attrs[f.name] = other_value

        return JobConfig(**current_attrs)

    def __post_init__(self):
        """Sanitizes and validates the configuration."""
        if self.shots is not None and self.shots <= 0:
            raise ValueError(f"Shots must be a positive integer. Got {self.shots}.")

        if isinstance(self.qpu_system, str):
            # Defer resolution - will be resolved in QoroService.__init__() after fetch_qpu_systems()
            # This allows JobConfig to be created before QoroService exists
            pass
        elif self.qpu_system is not None and not isinstance(self.qpu_system, QPUSystem):
            raise TypeError(
                f"Expected a QPUSystem instance or str, got {type(self.qpu_system)}"
            )

        if self.use_circuit_packing is not None and not isinstance(
            self.use_circuit_packing, bool
        ):
            raise TypeError(f"Expected a bool, got {type(self.use_circuit_packing)}")


class MaxRetriesReachedError(Exception):
    """Exception raised when the maximum number of retries is reached."""

    def __init__(self, job_id, retries):
        self.job_id = job_id
        self.retries = retries
        self.message = (
            f"Maximum retries reached: {retries} retries attempted for job {job_id}"
        )
        super().__init__(self.message)


_DEFAULT_QPU_SYSTEM = QPUSystem(name="qoro_maestro", supports_expval=True)

_DEFAULT_JOB_CONFIG = JobConfig(
    shots=1000, qpu_system=_DEFAULT_QPU_SYSTEM, use_circuit_packing=False
)


class QoroService(CircuitRunner):
    """A client for interacting with the Qoro Quantum Service API.

    This class provides methods to submit circuits, check job status,
    and retrieve results from the Qoro platform.
    """

    def __init__(
        self,
        auth_token: str | None = None,
        config: JobConfig | None = None,
        polling_interval: float = 3.0,
        max_retries: int = 5000,
    ):
        """Initializes the QoroService client.

        Args:
            auth_token (str | None, optional):
                The authentication token for the Qoro API. If not provided,
                it will be read from the QORO_API_KEY in a .env file.
            config (JobConfig | None, optional):
                A JobConfig object containing default job settings. If not
                provided, a default configuration will be created.
            polling_interval (float, optional):
                The interval in seconds for polling job status. Defaults to 3.0.
            max_retries (int, optional):
                The maximum number of retries for polling. Defaults to 5000.
        """

        # Set up auth_token first (needed for API calls like fetch_qpu_systems)
        if auth_token is None:
            try:
                auth_token = dotenv_values()["QORO_API_KEY"]
            except KeyError:
                raise ValueError("Qoro API key not provided nor found in a .env file.")

        self.auth_token = "Bearer " + auth_token
        self.polling_interval = polling_interval
        self.max_retries = max_retries

        # Fetch QPU systems (needs auth_token to be set)
        self.fetch_qpu_systems()

        # Set up config
        if config is None:
            config = _DEFAULT_JOB_CONFIG

        # Resolve string qpu_system names and validate that one is present.
        self.config = self._resolve_and_validate_qpu_system(config)

        super().__init__(shots=self.config.shots)

    @property
    def supports_expval(self) -> bool:
        """
        Whether the backend supports expectation value measurements.
        """
        return self.config.qpu_system.supports_expval and not self.config.force_sampling

    @property
    def is_async(self) -> bool:
        """
        Whether the backend executes circuits asynchronously.
        """
        return True

    def _resolve_and_validate_qpu_system(self, config: JobConfig) -> JobConfig:
        """Ensures the config has a valid QPUSystem object, resolving from string if needed."""
        if config.qpu_system is None:
            raise ValueError(
                "JobConfig must have a qpu_system. It cannot be None. "
                "Please provide a QPUSystem object or a valid system name string."
            )

        if isinstance(config.qpu_system, str):
            resolved_qpu = get_qpu_system(config.qpu_system)
            return replace(config, qpu_system=resolved_qpu)

        return config

    def _make_request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """
        Make an authenticated HTTP request to the Qoro API.

        This internal method centralizes all API communication, handling authentication
        headers and error responses consistently.

        Args:
            method (str): HTTP method to use (e.g., 'get', 'post', 'delete').
            endpoint (str): API endpoint path (without base URL).
            **kwargs: Additional arguments to pass to requests.request(), such as
                'json', 'timeout', 'params', etc.

        Returns:
            requests.Response: The HTTP response object from the API.

        Raises:
            requests.exceptions.HTTPError: If the response status code is 400 or above.
        """
        url = f"{API_URL}/{endpoint}"

        headers = {"Authorization": self.auth_token}

        if method.upper() in ["POST", "PUT", "PATCH"]:
            headers["Content-Type"] = "application/json"

        # Allow overriding default headers
        if "headers" in kwargs:
            headers.update(kwargs.pop("headers"))

        response = session.request(method, url, headers=headers, **kwargs)

        # Raise with comprehensive error details if request failed
        if response.status_code >= 400:
            _raise_with_details(response)

        return response

    def _extract_job_id(self, execution_result: ExecutionResult) -> str:
        job_id = execution_result.job_id
        if job_id is None:
            raise ValueError(
                "ExecutionResult must have a job_id. "
                "This ExecutionResult appears to be from a synchronous backend."
            )
        return job_id

    def test_connection(self):
        """
        Test the connection to the Qoro API.

        Sends a simple GET request to verify that the API is reachable and
        the authentication token is valid.

        Returns:
            requests.Response: The response from the API ping endpoint.

        Raises:
            requests.exceptions.HTTPError: If the connection fails or authentication
                is invalid.
        """
        return self._make_request("get", "", timeout=10)

    def fetch_qpu_systems(self) -> list[QPUSystem]:
        """
        Get the list of available QPU systems from the Qoro API.

        Returns:
            List of QPUSystem objects.
        """
        response = self._make_request("get", "qpusystem/", timeout=10)
        systems = parse_qpu_systems(response.json())
        update_qpu_systems_cache(systems)
        return systems

    @staticmethod
    def _compress_data(value) -> bytes:
        return base64.b64encode(gzip.compress(value.encode("utf-8"))).decode("utf-8")

    def _split_circuits(self, circuits: dict[str, str]) -> list[dict[str, str]]:
        """
        Splits circuits into chunks by estimating payload size with a simplified,
        consistent overhead calculation.
        Assumes that BASE64 encoding produces ASCI characters, which are 1 byte each.
        """
        max_payload_bytes = _MAX_PAYLOAD_SIZE_MB * 1024 * 1024
        circuit_chunks = []
        current_chunk = {}

        # Start with size 2 for the opening and closing curly braces '{}'
        current_chunk_size_bytes = 2

        for key, value in circuits.items():
            compressed_value = self._compress_data(value)

            item_size_bytes = len(key) + len(compressed_value) + 6

            # If adding this item would exceed the limit, finalize the current chunk.
            # This check only runs if the chunk is not empty.
            if current_chunk and (
                current_chunk_size_bytes + item_size_bytes > max_payload_bytes
            ):
                circuit_chunks.append(current_chunk)

                # Start a new chunk
                current_chunk = {}
                current_chunk_size_bytes = 2

            # Add the new item to the current chunk and update its size
            current_chunk[key] = compressed_value
            current_chunk_size_bytes += item_size_bytes

        # Add the last remaining chunk if it's not empty
        if current_chunk:
            circuit_chunks.append(current_chunk)

        return circuit_chunks

    def submit_circuits(
        self,
        circuits: dict[str, str],
        ham_ops: str | None = None,
        job_type: JobType | None = None,
        override_config: JobConfig | None = None,
    ) -> ExecutionResult:
        """
        Submit quantum circuits to the Qoro API for execution.

        This method first initializes a job and then sends the circuits in
        one or more chunks, associating them all with a single job ID.

        Args:
            circuits (dict[str, str]):
                Dictionary mapping unique circuit IDs to QASM circuit strings.
            ham_ops (str | None, optional):
                String representing the Hamiltonian operators to measure, semicolon-separated.
                Each term is a combination of Pauli operators, e.g. "XYZ;XXZ;ZIZ".
                If None, no Hamiltonian operators will be measured.
            job_type (JobType | None, optional):
                Type of job to execute (e.g., SIMULATE, EXECUTE, EXPECTATION, CIRCUIT_CUT).
                If not provided, the job type will be determined from the service configuration.
            override_config (JobConfig | None, optional):
                Configuration object to override the service's default settings.
                If not provided, default values are used.

        Raises:
            ValueError: If more than one circuit is submitted for a CIRCUIT_CUT job,
                        or if any circuit is not valid QASM.
            requests.exceptions.HTTPError: If any API request fails.

        Returns:
            ExecutionResult: Contains job_id for asynchronous execution. Use the job_id
                to poll for results using backend.poll_job_status() and get_job_results().
        """
        # Create final job configuration by layering configurations:
        #    service defaults -> user overrides
        if override_config:
            config = self.config.override(override_config)
            job_config = self._resolve_and_validate_qpu_system(config)
        else:
            job_config = self.config

        # Handle Hamiltonian operators: validate compatibility and auto-infer job type
        if ham_ops is not None:
            # Validate that if job_type is explicitly set, it must be EXPECTATION
            if job_type is not None and job_type != JobType.EXPECTATION:
                raise ValueError(
                    "Hamiltonian operators are only supported for EXPECTATION job type."
                )
            # Auto-infer job type if not explicitly set
            if job_type is None:
                job_type = JobType.EXPECTATION

            # Validate observables format

            terms = ham_ops.split(";")
            if len(terms) == 0:
                raise ValueError(
                    "Hamiltonian operators must be non-empty semicolon-separated strings."
                )
            ham_ops_length = len(terms[0])
            if not all(len(term) == ham_ops_length for term in terms):
                raise ValueError("All Hamiltonian operators must have the same length.")
            # Validate that each term only contains I, X, Y, Z
            valid_paulis = {"I", "X", "Y", "Z"}
            if not all(all(c in valid_paulis for c in term) for term in terms):
                raise ValueError(
                    "Hamiltonian operators must contain only I, X, Y, Z characters."
                )

        if job_type is None:
            job_type = JobType.SIMULATE

        # Validate circuits
        if job_type == JobType.CIRCUIT_CUT and len(circuits) > 1:
            raise ValueError("Only one circuit allowed for circuit-cutting jobs.")

        for key, circuit in circuits.items():
            if not is_valid_qasm(circuit):
                # Get the actual error message for better error reporting
                try:
                    validate_qasm(circuit)
                except SyntaxError as e:
                    raise ValueError(f"Circuit '{key}' is not a valid QASM: {e}") from e

        # Initialize the job without circuits to get a job_id
        init_payload = {
            "tag": job_config.tag,
            "job_type": job_type.value,
            "qpu_system_name": (
                job_config.qpu_system.name if job_config.qpu_system else None
            ),
            "use_packing": job_config.use_circuit_packing or False,
        }

        init_response = self._make_request(
            "post", "job/init/", json=init_payload, timeout=100
        )
        if init_response.status_code not in [HTTPStatus.OK, HTTPStatus.CREATED]:
            _raise_with_details(init_response)
        job_id = init_response.json()["job_id"]

        # Split circuits and add them to the created job
        circuit_chunks = self._split_circuits(circuits)
        num_chunks = len(circuit_chunks)

        for i, chunk in enumerate(circuit_chunks):
            is_last_chunk = i == num_chunks - 1
            add_circuits_payload = {
                "circuits": chunk,
                "mode": "append",
                "finalized": "true" if is_last_chunk else "false",
            }

            # Include shots/ham_ops in add_circuits payload
            if ham_ops is not None:
                add_circuits_payload["observables"] = ham_ops
            else:
                add_circuits_payload["shots"] = job_config.shots

            add_circuits_response = self._make_request(
                "post",
                f"job/{job_id}/add_circuits/",
                json=add_circuits_payload,
                timeout=100,
            )
            if add_circuits_response.status_code != HTTPStatus.OK:
                _raise_with_details(add_circuits_response)

        return ExecutionResult(results=None, job_id=job_id)

    def delete_job(self, execution_result: ExecutionResult) -> requests.Response:
        """
        Delete a job from the Qoro Database.

        Args:
            execution_result: An ExecutionResult instance with a job_id to delete.
        Returns:
            requests.Response: The response from the API.
        Raises:
            ValueError: If the ExecutionResult does not have a job_id.
        """
        job_id = self._extract_job_id(execution_result)
        return self._make_request(
            "delete",
            f"job/{job_id}",
            timeout=50,
        )

    def cancel_job(self, execution_result: ExecutionResult) -> requests.Response:
        """
        Cancel a job on the Qoro Service.

        Args:
            execution_result: An ExecutionResult instance with a job_id to cancel.
        Returns:
            requests.Response: The response from the API. Use response.json() to get
                the cancellation details (status, job_id, circuits_cancelled).
        Raises:
            ValueError: If the ExecutionResult does not have a job_id.
            requests.exceptions.HTTPError: If the cancellation fails (e.g., 403 Forbidden,
                or 409 Conflict if job is not in a cancellable state).
        """
        job_id = self._extract_job_id(execution_result)
        return self._make_request(
            "post",
            f"job/{job_id}/cancel/",
            timeout=50,
        )

    def get_job_results(self, execution_result: ExecutionResult) -> ExecutionResult:
        """
        Get the results of a job from the Qoro Database.

        Args:
            execution_result: An ExecutionResult instance with a job_id to fetch results for.

        Returns:
            ExecutionResult: A new ExecutionResult instance with results populated.

        Raises:
            ValueError: If the ExecutionResult does not have a job_id.
            requests.exceptions.HTTPError: If the job results are not available
                (e.g., job is still running) or if the request fails.
        """
        job_id = self._extract_job_id(execution_result)

        try:
            response = self._make_request(
                "get",
                f"job/{job_id}/resultsV2/?limit=100&offset=0",
                timeout=100,
            )
        except requests.exceptions.HTTPError as e:
            # Provide a more specific error message for 400 Bad Request
            if e.response.status_code == HTTPStatus.BAD_REQUEST:
                raise requests.exceptions.HTTPError(
                    "400 Bad Request: Job results not available, likely job is still running"
                ) from e
            # Re-raise any other HTTP error
            raise e

        # If the request was successful, process the data
        data = response.json()

        for result in data["results"]:
            result["results"] = _decode_qh1_b64(result["results"])

        # Return a new ExecutionResult with results populated
        return execution_result.with_results(data["results"])

    def poll_job_status(
        self,
        execution_result: ExecutionResult,
        loop_until_complete: bool = False,
        on_complete: Callable[[requests.Response], None] | None = None,
        verbose: bool = True,
        progress_callback: Callable[[int, str], None] | None = None,
    ) -> JobStatus:
        """
        Get the status of a job and optionally execute a function on completion.

        Args:
            execution_result: An ExecutionResult instance with a job_id to check.
            loop_until_complete (bool): If True, polls until the job is complete or failed.
            on_complete (Callable, optional): A function to call with the final response
                object when the job finishes.
            verbose (bool, optional): If True, prints polling status to the logger.
            progress_callback (Callable, optional): A function for updating progress bars.
                Takes `(retry_count, status)`.

        Returns:
            JobStatus: The current job status.

        Raises:
            ValueError: If the ExecutionResult does not have a job_id.
        """
        job_id = self._extract_job_id(execution_result)

        polling_status = None

        # Decide once at the start which update function to use
        if progress_callback:
            update_fn = progress_callback
        elif verbose:
            # Use Rich's status for overwriting polling messages
            polling_status = Console(file=None).status("", spinner="aesthetic")
            polling_status.start()

            def update_polling_status(retry_count, job_status):
                status_msg = (
                    f"Job [cyan]{job_id.split('-')[0]}[/cyan] is {job_status}. "
                    f"Polling attempt {retry_count} / {self.max_retries}"
                )
                polling_status.update(status_msg)

            update_fn = update_polling_status
        else:
            update_fn = lambda _, __: None

        try:
            if not loop_until_complete:
                response = self._make_request(
                    "get", f"job/{job_id}/status/", timeout=200
                )
                return JobStatus(response.json()["status"])

            terminal_statuses = {
                JobStatus.COMPLETED,
                JobStatus.FAILED,
                JobStatus.CANCELLED,
            }

            for retry_count in range(1, self.max_retries + 1):
                response = self._make_request(
                    "get", f"job/{job_id}/status/", timeout=200
                )
                status = JobStatus(response.json()["status"])

                if status in terminal_statuses:
                    if on_complete:
                        on_complete(response)
                    return status

                update_fn(retry_count, status.value)
                time.sleep(self.polling_interval)

            raise MaxRetriesReachedError(job_id, self.max_retries)
        finally:
            if polling_status:
                polling_status.stop()
