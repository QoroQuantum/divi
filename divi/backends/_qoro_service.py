# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import base64
import gzip
import itertools
import json
import logging
import os
import time
import warnings
from collections.abc import Callable, Mapping
from dataclasses import replace
from enum import Enum
from http import HTTPStatus

import requests
from dotenv import dotenv_values, find_dotenv
from qiskit import QuantumCircuit
from requests.adapters import HTTPAdapter, Retry
from rich.console import Console

from divi.backends import CircuitRunner
from divi.backends._config import ExecutionConfig, JobConfig
from divi.backends._execution_result import ExecutionResult
from divi.backends._results_processing import _decode_qh1_b64
from divi.backends._shot_allocation import (
    from_wire,
    restrict_to_chunk,
    to_wire,
    validate,
)
from divi.backends._systems import (
    QPUSystem,
    SimulatorCluster,
    get_qpu_system,
    get_simulator_cluster,
    parse_qpu_systems,
    parse_simulator_clusters,
    update_qpu_systems_cache,
    update_simulator_clusters_cache,
)
from divi.circuits import is_valid_qasm, validate_qasm
from divi.circuits._qasm_validation import _format_validation_error_with_context
from divi.hamiltonians import compress_ham_ops

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
    """Run circuits and return measurement count histograms."""

    EXPECTATION = "EXPECTATION"
    """Compute expectation values for Hamiltonian operators."""


class MaxRetriesReachedError(Exception):
    """Exception raised when the maximum number of retries is reached."""

    def __init__(self, job_id, retries):
        self.job_id = job_id
        self.retries = retries
        self.message = (
            f"Maximum retries reached: {retries} retries attempted for job {job_id}"
        )
        super().__init__(self.message)


_DEFAULT_SIMULATOR_CLUSTER = SimulatorCluster(name="qoro_maestro")

_DEFAULT_JOB_CONFIG = JobConfig(
    shots=1000, simulator_cluster=_DEFAULT_SIMULATOR_CLUSTER, use_circuit_packing=False
)


class QoroService(CircuitRunner):
    """A client for interacting with the Qoro Quantum Service API.

    This class provides methods to submit circuits, check job status,
    and retrieve results from the Qoro platform.
    """

    def __init__(
        self,
        auth_token: str | None = None,
        job_config: JobConfig | None = None,
        execution_config: ExecutionConfig | None = None,
        polling_interval: float = 3.0,
        max_retries: int = 5000,
        track_depth: bool = False,
    ):
        """Initializes the QoroService client.

        Args:
            auth_token (str | None, optional):
                The authentication token for the Qoro API. If not provided,
                it will be read from ``QORO_API_KEY`` in a ``.env`` file,
                falling back to the ``QORO_API_KEY`` environment variable.
            job_config (JobConfig | None, optional):
                A JobConfig object containing default job settings. If not
                provided, a default configuration will be created. If the
                job_config has neither ``simulator_cluster`` nor ``qpu_system``,
                it defaults to the ``qoro_maestro`` simulator cluster with a
                warning.
            execution_config (ExecutionConfig | None, optional):
                Default execution configuration for submitted jobs. When
                provided, every call to :meth:`submit_circuits` will use
                this config unless an explicit ``execution_config`` argument
                overrides it.
            polling_interval (float, optional):
                The interval in seconds for polling job status. Defaults to 3.0.
            max_retries (int, optional):
                The maximum number of retries for polling. Defaults to 5000.
            track_depth (bool, optional):
                If True, record circuit depth for each submitted batch.
                Access via :attr:`~divi.backends.CircuitRunner.depth_history` after execution. Defaults to False.
        """

        # Set up auth_token first (needed for API calls like fetch_simulator_clusters)
        if auth_token is None:
            try:
                env_path = find_dotenv(usecwd=True)
                auth_token = dotenv_values(env_path)["QORO_API_KEY"]
            except KeyError:
                auth_token = os.environ.get("QORO_API_KEY")
                if auth_token is None:
                    raise ValueError(
                        "Qoro API key not provided nor found in a .env file "
                        "or QORO_API_KEY environment variable."
                    )

        self.auth_token = "Bearer " + auth_token
        self.polling_interval = polling_interval
        self.max_retries = max_retries

        # Fetch available systems (needs auth_token to be set)
        self.fetch_qpu_systems()
        self.fetch_simulator_clusters()

        # Set up job config
        if job_config is None:
            job_config = _DEFAULT_JOB_CONFIG

        self.job_config = job_config

        self.execution_config = execution_config

        super().__init__(shots=self.job_config.shots, track_depth=track_depth)

    @property
    def supports_expval(self) -> bool:
        """
        Whether the backend supports expectation value measurements.
        """
        target = self.job_config.simulator_cluster or self.job_config.qpu_system
        return target.supports_expval and not self.job_config.force_sampling

    @property
    def job_config(self) -> JobConfig:
        """The service's default job configuration."""
        return self._job_config

    @job_config.setter
    def job_config(self, value: JobConfig) -> None:
        self._job_config = self._resolve_and_validate_target(value)

    @property
    def execution_config(self) -> ExecutionConfig | None:
        """The service's default execution configuration."""
        return self._execution_config

    @execution_config.setter
    def execution_config(self, value: ExecutionConfig | None) -> None:
        self._execution_config = value

    @property
    def is_async(self) -> bool:
        """
        Whether the backend executes circuits asynchronously.
        """
        return True

    def _resolve_and_validate_target(self, config: JobConfig) -> JobConfig:
        """Ensures the config has a valid target, resolving strings if needed.

        If neither ``simulator_cluster`` nor ``qpu_system`` is set, defaults to
        the ``qoro_maestro`` simulator cluster with a warning.
        """
        if config.simulator_cluster is None and config.qpu_system is None:
            warnings.warn(
                "No simulator_cluster or qpu_system specified in JobConfig. "
                f"Defaulting to simulator cluster '{_DEFAULT_SIMULATOR_CLUSTER.name}'.",
                stacklevel=2,
            )
            return replace(config, simulator_cluster=_DEFAULT_SIMULATOR_CLUSTER)

        if isinstance(config.simulator_cluster, str):
            resolved = get_simulator_cluster(config.simulator_cluster)
            return replace(config, simulator_cluster=resolved)

        if isinstance(config.qpu_system, str):
            resolved = get_qpu_system(config.qpu_system)
            return replace(config, qpu_system=resolved)

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

    def fetch_simulator_clusters(self) -> list[SimulatorCluster]:
        """
        Get the list of available simulator clusters from the Qoro API.

        Returns:
            List of SimulatorCluster objects.
        """
        response = self._make_request("get", "simulatorcluster/", timeout=10)
        clusters = parse_simulator_clusters(response.json())
        update_simulator_clusters_cache(clusters)
        return clusters

    def get_credit_balance(self) -> dict:
        """
        Get the current credit balance for the authenticated user.

        Returns:
            dict: A dictionary containing the credit account information::

                {
                    "balance": "500.00",
                    "total_used": "0",
                    "created_at": "2026-01-01T00:00:00Z",
                    "updated_at": "2026-01-01T00:00:00Z"
                }

        Raises:
            requests.exceptions.HTTPError: If the request fails (e.g., 401
                if the token is deactivated).
        """
        response = self._make_request("get", "credits/", timeout=10)
        return response.json()

    def get_credit_transactions(self, page: int = 1, page_size: int = 20) -> dict:
        """
        Get paginated credit transaction history for the authenticated user.

        Args:
            page (int, optional): Page number to retrieve. Defaults to 1.
            page_size (int, optional): Number of transactions per page.
                Defaults to 20. Maximum is 100.

        Returns:
            dict: A paginated response containing transaction records::

                {
                    "count": 1,
                    "total_pages": 1,
                    "next": null,
                    "previous": null,
                    "results": [
                        {
                            "id": 1,
                            "amount": "500.00",
                            "balance_after": "500.00",
                            "transaction_type": "PURCHASE",
                            "description": "...",
                            "job_id": null,
                            "created_at": "2026-01-01T00:00:00Z"
                        }
                    ]
                }

        Raises:
            requests.exceptions.HTTPError: If the request fails (e.g., 401
                if the token is deactivated).
        """
        response = self._make_request(
            "get",
            "credits/transactions/",
            params={"page": page, "page_size": page_size},
            timeout=10,
        )
        return response.json()

    @staticmethod
    def _compress_data(value) -> bytes:
        return base64.b64encode(gzip.compress(value.encode("utf-8"))).decode("utf-8")

    def _split_circuits(self, circuits: Mapping[str, str]) -> list[dict[str, str]]:
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
        circuits: Mapping[str, str],
        ham_ops: str | None = None,
        circuit_ham_map: list[list[int]] | None = None,
        shot_groups: list[list[int]] | None = None,
        job_type: JobType | None = None,
        override_execution_config: ExecutionConfig | None = None,
        override_job_config: JobConfig | None = None,
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
                Multiple groups can be pipe-delimited (e.g. "XYZ;XXZ|ZI;IZ") when
                ``circuit_ham_map`` is provided to assign each group to a slice of
                circuits. If None, no Hamiltonian operators will be measured.
            circuit_ham_map (list[list[int]] | None, optional):
                Maps each ``|``-delimited group in ``ham_ops`` to a ``[start, end)``
                slice of the ordered circuit list.  Must have the same length as
                ``ham_ops.split("|")``.  When None, a single ``ham_ops`` group is
                applied to all circuits.
            shot_groups (list[list[int]] | None, optional):
                Per-circuit shot allocation as ``[start, end, shots]`` triples
                covering the iteration order of ``circuits``. Mutually exclusive
                with the service-level ``shots`` field. When provided, ranges
                spanning multiple internal chunks are re-indexed automatically.
            job_type (JobType | None, optional):
                Type of job to execute (EXECUTE or EXPECTATION).
                If not provided, defaults to EXECUTE.
            override_execution_config (ExecutionConfig | None, optional):
                Execution configuration override for this submission. When
                provided, its non-None fields override the service-level
                ``execution_config`` set in the constructor. When omitted, the
                service-level default is used (if any). The merged config is
                sent inline to ``job/init`` as ``execution_configuration``.
            override_job_config (JobConfig | None, optional):
                Configuration object to override the service's default settings.
                If not provided, default values are used.

        Raises:
            ValueError: If any circuit is not valid QASM.
            requests.exceptions.HTTPError: If any API request fails.

        Returns:
            ExecutionResult: Contains job_id for asynchronous execution. Use the job_id
                to poll for results using backend.poll_job_status() and get_job_results().
        """
        # Create final job configuration by layering configurations:
        #    service defaults -> user overrides
        if override_job_config:
            config = self.job_config.override(override_job_config)
            job_config = self._resolve_and_validate_target(config)
        else:
            job_config = self.job_config

        if ham_ops is not None and shot_groups is not None:
            raise ValueError(
                "shot_groups is incompatible with ham_ops: EXPECTATION jobs "
                "compute expectation values analytically on the backend and "
                "ignore shot counts. Pass exactly one."
            )

        shot_ranges = None
        if shot_groups is not None:
            shot_ranges = from_wire(shot_groups)
            validate(shot_ranges, len(circuits))

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

            # Validate observables format (each |-delimited group independently)
            valid_paulis = {"I", "X", "Y", "Z"}
            ham_groups = ham_ops.split("|")
            for group in ham_groups:
                terms = group.split(";")
                if len(terms) == 0:
                    raise ValueError(
                        "Hamiltonian operators must be non-empty semicolon-separated strings."
                    )
                ham_ops_length = len(terms[0])
                if not all(len(term) == ham_ops_length for term in terms):
                    raise ValueError(
                        "All Hamiltonian operators must have the same length."
                    )
                if not all(all(c in valid_paulis for c in term) for term in terms):
                    raise ValueError(
                        "Hamiltonian operators must contain only I, X, Y, Z characters."
                    )

            # Validate circuit_ham_map consistency
            if circuit_ham_map is not None:
                if len(circuit_ham_map) != len(ham_groups):
                    raise ValueError(
                        f"circuit_ham_map length ({len(circuit_ham_map)}) must match "
                        f"number of ham_ops groups ({len(ham_groups)})."
                    )

        if job_type is None:
            job_type = JobType.EXECUTE

        # Validate circuits
        for key, circuit in circuits.items():
            if not is_valid_qasm(circuit):
                # Get the actual error message for better error reporting
                try:
                    validate_qasm(circuit)
                except SyntaxError as e:
                    msg = _format_validation_error_with_context(circuit, e)
                    raise ValueError(
                        f"Circuit '{key}' is not a valid QASM: {msg}"
                    ) from e

        # Track circuit depth if enabled
        if self.track_depth:
            self._depth_history.append(
                [
                    QuantumCircuit.from_qasm_str(qasm).depth()
                    for qasm in circuits.values()
                ]
            )

        # Resolve execution config: service default -> explicit override
        if override_execution_config is not None and self.execution_config is not None:
            execution_config = self.execution_config.override(override_execution_config)
        elif override_execution_config is not None:
            execution_config = override_execution_config
        else:
            execution_config = self.execution_config

        # Initialize the job without circuits to get a job_id
        init_payload = {
            "tag": job_config.tag,
            "job_type": job_type.value,
            "use_packing": job_config.use_circuit_packing or False,
        }
        if job_config.simulator_cluster:
            init_payload["simulator_cluster"] = job_config.simulator_cluster.name
        elif job_config.qpu_system:
            init_payload["qpu_system_name"] = job_config.qpu_system.name
        if execution_config is not None:
            init_payload["execution_configuration"] = execution_config.to_payload()

        init_response = self._make_request(
            "post", "job/init/", json=init_payload, timeout=100
        )
        if init_response.status_code not in [HTTPStatus.OK, HTTPStatus.CREATED]:
            _raise_with_details(init_response)
        job_id = init_response.json()["job_id"]

        # Split circuits and add them to the created job
        circuit_chunks = self._split_circuits(circuits)
        num_chunks = len(circuit_chunks)
        compressed_ham_ops = compress_ham_ops(ham_ops) if ham_ops is not None else None

        # Per-chunk starting offset into the global circuit list, used to
        # re-index ``shot_groups`` when chunking. Built up-front so the loop
        # body has no manual accumulator to maintain.
        chunk_offsets = list(
            itertools.accumulate((len(c) for c in circuit_chunks), initial=0)
        )
        for i, (chunk, chunk_offset) in enumerate(zip(circuit_chunks, chunk_offsets)):
            is_last_chunk = i == num_chunks - 1
            add_circuits_payload = {
                "circuits": chunk,
                "mode": "append",
                "finalized": "true" if is_last_chunk else "false",
            }

            # Include shots/ham_ops in add_circuits payload
            if compressed_ham_ops is not None:
                add_circuits_payload["observables"] = compressed_ham_ops
                if circuit_ham_map is not None:
                    add_circuits_payload["circuit_ham_map"] = circuit_ham_map
            elif shot_ranges is not None:
                add_circuits_payload["shot_groups"] = to_wire(
                    restrict_to_chunk(shot_ranges, chunk_offset, len(chunk))
                )
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

    def set_execution_config(
        self,
        execution_result: ExecutionResult,
        config: ExecutionConfig,
    ) -> dict:
        """Set or overwrite the execution configuration for a job.

        The job must be in ``PENDING`` status. Re-calling this method
        overwrites any previously set configuration.

        Args:
            execution_result: An ExecutionResult instance whose ``job_id``
                identifies the target job.
            config: The execution configuration to attach.

        Returns:
            dict: The API response containing ``status``, ``job_id`` and
                ``execution_configuration``.

        Raises:
            ValueError: If the ExecutionResult does not have a job_id.
            requests.exceptions.HTTPError:
                - 400: Validation errors (unknown ``api_meta`` keys, wrong
                  types, payload too large).
                - 403: ``bond_dimension`` exceeds the user's tier cap.
                - 409: Job is not in ``PENDING`` status.
        """
        job_id = self._extract_job_id(execution_result)
        response = self._make_request(
            "post",
            f"job/{job_id}/execution_config/",
            json=config.to_payload(),
            timeout=50,
        )
        return response.json()

    def get_execution_config(
        self,
        execution_result: ExecutionResult,
    ) -> ExecutionConfig:
        """Retrieve the execution configuration for an existing job.

        Args:
            execution_result: An ExecutionResult instance whose ``job_id``
                identifies the target job.

        Returns:
            ExecutionConfig: The execution configuration attached to the job.

        Raises:
            ValueError: If the ExecutionResult does not have a job_id.
            requests.exceptions.HTTPError:
                - 404: No execution configuration exists for this job.
        """
        job_id = self._extract_job_id(execution_result)
        response = self._make_request(
            "get",
            f"job/{job_id}/execution_config/",
            timeout=50,
        )
        data = response.json()
        return ExecutionConfig.from_response(data["execution_configuration"])

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

        all_results: list[dict] = []
        limit = 100
        offset = 0

        while True:
            try:
                response = self._make_request(
                    "get",
                    f"job/{job_id}/resultsV2/?limit={limit}&offset={offset}",
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

            data = response.json()

            for result in data["results"]:
                result["results"] = _decode_qh1_b64(result["results"])
            all_results.extend(data["results"])

            if data.get("next") is None:
                break
            offset += limit

        # Return a new ExecutionResult with results populated
        return execution_result.with_results(all_results)

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
