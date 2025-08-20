# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import base64
import gzip
import json
import logging
import time
from collections.abc import Callable
from enum import Enum
from http import HTTPStatus

import requests
from dotenv import dotenv_values
from requests.adapters import HTTPAdapter, Retry

from divi.exp.cirq import is_valid_qasm
from divi.interfaces import CircuitRunner
from divi.qpu_system import QPU, QPUSystem

API_URL = "https://app.qoroquantum.net/api"
MAX_PAYLOAD_SIZE_MB = 0.95

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
    raise requests.HTTPError(msg)


class JobStatus(Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class JobType(Enum):
    EXECUTE = "EXECUTE"
    SIMULATE = "SIMULATE"
    ESTIMATE = "ESTIMATE"
    CIRCUIT_CUT = "CIRCUIT_CUT"


class MaxRetriesReachedError(Exception):
    """Exception raised when the maximum number of retries is reached."""

    def __init__(self, retries):
        self.retries = retries
        self.message = f"Maximum retries reached: {retries} retries attempted"
        super().__init__(self.message)


def _parse_qpu_systems(json_data: list) -> list[QPUSystem]:
    return [
        QPUSystem(
            name=system_data["name"],
            qpus=[QPU(**qpu) for qpu in system_data.get("qpus", [])],
            access_level=system_data["access_level"],
        )
        for system_data in json_data
    ]


class QoroService(CircuitRunner):

    def __init__(
        self,
        auth_token: str | None = None,
        polling_interval: float = 3.0,
        max_retries: int = 5000,
        shots: int = 1000,
        qpu_system_name: str | QPUSystem | None = None,
        use_circuit_packing: bool = False,
    ):
        super().__init__(shots=shots)

        if auth_token is None:
            try:
                auth_token = dotenv_values()["QORO_API_KEY"]
            except KeyError:
                raise ValueError("Qoro API key not provided nor found in a .env file.")

        self.auth_token = "Bearer " + auth_token
        self.polling_interval = polling_interval
        self.max_retries = max_retries
        self._qpu_system_name = qpu_system_name
        self.use_circuit_packing = use_circuit_packing

    @property
    def qpu_system_name(self) -> str | QPUSystem | None:
        return self._qpu_system_name

    @qpu_system_name.setter
    def qpu_system_name(self, system_name: str | QPUSystem | None):
        """
        Set the QPU system for the service.

        Args:
            system_name (str | QPUSystem): The QPU system to set or the name as a string.
        """
        if isinstance(system_name, str):
            self._qpu_system_name = system_name
        elif isinstance(system_name, QPUSystem):
            self._qpu_system_name = system_name.name
        elif system_name is None:
            self._qpu_system_name = None
        else:
            raise TypeError("Expected a QPUSystem instance or str.")

    def _make_request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """A centralized helper for making API requests."""
        url = f"{API_URL}/{endpoint}"

        headers = {"Authorization": self.auth_token}

        if method.upper() in ["POST", "PUT", "PATCH"]:
            headers["Content-Type"] = "application/json"

        # Allow overriding default headers
        if "headers" in kwargs:
            headers.update(kwargs.pop("headers"))

        response = session.request(method, url, headers=headers, **kwargs)

        # Generic error handling for non-OK statuses
        if response.status_code >= 400:
            raise requests.exceptions.HTTPError(
                f"API Error: {response.status_code} {response.reason} for URL {response.url}"
            )

        return response

    def test_connection(self):
        """Test the connection to the Qoro API"""
        return self._make_request("get", "", timeout=10)

    def fetch_qpu_systems(self) -> list[QPUSystem]:
        """
        Get the list of available QPU systems from the Qoro API.

        Returns:
            List of QPUSystem objects.
        """
        response = self._make_request("get", "qpusystem/", timeout=10)
        return _parse_qpu_systems(response.json())

    @staticmethod
    def _compress_data(value) -> bytes:
        return base64.b64encode(gzip.compress(value.encode("utf-8"))).decode("utf-8")

    def _split_circuits(self, circuits: dict[str, str]) -> list[dict[str, str]]:
        """
        Splits circuits into chunks by estimating payload size with a simplified,
        consistent overhead calculation.
        Assumes that BASE64 encoding produces ASCI characters, which are 1 byte each.
        """
        max_payload_bytes = MAX_PAYLOAD_SIZE_MB * 1024 * 1024
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
        tag: str = "default",
        job_type: JobType = JobType.SIMULATE,
        qpu_system_name: str | None = None,
        override_circuit_packing: bool | None = None,
    ):
        """
        Submit quantum circuits to the Qoro API for execution.

        Args:
            circuits (dict[str, str]):
                Dictionary mapping unique circuit IDs to QASM circuit strings.
            tag (str, optional):
                Tag to associate with the job for identification. Defaults to "default".
            job_type (JobType, optional):
                Type of job to execute (e.g., SIMULATE, EXECUTE, ESTIMATE, CIRCUIT_CUT). Defaults to JobType.SIMULATE.
            use_packing (bool):
                Whether to use circuit packing optimization. Defaults to False.

        Raises:
            ValueError: If more than one circuit is submitted for a CIRCUIT_CUT job.

        Returns:
            str or list[str]:
                The job ID(s) of the created job(s). Returns a single job ID if only one job is created,
                otherwise returns a list of job IDs if the circuits are split into multiple jobs due to payload size.
        """

        if job_type == JobType.CIRCUIT_CUT and len(circuits) > 1:
            raise ValueError("Only one circuit allowed for circuit-cutting jobs.")

        for key, circuit in circuits.items():
            if not is_valid_qasm(circuit):
                raise ValueError(f"Circuit {key} is not a valid QASM string.")

        circuit_chunks = self._split_circuits(circuits)

        payload = {
            "shots": self.shots,
            "tag": tag,
            "job_type": job_type.value,
            "qpu_system_name": qpu_system_name or self.qpu_system_name,
            "use_packing": (
                override_circuit_packing
                if override_circuit_packing is not None
                else self.use_circuit_packing
            ),
        }

        job_ids = []
        for chunk in circuit_chunks:
            payload["circuits"] = chunk

            response = self._make_request(
                "post",
                "job/",
                json=payload,
                timeout=100,
            )

            if response.status_code == HTTPStatus.CREATED:
                job_ids.append(response.json()["job_id"])
            else:
                _raise_with_details(response)

        return job_ids if len(job_ids) > 1 else job_ids[0]

    def delete_job(self, job_ids):
        """
        Delete a job from the Qoro Database.

        Args:
            job_id: The ID of the jobs to be deleted
        Returns:
            response: The response from the API
        """
        if not isinstance(job_ids, list):
            job_ids = [job_ids]

        responses = [
            self._make_request(
                "delete",
                f"job/{job_id}",
                timeout=50,
            )
            for job_id in job_ids
        ]

        return responses if len(responses) > 1 else responses[0]

    def get_job_results(self, job_ids):
        """
        Get the results of a job from the Qoro Database.

        Args:
            job_id: The ID of the job to get results from
        Returns:
            results: The results of the job
        """
        if not isinstance(job_ids, list):
            job_ids = [job_ids]

        responses = [
            self._make_request(
                "get",
                f"job/{job_id}/results",
                timeout=100,
            )
            for job_id in job_ids
        ]

        if all(response.status_code == HTTPStatus.OK for response in responses):
            responses = [response.json() for response in responses]
            return sum(responses, [])
        elif any(
            response.status_code == HTTPStatus.BAD_REQUEST for response in responses
        ):
            raise requests.exceptions.HTTPError(
                "400 Bad Request: Job results not available, likely job is still running"
            )
        else:
            for response in responses:
                if response.status_code not in [HTTPStatus.OK, HTTPStatus.BAD_REQUEST]:
                    raise requests.exceptions.HTTPError(
                        f"{response.status_code}: {response.reason}"
                    )

    def poll_job_status(
        self,
        job_ids: str | list[str],
        loop_until_complete: bool = False,
        on_complete: Callable | None = None,
        verbose: bool = True,
        pbar_update_fn: Callable | None = None,
    ):
        """
        Get the status of a job and optionally execute function *on_complete* on the results
        if the status is COMPLETE.

        Args:
            job_ids: The job id of the jobs to check
            loop_until_complete (bool): A flag to loop until the job is completed
            on_complete (optional): A function to be called when the job is completed
            polling_interval (optional): The time to wait between retries
            verbose (optional): A flag to print the when retrying
            pbar_update_fn (optional): A function for updating progress bars while polling.
        Returns:
            status: The status of the job
        """
        if not isinstance(job_ids, list):
            job_ids = [job_ids]

        if not loop_until_complete:
            statuses = [
                self._make_request(
                    "get",
                    f"job/{job_id}/status/",
                    timeout=200,
                ).json()["status"]
                for job_id in job_ids
            ]
            return statuses if len(statuses) > 1 else statuses[0]

        pending_job_ids = set(job_ids)
        responses = []
        for retry_count in range(1, self.max_retries + 1):
            # Exit early if all jobs are done
            if not pending_job_ids:
                break

            for job_id in list(pending_job_ids):
                response = self._make_request(
                    "get",
                    f"job/{job_id}/status/",
                    timeout=200,
                )

                if response.json()["status"] == JobStatus.COMPLETED.value:
                    pending_job_ids.remove(job_id)
                    responses.append(response)

            # Exit before sleeping if no jobs are pending
            if not pending_job_ids:
                break

            time.sleep(self.polling_interval)

            if verbose:
                if pbar_update_fn:
                    pbar_update_fn(retry_count)
                else:
                    logger.info(
                        rf"\cPolling {retry_count} / {self.max_retries} retries\r"
                    )

        if not pending_job_ids:
            if on_complete:
                on_complete(responses)
            return JobStatus.COMPLETED
        else:
            raise MaxRetriesReachedError(retry_count)
