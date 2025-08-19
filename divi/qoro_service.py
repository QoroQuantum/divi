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
from typing import Optional

import requests
from requests.adapters import HTTPAdapter, Retry

from divi.exp.cirq import is_valid_qasm
from divi.interfaces import CircuitRunner
from divi.qpu_system import QPUSystem, parse_qpu_systems

API_URL = "https://app.qoroquantum.net/api"
MAX_PAYLOAD_SIZE_MB = 0.95

session = requests.Session()
retries = Retry(
    total=5,
    backoff_factor=0.1,
    status_forcelist=[502],
    allowed_methods=["GET", "POST", "DELETE"],
)

session.mount("http://", HTTPAdapter(max_retries=retries))
session.mount("https://", HTTPAdapter(max_retries=retries))

logger = logging.getLogger(__name__)


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

    def __init__(self, retries, message="Maximum retries reached"):
        self.retries = retries
        self.message = f"{message}: {retries} retries attempted"
        super().__init__(self.message)


class QoroService(CircuitRunner):

    def __init__(
        self,
        auth_token: str,
        polling_interval: float = 3.0,
        max_retries: int = 5000,
        shots: int = 1000,
        qpu_system_name: Optional[str | QPUSystem] = None,
        use_circuit_packing: bool = False,
    ):
        super().__init__(shots=shots)

        self.auth_token = "Bearer " + auth_token
        self.polling_interval = polling_interval
        self.max_retries = max_retries
        self._qpu_system_name = qpu_system_name
        self.use_circuit_packing = use_circuit_packing

    @property
    def qpu_system_name(self) -> Optional[str | QPUSystem]:
        return self._qpu_system_name

    @qpu_system_name.setter
    def qpu_system_name(self, system_name: str | QPUSystem):
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

    def test_connection(self):
        """Test the connection to the Qoro API"""
        response = session.get(
            API_URL, headers={"Authorization": self.auth_token}, timeout=10
        )

        if response.status_code != HTTPStatus.OK:
            raise requests.exceptions.HTTPError(
                f"Connection failed with error: {response.status_code}: {response.reason}"
            )

        return response

    def submit_circuits(
        self,
        circuits: dict[str, str],
        tag: str = "default",
        job_type: JobType = JobType.SIMULATE,
        qpu_system_name: Optional[str] = None,
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

        def _compress_data(value) -> bytes:
            return base64.b64encode(gzip.compress(value.encode("utf-8"))).decode(
                "utf-8"
            )

        def _split_circuits(circuits: dict[str, str]) -> list[dict[str, str]]:
            """
            Split circuits into smaller chunks if the payload size exceeds the maximum allowed size.

            Args:
                circuits: Dictionary of circuits to be sent

            Returns:
                List of circuit chunks
            """

            def _estimate_size(data):
                payload_json = json.dumps(data)
                return len(payload_json.encode("utf-8")) / 1024 / 1024

            circuit_chunks = []
            current_chunk = {}
            current_size = 0

            for key, value in circuits.items():
                compressed_value = _compress_data(value)
                estimated_size = _estimate_size({key: compressed_value})

                if current_size + estimated_size > MAX_PAYLOAD_SIZE_MB:
                    circuit_chunks.append(current_chunk)
                    current_chunk = {key: compressed_value}
                    current_size = estimated_size
                else:
                    current_chunk[key] = compressed_value
                    current_size += estimated_size

            if current_chunk:
                circuit_chunks.append(current_chunk)

            return circuit_chunks

        circuit_chunks = _split_circuits(circuits)

        job_ids = []
        for chunk in circuit_chunks:
            response = session.post(
                API_URL + "/job/",
                headers={
                    "Authorization": self.auth_token,
                    "Content-Type": "application/json",
                },
                json={
                    "circuits": chunk,
                    "shots": self.shots,
                    "tag": tag,
                    "job_type": job_type.value,
                    "qpu_system_name": (
                        self.qpu_system_name
                        if self.qpu_system_name is not None
                        else qpu_system_name
                    ),
                    "use_packing": (
                        override_circuit_packing
                        if override_circuit_packing is not None
                        else self.use_circuit_packing
                    ),
                },
                timeout=100,
            )

            if response.status_code == HTTPStatus.CREATED:
                job_ids.append(response.json()["job_id"])
            else:
                raise requests.exceptions.HTTPError(
                    f"{response.status_code}: {response.reason}"
                )

        return job_ids if len(job_ids) > 1 else job_ids[0]

    def qpu_systems(self) -> list[QPUSystem]:
        """
        Get the list of available QPU systems from the Qoro API.

        Returns:
            List of QPUSystem objects.
        """
        response = session.get(
            API_URL + "/qpusystem/",
            headers={"Authorization": self.auth_token},
            timeout=10,
        )

        if response.status_code == HTTPStatus.OK:
            return parse_qpu_systems(response.json())

        else:
            raise requests.exceptions.HTTPError(
                f"{response.status_code}: {response.reason}"
            )

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

        responses = []

        for job_id in job_ids:
            response = session.delete(
                API_URL + f"/job/{job_id}",
                headers={"Authorization": self.auth_token},
                timeout=50,
            )

            responses.append(response)

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

        responses = []
        for job_id in job_ids:
            response = session.get(
                API_URL + f"/job/{job_id}/results",
                headers={"Authorization": self.auth_token},
                timeout=100,
            )
            responses.append(response)

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
            max_retries (optional): The maximum number of retries
            verbose (optional): A flag to print the when retrying
            pbar_update_fn (optional): A function for updating progress bars while polling.
        Returns:
            status: The status of the job
        """
        if not isinstance(job_ids, list):
            job_ids = [job_ids]

        def _poll_job_status(job_id):
            response = session.get(
                API_URL + f"/job/{job_id}/status/",
                headers={
                    "Authorization": self.auth_token,
                    "Content-Type": "application/json",
                },
                timeout=200,
            )

            if response.status_code == HTTPStatus.OK:
                return response.json()["status"], response
            else:
                raise requests.exceptions.HTTPError(
                    f"{response.status_code}: {response.reason}"
                )

        if loop_until_complete:
            retries = 0
            completed = False
            while True:
                responses = []
                statuses = []

                for job_id in job_ids:
                    job_status, response = _poll_job_status(job_id)
                    statuses.append(job_status)
                    responses.append(response)

                if all(status == JobStatus.COMPLETED.value for status in statuses):
                    responses = [response.json() for response in responses]
                    completed = True
                    break

                if retries >= self.max_retries:
                    break

                retries += 1

                time.sleep(self.polling_interval)

                if verbose:
                    if pbar_update_fn:
                        pbar_update_fn(retries)
                    else:
                        logger.info(
                            rf"\cPolling {retries} / {self.max_retries} retries\r"
                        )

            if completed and on_complete:
                on_complete(responses)
                return JobStatus.COMPLETED
            elif completed:
                return JobStatus.COMPLETED
            else:
                raise MaxRetriesReachedError(retries)
        else:
            statuses = [_poll_job_status(job_id)[0] for job_id in job_ids]
            return statuses if len(statuses) > 1 else statuses[0]
