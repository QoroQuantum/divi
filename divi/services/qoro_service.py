import base64
import gzip
import json
import time
from enum import Enum
from http import HTTPStatus
import requests
from requests.adapters import HTTPAdapter, Retry


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


class JobStatus(Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class JobTypes(Enum):
    EXECUTE = "EXECUTE"
    SIMULATE = "SIMULATE"
    ESTIMATE = "ESTIMATE"


class MaxRetriesReachedError(Exception):
    """Exception raised when the maximum number of retries is reached."""

    def __init__(self, retries, message="Maximum retries reached"):
        self.retries = retries
        self.message = f"{message}: {retries} retries attempted"
        super().__init__(self.message)


class QoroService:

    def __init__(self, auth_token) -> None:
        self.auth_token = "Bearer " + auth_token

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

    def send_circuits(
        self, circuits, shots=1000, tag="default", job_type=JobTypes.SIMULATE
    ):
        """
        Send circuits to the Qoro API for execution

        Args:
            circuits: list of circuits to be sent as QASM strings
            shots (optional): number of shots to be executed for each circuit, default 1000
            tag (optional): tag to be used for the job, defaut "default"
        Returns:
            job_id: The job id of the job created
        """

        def _compress_data(value):
            return base64.b64encode(gzip.compress(value.encode("utf-8"))).decode(
                "utf-8"
            )

        def _split_circuits(circuits):
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
                    "shots": shots,
                    "tag": tag,
                    "type": job_type.value,
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
        job_ids,
        loop_until_complete=False,
        on_complete=None,
        timeout=3,
        max_retries=5000,
        verbose=True,
    ):
        """
        Get the status of a job and optionally execute function *on_complete* on the results
        if the status is COMPLETE.

        Args:
            job_id: The job id of the jobs to check
            loop_until_complete (optional): A flag to loop until the job is completed
            on_complete (optional): A function to be called when the job is completed
            timeout (optional): The time to wait between retries
            max_retries (optional): The maximum number of retries
            verbose (optional): A flag to print the when retrying
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

                if retries >= max_retries:
                    break

                retries += 1

                time.sleep(timeout)

                if verbose:
                    print(
                        f"Polling job {job_ids}: {retries} times / {max_retries} retries",
                        f"Run time: {retries*timeout} seconds",
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
