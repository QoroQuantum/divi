import os
import time
from enum import Enum
from http import HTTPStatus

import requests

LOCAL = os.environ.get("LOCAL", "True") == "True"
if LOCAL:
    API_URL = "http://127.0.0.1:8000/api"
else:
    API_URL = "https://app.qoroquantum.net/api"


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

        response = requests.get(
            API_URL, headers={"Authorization": self.auth_token}, timeout=10
        )
        if response.status_code == HTTPStatus.OK:
            print("Connection successful")
        else:
            print("Connection failed")
        return response

    def send_circuits(
        self, circuits, shots=1000, tag="default", job_type=JobTypes.SIMULATE
    ):
        """
        Send circuits to the Qoro API for execution

        args:
            circuits: list of circuits to be sent as QASM strings
            shots (optional): number of shots to be executed for each circuit, default 1000
            tag (optional): tag to be used for the job, defaut "default"
        return:
            job_id: The job id of the job created
        """
        data = {
            "circuits": circuits,
            "shots": shots,
            "tag": tag,
            "type": job_type.value,
        }
        response = requests.post(
            API_URL + "/job/",
            headers={
                "Authorization": self.auth_token,
                "Content-Type": "application/json",
            },
            json=data,
            timeout=10,
        )
        if response.status_code == HTTPStatus.CREATED:
            job_id = response.json()["job_id"]
            return job_id
        elif response.status_code == HTTPStatus.UNAUTHORIZED:
            raise requests.exceptions.HTTPError("401 Unauthorized: Invalid API token")
        else:
            raise requests.exceptions.HTTPError(
                f"{response.status_code}: {response.reason}"
            )

    def delete_job(self, job_id):
        """
        Delete a job from the Qoro Database.

        args:
            job_id: The ID of the job to be deleted
        return:
            response: The response from the API
        """
        response = requests.delete(
            API_URL + f"/job/{job_id}",
            headers={"Authorization": self.auth_token},
            timeout=10,
        )
        return response

    def get_job_results(self, job_id):
        """
        Get the results of a job from the Qoro Database.

        args:
            job_id: The ID of the job to get results from
        return:
            results: The results of the job
        """
        response = requests.get(
            API_URL + f"/job/{job_id}/results",
            headers={"Authorization": self.auth_token},
            timeout=10,
        )
        if response.status_code == HTTPStatus.OK:
            return response.json()
        elif response.status_code == HTTPStatus.BAD_REQUEST:
            raise requests.exceptions.HTTPError(
                "400 Bad Request: Job results not available, likely job is still running"
            )
        else:
            raise requests.exceptions.HTTPError(
                f"{response.status_code}: {response.reason}"
            )

    def job_status(
        self,
        job_id,
        loop_until_complete=False,
        on_complete=None,
        timeout=5,
        max_retries=100,
        verbose=True,
    ):
        """
        Get the status of a job and optionally execute function *on_complete* on the results
        if the status is COMPLETE.

        args:
            job_id: The job id of the job
            loop_until_complete (optional): A flag to loop until the job is completed
            on_complete (optional): A function to be called when the job is completed
            timeout (optional): The time to wait between retries
            max_retries (optional): The maximum number of retries
            verbose (optional): A flag to print the when retrying
        return:
            status: The status of the job
        """

        def _poll_job_status():
            response = requests.get(
                API_URL + f"/job/{job_id}/status",
                headers={
                    "Authorization": self.auth_token,
                    "Content-Type": "application/json",
                },
                timeout=200,
            )
            if response.status_code == HTTPStatus.OK:
                return response.json()["status"], response
            else:
                raise ("Error getting job status")

        if loop_until_complete:
            retries = 0
            completed = False
            while True:
                job_status, response = _poll_job_status()
                if job_status == JobStatus.COMPLETED.value:
                    results = response.json()["results"]
                    completed = True
                    break
                if retries >= max_retries:
                    break
                retries += 1
                time.sleep(timeout)
                if verbose:
                    print(
                        f"Retrying: {retries} times",
                        f"Run time: {retries*timeout} seconds",
                    )
            if completed and on_complete:
                return on_complete(results)
            elif completed:
                return JobStatus.COMPLETED
            else:
                raise MaxRetriesReachedError(retries)
        else:
            return _poll_job_status()[0]
