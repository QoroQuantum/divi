import requests
import json
import time

from enum import Enum

API_URL = "https://app.qoroquantum.net/api/"


class JobStatus(Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class QoroService:

    def __init__(self, auth_token) -> None:
        self.auth_token = "Bearer " + auth_token

    def test_connection(self):
        """Test the connection to the Qoro API"""

        response = requests.get(API_URL+"/",
                                headers={"Authorization": self.auth_token},
                                timeout=10
                                )
        if response.status_code == 200:
            print("Connection successful")
        else:
            print("Connection failed")
        return response

    def declare_architecture(self, system_name, qubits, classical_bits, architectures, system_kinds):
        """
        Declare a QPU architecture to the Qoro API.

        args:
            system_name: The name of the QPU system
            qubits (list): The number of qubits per system
            classical_bits (list): The number of classical bits per system
            architecture (list): The architectures of the system
            system_kind (list): The kinds of systems
            names (list): The names of the QPU systems
        return:
            The system ID of the system created
        """
        assert len(qubits) == len(classical_bits) == len(architectures) == len(
            system_kinds), "All lists of the QPU systems must be of the same length"
        
        system_details = zip(qubits, classical_bits,
                             architectures, system_kinds)
        system_info = {
            "qpus": [{"qubits": details[0],
                      "classical_bits": details[1],
                      "architecture": details[2],
                      "system_kind": details[3]} for details in system_details],
            "name": system_name
        }
        response = requests.post(API_URL+"/qpusystem/",
                                 data=json.dumps(system_info),
                                 headers={
                                     "Authorization": self.auth_token,
                                     "Content-Type": "application/json"},
                                 timeout=10
                                 )
        if response.status_code == 201:
            return response.json()['id']
        else:
            raise ("Error setting QPU configuration", response.reason)

    def send_circuits(self, circuits, shots=1000, tag="default"):
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
            "tag": tag
        }
        response = requests.post(API_URL+"/job/",
                                 headers={"Authorization": self.auth_token,
                                          "Content-Type": "application/json"},
                                 json=data,
                                 timeout=10
                                 )
        if response.status_code == 201:
            job_id = response.json()['job_id']
            return job_id
        else:
            print("Failed to send circuits")

    def job_status(self, job_id, loop_until_complete=False, on_complete=None,  timeout=5, max_retries=50):
        """
        Get the status of a job and optionally execute function *on_complete* on the results 
        if the status is COMPLETE.

        args:
            job_id: The job id of the job            
            loop_until_complete (optional): A flag to loop until the job is completed
            on_complete (optional): A function to be called when the job is completed
            timeout (optional): The time to wait between retries
            max_retries (optional): The maximum number of retries
        return:
            status: The status of the job
        """
        def _poll_job_status():
            response = requests.get(API_URL+f"/job/{job_id}/status",
                                    headers={"Authorization": self.auth_token,
                                             "Content-Type": "application/json"},
                                    timeout=10
                                    )
            if response.status_code == 200:
                return response.json()['status'], response
            else:
                raise ("Error getting job status")

        if loop_until_complete:
            retries = 0
            completed = False
            while True:
                job_status, response = _poll_job_status()
                if job_status == JobStatus.COMPLETED:
                    results = response.json()['results']
                    completed = True
                    break
                if retries >= max_retries:
                    break
                retries += 1
                time.sleep(timeout)
            if completed and on_complete:
                return on_complete(results)
            elif completed:
                return results
            else:
                raise ("Max retries reached, job did not complete")
        else:
            return _poll_job_status()
