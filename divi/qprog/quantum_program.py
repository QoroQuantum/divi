import pickle
from abc import ABC, abstractmethod
from typing import Optional

from divi.services import QoroService
from divi.simulator.parallel_simulator import ParallelSimulator


class QuantumProgram(ABC):
    def __init__(self, qoro_service: Optional[QoroService] = None, **kwargs):
        self.circuits = []
        if (m_list_circuits := kwargs.pop("circuits", None)) is not None:
            self.circuits = m_list_circuits

        self._total_circuit_count = 0

        self.qoro_service = qoro_service
        self.job_id = None

    @property
    def total_circuit_count(self):
        return self._total_circuit_count

    @total_circuit_count.setter
    def _(self, value):
        raise RuntimeError("Can not set total circuit count value.")

    @abstractmethod
    def _generate_circuits(self, params=None, **kwargs):
        pass

    @abstractmethod
    def run(self, store_data=False, data_file=None):
        pass

    @abstractmethod
    def _post_process_results(self, job_id=None, results=None):
        pass

    def _prepare_and_send_circuits(self):
        job_circuits = {}

        for circuit in self.circuits:
            for tag, qasm_circuit in zip(circuit.tags, circuit.qasm_circuits):
                job_circuits[tag] = qasm_circuit

        self._total_circuit_count += len(job_circuits)

        if self.qoro_service is not None:
            self.job_id = self.qoro_service.send_circuits(
                job_circuits, shots=self.shots
            )
            return self.job_id, "job_id"
        else:
            circuit_simulator = ParallelSimulator()
            circuit_results = circuit_simulator.simulate(job_circuits, shots=self.shots)
            return circuit_results, "circuit_results"

    def _dispatch_circuits_and_process_results(self, store_data=False, data_file=None):
        """
        Run an iteration of the program. The outputs are stored in the Program object.
        Optionally, the data can be stored in a file.

        Args:
            store_data (bool): Whether to store the data for the iteration
            data_file (str): The file to store the data in
        """

        results, backend_return_type = self._prepare_and_send_circuits()

        if backend_return_type == "job_id":
            result = self._post_process_results(job_id=results)
        elif backend_return_type == "circuit_results":
            result = self._post_process_results(results=results)

        if store_data:
            self.save_iteration(data_file)

        return result

    def save_iteration(self, data_file):
        """
        Save the current iteration of the program to a file.

        Args:
            data_file (str): The file to save the iteration to.
        """

        with open(data_file, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def import_iteration(data_file):
        """
        Import an iteration of the program from a file.

        Args:
            data_file (str): The file to import the iteration from.
        """

        with open(data_file, "rb") as f:
            return pickle.load(f)
