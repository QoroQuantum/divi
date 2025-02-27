import pickle
from abc import ABC, abstractmethod
from typing import Optional

from qiskit.result import marginal_counts, sampled_expectation_value

from divi.services import QoroService
from divi.services.qoro_service import JobStatus
from divi.simulator.parallel_simulator import ParallelSimulator


class QuantumProgram(ABC):
    def __init__(
        self, shots: int = 5000, qoro_service: Optional[QoroService] = None, **kwargs
    ):
        self.circuits = []
        if (m_list_circuits := kwargs.pop("circuits", None)) is not None:
            self.circuits = m_list_circuits

        self._total_circuit_count = 0

        self.shots = shots
        self.qoro_service = qoro_service
        self.job_id = None
        self.run_time = 0

    @property
    def total_circuit_count(self):
        return self._total_circuit_count

    @total_circuit_count.setter
    def _(self, value):
        raise RuntimeError("Can not set total circuit count value.")

    @property
    def total_run_time(self):
        return self.run_time

    @total_run_time.setter
    def _(self, value):
        raise RuntimeError("Can not set total run time value.")

    @abstractmethod
    def _generate_circuits(self, params=None, **kwargs):
        pass

    @abstractmethod
    def run(self, store_data=False, data_file=None):
        pass

    def _post_process_results(self, results):
        """
        Post-process the results of the VQE problem.

        Returns:
            (dict) The energies for each parameter set grouping.
        """

        losses = {}

        for p, _ in enumerate(self.params):
            losses[p] = 0
            cur_result = {
                key: value for key, value in results.items() if key.startswith(f"{p}")
            }

            marginal_results = []
            for param_id, shots_dict in cur_result.items():
                ham_op_index = int(param_id.split("_")[-1])
                ham_op_metadata = self.expval_hamiltonian_metadata[ham_op_index]
                pair = (
                    ham_op_metadata,
                    marginal_counts(shots_dict, ham_op_metadata[0].tolist()),
                )
                marginal_results.append(pair)

            for ham_op_metadata, marginal_shots in marginal_results:
                exp_value = sampled_expectation_value(
                    marginal_shots, "Z" * len(ham_op_metadata[0])
                )
                losses[p] += ham_op_metadata[1] * exp_value

        return losses

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

        def add_run_time(response):
            self.run_time += float(response["run_time"])

        if backend_return_type == "job_id":
            job_id = results
            if job_id is not None and self.qoro_service is not None:
                status = self.qoro_service.job_status(
                    self.job_id,
                    loop_until_complete=True,
                    on_complete=add_run_time,
                )
                if status != JobStatus.COMPLETED:
                    raise Exception(
                        "Job has not completed yet, cannot post-process results"
                    )
                results = self.qoro_service.get_job_results(self.job_id)

        results = {r["label"]: r["results"] for r in results}

        result = self._post_process_results(results)

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
