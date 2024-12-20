from divi.services.qoro_service import JobTypes
from divi.simulator.parallel_simulator import ParallelSimulator


class QuantumProgram:
    def __init__(self, qoro_service=None):
        self.qoro_service = qoro_service
        self.job_id = None

    def _prepare_and_send_circuits(self):
        job_circuits = {}

        # This minor type check is needed because of how the VQE class is implemented. Revert to self.circuits once the behaviour is atomized
        outer_loop_circuits = (
            [self.circuits]
            if isinstance(self.circuits, list)
            else self.circuits.values()
        )

        for circuits in outer_loop_circuits:
            for circuit in circuits:
                job_circuits[circuit.tag] = circuit.qasm_circuit

        if self.qoro_service is not None:
            job_id = self.qoro_service.send_circuits(
                job_circuits, shots=self.shots, job_type=self.job_type
            )
            self.job_id = job_id if job_id is not None else None
            return job_id, "job_id"
        else:
            circuit_simulator = ParallelSimulator()
            circuit_results = circuit_simulator.simulate(job_circuits, shots=self.shots)
            return circuit_results, "circuit_results"

    def run_iteration(self, store_data=False, data_file=None, type=JobTypes.SIMULATE):
        """
        Run an iteration of the program. The outputs are stored in the VQE object. Optionally, the data can be stored in a file.

        args:
            store_data (bool): Whether to store the data for the iteration
            data_file (str): The file to store the data in
        """

        self._run_optimize()
        self._generate_circuits()
        results, param = self._prepare_and_send_circuits()

        if param == "job_id":
            self._post_process_results(job_id=results)
        elif param == "circuit_results":
            self._post_process_results(results=results)

        if store_data:
            self.save_iteration(data_file)

    def save_iteration(self, data_file):
        """
        Save the current iteration of the program to a file.

        args:
            data_file (str): The file to save the iteration to.
        """
        import pickle

        with open(data_file, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def import_iteration(data_file):
        """
        Import an iteration of the program from a file.

        args:
            data_file (str): The file to import the iteration from.
        """
        import pickle

        with open(data_file, "rb") as f:
            return pickle.load(f)
