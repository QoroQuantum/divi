import logging
import pickle
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from qiskit.result import marginal_counts, sampled_expectation_value
from scipy.optimize import minimize

from divi.parallel_simulator import ParallelSimulator
from divi.qprog.optimizers import Optimizers
from divi.services import QoroService
from divi.services.qoro_service import JobStatus

# Set up your logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(ch)

# Suppress debug logs from external libraries
logging.getLogger().setLevel(logging.WARNING)


class QuantumProgram(ABC):
    def __init__(
        self, shots: int = 5000, qoro_service: Optional[QoroService] = None, **kwargs
    ):
        self.circuits = []
        if (m_list_circuits := kwargs.pop("circuits", None)) is not None:
            self.circuits = m_list_circuits

        self._total_circuit_count = 0
        self._total_run_time = 0
        self._curr_params = []

        # Lets child classes adapt their optimization
        # step for grad calculation routine
        self._grad_mode = False

        self.shots = shots
        self.qoro_service = qoro_service
        self.job_id = None

    @property
    def total_circuit_count(self):
        return self._total_circuit_count

    @total_circuit_count.setter
    def total_circuit_count(self, _):
        raise RuntimeError("Can not set total circuit count value.")

    @property
    def total_run_time(self):
        return self._total_run_time

    @total_run_time.setter
    def total_run_time(self, _):
        raise RuntimeError("Can not set total run time value.")

    def _reset_params(self):
        self._curr_params = []

    @abstractmethod
    def _generate_circuits(self, **kwargs):
        pass

    @abstractmethod
    def run(self, store_data=False, data_file=None):
        pass

    def _run_optimization_circuits(self, store_data, data_file):
        self.circuits[:] = []
        self._generate_circuits()
        losses = self._dispatch_circuits_and_process_results(
            store_data=store_data, data_file=data_file
        )

        return losses

    def _update_mc_params(self):
        """
        Updates the parameters based on previous MC iteration.
        """

        if self.current_iteration == 0:
            self._reset_params()
            self._curr_params = [
                np.random.uniform(0, 2 * np.pi, self.n_layers * self.n_params)
                for _ in range(self.optimizer.n_param_sets)
            ]

            self.current_iteration += 1

            return

        self._curr_params = self.optimizer.compute_new_parameters(
            self._curr_params,
            self.current_iteration,
            losses=self.losses[-1],
        )

        self.current_iteration += 1

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
            self._total_run_time += float(response["run_time"])

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

    def _post_process_results(self, results):
        """
        Post-process the results of the quantum problem.

        Returns:
            (dict) The energies for each parameter set grouping.
        """

        losses = {}

        for p, _ in enumerate(self._curr_params):
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

    def run(self, store_data=False, data_file=None):
        """
        Run the QAOA problem. The outputs are stored in the QAOA object. Optionally, the data can be stored in a file.

        Args:
            store_data (bool): Whether to store the data for the iteration
            data_file (str): The file to store the data in
        """

        if self.optimizer == Optimizers.MONTE_CARLO:
            logger.debug(f"Finished iteration {self.current_iteration}")
            while self.current_iteration < self.max_iterations:

                self._update_mc_params()

                curr_losses = self._run_optimization_circuits(store_data, data_file)

                self.losses.append(curr_losses)

                logger.debug(f"Finished iteration {self.current_iteration}")

            return self._total_circuit_count, self._total_run_time

        elif self.optimizer in (Optimizers.NELDER_MEAD, Optimizers.L_BFGS_B):
            logger.debug(f"Finished iteration {self.current_iteration}")

            def cost_fn(params):
                self._curr_params = np.atleast_2d(params)

                losses = self._run_optimization_circuits(store_data, data_file)

                return losses[0]

            def grad_fn(params):
                self._grad_mode = True

                shift_mask = self.optimizer.compute_parameter_shift_mask(len(params))

                self._curr_params = shift_mask + params

                exp_vals = self._run_optimization_circuits(store_data, data_file)

                grads = np.zeros_like(params)
                for i in range(len(params)):
                    grads[i] = 0.5 * (exp_vals[2 * i] - exp_vals[2 * i + 1])

                self._grad_mode = False

                return grads

            def _iteration_counter(intermediate_result):
                self.losses.append({0: intermediate_result.fun})
                self.final_params = np.atleast_2d(intermediate_result.x)

                self.current_iteration += 1
                logger.debug(f"Finished iteration {self.current_iteration}")

            self._reset_params()

            self._curr_params = [
                np.random.uniform(0, 2 * np.pi, self.n_layers * self.n_params)
                for _ in range(self.optimizer.n_param_sets)
            ]

            self._minimize_res = minimize(
                fun=cost_fn,
                x0=self._curr_params[0],
                method=self.optimizer.value,
                jac=grad_fn if self.optimizer == Optimizers.L_BFGS_B else None,
                callback=_iteration_counter,
                options={"maxiter": self.max_iterations},
            )

            if self.max_iterations == 1:
                # Need to handle this edge case for single
                # iteration optimization
                self.current_iteration += 1

            return self._total_circuit_count, self._total_run_time

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
