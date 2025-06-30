import logging
import pickle
from abc import ABC, abstractmethod
from functools import partial
from typing import Optional

import numpy as np
from qiskit.result import marginal_counts, sampled_expectation_value
from scipy.optimize import OptimizeResult, minimize

from divi import QoroService
from divi.circuits import Circuit, MetaCircuit
from divi.interfaces import CircuitRunner
from divi.qem import _NoMitigation
from divi.qoro_service import JobStatus
from divi.qprog.optimizers import Optimizers

# Set up your logger
logger = logging.getLogger(__name__)


class QuantumProgram(ABC):
    def __init__(
        self,
        backend: CircuitRunner,
        seed: Optional[int] = None,
        **kwargs,
    ):
        """
        Initializes the QuantumProgram class.

        If a child class represents a hybrid quantum-classical algorithm,
        the instance variables `n_layers` and `n_params` must be set, where:
        - `n_layers` is the number of layers in the quantum circuit.
        - `n_params` is the number of parameters per layer.

        For exotic algorithms where these variables may not be applicable,
        the `_initialize_params` method should be overridden to set the parameters.

        Args:
            shots (int): The number of shots for quantum circuit execution.
                Must be a positive integer. Defaults to 5000.
            qoro_service (QoroService): An instance of QoroService to handle.
                Defaults to None, which corresponds to local simulation.
            seed (int): A seed for numpy's random number generator, which will
                be used for the parameter initialization.
                Defaults to None.

            **kwargs: Additional keyword arguments that influence behaviour.
                - grouping_strategy (Optional[Any]): A strategy for grouping operations, used in Pennylane's transforms.
                    Defaults to None.
                - qem_protocol (Optional[QEMProtocol]): the quantum error mitigation protocol to apply.
                    Must be of type QEMProtocol. Defaults to None.

                The following key values are reserved for internal use and should not be set by the user:
                - losses (Optional[list]): A list to initialize the `losses` attribute. Defaults to an empty list.
                - final_params (Optional[list]): A list to initialize the `final_params` attribute. Defaults to an empty list.

        """

        # Shared Variables
        self.losses = kwargs.pop("losses", [])
        self.final_params = kwargs.pop("final_params", [])

        self.circuits: list[Circuit] = []

        self._total_circuit_count = 0
        self._total_run_time = 0.0
        self._curr_params = []

        self._seed = seed
        self._rng = np.random.default_rng(self._seed)

        # Lets child classes adapt their optimization
        # step for grad calculation routine
        self._grad_mode = False

        self.backend = backend
        self.job_id = kwargs.get("job_id", None)

        # Needed for Pennylane's transforms
        self._grouping_strategy = kwargs.pop("grouping_strategy", None)

        self._qem_protocol = kwargs.pop("qem_protocol", None) or _NoMitigation()

        self._meta_circuit_factory = partial(
            MetaCircuit,
            grouping_strategy=self._grouping_strategy,
            qem_protocol=self._qem_protocol,
        )

    @property
    def total_circuit_count(self):
        return self._total_circuit_count

    @property
    def total_run_time(self):
        return self._total_run_time

    @property
    def meta_circuits(self):
        return self._meta_circuits

    @abstractmethod
    def _create_meta_circuits_dict(self) -> dict[str, MetaCircuit]:
        pass

    @abstractmethod
    def _generate_circuits(self, **kwargs):
        pass

    @abstractmethod
    def run(self, store_data=False, data_file=None):
        pass

    def _initialize_params(self):
        self._curr_params = np.array(
            [
                self._rng.uniform(0, 2 * np.pi, self.n_layers * self.n_params)
                for _ in range(self.optimizer.n_param_sets)
            ]
        )

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
            self._initialize_params()

            self.current_iteration += 1

            return

        self._curr_params = self.optimizer.compute_new_parameters(
            self._curr_params,
            self.current_iteration,
            losses=self.losses[-1],
            rng=self._rng,
        )

        self.current_iteration += 1

    def _prepare_and_send_circuits(self):
        job_circuits = {}

        for circuit in self.circuits:
            for tag, qasm_circuit in zip(circuit.tags, circuit.qasm_circuits):
                job_circuits[tag] = qasm_circuit

        self._total_circuit_count += len(job_circuits)

        backend_output = self.backend.submit_circuits(job_circuits)

        if isinstance(self.backend, QoroService):
            self.job_id = backend_output

        return backend_output

    def _dispatch_circuits_and_process_results(self, store_data=False, data_file=None):
        """
        Run an iteration of the program. The outputs are stored in the Program object.
        Optionally, the data can be stored in a file.

        Args:
            store_data (bool): Whether to store the data for the iteration
            data_file (str): The file to store the data in
        """

        results = self._prepare_and_send_circuits()

        def add_run_time(response):
            if isinstance(response, dict):
                self._total_run_time += float(response["run_time"])
            elif isinstance(response, list):
                self._total_run_time += sum(float(r["run_time"]) for r in response)

        if isinstance(self.backend, QoroService):
            status = self.backend.poll_job_status(
                self.job_id,
                loop_until_complete=True,
                on_complete=add_run_time,
            )

            if status != JobStatus.COMPLETED:
                raise Exception(
                    "Job has not completed yet, cannot post-process results"
                )

            results = self.backend.get_job_results(self.job_id)

        results = {r["label"]: r["results"] for r in results}

        result = self._post_process_results(results)

        if store_data:
            self.save_iteration(data_file)

        return result

    def _post_process_results(
        self, results: dict[str, dict[str, int]]
    ) -> dict[int, float]:
        """
        Post-process the results of the quantum problem.

        Args:
            results (dict): The shot histograms of the quantum execution step.
                The keys should be strings of format {param_id}_*_{measurement_group_id}.
                i.e. An underscore-separated bunch of metadata, starting always with
                the index of some parameter and ending with the index of some measurement group.
                Any extra piece of metadata that might be relevant to the specific application can
                be kept in the middle.

        Returns:
            (dict) The energies for each parameter set grouping, where the dict keys
                correspond to the parameter indices.
        """

        losses = {}
        measurement_groups = self._meta_circuits["cost_circuit"].measurement_groups

        for p in range(self._curr_params.shape[0]):
            # Extract relevant entries from the execution results dict
            param_results = {k: v for k, v in results.items() if k.startswith(f"{p}_")}

            # Compute the marginal results for each observable
            marginal_results = []
            for group_idx, curr_measurement_group in enumerate(measurement_groups):
                group_results = {
                    k: v
                    for k, v in param_results.items()
                    if k.endswith(f"_{group_idx}")
                }

                curr_marginal_results = []
                for observable in curr_measurement_group:
                    intermediate_exp_values = [
                        sampled_expectation_value(
                            marginal_counts(shots_dict, observable.wires.tolist()),
                            "Z" * len(observable.wires),
                        )
                        for shots_dict in group_results.values()
                    ]

                    mitigated_exp_value = self._qem_protocol.postprocess_results(
                        intermediate_exp_values
                    )

                    curr_marginal_results.append(mitigated_exp_value)

                marginal_results.append(
                    curr_marginal_results
                    if len(curr_marginal_results) > 1
                    else curr_marginal_results[0]
                )

            pl_loss = (
                self._meta_circuits["cost_circuit"]
                .postprocessing_fn(marginal_results)[0]
                .item()
            )

            losses[p] = pl_loss + self.loss_constant

        return losses

    def run(self, progress_queue=None, store_data=False, data_file=None):
        """
        Run the QAOA problem. The outputs are stored in the QAOA object. Optionally, the data can be stored in a file.

        Args:
            store_data (bool): Whether to store the data for the iteration
            data_file (str): The file to store the data in
        """

        if progress_queue is not None:
            progress_queue.put(
                {
                    "job_id": self.job_id,
                    "message": "Finished Setup",
                    "progress": 0,
                }
            )
        else:
            logger.info("Finished Setup")

        if self.optimizer == Optimizers.MONTE_CARLO:
            while self.current_iteration < self.max_iterations:

                self._update_mc_params()

                if progress_queue is not None:
                    progress_queue.put(
                        {
                            "job_id": self.job_id,
                            "message": f"Running Iteration #{self.current_iteration} circuits",
                            "progress": 0,
                        }
                    )
                else:
                    logger.info(
                        f"Running Iteration #{self.current_iteration} circuits\r"
                    )

                curr_losses = self._run_optimization_circuits(store_data, data_file)

                if progress_queue is not None:
                    progress_queue.put(
                        {
                            "job_id": self.job_id,
                            "progress": 1,
                        }
                    )
                else:
                    logger.info(f"Finished Iteration #{self.current_iteration}\r\n")

                self.losses.append(curr_losses)

            self.final_params[:] = np.atleast_2d(self._curr_params)

        elif self.optimizer in (Optimizers.NELDER_MEAD, Optimizers.L_BFGS_B):

            def cost_fn(params):
                task_name = "💸 Computing Cost 💸"

                if progress_queue is not None:
                    progress_queue.put(
                        {
                            "job_id": self.job_id,
                            "message": task_name,
                            "progress": 0,
                        }
                    )
                else:
                    logger.info(
                        f"Running Iteration #{self.current_iteration + 1} circuits: {task_name}\r"
                    )

                self._curr_params = np.atleast_2d(params)

                losses = self._run_optimization_circuits(store_data, data_file)

                return losses[0]

            def grad_fn(params):
                self._grad_mode = True

                task_name = "📈 Computing Gradients 📈"

                if progress_queue is not None:
                    progress_queue.put(
                        {
                            "job_id": self.job_id,
                            "message": task_name,
                            "progress": 0,
                        }
                    )
                else:
                    logger.info(
                        f"Running Iteration #{self.current_iteration + 1} circuits: {task_name}\r"
                    )

                shift_mask = self.optimizer.compute_parameter_shift_mask(len(params))

                self._curr_params = shift_mask + params

                exp_vals = self._run_optimization_circuits(store_data, data_file)

                grads = np.zeros_like(params)
                for i in range(len(params)):
                    grads[i] = 0.5 * (exp_vals[2 * i] - exp_vals[2 * i + 1])

                self._grad_mode = False

                return grads

            def _iteration_counter(intermediate_result: OptimizeResult):
                self.losses.append({0: intermediate_result.fun})

                self.final_params[:] = np.atleast_2d(intermediate_result.x)

                self.current_iteration += 1

                if progress_queue is not None:
                    progress_queue.put(
                        {
                            "job_id": self.job_id,
                            "progress": 1,
                        }
                    )
                else:
                    logger.info(f"Finished Iteration #{self.current_iteration}\r\n")

            self._initialize_params()
            self._minimize_res = minimize(
                fun=cost_fn,
                x0=self._curr_params[0],
                method=self.optimizer.value,
                jac=grad_fn if self.optimizer == Optimizers.L_BFGS_B else None,
                callback=_iteration_counter,
                options={
                    "maxiter": (
                        # Need to add one more iteration for Nelder-Mead's simplex initialization step
                        self.max_iterations + 1
                        if self.optimizer == Optimizers.NELDER_MEAD
                        else self.max_iterations
                    )
                },
            )

        if progress_queue:
            progress_queue.put(
                {
                    "job_id": self.job_id,
                    "progress": 0,
                    "final_status": "Success",
                }
            )
        else:
            logger.info(f"Finished Optimization!")

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
