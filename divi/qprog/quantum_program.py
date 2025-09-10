# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import logging
import pickle
from abc import ABC, abstractmethod
from functools import partial
from itertools import groupby
from queue import Queue

import numpy as np
from pennylane.measurements import ExpectationMP
from scipy.optimize import OptimizeResult

from divi import QoroService
from divi.circuits import Circuit, MetaCircuit
from divi.interfaces import CircuitRunner
from divi.qem import _NoMitigation
from divi.qoro_service import JobStatus
from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer
from divi.reporter import LoggingProgressReporter, QueueProgressReporter

logger = logging.getLogger(__name__)


def _compute_parameter_shift_mask(n_params):
    mask_arr = np.arange(0, 2 * n_params, 2)
    mask_arr[0] = 1

    binary_matrix = ((mask_arr[:, np.newaxis] & (1 << np.arange(n_params))) > 0).astype(
        np.float64
    )

    binary_matrix = binary_matrix.repeat(2, axis=0)
    binary_matrix[1::2] *= -1
    binary_matrix *= 0.5 * np.pi

    return binary_matrix


class QuantumProgram(ABC):
    def __init__(
        self,
        backend: CircuitRunner,
        seed: int | None = None,
        progress_queue: Queue | None = None,
        has_final_computation: bool = False,
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
            backend (CircuitRunner): An instance of a CircuitRunner object, which
                can either be ParallelSimulator or QoroService.
            seed (int): A seed for numpy's random number generator, which will
                be used for the parameter initialization.
                Defaults to None.
            progress_queue (Queue): a queue for progress bar updates.
            has_final_computation (bool): Whether the program includes a final
                computation step after optimization. This affects progress reporting.

            **kwargs: Additional keyword arguments that influence behaviour.
                - grouping_strategy (Literal["default", "wires", "qwc"]): A strategy for grouping operations, used in Pennylane's transforms.
                    Defaults to None.
                - qem_protocol (QEMProtocol, optional): the quantum error mitigation protocol to apply.
                    Must be of type QEMProtocol. Defaults to None.

                The following key values are reserved for internal use and should not be set by the user:
                - losses (list, optional): A list to initialize the `losses` attribute. Defaults to an empty list.
                - final_params (list, optional): A list to initialize the `final_params` attribute. Defaults to an empty list.

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
        self._progress_queue = progress_queue
        if progress_queue and self.job_id:
            self.reporter = QueueProgressReporter(
                self.job_id, progress_queue, has_final_computation=has_final_computation
            )
        else:
            self.reporter = LoggingProgressReporter()

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

    def _prepare_and_send_circuits(self):
        job_circuits = {}

        for circuit in self.circuits:
            for tag, qasm_circuit in zip(circuit.tags, circuit.qasm_circuits):
                job_circuits[tag] = qasm_circuit

        self._total_circuit_count += len(job_circuits)

        backend_output = self.backend.submit_circuits(job_circuits)

        if isinstance(self.backend, QoroService):
            self._curr_service_job_id = backend_output

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
                self._total_run_time += sum(
                    float(r.json()["run_time"]) for r in response
                )

        if isinstance(self.backend, QoroService):
            update_function = lambda n_polls, status: self.reporter.info(
                message="",
                poll_attempt=n_polls,
                max_retries=self.backend.max_retries,
                service_job_id=self._curr_service_job_id,
                job_status=status,
            )

            status = self.backend.poll_job_status(
                self._curr_service_job_id,
                loop_until_complete=True,
                on_complete=add_run_time,
                verbose=False,  # Disable the default logger in QoroService
                poll_callback=update_function,  # Use the new, more generic name
            )

            if status != JobStatus.COMPLETED:
                raise Exception(
                    "Job has not completed yet, cannot post-process results"
                )

            results = self.backend.get_job_results(self._curr_service_job_id)

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

        # Define key functions for both levels of grouping
        get_param_id = lambda item: int(item[0].split("_")[0])
        get_group_id = lambda item: int(item[0].split("_")[-1])

        # --- OPTIMIZATION START ---
        # Group the pre-sorted results by parameter ID.
        # This is highly memory-efficient as it processes results as a stream.
        for p, param_group_iterator in groupby(results.items(), key=get_param_id):

            # This dictionary is now built from a lazy iterator, which is more efficient.
            # It consumes the inner iterator to create a lookup of histograms for the current 'p'.
            shots_by_group_idx = {
                gid: [value for _, value in group]
                for gid, group in groupby(param_group_iterator, key=get_group_id)
            }

            # --- The rest of the logic is the same, but operates on the efficiently grouped data ---
            marginal_results = []
            for group_idx, curr_measurement_group in enumerate(measurement_groups):
                group_shot_histograms = shots_by_group_idx.get(group_idx, [])

                if not group_shot_histograms:
                    continue

                curr_marginal_results = []
                for observable in curr_measurement_group:
                    intermediate_exp_values = [
                        ExpectationMP(observable).process_counts(
                            shots_dict,
                            tuple(reversed(range(len(next(iter(shots_dict.keys())))))),
                        )
                        for shots_dict in group_shot_histograms
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

    def run(self, store_data=False, data_file=None):
        """
        Run the QAOA problem. The outputs are stored in the QAOA object. Optionally, the data can be stored in a file.

        Args:
            store_data (bool): Whether to store the data for the iteration
            data_file (str): The file to store the data in
        """

        def cost_fn(params):
            self.reporter.info(
                message="💸 Computing Cost 💸", iteration=self.current_iteration
            )

            self._curr_params = np.atleast_2d(params)

            losses = self._run_optimization_circuits(store_data, data_file)

            losses = np.fromiter(losses.values(), dtype=np.float64)

            if params.ndim > 1:
                return losses
            else:
                return losses.item()

        self._grad_shift_mask = _compute_parameter_shift_mask(
            self.n_layers * self.n_params
        )

        def grad_fn(params):
            self._grad_mode = True

            self.reporter.info(
                message="📈 Computing Gradients 📈", iteration=self.current_iteration
            )

            self._curr_params = self._grad_shift_mask + params

            exp_vals = self._run_optimization_circuits(store_data, data_file)
            exp_vals_arr = np.fromiter(exp_vals.values(), dtype=np.float64)

            pos_shifts = exp_vals_arr[::2]
            neg_shifts = exp_vals_arr[1::2]
            grads = 0.5 * (pos_shifts - neg_shifts)

            self._grad_mode = False

            return grads

        def _iteration_counter(intermediate_result: OptimizeResult):
            self.losses.append(
                dict(
                    zip(range(self.optimizer.n_param_sets), (intermediate_result.fun,))
                )
            )

            self.final_params[:] = np.atleast_2d(intermediate_result.x)

            self.current_iteration += 1

            self.reporter.update(iteration=self.current_iteration)

            if (
                isinstance(self.optimizer, ScipyOptimizer)
                and self.optimizer.method == ScipyMethod.COBYLA
                and intermediate_result.nit + 1 == self.max_iterations
            ):
                raise StopIteration

        self.reporter.info(message="Finished Setup")

        self._initialize_params()
        self._minimize_res = self.optimizer.optimize(
            cost_fn=cost_fn,
            initial_params=self._curr_params,
            callback_fn=_iteration_counter,
            jac=grad_fn,
            maxiter=self.max_iterations,
            rng=self._rng,
        )
        self.final_params[:] = np.atleast_2d(self._minimize_res.x)

        self.reporter.info(message="Finished Optimization!")

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
