import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Literal, get_args

import networkx as nx
import numpy as np
import pennylane as qml
import pennylane.qaoa as pqaoa
import rustworkx as rx
from qiskit.result import marginal_counts
from scipy.optimize import minimize

from divi.circuit import Circuit
from divi.qprog import QuantumProgram
from divi.qprog.optimizers import Optimizers
from divi.qprog.utils import counts_to_expectation_value
from divi.services.qoro_service import JobStatus, JobTypes

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

_SUPPORTED_PROBLEMS_LITERAL = Literal[
    "max_clique",
    "max_independent_set",
    "max_weight_cycle",
    "maxcut",
    "min_vertex_cover",
]
_SUPPORTED_PROBLEMS = get_args(_SUPPORTED_PROBLEMS_LITERAL)

_SUPPPORTED_INITIAL_STATES_LITERAL = Literal[
    "Zeros", "Ones", "Superposition", "Recommended"
]

# Recommended initial state as per Pennylane's documentation.
# Values are of the format (Constrained, Unonstrained).
# Value is duplicated if not applicable to the problem
_PROBLEM_TO_INITIAL_STATE_MAP = dict(
    zip(
        _SUPPORTED_PROBLEMS,
        [
            ("Zeros", "Superposition"),
            ("Zeros", "Superposition"),
            ("Superposition", "Superposition"),
            ("Superposition", "Superposition"),
            ("Ones", "Superposition"),
        ],
    )
)


def _resolve_circuit_layers(problem, graph, initial_state, **kwargs):
    """
    Generates the cost and mixer hamiltonians for a given problem, in addition to
    optional metadata returned by Pennylane if applicable
    """

    if problem in (
        "max_clique",
        "max_independent_set",
        "max_weight_cycle",
        "min_vertex_cover",
    ):
        is_constrained = kwargs.pop("is_constrained", True)
        params = (graph, is_constrained)
    else:
        params = (graph,)

    if initial_state == "Recommended":
        resolved_initial_state = _PROBLEM_TO_INITIAL_STATE_MAP[problem][
            0 if is_constrained else 1
        ]
    else:
        resolved_initial_state = initial_state

    return *getattr(pqaoa, problem)(*params), resolved_initial_state


class QAOA(QuantumProgram):
    def __init__(
        self,
        problem: _SUPPORTED_PROBLEMS_LITERAL,
        graph: nx.Graph | rx.PyGraph,
        n_layers: int,
        initial_state: _SUPPPORTED_INITIAL_STATES_LITERAL = "Recommended",
        optimizer=Optimizers.MONTE_CARLO,
        max_iterations=10,
        shots=5000,
        **kwargs,
    ):
        """
        Initialize the QAOA problem.

        args:
            problem (str): The graph problem to solve.
            graph (networkx.Graph or rustworkx.PyGraph): The graph representing the problem
            n_layers (int): number of QAOA layers
            initial_state (str): The initial state of the circuit
        """
        self.graph = graph

        if problem not in _SUPPORTED_PROBLEMS:
            raise ValueError(
                f"Unsupported Problem. Got {
                    problem}. Must be one of: {_SUPPORTED_PROBLEMS}"
            )
        self.problem = problem

        if initial_state not in get_args(_SUPPPORTED_INITIAL_STATES_LITERAL):
            raise ValueError(
                f"Unsupported Initial State. Got {initial_state}. Must be one of: {
                    get_args(_SUPPPORTED_INITIAL_STATES_LITERAL)}"
            )

        if n_layers < 1 or not isinstance(n_layers, int):
            raise ValueError(
                f"Number of layers should be a positive integer. Got {
                    n_layers}."
            )
        self.n_layers = n_layers

        self.optimizer = optimizer
        self.current_iteration = 0
        self.max_iterations = max_iterations
        self.params = []
        self.num_qubits = graph.number_of_nodes()
        self.shots = shots
        self.losses = []

        (
            self.cost_hamiltonian,
            self.mixer_hamiltonian,
            *self.problem_metadata,
            self.initial_state,
        ) = _resolve_circuit_layers(problem, graph, initial_state, **kwargs)

        kwargs.pop("is_constrained", None)
        super().__init__(**kwargs)

    def _reset_params(self):
        self.params = []

    def _run_optimize(self):
        """
        Run the optimization step for the QAOA problem.
        """
        num_param_sets = self.optimizer.num_param_sets()

        if self.current_iteration == 0:
            self._reset_params()
            self.params = [
                np.random.uniform(0, 2 * np.pi, 2) for _ in range(num_param_sets)
            ]
        else:
            # Optimize the QAOA problem.
            if self.optimizer == Optimizers.NELDER_MEAD:
                raise NotImplementedError

            elif self.optimizer == Optimizers.MONTE_CARLO:
                self.params = self.optimizer.compute_new_parameters(
                    self.params,
                    self.current_iteration,
                    losses=self.losses[self.current_iteration - 1],
                )
            else:
                raise NotImplementedError

        self.current_iteration += 1

    def _post_process_results(self, job_id=None, results=None):
        """
        Post-process the results of the QAOA problem.

        return:
            (dict) The losses for each parameter set grouping.
        """

        def process_results(results):
            processed_results = {}
            for r in results:
                processed_results[r["label"]] = r["results"]
            return processed_results

        if job_id is not None and self.qoro_service is not None:
            status = self.qoro_service.job_status(
                self.job_id, loop_until_complete=True)
            if status != JobStatus.COMPLETED:
                raise Exception(
                    "Job has not completed yet, cannot post-process results"
                )
            results = self.qoro_service.get_job_results(self.job_id)

        results = process_results(results)
        losses = {}

        for p, _ in enumerate(self.params):
            losses[p] = 0
            cur_result = {
                key: value for key, value in results.items() if key.startswith(f"{p}")
            }

            marginal_results = []
            for c in cur_result.keys():
                ham_op_index = int(c.split("_")[-1])
                ham_op = self.cost_hamiltonian[ham_op_index]
                pair = (
                    ham_op,
                    cur_result[c],
                    marginal_counts(cur_result[c], ham_op.wires.tolist()),
                )
                marginal_results.append(pair)

            for result in marginal_results:
                losses[p] += float(result[0].scalar) * counts_to_expectation_value(
                    result[2]
                )

        self.losses.append(losses)

        return losses

    def _generate_circuits(self, params=None):
        """
        Generate the circuits for the QAOA problem.

        In this method, we generate bulk circuits based on the selected parameters.
        """

        def qaoa_layer(gamma, alpha):
            pqaoa.cost_layer(gamma, self.cost_hamiltonian)
            pqaoa.mixer_layer(alpha, self.mixer_hamiltonian)

        def _prepare_circuit(hamiltonian_term, params):
            """
            Prepare the circuit for the QAOA problem.
            args:
                hamiltonian (qml.Hamiltonian): The Hamiltonian term to measure
            """

            if self.initial_state == "Ones":
                qml.PauliX(wires=range(self.num_qubits))
            elif self.initial_state == "Superposition":
                qml.Hadamard(wires=range(self.num_qubits))

            qml.layer(qaoa_layer, self.n_layers,
                      gamma=params[0], alpha=params[1])

            return qml.sample(hamiltonian_term)

        params = self.params if params is None else [params]

        for p, params_group in enumerate(params):
            for i, term in enumerate(self.cost_hamiltonian):
                qscript = qml.tape.make_qscript(
                    _prepare_circuit)(term, params_group)
                self.circuits.append(Circuit(qscript, tag=f"{p}_{i}"))

    def run(self, store_data=False, data_file=None):
        """
        Run the QAOA problem. The outputs are stored in the QAOA object. Optionally, the data can be stored in a file.

        args:
            store_data (bool): Whether to store the data for the iteration
            data_file (str): The file to store the data in
        """

        if self.optimizer == Optimizers.MONTE_CARLO:
            while self.current_iteration < self.max_iterations:
                logger.debug(f"Running iteration {self.current_iteration}")
                self.run_iteration(store_data, data_file)

        elif self.optimizer == Optimizers.NELDER_MEAD:

            def cost_function(params):
                self._generate_circuits(params)
                results, param = self._prepare_and_send_circuits()
                if param == "job_id":
                    losses = self._post_process_results(job_id=results)
                elif param == "circuit_results":
                    losses = self._post_process_results(results=results)

                self.losses.append(losses)

                return losses[0]

            def optimizer_loop_body():
                result = minimize(
                    cost_function,
                    self.params[0],
                    method="Nelder-Mead",
                    options={"maxiter": self.max_iterations},
                )
                return result.fun

            self._reset_params()

            self.params = [
                np.random.uniform(0, 2 * np.pi, 2)
                for _ in range(self.optimizer.num_param_sets())
            ]

            return [optimizer_loop_body()]
