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
from divi.services.qoro_service import JobStatus, JobTypes
from divi.simulator.parallel_simulator import ParallelSimulator

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
        Initialize the VQE problem.

        args:
            problem (str): The graph problem to solve.
            graph (networkx.Graph or rustworkx.PyGraph): The graph representing the problem
            n_layers (int): number of QAOA layers
            initial_state (str): The initial state of the circuit
        """
        self.graph = graph

        if problem not in _SUPPORTED_PROBLEMS:
            raise ValueError(
                f"Unsupported Problem. Got {problem}. Must be one of: {_SUPPORTED_PROBLEMS}"
            )
        self.problem = problem

        if initial_state not in get_args(_SUPPPORTED_INITIAL_STATES_LITERAL):
            raise ValueError(
                f"Unsupported Initial State. Got {initial_state}. Must be one of: {get_args(_SUPPPORTED_INITIAL_STATES_LITERAL)}"
            )

        if n_layers < 1 or not isinstance(n_layers, int):
            raise ValueError(
                f"Number of layers should be a positive integer. Got {n_layers}."
            )
        self.n_layers = n_layers

        self.optimizer = optimizer
        self.current_iteration = 0
        self.max_iterations = max_iterations
        self.params = []
        self.num_qubits = graph.number_of_nodes()
        self.circuits = []
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
        self.params = np.random.random(2)

    def _run_optimize(self):
        """
        Run the optimization step for the VQE problem.
        """
        if self.current_iteration == 0:
            self._reset_params()
        else:
            self._optimize()

        self.current_iteration += 1

    def _post_process_results(self, job_id=None, results=None):
        """
        Post-process the results of the VQE problem.

        return:
            (dict) The energies for each bond length, ansatz, and parameter set grouping.
        """

        def process_results(results):
            processed_results = {}
            for r in results:
                processed_results[r["label"]] = r["results"]
            return processed_results

        def expectation_value(results):
            eigenvalue = 0
            total_shots = 0

            for key, val in results.items():
                if key.count("1") % 2 == 1:
                    eigenvalue += -val
                else:
                    eigenvalue += val
                total_shots += val

            return eigenvalue / total_shots

        if job_id is not None and self.qoro_service is not None:
            status = self.qoro_service.job_status(self.job_id, loop_until_complete=True)
            if status != JobStatus.COMPLETED:
                raise Exception(
                    "Job has not completed yet, cannot post-process results"
                )
            results = self.qoro_service.get_job_results(self.job_id)

        results = process_results(results)

        curr_loss = 0.0
        marginal_results = []
        for c in results.keys():
            ham_op = self.cost_hamiltonian[int(c)]
            pair = (
                ham_op,
                marginal_counts(results[c], ham_op.wires.tolist()),
            )
            marginal_results.append(pair)

        for result in marginal_results:
            curr_loss += float(result[0].scalar) * expectation_value(result[1])

        self.losses.append(curr_loss)
        return curr_loss

    def run_iteration(self, store_data=False, data_file=None, type=JobTypes.EXECUTE):
        """
        Run an iteration of the VQE problem. The outputs are stored in the VQE object. Optionally, the data can be stored in a file.

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

    def _generate_circuits(self):
        """
        Generate the circuits for the QAOA problem.

        In this method, we generate bulk circuits based on the selected parameters.
        """

        def qaoa_layer(gamma, alpha):
            pqaoa.cost_layer(gamma, self.cost_hamiltonian)
            pqaoa.mixer_layer(alpha, self.mixer_hamiltonian)

        def _prepare_circuit(hamiltonian_term):
            """
            Prepare the circuit for the VQE problem.
            args:
                hamiltonian (qml.Hamiltonian): The Hamiltonian term to measure
            """

            if self.initial_state == "Ones":
                qml.PauliX(wires=range(self.num_qubits))
            elif self.initial_state == "Superposition":
                qml.Hadamard(wires=range(self.num_qubits))

            qml.layer(
                qaoa_layer, self.n_layers, gamma=self.params[0], alpha=self.params[1]
            )

            return qml.sample(hamiltonian_term)

        for i, term in enumerate(self.cost_hamiltonian):
            qscript = qml.tape.make_qscript(_prepare_circuit)(term)
            self.circuits.append(Circuit(qscript, tag=f"{i}"))

    def run(self, store_data=False, data_file=None, type=JobTypes.EXECUTE):
        """
        Run the VQE problem. The outputs are stored in the VQE object. Optionally, the data can be stored in a file.

        args:
            store_data (bool): Whether to store the data for the iteration
            data_file (str): The file to store the data in
        """

        if self.optimizer == Optimizers.MONTE_CARLO:
            while self.current_iteration < self.max_iterations:
                logger.debug(f"Running iteration {self.current_iteration}")
                self.run_iteration(store_data, data_file, type)

        elif self.optimizer == Optimizers.NELDER_MEAD:

            def cost_function(params, bond_length_index, ansatz):
                self.params[bond_length_index][ansatz] = params
                self._generate_circuits(params)
                results, param = self._prepare_and_send_circuits()
                if param == "job_id":
                    energies = self._post_process_results(job_id=results)
                elif param == "circuit_results":
                    energies = self._post_process_results(results=results)
                self.energies.append(energies)
                return energies[bond_length_index][ansatz][0]

            def optimize_single(args):
                i, ansatz = args
                logger.debug("Running optimization for bond length:", i, ansatz)
                params = self.params[i][ansatz][0]
                result = minimize(
                    cost_function,
                    params,
                    args=(i, ansatz),
                    method="Nelder-Mead",
                    options={"maxiter": self.max_iterations},
                )
                return i, ansatz, result.fun

            self._reset_params()
            num_param_sets = 1
            args = []
            energies = {}
            for i in range(len(self.bond_lengths)):
                energies[i] = {}
                for ansatz in self.ansatze:
                    energies[i][ansatz] = {}
                    num_params = ansatz.num_params(self.num_qubits)
                    self.params[i][ansatz] = [
                        np.random.uniform(0, 2 * np.pi, num_params)
                        for _ in range(num_param_sets)
                    ]
                    args.append((i, ansatz))

            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(optimize_single, arg) for arg in args]
                for future in futures:
                    i, ansatz, energy = future.result()
                    energies[i][ansatz][0] = energy

            return energies

    def _prepare_and_send_circuits(self):
        job_circuits = {}
        for circuit in self.circuits:
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

    def _optimize(self):
        """
        Optimize the VQE problem.
        """
        if self.optimizer == Optimizers.NELDER_MEAD:
            raise NotImplementedError

        elif self.optimizer == Optimizers.MONTE_CARLO:
            losses = self.losses[self.current_iteration - 1]
            breakpoint()
            smallest_loss_keys = sorted(losses, key=lambda k: losses[k])[
                : self.optimizer.samples()
            ]
            new_params = []
            for key in smallest_loss_keys:
                new_param_set = self.optimizer.update_params(
                    self.params[int(key)], self.current_iteration
                )
                new_params.extend(new_param_set)
            self.params = new_params
        else:
            raise NotImplementedError
