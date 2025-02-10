import logging
import re
from typing import Literal, get_args

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pennylane as qml
import pennylane.qaoa as pqaoa
import rustworkx as rx
import sympy as sp
from qiskit.result import marginal_counts, sampled_expectation_value
from scipy.optimize import minimize

from divi.circuit import MetaCircuit
from divi.qprog import QuantumProgram
from divi.qprog.optimizers import Optimizers

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

_SUPPORTED_INITIAL_STATES_LITERAL = Literal[
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

    is_constrained = kwargs.pop("is_constrained", True)

    if problem in (
        "max_clique",
        "max_independent_set",
        "max_weight_cycle",
        "min_vertex_cover",
    ):
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
        initial_state: _SUPPORTED_INITIAL_STATES_LITERAL = "Recommended",
        optimizer=Optimizers.MONTE_CARLO,
        max_iterations=10,
        shots=5000,
        **kwargs,
    ):
        """
        Initialize the QAOA problem.

        Args:
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

        if initial_state not in get_args(_SUPPORTED_INITIAL_STATES_LITERAL):
            raise ValueError(
                f"Unsupported Initial State. Got {initial_state}. Must be one of: {get_args(_SUPPORTED_INITIAL_STATES_LITERAL)}"
            )

        if n_layers < 1 or not isinstance(n_layers, int):
            raise ValueError(
                f"Number of layers should be a positive integer. Got {n_layers}."
            )

        # Local Variables
        self.n_layers = n_layers
        self.optimizer = optimizer
        self.shots = shots
        self.max_iterations = max_iterations
        self.n_qubits = graph.number_of_nodes()
        self.current_iteration = 0
        self.params = []
        self._solution_nodes = None

        # Shared Variables
        self.losses = []
        if (m_list_losses := kwargs.pop("losses", None)) is not None:
            self.losses = m_list_losses

        self.probs = []
        if (m_list_probs := kwargs.pop("probs", None)) is not None:
            self.probs = m_list_probs

        (
            self.cost_hamiltonian,
            self.mixer_hamiltonian,
            *self.problem_metadata,
            self.initial_state,
        ) = _resolve_circuit_layers(problem, graph, initial_state, **kwargs)

        self._meta_circuits = self._create_meta_circuits()

        kwargs.pop("is_constrained", None)
        super().__init__(**kwargs)

    def _reset_params(self):
        self.params = []

    def _run_optimize(self):
        """
        Run the optimization step for the QAOA problem.
        """
        n_param_sets = self.optimizer.n_param_sets

        if self.current_iteration == 0:
            self._reset_params()
            self.params = [
                np.random.uniform(0, 2 * np.pi, self.n_layers * 2)
                for _ in range(n_param_sets)
            ]
        else:
            # Optimize the QAOA problem.
            if self.optimizer == Optimizers.MONTE_CARLO:
                self.params = self.optimizer.compute_new_parameters(
                    self.params,
                    self.current_iteration,
                    losses=self.losses[self.current_iteration - 1],
                )
            else:
                raise NotImplementedError

        self.current_iteration += 1

    def _post_process_results(self, results):
        """
        Post-process the results of the QAOA problem.

        Returns:
            (dict) The losses for each parameter set grouping.
        """

        losses = {}
        if self._is_compute_probabilies:
            probs = {
                outer_k: {
                    inner_k: inner_v / self.shots
                    for inner_k, inner_v in outer_v.items()
                }
                for outer_k, outer_v in results.items()
            }

        for p, _ in enumerate(self.params):
            if self._is_compute_probabilies:
                break

            losses[p] = 0
            cur_result = {
                key: value for key, value in results.items() if key.startswith(f"{p}")
            }

            marginal_results = []
            for param_id, shots_dict in cur_result.items():
                ham_op_index = int(param_id.split("_")[-1])
                ham_op = self.cost_hamiltonian[ham_op_index]
                pair = (
                    ham_op,
                    marginal_counts(shots_dict, ham_op.wires.tolist()),
                )
                marginal_results.append(pair)

            for ham_op, marginal_shots in marginal_results:
                exp_value = sampled_expectation_value(
                    marginal_shots, "Z" * len(ham_op.wires)
                )
                losses[p] += float(ham_op.scalar) * exp_value

        if self._is_compute_probabilies:
            self.probs.append(probs)
        else:
            self.losses.append(losses)

        return losses

    def _create_meta_circuits(self):
        """
        Generate the meta circuits for the QAOA problem.

        In this method, we generate bulk circuits based on the selected parameters.
        """

        betas = sp.symarray("β", self.n_layers)
        gammas = sp.symarray("γ", self.n_layers)

        sym_params = np.vstack((betas, gammas)).transpose()

        def _qaoa_layer(params):
            gamma, beta = params
            pqaoa.cost_layer(gamma, self.cost_hamiltonian)
            pqaoa.mixer_layer(beta, self.mixer_hamiltonian)

        def _prepare_circuit(hamiltonian, params, final_measurement):
            """
            Prepare the circuit for the QAOA problem.
            Args:
                hamiltonian (qml.Hamiltonian): The Hamiltonian term to measure
            """

            # Note: could've been done as qml.[Insert Gate](wires=range(self.n_qubits))
            # but there seems to be a bug with program capture in Pennylane.
            # Maybe check when a new version comes out?
            if self.initial_state == "Ones":
                for i in range(self.n_qubits):
                    qml.PauliX(wires=i)
            elif self.initial_state == "Superposition":
                for i in range(self.n_qubits):
                    qml.Hadamard(wires=i)

            qml.layer(_qaoa_layer, self.n_layers, params)

            if final_measurement:
                return qml.probs()
            else:
                return [qml.sample(term) for term in hamiltonian]

        return {
            "opt_circuit": MetaCircuit(
                qml.tape.make_qscript(_prepare_circuit)(
                    self.cost_hamiltonian, sym_params, final_measurement=False
                ),
                symbols=sym_params.flatten(),
            ),
            "meas_circuit": MetaCircuit(
                qml.tape.make_qscript(_prepare_circuit)(
                    self.cost_hamiltonian, sym_params, final_measurement=True
                ),
                symbols=sym_params.flatten(),
            ),
        }

    def _generate_circuits(self, params=None, **kwargs):
        """
        Generate the circuits for the QAOA problem.

        In this method, we generate bulk circuits based on the selected parameters.
        """

        # Clear the previous circuit batch
        self.circuits[:] = []

        circuit_type = (
            "opt_circuit"
            if not kwargs.pop("measurement_phase", False)
            else "meas_circuit"
        )

        params = self.params if params is None else [params]

        for p, params_group in enumerate(params):
            circuit = self._meta_circuits[circuit_type].initialize_circuit_from_params(
                params_group, tag_prefix=f"{p}"
            )

            self.circuits.append(circuit)

    def run(self, store_data=False, data_file=None):
        """
        Run the QAOA problem. The outputs are stored in the QAOA object. Optionally, the data can be stored in a file.

        Args:
            store_data (bool): Whether to store the data for the iteration
            data_file (str): The file to store the data in
        """

        if self.optimizer == Optimizers.MONTE_CARLO:
            while self.current_iteration < self.max_iterations:
                logger.debug(f"Running iteration {self.current_iteration}")

                self._run_optimize()

                self._is_compute_probabilies = False
                self._generate_circuits(final_measurement=False)
                self._dispatch_circuits_and_process_results(
                    store_data=store_data, data_file=data_file
                )

                self._is_compute_probabilies = True
                self._generate_circuits(final_measurement=True)
                self._dispatch_circuits_and_process_results(
                    store_data=store_data, data_file=data_file
                )

            return self.total_circuit_count

        elif self.optimizer == Optimizers.NELDER_MEAD:

            def cost_function(params):
                self._is_compute_probabilies = False
                self._generate_circuits(params, measurement_phase=False)
                losses = self._dispatch_circuits_and_process_results(
                    store_data=store_data, data_file=data_file
                )

                self.losses.append(losses)

                self._is_compute_probabilies = True
                self._generate_circuits(params, measurement_phase=True)
                self._dispatch_circuits_and_process_results(
                    store_data=store_data, data_file=data_file
                )

                return losses[0]

            self._reset_params()

            self.params = [
                np.random.uniform(0, 2 * np.pi, self.n_layers * 2)
                for _ in range(self.optimizer.n_param_sets)
            ]

            minimize(
                cost_function,
                self.params[0],
                method="Nelder-Mead",
                options={"maxiter": self.max_iterations},
            )

            return self.total_circuit_count

    def compute_final_solution(self):
        # Convert losses dict to list to apply ordinal operations
        final_losses_list = list(self.losses[-1].values())

        # Get the index of the smallest loss in the last operation
        best_solution_idx = min(
            range(len(final_losses_list)),
            key=lambda x: final_losses_list.__getitem__(x),
        )

        # Retrieve the probability distribution dictionary of the best solution
        best_solution_probs = self.probs[-1][f"{best_solution_idx}_0"]

        # Retrieve the bitstring with the actual best solution
        # Reverse to account for the endianness difference
        best_solution_bitstring = max(best_solution_probs, key=best_solution_probs.get)[
            ::-1
        ]

        self._solution_nodes = [
            m.start() for m in re.finditer("1", best_solution_bitstring)
        ]

        return self._solution_nodes

    def draw_solution(self):
        if not self._solution_nodes:
            self.compute_final_solution()

        # Create a dictionary for node colors
        node_colors = [
            "red" if node in self._solution_nodes else "lightblue"
            for node in self.graph.nodes()
        ]

        plt.figure(figsize=(10, 8))

        pos = nx.spring_layout(self.graph)

        nx.draw_networkx_nodes(self.graph, pos, node_color=node_colors, node_size=500)
        nx.draw_networkx_edges(self.graph, pos)
        nx.draw_networkx_labels(self.graph, pos, font_size=10, font_weight="bold")

        # Remove axes
        plt.axis("off")

        # Show the plot
        plt.tight_layout()
        plt.show()
