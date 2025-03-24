import re
from typing import Literal, get_args

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pennylane as qml
import pennylane.qaoa as pqaoa
import rustworkx as rx
import sympy as sp

from divi.circuit import MetaCircuit
from divi.qprog import QuantumProgram
from divi.qprog.optimizers import Optimizers

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
# Values are of the format (Constrained, Unconstrained).
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
        n_layers: int = 1,
        initial_state: _SUPPORTED_INITIAL_STATES_LITERAL = "Recommended",
        optimizer=Optimizers.MONTE_CARLO,
        max_iterations=10,
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
                f"Unsupported Problem. Got '{problem}'. Must be one of: {_SUPPORTED_PROBLEMS}"
            )
        self.problem = problem

        if initial_state not in get_args(_SUPPORTED_INITIAL_STATES_LITERAL):
            raise ValueError(
                f"Unsupported Initial State. Got {initial_state}. Must be one of: {get_args(_SUPPORTED_INITIAL_STATES_LITERAL)}"
            )

        # Local Variables
        self.n_layers = n_layers
        self.optimizer = optimizer
        self.max_iterations = max_iterations
        self.n_qubits = graph.number_of_nodes()
        self.current_iteration = 0
        self._solution_nodes = None
        self.n_params = 2
        self._is_compute_probabilies = False

        # Shared Variables
        self.losses = []
        if (m_list_losses := kwargs.pop("losses", None)) is not None:
            self.losses = m_list_losses

        self.probs = {}
        if (m_dict_probs := kwargs.pop("probs", None)) is not None:
            self.probs = m_dict_probs

        (
            self.cost_hamiltonian,
            self.mixer_hamiltonian,
            *self.problem_metadata,
            self.initial_state,
        ) = _resolve_circuit_layers(problem, graph, initial_state, **kwargs)

        self.expval_hamiltonian_metadata = {
            i: (term.wires, float(term.scalar))
            for i, term in enumerate(self.cost_hamiltonian)
        }

        self._meta_circuits = self._create_meta_circuits()

        kwargs.pop("is_constrained", None)
        super().__init__(**kwargs)

    def _create_meta_circuits(self):
        """
        Generate the meta circuits for the QAOA problem.

        In this method, we generate the scaffolding for the circuits that will be
        executed during optimization.
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
            "cost_circuit": MetaCircuit(
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

    def _generate_circuits(self):
        """
        Generate the circuits for the QAOA problem.

        In this method, we generate bulk circuits based on the selected parameters.
        """

        circuit_type = (
            "cost_circuit" if not self._is_compute_probabilies else "meas_circuit"
        )

        for p, params_group in enumerate(self._curr_params):
            circuit = self._meta_circuits[circuit_type].initialize_circuit_from_params(
                params_group, tag_prefix=f"{p}"
            )

            self.circuits.append(circuit)

    def _post_process_results(self, results):
        """
        Post-process the results of the QAOA problem.

        Returns:
            (dict) The losses for each parameter set grouping.
        """

        if self._is_compute_probabilies:
            return {
                outer_k: {
                    inner_k: inner_v / self.shots
                    for inner_k, inner_v in outer_v.items()
                }
                for outer_k, outer_v in results.items()
            }

        losses = super()._post_process_results(results)

        return losses

    def _run_final_measurement(self):
        self._is_compute_probabilies = True

        self._curr_params = np.array(self.final_params)

        self.circuits[:] = []

        self._generate_circuits()

        self.probs.update(self._dispatch_circuits_and_process_results())

        self._is_compute_probabilies = False

    def compute_final_solution(self):
        # Convert losses dict to list to apply ordinal operations
        final_losses_list = list(self.losses[-1].values())

        # Get the index of the smallest loss in the last operation
        best_solution_idx = min(
            range(len(final_losses_list)),
            key=lambda x: final_losses_list.__getitem__(x),
        )

        # Insert the measurement circuit here
        self._run_final_measurement()

        # Retrieve the probability distribution dictionary of the best solution
        best_solution_probs = self.probs[f"{best_solution_idx}_0"]

        # Retrieve the bitstring with the actual best solution
        # Reverse to account for the endianness difference
        best_solution_bitstring = max(best_solution_probs, key=best_solution_probs.get)[
            ::-1
        ]

        self._solution_nodes = [
            m.start() for m in re.finditer("1", best_solution_bitstring)
        ]

        return self._total_circuit_count, self._total_run_time

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
