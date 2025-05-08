import re
from functools import reduce
from typing import Literal, Optional, get_args
from warnings import warn

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pennylane as qml
import pennylane.qaoa as pqaoa
import rustworkx as rx
import scipy.sparse as sps
import sympy as sp
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.problems import VarType

from divi.circuit import MetaCircuit
from divi.qprog import QuantumProgram
from divi.qprog.optimizers import Optimizers
from divi.utils import convert_qubo_matrix_to_pennylane_ising

GraphProblemTypes = nx.Graph | rx.PyGraph
QUBOProblemTypes = list | np.ndarray | sps.spmatrix | QuadraticProgram

_SUPPORTED_GRAPH_PROBLEMS_LITERAL = Literal[
    "max_clique",
    "max_independent_set",
    "max_weight_cycle",
    "maxcut",
    "min_vertex_cover",
]
_SUPPORTED_PROBLEMS = get_args(_SUPPORTED_GRAPH_PROBLEMS_LITERAL)

_SUPPORTED_INITIAL_STATES_LITERAL = Literal[
    "Zeros", "Ones", "Superposition", "Recommended"
]

# Recommended initial state as per Pennylane's documentation.
# Values are of the format (Constrained, Unconstrained).
# Value is duplicated if not applicable to the problem
_GRAPH_PROBLEM_TO_INITIAL_STATE_MAP = dict(
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


def _convert_quadratic_program_to_pennylane_ising(qp: QuadraticProgram):
    qiskit_sparse_op, constant = qp.to_ising()

    pauli_list = qiskit_sparse_op.paulis

    pennylane_ising = 0.0
    for pauli_string, coeff in zip(pauli_list.z, qiskit_sparse_op.coeffs):
        sanitized_coeff = coeff.real if np.isreal(coeff) else coeff

        curr_term = (
            reduce(
                lambda x, y: x @ y,
                map(lambda x: qml.Z(x), np.flatnonzero(pauli_string)),
            )
            * sanitized_coeff.item()
        )

        pennylane_ising += curr_term

    return pennylane_ising, constant.item(), pauli_list.num_qubits


def _resolve_circuit_layers(
    initial_state, problem, graph_problem, **kwargs
) -> tuple[qml.operation.Operator, qml.operation.Operator, Optional[dict], str]:
    """
    Generates the cost and mixer hamiltonians for a given problem, in addition to
    optional metadata returned by Pennylane if applicable
    """

    if isinstance(problem, GraphProblemTypes):
        is_constrained = kwargs.pop("is_constrained", True)

        if graph_problem in (
            "max_clique",
            "max_independent_set",
            "max_weight_cycle",
            "min_vertex_cover",
        ):
            params = (problem, is_constrained)
        else:
            params = (problem,)

        if initial_state == "Recommended":
            resolved_initial_state = _GRAPH_PROBLEM_TO_INITIAL_STATE_MAP[graph_problem][
                0 if is_constrained else 1
            ]
        else:
            resolved_initial_state = initial_state

        return *getattr(pqaoa, graph_problem)(*params), resolved_initial_state
    else:
        if isinstance(problem, QuadraticProgram):
            cost_hamiltonian, constant, n_qubits = (
                _convert_quadratic_program_to_pennylane_ising(problem)
            )
        else:
            cost_hamiltonian, constant = convert_qubo_matrix_to_pennylane_ising(problem)

            n_qubits = problem.shape[0]

        return (
            cost_hamiltonian,
            pqaoa.x_mixer(range(n_qubits)),
            {"constant": constant},
            "Superposition",
        )


class QAOA(QuantumProgram):
    def __init__(
        self,
        problem: GraphProblemTypes | QUBOProblemTypes,
        graph_problem: Optional[_SUPPORTED_GRAPH_PROBLEMS_LITERAL] = None,
        n_layers: int = 1,
        initial_state: _SUPPORTED_INITIAL_STATES_LITERAL = "Recommended",
        optimizer=Optimizers.MONTE_CARLO,
        max_iterations=10,
        **kwargs,
    ):
        """
        Initialize the QAOA problem.

        Args:
            problem: The problem to solve, can either be a graph or a QUBO.
                For graph inputs, the graph problem to solve must be provided
                through the `graph_problem` variable.
            graph_problem (str): The graph problem to solve.
            n_layers (int): number of QAOA layers
            initial_state (str): The initial state of the circuit
        """

        if isinstance(problem, QUBOProblemTypes):
            if graph_problem is not None:
                warn("Ignoring the 'problem' argument as it is not applicable to QUBO.")

            self.graph_problem = None

            if isinstance(problem, QuadraticProgram):
                if any(var.vartype != VarType.BINARY for var in problem.variables):
                    warn(
                        "Quadratic Program contains non-binary variables. Converting to QUBO."
                    )
                    self._qp_converter = QuadraticProgramToQubo()
                    problem = self._qp_converter.convert(problem)

                self.n_qubits = problem.get_num_vars()
            else:
                if isinstance(problem, list):
                    problem = np.array(problem)

                if problem.ndim != 2 or problem.shape[0] != problem.shape[1]:
                    raise ValueError(
                        "Invalid QUBO matrix."
                        f" Got array of shape {problem.shape}."
                        " Must be a square matrix."
                    )

                self.n_qubits = problem.shape[1]
        else:
            if graph_problem not in _SUPPORTED_PROBLEMS:
                raise ValueError(
                    f"Unsupported Problem. Got '{graph_problem}'. Must be one of: {_SUPPORTED_PROBLEMS}"
                )

            self.graph_problem = graph_problem
            self.n_qubits = problem.number_of_nodes()

        self.problem = problem

        if initial_state not in get_args(_SUPPORTED_INITIAL_STATES_LITERAL):
            raise ValueError(
                f"Unsupported Initial State. Got {initial_state}. Must be one of: {get_args(_SUPPORTED_INITIAL_STATES_LITERAL)}"
            )

        # Local Variables
        self.n_layers = n_layers
        self.optimizer = optimizer
        self.max_iterations = max_iterations
        self.current_iteration = 0
        self._solution_nodes = None
        self.n_params = 2
        self._is_compute_probabilies = False

        # Shared Variables
        self.probs = {}
        if (m_dict_probs := kwargs.pop("probs", None)) is not None:
            self.probs = m_dict_probs

        (
            self.cost_hamiltonian,
            self.mixer_hamiltonian,
            *problem_metadata,
            self.initial_state,
        ) = _resolve_circuit_layers(
            initial_state=initial_state,
            problem=problem,
            graph_problem=graph_problem,
            **kwargs,
        )
        self.problem_metadata = problem_metadata[0] if problem_metadata else {}

        self.loss_constant = self.problem_metadata.get("constant", 0.0)

        kwargs.pop("is_constrained", None)
        super().__init__(**kwargs)

        self._meta_circuits = self._create_meta_circuits_dict()

    @property
    def solution(self):
        return (
            self._solution_nodes
            if self.graph_problem is not None
            else self._solution_bitstring
        )

    @solution.setter
    def solution(self, value):
        raise RuntimeError(
            "The solution property is read-only. Use compute_final_solution() to get the solution."
        )

    def _create_meta_circuits_dict(self) -> dict[str, MetaCircuit]:
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
                return qml.expval(hamiltonian)

        return {
            "cost_circuit": MetaCircuit(
                qml.tape.make_qscript(_prepare_circuit)(
                    self.cost_hamiltonian, sym_params, final_measurement=False
                ),
                symbols=sym_params.flatten(),
                grouping_strategy=self._grouping_strategy,
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

        if isinstance(self.problem, QUBOProblemTypes):
            self._solution_bitstring = np.fromiter(
                best_solution_bitstring, dtype=np.int32
            )

        if isinstance(self.problem, GraphProblemTypes):
            self._solution_nodes = [
                m.start() for m in re.finditer("1", best_solution_bitstring)
            ]

        return self._total_circuit_count, self._total_run_time

    def draw_solution(self):
        if self.graph_problem is None:
            raise RuntimeError(
                "The problem is not a graph problem. Cannot draw solution."
            )

        if not self._solution_nodes:
            self.compute_final_solution()

        # Create a dictionary for node colors
        node_colors = [
            "red" if node in self._solution_nodes else "lightblue"
            for node in self.problem.nodes()
        ]

        plt.figure(figsize=(10, 8))

        pos = nx.spring_layout(self.problem)

        nx.draw_networkx_nodes(self.problem, pos, node_color=node_colors, node_size=500)
        nx.draw_networkx_edges(self.problem, pos)
        nx.draw_networkx_labels(self.problem, pos, font_size=10, font_weight="bold")

        # Remove axes
        plt.axis("off")

        # Show the plot
        plt.tight_layout()
        plt.show()
