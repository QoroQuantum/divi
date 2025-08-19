# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import logging
import re
from enum import Enum
from functools import reduce
from typing import Literal, get_args
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

from divi.circuits import MetaCircuit
from divi.qprog import QuantumProgram
from divi.qprog.optimizers import Optimizer
from divi.utils import convert_qubo_matrix_to_pennylane_ising

logger = logging.getLogger(__name__)

GraphProblemTypes = nx.Graph | rx.PyGraph
QUBOProblemTypes = list | np.ndarray | sps.spmatrix | QuadraticProgram


def draw_graph_solution_nodes(main_graph, partition_nodes):
    # Create a dictionary for node colors
    node_colors = [
        "red" if node in partition_nodes else "lightblue" for node in main_graph.nodes()
    ]

    plt.figure(figsize=(10, 8))

    pos = nx.spring_layout(main_graph)

    nx.draw_networkx_nodes(main_graph, pos, node_color=node_colors, node_size=500)
    nx.draw_networkx_edges(main_graph, pos)
    nx.draw_networkx_labels(main_graph, pos, font_size=10, font_weight="bold")

    # Remove axes
    plt.axis("off")

    # Show the plot
    plt.tight_layout()
    plt.show()


class GraphProblem(Enum):
    MAX_CLIQUE = ("max_clique", "Zeros", "Superposition")
    MAX_INDEPENDENT_SET = ("max_independent_set", "Zeros", "Superposition")
    MAX_WEIGHT_CYCLE = ("max_weight_cycle", "Superposition", "Superposition")
    MAXCUT = ("maxcut", "Superposition", "Superposition")
    MIN_VERTEX_COVER = ("min_vertex_cover", "Ones", "Superposition")

    # This is an internal problem with no pennylane equivalent
    EDGE_PARTITIONING = ("", "", "")

    def __init__(
        self,
        pl_string: str,
        constrained_initial_state: str,
        unconstrained_initial_state: str,
    ):
        self.pl_string = pl_string

        # Recommended initial state as per Pennylane's documentation.
        # Value is duplicated if not applicable to the problem
        self.constrained_initial_state = constrained_initial_state
        self.unconstrained_initial_state = unconstrained_initial_state


_SUPPORTED_INITIAL_STATES_LITERAL = Literal[
    "Zeros", "Ones", "Superposition", "Recommended"
]


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
) -> tuple[qml.operation.Operator, qml.operation.Operator, dict | None, str]:
    """
    Generates the cost and mixer hamiltonians for a given problem, in addition to
    optional metadata returned by Pennylane if applicable
    """

    if isinstance(problem, GraphProblemTypes):
        is_constrained = kwargs.pop("is_constrained", True)

        if graph_problem == GraphProblem.MAXCUT:
            params = (problem,)
        else:
            params = (problem, is_constrained)

        if initial_state == "Recommended":
            resolved_initial_state = (
                graph_problem.constrained_initial_state
                if is_constrained
                else graph_problem.constrained_initial_state
            )
        else:
            resolved_initial_state = initial_state

        return *getattr(pqaoa, graph_problem.pl_string)(*params), resolved_initial_state
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
        graph_problem: GraphProblem | None = None,
        n_layers: int = 1,
        initial_state: _SUPPORTED_INITIAL_STATES_LITERAL = "Recommended",
        optimizer: Optimizer = Optimizer.MONTE_CARLO,
        max_iterations: int = 10,
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
                if (
                    any(var.vartype != VarType.BINARY for var in problem.variables)
                    or problem.linear_constraints
                    or problem.quadratic_constraints
                ):
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
            if not isinstance(graph_problem, GraphProblem):
                raise ValueError(
                    f"Unsupported Problem. Got '{graph_problem}'. Must be one of type divi.qprog.GraphProblem."
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
        self.n_params = 2
        self._is_compute_probabilites = False

        # Shared Variables
        self.probs = kwargs.pop("probs", {})
        self._solution_nodes = kwargs.pop("solution_nodes", [])
        self._solution_bitstring = kwargs.pop("solution_bitstring", [])

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
            "cost_circuit": self._meta_circuit_factory(
                qml.tape.make_qscript(_prepare_circuit)(
                    self.cost_hamiltonian, sym_params, final_measurement=False
                ),
                symbols=sym_params.flatten(),
            ),
            "meas_circuit": self._meta_circuit_factory(
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
            "cost_circuit" if not self._is_compute_probabilites else "meas_circuit"
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

        if self._is_compute_probabilites:
            return {
                outer_k: {
                    inner_k: inner_v / self.backend.shots
                    for inner_k, inner_v in outer_v.items()
                }
                for outer_k, outer_v in results.items()
            }

        losses = super()._post_process_results(results)

        return losses

    def _run_final_measurement(self):
        self._is_compute_probabilites = True

        self._curr_params = np.array(self.final_params)

        self.circuits[:] = []

        self._generate_circuits()

        self.probs.update(self._dispatch_circuits_and_process_results())

        self._is_compute_probabilites = False

    def compute_final_solution(self):
        """
        Computes and extracts the final solution from the QAOA optimization process.
        This method performs the following steps:
            1. Identifies the best solution index based on the lowest loss value from the last optimization step.
            2. Executes the final measurement circuit to obtain the probability distributions of solutions.
            3. Retrieves the bitstring representing the best solution, correcting for endianness.
            4. Depending on the problem type:
            - For QUBO problems, stores the solution as a NumPy array of bits.
            - For graph problems, stores the solution as a list of node indices corresponding to '1's in the bitstring.
            5. Returns the total circuit count and total runtime for the optimization process.

        Returns:
            tuple: A tuple containing:
            - int: The total number of circuits executed.
            - float: The total runtime of the optimization process.
        """

        if self._progress_queue:
            self._progress_queue.put(
                {
                    "job_id": self.job_id,
                    "message": "🏁 Computing Final Solution 🏁",
                    "progress": 0,
                }
            )
        else:
            logger.info("🏁 Computing Final Solution 🏁")

        # Convert losses dict to list to apply ordinal operations
        final_losses_list = list(self.losses[-1].values())

        # Get the index of the smallest loss in the last operation
        best_solution_idx = min(
            range(len(final_losses_list)),
            key=lambda x: final_losses_list.__getitem__(x),
        )

        # Insert the measurement circuit here
        self._run_final_measurement()

        # Find the key matching the best_solution_idx with possible metadata in between
        pattern = re.compile(rf"^{best_solution_idx}(?:_[^_]*)*_0$")
        matching_keys = [k for k in self.probs.keys() if pattern.match(k)]

        # Some minor sanity checks
        if len(matching_keys) == 0:
            raise RuntimeError("No matching key found.")
        if len(matching_keys) > 1:
            raise RuntimeError(f"More than one matching key found.")

        best_solution_key = matching_keys[0]
        # Retrieve the probability distribution dictionary of the best solution
        best_solution_probs = self.probs[best_solution_key]

        # Retrieve the bitstring with the actual best solution
        # Reverse to account for the endianness difference
        best_solution_bitstring = max(best_solution_probs, key=best_solution_probs.get)[
            ::-1
        ]

        if isinstance(self.problem, QUBOProblemTypes):
            self._solution_bitstring[:] = np.fromiter(
                best_solution_bitstring, dtype=np.int32
            )

        if isinstance(self.problem, GraphProblemTypes):
            self._solution_nodes[:] = [
                m.start() for m in re.finditer("1", best_solution_bitstring)
            ]

        if self._progress_queue:
            self._progress_queue.put(
                {
                    "job_id": self.job_id,
                    "progress": 0,
                    "final_status": "Success",
                }
            )
        else:
            logger.info(f"Computed Solution!")

        return self._total_circuit_count, self._total_run_time

    def draw_solution(self):
        if self.graph_problem is None:
            raise RuntimeError(
                "The problem is not a graph problem. Cannot draw solution."
            )

        if not self._solution_nodes:
            self.compute_final_solution()

        draw_graph_solution_nodes(self.problem, self._solution_nodes)
