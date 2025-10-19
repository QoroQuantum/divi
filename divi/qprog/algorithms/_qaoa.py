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
from divi.qprog.optimizers import MonteCarloOptimizer, Optimizer
from divi.qprog.variational_quantum_algorithm import VariationalQuantumAlgorithm
from divi.utils import convert_qubo_matrix_to_pennylane_ising

logger = logging.getLogger(__name__)

GraphProblemTypes = nx.Graph | rx.PyGraph
QUBOProblemTypes = list | np.ndarray | sps.spmatrix | QuadraticProgram


def draw_graph_solution_nodes(main_graph: nx.Graph, partition_nodes):
    """Visualize a graph with solution nodes highlighted.

    Draws the graph with nodes colored to distinguish solution nodes (red) from
    other nodes (light blue).

    Args:
        main_graph (nx.Graph): NetworkX graph to visualize.
        partition_nodes: Collection of node indices that are part of the solution.
    """
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
    """Enumeration of supported graph problems for QAOA.

    Each problem type defines:
    - pl_string: The corresponding PennyLane function name
    - constrained_initial_state: Recommended initial state for constrained problems
    - unconstrained_initial_state: Recommended initial state for unconstrained problems
    """

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
        """Initialize the GraphProblem enum value.

        Args:
            pl_string (str): The corresponding PennyLane function name.
            constrained_initial_state (str): Recommended initial state for constrained problems.
            unconstrained_initial_state (str): Recommended initial state for unconstrained problems.
        """
        self.pl_string = pl_string

        # Recommended initial state as per Pennylane's documentation.
        # Value is duplicated if not applicable to the problem
        self.constrained_initial_state = constrained_initial_state
        self.unconstrained_initial_state = unconstrained_initial_state


_SUPPORTED_INITIAL_STATES_LITERAL = Literal[
    "Zeros", "Ones", "Superposition", "Recommended"
]


def _convert_quadratic_program_to_pennylane_ising(qp: QuadraticProgram):
    """Convert a Qiskit QuadraticProgram to a PennyLane Ising Hamiltonian.

    Args:
        qp (QuadraticProgram): Qiskit QuadraticProgram to convert.

    Returns:
        tuple[qml.Hamiltonian, float, int]: (pennylane_ising, constant, n_qubits) where:
            - pennylane_ising: The Ising Hamiltonian in PennyLane format
            - constant: The constant term
            - n_qubits: Number of qubits required
    """
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
    """Generate the cost and mixer Hamiltonians for a given problem.

    Args:
        initial_state (str): The initial state specification.
        problem (GraphProblemTypes | QUBOProblemTypes): The problem to solve (graph or QUBO).
        graph_problem (GraphProblem | None): The graph problem type (if applicable).
        **kwargs: Additional keyword arguments.

    Returns:
        tuple[qml.operation.Operator, qml.operation.Operator, dict | None, str]: (cost_hamiltonian, mixer_hamiltonian, metadata, resolved_initial_state)
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


class QAOA(VariationalQuantumAlgorithm):
    """Quantum Approximate Optimization Algorithm (QAOA) implementation.

    QAOA is a hybrid quantum-classical algorithm designed to solve combinatorial
    optimization problems. It alternates between applying a cost Hamiltonian
    (encoding the problem) and a mixer Hamiltonian (enabling exploration).

    The algorithm can solve:
    - Graph problems (MaxCut, Max Clique, etc.)
    - QUBO (Quadratic Unconstrained Binary Optimization) problems
    - Quadratic programs (converted to QUBO)

    Attributes:
        problem (GraphProblemTypes | QUBOProblemTypes): The problem instance to solve.
        graph_problem (GraphProblem | None): The graph problem type (if applicable).
        n_layers (int): Number of QAOA layers.
        n_qubits (int): Number of qubits required.
        cost_hamiltonian (qml.Hamiltonian): The cost Hamiltonian encoding the problem.
        mixer_hamiltonian (qml.Hamiltonian): The mixer Hamiltonian for exploration.
        initial_state (str): The initial quantum state.
        problem_metadata (dict | None): Additional metadata from problem setup.
        loss_constant (float): Constant term from the problem.
        optimizer (Optimizer): Classical optimizer for parameter updates.
        max_iterations (int): Maximum number of optimization iterations.
        current_iteration (int): Current optimization iteration.
        _n_params (int): Number of parameters per layer (always 2 for QAOA).
        _solution_nodes (list[int] | None): Solution nodes for graph problems.
        _solution_bitstring (np.ndarray | None): Solution bitstring for QUBO problems.
    """

    def __init__(
        self,
        problem: GraphProblemTypes | QUBOProblemTypes,
        *,
        graph_problem: GraphProblem | None = None,
        n_layers: int = 1,
        initial_state: _SUPPORTED_INITIAL_STATES_LITERAL = "Recommended",
        optimizer: Optimizer | None = None,
        max_iterations: int = 10,
        **kwargs,
    ):
        """Initialize the QAOA problem.

        Args:
            problem (GraphProblemTypes | QUBOProblemTypes): The problem to solve, can either be a graph or a QUBO.
                For graph inputs, the graph problem to solve must be provided
                through the `graph_problem` variable.
            graph_problem (GraphProblem | None): The graph problem to solve. Defaults to None.
            n_layers (int): Number of QAOA layers. Defaults to 1.
            initial_state (_SUPPORTED_INITIAL_STATES_LITERAL): The initial state of the circuit. Defaults to "Recommended".
            optimizer (Optimizer | None): The optimizer to use. Defaults to MonteCarloOptimizer.
            max_iterations (int): Maximum number of optimization iterations. Defaults to 10.
            **kwargs: Additional keyword arguments passed to the parent class.
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
        self.max_iterations = max_iterations
        self.current_iteration = 0
        self._n_params = 2
        self.optimizer = optimizer if optimizer is not None else MonteCarloOptimizer()

        self._solution_nodes = []
        self._solution_bitstring = []

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

        if "constant" in self.problem_metadata:
            self.loss_constant = self.problem_metadata.get("constant")
            try:
                self.loss_constant = self.loss_constant.item()
            except AttributeError:
                pass
        else:
            self.loss_constant = 0.0

        kwargs.pop("is_constrained", None)
        super().__init__(**kwargs)

        self._meta_circuits = self._create_meta_circuits_dict()

    @property
    def solution(self):
        """Get the solution found by QAOA optimization.

        Returns:
            list[int] | np.ndarray: For graph problems, returns a list of selected node indices.
                For QUBO problems, returns a list/array of binary values.
        """
        return (
            self._solution_nodes
            if self.graph_problem is not None
            else self._solution_bitstring
        )

    def _create_meta_circuits_dict(self) -> dict[str, MetaCircuit]:
        """Generate the meta circuits for the QAOA problem.

        Creates both cost and measurement circuits for the QAOA algorithm.
        The cost circuit is used during optimization, while the measurement
        circuit is used for final solution extraction.

        Returns:
            dict[str, MetaCircuit]: Dictionary containing cost_circuit and meas_circuit.
        """

        betas = sp.symarray("β", self.n_layers)
        gammas = sp.symarray("γ", self.n_layers)

        sym_params = np.vstack((betas, gammas)).transpose()

        def _qaoa_layer(params):
            gamma, beta = params
            pqaoa.cost_layer(gamma, self.cost_hamiltonian)
            pqaoa.mixer_layer(beta, self.mixer_hamiltonian)

        def _prepare_circuit(hamiltonian, params, final_measurement):
            """Prepare the circuit for the QAOA problem.

            Args:
                hamiltonian (qml.Hamiltonian): The Hamiltonian term to measure.
                params (np.ndarray): The QAOA parameters (betas and gammas).
                final_measurement (bool): Whether to perform final measurement.
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
                grouping_strategy="wires",
            ),
        }

    def _generate_circuits(self):
        """Generate the circuits for the QAOA problem.

        Generates circuits for each parameter set in the current parameters.
        The circuit type depends on whether we're computing probabilities
        (for final solution extraction) or just expectation values (for optimization).
        """

        circuit_type = (
            "cost_circuit" if not self._is_compute_probabilites else "meas_circuit"
        )

        for p, params_group in enumerate(self._curr_params):
            circuit = self._meta_circuits[circuit_type].initialize_circuit_from_params(
                params_group, tag_prefix=f"{p}"
            )

            self._circuits.append(circuit)

    def _post_process_results(self, results):
        """Post-process the results of the QAOA problem.

        Args:
            results (dict[str, dict[str, int]]): Raw results from circuit execution.

        Returns:
            dict[str, dict[str, float]] | dict[int, float]: The losses for each parameter set grouping, or probability
                distributions if computing probabilities.
        """

        if self._is_compute_probabilites:
            return self._process_probability_results(results)

        losses = super()._post_process_results(results)
        return losses

    def _perform_final_computation(self, **kwargs):
        """Extract the optimal solution from the QAOA optimization process.

        This method performs the following steps:
        1. Executes measurement circuits with the best parameters (those that achieved the lowest loss).
        2. Retrieves the bitstring representing the best solution, correcting for endianness.
        3. Depending on the problem type:
           - For QUBO problems, stores the solution as a NumPy array of bits.
           - For graph problems, stores the solution as a list of node indices corresponding to '1's in the bitstring.

        Returns:
            tuple[int, float]: A tuple containing:
                - int: The total number of circuits executed.
                - float: The total runtime of the optimization process.
        """

        self.reporter.info(message="🏁 Computing Final Solution 🏁")

        self._run_solution_measurement()

        best_measurement_probs = next(iter(self._best_probs.values()))

        # Endianness is corrected in _post_process_results
        best_solution_bitstring = max(
            best_measurement_probs, key=best_measurement_probs.get
        )

        if isinstance(self.problem, QUBOProblemTypes):
            self._solution_bitstring[:] = np.fromiter(
                best_solution_bitstring, dtype=np.int32
            )

        if isinstance(self.problem, GraphProblemTypes):
            self._solution_nodes[:] = [
                m.start() for m in re.finditer("1", best_solution_bitstring)
            ]

        return self._total_circuit_count, self._total_run_time

    def draw_solution(self):
        """Visualize the solution found by QAOA for graph problems.

        Draws the graph with solution nodes highlighted in red and other nodes
        in light blue. If the solution hasn't been computed yet, it will be
        calculated first.

        Raises:
            RuntimeError: If called on a QUBO problem instead of a graph problem.

        Note:
            This method only works for graph problems. For QUBO problems, access
            the solution directly via the `solution` property.
        """
        if self.graph_problem is None:
            raise RuntimeError(
                "The problem is not a graph problem. Cannot draw solution."
            )

        if not self._solution_nodes:
            self._perform_final_computation()

        draw_graph_solution_nodes(self.problem, self._solution_nodes)
