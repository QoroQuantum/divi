# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import logging
import pickle
from collections.abc import Callable, Container
from enum import Enum
from typing import Any, Literal, get_args
from warnings import warn

import dimod
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pennylane as qml
import pennylane.qaoa as pqaoa
import sympy as sp

from divi.circuits import CircuitBundle, MetaCircuit
from divi.qprog._hamiltonians import (
    ExactTrotterization,
    TrotterizationStrategy,
    _clean_hamiltonian,
    convert_qubo_matrix_to_pennylane_ising,
)
from divi.qprog.typing import GraphProblemTypes, QUBOProblemTypes, qubo_to_matrix
from divi.qprog.variational_quantum_algorithm import VariationalQuantumAlgorithm

logger = logging.getLogger(__name__)


def _extract_loss_constant(
    problem_metadata: dict, constant_from_hamiltonian: float
) -> float:
    """Extract and combine loss constants from problem metadata and hamiltonian.

    Args:
        problem_metadata: Metadata dictionary that may contain a "constant" key.
        constant_from_hamiltonian: Constant extracted from the hamiltonian.

    Returns:
        Combined loss constant.
    """
    pre_calculated_constant = 0.0
    if "constant" in problem_metadata:
        pre_calculated_constant = problem_metadata.get("constant")
        try:
            pre_calculated_constant = pre_calculated_constant.item()
        except (AttributeError, TypeError):
            # If .item() doesn't exist or fails, ensure it's a float
            pre_calculated_constant = float(pre_calculated_constant)

    return pre_calculated_constant + constant_from_hamiltonian


def draw_graph_solution_nodes(main_graph: nx.Graph, partition_nodes: Container[Any]):
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
        qubo_matrix = qubo_to_matrix(problem)

        cost_hamiltonian, constant = convert_qubo_matrix_to_pennylane_ising(qubo_matrix)

        n_qubits = qubo_matrix.shape[0]

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
    - BinaryQuadraticModel from dimod

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
        _solution_bitstring (npt.NDArray[np.int32] | None): Solution bitstring for QUBO problems.
    """

    def __init__(
        self,
        problem: GraphProblemTypes | QUBOProblemTypes,
        *,
        graph_problem: GraphProblem | None = None,
        initial_state: _SUPPORTED_INITIAL_STATES_LITERAL = "Recommended",
        decode_solution_fn: Callable[[str], Any] | None = None,
        trotterization_strategy: TrotterizationStrategy | None = None,
        max_iterations: int = 10,
        n_layers: int = 1,
        **kwargs,
    ):
        """Initialize the QAOA problem.

        Args:
            problem (GraphProblemTypes | QUBOProblemTypes): The problem to solve, can either be a graph or a QUBO.
                For graph inputs, the graph problem to solve must be provided
                through the `graph_problem` variable.
            graph_problem (GraphProblem | None): The graph problem to solve. Defaults to None.
            initial_state (_SUPPORTED_INITIAL_STATES_LITERAL): The initial state of the circuit. Defaults to "Recommended".
            decode_solution_fn (callable[[str], Any] | None): Optional decoder for bitstrings.
                If not provided, a default decoder is selected based on problem type.
            trotterization_strategy (TrotterizationStrategy | None): The trotterization strategy to use. Defaults to ExactTrotterization.
            max_iterations (int): Maximum number of optimization iterations. Defaults to 10.
            n_layers (int): Number of QAOA layers. Defaults to 1.
            **kwargs: Additional keyword arguments passed to the parent class, including `optimizer`.
        """
        self.graph_problem = graph_problem

        # Validate and process problem (needed to determine decode function)
        # This sets n_qubits which is needed before parent init
        self.problem = self._validate_and_set_problem(problem, graph_problem)

        if decode_solution_fn is not None:
            kwargs["decode_solution_fn"] = decode_solution_fn

        super().__init__(**kwargs)

        # Validate initial state
        if initial_state not in get_args(_SUPPORTED_INITIAL_STATES_LITERAL):
            raise ValueError(
                f"Unsupported Initial State. Got {initial_state}. Must be one of: {get_args(_SUPPORTED_INITIAL_STATES_LITERAL)}"
            )

        if trotterization_strategy is None:
            trotterization_strategy = ExactTrotterization()

        # Initialize local state
        self.n_layers = n_layers
        self.max_iterations = max_iterations
        self.current_iteration = 0
        self.trotterization_strategy = trotterization_strategy
        self._n_params = 2

        self._solution_nodes = []
        self._solution_bitstring = []
        self._cost_circuit_by_hamiltonian: dict[int, MetaCircuit] = {}
        self._hamiltonian_samples: list[qml.operation.Operator] | None = None
        self._curr_ham: qml.operation.Operator | None = None

        # Resolve hamiltonians and problem metadata
        (
            cost_hamiltonian,
            self._mixer_hamiltonian,
            *problem_metadata,
            self.initial_state,
        ) = _resolve_circuit_layers(
            initial_state=initial_state,
            problem=self.problem,
            graph_problem=self.graph_problem,
            **kwargs,
        )
        self.problem_metadata = problem_metadata[0] if problem_metadata else {}

        # Extract and combine constants
        self._cost_hamiltonian, constant_from_hamiltonian = _clean_hamiltonian(
            cost_hamiltonian
        )

        self.loss_constant = _extract_loss_constant(
            self.problem_metadata, constant_from_hamiltonian
        )

        # Extract wire labels from the cost Hamiltonian to ensure consistency
        self._circuit_wires = tuple(self._cost_hamiltonian.wires)

        # Set up decode function based on problem type if user didn't provide one
        if decode_solution_fn is None:
            if isinstance(self.problem, QUBOProblemTypes):
                # For QUBO: convert bitstring to numpy array of int32
                self._decode_solution_fn = lambda bitstring: np.fromiter(
                    bitstring, dtype=np.int32
                )
            elif isinstance(self.problem, GraphProblemTypes):
                # For Graph: map bitstring positions to graph node labels
                circuit_wires = self._circuit_wires  # Capture for closure
                self._decode_solution_fn = lambda bitstring: [
                    circuit_wires[idx]
                    for idx, bit in enumerate(bitstring)
                    if bit == "1" and idx < len(circuit_wires)
                ]

    def _save_subclass_state(self) -> dict[str, Any]:
        """Save QAOA-specific runtime state."""
        return {
            "problem_metadata": self.problem_metadata,
            "solution_nodes": self._solution_nodes,
            "solution_bitstring": self._solution_bitstring,
            "loss_constant": self.loss_constant,
            "trotterization_strategy": pickle.dumps(
                self.trotterization_strategy, protocol=pickle.HIGHEST_PROTOCOL
            ).hex(),
        }

    def _load_subclass_state(self, state: dict[str, Any]) -> None:
        """Load QAOA-specific state.

        Raises:
            KeyError: If any required state key is missing (indicates checkpoint corruption).
        """
        required_keys = [
            "problem_metadata",
            "solution_nodes",  # Key must exist, but value can be None if final computation hasn't run
            "solution_bitstring",  # Key must exist, but value can be None if final computation hasn't run
            "loss_constant",
        ]
        missing_keys = [key for key in required_keys if key not in state]
        if missing_keys:
            raise KeyError(
                f"Corrupted checkpoint: missing required state keys: {missing_keys}"
            )

        self.problem_metadata = state["problem_metadata"]
        # solution_nodes and solution_bitstring can be None if final computation hasn't run
        # Convert None to empty list to match initialization behavior
        self._solution_nodes = (
            state["solution_nodes"] if state["solution_nodes"] is not None else []
        )
        self._solution_bitstring = (
            state["solution_bitstring"]
            if state["solution_bitstring"] is not None
            else []
        )
        self.loss_constant = state["loss_constant"]
        self.trotterization_strategy = pickle.loads(
            bytes.fromhex(state["trotterization_strategy"])
        )

    def _validate_and_set_problem(
        self,
        problem: GraphProblemTypes | QUBOProblemTypes,
        graph_problem: GraphProblem | None,
    ) -> GraphProblemTypes | QUBOProblemTypes:
        """Validate and process the problem input, setting n_qubits and graph_problem.

        Args:
            problem: The problem to solve (graph or QUBO).
            graph_problem: The graph problem type (if applicable).

        Returns:
            The processed problem instance.

        Raises:
            ValueError: If problem type or graph_problem is invalid.
        """
        if isinstance(problem, QUBOProblemTypes):
            if graph_problem is not None:
                warn("Ignoring the 'problem' argument as it is not applicable to QUBO.")

            self.graph_problem = None
            return self._process_qubo_problem(problem)
        else:
            return self._process_graph_problem(problem, graph_problem)

    def _process_qubo_problem(self, problem: QUBOProblemTypes) -> QUBOProblemTypes:
        """Process QUBO problem, converting if necessary and setting n_qubits.

        Args:
            problem: QUBO problem (BinaryQuadraticModel, list, array, or sparse matrix).

        Returns:
            Processed QUBO problem.

        Raises:
            ValueError: If QUBO matrix has invalid shape or BinaryQuadraticModel has non-binary variables.
        """
        # Handle BinaryQuadraticModel
        if isinstance(problem, dimod.BinaryQuadraticModel):
            if problem.vartype != dimod.Vartype.BINARY:
                raise ValueError(
                    f"BinaryQuadraticModel must have vartype='BINARY', got {problem.vartype}"
                )
            self.n_qubits = len(problem.variables)
            return problem

        # Handle list input
        if isinstance(problem, list):
            problem = np.array(problem)

        # Validate matrix shape
        if problem.ndim != 2 or problem.shape[0] != problem.shape[1]:
            raise ValueError(
                "Invalid QUBO matrix."
                f" Got array of shape {problem.shape}."
                " Must be a square matrix."
            )

        self.n_qubits = problem.shape[1]

        return problem

    def _process_graph_problem(
        self,
        problem: GraphProblemTypes,
        graph_problem: GraphProblem | None,
    ) -> GraphProblemTypes:
        """Process graph problem, validating graph_problem and setting n_qubits.

        Args:
            problem: Graph problem (NetworkX or RustworkX graph).
            graph_problem: The graph problem type.

        Returns:
            The graph problem instance.

        Raises:
            ValueError: If graph_problem is not a valid GraphProblem enum.
        """
        if not isinstance(graph_problem, GraphProblem):
            raise ValueError(
                f"Unsupported Problem. Got '{graph_problem}'. Must be one of type divi.qprog.GraphProblem."
            )

        self.graph_problem = graph_problem
        self.n_qubits = problem.number_of_nodes()
        return problem

    @property
    def cost_hamiltonian(self) -> qml.operation.Operator:
        """The cost Hamiltonian for the QAOA problem."""
        return self._cost_hamiltonian

    @property
    def mixer_hamiltonian(self) -> qml.operation.Operator:
        """The mixer Hamiltonian for the QAOA problem."""
        return self._mixer_hamiltonian

    @property
    def solution(self):
        """Get the solution found by QAOA optimization.

        Returns:
            list[int] | npt.NDArray[np.int32]: For graph problems, returns a list of selected node indices.
                For QUBO problems, returns a list/array of binary values.
        """
        return (
            self._solution_nodes
            if self.graph_problem is not None
            else self._solution_bitstring
        )

    def _build_qaoa_ops(self, cost_hamiltonian: qml.operation.Operator) -> list:
        """Build QAOA layer ops for a given cost Hamiltonian."""
        betas = sp.symarray("β", self.n_layers)
        gammas = sp.symarray("γ", self.n_layers)
        sym_params = np.vstack((betas, gammas)).transpose()

        ops = []
        if self.initial_state == "Ones":
            for wire in self._circuit_wires:
                ops.append(qml.PauliX(wires=wire))
        elif self.initial_state == "Superposition":
            for wire in self._circuit_wires:
                ops.append(qml.Hadamard(wires=wire))

        for layer_params in sym_params:
            gamma, beta = layer_params
            ops.append(pqaoa.cost_layer(gamma, cost_hamiltonian))
            ops.append(pqaoa.mixer_layer(beta, self._mixer_hamiltonian))

        return ops

    def _create_meta_circuits_dict(self) -> dict[str, MetaCircuit]:
        """Generate meta circuits for the QAOA problem.

        Uses self._curr_ham if set (temporary, from _generate_circuits), else
        self._cost_hamiltonian. When _curr_ham is set, returns only the circuit
        type needed for the current mode (cost or meas per self._is_compute_probabilities).
        Otherwise returns both cost_circuit and meas_circuit for lazy init.
        """
        ham = self._curr_ham if self._curr_ham is not None else self._cost_hamiltonian
        ops = self._build_qaoa_ops(ham)
        betas = sp.symarray("β", self.n_layers)
        gammas = sp.symarray("γ", self.n_layers)
        sym_params = np.vstack((betas, gammas)).transpose()

        result: dict[str, MetaCircuit] = {}
        build_both = self._curr_ham is None

        if build_both or not self._is_compute_probabilities:
            result["cost_circuit"] = self._meta_circuit_factory(
                qml.tape.QuantumScript(ops=ops, measurements=[qml.expval(ham)]),
                symbols=sym_params.flatten(),
            )
        if build_both or self._is_compute_probabilities:
            result["meas_circuit"] = self._meta_circuit_factory(
                qml.tape.QuantumScript(ops=ops, measurements=[qml.probs()]),
                symbols=sym_params.flatten(),
                grouping_strategy="wires",
            )

        return result

    def _get_cost_circuit_for_hamiltonian(self, ham_id: int) -> MetaCircuit:
        """Return the cost circuit for a given Hamiltonian sample."""
        return self._cost_circuit_by_hamiltonian.get(
            ham_id, self.meta_circuits["cost_circuit"]
        )

    def _generate_circuits(self, **kwargs) -> list[CircuitBundle]:
        """Generate the circuits for the QAOA problem.

        Generates circuits for each parameter set and Hamiltonian sample.
        Single-sample uses [self._cost_hamiltonian]; multi-sample QDrift uses
        multiple samples. Same interface for both.
        """
        if self._hamiltonian_samples is None:
            raise RuntimeError(
                "_hamiltonian_samples must be set before _generate_circuits; "
                "call _run_optimization_circuits or _run_solution_measurement first."
            )

        use_meas = self._is_compute_probabilities

        circuit_bundles: list[CircuitBundle] = []
        try:
            for ham_id, sample_hamiltonian in enumerate(self._hamiltonian_samples):
                self._curr_ham = sample_hamiltonian
                circuits = self._create_meta_circuits_dict()
                circuit = circuits["meas_circuit" if use_meas else "cost_circuit"]

                if not use_meas:
                    self._cost_circuit_by_hamiltonian[ham_id] = circuit

                for p, params_group in enumerate(self._curr_params):
                    bundle = circuit.initialize_circuit_from_params(
                        params_group, param_idx=p, hamiltonian_id=ham_id
                    )
                    circuit_bundles.append(bundle)
        finally:
            self._curr_ham = None
            self._hamiltonian_samples = None

        return circuit_bundles

    def _run_optimization_circuits(self, **kwargs) -> dict[int, float]:
        strategy = self.trotterization_strategy
        n_samples = getattr(strategy, "n_hamiltonians_per_iteration", 1)

        if n_samples > 1:
            if self.backend.supports_expval:
                raise ValueError(
                    "Multi-sample QDrift is not supported with backends that compute "
                    "expectation values directly; each Hamiltonian sample requires "
                    "different observables. Use a shot-based backend instead."
                )
            self._cost_circuit_by_hamiltonian.clear()

        self._hamiltonian_samples = [
            strategy.process_hamiltonian(self._cost_hamiltonian)
            for _ in range(n_samples)
        ]

        if strategy.stateful:
            # Invalidate meta-circuits to force rebuild
            self._meta_circuits = None

        return super()._run_optimization_circuits(**kwargs)

    def _run_solution_measurement(self) -> None:
        """Execute measurement circuits, sampling Hamiltonians when using multi-sample QDrift."""
        strategy = self.trotterization_strategy
        n_samples = getattr(strategy, "n_hamiltonians_per_iteration", 1)
        self._hamiltonian_samples = [
            strategy.process_hamiltonian(self._cost_hamiltonian)
            for _ in range(n_samples)
        ]
        super()._run_solution_measurement()

    def _perform_final_computation(self, **kwargs):
        """Extract the optimal solution from the QAOA optimization process.

        This method performs the following steps:
        1. Executes measurement circuits with the best parameters (those that achieved the lowest loss).
        2. Retrieves the bitstring representing the best solution, correcting for endianness.
        3. Uses the `decode_solution_fn` (configured in constructor based on problem type) to decode
           the bitstring into the appropriate format:
           - For QUBO problems: NumPy array of bits (int32).
           - For graph problems: List of node indices corresponding to '1's in the bitstring.
        4. Stores the decoded solution in the appropriate attribute.

        Returns:
            tuple[int, float]: A tuple containing:
                - int: The total number of circuits executed.
                - float: The total runtime of the optimization process.
        """

        self.reporter.info(message="🏁 Computing Final Solution 🏁", overwrite=True)

        self._run_solution_measurement()

        best_measurement_probs = next(iter(self._best_probs.values()))

        # Endianness is corrected in _post_process_results
        best_solution_bitstring = max(
            best_measurement_probs, key=best_measurement_probs.get
        )

        # Use decode function to get the decoded solution
        decoded_solution = self._decode_solution_fn(best_solution_bitstring)

        # Store in appropriate attribute based on problem type
        if isinstance(self.problem, QUBOProblemTypes):
            self._solution_bitstring[:] = decoded_solution
        elif isinstance(self.problem, GraphProblemTypes):
            self._solution_nodes[:] = decoded_solution

        self.reporter.info(message="🏁 Computed Final Solution! 🏁")

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
