# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import logging
import pickle
from collections.abc import Callable, Container
from enum import Enum
from typing import Any
from warnings import warn

import dimod
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pennylane as qml
import pennylane.qaoa as pqaoa
import sympy as sp

from divi.circuits import MetaCircuit
from divi.hamiltonians import (
    ExactTrotterization,
    TrotterizationStrategy,
    _clean_hamiltonian,
    _is_empty_hamiltonian,
    convert_qubo_matrix_to_pennylane_ising,
)
from divi.pipeline.stages import TrotterSpecStage
from divi.qprog.algorithms._initial_state import (
    build_initial_state_ops,
    validate_initial_state,
)
from divi.qprog.variational_quantum_algorithm import (
    VariationalQuantumAlgorithm,
    _extract_param_set_idx,
)
from divi.typing import GraphProblemTypes, QUBOProblemTypes, qubo_to_matrix

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
        _n_params_per_layer (int): Number of parameters per layer (always 2 for QAOA).
        _solution_nodes (list[int] | None): Solution nodes for graph problems.
        _solution_bitstring (npt.NDArray[np.int32] | None): Solution bitstring for QUBO problems.
    """

    def __init__(
        self,
        problem: GraphProblemTypes | QUBOProblemTypes,
        *,
        graph_problem: GraphProblem | None = None,
        initial_state: str = "Recommended",
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
            initial_state (str): The initial state of the circuit. Defaults to "Recommended".
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

        if trotterization_strategy is None:
            trotterization_strategy = ExactTrotterization()

        # Initialize local state
        self.n_layers = n_layers
        self.max_iterations = max_iterations
        self.current_iteration = 0
        self.trotterization_strategy = trotterization_strategy
        self._n_params_per_layer = 2

        self._solution_nodes = []
        self._solution_bitstring = []
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

        # Validate the *resolved* initial state ("Recommended" has been
        # mapped to a concrete value by _resolve_circuit_layers).
        validate_initial_state(self.initial_state, self.n_qubits)
        self.problem_metadata = problem_metadata[0] if problem_metadata else {}

        # Extract and combine constants
        self._cost_hamiltonian, constant_from_hamiltonian = _clean_hamiltonian(
            cost_hamiltonian
        )
        if _is_empty_hamiltonian(self._cost_hamiltonian):
            raise ValueError("Hamiltonian contains only constant terms.")

        self.loss_constant = _extract_loss_constant(
            self.problem_metadata, constant_from_hamiltonian
        )

        # Extract wire labels from the cost Hamiltonian to ensure consistency
        self._circuit_wires = tuple(self._cost_hamiltonian.wires)

        # Cache symbolic parameters for the ansatz (used by meta circuit factory)
        betas = sp.symarray("β", self.n_layers)
        gammas = sp.symarray("γ", self.n_layers)
        self._sym_params = np.vstack((betas, gammas)).transpose()

        # Build pipelines

        self._build_pipelines()

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

    def _build_pipelines(self) -> None:
        self._cost_pipeline = self._build_cost_pipeline(
            TrotterSpecStage(
                trotterization_strategy=self.trotterization_strategy,
                meta_circuit_factory=self._cost_meta_circuit_factory,
            )
        )
        self._measurement_pipeline = self._build_measurement_pipeline()

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
        ops = build_initial_state_ops(self.initial_state, self._circuit_wires)

        for layer_params in self._sym_params:
            gamma, beta = layer_params
            ops.append(pqaoa.cost_layer(gamma, cost_hamiltonian))
            ops.append(pqaoa.mixer_layer(beta, self._mixer_hamiltonian))

        return ops

    def _cost_meta_circuit_factory(
        self, processed_ham: qml.operation.Operator, ham_id: int
    ) -> MetaCircuit:
        """Build a cost MetaCircuit for a given (possibly QDrift-sampled) Hamiltonian."""
        return MetaCircuit(
            source_circuit=qml.tape.QuantumScript(
                ops=self._build_qaoa_ops(processed_ham),
                measurements=[qml.expval(processed_ham)],
            ),
            symbols=self._sym_params.flatten(),
            precision=self._precision,
        )

    def _create_meta_circuit_factories(self) -> dict[str, MetaCircuit]:
        """Generate meta-circuit factories for the QAOA problem."""
        ops = self._build_qaoa_ops(self._cost_hamiltonian)

        return {
            "cost_circuit": MetaCircuit(
                source_circuit=qml.tape.QuantumScript(
                    ops=ops, measurements=[qml.expval(self._cost_hamiltonian)]
                ),
                symbols=self._sym_params.flatten(),
                precision=self._precision,
            ),
            "meas_circuit": MetaCircuit(
                source_circuit=qml.tape.QuantumScript(
                    ops=ops, measurements=[qml.probs()]
                ),
                symbols=self._sym_params.flatten(),
                precision=self._precision,
            ),
        }

    def _run_optimization_circuits(self, **kwargs) -> dict[int, float]:
        """Run cost evaluation via the pipeline."""

        env = self._build_pipeline_env()
        result = self._cost_pipeline.run(
            initial_spec=self._cost_hamiltonian,
            env=env,
        )
        self._total_circuit_count += env.artifacts.get("circuit_count", 0)
        self._total_run_time += env.artifacts.get("run_time", 0.0)
        self._current_execution_result = env.artifacts.get("_current_execution_result")

        return {
            _extract_param_set_idx(key): value + self.loss_constant
            for key, value in result.items()
        }

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

        # Endianness is corrected in the pipeline's format dispatch
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
