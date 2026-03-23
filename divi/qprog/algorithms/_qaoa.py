# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import logging
import pickle
from collections.abc import Callable, Container
from enum import Enum
from typing import Any, Literal
from warnings import warn

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pennylane as qml
import pennylane.qaoa as pqaoa
import sympy as sp

from divi.circuits import MetaCircuit
from divi.hamiltonians import (
    ExactTrotterization,
    IsingEncoding,
    TrotterizationStrategy,
    _clean_hamiltonian,
    _is_empty_hamiltonian,
    _resolve_ising_converter,
    normalize_binary_polynomial_problem,
)
from divi.pipeline.stages import TrotterSpecStage
from divi.qprog.algorithms._initial_state import (
    InitialState,
    OnesState,
    SuperpositionState,
    WState,
    ZerosState,
    build_block_xy_mixer_graph,
)
from divi.qprog.variational_quantum_algorithm import (
    VariationalQuantumAlgorithm,
    _extract_param_set_idx,
)
from divi.typing import (
    BinaryPolynomialProblem,
    GraphProblemTypes,
    HUBOProblemTypes,
    QUBOProblemTypes,
)

logger = logging.getLogger(__name__)


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

    Each variant stores its PennyLane function name and the recommended
    :class:`InitialState` classes for constrained / unconstrained modes.
    """

    MAX_CLIQUE = ("max_clique", ZerosState, SuperpositionState)
    MAX_INDEPENDENT_SET = ("max_independent_set", ZerosState, SuperpositionState)
    MAX_WEIGHT_CYCLE = ("max_weight_cycle", SuperpositionState, SuperpositionState)
    MAXCUT = ("maxcut", SuperpositionState, SuperpositionState)
    MIN_VERTEX_COVER = ("min_vertex_cover", OnesState, SuperpositionState)

    # Internal problem with no PennyLane equivalent
    EDGE_PARTITIONING = ("", SuperpositionState, SuperpositionState)

    def __init__(
        self,
        pl_string: str,
        constrained_state_cls: type[InitialState],
        unconstrained_state_cls: type[InitialState],
    ):
        self.pl_string = pl_string
        self._constrained_state_cls = constrained_state_cls
        self._unconstrained_state_cls = unconstrained_state_cls

    def default_initial_state(self, *, is_constrained: bool) -> InitialState:
        """Return the recommended initial state for this problem."""
        cls = (
            self._constrained_state_cls
            if is_constrained
            else self._unconstrained_state_cls
        )
        return cls()

    def resolve(
        self,
        problem: GraphProblemTypes,
        *,
        is_constrained: bool,
    ) -> tuple[qml.operation.Operator, qml.operation.Operator, dict, InitialState]:
        """Return (cost_ham, mixer_ham, metadata, initial_state) for a graph problem."""
        params = (
            (problem,) if self == GraphProblem.MAXCUT else (problem, is_constrained)
        )
        result = getattr(pqaoa, self.pl_string)(*params)
        cost_ham, mixer_ham = result[0], result[1]
        metadata = result[2] if len(result) > 2 else {}
        return (
            cost_ham,
            mixer_ham,
            metadata,
            self.default_initial_state(is_constrained=is_constrained),
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
        problem: GraphProblemTypes | QUBOProblemTypes | HUBOProblemTypes,
        *,
        graph_problem: GraphProblem | None = None,
        initial_state: InitialState | None = None,
        decode_solution_fn: Callable[[str], Any] | None = None,
        hamiltonian_builder: Literal["native", "quadratized"] = "native",
        quadratization_strength: float = 10.0,
        trotterization_strategy: TrotterizationStrategy | None = None,
        max_iterations: int = 10,
        n_layers: int = 1,
        **kwargs,
    ):
        """Initialize the QAOA problem.

        Args:
            problem (GraphProblemTypes | QUBOProblemTypes | HUBOProblemTypes): The problem to solve, can either be a graph or a binary polynomial optimization problem.
                For graph inputs, the graph problem to solve must be provided
                through the `graph_problem` variable.
            graph_problem (GraphProblem | None): The graph problem to solve. Defaults to None.
            initial_state (InitialState | None): Initial quantum state preparation.
                Pass an :class:`InitialState` subclass instance (e.g.
                ``SuperpositionState()``, ``WState(block_size, n_blocks)``).
                If ``None`` (default), graph problems use the recommended state
                per problem type; QUBO/HUBO problems default to
                ``SuperpositionState()``. When ``WState`` is passed, the mixer
                is automatically set to the XY mixer.
            decode_solution_fn (callable[[str], Any] | None): Optional decoder for bitstrings.
                If not provided, a default decoder is selected based on problem type.
            hamiltonian_builder: Hamiltonian conversion backend for binary polynomial
                problems. Accepts "native" or "quadratized".
            quadratization_strength: Penalty strength used when
                `hamiltonian_builder="quadratized"`.
            trotterization_strategy (TrotterizationStrategy | None): The trotterization strategy to use. Defaults to ExactTrotterization.
            max_iterations (int): Maximum number of optimization iterations. Defaults to 10.
            n_layers (int): Number of QAOA layers. Defaults to 1.
            **kwargs: Additional keyword arguments passed to the parent class, including `optimizer`.
        """
        if initial_state is not None and not isinstance(initial_state, InitialState):
            raise TypeError(
                f"initial_state must be an InitialState instance or None, "
                f"got {type(initial_state).__name__}"
            )

        self.graph_problem = graph_problem
        self._ising_encoding: IsingEncoding | None = None

        # Validate and process problem (needed to determine decode function)
        # This sets n_qubits which is needed before parent init
        self.problem = self._validate_and_set_problem(
            problem,
            graph_problem,
            hamiltonian_builder=hamiltonian_builder,
            quadratization_strength=quadratization_strength,
        )

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
        if isinstance(self.problem, GraphProblemTypes):
            (
                cost_hamiltonian,
                self._mixer_hamiltonian,
                self.problem_metadata,
                default_state,
            ) = self.graph_problem.resolve(
                self.problem,
                is_constrained=kwargs.get("is_constrained", True),
            )
            self.initial_state = (
                initial_state if initial_state is not None else default_state
            )
        else:
            # QUBO / HUBO problems
            if self._ising_encoding is None:
                raise ValueError(
                    "Missing Ising encoding for binary polynomial problem."
                )
            cost_hamiltonian = self._ising_encoding.operator

            if initial_state is None:
                initial_state = SuperpositionState()
            self.initial_state = initial_state

            # Auto-select mixer: XY for WState, X otherwise
            if isinstance(self.initial_state, WState):
                graph = build_block_xy_mixer_graph(
                    self.initial_state.block_size,
                    self.initial_state.n_blocks,
                    range(self.n_qubits),
                )
                self._mixer_hamiltonian = pqaoa.xy_mixer(graph)
            else:
                self._mixer_hamiltonian = pqaoa.x_mixer(range(self.n_qubits))
        if not isinstance(self.problem, GraphProblemTypes):
            self.problem_metadata = self._ising_encoding.metadata or {}

        # Extract and combine constants
        self._cost_hamiltonian, constant_from_hamiltonian = _clean_hamiltonian(
            cost_hamiltonian
        )
        if _is_empty_hamiltonian(self._cost_hamiltonian):
            raise ValueError("Hamiltonian contains only constant terms.")

        if self._ising_encoding is not None:
            self.loss_constant = (
                self._ising_encoding.constant + constant_from_hamiltonian
            )
        else:
            self.loss_constant = constant_from_hamiltonian

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
            if self.graph_problem is None:
                # Binary polynomial problems decode via the converter's decode_fn.
                self._decode_solution_fn = self._ising_encoding.decode_fn
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
        problem: GraphProblemTypes | QUBOProblemTypes | HUBOProblemTypes,
        graph_problem: GraphProblem | None,
        *,
        hamiltonian_builder: Literal["native", "quadratized"],
        quadratization_strength: float,
    ) -> GraphProblemTypes | BinaryPolynomialProblem:
        """Validate and process the problem input, setting n_qubits and graph_problem.

        Args:
            problem: The problem to solve (graph or binary polynomial optimization).
            graph_problem: The graph problem type (if applicable).

        Returns:
            The processed problem instance.

        Raises:
            ValueError: If problem type or graph_problem is invalid.
        """
        if isinstance(problem, GraphProblemTypes):
            return self._process_graph_problem(problem, graph_problem)
        else:
            if graph_problem is not None:
                warn(
                    "Ignoring the 'graph_problem' argument as it is not applicable to binary polynomial inputs."
                )

            self.graph_problem = None
            return self._process_binary_problem(
                problem,
                hamiltonian_builder=hamiltonian_builder,
                quadratization_strength=quadratization_strength,
            )

    def _process_binary_problem(
        self,
        problem: QUBOProblemTypes | HUBOProblemTypes,
        *,
        hamiltonian_builder: Literal["native", "quadratized"],
        quadratization_strength: float,
    ) -> BinaryPolynomialProblem:
        """Normalize binary optimization input and convert to Ising."""
        canonical_problem = normalize_binary_polynomial_problem(problem)
        converter = _resolve_ising_converter(
            hamiltonian_builder, quadratization_strength=quadratization_strength
        )
        self._ising_encoding = converter.convert(canonical_problem)
        self.n_qubits = len(self._ising_encoding.operator.wires)

        return canonical_problem

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
            For graph problems, a list of selected node indices.
            For QUBO problems, a list of binary values.
            For HUBO problems with non-integer variable names, a dictionary
            mapping variable names to binary values.
        """
        if self.graph_problem is not None:
            return self._solution_nodes
        if isinstance(self.problem, BinaryPolynomialProblem):
            vo = self.problem.variable_order
            if vo != tuple(range(self.problem.n_vars)):
                return dict(zip(vo, self._solution_bitstring))
        return self._solution_bitstring

    def _build_qaoa_ops(self, cost_hamiltonian: qml.operation.Operator) -> list:
        """Build QAOA layer ops for a given cost Hamiltonian."""
        ops = self.initial_state.build(self._circuit_wires)

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

        # Decode the best bitstring via the problem-specific decode function
        decoded_solution = self._decode_solution_fn(best_solution_bitstring)

        if isinstance(self.problem, GraphProblemTypes):
            self._solution_nodes[:] = decoded_solution
        elif decoded_solution is not None:
            try:
                self._solution_bitstring[:] = decoded_solution
            except (TypeError, ValueError):
                # decode returned a non-array type (e.g. tour list) — store raw bitstring
                self._solution_bitstring = np.array(
                    [int(b) for b in best_solution_bitstring], dtype=np.int32
                )
        else:
            # decode returned None (infeasible) — store raw bitstring
            self._solution_bitstring = np.array(
                [int(b) for b in best_solution_bitstring], dtype=np.int32
            )

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
