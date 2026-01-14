# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import heapq
import logging
import pickle
from abc import abstractmethod
from datetime import datetime
from functools import partial
from itertools import groupby
from pathlib import Path
from queue import Queue
from typing import Any, NamedTuple
from warnings import warn

import numpy as np
import numpy.typing as npt
import pennylane as qml
from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator
from scipy.optimize import OptimizeResult

from divi.backends import (
    CircuitRunner,
    convert_counts_to_probs,
    reverse_dict_endianness,
)
from divi.circuits import CircuitBundle, MetaCircuit
from divi.circuits.qem import _NoMitigation
from divi.qprog._expectation import _batched_expectation
from divi.qprog._hamiltonians import convert_hamiltonian_to_pauli_string
from divi.qprog.checkpointing import (
    PROGRAM_STATE_FILE,
    CheckpointConfig,
    _atomic_write,
    _ensure_checkpoint_dir,
    _get_checkpoint_subdir_path,
    _load_and_validate_pydantic_model,
    resolve_checkpoint_path,
)
from divi.qprog.exceptions import _CancelledError
from divi.qprog.optimizers import (
    MonteCarloOptimizer,
    Optimizer,
    PymooOptimizer,
    ScipyMethod,
    ScipyOptimizer,
)
from divi.qprog.quantum_program import QuantumProgram

logger = logging.getLogger(__name__)


class SolutionEntry(NamedTuple):
    """A solution entry with bitstring, probability, and optional decoded value.

    Args:
        bitstring: Binary string representing a computational basis state.
        prob: Measured probability in range [0.0, 1.0].
        decoded: Optional problem-specific decoded representation. Defaults to None.
    """

    bitstring: str
    prob: float
    decoded: Any | None = None


class SubclassState(BaseModel):
    """Container for subclass-specific state."""

    data: dict[str, Any] = Field(default_factory=dict)


class OptimizerConfig(BaseModel):
    """Configuration for reconstructing an optimizer."""

    type: str
    config: dict[str, Any] = Field(default_factory=dict)


class ProgramState(BaseModel):
    """Pydantic model for VariationalQuantumAlgorithm state."""

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    # Metadata
    program_type: str = Field(validation_alias="_serialized_program_type")
    version: str = "1.0"
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

    # Core Algorithm State (mapped to private attributes)
    current_iteration: int
    max_iterations: int
    losses_history: list[dict[str, float]] = Field(validation_alias="_losses_history")
    best_loss: float = Field(validation_alias="_best_loss")
    best_probs: dict[str, float] = Field(validation_alias="_best_probs")
    total_circuit_count: int = Field(validation_alias="_total_circuit_count")
    total_run_time: float = Field(validation_alias="_total_run_time")
    seed: int | None = Field(validation_alias="_seed")
    grouping_strategy: str = Field(validation_alias="_grouping_strategy")

    # Arrays
    curr_params: list[list[float]] | None = Field(
        default=None, validation_alias="_curr_params"
    )
    best_params: list[float] | None = Field(
        default=None, validation_alias="_best_params"
    )
    final_params: list[float] | None = Field(
        default=None, validation_alias="_final_params"
    )

    # Complex State (mapped to new adapter properties)
    rng_state_bytes: bytes | None = Field(
        default=None, validation_alias="_serialized_rng_state"
    )
    optimizer_config: OptimizerConfig = Field(
        validation_alias="_serialized_optimizer_config"
    )
    subclass_state: SubclassState = Field(validation_alias="_serialized_subclass_state")

    @field_serializer("rng_state_bytes")
    def serialize_bytes(self, v: bytes | None, _info):
        return v.hex() if v is not None else None

    @field_validator("rng_state_bytes", mode="before")
    @classmethod
    def validate_bytes(cls, v):
        return bytes.fromhex(v) if isinstance(v, str) else v

    @field_serializer("curr_params", "best_params", "final_params")
    def serialize_arrays(self, v: npt.NDArray | list | None, _info):
        if isinstance(v, np.ndarray):
            return v.tolist()
        return v

    def restore(self, program: "VariationalQuantumAlgorithm") -> None:
        """Apply this state object back to a program instance."""
        # 1. Bulk restore standard attributes
        for name, field in self.model_fields.items():
            target_attr = field.validation_alias or name

            # Skip adapter properties (they are read-only / calculated)
            if target_attr.startswith("_serialized_"):
                continue

            val = getattr(self, name)

            # Handle numpy conversion
            if "params" in target_attr and val is not None:
                val = np.array(val)

            if hasattr(program, target_attr):
                setattr(program, target_attr, val)

        # 2. Restore complex state
        if self.rng_state_bytes:
            program._rng.bit_generator.state = pickle.loads(self.rng_state_bytes)

        program._load_subclass_state(self.subclass_state.data)


def _compute_parameter_shift_mask(n_params: int) -> npt.NDArray[np.float64]:
    """
    Generate a binary matrix mask for the parameter shift rule.
    This mask is used to determine the shifts to apply to each parameter
    when computing gradients via the parameter shift rule in quantum algorithms.

    Args:
        n_params (int): The number of parameters in the quantum circuit.

    Returns:
        npt.NDArray[np.float64]: A (2 * n_params, n_params) matrix where each row encodes
            the shift to apply to each parameter for a single evaluation.
            The values are multiples of 0.5 * pi, with alternating signs.
    """
    mask_arr = np.arange(0, 2 * n_params, 2)
    mask_arr[0] = 1

    binary_matrix = ((mask_arr[:, np.newaxis] & (1 << np.arange(n_params))) > 0).astype(
        np.float64
    )

    binary_matrix = binary_matrix.repeat(2, axis=0)
    binary_matrix[1::2] *= -1
    binary_matrix *= 0.5 * np.pi

    return binary_matrix


class VariationalQuantumAlgorithm(QuantumProgram):
    """Base class for variational quantum algorithms.

    This class provides the foundation for implementing variational quantum
    algorithms in Divi. It handles circuit execution, parameter optimization,
    and result management for algorithms that optimize parameterized quantum
    circuits to minimize cost functions.

    Variational algorithms work by:
    1. Generating parameterized quantum circuits
    2. Executing circuits on quantum hardware/simulators
    3. Computing expectation values of cost Hamiltonians
    4. Using classical optimizers to update parameters
    5. Iterating until convergence

    Attributes:
        _losses_history (list[dict]): History of loss values during optimization.
        _final_params (npt.NDArray[np.float64]): Final optimized parameters.
        _best_params (npt.NDArray[np.float64]): Parameters that achieved the best loss.
        _best_loss (float): Best loss achieved during optimization.
        _circuits (list[Circuit]): Generated quantum circuits.
        _total_circuit_count (int): Total number of circuits executed.
        _total_run_time (float): Total execution time in seconds.
        _curr_params (npt.NDArray[np.float64]): Current parameter values.
        _seed (int | None): Random seed for parameter initialization.
        _rng (np.random.Generator): Random number generator.
        _grad_mode (bool): Whether currently computing gradients.
        _grouping_strategy (str): Strategy for grouping quantum operations.
        _qem_protocol (QEMProtocol): Quantum error mitigation protocol.
        _cancellation_event (Event | None): Event for graceful termination.
        _meta_circuit_factory (callable): Factory for creating MetaCircuit instances.
    """

    def __init__(
        self,
        backend: CircuitRunner,
        optimizer: Optimizer | None = None,
        seed: int | None = None,
        progress_queue: Queue | None = None,
        **kwargs,
    ):
        """Initialize the VariationalQuantumAlgorithm.

        This constructor is specifically designed for hybrid quantum-classical
        variational algorithms. The instance variables `n_layers` and `n_params`
        must be set by subclasses, where:
        - `n_layers` is the number of layers in the quantum circuit.
        - `n_params` is the number of parameters per layer.

        For exotic variational algorithms where these variables may not be applicable,
        the `_initialize_params` method should be overridden to set the parameters.

        Args:
            backend (CircuitRunner): Quantum circuit execution backend.
            optimizer (Optimizer | None): The optimizer to use for parameter optimization.
                Defaults to MonteCarloOptimizer().
            seed (int | None): Random seed for parameter initialization. Defaults to None.
            progress_queue (Queue | None): Queue for progress reporting. Defaults to None.

        Keyword Args:
            initial_params (npt.NDArray[np.float64] | None): Initial parameters with shape
                (n_param_sets, n_layers * n_params). If provided, these will be set as
                the current parameters via the `curr_params` setter (which includes validation).
                Defaults to None.
            grouping_strategy (str): Strategy for grouping operations in Pennylane transforms.
                Options: "default", "wires", "qwc". Defaults to "qwc".
            qem_protocol (QEMProtocol | None): Quantum error mitigation protocol to apply. Defaults to None.
            precision (int): Number of decimal places for parameter values in QASM conversion.
                Defaults to 8.

                Note: Higher precision values result in longer QASM strings, which increases
                the amount of data sent to cloud backends. For most use cases, the default
                precision of 8 decimal places provides sufficient accuracy while keeping
                QASM sizes manageable. Consider reducing precision if you need to minimize
                data transfer overhead, or increase it only if you require higher numerical
                precision in your circuit parameters.
            decode_solution_fn (callable[[str], Any] | None): Function to decode bitstrings
                into problem-specific solution representations. Called during final computation
                and when `get_top_solutions(include_decoded=True)` is used. The function should
                take a binary string (e.g., "0101") and return a decoded representation
                (e.g., a list of indices, numpy array, or custom object). Defaults to
                `lambda bitstring: bitstring` (identity function).
        """

        super().__init__(
            backend=backend, seed=seed, progress_queue=progress_queue, **kwargs
        )

        # --- Optimization Results & History ---
        self._losses_history = []
        self._best_params = []
        self._final_params = []
        self._best_loss = float("inf")
        self._best_probs = {}
        self._curr_params = kwargs.pop("initial_params", None)

        # --- Random Number Generation ---
        self._seed = seed
        self._rng = np.random.default_rng(self._seed)

        # --- Computation Mode Flags ---
        # Lets child classes adapt their optimization step for grad calculation routine
        self._grad_mode = False
        self._is_compute_probabilities = False

        # --- Optimizer Configuration ---
        self.optimizer = optimizer if optimizer is not None else MonteCarloOptimizer()

        # --- Backend & Circuit Configuration ---
        if backend and backend.supports_expval:
            grouping_strategy = kwargs.pop("grouping_strategy", None)
            if grouping_strategy is not None and grouping_strategy != "_backend_expval":
                warn(
                    "Backend supports direct expectation value calculation, but a grouping_strategy was provided. "
                    "The grouping strategy will be ignored.",
                    UserWarning,
                )
            self._grouping_strategy = "_backend_expval"
        else:
            self._grouping_strategy = kwargs.pop("grouping_strategy", "qwc")

        self._qem_protocol = kwargs.pop("qem_protocol", None) or _NoMitigation()
        self._precision = kwargs.pop("precision", 8)

        # --- Solution Decoding ---
        self._decode_solution_fn = kwargs.pop(
            "decode_solution_fn", lambda bitstring: bitstring
        )

        # --- Circuit Factory & Templates ---
        self._meta_circuits = None
        self._meta_circuit_factory = partial(
            MetaCircuit,
            # No grouping strategy for expectation value measurements
            grouping_strategy=self._grouping_strategy,
            qem_protocol=self._qem_protocol,
            precision=self._precision,
        )

        # --- Control Flow ---
        self._cancellation_event = None

    @property
    @abstractmethod
    def cost_hamiltonian(self) -> qml.operation.Operator:
        """The cost Hamiltonian for the variational problem."""
        pass

    @property
    def total_circuit_count(self) -> int:
        """Get the total number of circuits executed.

        Returns:
            int: Cumulative count of circuits submitted for execution.
        """
        return self._total_circuit_count

    @property
    def total_run_time(self) -> float:
        """Get the total runtime across all circuit executions.

        Returns:
            float: Cumulative execution time in seconds.
        """
        return self._total_run_time

    @property
    def meta_circuits(self) -> dict[str, MetaCircuit]:
        """Get the meta-circuit templates used by this program.

        Returns:
            dict[str, MetaCircuit]: Dictionary mapping circuit names to their
                MetaCircuit templates.
        """
        return self._meta_circuits

    @property
    def n_params(self):
        """Get the total number of parameters in the quantum circuit.

        Returns:
            int: Total number of trainable parameters (n_layers * n_params_per_layer).
        """
        return self._n_params

    def _has_run_optimization(self) -> bool:
        """Check if optimization has been run at least once.

        Returns:
            bool: True if optimization has been run, False otherwise.
        """
        return len(self._losses_history) > 0

    @property
    def losses_history(self) -> list[dict]:
        """Get a copy of the optimization loss history.

        Each entry is a dictionary mapping parameter indices to loss values.

        Returns:
            list[dict]: Copy of the loss history. Modifications to this list
                will not affect the internal state.
        """
        if not self._has_run_optimization():
            warn(
                "losses_history is empty. Optimization has not been run yet. "
                "Call run() to execute the optimization.",
                UserWarning,
                stacklevel=2,
            )
        return self._losses_history.copy()

    @property
    def min_losses_per_iteration(self) -> list[float]:
        """Get the minimum loss value for each iteration.

        Returns a list where each element is the minimum (best) loss value
        across all parameter sets for that iteration.

        Returns:
            list[float]: List of minimum loss values, one per iteration.
        """
        if not self._has_run_optimization():
            warn(
                "min_losses_per_iteration is empty. Optimization has not been run yet. "
                "Call run() to execute the optimization.",
                UserWarning,
                stacklevel=2,
            )
        return [min(loss_dict.values()) for loss_dict in self._losses_history]

    @property
    def final_params(self) -> npt.NDArray[np.float64]:
        """Get a copy of the final optimized parameters.

        Returns:
            npt.NDArray[np.float64]: Copy of the final parameters. Modifications to this array
                will not affect the internal state.
        """
        if len(self._final_params) == 0 or not self._has_run_optimization():
            warn(
                "final_params is not available. Optimization has not been run yet. "
                "Call run() to execute the optimization.",
                UserWarning,
                stacklevel=2,
            )
        return self._final_params.copy()

    @property
    def best_params(self) -> npt.NDArray[np.float64]:
        """Get a copy of the parameters that achieved the best (lowest) loss.

        Returns:
            npt.NDArray[np.float64]: Copy of the best parameters. Modifications to this array
                will not affect the internal state.
        """
        if len(self._best_params) == 0 or not self._has_run_optimization():
            warn(
                "best_params is not available. Optimization has not been run yet. "
                "Call run() to execute the optimization.",
                UserWarning,
                stacklevel=2,
            )
        return self._best_params.copy()

    @property
    def best_loss(self) -> float:
        """Get the best loss achieved so far.

        Returns:
            float: The best loss achieved so far.
        """
        if not self._has_run_optimization():
            warn(
                "best_loss has not been computed yet. Optimization has not been run. "
                "Call run() to execute the optimization.",
                UserWarning,
                stacklevel=2,
            )
        elif self._best_loss == float("inf"):
            # Defensive check: if optimization ran but best_loss is still inf, something is wrong
            raise RuntimeError(
                "best_loss is still infinite after optimization. This indicates a problem "
                "with the optimization process. The optimization callback may not have executed "
                "correctly, or all computed losses were infinite."
            )
        return self._best_loss

    @property
    def best_probs(self) -> dict[str, float]:
        """Get normalized probabilities for the best parameters.

        This property provides access to the probability distribution computed
        by running measurement circuits with the best parameters found during
        optimization. The distribution maps bitstrings (computational basis states)
        to their measured probabilities.

        The probabilities are normalized and have deterministic ordering when
        iterated (dictionary insertion order is preserved in Python 3.7+).

        Returns:
            dict[str, float]: Dictionary mapping bitstrings to probabilities.
                Keys are binary strings (e.g., "0101"), values are probabilities
                in the range [0.0, 1.0]. Returns an empty dict if final computation
                has not been performed.

        Raises:
            RuntimeError: If attempting to access probabilities before running
                the algorithm with final computation enabled.

        Note:
            To populate this distribution, you must run the algorithm with
            `perform_final_computation=True` (the default):

            >>> program.run(perform_final_computation=True)
            >>> probs = program.best_probs

        Example:
            >>> program.run()
            >>> probs = program.best_probs
            >>> for bitstring, prob in probs.items():
            ...     print(f"{bitstring}: {prob:.2%}")
            0101: 42.50%
            1010: 31.20%
            ...
        """
        if not self._best_probs:
            warn(
                "best_probs is empty. Either optimization has not been run yet, "
                "or final computation was not performed. Call run() to execute the optimization.",
                UserWarning,
                stacklevel=2,
            )
        return self._best_probs.copy()

    def get_top_solutions(
        self, n: int = 10, *, min_prob: float = 0.0, include_decoded: bool = False
    ) -> list[SolutionEntry]:
        """Get the top-N solutions sorted by probability.

        This method extracts the most probable solutions from the measured
        probability distribution. Solutions are sorted by probability (descending)
        with deterministic tie-breaking using lexicographic ordering of bitstrings.

        Args:
            n (int): Maximum number of solutions to return. Must be non-negative.
                If n is 0 or negative, returns an empty list. If n exceeds the
                number of available solutions (after filtering), returns all
                available solutions. Defaults to 10.
            min_prob (float): Minimum probability threshold for including solutions.
                Only solutions with probability >= min_prob will be included.
                Must be in range [0.0, 1.0]. Defaults to 0.0 (no filtering).
            include_decoded (bool): Whether to populate the `decoded` field of
                each SolutionEntry by calling the `decode_solution_fn` provided
                in the constructor. If False, the decoded field will be None.
                Defaults to False.

        Returns:
            list[SolutionEntry]: List of solution entries sorted by probability
                (descending), then by bitstring (lexicographically ascending)
                for deterministic tie-breaking. Returns an empty list if no
                probability distribution is available or n <= 0.

        Raises:
            RuntimeError: If probability distribution is not available because
                optimization has not been run or final computation was not performed.
            ValueError: If min_prob is not in range [0.0, 1.0] or n is negative.

        Note:
            The probability distribution must be computed by running the algorithm
            with `perform_final_computation=True` (the default):

            >>> program.run(perform_final_computation=True)
            >>> top_10 = program.get_top_solutions(n=10)

        Example:
            >>> # Get top 5 solutions with probability >= 5%
            >>> program.run()
            >>> solutions = program.get_top_solutions(n=5, min_prob=0.05)
            >>> for sol in solutions:
            ...     print(f"{sol.bitstring}: {sol.prob:.2%}")
            1010: 42.50%
            0101: 31.20%
            1100: 15.30%
            0011: 8.50%
            1111: 2.50%

            >>> # Get solutions with decoding
            >>> solutions = program.get_top_solutions(n=3, include_decoded=True)
            >>> for sol in solutions:
            ...     print(f"{sol.bitstring} -> {sol.decoded}")
            1010 -> [0, 2]
            0101 -> [1, 3]
            ...
        """
        # Validate inputs
        if n < 0:
            raise ValueError(f"n must be non-negative, got {n}")
        if not (0.0 <= min_prob <= 1.0):
            raise ValueError(f"min_prob must be in range [0.0, 1.0], got {min_prob}")

        # Handle edge case: n == 0
        if n == 0:
            return []

        # Require probability distribution to exist
        if not self._best_probs:
            raise RuntimeError(
                "No probability distribution available. The final computation step "
                "must be performed to compute the probability distribution. "
                "Call run(perform_final_computation=True) to execute optimization "
                "and compute the distribution."
            )
        # Extract the probability distribution (nested by parameter set)
        # _best_probs structure: {tag: {bitstring: prob}}
        probs_dict = next(iter(self._best_probs.values()))

        # Filter by minimum probability and get top n sorted by probability (descending),
        # then bitstring (ascending) for deterministic tie-breaking
        top_items = heapq.nsmallest(
            n,
            filter(
                lambda bitstring_prob: bitstring_prob[1] >= min_prob, probs_dict.items()
            ),
            key=lambda bitstring_prob: (-bitstring_prob[1], bitstring_prob[0]),
        )

        # Decode items if needed
        decoded_values = map(
            lambda item: self._decode_solution_fn(item[0]) if include_decoded else None,
            top_items,
        )

        # Build result list
        return list(
            map(
                lambda item, decoded: SolutionEntry(
                    bitstring=item[0], prob=item[1], decoded=decoded
                ),
                top_items,
                decoded_values,
            )
        )

    @property
    def curr_params(self) -> npt.NDArray[np.float64]:
        """Get the current parameters.

        These are the parameters used for optimization. They can be accessed
        and modified at any time, including during optimization.

        Returns:
            npt.NDArray[np.float64]: Current parameters. If not yet initialized,
                they will be generated automatically.
        """
        if self._curr_params is None:
            self._initialize_params()
        return self._curr_params.copy()

    @curr_params.setter
    def curr_params(self, value: npt.NDArray[np.float64] | None):
        """
        Set the current parameters.

        Args:
            value (npt.NDArray[np.float64] | None): Parameters with shape
                (n_param_sets, n_layers * n_params), or None to reset
                to uninitialized state.

        Raises:
            ValueError: If parameters have incorrect shape.
        """
        if value is not None:
            self._validate_initial_params(value)
            self._curr_params = value.copy()
        else:
            # Reset to uninitialized state
            self._curr_params = None

    # --- Serialization Adapters (For Pydantic) ---
    @property
    def _serialized_program_type(self) -> str:
        return type(self).__name__

    @property
    def _serialized_rng_state(self) -> bytes:
        return pickle.dumps(self._rng.bit_generator.state)

    @property
    def _serialized_optimizer_config(self) -> OptimizerConfig:
        config_dict = self.optimizer.get_config()
        return OptimizerConfig(type=config_dict.pop("type"), config=config_dict)

    @property
    def _serialized_subclass_state(self) -> SubclassState:
        return SubclassState(data=self._save_subclass_state())

    @property
    def meta_circuits(self) -> dict[str, MetaCircuit]:
        """Get the meta-circuit templates used by this program.

        Returns:
            dict[str, MetaCircuit]: Dictionary mapping circuit names to their
                MetaCircuit templates.
        """
        # Lazy initialization: each instance has its own _meta_circuits.
        # Note: When used with ProgramBatch, meta_circuits is initialized sequentially
        # in the main thread before parallel execution to avoid thread-safety issues.
        if self._meta_circuits is None:
            self._meta_circuits = self._create_meta_circuits_dict()
        return self._meta_circuits

    @abstractmethod
    def _create_meta_circuits_dict(self) -> dict[str, MetaCircuit]:
        pass

    @abstractmethod
    def _generate_circuits(self, **kwargs) -> list[CircuitBundle]:
        """Generate quantum circuits for execution.

        This method should generate and return a list of Circuit objects based on
        the current algorithm state and parameters. The circuits will be executed
        by the backend.

        Args:
            **kwargs: Additional keyword arguments for circuit generation.

        Returns:
            list[CircuitBundle]: List of Circuit objects to be executed.
        """
        pass

    @abstractmethod
    def _save_subclass_state(self) -> dict[str, Any]:
        """Hook method for subclasses to save additional state.

        Subclasses must override this method to return a dictionary of
        state variables that should be included in the checkpoint.

        Returns:
            dict[str, Any]: Dictionary of subclass-specific state.
        """
        pass

    @abstractmethod
    def _load_subclass_state(self, state: dict[str, Any]) -> None:
        """Hook method for subclasses to load additional state.

        Subclasses must override this method to restore state variables
        from the checkpoint dictionary. This is called after instance creation.

        Args:
            state (dict[str, Any]): Dictionary of subclass-specific state.
        """
        pass

    def _get_optimizer_config(self) -> OptimizerConfig:
        """Extract optimizer configuration for checkpoint reconstruction.

        Returns:
            OptimizerConfig: Configuration object for the current optimizer.

        Raises:
            NotImplementedError: If the optimizer does not support state saving.
        """
        config_dict = self.optimizer.get_config()
        return OptimizerConfig(
            type=config_dict.pop("type"),
            config=config_dict,
        )

    def save_state(self, checkpoint_config: CheckpointConfig) -> str:
        """Save the program state to a checkpoint directory."""
        if self.current_iteration == 0 and len(self._losses_history) == 0:
            raise RuntimeError("Cannot save checkpoint: optimization has not been run.")

        if checkpoint_config.checkpoint_dir is None:
            raise ValueError(
                "checkpoint_config.checkpoint_dir must be a non-None Path."
            )

        main_dir = _ensure_checkpoint_dir(checkpoint_config.checkpoint_dir)
        checkpoint_path = _get_checkpoint_subdir_path(main_dir, self.current_iteration)
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # 1. Save optimizer
        self.optimizer.save_state(checkpoint_path)

        # 2. Save Program State (Pydantic pulls data via validation_aliases)
        state = ProgramState.model_validate(self)

        state_file = checkpoint_path / PROGRAM_STATE_FILE
        _atomic_write(state_file, state.model_dump_json(indent=2))

        return checkpoint_path

    @classmethod
    def load_state(
        cls,
        checkpoint_dir: Path | str,
        backend: CircuitRunner,
        subdirectory: str | None = None,
        **kwargs,
    ) -> "VariationalQuantumAlgorithm":
        """Load program state from a checkpoint directory."""
        checkpoint_path = resolve_checkpoint_path(checkpoint_dir, subdirectory)
        state_file = checkpoint_path / PROGRAM_STATE_FILE

        # 1. Load Pydantic Model
        state = _load_and_validate_pydantic_model(
            state_file,
            ProgramState,
            required_fields=["program_type", "current_iteration"],
        )

        # 2. Reconstruct Optimizer
        opt_config = state.optimizer_config
        if opt_config.type == "MonteCarloOptimizer":
            optimizer = MonteCarloOptimizer.load_state(checkpoint_path)
        elif opt_config.type == "PymooOptimizer":
            optimizer = PymooOptimizer.load_state(checkpoint_path)
        else:
            raise ValueError(f"Unsupported optimizer type: {opt_config.type}")

        # 3. Create Instance
        program = cls(backend=backend, optimizer=optimizer, seed=state.seed, **kwargs)

        # 4. Restore State
        state.restore(program)

        return program

    def get_expected_param_shape(self) -> tuple[int, int]:
        """
        Get the expected shape for initial parameters.

        Returns:
            tuple[int, int]: Shape (n_param_sets, n_layers * n_params) that
                initial parameters should have for this quantum program.
        """
        return (self.optimizer.n_param_sets, self.n_layers * self.n_params)

    def _validate_initial_params(self, params: npt.NDArray[np.float64]):
        """
        Validate user-provided initial parameters.

        Args:
            params (npt.NDArray[np.float64]): Parameters to validate.

        Raises:
            ValueError: If parameters have incorrect shape.
        """
        expected_shape = self.get_expected_param_shape()

        if params.shape != expected_shape:
            raise ValueError(
                f"Initial parameters must have shape {expected_shape}, "
                f"got {params.shape}"
            )

    def _initialize_params(self):
        """
        Initialize the circuit parameters randomly.

        Generates random parameters with values uniformly distributed between
        0 and 2π. The number of parameter sets depends on the optimizer being used.
        """
        total_params = self.n_layers * self.n_params
        self._curr_params = self._rng.uniform(
            0, 2 * np.pi, (self.optimizer.n_param_sets, total_params)
        )

    def _run_optimization_circuits(self, **kwargs) -> dict[int, float]:
        self._curr_circuits = self._generate_circuits(**kwargs)

        if self.backend.supports_expval:
            kwargs["ham_ops"] = convert_hamiltonian_to_pauli_string(
                self.cost_hamiltonian, self.n_qubits
            )

        losses = self._dispatch_circuits_and_process_results(**kwargs)

        return losses

    def _post_process_results(
        self, results: dict[str, dict[str, int]], **kwargs
    ) -> dict[int, float]:
        """
        Post-process the results of the quantum problem.

        Args:
            results (dict[str, dict[str, int]]): The shot histograms of the quantum execution step.
                The keys should be strings of format {param_id}_*_{measurement_group_id}.
                i.e. an underscore-separated bunch of metadata, starting always with
                the index of some parameter and ending with the index of some measurement group.
                Any extra piece of metadata that might be relevant to the specific
                application can be kept in the middle.

        Returns:
            dict[int, float]: The energies for each parameter set grouping, where the dict keys
                correspond to the parameter indices.
        """
        if self._is_compute_probabilities:
            probs = convert_counts_to_probs(results, self.backend.shots)
            return reverse_dict_endianness(probs)

        if not (self._cancellation_event and self._cancellation_event.is_set()):
            self.reporter.info(
                message="Post-processing output", iteration=self.current_iteration
            )

        losses = {}
        measurement_groups = self.meta_circuits["cost_circuit"].measurement_groups

        # Define key functions for grouping
        get_param_id = lambda item: int(item[0].split("_")[0])
        get_qem_id = lambda item: int(item[0].split("_")[1].split(":")[1])

        # Group the pre-sorted results by parameter ID.
        for p, param_group_iterator in groupby(results.items(), key=get_param_id):
            param_group_iterator = list(param_group_iterator)

            # Group by QEM ID to handle error mitigation
            qem_groups = {
                gid: [value for _, value in group]
                for gid, group in groupby(param_group_iterator, key=get_qem_id)
            }

            # Apply QEM protocol to expectation values (common for both backends)
            apply_qem = lambda exp_matrix: [
                self._qem_protocol.postprocess_results(exp_vals)
                for exp_vals in exp_matrix
            ]

            if self.backend.supports_expval:
                ham_ops = kwargs.get("ham_ops")
                if ham_ops is None:
                    raise ValueError(
                        "Hamiltonian operators (ham_ops) are required when using a backend "
                        "that supports expectation values, but were not provided."
                    )
                marginal_results = [
                    apply_qem(
                        np.array(
                            [
                                [shot_dict[op] for op in ham_ops.split(";")]
                                for shot_dict in shots_dicts
                            ]
                        ).T
                    )
                    for shots_dicts in sorted(qem_groups.values())
                ] or []
            else:
                shots_by_qem_idx = zip(*qem_groups.values())
                marginal_results = []
                for shots_dicts, curr_measurement_group in zip(
                    shots_by_qem_idx, measurement_groups
                ):
                    wire_order = tuple(reversed(self.cost_hamiltonian.wires))
                    exp_matrix = _batched_expectation(
                        shots_dicts, curr_measurement_group, wire_order
                    )
                    mitigated = apply_qem(exp_matrix)
                    marginal_results.append(
                        mitigated if len(mitigated) > 1 else mitigated[0]
                    )

            pl_loss = (
                self.meta_circuits["cost_circuit"]
                .postprocessing_fn(marginal_results)
                .item()
            )

            losses[p] = pl_loss + self.loss_constant

        return losses

    def _perform_final_computation(self, **kwargs) -> None:
        """
        Perform final computations after optimization is complete.

        This is an optional hook method that subclasses can override to perform
        any post-optimization processing, such as extracting solutions, running
        final measurements, or computing additional metrics.

        Args:
            **kwargs: Additional keyword arguments for subclasses.

        Note:
            The default implementation does nothing. Subclasses should override
            this method if they need post-optimization processing.
        """
        pass

    def run(
        self,
        perform_final_computation: bool = True,
        checkpoint_config: CheckpointConfig | None = None,
        **kwargs,
    ) -> tuple[int, float]:
        """Run the variational quantum algorithm.

        The outputs are stored in the algorithm object.

        Args:
            perform_final_computation (bool): Whether to perform final computation after optimization completes.
                Typically, this step involves sampling with the best found parameters to extract
                solution probability distributions. Set this to False in warm-starting or pre-training
                routines where the final sampling step is not needed. Defaults to True.
            checkpoint_config (CheckpointConfig | None): Checkpoint configuration.
                If None, no checkpointing is performed.
            **kwargs: Additional keyword arguments for subclasses.

        Returns:
            tuple[int, float]: A tuple containing (total_circuit_count, total_run_time).
        """
        # Initialize checkpointing
        if checkpoint_config is None:
            checkpoint_config = CheckpointConfig()

        if checkpoint_config.checkpoint_dir:
            logger.info(
                f"Using checkpoint directory: {checkpoint_config.checkpoint_dir}"
            )

        # Extract max_iterations from kwargs if present (for compatibility with subclasses)
        max_iterations = kwargs.pop("max_iterations", self.max_iterations)
        if max_iterations != self.max_iterations:
            self.max_iterations = max_iterations

        # Warn if max_iterations is less than current_iteration (regardless of how it was set)
        if self.max_iterations < self.current_iteration:
            warn(
                f"max_iterations ({self.max_iterations}) is less than current_iteration "
                f"({self.current_iteration}). The optimization will not run additional "
                f"iterations since the maximum has already been reached.",
                UserWarning,
            )

        def cost_fn(params):
            self.reporter.info(
                message="💸 Computing Cost 💸", iteration=self.current_iteration
            )

            self._curr_params = np.atleast_2d(params)

            losses = self._run_optimization_circuits(**kwargs)

            losses = np.fromiter(losses.values(), dtype=np.float64)

            if params.ndim > 1:
                return losses
            else:
                return losses.item()

        self._grad_shift_mask = _compute_parameter_shift_mask(
            self.n_layers * self.n_params
        )

        def grad_fn(params):
            self._grad_mode = True

            self.reporter.info(
                message="📈 Computing Gradients 📈", iteration=self.current_iteration
            )

            self._curr_params = self._grad_shift_mask + params

            exp_vals = self._run_optimization_circuits(**kwargs)
            exp_vals_arr = np.fromiter(exp_vals.values(), dtype=np.float64)

            pos_shifts = exp_vals_arr[::2]
            neg_shifts = exp_vals_arr[1::2]
            grads = 0.5 * (pos_shifts - neg_shifts)

            self._grad_mode = False

            return grads

        def _iteration_counter(intermediate_result: OptimizeResult):

            self._losses_history.append(
                dict(
                    zip(
                        [str(i) for i in range(len(intermediate_result.x))],
                        intermediate_result.fun,
                    )
                )
            )

            current_loss = np.min(intermediate_result.fun)
            if current_loss < self._best_loss:
                self._best_loss = current_loss
                best_idx = np.argmin(intermediate_result.fun)

                self._best_params = intermediate_result.x[best_idx].copy()

            self.current_iteration += 1

            self.reporter.update(iteration=self.current_iteration)

            # Checkpointing
            if checkpoint_config._should_checkpoint(self.current_iteration):
                self.save_state(checkpoint_config)

            if self._cancellation_event and self._cancellation_event.is_set():
                raise _CancelledError("Cancellation requested by batch.")

            # The scipy implementation of COBYLA interprets the `maxiter` option
            # as the maximum number of function evaluations, not iterations.
            # To provide a consistent user experience, we disable `scipy`'s
            # `maxiter` and manually stop the optimization from the callback
            # when the desired number of iterations is reached.
            if (
                isinstance(self.optimizer, ScipyOptimizer)
                and self.optimizer.method == ScipyMethod.COBYLA
                and intermediate_result.nit + 1 == self.max_iterations
            ):
                raise StopIteration

        self.reporter.info(message="Finished Setup")

        if self._curr_params is None:
            self._initialize_params()
        else:
            self._validate_initial_params(self._curr_params)

        try:
            self._minimize_res = self.optimizer.optimize(
                cost_fn=cost_fn,
                initial_params=self._curr_params,
                callback_fn=_iteration_counter,
                jac=grad_fn,
                max_iterations=self.max_iterations,
                rng=self._rng,
            )
        except _CancelledError:
            # The optimizer was stopped by our callback. This is not a real
            # error, just a signal to exit this task cleanly.
            return self._total_circuit_count, self._total_run_time

        self._final_params = self._minimize_res.x

        if perform_final_computation:
            self._perform_final_computation(**kwargs)

        self.reporter.info(message="Finished successfully!")

        return self.total_circuit_count, self.total_run_time

    def _run_solution_measurement(self) -> None:
        """Execute measurement circuits to obtain probability distributions for solution extraction."""
        if self._best_params is None:
            raise RuntimeError(
                "Optimization has not been run, no best parameters available."
            )

        if "meas_circuit" not in self.meta_circuits:
            raise NotImplementedError(
                f"{type(self).__name__} does not implement a 'meas_circuit'."
            )

        self._is_compute_probabilities = True

        # Compute probabilities for best parameters (the ones that achieved best loss)
        self._curr_params = np.atleast_2d(self._best_params)
        self._curr_circuits = self._generate_circuits()
        best_probs = self._dispatch_circuits_and_process_results()
        self._best_probs.update(best_probs)

        self._is_compute_probabilities = False
