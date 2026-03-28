# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import logging
import pickle
from abc import abstractmethod
from datetime import datetime
from pathlib import Path
from queue import Queue
from typing import Any, Literal, NamedTuple
from warnings import warn

import numpy as np
import numpy.typing as npt
import pennylane as qml
from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator
from scipy.optimize import OptimizeResult

from divi.backends import CircuitRunner
from divi.circuits import MetaCircuit
from divi.pipeline import CircuitPipeline, PipelineEnv, Stage
from divi.pipeline.stages import (
    CircuitSpecStage,
    MeasurementStage,
    ParameterBindingStage,
    PauliTwirlStage,
    QEMStage,
)
from divi.qprog.checkpointing import (
    PROGRAM_STATE_FILE,
    CheckpointConfig,
    _atomic_write,
    _ensure_checkpoint_dir,
    _get_checkpoint_subdir_path,
    _load_and_validate_pydantic_model,
    resolve_checkpoint_path,
)
from divi.qprog.early_stopping import EarlyStopping, StopReason
from divi.qprog.exceptions import _CancelledError
from divi.qprog.optimizers import (
    MonteCarloOptimizer,
    Optimizer,
    PymooOptimizer,
    ScipyMethod,
    ScipyOptimizer,
)
from divi.qprog.quantum_program import QuantumProgram
from divi.viz import ProgramViz

logger = logging.getLogger(__name__)

PARAM_SET_AXIS = "param_set"


def _extract_param_set_idx(key: tuple) -> int:
    """Extract the param_set index from a pipeline result key."""
    for axis_name, idx in key:
        if axis_name == PARAM_SET_AXIS:
            return idx
    raise KeyError(f"No '{PARAM_SET_AXIS}' axis found in pipeline result key: {key}")


_RUN_INSTRUCTION = "Call run() to execute the optimization."

ParamHistoryMode = Literal["all_evaluated", "best_per_iteration"]


class SolutionEntry(NamedTuple):
    """A solution entry with bitstring, probability, and optional decoded value.

    Args:
        bitstring: Binary string representing a computational basis state.
        prob: Measured probability in range [0.0, 1.0].
        decoded: Optional problem-specific decoded representation. Defaults to None.
        energy: Optional objective energy for this solution. Defaults to None.
    """

    bitstring: str
    prob: float
    decoded: Any | None = None
    energy: float | None = None


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
    param_history: list[list[list[float]]] = Field(
        default_factory=list, validation_alias="_param_history"
    )
    best_loss: float = Field(validation_alias="_best_loss")
    best_probs: dict[str, float] = Field(validation_alias="_best_probs")
    total_circuit_count: int = Field(validation_alias="_total_circuit_count")
    total_run_time: float = Field(validation_alias="_total_run_time")
    seed: int | None = Field(validation_alias="_seed")
    stop_reason: str | None = Field(
        default=None, validation_alias="_serialized_stop_reason"
    )
    grouping_strategy: str | None = Field(validation_alias="_grouping_strategy")

    # Arrays
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

    @field_validator("param_history", mode="before")
    @classmethod
    def normalize_param_history(cls, v):
        """Accept nested lists or per-iteration ndarray snapshots from disk or program."""
        if not v:
            return []
        rows: list[list[list[float]]] = []
        for item in v:
            arr = np.asarray(item, dtype=np.float64)
            rows.append(arr.tolist())
        return rows

    @field_serializer("best_params", "final_params")
    def serialize_arrays(self, v: npt.NDArray | list | None, _info):
        if isinstance(v, np.ndarray):
            return v.tolist()
        return v

    def restore(self, program: "VariationalQuantumAlgorithm") -> None:
        """Apply this state object back to a program instance."""
        # 1. Bulk restore standard attributes
        for name, field in self.__class__.model_fields.items():
            target_attr = field.validation_alias or name

            # Skip adapter properties (they are read-only / calculated)
            if target_attr.startswith("_serialized_"):
                continue

            val = getattr(self, name)

            if target_attr == "_param_history" and val is not None:
                val = [np.asarray(block, dtype=np.float64) for block in val]
            # Handle numpy conversion
            elif "params" in target_attr and val is not None:
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
    mask_arr = 1 << np.arange(n_params)

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
        _param_history (list[npt.NDArray]): Raw per-callback parameter batches;
            use :meth:`param_history` to read copies with optional filtering.
        _final_params (npt.NDArray[np.float64]): Final optimized parameters.
        _best_params (npt.NDArray[np.float64]): Parameters that achieved the best loss.
        _best_loss (float): Best loss achieved during optimization.
        _circuits (list[Circuit]): Generated quantum circuits.
        _total_circuit_count (int): Total number of circuits executed.
        _total_run_time (float): Total execution time in seconds.
        _seed (int | None): Random seed for parameter initialization.
        _rng (np.random.Generator): Random number generator.

        _grouping_strategy (str): Strategy for grouping quantum operations.
        _qem_protocol (QEMProtocol): Quantum error mitigation protocol.
        _cancellation_event (Event | None): Event for graceful termination.
        _meta_circuit_factories (dict): Lazily-built mapping of circuit names to MetaCircuit factories.
    """

    def __init__(
        self,
        backend: CircuitRunner,
        optimizer: Optimizer | None = None,
        seed: int | None = None,
        progress_queue: Queue | None = None,
        early_stopping: EarlyStopping | None = None,
        **kwargs,
    ):
        """Initialize the VariationalQuantumAlgorithm.

        This constructor is specifically designed for hybrid quantum-classical
        variational algorithms. The instance variables `n_layers` and
        `n_params_per_layer` must be set by subclasses, where:
        - `n_layers` is the number of layers in the quantum circuit.
        - `n_params_per_layer` is the number of parameters per layer.

        For exotic variational algorithms where these variables may not be applicable,
        the `_initialize_param_sets` method should be overridden to generate the
        starting parameters for a fresh optimization run.

        Args:
            backend (CircuitRunner): Quantum circuit execution backend.
            optimizer (Optimizer | None): The optimizer to use for parameter optimization.
                Defaults to MonteCarloOptimizer().
            seed (int | None): Random seed for parameter initialization. Defaults to None.
            progress_queue (Queue | None): Queue for progress reporting. Defaults to None.
            early_stopping (EarlyStopping | None): Early stopping controller. When
                provided, the optimization loop will be halted if any of the
                configured criteria are met (e.g. patience exceeded, gradient
                below threshold, cost variance settled). Defaults to None.

        Keyword Args:
            grouping_strategy (str): Strategy for grouping operations in Pennylane transforms.
                Options: "default", "wires", "qwc". Defaults to "qwc".
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
        self._param_history: list[npt.NDArray[np.float64]] = []
        self._best_params = []
        self._final_params = []
        self._best_loss = float("inf")
        self._best_probs = {}
        self.optimize_result: OptimizeResult | None = None
        """Raw result object returned by the underlying optimizer, or ``None``
        before :meth:`run` is called.

        Always populated after :meth:`run` completes.  When optimization
        converges normally, ``success`` is ``True``.  When early stopping
        or cancellation terminates the run, ``success`` is ``False`` and the
        ``message`` field describes the reason.

        See :class:`scipy.optimize.OptimizeResult` for the full specification.
        """
        # --- Random Number Generation ---
        self._seed = seed
        self._rng = np.random.default_rng(self._seed)

        # --- Optimizer Configuration ---
        self.optimizer = optimizer if optimizer is not None else MonteCarloOptimizer()

        # --- Early Stopping ---
        self._early_stopping = early_stopping
        self._stop_reason: StopReason | None = None

        # --- Backend & Circuit Configuration ---
        _UNSET = object()
        grouping_strategy = kwargs.pop("grouping_strategy", _UNSET)
        if self.backend.supports_expval and grouping_strategy not in (
            None,
            "_backend_expval",
        ):
            if grouping_strategy is not _UNSET:
                warn(
                    "Backend supports direct expectation value calculation, but a "
                    "grouping_strategy was provided. Overriding to use the "
                    "backend's native expval support.",
                    UserWarning,
                    stacklevel=2,
                )
            self._grouping_strategy = "_backend_expval"
        else:
            self._grouping_strategy = (
                grouping_strategy if grouping_strategy is not _UNSET else "qwc"
            )

        self._precision = kwargs.pop("precision", 8)

        # --- Solution Decoding ---
        self._decode_solution_fn = kwargs.pop(
            "decode_solution_fn", lambda bitstring: bitstring
        )

        # --- Circuit Factory & Templates ---
        self._meta_circuit_factories = None

        # --- Control Flow ---
        self._cancellation_event = None

    @property
    def cost_hamiltonian(self) -> qml.operation.Operator:
        """The cost Hamiltonian for the variational problem."""
        return self._cost_hamiltonian

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
    def meta_circuit_factories(self) -> dict[str, MetaCircuit]:
        """Get the meta-circuit factories used by this program.

        Returns:
            dict[str, MetaCircuit]: Dictionary mapping circuit names to their
                MetaCircuit factories.
        """
        return self._meta_circuit_factories

    @property
    def n_params_per_layer(self):
        """Number of trainable parameters per layer.

        Subclasses must set ``_n_params_per_layer`` (or override this property)
        so that the base class can compute the total parameter count as
        ``n_layers * n_params_per_layer``.

        Returns:
            int: Trainable parameters per layer.
        """
        return self._n_params_per_layer

    def _has_run_optimization(self) -> bool:
        """Check if optimization has been run at least once.

        Returns:
            bool: True if optimization has been run, False otherwise.
        """
        return len(self._losses_history) > 0

    @property
    def stop_reason(self) -> StopReason | None:
        """Reason the optimization was stopped early, or ``None``.

        Returns:
            StopReason | None: The :class:`~divi.qprog.early_stopping.StopReason`
                that triggered early stopping, or ``None`` if optimization
                completed normally or has not been run yet.
        """
        return self._stop_reason

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
                f"{_RUN_INSTRUCTION}",
                UserWarning,
                stacklevel=2,
            )
        return self._losses_history.copy()

    def param_history(
        self,
        mode: ParamHistoryMode = "all_evaluated",
    ) -> list[npt.NDArray[np.float64]]:
        """Parameter vectors recorded at each optimization callback.

        Args:
            mode: Which rows to return for each iteration:

                * ``"all_evaluated"`` — full batch from the callback, shape
                  ``(n_param_sets, n_params)`` per iteration (mirrors
                  :attr:`losses_history` population layout).
                * ``"best_per_iteration"`` — single best member by loss for
                  that iteration, shape ``(1, n_params)`` per iteration.

        Returns:
            list[npt.NDArray[np.float64]]: One array per completed callback.
            Use ``numpy.vstack(...)`` for a 2D sample matrix (e.g. PCA).

        Raises:
            RuntimeError: If internal loss and parameter histories are out of sync.
        """
        if not self._has_run_optimization():
            warn(
                "Parameter history is unavailable because optimization has not "
                f"been run yet. {_RUN_INSTRUCTION}",
                UserWarning,
                stacklevel=2,
            )
            return []

        if mode == "all_evaluated":
            return [row.copy() for row in self._param_history]

        if len(self._losses_history) != len(self._param_history):
            raise RuntimeError(
                "losses_history and _param_history length mismatch; cannot select "
                "best_per_iteration rows."
            )

        best_blocks: list[npt.NDArray[np.float64]] = []
        for loss_dict, block in zip(
            self._losses_history, self._param_history, strict=True
        ):
            arr = np.atleast_2d(np.asarray(block, dtype=np.float64))
            n_rows = arr.shape[0]
            best_idx = min(
                range(n_rows),
                key=lambda j: float(loss_dict[str(j)]),
            )
            best_blocks.append(arr[best_idx : best_idx + 1].copy())
        return best_blocks

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
                f"{_RUN_INSTRUCTION}",
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
                f"{_RUN_INSTRUCTION}",
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
                f"{_RUN_INSTRUCTION}",
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
                f"{_RUN_INSTRUCTION}",
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
    def viz(self):
        """Access visualization helpers for this variational program.

        The returned object exposes a thin convenience wrapper over the
        standalone :mod:`divi.viz` API, so scans can be written either as
        ``divi.viz.scan_1d(program, ...)`` or ``program.viz.scan_1d(...)``.

        Returns:
            ProgramViz: Convenience wrapper bound to this program instance.
        """
        return ProgramViz(self)

    @property
    def best_probs(self) -> dict[str, dict[str, float]]:
        """Get normalized probabilities for the best parameters.

        This property provides access to the probability distribution computed
        by running measurement circuits with the best parameters found during
        optimization. The distribution maps bitstrings (computational basis states)
        to their measured probabilities.

        The probabilities are normalized and have deterministic ordering when
        iterated (dictionary insertion order is preserved in Python 3.7+).

        Returns:
            dict[str, dict[str, float]]: Dictionary mapping parameter-set keys to
                bitstring probability dictionaries. Bitstrings are binary strings
                (e.g., "0101"), values are probabilities in range [0.0, 1.0].
                Returns an empty dict if final computation has not been performed.

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
                f"or final computation was not performed. {_RUN_INSTRUCTION}",
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
        top_items = sorted(
            filter(
                lambda bitstring_prob: bitstring_prob[1] >= min_prob, probs_dict.items()
            ),
            key=lambda bitstring_prob: (-bitstring_prob[1], bitstring_prob[0]),
        )[:n]

        # Build result list (decode on demand)
        return [
            SolutionEntry(
                bitstring=bitstring,
                prob=prob,
                decoded=(
                    self._decode_solution_fn(bitstring) if include_decoded else None
                ),
            )
            for bitstring, prob in top_items
        ]

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
    def _serialized_stop_reason(self) -> str | None:
        return self._stop_reason.value if self._stop_reason is not None else None

    @property
    def meta_circuit_factories(self) -> dict[str, MetaCircuit]:
        """Get the meta-circuit factories used by this program.

        Returns:
            dict[str, MetaCircuit]: Dictionary mapping circuit names to their
                MetaCircuit factories.
        """
        # Lazy initialization: each instance has its own _meta_circuit_factories.
        # Note: When used with ProgramEnsemble, meta_circuit_factories is initialized sequentially
        # in the main thread before parallel execution to avoid thread-safety issues.
        if self._meta_circuit_factories is None:
            self._meta_circuit_factories = self._create_meta_circuit_factories()

        return self._meta_circuit_factories

    @abstractmethod
    def _create_meta_circuit_factories(self) -> dict[str, MetaCircuit]:
        pass

    def _save_subclass_state(self) -> dict[str, Any]:
        """Hook method for subclasses to save additional state.

        Override to return a dictionary of state variables that should be
        included in the checkpoint. Default returns an empty dict.

        Returns:
            dict[str, Any]: Dictionary of subclass-specific state.
        """
        return {}

    def _load_subclass_state(self, state: dict[str, Any]) -> None:
        """Hook method for subclasses to load additional state.

        Override to restore state variables from the checkpoint dictionary.
        Default is a no-op.

        Args:
            state (dict[str, Any]): Dictionary of subclass-specific state.
        """

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
            tuple[int, int]: Shape (n_param_sets, n_layers * n_params_per_layer) that
                initial parameters should have for this quantum program.
        """
        return (self.optimizer.n_param_sets, self.n_layers * self.n_params_per_layer)

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

    def _initialize_param_sets(self) -> npt.NDArray[np.float64]:
        """Generate fresh parameter sets for a new optimization run."""
        total_params = self.n_layers * self.n_params_per_layer
        return self._rng.uniform(
            0, 2 * np.pi, (self.optimizer.n_param_sets, total_params)
        )

    def _optimizer_has_resume_state(self) -> bool:
        """Return True when the optimizer already carries resumable state."""
        if isinstance(self.optimizer, MonteCarloOptimizer):
            return self.optimizer._curr_population is not None
        if isinstance(self.optimizer, PymooOptimizer):
            return self.optimizer._curr_algorithm_obj is not None
        return False

    def _resolve_initial_param_sets(
        self, initial_params: npt.NDArray[np.float64] | None
    ) -> npt.NDArray[np.float64] | None:
        """Resolve the initial parameter sets for a fresh or resumed run."""
        if initial_params is not None and self._optimizer_has_resume_state():
            raise ValueError(
                "initial_params cannot be provided when resuming from optimizer state. "
                "Load a fresh program instance or reset the optimizer first."
            )

        if initial_params is not None:
            validated = np.atleast_2d(initial_params)
            self._validate_initial_params(validated)
            return validated.copy()

        if self._optimizer_has_resume_state():
            return None

        return self._initialize_param_sets()

    # ------------------------------------------------------------------ #
    # Pipeline builders
    # ------------------------------------------------------------------ #

    def dry_run(self) -> dict:
        """Run forward pass on all pipelines and print fan-out analysis."""
        if self._curr_params is None:
            self._initialize_params()
        return super().dry_run()

    def _get_dry_run_pipelines(self) -> dict[str, tuple]:
        factories = self.meta_circuit_factories
        result = {"cost": (self._cost_pipeline, factories["cost_circuit"])}
        if hasattr(self, "_measurement_pipeline") and self._measurement_pipeline:
            result["measurement"] = (
                self._measurement_pipeline,
                factories.get("meas_circuit", factories["cost_circuit"]),
            )
        return result

    def _build_pipeline_env(self, **overrides) -> PipelineEnv:
        """Construct a PipelineEnv, injecting the current parameter sets."""
        overrides.setdefault("param_sets", self._curr_params)
        return super()._build_pipeline_env(**overrides)

    def _build_cost_pipeline(self, spec_stage: Stage) -> CircuitPipeline:
        """Build the cost-evaluation pipeline.

        Stages: spec_stage → ParameterBinding → QEM → Measurement.

        QEM must come before Measurement so that reduce sees full-H
        scalars (after MeasurementStage recombines observable groups)
        rather than per-group values that don't match the classical
        simulation context.

        Args:
            spec_stage: A SpecStage producing MetaCircuit(s) from the
                cost Hamiltonian (e.g. TrotterSpecStage).
        """

        stages = [
            spec_stage,
            ParameterBindingStage(),
            QEMStage(protocol=self._qem_protocol),
            MeasurementStage(grouping_strategy=self._grouping_strategy),
        ]
        n_twirls = getattr(self._qem_protocol, "n_twirls", 0)
        if n_twirls > 0:
            stages.append(PauliTwirlStage(n_twirls=n_twirls))
        return CircuitPipeline(stages=stages)

    def _build_measurement_pipeline(self) -> CircuitPipeline:
        """Build the measurement pipeline for solution extraction.

        Stages: SingleCircuitSpec → Measurement → ParameterBinding.

        Note: QEM is intentionally excluded — ZNE error mitigation applies
        only to cost evaluation (expectation values), not probability extraction.
        """

        return CircuitPipeline(
            stages=[
                CircuitSpecStage(),
                MeasurementStage(),
                ParameterBindingStage(),
            ]
        )

    def _get_cost_pipeline_initial_spec(self) -> Any:
        """Return the initial spec passed into ``_cost_pipeline``."""
        return self.meta_circuit_factories["cost_circuit"]

    # ------------------------------------------------------------------ #
    # Execution
    # ------------------------------------------------------------------ #

    def _evaluate_cost_param_sets(
        self, param_sets: npt.NDArray[np.float64], **kwargs
    ) -> dict[int, float]:
        """Evaluate the cost pipeline for the provided parameter sets.

        Subclasses should prefer overriding the initial-spec hook over
        replacing the full evaluator.
        """
        normalized_param_sets = np.atleast_2d(param_sets)

        env = self._build_pipeline_env(param_sets=normalized_param_sets)
        result = self._cost_pipeline.run(
            initial_spec=self._get_cost_pipeline_initial_spec(),
            env=env,
        )
        self._total_circuit_count += env.artifacts.get("circuit_count", 0)
        self._total_run_time += env.artifacts.get("run_time", 0.0)
        self._current_execution_result = env.artifacts.get("_current_execution_result")

        indexed = {
            _extract_param_set_idx(key): value + self.loss_constant
            for key, value in result.items()
        }
        return dict(sorted(indexed.items()))

    def _evaluate_gradient_at(
        self, params: npt.NDArray[np.float64], **kwargs
    ) -> npt.NDArray[np.float64]:
        """Evaluate the parameter-shift gradient at a single parameter vector."""
        shifted_param_sets = self._grad_shift_mask + params
        exp_vals = self._evaluate_cost_param_sets(shifted_param_sets, **kwargs)
        exp_vals_arr = np.asarray(
            [value for _, value in sorted(exp_vals.items())],
            dtype=np.float64,
        )

        pos_shifts = exp_vals_arr[::2]
        neg_shifts = exp_vals_arr[1::2]
        return 0.5 * (pos_shifts - neg_shifts)

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

    def run(
        self,
        initial_params: npt.NDArray[np.float64] | None = None,
        perform_final_computation: bool = True,
        checkpoint_config: CheckpointConfig | None = None,
        **kwargs,
    ) -> tuple[int, float]:
        """Run the variational quantum algorithm.

        The outputs are stored in the algorithm object.

        Args:
            initial_params (npt.NDArray[np.float64] | None): Optional initial parameter
                sets for a fresh optimization run. Must have shape
                ``(n_param_sets, n_layers * n_params_per_layer)``. Cannot be
                combined with a checkpoint-resumed optimizer state.
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

            losses = self._evaluate_cost_param_sets(np.atleast_2d(params), **kwargs)

            losses = np.asarray(
                [value for _, value in sorted(losses.items())],
                dtype=np.float64,
            )

            if params.ndim > 1:
                return losses
            else:
                return losses.item()

        self._grad_shift_mask = _compute_parameter_shift_mask(
            self.n_layers * self.n_params_per_layer
        )

        last_grad_norm: float | None = None

        def grad_fn(params):
            nonlocal last_grad_norm

            self.reporter.info(
                message="📈 Computing Gradients 📈", iteration=self.current_iteration
            )

            grads = self._evaluate_gradient_at(params, **kwargs)

            last_grad_norm = float(np.linalg.norm(grads))

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

            self._param_history.append(
                np.atleast_2d(
                    np.asarray(intermediate_result.x, dtype=np.float64)
                ).copy()
            )

            current_loss = np.min(intermediate_result.fun)
            if current_loss < self._best_loss:
                self._best_loss = current_loss
                best_idx = np.argmin(intermediate_result.fun)
                self._best_params = intermediate_result.x[best_idx].copy()

            self.current_iteration += 1

            self.reporter.update(
                iteration=self.current_iteration, loss=float(current_loss)
            )

            # Checkpointing
            if checkpoint_config._should_checkpoint(self.current_iteration):
                self.save_state(checkpoint_config)

            if self._cancellation_event and self._cancellation_event.is_set():
                raise _CancelledError("Cancellation requested by batch.")

            # --- Early stopping ---
            if self._early_stopping is not None:
                reason = self._early_stopping.check(
                    current_loss,
                    grad_norm=last_grad_norm,
                )
                if reason is not None:
                    self._stop_reason = reason
                    self.reporter.info(
                        message=f"Early stopping triggered: {reason.value}",
                        iteration=self.current_iteration,
                    )
                    raise StopIteration

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

        resolved_initial_params = self._resolve_initial_param_sets(initial_params)

        try:
            self.optimize_result = self.optimizer.optimize(
                cost_fn=cost_fn,
                initial_params=resolved_initial_params,
                callback_fn=_iteration_counter,
                jac=grad_fn,
                max_iterations=self.max_iterations,
                rng=self._rng,
            )
        except (_CancelledError, StopIteration) as exc:
            if isinstance(exc, _CancelledError):
                message = "Optimization cancelled."
            else:
                reason = self._stop_reason.value if self._stop_reason else "Stopped"
                message = f"Early stopping: {reason}"

            self.optimize_result = OptimizeResult(
                x=np.atleast_2d(self._best_params),
                fun=np.atleast_1d(self._best_loss),
                nit=self.current_iteration,
                success=False,
                message=message,
            )

            if isinstance(exc, _CancelledError):
                return self._total_circuit_count, self._total_run_time
        else:
            self.optimize_result.success = True
            self.optimize_result.message = "Optimization converged."

            # Set _best_params from final result (source of truth)
            x = np.atleast_2d(self.optimize_result.x)
            self._best_params = x[np.argmin(self.optimize_result.fun)].copy()

        self._final_params = self.optimize_result.x

        if perform_final_computation:
            self._perform_final_computation(**kwargs)

        self.reporter.info(message="Finished successfully!")

        return self.total_circuit_count, self.total_run_time

    def _run_solution_measurement_for(
        self, param_sets: npt.NDArray[np.float64]
    ) -> None:
        """Execute measurement circuits via the pipeline for the provided parameter sets."""
        env = self._build_pipeline_env(param_sets=np.atleast_2d(param_sets))
        result = self._measurement_pipeline.run(
            initial_spec=self.meta_circuit_factories["meas_circuit"],
            env=env,
        )
        self._total_circuit_count += env.artifacts.get("circuit_count", 0)
        self._total_run_time += env.artifacts.get("run_time", 0.0)
        self._current_execution_result = env.artifacts.get("_current_execution_result")

        indexed = {_extract_param_set_idx(key): value for key, value in result.items()}
        self._best_probs = dict(sorted(indexed.items()))
