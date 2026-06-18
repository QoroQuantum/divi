# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import logging
import pickle
from abc import abstractmethod
from datetime import datetime
from pathlib import Path
from queue import Queue
from typing import Any, ClassVar, Literal, TypeAlias
from warnings import warn

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator
from qiskit.quantum_info import SparsePauliOp
from scipy.optimize import OptimizeResult

from divi.backends import CircuitRunner
from divi.circuits import MetaCircuit
from divi.exceptions import ExecutionCancelledError
from divi.pipeline import (
    CircuitPipeline,
    GroupingStrategy,
    PipelineEnv,
    PipelineSet,
    ResultFormat,
    ShotDistStrategy,
    Stage,
)
from divi.pipeline._compilation import _extract_param_set_idx
from divi.pipeline.stages import (
    CircuitSpecStage,
    MeasurementStage,
    ParameterBindingStage,
)
from divi.qprog import ObservableMeasuringMixin
from divi.qprog._solution_sampling_mixin import SolutionSamplingMixin
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
from divi.qprog.optimizers import (
    MonteCarloOptimizer,
    Optimizer,
    PymooOptimizer,
    ScipyMethod,
    ScipyOptimizer,
)
from divi.qprog.quantum_program import QuantumProgram
from divi.reporting import TerminalStatus
from divi.viz import ProgramViz

logger = logging.getLogger(__name__)

_RUN_INSTRUCTION = "Call run() to execute the optimization."

ParamHistoryMode: TypeAlias = Literal["all_evaluated", "best_per_iteration"]


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
    # Only solution-sampling programs (SolutionSamplingMixin) carry _best_probs;
    # it maps a parameter-set index to that set's {bitstring: probability} dict.
    best_probs: dict[int, dict[str, float]] = Field(
        default_factory=dict, validation_alias="_best_probs"
    )
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
            alias = field.validation_alias
            target_attr = alias if isinstance(alias, str) else name

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


class VariationalQuantumAlgorithm(ObservableMeasuringMixin, QuantumProgram):
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
        _losses_history: History of loss values during optimization.
        _param_history: Raw per-callback parameter batches;
            use :meth:`param_history` to read copies with optional filtering.
        _final_params: Final optimized parameters.
        _best_params: Parameters that achieved the best loss.
        _best_loss (float): Best loss achieved during optimization.
        _circuits: Generated quantum circuits.
        _total_circuit_count (int): Total number of circuits executed.
        _total_run_time (float): Total execution time in seconds.
        _seed: Random seed for parameter initialization.
        _rng: Random number generator.

        _grouping_strategy: Strategy for grouping quantum operations.
        _qem_protocol: Quantum error mitigation protocol.
        _cancellation_event: Event for graceful termination.
        _meta_circuit_factories (dict): Lazily-built mapping of circuit names to MetaCircuit factories.
    """

    # Subclass-populated declarations.
    #
    # ``_supports_fixed_param_scans`` defaults to True; override to False for
    # VQAs whose parameter space varies during optimization (e.g. depth
    # schedules) so ``divi.viz`` fixed-parameter scans reject them. The rest
    # have no default — each concrete VQA must assign them during ``__init__``
    # (or override as a property) or the corresponding methods will raise
    # AttributeError.
    _supports_fixed_param_scans: ClassVar[bool] = True
    current_iteration: int
    n_layers: int
    loss_constant: float
    cost_hamiltonian: SparsePauliOp
    """The cost Hamiltonian for the variational problem."""

    _grouping_strategy: GroupingStrategy
    _shot_distribution: ShotDistStrategy | None
    _best_params: npt.NDArray[np.float64]
    _final_params: npt.NDArray[np.float64]
    _meta_circuit_factories: dict[str, MetaCircuit] | None

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
            grouping_strategy (str): Strategy for partitioning Hamiltonian terms
                into compatible measurement groups; one circuit is executed per
                group. Options: ``"qwc"`` (qubit-wise-commuting — most
                compact), ``"wires"`` (group by support wires), or ``None``
                (one circuit per term). Defaults to ``"qwc"``.
            shot_distribution (str or callable, optional): Focus the backend's
                shot budget on the Hamiltonian terms that matter most.
                Without this option, every measurement group is sampled with
                the backend's full shot count, even tiny terms with little
                impact on the final energy. With ``shot_distribution`` set,
                the same total budget is split across groups according to
                their importance — reducing variance without spending more
                shots.

                Available strategies:

                - ``"uniform"`` — equal split across groups.
                - ``"weighted"`` — proportional to per-group coefficient L1
                  norm; dominant Hamiltonian terms get more shots.
                - ``"weighted_random"`` — multinomial sample of the same
                  probabilities; may drop more low-weight groups than the
                  deterministic ``"weighted"`` for the same budget.
                - A callable ``(group_l1_norms, total_shots) -> per_group_shots``
                  for fully custom allocation.

                Example::

                    vqe = MyVQA(
                        backend=QiskitSimulator(shots=1000),
                        shot_distribution="weighted",
                    )
                    vqe.run()

                Only valid when sampling is actually used. Setting it on a
                backend that computes expectation values analytically
                (``grouping_strategy="_backend_expval"``) is rejected because
                shots are ignored in that mode. Defaults to ``None`` (every
                group receives the full shot budget).
            precision (int): Forwarded to
                :class:`~divi.qprog.QuantumProgram` — decimal places for
                numeric parameter values in QASM conversion. Higher values
                produce longer QASM strings (more data sent to cloud
                backends); lower values trade resolution for compactness.
                Defaults to :data:`~divi.circuits.DEFAULT_PRECISION`.

        Note:
            Solution-extracting subclasses (VQE/QAOA/PCE) also accept
            ``decode_solution_fn`` via
            :class:`~divi.qprog.SolutionSamplingMixin`.
        """

        program_id = kwargs.pop("program_id", None)

        super().__init__(
            backend=backend,
            seed=seed,
            progress_queue=progress_queue,
            program_id=program_id,
            **kwargs,
        )

        # --- Optimization Results & History ---
        self._losses_history = []
        self._param_history: list[npt.NDArray[np.float64]] = []
        self._best_params = np.array([], dtype=np.float64)
        self._final_params = np.array([], dtype=np.float64)
        self._best_loss = float("inf")
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
        self._rng = np.random.default_rng(self._seed)
        # The optimizer draws perturbations from an independent stream, spawned
        # from the same seed lineage, so env/metric/QDrift draws on self._rng can
        # never shift optimizer randomness (and vice versa). Reproducible under a
        # fixed seed; a stochastic optimizer's stream differs once from when it
        # shared self._rng.
        self._optimizer_rng = self._rng.spawn(1)[0]

        # --- Optimizer Configuration ---
        self.optimizer = optimizer if optimizer is not None else MonteCarloOptimizer()

        # --- Early Stopping ---
        self._early_stopping = early_stopping
        self._stop_reason: StopReason | None = None

        # --- Circuit Factory & Templates ---
        self._meta_circuit_factories = None

    def _has_run_optimization(self) -> bool:
        """Check if optimization has been run at least once.

        Returns:
            bool: True if optimization has been run, False otherwise.
        """
        return len(self._losses_history) > 0

    def has_results(self) -> bool:
        return self._has_run_optimization()

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

    @property
    @abstractmethod
    def n_params_per_layer(self) -> int:
        """Number of trainable parameters per ansatz layer.

        Used by the base class to compute the total parameter count as
        ``n_layers * n_params_per_layer``.
        """

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

    def save_state(self, checkpoint_config: CheckpointConfig) -> Path:
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
            return self.optimizer._has_checkpoint
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

    @property
    def _cost_pipeline(self) -> CircuitPipeline:
        """The cost-evaluation pipeline that drives optimization."""
        return self._pipelines["cost"]

    def _build_pipeline_env(self, **overrides) -> PipelineEnv:
        """Construct a PipelineEnv for the provided parameter sets.

        When no ``param_sets`` override is given, defaults to a deterministic
        zeros placeholder of the right shape — building an env must not draw from
        the program RNG (callers that need real initial parameters draw them via
        :meth:`_resolve_initial_param_sets`; spec-stage / dry-run paths never bind
        these values).
        """
        if "param_sets" not in overrides:
            total_params = self.n_layers * self.n_params_per_layer
            overrides["param_sets"] = np.zeros(
                (self.optimizer.n_param_sets, total_params)
            )
        if "rng" not in overrides:
            overrides["rng"] = self._rng
        return super()._build_pipeline_env(**overrides)

    @property
    def _loss_constant_consumed(self) -> bool:
        """Whether a cost-pipeline component already folds ``loss_constant`` in.

        When ``True``, :meth:`_evaluate_cost_param_sets` skips its post-reduction
        add to avoid double-counting. ``False`` for vanilla VQE/QAOA/CustomVQA;
        data-binding subclasses override it.
        """
        return False

    # ------------------------------------------------------------------ #
    # Pipeline assembly — one generic builder shared by every named pipeline.
    # ------------------------------------------------------------------ #

    def _assemble_pipeline(
        self,
        spec_stage: Stage,
        terminal_stage: Stage,
        *,
        result_format: ResultFormat,
        extra_stages: tuple[Stage, ...] = (),
    ) -> CircuitPipeline:
        """Assemble a variational pipeline with parameter binding."""
        mitigation_stages = self._mitigation_stages(result_format)
        bind_early = (
            bool(mitigation_stages) and self._qem_protocol.requires_bound_params
        )

        stages: list[Stage] = [spec_stage, *extra_stages]
        if bind_early:
            stages.append(ParameterBindingStage())
        stages.extend(mitigation_stages)
        stages.append(terminal_stage)
        if not bind_early:
            stages.append(ParameterBindingStage())
        return CircuitPipeline(stages=stages)

    def _expectation_pipeline(self) -> CircuitPipeline:
        """The canonical pipeline measuring an arbitrary observable-carrying
        MetaCircuit as expectation values on this program's ansatz.

        Used both as the default ``"cost"`` pipeline (VQE/CustomVQA) and, on
        demand, by natural-gradient metric estimators (which supply their own
        MetaCircuit). QAOA/PCE replace ``"cost"`` with a specialized pipeline but
        still expose this for the metric.
        """
        return self._assemble_pipeline(
            CircuitSpecStage(),
            MeasurementStage(
                grouping_strategy=self._grouping_strategy,
                shot_distribution=self._shot_distribution,
            ),
            result_format=ResultFormat.EXPVALS,
        )

    def _build_pipelines(self) -> PipelineSet:
        """Register the ``"cost"`` pipeline (an expectation measurement of the cost
        ansatz). ``"sample"`` is added by :class:`SolutionSamplingMixin`; QAOA/PCE
        replace ``"cost"`` — all via cooperative ``super()._build_pipelines().with_(...)``.
        """
        return PipelineSet(
            {
                "cost": (
                    self._expectation_pipeline(),
                    lambda: self.meta_circuit_factories["cost_circuit"],
                ),
            }
        )

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
        result = self._run_pipeline("cost", param_sets=np.atleast_2d(param_sets))

        constant = 0.0 if self._loss_constant_consumed else self.loss_constant
        indexed = {
            _extract_param_set_idx(key): float(value[0]) + constant
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
            [value for value in exp_vals.values()],
            dtype=np.float64,
        )

        pos_shifts = exp_vals_arr[::2]
        neg_shifts = exp_vals_arr[1::2]
        return 0.5 * (pos_shifts - neg_shifts)

    def _coerce_sample_params(
        self, params: npt.NDArray[np.float64] | None
    ) -> npt.NDArray[np.float64]:
        """Resolve and validate parameters for :meth:`SolutionSamplingMixin.sample_solution`.

        With ``params=None`` falls back to the trained ``_best_params`` (raising if
        optimization has not run); otherwise validates the trailing axis against
        ``n_layers * n_params_per_layer``. Supplies the variational parameter-model
        knowledge that :class:`SolutionSamplingMixin` is agnostic to.
        """
        if params is None:
            if len(self._best_params) == 0:
                raise RuntimeError(
                    "sample_solution() was called without explicit `params` "
                    "but no trained parameters are available. Either pass "
                    "`params=...` or call run() first."
                )
            return self._best_params

        params_arr = np.asarray(params, dtype=np.float64)
        expected = self.n_layers * self.n_params_per_layer
        if params_arr.shape[-1] != expected:
            raise ValueError(
                f"params last-axis size ({params_arr.shape[-1]}) does not "
                f"match n_layers * n_params_per_layer ({expected})."
            )
        return params_arr

    def run(
        self,
        initial_params: npt.NDArray[np.float64] | None = None,
        perform_final_computation: bool = True,
        checkpoint_config: CheckpointConfig | None = None,
        **kwargs,
    ) -> "VariationalQuantumAlgorithm":
        """Run the variational quantum algorithm.

        The outputs are stored in the algorithm object and can be accessed via
        properties such as ``total_circuit_count``, ``total_run_time``,
        ``losses_history``, and ``best_params``.

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
            VariationalQuantumAlgorithm: Returns ``self`` for method chaining.
        """
        # Initialize checkpointing
        if checkpoint_config is None:
            checkpoint_config = CheckpointConfig()

        if checkpoint_config.checkpoint_dir:
            logger.info(
                f"Using checkpoint directory: {checkpoint_config.checkpoint_dir}"
            )

        self.optimizer.validate_program(self)

        if (
            checkpoint_config.checkpoint_dir is not None
            and not self.optimizer.supports_checkpointing
        ):
            raise ValueError(
                f"{type(self.optimizer).__name__} does not support checkpointing, "
                "but checkpoint_config.checkpoint_dir was set. Remove the "
                "checkpoint directory or use a checkpointing-capable optimizer "
                "(e.g. MonteCarloOptimizer, PymooOptimizer, GridSearchOptimizer)."
            )

        # Extract max_iterations from kwargs if present (for compatibility with subclasses)
        max_iterations = kwargs.pop("max_iterations", self.max_iterations)
        if max_iterations != self.max_iterations:
            self.max_iterations = max_iterations

        if self.max_iterations <= self.current_iteration:
            warn(
                f"max_iterations ({self.max_iterations}) is less than or equal to "
                f"current_iteration ({self.current_iteration}). The optimization will "
                f"not run additional iterations since the maximum has already been "
                f"reached.",
                UserWarning,
            )

        def cost_fn(params):
            self._evaluation_counter += 1
            self.reporter.info(
                message="💸 Computing Cost 💸", iteration=self.current_iteration
            )

            losses = self._evaluate_cost_param_sets(np.atleast_2d(params), **kwargs)

            losses = np.asarray(
                [value for value in losses.values()],
                dtype=np.float64,
            )

            if params.ndim > 1:
                return losses
            else:
                return losses.item()

        self._grad_shift_mask = _compute_parameter_shift_mask(
            self.n_layers * self.n_params_per_layer
        )

        # Let the optimizer contribute any extra evaluators it needs (e.g. a
        # metric-based optimizer binds its metric estimator to this program and
        # returns a fused gradient + ``metric_fn``).
        extra_evaluators = self.optimizer.build_evaluators(self)
        jac_fn = extra_evaluators.get("jac")

        last_grad_norm: float | None = None

        def grad_fn(params):
            nonlocal last_grad_norm

            self.reporter.info(
                message="📈 Computing Gradients 📈", iteration=self.current_iteration
            )

            if jac_fn is not None:
                grads = jac_fn(params)
            else:
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
                raise ExecutionCancelledError("Cancellation requested by batch.")

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

        optimize_kwargs: dict[str, Any] = dict(
            cost_fn=cost_fn,
            initial_params=resolved_initial_params,
            callback_fn=_iteration_counter,
            jac=grad_fn,
            max_iterations=self.max_iterations,
            rng=self._optimizer_rng,
        )
        # Forward every extra evaluator the optimizer declared except ``jac``,
        # which is already folded into ``grad_fn`` above. Each optimizer pops the
        # keys it understands (e.g. ``metric_fn``, ``fidelity_fn``) and ignores
        # the rest.
        for key, evaluator in extra_evaluators.items():
            if key != "jac":
                optimize_kwargs[key] = evaluator

        with self._install_cancellation_handler():
            try:
                self.optimize_result = self.optimizer.optimize(**optimize_kwargs)
            except StopIteration:
                reason = self._stop_reason.value if self._stop_reason else "Stopped"
                self.optimize_result = OptimizeResult(
                    x=np.atleast_2d(self._best_params),
                    fun=np.atleast_1d(self._best_loss),
                    nit=self.current_iteration,
                    success=False,
                    message=f"Early stopping: {reason}",
                )
            except ExecutionCancelledError as exc:
                # ``KeyboardInterrupt`` is deliberately NOT caught here:
                # the second Ctrl+C re-raises ``KeyboardInterrupt`` from
                # the signal handler as the documented hard-abort path,
                # and intercepting it would defeat that.
                message = "Cancelled by user"
                self.optimize_result = OptimizeResult(
                    x=np.atleast_2d(self._best_params),
                    fun=np.atleast_1d(self._best_loss),
                    nit=self.current_iteration,
                    success=False,
                    message=message,
                )
                # The pipeline already best-effort-cancelled the in-flight
                # job when it raised; no redundant call needed here.
                self.reporter.info(
                    message=message, final_status=TerminalStatus.CANCELLED
                )
                raise ExecutionCancelledError(message) from exc
            else:
                self.optimize_result.success = True
                self.optimize_result.message = "Optimization converged."

                # Set _best_params from final result (source of truth)
                x = np.atleast_2d(self.optimize_result.x)
                self._best_params = x[np.argmin(self.optimize_result.fun)].copy()

        # Canonical 1-D best parameters (the optimizer result contract); the
        # early-stop/cancel branches above carry a 2-D (1, n) best, so squeeze.
        self._final_params = np.atleast_1d(np.asarray(self.optimize_result.x).squeeze())

        if perform_final_computation and isinstance(self, SolutionSamplingMixin):
            self.sample_solution(**kwargs)

        self.reporter.info(
            message="Finished successfully!", final_status=TerminalStatus.SUCCESS
        )

        return self
