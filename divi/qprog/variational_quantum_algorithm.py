# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import copy
import logging
import pickle
from abc import abstractmethod
from collections.abc import Sequence
from functools import cached_property, wraps
from pathlib import Path
from typing import Any, ClassVar, Literal, Self, TypeAlias, cast
from warnings import warn

import numpy as np
import numpy.typing as npt
from qiskit.quantum_info import SparsePauliOp
from scipy.optimize import OptimizeResult

from divi.backends import CircuitRunner
from divi.circuits import MetaCircuit
from divi.exceptions import ExecutionCancelledError
from divi.pipeline import (
    CircuitPipeline,
    CircuitPreprocessor,
    GroupingStrategy,
    PipelineCadence,
    PipelineEnv,
    ResultFormat,
    ShotDistStrategy,
    Stage,
)
from divi.pipeline import cost_preprocessor as _default_cost_preprocessor
from divi.pipeline.stages import ParameterBindingStage
from divi.qprog import ObservableMeasuringMixin
from divi.qprog._program_checkpoint import (
    OptimizerConfig,
    SubclassState,
    VQACheckpoint,
)
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
    GridSearchOptimizer,
    MonteCarloOptimizer,
    Optimizer,
    PymooOptimizer,
    ScipyMethod,
    ScipyOptimizer,
)
from divi.qprog.quantum_program import QuantumProgram, reject_unclaimed_run_kwargs
from divi.reporting._events import (
    ProgressEvent,
    TerminalStatus,
)
from divi.viz import ProgramViz

logger = logging.getLogger(__name__)

_RUN_INSTRUCTION = "Call run() to execute the optimization."

# Every optimizer whose ``supports_checkpointing`` is True must appear here, or
# its checkpoints would be written but never loadable.
_CHECKPOINTABLE_OPTIMIZERS: dict[str, type[Optimizer]] = {
    "GridSearchOptimizer": GridSearchOptimizer,
    "MonteCarloOptimizer": MonteCarloOptimizer,
    "PymooOptimizer": PymooOptimizer,
}

ParamHistoryMode: TypeAlias = Literal["all_evaluated", "best_per_iteration"]


def _with_optimisation_progress(run_method):
    """Bind one direct session around an otherwise unbound VQA run."""

    @wraps(run_method)
    def wrapped(self, *args, **kwargs):
        max_iterations = kwargs.get("max_iterations", self.max_iterations)
        total = max(0, max_iterations - self.current_iteration)
        with self._ensure_progress_session(label="Optimising", total=total):
            return run_method(self, *args, **kwargs)

    return wrapped


def _compute_parameter_shift_rule(
    frequencies: Sequence[tuple[float, int]],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Build a parameter-shift rule from each parameter's frequency family.

    A parameter whose energy carries frequencies ``{omega, ..., order * omega}``
    is differentiated exactly by ``2 * order`` evaluations at equidistant shifts;
    ``(1.0, 1)`` is the familiar two-term ``+-pi/2`` rule. Parameters may need
    different term counts, so the evaluations form one flat batch.

    Args:
        frequencies: One ``(omega, order)`` pair per parameter.

    Returns:
        ``(shifts, weights)``: a ``(n_evaluations, n_params)`` array of offsets to
        add to the parameter vector, and a ``(n_params, n_evaluations)`` array
        whose product with the evaluated values is the derivative.

    Raises:
        ValueError: If any ``omega`` is not positive or ``order`` is below 1.
    """
    n_params = len(frequencies)
    shift_blocks = []
    weight_blocks = []
    for index, (omega, order) in enumerate(frequencies):
        if not omega > 0:
            raise ValueError(f"omega must be positive; got {omega}.")
        if order < 1:
            raise ValueError(f"order must be at least 1; got {order}.")

        term = np.arange(1, 2 * order + 1)
        shifts = (2 * term - 1) * np.pi / (2 * order * omega)
        # The energy has period 2 pi / omega, so folding the shifts into its
        # principal interval keeps the rule exact and the single-frequency case
        # at the conventional +-pi/2.
        period = 2.0 * np.pi / omega
        shifts = (shifts + 0.5 * period) % period - 0.5 * period
        coefficients = (
            (-1.0) ** (term - 1)
            * omega
            / (4 * order * np.sin((2 * term - 1) * np.pi / (4 * order)) ** 2)
        )

        block = np.zeros((len(shifts), n_params))
        block[:, index] = shifts
        shift_blocks.append(block)
        weights = np.zeros((n_params, len(shifts)))
        weights[index, :] = coefficients
        weight_blocks.append(weights)

    if not shift_blocks:
        return np.zeros((0, 0)), np.zeros((0, 0))
    return np.vstack(shift_blocks), np.hstack(weight_blocks)


def _argmin_finite(values: npt.NDArray[np.float64]) -> int | None:
    """Index of the smallest finite entry, or ``None`` if every entry is non-finite.

    Keeps a NaN/inf loss from being selected as the best parameter set.
    """
    arr = np.asarray(values, dtype=np.float64).ravel()
    finite = np.isfinite(arr)
    if not finite.any():
        return None
    return int(np.argmin(np.where(finite, arr, np.inf)))


class VariationalQuantumAlgorithm(ObservableMeasuringMixin, QuantumProgram):
    """Base class for variational quantum algorithms.

    This class provides the foundation for implementing variational quantum
    algorithms in Divi. It handles circuit execution, parameter optimisation,
    and result management for algorithms that optimise parameterised quantum
    circuits to minimise cost functions.

    Variational algorithms work by:
    1. Generating parameterised quantum circuits
    2. Executing circuits on quantum hardware/simulators
    3. Computing expectation values of cost Hamiltonians
    4. Using classical optimizers to update parameters
    5. Iterating until convergence

    Attributes:
        _losses_history: History of loss values during optimisation.
        _param_history: Raw per-callback parameter batches;
            use :meth:`param_history` to read copies with optional filtering.
        _final_params: Final optimised parameters.
        _best_params: Parameters that achieved the best loss.
        _best_loss (float): Best loss achieved during optimisation.
        _circuits: Generated quantum circuits.
        _total_circuit_count (int): Total number of circuits executed.
        _total_run_time (float): Total execution time in seconds.
        _seed: Random seed for parameter initialisation.
        _rng: Random number generator.

        _grouping_strategy: Strategy for grouping quantum operations.
        _qem_protocol: Quantum error mitigation protocol.
        _cancellation_event: Event for graceful termination.
        _cost_circuit: Lazily-built cost MetaCircuit for this program (or ``None``).
    """

    # Subclass-populated declarations.
    #
    # ``_supports_fixed_param_scans`` defaults to True; override to False for
    # VQAs whose parameter space varies during optimisation (e.g. depth
    # schedules) so ``divi.viz`` fixed-parameter scans reject them. The rest
    # have no default — each concrete VQA must assign them during ``__init__``
    # (or override as a property) or the corresponding methods will raise
    # AttributeError.
    _supports_fixed_param_scans: ClassVar[bool] = True
    current_iteration: int
    max_iterations: int
    n_layers: int
    loss_constant: float
    cost_hamiltonian: SparsePauliOp
    """The cost Hamiltonian for the variational problem."""

    _grouping_strategy: GroupingStrategy
    _shot_distribution: ShotDistStrategy | None
    _best_params: npt.NDArray[np.float64]
    _final_params: npt.NDArray[np.float64]
    _cost_circuit: MetaCircuit | None

    def _make_checkpoint(
        self,
        checkpoint_dir: Path,
    ) -> VQACheckpoint:
        return VQACheckpoint.from_program(self, kind="program_completion")

    def _restore_checkpoint(self, checkpoint_json: str, checkpoint_dir: Path) -> bool:
        checkpoint = VQACheckpoint.model_validate_json(checkpoint_json)
        if checkpoint.program_type != type(self).__name__:
            raise ValueError(
                f"Checkpoint is for {checkpoint.program_type}, not {type(self).__name__}."
            )
        if checkpoint.kind != "program_completion":
            raise ValueError("Expected a completed VQA checkpoint.")
        self._apply_checkpoint_state(checkpoint)
        return True

    def __init__(
        self,
        backend: CircuitRunner,
        optimizer: Optimizer | None = None,
        seed: int | None = None,
        early_stopping: EarlyStopping | None = None,
        **kwargs,
    ):
        """Initialise the VariationalQuantumAlgorithm.

        This constructor is specifically designed for hybrid quantum-classical
        variational algorithms. The instance variables `n_layers` and
        `n_params_per_layer` must be set by subclasses, where:
        - `n_layers` is the number of layers in the quantum circuit.
        - `n_params_per_layer` is the number of parameters per layer.

        For exotic variational algorithms where these variables may not be applicable,
        the `_initialize_param_sets` method should be overridden to generate the
        starting parameters for a fresh optimisation run.

        Args:
            backend (CircuitRunner): Quantum circuit execution backend.
            optimizer (Optimizer): The optimizer to use for parameter optimisation.
                Required — passing ``None`` (or omitting it) raises ``ValueError``.
            seed (int | None): Random seed for parameter initialisation. Defaults to None.
            early_stopping (EarlyStopping | None): Early stopping controller. When
                provided, the optimisation loop will be halted if any of the
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

        super().__init__(
            backend=backend,
            seed=seed,
            **kwargs,
        )

        # --- Optimisation Results & History ---
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
        if optimizer is None:
            raise ValueError(
                "A VariationalQuantumAlgorithm requires an explicit optimizer; "
                "pass one (e.g. optimizer=MonteCarloOptimizer())."
            )
        self.optimizer = optimizer

        # --- Early Stopping ---
        self._early_stopping = early_stopping
        self._stop_reason: StopReason | None = None

        # --- Circuit Factory & Templates ---
        self._cost_circuit = None

    def _has_run_optimization(self) -> bool:
        """Check if optimisation has been run at least once.

        Returns:
            bool: True if optimisation has been run, False otherwise.
        """
        return len(self._losses_history) > 0

    def has_results(self) -> bool:
        return self._has_run_optimization()

    @property
    def stop_reason(self) -> StopReason | None:
        """Reason the optimisation was stopped early, or ``None``.

        Returns:
            StopReason | None: The :class:`~divi.qprog.early_stopping.StopReason`
                that triggered early stopping, or ``None`` if optimisation
                completed normally or has not been run yet.
        """
        return self._stop_reason

    @property
    def losses_history(self) -> list[dict]:
        """Get a copy of the optimisation loss history.

        Each entry is a dictionary mapping parameter indices to loss values.

        Returns:
            list[dict]: Copy of the loss history. Modifications to this list
                will not affect the internal state.
        """
        if not self._has_run_optimization():
            warn(
                "losses_history is empty. Optimisation has not been run yet. "
                f"{_RUN_INSTRUCTION}",
                UserWarning,
                stacklevel=2,
            )
        return self._losses_history.copy()

    def param_history(
        self,
        mode: ParamHistoryMode = "all_evaluated",
    ) -> list[npt.NDArray[np.float64]]:
        """Parameter vectors recorded at each optimisation callback.

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
                "Parameter history is unavailable because optimisation has not "
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
                "min_losses_per_iteration is empty. Optimisation has not been run yet. "
                f"{_RUN_INSTRUCTION}",
                UserWarning,
                stacklevel=2,
            )
        return [min(loss_dict.values()) for loss_dict in self._losses_history]

    @property
    def final_params(self) -> npt.NDArray[np.float64]:
        """Get a copy of the final optimised parameters.

        Returns:
            npt.NDArray[np.float64]: Copy of the final parameters. Modifications to this array
                will not affect the internal state.
        """
        if len(self._final_params) == 0 or not self._has_run_optimization():
            warn(
                "final_params is not available. Optimisation has not been run yet. "
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
                "best_params is not available. Optimisation has not been run yet. "
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
                "best_loss has not been computed yet. Optimisation has not been run. "
                f"{_RUN_INSTRUCTION}",
                UserWarning,
                stacklevel=2,
            )
        elif self._best_loss == float("inf"):
            # Defensive check: if optimisation ran but best_loss is still inf, something is wrong
            raise RuntimeError(
                "best_loss is still infinite after optimisation. This indicates a problem "
                "with the optimisation process. The optimisation callback may not have executed "
                "correctly, or all computed losses were infinite."
            )
        return self._best_loss

    @property
    def viz(self):
        """Access visualisation helpers for this variational program.

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
    def cost_circuit(self) -> MetaCircuit:
        """The cost MetaCircuit for this program (lazily built, cached).

        Note: When used with ProgramEnsemble, this is initialised sequentially
        in the main thread before parallel execution to avoid thread-safety issues.
        """
        if self._cost_circuit is None:
            self._cost_circuit = self._create_cost_circuit()
        return self._cost_circuit

    @abstractmethod
    def _create_cost_circuit(self) -> MetaCircuit:
        pass

    @property
    @abstractmethod
    def n_params_per_layer(self) -> int:
        """Number of trainable parameters per ansatz layer.

        Used by the base class to compute the total parameter count as
        ``n_layers * n_params_per_layer``.
        """

    @property
    def n_params(self) -> int:
        """Total trainable parameters, ``n_layers * n_params_per_layer``."""
        return self.n_layers * self.n_params_per_layer

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
            raise RuntimeError("Cannot save checkpoint: optimisation has not been run.")

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
        state = VQACheckpoint.from_program(self, kind="iteration")

        state_file = checkpoint_path / PROGRAM_STATE_FILE
        _atomic_write(state_file, state.model_dump_json(indent=2))

        return checkpoint_path

    @classmethod
    def _resolve_checkpoint_path(
        cls,
        checkpoint_dir: Path | str,
        subdirectory: str | None = None,
    ) -> Path:
        """Resolve a program checkpoint, allowing subclasses to add nesting."""
        return resolve_checkpoint_path(checkpoint_dir, subdirectory)

    @classmethod
    def _load_checkpoint_state(
        cls,
        checkpoint_dir: Path | str,
        subdirectory: str | None = None,
    ) -> tuple[Path, VQACheckpoint]:
        """Read and validate the program portion of a checkpoint."""
        checkpoint_path = cls._resolve_checkpoint_path(checkpoint_dir, subdirectory)
        state = _load_and_validate_pydantic_model(
            checkpoint_path / PROGRAM_STATE_FILE,
            VQACheckpoint,
            required_fields=["kind", "program_type", "current_iteration"],
        )
        if state.kind != "iteration":
            raise ValueError("Expected an iterative VQA checkpoint.")
        if state.program_type != cls.__name__:
            raise ValueError(
                f"Checkpoint contains {state.program_type}, not {cls.__name__}."
            )
        return checkpoint_path, state

    @staticmethod
    def _load_checkpoint_optimizer(
        checkpoint_path: Path, state: VQACheckpoint
    ) -> Optimizer:
        """Reconstruct the optimizer recorded in a program checkpoint."""
        opt_config = state.optimizer_config
        if opt_config is None:
            raise ValueError("Iterative checkpoint has no optimizer configuration.")
        optimizer_class = _CHECKPOINTABLE_OPTIMIZERS.get(opt_config.type)
        if optimizer_class is None:
            supported = ", ".join(sorted(_CHECKPOINTABLE_OPTIMIZERS))
            raise ValueError(
                f"Unsupported optimizer type: {opt_config.type}. "
                f"Checkpoints can be loaded for: {supported}."
            )
        return optimizer_class.load_state(checkpoint_path)

    def _restore_state(
        self,
        checkpoint_dir: Path | str,
        subdirectory: str | None = None,
    ) -> Self:
        """Restore a checkpoint onto this already-constructed program."""
        checkpoint_path, state = type(self)._load_checkpoint_state(
            checkpoint_dir, subdirectory
        )
        return self._restore_loaded_checkpoint(checkpoint_path, state)

    def _restore_loaded_checkpoint(
        self, checkpoint_path: Path, state: VQACheckpoint
    ) -> Self:
        """Apply an already loaded checkpoint to this program."""
        optimizer = self._load_checkpoint_optimizer(checkpoint_path, state)
        self._apply_checkpoint_state(state, optimizer=optimizer)
        return self

    def _apply_checkpoint_state(
        self,
        state: VQACheckpoint,
        *,
        optimizer: Optimizer | None = None,
    ) -> None:
        """Apply validated state atomically to this program instance."""
        attributes = {
            name: (
                value.copy()
                if isinstance(value, (dict, list, set, np.ndarray))
                else value
            )
            for name, value in self.__dict__.items()
        }
        rng_state = copy.deepcopy(self._rng.bit_generator.state)
        try:
            if optimizer is not None:
                self.optimizer = optimizer
            state.restore(self)
        except Exception:
            self.__dict__.clear()
            self.__dict__.update(attributes)
            self._rng.bit_generator.state = rng_state
            raise

    @classmethod
    def load_state(
        cls,
        checkpoint_dir: Path | str,
        backend: CircuitRunner,
        subdirectory: str | None = None,
        **kwargs,
    ) -> Self:
        """Load program state from a checkpoint directory."""
        checkpoint_path, state = cls._load_checkpoint_state(
            checkpoint_dir, subdirectory
        )
        optimizer = cls._load_checkpoint_optimizer(checkpoint_path, state)
        program = cls(backend=backend, optimizer=optimizer, seed=state.seed, **kwargs)
        state.restore(program)
        return program

    def get_expected_param_shape(self) -> tuple[int, int]:
        """
        Get the expected shape for initial parameters.

        Returns:
            tuple[int, int]: Shape (n_param_sets, n_layers * n_params_per_layer) that
                initial parameters should have for this quantum program.
        """
        return (self.optimizer.n_param_sets, self.n_params)

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
        """Generate fresh parameter sets for a new optimisation run."""
        return self._rng.uniform(
            0, 2 * np.pi, (self.optimizer.n_param_sets, self.n_params)
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

    def _build_pipeline_env(self, **overrides) -> PipelineEnv:
        """Construct a PipelineEnv for the provided parameter sets.

        When no ``param_sets`` override is given, defaults to a deterministic
        zeros placeholder of the right shape — building an env must not draw from
        the program RNG (callers that need real initial parameters draw them via
        :meth:`_resolve_initial_param_sets`; spec-stage / dry-run paths never bind
        these values).
        """
        if "param_sets" not in overrides:
            overrides["param_sets"] = np.zeros(
                (self.optimizer.n_param_sets, self.n_params)
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
        return CircuitPipeline(
            stages=stages,
            suppress_performance_warnings=self._suppress_performance_warnings,
        )

    def _bindable_parameter_count(self) -> int:
        return self.n_params

    def _validate_before_preview(self) -> None:
        """Reject counts a run would reject, so the preview fails where it is cheap."""
        super()._validate_before_preview()
        if self.n_layers < 1:
            raise ValueError(
                f"n_layers must be >= 1, got {self.n_layers}; a circuit with no "
                "ansatz layers has nothing to optimize."
            )
        if self.max_iterations < 1:
            raise ValueError(
                f"max_iterations must be >= 1, got {self.max_iterations}; run() "
                "would warn and return without optimising anything."
            )

    def _dry_run_env(
        self, preprocessor: CircuitPreprocessor, rng: np.random.Generator
    ) -> PipelineEnv:
        """Preview a one-time readout at a single parameter set.

        The optimizer evaluates its recurring pipelines over a whole working set
        (a population, for population optimizers), which is what the default env
        supplies. A ``ONCE`` routine instead runs after optimisation at the
        trained parameters — one set — so binding the working set there would
        report circuits the run never submits.
        """
        # A Hamiltonian-seeded program declares no width on its seed, so fall back
        # to the program's own.
        n_rows = (
            1
            if preprocessor.cadence is PipelineCadence.ONCE
            else self.optimizer.n_param_sets
        )
        n_params = self._routine_parameter_count(preprocessor) or self.n_params
        return self._build_pipeline_env(
            rng=rng,
            progress_emitter=None,
            param_sets=np.zeros((n_rows, n_params)),
        )

    def _preprocessors(self) -> tuple[CircuitPreprocessor, ...]:
        """The cost routine plus whatever the optimizer drives alongside it.

        ``SolutionSamplingMixin`` adds the sample routine cooperatively, and a
        metric optimizer adds the routines it drives, so this is every routine a
        run submits. An optimizer that cannot drive this program raises
        :class:`~divi.pipeline.ContractViolation` from here — the same refusal
        ``run()`` makes, reached without submitting anything.
        """
        return (
            *super()._preprocessors(),
            self.cost_preprocessor(),
            *self.optimizer.preprocessors(self),
        )

    def cost_preprocessor(self) -> CircuitPreprocessor:
        """The preprocessor driving optimisation: expectation of the cost observable.

        Pass it to :meth:`~divi.qprog.QuantumProgram.evaluate` to measure the
        cost at chosen parameters, e.g.
        ``program.evaluate(params, program.cost_preprocessor())``.
        """
        return _default_cost_preprocessor()

    def _initial_spec(self) -> Any:
        """The cost ansatz seed. QAOA overrides this with its cost Hamiltonian."""
        return self.cost_circuit

    def _post_spec_batch(self):
        """The cohort of seed circuits emitted by the cost pipeline's spec stage.

        Runs the (memoized) cost pipeline's spec stage, reusing its cached
        forward-pass output so a metric estimator measures on the same sampled
        batch the cost evaluation used this iteration (deterministic
        per-evaluation seeding reproduces it on a cache miss).
        """
        pipeline = self._build_preprocessor_pipeline(self.cost_preprocessor())
        return pipeline.run_spec_stage(
            self._initial_spec(), self._build_pipeline_env()
        ).batch

    # ------------------------------------------------------------------ #
    # Execution
    # ------------------------------------------------------------------ #

    def _evaluate_cost_param_sets(
        self,
        param_sets: npt.NDArray[np.float64],
        *,
        shots: int | None = None,
        collect_variance: bool = False,
        **kwargs,
    ) -> dict[int, float]:
        """Evaluate the cost pipeline for the provided parameter sets.

        ``shots`` overrides the per-evaluation measurement budget without
        mutating the immutable backend (see :attr:`PipelineEnv.effective_shots`).
        ``collect_variance`` asks the pipeline to also estimate the shot-noise
        variance of each cost value, stashed on ``_last_cost_variance`` and read
        back via :meth:`_cost_shot_variances`.

        Subclasses should prefer overriding the initial-spec hook over
        replacing the full evaluator.
        """
        out = self.evaluate(
            np.atleast_2d(param_sets),
            self.cost_preprocessor(),
            shots=shots,
            return_variance=collect_variance,
        )
        if collect_variance:
            result, _ = cast("tuple[dict[int, Any], dict[int, float]]", out)
        else:
            result = cast("dict[int, Any]", out)

        constant = 0.0 if self._loss_constant_consumed else self.loss_constant
        return {idx: float(value[0]) + constant for idx, value in result.items()}

    def _parameter_frequencies(self) -> Sequence[tuple[float, int]] | None:
        """Per-parameter ``(omega, order)`` for the shift rule, or ``None``.

        ``None`` takes the two-term rule for every parameter. Subclasses that own
        an ansatz delegate to it.
        """
        return None

    @cached_property
    def _grad_shift_rule(
        self,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """The ``(shifts, weights)`` this program differentiates with.

        Built on first use so a gradient-free run never pays for it, and so a
        parameterisation with no tractable exact rule only raises when a gradient
        is actually requested.
        """
        frequencies = self._parameter_frequencies()
        if frequencies is None:
            frequencies = [(1.0, 1)] * self.n_params
        if len(frequencies) != self.n_params:
            raise ValueError(
                f"{type(self).__name__} declared {len(frequencies)} parameter "
                f"frequencies but has {self.n_params} parameters."
            )
        return _compute_parameter_shift_rule(frequencies)

    def _evaluate_gradient_at(
        self, params: npt.NDArray[np.float64], **kwargs
    ) -> npt.NDArray[np.float64]:
        """Evaluate the parameter-shift gradient at a single parameter vector."""
        shifts, weights = self._grad_shift_rule
        exp_vals = self._evaluate_cost_param_sets(shifts + params, **kwargs)
        return weights @ np.asarray(list(exp_vals.values()), dtype=np.float64)

    def _resolve_sample_params(
        self, params: npt.NDArray[np.float64] | None
    ) -> npt.NDArray[np.float64]:
        """Resolve and validate parameters for :meth:`~SolutionSamplingMixin.sample_solution`.

        Supplies the variational parameter-model knowledge that
        :class:`~divi.qprog._solution_sampling_mixin.SolutionSamplingMixin` is
        agnostic to: ``None`` falls back to ``_best_params`` (raising if
        optimisation has not run), and explicit params are validated against
        ``n_layers * n_params_per_layer``.
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
        expected = self.n_params
        if params_arr.shape[-1] != expected:
            raise ValueError(
                f"params last-axis size ({params_arr.shape[-1]}) does not "
                f"match n_layers * n_params_per_layer ({expected})."
            )
        return params_arr

    @_with_optimisation_progress
    def run(
        self,
        initial_params: npt.NDArray[np.float64] | None = None,
        perform_final_computation: bool = True,
        checkpoint_config: CheckpointConfig | None = None,
        **kwargs,
    ) -> Self:
        """Run the variational quantum algorithm.

        The outputs are stored in the algorithm object and can be accessed via
        properties such as ``total_circuit_count``, ``total_run_time``,
        ``losses_history``, and ``best_params``.

        Args:
            initial_params (npt.NDArray[np.float64] | None): Optional initial parameter
                sets for a fresh optimisation run. Must have shape
                ``(n_param_sets, n_layers * n_params_per_layer)``. Cannot be
                combined with a checkpoint-resumed optimizer state.
            perform_final_computation (bool): Whether to perform final computation after optimisation completes.
                Typically, this step involves sampling with the best found parameters to extract
                solution probability distributions. Set this to False in warm-starting or pre-training
                routines where the final sampling step is not needed. Defaults to True.
            checkpoint_config (CheckpointConfig | None): Checkpoint configuration.
                If None, no checkpointing is performed.
            **kwargs: Additional keyword arguments for subclasses.

        Returns:
            VariationalQuantumAlgorithm: Returns ``self`` for method chaining.
        """
        # Initialise checkpointing
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

        reject_unclaimed_run_kwargs(self, kwargs)

        # ``max_iterations`` is a total, not a per-call count: a resumed run (from a
        # checkpoint, or after raising the limit) spends only what is left, and a
        # program already at the limit spends nothing. Optimizers receive the
        # remaining count, so none of them needs to know about resumption.
        iterations_remaining = self.max_iterations - self.current_iteration
        if iterations_remaining <= 0:
            warn(
                f"This program has already run {self.current_iteration} of "
                f"max_iterations={self.max_iterations} iterations, so run() has "
                "nothing left to do. Raise max_iterations to continue.",
                UserWarning,
            )
            return self

        def cost_fn(params, *, shots=None, return_variance=False):
            self._evaluation_counter += 1
            self._progress_emitter(
                ProgressEvent.show(
                    self._progress_key,
                    f"Iteration #{self.current_iteration + 1}: Optimising",
                )
            )

            values_map = self._evaluate_cost_param_sets(
                np.atleast_2d(params),
                shots=shots,
                collect_variance=return_variance,
                **kwargs,
            )
            losses = np.asarray(list(values_map.values()), dtype=np.float64)
            losses = losses if params.ndim > 1 else losses.item()
            if not return_variance:
                return losses

            var_map = self._cost_shot_variances(values_map)
            variances = np.asarray(list(var_map.values()), dtype=np.float64)
            variances = variances if params.ndim > 1 else variances.item()
            return losses, variances

        # Advertise the shot-variance channel so variance-aware optimizers (e.g.
        # QUIVER) can use it without sniffing this closure's signature.
        setattr(cost_fn, "supports_variance", True)

        # Let the optimizer contribute any extra evaluators it needs (e.g. a
        # metric-based optimizer binds its metric estimator to this program and
        # returns a fused gradient + ``metric_fn``).
        extra_evaluators = self.optimizer.build_evaluators(self)
        jac_fn = extra_evaluators.get("jac")

        last_grad_norm: float | None = None
        last_checkpointed_iteration: int | None = None

        def _flush_final_checkpoint(*, force: bool = False):
            """Checkpoint the last iteration if the interval did not already."""
            if (
                checkpoint_config.checkpoint_dir is not None
                and self.current_iteration > 0
                and (force or self.current_iteration != last_checkpointed_iteration)
            ):
                self.save_state(checkpoint_config)

        def grad_fn(params):
            nonlocal last_grad_norm

            self._progress_emitter(
                ProgressEvent.show(
                    self._progress_key,
                    f"Iteration #{self.current_iteration + 1}: Computing gradients",
                )
            )

            if jac_fn is not None:
                grads = jac_fn(params)
            else:
                grads = self._evaluate_gradient_at(params, **kwargs)

            last_grad_norm = float(np.linalg.norm(grads))

            return grads

        def _iteration_counter(intermediate_result: OptimizeResult):
            nonlocal last_checkpointed_iteration

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

            fun = np.asarray(intermediate_result.fun, dtype=np.float64).ravel()
            best_idx = _argmin_finite(fun)
            if best_idx is None:
                current_loss = float("nan")
            else:
                current_loss = float(fun[best_idx])
                if current_loss < self._best_loss:
                    self._best_loss = current_loss
                    self._best_params = intermediate_result.x[best_idx].copy()

            self.current_iteration += 1

            self._progress_emitter(
                ProgressEvent.advance(
                    self._progress_key,
                    loss=float(current_loss),
                )
            )

            # Checkpointing
            if checkpoint_config._should_checkpoint(self.current_iteration):
                self.save_state(checkpoint_config)
                last_checkpointed_iteration = self.current_iteration

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
                    self._progress_emitter(
                        ProgressEvent.show(
                            self._progress_key,
                            f"Early stopping triggered: {reason.value}",
                        )
                    )
                    raise StopIteration

            # The scipy implementation of COBYLA interprets the `maxiter` option
            # as the maximum number of function evaluations, not iterations.
            # To provide a consistent user experience, we disable `scipy`'s
            # `maxiter` and manually stop the optimisation from the callback
            # when the desired number of iterations is reached.
            # Counted against the program's own total rather than scipy's per-call
            # ``nit``, so a resumed run stops at the limit instead of restarting it.
            if (
                isinstance(self.optimizer, ScipyOptimizer)
                and self.optimizer.method == ScipyMethod.COBYLA
                and self.current_iteration >= self.max_iterations
            ):
                raise StopIteration

        self._progress_emitter(ProgressEvent.show(self._progress_key, "Finished Setup"))

        resolved_initial_params = self._resolve_initial_param_sets(initial_params)

        optimize_kwargs: dict[str, Any] = dict(
            cost_fn=cost_fn,
            initial_params=resolved_initial_params,
            callback_fn=_iteration_counter,
            jac=grad_fn,
            max_iterations=iterations_remaining,
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
                self._progress_emitter(
                    ProgressEvent.finish(
                        self._progress_key,
                        TerminalStatus.CANCELLED,
                        detail=message,
                    )
                )
                try:
                    _flush_final_checkpoint()
                except Exception:
                    logger.warning(
                        "Failed to write a final checkpoint after cancellation.",
                        exc_info=True,
                    )
                raise ExecutionCancelledError(message) from exc
            else:
                self.optimize_result.success = True
                self.optimize_result.message = "Optimisation converged."

                # Set _best_params from final result (source of truth); a
                # non-finite loss never wins (falls back to the iteration-tracked
                # best when every final loss is non-finite).
                x = np.atleast_2d(self.optimize_result.x)
                best_idx = _argmin_finite(self.optimize_result.fun)
                if best_idx is not None:
                    self._best_params = x[best_idx].copy()

        # Canonical 1-D best parameters (the optimizer result contract); the
        # early-stop/cancel branches above carry a 2-D (1, n) best, so squeeze.
        self._final_params = np.atleast_1d(np.asarray(self.optimize_result.x).squeeze())

        if perform_final_computation and isinstance(self, SolutionSamplingMixin):
            self.sample_solution(**kwargs)
            _flush_final_checkpoint(force=True)
        else:
            _flush_final_checkpoint()

        self._progress_emitter(
            ProgressEvent.finish(
                self._progress_key,
                TerminalStatus.SUCCESS,
                detail="Finished successfully!",
            )
        )

        return self
