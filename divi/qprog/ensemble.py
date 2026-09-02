# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import atexit
import logging
import os
from abc import ABC, abstractmethod
from collections.abc import Callable, Container, Hashable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from dataclasses import asdict, dataclass, replace
from enum import Enum
from pathlib import Path
from threading import Event
from typing import Any, Self, cast
from warnings import warn

import numpy as np
import numpy.typing as npt

from divi.backends import CircuitRunner
from divi.exceptions import ExecutionCancelledError
from divi.pipeline import EnsembleReports
from divi.qprog._batch_coordinator import (
    BatchConfig,
    BatchMode,
    _BatchCoordinator,
    _ProxyBackend,
)
from divi.qprog._ensemble_checkpoint import (
    PROGRAM_COMPLETION_FILE,
    ROUND_COMPLETION_FILE,
    ROUND_START_FILE,
    ProgramRoundRecord,
    RoundCheckpoint,
    _encode_program_id,
    _program_checkpoint_path,
    _resolve_ensemble_checkpoint,
    _round_dir,
)
from divi.qprog._program_checkpoint import ProgramCheckpoint
from divi.qprog._solution_sampling_mixin import SolutionSamplingMixin
from divi.qprog.checkpointing import (
    CheckpointConfig,
    CheckpointNotFoundError,
    _atomic_write,
    _ensure_checkpoint_dir,
)
from divi.qprog.quantum_program import QuantumProgram
from divi.qprog.variational_quantum_algorithm import VariationalQuantumAlgorithm
from divi.reporting._events import (
    ProgressEmitter,
    ProgressEvent,
    ProgressScope,
    TerminalStatus,
    discard_progress_event,
)
from divi.reporting._logging import log_progress_event
from divi.reporting._rich import render_failure
from divi.reporting._session import ProgressSession, _environment_disables_progress
from divi.reporting._state import ProgressState

__all__ = [
    "BatchConfig",
    "BatchMode",
    "ProgramEnsemble",
    "ReportingLevel",
    "RoundRecord",
    "WorkflowStatus",
]

logger = logging.getLogger(__name__)


def _qualified_type(value: Any) -> str:
    cls = value if isinstance(value, type) else type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


@dataclass
class _RoundCheckpointSession:
    checkpoint_path_by_program: dict[QuantumProgram, Path]
    iterative_config_by_program: dict[QuantumProgram, CheckpointConfig]
    restored_programs: set[QuantumProgram]
    recovered_circuit_count: int = 0
    recovered_run_time: float = 0.0

    @classmethod
    def inactive(cls) -> Self:
        """Session for a round that is not being checkpointed."""
        return cls({}, {}, set())

    @classmethod
    def prepare(
        cls,
        *,
        checkpoint_config: CheckpointConfig,
        round_path: Path,
        ensemble_type: str,
        round_index: int,
        ensemble_state: dict[str, Any],
        programs: list[QuantumProgram],
        child_recovery_states: list[ProgramRoundRecord],
        interrupted_checkpoint: RoundCheckpoint | None,
    ) -> Self:
        round_path.mkdir(parents=True, exist_ok=True)
        restoring_children = interrupted_checkpoint is not None
        if interrupted_checkpoint is None:
            interrupted_checkpoint = RoundCheckpoint(
                kind="round_start",
                ensemble_type=ensemble_type,
                round_index=round_index,
                ensemble_state=ensemble_state,
                programs=child_recovery_states,
            )
            _atomic_write(
                round_path / ROUND_START_FILE,
                interrupted_checkpoint.model_dump_json(indent=2, exclude_none=True),
            )
        else:
            cls._validate_structure(interrupted_checkpoint, child_recovery_states)

        saved_recovery_states = interrupted_checkpoint.round_start_data()
        recovery_state_by_program = dict(zip(programs, saved_recovery_states))
        checkpoint_path_by_program = {
            program: _program_checkpoint_path(round_path, slot)
            for slot, program in enumerate(programs)
        }
        for path in checkpoint_path_by_program.values():
            path.mkdir(parents=True, exist_ok=True)

        session = cls(
            checkpoint_path_by_program=checkpoint_path_by_program,
            iterative_config_by_program={
                program: replace(
                    checkpoint_config,
                    checkpoint_dir=checkpoint_path_by_program[program],
                )
                for program in programs
                if cls._supports_iterative(program)
            },
            restored_programs=set(),
        )
        if restoring_children:
            session._recover(programs, recovery_state_by_program)
        return session

    @staticmethod
    def _validate_structure(
        persisted_checkpoint: RoundCheckpoint,
        child_recovery_states: list[ProgramRoundRecord],
    ) -> None:
        def key(entry: ProgramRoundRecord) -> tuple[Any, ...]:
            return entry.program_id, entry.program_type

        persisted_states = persisted_checkpoint.round_start_data()
        if len(persisted_states) != len(child_recovery_states) or any(
            key(saved) != key(fresh)
            for saved, fresh in zip(persisted_states, child_recovery_states)
        ):
            raise ValueError(
                "Child recovery states do not match reconstructed programs."
            )

    @staticmethod
    def _supports_iterative(program: QuantumProgram) -> bool:
        return (
            isinstance(program, VariationalQuantumAlgorithm)
            and program.optimizer.supports_checkpointing
            and program._early_stopping is None
        )

    def _recover(
        self,
        programs: list[QuantumProgram],
        recovery_state_by_program: dict[QuantumProgram, ProgramRoundRecord],
    ) -> None:
        circuits = 0
        runtime = 0.0
        for slot, program in enumerate(programs):
            recovery_state = recovery_state_by_program[program]
            path = self.checkpoint_path_by_program[program]

            restored = False
            marker_path = path / PROGRAM_COMPLETION_FILE
            if marker_path.is_file():
                try:
                    checkpoint_json = marker_path.read_text()
                    metadata = ProgramCheckpoint.model_validate_json(
                        checkpoint_json, extra="ignore"
                    )
                    if metadata.program_type != type(program).__name__:
                        raise ValueError("Completed child checkpoint type changed.")
                    recovered_circuits, recovered_runtime = self._recovered_accounting(
                        recovery_state, metadata
                    )
                    if not program._restore_checkpoint(checkpoint_json, path):
                        raise ValueError("Child does not support checkpoint restore.")
                except Exception:
                    logger.warning(
                        "Could not restore completed ensemble child in slot %d; "
                        "trying iterative recovery.",
                        slot,
                        exc_info=True,
                    )
                else:
                    self.restored_programs.add(program)
                    circuits += recovered_circuits
                    runtime += recovered_runtime
                    restored = True

            if restored or program not in self.iterative_config_by_program:
                continue
            try:
                vqa = cast(VariationalQuantumAlgorithm, program)
                checkpoint_path, checkpoint = type(vqa)._load_checkpoint_state(path)
                recovered_circuits, recovered_runtime = self._recovered_accounting(
                    recovery_state, checkpoint
                )
                vqa._restore_loaded_checkpoint(checkpoint_path, checkpoint)
            except Exception:
                logger.warning(
                    "Could not restore iterative ensemble child in slot %d; "
                    "restarting it.",
                    slot,
                    exc_info=True,
                )
            else:
                circuits += recovered_circuits
                runtime += recovered_runtime

        self.recovered_circuit_count = circuits
        self.recovered_run_time = runtime

    @staticmethod
    def _recovered_accounting(
        recovery_state: ProgramRoundRecord, checkpoint: ProgramCheckpoint
    ) -> tuple[int, float]:
        circuits = (
            checkpoint.total_circuit_count - recovery_state.circuit_count_at_round_start
        )
        runtime = checkpoint.total_run_time - recovery_state.run_time_at_round_start
        if circuits < 0 or runtime < 0:
            raise ValueError("Recovered child accounting is negative.")
        return circuits, runtime

    def child_config(self, program: QuantumProgram) -> CheckpointConfig | None:
        return self.iterative_config_by_program.get(program)

    def was_restored(self, program: QuantumProgram) -> bool:
        return program in self.restored_programs

    def commit_completed(self, program: QuantumProgram) -> None:
        path = self.checkpoint_path_by_program.get(program)
        if path is None:
            return
        checkpoint = program._make_checkpoint(path)
        if checkpoint is None:
            return
        _atomic_write(
            path / PROGRAM_COMPLETION_FILE,
            checkpoint.model_dump_json(indent=2),
        )

    def execute(
        self,
        program: QuantumProgram,
        operation: Callable[[QuantumProgram], Any],
        *,
        commit_completion: bool,
    ) -> Any:
        if self.was_restored(program):
            return program
        result = operation(program)
        if commit_completion:
            self.commit_completed(program)
        return result


class ReportingLevel(str, Enum):
    """Amount of live progress shown for an ensemble workflow."""

    OFF = "off"
    COMPACT = "compact"
    FULL = "full"


class WorkflowStatus(str, Enum):
    """How a workflow round, or a whole workflow run, ended.

    :attr:`RoundRecord.status` takes ``COMPLETE``, ``FAILED``, or
    ``CANCELLED``. :attr:`~divi.qprog.ensemble.ProgramEnsemble.stop_reason`
    takes those plus ``MAX_ROUNDS``, which describes the run rather than any
    single round.
    """

    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"
    MAX_ROUNDS = "max_rounds"


@dataclass(frozen=True)
class RoundRecord:
    """Immutable accounting summary for one ensemble round.

    ``circuit_count`` and ``run_time`` are per-round deltas, not cumulative
    totals. ``error`` carries the formatted exception for failed rounds and
    is ``None`` otherwise.
    """

    #: Round number, counting from 1.
    number: int
    program_count: int
    circuit_count: int
    run_time: float
    status: WorkflowStatus
    error: str | None = None


#: Largest ensemble size for which :meth:`ProgramEnsemble.run` will allocate
#: one executor thread per program under the default wait-for-all barrier.
_BARRIER_PROGRAM_LIMIT = 256

#: Above this many ``max_concurrent_programs``, ``run`` warns to flag a
#: likely misuse of the knob (e.g. the user wanted ``max_batch_size``).
_CONCURRENT_PROGRAMS_SOFT_CAP = 1024


def _resolve_worker_count(batch_config: BatchConfig, n_programs: int) -> int:
    """Size the executor pool for one dispatch.

    In barrier mode the pool must let the barrier predicate fill a batch.
    When ``max_batch_size`` is set the pool is capped at
    ``min(max_batch_size, n_programs)`` so the barrier and the batch size
    align (one full wave per flush), no more threads are spawned than needed
    (avoiding macOS's per-process thread cap on large ensembles), and threads
    recycle for the next wave after each flush.

    Raises:
        RuntimeError: If the default wait-for-all barrier would need more than
            :data:`_BARRIER_PROGRAM_LIMIT` threads.
    """
    default_workers = (os.cpu_count() or 1) + 4
    if batch_config.mode is not BatchMode.MERGED:
        return default_workers

    if batch_config.max_concurrent_programs is not None:
        if batch_config.max_concurrent_programs == -1:
            if batch_config.max_batch_size is not None:
                return min(batch_config.max_batch_size, n_programs)
            return n_programs
        n_workers = batch_config.max_concurrent_programs
        if n_workers > _CONCURRENT_PROGRAMS_SOFT_CAP:
            warn(
                f"max_concurrent_programs={n_workers} spawns that many "
                f"executor threads; if you meant to merge submissions, "
                f"set max_batch_size on BatchConfig instead.",
                UserWarning,
                stacklevel=3,
            )
        return n_workers

    if batch_config.max_batch_size is not None:
        return min(batch_config.max_batch_size, n_programs)
    if n_programs <= _BARRIER_PROGRAM_LIMIT:
        return max(n_programs, default_workers)
    raise RuntimeError(
        f"Ensemble has {n_programs} programs, exceeding the "
        f"wait-for-all barrier limit ({_BARRIER_PROGRAM_LIMIT}). Set "
        f"BatchConfig(max_batch_size=N) for early-flush, "
        f"BatchConfig(max_concurrent_programs=N) to bypass the cap, or "
        f"BatchConfig(mode=BatchMode.OFF)."
    )


def _validate_sampling_programs(
    programs: dict[Any, QuantumProgram], *, action: str
) -> None:
    """Require variational programs that expose solution sampling."""
    for prog_id, program in programs.items():
        if not isinstance(program, VariationalQuantumAlgorithm) or not isinstance(
            program, SolutionSamplingMixin
        ):
            raise TypeError(
                f"{action} requires variational solution-sampling sub-programs "
                f"(VariationalQuantumAlgorithm + SolutionSamplingMixin); program "
                f"{prog_id!r} is {type(program).__name__}."
            )


def _resolve_sampling_params(
    programs: dict[Any, QuantumProgram],
    params_per_program: dict[Any, npt.NDArray[np.float64]] | None,
    *,
    suppress_strict_warning: bool,
) -> dict[Any, npt.NDArray[np.float64]]:
    """Resolve one parameter array per program for :meth:`sample_solution`.

    Entries in ``params_per_program`` win; programs missing from it fall back
    to their own ``_best_params``.

    Raises:
        RuntimeError: If ``programs`` is empty, or a program has neither an
            explicit entry nor stored parameters.
        ValueError: If ``params_per_program`` names unknown programs, or a
            resolved array's last axis does not match the program's
            ``n_layers * n_params_per_layer``.
        TypeError: If any program is not a
            :class:`~divi.qprog.VariationalQuantumAlgorithm`.
    """
    if len(programs) == 0:
        raise RuntimeError("No programs to sample.")

    _validate_sampling_programs(programs, action="sample_solution")

    if params_per_program is not None:
        unknown = set(params_per_program) - set(programs)
        if unknown:
            raise ValueError(
                f"params_per_program contains keys not in this ensemble: "
                f"{list(unknown)!r}. Valid program IDs: "
                f"{list(programs.keys())!r}."
            )

    resolved: dict[Any, npt.NDArray[np.float64]] = {}
    fallbacks: list[Any] = []
    for prog_id, program in programs.items():
        if params_per_program is not None and prog_id in params_per_program:
            arr = np.asarray(params_per_program[prog_id], dtype=np.float64)
        else:
            arr = np.asarray(
                getattr(program, "_best_params", np.array([], dtype=np.float64)),
                dtype=np.float64,
            )
            if params_per_program is not None:
                fallbacks.append(prog_id)

        if arr.size == 0:
            raise RuntimeError(
                f"Program {prog_id!r}: no parameters available. "
                f"Pass params_per_program[{prog_id!r}]=... or call "
                f"run() on the ensemble first."
            )

        n_layers = getattr(program, "n_layers", None)
        n_params_per_layer = getattr(program, "n_params_per_layer", None)
        if n_layers is not None and n_params_per_layer is not None:
            expected = n_layers * n_params_per_layer
            if arr.shape[-1] != expected:
                raise ValueError(
                    f"Program {prog_id!r}: params last-axis size "
                    f"({arr.shape[-1]}) does not match "
                    f"n_layers * n_params_per_layer ({expected})."
                )

        resolved[prog_id] = arr

    if fallbacks and not suppress_strict_warning:
        warn(
            f"params_per_program is missing keys for programs "
            f"{list(fallbacks)!r}; falling back to each program's "
            f"_best_params. Pass suppress_strict_warning=True to silence.",
            UserWarning,
            stacklevel=3,
        )

    return resolved


class ProgramEnsemble(ABC):
    """This abstract class provides the basic scaffolding for higher-order
    computations that require more than one quantum program to achieve its goal.

    :meth:`run` executes the ensemble as a loop of *rounds*. Each round
    materialises a fresh program map, runs those programs in parallel, and
    folds their results into a workflow state the next round can use. The loop
    ends when :meth:`is_complete` returns ``True`` or a ``max_rounds`` limit is
    hit.

    Subclasses must implement:
        1. `create_programs(state)`: Generates the independent programs needed
            for the coming round, keyed by program identifier. It receives the
            current workflow state, so an adaptive ensemble can choose this
            round's programs from what the last round measured.

        2. `aggregate_results`: Aggregates the results of the programs after
            they are done executing. This function should be aware of the
            different formats the programs might have (counts dictionary,
            expectation value, etc) and handle such cases accordingly. Only the
            final round's programs remain in ``self.programs``.

    Three further hooks are optional and default to single-round behaviour:
    :meth:`initial_state`, :meth:`update_state`, and :meth:`is_complete`.
    """

    def __init__(
        self,
        backend: CircuitRunner,
        *,
        sampling_backend: CircuitRunner | None = None,
        reporting_level: ReportingLevel = ReportingLevel.COMPACT,
    ):
        """Initialise the ensemble.

        Args:
            backend: Backend used to execute every sub-program's circuits.
            sampling_backend: Backend used for the ensemble's final sampling
                phase. ``None`` reuses ``backend``.
            reporting_level: How much live progress to render. See
                :class:`~divi.qprog.ReportingLevel`; defaults to ``COMPACT``.
                The ``DIVI_DISABLE_PROGRESS`` environment variable suppresses
                all visual output regardless of this setting.
        """
        super().__init__()

        self.backend = backend
        self._sampling_backend = sampling_backend
        self._executor = None
        self._programs = {}
        self._coordinator: _BatchCoordinator | None = None
        # Real backend per program, saved before batching swaps in a proxy.
        self._program_original_backend: dict[QuantumProgram, CircuitRunner] = {}
        # Per-program counter values captured at the start of each dispatch.
        self._dispatch_count_baseline: dict[QuantumProgram, tuple[int, float]] = {}
        self.futures: list[Future] = []

        self._total_circuit_count = 0
        self._total_run_time = 0.0
        # Normalize so a plain string works wherever the enum does.
        self.reporting_level = ReportingLevel(reporting_level)
        self._workflow_state: Any = None
        self._stop_reason: WorkflowStatus | None = None
        self._round_history: list[RoundRecord] = []
        self._round_index = 0
        # ``(round_number, max_rounds)`` while a workflow round is dispatching;
        # ``None`` for a standalone ``run_one_round``.
        self._round_context: tuple[int, int | None] | None = None
        # True between ``create_programs()`` and the dispatch that consumes the
        # resulting map. Lets ``run()`` distinguish a caller-materialised first
        # round from the previous round's spent programs.
        self._programs_pending = False
        # Set by join() when a round is cut short by KeyboardInterrupt, so
        # run() can stop the workflow instead of starting another round.
        self._round_cancelled = False
        self._resumed_from_checkpoint = False
        self._interrupted_checkpoint: RoundCheckpoint | None = None
        self._restored_checkpoint_root: Path | None = None

        self._progress_session: ProgressSession | None = None
        self._progress_bindings: ExitStack | None = None
        self._progress_emitter: ProgressEmitter = (
            discard_progress_event
            if self.reporting_level is ReportingLevel.OFF
            or _environment_disables_progress()
            else log_progress_event
        )
        self._preparation_registered = False
        self._workflow_registered = False
        self._workflow_message: str | None = None
        self._cancellation_event = Event()
        self._future_to_program: dict[Future, QuantumProgram] = {}

    @property
    def sampling_backend(self) -> CircuitRunner | None:
        """Backend dedicated to final solution sampling, when configured."""
        return self._sampling_backend

    @property
    def total_circuit_count(self):
        """
        Get the total number of circuits executed across all programs in the ensemble.

        Returns:
            int: Cumulative count of circuits submitted by all programs.
        """
        return self._total_circuit_count

    @property
    def total_run_time(self):
        """
        Get the total runtime across all programs in the ensemble.

        Returns:
            float: Cumulative execution time in seconds across all programs.
        """
        return self._total_run_time

    @property
    def round_history(self) -> tuple[RoundRecord, ...]:
        """Completed workflow rounds, in execution order."""
        return tuple(self._round_history)

    @property
    def workflow_state(self) -> Any:
        """Latest state produced by :meth:`update_state`, or whatever
        :meth:`initial_state` returned if no round has completed yet. Reset at
        the start of every :meth:`run`.
        """
        return self._workflow_state

    @property
    def stop_reason(self) -> WorkflowStatus | None:
        """Why the most recent :meth:`run` stopped, or ``None`` before the
        first run. See :class:`WorkflowStatus`.
        """
        return self._stop_reason

    @property
    def programs(self) -> dict:
        """
        Get a copy of the programs dictionary.

        Returns:
            dict: Copy of the programs dictionary mapping program IDs to
                QuantumProgram instances. Modifications to this dict will not
                affect the internal state.
        """
        return self._programs.copy()

    @programs.setter
    def programs(self, value: dict):
        """Set the programs dictionary."""
        self._programs = value

    def dry_run(self, *, force_circuit_generation: bool = False) -> EnsembleReports:
        """Preview every sub-program's circuit fan-out without executing anything.

        Calls :meth:`~divi.qprog.QuantumProgram.dry_run` on each program and
        keys the results by program identifier. Pass the returned dict to
        :func:`~divi.pipeline.format_dry_run` for the tree output.

        Args:
            force_circuit_generation: Forwarded to each sub-program's
                :meth:`~divi.qprog.QuantumProgram.dry_run`. If ``True``, every
                stage runs its full ``expand`` path so the trace contains real
                DAGs and QASM strings. Defaults to ``False``.

        Raises:
            RuntimeError: If no programs exist, or a sub-program's dry-run fails.

        Example:
            >>> from divi.pipeline import format_dry_run
            >>> ensemble.create_programs()
            >>> reports = ensemble.dry_run()
            >>> format_dry_run(reports, style="grouped")  # pretty-print to stdout
        """
        if len(self._programs) == 0:
            raise RuntimeError("No programs to dry-run. Call create_programs() first.")
        reports: EnsembleReports = {}
        for program_id, program in self._programs.items():
            try:
                reports[program_id] = program.dry_run(
                    force_circuit_generation=force_circuit_generation
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Dry run failed for program {program_id!r}: {exc}"
                ) from exc
        return reports

    def initial_state(self) -> Any:
        """Return the state supplied to the first workflow round."""
        return None

    def _save_workflow_checkpoint_state(
        self, state: Any, round_dir: Path, stem: str
    ) -> dict[str, Any]:
        """Serialize mutable workflow state for an ensemble checkpoint."""
        if state is not None:
            raise NotImplementedError(
                f"{type(self).__name__} must implement workflow-state checkpointing."
            )
        return {}

    def _load_workflow_checkpoint_state(
        self, payload: dict[str, Any], round_dir: Path, stem: str
    ) -> Any:
        """Deserialize mutable workflow state from an ensemble checkpoint."""
        if payload:
            raise NotImplementedError(
                f"{type(self).__name__} must implement workflow-state checkpointing."
            )
        return None

    def update_state(self, state: Any) -> Any:
        """Reduce the finished round's results into the next round's state.

        Read the results off ``self.programs``, which still holds the round
        that just completed. Anything later rounds need must be folded into
        the returned state — the program map is replaced each round.
        """
        return state

    def is_complete(self, state: Any) -> bool:
        """Return whether the workflow should stop before running another round.

        Checked at the top of every iteration, including before the first
        round — a ``state`` that already satisfies this runs zero rounds. The
        default stops after one round, which is what the one-shot built-in
        ensembles need. Overrides that need a round count should read
        ``len(self.round_history)``, which is reset on every new
        :meth:`run`.
        """
        return self._round_index >= 1

    @abstractmethod
    def create_programs(self, state: Any = None):
        """Generate and populate the programs dictionary for ensemble execution.

        This method must be implemented by subclasses to create the quantum programs
        that will be executed as part of the ensemble. The method operates via side effects:
        it populates `self._programs` (or `self.programs`) with a dictionary mapping
        program identifiers to `QuantumProgram` instances.

        Implementation Notes:
            - Subclasses should call `super().create_programs()` first to
              validate that no programs already exist.
            - An override must accept the state argument; `run()` passes the
              current workflow state positionally on every round.
            - After calling super(), subclasses should populate `self.programs` or
              `self._programs` with their program instances.
            - Program identifiers can be any hashable type (e.g., strings, tuples).
              Common patterns include strings like "program_1", "program_2" or tuples like
              ('A', 5) for partitioned problems.

        Side Effects:
            - Populates `self._programs` with program instances.

        Raises:
            RuntimeError: If programs already exist (should call `reset()` first).

        Example:
            >>> def create_programs(self, state=None):
            ...     super().create_programs()
            ...     self.programs = {
            ...         "prog1": QAOA(...),
            ...         "prog2": QAOA(...),
            ...     }
        """
        if self._executor is not None:
            raise RuntimeError(
                "Cannot create programs while an ensemble round is running."
            )
        if self._programs:
            raise RuntimeError(
                "Some programs already exist. Complete the pending round with run() "
                "or call reset() before creating a new program map."
            )
        self._programs_pending = True

    def _clear_completed_round(self) -> None:
        """Discard program instances only after their round has completed."""
        if self._executor is not None:
            raise RuntimeError(
                "Cannot replace programs while an ensemble round is running."
            )
        self._programs.clear()
        self._programs_pending = False

    def _reset_workflow_state(self) -> None:
        """Clear per-workflow state without touching programs or lifetime totals.

        Called at the start of every :meth:`run` so a repeated workflow starts
        from round zero. Cumulative circuit and runtime counters survive, as
        does any program map the caller materialised itself.
        """
        self._workflow_state = None
        self._stop_reason = None
        self._round_history.clear()
        self._round_index = 0
        self._round_context = None
        self._resumed_from_checkpoint = False
        self._interrupted_checkpoint = None
        self._restored_checkpoint_root = None

    def reset(self):
        """
        Reset the ensemble to its initial state.

        Clears all programs, stops any running executors, closes the queued
        progress session, and restores program emitters. This allows the
        ensemble to be reused for a new set of programs. Also clears
        :attr:`workflow_state`, :attr:`stop_reason` and
        :attr:`round_history`; the lifetime :attr:`total_circuit_count` and
        :attr:`total_run_time` counters survive.

        Note:
            Any running programs will be forcefully stopped. Results from incomplete
            programs will be lost.
        """
        if self._progress_bindings is not None:
            for program in self._programs.values():
                self._emit_progress_message(
                    program._progress_key,
                    final_status=TerminalStatus.CANCELLED,
                    message="Reset by caller",
                )
            self._finish_workflow_progress(TerminalStatus.CANCELLED)

        # Stop workers before restoring their temporary backends and emitter
        # bindings. A worker may own a nested progress context, so allowing it
        # to outlive teardown can restore an emitter for an already-closed
        # session.
        self._cancellation_event.set()
        if self._coordinator is not None:
            self._coordinator.shutdown()
            self._coordinator = None

        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None
            self.futures.clear()

        self._restore_program_backends()
        self._teardown_progress_session()

        self._programs.clear()
        self._programs_pending = False
        self._reset_workflow_state()

    def _restore_program_backends(self) -> None:
        """Undo the backend swap done for a dispatch with a backend override.

        Batched dispatch replaces each sub-program's ``backend`` with a
        ``_ProxyBackend`` bound to the coordinator; an un-batched dispatch
        with a ``backend`` override swaps in that backend directly. Either
        way, restore the original so each program is usable directly or in a
        later dispatch. Idempotent: the snapshot is cleared after restoring.
        """
        for program, backend in self._program_original_backend.items():
            program.backend = backend
        self._program_original_backend.clear()

    def _atexit_cleanup_hook(self):
        # This hook is only registered for non-blocking runs.
        if self._executor is not None:
            warn(
                "A non-blocking ProgramEnsemble run was not explicitly closed with "
                "'join()'. The batch was cleaned up automatically on exit.",
                UserWarning,
            )
            self.reset()

    def _add_program_to_executor(
        self,
        program: QuantumProgram,
        task_fn: Callable[..., Any],
    ) -> Future:
        """
        Add a quantum program to the thread pool executor for execution.

        Sets up the program with cancellation support and progress tracking, then
        submits it for execution in a separate thread.  The program is
        automatically deregistered from the batch coordinator when it finishes.

        Args:
            program (QuantumProgram): The quantum program to execute.
            task_fn: The callable that consumes the program and produces the
                per-program result. Provided by the caller (e.g. ``run()`` or
                ``sample_solution()``).

        Returns:
            Future: A Future object representing the program's execution.
        """
        if hasattr(program, "_set_cancellation_event"):
            program._set_cancellation_event(self._cancellation_event)

        coordinator = self._coordinator
        program_key = program._progress_key

        def _coordinated_task(prog):
            try:
                return task_fn(prog)
            finally:
                if coordinator is not None:
                    coordinator.deregister_program(program_key)

        if self._executor is None:
            raise RuntimeError(
                "Cannot submit program: executor is not initialised. Call run() first."
            )
        return self._executor.submit(_coordinated_task, program)

    def run_one_round(
        self,
        blocking: bool = False,
        *,
        batch_config: BatchConfig = BatchConfig(),
        _checkpoint_session: _RoundCheckpointSession | None = None,
    ):
        """
        Execute the currently materialised programs once.

        Prefer :meth:`run`, which materialises programs and drives rounds for
        you. Reach for this only to dispatch without blocking, or to drive
        rounds yourself.

        Starts all quantum programs in parallel using a thread pool. Can run in
        blocking or non-blocking mode.

        Args:
            blocking (bool, optional): If True, waits for all programs to complete
                before returning. If False, returns immediately and programs run in
                the background. Defaults to False.
            batch_config (BatchConfig): Configuration for circuit batching.
                Controls whether submissions are merged and how.  Defaults to
                ``BatchConfig()`` which merges all submissions via a wait-for-all
                barrier.  Use ``BatchConfig(mode=BatchMode.OFF)`` to let each
                program submit independently, or
                ``BatchConfig(max_batch_size=50)`` to cap the number of circuits
                per merged backend call.

                The default barrier requires one executor thread per program,
                so it is capped at ``256`` programs.  Larger ensembles must
                opt into ``max_batch_size`` (bounded pool, smaller merges),
                ``max_concurrent_programs`` (explicit pool size, ideal for
                cloud submission of one large merged job), or
                ``BatchMode.OFF``.

        Returns:
            ProgramEnsemble: Returns self for method chaining.

        Raises:
            RuntimeError: If an ensemble is already running, if no programs
                have been created, or if the ensemble exceeds 256 programs
                without an explicit batching strategy.
            ValueError: If ``blocking=False`` while ``sampling_backend`` is
                configured — the sampling phase needs training to finish first
                and so cannot be split across a non-blocking dispatch.

        Note:
            In non-blocking mode, call `join()` later to wait for completion and
            collect results.
        """
        if self._sampling_backend is not None and not blocking:
            raise ValueError(
                "run_one_round(blocking=False) can't honour sampling_backend: "
                "sampling needs training to finish first. Pass blocking=True, "
                "or unset sampling_backend."
            )
        dispatched = self._dispatch_round(
            blocking=blocking,
            batch_config=batch_config,
            checkpoint_session=_checkpoint_session,
        )
        # Submitted — a later run() must not treat this map as pending.
        self._programs_pending = False
        return dispatched

    def _dispatch_round(
        self,
        *,
        blocking: bool,
        batch_config: BatchConfig,
        checkpoint_session: _RoundCheckpointSession | None = None,
    ) -> Self:
        """Dispatch the currently materialised programs for one round.

        Without ``sampling_backend`` this is a plain dispatch. With it,
        training runs to completion first with each program's final sampling
        disabled, then a separate sampling-only dispatch runs on
        ``sampling_backend`` — so the first dispatch is always blocking
        regardless of ``blocking``.
        """
        session = checkpoint_session or _RoundCheckpointSession.inactive()

        if self._sampling_backend is None:

            def task_fn(program: QuantumProgram):
                child_config = session.child_config(program)

                def operation(child: QuantumProgram):
                    if child_config is None:
                        return child.run()
                    return child.run(checkpoint_config=child_config)

                return session.execute(program, operation, commit_completion=True)

            return self._dispatch(
                task_fn=task_fn,
                blocking=blocking,
                batch_config=batch_config,
            )

        _validate_sampling_programs(self._programs, action="sampling_backend workflow")

        def _train_without_sampling(program: QuantumProgram):
            child_config = session.child_config(program)

            def operation(child: QuantumProgram):
                vqa = cast(VariationalQuantumAlgorithm, child)
                if child_config is None:
                    return vqa.run(perform_final_computation=False)
                return vqa.run(
                    perform_final_computation=False,
                    checkpoint_config=child_config,
                )

            return session.execute(program, operation, commit_completion=False)

        self._dispatch(
            task_fn=_train_without_sampling, blocking=True, batch_config=batch_config
        )
        if self._round_cancelled:
            return self
        resolved = _resolve_sampling_params(
            self._programs, None, suppress_strict_warning=False
        )
        return self._dispatch_sample_solution(
            resolved,
            backend=self._sampling_backend,
            blocking=blocking,
            batch_config=batch_config,
            restored_programs=session.restored_programs,
            on_sampled=session.commit_completed,
        )

    def run(
        self,
        *,
        max_rounds: int | None = None,
        batch_config: BatchConfig = BatchConfig(),
        checkpoint_config: CheckpointConfig = CheckpointConfig(),
    ) -> Self:
        """Run this ensemble's state-dependent workflow to completion.

        Each round materialises a fresh program map, runs it synchronously,
        then lets the subclass reduce its results into the next workflow state.
        ``run_one_round`` remains available for callers needing direct control
        of a previously materialised program map.

        A program map materialised by the caller before this call is used
        as-is for the first round; later rounds always come from
        :meth:`create_programs`.

        Args:
            max_rounds: Stop after this many rounds even if
                :meth:`is_complete` is still false. ``None`` runs until
                convergence.
            batch_config: Forwarded to each round's dispatch.
            checkpoint_config: Checkpoint directory and child VQA checkpoint
                interval. The ensemble writes workflow state at every round
                boundary.

        Returns:
            ProgramEnsemble: Returns self for method chaining. The terminal
            reason is in :attr:`stop_reason` and the latest state in
            :attr:`workflow_state`.

        Raises:
            RuntimeError: If a round's programs fail. :attr:`stop_reason` and
                :attr:`round_history` are populated before the exception
                propagates, so catch it and inspect them to see which round
                failed and why. An exception raised by :meth:`update_state` is
                recorded the same way, so a ``FAILED`` round may still have
                executed its circuits successfully.
        """
        if max_rounds is not None and max_rounds < 1:
            raise ValueError("max_rounds must be >= 1 when provided.")
        if self._executor is not None:
            raise RuntimeError("An ensemble is already being run.")

        interrupted_checkpoint = self._interrupted_checkpoint
        self._interrupted_checkpoint = None

        if self._resumed_from_checkpoint:
            if (
                checkpoint_config.checkpoint_dir is None
                and self._restored_checkpoint_root is not None
            ):
                checkpoint_config = replace(
                    checkpoint_config,
                    checkpoint_dir=self._restored_checkpoint_root,
                )
            state = self._workflow_state
            self._resumed_from_checkpoint = False
        else:
            if checkpoint_config.checkpoint_dir is not None:
                try:
                    _resolve_ensemble_checkpoint(checkpoint_config.checkpoint_dir)
                except CheckpointNotFoundError:
                    pass
                else:
                    raise RuntimeError(
                        "Checkpoint directory already contains an ensemble "
                        "checkpoint. Restore it or use a fresh directory."
                    )
            self._reset_workflow_state()
            state = self.initial_state()
            self._workflow_state = state
        use_caller_programs = self._programs_pending and bool(self._programs)

        while True:
            if self.is_complete(state):
                self._stop_reason = WorkflowStatus.COMPLETE
                break
            if max_rounds is not None and self._round_index >= max_rounds:
                self._stop_reason = WorkflowStatus.MAX_ROUNDS
                break

            # Claim the round number up front so every outcome records the
            # same index.
            self._round_index += 1
            circuits_before = self.total_circuit_count
            runtime_before = self.total_run_time
            self._round_context = (self._round_index, max_rounds)
            self._round_cancelled = False
            program_count = len(self._programs)
            round_input_snapshot = None
            try:
                if (
                    interrupted_checkpoint is None
                    and checkpoint_config.checkpoint_dir is not None
                ):
                    round_input_snapshot = self._save_round_input_state(
                        state, checkpoint_config
                    )
                if not use_caller_programs:
                    self._clear_completed_round()
                    self.create_programs(state)
                use_caller_programs = False
                program_count = len(self._programs)
                checkpoint_session = self._prepare_checkpoint_session(
                    state,
                    checkpoint_config,
                    interrupted_checkpoint,
                    round_input_snapshot,
                )
                interrupted_checkpoint = None
                self._total_circuit_count += checkpoint_session.recovered_circuit_count
                self._total_run_time += checkpoint_session.recovered_run_time
                self.run_one_round(
                    blocking=True,
                    batch_config=batch_config,
                    _checkpoint_session=checkpoint_session,
                )
                self._programs_pending = False
                if self._round_cancelled:
                    # Ctrl-C during the round: don't reduce partial results
                    # into the state, and don't start another round.
                    self._stop_reason = WorkflowStatus.CANCELLED
                    self._record_round(
                        program_count,
                        circuits_before,
                        runtime_before,
                        status=WorkflowStatus.CANCELLED,
                    )
                    break
                state = self.update_state(state)
                completed_record = self._round_record(
                    program_count,
                    circuits_before,
                    runtime_before,
                    status=WorkflowStatus.COMPLETE,
                )
                if checkpoint_config.checkpoint_dir is not None:
                    self._save_completed_round_checkpoint(
                        checkpoint_config,
                        state,
                        [*self._round_history, completed_record],
                    )
            except KeyboardInterrupt:
                # Ctrl-C outside the dispatch, most likely in update_state's
                # classical reduction. Leave the state at its last complete
                # round rather than reducing a partial one.
                self._stop_reason = WorkflowStatus.CANCELLED
                self._programs_pending = False
                self._record_round(
                    program_count,
                    circuits_before,
                    runtime_before,
                    status=WorkflowStatus.CANCELLED,
                )
                break
            except Exception as exc:
                self._stop_reason = WorkflowStatus.FAILED
                self._programs_pending = False
                self._record_round(
                    program_count,
                    circuits_before,
                    runtime_before,
                    status=WorkflowStatus.FAILED,
                    error=f"{type(exc).__name__}: {exc}",
                )
                raise
            finally:
                self._round_context = None

            self._workflow_state = state
            self._round_history.append(completed_record)

        self._workflow_state = state
        return self

    def _child_recovery_states(self) -> list[ProgramRoundRecord]:
        return [
            ProgramRoundRecord(
                program_id=_encode_program_id(program_id),
                program_type=_qualified_type(program),
                circuit_count_at_round_start=program.total_circuit_count,
                run_time_at_round_start=program.total_run_time,
            )
            for program_id, program in self._programs.items()
        ]

    def _prepare_checkpoint_session(
        self,
        state: Any,
        checkpoint_config: CheckpointConfig,
        interrupted_checkpoint: RoundCheckpoint | None = None,
        round_input_snapshot: tuple[Path, dict[str, Any]] | None = None,
    ) -> _RoundCheckpointSession:
        if interrupted_checkpoint is None and checkpoint_config.checkpoint_dir is None:
            return _RoundCheckpointSession.inactive()

        if interrupted_checkpoint is not None:
            checkpoint_dir = checkpoint_config.checkpoint_dir
            if checkpoint_dir is None:
                raise ValueError("A checkpoint directory is required.")
            round_path = _round_dir(
                Path(checkpoint_dir), interrupted_checkpoint.round_index
            )
            ensemble_state_payload = interrupted_checkpoint.ensemble_state
        else:
            if round_input_snapshot is None:
                checkpoint_dir = checkpoint_config.checkpoint_dir
                if checkpoint_dir is None:
                    raise ValueError("A checkpoint directory is required.")
                root = _ensure_checkpoint_dir(checkpoint_dir)
                round_path = _round_dir(root, self._round_index)
                # The workflow-state artifact is written here, before prepare().
                round_path.mkdir(parents=True, exist_ok=True)
                ensemble_state_payload = self._save_workflow_checkpoint_state(
                    state, round_path, "input_state"
                )
            else:
                round_path, ensemble_state_payload = round_input_snapshot

        child_recovery_states = self._child_recovery_states()
        return _RoundCheckpointSession.prepare(
            checkpoint_config=checkpoint_config,
            round_path=round_path,
            ensemble_type=_qualified_type(self),
            round_index=self._round_index,
            ensemble_state=ensemble_state_payload,
            programs=list(self._programs.values()),
            child_recovery_states=child_recovery_states,
            interrupted_checkpoint=interrupted_checkpoint,
        )

    def _save_round_input_state(
        self, state: Any, checkpoint_config: CheckpointConfig
    ) -> tuple[Path, dict[str, Any]]:
        """Snapshot round input before program construction consumes RNG state."""
        checkpoint_dir = checkpoint_config.checkpoint_dir
        if checkpoint_dir is None:
            raise ValueError("A checkpoint directory is required.")
        root = _ensure_checkpoint_dir(checkpoint_dir)
        round_path = _round_dir(root, self._round_index)
        round_path.mkdir(parents=True, exist_ok=True)
        ensemble_state_payload = self._save_workflow_checkpoint_state(
            state, round_path, "input_state"
        )
        return round_path, ensemble_state_payload

    @staticmethod
    def _serialize_round_history(
        history: list[RoundRecord],
    ) -> list[dict[str, Any]]:
        return [{**asdict(record), "status": record.status.value} for record in history]

    def _save_completed_round_checkpoint(
        self,
        checkpoint_config: CheckpointConfig,
        state: Any,
        history: list[RoundRecord],
    ) -> Path:
        checkpoint_dir = checkpoint_config.checkpoint_dir
        if checkpoint_dir is None:
            raise ValueError("A checkpoint directory is required.")
        root = _ensure_checkpoint_dir(checkpoint_dir)
        round_path = _round_dir(root, self._round_index)
        round_path.mkdir(parents=True, exist_ok=True)
        ensemble_state_payload = self._save_workflow_checkpoint_state(
            state, round_path, "output_state"
        )
        completed = RoundCheckpoint(
            kind="round_completion",
            ensemble_type=_qualified_type(self),
            round_index=self._round_index,
            ensemble_state=ensemble_state_payload,
            round_history=self._serialize_round_history(history),
            total_circuit_count=self.total_circuit_count,
            total_run_time=self.total_run_time,
        )
        _atomic_write(
            round_path / ROUND_COMPLETION_FILE,
            completed.model_dump_json(indent=2, exclude_none=True),
        )
        return round_path

    def save_state(self, checkpoint_config: CheckpointConfig) -> Path:
        """Persist the latest successfully completed ensemble round."""
        if self._executor is not None:
            raise RuntimeError("Cannot save an ensemble while it is running.")
        if (
            not self._round_history
            or self._round_history[-1].status is not WorkflowStatus.COMPLETE
        ):
            raise RuntimeError("Cannot save an ensemble before a round has completed.")
        if checkpoint_config.checkpoint_dir is None:
            raise ValueError(
                "checkpoint_config.checkpoint_dir must be a non-None Path."
            )
        return self._save_completed_round_checkpoint(
            checkpoint_config, self._workflow_state, list(self._round_history)
        )

    @staticmethod
    def _deserialize_round_history(
        history: list[dict[str, Any]],
    ) -> list[RoundRecord]:
        return [
            RoundRecord(
                number=record["number"],
                program_count=record["program_count"],
                circuit_count=record["circuit_count"],
                run_time=record["run_time"],
                status=WorkflowStatus(record["status"]),
                error=record.get("error"),
            )
            for record in history
        ]

    def restore_state(
        self,
        checkpoint_dir: Path | str,
        subdirectory: str | None = None,
    ) -> Self:
        """Restore the latest ensemble checkpoint onto this instance."""
        if self._executor is not None:
            raise RuntimeError("Cannot restore an ensemble while it is running.")
        if self._programs:
            raise RuntimeError(
                "Cannot restore onto an ensemble with pre-materialized programs."
            )
        checkpoint = _resolve_ensemble_checkpoint(checkpoint_dir, subdirectory)
        round_path = _round_dir(Path(checkpoint_dir), checkpoint.round_index)
        expected_type = _qualified_type(self)
        if checkpoint.ensemble_type != expected_type:
            raise ValueError(
                f"Checkpoint contains {checkpoint.ensemble_type}, "
                f"not {expected_type}."
            )
        if checkpoint.kind == "round_completion":
            round_history_data, total_circuit_count, total_run_time = (
                checkpoint.round_completion_data()
            )
            history = self._deserialize_round_history(round_history_data)
            round_index = checkpoint.round_index
            interrupted_checkpoint = None
        else:
            interrupted_checkpoint = checkpoint
            history = []
            total_circuit_count = 0
            total_run_time = 0.0
            for candidate_index in range(interrupted_checkpoint.round_index - 1, 0, -1):
                try:
                    completed_checkpoint = _resolve_ensemble_checkpoint(
                        checkpoint_dir, f"round_{candidate_index:03d}"
                    )
                except CheckpointNotFoundError:
                    continue
                if completed_checkpoint.kind == "round_completion":
                    if completed_checkpoint.ensemble_type != expected_type:
                        raise ValueError(
                            "Checkpoint history contains an earlier completed round "
                            "from a different ensemble type."
                        )
                    round_history_data, total_circuit_count, total_run_time = (
                        completed_checkpoint.round_completion_data()
                    )
                    history = self._deserialize_round_history(round_history_data)
                    break
            round_index = interrupted_checkpoint.round_index - 1

        stem = (
            "output_state" if checkpoint.kind == "round_completion" else "input_state"
        )
        state = self._load_workflow_checkpoint_state(
            checkpoint.ensemble_state, round_path, stem
        )

        self._workflow_state = state
        self._round_history = history
        self._round_index = round_index
        self._total_circuit_count = total_circuit_count
        self._total_run_time = total_run_time
        self._stop_reason = None
        self._round_context = None
        self._programs_pending = False
        self._interrupted_checkpoint = interrupted_checkpoint
        self._restored_checkpoint_root = Path(checkpoint_dir)
        self._resumed_from_checkpoint = True
        return self

    def _round_record(
        self,
        program_count: int,
        circuits_before: int,
        runtime_before: float,
        *,
        status: WorkflowStatus,
        error: str | None = None,
    ) -> RoundRecord:
        return RoundRecord(
            number=self._round_index,
            program_count=program_count,
            circuit_count=self.total_circuit_count - circuits_before,
            run_time=self.total_run_time - runtime_before,
            status=status,
            error=error,
        )

    def _record_round(
        self,
        program_count: int,
        circuits_before: int,
        runtime_before: float,
        *,
        status: WorkflowStatus,
        error: str | None = None,
    ) -> None:
        """Append a :class:`RoundRecord` for the round now in progress."""
        self._round_history.append(
            self._round_record(
                program_count,
                circuits_before,
                runtime_before,
                status=status,
                error=error,
            )
        )

    def _dispatch(
        self,
        task_fn: Callable[..., Any],
        blocking: bool,
        batch_config: BatchConfig,
        backend: CircuitRunner | None = None,
    ):
        """Drive the ensemble lifecycle using ``task_fn`` per sub-program.

        Shared by :meth:`run` (training) and :meth:`sample_solution` (sampling).
        Owns the executor, ``_BatchCoordinator``, queued progress session,
        submission loop, error cleanup, and the blocking/non-blocking return.
        """
        if self._executor is not None:
            raise RuntimeError("An ensemble is already being run.")

        if len(self._programs) == 0:
            raise RuntimeError("No programs to run.")

        batching_enabled = batch_config.mode is BatchMode.MERGED

        # Validate that all program instances are unique to prevent thread-safety issues
        program_instances = list(self._programs.values())
        if len(set(program_instances)) != len(program_instances):
            raise RuntimeError(
                "Duplicate program instances detected in ensemble. "
                "QuantumProgram instances are stateful and NOT thread-safe. "
                "You must provide a unique instance for each program ID."
            )

        program_keys = [program._progress_key for program in program_instances]
        if len(set(program_keys)) != len(program_keys):
            raise RuntimeError(
                "Duplicate progress keys detected in ensemble. "
                "Copied or pickle-restored QuantumProgram instances retain their "
                "source identity and cannot run alongside it; construct each "
                "ensemble program independently."
            )

        n_workers = _resolve_worker_count(batch_config, len(self._programs))

        self._cancellation_event = Event()
        self.futures.clear()
        self._future_to_program.clear()
        # Per-program counter values at dispatch start; join() adds the delta.
        self._dispatch_count_baseline = {
            program: (program._total_circuit_count, program._total_run_time)
            for program in self._programs.values()
        }

        # Setup → registration → executor.submit happen sequentially on the
        # main thread.  If any of them raises after partial work is done
        # (e.g. some programs registered with the coordinator but the
        # remainder didn't reach _add_program_to_executor), the barrier
        # invariant ``len(_pending) >= len(_active_programs)`` would never
        # hold for survivors and they'd hang forever.  Tear everything
        # down on failure so the caller can retry cleanly.
        try:
            self._start_progress_session(batching_enabled)
            self._executor = ThreadPoolExecutor(max_workers=n_workers)
            if batching_enabled:
                self._install_coordinator(batch_config, n_workers, backend)
            elif backend is not None:
                for program in self._programs.values():
                    self._program_original_backend[program] = program.backend
                    program.backend = backend

            for program in self._programs.values():
                future = self._add_program_to_executor(program, task_fn)
                self.futures.append(future)
                self._future_to_program[future] = program

        except BaseException:
            # Tear down any half-submitted futures and restore all temporary
            # backend/emitter bindings so the caller can retry cleanly. Signal
            # every submitted worker and retain ownership until it exits;
            # otherwise an unbatched worker could outlive the closed session
            # and race a subsequent dispatch of the same stateful program.
            self._cancellation_event.set()
            try:
                if self._coordinator is not None:
                    self._coordinator.shutdown()
                if self._executor is not None:
                    self._executor.shutdown(wait=True)
            finally:
                self._coordinator = None
                self._executor = None
                self._restore_program_backends()
                self._teardown_progress_session()
                self.futures.clear()
                self._future_to_program.clear()
            raise

        if not blocking:
            # Arm safety net
            atexit.register(self._atexit_cleanup_hook)
        else:
            self.join()

        return self

    def _dispatch_sample_solution(
        self,
        resolved: dict[Any, npt.NDArray[np.float64]],
        *,
        backend: CircuitRunner | None,
        blocking: bool,
        batch_config: BatchConfig,
        restored_programs: Container[QuantumProgram] = frozenset(),
        on_sampled: Callable[[QuantumProgram], None] | None = None,
    ) -> Self:
        program_to_id = {program: pid for pid, program in self._programs.items()}

        def _sample_solution_task(program: VariationalQuantumAlgorithm):
            if program in restored_programs:
                return program
            result = cast(SolutionSamplingMixin, program).sample_solution(
                resolved[program_to_id[program]], backend=program.backend
            )
            if on_sampled is not None:
                on_sampled(program)
            return result

        return self._dispatch(
            task_fn=_sample_solution_task,
            blocking=blocking,
            batch_config=batch_config,
            backend=backend,
        )

    def sample_solution(
        self,
        params_per_program: dict[Any, npt.NDArray[np.float64]] | None = None,
        *,
        backend: CircuitRunner | None = None,
        blocking: bool = False,
        batch_config: BatchConfig = BatchConfig(),
        suppress_strict_warning: bool = False,
    ) -> Self:
        """Sample every sub-program's circuit with trained parameters.

        Runs only the final measurement step on each sub-program — no
        EXPECTATION jobs are dispatched. Two usage paths:

        * ``params_per_program=None``: each sub-program uses its own
          ``_best_params`` (typically populated by a prior ``run()`` or a
          loaded checkpoint). A program with no trained parameters raises
          :class:`RuntimeError` upfront with the program ID.
        * ``params_per_program={program_id: params, ...}``: per-program
          parameter sets. Unknown program IDs raise :class:`ValueError`.
          Program IDs that are present in the ensemble but missing from
          the dict fall back to that program's own ``_best_params`` and
          emit a :class:`UserWarning` listing the fallbacks (silence with
          ``suppress_strict_warning=True``).

        Mirrors :meth:`run` for everything else — executor pool sizing,
        merged batching via :class:`_BatchCoordinator`, progress UI,
        cancellation, and the blocking / non-blocking return contract.
        No optimizer state on any sub-program is mutated.

        Args:
            params_per_program: Optional mapping from program ID to
                parameter set. See above for resolution semantics.
            backend: Backend used for this sampling dispatch. ``None`` uses the
                configured sampling backend, falling back to the ensemble's
                primary backend.
            blocking: If ``True``, waits for all programs to complete
                before returning. Defaults to ``False``.
            batch_config: Same semantics as :meth:`run`.
            suppress_strict_warning: When ``True``, silences the
                fallback warning emitted when ``params_per_program`` is
                missing entries for some programs.

        Returns:
            ProgramEnsemble: ``self`` for method chaining.

        Raises:
            RuntimeError: If the ensemble has no programs, if it is
                already running, or if a sub-program has no parameters
                available (no dict entry and empty ``_best_params``).
            ValueError: If ``params_per_program`` contains unknown program
                IDs, or any resolved parameter set has the wrong shape
                for its sub-program's ``n_layers * n_params_per_layer``.
            TypeError: If any sub-program is not a
                :class:`~divi.qprog.VariationalQuantumAlgorithm`.
        """
        resolved = _resolve_sampling_params(
            self._programs,
            params_per_program,
            suppress_strict_warning=suppress_strict_warning,
        )

        selected_backend = backend if backend is not None else self._sampling_backend
        return self._dispatch_sample_solution(
            resolved,
            backend=selected_backend,
            blocking=blocking,
            batch_config=batch_config,
        )

    def check_all_done(self) -> bool:
        """
        Check if all programs in the ensemble have completed execution.

        Returns:
            bool: True if all programs are finished (successfully or with errors),
                False if any are still running.
        """
        if not self.futures:
            warn(
                "check_all_done called with no active futures — run() has "
                "not been invoked (or the ensemble has been reset).",
                UserWarning,
                stacklevel=2,
            )
        return all(future.done() for future in self.futures)

    def _collect_completed_results(self, completed_futures: list):
        """Collect completed program instances from futures.

        Args:
            completed_futures: List to append program instances to.
        """
        for future in self.futures:
            if future.done() and not future.cancelled():
                try:
                    completed_futures.append(future.result())
                except Exception:
                    pass  # Skip failed futures

    def _install_coordinator(
        self,
        batch_config: BatchConfig,
        n_workers: int,
        backend: CircuitRunner | None = None,
    ) -> None:
        """Create the batch coordinator and route every program through it."""
        selected_backend = self.backend if backend is None else backend
        self._coordinator = _BatchCoordinator(
            selected_backend,
            progress_emitter=self._progress_emitter,
            batch_config=batch_config,
            preparation_key=(
                ("preparation", id(self)) if self._preparation_registered else None
            ),
            n_workers=n_workers,
            cancellation_event=self._cancellation_event,
        )
        for program in self._programs.values():
            program_key = program._progress_key
            self._coordinator.register_program(program_key)
            self._program_original_backend[program] = program.backend
            program.backend = _ProxyBackend(
                selected_backend, self._coordinator, program_key
            )

    def _start_progress_session(self, batching_enabled: bool) -> None:
        """Create one dispatch session and bind all participating emitters."""
        self._teardown_progress_session()
        self._preparation_registered = False
        self._workflow_registered = False
        self._workflow_message = None

        if (
            self.reporting_level is ReportingLevel.OFF
            or _environment_disables_progress()
        ):
            self._progress_session = None
            progress_emitter = discard_progress_event
        else:
            state = ProgressState(
                hide_successful_programs=(
                    self.reporting_level is ReportingLevel.COMPACT
                )
            )
            self._progress_session = ProgressSession.queued(state)
            progress_emitter = self._progress_session.emit

        bindings = ExitStack()
        self._progress_bindings = bindings
        self._progress_emitter = progress_emitter
        try:
            for program in self._programs.values():
                bindings.enter_context(program._bind_progress_emitter(progress_emitter))

            if self._progress_session is None:
                return

            if self._round_context is not None:
                round_number, max_rounds = self._round_context
                round_label = (
                    f"Round {round_number}/{max_rounds}"
                    if max_rounds is not None
                    else f"Round {round_number}"
                )
                self._workflow_message = (
                    f"{round_label} — {len(self._programs)} programs"
                )
                workflow_key = ("workflow", id(self))
                progress_emitter(
                    ProgressEvent.register(
                        workflow_key,
                        ProgressScope.WORKFLOW,
                        "Workflow",
                        None,
                    )
                )
                progress_emitter(
                    ProgressEvent.show(workflow_key, self._workflow_message)
                )
                self._workflow_registered = True

            if self.reporting_level is ReportingLevel.COMPACT and batching_enabled:
                progress_emitter(
                    ProgressEvent.register(
                        ("preparation", id(self)),
                        ProgressScope.PREPARATION,
                        "Submitting circuits",
                        len(self._programs),
                    )
                )
                self._preparation_registered = True

            for map_key, program in self._programs.items():
                total = getattr(
                    program,
                    "_expected_total_iterations",
                    getattr(self, "max_iterations", 1),
                )
                progress_emitter(
                    ProgressEvent.register(
                        program._progress_key,
                        ProgressScope.PROGRAM,
                        f"Program {map_key}",
                        total,
                        visible=self.reporting_level is ReportingLevel.FULL,
                    )
                )
        except BaseException:
            self._teardown_progress_session()
            raise

    def _teardown_progress_session(self) -> None:
        """Drain/close the session, then restore every bound program emitter."""
        session = self._progress_session
        bindings, self._progress_bindings = self._progress_bindings, None
        try:
            if session is not None:
                session.close()
        finally:
            if bindings is not None:
                bindings.close()
            self._progress_emitter = (
                discard_progress_event
                if self.reporting_level is ReportingLevel.OFF
                or _environment_disables_progress()
                else log_progress_event
            )

    def _emit_progress_message(
        self,
        program_key: Hashable | None,
        *,
        final_status: TerminalStatus | None = None,
        message: str | None = None,
    ) -> None:
        """Emit typed program feedback through the dispatch's single writer."""
        if program_key is None:
            return
        if message is not None:
            self._progress_emitter(ProgressEvent.show(program_key, message))
        if final_status is not None:
            self._progress_emitter(
                ProgressEvent.finish(
                    program_key,
                    final_status,
                    detail=message,
                )
            )

    def _emit_workflow_stage(self, message: str, *, final: bool = False) -> None:
        """Name the stage a multi-stage ``update_state`` is currently in.

        Active dispatches route the phase through their queued session;
        classical reduction after dispatch uses the normal logging emitter.
        """
        if final:
            logger.info(message)
            return
        if self.reporting_level is ReportingLevel.OFF:
            return
        if not self._workflow_registered:
            return
        self._progress_emitter(ProgressEvent.show(("workflow", id(self)), message))

    def _finish_workflow_progress(self, final_status: TerminalStatus) -> None:
        """Finish the registered standing rows for this workflow round."""
        if self._preparation_registered:
            self._progress_emitter(
                ProgressEvent.finish(
                    ("preparation", id(self)),
                    final_status,
                )
            )
            self._preparation_registered = False
        if self._workflow_registered:
            self._progress_emitter(
                ProgressEvent.finish(
                    ("workflow", id(self)),
                    final_status,
                    detail=self._workflow_message,
                )
            )
            self._workflow_registered = False

    def _stop_remaining_programs(
        self,
        *,
        pending_status: TerminalStatus,
        pending_message: str,
        running_status: TerminalStatus,
        running_message: str,
        failed_status: TerminalStatus = TerminalStatus.FAILED,
        failed_message: str = "Job failed",
    ) -> None:
        """Signal all remaining programs to stop and wait for them to finish.

        Shared mechanical core used by both the cancellation path
        (``KeyboardInterrupt``) and the failure path (program exception).

        Args:
            pending_status: Progress bar ``final_status`` for futures that
                were successfully cancelled before they started.
            pending_message: Progress bar ``message`` for those futures.
            running_status: Progress bar ``final_status`` for futures that
                were already running and had to be waited on.
            running_message: Progress bar ``message`` for those futures.
            failed_status: Progress bar ``final_status`` for futures that
                already finished with an exception (e.g. batch coordinator
                failed all waiting programs).
            failed_message: Progress bar ``message`` for those futures.
        """
        self._cancellation_event.set()

        # Cancel the coordinator first — it unblocks all programs waiting
        # on the barrier and cancels real backend jobs.
        if self._coordinator is not None:
            self._coordinator.cancel()

        successfully_cancelled = []
        unstoppable_futures = []

        for future, program in self._future_to_program.items():
            if future.done():
                # Mark already-failed futures so their progress bars don't
                # freeze.  Futures that completed successfully are fine.
                if not future.cancelled():
                    try:
                        exc = future.exception()
                    except Exception:
                        exc = True  # defensive; treat as failed
                    if exc is not None:
                        # Workers that raised ExecutionCancelledError were
                        # cooperating with the user's cancel; everything
                        # else is a real failure.
                        if isinstance(exc, ExecutionCancelledError):
                            status, message = (
                                TerminalStatus.CANCELLED,
                                "Cancelled by user",
                            )
                        else:
                            status, message = failed_status, failed_message
                        self._emit_progress_message(
                            program._progress_key,
                            final_status=status,
                            message=message,
                        )
                continue

            cancel_result = future.cancel()
            if cancel_result:
                successfully_cancelled.append(program)
            else:
                # Already running — cancel the backend job directly only
                # when there is no coordinator (the coordinator already
                # cancelled real backend jobs above; the proxy has no job_id).
                if self._coordinator is None:
                    program.cancel_unfinished_job()
                unstoppable_futures.append(future)
                self._emit_progress_message(
                    program._progress_key,
                    message="Finishing... ⏳",
                )

        # Immediately mark successfully cancelled tasks
        for program in successfully_cancelled:
            self._emit_progress_message(
                program._progress_key,
                final_status=pending_status,
                message=pending_message,
            )

        # Wait for running tasks to finish
        for future in as_completed(unstoppable_futures):
            program = self._future_to_program[future]
            self._emit_progress_message(
                program._progress_key,
                final_status=running_status,
                message=running_message,
            )

    def _handle_cancellation(self):
        """Handle cancellation gracefully with accurate progress feedback.

        With the batch coordinator active, cancellation works as follows:
        1. ``coordinator.cancel()`` sets the cancelled flag, cancels any
           in-flight backend jobs, and resolves pending futures with
           ``ExecutionCancelledError``.
        2. The program threads see the ``ExecutionCancelledError`` (or the
           ``_cancellation_event``) and exit.
        3. We wait for all still-running futures and mark them in the
           progress bar.

        Without the coordinator the legacy path applies: we try
        ``future.cancel()`` for pending tasks and ``cancel_unfinished_job()``
        for running ones.
        """
        self._stop_remaining_programs(
            pending_status=TerminalStatus.CANCELLED,
            pending_message="Cancelled by user",
            running_status=TerminalStatus.ABORTED,
            running_message="Stopped after current iteration",
        )
        self._report_failed_programs()

    def _report_failed_programs(self) -> None:
        """Render a Rich panel + traceback for any future that finished with
        a non-cancellation exception.

        Called from the cancellation path so users still see what crashed
        — otherwise the failure detail disappears into the progress row.
        Failures that happened before the cancel was requested still get
        the same panel treatment the no-cancel failure path produces.
        """
        failures: list[tuple[QuantumProgram, BaseException]] = []
        for future, program in self._future_to_program.items():
            if not future.done() or future.cancelled():
                continue
            try:
                exc = future.exception()
            except Exception:
                continue
            if exc is None or isinstance(exc, ExecutionCancelledError):
                continue
            failures.append((program, exc))

        if not failures:
            return

        console = (
            self._progress_session.console
            if self._progress_session is not None
            else None
        )
        for program, exc in failures:
            map_key = next(
                (
                    key
                    for key, candidate in self._programs.items()
                    if candidate is program
                ),
                None,
            )
            label = f" (Program {map_key})" if map_key is not None else ""
            render_failure(exc, label=label, console=console)

    def _handle_failure(self, failed_future: Future | None) -> None:
        """Handle a program failure by stopping remaining programs.

        Marks the failed program's progress bar as failed, then stops
        all other running programs using the same mechanism as
        cancellation.

        Args:
            failed_future: The future that raised the exception, or
                ``None`` if it could not be identified.
        """
        if failed_future is not None:
            failed_program = self._future_to_program.get(failed_future)
            if failed_program is not None:
                self._emit_progress_message(
                    failed_program._progress_key,
                    final_status=TerminalStatus.FAILED,
                    message="Job failed",
                )

        self._stop_remaining_programs(
            pending_status=TerminalStatus.CANCELLED,
            pending_message="Cancelled due to failure",
            running_status=TerminalStatus.ABORTED,
            running_message="Aborted due to failure",
        )
        # Every failure gets a panel, not just the one join() spotted first.
        self._report_failed_programs()

    def join(self):
        """
        Wait for all programs in the ensemble to complete and collect results.

        Blocks until all programs finish execution, aggregating their circuit counts
        and run times. Handles keyboard interrupts gracefully by attempting to cancel
        remaining programs.

        Returns:
            bool or None: Returns False if interrupted by KeyboardInterrupt, None otherwise.

        Raises:
            RuntimeError: If any program fails with an exception, after cancelling
                remaining programs.

        Note:
            This method should be called after `run_one_round(blocking=False)`
            to wait for completion. It's automatically called by `run()` and by
            `run_one_round(blocking=True)`.
        """
        if self._executor is None:
            return

        completed_futures = []
        try:
            # The as_completed iterator will yield futures as they finish.
            # If a task fails, future.result() will raise the exception immediately.
            for future in as_completed(self.futures):
                completed_futures.append(future.result())
                program = self._future_to_program.get(future)
                if program is not None:
                    self._emit_progress_message(
                        program._progress_key,
                        final_status=TerminalStatus.SUCCESS,
                    )
            self._finish_workflow_progress(TerminalStatus.SUCCESS)

        except KeyboardInterrupt:
            self._round_cancelled = True
            if self._workflow_registered:
                self._progress_emitter(
                    ProgressEvent.show(
                        ("workflow", id(self)),
                        "Shutdown signal received; waiting for programs to finish",
                    )
                )
            self._handle_cancellation()
            self._finish_workflow_progress(TerminalStatus.CANCELLED)

            # Re-collect all completed results from scratch to avoid duplicates
            # from the as_completed loop above.
            completed_futures.clear()
            self._collect_completed_results(completed_futures)

            return False

        except Exception as e:
            # A task has failed. Identify the culprit and stop everything.
            failed_future = None
            # Count programs that finished successfully *before* we stop
            # anything — programs interrupted by the cancellation event
            # should not count as "completed".
            n_already_done = 0
            for f in self.futures:
                if f.done() and not f.cancelled():
                    try:
                        exc = f.exception()
                    except Exception:
                        exc = True
                    if exc is not None:
                        if failed_future is None:
                            failed_future = f
                    else:
                        n_already_done += 1

            # _handle_failure renders a panel per failed program, so every
            # failure is reported rather than only this first one.
            self._handle_failure(failed_future)
            self._finish_workflow_progress(TerminalStatus.FAILED)

            # Re-collect all completed results from scratch to avoid duplicates
            # from the as_completed loop above.
            completed_futures.clear()
            self._collect_completed_results(completed_futures)

            n_total = len(self._programs)
            raise RuntimeError(
                f"Ensemble execution failed: {n_already_done}/{n_total} programs "
                f"completed before failure."
            ) from e

        finally:
            # Aggregate results from completed program instances.
            # run() returns self, so completed_futures contains programs.
            if completed_futures:
                baseline = self._dispatch_count_baseline
                self._total_circuit_count += sum(
                    p._total_circuit_count - baseline.get(p, (0, 0.0))[0]
                    for p in completed_futures
                )
                # For async backends the individual programs don't track runtime
                # (the proxy returns sync results). Use the coordinator's total
                # which is captured from the real backend's poll responses.
                if (
                    self._coordinator is not None
                    and self._coordinator.total_runtime > 0
                ):
                    self._total_run_time += self._coordinator.total_runtime
                else:
                    self._total_run_time += sum(
                        p._total_run_time - baseline.get(p, (0, 0.0))[1]
                        for p in completed_futures
                    )
            self.futures.clear()

            # A second KeyboardInterrupt lands most often in the executor
            # shutdown below, so the display/listener teardown gets its own
            # finally — otherwise an orphaned Live would block every later
            # dispatch on this console.
            try:
                # Shutdown coordinator
                if self._coordinator is not None:
                    self._coordinator.shutdown()
                    self._coordinator = None

                # Shutdown executor and wait for all threads to complete
                # This is critical for Python 3.12 to prevent process hangs
                if self._executor is not None:
                    executor, self._executor = self._executor, None
                    executor.shutdown(wait=True)
            finally:
                self._restore_program_backends()
                self._teardown_progress_session()

        # After successful cleanup, try to unregister the hook.
        try:
            atexit.unregister(self._atexit_cleanup_hook)
        except TypeError:
            pass

    def _check_ready_for_aggregation(self):
        """Validate that programs exist, are complete, and results are ready."""
        if len(self._programs) == 0:
            raise RuntimeError("No programs to aggregate. Run create_programs() first.")

        if self._executor is not None:
            self.join()

        for program in self._programs.values():
            if not program.has_results() and not getattr(program, "_best_probs", None):
                raise RuntimeError(
                    "Some/All programs have no results. "
                    "Did you call run() or sample_solution()?"
                )

    @abstractmethod
    def aggregate_results(self) -> Any:
        """
        Aggregate results from all programs in the ensemble after execution.

        This is an abstract method that must be implemented by subclasses. The base
        implementation performs validation checks:
        - Ensures programs have been created
        - Waits for any running programs to complete (calls join() if needed)
        - Verifies that all programs have completed execution (non-empty losses_history)

        Subclasses should call super().aggregate_results() first, then implement
        their own aggregation logic to combine results from all programs. The
        aggregation should handle different result formats (counts dictionary,
        expectation values, etc.) as appropriate for the specific use case.

        Returns:
            The aggregated result, format depends on the subclass implementation.

        Raises:
            RuntimeError: If no programs exist, or if programs haven't completed
                execution (empty losses_history).
        """
        self._check_ready_for_aggregation()

    def get_top_solutions(self, n=10, *, strategy=None):
        """Get the top-N global solutions from partition aggregation.

        Available on subclasses that aggregate per-partition results
        (e.g., ``PartitioningProgramEnsemble``).

        Args:
            n (int): Number of top solutions to return. Must be >= 1.
            strategy: An :class:`~divi.qprog.AggregationStrategy` controlling how
                per-partition candidates are combined. Defaults to
                :class:`~divi.qprog.BeamSearchStrategy`.

        Returns:
            Subclass-specific format. See subclass documentation.

        Raises:
            NotImplementedError: If the subclass does not override this method.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support get_top_solutions."
        )
