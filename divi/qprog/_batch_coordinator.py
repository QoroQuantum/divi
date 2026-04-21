# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Batch coordinator for ProgramEnsemble circuit submission.

Provides a proxy backend and coordinator that merge circuit submissions
from multiple QuantumProgram instances into single backend calls,
improving backend utilization.
"""

import logging
from collections.abc import Mapping
from concurrent.futures import Future
from dataclasses import dataclass
from enum import Enum
from queue import Queue
from threading import Event, Lock, Thread
from typing import NamedTuple

from divi.backends import CircuitRunner, JobStatus
from divi.backends._execution_result import ExecutionResult
from divi.backends._shot_allocation import from_wire, to_wire
from divi.exceptions import ExecutionCancelledError
from divi.reporting import BATCH_COLORS

logger = logging.getLogger(__name__)

# Separator used to namespace circuit tags per program.
# Chosen because it never appears in CircuitTag encoded strings.
_TAG_SEP = "@"


class BatchMode(Enum):
    """Controls whether circuit submissions are merged across programs.

    Attributes:
        MERGED: Circuit submissions from all programs are merged into single
            backend calls via the batch coordinator.
        OFF: Each program submits circuits independently to the backend.
    """

    MERGED = "merged"
    OFF = "off"


@dataclass(frozen=True)
class BatchConfig:
    """Configuration for circuit batching in :meth:`ProgramEnsemble.run`.

    Attributes:
        mode: Whether to merge circuit submissions across programs.
            Defaults to :attr:`~divi.qprog.ensemble.BatchMode.MERGED`.
        max_batch_size: Maximum number of circuits per merged backend call.
            When set, the coordinator flushes early once the pending circuit
            count reaches this limit instead of waiting for every active
            program to submit.  ``None`` (the default) preserves the
            wait-for-all barrier behavior.  Only meaningful when
            ``mode`` is :attr:`~divi.qprog.ensemble.BatchMode.MERGED`.
        _sort_programs: Whether to sort programs by key before merging their
            circuits into a single backend call.  Defaults to ``False``.

            When ``False`` (default), circuits are merged in submission-arrival
            order — the order in which program threads reach the flush barrier.
            This preserves backward-compatible behavior.

            Set to ``True`` to merge circuits in a consistent, key-sorted order
            regardless of thread scheduling.  This ensures that
            position-dependent seeds (e.g. ``seed + circuit_index`` in
            :class:`~divi.backends.QiskitSimulator` deterministic mode)
            map to the same circuit on every run, making seeded experiments
            fully reproducible.  Only meaningful when ``mode`` is
            :attr:`~divi.qprog.ensemble.BatchMode.MERGED`.
    """

    mode: BatchMode = BatchMode.MERGED
    max_batch_size: int | None = None
    _sort_programs: bool = False

    def __post_init__(self):
        if self.max_batch_size is not None and self.max_batch_size < 1:
            raise ValueError(
                f"max_batch_size must be >= 1 or None, got {self.max_batch_size}"
            )
        if self.mode is BatchMode.OFF and self.max_batch_size is not None:
            raise ValueError("max_batch_size has no effect when mode is BatchMode.OFF.")
        if self.mode is BatchMode.OFF and self._sort_programs:
            raise ValueError("_sort_programs has no effect when mode is BatchMode.OFF.")


class _PendingEntry(NamedTuple):
    """A single program's pending submission awaiting the flush barrier."""

    circuits: dict[str, str]
    kwargs: dict
    future: Future


# Batch dict type used throughout the coordinator.
_Batch = dict[str, _PendingEntry]


class _FlushGroup:
    """Tracks one merged submission: the per-program futures and the backend job."""

    __slots__ = ("futures", "execution_result", "color", "program_keys", "label")

    def __init__(self, futures: dict[str, Future], color: str, label: str = ""):
        self.futures = futures
        self.program_keys = set(futures.keys())
        self.color = color
        self.label = label
        self.execution_result: ExecutionResult | None = None


def _fail_futures(batch: _Batch, exc: BaseException) -> None:
    """Set *exc* on all unresolved futures in *batch*."""
    for entry in batch.values():
        if not entry.future.done():
            entry.future.set_exception(exc)


class _BatchCoordinator:
    """Coordinates circuit submissions from multiple programs into merged jobs.

    Programs register before execution and deregister when they finish.
    Each call to :meth:`submit` blocks until the barrier is met (all active
    programs have submitted) and the merged job returns results.  Multiple
    flush groups can be in-flight concurrently.
    """

    def __init__(
        self,
        real_backend: CircuitRunner,
        progress_queue: Queue | None = None,
        batch_config: BatchConfig | None = None,
    ):
        self._real_backend = real_backend
        self._progress_queue = progress_queue
        self._batch_config = batch_config or BatchConfig()
        self._lock = Lock()
        self._cancelled = Event()

        # Programs currently executing (not yet finished their run()).
        self._active_programs: set[str] = set()

        # Pending submissions waiting for the barrier.
        self._pending: _Batch = {}

        # In-flight flush groups (background threads processing backend jobs).
        self._in_flight: list[_FlushGroup] = []
        self._in_flight_lock = Lock()

        # Cumulative runtime tracked from async backend responses.
        self._total_runtime = 0.0

        # Color cycling for flush group indicators.
        self._color_index = 0

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_program(self, program_key: str) -> None:
        """Register a program as active before it starts executing."""
        with self._lock:
            self._active_programs.add(program_key)

    def deregister_program(self, program_key: str) -> None:
        """Remove a program from the active set.

        If the reduced active set means the barrier is now met for the
        current pending batch, a flush is triggered.
        """
        with self._lock:
            self._active_programs.discard(program_key)
            if self._should_flush():
                self._trigger_flush()

    # ------------------------------------------------------------------
    # Submission (called from _ProxyBackend.submit_circuits)
    # ------------------------------------------------------------------

    def submit(
        self,
        program_key: str,
        prefixed_circuits: dict[str, str],
        **kwargs,
    ) -> tuple[list[dict], float]:
        """Submit circuits and block until the merged job returns results.

        Args:
            program_key: Unique identifier for the calling program.
            prefixed_circuits: Circuit dict with tags already namespaced.
            **kwargs: Backend kwargs forwarded to ``submit_circuits``.

        Returns:
            Tuple of (demuxed results list, per-program runtime share).

        Raises:
            ExecutionCancelledError: If the coordinator has been cancelled.
        """
        future: Future = Future()

        with self._lock:
            if self._cancelled.is_set():
                raise ExecutionCancelledError("Batch coordinator has been cancelled.")

            self._pending[program_key] = _PendingEntry(
                prefixed_circuits, kwargs, future
            )

            if self._should_flush():
                self._trigger_flush()

        # Block until this program's results are ready.
        return future.result()

    # ------------------------------------------------------------------
    # Barrier & flush
    # ------------------------------------------------------------------

    def _pending_circuit_count(self) -> int:
        """Total circuits across all pending submissions (lock must be held)."""
        return sum(len(entry.circuits) for entry in self._pending.values())

    def _should_flush(self) -> bool:
        """Check whether the barrier condition is met (lock must be held)."""
        if not self._pending:
            return False
        # Barrier: all active programs have submitted.
        if len(self._pending) >= len(self._active_programs):
            return True
        # Circuit-count cap: flush early when pending circuits hit the limit.
        if (
            self._batch_config.max_batch_size is not None
            and self._pending_circuit_count() >= self._batch_config.max_batch_size
        ):
            return True
        return False

    def _next_color(self) -> str:
        """Return the next color in the cycle (lock must be held)."""
        color = BATCH_COLORS[self._color_index % len(BATCH_COLORS)]
        self._color_index += 1
        return color

    def _trigger_flush(self) -> None:
        """Snapshot the pending batch and dispatch in a background thread.

        Must be called with ``self._lock`` held.
        """

        pending_items = (
            sorted(self._pending.items())
            if self._batch_config._sort_programs
            else list(self._pending.items())
        )
        batch = dict(pending_items)
        self._pending.clear()

        color = self._next_color()
        flush_group = _FlushGroup(
            futures={key: entry.future for key, entry in batch.items()},
            color=color,
        )
        with self._in_flight_lock:
            self._in_flight.append(flush_group)

        thread = Thread(
            target=self._do_flush,
            args=(batch, flush_group),
            daemon=True,
        )
        thread.start()

    def _send_batch_progress(
        self,
        flush_group: _FlushGroup,
        *,
        n_circuits: int = 0,
        n_programs: int = 0,
        **kwargs,
    ) -> None:
        """Send a batch-level progress message to the queue."""
        if self._progress_queue is None:
            return
        msg = {
            "batch": True,
            "batch_id": id(flush_group),
            "batch_label": flush_group.label,
            "batch_color": flush_group.color,
            "program_keys": list(flush_group.program_keys),
            "n_circuits": n_circuits,
            "n_programs": n_programs,
            "progress": 0,
            **kwargs,
        }
        self._progress_queue.put(msg)

    @staticmethod
    def _merge_circuits_and_kwargs(
        batch: _Batch,
    ) -> tuple[dict[str, str], dict]:
        """Merge circuits from all programs and build unified submit kwargs.

        When all programs share identical kwargs the circuits are merged in
        batch iteration order and the common kwargs are returned directly,
        with one exception: ``shot_groups`` indices are per-program and must
        be re-offset to point into the merged circuit list.

        When programs have different ``ham_ops``, circuits are **reordered** so
        that circuits sharing the same ``ham_ops`` are contiguous.  The
        individual ``ham_ops`` strings are combined with ``|`` and a
        ``circuit_ham_map`` is computed so the backend routes each group to the
        correct circuit slice. Combining heterogeneous ``ham_ops`` with
        ``shot_groups`` is currently rejected because the circuit reordering
        would require reshuffling the per-circuit shot allocation in lockstep.

        Returns:
            ``(merged_circuits, submit_kwargs)``
        """
        all_kwargs = [entry.kwargs for entry in batch.values()]

        # Fast path: all kwargs identical → simple merge.
        if all(kw == all_kwargs[0] for kw in all_kwargs):
            merged: dict[str, str] = {}
            offset = 0
            template_shot_groups = all_kwargs[0].get("shot_groups")
            merged_ranges = [] if template_shot_groups is not None else None
            for entry in batch.values():
                if merged_ranges is not None:
                    for r in from_wire(entry.kwargs["shot_groups"]):
                        merged_ranges.append(r.shift(offset))
                merged.update(entry.circuits)
                offset += len(entry.circuits)

            merged_kw = dict(all_kwargs[0])
            if merged_ranges is not None:
                merged_kw["shot_groups"] = to_wire(merged_ranges)
            return merged, merged_kw

        # Slow path: kwargs differ across programs.  We support two
        # independent forms of variation: per-program ``shot_groups`` (sampling
        # mode) and per-program ``ham_ops`` (expval mode). Combining the two
        # would require reshuffling the per-circuit shot allocation when
        # circuits are reordered by observable group, which is out of scope.
        any_shot_groups = any(
            entry.kwargs.get("shot_groups") is not None for entry in batch.values()
        )
        any_ham_ops = any(
            entry.kwargs.get("ham_ops") is not None for entry in batch.values()
        )
        if any_shot_groups and any_ham_ops:
            raise ValueError(
                "Cannot merge programs that combine per-program 'shot_groups' "
                "with heterogeneous 'ham_ops'. Submit such programs in "
                "separate batches."
            )

        if any_shot_groups:
            # Programs differ only in shot_groups (sampling mode). Merge
            # circuits in batch order and re-offset each program's
            # shot_groups indices into the merged circuit list.
            merged = {}
            merged_ranges = []
            offset = 0
            for entry in batch.values():
                shot_groups_in = entry.kwargs.get("shot_groups")
                if shot_groups_in is None:
                    raise ValueError(
                        "Cannot merge a mix of programs with and without "
                        "'shot_groups' in the same sub-batch. Either set "
                        "shot_distribution on every program or none."
                    )
                for r in from_wire(shot_groups_in):
                    merged_ranges.append(r.shift(offset))
                merged.update(entry.circuits)
                offset += len(entry.circuits)

            # Use the first program's kwargs as a template, replacing
            # shot_groups with the merged version. All other kwargs must
            # match (the equality check above failed only on shot_groups).
            merged_kw = dict(all_kwargs[0])
            merged_kw["shot_groups"] = to_wire(merged_ranges)
            return merged, merged_kw

        # Group by ham_ops so circuits with the same observable are contiguous.
        ham_to_programs: dict[str, list[str]] = {}
        for prog_key, entry in batch.items():
            ham = entry.kwargs.get("ham_ops")
            if ham is None:
                raise ValueError(
                    f"Program {prog_key!r} has no 'ham_ops' but its kwargs "
                    f"differ from other programs in the sub-batch. "
                    f"Cannot merge programs with incompatible submit kwargs."
                )
            ham_to_programs.setdefault(ham, []).append(prog_key)

        merged = {}
        ham_groups: list[str] = []
        circuit_ham_map: list[list[int]] = []
        offset = 0

        for ham, prog_keys in ham_to_programs.items():
            group_start = offset
            for pk in prog_keys:
                circuits = batch[pk].circuits
                merged.update(circuits)
                offset += len(circuits)
            ham_groups.append(ham)
            circuit_ham_map.append([group_start, offset])

        # Base kwargs from first program, replacing ham_ops with merged version.
        merged_kw = {k: v for k, v in all_kwargs[0].items() if k != "ham_ops"}
        merged_kw["ham_ops"] = "|".join(ham_groups)
        merged_kw["circuit_ham_map"] = circuit_ham_map
        return merged, merged_kw

    @staticmethod
    def _split_by_ham_ops(batch: _Batch) -> list[_Batch]:
        """Split a batch into compatible sub-batches.

        Programs with ``ham_ops`` cannot be merged with programs without it
        because the backend treats the two as fundamentally different job types
        (observable evaluation vs shot-based sampling).  This method partitions
        the batch so each sub-batch can be merged safely.
        """
        with_ham: _Batch = {}
        without_ham: _Batch = {}
        for prog_key, entry in batch.items():
            if entry.kwargs.get("ham_ops") is not None:
                with_ham[prog_key] = entry
            else:
                without_ham[prog_key] = entry

        return [sb for sb in (with_ham, without_ham) if sb]

    def _do_flush(self, batch: _Batch, flush_group: _FlushGroup) -> None:
        """Merge circuits, submit to real backend, demux results, resolve futures.

        Programs are first split into compatible sub-batches (with/without
        ``ham_ops``) so that each backend call receives a uniform set of
        kwargs.  Within a sub-batch, programs with different ``ham_ops``
        values are merged using ``circuit_ham_map``.
        """
        sub_flush_groups: list[_FlushGroup] = []
        try:
            if self._cancelled.is_set():
                _fail_futures(
                    batch,
                    ExecutionCancelledError("Batch coordinator has been cancelled."),
                )
                return

            sub_batches = self._split_by_ham_ops(batch)
            total_runtime = 0.0

            # Build (sub_batch, flush_group) pairs. When there's only one
            # sub-batch reuse the parent flush_group; otherwise create
            # labelled sub-groups so progress lines are distinct.
            sub_groups: list[tuple[_Batch, _FlushGroup]] = []
            if len(sub_batches) == 1:
                sub_groups.append((sub_batches[0], flush_group))
            else:
                for sub_batch in sub_batches:
                    has_ham = any(e.kwargs.get("ham_ops") for e in sub_batch.values())
                    sub_fg = _FlushGroup(
                        futures={k: e.future for k, e in sub_batch.items()},
                        color=flush_group.color,
                        label="expval" if has_ham else "shots",
                    )
                    sub_flush_groups.append(sub_fg)
                    with self._in_flight_lock:
                        self._in_flight.append(sub_fg)
                    sub_groups.append((sub_batch, sub_fg))

            for sub_batch, sub_fg in sub_groups:
                runtime = self._submit_sub_batch(sub_batch, sub_fg)
                total_runtime += runtime

            with self._lock:
                self._total_runtime += total_runtime

        except ExecutionCancelledError:
            self._send_batch_progress(flush_group, final_status="Cancelled")
            _fail_futures(
                batch, ExecutionCancelledError("Batch coordinator has been cancelled.")
            )
        except Exception as exc:
            self._send_batch_progress(flush_group, final_status="Failed")
            _fail_futures(batch, exc)
        finally:
            with self._in_flight_lock:
                for fg in [flush_group, *sub_flush_groups]:
                    if fg in self._in_flight:
                        self._in_flight.remove(fg)

    def _submit_sub_batch(self, sub_batch: _Batch, flush_group: _FlushGroup) -> float:
        """Merge, submit, poll, demux, and resolve a single compatible sub-batch.

        Returns the runtime reported by the backend.
        """
        merged_circuits, submit_kwargs = self._merge_circuits_and_kwargs(sub_batch)

        n_circuits = len(merged_circuits)
        n_programs = len(sub_batch)

        self._send_batch_progress(
            flush_group,
            n_circuits=n_circuits,
            n_programs=n_programs,
        )

        execution_result = self._real_backend.submit_circuits(
            merged_circuits, **submit_kwargs
        )
        flush_group.execution_result = execution_result

        if self._cancelled.is_set():
            raise ExecutionCancelledError("Batch coordinator has been cancelled.")

        # --- Collect results (sync or async) ---
        runtime = 0.0
        if execution_result.job_id is not None:
            results_list, runtime = self._poll_and_get_results(
                execution_result, flush_group, n_circuits, n_programs
            )
        else:
            results_list = execution_result.results
            if results_list is None:
                raise ValueError("ExecutionResult has neither results nor job_id.")

        self._send_batch_progress(
            flush_group,
            n_circuits=n_circuits,
            n_programs=n_programs,
            final_status="Success",
        )

        # --- Demultiplex results by tag prefix ---
        program_results: dict[str, list[dict]] = {}
        for item in results_list:
            prefix, original_label = item["label"].split(_TAG_SEP, 1)
            program_results.setdefault(prefix, []).append(
                {"label": original_label, "results": item["results"]}
            )

        # --- Resolve futures ---
        per_program_runtime = runtime / n_programs if n_programs > 0 else 0.0
        for prog_key, entry in sub_batch.items():
            if not entry.future.done():
                entry.future.set_result(
                    (program_results.get(prog_key, []), per_program_runtime)
                )

        return runtime

    # ------------------------------------------------------------------
    # Async backend helpers
    # ------------------------------------------------------------------

    def _poll_and_get_results(
        self,
        execution_result: ExecutionResult,
        flush_group: _FlushGroup,
        n_circuits: int,
        n_programs: int,
    ) -> tuple[list[dict], float]:
        """Poll an async job to completion and return (results, runtime)."""
        runtime = 0.0

        def _on_complete(response):
            nonlocal runtime
            if isinstance(response, dict):
                runtime = float(response.get("run_time", 0))
            elif isinstance(response, list):
                runtime = sum(float(r.json()["run_time"]) for r in response)

        def _progress_callback(n_polls, job_status):
            self._send_batch_progress(
                flush_group,
                n_circuits=n_circuits,
                n_programs=n_programs,
                service_job_id=execution_result.job_id,
                job_status=job_status,
                poll_attempt=n_polls,
                max_retries=getattr(self._real_backend, "max_retries", 0),
            )

        status = self._real_backend.poll_job_status(
            execution_result,
            loop_until_complete=True,
            on_complete=_on_complete,
            verbose=False,
            progress_callback=_progress_callback,
        )

        if status == JobStatus.FAILED:
            raise RuntimeError(
                f"Merged batch job {execution_result.job_id} has failed."
            )
        if status == JobStatus.CANCELLED:
            raise ExecutionCancelledError(
                f"Merged batch job {execution_result.job_id} was cancelled."
            )
        if status != JobStatus.COMPLETED:
            raise RuntimeError(
                f"Merged batch job {execution_result.job_id} "
                f"ended with unexpected status: {status}"
            )

        completed = self._real_backend.get_job_results(execution_result)
        return completed.results, runtime

    # ------------------------------------------------------------------
    # Cancellation & shutdown
    # ------------------------------------------------------------------

    def cancel(self) -> None:
        """Cancel all pending and in-flight operations.

        Safe to call multiple times — subsequent calls are no-ops.
        """
        if self._cancelled.is_set():
            return
        self._cancelled.set()

        # Cancel in-flight backend jobs
        with self._in_flight_lock:
            for group in self._in_flight:
                if (
                    group.execution_result is not None
                    and group.execution_result.job_id is not None
                ):
                    try:
                        self._real_backend.cancel_job(group.execution_result)
                    except Exception:
                        logger.debug(
                            "Failed to cancel in-flight batch job", exc_info=True
                        )

        # Resolve any pending futures that haven't been flushed yet
        with self._lock:
            _fail_futures(
                self._pending,
                ExecutionCancelledError("Batch coordinator has been cancelled."),
            )
            self._pending.clear()

    def shutdown(self) -> None:
        """Clean up coordinator state."""
        self.cancel()
        self._active_programs.clear()

    @property
    def total_runtime(self) -> float:
        """Cumulative backend runtime across all flushed jobs."""
        return self._total_runtime


class _ProxyBackend(CircuitRunner):
    """Transparent backend proxy that routes submissions through a coordinator.

    From the program's perspective this behaves like a synchronous backend:
    ``submit_circuits`` blocks until the coordinator has flushed the merged
    job and demultiplexed results back.
    """

    def __init__(
        self,
        real_backend: CircuitRunner,
        coordinator: _BatchCoordinator,
        program_key: str,
    ):
        super().__init__(shots=real_backend.shots)
        self._real = real_backend
        self._coordinator = coordinator
        self._program_key = program_key

    # --- Delegated properties ---

    @property
    def supports_expval(self) -> bool:
        return self._real.supports_expval

    @property
    def is_async(self) -> bool:
        # Results come back synchronously from the coordinator.
        return False

    @property
    def little_endian_bitstrings(self) -> bool:
        return self._real.little_endian_bitstrings

    @property
    def max_retries(self):
        return getattr(self._real, "max_retries", 0)

    # --- Intercepted methods ---

    def submit_circuits(self, circuits: Mapping[str, str], **kwargs) -> ExecutionResult:
        """Prefix tags, submit to coordinator, return sync results."""
        prefixed = {
            f"{self._program_key}{_TAG_SEP}{tag}": qasm
            for tag, qasm in circuits.items()
        }

        results, _runtime = self._coordinator.submit(
            self._program_key, prefixed, **kwargs
        )
        return ExecutionResult(results=results)
