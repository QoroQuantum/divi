# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Batch coordinator for ProgramBatch circuit submission.

Provides a proxy backend and coordinator that merge circuit submissions
from multiple QuantumProgram instances into single backend calls,
improving backend utilization.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from concurrent.futures import Future
from queue import Queue
from threading import Event, Lock, Thread

from divi.backends import CircuitRunner, JobStatus
from divi.backends._execution_result import ExecutionResult
from divi.qprog.exceptions import _CancelledError
from divi.reporting import BATCH_COLORS

logger = logging.getLogger(__name__)

# Separator used to namespace circuit tags per program.
# Chosen because it never appears in CircuitTag encoded strings.
_TAG_SEP = "@"


class _FlushGroup:
    """Tracks one merged submission: the per-program futures and the backend job."""

    __slots__ = ("futures", "execution_result", "color", "program_keys")

    def __init__(self, futures: dict[str, Future], color: str):
        self.futures = futures
        self.program_keys = set(futures.keys())
        self.color = color
        self.execution_result: ExecutionResult | None = None


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
    ):
        self._real_backend = real_backend
        self._progress_queue = progress_queue
        self._lock = Lock()
        self._cancelled = Event()

        # Programs currently executing (not yet finished their run()).
        self._active_programs: set[str] = set()

        # Pending submissions waiting for the barrier.
        # Maps program_key -> (prefixed_circuits, kwargs, Future)
        self._pending: dict[str, tuple[dict[str, str], dict, Future]] = {}

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
            _CancelledError: If the coordinator has been cancelled.
        """
        future: Future = Future()

        with self._lock:
            if self._cancelled.is_set():
                raise _CancelledError("Batch coordinator has been cancelled.")

            self._pending[program_key] = (prefixed_circuits, kwargs, future)

            if self._should_flush():
                self._trigger_flush()

        # Block until this program's results are ready.
        return future.result()

    # ------------------------------------------------------------------
    # Barrier & flush
    # ------------------------------------------------------------------

    def _should_flush(self) -> bool:
        """Check whether the barrier condition is met (lock must be held)."""
        if not self._pending:
            return False
        return len(self._pending) >= len(self._active_programs)

    def _next_color(self) -> str:
        """Return the next color in the cycle (lock must be held)."""
        color = BATCH_COLORS[self._color_index % len(BATCH_COLORS)]
        self._color_index += 1
        return color

    def _trigger_flush(self) -> None:
        """Snapshot the pending batch and dispatch in a background thread.

        Must be called with ``self._lock`` held.
        """
        batch = dict(self._pending)
        self._pending.clear()

        color = self._next_color()
        flush_group = _FlushGroup(
            futures={key: entry[2] for key, entry in batch.items()},
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
            "batch_color": flush_group.color,
            "program_keys": list(flush_group.program_keys),
            "n_circuits": n_circuits,
            "n_programs": n_programs,
            "progress": 0,
            **kwargs,
        }
        self._progress_queue.put(msg)

    @staticmethod
    def _kwargs_key(kwargs: dict) -> tuple:
        """Return a hashable key for grouping programs with compatible kwargs."""
        return tuple(sorted(kwargs.items()))

    def _do_flush(
        self,
        batch: dict[str, tuple[dict[str, str], dict, Future]],
        flush_group: _FlushGroup,
    ) -> None:
        """Merge circuits, submit to real backend, demux results, resolve futures.

        Programs with identical submit kwargs are merged into a single backend
        call.  Programs with different kwargs (e.g. different ``ham_ops``) are
        submitted in separate calls within the same flush.
        """
        try:
            if self._cancelled.is_set():
                for _, (_, _, fut) in batch.items():
                    if not fut.done():
                        fut.set_exception(
                            _CancelledError("Batch coordinator has been cancelled.")
                        )
                return

            n_circuits = sum(len(circuits) for circuits, _, _ in batch.values())
            n_programs = len(batch)

            # Notify progress: batch submitted
            self._send_batch_progress(
                flush_group,
                n_circuits=n_circuits,
                n_programs=n_programs,
                message="Submitting",
            )

            # --- Group programs by compatible kwargs ---
            groups: dict[tuple, list[str]] = {}
            for prog_key, (_, kwargs, _) in batch.items():
                key = self._kwargs_key(kwargs)
                groups.setdefault(key, []).append(prog_key)

            # --- Submit each group and collect results ---
            all_results: list[dict] = []
            total_runtime = 0.0

            for _kw_key, prog_keys in groups.items():
                merged_circuits: dict[str, str] = {}
                group_kwargs = batch[prog_keys[0]][1]
                for pk in prog_keys:
                    merged_circuits.update(batch[pk][0])

                execution_result = self._real_backend.submit_circuits(
                    merged_circuits, **group_kwargs
                )
                flush_group.execution_result = execution_result

                runtime = 0.0
                if execution_result.job_id is not None:
                    results_list, runtime = self._poll_and_get_results(
                        execution_result, flush_group, n_circuits, n_programs
                    )
                else:
                    results_list = execution_result.results
                    if results_list is None:
                        raise ValueError(
                            "ExecutionResult has neither results nor job_id."
                        )

                all_results.extend(results_list)
                total_runtime += runtime

            # Track cumulative runtime
            self._total_runtime += total_runtime

            # Notify progress: batch complete
            self._send_batch_progress(
                flush_group,
                n_circuits=n_circuits,
                n_programs=n_programs,
                final_status="Success",
            )

            # --- Demultiplex results by tag prefix ---
            program_results: dict[str, list[dict]] = {}
            for item in all_results:
                label = item["label"]
                prefix, original_label = label.split(_TAG_SEP, 1)
                prog_key = prefix
                program_results.setdefault(prog_key, []).append(
                    {"label": original_label, "results": item["results"]}
                )

            # --- Resolve futures ---
            per_program_runtime = total_runtime / n_programs if n_programs > 0 else 0.0

            for prog_key, (_, _, fut) in batch.items():
                if not fut.done():
                    fut.set_result(
                        (program_results.get(prog_key, []), per_program_runtime)
                    )

        except _CancelledError:
            self._send_batch_progress(flush_group, final_status="Cancelled")
            for _, (_, _, fut) in batch.items():
                if not fut.done():
                    fut.set_exception(
                        _CancelledError("Batch coordinator has been cancelled.")
                    )
        except Exception as exc:
            self._send_batch_progress(flush_group, final_status="Failed")
            for _, (_, _, fut) in batch.items():
                if not fut.done():
                    fut.set_exception(exc)
        finally:
            with self._in_flight_lock:
                if flush_group in self._in_flight:
                    self._in_flight.remove(flush_group)

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
            raise _CancelledError(
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
        """Cancel all pending and in-flight operations."""
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
            for _, (_, _, fut) in self._pending.items():
                if not fut.done():
                    fut.set_exception(
                        _CancelledError("Batch coordinator has been cancelled.")
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
        self._last_runtime = 0.0

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

        results, runtime = self._coordinator.submit(
            self._program_key, prefixed, **kwargs
        )
        self._last_runtime = runtime

        return ExecutionResult(results=results)
