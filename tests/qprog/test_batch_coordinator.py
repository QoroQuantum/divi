# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for _BatchCoordinator, _ProxyBackend, and related helpers."""

import time
from concurrent.futures import Future
from dataclasses import replace
from threading import Barrier, Event, Lock, Thread

import pytest

from divi.backends import (
    AsyncJobBackend,
    CircuitRunner,
    ExecutionResult,
    JobCancelledError,
    JobStatus,
    JobTimedOutError,
)
from divi.circuits._payloads import bound_circuits
from divi.exceptions import ExecutionCancelledError
from divi.qprog import BatchConfig, BatchMode
from divi.qprog._batch_coordinator import (
    _BATCH_COLORS,
    _Batch,
    _BatchCoordinator,
    _fail_futures,
    _FlushGroup,
    _PendingEntry,
    _ProxyBackend,
    _route_batch_circuits,
)
from divi.reporting._events import (
    EventKind,
    ProgressEvent,
    ProgressScope,
    TerminalStatus,
)
from divi.reporting._state import ProgressState
from tests._helpers import exact_match


class FakeSyncBackend(CircuitRunner):
    """Minimal synchronous backend that echoes circuit labels as results."""

    def __init__(self, shots: int = 100, resolves_parameters: bool = False):
        super().__init__(shots=shots)
        self.submitted: list[dict[str, str]] = []
        self._resolves_parameters = resolves_parameters

    @property
    def resolves_parameters(self) -> bool:
        return self._resolves_parameters

    @property
    def is_async(self) -> bool:
        return False

    @property
    def supports_expval(self) -> bool:
        return False

    def submit_circuits(self, payloads, **kwargs) -> ExecutionResult:
        circuits = bound_circuits(payloads)
        self.submitted.append(circuits)
        results = [
            {"label": label, "results": {"00": self._shots}} for label in circuits
        ]
        return ExecutionResult(results=results)


class FakeExpvalBackend(CircuitRunner):
    """Synchronous backend that supports expval and records kwargs."""

    def __init__(self, shots: int = 100):
        super().__init__(shots=shots)
        self.call_log: list[tuple[dict, dict]] = []

    @property
    def is_async(self) -> bool:
        return False

    @property
    def supports_expval(self) -> bool:
        return True

    def submit_circuits(self, payloads, **kwargs) -> ExecutionResult:
        circuits = bound_circuits(payloads)
        self.call_log.append((circuits, dict(kwargs)))
        results = [{"label": label, "results": {"expval": 0.5}} for label in circuits]
        return ExecutionResult(results=results)


class _FakeAsyncBackend(FakeSyncBackend):
    """Production-shaped async backend that records how it is driven."""

    def __init__(self, max_retries: int | None = None):
        super().__init__()
        self.max_retries = max_retries
        self._submitted_circuits: dict[str, str] = {}
        self.jobs: list[ExecutionResult] = []
        self.poll_calls: list[dict] = []
        self.fetched: list[ExecutionResult] = []
        self.cancelled_jobs: list[ExecutionResult] = []

    @property
    def is_async(self) -> bool:
        return True

    def submit_circuits(self, payloads, **kwargs) -> ExecutionResult:
        self._submitted_circuits = bound_circuits(payloads)
        job = ExecutionResult(results=None, job_id=f"job-{len(self.jobs) + 1}")
        self.jobs.append(job)
        return job

    def poll_job_status(
        self,
        execution_result,
        loop_until_complete=False,
        verbose=True,
        progress_callback=None,
        cancellation_event=None,
    ):
        self.poll_calls.append(
            {
                "loop_until_complete": loop_until_complete,
                "verbose": verbose,
                "cancellation_event": cancellation_event,
            }
        )
        if progress_callback is not None:
            progress_callback(1, "RUNNING")
        return JobStatus.COMPLETED

    def get_job_results(self, execution_result) -> ExecutionResult:
        self.fetched.append(execution_result)
        return ExecutionResult(
            results=[
                {"label": label, "results": {"00": 100}}
                for label in self._submitted_circuits
            ],
            run_time=2.5,
        )

    def cancel_job(self, execution_result):
        self.cancelled_jobs.append(execution_result)


class _BlockingAsyncBackend(_FakeAsyncBackend):
    """Async backend whose polling blocks until the cancellation event is set."""

    def __init__(self, cancel_error: Exception | None = None):
        super().__init__()
        self.polling = Event()
        self._cancel_error = cancel_error

    def poll_job_status(
        self,
        execution_result,
        loop_until_complete=False,
        verbose=True,
        progress_callback=None,
        cancellation_event=None,
    ):
        self.polling.set()
        cancellation_event.wait(timeout=10)
        raise ExecutionCancelledError("Polling interrupted.")

    def cancel_job(self, execution_result):
        super().cancel_job(execution_result)
        if self._cancel_error is not None:
            raise self._cancel_error


class _SubmissionCounter:
    """Progress emitter that signals once ``target`` programs have first submitted."""

    def __init__(self, target: int):
        self._target = target
        self._count = 0
        self._lock = Lock()
        self.reached = Event()

    def __call__(self, event: ProgressEvent) -> None:
        if event.kind is EventKind.ADVANCE:
            with self._lock:
                self._count += 1
                if self._count >= self._target:
                    self.reached.set()


class _UnknownStatusAsyncBackend(_FakeAsyncBackend):
    """Async backend that reports a valid backend-specific polling status."""

    def poll_job_status(
        self,
        execution_result,
        loop_until_complete=False,
        verbose=True,
        progress_callback=None,
        cancellation_event=None,
    ):
        if progress_callback is not None:
            progress_callback(2, "BACKEND_SPECIFIC_WAIT")
        return JobStatus.COMPLETED


class _TerminalErrorAsyncBackend(_FakeAsyncBackend):
    """Async backend that polls once before raising a terminal job error."""

    def __init__(self, error_type, status: JobStatus) -> None:
        super().__init__()
        self.error_type = error_type
        self.status = status

    def poll_job_status(
        self,
        execution_result,
        loop_until_complete=False,
        verbose=True,
        progress_callback=None,
        cancellation_event=None,
    ):
        del loop_until_complete, verbose, cancellation_event
        if progress_callback is not None:
            progress_callback(1, JobStatus.RUNNING.value)
        raise self.error_type(execution_result.job_id)


def _report_run_times(mocker, backend, outcomes):
    """Make each submission report the next run time, or raise the next error."""
    submit = backend.submit_circuits
    outcomes = iter(outcomes)

    def timed_submit(payloads, **kwargs):
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return replace(submit(payloads, **kwargs), run_time=outcome)

    mocker.patch.object(backend, "submit_circuits", side_effect=timed_submit)


def _make_entry(circuits: dict[str, str], kwargs: dict | None = None) -> _PendingEntry:
    """Create a _PendingEntry with a fresh Future."""
    return _PendingEntry(circuits, kwargs or {}, Future())


def _flush_group_for(batch: _Batch) -> _FlushGroup:
    return _FlushGroup(program_keys=tuple(batch), color="green")


def _run_flush(coord: _BatchCoordinator, batch: _Batch) -> None:
    """Flush ``batch`` on the calling thread, tracked in flight as a real flush is."""
    flush_group = _flush_group_for(batch)
    with coord._in_flight_lock:
        coord._in_flight.append(flush_group)
    coord._do_flush(batch, flush_group)


def _submit_in_background(
    coord: _BatchCoordinator, program_key, circuits: dict[str, str], **kwargs
) -> Future:
    """Run ``coord.submit`` on a worker thread; the future carries its outcome."""
    outcome: Future = Future()

    def _run():
        try:
            outcome.set_result(coord.submit(program_key, circuits, **kwargs))
        except BaseException as exc:
            outcome.set_exception(exc)

    Thread(target=_run, daemon=True).start()
    return outcome


def _batch_registrations(emitted: list[ProgressEvent]) -> list[ProgressEvent]:
    return [
        event
        for event in emitted
        if event.kind is EventKind.REGISTER and event.scope is ProgressScope.BATCH
    ]


_SINGLE_AND_MIXED_HAM_OPS = pytest.mark.parametrize(
    "kwargs_by_program",
    [{"p1": {}}, {"p_ham": {"ham_ops": "Z"}, "p_shots": {}}],
    ids=["single_job", "mixed_ham_ops"],
)


def test_routed_labels_do_not_depend_on_merge_order():
    """Backends seed circuits per label, so the label must not encode position."""
    first = {"a": _make_entry({"c": "qa"}), "b": _make_entry({"c": "qb"})}
    second = {"b": first["b"], "a": first["a"]}

    _, routes_first = _route_batch_circuits(first)
    _, routes_second = _route_batch_circuits(second)

    assert routes_first == routes_second
    assert len(routes_first) == 2


class TestFailFutures:
    def test_sets_exception_on_all_unresolved(self):
        batch: _Batch = {
            "a": _make_entry({"c1": "q"}),
            "b": _make_entry({"c2": "q"}),
        }
        exc = RuntimeError("boom")
        _fail_futures(batch, exc)

        for entry in batch.values():
            with pytest.raises(RuntimeError, match="boom"):
                entry.future.result(timeout=0)

    def test_skips_already_resolved(self):
        batch: _Batch = {"a": _make_entry({"c": "q"})}
        batch["a"].future.set_result("ok")

        # Should not raise — already resolved future is skipped.
        _fail_futures(batch, RuntimeError("boom"))
        assert batch["a"].future.result() == "ok"


def test_flush_group_starts_without_a_job():
    fg = _FlushGroup(program_keys=("prog_a", "prog_b"), color="green", label="expval")
    assert fg.program_keys == ("prog_a", "prog_b")
    assert fg.color == "green"
    assert fg.label == "expval"
    assert fg.execution_result is None


class TestBatchConfig:
    def test_defaults(self):
        cfg = BatchConfig()
        assert cfg.mode is BatchMode.MERGED
        assert cfg.max_batch_size is None
        assert cfg.max_concurrent_programs is None
        assert cfg._sort_programs is False

    def test_sort_programs_true_is_accepted(self):
        cfg = BatchConfig(_sort_programs=True)
        assert cfg._sort_programs is True

    def test_max_batch_size_zero_raises(self):
        with pytest.raises(ValueError, match="max_batch_size must be >= 1"):
            BatchConfig(max_batch_size=0)

    def test_mode_off_with_max_batch_size_raises(self):
        with pytest.raises(ValueError, match="max_batch_size has no effect"):
            BatchConfig(mode=BatchMode.OFF, max_batch_size=10)

    def test_mode_off_with_sort_true_raises(self):
        with pytest.raises(ValueError, match="_sort_programs has no effect"):
            BatchConfig(mode=BatchMode.OFF, _sort_programs=True)

    def test_max_batch_size_one_accepted(self):
        assert BatchConfig(max_batch_size=1).max_batch_size == 1

    @pytest.mark.parametrize(
        "value",
        [
            pytest.param(1, id="one"),
            pytest.param(64, id="sixty_four"),
            pytest.param(-1, id="minus_one_unbounded"),
        ],
    )
    def test_max_concurrent_programs_accepted(self, value):
        cfg = BatchConfig(max_concurrent_programs=value)
        assert cfg.max_concurrent_programs == value

    @pytest.mark.parametrize(
        "value",
        [pytest.param(0, id="zero"), pytest.param(-2, id="other_negative")],
    )
    def test_max_concurrent_programs_rejected(self, value):
        with pytest.raises(ValueError, match="max_concurrent_programs must be >= 1"):
            BatchConfig(max_concurrent_programs=value)

    def test_mode_off_with_max_concurrent_programs_raises(self):
        with pytest.raises(ValueError, match="max_concurrent_programs has no effect"):
            BatchConfig(mode=BatchMode.OFF, max_concurrent_programs=10)


class TestMergeCircuitsAndKwargs:
    """Tests for _BatchCoordinator._merge_circuits_and_kwargs."""

    def test_identical_kwargs_fast_path(self):
        """When all programs share identical kwargs, circuits merge directly."""
        batch: _Batch = {
            "p1": _make_entry({"p1@c1": "q1", "p1@c2": "q2"}, {"shots": 100}),
            "p2": _make_entry({"p2@c1": "q3"}, {"shots": 100}),
        }
        merged, kw = _BatchCoordinator._merge_circuits_and_kwargs(batch)

        assert merged == {"p1@c1": "q1", "p1@c2": "q2", "p2@c1": "q3"}
        assert kw == {"shots": 100}

    def test_different_ham_ops_produces_circuit_ham_map(self):
        """Programs with different ham_ops get reordered with circuit_ham_map."""
        batch: _Batch = {
            "p1": _make_entry({"p1@c1": "q1", "p1@c2": "q2"}, {"ham_ops": "Z0"}),
            "p2": _make_entry({"p2@c1": "q3"}, {"ham_ops": "Z1"}),
        }
        merged, kw = _BatchCoordinator._merge_circuits_and_kwargs(batch)

        assert len(merged) == 3
        assert kw["ham_ops"] == "Z0|Z1"
        assert kw["circuit_ham_map"] == [[0, 2], [2, 3]]

    def test_same_ham_ops_grouped(self):
        """Programs sharing the same ham_ops end up in one contiguous slice."""
        batch: _Batch = {
            "p1": _make_entry({"p1@c1": "q1"}, {"ham_ops": "XX"}),
            "p2": _make_entry({"p2@c1": "q2"}, {"ham_ops": "ZZ"}),
            "p3": _make_entry({"p3@c1": "q3"}, {"ham_ops": "XX"}),
        }
        merged, kw = _BatchCoordinator._merge_circuits_and_kwargs(batch)

        # p1 and p3 share "XX" so they should be contiguous.
        assert kw["ham_ops"] == "XX|ZZ"
        assert kw["circuit_ham_map"] == [[0, 2], [2, 3]]

    def test_different_ham_ops_keep_shared_kwargs(self):
        batch: _Batch = {
            "p1": _make_entry({"p1@c1": "q1"}, {"shots": 100, "ham_ops": "Z0"}),
            "p2": _make_entry({"p2@c1": "q2"}, {"shots": 100, "ham_ops": "Z1"}),
        }
        _, kw = _BatchCoordinator._merge_circuits_and_kwargs(batch)

        assert kw == {
            "shots": 100,
            "ham_ops": "Z0|Z1",
            "circuit_ham_map": [[0, 1], [1, 2]],
        }

    def test_different_ham_ops_with_diverging_other_kwargs_raises(self):
        batch: _Batch = {
            "p1": _make_entry({"p1@c1": "q1"}, {"shots": 100, "ham_ops": "Z0"}),
            "p2": _make_entry({"p2@c1": "q2"}, {"shots": 200, "ham_ops": "Z1"}),
        }
        with pytest.raises(
            ValueError,
            match=exact_match(
                "Cannot merge programs whose kwargs differ in keys other than "
                "'ham_ops'. Submit such programs in separate batches."
            ),
        ):
            _BatchCoordinator._merge_circuits_and_kwargs(batch)


class TestMergeCircuitsAndKwargsShotGroups:
    """Tests for shot_groups behavior in _merge_circuits_and_kwargs.

    When programs in an ensemble use shot_distribution, each program's
    submit_kwargs include a ``shot_groups`` payload whose indices are
    relative to that program's own circuit list.  After merging multiple
    programs, those indices must be re-offset to point into the merged
    circuit list, otherwise the backend will see ranges that don't cover
    every circuit.
    """

    def test_identical_shot_groups_reindexed_per_program(self):
        """Two programs with identical encoded shot_groups must be expanded
        into a merged shot_groups whose ranges cover ALL merged circuits."""
        batch: _Batch = {
            "p1": _make_entry(
                {"p1@c1": "q1", "p1@c2": "q2", "p1@c3": "q3"},
                {"shot_groups": [[0, 3, 100]]},
            ),
            "p2": _make_entry(
                {"p2@c1": "q4", "p2@c2": "q5", "p2@c3": "q6"},
                {"shot_groups": [[0, 3, 100]]},
            ),
        }
        merged, kw = _BatchCoordinator._merge_circuits_and_kwargs(batch)
        assert len(merged) == 6
        # The merged shot_groups must cover all 6 circuits (not just first 3).
        flat = []
        for s, e, shots in kw["shot_groups"]:
            flat.extend([shots] * (e - s))
        assert len(flat) == 6
        assert all(s == 100 for s in flat)

    def test_distinct_shot_groups_per_program_reindexed(self):
        """Programs with different shot allocations get correctly stitched."""
        batch: _Batch = {
            "p1": _make_entry(
                {"p1@c1": "q1", "p1@c2": "q2"},
                {"shot_groups": [[0, 1, 50], [1, 2, 200]]},
            ),
            "p2": _make_entry(
                {"p2@c1": "q3", "p2@c2": "q4"},
                {"shot_groups": [[0, 2, 300]]},
            ),
        }
        merged, kw = _BatchCoordinator._merge_circuits_and_kwargs(batch)
        assert len(merged) == 4
        flat = []
        for s, e, shots in kw["shot_groups"]:
            flat.extend([shots] * (e - s))
        # p1's allocation: [50, 200], p2's: [300, 300] -> merged [50, 200, 300, 300]
        assert flat == [50, 200, 300, 300]

    @pytest.mark.parametrize(
        ("programs", "expected"),
        [
            pytest.param(
                [(2, [[0, 2, 100]])] * 3,
                [[0, 2, 100], [2, 4, 100], [4, 6, 100]],
                id="identical_kwargs",
            ),
            pytest.param(
                [
                    (2, [[0, 1, 50], [1, 2, 200]]),
                    (1, [[0, 1, 300]]),
                    (2, [[0, 2, 400]]),
                ],
                [[0, 1, 50], [1, 2, 200], [2, 3, 300], [3, 5, 400]],
                id="distinct_shot_groups",
            ),
        ],
    )
    def test_shot_groups_are_offset_into_merged_circuit_list(self, programs, expected):
        batch: _Batch = {
            f"p{i}": _make_entry(
                {f"p{i}@c{j}": "q" for j in range(n_circuits)},
                {"shot_groups": shot_groups},
            )
            for i, (n_circuits, shot_groups) in enumerate(programs)
        }
        _, kw = _BatchCoordinator._merge_circuits_and_kwargs(batch)

        assert kw["shot_groups"] == expected

    def test_mixed_with_without_shot_groups_raises(self):
        """Programs that mix shot_groups-set and shot_groups-unset can't merge."""
        batch: _Batch = {
            "p1": _make_entry(
                {"p1@c1": "q1"},
                {"shot_groups": [[0, 1, 100]]},
            ),
            "p2": _make_entry(
                {"p2@c1": "q2"},
                {"shots": 100},  # no shot_groups
            ),
        }
        with pytest.raises(ValueError, match="mix of programs"):
            _BatchCoordinator._merge_circuits_and_kwargs(batch)

    def test_shot_groups_with_diverging_other_kwargs_raises(self):
        """Programs that share shot_groups but differ in any other kwarg
        must raise rather than silently discarding the diverging value."""
        batch: _Batch = {
            "p1": _make_entry(
                {"p1@c1": "q1"},
                {"shots": 100, "shot_groups": [[0, 1, 100]]},
            ),
            "p2": _make_entry(
                {"p2@c1": "q2"},
                {"shots": 200, "shot_groups": [[0, 1, 200]]},
            ),
        }
        with pytest.raises(ValueError, match="keys other than 'shot_groups'"):
            _BatchCoordinator._merge_circuits_and_kwargs(batch)

    def test_shot_groups_with_different_ham_ops_raises(self):
        """Combining shot_groups with heterogeneous ham_ops would require
        reordering shots in lockstep with circuit reordering. Out of scope
        for v1 — must raise a clear error rather than misbehave."""
        batch: _Batch = {
            "p1": _make_entry(
                {"p1@c1": "q1"},
                {"ham_ops": "Z", "shot_groups": [[0, 1, 100]]},
            ),
            "p2": _make_entry(
                {"p2@c1": "q2"},
                {"ham_ops": "X", "shot_groups": [[0, 1, 200]]},
            ),
        }
        with pytest.raises(ValueError, match="shot_groups"):
            _BatchCoordinator._merge_circuits_and_kwargs(batch)


class TestSplitByHamOps:
    """Tests for _BatchCoordinator._split_by_ham_ops."""

    def test_all_with_ham_ops(self):
        batch: _Batch = {
            "p1": _make_entry({}, {"ham_ops": "Z"}),
            "p2": _make_entry({}, {"ham_ops": "X"}),
        }
        result = _BatchCoordinator._split_by_ham_ops(batch)
        assert len(result) == 1
        assert set(result[0].keys()) == {"p1", "p2"}

    def test_all_without_ham_ops(self):
        batch: _Batch = {
            "p1": _make_entry({}, {}),
            "p2": _make_entry({}, {}),
        }
        result = _BatchCoordinator._split_by_ham_ops(batch)
        assert len(result) == 1
        assert set(result[0].keys()) == {"p1", "p2"}

    def test_mixed_splits_into_two(self):
        batch: _Batch = {
            "p1": _make_entry({}, {"ham_ops": "Z"}),
            "p2": _make_entry({}, {}),
            "p3": _make_entry({}, {"ham_ops": "X"}),
        }
        result = _BatchCoordinator._split_by_ham_ops(batch)
        assert len(result) == 2

        with_ham = result[0]
        without_ham = result[1]
        assert set(with_ham.keys()) == {"p1", "p3"}
        assert set(without_ham.keys()) == {"p2"}

    def test_empty_batch(self):
        assert _BatchCoordinator._split_by_ham_ops({}) == []


class TestRegistrationAndBarrier:
    def test_register_and_deregister(self):
        coord = _BatchCoordinator(FakeSyncBackend())
        coord.register_program("a")
        coord.register_program("b")
        assert coord._active_programs == {"a", "b"}

        coord.deregister_program("a")
        assert coord._active_programs == {"b"}

    def test_deregister_unknown_is_safe(self):
        coord = _BatchCoordinator(FakeSyncBackend())
        coord.deregister_program("nonexistent")  # should not raise

    def test_should_flush_when_all_submitted(self):
        coord = _BatchCoordinator(FakeSyncBackend())
        coord.register_program("a")
        coord.register_program("b")

        # One pending — not ready.
        coord._pending["a"] = _make_entry({})
        assert not coord._should_flush()

        # Both pending — ready.
        coord._pending["b"] = _make_entry({})
        assert coord._should_flush()


class TestNWorkersBarrierCap:
    """The ``n_workers`` cap on the barrier predicate keeps the wait-for-all
    barrier satisfiable when ``_active_programs`` exceeds executor capacity."""

    @pytest.mark.parametrize(
        "coordinator_kwargs, active, flush_at",
        [
            pytest.param({}, "abc", 3, id="no_cap_waits_for_every_active"),
            pytest.param(
                {"n_workers": 2}, "abcde", 2, id="cap_fires_below_full_active"
            ),
            pytest.param(
                {"n_workers": 14}, "ab", 2, id="cap_dormant_when_active_below_cap"
            ),
        ],
    )
    def test_flushes_at_min_of_active_and_cap(
        self, coordinator_kwargs, active, flush_at
    ):
        coord = _BatchCoordinator(FakeSyncBackend(), **coordinator_kwargs)
        for key in active:
            coord.register_program(key)
        for key in active[: flush_at - 1]:
            coord._pending[key] = _make_entry({"c": "q"})
        assert not coord._should_flush()
        coord._pending[active[flush_at - 1]] = _make_entry({"c": "q"})
        assert coord._should_flush()

    def test_predicate_collapses_when_active_drops_below_cap(self):
        coord = _BatchCoordinator(FakeSyncBackend(), n_workers=4)
        for key in ("a", "b", "c", "d", "e"):
            coord.register_program(key)
        # 3 pending against active=5, cap=4 → min is 4, not satisfied yet.
        coord._pending["a"] = _make_entry({"c": "q"})
        coord._pending["b"] = _make_entry({"c": "q"})
        coord._pending["c"] = _make_entry({"c": "q"})
        assert not coord._should_flush()
        # Shrink active below cap directly so the predicate, not the
        # deregister side-effect, is what's under test.
        coord._active_programs.discard("d")
        coord._active_programs.discard("e")
        assert coord._should_flush()

    def test_deadlock_repro_programs_exceed_pool(self):
        """Bounded pool + many programs + 1-circuit submits must flush."""
        backend = FakeSyncBackend()
        n_workers = 4
        n_programs = 16
        coord = _BatchCoordinator(
            backend,
            batch_config=BatchConfig(max_batch_size=n_programs),
            n_workers=n_workers,
        )
        for i in range(n_programs):
            coord.register_program(f"p{i}")

        gate = Barrier(n_workers)
        results = {}
        results_lock = Lock()

        def _submit(key):
            gate.wait(timeout=5)
            res, _runtime = coord.submit(key, {"c": "qasm"}, shots=100)
            with results_lock:
                results[key] = res

        threads = [Thread(target=_submit, args=(f"p{i}",)) for i in range(n_workers)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(results) == n_workers
        for i in range(n_workers):
            assert len(results[f"p{i}"]) == 1

    def test_pool_sized_to_programs_yields_one_flush(self):
        """When ``n_workers == n_programs`` the barrier admits a single
        merged backend call — the cloud-merge recipe."""
        backend = FakeSyncBackend()
        n = 64
        coord = _BatchCoordinator(backend, n_workers=n)
        for i in range(n):
            coord.register_program(f"p{i}")

        gate = Barrier(n)
        results: dict[str, list] = {}
        results_lock = Lock()

        def _submit(key):
            gate.wait(timeout=10)
            res, _runtime = coord.submit(key, {"c": "qasm"}, shots=100)
            with results_lock:
                results[key] = res

        threads = [Thread(target=_submit, args=(f"p{i}",)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=20)

        assert len(results) == n
        # Single merged backend call for all 64 programs.
        assert len(backend.submitted) == 1
        assert len(backend.submitted[0]) == n


class TestFlushWithSyncBackend:
    """Integration tests using FakeSyncBackend to verify the full
    submit → barrier → merge → demux → resolve cycle."""

    def test_two_programs_single_flush(self):
        """Two programs submit concurrently; results are demuxed correctly."""
        backend = FakeSyncBackend()
        coord = _BatchCoordinator(backend)
        coord.register_program("p1")
        coord.register_program("p2")

        results = {}
        barrier = Barrier(2)

        def _submit(key, circuits):
            barrier.wait(timeout=5)
            results[key] = coord.submit(key, circuits)

        t1 = Thread(
            target=_submit,
            args=("p1", {"c1": "q1", "c2": "q2"}),
        )
        t2 = Thread(
            target=_submit,
            args=("p2", {"c1": "q3"}),
        )
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        # Both programs should have received their demuxed results.
        p1_results, p1_runtime = results["p1"]
        p2_results, p2_runtime = results["p2"]

        assert len(p1_results) == 2
        assert len(p2_results) == 1
        assert all(r["label"].startswith("c") for r in p1_results)
        assert p2_results[0]["label"] == "c1"

        # Backend should have been called exactly once (merged).
        assert len(backend.submitted) == 1
        assert len(backend.submitted[0]) == 3

    def test_result_metadata_reaches_the_program(self, mocker):
        backend = FakeSyncBackend()
        mocker.patch.object(
            backend,
            "submit_circuits",
            lambda payloads, **kw: ExecutionResult(
                results=[
                    {"label": label, "results": {}, "metadata": {"depth": 3}}
                    for label in payloads
                ]
            ),
        )
        coord = _BatchCoordinator(backend)
        coord.register_program("p1")

        results, _ = coord.submit("p1", {"c1": "q"})

        assert results == [{"label": "c1", "results": {}, "metadata": {"depth": 3}}]

    def test_base_exception_in_flush_fails_futures(self, mocker):
        backend = FakeSyncBackend()
        mocker.patch.object(
            backend,
            "submit_circuits",
            lambda payloads, **kw: ExecutionResult(results=None, job_id="fake"),
        )
        coord = _BatchCoordinator(backend)

        def raise_base(self, *a, **k):
            raise SystemExit("backend died")

        mocker.patch.object(_BatchCoordinator, "_poll_and_get_results", raise_base)

        batch = {"p1": _make_entry({"c1": "q"}, {})}
        _run_flush(coord, batch)

        future = batch["p1"].future
        assert future.done()
        with pytest.raises(SystemExit):
            future.result()

    def test_shutdown_joins_and_clears_flush_threads(self):
        backend = FakeSyncBackend()
        coord = _BatchCoordinator(backend)
        coord.register_program("p1")
        coord.submit("p1", {"c1": "q"})

        assert coord._flush_threads

        coord.shutdown()

        assert coord._flush_threads == []

    def test_three_programs_single_flush(self):
        """Three programs all reach the barrier together."""
        backend = FakeSyncBackend()
        coord = _BatchCoordinator(backend)
        for key in ("a", "b", "c"):
            coord.register_program(key)

        results = {}
        barrier = Barrier(3)

        def _submit(key):
            barrier.wait(timeout=5)
            circuits = {"circ": f"qasm_{key}"}
            results[key] = coord.submit(key, circuits)

        threads = [Thread(target=_submit, args=(k,)) for k in ("a", "b", "c")]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        for key in ("a", "b", "c"):
            r, _ = results[key]
            assert len(r) == 1
            assert r[0]["label"] == "circ"

        assert len(backend.submitted) == 1

    def test_deregister_triggers_flush_for_remaining(self):
        """When a program deregisters, the barrier shrinks and pending
        submissions flush immediately."""
        backend = FakeSyncBackend()
        coord = _BatchCoordinator(backend)
        coord.register_program("p1")
        coord.register_program("p2")

        result_holder = {}
        submitted_event = Event()

        def _submit_p1():
            result_holder["p1"] = coord.submit("p1", {"c1": "q1"})
            submitted_event.set()

        t = Thread(target=_submit_p1)
        t.start()

        # p1 is now blocked waiting for p2. Deregistering p2 should flush.
        time.sleep(0.1)  # Give p1's thread time to submit
        coord.deregister_program("p2")
        t.join(timeout=10)

        assert submitted_event.is_set()
        p1_results, _ = result_holder["p1"]
        assert len(p1_results) == 1

    def test_multiple_flush_rounds(self):
        """Programs go through multiple submit rounds (like VQE iterations)."""
        backend = FakeSyncBackend()
        coord = _BatchCoordinator(backend)
        coord.register_program("p1")
        coord.register_program("p2")

        n_rounds = 3
        all_results = {"p1": [], "p2": []}
        barrier = Barrier(2)

        def _run_rounds(key):
            for i in range(n_rounds):
                barrier.wait(timeout=5)
                circuits = {f"r{i}": f"qasm_{key}_{i}"}
                res, _ = coord.submit(key, circuits)
                all_results[key].append(res)

        t1 = Thread(target=_run_rounds, args=("p1",))
        t2 = Thread(target=_run_rounds, args=("p2",))
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        # Each round should produce exactly one merged backend call.
        assert len(backend.submitted) == n_rounds
        for key in ("p1", "p2"):
            assert len(all_results[key]) == n_rounds
            for i, round_results in enumerate(all_results[key]):
                assert len(round_results) == 1
                assert round_results[0]["label"] == f"r{i}"

    def test_sort_programs_true_produces_deterministic_circuit_order(self):
        """With _sort_programs=True the merged batch is always in key-sorted
        order regardless of which thread reaches the barrier first."""
        # Run the same two-program flush 20 times and confirm that the circuit
        # ordering in the merged backend call is always "p1" circuits before
        # "p2" circuits (sorted keys).
        for _ in range(20):
            backend = FakeSyncBackend()
            coord = _BatchCoordinator(
                backend, batch_config=BatchConfig(_sort_programs=True)
            )
            coord.register_program("p2")  # register in reverse order on purpose
            coord.register_program("p1")

            barrier = Barrier(2)

            def _submit(key):
                barrier.wait(timeout=5)
                coord.submit(key, {"c": f"qasm_{key}"})

            threads = [Thread(target=_submit, args=(k,)) for k in ("p2", "p1")]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10)

            # The merged backend call must always list p1's circuit before p2's.
            assert len(backend.submitted) == 1
            merged_values = list(backend.submitted[0].values())
            p1_pos = merged_values.index("qasm_p1")
            p2_pos = merged_values.index("qasm_p2")
            assert (
                p1_pos < p2_pos
            ), f"Expected p1 before p2 (sorted), got order: {merged_values}"

    @pytest.mark.parametrize(
        ("sort_programs", "expected_order"),
        [(False, ["q2", "q1"]), (True, ["q1", "q2"])],
        ids=["arrival_order", "key_order"],
    )
    def test_merge_order_follows_sort_setting(self, sort_programs, expected_order):
        backend = FakeSyncBackend()
        first_submitted = _SubmissionCounter(1)
        coord = _BatchCoordinator(
            backend,
            progress_emitter=first_submitted,
            batch_config=BatchConfig(_sort_programs=sort_programs),
            preparation_key="preparation",
        )
        coord.register_program("p1")
        coord.register_program("p2")

        early = _submit_in_background(coord, "p2", {"c": "q2"})
        assert first_submitted.reached.wait(timeout=5)
        coord.submit("p1", {"c": "q1"})
        early.result(timeout=5)

        assert [list(batch.values()) for batch in backend.submitted] == [expected_order]


class TestHamOpsSplitting:
    """Tests that mixed ham_ops batches are split into separate backend calls."""

    def test_mixed_batch_produces_two_backend_calls(self):
        """Programs with/without ham_ops are submitted separately."""
        backend = FakeExpvalBackend()
        coord = _BatchCoordinator(backend)
        coord.register_program("expval_prog")
        coord.register_program("shots_prog")

        results = {}
        barrier = Barrier(2)

        def _submit(key, circuits, **kwargs):
            barrier.wait(timeout=5)
            results[key] = coord.submit(key, circuits, **kwargs)

        t1 = Thread(
            target=_submit,
            args=("expval_prog", {"c1": "q1"}),
            kwargs={"ham_ops": "Z0 Z1"},
        )
        t2 = Thread(
            target=_submit,
            args=("shots_prog", {"c1": "q2", "c2": "q3"}),
        )
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        # Two separate backend calls (one for expval, one for shots).
        assert len(backend.call_log) == 2

        # Verify each program got its own results back.
        expval_res, _ = results["expval_prog"]
        shots_res, _ = results["shots_prog"]
        assert len(expval_res) == 1
        assert len(shots_res) == 2

    def test_homogeneous_ham_ops_single_call(self):
        """Programs all having ham_ops produce a single merged call."""
        backend = FakeExpvalBackend()
        coord = _BatchCoordinator(backend)
        coord.register_program("p1")
        coord.register_program("p2")

        results = {}
        barrier = Barrier(2)

        def _submit(key, ham):
            barrier.wait(timeout=5)
            results[key] = coord.submit(key, {"c1": "qasm"}, ham_ops=ham)

        t1 = Thread(target=_submit, args=("p1", "Z0"))
        t2 = Thread(target=_submit, args=("p2", "Z0"))
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        # Same ham_ops → single merged call.
        assert len(backend.call_log) == 1
        _, kw = backend.call_log[0]
        assert kw["ham_ops"] == "Z0"
        assert "circuit_ham_map" not in kw  # fast path, identical kwargs

    def test_different_ham_ops_merged_with_map(self):
        """Programs with different ham_ops get merged with circuit_ham_map."""
        backend = FakeExpvalBackend()
        coord = _BatchCoordinator(backend)
        coord.register_program("p1")
        coord.register_program("p2")

        results = {}
        barrier = Barrier(2)

        def _submit(key, ham):
            barrier.wait(timeout=5)
            results[key] = coord.submit(key, {"c1": "qasm"}, ham_ops=ham)

        t1 = Thread(target=_submit, args=("p1", "Z0"))
        t2 = Thread(target=_submit, args=("p2", "X1"))
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        # Different ham_ops → single call with circuit_ham_map.
        assert len(backend.call_log) == 1
        _, kw = backend.call_log[0]
        assert "Z0" in kw["ham_ops"]
        assert "X1" in kw["ham_ops"]
        assert "circuit_ham_map" in kw


class TestBatchProgress:
    def test_flush_emits_typed_batch_registration_and_finish(self):
        """A merged submission owns one typed batch target lifecycle."""
        emitted: list[ProgressEvent] = []
        backend = FakeSyncBackend()
        coord = _BatchCoordinator(backend, progress_emitter=emitted.append)
        coord.register_program("p1")

        coord.submit("p1", {"c1": "q1"})

        batch_events = [
            event
            for event in emitted
            if event.scope is ProgressScope.BATCH
            or event.progress_key == emitted[-1].progress_key
        ]
        register, finish = batch_events
        assert register.kind is EventKind.REGISTER
        assert register.scope is ProgressScope.BATCH
        assert register.label == "Batch (1 circuit, 1 program)"
        assert register.program_keys == ("p1",)
        assert finish.kind is EventKind.FINISH
        assert finish.progress_key == register.progress_key
        assert finish.terminal_status is TerminalStatus.SUCCESS

    def test_default_no_op_emitter_keeps_standalone_coordinator_quiet(
        self, capsys, caplog
    ):
        backend = FakeSyncBackend()
        coord = _BatchCoordinator(backend)
        coord.register_program("p1")

        coord.submit("p1", {"c1": "q1"})

        assert [list(batch.values()) for batch in backend.submitted] == [["q1"]]
        assert capsys.readouterr() == ("", "")
        assert caplog.records == []

    def test_plural_batch_label(self):
        emitted: list[ProgressEvent] = []
        coord = _BatchCoordinator(FakeSyncBackend(), progress_emitter=emitted.append)
        coord.register_program("p1")
        coord.register_program("p2")

        outcomes = [
            _submit_in_background(coord, "p1", {"c1": "q", "c2": "q"}),
            _submit_in_background(coord, "p2", {"c1": "q"}),
        ]
        for outcome in outcomes:
            outcome.result(timeout=5)

        assert [event.label for event in _batch_registrations(emitted)] == [
            "Batch (3 circuits, 2 programs)"
        ]

    def test_batch_colours_cycle_through_the_palette(self):
        emitted: list[ProgressEvent] = []
        coord = _BatchCoordinator(FakeSyncBackend(), progress_emitter=emitted.append)
        coord.register_program("p1")

        for index in range(len(_BATCH_COLORS) + 1):
            coord.submit("p1", {f"c{index}": "q"})

        assert [event.batch_color for event in _batch_registrations(emitted)] == [
            *_BATCH_COLORS,
            _BATCH_COLORS[0],
        ]

    def test_deregister_with_nothing_pending_starts_no_batch(self):
        emitted: list[ProgressEvent] = []
        backend = FakeSyncBackend()
        coord = _BatchCoordinator(backend, progress_emitter=emitted.append)
        coord.register_program("p1")
        coord.register_program("p2")

        coord.deregister_program("p2")
        coord.submit("p1", {"c1": "q"})

        assert [list(batch.values()) for batch in backend.submitted] == [["q"]]
        assert [event.batch_color for event in _batch_registrations(emitted)] == [
            _BATCH_COLORS[0]
        ]

    @pytest.mark.parametrize(
        ("backend_cls", "job_status"),
        [(FakeSyncBackend, None), (_FakeAsyncBackend, JobStatus.COMPLETED)],
        ids=["sync", "async"],
    )
    def test_successful_batch_finish_reports_job_status(self, backend_cls, job_status):
        emitted: list[ProgressEvent] = []
        coord = _BatchCoordinator(backend_cls(), progress_emitter=emitted.append)
        coord.register_program("p1")

        coord.submit("p1", {"c1": "q"})

        finish = emitted[-1]
        assert finish.kind is EventKind.FINISH
        assert finish.terminal_status is TerminalStatus.SUCCESS
        assert finish.job_status is job_status

    def test_polling_reports_the_backend_retry_limit(self):
        emitted: list[ProgressEvent] = []
        coord = _BatchCoordinator(
            _FakeAsyncBackend(max_retries=7), progress_emitter=emitted.append
        )
        coord.register_program("p1")

        coord.submit("p1", {"c1": "q"})

        polling = [event for event in emitted if event.kind is EventKind.POLLING]
        assert [event.max_retries for event in polling] == [7]

    def test_polling_waits_quietly_and_cancellably_for_the_submitted_job(self):
        cancellation_event = Event()
        backend = _FakeAsyncBackend()
        coord = _BatchCoordinator(backend, cancellation_event=cancellation_event)
        coord.register_program("p1")

        coord.submit("p1", {"c1": "q"})

        assert backend.poll_calls == [
            {
                "loop_until_complete": True,
                "verbose": False,
                "cancellation_event": cancellation_event,
            }
        ]
        assert backend.fetched == backend.jobs

    def test_sequential_flushes_keep_distinct_batch_lifecycles(self):
        emitted: list[ProgressEvent] = []
        coord = _BatchCoordinator(FakeSyncBackend(), progress_emitter=emitted.append)
        coord.register_program("p1")

        for index in range(32):
            coord.submit("p1", {f"c{index}": "qasm"})

        targets = [event.progress_key for event in _batch_registrations(emitted)]
        assert len(targets) == 32
        assert len(set(targets)) == 32

        state = ProgressState()
        state.apply(
            ProgressEvent.register("p1", ProgressScope.PROGRAM, "Program p1", 32)
        )
        for event in emitted:
            state.apply(event)
        assert (
            sum(
                target.scope is ProgressScope.BATCH for target in state.targets.values()
            )
            == 32
        )

    def test_mixed_ham_ops_registers_labelled_batch_keys(self):
        """Sub-batches from ham_ops splitting include labels."""
        emitted: list[ProgressEvent] = []
        backend = FakeExpvalBackend()
        coord = _BatchCoordinator(backend, progress_emitter=emitted.append)
        coord.register_program("p1")
        coord.register_program("p2")

        barrier = Barrier(2)

        def _submit(key, **kwargs):
            barrier.wait(timeout=5)
            coord.submit(key, {"c1": "qasm"}, **kwargs)

        t1 = Thread(target=_submit, args=("p1",), kwargs={"ham_ops": "Z"})
        t2 = Thread(target=_submit, args=("p2",))
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        registrations = _batch_registrations(emitted)
        assert [event.label for event in registrations] == [
            "Batch expval (1 circuit, 1 program)",
            "Batch shots (1 circuit, 1 program)",
        ]
        assert [event.batch_color for event in registrations] == [
            _BATCH_COLORS[0],
            _BATCH_COLORS[0],
        ]

    def test_first_program_submit_advances_preparation_once(self):
        emitted: list[ProgressEvent] = []
        backend = FakeSyncBackend()
        coord = _BatchCoordinator(
            backend,
            progress_emitter=emitted.append,
            preparation_key="preparation",
        )
        coord.register_program("p1")

        coord.submit("p1", {"c1": "q1"})
        coord.submit("p1", {"c2": "q2"})

        prep_events = [
            event for event in emitted if event.progress_key == "preparation"
        ]
        assert prep_events == [ProgressEvent.advance("preparation")]

    def test_polling_normalises_job_status_to_canonical_enum(self):
        emitted: list[ProgressEvent] = []
        backend = _FakeAsyncBackend()
        backend.submit_circuits({"c1": "qasm"})
        coord = _BatchCoordinator(backend, progress_emitter=emitted.append)

        completed = coord._poll_and_get_results(
            ExecutionResult(results=None, job_id="job-123"),
            job_id="job-123",
            batch_progress_key="registered-batch",
        )

        event = emitted[-1]
        assert isinstance(backend, AsyncJobBackend)
        assert completed.results[0]["label"] == "c1"
        assert completed.run_time == 2.5
        assert event.kind is EventKind.POLLING
        assert event.progress_key == "registered-batch"
        assert event.job_status is JobStatus.RUNNING
        assert event.max_retries is None

    def test_backend_specific_polling_status_is_reported_without_aborting(self):
        emitted: list[ProgressEvent] = []
        backend = _UnknownStatusAsyncBackend()
        backend.submit_circuits({"c1": "qasm"})
        coord = _BatchCoordinator(backend, progress_emitter=emitted.append)

        completed = coord._poll_and_get_results(
            ExecutionResult(results=None, job_id="job-123"),
            job_id="job-123",
            batch_progress_key="registered-batch",
        )

        assert completed.results[0]["label"] == "c1"
        assert completed.run_time == 2.5
        assert emitted == [
            ProgressEvent.show(
                "registered-batch",
                "Backend status BACKEND_SPECIFIC_WAIT for job job-123 (attempt 2)",
            )
        ]

    def test_batch_finish_clears_program_membership_through_reducer(self):
        emitted: list[ProgressEvent] = []
        coord = _BatchCoordinator(FakeSyncBackend(), progress_emitter=emitted.append)
        coord.register_program("p1")
        coord.submit("p1", {"c1": "q1"})
        batch_events = [
            event
            for event in emitted
            if event.kind in {EventKind.REGISTER, EventKind.FINISH}
        ]

        state = ProgressState()
        state.apply(
            ProgressEvent.register("p1", ProgressScope.PROGRAM, "Program p1", 1)
        )
        state.apply(batch_events[0])
        assert state.get("p1").batch_color == batch_events[0].batch_color

        state.apply(batch_events[-1])
        assert state.get("p1").batch_color == ""

    def test_malformed_result_finishes_batch_as_failed(self, mocker):
        emitted: list[ProgressEvent] = []
        backend = FakeSyncBackend()
        mocker.patch.object(
            backend,
            "submit_circuits",
            return_value=ExecutionResult(
                results=[{"label": "missing-prefix-separator", "results": {}}]
            ),
        )
        coord = _BatchCoordinator(backend, progress_emitter=emitted.append)
        coord.register_program("p1")

        with pytest.raises(ValueError, match="unknown circuit label"):
            coord.submit("p1", {"c1": "q1"})

        register, finish = emitted
        assert register.kind is EventKind.REGISTER
        assert finish.kind is EventKind.FINISH
        assert finish.progress_key == register.progress_key
        assert finish.terminal_status is TerminalStatus.FAILED
        assert finish.detail.startswith("ValueError:")

    def test_backend_failure_finishes_batch_as_failed(self, mocker):
        emitted: list[ProgressEvent] = []
        backend = FakeSyncBackend()
        mocker.patch.object(
            backend, "submit_circuits", side_effect=RuntimeError("backend failed")
        )
        coord = _BatchCoordinator(backend, progress_emitter=emitted.append)
        coord.register_program("p1")

        with pytest.raises(RuntimeError, match="backend failed"):
            coord.submit("p1", {"c1": "q1"})

        register, finish = emitted
        assert register.kind is EventKind.REGISTER
        assert finish.kind is EventKind.FINISH
        assert finish.progress_key == register.progress_key
        assert finish.terminal_status is TerminalStatus.FAILED
        assert finish.detail == "RuntimeError: backend failed"

    @pytest.mark.parametrize(
        ("error_type", "job_status", "terminal_status"),
        [
            (JobTimedOutError, JobStatus.TIMED_OUT, TerminalStatus.FAILED),
            (JobCancelledError, JobStatus.CANCELLED, TerminalStatus.CANCELLED),
        ],
    )
    def test_terminal_backend_error_preserves_job_status_in_final_state(
        self, error_type, job_status, terminal_status
    ):
        emitted: list[ProgressEvent] = []
        backend = _TerminalErrorAsyncBackend(error_type, job_status)
        coord = _BatchCoordinator(backend, progress_emitter=emitted.append)
        coord.register_program("p1")

        with pytest.raises(error_type):
            coord.submit("p1", {"c1": "q1"})

        register, polling, finish = emitted
        assert polling.job_status is JobStatus.RUNNING
        assert finish.terminal_status is terminal_status
        assert finish.job_status is job_status

        state = ProgressState()
        state.apply(
            ProgressEvent.register("p1", ProgressScope.PROGRAM, "Program p1", 1)
        )
        state.apply(register)
        state.apply(polling)
        state.apply(finish)
        target = state.get(register.progress_key)
        assert target.terminal_status is terminal_status
        assert target.job_status is job_status

    def test_cancellation_finishes_batch_as_cancelled(self, mocker):
        emitted: list[ProgressEvent] = []
        cancellation_event = Event()
        backend = FakeSyncBackend()

        def _cancel_during_submit(payloads, **kwargs):
            cancellation_event.set()
            return ExecutionResult(results=[{"label": "0", "results": {}}])

        mocker.patch.object(
            backend, "submit_circuits", side_effect=_cancel_during_submit
        )
        coord = _BatchCoordinator(
            backend,
            progress_emitter=emitted.append,
            cancellation_event=cancellation_event,
        )
        coord.register_program("p1")

        with pytest.raises(ExecutionCancelledError):
            coord.submit("p1", {"c1": "q1"})

        register, finish = emitted
        assert register.kind is EventKind.REGISTER
        assert finish.kind is EventKind.FINISH
        assert finish.progress_key == register.progress_key
        assert finish.terminal_status is TerminalStatus.CANCELLED

    def test_success_follows_parsing_but_precedes_future_release(self):
        entry = _make_entry({"c1": "qasm"})
        batch = {"p1": entry}
        future_done_at_success: list[bool] = []

        def _record(event: ProgressEvent) -> None:
            if (
                event.kind is EventKind.FINISH
                and event.terminal_status is TerminalStatus.SUCCESS
            ):
                future_done_at_success.append(entry.future.done())

        coord = _BatchCoordinator(_FakeAsyncBackend(), progress_emitter=_record)

        coord._submit_sub_batch(batch, _flush_group_for(batch))

        assert future_done_at_success == [False]
        _, run_time = entry.future.result(timeout=0)
        assert run_time == 2.5


class TestCancellation:
    def test_cancel_rejects_new_submissions(self):
        coord = _BatchCoordinator(FakeSyncBackend())
        coord.register_program("p1")
        coord.cancel()

        with pytest.raises(ExecutionCancelledError):
            coord.submit("p1", {"c": "q"})

    def test_cancel_resolves_pending_futures(self):
        coord = _BatchCoordinator(FakeSyncBackend())
        coord.register_program("p1")
        coord.register_program("p2")

        # Add a pending entry that hasn't flushed yet.
        entry = _make_entry({"c": "q"})
        coord._pending["p1"] = entry

        coord.cancel()

        with pytest.raises(ExecutionCancelledError):
            entry.future.result(timeout=0)

    def test_shutdown_clears_active_programs(self):
        coord = _BatchCoordinator(FakeSyncBackend())
        coord.register_program("p1")
        coord.shutdown()

        assert len(coord._active_programs) == 0

    def test_flush_after_cancel_resolves_with_error(self):
        """If cancel is called while a flush is in progress, futures get
        the cancellation error."""
        backend = FakeSyncBackend()
        coord = _BatchCoordinator(backend)
        coord.register_program("p1")
        coord.register_program("p2")

        # Manually trigger a cancel before p2 submits.
        result_holder = {}
        error_holder = {}
        barrier = Barrier(2)

        def _submit_p1():
            barrier.wait(timeout=5)
            try:
                result_holder["p1"] = coord.submit("p1", {"c1": "q"})
            except ExecutionCancelledError as e:
                error_holder["p1"] = e

        t = Thread(target=_submit_p1)
        t.start()

        barrier.wait(timeout=5)
        time.sleep(0.1)
        coord.cancel()
        t.join(timeout=10)

        assert "p1" in error_holder

    def test_flush_started_after_cancellation_never_reaches_backend(self):
        cancellation_event = Event()
        backend = FakeSyncBackend()
        coord = _BatchCoordinator(backend, cancellation_event=cancellation_event)
        batch = {"p1": _make_entry({"c1": "q"})}

        cancellation_event.set()
        _run_flush(coord, batch)

        with pytest.raises(ExecutionCancelledError):
            batch["p1"].future.result(timeout=0)
        assert backend.submitted == []

    @_SINGLE_AND_MIXED_HAM_OPS
    def test_cancel_cancels_the_job_being_polled(self, kwargs_by_program):
        backend = _BlockingAsyncBackend()
        coord = _BatchCoordinator(backend)
        for key in kwargs_by_program:
            coord.register_program(key)

        outcomes = [
            _submit_in_background(coord, key, {"c": "q"}, **kwargs)
            for key, kwargs in kwargs_by_program.items()
        ]
        assert backend.polling.wait(timeout=5)
        coord.cancel()

        for outcome in outcomes:
            with pytest.raises(ExecutionCancelledError):
                outcome.result(timeout=5)
        assert [job.job_id for job in backend.cancelled_jobs] == ["job-1"]

    @_SINGLE_AND_MIXED_HAM_OPS
    def test_cancel_after_finished_flush_cancels_no_job(self, kwargs_by_program):
        backend = _FakeAsyncBackend()
        coord = _BatchCoordinator(backend)
        batch = {
            key: _make_entry({"c": "q"}, kwargs)
            for key, kwargs in kwargs_by_program.items()
        }
        _run_flush(coord, batch)

        coord.cancel()

        assert len(backend.jobs) == len(kwargs_by_program)
        assert backend.cancelled_jobs == []

    def test_cancel_job_failure_still_fails_every_waiting_program(self):
        backend = _BlockingAsyncBackend(cancel_error=RuntimeError("cancel refused"))
        both_submitted = _SubmissionCounter(2)
        coord = _BatchCoordinator(
            backend,
            progress_emitter=both_submitted,
            batch_config=BatchConfig(max_batch_size=2),
            preparation_key="preparation",
        )
        for key in ("a", "b", "c"):
            coord.register_program(key)

        in_flight = _submit_in_background(coord, "a", {"c1": "q", "c2": "q"})
        assert backend.polling.wait(timeout=5)
        pending = _submit_in_background(coord, "b", {"c1": "q"})
        assert both_submitted.reached.wait(timeout=5)
        coord.cancel()

        for outcome in (in_flight, pending):
            with pytest.raises(ExecutionCancelledError):
                outcome.result(timeout=5)
        assert [job.job_id for job in backend.cancelled_jobs] == ["job-1"]


def test_partial_subbatch_failure_keeps_the_successful_share(mocker):
    """Sub-batch 0 succeeds and sub-batch 1 raises: sub-batch 0's program still
    receives its run time, and only sub-batch 1's program fails."""
    backend = FakeSyncBackend()
    _report_run_times(mocker, backend, [7.5, RuntimeError("second fails")])
    coord = _BatchCoordinator(backend)
    batch = {
        "p_with_ham": _make_entry({"c1": "q"}, {"ham_ops": "Z"}),
        "p_no_ham": _make_entry({"c2": "q"}, {}),
    }
    _run_flush(coord, batch)

    _, run_time = batch["p_with_ham"].future.result(timeout=0)
    assert run_time == 7.5
    with pytest.raises(RuntimeError, match="second fails"):
        batch["p_no_ham"].future.result(timeout=0)


class TestProxyBackend:
    def test_delegates_properties(self):
        real = FakeSyncBackend(shots=200)
        coord = _BatchCoordinator(real)
        proxy = _ProxyBackend(real, coord, "prog_1")

        assert proxy.shots == 200
        assert proxy.supports_expval == real.supports_expval
        assert proxy.is_async is False
        assert proxy.max_retries == 0

    def test_never_defers_parameter_binding(self):
        """The proxy submits a bound label -> qasm mapping, so it must report
        resolves_parameters False even when the real backend resolves them —
        otherwise the pipeline emits parametric payloads the proxy cannot flatten."""
        real = FakeSyncBackend(shots=200, resolves_parameters=True)
        coord = _BatchCoordinator(real)
        proxy = _ProxyBackend(real, coord, "prog_1")

        assert real.resolves_parameters is True
        assert proxy.resolves_parameters is False

    def test_proxy_integrates_with_coordinator_barrier(self, mocker):
        """Two proxies submit through the coordinator and each gets its own
        results and an even share of the merged job's run time."""
        backend = FakeSyncBackend()
        _report_run_times(mocker, backend, [6.0])
        coord = _BatchCoordinator(backend)
        coord.register_program("p1")
        coord.register_program("p2")

        proxy1 = _ProxyBackend(backend, coord, "p1")
        proxy2 = _ProxyBackend(backend, coord, "p2")

        results = {}
        barrier = Barrier(2)

        def _submit(proxy, key):
            barrier.wait(timeout=5)
            results[key] = proxy.submit_circuits({f"c_{key}": f"qasm_{key}"})

        t1 = Thread(target=_submit, args=(proxy1, "p1"))
        t2 = Thread(target=_submit, args=(proxy2, "p2"))
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        # Each proxy gets only its own results.
        assert len(results["p1"].results) == 1
        assert results["p1"].results[0]["label"] == "c_p1"
        assert len(results["p2"].results) == 1
        assert results["p2"].results[0]["label"] == "c_p2"
        assert results["p1"].run_time == results["p2"].run_time == 3.0

        # Single merged backend call.
        assert len(backend.submitted) == 1

    def test_demux_is_independent_of_program_key_and_tag_delimiters(self):
        backend = FakeSyncBackend()
        coord = _BatchCoordinator(backend)
        keys = ("program@one", "program@two")
        tags = {keys[0]: "tag@alpha", keys[1]: "tag@beta"}
        for key in keys:
            coord.register_program(key)

        proxies = {key: _ProxyBackend(backend, coord, key) for key in keys}
        results = {}
        barrier = Barrier(2)

        def _submit(key):
            barrier.wait(timeout=5)
            results[key] = proxies[key].submit_circuits({tags[key]: f"qasm_{key}"})

        threads = [Thread(target=_submit, args=(key,)) for key in keys]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

        for key in keys:
            assert results[key].results == [
                {"label": tags[key], "results": {"00": 100}}
            ]

    def test_little_endian_bitstrings_delegated(self):
        """little_endian_bitstrings is delegated to the real backend."""
        real = FakeSyncBackend()
        real.little_endian_bitstrings = True  # type: ignore[attr-defined]
        coord = _BatchCoordinator(real)
        proxy = _ProxyBackend(real, coord, "p")
        assert proxy.little_endian_bitstrings is True


class TestMaxBatchSize:
    def test_flush_triggered_by_circuit_limit(self):
        """Pending circuits reaching the limit triggers a flush even when
        not all programs have submitted."""
        coord = _BatchCoordinator(
            FakeSyncBackend(), batch_config=BatchConfig(max_batch_size=3)
        )
        coord.register_program("a")
        coord.register_program("b")
        coord.register_program("c")

        # Two programs with combined 3 circuits should trigger flush.
        coord._pending["a"] = _make_entry({"a@c1": "q", "a@c2": "q"})
        coord._pending["b"] = _make_entry({"b@c1": "q"})
        assert coord._should_flush()

    def test_no_flush_below_limit(self):
        """Below the circuit limit and not all submitted → no flush."""
        coord = _BatchCoordinator(
            FakeSyncBackend(), batch_config=BatchConfig(max_batch_size=5)
        )
        coord.register_program("a")
        coord.register_program("b")
        coord.register_program("c")

        coord._pending["a"] = _make_entry({"a@c1": "q", "a@c2": "q"})
        assert not coord._should_flush()

    def test_barrier_still_works_below_limit(self):
        """All programs submitted but below limit → still flushes (barrier)."""
        coord = _BatchCoordinator(
            FakeSyncBackend(), batch_config=BatchConfig(max_batch_size=100)
        )
        coord.register_program("a")
        coord.register_program("b")

        coord._pending["a"] = _make_entry({"a@c1": "q"})
        coord._pending["b"] = _make_entry({"b@c1": "q"})
        assert coord._should_flush()

    def test_partial_flush_integration(self):
        """Threaded: A+B flush early via limit, C flushes after A/B deregister."""
        backend = FakeSyncBackend()
        coord = _BatchCoordinator(backend, batch_config=BatchConfig(max_batch_size=2))
        coord.register_program("a")
        coord.register_program("b")
        coord.register_program("c")

        results = {}
        ab_barrier = Barrier(2)
        ab_done = Event()

        def _submit(key, circuits):
            results[key] = coord.submit(key, circuits)

        def _submit_ab(key, circuits):
            ab_barrier.wait(timeout=5)
            _submit(key, circuits)
            ab_done.set()

        t_a = Thread(
            target=_submit_ab,
            args=("a", {"c1": "q"}),
        )
        t_b = Thread(
            target=_submit_ab,
            args=("b", {"c1": "q"}),
        )
        t_a.start()
        t_b.start()
        t_a.join(timeout=10)
        t_b.join(timeout=10)

        # A+B should have flushed (2 circuits == limit).
        assert "a" in results
        assert "b" in results

        # Deregister a and b so c can flush on its own.
        coord.deregister_program("a")
        coord.deregister_program("b")

        t_c = Thread(
            target=_submit,
            args=("c", {"c1": "q"}),
        )
        t_c.start()
        t_c.join(timeout=10)

        assert "c" in results
        # Two backend calls: one for A+B, one for C.
        assert len(backend.submitted) == 2

    def test_single_program_exceeds_limit(self):
        """A single program submitting more circuits than the limit still works."""
        backend = FakeSyncBackend()
        coord = _BatchCoordinator(backend, batch_config=BatchConfig(max_batch_size=2))
        coord.register_program("p1")

        # Single program: barrier triggers immediately regardless of limit.
        result = coord.submit(
            "p1",
            {"c1": "q", "c2": "q", "c3": "q"},
        )
        assert len(result[0]) == 3
        assert len(backend.submitted) == 1

    def test_max_batch_size_none_default(self):
        """None preserves the wait-for-all barrier behaviour."""
        coord = _BatchCoordinator(FakeSyncBackend(), batch_config=BatchConfig())
        coord.register_program("a")
        coord.register_program("b")

        # Only one submitted → not all → should not flush.
        coord._pending["a"] = _make_entry({"a@c1": "q", "a@c2": "q", "a@c3": "q"})
        assert not coord._should_flush()

        # Both submitted → should flush.
        coord._pending["b"] = _make_entry({"b@c1": "q"})
        assert coord._should_flush()

    def test_pending_circuit_count(self):
        """_pending_circuit_count correctly sums circuits."""
        coord = _BatchCoordinator(FakeSyncBackend())
        coord._pending["a"] = _make_entry({"c1": "q", "c2": "q"})
        coord._pending["b"] = _make_entry({"c3": "q"})
        assert coord._pending_circuit_count() == 3
