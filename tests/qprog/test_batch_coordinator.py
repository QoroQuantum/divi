# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for _BatchCoordinator, _ProxyBackend, and related helpers."""

import time
from concurrent.futures import Future
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
    _Batch,
    _BatchCoordinator,
    _fail_futures,
    _FlushGroup,
    _PendingEntry,
    _ProxyBackend,
)
from divi.reporting._events import (
    EventKind,
    ProgressEvent,
    ProgressScope,
    TerminalStatus,
)
from divi.reporting._state import ProgressState


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
    """Production-shaped async backend for typed polling-event tests."""

    def __init__(self):
        super().__init__()
        self._submitted_circuits: dict[str, str] = {}

    @property
    def is_async(self) -> bool:
        return True

    def submit_circuits(self, payloads, **kwargs) -> ExecutionResult:
        self._submitted_circuits = bound_circuits(payloads)
        return ExecutionResult(results=None, job_id="job-123")

    def poll_job_status(
        self,
        execution_result,
        loop_until_complete=False,
        on_complete=None,
        verbose=True,
        progress_callback=None,
        cancellation_event=None,
    ):
        if progress_callback is not None:
            progress_callback(1, "RUNNING")
        if on_complete is not None:
            on_complete({"run_time": 2.5})
        return JobStatus.COMPLETED

    def get_job_results(self, execution_result) -> ExecutionResult:
        return ExecutionResult(
            results=[
                {"label": label, "results": {"00": 100}}
                for label in self._submitted_circuits
            ]
        )

    def cancel_job(self, execution_result):
        return None


class _UnknownStatusAsyncBackend(_FakeAsyncBackend):
    """Async backend that reports a valid backend-specific polling status."""

    def poll_job_status(
        self,
        execution_result,
        loop_until_complete=False,
        on_complete=None,
        verbose=True,
        progress_callback=None,
        cancellation_event=None,
    ):
        if progress_callback is not None:
            progress_callback(2, "BACKEND_SPECIFIC_WAIT")
        if on_complete is not None:
            on_complete({"run_time": 2.5})
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
        on_complete=None,
        verbose=True,
        progress_callback=None,
        cancellation_event=None,
    ):
        del loop_until_complete, verbose, cancellation_event
        if progress_callback is not None:
            progress_callback(1, JobStatus.RUNNING.value)
        if on_complete is not None:
            on_complete({"run_time": 2.5, "status": self.status.value})
        raise self.error_type(execution_result.job_id)


def _make_entry(circuits: dict[str, str], kwargs: dict | None = None) -> _PendingEntry:
    """Create a _PendingEntry with a fresh Future."""
    return _PendingEntry(circuits, kwargs or {}, Future())


class TestPendingEntry:
    def test_named_access(self):
        entry = _make_entry({"t": "qasm"}, {"ham_ops": "Z"})
        assert entry.circuits == {"t": "qasm"}
        assert entry.kwargs == {"ham_ops": "Z"}
        assert isinstance(entry.future, Future)

    def test_unpacking(self):
        entry = _make_entry({"t": "qasm"})
        circuits, kwargs, future = entry
        assert circuits == {"t": "qasm"}
        assert kwargs == {}
        assert isinstance(future, Future)


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


def test_program_keys_from_futures():
    fg = _FlushGroup(
        futures={"prog_a": Future(), "prog_b": Future()},
        color="green",
        label="expval",
    )
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

    def test_max_concurrent_programs_accepted(self):
        cfg = BatchConfig(max_concurrent_programs=64)
        assert cfg.max_concurrent_programs == 64

    def test_max_concurrent_programs_zero_raises(self):
        with pytest.raises(ValueError, match="max_concurrent_programs must be >= 1"):
            BatchConfig(max_concurrent_programs=0)

    def test_max_concurrent_programs_negative_one_accepted(self):
        cfg = BatchConfig(max_concurrent_programs=-1)
        assert cfg.max_concurrent_programs == -1

    def test_max_concurrent_programs_other_negatives_raise(self):
        with pytest.raises(ValueError, match="max_concurrent_programs must be >= 1"):
            BatchConfig(max_concurrent_programs=-2)

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

    def test_no_cap_waits_for_every_active(self):
        coord = _BatchCoordinator(FakeSyncBackend())
        for key in ("a", "b", "c"):
            coord.register_program(key)
        coord._pending["a"] = _make_entry({"c": "q"})
        coord._pending["b"] = _make_entry({"c": "q"})
        assert not coord._should_flush()
        coord._pending["c"] = _make_entry({"c": "q"})
        assert coord._should_flush()

    def test_cap_fires_below_full_active(self):
        coord = _BatchCoordinator(FakeSyncBackend(), n_workers=2)
        for key in ("a", "b", "c", "d", "e"):
            coord.register_program(key)
        coord._pending["a"] = _make_entry({"c": "q"})
        assert not coord._should_flush()
        coord._pending["b"] = _make_entry({"c": "q"})
        assert coord._should_flush()

    def test_cap_dormant_when_active_smaller_than_cap(self):
        coord = _BatchCoordinator(FakeSyncBackend(), n_workers=14)
        coord.register_program("a")
        coord.register_program("b")
        coord._pending["a"] = _make_entry({"c": "q"})
        assert not coord._should_flush()
        coord._pending["b"] = _make_entry({"c": "q"})
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
        flush_group = _FlushGroup(
            futures={k: e.future for k, e in batch.items()}, color="green"
        )
        with coord._in_flight_lock:
            coord._in_flight.append(flush_group)

        coord._do_flush(batch, flush_group)

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

    def test_sort_programs_false_can_produce_arrival_order(self):
        """With _sort_programs=False (default) the batch is flushed in arrival
        order, so a single-threaded submission preserves insertion sequence."""
        backend = FakeSyncBackend()
        coord = _BatchCoordinator(
            backend, batch_config=BatchConfig(_sort_programs=False)
        )
        # Register and immediately submit sequentially (no concurrency) so
        # the insertion order is deterministic: "p2" then "p1".
        coord.register_program("p2")
        coord.register_program("p1")

        futures = {}
        with coord._lock:
            futures["p2"] = Future()
            coord._pending["p2"] = _PendingEntry({"c": "q2"}, {}, futures["p2"])
            futures["p1"] = Future()
            coord._pending["p1"] = _PendingEntry({"c": "q1"}, {}, futures["p1"])
            coord._trigger_flush()

        # Collect results so the flush thread can finish.
        for f in futures.values():
            f.result(timeout=5)

        assert len(backend.submitted) == 1
        merged_values = list(backend.submitted[0].values())
        p2_pos = merged_values.index("q2")
        p1_pos = merged_values.index("q1")
        # With _sort_programs=False the insertion order (p2, then p1) is preserved.
        assert (
            p2_pos < p1_pos
        ), f"Expected p2 before p1 (arrival order), got: {merged_values}"


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

    def test_default_no_op_emitter_keeps_standalone_coordinator_quiet(self):
        backend = FakeSyncBackend()
        coord = _BatchCoordinator(backend)
        coord.register_program("p1")
        coord.submit("p1", {"c1": "q1"})

    def test_batch_registration_color_cycles(self):
        """Each flush group gets the next color in the cycle."""
        emitted: list[ProgressEvent] = []
        backend = FakeSyncBackend()
        coord = _BatchCoordinator(backend, progress_emitter=emitted.append)

        coord.register_program("p1")
        coord.submit("p1", {"c1": "q1"})
        coord.submit("p1", {"c2": "q2"})

        colors = [
            event.batch_color
            for event in emitted
            if event.kind is EventKind.REGISTER and event.scope is ProgressScope.BATCH
        ]
        assert colors[0] != colors[1]

    def test_sequential_flushes_keep_distinct_batch_lifecycles(self):
        emitted: list[ProgressEvent] = []
        coord = _BatchCoordinator(FakeSyncBackend(), progress_emitter=emitted.append)
        coord.register_program("p1")

        for index in range(32):
            coord.submit("p1", {f"c{index}": "qasm"})

        registrations = [
            event
            for event in emitted
            if event.kind is EventKind.REGISTER and event.scope is ProgressScope.BATCH
        ]
        targets = [event.progress_key for event in registrations]
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

        labels = {
            event.label
            for event in emitted
            if event.kind is EventKind.REGISTER and event.scope is ProgressScope.BATCH
        }
        assert labels == {
            "Batch expval (1 circuit, 1 program)",
            "Batch shots (1 circuit, 1 program)",
        }

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

        results, runtime = coord._poll_and_get_results(
            ExecutionResult(results=None, job_id="job-123"),
            batch_progress_key="registered-batch",
        )

        event = emitted[-1]
        assert isinstance(backend, AsyncJobBackend)
        assert results[0]["label"] == "c1"
        assert runtime == 2.5
        assert event.kind is EventKind.POLLING
        assert event.progress_key == "registered-batch"
        assert event.job_status is JobStatus.RUNNING
        assert event.max_retries is None

    def test_backend_specific_polling_status_is_reported_without_aborting(self):
        emitted: list[ProgressEvent] = []
        backend = _UnknownStatusAsyncBackend()
        backend.submit_circuits({"c1": "qasm"})
        coord = _BatchCoordinator(backend, progress_emitter=emitted.append)

        results, runtime = coord._poll_and_get_results(
            ExecutionResult(results=None, job_id="job-123"),
            batch_progress_key="registered-batch",
        )

        assert results[0]["label"] == "c1"
        assert runtime == 2.5
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

    def test_success_follows_parsing_and_accounting_but_precedes_future_release(self):
        entry = _make_entry({"c1": "qasm"})
        flush_group = _FlushGroup({"p1": entry.future}, "cyan")
        observations: list[tuple[float, bool]] = []
        coord: _BatchCoordinator

        def _record(event: ProgressEvent) -> None:
            if (
                event.kind is EventKind.FINISH
                and event.terminal_status is TerminalStatus.SUCCESS
            ):
                observations.append((coord.total_runtime, entry.future.done()))

        coord = _BatchCoordinator(_FakeAsyncBackend(), progress_emitter=_record)

        runtime = coord._submit_sub_batch({"p1": entry}, flush_group)

        assert runtime == 2.5
        assert observations == [(2.5, False)]
        assert entry.future.done()


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


class TestTotalRuntime:
    def test_runtime_zero_for_sync_backend(self):
        """Sync backends report no runtime (no polling)."""
        backend = FakeSyncBackend()
        coord = _BatchCoordinator(backend)
        coord.register_program("p1")
        coord.submit("p1", {"c1": "q"})

        # Sync backend → no runtime tracking.
        assert coord.total_runtime == 0.0

    def test_partial_subbatch_failure_preserves_credit(self, mocker):
        """Sub-batch 0 succeeds, sub-batch 1 raises → coordinator keeps the
        credit from sub-batch 0."""
        backend = FakeSyncBackend()
        # Force the async branch so _submit_sub_batch's runtime
        # accumulation runs (sync branch always reports runtime=0).
        mocker.patch.object(
            backend,
            "submit_circuits",
            lambda payloads, **kw: ExecutionResult(results=None, job_id="fake"),
        )
        coord = _BatchCoordinator(backend)

        batch = {
            "p_with_ham": _make_entry({"c1": "q"}, {"ham_ops": "Z"}),
            "p_no_ham": _make_entry({"c2": "q"}, {}),
        }

        poll_calls = {"n": 0}

        def fake_poll(
            self,
            execution_result,
            batch_progress_key,
        ):
            poll_calls["n"] += 1
            if poll_calls["n"] == 1:
                return [{"label": "0", "results": {}}], 7.5
            raise RuntimeError("second sub-batch fails")

        mocker.patch.object(_BatchCoordinator, "_poll_and_get_results", fake_poll)

        flush_group = _FlushGroup(
            futures={k: e.future for k, e in batch.items()}, color="green"
        )
        with coord._in_flight_lock:
            coord._in_flight.append(flush_group)

        coord._do_flush(batch, flush_group)

        assert coord.total_runtime == 7.5

    def test_runtime_credited_before_futures_resolved(self, mocker):
        """The flush runs on a daemon thread, and resolving a program's future
        unblocks it — which lets the ensemble's join() read ``total_runtime``.
        So the credit must land *before* the futures resolve; otherwise that
        read races the credit and can drop the flush's runtime.
        """
        backend = FakeSyncBackend()
        mocker.patch.object(
            backend,
            "submit_circuits",
            lambda payloads, **kw: ExecutionResult(results=None, job_id="fake"),
        )
        coord = _BatchCoordinator(backend)

        runtime_seen_at_resolution = []

        class _RecordingFuture(Future):
            def set_result(self, result):
                runtime_seen_at_resolution.append(coord.total_runtime)
                super().set_result(result)

        batch = {
            "p1": _PendingEntry({"c1": "q"}, {}, _RecordingFuture()),
            "p2": _PendingEntry({"c2": "q"}, {}, _RecordingFuture()),
        }

        def fake_poll(
            self,
            execution_result,
            batch_progress_key,
        ):
            return [], 6.0

        mocker.patch.object(_BatchCoordinator, "_poll_and_get_results", fake_poll)

        flush_group = _FlushGroup(
            futures={k: e.future for k, e in batch.items()}, color="green"
        )
        with coord._in_flight_lock:
            coord._in_flight.append(flush_group)

        coord._do_flush(batch, flush_group)

        # Both futures observed the full runtime already credited when they
        # resolved — never 0.0 (which is what a resolve-then-credit order
        # would record).
        assert runtime_seen_at_resolution == [6.0, 6.0]
        assert coord.total_runtime == 6.0


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

    def test_proxy_integrates_with_coordinator_barrier(self):
        """Two proxies submit through the coordinator and results are correct."""
        backend = FakeSyncBackend()
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
