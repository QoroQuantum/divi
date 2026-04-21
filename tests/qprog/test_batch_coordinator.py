# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for _BatchCoordinator, _ProxyBackend, and related helpers."""

from concurrent.futures import Future
from queue import Queue
from threading import Barrier, Event, Thread

import pytest

from divi.backends import CircuitRunner, ExecutionResult
from divi.exceptions import ExecutionCancelledError
from divi.qprog._batch_coordinator import (
    _TAG_SEP,
    BatchConfig,
    BatchMode,
    _Batch,
    _BatchCoordinator,
    _fail_futures,
    _FlushGroup,
    _PendingEntry,
    _ProxyBackend,
)


class FakeSyncBackend(CircuitRunner):
    """Minimal synchronous backend that echoes circuit labels as results."""

    def __init__(self, shots: int = 100):
        super().__init__(shots=shots)
        self.submitted: list[dict[str, str]] = []

    @property
    def is_async(self) -> bool:
        return False

    @property
    def supports_expval(self) -> bool:
        return False

    def submit_circuits(self, circuits, **kwargs) -> ExecutionResult:
        self.submitted.append(dict(circuits))
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

    def submit_circuits(self, circuits, **kwargs) -> ExecutionResult:
        self.call_log.append((dict(circuits), dict(kwargs)))
        results = [{"label": label, "results": {"expval": 0.5}} for label in circuits]
        return ExecutionResult(results=results)


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


class TestFlushGroup:
    def test_program_keys_from_futures(self):
        fg = _FlushGroup(
            futures={"prog_a": Future(), "prog_b": Future()},
            color="green",
            label="expval",
        )
        assert fg.program_keys == {"prog_a", "prog_b"}
        assert fg.color == "green"
        assert fg.label == "expval"
        assert fg.execution_result is None


class TestBatchConfig:
    def test_defaults(self):
        cfg = BatchConfig()
        assert cfg.mode is BatchMode.MERGED
        assert cfg.max_batch_size is None
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
            args=("p1", {f"p1{_TAG_SEP}c1": "q1", f"p1{_TAG_SEP}c2": "q2"}),
        )
        t2 = Thread(
            target=_submit,
            args=("p2", {f"p2{_TAG_SEP}c1": "q3"}),
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
            circuits = {f"{key}{_TAG_SEP}circ": f"qasm_{key}"}
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
            result_holder["p1"] = coord.submit("p1", {f"p1{_TAG_SEP}c1": "q1"})
            submitted_event.set()

        t = Thread(target=_submit_p1)
        t.start()

        # p1 is now blocked waiting for p2. Deregistering p2 should flush.
        import time

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
                circuits = {f"{key}{_TAG_SEP}r{i}": f"qasm_{key}_{i}"}
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
                coord.submit(key, {f"{key}{_TAG_SEP}c": f"qasm_{key}"})

            threads = [Thread(target=_submit, args=(k,)) for k in ("p2", "p1")]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10)

            # The merged backend call must always list p1's circuit before p2's.
            assert len(backend.submitted) == 1
            merged_keys = list(backend.submitted[0].keys())
            p1_pos = next(i for i, k in enumerate(merged_keys) if k.startswith("p1"))
            p2_pos = next(i for i, k in enumerate(merged_keys) if k.startswith("p2"))
            assert (
                p1_pos < p2_pos
            ), f"Expected p1 before p2 (sorted), got order: {merged_keys}"

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
            coord._pending["p2"] = _PendingEntry(
                {f"p2{_TAG_SEP}c": "q2"}, {}, futures["p2"]
            )
            futures["p1"] = Future()
            coord._pending["p1"] = _PendingEntry(
                {f"p1{_TAG_SEP}c": "q1"}, {}, futures["p1"]
            )
            coord._trigger_flush()

        # Collect results so the flush thread can finish.
        for f in futures.values():
            f.result(timeout=5)

        assert len(backend.submitted) == 1
        merged_keys = list(backend.submitted[0].keys())
        p2_pos = next(i for i, k in enumerate(merged_keys) if k.startswith("p2"))
        p1_pos = next(i for i, k in enumerate(merged_keys) if k.startswith("p1"))
        # With _sort_programs=False the insertion order (p2, then p1) is preserved.
        assert (
            p2_pos < p1_pos
        ), f"Expected p2 before p1 (arrival order), got: {merged_keys}"


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
            prefixed = {f"{key}{_TAG_SEP}{t}": q for t, q in circuits.items()}
            results[key] = coord.submit(key, prefixed, **kwargs)

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
            prefixed = {f"{key}{_TAG_SEP}c1": "qasm"}
            results[key] = coord.submit(key, prefixed, ham_ops=ham)

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
            prefixed = {f"{key}{_TAG_SEP}c1": "qasm"}
            results[key] = coord.submit(key, prefixed, ham_ops=ham)

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
    def test_progress_messages_sent_to_queue(self):
        """Flush sends start and success progress messages."""
        queue = Queue()
        backend = FakeSyncBackend()
        coord = _BatchCoordinator(backend, progress_queue=queue)
        coord.register_program("p1")

        # Single program → barrier met immediately on submit.
        coord.submit("p1", {f"p1{_TAG_SEP}c1": "q1"})

        messages = []
        while not queue.empty():
            messages.append(queue.get_nowait())

        # At minimum: one start message and one success message.
        assert len(messages) >= 2
        assert messages[0]["batch"] is True
        assert messages[0]["n_circuits"] == 1
        assert messages[0]["n_programs"] == 1
        assert messages[-1].get("final_status") == "Success"

    def test_no_progress_without_queue(self):
        """When no queue is provided, nothing breaks."""
        backend = FakeSyncBackend()
        coord = _BatchCoordinator(backend, progress_queue=None)
        coord.register_program("p1")
        coord.submit("p1", {f"p1{_TAG_SEP}c1": "q1"})
        # No assertion needed — just verifying no error.

    def test_color_cycling(self):
        """Each flush group gets the next color in the cycle."""
        queue = Queue()
        backend = FakeSyncBackend()
        coord = _BatchCoordinator(backend, progress_queue=queue)

        coord.register_program("p1")
        coord.submit("p1", {f"p1{_TAG_SEP}c1": "q1"})
        coord.submit("p1", {f"p1{_TAG_SEP}c2": "q2"})

        messages = []
        while not queue.empty():
            messages.append(queue.get_nowait())

        colors = [m["batch_color"] for m in messages if "final_status" not in m]
        # First flush → first color, second flush → second color.
        assert colors[0] != colors[1]

    def test_mixed_ham_ops_sends_labelled_messages(self):
        """Sub-batches from ham_ops splitting include labels."""
        queue = Queue()
        backend = FakeExpvalBackend()
        coord = _BatchCoordinator(backend, progress_queue=queue)
        coord.register_program("p1")
        coord.register_program("p2")

        barrier = Barrier(2)

        def _submit(key, **kwargs):
            barrier.wait(timeout=5)
            coord.submit(key, {f"{key}{_TAG_SEP}c1": "qasm"}, **kwargs)

        t1 = Thread(target=_submit, args=("p1",), kwargs={"ham_ops": "Z"})
        t2 = Thread(target=_submit, args=("p2",))
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        messages = []
        while not queue.empty():
            messages.append(queue.get_nowait())

        labels = {m.get("batch_label") for m in messages}
        assert "expval" in labels
        assert "shots" in labels


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
                result_holder["p1"] = coord.submit("p1", {f"p1{_TAG_SEP}c1": "q"})
            except ExecutionCancelledError as e:
                error_holder["p1"] = e

        t = Thread(target=_submit_p1)
        t.start()

        barrier.wait(timeout=5)
        import time

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
        coord.submit("p1", {f"p1{_TAG_SEP}c1": "q"})

        # Sync backend → no runtime tracking.
        assert coord.total_runtime == 0.0


class TestProxyBackend:
    def test_delegates_properties(self):
        real = FakeSyncBackend(shots=200)
        coord = _BatchCoordinator(real)
        proxy = _ProxyBackend(real, coord, "prog_1")

        assert proxy.shots == 200
        assert proxy.supports_expval == real.supports_expval
        assert proxy.is_async is False
        assert proxy.max_retries == 0

    def test_submit_prefixes_tags_and_returns_results(self):
        """Proxy prefixes circuit tags and returns demuxed results."""
        backend = FakeSyncBackend()
        coord = _BatchCoordinator(backend)
        coord.register_program("prog_1")

        proxy = _ProxyBackend(backend, coord, "prog_1")
        result = proxy.submit_circuits({"circuit_a": "qasm_a", "circuit_b": "qasm_b"})

        assert result.results is not None
        assert len(result.results) == 2
        labels = {r["label"] for r in result.results}
        assert labels == {"circuit_a", "circuit_b"}

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
            args=("a", {f"a{_TAG_SEP}c1": "q"}),
        )
        t_b = Thread(
            target=_submit_ab,
            args=("b", {f"b{_TAG_SEP}c1": "q"}),
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
            args=("c", {f"c{_TAG_SEP}c1": "q"}),
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
            {f"p1{_TAG_SEP}c1": "q", f"p1{_TAG_SEP}c2": "q", f"p1{_TAG_SEP}c3": "q"},
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
