# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import time
from queue import Queue
from threading import Event, Lock, Thread

import pytest
from rich.live import Live
from rich.progress import Progress, TextColumn

from divi.reporting import (
    TerminalStatus,
    handle_batch_message,
    make_progress_bar,
    make_progress_display,
    progress_disabled,
    queue_listener,
)
from divi.reporting._pbar import (
    ConditionalSpinnerColumn,
    PhaseStatusColumn,
    _ProgramOnlyColumn,
)


class TestConditionalSpinnerColumn:
    """Tests for ConditionalSpinnerColumn."""

    @pytest.fixture
    def column(self):
        """Fixture providing a ConditionalSpinnerColumn instance."""
        return ConditionalSpinnerColumn()

    def _create_task(self, mocker, fields=None, **kwargs):
        """Helper to create a mock task with fields."""
        task = mocker.Mock()
        task.fields = fields or {}
        for key, value in kwargs.items():
            setattr(task, key, value)
        return task

    @pytest.mark.parametrize("status", ["Success", "Failed", "Cancelled", "Aborted"])
    def test_render_with_final_status_hides_spinner(self, mocker, column, status):
        """Test that spinner is hidden when final_status is in FINAL_STATUSES."""
        task = self._create_task(mocker, {"final_status": status})
        result = column.render(task)
        assert str(result) == ""

    @pytest.mark.parametrize(
        "fields",
        [
            {},
            {"final_status": "Running"},
            {"final_status": "Pending"},
        ],
    )
    def test_render_shows_spinner(self, mocker, column, fields):
        """Test that spinner is shown when final_status is not in FINAL_STATUSES."""
        task = self._create_task(mocker, fields, get_time=mocker.Mock(return_value=0.0))
        result = column.render(task)
        assert result != ""


class TestPhaseStatusColumn:
    """Tests for PhaseStatusColumn."""

    @pytest.fixture
    def column(self):
        """Fixture providing a PhaseStatusColumn instance."""
        return PhaseStatusColumn()

    def _create_task(self, mocker, fields):
        """Helper to create a mock task with fields."""
        task = mocker.Mock()
        task.fields = fields
        return task

    @pytest.mark.parametrize(
        "status,emoji",
        [
            ("Success", "✅"),
            ("Failed", "❌"),
            ("Cancelled", "⏹️"),
            ("Aborted", "⚠️"),
        ],
    )
    def test_render_final_status(self, mocker, column, status, emoji):
        """Test rendering with different final_status values."""
        task = self._create_task(mocker, {"final_status": status})
        result = column.render(task)

        assert status in str(result)
        assert emoji in str(result)

    def test_render_final_status_keeps_last_loss(self, mocker, column):
        """Test final status rendering keeps the last loss when present."""
        task = self._create_task(
            mocker, {"final_status": "Success", "loss": -0.123456789}
        )
        result = column.render(task)
        result_str = str(result)

        assert "Success" in result_str
        assert "loss: -0.123457" in result_str

    def test_render_with_message_only(self, mocker, column):
        """Test rendering with just a message (no final_status)."""
        task = self._create_task(mocker, {"message": "Processing..."})
        result = column.render(task)

        assert "Processing..." in str(result)

    def test_render_with_message_and_loss(self, mocker, column):
        """Test rendering includes formatted loss when present."""
        task = self._create_task(
            mocker, {"message": "Optimizing...", "loss": -0.123456789}
        )
        result = column.render(task)
        result_str = str(result)

        assert "Optimizing..." in result_str
        assert "loss: -0.123457" in result_str

    @pytest.mark.parametrize(
        "fields,expected_strings,check_highlighting",
        [
            (
                {
                    "message": "Running job",
                    "service_job_id": "service_abc-123",
                    "job_status": "COMPLETED",
                },
                ["service_abc", "complete"],
                False,
            ),
            (
                {
                    "message": "Running job",
                    "service_job_id": "service_xyz-456",
                    "job_status": "PENDING",
                    "poll_attempt": 3,
                    "max_retries": 10,
                },
                ["service_xyz", "PENDING", "Polling attempt 3 / 10"],
                False,
            ),
            (
                {
                    "message": "Running job",
                    "service_job_id": "service_test-789",
                },
                ["Running job"],
                True,
            ),
        ],
    )
    def test_render_with_service_job_id(
        self, mocker, column, fields, expected_strings, check_highlighting
    ):
        """Test rendering with service_job_id in various scenarios."""
        task = self._create_task(mocker, fields)
        result = column.render(task)

        result_str = str(result).lower()
        for expected in expected_strings:
            assert expected.lower() in result_str

        # Check highlighting when only service_job_id is present (no job_status)
        if check_highlighting:
            assert hasattr(result, "highlight_words")

    @pytest.mark.parametrize(
        "fields",
        [
            {"service_job_id": "service_def-012", "job_status": "RUNNING"},
            {
                "service_job_id": "service_ghi-345",
                "job_status": "PENDING",
                "poll_attempt": 0,
                "max_retries": 10,
            },
        ],
    )
    def test_render_with_service_job_id_no_polling_info(self, mocker, column, fields):
        """Test rendering with service_job_id but no polling info shown."""
        task = self._create_task(mocker, {"message": "Running job", **fields})
        result = column.render(task)

        result_str = str(result)
        assert "Running job" in result_str
        assert "Polling attempt" not in result_str


def test_make_progress_bar():
    """Test creating progress bar."""
    progress = make_progress_bar()

    assert progress is not None
    task_id = progress.add_task("test", total=1)
    assert isinstance(task_id, int)
    assert task_id in progress._tasks


def _make_simple_progress() -> Progress:
    """A minimal Progress instance (no columns needed for state tests)."""
    return Progress(TextColumn("{task.fields[job_name]}"), auto_refresh=False)


def _task_fields(progress: Progress, task_id) -> dict:
    """Read fields dict from a Rich Progress task."""
    return progress._tasks[task_id].fields


def _task_visible(progress: Progress, task_id) -> bool:
    return progress._tasks[task_id].visible


class TestHandleBatchMessage:
    """Tests for handle_batch_message progress bar state management.

    After the two-Progress collapse, batch rows live in the same
    ``Progress`` as program rows, distinguished by the ``row_kind``
    field.
    """

    @pytest.fixture
    def progress(self):
        return _make_simple_progress()

    @pytest.fixture
    def lock(self):
        return Lock()

    def test_creates_row_lazily(self, progress, lock):
        """First message for a batch_id creates a new task row."""
        batch_task_ids: dict[int, int] = {}

        handle_batch_message(
            {
                "batch_id": 1,
                "batch_color": "green",
                "batch_label": "",
                "n_circuits": 5,
                "n_programs": 2,
                "program_keys": [],
            },
            progress,
            batch_task_ids,
            lock,
        )

        assert 1 in batch_task_ids
        task_id = batch_task_ids[1]
        assert _task_visible(progress, task_id)
        fields = _task_fields(progress, task_id)
        assert fields["row_kind"] == "batch"
        assert "5 circuits" in fields["job_name"]
        assert "2 programs" in fields["job_name"]
        assert fields["batch_color"] == "green"

    def test_reuses_existing_row(self, progress, lock):
        """Subsequent messages with the same batch_id reuse the same task row."""
        batch_task_ids: dict[int, int] = {}

        for _ in range(3):
            handle_batch_message(
                {
                    "batch_id": 42,
                    "batch_color": "cyan",
                    "n_circuits": 10,
                    "n_programs": 3,
                    "program_keys": [],
                },
                progress,
                batch_task_ids,
                lock,
            )

        assert len(batch_task_ids) == 1
        assert len(progress.tasks) == 1

    def test_labelled_batch_prefix(self, progress, lock):
        """Label is included in the job_name prefix."""
        batch_task_ids: dict[int, int] = {}

        handle_batch_message(
            {
                "batch_id": 1,
                "batch_color": "red",
                "batch_label": "expval",
                "n_circuits": 3,
                "n_programs": 1,
                "program_keys": [],
            },
            progress,
            batch_task_ids,
            lock,
        )

        fields = _task_fields(progress, batch_task_ids[1])
        assert fields["job_name"].startswith("Batch (expval)")

    def test_unlabelled_batch_prefix(self, progress, lock):
        """Without a label, prefix is just 'Batch'."""
        batch_task_ids: dict[int, int] = {}

        handle_batch_message(
            {
                "batch_id": 1,
                "batch_color": "blue",
                "batch_label": "",
                "n_circuits": 2,
                "n_programs": 1,
                "program_keys": [],
            },
            progress,
            batch_task_ids,
            lock,
        )

        fields = _task_fields(progress, batch_task_ids[1])
        assert fields["job_name"].startswith("Batch:")

    def test_final_status_hides_row_and_cleans_up(self, progress, lock):
        """A final_status message hides the row and removes the batch_id mapping."""
        batch_task_ids: dict[int, int] = {}

        handle_batch_message(
            {
                "batch_id": 7,
                "batch_color": "green",
                "n_circuits": 1,
                "n_programs": 1,
                "program_keys": [],
            },
            progress,
            batch_task_ids,
            lock,
        )
        task_id = batch_task_ids[7]
        assert _task_visible(progress, task_id)

        handle_batch_message(
            {
                "batch_id": 7,
                "batch_color": "green",
                "n_circuits": 1,
                "n_programs": 1,
                "final_status": "Success",
                "program_keys": [],
            },
            progress,
            batch_task_ids,
            lock,
        )

        assert 7 not in batch_task_ids
        assert not _task_visible(progress, task_id)

    def test_program_color_set_on_start(self, progress, lock):
        """Active batch colors the participating program rows; coloring
        works by matching ``program_key`` task fields, not via a parallel
        index."""
        p1_tid = progress.add_task(
            "",
            job_name="prog1",
            batch_color="",
            total=10,
            row_kind="program",
            program_key="p1",
        )
        p2_tid = progress.add_task(
            "",
            job_name="prog2",
            batch_color="",
            total=10,
            row_kind="program",
            program_key="p2",
        )
        batch_task_ids: dict[int, int] = {}

        handle_batch_message(
            {
                "batch_id": 1,
                "batch_color": "magenta",
                "n_circuits": 4,
                "n_programs": 2,
                "program_keys": ["p1", "p2"],
            },
            progress,
            batch_task_ids,
            lock,
        )

        assert _task_fields(progress, p1_tid)["batch_color"] == "magenta"
        assert _task_fields(progress, p2_tid)["batch_color"] == "magenta"

    def test_program_color_cleared_on_final(self, progress, lock):
        """Final status clears batch color from program rows."""
        p1_tid = progress.add_task(
            "",
            job_name="prog1",
            batch_color="cyan",
            total=10,
            row_kind="program",
            program_key="p1",
        )
        batch_task_ids: dict[int, int] = {}

        for msg in [
            {
                "batch_id": 1,
                "batch_color": "cyan",
                "n_circuits": 1,
                "n_programs": 1,
                "program_keys": ["p1"],
            },
            {
                "batch_id": 1,
                "batch_color": "cyan",
                "n_circuits": 1,
                "n_programs": 1,
                "final_status": "Success",
                "program_keys": ["p1"],
            },
        ]:
            handle_batch_message(msg, progress, batch_task_ids, lock)

        assert _task_fields(progress, p1_tid)["batch_color"] == ""

    def test_polling_fields_forwarded(self, progress, lock):
        """Polling metadata is forwarded to the batch task."""
        batch_task_ids: dict[int, int] = {}

        handle_batch_message(
            {
                "batch_id": 1,
                "batch_color": "green",
                "n_circuits": 1,
                "n_programs": 1,
                "program_keys": [],
                "service_job_id": "job-abc",
                "job_status": "PENDING",
                "poll_attempt": 3,
                "max_retries": 100,
            },
            progress,
            batch_task_ids,
            lock,
        )

        fields = _task_fields(progress, batch_task_ids[1])
        assert fields["service_job_id"] == "job-abc"
        assert fields["job_status"] == "PENDING"
        assert fields["poll_attempt"] == 3
        assert fields["max_retries"] == 100

    def test_multiple_concurrent_batches(self, progress, lock):
        """Multiple batch_ids can coexist simultaneously in the same Progress."""
        batch_task_ids: dict[int, int] = {}

        for bid, color, label in [(1, "green", "expval"), (2, "cyan", "shots")]:
            handle_batch_message(
                {
                    "batch_id": bid,
                    "batch_color": color,
                    "batch_label": label,
                    "n_circuits": 3,
                    "n_programs": 1,
                    "program_keys": [],
                },
                progress,
                batch_task_ids,
                lock,
            )

        assert len(batch_task_ids) == 2
        batch_rows = [t for t in progress.tasks if t.fields["row_kind"] == "batch"]
        assert len(batch_rows) == 2

        fields_1 = _task_fields(progress, batch_task_ids[1])
        fields_2 = _task_fields(progress, batch_task_ids[2])
        assert "expval" in fields_1["job_name"]
        assert "shots" in fields_2["job_name"]
        assert fields_1["batch_color"] == "green"
        assert fields_2["batch_color"] == "cyan"

    def test_final_one_batch_keeps_other(self, progress, lock):
        """Finalizing one batch doesn't affect another active batch."""
        batch_task_ids: dict[int, int] = {}

        for bid in (10, 20):
            handle_batch_message(
                {
                    "batch_id": bid,
                    "batch_color": "green",
                    "n_circuits": 1,
                    "n_programs": 1,
                    "program_keys": [],
                },
                progress,
                batch_task_ids,
                lock,
            )

        tid_20 = batch_task_ids[20]

        handle_batch_message(
            {
                "batch_id": 10,
                "batch_color": "green",
                "n_circuits": 1,
                "n_programs": 1,
                "final_status": "Success",
                "program_keys": [],
            },
            progress,
            batch_task_ids,
            lock,
        )

        assert 10 not in batch_task_ids
        assert 20 in batch_task_ids
        assert _task_visible(progress, tid_20)


class TestProgramOnlyColumn:
    """The conditional column wrappers must render only on program rows
    and emit empty text for batch (or any non-program) rows."""

    def _task(self, mocker, row_kind):
        task = mocker.Mock()
        task.fields = {"row_kind": row_kind}
        return task

    def test_renders_inner_for_program_row(self, mocker):
        inner = mocker.Mock()
        col = _ProgramOnlyColumn(inner)
        col.render(self._task(mocker, "program"))
        inner.render.assert_called_once()

    def test_empty_text_for_batch_row(self, mocker):
        inner = mocker.Mock()
        col = _ProgramOnlyColumn(inner)
        result = col.render(self._task(mocker, "batch"))
        assert str(result) == ""
        inner.render.assert_not_called()

    def test_empty_text_for_unknown_row_kind(self, mocker):
        inner = mocker.Mock()
        col = _ProgramOnlyColumn(inner)
        result = col.render(self._task(mocker, None))
        assert str(result) == ""
        inner.render.assert_not_called()


class TestProgressDisabled:
    """The ``DIVI_DISABLE_PROGRESS`` env var controls whether the rich
    progress UI is disabled.  Parsing is centralized in
    :func:`progress_disabled`; both the standalone helper and the factory
    short-circuit must agree on what 'truthy' means."""

    @pytest.mark.parametrize(
        "value", ["1", "true", "TRUE", "yes", "YES", "on", "ON", " True "]
    )
    def test_truthy_values_disable(self, monkeypatch, value):
        monkeypatch.setenv("DIVI_DISABLE_PROGRESS", value)
        assert progress_disabled() is True

    @pytest.mark.parametrize("value", ["0", "false", "no", "off", "", "random"])
    def test_falsy_values_keep_enabled(self, monkeypatch, value):
        monkeypatch.setenv("DIVI_DISABLE_PROGRESS", value)
        assert progress_disabled() is False

    def test_unset_env_var_keeps_enabled(self, monkeypatch):
        monkeypatch.delenv("DIVI_DISABLE_PROGRESS", raising=False)
        assert progress_disabled() is False


class TestFactoryShortCircuit:
    """Both display factories must return all-``None`` tuples when the env
    var disables progress, so callers' ``is not None`` guards transparently
    skip UI work without an explicit branch."""

    def test_make_progress_display_returns_nones_when_disabled(self, monkeypatch):
        monkeypatch.setenv("DIVI_DISABLE_PROGRESS", "1")
        progress, live = make_progress_display()
        assert progress is None
        assert live is None

    def test_make_progress_display_returns_real_objects_when_enabled(self, monkeypatch):
        monkeypatch.delenv("DIVI_DISABLE_PROGRESS", raising=False)
        progress, live = make_progress_display()
        try:
            assert isinstance(progress, Progress)
            assert isinstance(live, Live)
        finally:
            if live is not None:
                live.stop()


class TestQueueListenerCrashSafety:
    """Regression for fix #3: a malformed message must not kill the
    listener thread; ``queue.task_done()`` must fire for every ``get()``
    so ``Queue.join()`` can never block forever.
    """

    def test_unknown_job_id_does_not_kill_listener(self, mocker):
        q: Queue = Queue()
        progress_bar = mocker.MagicMock()
        progress_bar.console = mocker.MagicMock()
        pb_task_map = {"known": 1}
        done = Event()
        lock = Lock()

        # First message references an unknown job_id (would raise KeyError
        # pre-fix); second references a known one and must be processed.
        q.put({"job_id": "unknown", "progress": 1})
        q.put({"job_id": "known", "progress": 2, "message": "ok"})

        thread = Thread(
            target=queue_listener,
            args=(q, progress_bar, pb_task_map, done, lock),
        )
        thread.start()
        # Wait for the queue to drain — this is the load-bearing assertion.
        # Pre-fix the listener died on the first message and the second
        # message's task_done() never fired, so ``q.join()`` would hang.
        try:
            q.join()
        finally:
            done.set()
            thread.join(timeout=2)

        assert not thread.is_alive(), "listener thread should exit on done_event"
        # The second (valid) message reached the progress bar.
        progress_bar.update.assert_called_once()
        # The malformed message was specifically logged as an unknown job_id.
        log_calls = [str(c) for c in progress_bar.console.log.call_args_list]
        assert any(
            "unknown" in c for c in log_calls
        ), f"expected an 'unknown job_id' log entry, got {log_calls}"

    def test_success_status_fills_bar_when_no_incremental_advance(self, mocker):
        """A Success final_status with progress=0 must mark the bar full
        so the task doesn't render as 0/N when its program never ticked
        the counter (e.g. single-shot TimeEvolution programs)."""
        q: Queue = Queue()
        progress_bar = mocker.MagicMock()
        progress_bar.console = mocker.MagicMock()
        # Rich Progress keeps tasks in a private dict; emulate that.
        task = mocker.Mock(total=1)
        progress_bar._tasks = {1: task}
        pb_task_map = {"j1": 1}
        done = Event()
        lock = Lock()

        q.put(
            {
                "job_id": "j1",
                "progress": 0,
                "message": "Finished successfully!",
                "final_status": "Success",
            }
        )

        thread = Thread(
            target=queue_listener,
            args=(q, progress_bar, pb_task_map, done, lock),
        )
        thread.start()
        try:
            q.join()
        finally:
            done.set()
            thread.join(timeout=2)

        progress_bar.update.assert_called_once()
        kwargs = progress_bar.update.call_args.kwargs
        assert kwargs.get("completed") == 1
        # ``advance`` must not coexist with ``completed`` — Rich would
        # otherwise apply both and overshoot the total.
        assert "advance" not in kwargs
        assert kwargs.get("final_status") == "Success"

    def test_hide_program_rows_reveals_failed_row(self, mocker):
        """With ``hide_program_rows=True`` (large ensemble: rows created
        invisible by the ensemble), a non-Success terminal status must
        flip ``visible=True`` so failures stay diagnosable.  Successful
        rows stay hidden."""
        q: Queue = Queue()
        progress_bar = mocker.MagicMock()
        progress_bar.console = mocker.MagicMock()
        progress_bar._tasks = {1: mocker.Mock(total=1), 2: mocker.Mock(total=1)}
        pb_task_map = {"j_ok": 1, "j_fail": 2}
        done = Event()
        lock = Lock()

        q.put(
            {
                "job_id": "j_ok",
                "progress": 0,
                "message": "Finished successfully!",
                "final_status": TerminalStatus.SUCCESS,
            }
        )
        q.put(
            {
                "job_id": "j_fail",
                "progress": 0,
                "message": "boom",
                "final_status": TerminalStatus.FAILED,
            }
        )

        thread = Thread(
            target=queue_listener,
            args=(q, progress_bar, pb_task_map, done, lock),
            kwargs={"hide_program_rows": True},
        )
        thread.start()
        try:
            q.join()
        finally:
            done.set()
            thread.join(timeout=2)

        calls = {c.args[0]: c.kwargs for c in progress_bar.update.call_args_list}
        # Success: row stays hidden (no visibility change emitted).
        assert "visible" not in calls[1]
        # Failure: row is revealed.
        assert calls[2].get("visible") is True

    def test_hide_program_rows_default_keeps_visibility_untouched(self, mocker):
        """Default behavior (small ensembles): listener leaves visibility
        alone on any terminal status."""
        q: Queue = Queue()
        progress_bar = mocker.MagicMock()
        progress_bar.console = mocker.MagicMock()
        progress_bar._tasks = {1: mocker.Mock(total=1)}
        pb_task_map = {"j_ok": 1}
        done = Event()
        lock = Lock()

        q.put(
            {
                "job_id": "j_ok",
                "progress": 0,
                "message": "Finished successfully!",
                "final_status": TerminalStatus.SUCCESS,
            }
        )

        thread = Thread(
            target=queue_listener,
            args=(q, progress_bar, pb_task_map, done, lock),
        )
        thread.start()
        try:
            q.join()
        finally:
            done.set()
            thread.join(timeout=2)

        kwargs = progress_bar.update.call_args.kwargs
        assert "visible" not in kwargs

    def test_progress_bar_update_failure_is_swallowed(self, mocker):
        """If ``progress_bar.update`` raises (e.g. Rich teardown), the
        listener must keep going and ``task_done()`` must still fire."""
        q: Queue = Queue()
        progress_bar = mocker.MagicMock()
        progress_bar.console = mocker.MagicMock()
        progress_bar.update.side_effect = RuntimeError("Rich is gone")
        pb_task_map = {"j1": 1, "j2": 2}
        done = Event()
        lock = Lock()

        q.put({"job_id": "j1", "progress": 1})
        q.put({"job_id": "j2", "progress": 1})

        thread = Thread(
            target=queue_listener,
            args=(q, progress_bar, pb_task_map, done, lock),
        )
        thread.start()
        try:
            q.join()  # must return even though update() raises
        finally:
            done.set()
            thread.join(timeout=2)

        assert not thread.is_alive()
        # Both messages were attempted.
        assert progress_bar.update.call_count == 2

    def test_queue_get_exception_is_logged(self, mocker):
        mock_queue = mocker.MagicMock()
        progress_bar = mocker.MagicMock()
        done = Event()
        pb_task_map: dict = {}
        lock = Lock()
        mock_queue.get.side_effect = Exception("Unexpected queue error")
        progress_bar.console = mocker.MagicMock()

        thread = Thread(
            target=queue_listener,
            args=(mock_queue, progress_bar, pb_task_map, done, lock),
        )
        thread.start()
        time.sleep(0.2)
        done.set()
        thread.join(timeout=1)

        progress_bar.console.log.assert_called()
        assert "queue.get failed" in str(progress_bar.console.log.call_args)

    def test_optional_message_fields_forwarded_to_update(self, mocker):
        q: Queue = Queue()
        progress_bar = mocker.MagicMock()
        done = Event()
        pb_task_map = {"job1": 1}
        lock = Lock()

        q.put(
            {
                "job_id": "job1",
                "progress": 1,
                "poll_attempt": 3,
                "max_retries": 5,
                "service_job_id": "service_123",
                "job_status": "running",
                "message": "Processing...",
                "final_status": "completed",
                "loss": -0.321,
            }
        )

        thread = Thread(
            target=queue_listener,
            args=(q, progress_bar, pb_task_map, done, lock),
        )
        thread.start()
        time.sleep(0.1)
        done.set()
        thread.join(timeout=1)

        progress_bar.update.assert_called_once()
        call_kwargs = progress_bar.update.call_args[1]
        assert call_kwargs["poll_attempt"] == 3
        assert call_kwargs["max_retries"] == 5
        assert call_kwargs["service_job_id"] == "service_123"
        assert call_kwargs["job_status"] == "running"
        assert call_kwargs["loss"] == -0.321

    def test_basic_program_updates(self, mocker):
        q: Queue = Queue()
        progress_bar = mocker.MagicMock()
        done = Event()
        lock = Lock()
        pb_task_map = {"job1": 1, "job2": 2}

        q.put(
            {
                "job_id": "job1",
                "progress": 1,
                "message": "step 1",
                "final_status": "running",
            }
        )
        q.put({"job_id": "job2", "progress": 1, "poll_attempt": 3})

        thread = Thread(
            target=queue_listener,
            args=(q, progress_bar, pb_task_map, done, lock),
        )
        thread.start()
        time.sleep(0.1)
        done.set()
        thread.join(timeout=1)
        assert not thread.is_alive()

        expected_calls = [
            mocker.call(1, advance=1, message="step 1", final_status="running"),
            mocker.call(2, advance=1, poll_attempt=3),
        ]
        progress_bar.update.assert_has_calls(expected_calls, any_order=True)
        assert q.empty()
