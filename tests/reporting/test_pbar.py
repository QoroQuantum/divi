# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from threading import Lock

import pytest
from rich.live import Live
from rich.progress import Progress, TextColumn

from divi.reporting._pbar import (
    ConditionalSpinnerColumn,
    PhaseStatusColumn,
    handle_batch_message,
    make_batch_display,
    make_progress_bar,
    make_progress_display,
    progress_disabled,
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


class TestMakeProgressBar:
    """Tests for make_progress_bar function."""

    def test_make_progress_bar(self):
        """Test creating progress bar."""
        progress = make_progress_bar()

        assert progress is not None
        assert hasattr(progress, "add_task")


# ---------------------------------------------------------------------------
# Helpers for batch message tests
# ---------------------------------------------------------------------------


def _make_simple_progress() -> Progress:
    """A minimal Progress instance (no columns needed for state tests)."""
    return Progress(TextColumn("{task.fields[job_name]}"), auto_refresh=False)


def _task_fields(progress: Progress, task_id) -> dict:
    """Read fields dict from a Rich Progress task."""
    return progress._tasks[task_id].fields


def _task_visible(progress: Progress, task_id) -> bool:
    return progress._tasks[task_id].visible


# ---------------------------------------------------------------------------
# handle_batch_message
# ---------------------------------------------------------------------------


class TestHandleBatchMessage:
    """Tests for handle_batch_message progress bar state management."""

    @pytest.fixture
    def batch_progress(self):
        return _make_simple_progress()

    @pytest.fixture
    def program_progress(self):
        return _make_simple_progress()

    @pytest.fixture
    def lock(self):
        return Lock()

    def test_creates_row_lazily(self, batch_progress, program_progress, lock):
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
            batch_progress,
            batch_task_ids,
            program_progress,
            {},
            lock,
        )

        assert 1 in batch_task_ids
        task_id = batch_task_ids[1]
        assert _task_visible(batch_progress, task_id)
        fields = _task_fields(batch_progress, task_id)
        assert "5 circuits" in fields["job_name"]
        assert "2 programs" in fields["job_name"]
        assert fields["batch_color"] == "green"

    def test_reuses_existing_row(self, batch_progress, program_progress, lock):
        """Subsequent messages with the same batch_id reuse the same task row."""
        batch_task_ids: dict[int, int] = {}

        for i in range(3):
            handle_batch_message(
                {
                    "batch_id": 42,
                    "batch_color": "cyan",
                    "n_circuits": 10,
                    "n_programs": 3,
                    "program_keys": [],
                },
                batch_progress,
                batch_task_ids,
                program_progress,
                {},
                lock,
            )

        # Only one task row should exist.
        assert len(batch_task_ids) == 1
        assert len(batch_progress.tasks) == 1

    def test_labelled_batch_prefix(self, batch_progress, program_progress, lock):
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
            batch_progress,
            batch_task_ids,
            program_progress,
            {},
            lock,
        )

        fields = _task_fields(batch_progress, batch_task_ids[1])
        assert fields["job_name"].startswith("Batch (expval)")

    def test_unlabelled_batch_prefix(self, batch_progress, program_progress, lock):
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
            batch_progress,
            batch_task_ids,
            program_progress,
            {},
            lock,
        )

        fields = _task_fields(batch_progress, batch_task_ids[1])
        assert fields["job_name"].startswith("Batch:")

    def test_final_status_hides_row_and_cleans_up(
        self, batch_progress, program_progress, lock
    ):
        """A final_status message hides the row and removes the batch_id mapping."""
        batch_task_ids: dict[int, int] = {}

        # Create the row first.
        handle_batch_message(
            {
                "batch_id": 7,
                "batch_color": "green",
                "n_circuits": 1,
                "n_programs": 1,
                "program_keys": [],
            },
            batch_progress,
            batch_task_ids,
            program_progress,
            {},
            lock,
        )
        task_id = batch_task_ids[7]
        assert _task_visible(batch_progress, task_id)

        # Now send success.
        handle_batch_message(
            {
                "batch_id": 7,
                "batch_color": "green",
                "n_circuits": 1,
                "n_programs": 1,
                "final_status": "Success",
                "program_keys": [],
            },
            batch_progress,
            batch_task_ids,
            program_progress,
            {},
            lock,
        )

        assert 7 not in batch_task_ids
        assert not _task_visible(batch_progress, task_id)

    def test_program_color_set_on_start(self, batch_progress, program_progress, lock):
        """Active batch colors the participating program rows."""
        # Create program rows.
        p1_tid = program_progress.add_task(
            "", job_name="prog1", batch_color="", total=10
        )
        p2_tid = program_progress.add_task(
            "", job_name="prog2", batch_color="", total=10
        )
        program_key_to_task_ids = {"p1": [p1_tid], "p2": [p2_tid]}
        batch_task_ids: dict[int, int] = {}

        handle_batch_message(
            {
                "batch_id": 1,
                "batch_color": "magenta",
                "n_circuits": 4,
                "n_programs": 2,
                "program_keys": ["p1", "p2"],
            },
            batch_progress,
            batch_task_ids,
            program_progress,
            program_key_to_task_ids,
            lock,
        )

        assert _task_fields(program_progress, p1_tid)["batch_color"] == "magenta"
        assert _task_fields(program_progress, p2_tid)["batch_color"] == "magenta"

    def test_program_color_cleared_on_final(
        self, batch_progress, program_progress, lock
    ):
        """Final status clears batch color from program rows."""
        p1_tid = program_progress.add_task(
            "", job_name="prog1", batch_color="cyan", total=10
        )
        program_key_to_task_ids = {"p1": [p1_tid]}
        batch_task_ids: dict[int, int] = {}

        # Create, then finalize.
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
            handle_batch_message(
                msg,
                batch_progress,
                batch_task_ids,
                program_progress,
                program_key_to_task_ids,
                lock,
            )

        assert _task_fields(program_progress, p1_tid)["batch_color"] == ""

    def test_polling_fields_forwarded(self, batch_progress, program_progress, lock):
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
            batch_progress,
            batch_task_ids,
            program_progress,
            {},
            lock,
        )

        fields = _task_fields(batch_progress, batch_task_ids[1])
        assert fields["service_job_id"] == "job-abc"
        assert fields["job_status"] == "PENDING"
        assert fields["poll_attempt"] == 3
        assert fields["max_retries"] == 100

    def test_multiple_concurrent_batches(self, batch_progress, program_progress, lock):
        """Multiple batch_ids can coexist simultaneously."""
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
                batch_progress,
                batch_task_ids,
                program_progress,
                {},
                lock,
            )

        assert len(batch_task_ids) == 2
        assert len(batch_progress.tasks) == 2

        fields_1 = _task_fields(batch_progress, batch_task_ids[1])
        fields_2 = _task_fields(batch_progress, batch_task_ids[2])
        assert "expval" in fields_1["job_name"]
        assert "shots" in fields_2["job_name"]
        assert fields_1["batch_color"] == "green"
        assert fields_2["batch_color"] == "cyan"

    def test_final_one_batch_keeps_other(self, batch_progress, program_progress, lock):
        """Finalizing one batch doesn't affect another active batch."""
        batch_task_ids: dict[int, int] = {}

        # Start two batches.
        for bid in (10, 20):
            handle_batch_message(
                {
                    "batch_id": bid,
                    "batch_color": "green",
                    "n_circuits": 1,
                    "n_programs": 1,
                    "program_keys": [],
                },
                batch_progress,
                batch_task_ids,
                program_progress,
                {},
                lock,
            )

        tid_20 = batch_task_ids[20]

        # Finalize batch 10.
        handle_batch_message(
            {
                "batch_id": 10,
                "batch_color": "green",
                "n_circuits": 1,
                "n_programs": 1,
                "final_status": "Success",
                "program_keys": [],
            },
            batch_progress,
            batch_task_ids,
            program_progress,
            {},
            lock,
        )

        assert 10 not in batch_task_ids
        assert 20 in batch_task_ids
        assert _task_visible(batch_progress, tid_20)


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

    def test_make_batch_display_returns_nones_when_disabled(self, monkeypatch):
        monkeypatch.setenv("DIVI_DISABLE_PROGRESS", "1")
        batch, progress, live = make_batch_display()
        assert batch is None
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

    def test_make_batch_display_returns_real_objects_when_enabled(self, monkeypatch):
        monkeypatch.delenv("DIVI_DISABLE_PROGRESS", raising=False)
        batch, progress, live = make_batch_display()
        try:
            assert isinstance(batch, Progress)
            assert isinstance(progress, Progress)
            assert isinstance(live, Live)
        finally:
            if live is not None:
                live.stop()
