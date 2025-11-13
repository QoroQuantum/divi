# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from divi.reporting._pbar import (
    ConditionalSpinnerColumn,
    PhaseStatusColumn,
    make_progress_bar,
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

    def test_render_with_message_only(self, mocker, column):
        """Test rendering with just a message (no final_status)."""
        task = self._create_task(mocker, {"message": "Processing..."})
        result = column.render(task)

        assert "Processing..." in str(result)

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

    @pytest.mark.parametrize("is_jupyter", [True, False, None])
    def test_make_progress_bar(self, is_jupyter):
        """Test creating progress bar with different jupyter settings."""
        kwargs = {} if is_jupyter is None else {"is_jupyter": is_jupyter}
        progress = make_progress_bar(**kwargs)

        assert progress is not None
        assert hasattr(progress, "add_task")
