# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for backend tests."""

from http import HTTPStatus

import pytest

from divi.backends import QoroService, _qoro_service


@pytest.fixture
def qoro_service(api_key):
    """Provides a QoroService instance with a real API token for integration tests."""
    return QoroService(auth_token=api_key)


@pytest.fixture
def qoro_service_factory():
    """Provides a factory to create mocked QoroService instances.

    Temporarily replaces ``_make_request`` during construction so that
    ``fetch_qpu_systems`` and ``fetch_simulator_clusters`` receive empty
    responses. The original method is restored immediately after construction,
    so tests can set up their own mocks freely.
    """

    class _EmptyResponse:
        @staticmethod
        def json():
            return []

    def _factory(**kwargs):
        config = {
            "auth_token": "mock_token",
            "max_retries": 3,
            "polling_interval": 0.01,
        }
        config.update(kwargs)

        original = QoroService._make_request
        QoroService._make_request = lambda self, *a, **kw: _EmptyResponse()
        try:
            service = QoroService(**config)
        finally:
            QoroService._make_request = original
        return service

    return _factory


@pytest.fixture
def submit_circuits_mock(mocker, qoro_service_factory):
    """Mocks the dependencies for submit_circuits and returns the make_request mock."""
    mocker.patch(f"{_qoro_service.__name__}.is_valid_qasm", return_value=True)

    mock_init_response = mocker.MagicMock()
    mock_init_response.status_code = HTTPStatus.CREATED
    mock_init_response.json.return_value = {"job_id": "mock_job_id"}

    mock_add_response = mocker.MagicMock()
    mock_add_response.status_code = HTTPStatus.OK

    service_instance = qoro_service_factory()
    mock_make_request = mocker.patch.object(
        service_instance,
        "_make_request",
        side_effect=[mock_init_response, mock_add_response, mock_add_response],
    )

    return service_instance, mock_make_request


@pytest.fixture
def circuits():
    """Provides a dictionary of test circuits."""
    test_qasm = (
        'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[4];\ncreg c[4];\n'
        "x q[0];\nx q[1];\nry(0) q[2];\ncx q[2],q[3];\ncx q[2],q[0];"
        "cx q[3],q[1];\nmeasure q[0] -> c[0];\nmeasure q[1] -> c[1];"
        "\nmeasure q[2] -> c[2];\nmeasure q[3] -> c[3];\n"
    )
    return {f"circuit_{i}": test_qasm for i in range(10)}
