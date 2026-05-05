# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for QUBO/HUBO characterization feature.

Tests cover:
- Wire-format serialization from all supported QUBO/HUBO types
- CharacterizationResult dataclass properties and display
- QoroService.characterize_and_validate submit + fetch flows (mocked HTTP)
- Top-level divi.backends.characterize_and_validate convenience function
"""

from http import HTTPStatus

import numpy as np
import pytest
import requests

from divi.backends import (
    CharacterizationOptions,
    ExecutionResult,
    characterize_and_validate,
    get_characterization_result,
)
from divi.backends._characterization import (
    CharacterizationResult,
    _serialize_qubo_for_wire,
    _wrap_response,
)
from divi.backends._qoro_service import JobType, QoroService
from divi.qprog.problems._binary import BinaryOptimizationProblem


def _is_number(val) -> bool:
    """``isinstance(val, (int, float))`` excluding ``bool``.

    ``bool`` is a subclass of ``int``; without this guard, a boolean-valued
    field would silently pass numeric type assertions.
    """
    return isinstance(val, (int, float)) and not isinstance(val, bool)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_REPORT = {
    "quality_score": 78.5,
    "concentration_ratio": 3.2,
    "approximation_ratio": 0.92,
    "feasibility_rate": 0.85,
    "state_probabilities": [
        {"state": "01", "is_target": True, "probability": 0.35, "energy": -1.0},
        {"state": "10", "is_target": True, "probability": 0.30, "energy": -1.0},
        {"state": "00", "is_target": False, "probability": 0.20, "energy": 0.0},
        {"state": "11", "is_target": False, "probability": 0.15, "energy": 2.0},
    ],
    "best_parameters": {"gamma": 1.2, "beta": 0.7, "probability": 0.15},
    "penalty_recommendation": 2.5,
    "penalty_tuning": {"is_well_tuned": True, "optimal_lambda": 2.5},
    "sensitivity": [
        {"qubit": 0, "sensitivity": 0.42},
        {"qubit": 1, "sensitivity": 0.18},
    ],
}

SAMPLE_HARDNESS = {
    "difficulty": "medium",
    "spectral_gap": 0.35,
    "condition_number": 4.2,
    "degeneracy": 2,
}

# Mirrors the structure produced by usher's ``build_recommendations`` —
# top-level field on the response, list of dicts with ``level``/``metric``/
# ``text``/``html`` keys.
SAMPLE_RECOMMENDATIONS = [
    {
        "level": "info",
        "metric": "quality_score",
        "text": "Quality is moderate — consider tuning penalty parameters.",
        "html": "<strong>Quality is moderate</strong> — consider tuning penalty parameters.",
    },
    {
        "level": "warn",
        "metric": "feasibility_rate",
        "text": "Feasibility rate is low; review constraint penalties.",
        "html": "<strong>Feasibility rate is low</strong>; review constraint penalties.",
    },
]

SAMPLE_RESPONSE = {
    "job_id": "abc-123",
    "status": "COMPLETED",
    "hardness": SAMPLE_HARDNESS,
    "report": SAMPLE_REPORT,
    "recommendations": SAMPLE_RECOMMENDATIONS,
    "created_at": "2026-04-25T12:00:00Z",
    "completed_at": "2026-04-25T12:00:01Z",
}


@pytest.fixture
def characterization_result():
    """A fully-populated CharacterizationResult for testing."""
    return CharacterizationResult(
        job_id="abc-123",
        status="COMPLETED",
        hardness=SAMPLE_HARDNESS,
        report=SAMPLE_REPORT,
        recommendations=SAMPLE_RECOMMENDATIONS,
        created_at="2026-04-25T12:00:00Z",
        completed_at="2026-04-25T12:00:01Z",
        html="<div>stub</div>",
    )


@pytest.fixture
def empty_result():
    """A minimal CharacterizationResult with no report or hardness."""
    return CharacterizationResult(
        job_id="xyz-789",
        status="FAILED",
        html="",
    )


@pytest.fixture
def qoro_service_factory():
    """Factory for creating mocked QoroService instances."""

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

        def _stub(self, method, endpoint, **kwargs):
            return _EmptyResponse()

        QoroService._make_request = _stub  # type: ignore[method-assign]
        try:
            service = QoroService(**config)
        finally:
            QoroService._make_request = original  # type: ignore[method-assign]
        return service

    return _factory


# ---------------------------------------------------------------------------
# _serialize_qubo_for_wire
# ---------------------------------------------------------------------------


class TestSerializeQuboForWire:
    """Tests for serializing a BinaryOptimizationProblem to wire format."""

    def test_ndarray_input(self):
        """Diagonal entries become single-index keys, off-diagonals tuple keys."""
        problem = BinaryOptimizationProblem(np.array([[-1.0, 2.0], [0.0, -1.0]]))
        wire = _serialize_qubo_for_wire(problem)
        assert all(isinstance(k, str) for k in wire)
        assert wire == {"0": -1.0, "0,1": 2.0, "1": -1.0}

    def test_zero_coefficients_skipped(self):
        problem = BinaryOptimizationProblem({(0,): -1.0, (0, 1): 0.0, (1,): -1.0})
        wire = _serialize_qubo_for_wire(problem)
        assert "0,1" not in wire

    def test_hubo_input(self):
        """HUBO terms (degree > 2) serialize with multi-index keys."""
        problem = BinaryOptimizationProblem({(0,): -1.0, (0, 1): 2.0, (0, 1, 2): 3.0})
        wire = _serialize_qubo_for_wire(problem)
        assert wire["0"] == -1.0
        assert wire["0,1"] == 2.0
        assert wire["0,1,2"] == 3.0


# ---------------------------------------------------------------------------
# CharacterizationResult
# ---------------------------------------------------------------------------


class TestCharacterizationResult:
    """Tests for the characterization result dataclass."""

    def test_quality_score(self, characterization_result):
        assert characterization_result.quality_score == 78.5

    def test_concentration_ratio(self, characterization_result):
        assert characterization_result.concentration_ratio == 3.2

    def test_is_well_tuned(self, characterization_result):
        assert characterization_result.is_well_tuned is True

    def test_empty_result_properties(self, empty_result):
        assert empty_result.quality_score is None
        assert empty_result.concentration_ratio is None
        assert empty_result.is_well_tuned is None

    def test_summary_contains_key_metrics(self, characterization_result):
        s = characterization_result.summary()
        assert "78.5" in s
        assert "3.2" in s
        assert "medium" in s
        assert "abc-123" in s

    def test_repr_equals_summary(self, characterization_result):
        assert repr(characterization_result) == characterization_result.summary()

    def test_repr_html_returns_stored_html(self):
        """``_repr_html_`` returns whatever HTML was stored at construction."""
        result = CharacterizationResult(
            job_id="abc-123",
            status="COMPLETED",
            html="<div>server-rendered</div>",
        )
        assert result._repr_html_() == "<div>server-rendered</div>"

    def test_recommendations_field_carries_server_value(self):
        """The ``recommendations`` dataclass field stores the server's list."""
        recs = [
            {
                "level": "warn",
                "metric": "quality_score",
                "text": "X",
                "html": "<strong>X</strong>",
            }
        ]
        result = CharacterizationResult(
            job_id="r1",
            status="COMPLETED",
            recommendations=recs,
            html="",
        )
        assert result.recommendations == recs

    def test_recommendations_empty_list_when_missing(self, empty_result):
        """``recommendations`` is an empty list when the server omits the field.

        Iterating, ``len()``, and truthiness checks must all behave like a
        list — never raise ``TypeError`` because of a ``None`` slipping
        through.
        """
        assert empty_result.recommendations == []
        # Smoke-test the patterns user code is likely to use:
        assert list(empty_result.recommendations) == []
        assert len(empty_result.recommendations) == 0
        assert not empty_result.recommendations

    def test_display_renders_server_recommendations(self, capsys):
        """``display()`` surfaces server-supplied recommendations."""
        result = CharacterizationResult(
            job_id="d1",
            status="COMPLETED",
            recommendations=[
                {
                    "level": "action",
                    "metric": "feasibility_rate",
                    "text": "Feasibility rate is only 30%",
                    "html": "Feasibility rate is only <strong>30%</strong>",
                },
            ],
            html="",
        )
        result.display()
        captured = capsys.readouterr().out
        assert "Recommendations" in captured
        assert "Feasibility rate is only" in captured
        # HTML <strong> stripped to plain text in the terminal output.
        assert "<strong>" not in captured

    def test_display_does_not_read_recommendations_from_report_dict(self, capsys):
        """Regression: ``display()`` reads the ``recommendations`` dataclass
        field, not ``self.report["recommendations"]``.

        The ``report`` dict is opaque pass-through from Composer (the upstream
        validation service); whatever Composer puts under any key name is
        Composer's business. Usher exposes the structured user-facing
        recommendations as a *separate* top-level response field, which divi
        stores on the dedicated dataclass field.

        Mechanism: the test plants a ``list[str]`` under
        ``report["recommendations"]``. If a future change re-introduces a
        property/getter that reads from there, ``display()`` will iterate
        the strings and crash with ``AttributeError`` (strings have no
        ``.get()``) — the same crash that motivated this refactor. The
        ``== []`` assertion additionally pins that ``result.recommendations``
        is not shadowed by such a getter.
        """
        result = CharacterizationResult(
            job_id="r1",
            status="COMPLETED",
            report={
                "quality_score": 50.0,
                "recommendations": [
                    "Composer string 1",
                    "Composer string 2",
                ],
            },
            html="",
        )
        # The dataclass field, not ``report``, is the source of truth.
        assert result.recommendations == []
        # And ``display()`` must not iterate ``report["recommendations"]``.
        result.display()  # AttributeError would propagate if it did

    def test_wrap_response_collapses_null_recommendations_to_list(self, mocker):
        """``_wrap_response`` must turn a missing or null ``recommendations``
        field into the empty-list contract."""
        service = mocker.MagicMock()
        service._fetch_characterization_html.return_value = ""
        # explicit null
        result_null = _wrap_response(
            {"job_id": "n1", "status": "COMPLETED", "recommendations": None},
            service,
        )
        # missing key
        result_missing = _wrap_response(
            {"job_id": "n2", "status": "COMPLETED"},
            service,
        )
        assert result_null.recommendations == []
        assert result_missing.recommendations == []

    def test_wrap_response_passes_recommendations_through(self, mocker):
        """``_wrap_response`` must pass a non-empty ``recommendations`` list
        through unchanged onto the dataclass field.

        This is the direct regression test for the original bug: the previous
        ``_wrap_response`` silently dropped the top-level ``recommendations``
        field, leaving the result blank even when the server provided one.
        """
        service = mocker.MagicMock()
        service._fetch_characterization_html.return_value = ""
        recs = [
            {
                "level": "warn",
                "metric": "feasibility_rate",
                "text": "Feasibility is low.",
                "html": "<strong>Feasibility</strong> is low.",
            },
        ]
        result = _wrap_response(
            {"job_id": "p1", "status": "COMPLETED", "recommendations": recs},
            service,
        )
        assert result.recommendations == recs


# ---------------------------------------------------------------------------
# QoroService.characterize_and_validate (mocked HTTP)
# ---------------------------------------------------------------------------


class TestQoroServiceCharacterize:
    """Tests for the consolidated submit + fetch flow in QoroService."""

    def test_full_submit_flow(self, mocker, qoro_service_factory):
        service = qoro_service_factory()

        mock_init = mocker.MagicMock()
        mock_init.json.return_value = {"job_id": "val-job-123"}
        mock_submit = mocker.MagicMock(status_code=HTTPStatus.OK)
        mock_result = mocker.MagicMock()
        mock_result.json.return_value = SAMPLE_RESPONSE

        mock_req = mocker.patch.object(
            service,
            "_make_request",
            side_effect=[mock_init, mock_submit, mock_result],
        )
        mocker.patch.object(
            service, "_fetch_characterization_html", return_value="<div/>"
        )

        result = service.characterize_and_validate(
            qubo={"0,0": -1.0, "0,1": 2.0},
            target_states=["01", "10"],
        )

        # service.characterize_and_validate returns the raw API response dict;
        # the rich wrapper is built by divi.backends.characterize_and_validate().
        assert isinstance(result, dict)
        assert result["status"] == "COMPLETED"
        assert result["report"]["quality_score"] == 78.5
        assert result["hardness"]["difficulty"] == "medium"

        # init → submit → result: three _make_request calls, all routed
        # through the same wrapper (submit uses retry=False).
        assert mock_req.call_count == 3
        init_call, submit_call, _ = mock_req.call_args_list
        assert init_call.args == ("post", "job/init/")
        assert init_call.kwargs["json"]["job_type"] == "VALIDATE"
        assert submit_call.args == ("post", "job/val-job-123/submit_qubo/")
        assert submit_call.kwargs["retry"] is False
        assert submit_call.kwargs["json"]["qubo"] == {"0,0": -1.0, "0,1": 2.0}
        assert submit_call.kwargs["json"]["target_states"] == ["01", "10"]

    def test_options_pass_through(self, mocker, qoro_service_factory):
        service = qoro_service_factory()

        mock_init = mocker.MagicMock()
        mock_init.json.return_value = {"job_id": "val-opt-123"}
        mock_submit = mocker.MagicMock(status_code=HTTPStatus.OK)
        mock_result = mocker.MagicMock()
        mock_result.json.return_value = SAMPLE_RESPONSE

        mock_req = mocker.patch.object(
            service,
            "_make_request",
            side_effect=[mock_init, mock_submit, mock_result],
        )
        mocker.patch.object(
            service, "_fetch_characterization_html", return_value="<div/>"
        )

        options = {
            "ansatz": {"mixer": "x", "layers": 1},
            "analysis": {"sensitivity": True, "parameter_sweep": True},
        }

        service.characterize_and_validate(
            qubo={"0,0": -1.0},
            target_states=["0"],
            options=options,
        )

        assert mock_req.call_args_list[1].kwargs["json"]["options"] == options

    def test_hardness_only_option_pipes_through(self, mocker, qoro_service_factory):
        """Hardness-only is just a parameterization of the submit flow."""
        service = qoro_service_factory()

        mock_init = mocker.MagicMock()
        mock_init.json.return_value = {"job_id": "hard-123"}
        mock_submit = mocker.MagicMock(status_code=HTTPStatus.OK)
        mock_result = mocker.MagicMock()
        mock_result.json.return_value = {"hardness": {}, "status": "COMPLETED"}

        mock_req = mocker.patch.object(
            service,
            "_make_request",
            side_effect=[mock_init, mock_submit, mock_result],
        )
        mocker.patch.object(
            service, "_fetch_characterization_html", return_value="<div/>"
        )

        service.characterize_and_validate(
            qubo={"0,0": -1.0},
            target_states=[],
            options={"analysis": {"hardness_only": True}},
        )

        submit_payload = mock_req.call_args_list[1].kwargs["json"]
        assert submit_payload["target_states"] == []
        assert submit_payload["options"]["analysis"]["hardness_only"] is True

    def test_fetch_by_job_id_skips_submit(self, mocker, qoro_service_factory):
        """Passing ``job_id`` only triggers the fetch step — no init/submit."""
        service = qoro_service_factory()

        mock_resp = mocker.MagicMock()
        mock_resp.json.return_value = SAMPLE_RESPONSE

        mock_req = mocker.patch.object(service, "_make_request", return_value=mock_resp)

        result = service.characterize_and_validate(job_id="abc-123")

        assert isinstance(result, dict)
        assert result["job_id"] == "abc-123"
        assert result["report"]["quality_score"] == 78.5

        # Only the GET to validation_result/, no init/submit traffic.
        assert mock_req.call_count == 1
        assert mock_req.call_args.args == ("get", "job/abc-123/validation_result/")

    def test_requires_qubo_or_job_id(self, qoro_service_factory):
        service = qoro_service_factory()
        with pytest.raises(ValueError, match="qubo.*or.*job_id"):
            service.characterize_and_validate()


# ---------------------------------------------------------------------------
# Top-level characterize_and_validate() function
# ---------------------------------------------------------------------------


class TestTopLevelCharacterize:
    """Tests for the divi.backends.characterize_and_validate convenience function."""

    def test_characterize_with_ndarray(self, mocker, qoro_service_factory):
        service = qoro_service_factory()

        mock_init = mocker.MagicMock()
        mock_init.json.return_value = {"job_id": "top-123"}
        mock_result = mocker.MagicMock()
        mock_result.json.return_value = SAMPLE_RESPONSE

        mock_submit = mocker.MagicMock(status_code=HTTPStatus.OK)
        mocker.patch.object(
            service,
            "_make_request",
            side_effect=[mock_init, mock_submit, mock_result],
        )
        mocker.patch.object(
            service, "_fetch_characterization_html", return_value="<div/>"
        )

        problem = BinaryOptimizationProblem(np.array([[-1.0, 2.0], [0.0, -1.0]]))
        result = characterize_and_validate(
            problem, target_states=["01", "10"], service=service
        )

        assert isinstance(result, CharacterizationResult)
        assert result.quality_score == 78.5

    def test_characterize_with_options(self, mocker, qoro_service_factory):
        service = qoro_service_factory()

        mock_init = mocker.MagicMock()
        mock_init.json.return_value = {"job_id": "top-opt-123"}
        mock_submit = mocker.MagicMock(status_code=HTTPStatus.OK)
        mock_result = mocker.MagicMock()
        mock_result.json.return_value = SAMPLE_RESPONSE

        mock_req = mocker.patch.object(
            service,
            "_make_request",
            side_effect=[mock_init, mock_submit, mock_result],
        )
        mocker.patch.object(
            service, "_fetch_characterization_html", return_value="<div/>"
        )

        characterize_and_validate(
            BinaryOptimizationProblem(np.array([[-1.0, 2.0], [0.0, -1.0]])),
            target_states=["01"],
            service=service,
            options=CharacterizationOptions(
                sensitivity=True,
                parameter_sweep=True,
                auto_tune=True,
            ),
        )

        options = mock_req.call_args_list[1].kwargs["json"]["options"]
        assert options["analysis"]["sensitivity"] is True
        assert options["analysis"]["parameter_sweep"] is True
        assert options["analysis"]["auto_tune"] is True

    def test_characterize_with_fixed_gamma_beta(self, mocker, qoro_service_factory):
        service = qoro_service_factory()

        mock_init = mocker.MagicMock()
        mock_init.json.return_value = {"job_id": "top-fixed-123"}
        mock_submit = mocker.MagicMock(status_code=HTTPStatus.OK)
        mock_result = mocker.MagicMock()
        mock_result.json.return_value = SAMPLE_RESPONSE

        mock_req = mocker.patch.object(
            service,
            "_make_request",
            side_effect=[mock_init, mock_submit, mock_result],
        )
        mocker.patch.object(
            service, "_fetch_characterization_html", return_value="<div/>"
        )

        characterize_and_validate(
            BinaryOptimizationProblem(np.array([[-1.0, 2.0], [0.0, -1.0]])),
            target_states=["01"],
            service=service,
            options=CharacterizationOptions(gamma=1.0, beta=0.5),
        )

        analysis = mock_req.call_args_list[1].kwargs["json"]["options"]["analysis"]
        assert analysis["gamma"] == 1.0
        assert analysis["beta"] == 0.5

    def test_parameter_sweep_with_fixed_gamma_raises(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            CharacterizationOptions(parameter_sweep=True, gamma=1.0)

    def test_ansatz_auto_warmstart_rejected(self):
        with pytest.raises(ValueError, match="auto_warmstart.*managed by the backend"):
            CharacterizationOptions(ansatz={"mixer": "x", "auto_warmstart": True})

    def test_characterize_with_penalty_qubos(self, mocker, qoro_service_factory):
        service = qoro_service_factory()

        mock_init = mocker.MagicMock()
        mock_init.json.return_value = {"job_id": "pen-123"}
        mock_submit = mocker.MagicMock(status_code=HTTPStatus.OK)
        mock_result = mocker.MagicMock()
        mock_result.json.return_value = SAMPLE_RESPONSE

        mock_req = mocker.patch.object(
            service,
            "_make_request",
            side_effect=[mock_init, mock_submit, mock_result],
        )
        mocker.patch.object(
            service, "_fetch_characterization_html", return_value="<div/>"
        )

        cost_q = BinaryOptimizationProblem(np.array([[-1.0, 0.0], [0.0, -1.0]]))
        pen_q = BinaryOptimizationProblem(np.array([[0.0, 2.0], [0.0, 0.0]]))

        characterize_and_validate(
            BinaryOptimizationProblem(np.array([[-1.0, 2.0], [0.0, -1.0]])),
            target_states=["01"],
            service=service,
            options=CharacterizationOptions(cost_qubo=cost_q, penalty_qubo=pen_q),
        )

        options = mock_req.call_args_list[1].kwargs["json"]["options"]
        assert "cost_qubo" in options
        assert "penalty_qubo" in options
        # Diagonal terms serialize as single-index keys via the canonical form.
        assert options["cost_qubo"]["0"] == -1.0


# ---------------------------------------------------------------------------
# JobType enum
# ---------------------------------------------------------------------------


class TestJobTypeCharacterize:
    """Tests for the CHARACTERIZE enum member."""

    def test_member_exists_with_validate_wire_value(self):
        # Wire value must remain ``"VALIDATE"`` — server compatibility.
        assert JobType.CHARACTERIZE.value == "VALIDATE"

    def test_member_in_all_values(self):
        values = [j.value for j in JobType]
        assert "VALIDATE" in values


# ---------------------------------------------------------------------------
# E2E tests for validation HTML feature (prod usher endpoints)
# ---------------------------------------------------------------------------


@pytest.fixture
def qoro_service(api_key):
    """Live ``QoroService`` for E2E tests in this module.

    Mirrors the fixture in ``test_qoro_service.py``; defined locally because
    fixtures in test modules don't propagate across files.
    """
    return QoroService(auth_token=api_key)


@pytest.mark.requires_api_key
class TestQoroServiceValidationHtmlE2E:
    """E2E regression tests for the prod ``validation_result`` endpoints.

    Pins the wire contract between divi and usher / Composer for the QUBO
    characterization feature against the real prod API. Every test reads
    from one of two class-scoped fixtures so the prod account sees at most
    two VALIDATE submissions per CI run.

    Endpoints exercised:
      - ``POST /api/job/init/``                         (job creation)
      - ``POST /api/job/<pk>/submit_qubo/``             (validation submit)
      - ``GET  /api/job/<pk>/validation_result/``       (JSON, via ``characterize_and_validate``)
      - ``GET  /api/job/<pk>/validation_result/html/``  (HTML)

    Deferred (not covered here — would need orchestration we can't drive
    from the public API):
      - 401 specifically for the validation path (covered generally by
        ``test_qoro_service.py::test_initialization_with_deactivated_token``,
        which fails earlier at ``QoroService`` construction).
      - 402 insufficient credits (would need a quota-exhausted test account).
      - 409 for RUNNING / FAILED VALIDATE jobs (would need a job mid-flight
        on the deterministic synchronous Composer path).
      - 502 Composer-down (can't trigger from outside).
      - 403 cross-user access (would need a second test account's job ID).
    """

    @pytest.fixture(scope="class")
    def completed_validation_job(self, api_key):
        """Submit one rich VALIDATE job and clean up afterwards.

        Class-scoped: shared by every test in this class. Configured with
        ``parameter_sweep=True`` and ``sensitivity=True`` so the response
        carries every optional field the wire contract supports — letting
        a single submission anchor the full-shape audit, the HTML render,
        recommendations parsing, and the fetch-by-job-id round-trip.
        """
        service = QoroService(auth_token=api_key)
        # 2-qubit QUBO with a concrete optimum at ``00``. Larger than this
        # adds prod cost without unlocking new wire-contract surface.
        problem = BinaryOptimizationProblem(np.array([[1.0, -1.0], [-1.0, 1.0]]))
        result = characterize_and_validate(
            problem,
            target_states=["00"],
            service=service,
            options=CharacterizationOptions(
                parameter_sweep=True,
                sensitivity=True,
                ansatz={"mixer": "x", "layers": 1},
            ),
        )
        try:
            yield service, result
        finally:
            try:
                service.delete_job(ExecutionResult(job_id=result.job_id))
            except Exception:
                # Don't mask a real test failure with cleanup noise.
                pass

    @pytest.fixture(scope="class")
    def hardness_only_job(self, api_key):
        """Submit one VALIDATE job with ``target_states=[]`` and no sweep.

        Exercises the lightest characterization path — Composer still
        produces a hardness analysis, but no parameter-sweep / per-state /
        sensitivity data. Confirms the wire contract degrades gracefully
        when the user only wants structural diagnostics.
        """
        service = QoroService(auth_token=api_key)
        problem = BinaryOptimizationProblem(np.array([[1.0, -1.0], [-1.0, 1.0]]))
        result = characterize_and_validate(
            problem,
            target_states=[],
            service=service,
            options=CharacterizationOptions(parameter_sweep=False, sensitivity=False),
        )
        try:
            yield service, result
        finally:
            try:
                service.delete_job(ExecutionResult(job_id=result.job_id))
            except Exception:
                pass

    def test_completed_job_full_shape(self, completed_validation_job):
        """Every user-facing field on a sweep+sensitivity result is the right
        shape and contains the keys downstream code reads.

        This is the wire-contract regression test: if usher / Composer
        renames or reshapes any of the fields divi exposes through
        ``CharacterizationResult``, this test fails. Offline tests can't
        catch shape drift because their mocks were authored against the
        expected shape, not the actual one.

        Asserts types and required keys, not specific numeric values
        (which would be flaky against the live service).
        """
        _, result = completed_validation_job

        # Top-level dataclass fields.
        assert isinstance(result.job_id, str) and result.job_id
        assert result.status == "COMPLETED"
        assert isinstance(result.created_at, str) and result.created_at
        assert isinstance(result.completed_at, str) and result.completed_at
        assert isinstance(result.report, dict) and result.report
        assert isinstance(result.hardness, dict) and result.hardness
        assert isinstance(result.recommendations, list)
        assert isinstance(result.html, str) and result.html

        # Hardness — structural metrics. ``difficulty`` drives the
        # recommendation rules; ``spectral_gap`` and ``condition_number``
        # are read by the ``summary()`` and ``_render()`` paths. All three
        # must be present for the rich display to render correctly.
        assert "difficulty" in result.hardness
        assert isinstance(result.hardness["difficulty"], str)
        assert "spectral_gap" in result.hardness
        assert _is_number(result.hardness["spectral_gap"])
        assert "condition_number" in result.hardness
        assert _is_number(result.hardness["condition_number"])

        # Quality / concentration — both should be numeric in [0, *) with
        # quality bounded at 100.
        assert _is_number(result.quality_score)
        assert 0.0 <= result.quality_score <= 100.0
        assert _is_number(result.concentration_ratio)
        assert result.concentration_ratio >= 0.0

        # Best parameters — the sweep's headline output. ``gamma`` and
        # ``beta`` are the keys the tutorial and ``display()`` both read.
        bp = result.best_parameters
        assert isinstance(bp, dict)
        assert "gamma" in bp and _is_number(bp["gamma"])
        assert "beta" in bp and _is_number(bp["beta"])

        # State probabilities — list of per-state rows, each carrying at
        # least ``state`` (str) and ``probability`` (numeric).
        sp = result.state_probabilities
        assert isinstance(sp, list) and sp
        for row in sp:
            assert isinstance(row.get("state"), str)
            assert _is_number(row.get("probability"))

        # Sensitivity — per-qubit rows. Some servers emit ``score``, some
        # emit ``sensitivity``; the renderer accepts either, so the test
        # accepts either.
        sens = result.sensitivity
        assert isinstance(sens, list) and sens
        for row in sens:
            assert "qubit" in row
            assert "score" in row or "sensitivity" in row

        # Recommendations — already pinned by
        # ``test_recommendations_have_structured_shape``; smoke-check here
        # that the field came through.
        for rec in result.recommendations:
            assert set(rec.keys()) >= {"level", "metric", "text", "html"}

    def test_recommendations_have_structured_shape(self, completed_validation_job):
        """``result.recommendations`` is a list of structured dicts.

        Regression for the divi/usher shape mismatch — usher returns
        ``recommendations`` as a top-level field with
        ``{level, metric, text, html}`` dicts; if ``_wrap_response`` ever
        drops it again or a property reads from ``self.report`` instead of
        the dataclass field, this test fails.

        Deliberately does **not** assert ``len(recs) > 0`` — whether any
        rules fire is server-side policy, not divi's contract.
        """
        _, result = completed_validation_job

        recs = result.recommendations
        assert isinstance(recs, list)
        for rec in recs:
            assert isinstance(rec, dict)
            assert set(rec.keys()) >= {"level", "metric", "text", "html"}
            assert rec["level"] in {"info", "warn", "action"}
            assert isinstance(rec["text"], str) and rec["text"]
            assert "<" not in rec["text"], "text field must be plain (no markup)"
            # ``html`` carries the same content as ``text``, optionally with
            # inline ``<strong>`` markup. Pinning markup *presence* would be
            # a server-style assumption, not a contract: a future rec rule
            # that emits a plain message would then fail spuriously here.
            assert isinstance(rec["html"], str) and rec["html"]

    def test_html_report_renders(self, completed_validation_job):
        """The cached ``result.html`` is a self-contained validation report doc.

        Uses the public ``CharacterizationResult.html`` field rather than
        re-calling ``_fetch_characterization_html`` so the test covers the
        same artifact users actually consume (e.g. via ``_repr_html_`` in
        a notebook), with no extra prod call.
        """
        _, result = completed_validation_job
        assert isinstance(result.html, str)
        # Root container of the validation-report template; present in
        # every successful render.
        assert 'class="qvr-root"' in result.html

    def test_fetch_by_job_id_returns_equivalent_result(
        self, completed_validation_job, qoro_service
    ):
        """``get_characterization_result(job_id)`` re-reads without resubmitting.

        Pins the read-only code path: a second call with only the prior
        job's ``job_id`` must return a ``CharacterizationResult`` whose
        data fields equal the original. Catches drift between the submit-
        then-fetch path (``characterize_and_validate``) and the pure-fetch path
        (``get_characterization_result``) in ``QoroService.characterize_and_validate``.

        The ``html`` field is excluded from the equality check by the
        dataclass (``compare=False``); we additionally assert it's
        non-empty on the refetch so the lazy HTML fetch isn't silently
        broken on this code path.
        """
        _, original = completed_validation_job

        refetched = get_characterization_result(
            job_id=original.job_id,
            service=qoro_service,
        )

        # Pin ``job_id`` explicitly so this test still catches drift even
        # if a future dataclass refactor marks any field ``compare=False``.
        assert refetched.job_id == original.job_id
        # Dataclass ``__eq__`` compares every field except ``html``
        # (marked ``compare=False``).
        assert refetched == original
        # And the second HTML fetch returns a non-empty document — the
        # lazy HTML fetch must work on the pure-fetch code path too.
        assert isinstance(refetched.html, str)
        assert 'class="qvr-root"' in refetched.html

    def test_hardness_only_mode_returns_valid_report(self, hardness_only_job):
        """``target_states=[]`` with no sweep / sensitivity still completes.

        The lightest configuration: hardness analysis only. Pins that
        Composer accepts an empty target list and that divi parses the
        resulting (degenerate) response without crashing on the missing
        sweep / sensitivity fields.
        """
        _, result = hardness_only_job

        assert result.status == "COMPLETED"
        # Hardness must be present — that's the whole point of this mode.
        assert isinstance(result.hardness, dict) and result.hardness
        assert "difficulty" in result.hardness

        # Composer runs its full pipeline regardless of the client's
        # ``CharacterizationOptions`` flags, so ``best_parameters`` and
        # ``sensitivity`` may still come back populated even though we
        # didn't request a sweep or sensitivity. The contract for *this
        # mode* is "if those fields are present, they have the same shape
        # as in the rich path" — not "they're absent."
        bp = result.best_parameters
        if bp is not None:
            assert isinstance(bp, dict)
            assert "gamma" in bp and _is_number(bp["gamma"])
            assert "beta" in bp and _is_number(bp["beta"])

        sens = result.sensitivity
        if sens is not None:
            assert isinstance(sens, list)
            for row in sens:
                assert "qubit" in row
                assert "score" in row or "sensitivity" in row

        # Recommendations may fire from hardness alone (e.g. a "hard"
        # difficulty triggers a rec); always a list of structured dicts
        # when any are emitted.
        assert isinstance(result.recommendations, list)

    def test_nonexistent_job_returns_404(self, qoro_service):
        """A bogus UUID surfaces as ``HTTPError 404`` through the public API.

        Exercises ``get_characterization_result`` rather than reaching into
        ``_make_request``: this is the real user-facing contract, and a
        future refactor that moves 4xx handling out of ``_make_request``
        into a higher layer must still produce the same exception type
        and status code on this path.
        """
        with pytest.raises(requests.HTTPError) as exc:
            get_characterization_result(
                job_id="00000000-0000-0000-0000-000000000000",
                service=qoro_service,
            )
        assert exc.value.response.status_code == 404
