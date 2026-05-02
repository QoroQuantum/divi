# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for QUBO/HUBO characterization feature.

Tests cover:
- Wire-format serialization from all supported QUBO/HUBO types
- CharacterizationResult dataclass properties and display
- QoroService.characterize submit + fetch flows (mocked HTTP)
- Top-level divi.backends.characterize convenience function
"""

from http import HTTPStatus

import numpy as np
import pytest

from divi.backends import CharacterizationOptions, characterize
from divi.backends._characterization import (
    CharacterizationResult,
    _serialize_qubo_for_wire,
)
from divi.backends._qoro_service import JobType, QoroService
from divi.qprog.problems._binary import BinaryOptimizationProblem

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

SAMPLE_RESPONSE = {
    "job_id": "abc-123",
    "status": "COMPLETED",
    "hardness": SAMPLE_HARDNESS,
    "report": SAMPLE_REPORT,
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
        QoroService._make_request = lambda self, *a, **kw: _EmptyResponse()
        try:
            service = QoroService(**config)
        finally:
            QoroService._make_request = original
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
        assert empty_result.recommendations == []

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

    def test_recommendations_property_exposes_server_field(self):
        recs = [
            {"level": "warn", "metric": "quality_score", "html": "<strong>X</strong>"}
        ]
        result = CharacterizationResult(
            job_id="r1",
            status="COMPLETED",
            report={"recommendations": recs},
            html="",
        )
        assert result.recommendations == recs

    def test_recommendations_property_empty_when_missing(self, empty_result):
        assert empty_result.recommendations == []

    def test_display_renders_server_recommendations(self, capsys):
        """``display()`` surfaces server-supplied recommendations."""
        result = CharacterizationResult(
            job_id="d1",
            status="COMPLETED",
            report={
                "recommendations": [
                    {
                        "level": "action",
                        "metric": "feasibility_rate",
                        "html": "Feasibility rate is only <strong>30%</strong>",
                    },
                ],
            },
            html="",
        )
        result.display()
        captured = capsys.readouterr().out
        assert "Recommendations" in captured
        assert "Feasibility rate is only" in captured
        # HTML <strong> stripped to plain text in the terminal output.
        assert "<strong>" not in captured


# ---------------------------------------------------------------------------
# QoroService.characterize (mocked HTTP)
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

        result = service.characterize(
            qubo={"0,0": -1.0, "0,1": 2.0},
            target_states=["01", "10"],
        )

        # service.characterize returns the raw API response dict;
        # the rich wrapper is built by divi.backends.characterize().
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

        service.characterize(
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

        service.characterize(
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

        result = service.characterize(job_id="abc-123")

        assert isinstance(result, dict)
        assert result["job_id"] == "abc-123"
        assert result["report"]["quality_score"] == 78.5

        # Only the GET to validation_result/, no init/submit traffic.
        assert mock_req.call_count == 1
        assert mock_req.call_args.args == ("get", "job/abc-123/validation_result/")

    def test_requires_qubo_or_job_id(self, qoro_service_factory):
        service = qoro_service_factory()
        with pytest.raises(ValueError, match="qubo.*or.*job_id"):
            service.characterize()


# ---------------------------------------------------------------------------
# Top-level characterize() function
# ---------------------------------------------------------------------------


class TestTopLevelCharacterize:
    """Tests for the divi.backends.characterize convenience function."""

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
        result = characterize(problem, target_states=["01", "10"], service=service)

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

        characterize(
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

        characterize(
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

        characterize(
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
