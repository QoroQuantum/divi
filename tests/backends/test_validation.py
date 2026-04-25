# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for QUBO/HUBO validation feature.

Tests cover:
- Wire-format serialization from all supported QUBO/HUBO types
- QUBOValidationResult dataclass properties and display
- QoroService.validate_qubo 3-step flow (mocked HTTP)
- QoroService.validate_hardness flow (mocked HTTP)
- QoroService.get_validation_result (mocked HTTP)
- Top-level divi.validate.validate convenience function
"""

from http import HTTPStatus

import dimod
import numpy as np
import pytest
import scipy.sparse as sps

from divi.backends._validation import (
    QUBOValidationResult,
    _serialize_qubo_for_wire,
)


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

SAMPLE_VALIDATION_RESPONSE = {
    "job_id": "abc-123",
    "status": "COMPLETED",
    "hardness": SAMPLE_HARDNESS,
    "report": SAMPLE_REPORT,
    "created_at": "2026-04-25T12:00:00Z",
    "completed_at": "2026-04-25T12:00:01Z",
}


@pytest.fixture
def validation_result():
    """A fully-populated QUBOValidationResult for testing."""
    return QUBOValidationResult(
        job_id="abc-123",
        status="COMPLETED",
        hardness=SAMPLE_HARDNESS,
        report=SAMPLE_REPORT,
        created_at="2026-04-25T12:00:00Z",
        completed_at="2026-04-25T12:00:01Z",
    )


@pytest.fixture
def empty_result():
    """A minimal QUBOValidationResult with no report or hardness."""
    return QUBOValidationResult(
        job_id="xyz-789",
        status="FAILED",
    )


@pytest.fixture
def qoro_service_factory():
    """Factory for creating mocked QoroService instances."""
    from divi.backends._qoro_service import QoroService

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
    """Tests for converting various QUBO/HUBO types to wire format."""

    def test_ndarray(self):
        Q = np.array([[-1.0, 2.0], [0.0, -1.0]])
        wire = _serialize_qubo_for_wire(Q)

        assert isinstance(wire, dict)
        assert all(isinstance(k, str) for k in wire)
        assert wire["0,0"] == -1.0
        assert wire["0,1"] == 2.0
        assert wire["1,1"] == -1.0
        # zero entries should be excluded
        assert "1,0" not in wire

    def test_ndarray_non_square_raises(self):
        with pytest.raises(ValueError, match="square"):
            _serialize_qubo_for_wire(np.array([[1, 2, 3], [4, 5, 6]]))

    def test_sparse_matrix(self):
        data = np.array([-1.0, 2.0, -1.0])
        row = np.array([0, 0, 1])
        col = np.array([0, 1, 1])
        Q = sps.coo_matrix((data, (row, col)), shape=(2, 2))

        wire = _serialize_qubo_for_wire(Q)
        assert wire["0,0"] == -1.0
        assert wire["0,1"] == 2.0
        assert wire["1,1"] == -1.0

    def test_bqm(self):
        bqm = dimod.BinaryQuadraticModel(
            {0: -1.0, 1: -1.0},
            {(0, 1): 2.0},
            vartype=dimod.BINARY,
        )
        wire = _serialize_qubo_for_wire(bqm)
        assert wire["0,0"] == -1.0
        assert wire["1,1"] == -1.0
        # dimod may iterate quadratic as (1,0) or (0,1) — accept either
        quad_key = "0,1" if "0,1" in wire else "1,0"
        assert wire[quad_key] == 2.0

    def test_binary_polynomial(self):
        poly = dimod.BinaryPolynomial(
            {frozenset({0}): -1.0, frozenset({0, 1}): 2.0, frozenset({1}): -1.0},
            dimod.BINARY,
        )
        wire = _serialize_qubo_for_wire(poly)
        assert wire["0"] == -1.0
        assert wire["1"] == -1.0
        assert wire["0,1"] == 2.0

    def test_dict_with_tuple_keys(self):
        qubo = {(0, 0): -1.0, (0, 1): 2.0, (1, 1): -1.0}
        wire = _serialize_qubo_for_wire(qubo)
        assert wire["0,0"] == -1.0
        assert wire["0,1"] == 2.0
        assert wire["1,1"] == -1.0

    def test_dict_with_string_keys_passthrough(self):
        qubo = {"0,0": -1.0, "0,1": 2.0}
        wire = _serialize_qubo_for_wire(qubo)
        assert wire == qubo

    def test_dict_skips_zero_coefficients(self):
        qubo = {(0, 0): -1.0, (0, 1): 0.0, (1, 1): -1.0}
        wire = _serialize_qubo_for_wire(qubo)
        assert "0,1" not in wire

    def test_list_converted_to_ndarray(self):
        Q = [[-1.0, 2.0], [0.0, -1.0]]
        wire = _serialize_qubo_for_wire(Q)
        assert wire["0,0"] == -1.0
        assert wire["0,1"] == 2.0

    def test_binary_optimization_problem(self):
        from divi.qprog.problems._binary import BinaryOptimizationProblem

        problem = BinaryOptimizationProblem(
            np.array([[-1.0, 2.0], [0.0, -1.0]])
        )
        wire = _serialize_qubo_for_wire(problem)
        assert isinstance(wire, dict)
        # Should contain entries for the nonzero terms
        assert len(wire) > 0

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError, match="Cannot serialize"):
            _serialize_qubo_for_wire(42)

    def test_ndarray_symmetric_folds_to_upper_triangular(self):
        """Symmetric ndarray must be folded: wire[i,j] = Q[i,j] + Q[j,i]."""
        Q = np.array([[-1.0, -2.0], [-2.0, -1.0]])  # symmetric
        wire = _serialize_qubo_for_wire(Q)
        assert wire["0,0"] == -1.0
        assert wire["1,1"] == -1.0
        assert wire["0,1"] == -4.0  # -2.0 + -2.0
        # No lower-triangular keys
        assert "1,0" not in wire

    def test_ndarray_upper_triangular_preserved(self):
        """Already upper-triangular ndarray should pass through correctly."""
        Q = np.array([[-1.0, 2.0], [0.0, -1.0]])
        wire = _serialize_qubo_for_wire(Q)
        assert wire["0,1"] == 2.0  # 2.0 + 0.0
        assert "1,0" not in wire

    def test_sparse_symmetric_folds_to_upper_triangular(self):
        """Symmetric sparse matrix must be folded to upper-triangular."""
        data = np.array([-1.0, -2.0, -2.0, -1.0])
        row = np.array([0, 0, 1, 1])
        col = np.array([0, 1, 0, 1])
        Q = sps.coo_matrix((data, (row, col)), shape=(2, 2))
        wire = _serialize_qubo_for_wire(Q)
        assert wire["0,0"] == -1.0
        assert wire["1,1"] == -1.0
        assert wire["0,1"] == -4.0  # -2.0 + -2.0
        assert "1,0" not in wire


# ---------------------------------------------------------------------------
# QUBOValidationResult
# ---------------------------------------------------------------------------


class TestQUBOValidationResult:
    """Tests for the validation result dataclass."""

    def test_quality_score(self, validation_result):
        assert validation_result.quality_score == 78.5

    def test_concentration_ratio(self, validation_result):
        assert validation_result.concentration_ratio == 3.2

    def test_best_parameters(self, validation_result):
        bp = validation_result.best_parameters
        assert bp["gamma"] == 1.2
        assert bp["beta"] == 0.7

    def test_penalty_recommendation(self, validation_result):
        assert validation_result.penalty_recommendation == 2.5

    def test_is_well_tuned(self, validation_result):
        assert validation_result.is_well_tuned is True

    def test_feasibility_rate(self, validation_result):
        assert validation_result.feasibility_rate == 0.85

    def test_approximation_ratio(self, validation_result):
        assert validation_result.approximation_ratio == 0.92

    def test_state_probabilities(self, validation_result):
        sp = validation_result.state_probabilities
        assert len(sp) == 4
        assert sp[0]["state"] == "01"

    def test_sensitivity(self, validation_result):
        sens = validation_result.sensitivity
        assert len(sens) == 2
        assert sens[0]["qubit"] == 0

    def test_empty_result_properties(self, empty_result):
        assert empty_result.quality_score is None
        assert empty_result.concentration_ratio is None
        assert empty_result.best_parameters is None
        assert empty_result.penalty_recommendation is None
        assert empty_result.is_well_tuned is None
        assert empty_result.feasibility_rate is None
        assert empty_result.state_probabilities is None

    def test_summary_contains_key_metrics(self, validation_result):
        s = validation_result.summary()
        assert "78.5" in s
        assert "3.2" in s
        assert "medium" in s
        assert "abc-123" in s

    def test_repr_equals_summary(self, validation_result):
        assert repr(validation_result) == validation_result.summary()

    def test_repr_html_returns_string(self, validation_result):
        html = validation_result._repr_html_()
        assert isinstance(html, str)
        assert "qvr-root" in html
        assert "Quality Score" in html
        assert "78.5" in html

    def test_repr_html_hardness_section(self, validation_result):
        html = validation_result._repr_html_()
        assert "Hardness Analysis" in html
        assert "Medium" in html  # difficulty badge
        assert "Spectral Gap" in html

    def test_repr_html_state_probabilities_table(self, validation_result):
        html = validation_result._repr_html_()
        assert "State Probabilities" in html
        assert "01" in html
        assert "✓" in html  # target marker
        assert "✗" in html  # non-target marker

    def test_repr_html_best_parameters(self, validation_result):
        html = validation_result._repr_html_()
        assert "Best Parameters" in html
        assert "1.2" in html  # gamma

    def test_repr_html_penalty_tuning(self, validation_result):
        html = validation_result._repr_html_()
        assert "Penalty Tuning" in html
        assert "2.5" in html  # lambda

    def test_repr_html_recommendations(self, validation_result):
        html = validation_result._repr_html_()
        assert "Recommendations" in html

    def test_repr_html_empty_result(self, empty_result):
        html = empty_result._repr_html_()
        assert isinstance(html, str)
        assert "qvr-root" in html
        assert "FAILED" in html

    def test_frozen(self, validation_result):
        with pytest.raises(AttributeError):
            validation_result.status = "FAILED"

    def test_recommendations_low_quality(self):
        result = QUBOValidationResult(
            job_id="low-q",
            status="COMPLETED",
            report={"quality_score": 20.0},
        )
        html = result._repr_html_()
        assert "Low quality" in html

    def test_recommendations_low_feasibility(self):
        result = QUBOValidationResult(
            job_id="low-f",
            status="COMPLETED",
            report={"feasibility_rate": 0.3},
        )
        html = result._repr_html_()
        assert "30%" in html

    def test_recommendations_hard_problem(self):
        result = QUBOValidationResult(
            job_id="hard-p",
            status="COMPLETED",
            hardness={"difficulty": "hard"},
            report={},
        )
        html = result._repr_html_()
        assert "hard" in html.lower()


# ---------------------------------------------------------------------------
# QoroService.validate_qubo (mocked HTTP)
# ---------------------------------------------------------------------------


class TestQoroServiceValidateQubo:
    """Tests for the 3-step validation flow in QoroService."""

    def test_validate_qubo_full_flow(self, mocker, qoro_service_factory):
        service = qoro_service_factory()

        mock_init = mocker.MagicMock()
        mock_init.json.return_value = {"job_id": "val-job-123"}

        mock_result = mocker.MagicMock()
        mock_result.json.return_value = SAMPLE_VALIDATION_RESPONSE

        # init + result go through _make_request
        mock_req = mocker.patch.object(
            service,
            "_make_request",
            side_effect=[mock_init, mock_result],
        )

        # submit goes through requests.post directly (non-retrying)
        mock_submit = mocker.MagicMock()
        mock_submit.status_code = HTTPStatus.OK
        mock_submit.json.return_value = {"status": "COMPLETED"}
        mock_post = mocker.patch("divi.backends._qoro_service.requests.post", return_value=mock_submit)

        result = service.validate_qubo(
            qubo={"0,0": -1.0, "0,1": 2.0},
            target_states=["01", "10"],
        )

        assert isinstance(result, QUBOValidationResult)
        assert result.status == "COMPLETED"
        assert result.quality_score == 78.5
        assert result.hardness["difficulty"] == "medium"

        # Verify init + result calls via _make_request
        assert mock_req.call_count == 2
        init_call = mock_req.call_args_list[0]
        assert init_call.args == ("post", "job/init/")
        assert init_call.kwargs["json"]["job_type"] == "VALIDATE"

        # Verify submit call via requests.post
        assert mock_post.call_count == 1
        submit_payload = mock_post.call_args.kwargs["json"]
        assert submit_payload["qubo"] == {"0,0": -1.0, "0,1": 2.0}
        assert submit_payload["target_states"] == ["01", "10"]

    def test_validate_qubo_with_options(self, mocker, qoro_service_factory):
        service = qoro_service_factory()

        mock_init = mocker.MagicMock()
        mock_init.json.return_value = {"job_id": "val-opt-123"}
        mock_result = mocker.MagicMock()
        mock_result.json.return_value = SAMPLE_VALIDATION_RESPONSE

        mock_req = mocker.patch.object(
            service,
            "_make_request",
            side_effect=[mock_init, mock_result],
        )

        mock_submit = mocker.MagicMock(status_code=HTTPStatus.OK)
        mock_post = mocker.patch("divi.backends._qoro_service.requests.post", return_value=mock_submit)

        options = {
            "ansatz": {"mixer": "x", "layers": 1},
            "analysis": {"sensitivity": True, "parameter_sweep": True},
        }

        service.validate_qubo(
            qubo={"0,0": -1.0},
            target_states=["0"],
            options=options,
        )

        submit_payload = mock_post.call_args.kwargs["json"]
        assert submit_payload["options"] == options


class TestQoroServiceValidateHardness:
    """Tests for the hardness-only shortcut."""

    def test_validate_hardness_returns_dict(self, mocker, qoro_service_factory):
        service = qoro_service_factory()

        mock_init = mocker.MagicMock()
        mock_init.json.return_value = {"job_id": "hard-123"}
        mock_submit = mocker.MagicMock(status_code=HTTPStatus.OK)
        mock_result = mocker.MagicMock()
        mock_result.json.return_value = {
            "hardness": SAMPLE_HARDNESS,
            "status": "COMPLETED",
        }

        mocker.patch.object(
            service,
            "_make_request",
            side_effect=[mock_init, mock_submit, mock_result],
        )

        result = service.validate_hardness(
            qubo={"0,0": -1.0, "0,1": 2.0},
            n_qubits=2,
        )

        assert isinstance(result, dict)
        assert result["difficulty"] == "medium"
        assert result["spectral_gap"] == 0.35

    def test_validate_hardness_submits_empty_targets(
        self, mocker, qoro_service_factory
    ):
        service = qoro_service_factory()

        mock_init = mocker.MagicMock()
        mock_init.json.return_value = {"job_id": "hard-456"}
        mock_submit = mocker.MagicMock(status_code=HTTPStatus.OK)
        mock_result = mocker.MagicMock()
        mock_result.json.return_value = {"hardness": {}, "status": "COMPLETED"}

        mock_req = mocker.patch.object(
            service,
            "_make_request",
            side_effect=[mock_init, mock_submit, mock_result],
        )

        service.validate_hardness(qubo={"0,0": -1.0})

        submit_payload = mock_req.call_args_list[1].kwargs["json"]
        assert submit_payload["target_states"] == []
        assert submit_payload["options"]["analysis"]["hardness_only"] is True


class TestQoroServiceGetValidationResult:
    """Tests for fetching existing validation results."""

    def test_get_validation_result(self, mocker, qoro_service_factory):
        service = qoro_service_factory()

        mock_resp = mocker.MagicMock()
        mock_resp.json.return_value = SAMPLE_VALIDATION_RESPONSE

        mocker.patch.object(service, "_make_request", return_value=mock_resp)

        result = service.get_validation_result("abc-123")

        assert isinstance(result, QUBOValidationResult)
        assert result.job_id == "abc-123"
        assert result.quality_score == 78.5


# ---------------------------------------------------------------------------
# Top-level validate() function
# ---------------------------------------------------------------------------


class TestTopLevelValidate:
    """Tests for divi.validate.validate convenience function."""

    def test_validate_with_ndarray(self, mocker, qoro_service_factory):
        service = qoro_service_factory()

        mock_init = mocker.MagicMock()
        mock_init.json.return_value = {"job_id": "top-123"}
        mock_result = mocker.MagicMock()
        mock_result.json.return_value = SAMPLE_VALIDATION_RESPONSE

        mocker.patch.object(
            service,
            "_make_request",
            side_effect=[mock_init, mock_result],
        )
        mock_submit = mocker.MagicMock(status_code=HTTPStatus.OK)
        mocker.patch("divi.backends._qoro_service.requests.post", return_value=mock_submit)

        from divi.validate import validate

        Q = np.array([[-1.0, 2.0], [0.0, -1.0]])
        result = validate(Q, target_states=["01", "10"], service=service)

        assert isinstance(result, QUBOValidationResult)
        assert result.quality_score == 78.5

    def test_validate_with_options(self, mocker, qoro_service_factory):
        service = qoro_service_factory()

        mock_init = mocker.MagicMock()
        mock_init.json.return_value = {"job_id": "top-opt-123"}
        mock_result = mocker.MagicMock()
        mock_result.json.return_value = SAMPLE_VALIDATION_RESPONSE

        mocker.patch.object(
            service,
            "_make_request",
            side_effect=[mock_init, mock_result],
        )
        mock_submit = mocker.MagicMock(status_code=HTTPStatus.OK)
        mock_post = mocker.patch("divi.backends._qoro_service.requests.post", return_value=mock_submit)

        from divi.validate import validate

        validate(
            np.array([[-1.0, 2.0], [0.0, -1.0]]),
            target_states=["01"],
            service=service,
            sensitivity=True,
            parameter_sweep=True,
            auto_tune=True,
            gamma=1.0,
            beta=0.5,
        )

        submit_payload = mock_post.call_args.kwargs["json"]
        options = submit_payload["options"]
        assert options["analysis"]["sensitivity"] is True
        assert options["analysis"]["parameter_sweep"] is True
        assert options["analysis"]["auto_tune"] is True
        assert options["analysis"]["gamma"] == 1.0
        assert options["analysis"]["beta"] == 0.5

    def test_validate_with_penalty_qubos(self, mocker, qoro_service_factory):
        service = qoro_service_factory()

        mock_init = mocker.MagicMock()
        mock_init.json.return_value = {"job_id": "pen-123"}
        mock_result = mocker.MagicMock()
        mock_result.json.return_value = SAMPLE_VALIDATION_RESPONSE

        mocker.patch.object(
            service,
            "_make_request",
            side_effect=[mock_init, mock_result],
        )
        mock_submit = mocker.MagicMock(status_code=HTTPStatus.OK)
        mock_post = mocker.patch("divi.backends._qoro_service.requests.post", return_value=mock_submit)

        from divi.validate import validate

        cost_q = np.array([[-1.0, 0.0], [0.0, -1.0]])
        pen_q = np.array([[0.0, 2.0], [0.0, 0.0]])

        validate(
            np.array([[-1.0, 2.0], [0.0, -1.0]]),
            target_states=["01"],
            service=service,
            cost_qubo=cost_q,
            penalty_qubo=pen_q,
        )

        submit_payload = mock_post.call_args.kwargs["json"]
        options = submit_payload["options"]
        assert "cost_qubo" in options
        assert "penalty_qubo" in options
        assert options["cost_qubo"]["0,0"] == -1.0


# ---------------------------------------------------------------------------
# JobType enum
# ---------------------------------------------------------------------------


class TestJobTypeValidate:
    """Tests for the VALIDATE enum value."""

    def test_validate_value_exists(self):
        from divi.backends._qoro_service import JobType

        assert JobType.VALIDATE.value == "VALIDATE"

    def test_validate_in_all_values(self):
        from divi.backends._qoro_service import JobType

        values = [j.value for j in JobType]
        assert "VALIDATE" in values
