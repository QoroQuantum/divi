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

import dimod
import numpy as np
import pytest
import requests

from divi.backends import (
    CharacterizationOptions,
    CharacterizationResult,
    ExecutionResult,
    JobType,
    QoroService,
    characterize_and_validate,
    get_characterization_result,
)
from divi.backends._characterization import (
    _FACTORED_PROBE_MIN_QUBITS,
    _TRUNCATED_PAYLOAD_BUDGET_BYTES,
    _TRUNCATED_REL_ERROR_MAX,
    _payload_size,
    _qubo_to_dense,
    _serialize_qubo_factored,
    _serialize_qubo_for_wire,
    _serialize_qubo_legacy,
    _wrap_response,
)
from divi.qprog.problems import BinaryOptimizationProblem


def _is_number(val) -> bool:
    """``isinstance(val, (int, float))`` excluding ``bool``.

    ``bool`` is a subclass of ``int``; without this guard, a boolean-valued
    field would silently pass numeric type assertions.
    """
    return isinstance(val, (int, float)) and not isinstance(val, bool)


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


class TestSerializeQuboForWire:
    """Wire-format dispatch between legacy and factored encodings."""

    def test_small_qubo_uses_legacy(self):
        """``n`` below the probe threshold serializes as the comma-key dict."""
        problem = BinaryOptimizationProblem(np.array([[-1.0, 2.0], [0.0, -1.0]]))
        wire = _serialize_qubo_for_wire(problem)
        assert wire == {"0": -1.0, "0,1": 2.0, "1": -1.0}

    def test_zero_coefficients_skipped(self):
        problem = BinaryOptimizationProblem({(0,): -1.0, (0, 1): 0.0, (1,): -1.0})
        wire = _serialize_qubo_for_wire(problem)
        assert "0,1" not in wire

    def test_hubo_routes_to_legacy(self):
        """Any term of degree > 2 forces legacy regardless of qubit count."""
        n = _FACTORED_PROBE_MIN_QUBITS + 4
        terms = {(i,): -1.0 for i in range(n)}
        terms[(0, 1, 2)] = 3.0
        problem = BinaryOptimizationProblem(terms)
        wire = _serialize_qubo_for_wire(problem)
        assert isinstance(wire, dict) and "_format" not in wire
        assert wire["0,1,2"] == 3.0

    def test_low_rank_qubo_uses_factored(self):
        """``Q = u·uᵀ`` serializes as a rank-1 factored payload."""
        n = _FACTORED_PROBE_MIN_QUBITS
        rng = np.random.default_rng(seed=0)
        u = rng.standard_normal(n)
        Q = np.outer(u, u)
        terms = {}
        for i in range(n):
            terms[(i,)] = float(Q[i, i])
            for j in range(i + 1, n):
                terms[(i, j)] = float(Q[i, j] + Q[j, i])
        problem = BinaryOptimizationProblem(terms)
        wire = _serialize_qubo_for_wire(problem)
        assert wire["_format"] == "factored_v1"
        assert wire["n"] == n
        assert wire["k"] == 1
        assert len(wire["F"]) == 2 * n * wire["k"] * 8
        assert len(wire["diag"]) == 2 * n * 8
        # ``u·uᵀ`` is positive semidefinite, so the single sign is +1.
        assert wire["signs"] == [1.0]

    def test_full_rank_random_qubo_falls_back_to_legacy(self):
        """Random dense QUBOs are smaller as legacy and the dispatcher honours that."""
        n = _FACTORED_PROBE_MIN_QUBITS
        rng = np.random.default_rng(seed=1)
        terms = {}
        for i in range(n):
            terms[(i,)] = float(rng.standard_normal())
            for j in range(i + 1, n):
                terms[(i, j)] = float(rng.standard_normal())
        problem = BinaryOptimizationProblem(terms)
        wire = _serialize_qubo_for_wire(problem)
        assert isinstance(wire, dict) and "_format" not in wire

    def test_pure_diagonal_qubo_encodes_with_k_zero(self):
        """A diagonal-only QUBO yields ``k=0`` via the factored helper.

        Exercises the encoder directly because the top-level dispatcher
        may still pick legacy when the diagonal coefficients have short
        JSON representations.
        """
        n = _FACTORED_PROBE_MIN_QUBITS
        rng = np.random.default_rng(seed=4)
        coeffs = rng.standard_normal(n).astype(np.float64)
        problem = BinaryOptimizationProblem({(i,): float(coeffs[i]) for i in range(n)})
        wire = _serialize_qubo_factored(problem.canonical_problem)
        assert wire["k"] == 0
        assert wire["signs"] == []
        diag = np.frombuffer(bytes.fromhex(wire["diag"]), dtype=np.float64)
        np.testing.assert_allclose(diag, coeffs)

    def test_factored_decoding_reconstructs_qubo(self):
        """``F · diag(signs) · Fᵀ + diag(diag)`` round-trips back to ``Q``."""
        n = _FACTORED_PROBE_MIN_QUBITS
        rng = np.random.default_rng(seed=2)
        U = rng.standard_normal((n, 5))
        signs = rng.choice([-1.0, 1.0], size=5)
        Q = U @ np.diag(signs) @ U.T
        terms: dict = {}
        for i in range(n):
            if Q[i, i] != 0:
                terms[(i,)] = float(Q[i, i])
            for j in range(i + 1, n):
                if Q[i, j] != 0:
                    terms[(i, j)] = float(Q[i, j] + Q[j, i])
        problem = BinaryOptimizationProblem(terms)
        wire = _serialize_qubo_for_wire(problem)
        assert wire["_format"] == "factored_v1"

        F = np.frombuffer(bytes.fromhex(wire["F"]), dtype=np.float64).reshape(
            n, wire["k"]
        )
        diag = np.frombuffer(bytes.fromhex(wire["diag"]), dtype=np.float64)
        signs_arr = np.asarray(wire["signs"], dtype=np.float64)
        Q_decoded = F @ np.diag(signs_arr) @ F.T + np.diag(diag)
        # Tolerance tracks ``eigh``'s ``O(n · eps · ‖Q‖)`` backward error.
        np.testing.assert_allclose(
            Q_decoded, Q, rtol=1e-10, atol=1e-12 * max(1.0, float(np.abs(Q).max()))
        )

    def test_legacy_helper_directly(self):
        """``_serialize_qubo_legacy`` accepts HUBO of any degree."""
        problem = BinaryOptimizationProblem({(0,): -1.0, (0, 1): 2.0, (0, 1, 2): 3.0})
        wire = _serialize_qubo_legacy(problem.canonical_problem)
        assert wire == {"0": -1.0, "0,1": 2.0, "0,1,2": 3.0}

    def test_factored_helper_picks_smaller_k(self):
        """Encoder picks the residual=0 decomposition for a rank-1 ``u·uᵀ``."""
        rng = np.random.default_rng(seed=3)
        n = 16
        u = rng.standard_normal(n)
        Q = np.outer(u, u)
        terms = {}
        for i in range(n):
            terms[(i,)] = float(Q[i, i])
            for j in range(i + 1, n):
                terms[(i, j)] = float(Q[i, j] + Q[j, i])
        problem = BinaryOptimizationProblem(terms)
        wire = _serialize_qubo_factored(problem.canonical_problem)
        assert wire["_format"] == "factored_v1"
        assert wire["n"] == n
        assert wire["k"] == 1


class TestSerializeQuboMidScale:
    """Encoder behavior on middle-range QUBOs (n=128..512).

    Covers two real-world flavors:

    * **Penalty-decorated low-rank** — small structural rank plus a
      dense cardinality penalty. The full ``Q`` matrix stays low rank,
      so the lossless candidate already gives ``k ≪ n``.
    * **Penalty-plateau with diagonal bias** — same structure plus
      independent linear/diagonal contributions. The full ``Q`` is now
      effectively full rank, forcing the truncated (lossy) path.
    """

    @staticmethod
    def _terms_from_dense(Q):
        n = Q.shape[0]
        terms = {}
        for i in range(n):
            if Q[i, i] != 0:
                terms[(i,)] = float(Q[i, i])
            for j in range(i + 1, n):
                v = Q[i, j] + Q[j, i]
                if v != 0:
                    terms[(i, j)] = float(v)
        return terms

    @staticmethod
    def _structural_plus_cardinality(n, n_structural=8, cardinality=10.0, seed=0):
        rng = np.random.default_rng(seed)
        F = rng.standard_normal((n, n_structural)) * 100.0
        signs = rng.choice([-1.0, 1.0], size=n_structural)
        structural = F @ np.diag(signs) @ F.T
        Q = structural + cardinality * np.ones((n, n))
        return 0.5 * (Q + Q.T)

    @staticmethod
    def _esg_portfolio_qubo(n, seed=0):
        """Portfolio QUBO mirroring the production failure mode.

        Construction (lifted from a 1000-asset ESG portfolio benchmark):
        low-rank covariance, returns, ESG scores, sector indicators, and a
        cardinality penalty. The cardinality linearization on the diagonal
        scales ``|Q|_max`` with ``λ · target_count``, making relative
        reconstruction error robust to truncation at this scale.
        """
        rng = np.random.default_rng(seed)
        F_cov = rng.standard_normal((n, 50)) * 0.1
        Sigma = F_cov @ F_cov.T
        mu = rng.standard_normal(n) * 0.05
        sectors = rng.integers(0, 8, size=n)
        esg = rng.standard_normal(n) * 0.1
        lambda_card = 100.0
        target_count = max(1, n // 20)
        lambda_sector = 10.0
        Q = -1.0 * Sigma
        np.fill_diagonal(Q, -mu + esg - 2 * lambda_card * target_count)
        Q += lambda_card * np.ones_like(Q)
        for s in range(8):
            mask = (sectors == s).astype(np.float64)
            Q += lambda_sector * np.outer(mask, mask)
        return 0.5 * (Q + Q.T)

    @pytest.mark.parametrize("n", [128, 256, 512])
    def test_low_rank_penalty_qubo_picks_lossless_factored(self, n):
        """Penalty + low-rank structure → lossless factored at ``k = rank(Q)``.

        Spans the size range typical of mid-scale optimization problems
        (128..512 qubits). The full Q matrix is low rank, so the
        lossless ``residual=0`` candidate wins outright.
        """
        Q = self._structural_plus_cardinality(n, n_structural=8)
        problem = BinaryOptimizationProblem(self._terms_from_dense(Q))
        wire = _serialize_qubo_for_wire(problem)
        assert wire["_format"] == "factored_v1"
        # Q has rank ≤ structural rank + 1 (cardinality direction).
        assert wire["k"] <= 16
        # Reconstruction is at eigh-noise level (lossless).
        F = np.frombuffer(bytes.fromhex(wire["F"]), dtype=np.float64).reshape(
            n, wire["k"]
        )
        diag = np.frombuffer(bytes.fromhex(wire["diag"]), dtype=np.float64)
        signs_arr = np.asarray(wire["signs"], dtype=np.float64)
        Q_recon = F @ np.diag(signs_arr) @ F.T + np.diag(diag)
        rel_err = np.abs(Q_recon - Q).max() / np.abs(Q).max()
        assert rel_err < 1e-10

    def test_truncated_preserves_structural_eigenvalues(self):
        """Truncated payload keeps every off-diagonal eigenvalue above 1% of ``|λ_max|``."""
        n = 1000
        Q = self._esg_portfolio_qubo(n)
        problem = BinaryOptimizationProblem(self._terms_from_dense(Q))
        wire = _serialize_qubo_for_wire(problem)
        assert wire["_format"] == "factored_v1"
        assert 2 <= wire["k"] <= 32

    def test_esg_portfolio_qubo_uses_truncated_factored(self):
        """A production-scale ESG QUBO (n=1000) ships under the payload budget.

        Without truncation the lossless factored payload would be ~16 MB
        and the legacy payload ~14 MB — both rejected by typical 10 MB
        nginx body-size limits. The truncated candidate cuts at the
        cardinality direction's spectral gap, absorbs the dropped
        eigencomponents into the diagonal, and stays inside the
        documented error tolerance.
        """
        n = 1000
        Q = self._esg_portfolio_qubo(n)
        problem = BinaryOptimizationProblem(self._terms_from_dense(Q))
        wire = _serialize_qubo_for_wire(problem)
        assert wire["_format"] == "factored_v1"
        assert _payload_size(wire) < _TRUNCATED_PAYLOAD_BUDGET_BYTES

        F = np.frombuffer(bytes.fromhex(wire["F"]), dtype=np.float64).reshape(
            n, wire["k"]
        )
        diag = np.frombuffer(bytes.fromhex(wire["diag"]), dtype=np.float64)
        signs_arr = np.asarray(wire["signs"], dtype=np.float64)
        Q_recon = F @ np.diag(signs_arr) @ F.T + np.diag(diag)

        Q_max = float(np.abs(Q).max())
        rel_err = float(np.abs(Q_recon - Q).max() / Q_max)
        assert rel_err < _TRUNCATED_REL_ERROR_MAX
        # Diagonal absorption keeps Q[i,i] exact at eigh-noise level.
        diag_err = float(np.abs(np.diagonal(Q_recon - Q)).max())
        assert diag_err < 1e-9 * Q_max

    @pytest.mark.parametrize("n", [64, 128, 256, 512])
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_random_low_rank_plus_noise_produces_valid_payload(self, n, seed):
        """Encoder output is structurally valid across random inputs.

        Spans the realistic problem-size range (64..512) at multiple seeds
        to catch failure modes the focused unit tests miss — invalid
        signs, NaN/Inf in hex blobs, dimension mismatches, oversized
        payloads, or relative errors above the documented bound.
        """
        rng = np.random.default_rng(seed)
        rank = max(1, n // 8)
        U = rng.standard_normal((n, rank))
        Q = U @ U.T + 0.01 * rng.standard_normal((n, n))
        Q = 0.5 * (Q + Q.T)
        problem = BinaryOptimizationProblem(self._terms_from_dense(Q))
        wire = _serialize_qubo_for_wire(problem)

        # Either format is acceptable; whichever was chosen must be
        # structurally well-formed.
        if isinstance(wire, dict) and wire.get("_format") == "factored_v1":
            assert wire["n"] == n
            assert 0 <= wire["k"] <= n
            assert all(s in (-1.0, 1.0) for s in wire["signs"])
            F_bytes = bytes.fromhex(wire["F"])
            diag_bytes = bytes.fromhex(wire["diag"])
            assert len(F_bytes) == n * wire["k"] * 8
            assert len(diag_bytes) == n * 8
            F = np.frombuffer(F_bytes, dtype=np.float64).reshape(n, wire["k"])
            diag = np.frombuffer(diag_bytes, dtype=np.float64)
            assert np.isfinite(F).all() and np.isfinite(diag).all()
            # Reconstruction must obey the documented relative-error bound.
            signs_arr = np.asarray(wire["signs"], dtype=np.float64)
            Q_recon = F @ np.diag(signs_arr) @ F.T + np.diag(diag)
            Q_max = float(np.abs(Q).max())
            if Q_max > 0:
                rel_err = float(np.abs(Q_recon - Q).max() / Q_max)
                assert rel_err < _TRUNCATED_REL_ERROR_MAX
        else:
            # Legacy comma-key dict — must be exact.
            assert all(isinstance(k, str) for k in wire.keys())
            assert all(isinstance(v, (int, float)) for v in wire.values())

    def test_smooth_decay_no_gap_avoids_lossy_path(self):
        """Random full-rank QUBO with no spectral gap stays lossless.

        Without a clean gap the truncated candidate is either rejected
        (relative error exceeds the threshold) or beaten on byte size by
        the lossless paths; in both cases the dispatcher returns a
        lossless payload.
        """
        rng = np.random.default_rng(seed=11)
        n = _FACTORED_PROBE_MIN_QUBITS
        terms = {}
        for i in range(n):
            terms[(i,)] = float(rng.standard_normal())
            for j in range(i + 1, n):
                terms[(i, j)] = float(rng.standard_normal())
        problem = BinaryOptimizationProblem(terms)
        wire = _serialize_qubo_for_wire(problem)
        if isinstance(wire, dict) and wire.get("_format") == "factored_v1":
            n_w = wire["n"]
            F = np.frombuffer(bytes.fromhex(wire["F"]), dtype=np.float64).reshape(
                n_w, wire["k"]
            )
            diag = np.frombuffer(bytes.fromhex(wire["diag"]), dtype=np.float64)
            signs_arr = np.asarray(wire["signs"], dtype=np.float64)
            Q = _qubo_to_dense(problem.canonical_problem)
            Q_recon = F @ np.diag(signs_arr) @ F.T + np.diag(diag)
            assert np.abs(Q_recon - Q).max() < 1e-8 * max(1.0, np.abs(Q).max())


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

    def test_factored_payload_auto_attaches_n_qubits(
        self, mocker, qoro_service_factory
    ):
        """Factored submissions ship ``options['n_qubits']`` alongside the payload."""
        service = qoro_service_factory()

        mock_init = mocker.MagicMock()
        mock_init.json.return_value = {"job_id": "fac-123"}
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

        # Rank-1 dense QUBO at the probe threshold encodes with k=1,
        # forcing the dispatcher to pick factored.
        n = _FACTORED_PROBE_MIN_QUBITS
        rng = np.random.default_rng(seed=5)
        u = rng.standard_normal(n)
        Q = np.outer(u, u)
        terms = {(i,): float(Q[i, i]) for i in range(n)}
        for i in range(n):
            for j in range(i + 1, n):
                terms[(i, j)] = float(Q[i, j] + Q[j, i])

        characterize_and_validate(
            BinaryOptimizationProblem(terms),
            target_states=[],
            service=service,
        )

        submit_json = mock_req.call_args_list[1].kwargs["json"]
        assert submit_json["qubo"]["_format"] == "factored_v1"
        assert submit_json["options"]["n_qubits"] == n

    def test_factored_n_qubits_does_not_clobber_user_option(
        self, mocker, qoro_service_factory
    ):
        """A user-supplied ``options['n_qubits']`` survives the auto-attach pass."""
        service = qoro_service_factory()

        mock_init = mocker.MagicMock()
        mock_init.json.return_value = {"job_id": "fac-2"}
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

        n = _FACTORED_PROBE_MIN_QUBITS
        rng = np.random.default_rng(seed=5)
        u = rng.standard_normal(n)
        Q = np.outer(u, u)
        terms = {(i,): float(Q[i, i]) for i in range(n)}
        for i in range(n):
            for j in range(i + 1, n):
                terms[(i, j)] = float(Q[i, j] + Q[j, i])

        options = CharacterizationOptions()
        mocker.patch.object(options, "_to_wire", return_value={"n_qubits": 999})

        characterize_and_validate(
            BinaryOptimizationProblem(terms),
            target_states=[],
            service=service,
            options=options,
        )

        submit_json = mock_req.call_args_list[1].kwargs["json"]
        assert submit_json["options"]["n_qubits"] == 999

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
        # n=2 routes to legacy. Diagonal entries serialize as single-index keys.
        assert options["cost_qubo"]["0"] == -1.0


class TestJobTypeCharacterize:
    """Tests for the CHARACTERIZE enum member."""

    def test_member_exists_with_validate_wire_value(self):
        # Wire value must remain ``"VALIDATE"`` — server compatibility.
        assert JobType.CHARACTERIZE.value == "VALIDATE"

    def test_member_in_all_values(self):
        values = [j.value for j in JobType]
        assert "VALIDATE" in values


@pytest.fixture
def qoro_service(api_key):
    """Live ``QoroService`` for E2E tests in this module.

    Mirrors the fixture in ``test_qoro_service.py``; defined locally because
    fixtures in test modules don't propagate across files.
    """
    return QoroService(auth_token=api_key)


# Report/hardness matching the redesigned canonical composer-service schema.
CANONICAL_REPORT = {
    "formulation_quality": 72.0,
    "target_achievability": 61.0,
    "concentration_ratio": 1.4,
    "approximation_ratio": 0.87,
    "feasibility_rate": 0.9,
    "verdict": {
        "verdict": "promising",
        "rationale": "QAOA AR 0.87 exceeds the classical baseline 0.80.",
        "qaoa_approximation_ratio": 0.87,
    },
    "classical_baseline": {
        "greedy_energy": -2.0,
        "sa_energy": -2.0,
        "best_energy": -2.0,
        "exact_ground_energy": -2.0,
        "distinct_optima": 1,
    },
    "constraint_diagnostics": [
        {
            "index": 0,
            "type": "max_cardinality",
            "violation_rate": 0.1,
            "is_redundant": False,
        }
    ],
    "penalty_lambda_min_feasible": 2.0,
    "penalty_lambda_safe": 3.5,
    "penalty_recommendation": 3.5,
    "penalty_tuning": {"is_well_tuned": True, "optimal_lambda": 2.0},
}
CANONICAL_HARDNESS = {
    "difficulty": "moderate",
    "cost_gap": 1.0,
    "cost_gap_normalized": 0.33,
    "ground_state_degeneracy": 2,
    "treewidth_estimate": 2,
    "frustration_index": 0.25,
    "matrix_spectral_gap": 0.0,
}


class TestRedesignedResultFields:
    """Client reads the canonical redesigned schema (verdict, AR, cost gap)."""

    def _result(self):
        return CharacterizationResult(
            job_id="c1",
            status="COMPLETED",
            hardness=CANONICAL_HARDNESS,
            report=CANONICAL_REPORT,
            html="",
        )

    def test_quality_prefers_target_achievability(self):
        assert self._result().quality_score == 61.0

    def test_formulation_and_target_quality(self):
        r = self._result()
        assert r.formulation_quality == 72.0
        assert r.target_achievability == 61.0

    def test_approximation_ratio_is_real(self):
        assert self._result().approximation_ratio == 0.87

    def test_verdict(self):
        assert self._result().verdict["verdict"] == "promising"

    def test_classical_baseline(self):
        assert self._result().classical_baseline["best_energy"] == -2.0

    def test_hardness_cost_spectrum_fields(self):
        r = self._result()
        assert r.cost_gap == 1.0
        assert r.ground_state_degeneracy == 2
        assert r.treewidth_estimate == 2
        assert r.frustration_index == 0.25

    def test_penalty_interval(self):
        r = self._result()
        assert r.penalty_lambda_min_feasible == 2.0
        assert r.penalty_lambda_safe == 3.5

    def test_constraint_diagnostics(self):
        assert self._result().constraint_diagnostics[0]["type"] == "max_cardinality"

    def test_summary_shows_verdict_and_ar(self):
        s = self._result().summary()
        assert "promising" in s
        assert "0.87" in s

    def test_summary_shows_penalty_safe_range_and_diagnostics_count(self):
        s = self._result().summary()
        assert "λ ∈ [2.00, 3.50]" in s
        assert "Constraint Diagnostics: 1 constraint(s)" in s

    def test_summary_falls_back_to_safe_bound_without_min_feasible(self):
        report = {**CANONICAL_REPORT, "penalty_lambda_min_feasible": None}
        r = CharacterizationResult(
            job_id="c1",
            status="COMPLETED",
            hardness=CANONICAL_HARDNESS,
            report=report,
            html="",
        )
        assert "λ ≤ 3.50" in r.summary()

    def test_render_shows_penalty_interval_and_constraint_table(self, capsys):
        self._result().display()
        out = capsys.readouterr().out
        assert "Safe range" in out
        assert "2.00" in out and "3.50" in out
        assert "Constraint Diagnostics" in out
        assert "max_cardinality" in out
        assert "10.0%" in out

    def test_empty_result_new_fields_are_none(self):
        r = CharacterizationResult(job_id="x", status="FAILED", html="")
        assert r.formulation_quality is None
        assert r.verdict is None
        assert r.classical_baseline is None
        assert r.cost_gap is None
        assert r.constraint_diagnostics is None

    def test_field_falls_through_explicit_null_to_next_fallback(self):
        """A present-but-null key must not short-circuit the fallback chain.

        Regression test: ``_field`` used to check ``k in self.report`` before
        reading the value, so a key present with an explicit ``None`` (e.g. an
        optional analysis that wasn't requested) would short-circuit and
        return ``None`` instead of falling through to a populated fallback.
        """
        report = {"target_achievability": None, "formulation_quality": 72.0}
        r = CharacterizationResult(
            job_id="n1", status="COMPLETED", report=report, html=""
        )
        assert r.quality_score == 72.0


class TestNamedVariableBQMSerialization:
    """String-named BQMs must serialize to integer-index wire keys."""

    def test_string_named_bqm_serializes_to_integer_indices(self):
        bqm = dimod.BinaryQuadraticModel(
            {"a": -1.0, "b": -1.0}, {("a", "b"): 2.0}, 0.0, dimod.BINARY
        )
        problem = BinaryOptimizationProblem(bqm)
        wire = _serialize_qubo_for_wire(problem)
        # Keys are integer-index strings ("0", "0,1"), never the names "a"/"b".
        for key in wire:
            for part in key.split(","):
                assert part.lstrip("-").isdigit(), f"non-integer key part: {part!r}"
        assert set(wire.keys()) == {"0", "1", "0,1"}

    def test_out_of_order_variable_names_map_correctly(self):
        """Variable names that are non-trivially ordered must still map
        through ``variable_to_idx`` correctly.

        ``{"a": ..., "b": ...}`` can't distinguish a correct mapping from one
        that accidentally works only because insertion order matched index
        order. Non-integer labels are ordered lexicographically by ``repr``
        (see ``_default_variable_order``), so "x10" < "x2" < "x9" -- neither
        insertion order nor numeric order -- which is exactly what would
        expose an off-by-one or ordering bug in the remap.
        """
        bqm = dimod.BinaryQuadraticModel(
            {"x9": -1.0, "x10": -2.0, "x2": -3.0},
            {("x9", "x2"): 1.5, ("x10", "x9"): 0.5},
            0.0,
            dimod.BINARY,
        )
        problem = BinaryOptimizationProblem(bqm)
        idx = problem.canonical_problem.variable_to_idx
        assert idx == {"x10": 0, "x2": 1, "x9": 2}

        wire = _serialize_qubo_for_wire(problem)
        assert wire["0"] == -2.0  # x10
        assert wire["1"] == -3.0  # x2
        assert wire["2"] == -1.0  # x9
        assert wire["1,2"] == 1.5  # (x9, x2) -> sorted by index -> (1, 2)
        assert wire["0,2"] == 0.5  # (x10, x9) -> sorted by index -> (0, 2)


class TestFactoredWireContract:
    """The factored_v1 payload must satisfy composer's from_wire decode."""

    def test_factored_payload_matches_server_decode_contract(self):
        rng = np.random.default_rng(0)
        u = rng.standard_normal((80, 1))
        Q = u @ u.T  # rank-1 -> factored encoding wins over legacy
        terms: dict = {}
        for i in range(80):
            if abs(Q[i, i]) > 0:
                terms[(i,)] = float(Q[i, i])
            for j in range(i + 1, 80):
                v = Q[i, j] + Q[j, i]
                if abs(v) > 0:
                    terms[(i, j)] = float(v)
        wire = _serialize_qubo_for_wire(BinaryOptimizationProblem(terms))
        assert wire.get("_format") == "factored_v1"

        # Decode exactly as composer-service FactoredQUBO.from_wire does and
        # confirm the byte-length / sign contract and the reconstruction.
        n, k = wire["n"], wire["k"]
        f_bytes = bytes.fromhex(wire["F"])
        assert len(f_bytes) == n * k * 8
        f = np.frombuffer(f_bytes, dtype=np.float64).reshape(n, k)
        signs = np.asarray(wire["signs"], dtype=np.float64)
        assert len(signs) == k and np.all((signs == 1.0) | (signs == -1.0))
        diag_bytes = bytes.fromhex(wire["diag"])
        assert len(diag_bytes) == n * 8
        residual = np.frombuffer(diag_bytes, dtype=np.float64)
        q_recon = f @ np.diag(signs) @ f.T + np.diag(residual)
        assert np.allclose(q_recon, Q, atol=1e-6)


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

    @pytest.mark.xfail(
        reason="Composer Service 500: target_states is empty", strict=False
    )
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

    @pytest.mark.xfail(
        reason="Composer Service 500: target_states is empty", strict=False
    )
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
