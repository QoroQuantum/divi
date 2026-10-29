# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for QUBO/HUBO characterization feature.

Tests cover:
- Wire-format serialization from all supported QUBO/HUBO types
- CharacterizationResult sections and shortcuts over the service's report
- QoroService.characterize_and_validate submit + fetch flows (mocked HTTP)
- divi.backends.characterization convenience functions
"""

import io
import json
import logging
import re
from http import HTTPStatus

import dimod
import numpy as np
import pytest
import requests
from pydantic import ValidationError
from rich.console import Console

from divi.backends import (
    ExecutionResult,
    InsufficientCreditsError,
    JobStatus,
    JobTimedOutError,
    JobType,
    QoroService,
)
from divi.backends.characterization import (
    CharacterizationOptions,
    CharacterizationResult,
    characterize_and_validate,
    get_characterization_result,
)
from divi.backends.characterization._characterization import (
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
from divi.exceptions import (
    CharacterizationFailedError,
    CharacterizationSubmitError,
    ExecutionCancelledError,
)
from divi.qprog.problems import BinaryOptimizationProblem
from tests._helpers import exact_match
from tests.backends._helpers import (
    http_response,
    make_mock_add_response,
    make_mock_init_response,
    make_mock_status_response,
    patch_transport,
)


def _curve_entry(layers, gap_closed, gammas, betas):
    return {
        "layers": layers,
        "gammas": gammas,
        "betas": betas,
        "energy": -1.0 - gap_closed,
        "error_bound": 0.0,
        "gap_closed": gap_closed,
        "gap_closed_bounds": [gap_closed, gap_closed],
        "padded": False,
        "optimized": True,
        "p_ground": gap_closed,
        "method": "statevector",
        "method_label": "exact statevector simulation",
        "shots_for_confidence": 4,
    }


_CURVE = [
    _curve_entry(1, 0.69, [0.4], [0.8]),
    _curve_entry(2, 0.88, [0.3, 0.5], [0.9, 0.4]),
    _curve_entry(3, 0.94, [0.2, 0.4, 0.6], [1.0, 0.6, 0.3]),
]

# Shaped like the report the characterisation service returns.
REPORT = {
    "summary": "The exact optimum is already known classically; use QAOA here to benchmark it.",
    "problem": {"num_variables": 2, "num_terms": 3, "lightcone_limit": 22},
    "classical": {
        "best_energy": -1.0,
        "best_bitstring": "01",
        "optimum": -1.0,
        "bound_label": "exact enumeration",
    },
    "qaoa": {
        "requested_layers": 2,
        "curve": _CURVE,
        "recommended_layers": 3,
        "recommended_reason": "the deeper layer counts computed gain little over it",
        "at_requested": _CURVE[1],
    },
    "angles": {
        "layers": 2,
        "gammas": [0.3, 0.5],
        "betas": [0.9, 0.4],
        "source": "optimized",
        "source_label": "optimised at this layer count",
    },
    "references": [{"bitstring": "01", "energy": -1.0, "gap_to_best": 0.0}],
    "penalty": {"exact": True, "qaoa_lambda": 3.0, "recommended_lambda": 1.519},
    "limits": [],
    "recommendations": [
        {"level": "action", "topic": "qaoa", "text": "Use p = 3 layers."},
        {"level": "info", "topic": "qaoa", "text": "At p = 2 QAOA closes 88.0%."},
    ],
}

SAMPLE_RESPONSE = {
    "job_id": "abc-123",
    "status": "COMPLETED",
    "report": REPORT,
    "recommendations": REPORT["recommendations"],
    "created_at": "2026-04-25T12:00:00Z",
    "completed_at": "2026-04-25T12:00:01Z",
}


def _result(report=REPORT, **fields):
    return CharacterizationResult(
        job_id="abc-123", status="COMPLETED", report=report, **fields
    )


def _pydantic_error(location: str, message: str) -> str:
    """``match=`` pattern for one pydantic error: its location line, then its message."""
    return rf"{re.escape(location)}\n\s+{message}"


_INVALID_OPTIONS = [
    *(
        pytest.param(
            {removed: value},
            _pydantic_error(removed, "Extra inputs are not permitted"),
            id=f"removed-{removed}",
        )
        for removed, value in (
            ("preset", "deep"),
            ("subspace", {"auto_warmstart": True}),
            ("parameter_sweep", True),
            ("penalty_tuning", True),
            ("gamma", 1.0),
            ("cost_qubo", BinaryOptimizationProblem({(0,): -1.0})),
        )
    ),
    pytest.param(
        {"n_qubits": 0},
        _pydantic_error("n_qubits", "Input should be greater than 0"),
        id="n-qubits-zero",
    ),
    pytest.param(
        {"n_qubits": -3},
        _pydantic_error("n_qubits", "Input should be greater than 0"),
        id="n-qubits-negative",
    ),
    pytest.param(
        # Weight keys are qubit indices; negatives must not reach the wire.
        {"constraints": [{"type": "inequality", "bound": 10, "weights": {-1: 4}}]},
        _pydantic_error(
            "constraints.0.weights.-1.[key]",
            "Input should be greater than or equal to 0",
        ),
        id="negative-weight-index",
    ),
    pytest.param(
        {"constraints": [{"type": "max_cardinaltiy", "bound": 3}]},
        _pydantic_error("constraints.0.type", "Input should be 'max_cardinality'"),
        id="unknown-constraint-type",
    ),
    pytest.param(
        {"constraints": [{"type": "max_cardinality", "bound": 3, "lower_bound": 1}]},
        _pydantic_error("constraints.0.lower_bound", "Extra inputs are not permitted"),
        id="unknown-constraint-key",
    ),
    pytest.param(
        {"constraints": [{"type": "inequality", "bound": 10}]},
        "'inequality' requires a non-empty 'weights' mapping",
        id="inequality-without-weights",
    ),
    pytest.param(
        {"constraints": [{"type": "equality", "bound": 10}]},
        "'equality' requires a non-empty 'weights' mapping",
        id="equality-without-weights",
    ),
    pytest.param(
        {
            "n_qubits": 4,
            "constraints": [
                {"type": "inequality", "bound": 10, "weights": {0: 1, 7: 2}}
            ],
        },
        "weight index 7",
        id="weight-index-out-of-range",
    ),
    pytest.param(
        {
            "n_qubits": 4,
            "constraints": [{"type": "max_cardinality", "bound": 2, "qubits": [0, 9]}],
        },
        "qubit index 9",
        id="qubit-index-out-of-range",
    ),
    pytest.param(
        {"ansatz": {"mixer": "x", "layers": 0}},
        _pydantic_error("ansatz.layers", "Input should be greater than 0"),
        id="ansatz-layers-zero",
    ),
    pytest.param(
        {"ansatz": {"mixer": "x", "auto_warmstart": True}},
        _pydantic_error("ansatz.auto_warmstart", "Extra inputs are not permitted"),
        id="ansatz-unknown-key",
    ),
    pytest.param(
        {"ansatz": {"mixer": "bad"}},
        _pydantic_error("ansatz.mixer", "Input should be 'x', 'xy' or 'I'"),
        id="ansatz-unknown-mixer",
    ),
]


def _init_ok(mocker):
    """A successful ``job/init/`` response for job ``char-1``."""
    return http_response(mocker, body={"job_id": "char-1"})


def _mock_characterization_requests(mocker, service, job_id):
    """Stub the init/submit/status/result requests and return the request mock."""
    mock_init = mocker.MagicMock()
    mock_init.json.return_value = {"job_id": job_id}
    mock_submit = mocker.MagicMock(status_code=HTTPStatus.ACCEPTED)
    mock_result = mocker.MagicMock()
    mock_result.json.return_value = SAMPLE_RESPONSE

    mocker.patch.object(service, "_fetch_characterization_html", return_value="<div/>")
    return mocker.patch.object(
        service,
        "_make_request",
        side_effect=[
            mock_init,
            mock_submit,
            make_mock_status_response(mocker, JobStatus.COMPLETED),
            mock_result,
        ],
    )


def _terms_from_dense(Q):
    """Upper-triangular term dict for a symmetric dense ``Q``, skipping zeros."""
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


def _decode_factored(wire):
    """Rebuild ``F · diag(signs) · Fᵀ + diag(diag)`` from a ``factored_v1`` payload."""
    F = np.frombuffer(bytes.fromhex(wire["F"]), dtype=np.float64).reshape(
        wire["n"], wire["k"]
    )
    diag = np.frombuffer(bytes.fromhex(wire["diag"]), dtype=np.float64)
    return F @ np.diag(np.asarray(wire["signs"], dtype=np.float64)) @ F.T + np.diag(
        diag
    )


def _rank_one_problem(n, seed=5):
    """Dense rank-1 ``u·uᵀ`` QUBO, which the dispatcher encodes as factored."""
    u = np.random.default_rng(seed).standard_normal(n)
    return BinaryOptimizationProblem(_terms_from_dense(np.outer(u, u)))


class TestSerializeQuboForWire:
    """Wire-format dispatch between legacy and factored encodings."""

    def test_small_qubo_uses_legacy(self):
        """``n`` below the probe threshold serializes as the comma-key dict."""
        problem = BinaryOptimizationProblem(np.array([[-1.0, 2.0], [0.0, -1.0]]))
        wire = _serialize_qubo_for_wire(problem)
        assert wire == {"0": -1.0, "0,1": 2.0, "1": -1.0}

    def test_constant_offset_is_dropped_not_serialized(self):
        """A constant term (e.g. from expanding a cardinality penalty) must not
        become an empty wire key or crash the dense path."""
        problem = BinaryOptimizationProblem({(0,): -1.0, (0, 1): 2.0, (): 5.0})

        with pytest.warns(UserWarning, match="constant offset"):
            wire = _serialize_qubo_for_wire(problem)

        assert wire == {"0": -1.0, "0,1": 2.0}
        assert _qubo_to_dense(problem.canonical_problem).shape == (2, 2)

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
        wire = _serialize_qubo_for_wire(_rank_one_problem(n, seed=0))
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
        problem = BinaryOptimizationProblem(_terms_from_dense(Q))
        wire = _serialize_qubo_for_wire(problem)
        assert wire["_format"] == "factored_v1"

        assert set(wire["signs"]) <= {-1.0, 1.0}
        # Tolerance tracks ``eigh``'s ``O(n · eps · ‖Q‖)`` backward error.
        np.testing.assert_allclose(
            _decode_factored(wire),
            Q,
            rtol=1e-10,
            atol=1e-12 * max(1.0, float(np.abs(Q).max())),
        )

    def test_factored_round_trip_on_small_scale_qubo(self):
        """A QUBO scaled by 1e-3 with unit coefficients survives the noise floor.

        ``Q = 1e-3 · (J − I)`` has spectral radius exactly ``1e-3 · (n − 1)``
        and unit-magnitude (times the scale) entries, so an absolute noise
        threshold would drop its whole spectrum.
        """
        n = _FACTORED_PROBE_MIN_QUBITS
        Q = 1e-3 * (np.ones((n, n)) - np.eye(n))
        problem = BinaryOptimizationProblem(_terms_from_dense(Q))

        wire = _serialize_qubo_factored(problem.canonical_problem)

        np.testing.assert_allclose(_decode_factored(wire), Q, rtol=0, atol=1e-15)

    def test_factored_round_trip_at_unit_spectral_radius(self):
        """``Q = J / n`` (spectral radius exactly 1) decodes to its unit-rank self."""
        n = _FACTORED_PROBE_MIN_QUBITS
        Q = np.ones((n, n)) / n
        problem = BinaryOptimizationProblem(_terms_from_dense(Q))

        wire = _serialize_qubo_factored(problem.canonical_problem)

        assert wire["k"] == 1
        np.testing.assert_allclose(_decode_factored(wire), Q, rtol=0, atol=1e-15)

    def test_dense_low_rank_qubo_over_budget_still_prefers_smaller_factored(self):
        """Factored wins whenever it is smaller, even above the truncation budget.

        ``U·Uᵀ`` at ``n=400`` and rank 160 is ~1.0 MB factored against ~2.3 MB
        legacy, so the legacy dict would be the worse payload to ship.
        """
        rng = np.random.default_rng(seed=0)
        U = rng.standard_normal((400, 160))
        problem = BinaryOptimizationProblem(_terms_from_dense(U @ U.T))

        wire = _serialize_qubo_for_wire(problem)

        assert wire["_format"] == "factored_v1"
        assert wire["k"] == 160
        assert _payload_size(wire) > _TRUNCATED_PAYLOAD_BUDGET_BYTES
        assert _payload_size(wire) < _payload_size(
            _serialize_qubo_legacy(problem.canonical_problem)
        )

    def test_dense_quadratic_with_one_cubic_term_uses_legacy(self):
        """A single degree-3 term forces legacy even where factored would win."""
        n = _FACTORED_PROBE_MIN_QUBITS
        terms = _terms_from_dense(np.ones((n, n)))
        terms[(0, 1, 2)] = 3.0
        problem = BinaryOptimizationProblem(terms)

        wire = _serialize_qubo_for_wire(problem)

        assert "_format" not in wire
        assert wire["0,1,2"] == 3.0
        assert wire["0,1"] == 2.0
        assert len(wire) == n * (n + 1) // 2 + 1

    def test_legacy_helper_directly(self):
        """``_serialize_qubo_legacy`` accepts HUBO of any degree."""
        problem = BinaryOptimizationProblem({(0,): -1.0, (0, 1): 2.0, (0, 1, 2): 3.0})
        wire = _serialize_qubo_legacy(problem.canonical_problem)
        assert wire == {"0": -1.0, "0,1": 2.0, "0,1,2": 3.0}

    def test_factored_helper_picks_smaller_k(self):
        """Encoder picks the residual=0 decomposition for a rank-1 ``u·uᵀ``."""
        wire = _serialize_qubo_factored(_rank_one_problem(16, seed=3).canonical_problem)
        assert wire["_format"] == "factored_v1"
        assert wire["n"] == 16
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
        problem = BinaryOptimizationProblem(_terms_from_dense(Q))
        wire = _serialize_qubo_for_wire(problem)
        assert wire["_format"] == "factored_v1"
        # Q has rank ≤ structural rank + 1 (cardinality direction).
        assert wire["k"] <= 16
        # Reconstruction is at eigh-noise level (lossless).
        rel_err = np.abs(_decode_factored(wire) - Q).max() / np.abs(Q).max()
        assert rel_err < 1e-10

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
        problem = BinaryOptimizationProblem(_terms_from_dense(Q))
        wire = _serialize_qubo_for_wire(problem)
        assert wire["_format"] == "factored_v1"
        assert 2 <= wire["k"] <= 32
        assert _payload_size(wire) < _TRUNCATED_PAYLOAD_BUDGET_BYTES

        Q_recon = _decode_factored(wire)

        Q_max = float(np.abs(Q).max())
        rel_err = float(np.abs(Q_recon - Q).max() / Q_max)
        assert rel_err < _TRUNCATED_REL_ERROR_MAX
        # Diagonal absorption keeps Q[i,i] exact at eigh-noise level.
        diag_err = float(np.abs(np.diagonal(Q_recon - Q)).max())
        assert diag_err < 1e-9 * Q_max

    def test_budget_capped_truncation_is_accepted(self):
        """A dominant diagonal keeps the budget-capped candidate under the error gate.

        The off-diagonal part is a dense random matrix with more eigenvalues
        above the magnitude cut than the payload budget allows, so only the
        budget cap keeps the payload under the proxy limit.
        """
        n = 256
        rng = np.random.default_rng(seed=0)
        off = rng.standard_normal((n, n))
        off = 0.5 * (off + off.T)
        np.fill_diagonal(off, 0.0)
        Q = off + 1e4 * np.eye(n)
        k_budget = (_TRUNCATED_PAYLOAD_BUDGET_BYTES - 16 * n - 200) // (16 * n + 5)
        eigvals = np.abs(np.linalg.eigvalsh(off))
        assert np.sum(eigvals >= 1e-2 * eigvals.max()) > k_budget

        wire = _serialize_qubo_factored(
            BinaryOptimizationProblem(_terms_from_dense(Q)).canonical_problem
        )

        assert wire["k"] == k_budget
        assert _payload_size(wire) <= _TRUNCATED_PAYLOAD_BUDGET_BYTES
        rel_err = np.abs(_decode_factored(wire) - Q).max() / np.abs(Q).max()
        assert rel_err < _TRUNCATED_REL_ERROR_MAX

    @pytest.mark.parametrize("n", [64, 128, 256])
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_random_low_rank_plus_noise_produces_valid_payload(self, n, seed):
        """Encoder output is structurally valid across random inputs.

        Spans the realistic problem-size range (64..256) at multiple seeds
        to catch failure modes the focused unit tests miss — invalid
        signs, NaN/Inf in hex blobs, dimension mismatches, oversized
        payloads, or relative errors above the documented bound.
        """
        rng = np.random.default_rng(seed)
        rank = max(1, n // 8)
        U = rng.standard_normal((n, rank))
        Q = U @ U.T + 0.01 * rng.standard_normal((n, n))
        Q = 0.5 * (Q + Q.T)
        problem = BinaryOptimizationProblem(_terms_from_dense(Q))
        wire = _serialize_qubo_for_wire(problem)

        # Either format is acceptable; whichever was chosen must be
        # structurally well-formed.
        if isinstance(wire, dict) and wire.get("_format") == "factored_v1":
            assert wire["n"] == n
            assert 0 <= wire["k"] <= n
            assert all(s in (-1.0, 1.0) for s in wire["signs"])
            assert len(bytes.fromhex(wire["F"])) == n * wire["k"] * 8
            assert len(bytes.fromhex(wire["diag"])) == n * 8
            # Reconstruction must obey the documented relative-error bound.
            Q_recon = _decode_factored(wire)
            assert np.isfinite(Q_recon).all()
            Q_max = float(np.abs(Q).max())
            if Q_max > 0:
                rel_err = float(np.abs(Q_recon - Q).max() / Q_max)
                assert rel_err < _TRUNCATED_REL_ERROR_MAX
        else:
            # Legacy comma-key dict — must be exact.
            assert all(isinstance(k, str) for k in wire.keys())
            assert all(isinstance(v, (int, float)) for v in wire.values())


class TestCharacterizationResult:
    """The result exposes the service's report as it arrives, plus a few shortcuts."""

    def test_sections_are_the_report_as_sent(self):
        result = _result()

        assert result.summary == REPORT["summary"]
        assert result.recommendations == REPORT["recommendations"]
        assert result.problem is REPORT["problem"]
        assert result.classical is REPORT["classical"]
        assert result.qaoa is REPORT["qaoa"]
        assert result.angles is REPORT["angles"]
        assert result.references == REPORT["references"]
        assert result.penalty is REPORT["penalty"]
        assert result.limits == []

    def test_absent_sections_read_as_empty(self):
        result = _result(report={})

        assert (result.summary, result.recommendations, result.limits) == ("", [], [])
        assert result.problem == result.constraints == result.penalty == {}
        assert result.recommended_layers is None
        assert result.gap_closed is None
        assert result.recommended_lambda is None

    def test_shortcuts_read_the_report(self):
        result = _result()

        assert result.recommended_layers == 3
        assert result.gap_closed == 0.88
        assert result.recommended_lambda == 1.519

    def test_gap_closed_is_none_when_the_requested_layers_were_not_computed(self):
        result = _result(
            report={**REPORT, "qaoa": {**REPORT["qaoa"], "at_requested": None}}
        )

        assert result.gap_closed is None

    @pytest.mark.parametrize(
        ("layers", "expected"),
        [
            (None, [[0.2, 1.0, 0.4, 0.6, 0.6, 0.3]]),
            (2, [[0.3, 0.9, 0.5, 0.4]]),
        ],
        ids=["recommended", "explicit"],
    )
    def test_qaoa_initial_params_interleave_the_curve_angles(self, layers, expected):
        np.testing.assert_array_equal(_result().qaoa_initial_params(layers), expected)

    def test_qaoa_initial_params_fall_back_to_the_starting_angles(self):
        """With no layer count computed, the report's (heuristic) starting angles are used."""
        ramp = {
            "layers": 2,
            "gammas": [0.1, 0.3],
            "betas": [0.2, 0.4],
            "source": "ramp",
        }
        result = _result(
            report={"qaoa": {"curve": [], "recommended_layers": None}, "angles": ramp}
        )

        np.testing.assert_array_equal(
            result.qaoa_initial_params(), [[0.1, 0.2, 0.3, 0.4]]
        )

    def test_qaoa_initial_params_reject_layers_without_angles(self):
        with pytest.raises(
            ValueError,
            match=re.escape(
                "No angles at layers=7; the report has them for [1, 2, 3]."
            ),
        ):
            _result().qaoa_initial_params(layers=7)

    @pytest.mark.parametrize(
        "angles",
        [{"gammas": [0.4, 0.5], "betas": [0.8]}, {"gammas": [], "betas": []}],
        ids=["mismatched", "empty"],
    )
    def test_qaoa_initial_params_none_for_incomplete_angles(self, angles):
        result = _result(report={"qaoa": {}, "angles": angles})

        assert result.qaoa_initial_params() is None

    def test_repr_lists_the_summary_and_the_advice(self):
        assert repr(_result()) == "\n".join(
            [
                "CharacterizationResult(job_id='abc-123', status='COMPLETED')",
                f"  {REPORT['summary']}",
                "  [action] Use p = 3 layers.",
                "  [info] At p = 2 QAOA closes 88.0%.",
            ]
        )

    @staticmethod
    def _displayed(mocker, result) -> str:
        console = Console(record=True, width=120, color_system=None, file=io.StringIO())
        mocker.patch(
            "divi.backends.characterization._characterization.Console",
            return_value=console,
        )
        result.display()
        return " ".join(console.export_text().split())

    def test_display_lays_out_the_report_wording(self, mocker):
        limit = "Sampling outcomes need the full statevector [22 variables]."
        text = self._displayed(mocker, _result(report={**REPORT, "limits": [limit]}))

        for expected in (
            REPORT["summary"],
            "• Use p = 3 layers.",
            "• At p = 2 QAOA closes 88.0%.",
            "2 │ 88.0% │ 0.8800 │ 4 │ exact statevector simulation",
            "3 ★ │ 94.0% │ 0.9400 │ 4 │ exact statevector simulation",
            f"★ recommended: {REPORT['qaoa']['recommended_reason']}",
            "γ = [0.3000, 0.5000]",
            "optimised at this layer count",
            limit,
        ):
            assert expected in text

    def test_display_of_an_empty_report_shows_only_its_frame(self, mocker):
        text = self._displayed(mocker, _result(report={}))

        assert "QUBO Characterization Report — job abc-123" in text
        assert "QAOA by layer count" not in text

    def test_repr_html_returns_the_stored_html(self):
        assert _result(html="<div>r</div>")._repr_html_() == "<div>r</div>"


class TestWrapResponse:
    """Turning the service's response into a result, or into the matching error."""

    @pytest.mark.parametrize(
        ("status", "error"),
        [
            ("FAILED", CharacterizationFailedError),
            ("CANCELLED", ExecutionCancelledError),
            ("INSUFFICIENT_CREDITS", InsufficientCreditsError),
            ("TIMED_OUT", JobTimedOutError),
            ("RUNNING", CharacterizationFailedError),
        ],
    )
    def test_unfinished_jobs_raise(self, mocker, status, error):
        with pytest.raises(error):
            _wrap_response({"job_id": "j", "status": status}, mocker.MagicMock())

    def test_a_completed_job_without_a_report_raises(self, mocker):
        with pytest.raises(CharacterizationFailedError, match="carried no report"):
            _wrap_response({"job_id": "j", "status": "COMPLETED"}, mocker.MagicMock())

    def test_the_report_and_html_are_kept(self, mocker):
        service = mocker.MagicMock()
        service._fetch_characterization_html.return_value = "<div/>"

        result = _wrap_response(SAMPLE_RESPONSE, service)

        assert result == _result(
            created_at="2026-04-25T12:00:00Z", completed_at="2026-04-25T12:00:01Z"
        )
        assert result.html == "<div/>"

    def test_an_unreachable_html_page_leaves_the_html_empty(self, mocker, caplog):
        service = mocker.MagicMock()
        service._fetch_characterization_html.side_effect = requests.ConnectionError(
            "down"
        )

        with caplog.at_level(logging.WARNING):
            result = _wrap_response(SAMPLE_RESPONSE, service)

        assert result.html == ""
        assert "Could not fetch HTML report for job abc-123" in caplog.text


class TestQoroServiceCharacterize:
    """Tests for the consolidated submit + fetch flow in QoroService."""

    def test_full_submit_flow(self, mocker, qoro_service_factory):
        service = qoro_service_factory()
        mock_req = _mock_characterization_requests(mocker, service, "val-job-123")

        result = service.characterize_and_validate(
            qubo={"0,0": -1.0, "0,1": 2.0},
            reference_states=["01", "10"],
        )

        # service.characterize_and_validate returns the raw API response dict;
        # the result object is built by divi.backends.characterize_and_validate().
        assert isinstance(result, dict)
        assert result["status"] == "COMPLETED"
        assert result["report"] == REPORT

        # init → submit → status → result, all routed through the same
        # wrapper (init and submit use retry=False).
        init_call, submit_call, status_call, result_call = mock_req.call_args_list
        assert status_call.args == ("get", "job/val-job-123/status/")
        assert result_call.args == ("get", "job/val-job-123/validation_result/")
        assert init_call.args == ("post", "job/init/")
        assert init_call.kwargs["retry"] is False
        assert init_call.kwargs["json"]["job_type"] == "VALIDATE"
        assert submit_call.args == ("post", "job/val-job-123/submit_qubo/")
        assert submit_call.kwargs["retry"] is False
        assert submit_call.kwargs["json"]["qubo"] == {"0,0": -1.0, "0,1": 2.0}
        assert submit_call.kwargs["json"]["reference_states"] == ["01", "10"]

    def test_options_pass_through(self, mocker, qoro_service_factory):
        service = qoro_service_factory()
        mock_req = _mock_characterization_requests(mocker, service, "val-opt-123")

        options = {
            "ansatz": {"mixer": "x", "layers": 1},
            "constraints": [{"type": "max_cardinality", "bound": 1}],
        }

        service.characterize_and_validate(
            qubo={"0,0": -1.0},
            reference_states=["0"],
            options=options,
        )

        assert mock_req.call_args_list[1].kwargs["json"]["options"] == options

    @pytest.mark.parametrize("reference_states", [None, []])
    def test_submit_requires_reference_states(
        self, qoro_service_factory, reference_states
    ):
        service = qoro_service_factory()

        with pytest.raises(ValueError, match="reference state"):
            service.characterize_and_validate(
                qubo={"0,0": -1.0},
                reference_states=reference_states,
            )

    def test_fetch_by_job_id_skips_submit(self, mocker, qoro_service_factory):
        """Fetching waits for completion, then reads without init/submit."""
        service = qoro_service_factory(polling_interval=0.0)

        mock_running = mocker.MagicMock()
        mock_running.json.return_value = {"status": "RUNNING"}
        mock_completed = mocker.MagicMock()
        mock_completed.json.return_value = {"status": "COMPLETED"}

        mock_resp = mocker.MagicMock()
        mock_resp.json.return_value = SAMPLE_RESPONSE

        mock_req = mocker.patch.object(
            service,
            "_make_request",
            side_effect=[mock_running, mock_completed, mock_resp],
        )

        result = service.characterize_and_validate(job_id="abc-123")

        assert isinstance(result, dict)
        assert result["job_id"] == "abc-123"
        assert result["report"] == REPORT

        assert [call.args for call in mock_req.call_args_list] == [
            ("get", "job/abc-123/status/"),
            ("get", "job/abc-123/status/"),
            ("get", "job/abc-123/validation_result/"),
        ]

    @pytest.mark.parametrize(
        ("status", "error_type"),
        [
            ("FAILED", CharacterizationFailedError),
            ("CANCELLED", ExecutionCancelledError),
            ("INSUFFICIENT_CREDITS", InsufficientCreditsError),
            ("TIMED_OUT", JobTimedOutError),
        ],
    )
    def test_fetch_by_job_id_surfaces_terminal_status(
        self, mocker, qoro_service_factory, status, error_type
    ):
        service = qoro_service_factory(polling_interval=0.0)
        mock_status = mocker.MagicMock()
        mock_status.json.return_value = {"status": status}
        mock_req = mocker.patch.object(
            service, "_make_request", return_value=mock_status
        )

        with pytest.raises(error_type):
            get_characterization_result("terminal-job-1", service=service)

        assert mock_req.call_count == 1
        assert mock_req.call_args.args == ("get", "job/terminal-job-1/status/")

    def test_submit_waits_through_running_then_surfaces_a_failed_job(
        self, mocker, qoro_service_factory
    ):
        service = qoro_service_factory(polling_interval=0.0)
        mock_req = mocker.patch.object(
            service,
            "_make_request",
            side_effect=[
                make_mock_init_response(mocker, "char-1"),
                make_mock_add_response(mocker, HTTPStatus.ACCEPTED),
                make_mock_status_response(mocker, JobStatus.RUNNING),
                make_mock_status_response(mocker, JobStatus.FAILED),
            ],
        )

        result = service.characterize_and_validate(
            qubo={"0,0": -1.0}, reference_states=["0"]
        )

        assert result == {"job_id": "char-1", "status": "FAILED"}
        assert mock_req.call_count == 4

    def test_credit_rejection_at_submit_raises_insufficient_credits(
        self, mocker, qoro_service_factory
    ):
        service = qoro_service_factory()
        patch_transport(
            mocker,
            plain=[
                _init_ok(mocker),
                http_response(
                    mocker,
                    HTTPStatus.PAYMENT_REQUIRED,
                    body={"detail": "no credits"},
                    reason="Payment Required",
                ),
            ],
        )

        with pytest.raises(InsufficientCreditsError) as exc:
            service.characterize_and_validate(
                qubo={"0,0": -1.0}, reference_states=["0"]
            )

        assert exc.value.job_id == "char-1"

    def test_exhausted_polling_surfaces_the_job_id(self, mocker, qoro_service_factory):
        service = qoro_service_factory(polling_interval=0.0, max_retries=2)
        mocker.patch.object(
            service,
            "_make_request",
            side_effect=[
                make_mock_init_response(mocker, "char-1"),
                make_mock_add_response(mocker, HTTPStatus.ACCEPTED),
                make_mock_status_response(mocker, JobStatus.RUNNING),
                make_mock_status_response(mocker, JobStatus.RUNNING),
            ],
        )

        with pytest.raises(CharacterizationSubmitError) as exc:
            service.characterize_and_validate(
                qubo={"0,0": -1.0}, reference_states=["0"]
            )

        assert (exc.value.job_id, exc.value.phase) == ("char-1", "status polling")

    def test_interrupting_the_wait_does_not_cancel_the_job(
        self, mocker, qoro_service_factory
    ):
        service = qoro_service_factory()
        cancel = mocker.patch.object(service, "cancel_job")
        mocker.patch.object(
            service,
            "_make_request",
            side_effect=[
                make_mock_init_response(mocker, "char-1"),
                make_mock_add_response(mocker, HTTPStatus.ACCEPTED),
                KeyboardInterrupt,
            ],
        )

        with pytest.raises(KeyboardInterrupt):
            service.characterize_and_validate(
                qubo={"0,0": -1.0}, reference_states=["0"]
            )

        cancel.assert_not_called()

    def test_requires_qubo_or_job_id(self, qoro_service_factory):
        service = qoro_service_factory()
        with pytest.raises(ValueError, match="qubo.*or.*job_id"):
            service.characterize_and_validate()

    @pytest.mark.parametrize(
        "submit, plain, session, phase",
        [
            pytest.param(
                True,
                lambda m: [_init_ok(m), requests.ReadTimeout("timed out")],
                lambda m: [],
                "submission",
                id="submit-timeout",
            ),
            pytest.param(
                True,
                lambda m: [
                    _init_ok(m),
                    http_response(
                        m,
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                        body={},
                        reason="Internal Server Error",
                    ),
                ],
                lambda m: [],
                "submission",
                id="submit-server-error",
            ),
            pytest.param(
                True,
                lambda m: [_init_ok(m), http_response(m, body={})],
                lambda m: [requests.ReadTimeout("status poll timed out")],
                "status polling",
                id="submit-then-status-polling-timeout",
            ),
            pytest.param(
                True,
                lambda m: [_init_ok(m), http_response(m, body={})],
                lambda m: [
                    http_response(m, body={"status": JobStatus.COMPLETED.value}),
                    requests.ReadTimeout("result fetch timed out"),
                ],
                "result retrieval",
                id="result-timeout",
            ),
            pytest.param(
                False,
                lambda m: [],
                lambda m: [requests.ReadTimeout("timed out")],
                "status polling",
                id="status-polling-timeout",
            ),
        ],
    )
    def test_recoverable_failure_surfaces_the_job_id_and_phase(
        self, mocker, qoro_service_factory, submit, plain, session, phase
    ):
        """A timeout or 5xx after the job exists must not lose its id."""
        service = qoro_service_factory()
        patch_transport(mocker, plain=plain(mocker), session=session(mocker))

        with pytest.raises(CharacterizationSubmitError) as exc:
            if submit:
                service.characterize_and_validate(
                    qubo={"0,0": -1.0}, reference_states=["0"]
                )
            else:
                service.characterize_and_validate(job_id="char-1")

        assert (exc.value.job_id, exc.value.phase) == ("char-1", phase)
        assert isinstance(
            exc.value.__cause__, (requests.ReadTimeout, requests.HTTPError)
        )

    def test_job_id_is_logged_before_the_submit(
        self, mocker, qoro_service_factory, caplog
    ):
        """The id must reach the user when it is created, not only on failure."""
        service = qoro_service_factory()
        patch_transport(
            mocker, plain=[_init_ok(mocker), requests.ReadTimeout("timed out")]
        )

        with caplog.at_level(logging.INFO, logger="divi.backends.runners._qoro"):
            with pytest.raises(CharacterizationSubmitError):
                service.characterize_and_validate(
                    qubo={"0,0": -1.0}, reference_states=["0"]
                )

        assert "char-1" in caplog.text

    def test_a_rejected_submit_is_raised_as_is(self, mocker, qoro_service_factory):
        """A 4xx means the job never ran; it must not advise re-fetching."""
        service = qoro_service_factory()
        patch_transport(
            mocker,
            plain=[
                _init_ok(mocker),
                http_response(
                    mocker,
                    HTTPStatus.BAD_REQUEST,
                    body={"detail": "bad qubo"},
                    reason="Bad Request",
                ),
            ],
        )

        with pytest.raises(
            requests.HTTPError,
            match=exact_match('400 Bad Request: {"detail": "bad qubo"}'),
        ):
            service.characterize_and_validate(
                qubo={"0,0": -1.0}, reference_states=["0"]
            )

    def test_a_client_error_while_waiting_is_raised_as_is(
        self, mocker, qoro_service_factory
    ):
        service = qoro_service_factory()
        error = requests.HTTPError(
            "404 Not Found", response=mocker.MagicMock(status_code=HTTPStatus.NOT_FOUND)
        )
        mocker.patch.object(service, "_make_request", side_effect=error)

        with pytest.raises(requests.HTTPError) as exc_info:
            service.characterize_and_validate(job_id="char-1")

        assert exc_info.value is error

    def test_init_body_and_job_id_fallback(self, mocker, qoro_service_factory):
        service = qoro_service_factory()
        result = mocker.MagicMock(json=lambda: {"status": "COMPLETED"})
        mock_request = mocker.patch.object(
            service,
            "_make_request",
            side_effect=[
                make_mock_init_response(mocker, "char-1"),
                make_mock_add_response(mocker),
                make_mock_status_response(mocker, JobStatus.COMPLETED),
                result,
            ],
        )

        assert service.characterize_and_validate(
            qubo={"0,0": -1.0}, reference_states=["0"]
        ) == {"status": "COMPLETED", "job_id": "char-1"}
        init_call = mock_request.call_args_list[0]
        assert init_call.args == ("post", "job/init/")
        assert init_call.kwargs["json"] == {
            "job_type": "VALIDATE",
            "tag": "divi-characterize",
        }

    def test_the_html_report_is_returned_as_text(self, mocker, qoro_service_factory):
        service = qoro_service_factory()
        mock_request = mocker.patch.object(
            service, "_make_request", return_value=mocker.MagicMock(text="<div>r</div>")
        )

        assert service._fetch_characterization_html("char-1") == "<div>r</div>"
        assert mock_request.call_args.args == (
            "get",
            "job/char-1/validation_result/html/",
        )


class TestTopLevelCharacterize:
    """Tests for the divi.backends.characterize_and_validate convenience function."""

    def test_characterize_with_ndarray(self, mocker, qoro_service_factory):
        service = qoro_service_factory()
        _mock_characterization_requests(mocker, service, "top-123")

        problem = BinaryOptimizationProblem(np.array([[-1.0, 2.0], [0.0, -1.0]]))
        result = characterize_and_validate(
            problem, reference_states=["01", "10"], service=service
        )

        assert isinstance(result, CharacterizationResult)
        assert result.summary == REPORT["summary"]

    @pytest.mark.parametrize("kwargs, match", _INVALID_OPTIONS)
    def test_invalid_options_rejected(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            CharacterizationOptions(**kwargs)

    def test_factored_payload_auto_attaches_n_qubits(
        self, mocker, qoro_service_factory
    ):
        """Factored submissions ship ``options['n_qubits']`` alongside the payload."""
        service = qoro_service_factory()
        mock_req = _mock_characterization_requests(mocker, service, "fac-123")

        # Rank-1 dense QUBO at the probe threshold encodes with k=1,
        # forcing the dispatcher to pick factored.
        n = _FACTORED_PROBE_MIN_QUBITS

        characterize_and_validate(
            _rank_one_problem(n),
            reference_states=["0" * n],
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
        mock_req = _mock_characterization_requests(mocker, service, "fac-2")

        n = _FACTORED_PROBE_MIN_QUBITS

        options = CharacterizationOptions()
        mocker.patch.object(options, "_to_wire", return_value={"n_qubits": 999})

        characterize_and_validate(
            _rank_one_problem(n),
            reference_states=["0" * n],
            service=service,
            options=options,
        )

        submit_json = mock_req.call_args_list[1].kwargs["json"]
        assert submit_json["options"]["n_qubits"] == 999

    def test_options_wire_exactly(self):
        wire = CharacterizationOptions(
            ansatz={"mixer": "xy", "layers": 2},
            constraints=[
                {"type": "max_cardinality", "bound": 3},
                {"type": "inequality", "bound": 10, "weights": {0: 4, 1: 5}},
            ],
            n_qubits=5,
        )._to_wire()

        assert wire == {
            "ansatz": {"mixer": "xy", "layers": 2},
            "constraints": [
                {"type": "max_cardinality", "bound": 3},
                {"type": "inequality", "bound": 10, "weights": {0: 4, 1: 5}},
            ],
            "n_qubits": 5,
        }

    def test_default_options_send_nothing(self):
        assert CharacterizationOptions()._to_wire() is None

    def test_a_penalised_problem_sends_its_cost_and_penalty(
        self, mocker, qoro_service_factory
    ):
        """Zero-coefficient and constant terms never reach the component dicts."""
        service = qoro_service_factory()
        mock_req = _mock_characterization_requests(mocker, service, "pen-wire")
        problem = BinaryOptimizationProblem(
            {(0,): -1.0, (0, 2): 1.5, (0, 1): 0.0, (): 2.0},
            penalty={(1, 2): 3.0, (1,): -0.5, (): 1.0},
        )

        with pytest.warns(UserWarning, match="constant offset"):
            characterize_and_validate(
                problem, reference_states=["000"], service=service
            )

        options = mock_req.call_args_list[1].kwargs["json"]["options"]
        assert options["cost_qubo"] == {"0": -1.0, "0,2": 1.5}
        assert options["penalty_qubo"] == {"1,2": 3.0, "1": -0.5}

    def test_an_unpenalised_problem_sends_no_split(self, mocker, qoro_service_factory):
        service = qoro_service_factory()
        mock_req = _mock_characterization_requests(mocker, service, "no-pen")

        characterize_and_validate(
            BinaryOptimizationProblem({(0,): -1.0}),
            reference_states=["0"],
            service=service,
        )

        assert "options" not in mock_req.call_args_list[1].kwargs["json"]

    @pytest.mark.parametrize(
        ("reference_states", "message"),
        [
            ([], "at least one"),
            (["0x"], "binary"),
            (["0"], "2 bits"),
        ],
    )
    def test_reference_states_are_validated_before_submission(
        self, mocker, reference_states, message
    ):
        service = mocker.MagicMock()
        problem = BinaryOptimizationProblem(np.eye(2))

        with pytest.raises(ValueError, match=message):
            characterize_and_validate(
                problem,
                reference_states=reference_states,
                service=service,
            )

        service.characterize_and_validate.assert_not_called()

    def test_reference_state_dimension_uses_explicit_n_qubits(self, mocker):
        service = mocker.MagicMock()
        problem = BinaryOptimizationProblem(np.eye(2))

        with pytest.raises(ValueError, match="4 bits"):
            characterize_and_validate(
                problem,
                reference_states=["00"],
                service=service,
                options=CharacterizationOptions(n_qubits=4),
            )

        service.characterize_and_validate.assert_not_called()

    def test_constraint_indices_unchecked_without_n_qubits(self):
        """Without n_qubits there is nothing to check against; the server decides."""
        wire = CharacterizationOptions(
            constraints=[{"type": "max_cardinality", "bound": 2, "qubits": [0, 99]}]
        )._to_wire()

        assert wire["constraints"][0]["qubits"] == [0, 99]

    def test_constraint_weights_accept_json_string_keys(self):
        """A JSON round trip stringifies int keys; they must still validate."""
        wire = CharacterizationOptions(
            constraints=[{"type": "inequality", "bound": 10, "weights": {"0": 4}}]
        )._to_wire()

        assert wire["constraints"][0]["weights"] == {0: 4}

    def test_options_survive_a_model_dump_round_trip(self):
        """model_dump() emits explicit Nones; re-validating them must not raise."""
        original = CharacterizationOptions(ansatz={"layers": 2}, n_qubits=4)

        assert (
            CharacterizationOptions.model_validate(original.model_dump())._to_wire()
            == original._to_wire()
        )

    def test_reference_states_match_a_wider_explicit_n_qubits(self, mocker):
        service = mocker.MagicMock()
        service.characterize_and_validate.return_value = SAMPLE_RESPONSE
        service._fetch_characterization_html.return_value = ""

        characterize_and_validate(
            BinaryOptimizationProblem(np.eye(2)),
            reference_states=["0100"],
            service=service,
            options=CharacterizationOptions(n_qubits=4),
        )

        assert service.characterize_and_validate.call_args.kwargs[
            "reference_states"
        ] == ["0100"]

    def test_n_qubits_below_problem_size_rejected(self, mocker):
        service = mocker.MagicMock()

        with pytest.raises(
            ValueError,
            match=r"^n_qubits=1 is smaller than the problem's 2 represented variables\.$",
        ):
            characterize_and_validate(
                BinaryOptimizationProblem(np.eye(2)),
                reference_states=["0"],
                service=service,
                options=CharacterizationOptions(n_qubits=1),
            )

        service.characterize_and_validate.assert_not_called()


def test_job_type_characterize_has_validate_wire_value():
    # Wire value must remain ``"VALIDATE"`` — server compatibility.
    assert JobType.CHARACTERIZE.value == "VALIDATE"


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"n_qubits": np.int64(5)}, id="n-qubits"),
        pytest.param({"ansatz": {"layers": np.int32(2)}}, id="ansatz-layers"),
        pytest.param(
            {
                "constraints": [
                    {"type": "max_cardinality", "bound": 3, "qubits": np.arange(3)}
                ]
            },
            id="constraint-qubit-array",
        ),
        pytest.param(
            {
                "constraints": [
                    {
                        "type": "inequality",
                        "bound": np.int64(10),
                        "weights": {np.int64(0): np.float64(4.0), np.int32(1): 5},
                    }
                ]
            },
            id="constraint-numpy-weight-keys",
        ),
    ],
)
def test_options_accept_numpy_scalars_and_serialise_to_json(kwargs):
    wire = CharacterizationOptions(**kwargs)._to_wire()

    assert wire is not None
    json.dumps(wire)


def test_options_hold_numpy_scalars_as_builtin_types():
    options = CharacterizationOptions(n_qubits=np.uint64(2**63 + 5))

    assert options.n_qubits == 2**63 + 5
    assert type(options.n_qubits) is int


@pytest.mark.parametrize(
    "kwargs, message",
    [
        pytest.param({"n_qubits": True}, "valid integer", id="bool-as-int"),
        pytest.param({"n_qubits": 2.0}, "valid integer", id="float-as-int"),
        pytest.param({"n_qubits": np.float64(2.0)}, "valid integer", id="np-float"),
        pytest.param(
            {"ansatz": {"layers": True}}, "valid integer", id="bool-as-layers"
        ),
        pytest.param(
            {
                "constraints": [
                    {"type": "max_cardinality", "bound": 3, "qubits": [True]}
                ]
            },
            "valid integer",
            id="bool-qubit",
        ),
        pytest.param(
            {"constraints": [{"type": "equality", "bound": 3, "weights": {True: 1}}]},
            "valid integer",
            id="bool-weight-key",
        ),
    ],
)
def test_options_still_reject_non_integers(kwargs, message):
    with pytest.raises(ValidationError, match=message):
        CharacterizationOptions(**kwargs)


def test_reference_states_may_be_an_array(mocker, qoro_service_factory):
    service = qoro_service_factory()
    mock_req = _mock_characterization_requests(mocker, service, "array-states")

    characterize_and_validate(
        BinaryOptimizationProblem(np.eye(2)),
        reference_states=np.array(["01", "10"]),
        service=service,
    )

    submit_json = mock_req.call_args_list[1].kwargs["json"]
    assert submit_json["reference_states"] == ["01", "10"]
    json.dumps(submit_json)


def test_an_empty_reference_state_array_is_rejected_before_submission(mocker):
    service = mocker.MagicMock()

    with pytest.raises(ValueError, match="at least one"):
        characterize_and_validate(
            BinaryOptimizationProblem(np.eye(2)),
            reference_states=np.array([], dtype=str),
            service=service,
        )

    service.characterize_and_validate.assert_not_called()


@pytest.mark.parametrize("reference_states", [np.array([], dtype=str), ()])
def test_the_service_requires_a_reference_state_in_any_empty_container(
    qoro_service_factory, reference_states
):
    with pytest.raises(ValueError, match="reference state"):
        qoro_service_factory().characterize_and_validate(
            qubo={"0,0": -1.0}, reference_states=reference_states
        )


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


@pytest.mark.requires_api_key
class TestQoroServiceValidationHtmlE2E:
    """E2E regression tests for the prod ``validation_result`` endpoints.

    Pins the wire contract between divi and the Qoro API for the QUBO
    characterization feature against the real prod API. Every test reads
    from one class-scoped job, so the prod account sees one VALIDATE
    submission per CI run.

    Endpoints exercised:
      - ``POST /api/job/init/``                         (job creation)
      - ``POST /api/job/<pk>/submit_qubo/``             (validation submit)
      - ``GET  /api/job/<pk>/validation_result/``       (JSON, via ``characterize_and_validate``)
      - ``GET  /api/job/<pk>/validation_result/html/``  (HTML)
    """

    @pytest.fixture(scope="class")
    def completed_validation_job(self, api_key):
        """Submit one VALIDATE job with every option divi sends, and clean up afterwards.

        The problem carries a penalty and a constraint so the report includes
        its optional ``penalty`` and ``constraints`` sections. The server
        rejects unknown option keys with a 400, so a submission that succeeds
        proves divi's wire vocabulary is still a subset of the server's.
        """
        service = QoroService(auth_token=api_key)
        # Pick exactly one of two items worth 1 and 2: (x0 + x1 - 1)² penalty.
        problem = BinaryOptimizationProblem(
            {(0,): -1.0, (1,): -2.0},
            penalty={(0,): -1.0, (1,): -1.0, (0, 1): 2.0},
        )
        result = characterize_and_validate(
            problem,
            reference_states=["01"],
            service=service,
            options=CharacterizationOptions(
                ansatz={"mixer": "x", "layers": 1},
                constraints=[{"type": "eq_cardinality", "bound": 1}],
                n_qubits=2,
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

    def test_completed_job_full_shape(self, completed_validation_job):
        """Every section divi exposes arrives, with the keys its shortcuts read.

        Asserts types and keys, not specific numeric values (which would be
        flaky against the live service).
        """
        _, result = completed_validation_job

        assert isinstance(result.job_id, str) and result.job_id
        assert result.status == "COMPLETED"
        assert isinstance(result.created_at, str) and result.created_at
        assert isinstance(result.completed_at, str) and result.completed_at
        assert isinstance(result.summary, str) and result.summary
        assert {
            "problem",
            "classical",
            "qaoa",
            "angles",
            "references",
            "limits",
        } <= set(result.report)
        assert result.problem["num_variables"] == 2
        assert {"curve", "recommended_layers", "at_requested"} <= set(result.qaoa)
        assert {"gammas", "betas", "source", "convention"} <= set(result.angles)
        assert {"exact", "qaoa_lambda"} <= set(result.penalty)
        assert result.constraints["constraints"][0]["type"] == "eq_cardinality"
        assert result.qaoa_initial_params() is not None

    def test_recommendations_have_structured_shape(self, completed_validation_job):
        """Each recommendation is ``{level, topic, text}`` with plain text.

        Deliberately does **not** assert ``len(recs) > 0`` — whether any
        rules fire is server-side policy, not divi's contract.
        """
        _, result = completed_validation_job

        for rec in result.recommendations:
            assert set(rec) == {"level", "topic", "text"}
            assert rec["level"] in {"warning", "action", "info"}
            assert isinstance(rec["text"], str) and rec["text"]
            assert "<" not in rec["text"], "text field must be plain (no markup)"

    def test_html_report_renders(self, completed_validation_job):
        """The cached ``result.html`` is the self-contained validation report."""
        _, result = completed_validation_job
        assert 'class="qvr-root"' in result.html

    def test_fetch_by_job_id_returns_equivalent_result(
        self, completed_validation_job, qoro_service
    ):
        """``get_characterization_result(job_id)`` re-reads without resubmitting.

        The ``html`` field is excluded from the equality check by the
        dataclass (``compare=False``), so it is checked separately.
        """
        _, original = completed_validation_job

        refetched = get_characterization_result(
            job_id=original.job_id,
            service=qoro_service,
        )

        assert refetched.job_id == original.job_id
        assert refetched == original
        assert 'class="qvr-root"' in refetched.html

    def test_nonexistent_job_returns_404(self, qoro_service):
        """A bogus UUID surfaces as ``HTTPError 404`` through the public API."""
        with pytest.raises(requests.HTTPError) as exc:
            get_characterization_result(
                job_id="00000000-0000-0000-0000-000000000000",
                service=qoro_service,
            )
        assert exc.value.response.status_code == 404
