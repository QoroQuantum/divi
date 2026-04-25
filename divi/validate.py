# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Top-level QUBO / HUBO validation convenience API.

Usage::

    from divi.validate import validate, get_validation_result

    # Run a new validation (credit cost varies by QUBO size)
    result = validate(
        qubo_matrix,
        target_states=["01", "10"],
        sensitivity=True,
    )
    result.display()  # rich console report
    result             # rich HTML in Jupyter

    # Re-fetch a previous result by job ID (free)
    old_result = get_validation_result("4d0550f5-ffb0-...")
    old_result.display()
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from divi.backends._qoro_service import QoroService
    from divi.backends._validation import QUBOValidationResult
    from divi.hamiltonians._types import HUBOProblemTypes, QUBOProblemTypes
    from divi.qprog.problems._binary import BinaryOptimizationProblem


def validate(
    problem: "QUBOProblemTypes | HUBOProblemTypes | BinaryOptimizationProblem",
    target_states: list[str],
    *,
    service: "QoroService | None" = None,
    sensitivity: bool = False,
    parameter_sweep: bool = False,
    auto_tune: bool = False,
    cost_qubo=None,
    penalty_qubo=None,
    n_qubits: int | None = None,
    ansatz: dict | None = None,
    constraints: list | None = None,
    gamma: float | None = None,
    beta: float | None = None,
) -> "QUBOValidationResult":
    """One-call QUBO/HUBO validation with rich notebook display.

    Converts the problem to wire format, submits it to the Qoro
    Validation Service, and returns a :class:`QUBOValidationResult`
    that renders a styled report in Jupyter.

    Args:
        problem: QUBO matrix (ndarray, sparse, BQM), HUBO dict
            (BinaryPolynomial), or a
            :class:`~divi.qprog.problems.BinaryOptimizationProblem`.
        target_states: Bitstrings of the known optimal / target states.
        service: A :class:`~divi.backends.QoroService` instance.
            If ``None``, one is created from the default API key.
        sensitivity: If ``True``, request per-qubit sensitivity analysis.
        parameter_sweep: If ``True``, request a γ/β parameter sweep.
        auto_tune: If ``True``, request automatic penalty tuning.
        cost_qubo: Optional cost-only QUBO for penalty analysis
            (same types as *problem*).
        penalty_qubo: Optional penalty-only QUBO for penalty analysis.
        n_qubits: Explicit qubit count (auto-detected if omitted).
        ansatz: Ansatz configuration dict
            (e.g. ``{"mixer": "x", "layers": 1}``).
            Note: ``auto_warmstart`` is managed by the backend and
            cannot be overridden from the client.
        constraints: List of constraint descriptors.
        gamma: Fixed γ value (overrides parameter sweep).
        beta: Fixed β value (overrides parameter sweep).

    Returns:
        QUBOValidationResult: Rich result object. Displaying it in
        Jupyter shows a styled HTML report.

    Raises:
        requests.exceptions.HTTPError: On API errors.
        TypeError: If the problem type is not supported.

    Examples:
        >>> import numpy as np
        >>> from divi.validate import validate
        >>> Q = np.array([[-1, 2], [0, -1]])
        >>> result = validate(Q, target_states=["01", "10"])
        >>> result.quality_score
        78.5

    .. note::
        Credit cost scales with QUBO size.
    """
    from divi.backends._qoro_service import QoroService
    from divi.backends._validation import _serialize_qubo_for_wire

    # --- Build the service if not provided ---
    if service is None:
        service = QoroService()

    # --- Serialize the problem ---
    qubo_wire = _serialize_qubo_for_wire(problem)

    # --- Serialize optional cost/penalty QUBOs ---
    cost_wire = None
    if cost_qubo is not None:
        cost_wire = _serialize_qubo_for_wire(cost_qubo)

    penalty_wire = None
    if penalty_qubo is not None:
        penalty_wire = _serialize_qubo_for_wire(penalty_qubo)

    # --- Build options dict ---
    analysis: dict = {}
    if gamma is not None:
        analysis["gamma"] = gamma
    if beta is not None:
        analysis["beta"] = beta
    if sensitivity:
        analysis["sensitivity"] = True
    if parameter_sweep:
        analysis["parameter_sweep"] = True
    if auto_tune:
        analysis["auto_tune"] = True

    options: dict = {}
    if analysis:
        options["analysis"] = analysis
    if ansatz is not None:
        # auto_warmstart is managed server-side; strip it so users
        # can't accidentally override backend heuristics.
        ansatz = {k: v for k, v in ansatz.items() if k != "auto_warmstart"}
        options["ansatz"] = ansatz
    if cost_wire is not None:
        options["cost_qubo"] = cost_wire
    if penalty_wire is not None:
        options["penalty_qubo"] = penalty_wire
    if n_qubits is not None:
        options["n_qubits"] = n_qubits
    if constraints is not None:
        options["constraints"] = constraints

    return service.validate_qubo(
        qubo=qubo_wire,
        target_states=target_states,
        options=options or None,
    )


def get_validation_result(
    job_id: str,
    *,
    service: "QoroService | None" = None,
) -> "QUBOValidationResult":
    """Re-fetch a previous validation result by job ID.

    This does **not** cost any credits — it only retrieves the
    stored result from a previously completed validation run.

    Args:
        job_id: The UUID of a previously submitted validation job.
        service: A :class:`~divi.backends.QoroService` instance.
            If ``None``, one is created from the default API key.

    Returns:
        QUBOValidationResult: The full result including hardness,
        report, state probabilities, and any analysis data that was
        computed during the original run.

    Examples:
        >>> from divi.validate import get_validation_result
        >>> result = get_validation_result("4d0550f5-ffb0-...")
        >>> result.display()   # rich console report
        >>> result.quality_score
        45.89

    """
    from divi.backends._qoro_service import QoroService

    if service is None:
        service = QoroService()

    return service.get_validation_result(job_id)
