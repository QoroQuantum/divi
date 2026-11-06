# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Adjoint differentiation on the Qoro service."""

from collections.abc import Mapping, Sequence
from threading import Event
from typing import TYPE_CHECKING

import numpy as np

from divi.backends._base import ExecutionResult
from divi.backends._cancellation import _cancel_job_on_abort, raise_if_cancelled
from divi.backends._job_status import JobFailedError
from divi.backends._pauli_serde import pad_ham_ops
from divi.circuits._payloads import CircuitPayload
from divi.exceptions import VJPUnavailableError
from divi.qasm import count_qubits

from ._maestro import _compile_adjoint_programs, _vjp_result_entries

if TYPE_CHECKING:
    from ._qoro import JobType, QoroService


def run_adjoint_job(
    service: "QoroService",
    job_type: "JobType",
    payloads: Sequence[CircuitPayload],
    *,
    ham_ops: str,
    cotangents: Mapping[str, np.ndarray],
    cancellation_event: Event | None,
) -> ExecutionResult:
    """Run ``payloads`` as one ``job_type`` job whose parameter sets carry their
    cotangents, wait for it, and return the entries
    :meth:`~divi.backends.CircuitRunner.submit_vjp` specifies.

    The service differentiates each bound row over its gate angles; the angle
    map compiled here takes the gradients back to the payload's parameters.

    Raises:
        VJPUnavailableError: If a circuit cannot be differentiated, or the
            service fails the job or returns a gradient that does not fit.
        ValueError: If one parameter set, with its cotangent, exceeds the
            per-request payload cap.
    """
    raise_if_cancelled(
        cancellation_event, "Qoro job cancelled before any circuit was dispatched"
    )
    compiled = _compile_adjoint_programs(payloads)
    ham_ops = pad_ham_ops(
        ham_ops, None, [count_qubits(payload.circuit) for payload in payloads]
    )
    terms = ham_ops.split(";")

    job_config = service.job_config
    maestro_config, device_config = service._configs_for_target(job_config, None, None)
    envelope = service._fragment_envelope(ham_ops, None, None, job_config.shots)
    call_plan = service._build_parametric_call_plan(
        payloads, ham_ops, None, None, envelope, cotangents
    )
    submitted = service._dispatch_job(
        call_plan,
        job_config=job_config,
        maestro_config=maestro_config,
        device_config=device_config,
        job_type=job_type,
        envelope=envelope,
    )
    try:
        with _cancel_job_on_abort(service, submitted):
            service.poll_job_status(
                submitted,
                loop_until_complete=True,
                cancellation_event=cancellation_event,
            )
    except JobFailedError as exc:
        raise VJPUnavailableError(
            f"The service could not differentiate the circuits: {exc}"
        ) from exc
    fetched = service.get_job_results(submitted)
    if fetched.results is None:
        raise RuntimeError("Backend returned no results")

    by_label = {entry["label"]: entry for entry in fetched.results}
    time_taken = fetched.run_time / len(by_label)
    results: list[dict] = []
    for payload, (_, angle_matrix, _) in zip(payloads, compiled):
        labels = [label for label, _values in payload.parameter_sets]
        try:
            vjp = np.array(
                [by_label[label]["metadata"]["vjp"] for label in labels],
                dtype=np.float64,
            )
        except (KeyError, ValueError) as exc:
            raise VJPUnavailableError(
                "The service returned no gradient for every row."
            ) from exc
        if vjp.shape != (len(labels), angle_matrix.shape[0]):
            raise VJPUnavailableError(
                "The service differentiated a different circuit than the "
                "one compiled here."
            )
        raw = {
            "expectation_values": [
                [by_label[label]["results"][term] for term in terms] for label in labels
            ],
            "vjp": vjp,
        }
        results.extend(
            _vjp_result_entries(labels, terms, raw, angle_matrix, time_taken)
        )
    return fetched.with_results(results, run_time=fetched.run_time)
