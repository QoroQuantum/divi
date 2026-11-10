# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Backend submission of a traced batch."""

from collections.abc import Mapping, Sequence
from dataclasses import replace
from itertools import groupby
from operator import itemgetter
from typing import Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag

from divi.backends import AsyncJobBackend, ExecutionResult, JobStatus
from divi.backends._cancellation import _cancel_job_on_abort, _require_async_backend
from divi.circuits._conversions import (
    _format_bound_param,
    _sparse_pauli_op_to_ham_string,
    dag_to_qasm_body,
    lower_to_basis_gates,
)
from divi.circuits._payloads import CircuitPayload, flatten_bound_payloads, is_bound
from divi.exceptions import (
    ExecutionCancelledError,
    IncrementalEvolutionUnavailableError,
    VJPUnavailableError,
)
from divi.reporting._events import ProgressEvent

from ._compilation import _batch_has_free_parameters, _build_preamble, _compile_batch
from ._result_keys_operations import (
    OBS_GROUP_AXIS,
    PARAM_SET_AXIS,
    _collapse_to_parent_results,
    extract_param_set_idx,
)
from .abc import (
    ChildResults,
    ContractViolation,
    MetaCircuitBatch,
    PipelineEnv,
    PipelineTrace,
    ResultFormat,
)
from .stages import ParameterBindingStage


def _wait_for_async_result(
    backend: AsyncJobBackend, execution_result: ExecutionResult, env: "PipelineEnv"
) -> ExecutionResult:
    """Poll an async backend until job completes, then fetch results."""
    job_id = execution_result.job_id
    if job_id is None:
        raise ValueError("ExecutionResult must have a job_id for async completion")

    # Build the poll callback when progress delivery is bound.
    progress_callback = None
    if env._progress_emitter is not None and env._progress_key is not None:
        progress_emitter = env._progress_emitter
        progress_key = env._progress_key

        def progress_callback(n_polls, status):
            progress_emitter(
                ProgressEvent.polling(
                    progress_key,
                    job_id=job_id,
                    status=status,
                    attempt=n_polls,
                    limit=backend.max_retries,
                )
            )

    status = backend.poll_job_status(
        execution_result,
        loop_until_complete=True,
        verbose=progress_callback is None,
        progress_callback=progress_callback,
        cancellation_event=env.cancellation_event,
    )

    if status != JobStatus.COMPLETED:
        raise RuntimeError(
            f"Backend poll_job_status returned unexpected status {status} for job {job_id}"
        )

    return backend.get_job_results(execution_result)


def _resolve_branch_shot_count(
    branch_key: tuple,
    spec_keys: Sequence[tuple],
    per_group_shots: Mapping[tuple, dict[int, int]],
    per_param_group_shots: Mapping[tuple, dict[int, dict[int, int]]],
) -> int | None:
    """Resolve one compiled branch to its measurement-stage shot allocation."""
    branch_axes = set(branch_key)
    spec_match = next(
        (spec_key for spec_key in spec_keys if set(spec_key) <= branch_axes), None
    )
    if spec_match is None:
        return None

    obs_group = next(value for axis, value in branch_key if axis == OBS_GROUP_AXIS)
    if spec_match not in per_param_group_shots:
        return per_group_shots[spec_match].get(obs_group)

    param_set = next(value for axis, value in branch_key if axis == PARAM_SET_AXIS)
    return per_param_group_shots[spec_match].get(param_set, {}).get(obs_group)


def _build_shot_groups(
    circuits: Mapping[str, str],
    lineage_by_label: Mapping[str, tuple],
    per_group_shots: Mapping[tuple, dict[int, int]] | None = None,
    per_param_group_shots: Mapping[tuple, dict[int, dict[int, int]]] | None = None,
) -> list[list[int]] | None:
    """Translate measurement allocations into backend ``shot_groups``.

    The backend interface accepts ``shot_groups`` as a list of
    ``[start, end, shots]`` triples covering the iteration order of
    ``circuits``. Consecutive circuits with identical shot counts are
    collapsed into a single range. Returns ``None`` if no per-group
    allocation applies (every circuit's spec is unmapped).
    """
    per_group_shots = per_group_shots or {}
    per_param_group_shots = per_param_group_shots or {}
    spec_keys = list(dict.fromkeys((*per_param_group_shots, *per_group_shots)))
    per_circuit_shots = [
        _resolve_branch_shot_count(
            lineage_by_label[label],
            spec_keys,
            per_group_shots,
            per_param_group_shots,
        )
        for label in circuits
    ]

    if all(s is None for s in per_circuit_shots):
        return None

    # Collapse consecutive identical shot counts into [start, end, shots] ranges.
    shot_groups: list[list[int]] = []
    for shots, run in groupby(enumerate(per_circuit_shots), key=itemgetter(1)):
        positions = [i for i, _ in run]
        if shots is None:
            raise ValueError(
                f"Circuit at position {positions[0]} has no per-group shot allocation; "
                "per_group_shots must cover every submitted circuit."
            )
        shot_groups.append([positions[0], positions[-1] + 1, int(shots)])

    return shot_groups


def _measurement_artifacts(batch: MetaCircuitBatch) -> dict[str, Any]:
    """Reconstruct measurement submission metadata from circuit metadata.

    The measurement stage records these on each :class:`MetaCircuit`; this
    gathers them back into the artifact shape used for backend submission and
    dry-run reporting.
    """
    artifacts: dict[str, Any] = {}
    per_group_shots = {
        key: meta.group_shots for key, meta in batch.items() if meta.group_shots
    }
    if per_group_shots:
        artifacts["per_group_shots"] = per_group_shots
    per_param_group_shots = {
        key: meta.param_group_shots
        for key, meta in batch.items()
        if meta.param_group_shots
    }
    if per_param_group_shots:
        artifacts["per_param_group_shots"] = per_param_group_shots
    ham_ops = next(
        (m.backend_ham_ops for m in batch.values() if m.backend_ham_ops is not None),
        None,
    )
    if ham_ops is not None:
        artifacts["ham_ops"] = ham_ops
    return artifacts


def _record_submission_resources(
    env: PipelineEnv,
    circuit_count: int,
    ham_ops: str | None,
    shot_groups: Sequence[list[int]] | None,
) -> None:
    """Record samples requested by one pending backend submission."""
    if ham_ops is not None:
        device_shots = 0
    elif shot_groups is not None:
        device_shots = sum((end - start) * shots for start, end, shots in shot_groups)
    else:
        device_shots = circuit_count * env.effective_shots
    env.artifacts["device_shots"] = device_shots

    if env.estimator_samples is not None:
        env.artifacts["estimator_samples_requested"] = sum(
            env.estimator_samples.by_param_set
        )


def _raise_if_cancelled(env: PipelineEnv) -> None:
    """Raise :class:`ExecutionCancelledError` if the run was cancelled."""
    if env.cancellation_event is not None and env.cancellation_event.is_set():
        raise ExecutionCancelledError("Pipeline execution cancelled before dispatch")


def _record_no_submission(env: PipelineEnv) -> None:
    """Record the resources of a run that submits no circuit, unless cancelled."""
    _raise_if_cancelled(env)
    env.artifacts.update(circuit_count=0, backend_jobs=0, device_shots=0)
    env.artifacts.setdefault("run_time", 0.0)


def _default_execute_fn(
    trace: PipelineTrace,
    env: PipelineEnv,
) -> ChildResults:
    """Default execute: lower MetaCircuit batch to payloads, then run."""
    backend = env.backend
    if (
        _batch_has_free_parameters(trace.final_batch)
        and not backend.resolves_parameters
    ):
        raise ContractViolation(
            "Batch still carries free parameters at execution, but this "
            "backend cannot resolve them. Ensure ParameterBindingStage ran, "
            "or use a backend that sets resolves_parameters."
        )

    payloads, lineage_by_label, submit_kwargs = _lower_batch(
        trace.final_batch, env, trace.env_artifacts
    )
    result = backend.submit_circuits(
        payloads, cancellation_event=env.cancellation_event, **submit_kwargs
    )
    entries = _await_completed_entries(result, env)
    return _collapse_to_parent_results(
        {entry["label"]: entry["results"] for entry in entries}, lineage_by_label
    )


def _execute_vjp(
    trace: PipelineTrace,
    env: PipelineEnv,
    *,
    binding: ParameterBindingStage,
    cotangents: np.ndarray,
    vjps: dict[int, np.ndarray],
) -> ChildResults:
    """Submit the traced circuits for vector-Jacobian products, filling ``vjps``."""
    measured = {
        key: node
        for key, node in trace.stage_expansions[-2].batch.items()
        if not node.is_unmeasured
    }
    artifacts = _measurement_artifacts(measured)
    ham_ops = artifacts.get("ham_ops")
    # Backend expectation values are chosen only when every node shares one.
    observable = next(iter(measured.values())).observable
    if ham_ops is None or observable is None:
        raise VJPUnavailableError(
            "Vector-Jacobian products need the backend to compute expectation values."
        )
    if artifacts.get("per_group_shots") or artifacts.get("per_param_group_shots"):
        raise VJPUnavailableError(
            "Vector-Jacobian products cannot follow per-group shot allocations."
        )
    if any(
        node.result_format is not ResultFormat.EXPVALS for node in measured.values()
    ):
        raise VJPUnavailableError("Vector-Jacobian products need expectation values.")

    payloads, lineage_by_label, _ = _lower_batch(
        binding._run_qasm_payloads(measured), env, artifacts
    )
    param_set_of = {
        label: extract_param_set_idx(branch)
        for label, branch in lineage_by_label.items()
    }
    if sorted(param_set_of.values()) != list(range(len(cotangents))):
        raise VJPUnavailableError(
            "Vector-Jacobian products need exactly one circuit per parameter set."
        )

    (node,) = measured.values()
    (payload,) = payloads
    observables = observable if isinstance(observable, tuple) else (observable,)
    if cotangents.shape[1] != len(observables):
        raise ValueError(
            f"Got {cotangents.shape[1]} cotangents per parameter set for "
            f"{len(observables)} observables."
        )
    terms = ham_ops.split(";")
    slot = {term: i for i, term in enumerate(terms)}
    weights = np.zeros((len(observables), len(terms)))
    for row, obs in enumerate(observables):
        for term, coeff in zip(
            _sparse_pauli_op_to_ham_string(obs).split(";"), obs.coeffs
        ):
            weights[row, slot[term]] += coeff.real

    # Rounded as bound circuits render them, so cost and gradient share a point.
    rounded = tuple(
        (
            label,
            tuple(float(_format_bound_param(v, node.precision)) for v in values),
        )
        for label, values in payload.parameter_sets
    )
    result = env.backend.submit_vjp(
        [replace(payload, parameter_sets=rounded)],
        ham_ops=ham_ops,
        cotangents={
            label: cotangents[idx] @ weights for label, idx in param_set_of.items()
        },
        cancellation_event=env.cancellation_event,
    )
    entries = _await_completed_entries(result, env)
    # A parameter the circuit does not reference has zero derivative.
    column_of = {parameter: i for i, parameter in enumerate(node.parameters)}
    columns = [column_of[parameter] for parameter in payload.parameters]
    for entry in entries:
        vjp = np.zeros(len(node.parameters))
        vjp[columns] = entry["metadata"]["vjp"]
        vjps[param_set_of[entry["label"]]] = vjp
    return _collapse_to_parent_results(
        {entry["label"]: entry["results"] for entry in entries},
        lineage_by_label,
    )


def _execute_incremental(
    trace: PipelineTrace,
    env: PipelineEnv,
    initial_state: QuantumCircuit,
    steps: Sequence[int],
) -> list[ChildResults]:
    """Submit the traced circuit as one step repeated from ``initial_state``.

    Returns:
        The collapsed results at each entry of ``steps``, in order.

    Raises:
        IncrementalEvolutionUnavailableError: If the trace does not measure
            backend expectation values of exactly one bound circuit.
    """
    ham_ops = trace.env_artifacts.get("ham_ops")
    if ham_ops is None:
        raise IncrementalEvolutionUnavailableError(
            "Incremental evolution needs the backend to compute expectation values."
        )
    payloads, lineage_by_label, _ = _lower_batch(
        trace.final_batch, env, trace.env_artifacts
    )
    labels = [label for p in payloads for label, _values in p.parameter_sets]
    if len(labels) != 1 or not is_bound(payloads):
        raise IncrementalEvolutionUnavailableError(
            "Incremental evolution needs exactly one bound circuit."
        )
    (payload,), (label,), (node,) = payloads, labels, trace.final_batch.values()
    initial_qasm = _build_preamble(node.n_qubits) + dag_to_qasm_body(
        circuit_to_dag(lower_to_basis_gates(initial_state)),
        precision=node.precision,
    )
    result = env.backend.submit_incremental_evolution(
        initial_qasm,
        payload.circuit,
        steps=steps,
        ham_ops=ham_ops,
        cancellation_event=env.cancellation_event,
    )
    by_step = {
        int(entry["label"]): entry["results"]
        for entry in _await_completed_entries(result, env)
    }
    return [
        _collapse_to_parent_results({label: by_step[k]}, lineage_by_label)
        for k in steps
    ]


def _lower_batch(
    batch: MetaCircuitBatch, env: PipelineEnv, artifacts: Mapping[str, Any]
) -> tuple[list[CircuitPayload], dict[str, tuple], dict[str, Any]]:
    """Compile ``batch`` to payloads and record what their submission requests.

    Returns the payloads, their ``label -> branch key`` lineage, and the
    ``ham_ops``/``shot_groups`` keyword arguments for the backend.

    Raises:
        ExecutionCancelledError: If the run was cancelled before dispatch.
    """
    payloads, lineage_by_label = _compile_batch(batch, env.param_sets)
    env.artifacts["circuit_count"] = sum(
        len(payload.parameter_sets) for payload in payloads
    )

    submit_kwargs: dict[str, Any] = {}
    ham_ops = artifacts.get("ham_ops")
    if ham_ops is not None:
        submit_kwargs["ham_ops"] = ham_ops

    per_group_shots = artifacts.get("per_group_shots")
    per_param_group_shots = artifacts.get("per_param_group_shots")
    shot_groups = None
    if per_group_shots or per_param_group_shots:
        if not is_bound(payloads):
            raise ValueError(
                "Per-group shot allocation needs bound circuits, but the batch "
                "reached execution with free parameters for the backend to "
                "resolve. Add a ParameterBindingStage to the pipeline so each "
                "circuit is bound before its group's shots are applied."
            )
        shot_groups = _build_shot_groups(
            flatten_bound_payloads(payloads),
            lineage_by_label,
            per_group_shots,
            per_param_group_shots,
        )
        if shot_groups is not None:
            submit_kwargs["shot_groups"] = shot_groups

    _record_submission_resources(
        env,
        env.artifacts["circuit_count"],
        ham_ops,
        shot_groups,
    )
    _raise_if_cancelled(env)
    return payloads, lineage_by_label, submit_kwargs


def _await_completed_entries(result: ExecutionResult, env: PipelineEnv) -> list[dict]:
    """Wait for ``result``, record its jobs and run time, and return its entries."""
    env.artifacts["backend_jobs"] = result.backend_jobs

    if result.is_async():
        backend = _require_async_backend(env.backend)
        with _cancel_job_on_abort(backend, result):
            result = _wait_for_async_result(backend, result, env)

    if result.results is None:
        raise RuntimeError("Backend returned no results")
    env.artifacts["run_time"] = env.artifacts.get("run_time", 0.0) + result.run_time
    return result.results
