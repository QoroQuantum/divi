# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Run circuits on a Maestro server managed by QRMI."""

import importlib
import json
import logging
import os
import re
import time
from collections.abc import Mapping, Sequence
from threading import Event, RLock
from typing import Any

import numpy as np
from qiskit import QuantumCircuit

from divi.circuits._payloads import (
    CircuitBatch,
    CircuitPayload,
    flatten_bound_payloads,
)
from divi.exceptions import ExecutionCancelledError, VJPUnavailableError
from divi.qasm import count_clbits, count_qubits

from .._base import ExecutionResult
from .._cancellation import raise_if_cancelled
from .._pauli_serde import ham_ops_terms_for_circuit, pad_ham_ops
from .._shot_allocation import expand_shot_groups
from ._maestro import (
    MaestroBaseBackend,
    MaestroConfig,
    _bind_parameter_sets,
    _circuit_seed,
    _compile_adjoint_programs,
    _little_endian_counts,
    _lower_for_statevector,
    _prepare_noisy_circuits,
    _result_entry,
    _vjp_inputs,
    _vjp_result_entries,
)

logger = logging.getLogger(__name__)

#: Native request names for ``maestro.SimulatorType`` members.
_NATIVE_BACKENDS = {
    "QCSim": "qcsim",
    "CompositeQCSim": "composite_qcsim",
    "QiskitAer": "aer",
    "CompositeQiskitAer": "composite_aer",
    "Gpu": "gpu",
    "DistributedGpu": "distributed_gpu",
    "DistributedMpiGpu": "distributed_mpi_gpu",
    "QuestSim": "quest",
}

#: ``noise_realizations`` that Maestro's ``full_noise_execute`` and
#: ``full_noise_estimate`` default to, which a local run leaves in force.
_LOCAL_EXECUTE_REALIZATIONS = 64
_LOCAL_ESTIMATE_REALIZATIONS = 100

#: What the server reports for a session it no longer has, or QRMI for a
#: resource holding none.
_LOST_SESSION_MARKERS = ("session_not_found", "No session")

#: Task failures after which a local run would also have given up on adjoint
#: differentiation.
_VJP_UNAVAILABLE_CODES = frozenset({"insufficient_resources"})


class _TaskFailedError(RuntimeError):
    """A Maestro task the server ran to a failure."""

    def __init__(self, message: str, code: str | None):
        super().__init__(message)
        self.code = code


def _native_method(simulation_type: Any) -> str:
    """``MatrixProductState`` -> ``matrix_product_state``."""
    return re.sub(r"(?<!^)(?=[A-Z])", "_", simulation_type.name).lower()


def _simulator(config: MaestroConfig) -> dict[str, Any]:
    """The request's ``simulator`` object for ``config``."""
    options = config._simulator_options()
    for name in ("simulator_type", "simulation_type", "seed"):
        options.pop(name, None)
    return {
        "backend": _NATIVE_BACKENDS[config.simulator_type.name],
        "method": _native_method(config.simulation_type),
        "options": options,
    }


def _request(
    operation: str, qasm: str, simulator: dict[str, Any], **fields: Any
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "operation": operation,
        "circuit": {"source": qasm, "num_qubits": count_qubits(qasm)},
        "simulator": simulator,
        **fields,
    }


def _circuit_request(
    qasm: str,
    simulator: dict[str, Any],
    execution: dict[str, Any],
    *,
    shots: int,
    terms: list[str] | None,
) -> dict[str, Any]:
    """An ``execute`` request for ``shots``, or an ``estimate`` of ``terms``."""
    if terms is not None:
        return _request(
            "estimate", qasm, simulator, execution=execution, observables=terms
        )
    request = _request(
        "execute", qasm, simulator, execution=execution | {"shots": shots}
    )
    if clbits := count_clbits(qasm):
        request["circuit"]["num_clbits"] = clbits
    return request


def _noise_block(
    config: MaestroConfig, label: str, qasm: str, width: int, estimating: bool
) -> dict[str, Any] | None:
    """The request's ``noise`` object, or None when no channel acts on the circuit."""
    if config.noise_model is None or not (
        channels := config.noise_model.native_channels(
            width, include_readout=not estimating
        )
    ):
        return None
    noise = {
        "channels": channels,
        "realizations": config._noise_realization_kwargs().get(
            "noise_realizations",
            _LOCAL_ESTIMATE_REALIZATIONS if estimating else _LOCAL_EXECUTE_REALIZATIONS,
        ),
    }
    if (seed := _circuit_seed(config.noise_seed, label, qasm)) is not None:
        noise["seed"] = seed
    return noise


def _active_amplitudes(amplitudes: np.ndarray, active: Sequence[int]) -> np.ndarray:
    """The entries of a full-register state in which only ``active`` qubits vary.

    Idle qubits stay in ``|0>``, so these entries carry the whole state.
    """
    indices = [
        sum(((k >> bit) & 1) << qubit for bit, qubit in enumerate(active))
        for k in range(2 ** len(active))
    ]
    return amplitudes[indices]


class QRMIBackend(MaestroBaseBackend):
    """Run circuits on a QRMI-managed Maestro server.

    Each circuit is sent as a native Maestro request, which the server validates
    before queuing; an option it does not accept fails the submission. Results
    match :class:`~divi.backends.MaestroSimulator`'s for the same Maestro
    version, configuration and seeds.

    Args:
        resource: A ``qrmi.QuantumResource`` of type ``maestro-local``. Passed
            in rather than constructed, since QRMI cannot discover that type.
            A session its scheduler exported as
            ``<resource_id>_QRMI_JOB_ACQUISITION_TOKEN`` before the resource
            was built is used and never released.
        shots: Shots per circuit. Defaults to 5000.
        maestro_config: :class:`~divi.backends.MaestroConfig` selecting the
            simulator, simulation method, their options and any noise model,
            as for :class:`~divi.backends.MaestroSimulator`. Defaults to
            ``MaestroConfig()``.
        track_depth: Record per-circuit logical depth.
        force_sampling: If True, route observable measurements through
            shot-based sampling instead of the server's native estimation.
        poll_interval: Seconds between status polls.
    """

    def __init__(
        self,
        resource,
        *,
        shots: int = 5000,
        maestro_config: MaestroConfig | None = None,
        track_depth: bool = False,
        force_sampling: bool = False,
        poll_interval: float = 0.05,
    ):
        super().__init__(
            shots=shots,
            maestro_config=maestro_config,
            track_depth=track_depth,
            force_sampling=force_sampling,
        )
        self._qrmi = _import_qrmi()
        self._resource = resource
        self._poll_interval = poll_interval
        self._session: str | None = None
        self._borrowed_session = (
            f"{resource.resource_id()}_QRMI_JOB_ACQUISITION_TOKEN" in os.environ
        )
        # QRMI's resource methods take `&mut self`, so two threads calling one
        # resource raise "Already borrowed".
        self._lock = RLock()

    @property
    def is_async(self) -> bool:
        return False

    def _ensure_session(self) -> None:
        """Acquire a session unless one is held or borrowed. Call under the lock."""
        if self._session is None and not self._borrowed_session:
            self._session = self._resource.acquire()
            logger.info("Acquired Maestro session %s", self._session)

    def _replace_lost_session(self) -> None:
        """Acquire a fresh session, which this backend then owns. Call under the lock."""
        self._session = self._resource.acquire()
        self._borrowed_session = False
        logger.info("Replaced a lost Maestro session with %s", self._session)

    def close(self) -> None:
        """Release the session, if this backend acquired one."""
        with self._lock:
            session, self._session = self._session, None
            if session is None:
                return
            try:
                self._resource.release(session)
            except RuntimeError as exc:
                logger.debug("Maestro session %s already gone: %s", session, exc)

    def __enter__(self) -> "QRMIBackend":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def _run_task(self, request: dict, cancellation_event: Event | None) -> dict:
        """Submit one request and block until the server returns its result.

        Holds the resource lock for the whole task, so concurrent callers
        queue rather than interleave calls into one QRMI resource.
        """
        payload = self._qrmi.maestro.request_payload(request)
        with self._lock:
            return self._drive_task(payload, cancellation_event)

    def _drive_task(self, payload, cancellation_event: Event | None) -> dict:
        """Start one task, wait for it and return its result. Call under the lock.

        A cancellation that arrived while this thread waited for the lock stops
        it before dispatch; any exit other than completion or a reported
        failure stops the task on the server.
        """
        raise_if_cancelled(cancellation_event, "QRMI batch cancelled before dispatch")
        self._ensure_session()
        try:
            task_id = self._resource.task_start(payload)
        except self._qrmi.ConfigError as exc:
            if not any(marker in str(exc) for marker in _LOST_SESSION_MARKERS):
                raise
            logger.info("Maestro session lost (%s)", exc)
            self._replace_lost_session()
            task_id = self._resource.task_start(payload)

        status_type = self._qrmi.TaskStatus
        finished = (status_type.Completed, status_type.Failed, status_type.Cancelled)
        try:
            while True:
                if cancellation_event is not None and cancellation_event.is_set():
                    raise ExecutionCancelledError(
                        f"QRMI batch cancelled while task {task_id} was running"
                    )
                if (status := self._resource.task_status(task_id)) in finished:
                    break
                time.sleep(self._poll_interval)
        except BaseException:
            self._stop_task(task_id)
            raise

        if status != status_type.Completed:
            code, reason = self._task_failure(task_id)
            raise _TaskFailedError(
                f"Maestro task {task_id} ended as {status}: {reason}", code
            )
        return json.loads(self._resource.task_result(task_id).value)

    def _task_failure(self, task_id: str) -> tuple[str | None, str]:
        """The server's error code for a failed task, and its account of why."""
        try:
            logs = json.loads(self._resource.task_logs(task_id))
        except (RuntimeError, ValueError) as exc:
            return None, f"no diagnostics available ({exc})"
        error = logs.get("error")
        if isinstance(error, dict) and "message" in error:
            code = error.get("code")
            reason = f"{code}: {error['message']}"
        else:
            code, reason = None, logs.get("failure_message") or "no failure recorded"
        output = (logs.get("logs") or "").strip()
        return code, f"{reason}\n{output}" if output else reason

    def _stop_task(self, task_id: str) -> None:
        """Ask the server to stop a task; a refusal is logged, not raised.

        The server replies once the task's worker has exited.
        """
        try:
            self._resource.task_stop(task_id)
        except RuntimeError as exc:
            logger.warning("Could not stop Maestro task %s: %s", task_id, exc)

    def submit_circuits(
        self,
        payloads: Sequence[CircuitPayload] | CircuitBatch,
        *,
        ham_ops: str | None = None,
        circuit_ham_map: list[list[int]] | None = None,
        shot_groups: list[list[int]] | None = None,
        cancellation_event: Event | None = None,
        **kwargs,
    ) -> ExecutionResult:
        """Run each circuit as its own Maestro task, in order.

        Args:
            payloads: Bound circuits, as payloads or any collection of
                already-resolved circuits.
            ham_ops: Semicolon-separated Pauli terms, e.g. ``"ZI;IZ"``. Switches
                to Maestro's analytic estimator, returning expectation values
                instead of counts. Terms shorter than a circuit are padded onto
                its first qubits, with a warning.
            circuit_ham_map: Assigns each ``|``-delimited ``ham_ops`` group to a
                slice of the circuit list.
            shot_groups: Per-circuit ``[start, end, shots]`` triples. Sampling
                only.
            cancellation_event: Checked between circuits, and while one runs.
            **kwargs: Rejected with ``TypeError``.

        Returns:
            ExecutionResult of counts per label, or expectation values keyed by
            Pauli term, with the server's other outputs for each circuit under
            ``"metadata"``.
        """
        self._reject_unknown_options(kwargs)
        raise_if_cancelled(
            cancellation_event,
            "QRMI batch cancelled before any circuit was dispatched",
        )
        self._reject_shot_groups_with_ham_ops(ham_ops, shot_groups)

        circuits = flatten_bound_payloads(payloads)
        if not circuits:
            raise ValueError("submit_circuits needs at least one circuit.")

        labels = list(circuits)
        qasm_strings = list(circuits.values())
        widths = [count_qubits(q) for q in qasm_strings]
        if ham_ops is not None:
            ham_ops = pad_ham_ops(ham_ops, circuit_ham_map, widths)

        self._record_qasm_depths(qasm_strings)

        config = self.maestro_config
        qasm_strings = _prepare_noisy_circuits(
            config, qasm_strings, estimating=ham_ops is not None
        )
        simulator = _simulator(config)
        per_circuit_shots = expand_shot_groups(shot_groups, len(labels))

        results: list[dict] = []
        for index, (label, qasm) in enumerate(zip(labels, qasm_strings)):
            raise_if_cancelled(
                cancellation_event, "QRMI batch cancelled after partial completion"
            )
            terms = (
                None
                if ham_ops is None
                else ham_ops_terms_for_circuit(index, ham_ops, circuit_ham_map)
            )
            seed = _circuit_seed(config.seed, label, qasm)
            request = _circuit_request(
                qasm,
                simulator,
                {} if seed is None else {"seed": seed},
                shots=self.shots
                if per_circuit_shots is None
                else per_circuit_shots[index],
                terms=terms,
            )
            if noise := _noise_block(
                config, label, qasm, widths[index], terms is not None
            ):
                request["noise"] = noise

            raw = self._run_task(request, cancellation_event)
            if terms is None:
                counts = _little_endian_counts(raw["counts"], count_clbits(qasm))
                results.append(_result_entry(label, counts, raw, "counts"))
            else:
                expvals = dict(zip(terms, raw["expectation_values"]))
                results.append(_result_entry(label, expvals, raw, "expectation_values"))

        return ExecutionResult(
            results=results,
            run_time=sum(entry["metadata"]["time_taken"] for entry in results),
        )

    def submit_vjp(
        self,
        payloads: Sequence[CircuitPayload],
        *,
        ham_ops: str,
        cotangents: Mapping[str, np.ndarray],
        cancellation_event: Event | None = None,
    ) -> ExecutionResult:
        """Expectation values and vector-Jacobian products by adjoint differentiation.

        One ``adjoint_vjp`` task per payload. See
        :meth:`~divi.backends.CircuitRunner.submit_vjp`. Terms shorter than a
        circuit are padded onto its first qubits, as in :meth:`submit_circuits`.

        Raises:
            VJPUnavailableError: If
                :attr:`~divi.backends.MaestroBaseBackend.supports_vjp` is
                False, a payload's circuit cannot be differentiated, or the
                server refuses to on the configured simulator or runs out of
                memory.
        """
        simulator = _simulator(self._vjp_config())
        raise_if_cancelled(
            cancellation_event,
            "QRMI batch cancelled before any circuit was dispatched",
        )
        compiled = _compile_adjoint_programs(payloads)
        terms = pad_ham_ops(
            ham_ops, None, [count_qubits(payload.circuit) for payload in payloads]
        ).split(";")
        if self.track_depth:
            self._record_qasm_depths(_bind_parameter_sets(payloads))

        results: list[dict] = []
        for payload, (_program, angle_matrix, angle_offset) in zip(payloads, compiled):
            raise_if_cancelled(
                cancellation_event, "QRMI batch cancelled after partial completion"
            )
            labels, angles, weights = _vjp_inputs(
                payload, angle_matrix, angle_offset, cotangents
            )
            # Any bound row: parameter_rows supplies every row's gate angles.
            circuit = next(_bind_parameter_sets([payload]))
            request = _request(
                "adjoint_vjp",
                circuit,
                simulator,
                observables=terms,
                parameter_rows=angles.tolist(),
                cotangents=weights.tolist(),
            )
            try:
                raw = self._run_task(request, cancellation_event)
            except (self._qrmi.UnsupportedFunctionError, _TaskFailedError) as exc:
                if (
                    isinstance(exc, _TaskFailedError)
                    and exc.code not in _VJP_UNAVAILABLE_CODES
                ):
                    raise
                raise VJPUnavailableError(
                    f"Maestro cannot differentiate the circuit: {exc}"
                ) from exc
            results.extend(
                _vjp_result_entries(
                    labels, terms, raw, angle_matrix, raw["time_taken"] / len(labels)
                )
            )

        return ExecutionResult(
            results=results,
            run_time=sum(entry["metadata"]["time_taken"] for entry in results),
        )

    def statevector(
        self, circuit: str | QuantumCircuit, include_idle_qubits: bool = True
    ) -> np.ndarray:
        """Output amplitudes of ``circuit``, ignoring its final measurements, with
        qubit 0 the least-significant bit of the basis index.

        Args:
            circuit: OpenQASM 2.0 string or Qiskit circuit.
            include_idle_qubits: Span every qubit of the register. ``False`` keeps
                only the qubits a gate acts on, in ascending index order, so the
                vector is ``2**n_active`` long.

        Raises:
            ValueError: If the configuration has a noise model, which a state
                vector cannot carry.
        """
        config = self.maestro_config
        qasm, global_phase, active = _lower_for_statevector(
            config, circuit, include_idle_qubits
        )
        raw = self._run_task(_request("statevector", qasm, _simulator(config)), None)
        amplitudes = np.array([complex(re, im) for re, im in raw["amplitudes"]])
        if not include_idle_qubits:
            amplitudes = _active_amplitudes(amplitudes, active)
        return np.exp(1j * global_phase) * amplitudes


def _import_qrmi() -> Any:
    """The ``qrmi`` module and its ``qrmi.maestro`` helpers, imported on first use."""
    try:
        qrmi = importlib.import_module("qrmi")
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "QRMIBackend needs Qoro's fork of qrmi "
            "(https://github.com/QoroQuantum/qrmi), installed at the commit your "
            "Maestro server is tested against."
        ) from exc
    if not hasattr(qrmi.ResourceType, "MaestroLocal"):
        raise ImportError(
            "The installed qrmi has no maestro-local resource type, so it is "
            "the upstream build rather than Qoro's fork. Both report the same "
            "version; reinstall from a checkout of the fork."
        )
    importlib.import_module("qrmi.maestro")
    return qrmi
