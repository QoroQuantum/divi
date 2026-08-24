# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Run circuits on a Maestro server managed by QRMI.

Reaches the same simulator as :class:`~divi.backends.MaestroSimulator`, over a
Unix socket instead of in-process, and matches its conventions so results agree
across both.
"""

import importlib
import json
import logging
import os
import re
import time
from collections.abc import Sequence
from threading import Event, RLock
from typing import Any

from divi.circuits._payloads import CircuitBatch, CircuitPayload, bound_circuits
from divi.exceptions import ExecutionCancelledError
from divi.qasm import count_qubits

from .._base import CircuitRunner, ExecutionResult
from .._cancellation import raise_if_cancelled
from .._pauli_serde import ham_ops_terms_for_circuit, pad_ham_ops
from .._shot_allocation import per_circuit_or_none
from ._maestro import (
    MaestroConfig,
    _circuit_seed,
    _result_entry,
    require_maestro_config,
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

_NOISE_FIELDS = ("noise_model", "noise_seed", "noise_realizations")


def _native_method(simulation_type: Any) -> str:
    """``MatrixProductState`` -> ``matrix_product_state``."""
    return re.sub(r"(?<!^)(?=[A-Z])", "_", simulation_type.name).lower()


class QRMIBackend(CircuitRunner):
    """Run circuits on a QRMI-managed Maestro server.

    Each circuit is sent as a native Maestro request, which the server validates
    before queuing; an option it does not accept fails the submission.

    Args:
        resource: A ``qrmi.QuantumResource`` of type ``maestro-local``. Passed
            in rather than constructed, since QRMI cannot discover that type.
        shots: Shots per circuit.
        maestro_config: :class:`~divi.backends.MaestroConfig` selecting the
            simulator, simulation method and their options, as for
            :class:`~divi.backends.MaestroSimulator`. Noise is not supported.
            Defaults to ``MaestroConfig()``.
        track_depth: Record per-circuit logical depth.
        poll_interval: Seconds between status polls.
    """

    def __init__(
        self,
        resource,
        *,
        shots: int = 1024,
        maestro_config: MaestroConfig | None = None,
        track_depth: bool = False,
        poll_interval: float = 0.05,
    ):
        super().__init__(shots=shots, track_depth=track_depth)
        self._qrmi = _import_qrmi()
        self._resource = resource
        self.maestro_config = (
            maestro_config if maestro_config is not None else MaestroConfig()
        )
        self._poll_interval = poll_interval
        self._session: str | None = None
        self._prior_token: str | None = None
        # QRMI's resource methods take `&mut self`, so two threads calling one
        # resource raise "Already borrowed". Nothing is lost by serialising:
        # the server executes one task at a time regardless.
        self._lock = RLock()
        self._token_var = f"{resource.resource_id()}_QRMI_JOB_ACQUISITION_TOKEN"

    @property
    def maestro_config(self) -> MaestroConfig:
        """The Maestro settings every submission runs with."""
        return self._maestro_config

    @maestro_config.setter
    def maestro_config(self, value: MaestroConfig) -> None:
        require_maestro_config(value, "maestro_config")
        if noise := [
            name for name in _NOISE_FIELDS if getattr(value, name) is not None
        ]:
            raise ValueError(f"QRMIBackend does not support {noise}.")
        self._maestro_config = value

    @property
    def supports_expval(self) -> bool:
        return True

    @property
    def is_async(self) -> bool:
        # The server runs one task at a time locally; there is no queue to
        # hand back to the caller.
        return False

    def set_seed(self, seed: int) -> None:
        """Seed the simulation, as :meth:`MaestroSimulator.set_seed` does."""
        self.maestro_config = self.maestro_config.override(seed=seed)

    # --- Session -----------------------------------------------------------

    def _ensure_session(self, *, renew: bool = False) -> None:
        """Acquire a session unless the scheduler already exported one.

        Acquiring a second session would strand the plugin's. ``renew``
        replaces a session the server has already dropped — including one the
        scheduler owned, whose token is kept so :meth:`close` can put it back.
        """
        with self._lock:
            exported = os.environ.get(self._token_var)
            if not renew and (self._session is not None or exported is not None):
                return
            if self._session is not None:
                self._release(self._session)
            elif exported is not None:
                self._prior_token = exported
            self._session = self._resource.acquire()
            # QRMI reads the session from the environment on every task call,
            # not from the handle it just returned.
            os.environ[self._token_var] = self._session
            logger.info("Acquired Maestro session %s", self._session)

    def _release(self, session: str) -> None:
        """Release a session, tolerating one the server has already dropped."""
        try:
            self._resource.release(session)
        except RuntimeError as exc:
            logger.debug("Maestro session %s already gone: %s", session, exc)

    def close(self) -> None:
        """Release the session, if this backend acquired one."""
        with self._lock:
            session, self._session = self._session, None
            if session is None:
                return
            if self._prior_token is None:
                os.environ.pop(self._token_var, None)
            else:
                os.environ[self._token_var] = self._prior_token
                self._prior_token = None
            self._release(session)

    def __enter__(self) -> "QRMIBackend":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    # --- Execution ---------------------------------------------------------

    def _simulator(self) -> tuple[dict[str, Any], int | None]:
        """The request's ``simulator`` object, and the seed set on the config."""
        config = self.maestro_config
        options = config._simulator_options()
        options.pop("simulator_type", None)
        options.pop("simulation_type", None)
        seed = options.pop("seed", None)
        simulator = {
            "backend": _NATIVE_BACKENDS[config.simulator_type.name],
            "method": _native_method(config.simulation_type),
            "options": options,
        }
        return simulator, seed

    def _run_task(self, request: dict, cancellation_event: Event | None) -> dict:
        """Submit one request and block until the server returns its result.

        Holds the resource lock for the whole task, so concurrent callers
        queue rather than interleave calls into one QRMI resource.
        """
        payload = self._qrmi.Payload.MaestroLocal(
            input=json.dumps(request),
            job_type="request",
            qubits=request["circuit"]["num_qubits"],
            simulator_type=0,
            simulation_method=0,
            observables="",
            config="{}",
        )

        with self._lock:
            return self._drive_task(payload, cancellation_event)

    def _drive_task(self, payload, cancellation_event: Event | None) -> dict:
        # Re-checked inside the lock: a thread that queued behind a long task
        # would otherwise dispatch a new one after the cancel arrived.
        raise_if_cancelled(cancellation_event, "QRMI batch cancelled before dispatch")

        try:
            task_id = self._resource.task_start(payload)
        except RuntimeError as exc:
            if not _is_stale_session(exc):
                raise
            # The server drops a session whose lease lapsed between two
            # submissions.
            logger.info("Maestro session expired (%s); acquiring a new one", exc)
            self._ensure_session(renew=True)
            task_id = self._resource.task_start(payload)

        status_type = self._qrmi.TaskStatus
        while True:
            if cancellation_event is not None and cancellation_event.is_set():
                # Only unstarted tasks are dropped; one already in the simulator
                # runs to completion server-side.
                self._stop_task(task_id)
                raise ExecutionCancelledError(
                    f"QRMI batch cancelled while task {task_id} was running"
                )
            status = self._resource.task_status(task_id)
            if status == status_type.Completed:
                break
            if status in (status_type.Failed, status_type.Cancelled):
                raise RuntimeError(
                    f"Maestro task {task_id} ended as {status}: "
                    f"{self._task_logs(task_id)}"
                )
            time.sleep(self._poll_interval)

        return json.loads(self._resource.task_result(task_id).value)

    def _task_logs(self, task_id: str) -> str:
        """The server's retained diagnostics for a task, or why they are missing."""
        try:
            return self._resource.task_logs(task_id)
        except RuntimeError as exc:
            return f"no diagnostics available ({exc})"

    def _stop_task(self, task_id: str) -> None:
        """Ask the server to drop a task, tolerating one it has already lost.

        A raise here would surface a user's cancellation as a batch failure.
        """
        try:
            self._resource.task_stop(task_id)
        except RuntimeError as exc:
            logger.debug("Could not stop Maestro task %s: %s", task_id, exc)

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
            cancellation_event: Checked between circuits.
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

        circuits = bound_circuits(payloads)
        if not circuits:
            raise ValueError("submit_circuits needs at least one circuit.")

        labels = list(circuits)
        qasm_strings = list(circuits.values())
        widths = [count_qubits(q) for q in qasm_strings]
        if ham_ops is not None:
            ham_ops = pad_ham_ops(ham_ops, circuit_ham_map, widths)

        self._record_qasm_depths(qasm_strings)

        simulator, seed = self._simulator()
        per_circuit_shots = per_circuit_or_none(shot_groups, len(labels))

        self._ensure_session()

        results: list[dict] = []
        for index, (label, qasm) in enumerate(zip(labels, qasm_strings)):
            raise_if_cancelled(
                cancellation_event, "QRMI batch cancelled after partial completion"
            )

            request: dict[str, Any] = {
                "schema_version": 2,
                "circuit": {"source": qasm, "num_qubits": widths[index]},
                "simulator": simulator,
                "execution": {},
            }
            if (circuit_seed := _circuit_seed(seed, label, qasm)) is not None:
                request["execution"]["seed"] = circuit_seed

            if ham_ops is None:
                request["operation"] = "execute"
                request["execution"]["shots"] = (
                    self.shots
                    if per_circuit_shots is None
                    else per_circuit_shots[index]
                )
                raw = self._run_task(request, cancellation_event)
                # Maestro puts c[0] leftmost; Qiskit (little-endian) puts it rightmost.
                counts = {bits[::-1]: n for bits, n in raw["counts"].items()}
                results.append(_result_entry(label, counts, raw, "counts"))
            else:
                terms = ham_ops_terms_for_circuit(index, ham_ops, circuit_ham_map)
                request["operation"] = "estimate"
                request["observables"] = terms
                raw = self._run_task(request, cancellation_event)
                expvals = dict(zip(terms, raw["expectation_values"]))
                results.append(_result_entry(label, expvals, raw, "expectation_values"))

        return ExecutionResult(
            results=results,
            run_time=sum(entry["metadata"]["time_taken"] for entry in results),
        )


#: What the server says when it has forgotten a session. Matched rather than
#: any message mentioning "session", so a busy or rejected session is not
#: mistaken for an expired one and silently retried.
_STALE_SESSION_MARKERS = (
    "unknown session",
    "session does not exist",
    "session_not_found",
)


def _is_stale_session(exc: Exception) -> bool:
    """Whether a task failure means the server forgot our session."""
    message = str(exc).lower()
    return any(marker in message for marker in _STALE_SESSION_MARKERS)


def _import_qrmi() -> Any:
    """The ``qrmi`` module, imported on first use.

    Imported by name and returned as ``Any``: qrmi is a compiled pyo3
    extension shipping no stubs, so a type checker resolves none of its
    members.
    """
    try:
        qrmi = importlib.import_module("qrmi")
    except ImportError as exc:  # pragma: no cover - exercised by the extra
        raise ImportError(
            "QRMIBackend needs the qrmi package, built from a checkout that "
            "supports the maestro-local resource type."
        ) from exc

    # The fork carries the same version as upstream, so this is the only way
    # to tell them apart — and upstream would otherwise fail much later with a
    # bare AttributeError.
    if not hasattr(qrmi.ResourceType, "MaestroLocal"):
        raise ImportError(
            "The installed qrmi has no maestro-local resource type, so it is "
            "the upstream build rather than Qoro's fork. Both report the same "
            "version; reinstall from a checkout of the fork."
        )
    return qrmi
