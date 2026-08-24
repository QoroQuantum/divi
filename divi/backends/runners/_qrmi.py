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
import time
from collections.abc import Mapping, Sequence
from threading import Event, RLock
from typing import Any

from divi.circuits._payloads import CircuitBatch, CircuitPayload, bound_circuits
from divi.exceptions import ExecutionCancelledError

from .._base import CircuitRunner, ExecutionResult
from .._cancellation import raise_if_cancelled
from .._config import SimulationMethod, Simulator
from .._maestro_protocol import (
    MPS_AUTO_BOND_DIMENSION,
    MPS_QUBIT_THRESHOLD,
    counts_to_little_endian,
    expvals_from_result,
    qasm_n_qubits,
    strip_id_gates,
)
from .._pauli_serde import ham_ops_terms_for_circuit
from .._shot_allocation import per_circuit_or_none

logger = logging.getLogger(__name__)


class QRMIBackend(CircuitRunner):
    """Run circuits on a QRMI-managed Maestro server.

    Args:
        resource: A ``qrmi.QuantumResource`` of type ``maestro-local``. Passed
            in rather than constructed, since QRMI cannot discover that type.
        shots: Shots per circuit. Sent explicitly — Maestro defaults to 1.
        track_depth: Record per-circuit logical depth.
        simulator_type: A :class:`~divi.backends.Simulator`. Its ordinals shift
            on a server built ``-DNO_QISKIT_AER``, which the resource cannot
            report — read ``simulator`` off the result to see what ran.
        simulation_method: A :class:`~divi.backends.SimulationMethod`. ``None``
            picks statevector, or MPS once the widest circuit in a batch
            crosses ``mps_qubit_threshold``.
        max_bond_dimension: MPS bond-dimension cap. Defaults to 64 under
            auto-MPS.
        truncation_threshold: MPS singular-value truncation threshold.
        mps_qubit_threshold: Auto-MPS cutoff. Shares its default with
            :attr:`~divi.backends.MaestroConfig.mps_qubit_threshold`, so both
            channels route a batch the same way.
        poll_interval: Seconds between status polls. Stay under the server's
            ~10 s session-idle timeout.
    """

    def __init__(
        self,
        resource,
        *,
        shots: int = 1024,
        track_depth: bool = False,
        simulator_type: Simulator = Simulator.QCSim,
        simulation_method: SimulationMethod | None = None,
        max_bond_dimension: int | None = None,
        truncation_threshold: float | None = None,
        mps_qubit_threshold: int = MPS_QUBIT_THRESHOLD,
        poll_interval: float = 0.05,
    ):
        super().__init__(shots=shots, track_depth=track_depth)
        self._qrmi = _import_qrmi()
        self._resource = resource
        self._simulator_type = Simulator(simulator_type)
        self._simulation_method = (
            None if simulation_method is None else SimulationMethod(simulation_method)
        )
        self._max_bond_dimension = max_bond_dimension
        self._truncation_threshold = truncation_threshold
        self._mps_qubit_threshold = mps_qubit_threshold
        self._poll_interval = poll_interval
        self._session: str | None = None
        self._prior_token: str | None = None
        # QRMI's resource methods take `&mut self`, so two threads calling one
        # resource raise "Already borrowed". Nothing is lost by serialising:
        # the server executes one task at a time regardless.
        self._lock = RLock()
        self._token_var = f"{resource.resource_id()}_QRMI_JOB_ACQUISITION_TOKEN"

    @property
    def supports_expval(self) -> bool:
        return True

    @property
    def is_async(self) -> bool:
        # The server runs one task at a time locally; there is no queue to
        # hand back to the caller.
        return False

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

    def _resolve_method(self, max_qubits: int) -> tuple[SimulationMethod, int | None]:
        """Pick the simulation method and bond dimension for a whole batch.

        Sized from the widest circuit, as MaestroSimulator does, so a batch
        routes the same way through either channel.
        """
        method = self._simulation_method
        auto_mps = method is None and max_qubits > self._mps_qubit_threshold
        if method is None:
            method = (
                SimulationMethod.MatrixProductState
                if auto_mps
                else SimulationMethod.Statevector
            )

        bond_dimension = self._max_bond_dimension
        if bond_dimension is None and auto_mps:
            bond_dimension = MPS_AUTO_BOND_DIMENSION
        return method, bond_dimension

    def _config_json(self, shots: int, bond_dimension: int | None) -> str:
        config: dict[str, object] = {"shots": shots}
        if bond_dimension is not None:
            config["matrix_product_state_max_bond_dimension"] = bond_dimension
        if self._truncation_threshold is not None:
            config["matrix_product_state_truncation_threshold"] = (
                self._truncation_threshold
            )
        return json.dumps(config)

    def _run_task(
        self,
        qasm: str,
        *,
        n_qubits: int,
        method: SimulationMethod,
        config: str,
        job_type: str,
        observables: str,
        cancellation_event: Event | None,
    ) -> dict:
        """Submit one circuit and block until the server returns its result.

        Holds the resource lock for the whole task, so concurrent callers
        queue rather than interleave calls into one QRMI resource.
        """
        payload = self._qrmi.Payload.MaestroLocal(
            input=qasm,
            job_type=job_type,
            qubits=n_qubits,
            simulator_type=int(self._simulator_type),
            simulation_method=int(method),
            observables=observables,
            config=config,
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
            # The server drops a session after ~10s idle, which the gap
            # between two optimizer iterations clears easily.
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
                # Every circuit the server cannot run lands here, including
                # QASM it cannot parse. It reports no reason, so the log is
                # the only place to find one.
                raise RuntimeError(
                    f"Maestro task {task_id} ended as {status}. The server "
                    "gives no reason; check its log. Most often the circuit "
                    "used a gate Maestro does not implement, or QASM it could "
                    "not parse."
                )
            time.sleep(self._poll_interval)

        return json.loads(self._resource.task_result(task_id).value)

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
                instead of counts.
            circuit_ham_map: Assigns each ``|``-delimited ``ham_ops`` group to a
                slice of the circuit list.
            shot_groups: Per-circuit ``[start, end, shots]`` triples. Sampling
                only.
            cancellation_event: Checked between circuits.
            **kwargs: Ignored, for interface parity.

        Returns:
            ExecutionResult of counts per label, or expectation values keyed by
            Pauli term.
        """
        raise_if_cancelled(
            cancellation_event,
            "QRMI batch cancelled before any circuit was dispatched",
        )
        self._reject_shot_groups_with_ham_ops(ham_ops, shot_groups)

        circuits = bound_circuits(payloads)
        if not circuits:
            raise ValueError("submit_circuits needs at least one circuit.")

        labels = list(circuits)

        # Measured before stripping — Qiskit counts id as an operation, so
        # MaestroSimulator would otherwise disagree.
        self._record_qasm_depths(circuits.values())

        qasm_strings = [strip_id_gates(q) for q in circuits.values()]

        per_circuit_shots = per_circuit_or_none(shot_groups, len(labels))

        widths = [qasm_n_qubits(q, label) for label, q in zip(labels, qasm_strings)]
        method, bond_dimension = self._resolve_method(max(widths))

        self._ensure_session()

        results: list[dict] = []
        for index, (label, qasm) in enumerate(zip(labels, qasm_strings)):
            raise_if_cancelled(
                cancellation_event, "QRMI batch cancelled after partial completion"
            )

            if ham_ops is None:
                shots = (
                    per_circuit_shots[index]
                    if per_circuit_shots is not None
                    else self.shots
                )
                raw = self._run_task(
                    qasm,
                    n_qubits=widths[index],
                    method=method,
                    config=self._config_json(shots, bond_dimension),
                    job_type="execute",
                    observables="",
                    cancellation_event=cancellation_event,
                )
                results.append(
                    {
                        "label": label,
                        "results": counts_to_little_endian(raw["counts"]),
                    }
                )
            else:
                terms = ham_ops_terms_for_circuit(index, ham_ops, circuit_ham_map)
                raw = self._run_task(
                    qasm,
                    n_qubits=widths[index],
                    method=method,
                    config=self._config_json(self.shots, bond_dimension),
                    job_type="estimate",
                    observables=";".join(terms),
                    cancellation_event=cancellation_event,
                )
                results.append(
                    {
                        "label": label,
                        "results": expvals_from_result(raw, terms),
                    }
                )

        return ExecutionResult(results=results)


#: What the server says when it has forgotten a session. Matched rather than
#: any message mentioning "session", so a busy or rejected session is not
#: mistaken for an expired one and silently retried.
_STALE_SESSION_MARKERS = ("unknown session", "session does not exist")


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
