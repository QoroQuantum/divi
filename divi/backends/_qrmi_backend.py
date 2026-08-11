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
from collections.abc import Mapping, Sequence
from threading import Event, RLock
from typing import Any

from qiskit import QuantumCircuit

from divi.circuits._payloads import CircuitPayload, bound_circuits
from divi.exceptions import ExecutionCancelledError

from ._circuit_runner import CircuitRunner
from ._execution_result import ExecutionResult
from ._maestro_simulator import _MPS_AUTO_BOND_DIMENSION, _strip_id_gates
from ._pauli_serde import ham_ops_group_for_circuit
from ._shot_allocation import from_wire, per_circuit, validate

logger = logging.getLogger(__name__)

#: ``SimulatorType`` ordinal. Renumbered on a server built ``-DNO_QISKIT_AER``,
#: which the resource cannot report — read ``simulator`` off the result to see
#: what actually ran.
SIMULATOR_QCSIM = 1

METHOD_STATEVECTOR = 0
METHOD_MATRIX_PRODUCT_STATE = 1

#: Qubit count above which MPS is selected automatically, as in MaestroConfig.
_MPS_QUBIT_THRESHOLD = 22

_QREG_RE = re.compile(r"qreg\s+q\[(\d+)\]")


class QRMIBackend(CircuitRunner):
    """Run circuits on a QRMI-managed Maestro server.

    Args:
        resource: A ``qrmi.QuantumResource`` of type ``maestro-local``. Passed
            in rather than constructed, since QRMI cannot discover that type.
        shots: Shots per circuit. Sent explicitly — Maestro defaults to 1.
        track_depth: Record per-circuit logical depth.
        simulator_type: See :data:`SIMULATOR_QCSIM`.
        simulation_method: ``None`` picks statevector, or MPS once the widest
            circuit in a batch crosses ``mps_qubit_threshold``.
        max_bond_dimension: MPS bond-dimension cap. Defaults to 64 under
            auto-MPS.
        truncation_threshold: MPS singular-value truncation threshold.
        mps_qubit_threshold: Auto-MPS cutoff. Match
            :attr:`~divi.backends.MaestroConfig.mps_qubit_threshold` to keep
            both channels routing alike.
        poll_interval: Seconds between status polls. Stay under the server's
            ~10 s session-idle timeout.
    """

    def __init__(
        self,
        resource,
        *,
        shots: int = 1024,
        track_depth: bool = False,
        simulator_type: int = SIMULATOR_QCSIM,
        simulation_method: int | None = None,
        max_bond_dimension: int | None = None,
        truncation_threshold: float | None = None,
        mps_qubit_threshold: int = _MPS_QUBIT_THRESHOLD,
        poll_interval: float = 0.05,
    ):
        super().__init__(shots=shots, track_depth=track_depth)
        self._resource = resource
        self._simulator_type = simulator_type
        self._simulation_method = simulation_method
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

    def _resolve_method(self, max_qubits: int) -> tuple[int, int | None]:
        """Pick the simulation method and bond dimension for a whole batch.

        Sized from the widest circuit, as MaestroSimulator does, so a batch
        routes the same way through either channel.
        """
        method = self._simulation_method
        auto_mps = method is None and max_qubits > self._mps_qubit_threshold
        if method is None:
            method = METHOD_MATRIX_PRODUCT_STATE if auto_mps else METHOD_STATEVECTOR

        bond_dimension = self._max_bond_dimension
        if bond_dimension is None and auto_mps:
            bond_dimension = _MPS_AUTO_BOND_DIMENSION
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
        method: int,
        config: str,
        job_type: str,
        observables: str,
        cancellation_event: Event | None,
    ) -> dict:
        """Submit one circuit and block until the server returns its result.

        Holds the resource lock for the whole task, so concurrent callers
        queue rather than interleave calls into one QRMI resource.
        """
        qrmi = _import_qrmi()
        payload = qrmi.Payload.MaestroLocal(
            input=qasm,
            job_type=job_type,
            qubits=n_qubits,
            simulator_type=self._simulator_type,
            simulation_method=method,
            observables=observables,
            config=config,
        )

        with self._lock:
            return self._drive_task(qrmi, payload, cancellation_event)

    def _drive_task(self, qrmi, payload, cancellation_event: Event | None) -> dict:
        # Re-checked inside the lock: a thread that queued behind a long task
        # would otherwise dispatch a new one after the cancel arrived.
        _raise_if_cancelled(cancellation_event, "before dispatch")

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

        status_type = qrmi.TaskStatus
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
        payloads: Sequence[CircuitPayload] | Mapping[str, str],
        *,
        ham_ops: str | None = None,
        circuit_ham_map: list[list[int]] | None = None,
        shot_groups: list[list[int]] | None = None,
        cancellation_event: Event | None = None,
        **kwargs,
    ) -> ExecutionResult:
        """Run each circuit as its own Maestro task, in order.

        Args:
            payloads: Bound circuits, as payloads or a ``{label: qasm}`` mapping.
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
        _raise_if_cancelled(cancellation_event, "before any circuit was dispatched")

        if ham_ops is not None and shot_groups is not None:
            raise ValueError(
                "shot_groups is incompatible with ham_ops: Maestro computes "
                "expectation values analytically and ignores shot counts. "
                "Pass exactly one."
            )

        circuits = bound_circuits(payloads)
        if not circuits:
            raise ValueError("submit_circuits needs at least one circuit.")

        labels = list(circuits)

        if self.track_depth:
            # Measured before stripping, as MaestroSimulator does — Qiskit
            # counts id as an operation, so the two would disagree otherwise.
            self._depth_history.append(
                [QuantumCircuit.from_qasm_str(q).depth() for q in circuits.values()]
            )

        qasm_strings = [_strip_id_gates(q) for q in circuits.values()]

        if shot_groups is not None:
            shot_ranges = from_wire(shot_groups)
            validate(shot_ranges, len(labels))
            per_circuit_shots = per_circuit(shot_ranges, len(labels))
        else:
            per_circuit_shots = None

        widths = [_n_qubits(q, label) for label, q in zip(labels, qasm_strings)]
        method, bond_dimension = self._resolve_method(max(widths))

        self._ensure_session()

        results = []
        for index, (label, qasm) in enumerate(zip(labels, qasm_strings)):
            if cancellation_event is not None and cancellation_event.is_set():
                raise ExecutionCancelledError(
                    "QRMI batch cancelled after partial completion"
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
                # Maestro emits c[0] leftmost; Qiskit puts it rightmost.
                counts = {bits[::-1]: n for bits, n in raw["counts"].items()}
                results.append({"label": label, "results": counts})
            else:
                # A matched group holds no ``|``; the replace only flattens the
                # fall-back case where every group applies to this circuit.
                group = ham_ops_group_for_circuit(
                    index, ham_ops, circuit_ham_map
                ).replace("|", ";")
                raw = self._run_task(
                    qasm,
                    n_qubits=widths[index],
                    method=method,
                    config=self._config_json(self.shots, bond_dimension),
                    job_type="estimate",
                    observables=group,
                    cancellation_event=cancellation_event,
                )
                terms = group.split(";")
                expvals = dict(zip(terms, raw["expectation_values"]))
                results.append({"label": label, "results": expvals})

        return ExecutionResult(results=results)


def _raise_if_cancelled(event: Event | None, when: str) -> None:
    if event is not None and event.is_set():
        raise ExecutionCancelledError(f"QRMI batch cancelled {when}")


#: What the server says when it has forgotten a session. Matched rather than
#: any message mentioning "session", so a busy or rejected session is not
#: mistaken for an expired one and silently retried.
_STALE_SESSION_MARKERS = ("unknown session", "session does not exist")


def _is_stale_session(exc: Exception) -> bool:
    """Whether a task failure means the server forgot our session."""
    message = str(exc).lower()
    return any(marker in message for marker in _STALE_SESSION_MARKERS)


def _n_qubits(qasm: str, label: str) -> int:
    """Read the register width the server should allocate."""
    match = _QREG_RE.search(qasm)
    if match is None:
        raise ValueError(f"Circuit '{label}' declares no 'qreg q[N]'.")
    return int(match.group(1))


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
