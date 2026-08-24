# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for divi.backends.runners._qrmi."""

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock
from types import SimpleNamespace

import pytest

import divi.backends.runners._qrmi as qrmi_module
from divi.backends import QRMIBackend
from divi.exceptions import ExecutionCancelledError
from tests.backends._circuit_runner_contracts import SyncRunnerContractsBase
from tests.backends._qrmi_fakes import FAKE_QRMI, FakeQuantumResource, FakeTaskStatus

BELL_QASM = (
    'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncreg c[2];\n'
    "h q[0];\ncx q[0],q[1];\nmeasure q[0] -> c[0];\nmeasure q[1] -> c[1];\n"
)

#: Captured before the autouse fixture patches it away.
_REAL_IMPORT_QRMI = qrmi_module._import_qrmi


@pytest.fixture(autouse=True)
def fake_qrmi(mocker):
    """Swap the qrmi import for a stand-in, so no wheel is needed."""
    mocker.patch.object(qrmi_module, "_import_qrmi", return_value=FAKE_QRMI)


@pytest.fixture(autouse=True)
def clean_token_env(monkeypatch):
    """Keep the acquisition token out of the ambient environment."""
    monkeypatch.delenv("MAESTRO_LOCAL_QRMI_JOB_ACQUISITION_TOKEN", raising=False)


@pytest.fixture
def resource():
    return FakeQuantumResource()


@pytest.fixture
def backend(resource):
    with QRMIBackend(resource, shots=100) as runner:
        yield runner


class TestCapabilities:
    def test_reports_expval_support_and_synchronous_execution(self, backend):
        assert backend.supports_expval is True
        assert backend.is_async is False

    def test_never_resolves_parameters(self, backend):
        """Maestro's parser evaluates gate arguments at parse time, so a free
        identifier is a parse error — the pipeline must bind first."""
        assert backend.resolves_parameters is False


class TestSampling:
    def test_reverses_bitstrings_to_qiskit_order(self, resource):
        """Maestro writes c[0] leftmost; divi keys results Qiskit-style with
        c[0] rightmost. Without the flip, every result is silently mirrored."""
        resource._counts = {"100": 512}
        with QRMIBackend(resource, shots=512) as backend:
            results = backend.submit_circuits({"c0": BELL_QASM}).results

        assert results == [{"label": "c0", "results": {"001": 512}}]

    def test_sends_shots_explicitly(self, resource):
        """Maestro's own default is 1 shot, so omitting it returns one sample."""
        with QRMIBackend(resource, shots=4096) as backend:
            backend.submit_circuits({"c0": BELL_QASM})

        assert resource.last_config["shots"] == 4096

    def test_declares_the_register_width(self, resource, backend):
        backend.submit_circuits({"c0": BELL_QASM})

        assert resource.last_payload.qubits == 2

    def test_rejects_a_circuit_without_a_qreg(self, backend):
        with pytest.raises(ValueError, match="declares no 'qreg"):
            backend.submit_circuits({"c0": "OPENQASM 2.0;\n"})

    def test_strips_id_gates(self, resource, backend):
        """Maestro's QASM parser does not know the id gate."""
        qasm = BELL_QASM.replace("h q[0];", "h q[0];\nid q[1];\n")

        backend.submit_circuits({"c0": qasm})

        assert "id q[1]" not in resource.last_payload.input

    def test_routes_every_label_back(self, backend):
        results = backend.submit_circuits(
            {"a": BELL_QASM, "b": BELL_QASM, "c": BELL_QASM}
        ).results

        assert [row["label"] for row in results] == ["a", "b", "c"]

    def test_submits_one_task_per_circuit(self, resource, backend):
        backend.submit_circuits({"a": BELL_QASM, "b": BELL_QASM})

        assert len(resource.payloads) == 2
        assert all(p.job_type == "execute" for p in resource.payloads)


class TestExpectationValues:
    def test_keys_results_by_pauli_term(self, resource):
        resource._expectation_values = [0.25, -0.5]
        with QRMIBackend(resource, shots=100) as backend:
            results = backend.submit_circuits(
                {"c0": BELL_QASM}, ham_ops="ZZ;XX"
            ).results

        assert results == [{"label": "c0", "results": {"ZZ": 0.25, "XX": -0.5}}]

    def test_switches_the_job_type_and_forwards_observables(self, resource, backend):
        backend.submit_circuits({"c0": BELL_QASM}, ham_ops="ZZ;XX")

        assert resource.last_payload.job_type == "estimate"
        assert resource.last_payload.observables == "ZZ;XX"

    def test_passes_measurements_through(self, resource, backend):
        """Maestro's estimator ignores terminal measurement, so stripping it
        client-side was dead weight and left the two channels able to drift."""
        backend.submit_circuits({"c0": BELL_QASM}, ham_ops="ZZ")

        assert "measure q[0] -> c[0];" in resource.last_payload.input

    def test_assigns_groups_per_circuit(self, resource, backend):
        backend.submit_circuits(
            {"a": BELL_QASM, "b": BELL_QASM},
            ham_ops="ZZ|XX",
            circuit_ham_map=[[0, 1], [1, 2]],
        )

        assert [p.observables for p in resource.payloads] == ["ZZ", "XX"]

    def test_flattens_group_separators_for_unmapped_circuits(self, resource, backend):
        """A circuit outside every circuit_ham_map range gets the whole ham_ops
        string back, pipes included. Maestro splits only on ';', so an unflattened
        'XX|YY' would be evaluated as one pseudo-term worth 0.0 and keyed under a
        name nothing looks up — both observables silently missing."""
        resource._expectation_values = [0.1, 0.2, 0.3]

        backend.submit_circuits(
            {"aux": BELL_QASM}, ham_ops="ZZ|XX;YY", circuit_ham_map=[[1, 2]]
        )

        assert resource.last_payload.observables == "ZZ;XX;YY"

    def test_rejects_shot_groups(self, backend):
        with pytest.raises(ValueError, match="incompatible with ham_ops"):
            backend.submit_circuits(
                {"c0": BELL_QASM}, ham_ops="ZZ", shot_groups=[[0, 1, 10]]
            )


class TestSimulationMethod:
    def test_switches_to_mps_above_the_qubit_threshold(self, resource, backend):
        wide = BELL_QASM.replace("qreg q[2]", "qreg q[23]")

        backend.submit_circuits({"c0": wide})

        assert resource.last_payload.simulation_method == 1
        assert resource.last_config["matrix_product_state_max_bond_dimension"] == 64

    def test_explicit_method_disables_auto_selection(self, resource):
        wide = BELL_QASM.replace("qreg q[2]", "qreg q[23]")
        with QRMIBackend(resource, shots=100, simulation_method=0) as backend:
            backend.submit_circuits({"c0": wide})

        assert resource.last_payload.simulation_method == 0
        assert "matrix_product_state_max_bond_dimension" not in resource.last_config

    def test_stays_on_statevector_at_the_threshold(self, resource, backend):
        """The cutoff is strict, so exactly 22 qubits is still statevector —
        and no MPS-only knob is sent along with it."""
        at_threshold = BELL_QASM.replace("qreg q[2]", "qreg q[22]")

        backend.submit_circuits({"c0": at_threshold})

        assert resource.last_payload.simulation_method == 0
        assert "matrix_product_state_max_bond_dimension" not in resource.last_config

    def test_the_threshold_is_configurable(self, resource):
        """MaestroConfig exposes the same knob; both channels must be able to
        agree on where auto-MPS kicks in."""
        with QRMIBackend(resource, shots=100, mps_qubit_threshold=1) as backend:
            backend.submit_circuits({"c0": BELL_QASM})

        assert resource.last_payload.simulation_method == 1

    def test_sizes_the_batch_from_its_widest_circuit(self, resource, backend):
        """MaestroSimulator picks one method per batch from max qubits. A
        per-circuit choice would run the small circuit on statevector here and
        diverge from the in-process channel."""
        wide = BELL_QASM.replace("qreg q[2]", "qreg q[23]")

        backend.submit_circuits({"small": BELL_QASM, "wide": wide})

        assert [p.simulation_method for p in resource.payloads] == [1, 1]


def test_shot_groups_allocate_shots_per_circuit(resource, backend):
    backend.submit_circuits(
        {"a": BELL_QASM, "b": BELL_QASM}, shot_groups=[[0, 1, 10], [1, 2, 20]]
    )

    assert [json.loads(p.config)["shots"] for p in resource.payloads] == [10, 20]


def test_rejects_a_qrmi_without_maestro_local(mocker):
    """Qoro's fork and upstream report the same version, so the resource type
    is the only way to tell them apart — and upstream would otherwise fail far
    later with a bare AttributeError on Payload.MaestroLocal."""
    upstream = SimpleNamespace(ResourceType=SimpleNamespace(IBMQuantumSystem=object()))
    mocker.patch.dict(sys.modules, {"qrmi": upstream})

    with pytest.raises(ImportError, match="upstream build"):
        _REAL_IMPORT_QRMI()


def test_forwards_the_simulator_type(resource):
    with QRMIBackend(resource, shots=100, simulator_type=0) as backend:
        backend.submit_circuits({"c0": BELL_QASM})

    assert resource.last_payload.simulator_type == 0


def test_forwards_the_truncation_threshold(resource):
    with QRMIBackend(resource, shots=100, truncation_threshold=1e-8) as backend:
        backend.submit_circuits({"c0": BELL_QASM})

    assert resource.last_config["matrix_product_state_truncation_threshold"] == 1e-8


class TestSession:
    def test_acquires_once_across_submissions(self, resource, backend):
        backend.submit_circuits({"a": BELL_QASM})
        backend.submit_circuits({"b": BELL_QASM})

        assert len(resource.acquired) == 1

    def test_releases_on_close(self, resource):
        backend = QRMIBackend(resource, shots=100)
        backend.submit_circuits({"a": BELL_QASM})
        backend.close()

        assert resource.released == resource.acquired

    def test_close_is_idempotent(self, resource):
        backend = QRMIBackend(resource, shots=100)
        backend.submit_circuits({"a": BELL_QASM})
        backend.close()
        backend.close()

        assert len(resource.released) == 1

    def test_defers_to_a_token_the_scheduler_exported(self, resource, monkeypatch):
        """A resource manager acquires the session and exports the token;
        acquiring a second one would strand the plugin's."""
        monkeypatch.setenv("MAESTRO_LOCAL_QRMI_JOB_ACQUISITION_TOKEN", "7")

        with QRMIBackend(resource, shots=100) as backend:
            backend.submit_circuits({"a": BELL_QASM})

        assert resource.acquired == []
        assert resource.released == []

    def test_exports_the_token_it_acquired(self, resource):
        """QRMI reads the session from the environment, not from the handle."""
        with QRMIBackend(resource, shots=100) as backend:
            backend.submit_circuits({"a": BELL_QASM})
            assert os.environ["MAESTRO_LOCAL_QRMI_JOB_ACQUISITION_TOKEN"] == "1"

        assert "MAESTRO_LOCAL_QRMI_JOB_ACQUISITION_TOKEN" not in os.environ


class TestSessionRenewal:
    """The server drops a session after ~10s idle, which the gap between two
    optimizer iterations clears easily — so renewal is the steady state."""

    def _expire_once(self, resource, message="Failed to create task: Unknown session"):
        original = resource.task_start
        calls = {"n": 0}

        def fail_first(payload):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError(message)
            return original(payload)

        resource.task_start = fail_first
        return calls

    def test_reacquires_and_retries(self, resource, backend):
        self._expire_once(resource)

        results = backend.submit_circuits({"c0": BELL_QASM}).results

        assert len(resource.acquired) == 2
        assert results[0]["label"] == "c0"

    def test_releases_the_session_it_replaces(self, resource, backend):
        self._expire_once(resource)

        backend.submit_circuits({"c0": BELL_QASM})

        assert resource.released == [resource.acquired[0]]

    def test_restores_a_scheduler_token_on_close(self, resource, monkeypatch):
        """A resource manager owns the session and exports the token. If it
        expires we must acquire our own, but close() has to put the job's
        token back rather than leave ours in the environment."""
        monkeypatch.setenv("MAESTRO_LOCAL_QRMI_JOB_ACQUISITION_TOKEN", "scheduler")
        self._expire_once(resource)

        with QRMIBackend(resource, shots=100) as backend:
            backend.submit_circuits({"c0": BELL_QASM})
            assert os.environ["MAESTRO_LOCAL_QRMI_JOB_ACQUISITION_TOKEN"] != "scheduler"

        assert os.environ["MAESTRO_LOCAL_QRMI_JOB_ACQUISITION_TOKEN"] == "scheduler"

    def test_an_unrelated_failure_is_not_treated_as_expiry(self, resource, backend):
        """Only a forgotten session should be retried. Retrying anything else
        masks the real error behind a second, more confusing one."""
        self._expire_once(resource, message="session limit reached")

        with pytest.raises(RuntimeError, match="session limit reached"):
            backend.submit_circuits({"c0": BELL_QASM})

        assert len(resource.acquired) == 1


class TestFailureModes:
    def test_raises_when_the_task_fails(self, resource):
        resource._statuses = [FakeTaskStatus.Failed]
        with QRMIBackend(resource, shots=100) as backend:
            with pytest.raises(RuntimeError, match="ended as"):
                backend.submit_circuits({"c0": BELL_QASM})

    def test_polls_until_the_task_completes(self, resource, backend):
        """Only Completed ends the wait. Treating Queued as terminal would
        fetch a result the server has not written yet."""
        resource._statuses = [
            FakeTaskStatus.Queued,
            FakeTaskStatus.Running,
            FakeTaskStatus.Completed,
        ]
        polls = []
        original = resource.task_status
        resource.task_status = lambda t: (polls.append(t), original(t))[1]

        backend.submit_circuits({"c0": BELL_QASM})

        assert len(polls) == 3

    def test_rejects_an_empty_batch(self, backend):
        with pytest.raises(ValueError, match="at least one circuit"):
            backend.submit_circuits({})


class TestCancellation:
    # The pre-dispatch abort itself is covered by the shared contract suite;
    # only the paths past it are exercised here.

    def test_does_not_dispatch_after_a_cancel_arrives(self, resource, backend):
        """A thread queued behind a long task must not start a new one once the
        cancel has landed — it would submit work the user already stopped."""
        event = Event()
        event.set()
        # Past submit_circuits' own check, so only the in-lock recheck can
        # prevent dispatch.
        backend._lock.acquire()
        backend._lock.release()

        with pytest.raises(ExecutionCancelledError):
            backend._drive_task(object(), event)

        assert resource.payloads == []

    def test_a_failing_stop_still_reports_cancellation(self, resource, backend):
        """task_stop raising would surface a user's cancel as a batch failure,
        which the coordinator reports as Failed rather than Cancelled."""
        event = Event()
        original = resource.task_start

        def cancel_once_submitted(payload):
            event.set()
            return original(payload)

        resource.task_start = cancel_once_submitted

        def refuse(task_id):
            raise RuntimeError("task already gone")

        resource.task_stop = refuse

        with pytest.raises(ExecutionCancelledError):
            backend.submit_circuits({"a": BELL_QASM}, cancellation_event=event)

    def test_stops_a_task_cancelled_while_it_runs(self, resource, backend):
        """Polling is interruptible, so a cancel mid-task must tell the server
        to drop it rather than wait the circuit out."""
        event = Event()
        original = resource.task_start

        def cancel_once_submitted(payload):
            event.set()
            return original(payload)

        resource.task_start = cancel_once_submitted

        with pytest.raises(ExecutionCancelledError, match="while task"):
            backend.submit_circuits({"a": BELL_QASM}, cancellation_event=event)

        assert resource.stopped == ["task-0"]

    def test_aborts_between_circuits(self, resource, backend):
        event = Event()
        original = resource.task_result

        def cancel_after_first(task_id):
            event.set()
            return original(task_id)

        resource.task_result = cancel_after_first

        with pytest.raises(ExecutionCancelledError, match="partial completion"):
            backend.submit_circuits(
                {"a": BELL_QASM, "b": BELL_QASM}, cancellation_event=event
            )

        assert len(resource.payloads) == 1


def test_concurrent_submissions_never_interleave(resource, backend):
    """QRMI resource methods take `&mut self`, so overlapping calls into one
    resource raise "Already borrowed". Every method is probed, not just the
    poll: a lock narrowed to the poll loop would still let start and result
    interleave, which is the regression that breaks the borrow."""
    guard = Lock()
    depth = 0
    overlapped = False

    def probe(method):
        def wrapper(*args, **kwargs):
            nonlocal depth, overlapped
            with guard:
                depth += 1
                overlapped |= depth > 1
            try:
                time.sleep(0.001)
                return method(*args, **kwargs)
            finally:
                with guard:
                    depth -= 1

        return wrapper

    for name in ("acquire", "task_start", "task_status", "task_result"):
        setattr(resource, name, probe(getattr(resource, name)))

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [
            pool.submit(backend.submit_circuits, {f"c{i}": BELL_QASM}) for i in range(4)
        ]
        for future in futures:
            future.result()

    assert overlapped is False
    assert len(resource.payloads) == 4


class TestContracts(SyncRunnerContractsBase):
    @pytest.fixture
    def contract_runner_disabled(self, resource):
        return QRMIBackend(resource, shots=10, track_depth=False)

    @pytest.fixture
    def contract_runner_enabled(self, resource):
        return QRMIBackend(resource, shots=10, track_depth=True)

    @pytest.fixture
    def contract_runner_default(self, resource):
        return QRMIBackend(resource, shots=10)
