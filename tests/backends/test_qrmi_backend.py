# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for divi.backends.runners._qrmi."""

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock
from types import SimpleNamespace

import maestro
import pytest

import divi.backends.runners._qrmi as qrmi_module
from divi.backends import MaestroConfig, QRMIBackend
from divi.backends.runners._maestro import _circuit_seed
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

        assert results[0]["label"] == "c0"
        assert results[0]["results"] == {"001": 512}

    def test_returns_the_servers_other_outputs_as_metadata(self, backend):
        result = backend.submit_circuits({"c0": BELL_QASM})

        assert result.results[0]["metadata"]["seed"] == 7
        assert result.run_time == pytest.approx(0.001)

    def test_sends_a_native_execute_request(self, resource):
        with QRMIBackend(resource, shots=4096) as backend:
            backend.submit_circuits({"c0": BELL_QASM})

        request = resource.last_request
        assert resource.payloads[-1].job_type == "request"
        assert request["schema_version"] == 2
        assert request["operation"] == "execute"
        assert request["circuit"] == {"source": BELL_QASM, "num_qubits": 2}
        assert request["execution"] == {"shots": 4096}

    def test_routes_every_label_back(self, backend):
        results = backend.submit_circuits(
            {"a": BELL_QASM, "b": BELL_QASM, "c": BELL_QASM}
        ).results

        assert [row["label"] for row in results] == ["a", "b", "c"]

    def test_submits_one_task_per_circuit(self, resource, backend):
        backend.submit_circuits({"a": BELL_QASM, "b": BELL_QASM})

        assert len(resource.payloads) == 2

    def test_shot_groups_allocate_shots_per_circuit(self, resource, backend):
        backend.submit_circuits(
            {"a": BELL_QASM, "b": BELL_QASM}, shot_groups=[[0, 1, 10], [1, 2, 20]]
        )

        assert [r["execution"]["shots"] for r in resource.requests] == [10, 20]


class TestExpectationValues:
    def test_keys_results_by_pauli_term(self, resource):
        resource._expectation_values = [0.25, -0.5]
        with QRMIBackend(resource, shots=100) as backend:
            results = backend.submit_circuits(
                {"c0": BELL_QASM}, ham_ops="ZZ;XX"
            ).results

        assert results[0]["results"] == {"ZZ": 0.25, "XX": -0.5}

    def test_sends_an_estimate_request_without_shots(self, resource, backend):
        """The native estimator rejects execution.shots."""
        backend.submit_circuits({"c0": BELL_QASM}, ham_ops="ZZ;XX")

        request = resource.last_request
        assert request["operation"] == "estimate"
        assert request["observables"] == ["ZZ", "XX"]
        assert "shots" not in request["execution"]

    def test_pads_short_terms_onto_the_first_qubits(self, resource, backend):
        wide = BELL_QASM.replace("qreg q[2]", "qreg q[3]")

        with pytest.warns(UserWarning):
            backend.submit_circuits({"c0": wide}, ham_ops="ZZ")

        assert resource.last_request["observables"] == ["ZZI"]

    def test_assigns_groups_per_circuit(self, resource, backend):
        backend.submit_circuits(
            {"a": BELL_QASM, "b": BELL_QASM},
            ham_ops="ZZ|XX",
            circuit_ham_map=[[0, 1], [1, 2]],
        )

        assert [r["observables"] for r in resource.requests] == [["ZZ"], ["XX"]]

    def test_flattens_group_separators_for_unmapped_circuits(self, resource, backend):
        """A circuit outside every circuit_ham_map range gets every group's
        terms, each its own observable."""
        resource._expectation_values = [0.1, 0.2, 0.3]

        backend.submit_circuits(
            {"aux": BELL_QASM}, ham_ops="ZZ|XX;YY", circuit_ham_map=[[1, 2]]
        )

        assert resource.last_request["observables"] == ["ZZ", "XX", "YY"]

    def test_rejects_shot_groups(self, backend):
        with pytest.raises(ValueError, match="incompatible with ham_ops"):
            backend.submit_circuits(
                {"c0": BELL_QASM}, ham_ops="ZZ", shot_groups=[[0, 1, 10]]
            )


class TestMaestroConfig:
    def test_defaults_to_maestros_defaults(self, resource, backend):
        backend.submit_circuits({"c0": BELL_QASM})

        assert resource.last_request["simulator"] == {
            "backend": "qcsim",
            "method": "statevector",
            "options": {},
        }

    def test_names_the_backend_and_method_natively(self, resource):
        config = MaestroConfig(
            simulator_type="QiskitAer", simulation_type="MatrixProductState"
        )
        with QRMIBackend(resource, maestro_config=config) as backend:
            backend.submit_circuits({"c0": BELL_QASM})

        simulator = resource.last_request["simulator"]
        assert (simulator["backend"], simulator["method"]) == (
            "aer",
            "matrix_product_state",
        )

    def test_forwards_set_options_under_maestros_names(self, resource):
        config = MaestroConfig(max_bond_dimension=32, singular_value_threshold=1e-8)
        with QRMIBackend(resource, maestro_config=config) as backend:
            backend.submit_circuits({"c0": BELL_QASM})

        assert resource.last_request["simulator"]["options"] == {
            "max_bond_dimension": 32,
            "singular_value_threshold": 1e-8,
        }

    def test_derives_each_circuits_seed_as_maestro_simulator_does(self, resource):
        """The seed moves to execution.seed, where the server takes it once."""
        with QRMIBackend(resource, maestro_config=MaestroConfig(seed=5)) as backend:
            backend.submit_circuits({"a": BELL_QASM, "b": BELL_QASM})

        seeds = [r["execution"]["seed"] for r in resource.requests]
        assert seeds == [
            _circuit_seed(5, "a", BELL_QASM),
            _circuit_seed(5, "b", BELL_QASM),
        ]
        assert "seed" not in resource.last_request["simulator"]["options"]

    def test_set_seed_reaches_the_request(self, resource, backend):
        backend.set_seed(3)
        backend.submit_circuits({"a": BELL_QASM})

        assert resource.last_request["execution"]["seed"] == _circuit_seed(
            3, "a", BELL_QASM
        )

    def test_rejects_noise(self, resource):
        with pytest.raises(ValueError, match="noise_seed"):
            QRMIBackend(resource, maestro_config=MaestroConfig(noise_seed=1))

    def test_rejects_a_maestro_simulator_config(self, resource):
        with pytest.raises(TypeError, match="from_simulator_config"):
            QRMIBackend(resource, maestro_config=maestro.SimulatorConfig())


def test_rejects_a_qrmi_without_maestro_local(mocker):
    """Qoro's fork and upstream report the same version, so the resource type
    is the only way to tell them apart — and upstream would otherwise fail far
    later with a bare AttributeError on Payload.MaestroLocal."""
    upstream = SimpleNamespace(ResourceType=SimpleNamespace(IBMQuantumSystem=object()))
    mocker.patch.dict(sys.modules, {"qrmi": upstream})

    with pytest.raises(ImportError, match="upstream build"):
        _REAL_IMPORT_QRMI()


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
    """The server drops a session once its lease lapses between submissions."""

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
    def test_reports_the_servers_diagnostics_when_a_task_fails(self, resource):
        resource._statuses = [FakeTaskStatus.Failed]
        with QRMIBackend(resource, shots=100) as backend:
            with pytest.raises(RuntimeError, match="logs for task-0"):
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
