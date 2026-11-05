# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for divi.backends.runners._qrmi."""

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock

import maestro
import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp

import divi.backends.runners._qrmi as qrmi_module
from divi.backends import MaestroConfig, QRMIBackend
from divi.backends.runners._maestro import _circuit_seed
from divi.circuits._payloads import CircuitPayload
from divi.exceptions import ExecutionCancelledError, VJPUnavailableError
from tests.backends._circuit_runner_contracts import SyncRunnerContractsBase
from tests.backends._qrmi_fakes import (
    SESSION_NOT_FOUND,
    FakeConfigError,
    FakeInvalidInputError,
    FakeQuantumResource,
    FakeTaskStatus,
    FakeUnsupportedFunctionError,
    fake_qrmi_module,
)

BELL_QASM = (
    'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncreg c[2];\n'
    "h q[0];\ncx q[0],q[1];\nmeasure q[0] -> c[0];\nmeasure q[1] -> c[1];\n"
)

#: Captured before the autouse fixture patches it away.
_REAL_IMPORT_QRMI = qrmi_module._import_qrmi


@pytest.fixture(autouse=True)
def fake_qrmi(mocker):
    """Swap the qrmi import for a stand-in, so no wheel is needed."""
    mocker.patch.object(
        qrmi_module, "_import_qrmi", return_value=fake_qrmi_module(mocker)
    )


@pytest.fixture(autouse=True)
def clean_token_env(monkeypatch):
    """Keep the acquisition token out of the ambient environment."""
    monkeypatch.delenv("MAESTRO_LOCAL_QRMI_JOB_ACQUISITION_TOKEN", raising=False)


@pytest.fixture
def resource(mocker):
    return FakeQuantumResource(mocker)


@pytest.fixture
def backend(resource):
    with QRMIBackend(resource, shots=100, poll_interval=0) as runner:
        yield runner


def _fail_first_task_start(mocker, resource, error):
    """The first ``task_start`` raises ``error``; later ones go through."""
    start = resource.task_start

    def task_start(payload):
        if patched.call_count == 1:
            raise error
        return start(payload)

    patched = mocker.patch.object(resource, "task_start", side_effect=task_start)


def _cancel_on_task_start(mocker, resource, event):
    """Set ``event`` as each task is dispatched."""
    start = resource.task_start

    def task_start(payload):
        event.set()
        return start(payload)

    mocker.patch.object(resource, "task_start", side_effect=task_start)


class TestCapabilities:
    def test_rejects_a_qrmi_without_maestro_local(self, mocker):
        """Qoro's fork and upstream report the same version, so the resource type
        is the only way to tell them apart — and upstream would otherwise fail far
        later with a bare AttributeError on Payload.MaestroLocal."""
        upstream = mocker.MagicMock(spec=["ResourceType"])
        upstream.ResourceType = mocker.MagicMock(spec=["IBMQuantumSystem"])
        mocker.patch.dict(sys.modules, {"qrmi": upstream})

        with pytest.raises(ImportError, match="upstream build"):
            _REAL_IMPORT_QRMI()

    def test_reports_expval_support_and_synchronous_execution(self, backend):
        assert backend.supports_expval is True
        assert backend.is_async is False

    def test_never_resolves_parameters(self, backend):
        """Maestro's parser evaluates gate arguments at parse time, so a free
        identifier is a parse error — the pipeline must bind first."""
        assert backend.resolves_parameters is False

    def test_forced_sampling_turns_native_estimation_off(self, resource):
        backend = QRMIBackend(resource, force_sampling=True)

        assert backend.supports_expval is False
        assert backend.supports_vjp is False

    def test_defaults_to_maestro_simulators_shots(self, resource):
        assert QRMIBackend(resource).shots == 5000


class TestSampling:
    def test_reverses_bitstrings_to_qiskit_order(self, resource):
        """Maestro writes c[0] leftmost; divi keys results Qiskit-style with
        c[0] rightmost. Without the flip, every result is silently mirrored."""
        resource._counts = {"10": 512}
        with QRMIBackend(resource, shots=512) as backend:
            results = backend.submit_circuits({"c0": BELL_QASM}).results

        assert results[0]["label"] == "c0"
        assert results[0]["results"] == {"01": 512}

    def test_sizes_keys_and_the_request_to_the_classical_register(self, resource):
        """A register narrower than the qubits gets keys of its own width."""
        narrow = BELL_QASM.replace("creg c[2]", "creg c[1]").replace(
            "measure q[1] -> c[1];\n", ""
        )
        resource._counts = {"10": 512}
        with QRMIBackend(resource, shots=512) as backend:
            results = backend.submit_circuits({"c0": narrow}).results

        assert resource.last_request["circuit"]["num_clbits"] == 1
        assert results[0]["results"] == {"1": 512}

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
        assert request["circuit"] == {
            "source": BELL_QASM,
            "num_qubits": 2,
            "num_clbits": 2,
        }
        assert request["execution"] == {"shots": 4096}

    def test_sends_a_qiskit_circuit_as_openqasm(self, resource, backend):
        circuit = QuantumCircuit(2, 2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.measure([0, 1], [0, 1])

        backend.submit_circuits({"c0": circuit})

        sent = resource.last_request["circuit"]
        assert QuantumCircuit.from_qasm_str(sent["source"]) == circuit
        assert (sent["num_qubits"], sent["num_clbits"]) == (2, 2)

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
    def test_expectation_value_weights_each_terms_estimate(self, resource, backend):
        resource._expectation_values = [0.25, -0.5]

        value = backend.expectation_value(
            BELL_QASM, SparsePauliOp(["ZZ", "XX"], coeffs=[2.0, 1.0])
        )

        assert value == pytest.approx(0.0)
        assert resource.last_request["observables"] == ["ZZ", "XX"]

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

    def test_rejects_a_maestro_simulator_config(self, resource):
        with pytest.raises(TypeError, match="from_simulator_config"):
            QRMIBackend(resource, maestro_config=maestro.SimulatorConfig())


def _noisy_backend(resource, noise_model, **config):
    return QRMIBackend(
        resource, maestro_config=MaestroConfig(noise_model=noise_model, **config)
    )


def _depolarizing_and_readout():
    noise_model = maestro.NoiseModel()
    noise_model.set_all_depolarizing(2, 0.01)
    noise_model.set_readout_error(0, 0.02, 0.05)
    return noise_model


class TestNoise:
    def test_sends_the_models_channels(self, resource):
        with _noisy_backend(resource, _depolarizing_and_readout()) as backend:
            backend.submit_circuits({"c0": BELL_QASM})

        assert resource.last_request["noise"]["channels"] == [
            {"kind": "depolarizing", "targets": [0], "probability": 0.01},
            {"kind": "depolarizing", "targets": [1], "probability": 0.01},
            {
                "kind": "readout",
                "targets": [0],
                "p_meas1_prep0": 0.02,
                "p_meas0_prep1": 0.05,
            },
        ]

    @pytest.mark.parametrize(
        "ham_ops, realizations", [(None, 64), ("ZZ", 100)], ids=["execute", "estimate"]
    )
    def test_uses_the_realizations_a_local_run_defaults_to(
        self, resource, ham_ops, realizations
    ):
        noise_model = maestro.NoiseModel()
        noise_model.set_all_depolarizing(2, 0.01)
        with _noisy_backend(resource, noise_model) as backend:
            backend.submit_circuits({"c0": BELL_QASM}, ham_ops=ham_ops)

        assert resource.last_request["noise"]["realizations"] == realizations

    def test_one_realization_where_every_channel_is_exact(self, resource):
        noise_model = maestro.NoiseModel()
        noise_model.set_all_depolarizing(2, 0.01)
        with _noisy_backend(
            resource, noise_model, simulation_type="DensityMatrix"
        ) as backend:
            backend.submit_circuits({"c0": BELL_QASM})

        assert resource.last_request["noise"]["realizations"] == 1

    def test_derives_each_circuits_noise_seed_as_maestro_simulator_does(self, resource):
        with _noisy_backend(
            resource, _depolarizing_and_readout(), noise_seed=9
        ) as backend:
            backend.submit_circuits({"a": BELL_QASM})

        assert resource.last_request["noise"]["seed"] == _circuit_seed(
            9, "a", BELL_QASM
        )

    def test_an_estimate_leaves_readout_out_and_warns(self, resource):
        with _noisy_backend(resource, _depolarizing_and_readout()) as backend:
            with pytest.warns(UserWarning, match="Readout errors do not affect"):
                backend.submit_circuits({"c0": BELL_QASM}, ham_ops="ZZ")

        kinds = [c["kind"] for c in resource.last_request["noise"]["channels"]]
        assert kinds == ["depolarizing", "depolarizing"]

    def test_a_model_left_empty_by_the_circuit_runs_noiseless(self, resource):
        noise_model = maestro.NoiseModel()
        noise_model.set_readout_error(0, 0.02, 0.05)
        with _noisy_backend(resource, noise_model) as backend:
            with pytest.warns(UserWarning, match="Readout errors do not affect"):
                backend.submit_circuits({"c0": BELL_QASM}, ham_ops="ZZ")

        assert "noise" not in resource.last_request

    def test_identity_gates_carry_noise(self, resource):
        with _noisy_backend(resource, _depolarizing_and_readout()) as backend:
            backend.submit_circuits({"c0": BELL_QASM.replace("h q[0];", "id q[0];")})

        assert "u3(0,0,0) q[0];" in resource.last_request["circuit"]["source"]

    def test_noise_rules_out_vjps(self, resource):
        with _noisy_backend(resource, _depolarizing_and_readout()) as backend:
            assert backend.supports_vjp is False
            with pytest.raises(VJPUnavailableError, match="noiseless"):
                backend.submit_vjp(
                    [_vjp_payload()], ham_ops="ZZ", cotangents=_VJP_COTANGENTS
                )


class TestStatevector:
    def test_sends_the_lowered_circuit_and_applies_its_global_phase(
        self, resource, backend
    ):
        circuit = QuantumCircuit(2)
        circuit.global_phase = np.pi / 2

        amplitudes = backend.statevector(circuit)

        request = resource.last_request
        assert request["operation"] == "statevector"
        assert request["circuit"]["num_qubits"] == 2
        np.testing.assert_allclose(amplitudes, [0, 1j, 2j, 3j], atol=1e-12)

    def test_keeps_only_the_active_qubits_amplitudes_without_idle_qubits(
        self, resource, backend
    ):
        """Qubit 1 is idle, so only indices with bit 1 clear remain, in order."""
        circuit = QuantumCircuit(3)
        circuit.h(0)
        circuit.cx(0, 2)

        amplitudes = backend.statevector(circuit, include_idle_qubits=False)

        np.testing.assert_allclose(amplitudes, [0, 1, 4, 5])

    def test_pads_idle_qubits_before_any_measurement(self, resource, backend):
        """The server refuses a statevector whose measurement is not terminal."""
        circuit = QuantumCircuit(2, 1)
        circuit.h(0)
        circuit.measure(0, 0)

        backend.statevector(circuit)

        source = resource.last_request["circuit"]["source"]
        assert source.index("rz(0) q[1];") < source.index("measure")

    def test_refuses_a_noise_model(self, resource):
        backend = _noisy_backend(resource, _depolarizing_and_readout())

        with pytest.raises(ValueError, match="cannot apply the configured noise"):
            backend.statevector(BELL_QASM)


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

    def test_leaves_the_environment_alone(self, resource):
        """The resource keeps the session it acquired, so nothing is exported."""
        with QRMIBackend(resource, shots=100) as backend:
            backend.submit_circuits({"a": BELL_QASM})
            assert "MAESTRO_LOCAL_QRMI_JOB_ACQUISITION_TOKEN" not in os.environ


class TestSessionRenewal:
    """The server drops a session once its lease lapses between submissions."""

    def test_reacquires_and_retries(self, resource, backend, mocker):
        _fail_first_task_start(mocker, resource, FakeConfigError(SESSION_NOT_FOUND))

        results = backend.submit_circuits({"c0": BELL_QASM}).results

        assert len(resource.acquired) == 2
        assert results[0]["label"] == "c0"

    def test_does_not_release_the_session_the_server_dropped(
        self, resource, backend, mocker
    ):
        _fail_first_task_start(mocker, resource, FakeConfigError(SESSION_NOT_FOUND))

        backend.submit_circuits({"c0": BELL_QASM})

        assert resource.released == []

    def test_releases_only_the_session_it_acquired_after_a_scheduler_one_expires(
        self, resource, monkeypatch, mocker
    ):
        monkeypatch.setenv("MAESTRO_LOCAL_QRMI_JOB_ACQUISITION_TOKEN", "scheduler")
        _fail_first_task_start(mocker, resource, FakeConfigError(SESSION_NOT_FOUND))

        with QRMIBackend(resource, shots=100) as backend:
            backend.submit_circuits({"c0": BELL_QASM})

        assert resource.released == resource.acquired == ["1"]

    def test_a_configuration_error_is_not_mistaken_for_a_lost_session(
        self, resource, backend, mocker
    ):
        """QRMI raises ConfigError for a bad request too; retrying it on a new
        session would hide the error and could adopt a borrowed session."""
        _fail_first_task_start(
            mocker,
            resource,
            FakeConfigError("invalid configuration: invalid_config: no profile"),
        )

        with pytest.raises(FakeConfigError, match="invalid_config"):
            backend.submit_circuits({"c0": BELL_QASM})

        assert len(resource.acquired) == 1

    def test_a_submission_after_close_holds_its_own_session(self, resource, backend):
        backend.submit_circuits({"a": BELL_QASM})
        backend.close()
        backend.submit_circuits({"b": BELL_QASM})
        backend.close()

        assert resource.released == resource.acquired == ["1", "2"]

    def test_a_failure_of_another_kind_is_not_retried(self, resource, backend, mocker):
        """Retrying anything but a dead session would mask the real error."""
        _fail_first_task_start(mocker, resource, RuntimeError("session limit reached"))

        with pytest.raises(RuntimeError, match="session limit reached"):
            backend.submit_circuits({"c0": BELL_QASM})

        assert len(resource.acquired) == 1

    def test_gives_up_when_a_fresh_session_is_refused_too(
        self, resource, backend, mocker
    ):
        mocker.patch.object(
            resource, "task_start", side_effect=FakeConfigError(SESSION_NOT_FOUND)
        )

        with pytest.raises(FakeConfigError):
            backend.submit_circuits({"c0": BELL_QASM})

        assert len(resource.acquired) == 2


class TestFailureModes:
    def test_reports_the_servers_diagnostics_when_a_task_fails(self, resource):
        resource._statuses = [FakeTaskStatus.Failed]
        with QRMIBackend(resource, shots=100) as backend:
            with pytest.raises(RuntimeError, match="backend_unavailable: no such"):
                backend.submit_circuits({"c0": BELL_QASM})

    def test_polls_until_the_task_completes(self, resource, backend, mocker):
        """Only Completed ends the wait. Treating Queued as terminal would
        fetch a result the server has not written yet."""
        resource._statuses = [
            FakeTaskStatus.Queued,
            FakeTaskStatus.Running,
            FakeTaskStatus.Completed,
        ]
        polls = mocker.spy(resource, "task_status")

        backend.submit_circuits({"c0": BELL_QASM})

        assert polls.call_count == 3

    def test_rejects_an_empty_batch(self, backend):
        with pytest.raises(ValueError, match="at least one circuit"):
            backend.submit_circuits({})


class TestCancellation:
    # The pre-dispatch abort itself is covered by the shared contract suite;
    # only the paths past it are exercised here.

    def test_does_not_dispatch_after_a_cancel_arrives(self, resource, backend):
        """A cancel landing after submit_circuits' own check, while the request
        waits for the resource, must still stop it before dispatch."""
        event = Event()
        build_payload = backend._qrmi.maestro.request_payload.side_effect

        def cancel_then_build(request):
            event.set()
            return build_payload(request)

        backend._qrmi.maestro.request_payload.side_effect = cancel_then_build

        with pytest.raises(ExecutionCancelledError, match="before dispatch"):
            backend.submit_circuits({"a": BELL_QASM}, cancellation_event=event)

        assert resource.payloads == []

    def test_a_failing_stop_still_reports_cancellation(self, resource, backend, mocker):
        """task_stop raising would surface a user's cancel as a batch failure,
        which the coordinator reports as Failed rather than Cancelled."""
        event = Event()
        _cancel_on_task_start(mocker, resource, event)
        stop = mocker.patch.object(
            resource, "task_stop", side_effect=RuntimeError("task already gone")
        )

        with pytest.raises(ExecutionCancelledError):
            backend.submit_circuits({"a": BELL_QASM}, cancellation_event=event)

        stop.assert_called_once_with("task-0")

    def test_an_error_while_waiting_stops_the_task(self, resource, backend, mocker):
        """Otherwise the task runs on and blocks the session's next one."""
        mocker.patch.object(
            resource, "task_status", side_effect=RuntimeError("socket closed")
        )

        with pytest.raises(RuntimeError, match="socket closed"):
            backend.submit_circuits({"a": BELL_QASM})

        assert resource.stopped == ["task-0"]

    def test_stops_a_task_cancelled_while_it_runs(self, resource, backend, mocker):
        """Polling is interruptible, so a cancel mid-task must tell the server
        to drop it rather than wait the circuit out."""
        event = Event()
        _cancel_on_task_start(mocker, resource, event)

        with pytest.raises(ExecutionCancelledError, match="while task"):
            backend.submit_circuits({"a": BELL_QASM}, cancellation_event=event)

        assert resource.stopped == ["task-0"]

    def test_aborts_between_circuits(self, resource, backend, mocker):
        event = Event()
        result = resource.task_result

        def cancel_after_first(task_id):
            event.set()
            return result(task_id)

        mocker.patch.object(resource, "task_result", side_effect=cancel_after_first)

        with pytest.raises(ExecutionCancelledError, match="partial completion"):
            backend.submit_circuits(
                {"a": BELL_QASM, "b": BELL_QASM}, cancellation_event=event
            )

        assert len(resource.payloads) == 1


_VJP_ROWS = (("row_0", (0.3, -0.7)), ("row_1", (-1.2, 0.4)))
_VJP_COTANGENTS = {"row_0": np.array([1.0, 0.5]), "row_1": np.array([-0.3, 2.0])}


def _vjp_payload(body="ry(w_0) q[0];\nrz(w_1) q[1];\n"):
    return CircuitPayload(
        circuit=(
            'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncreg c[2];\n'
            + body
            + "measure q[0] -> c[0];\nmeasure q[1] -> c[1];\n"
        ),
        parameters=(Parameter("w_0"), Parameter("w_1")),
        parameter_sets=_VJP_ROWS,
    )


class TestVectorJacobianProducts:
    def _submit(self, backend):
        return backend.submit_vjp(
            [_vjp_payload()], ham_ops="ZZ;XI", cotangents=_VJP_COTANGENTS
        )

    def test_is_supported(self, backend):
        assert backend.supports_vjp is True

    def test_sends_one_adjoint_vjp_task_for_all_rows_of_a_payload(
        self, resource, backend
    ):
        self._submit(backend)

        (request,) = resource.requests
        assert request["operation"] == "adjoint_vjp"
        assert request["observables"] == ["ZZ", "XI"]
        assert request["parameter_rows"] == [[0.3, -0.7], [-1.2, 0.4]]
        assert request["cotangents"] == [[1.0, 0.5], [-0.3, 2.0]]
        assert request["simulator"]["backend"] == "qcsim"
        assert "execution" not in request

    def test_returns_each_rows_values_and_gradient(self, backend):
        results = self._submit(backend).results

        assert [entry["label"] for entry in results] == ["row_0", "row_1"]
        assert results[0]["results"] == {"ZZ": 0.5, "XI": 0.5}
        np.testing.assert_allclose(results[0]["metadata"]["vjp"], [1.0, 1.0])

    def test_maps_gradients_back_through_scaled_angles(self, resource, backend):
        """The server differentiates gate angles; ry(2*w_0) doubles w_0's share."""
        results = backend.submit_vjp(
            [_vjp_payload("ry(2*w_0) q[0];\nrz(w_1) q[1];\n")],
            ham_ops="ZZ;XI",
            cotangents=_VJP_COTANGENTS,
        ).results

        np.testing.assert_allclose(
            resource.last_request["parameter_rows"], [[0.6, -0.7], [-2.4, 0.4]]
        )
        np.testing.assert_allclose(results[0]["metadata"]["vjp"], [2.0, 1.0])

    def test_forced_sampling_rules_it_out(self, resource):
        backend = QRMIBackend(resource, force_sampling=True)

        with pytest.raises(VJPUnavailableError, match="native, noiseless"):
            self._submit(backend)

    def test_a_server_refusal_makes_it_unavailable(self, resource, backend, mocker):
        mocker.patch.object(
            resource,
            "task_start",
            side_effect=FakeUnsupportedFunctionError("unsupported_capability"),
        )

        with pytest.raises(VJPUnavailableError, match="cannot differentiate"):
            self._submit(backend)

    def test_running_out_of_memory_makes_it_unavailable(self, resource, backend):
        resource._statuses = [FakeTaskStatus.Failed]
        resource._failure = {
            "error": {"code": "insufficient_resources", "message": "state too large"}
        }

        with pytest.raises(VJPUnavailableError, match="insufficient_resources"):
            self._submit(backend)

    def test_a_rejected_request_is_not_hidden_as_unavailable(
        self, resource, backend, mocker
    ):
        """A malformed request is a bug to surface, not a reason to fall back."""
        mocker.patch.object(
            resource,
            "task_start",
            side_effect=FakeInvalidInputError("Unknown field: parameter_rows"),
        )

        with pytest.raises(FakeInvalidInputError):
            self._submit(backend)

    def test_a_cancel_arriving_first_dispatches_nothing(self, resource, backend):
        event = Event()
        event.set()

        with pytest.raises(ExecutionCancelledError):
            backend.submit_vjp(
                [_vjp_payload()],
                ham_ops="ZZ;XI",
                cotangents=_VJP_COTANGENTS,
                cancellation_event=event,
            )

        assert resource.payloads == []


class TestConcurrency:
    def test_concurrent_submissions_never_interleave(self, resource, backend):
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
                    overlapped = overlapped or depth > 1
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
                pool.submit(backend.submit_circuits, {f"c{i}": BELL_QASM})
                for i in range(4)
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
