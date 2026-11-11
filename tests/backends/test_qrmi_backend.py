# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import gc
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock

import maestro
import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.quantum_info import SparsePauliOp

import divi.backends.runners._qrmi as qrmi_module
from divi.backends import DeviceConfig, MaestroConfig, QRMITaskFailedError
from divi.backends.runners._maestro import _circuit_seed
from divi.exceptions import ExecutionCancelledError, VJPUnavailableError
from tests._helpers import exact_match
from tests.backends._circuit_runner_contracts import (
    CONTRACT_TEST_SHOTS,
    QASM_DEPTH_3,
    SyncRunnerContractsBase,
)
from tests.backends._helpers import (
    SHOT_GROUPS_WITH_HAM_OPS_MESSAGE,
    VJP_COTANGENTS,
    make_vjp_payload,
    padding_warning,
)
from tests.backends._qrmi_fakes import (
    SESSION_NOT_FOUND,
    FakeConfigError,
    FakeIBMResource,
    FakeInvalidInputError,
    FakeQuantumResource,
    FakeResourceType,
    FakeTaskStatus,
    FakeUnsupportedFunctionError,
    FakeUpstreamResourceType,
    fake_qrmi_module,
)

_BELL_QASM = QASM_DEPTH_3

# Long enough for a second caller to arrive, were nothing holding it back.
_OVERLAP_WAIT = 0.2

# Captured before the autouse fixture patches it away.
_REAL_IMPORT_QRMI = qrmi_module._import_qrmi

_IBM = FakeResourceType.IBMQuantumSystem


def _depolarizing(*, readout: bool = False):
    noise_model = maestro.NoiseModel()
    noise_model.set_all_depolarizing(2, 0.01)
    if readout:
        noise_model.set_readout_error(0, 0.02, 0.05)
    return noise_model


def _maestro_only(operation: str, resource_type=_IBM) -> str:
    return (
        f"{operation} needs a maestro-local resource; this resource is {resource_type}."
    )


class _OverlapProbe:
    """Records whether calls through :meth:`wrap` ever run at once.

    The first caller waits for a second, so a missing lock always overlaps.
    """

    def __init__(self):
        self._guard = Lock()
        self._inside = 0
        self._entered = False
        self._second = Event()
        self.overlapped = False

    def wrap(self, method):
        def wrapper(*args, **kwargs):
            with self._guard:
                self._inside += 1
                self.overlapped |= self._inside > 1
                first, self._entered = not self._entered, True
            if self.overlapped:
                self._second.set()
            try:
                if first:
                    self._second.wait(_OVERLAP_WAIT)
                return method(*args, **kwargs)
            finally:
                with self._guard:
                    self._inside -= 1

        return wrapper


def _submit_concurrently(backend, n: int = 4) -> None:
    with ThreadPoolExecutor(max_workers=n) as pool:
        futures = [
            pool.submit(backend.submit_circuits, {f"c{i}": _BELL_QASM})
            for i in range(n)
        ]
        for future in futures:
            future.result()


@pytest.fixture(autouse=True)
def fake_qrmi(mocker):
    """Swap the qrmi import for a stand-in, so no wheel is needed."""
    module = fake_qrmi_module(mocker)
    mocker.patch.object(qrmi_module, "_import_qrmi", return_value=module)
    return module


@pytest.fixture(autouse=True)
def clean_token_env(monkeypatch):
    """Keep the fakes' acquisition tokens out of the ambient environment."""
    for fake in (FakeQuantumResource, FakeIBMResource):
        monkeypatch.delenv(
            f"{fake.resource_id_value}_QRMI_JOB_ACQUISITION_TOKEN", raising=False
        )


@pytest.fixture
def resource(mocker):
    return FakeQuantumResource(mocker)


@pytest.fixture
def backend(make_qrmi_backend, resource):
    return make_qrmi_backend(resource, shots=100)


@pytest.fixture
def ibm_runtime():
    """Qiskit Runtime, whose job encoding the IBM path uses."""
    return pytest.importorskip("qiskit_ibm_runtime")


@pytest.fixture
def ibm_backend_factory(mocker, ibm_runtime):
    """qrmi's IBM backend factory, describing a generic five-qubit device."""
    decoder = pytest.importorskip(
        "qiskit_ibm_runtime.decoders.result_decoder"
    ).ResultDecoder
    factory = mocker.Mock(return_value=GenericBackendV2(num_qubits=5, seed=0))
    mocker.patch.object(
        qrmi_module,
        "_import_ibm_runtime",
        return_value=(factory, ibm_runtime.RuntimeEncoder, decoder),
    )
    return factory


@pytest.fixture
def ibm_resource(request, mocker, ibm_runtime, ibm_backend_factory):
    """An IBM resource, of the type an indirect parametrisation names."""
    return FakeIBMResource(
        mocker,
        ibm_runtime.RuntimeEncoder,
        resource_type=getattr(request, "param", _IBM),
    )


@pytest.fixture
def ibm_backend(make_qrmi_backend, ibm_resource):
    return make_qrmi_backend(ibm_resource, shots=100)


class TestConstruction:
    def test_rejects_a_resource_of_another_vendor(self, mocker, make_qrmi_backend):
        resource = FakeQuantumResource(
            mocker, resource_type=FakeResourceType.PasqalLocal
        )
        message = (
            "QRMIBackend runs maestro-local and IBM resources; got "
            f"{FakeResourceType.PasqalLocal}. maestro-local needs Qoro's fork of "
            "qrmi (https://github.com/QoroQuantum/qrmi)."
        )

        with pytest.raises(ValueError, match=exact_match(message)):
            make_qrmi_backend(resource)

    def test_imports_upstream_qrmi_without_maestro_helpers(self, mocker):
        """Upstream qrmi has no maestro-local type or ``qrmi.maestro`` module;
        importing that module would fail."""
        upstream = mocker.NonCallableMock(
            spec_set=["ResourceType"], ResourceType=FakeUpstreamResourceType
        )
        mocker.patch.dict(sys.modules, {"qrmi": upstream})

        assert _REAL_IMPORT_QRMI() is upstream

    def test_defaults_to_maestro_simulators_shots(self, resource, make_qrmi_backend):
        assert make_qrmi_backend(resource).shots == 5000


class TestProperties:
    def test_estimates_observables_natively(self, backend):
        assert backend.supports_expval is True

    def test_never_resolves_parameters(self, backend):
        """Maestro's parser evaluates gate arguments at parse time, so a free
        identifier is a parse error — the pipeline must bind first."""
        assert backend.resolves_parameters is False

    def test_forced_sampling_turns_native_estimation_off(
        self, resource, make_qrmi_backend
    ):
        backend = make_qrmi_backend(resource, force_sampling=True)

        assert backend.supports_expval is False
        assert backend.supports_vjp is False


class TestSampling:
    def test_keeps_maestros_bit_order(self, resource, backend):
        """Maestro writes c[0] leftmost, as divi keys results."""
        resource.counts = {"10": 512}

        (entry,) = backend.submit_circuits({"c0": _BELL_QASM}).results

        assert (entry["label"], entry["results"]) == ("c0", {"10": 512})

    def test_sizes_keys_and_the_request_to_the_classical_register(
        self, resource, backend
    ):
        """A register narrower than the qubits gets keys of its own width."""
        narrow = _BELL_QASM.replace("creg c[2]", "creg c[1]").replace(
            "measure q[1] -> c[1];\n", ""
        )
        resource.counts = {"10": 512}

        (entry,) = backend.submit_circuits({"c0": narrow}).results

        assert resource.last_request["circuit"]["num_clbits"] == 1
        assert entry["results"] == {"1": 512}

    def test_returns_the_servers_other_outputs_as_metadata(self, backend):
        result = backend.submit_circuits({"c0": _BELL_QASM})

        assert result.results[0]["metadata"]["seed"] == 7
        assert result.run_time == pytest.approx(0.001)

    def test_sends_a_native_execute_request(self, resource, make_qrmi_backend):
        make_qrmi_backend(resource, shots=4096).submit_circuits({"c0": _BELL_QASM})

        request = resource.last_request
        assert resource.payloads[-1].job_type == "request"
        assert request["schema_version"] == 2
        assert request["operation"] == "execute"
        assert request["circuit"] == {
            "source": _BELL_QASM,
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
            {"a": _BELL_QASM, "b": _BELL_QASM, "c": _BELL_QASM}
        ).results

        assert [row["label"] for row in results] == ["a", "b", "c"]

    def test_submits_one_task_per_circuit(self, resource, backend):
        backend.submit_circuits({"a": _BELL_QASM, "b": _BELL_QASM})

        assert len(resource.payloads) == 2

    def test_shot_groups_allocate_shots_per_circuit(self, resource, backend):
        backend.submit_circuits(
            {"a": _BELL_QASM, "b": _BELL_QASM}, shot_groups=[[0, 1, 10], [1, 2, 20]]
        )

        assert [r["execution"]["shots"] for r in resource.requests] == [10, 20]

    def test_rejects_an_empty_batch(self, backend):
        with pytest.raises(
            ValueError,
            match=exact_match("submit_circuits requires at least one payload."),
        ):
            backend.submit_circuits({})


class TestExpectationValues:
    def test_weights_each_terms_estimate(self, resource, backend):
        resource.expectation_values = [0.25, -0.5]

        value = backend.expectation_value(
            _BELL_QASM, SparsePauliOp(["ZZ", "XX"], coeffs=[2.0, 1.0])
        )

        assert value == pytest.approx(0.0)
        assert resource.last_request["observables"] == ["ZZ", "XX"]

    def test_sends_terms_qubit_zero_first(self, resource, backend):
        """Qiskit labels read right to left; the request reads qubit 0 first,
        so Qiskit's "ZI" (Z on qubit 1) is sent as "IZ"."""
        resource.expectation_values = [-1.0, 1.0]

        value = backend.expectation_value(
            _BELL_QASM, SparsePauliOp(["ZI", "IZ"], coeffs=[1.0, 0.5])
        )

        assert resource.last_request["observables"] == ["IZ", "ZI"]
        assert value == pytest.approx(-0.5)

    def test_rejects_a_non_hermitian_observable(self, resource, backend):
        with pytest.raises(ValueError, match="Hermitian"):
            backend.expectation_value(_BELL_QASM, SparsePauliOp("ZZ", coeffs=[1j]))
        assert resource.payloads == []

    def test_keys_results_by_pauli_term(self, resource, backend):
        resource.expectation_values = [0.25, -0.5]

        (entry,) = backend.submit_circuits({"c0": _BELL_QASM}, ham_ops="ZZ;XX").results

        assert entry["results"] == {"ZZ": 0.25, "XX": -0.5}

    def test_sends_an_estimate_request_without_shots(self, resource, backend):
        """The native estimator rejects execution.shots."""
        backend.submit_circuits({"c0": _BELL_QASM}, ham_ops="ZZ;XX")

        request = resource.last_request
        assert request["operation"] == "estimate"
        assert request["observables"] == ["ZZ", "XX"]
        assert "shots" not in request["execution"]

    def test_pads_short_terms_onto_the_first_qubits(self, resource, backend):
        wide = _BELL_QASM.replace("qreg q[2]", "qreg q[3]")

        with pytest.warns(UserWarning, match=exact_match(padding_warning("ZZ", "ZZI"))):
            backend.submit_circuits({"c0": wide}, ham_ops="ZZ")

        assert resource.last_request["observables"] == ["ZZI"]

    def test_assigns_groups_per_circuit(self, resource, backend):
        backend.submit_circuits(
            {"a": _BELL_QASM, "b": _BELL_QASM},
            ham_ops="ZZ|XX",
            circuit_ham_map=[[0, 1], [1, 2]],
        )

        assert [r["observables"] for r in resource.requests] == [["ZZ"], ["XX"]]

    def test_flattens_group_separators_for_unmapped_circuits(self, resource, backend):
        """A circuit outside every circuit_ham_map range gets every group's
        terms, each its own observable."""
        resource.expectation_values = [0.1, 0.2, 0.3]

        backend.submit_circuits(
            {"aux": _BELL_QASM}, ham_ops="ZZ|XX;YY", circuit_ham_map=[[1, 2]]
        )

        assert resource.last_request["observables"] == ["ZZ", "XX", "YY"]

    def test_rejects_shot_groups(self, backend):
        with pytest.raises(
            ValueError, match=exact_match(SHOT_GROUPS_WITH_HAM_OPS_MESSAGE)
        ):
            backend.submit_circuits(
                {"c0": _BELL_QASM}, ham_ops="ZZ", shot_groups=[[0, 1, 10]]
            )


class TestMaestroConfig:
    def test_defaults_to_maestros_defaults(self, resource, backend):
        backend.submit_circuits({"c0": _BELL_QASM})

        assert resource.last_request["simulator"] == {
            "backend": "qcsim",
            "method": "statevector",
            "options": {},
        }

    def test_names_the_backend_and_method_natively(self, resource, make_qrmi_backend):
        config = MaestroConfig(
            simulator_type="QiskitAer", simulation_type="MatrixProductState"
        )

        make_qrmi_backend(resource, maestro_config=config).submit_circuits(
            {"c0": _BELL_QASM}
        )

        simulator = resource.last_request["simulator"]
        assert (simulator["backend"], simulator["method"]) == (
            "aer",
            "matrix_product_state",
        )

    def test_forwards_set_options_under_maestros_names(
        self, resource, make_qrmi_backend
    ):
        config = MaestroConfig(max_bond_dimension=32, singular_value_threshold=1e-8)

        make_qrmi_backend(resource, maestro_config=config).submit_circuits(
            {"c0": _BELL_QASM}
        )

        assert resource.last_request["simulator"]["options"] == {
            "max_bond_dimension": 32,
            "singular_value_threshold": 1e-8,
        }

    def test_derives_each_circuits_seed_as_maestro_simulator_does(
        self, resource, make_qrmi_backend
    ):
        """The seed moves to execution.seed, where the server takes it once."""
        backend = make_qrmi_backend(resource, maestro_config=MaestroConfig(seed=5))

        backend.submit_circuits({"a": _BELL_QASM, "b": _BELL_QASM})

        seeds = [r["execution"]["seed"] for r in resource.requests]
        assert seeds == [
            _circuit_seed(5, "a", _BELL_QASM),
            _circuit_seed(5, "b", _BELL_QASM),
        ]
        assert "seed" not in resource.last_request["simulator"]["options"]

    def test_set_seed_reaches_the_request(self, resource, backend):
        backend.set_seed(3)
        backend.submit_circuits({"a": _BELL_QASM})

        assert resource.last_request["execution"]["seed"] == _circuit_seed(
            3, "a", _BELL_QASM
        )

    def test_rejects_a_maestro_simulator_config_at_construction(
        self, resource, make_qrmi_backend
    ):
        with pytest.raises(TypeError, match="from_simulator_config"):
            make_qrmi_backend(resource, maestro_config=maestro.SimulatorConfig())

    def test_rejects_a_maestro_simulator_config_on_assignment(self, backend):
        with pytest.raises(TypeError, match="from_simulator_config"):
            backend.maestro_config = maestro.SimulatorConfig()

    def test_takes_no_device_config_at_construction(self, resource, make_qrmi_backend):
        message = (
            "device_config applies to IBM resources; this resource is "
            f"{FakeResourceType.MaestroLocal}. Pass maestro_config instead."
        )

        with pytest.raises(ValueError, match=exact_match(message)):
            make_qrmi_backend(resource, device_config=DeviceConfig())

    def test_takes_no_device_config_on_assignment(self, backend):
        with pytest.raises(ValueError, match="device_config applies to IBM"):
            backend.device_config = DeviceConfig()
        assert backend.device_config is None


class TestNoise:
    @pytest.fixture
    def make_noisy_backend(self, resource, make_qrmi_backend):
        def _make(noise_model, **config):
            return make_qrmi_backend(
                resource,
                maestro_config=MaestroConfig(noise_model=noise_model, **config),
            )

        return _make

    def test_sends_the_models_channels(self, resource, make_noisy_backend):
        make_noisy_backend(_depolarizing(readout=True)).submit_circuits(
            {"c0": _BELL_QASM}
        )

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

    @pytest.mark.parametrize("ham_ops", [None, "ZZ"], ids=["execute", "estimate"])
    def test_unset_realizations_are_left_to_maestro(
        self, resource, make_noisy_backend, ham_ops
    ):
        make_noisy_backend(_depolarizing()).submit_circuits(
            {"c0": _BELL_QASM}, ham_ops=ham_ops
        )

        assert "realizations" not in resource.last_request["noise"]

    def test_forwards_explicit_realizations(self, resource, make_noisy_backend):
        make_noisy_backend(_depolarizing(), noise_realizations=7).submit_circuits(
            {"c0": _BELL_QASM}
        )

        assert resource.last_request["noise"]["realizations"] == 7

    def test_derives_each_circuits_noise_seed_as_maestro_simulator_does(
        self, resource, make_noisy_backend
    ):
        make_noisy_backend(_depolarizing(readout=True), noise_seed=9).submit_circuits(
            {"a": _BELL_QASM}
        )

        assert resource.last_request["noise"]["seed"] == _circuit_seed(
            9, "a", _BELL_QASM
        )

    def test_an_estimate_leaves_readout_out_and_warns(
        self, resource, make_noisy_backend
    ):
        backend = make_noisy_backend(_depolarizing(readout=True))

        with pytest.warns(UserWarning, match="Readout errors do not affect"):
            backend.submit_circuits({"c0": _BELL_QASM}, ham_ops="ZZ")

        kinds = [c["kind"] for c in resource.last_request["noise"]["channels"]]
        assert kinds == ["depolarizing", "depolarizing"]

    def test_the_readout_warning_points_at_the_callers_line(self, make_noisy_backend):
        backend = make_noisy_backend(_depolarizing(readout=True))

        with pytest.warns(UserWarning, match="Readout errors do not affect") as caught:
            backend.submit_circuits({"c0": _BELL_QASM}, ham_ops="ZZ")

        assert caught[0].filename == __file__

    def test_a_model_left_empty_by_the_circuit_runs_noiseless(
        self, resource, make_noisy_backend
    ):
        noise_model = maestro.NoiseModel()
        noise_model.set_readout_error(0, 0.02, 0.05)
        backend = make_noisy_backend(noise_model)

        with pytest.warns(UserWarning, match="Readout errors do not affect"):
            backend.submit_circuits({"c0": _BELL_QASM}, ham_ops="ZZ")

        assert "noise" not in resource.last_request

    def test_identity_gates_carry_noise(self, resource, make_noisy_backend):
        make_noisy_backend(_depolarizing(readout=True)).submit_circuits(
            {"c0": _BELL_QASM.replace("h q[0];", "id q[0];")}
        )

        assert "u3(0,0,0) q[0];" in resource.last_request["circuit"]["source"]

    def test_noise_rules_out_vjps(self, make_noisy_backend):
        backend = make_noisy_backend(_depolarizing(readout=True))

        assert backend.supports_vjp is False
        with pytest.raises(VJPUnavailableError, match="noiseless"):
            backend.submit_vjp(
                [make_vjp_payload()], ham_ops="ZZ", cotangents=VJP_COTANGENTS
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

    def test_refuses_a_noise_model(self, resource, make_qrmi_backend):
        backend = make_qrmi_backend(
            resource, maestro_config=MaestroConfig(noise_model=_depolarizing())
        )
        message = (
            "statevector cannot apply the configured noise model; use "
            "expectation_value, or a maestro_config without noise_model."
        )

        with pytest.raises(ValueError, match=exact_match(message)):
            backend.statevector(_BELL_QASM)


class TestSession:
    def test_acquires_once_across_submissions(self, resource, backend):
        backend.submit_circuits({"a": _BELL_QASM})
        backend.submit_circuits({"b": _BELL_QASM})

        assert len(resource.acquired) == 1

    def test_releases_on_close(self, resource, backend):
        backend.submit_circuits({"a": _BELL_QASM})
        backend.close()

        assert resource.released == resource.acquired

    def test_close_tolerates_a_session_the_server_already_dropped(
        self, resource, backend, caplog
    ):
        backend.submit_circuits({"a": _BELL_QASM})
        resource.release_error = RuntimeError("session_not_found")

        backend.close()

        assert "Could not release QRMI session 1: session_not_found" in caplog.text

    def test_close_is_idempotent(self, resource, backend):
        backend.submit_circuits({"a": _BELL_QASM})
        backend.close()
        backend.close()

        assert len(resource.released) == 1

    def test_releases_the_session_of_a_backend_never_closed(
        self, resource, make_qrmi_backend
    ):
        backend = make_qrmi_backend(resource)
        backend.submit_circuits({"a": _BELL_QASM})

        del backend
        gc.collect()

        assert resource.released == resource.acquired == ["1"]

    def test_a_closed_backend_releases_nothing_more_once_gone(
        self, resource, make_qrmi_backend
    ):
        backend = make_qrmi_backend(resource)
        backend.submit_circuits({"a": _BELL_QASM})
        backend.close()

        del backend
        gc.collect()

        assert resource.released == ["1"]

    def test_defers_to_a_token_the_scheduler_exported(
        self, resource, make_qrmi_backend, monkeypatch
    ):
        """A resource manager acquires the session and exports the token;
        acquiring a second one would strand the plugin's."""
        monkeypatch.setenv("MAESTRO_LOCAL_QRMI_JOB_ACQUISITION_TOKEN", "7")
        backend = make_qrmi_backend(resource)

        backend.submit_circuits({"a": _BELL_QASM})
        backend.close()

        assert resource.acquired == []
        assert resource.released == []

    def test_leaves_the_environment_alone(self, backend):
        """The resource keeps the session it acquired, so nothing is exported."""
        backend.submit_circuits({"a": _BELL_QASM})

        assert "MAESTRO_LOCAL_QRMI_JOB_ACQUISITION_TOKEN" not in os.environ

    def test_acquires_and_releases_a_session_on_ibm(self, ibm_resource, ibm_backend):
        ibm_backend.submit_circuits({"a": _BELL_QASM})
        ibm_backend.submit_circuits({"b": _BELL_QASM})
        ibm_backend.close()

        assert ibm_resource.released == ibm_resource.acquired == ["1"]


class TestSessionRenewal:
    """The server drops a session once its lease lapses between submissions."""

    def test_reacquires_and_retries(self, resource, backend):
        resource.start_errors = [FakeConfigError(SESSION_NOT_FOUND)]

        results = backend.submit_circuits({"c0": _BELL_QASM}).results

        assert len(resource.acquired) == 2
        assert results[0]["label"] == "c0"

    def test_does_not_release_the_session_the_server_dropped(self, resource, backend):
        resource.start_errors = [FakeConfigError(SESSION_NOT_FOUND)]

        backend.submit_circuits({"c0": _BELL_QASM})

        assert resource.released == []

    def test_releases_only_the_session_it_acquired_after_a_scheduler_one_expires(
        self, resource, make_qrmi_backend, monkeypatch
    ):
        monkeypatch.setenv("MAESTRO_LOCAL_QRMI_JOB_ACQUISITION_TOKEN", "scheduler")
        resource.start_errors = [FakeConfigError(SESSION_NOT_FOUND)]
        backend = make_qrmi_backend(resource)

        backend.submit_circuits({"c0": _BELL_QASM})
        backend.close()

        assert resource.released == resource.acquired == ["1"]

    def test_a_configuration_error_is_not_mistaken_for_a_lost_session(
        self, resource, backend
    ):
        """QRMI raises ConfigError for a bad request too; retrying it on a new
        session would hide the error and could adopt a borrowed session."""
        resource.start_errors = [
            FakeConfigError("invalid configuration: invalid_config: no profile")
        ]

        with pytest.raises(FakeConfigError, match="invalid_config"):
            backend.submit_circuits({"c0": _BELL_QASM})

        assert len(resource.acquired) == 1

    def test_a_submission_after_close_holds_its_own_session(self, resource, backend):
        backend.submit_circuits({"a": _BELL_QASM})
        backend.close()
        backend.submit_circuits({"b": _BELL_QASM})
        backend.close()

        assert resource.released == resource.acquired == ["1", "2"]

    def test_a_failure_of_another_kind_is_not_retried(self, resource, backend):
        """Retrying anything but a dead session would mask the real error."""
        resource.start_errors = [RuntimeError("session limit reached")]

        with pytest.raises(RuntimeError, match="session limit reached"):
            backend.submit_circuits({"c0": _BELL_QASM})

        assert len(resource.acquired) == 1

    def test_gives_up_when_a_fresh_session_is_refused_too(self, resource, backend):
        resource.start_errors = [
            FakeConfigError(SESSION_NOT_FOUND),
            FakeConfigError(SESSION_NOT_FOUND),
        ]

        with pytest.raises(FakeConfigError):
            backend.submit_circuits({"c0": _BELL_QASM})

        assert len(resource.acquired) == 2


class TestFailureModes:
    def test_reports_the_servers_diagnostics_when_a_task_fails(self, resource, backend):
        resource.statuses = [FakeTaskStatus.Failed]
        message = (
            f"Task task-0 ended as {FakeTaskStatus.Failed}: "
            "backend_unavailable: no such backend"
        )

        with pytest.raises(QRMITaskFailedError, match=exact_match(message)):
            backend.submit_circuits({"c0": _BELL_QASM})
        assert resource.stopped == []

    @pytest.mark.parametrize(
        ("failure", "reason"),
        [
            pytest.param(
                {"error": None, "failure_message": "cancelled by another client"},
                "cancelled by another client",
                id="failure-message",
            ),
            pytest.param({"error": None}, "no failure recorded", id="nothing-recorded"),
            pytest.param(
                {
                    "error": {"code": "oom", "message": "out of memory"},
                    "logs": "rank 0\n",
                },
                "oom: out of memory\nrank 0",
                id="with-logs",
            ),
        ],
    )
    def test_reports_each_shape_of_server_diagnostics(
        self, resource, backend, failure, reason
    ):
        resource.statuses = [FakeTaskStatus.Cancelled]
        resource.failure = failure
        message = f"Task task-0 ended as {FakeTaskStatus.Cancelled}: {reason}"

        with pytest.raises(QRMITaskFailedError, match=exact_match(message)):
            backend.submit_circuits({"c0": _BELL_QASM})

    def test_says_so_when_the_logs_cannot_be_read(self, resource, backend):
        resource.statuses = [FakeTaskStatus.Failed]
        resource.logs_error = RuntimeError("session gone")
        message = (
            f"Task task-0 ended as {FakeTaskStatus.Failed}: "
            "no diagnostics available (session gone)"
        )

        with pytest.raises(QRMITaskFailedError, match=exact_match(message)):
            backend.submit_circuits({"c0": _BELL_QASM})

    def test_says_so_when_the_logs_are_not_json(self, resource, backend):
        resource.statuses = [FakeTaskStatus.Failed]
        resource.logs = "not json"

        with pytest.raises(QRMITaskFailedError, match="no diagnostics available"):
            backend.submit_circuits({"c0": _BELL_QASM})


class TestPolling:
    def test_polls_until_the_task_completes(self, mocker, resource, backend):
        """Only Completed ends the wait. Treating Queued as terminal would
        fetch a result the server has not written yet."""
        resource.statuses = [
            FakeTaskStatus.Queued,
            FakeTaskStatus.Running,
            FakeTaskStatus.Completed,
        ]
        polls = mocker.spy(resource, "task_status")

        backend.submit_circuits({"c0": _BELL_QASM})

        assert polls.call_count == 3

    @pytest.mark.parametrize(
        ("kind", "interval"),
        [
            pytest.param("resource", 0.05, id="maestro-socket"),
            pytest.param("ibm_resource", 1.0, id="ibm-rest-api"),
        ],
    )
    def test_waits_on_the_cancellation_event_at_the_resources_interval(
        self, request, mocker, make_qrmi_backend, kind, interval
    ):
        """Waiting on the event lets a cancel end the wait at once."""
        resource = request.getfixturevalue(kind)
        resource.statuses = [FakeTaskStatus.Running]
        event = mocker.Mock(spec=Event)
        event.is_set.return_value = False

        make_qrmi_backend(resource).submit_circuits(
            {"c0": _BELL_QASM}, cancellation_event=event
        )

        event.wait.assert_called_once_with(interval)


class TestCancellation:
    def test_does_not_dispatch_after_a_cancel_arrives(
        self, fake_qrmi, resource, backend
    ):
        """A cancel landing after submit_circuits' own check, while the request
        waits for the resource, must still stop it before dispatch."""
        event = Event()
        build_payload = fake_qrmi.maestro.request_payload.side_effect

        def cancel_then_build(request):
            event.set()
            return build_payload(request)

        fake_qrmi.maestro.request_payload.side_effect = cancel_then_build

        with pytest.raises(
            ExecutionCancelledError,
            match=exact_match("QRMI batch cancelled before dispatch"),
        ):
            backend.submit_circuits({"a": _BELL_QASM}, cancellation_event=event)
        assert resource.payloads == []

    def test_a_failing_stop_still_reports_cancellation(self, mocker, resource, backend):
        """task_stop raising would surface a user's cancel as a batch failure,
        which the coordinator reports as Failed rather than Cancelled."""
        event = Event()
        resource.on_task_start = lambda _payload: event.set()
        resource.stop_error = RuntimeError("task already gone")
        stop = mocker.spy(resource, "task_stop")

        with pytest.raises(ExecutionCancelledError):
            backend.submit_circuits({"a": _BELL_QASM}, cancellation_event=event)

        stop.assert_called_once_with("task-0")

    def test_an_error_while_waiting_stops_the_task(self, resource, backend):
        """Otherwise the task runs on and blocks the session's next one."""
        resource.status_error = RuntimeError("socket closed")

        with pytest.raises(RuntimeError, match=exact_match("socket closed")):
            backend.submit_circuits({"a": _BELL_QASM})

        assert resource.stopped == ["task-0"]

    def test_stops_a_task_cancelled_while_it_runs(self, resource, backend):
        """Polling is interruptible, so a cancel mid-task must tell the server
        to drop it rather than wait the circuit out."""
        event = Event()
        resource.on_task_start = lambda _payload: event.set()

        with pytest.raises(
            ExecutionCancelledError,
            match=exact_match("QRMI batch cancelled while task task-0 was running"),
        ):
            backend.submit_circuits({"a": _BELL_QASM}, cancellation_event=event)

        assert resource.stopped == ["task-0"]

    def test_aborts_between_circuits(self, resource, backend):
        event = Event()
        resource.on_task_result = lambda _task_id: event.set()

        with pytest.raises(
            ExecutionCancelledError,
            match=exact_match("QRMI batch cancelled after partial completion"),
        ):
            backend.submit_circuits(
                {"a": _BELL_QASM, "b": _BELL_QASM}, cancellation_event=event
            )

        assert len(resource.payloads) == 1


class TestVectorJacobianProducts:
    _UNAVAILABLE = (
        "QRMIBackend computes vector-Jacobian products only on maestro-local "
        "resources, with native, noiseless estimation."
    )

    def _submit(self, backend):
        return backend.submit_vjp(
            [make_vjp_payload()], ham_ops="ZZ;XI", cotangents=VJP_COTANGENTS
        )

    def test_supports_vjp_on_maestro_local(self, backend):
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
            [make_vjp_payload("ry(2*w_0) q[0];\nrz(w_1) q[1];\n")],
            ham_ops="ZZ;XI",
            cotangents=VJP_COTANGENTS,
        ).results

        np.testing.assert_allclose(
            resource.last_request["parameter_rows"], [[0.6, -0.7], [-2.4, 0.4]]
        )
        np.testing.assert_allclose(results[0]["metadata"]["vjp"], [2.0, 1.0])

    def test_tracked_depths_have_one_entry_per_row(self, resource, make_qrmi_backend):
        backend = make_qrmi_backend(resource, track_depth=True)
        payload = make_vjp_payload()

        backend.submit_vjp([payload], ham_ops="ZZ;XI", cotangents=VJP_COTANGENTS)

        assert [len(batch) for batch in backend.depth_history] == [
            len(payload.parameter_sets)
        ]

    def test_forced_sampling_rules_it_out(self, resource, make_qrmi_backend):
        backend = make_qrmi_backend(resource, force_sampling=True)

        with pytest.raises(VJPUnavailableError, match=exact_match(self._UNAVAILABLE)):
            self._submit(backend)

    def test_a_server_refusal_makes_it_unavailable(self, resource, backend):
        resource.start_errors = [FakeUnsupportedFunctionError("unsupported_capability")]
        message = "Maestro cannot differentiate the circuit: unsupported_capability"

        with pytest.raises(VJPUnavailableError, match=exact_match(message)):
            self._submit(backend)

    def test_running_out_of_memory_makes_it_unavailable(self, resource, backend):
        resource.statuses = [FakeTaskStatus.Failed]
        resource.failure = {
            "error": {"code": "insufficient_resources", "message": "state too large"}
        }

        with pytest.raises(VJPUnavailableError, match="insufficient_resources"):
            self._submit(backend)

    def test_a_task_that_fails_otherwise_is_not_hidden_as_unavailable(
        self, resource, backend
    ):
        resource.statuses = [FakeTaskStatus.Failed]

        with pytest.raises(QRMITaskFailedError, match="backend_unavailable"):
            self._submit(backend)

    def test_a_rejected_request_is_not_hidden_as_unavailable(self, resource, backend):
        """A malformed request is a bug to surface, not a reason to fall back."""
        resource.start_errors = [FakeInvalidInputError("Unknown field: parameter_rows")]

        with pytest.raises(FakeInvalidInputError):
            self._submit(backend)

    def test_a_cancel_arriving_first_dispatches_nothing(self, resource, backend):
        event = Event()
        event.set()

        with pytest.raises(ExecutionCancelledError):
            backend.submit_vjp(
                [make_vjp_payload()],
                ham_ops="ZZ;XI",
                cotangents=VJP_COTANGENTS,
                cancellation_event=event,
            )

        assert resource.payloads == []


class TestConcurrency:
    def test_concurrent_submissions_never_interleave(self, resource, backend):
        """QRMI resource methods take `&mut self`, so overlapping calls into one
        resource raise "Already borrowed". Every method is probed, not just the
        poll: a lock narrowed to the poll loop would still let start and result
        interleave, which is the regression that breaks the borrow."""
        probe = _OverlapProbe()
        for name in ("acquire", "task_start", "task_status", "task_result"):
            setattr(resource, name, probe.wrap(getattr(resource, name)))

        _submit_concurrently(backend)

        assert probe.overlapped is False
        assert len(resource.payloads) == 4

    def test_compiles_one_ibm_batch_at_a_time(self, mocker, ibm_backend):
        """A pass manager keeps each run's state on itself, so concurrent runs
        on one would corrupt each other."""
        probe = _OverlapProbe()
        mocker.patch.object(
            qrmi_module,
            "generate_preset_pass_manager",
            return_value=mocker.Mock(run=probe.wrap(lambda circuits: circuits)),
        )

        _submit_concurrently(ibm_backend)

        assert probe.overlapped is False


class TestIBM:
    @pytest.mark.parametrize(
        "ibm_resource",
        [
            pytest.param(FakeResourceType.IBMQuantumSystem, id="quantum-system"),
            pytest.param(FakeResourceType.IBMQiskitRuntimeService, id="runtime"),
            pytest.param(
                FakeResourceType.IBMQuantumComputeService, id="compute-service"
            ),
        ],
        indirect=True,
    )
    def test_samples_on_every_ibm_resource_type(self, ibm_backend):
        (entry,) = ibm_backend.submit_circuits({"c0": _BELL_QASM}).results

        assert (entry["label"], entry["results"]) == ("c0", {"00": 4, "11": 6})

    def test_keys_counts_with_the_first_register_leftmost(
        self, ibm_resource, ibm_backend
    ):
        """As for Maestro: c[0] of the first register is the leftmost bit."""
        two_registers = (
            'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[3];\ncreg a[1];\n'
            "creg b[2];\nx q[0];\nmeasure q[0] -> a[0];\nmeasure q[1] -> b[0];\n"
            "measure q[2] -> b[1];\n"
        )
        ibm_resource.registers = {"a": {"1": 10}, "b": {"00": 10}}

        (entry,) = ibm_backend.submit_circuits({"c0": two_registers}).results

        assert entry["results"] == {"100": 10}

    def test_samples_a_batch_in_one_sampler_job(self, ibm_resource, ibm_backend):
        result = ibm_backend.submit_circuits({"c0": _BELL_QASM, "c1": _BELL_QASM})

        (payload,) = ibm_resource.payloads
        job = ibm_resource.last_request
        assert payload.program_id == "sampler"
        assert [shots for _source, _parameters, shots in job["pubs"]] == [100, 100]
        assert job["shots"] == 100
        assert result.backend_jobs == 1

    def test_runs_one_job_per_shot_count(self, ibm_resource, ibm_backend):
        """Qiskit Runtime has deprecated different shots across one job's pubs."""
        result = ibm_backend.submit_circuits(
            {"a": _BELL_QASM, "b": _BELL_QASM, "c": _BELL_QASM},
            shot_groups=[[0, 1, 30], [1, 2, 70], [2, 3, 30]],
        )

        jobs = ibm_resource.requests
        assert [job["shots"] for job in jobs] == [30, 70]
        assert [[pub[2] for pub in job["pubs"]] for job in jobs] == [[30, 30], [70]]
        assert [entry["label"] for entry in result.results] == ["a", "b", "c"]
        assert result.backend_jobs == 2

    def test_sends_each_circuit_compiled_for_the_backend_as_qpy(
        self, ibm_runtime, ibm_resource, ibm_backend, ibm_backend_factory
    ):
        ibm_backend.submit_circuits({"c0": _BELL_QASM})

        ibm_backend_factory.assert_called_once_with(
            ibm_resource, use_fractional_gates=False
        )
        ((compiled, _parameters, _shots),) = json.loads(
            ibm_resource.payloads[0].input, cls=ibm_runtime.RuntimeDecoder
        )["pubs"]
        basis = set(GenericBackendV2(num_qubits=5, seed=0).operation_names)
        assert {instruction.name for instruction in compiled.data} <= basis
        assert compiled.count_ops()["measure"] == 2

    def test_compiles_against_the_backend_fetched_once(
        self, ibm_backend, ibm_backend_factory
    ):
        ibm_backend.submit_circuits({"c0": _BELL_QASM})
        ibm_backend.submit_circuits({"c0": _BELL_QASM})

        ibm_backend_factory.assert_called_once()

    def test_rejects_circuits_without_classical_bits(self, ibm_resource, ibm_backend):
        unmeasured = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\nh q[0];\n'
        message = (
            "IBM resources return measurement counts, but circuits ['c1'] have "
            "no classical bits."
        )

        with pytest.raises(ValueError, match=exact_match(message)):
            ibm_backend.submit_circuits({"c0": _BELL_QASM, "c1": unmeasured})
        assert ibm_resource.payloads == []

    def test_deletes_each_job_once_read(self, ibm_resource, ibm_backend):
        ibm_backend.submit_circuits({"c0": _BELL_QASM})

        assert ibm_resource.stopped == ["task-0"]

    def test_still_returns_results_when_deleting_the_job_fails(
        self, ibm_resource, ibm_backend
    ):
        ibm_resource.stop_error = RuntimeError("job not found")

        (entry,) = ibm_backend.submit_circuits({"c0": _BELL_QASM}).results

        assert entry["results"] == {"00": 4, "11": 6}

    def test_keeps_a_job_whose_result_cannot_be_read(
        self, ibm_resource, ibm_backend, caplog
    ):
        """Deleting it would throw away results the device already produced."""
        ibm_resource.result_error = RuntimeError("S3 unavailable")

        with pytest.raises(RuntimeError, match=exact_match("S3 unavailable")):
            ibm_backend.submit_circuits({"c0": _BELL_QASM})

        assert ibm_resource.stopped == []
        assert (
            "IBM job task-0 finished but its result could not be read; the job is "
            "kept on the service."
        ) in caplog.text

    @pytest.mark.parametrize(
        "status",
        [
            pytest.param(FakeTaskStatus.Failed, id="failed"),
            pytest.param(FakeTaskStatus.Cancelled, id="cancelled"),
        ],
    )
    def test_reports_ibms_reason_and_the_jobs_logs_when_it_fails(
        self, ibm_resource, ibm_backend, status
    ):
        ibm_resource.statuses = [status]
        message = (
            f"Task task-0 ended as {status}: {ibm_resource.failure_reason}\n"
            f"{ibm_resource.logs.strip()}"
        )

        with pytest.raises(QRMITaskFailedError, match=exact_match(message)):
            ibm_backend.submit_circuits({"c0": _BELL_QASM})
        assert ibm_resource.stopped == ["task-0"]

    def test_a_failure_without_logs_still_gives_ibms_reason(
        self, ibm_resource, ibm_backend
    ):
        ibm_resource.statuses = [FakeTaskStatus.Failed]
        ibm_resource.logs_error = RuntimeError("no logs object")
        message = (
            f"Task task-0 ended as {FakeTaskStatus.Failed}: "
            f"{ibm_resource.failure_reason}"
        )

        with pytest.raises(QRMITaskFailedError, match=exact_match(message)):
            ibm_backend.submit_circuits({"c0": _BELL_QASM})

    def test_stops_a_job_cancelled_while_it_runs(self, ibm_resource, ibm_backend):
        event = Event()
        ibm_resource.on_task_start = lambda _payload: event.set()

        with pytest.raises(ExecutionCancelledError, match="while task task-0"):
            ibm_backend.submit_circuits({"c0": _BELL_QASM}, cancellation_event=event)

        assert ibm_resource.stopped == ["task-0"]

    def test_a_cancel_during_compilation_dispatches_nothing(
        self, ibm_resource, ibm_backend, ibm_backend_factory
    ):
        event = Event()
        device = ibm_backend_factory.return_value

        def fetch_and_cancel(*_args, **_kwargs):
            event.set()
            return device

        ibm_backend_factory.side_effect = fetch_and_cancel

        with pytest.raises(
            ExecutionCancelledError,
            match=exact_match("QRMI batch cancelled before dispatch"),
        ):
            ibm_backend.submit_circuits({"c0": _BELL_QASM}, cancellation_event=event)
        assert ibm_resource.payloads == []

    def test_estimates_nothing_natively(self, ibm_backend):
        assert ibm_backend.supports_expval is False
        assert ibm_backend.supports_vjp is False

    def test_rejects_ham_ops(self, ibm_backend):
        with pytest.raises(ValueError, match=exact_match(_maestro_only("ham_ops"))):
            ibm_backend.submit_circuits({"c0": _BELL_QASM}, ham_ops="ZZ")

    def test_rejects_expectation_values(self, ibm_backend):
        with pytest.raises(ValueError, match=exact_match(_maestro_only("ham_ops"))):
            ibm_backend.expectation_value(_BELL_QASM, SparsePauliOp("ZZ"))

    def test_rejects_statevectors(self, ibm_backend):
        with pytest.raises(ValueError, match=exact_match(_maestro_only("statevector"))):
            ibm_backend.statevector(_BELL_QASM)

    def test_rejects_gradients(self, ibm_backend):
        with pytest.raises(
            VJPUnavailableError,
            match=exact_match(TestVectorJacobianProducts._UNAVAILABLE),
        ):
            ibm_backend.submit_vjp(
                [make_vjp_payload()], ham_ops="ZZ", cotangents=VJP_COTANGENTS
            )

    def test_takes_no_maestro_config_at_construction(
        self, ibm_resource, make_qrmi_backend
    ):
        message = (
            "maestro_config applies to maestro-local resources; this resource is "
            f"{_IBM}. Pass device_config instead."
        )

        with pytest.raises(ValueError, match=exact_match(message)):
            make_qrmi_backend(ibm_resource, maestro_config=MaestroConfig())

    def test_takes_no_maestro_config_on_assignment(self, ibm_backend):
        with pytest.raises(ValueError, match="maestro_config applies to maestro-local"):
            ibm_backend.maestro_config = MaestroConfig()
        assert ibm_backend.maestro_config is None

    def test_set_seed_leaves_the_job_unseeded(self, ibm_resource, ibm_backend):
        ibm_backend.set_seed(7)
        ibm_backend.submit_circuits({"c0": _BELL_QASM})

        assert ibm_resource.last_request["options"] == {}


class TestIBMDeviceConfig:
    def test_compiles_at_the_level_qoros_service_defaults_to(self, mocker, ibm_backend):
        compile_spy = mocker.spy(qrmi_module, "generate_preset_pass_manager")

        ibm_backend.submit_circuits({"c0": _BELL_QASM})

        assert compile_spy.call_args.kwargs["optimization_level"] == 2

    @pytest.mark.parametrize(
        ("option", "value", "argument"),
        [
            pytest.param(
                "transpile_optimization_level", 0, "optimization_level", id="level"
            ),
            pytest.param(
                "transpile_layout_method", "trivial", "layout_method", id="layout"
            ),
            pytest.param(
                "transpile_routing_method", "basic", "routing_method", id="routing"
            ),
            pytest.param("transpile_seed", 3, "seed_transpiler", id="seed"),
            pytest.param(
                "transpile_approximation_degree",
                0.9,
                "approximation_degree",
                id="approximation",
            ),
            pytest.param(
                "transpile_initial_layout", [1, 0], "initial_layout", id="initial"
            ),
        ],
    )
    def test_transpile_keys_reach_the_pass_manager(
        self, mocker, ibm_resource, make_qrmi_backend, option, value, argument
    ):
        compile_spy = mocker.spy(qrmi_module, "generate_preset_pass_manager")
        backend = make_qrmi_backend(
            ibm_resource, device_config=DeviceConfig(**{option: value})
        )

        backend.submit_circuits({"c0": _BELL_QASM})

        assert compile_spy.call_args.kwargs[argument] == value

    @pytest.mark.parametrize(
        ("option", "value", "group", "name"),
        [
            pytest.param(
                "dynamical_decoupling_enable",
                True,
                "dynamical_decoupling",
                "enable",
                id="dd-enable",
            ),
            pytest.param(
                "dynamical_decoupling_sequence_type",
                "XY4",
                "dynamical_decoupling",
                "sequence_type",
                id="dd-sequence",
            ),
            pytest.param(
                "dynamical_decoupling_scheduling_method",
                "asap",
                "dynamical_decoupling",
                "scheduling_method",
                id="dd-scheduling",
            ),
            pytest.param(
                "dynamical_decoupling_extra_slack_distribution",
                "edges",
                "dynamical_decoupling",
                "extra_slack_distribution",
                id="dd-slack",
            ),
            pytest.param(
                "twirling_enable_gates",
                True,
                "twirling",
                "enable_gates",
                id="twirl-gates",
            ),
            pytest.param(
                "twirling_enable_measure",
                True,
                "twirling",
                "enable_measure",
                id="twirl-measure",
            ),
            pytest.param(
                "twirling_num_randomizations",
                "auto",
                "twirling",
                "num_randomizations",
                id="twirl-randomizations",
            ),
            pytest.param(
                "twirling_shots_per_randomization",
                64,
                "twirling",
                "shots_per_randomization",
                id="twirl-shots",
            ),
            pytest.param(
                "twirling_strategy",
                "active-circuit",
                "twirling",
                "strategy",
                id="twirl-strategy",
            ),
            pytest.param(
                "execution_rep_delay",
                0.00025,
                "execution",
                "rep_delay",
                id="rep-delay",
            ),
            pytest.param(
                "execution_init_qubits",
                False,
                "execution",
                "init_qubits",
                id="init-qubits",
            ),
        ],
    )
    def test_sampler_keys_become_options_qiskit_runtime_accepts(
        self, ibm_resource, ibm_runtime, make_qrmi_backend, option, value, group, name
    ):
        """The stand-in device ignores sampler options, so names are checked
        against Qiskit Runtime's own option classes, which reject unknown ones."""
        backend = make_qrmi_backend(
            ibm_resource, device_config=DeviceConfig(**{option: value})
        )

        backend.submit_circuits({"c0": _BELL_QASM})

        options = ibm_resource.last_request["options"]
        assert options == {group: {name: value}}
        option_class = {
            "dynamical_decoupling": ibm_runtime.options.DynamicalDecouplingOptions,
            "twirling": ibm_runtime.options.TwirlingOptions,
            "execution": ibm_runtime.options.SamplerExecutionOptionsV2,
        }[group]
        assert getattr(option_class(**options[group]), name) == value

    def test_fractional_gates_select_the_fractional_backend(
        self, ibm_resource, make_qrmi_backend, ibm_backend_factory
    ):
        backend = make_qrmi_backend(
            ibm_resource, device_config=DeviceConfig(use_fractional_gates=True)
        )

        backend.submit_circuits({"c0": _BELL_QASM})

        ibm_backend_factory.assert_called_once_with(
            ibm_resource, use_fractional_gates=True
        )

    def test_a_new_config_recompiles(self, ibm_backend, ibm_backend_factory):
        ibm_backend.submit_circuits({"c0": _BELL_QASM})
        ibm_backend.device_config = DeviceConfig(use_fractional_gates=True)
        ibm_backend.submit_circuits({"c0": _BELL_QASM})

        assert [call.kwargs for call in ibm_backend_factory.call_args_list] == [
            {"use_fractional_gates": False},
            {"use_fractional_gates": True},
        ]

    def test_none_restores_the_defaults(self, ibm_resource, make_qrmi_backend):
        backend = make_qrmi_backend(
            ibm_resource, device_config=DeviceConfig(twirling_enable_gates=True)
        )

        backend.device_config = None
        backend.submit_circuits({"c0": _BELL_QASM})

        assert backend.device_config is None
        assert ibm_resource.last_request["options"] == {}

    def test_rejects_keys_only_qoros_service_applies(
        self, ibm_resource, make_qrmi_backend
    ):
        config = DeviceConfig(aggressive_compiling=False, use_mitigation=True)
        message = (
            "QRMIBackend cannot apply device_config keys 'AGGRESSIVE_COMPILING', "
            "'USE_MITIGATION' on IBM resources; it takes the TRANSPILE_*, "
            "DYNAMICAL_DECOUPLING_*, TWIRLING_*, EXECUTION_REP_DELAY, "
            "EXECUTION_INIT_QUBITS and USE_FRACTIONAL_GATES keys."
        )

        with pytest.raises(ValueError, match=exact_match(message)):
            make_qrmi_backend(ibm_resource, device_config=config)

    def test_suggests_the_key_a_typo_meant(self, ibm_resource, make_qrmi_backend):
        config = DeviceConfig(twirling_enable_gate=True)

        with pytest.raises(ValueError, match="'TWIRLING_ENABLE_GATE' \\(did you mean "):
            make_qrmi_backend(ibm_resource, device_config=config)

    def test_rejects_another_type(self, ibm_backend):
        with pytest.raises(
            TypeError,
            match=exact_match("device_config must be a DeviceConfig, got dict."),
        ):
            ibm_backend.device_config = {"twirling_enable_gates": True}

    def test_a_rejected_config_keeps_the_previous_one(
        self, ibm_resource, make_qrmi_backend, ibm_backend_factory
    ):
        config = DeviceConfig(twirling_enable_gates=True)
        backend = make_qrmi_backend(ibm_resource, device_config=config)
        backend.submit_circuits({"c0": _BELL_QASM})

        with pytest.raises(ValueError, match="USE_MITIGATION"):
            backend.device_config = DeviceConfig(use_mitigation=True)
        backend.submit_circuits({"c0": _BELL_QASM})

        assert backend.device_config is config
        assert ibm_backend_factory.call_count == 1
        assert ibm_resource.last_request["options"] == {
            "twirling": {"enable_gates": True}
        }


class _QRMIContracts(SyncRunnerContractsBase):
    """Shared :class:`~divi.backends.CircuitRunner` behavioural contracts, on
    the resource a subclass's ``contract_resource`` fixture provides."""

    @pytest.fixture()
    def contract_runner_disabled(self, make_qrmi_backend, contract_resource):
        return make_qrmi_backend(
            contract_resource, shots=CONTRACT_TEST_SHOTS, track_depth=False
        )

    @pytest.fixture()
    def contract_runner_enabled(self, make_qrmi_backend, contract_resource):
        return make_qrmi_backend(
            contract_resource, shots=CONTRACT_TEST_SHOTS, track_depth=True
        )

    @pytest.fixture()
    def contract_runner_default(self, make_qrmi_backend, contract_resource):
        return make_qrmi_backend(contract_resource, shots=CONTRACT_TEST_SHOTS)


class TestIBMContracts(_QRMIContracts):
    @pytest.fixture()
    def contract_resource(self, ibm_resource):
        return ibm_resource


class TestContracts(_QRMIContracts):
    @pytest.fixture()
    def contract_resource(self, resource):
        return resource
