# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import warnings
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import pytest
from qiskit import QuantumCircuit, qasm2

pytest.importorskip("qiskit_aer")

from qiskit_aer import AerJob, AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError
from qiskit_ibm_runtime.fake_provider import FakeQuitoV2

from divi.backends import (
    ExecutionResult,
    QiskitSimulator,
    create_backend_from_properties,
)
from divi.backends.runners._qiskit import (
    FAKE_BACKENDS,
    _default_n_processes,
    _find_best_fake_backend,
)
from tests._helpers import exact_match
from tests.backends._circuit_runner_contracts import (
    CONTRACT_TEST_SHOTS,
    QASM_DEPTH_2,
    QASM_DEPTH_3,
    QASM_X_ON_FIRST_QUBIT,
    SyncRunnerContractsBase,
)
from tests.backends._helpers import (
    SHOT_GROUPS_WITH_HAM_OPS_MESSAGE,
    padding_warning,
    uncovered_circuits_message,
)

_DOUBLE_X_QASM = (
    'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\ncreg c[1];\n'
    "x q[0];\nx q[0];\nmeasure q[0] -> c[0];\n"
)

_WIDE_QASM = (
    'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[28];\ncreg c[28];\n'
    "h q[0];\nmeasure q[0] -> c[0];\n"
)

_NO_FAKE_BACKEND_MESSAGE = (
    "No fake backend available for circuit with 28 qubits. "
    "Please provide an explicit backend or use a smaller circuit."
)


def _gate(
    name: str,
    qubits: tuple[int, ...],
    *,
    length_ns: float | None = None,
    error: float = 0.0,
) -> dict:
    parameters = [{"name": "gate_error", "value": error}]
    if length_ns is not None:
        parameters.append({"name": "gate_length", "value": length_ns, "unit": "ns"})
    return {"gate": name, "qubits": list(qubits), "parameters": parameters}


def _calibrated_backend(
    n_qubits: int, *gates: dict, readout_length_ns: float | None = None
):
    """A backend carrying exactly ``gates``, on long-lived qubits."""
    qubit = [
        {"name": "T1", "value": 1e6, "unit": "us"},
        {"name": "T2", "value": 1e6, "unit": "us"},
    ]
    if readout_length_ns is not None:
        qubit.append(
            {"name": "readout_length", "value": readout_length_ns, "unit": "ns"}
        )
    return create_backend_from_properties(
        {"qubits": [qubit] * n_qubits, "gates": list(gates)}
    )


class _InProcessPool:
    """Stands in for ``multiprocessing.Pool``, mapping in the calling process."""

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def map(self, fn, iterable):
        return [fn(item) for item in iterable]


def test_fake_backend_sizes_match_their_keys():
    for n_qubits, backend_classes in FAKE_BACKENDS.items():
        assert [cls().num_qubits for cls in backend_classes] == [n_qubits] * len(
            backend_classes
        )


@pytest.mark.parametrize(
    "n_qubits, expected",
    [
        pytest.param(3, FAKE_BACKENDS[5], id="small_circuit_gets_5q_backend"),
        pytest.param(20, FAKE_BACKENDS[20], id="large_circuit_gets_20q_backend"),
        pytest.param(100, None, id="exceeds_all_backends"),
    ],
)
def test_find_best_fake_backend(n_qubits, expected):
    """_find_best_fake_backend picks the smallest fitting backend, or None if none fit."""
    circuit = QuantumCircuit(n_qubits)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure_all()

    assert _find_best_fake_backend(circuit) == expected


class TestQiskitSimulatorInit:
    """Tests for QiskitSimulator initialization."""

    def test_init_with_backend_and_noise_model_warns(self):
        """Test that warning is issued when both backend and noise_model are provided (line 88)."""
        backend = FakeQuitoV2()
        noise_model = NoiseModel()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            simulator = QiskitSimulator(qiskit_backend=backend, noise_model=noise_model)

            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)
            assert "Both `qiskit_backend` and `noise_model`" in str(w[0].message)
            assert "`noise_model` will be ignored" in str(w[0].message)

        # Verify simulator was still created
        assert simulator.qiskit_backend == backend
        assert simulator.noise_model == noise_model

    @pytest.mark.parametrize(
        "backend, noise_model",
        [(FakeQuitoV2(), None), ("auto", None), (None, NoiseModel())],
        ids=["backend", "auto_backend", "noise_model"],
    )
    def test_a_backend_or_noise_model_alone_forces_sampling_without_warning(
        self, backend, noise_model
    ):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            simulator = QiskitSimulator(qiskit_backend=backend, noise_model=noise_model)

        assert simulator.qiskit_backend is backend
        assert simulator.noise_model is noise_model
        assert simulator.supports_expval is False

    def test_init_accepts_a_single_process(self):
        assert QiskitSimulator(n_processes=1).n_processes == 1

    def test_shots_default_to_5000(self):
        assert QiskitSimulator().shots == 5000

    @pytest.mark.parametrize("bad_value", [0, -1, -100])
    def test_init_rejects_n_processes_below_one(self, bad_value):
        """Constructor raises ValueError when n_processes < 1."""
        with pytest.raises(
            ValueError, match=exact_match(f"n_processes must be >= 1, got {bad_value}")
        ):
            QiskitSimulator(n_processes=bad_value)

    def test_setter_rejects_n_processes_below_one(self):
        """The n_processes setter raises ValueError when value < 1."""
        sim = QiskitSimulator()
        with pytest.raises(
            ValueError, match=exact_match("n_processes must be >= 1, got 0")
        ):
            sim.n_processes = 0

    def test_init_none_uses_default(self, mocker):
        """When n_processes is None, the default is computed via _default_n_processes."""
        mocker.patch(
            "divi.backends.runners._qiskit._default_n_processes", return_value=5
        )
        sim = QiskitSimulator(n_processes=None)
        assert sim.n_processes == 5


class TestQiskitSimulatorProperties:
    """Tests for QiskitSimulator properties and methods."""

    def test_set_seed(self):
        """Test set_seed method (line 107)."""
        simulator = QiskitSimulator()
        assert simulator.simulation_seed is None

        simulator.set_seed(42)
        assert simulator.simulation_seed == 42

        simulator.set_seed(100)
        assert simulator.simulation_seed == 100

    @pytest.mark.parametrize(
        "force_sampling, expected",
        [
            pytest.param(False, True, id="default"),
            pytest.param(True, False, id="force_sampling"),
        ],
    )
    def test_supports_expval(self, force_sampling, expected):
        """force_sampling=True makes supports_expval return False."""
        simulator = QiskitSimulator(force_sampling=force_sampling)
        assert simulator.supports_expval is expected

    def test_is_async(self):
        """Test is_async property (line 121)."""
        simulator = QiskitSimulator()
        assert simulator.is_async is False

    def test_optimization_level_defaults_to_qiskits_choice(self):
        """Transpilation is the caller's to control; divi imposes no default."""
        assert QiskitSimulator().optimization_level is None

    @pytest.mark.parametrize("level", [None, 0, 2])
    def test_optimization_level_reaches_the_transpiler(self, mocker, level):
        """The setting is worthless unless it is threaded into transpile()."""
        spy = mocker.patch(
            "divi.backends.runners._qiskit.transpile",
            side_effect=lambda circuits, *a, **kw: circuits,
        )
        simulator = QiskitSimulator(shots=10, optimization_level=level)
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.measure_all()
        simulator.submit_circuits({"c0": qasm2.dumps(qc)})

        assert spy.call_args.kwargs["optimization_level"] == level

    @pytest.mark.parametrize(
        "in_worker_thread, expected", [(False, 4), (True, 1)], ids=["main", "worker"]
    )
    def test_transpile_forks_only_from_the_main_thread(
        self, mocker, in_worker_thread, expected
    ):
        """Forking transpile workers from a multithreaded process can deadlock."""
        spy = mocker.patch(
            "divi.backends.runners._qiskit.transpile",
            side_effect=lambda circuits, *a, **kw: circuits,
        )
        simulator = QiskitSimulator(n_processes=4, shots=10)
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.measure_all()
        circuits = {"c0": qasm2.dumps(qc)}

        if in_worker_thread:
            with ThreadPoolExecutor(max_workers=1) as executor:
                executor.submit(simulator.submit_circuits, circuits).result()
        else:
            simulator.submit_circuits(circuits)

        assert spy.call_args.kwargs["num_processes"] == expected


def _mock_aer_result(mocker, counts, metadata=None):
    """A mock Aer ``Result``; a list of ``counts`` is served one per circuit."""
    result = mocker.Mock()
    if isinstance(counts, list):
        result.get_counts.side_effect = counts
    else:
        result.get_counts.return_value = counts
    result.metadata = metadata or {"parallel_experiments": 1, "omp_nested": False}
    result.time_taken = 0.01
    return result


class TestQiskitSimulatorSubmitCircuits:
    """Tests for QiskitSimulator.submit_circuits method."""

    def _create_qasm_circuit(self, n_qubits=2):
        """Helper to create a QASM circuit string."""
        return f"""
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[{n_qubits}];
        creg c[{n_qubits}];
        h q[0];
        measure q[0] -> c[0];
        """

    def _setup_mock_aer_simulator(
        self, mocker, counts=None, metadata=None, use_from_backend=False
    ):
        """Helper to set up mock AerSimulator."""
        mock_aer = mocker.Mock()
        mock_aer.run.return_value.result.return_value = _mock_aer_result(
            mocker, {"0": 50, "1": 50} if counts is None else counts, metadata
        )

        if use_from_backend:
            return (
                mocker.patch(
                    "divi.backends.runners._qiskit.AerSimulator.from_backend",
                    return_value=mock_aer,
                ),
                mock_aer,
            )
        else:
            mocker.patch(
                "divi.backends.runners._qiskit.AerSimulator",
                return_value=mock_aer,
            )
            return mock_aer

    def _setup_mock_transpile(self, mocker, n_qubits=2, num_circuits=1):
        """Helper to set up mock transpile."""
        mock_transpiled = QuantumCircuit(n_qubits)
        mocker.patch(
            "divi.backends.runners._qiskit.transpile",
            return_value=[mock_transpiled] * num_circuits,
        )

    def test_submit_circuits_with_auto_backend(self, mocker):
        """'auto' simulates on the newest fake backend wide enough for the circuit."""
        simulator = QiskitSimulator(qiskit_backend="auto", shots=100)
        qasm = self._create_qasm_circuit(n_qubits=3)
        spy = mocker.spy(AerSimulator, "from_backend")

        result = simulator.submit_circuits({"test_circuit": qasm})

        assert isinstance(spy.call_args.args[0], FAKE_BACKENDS[5][-1])
        assert result.results[0]["label"] == "test_circuit"
        assert sum(result.results[0]["results"].values()) == 100

    def test_auto_backend_rejects_circuits_wider_than_every_fake_backend(self):
        with pytest.raises(ValueError, match=exact_match(_NO_FAKE_BACKEND_MESSAGE)):
            QiskitSimulator(qiskit_backend="auto").submit_circuits({"c0": _WIDE_QASM})

    @pytest.mark.parametrize(
        "deterministic", [False, True], ids=["batched", "deterministic"]
    )
    def test_the_seed_pins_the_sampling_stream(self, deterministic):
        def counts(seed):
            simulator = QiskitSimulator(
                shots=200,
                simulation_seed=seed,
                _deterministic_execution=deterministic,
            )
            result = simulator.submit_circuits(
                {"c0": self._create_qasm_circuit(), "c1": self._create_qasm_circuit()}
            )
            return [entry["results"] for entry in result.results]

        assert counts(7) == counts(7)
        assert counts(7) != counts(8)

    def test_noise_model_is_applied(self):
        noise_model = NoiseModel()
        noise_model.add_all_qubit_readout_error(ReadoutError([[0, 1], [1, 0]]))
        simulator = QiskitSimulator(shots=100, noise_model=noise_model)

        result = simulator.submit_circuits({"c0": _IDENTITY_QASM})

        assert result.results[0]["results"] == {"1": 100}

    def test_circuits_are_transpiled_to_the_backend(self):
        """``h`` runs as noisy ``sx`` pulses only if transpiled to the backend's basis."""
        backend = _calibrated_backend(
            1,
            _gate("sx", (0,), error=0.5),
            _gate("rz", (0,)),
        )
        simulator = QiskitSimulator(
            shots=200, qiskit_backend=backend, optimization_level=0, simulation_seed=3
        )
        qasm = self._create_qasm_circuit(n_qubits=1).replace(
            "h q[0];", "h q[0];\n        h q[0];"
        )

        counts = simulator.submit_circuits({"c0": qasm}).results[0]["results"]

        assert counts.get("1", 0) > 0

    @pytest.mark.parametrize(
        "seed, parallel_experiments",
        [(None, 2), (42, 1)],
        ids=["parallel_unseeded", "seeded_serial"],
    )
    def test_no_nondeterminism_warning_without_both_seed_and_parallelism(
        self, mocker, seed, parallel_experiments
    ):
        simulator = QiskitSimulator(shots=100, simulation_seed=seed)
        self._setup_mock_aer_simulator(
            mocker,
            metadata={
                "parallel_experiments": parallel_experiments,
                "omp_nested": False,
            },
        )
        self._setup_mock_transpile(mocker, num_circuits=2)
        mock_logger = mocker.patch("divi.backends.runners._qiskit.logger")

        qasm = self._create_qasm_circuit()
        simulator.submit_circuits({"c0": qasm, "c1": qasm})

        mock_logger.warning.assert_not_called()

    def test_submit_circuits_with_explicit_backend(self, mocker):
        """Test submit_circuits with explicit backend provided (line 194)."""
        backend = FakeQuitoV2()
        simulator = QiskitSimulator(qiskit_backend=backend, shots=100)

        circuits = {"test_circuit": self._create_qasm_circuit()}

        mock_from_backend = self._setup_mock_aer_simulator(
            mocker, use_from_backend=True
        )[0]
        self._setup_mock_transpile(mocker)

        result = simulator.submit_circuits(circuits)

        assert isinstance(result, ExecutionResult)
        assert result.results is not None
        assert len(result.results) == 1
        assert result.results[0]["label"] == "test_circuit"
        # Verify from_backend was called (line 200)
        mock_from_backend.assert_called_once_with(backend)

    def test_submit_circuits_non_deterministic_batch_execution(self, mocker):
        """Test non-deterministic batch execution path (lines 221-244)."""
        simulator = QiskitSimulator(shots=100, _deterministic_execution=False)

        qasm1 = self._create_qasm_circuit()
        qasm2 = self._create_qasm_circuit().replace("h q[0];", "x q[0];")
        circuits = {"circuit1": qasm1, "circuit2": qasm2}

        mock_aer = self._setup_mock_aer_simulator(
            mocker,
            counts=[
                {"0": 50, "1": 50},  # For circuit1
                {"0": 30, "1": 70},  # For circuit2
            ],
            metadata={"parallel_experiments": 2, "omp_nested": False},
        )
        self._setup_mock_transpile(mocker, num_circuits=2)

        result = simulator.submit_circuits(circuits)

        assert isinstance(result, ExecutionResult)
        assert result.results is not None
        assert len(result.results) == 2
        assert result.results[0]["label"] == "circuit1"
        assert result.results[1]["label"] == "circuit2"
        # Verify batch execution was used (not deterministic)
        assert mock_aer.run.called

    def test_submit_circuits_non_deterministic_with_seed_warns(self, mocker):
        """Test that warning is logged when parallel execution detected with seed (lines 230-236)."""
        simulator = QiskitSimulator(
            shots=100, simulation_seed=42, _deterministic_execution=False
        )

        qasm = self._create_qasm_circuit()
        circuits = {"circuit1": qasm, "circuit2": qasm}

        self._setup_mock_aer_simulator(
            mocker,
            metadata={"parallel_experiments": 2, "omp_nested": True},
        )
        self._setup_mock_transpile(mocker, num_circuits=2)

        mock_logger = mocker.patch("divi.backends.runners._qiskit.logger")

        simulator.submit_circuits(circuits)

        # A warning about parallel execution affecting determinism should be logged
        mock_logger.warning.assert_called_once()
        warning_msg = str(mock_logger.warning.call_args[0][0]).lower()
        assert "parallel" in warning_msg
        assert "not be deterministic" in warning_msg

    def test_shot_group_runs_warn_about_parallel_nondeterminism(self, mocker):
        simulator = QiskitSimulator(shots=100, simulation_seed=42)
        qasm = self._create_qasm_circuit()
        self._setup_mock_aer_simulator(
            mocker, metadata={"parallel_experiments": 2, "omp_nested": False}
        )
        self._setup_mock_transpile(mocker, num_circuits=2)
        mock_logger = mocker.patch("divi.backends.runners._qiskit.logger")

        simulator.submit_circuits(
            {"c0": qasm, "c1": qasm}, shot_groups=[[0, 1, 50], [1, 2, 80]]
        )

        assert mock_logger.warning.call_count == 2

    def test_submit_circuits_deterministic_with_backend(self, mocker):
        """Test deterministic execution with backend (line 147)."""
        backend = FakeQuitoV2()
        simulator = QiskitSimulator(
            qiskit_backend=backend,
            shots=100,
            _deterministic_execution=True,
            simulation_seed=42,
        )

        circuits = {"test_circuit": self._create_qasm_circuit()}

        mock_from_backend = self._setup_mock_aer_simulator(
            mocker, use_from_backend=True
        )[0]
        self._setup_mock_transpile(mocker)

        result = simulator.submit_circuits(circuits)

        assert isinstance(result, ExecutionResult)
        assert result.results is not None
        assert len(result.results) == 1
        assert result.results[0]["label"] == "test_circuit"
        # One simulator to transpile against, plus a fresh one per circuit.
        assert mock_from_backend.call_count == 2

    def test_submit_circuits_deterministic_without_backend(self, mocker):
        """Test deterministic execution without backend (noise_model path)."""
        noise_model = NoiseModel()
        simulator = QiskitSimulator(
            noise_model=noise_model,
            shots=100,
            _deterministic_execution=True,
            simulation_seed=42,
        )

        circuits = {"test_circuit": self._create_qasm_circuit()}

        self._setup_mock_aer_simulator(mocker)
        self._setup_mock_transpile(mocker)

        result = simulator.submit_circuits(circuits)

        assert isinstance(result, ExecutionResult)
        assert result.results is not None
        assert len(result.results) == 1
        assert result.results[0]["label"] == "test_circuit"


def _setup_qiskit_contract_mocks(mocker):
    """Mock AerSimulator/transpile for shared CircuitRunner contract tests."""
    mock_aer = mocker.Mock()
    mock_aer.run.return_value.result.return_value = _mock_aer_result(
        mocker, {"0": 50, "1": 50}
    )
    mocker.patch(
        "divi.backends.runners._qiskit.AerSimulator",
        return_value=mock_aer,
    )

    def _transpile_side_effect(circuits, *args, **kwargs):
        return [
            QuantumCircuit.from_qasm_str(QASM_DEPTH_2 if i == 0 else QASM_DEPTH_3)
            for i in range(len(circuits))
        ]

    mocker.patch(
        "divi.backends.runners._qiskit.transpile",
        side_effect=_transpile_side_effect,
    )
    return mock_aer


def _contract_qiskit_runner(**kwargs):
    return QiskitSimulator(shots=CONTRACT_TEST_SHOTS, **kwargs)


class TestContracts(SyncRunnerContractsBase):
    """Shared :class:`~divi.backends.CircuitRunner` behavioural contracts."""

    @pytest.fixture(autouse=True)
    def _qiskit_contract_mocks(self, mocker):
        return _setup_qiskit_contract_mocks(mocker)

    @pytest.fixture()
    def contract_runner_disabled(self):
        return _contract_qiskit_runner(track_depth=False)

    @pytest.fixture()
    def contract_runner_enabled(self):
        return _contract_qiskit_runner(track_depth=True)

    @pytest.fixture()
    def contract_runner_default(self):
        return _contract_qiskit_runner()


class TestExpvalSubmission:
    """Tests for QiskitSimulator expectation value estimation."""

    QASM_2Q = (
        'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncreg c[2];\n'
        "h q[0];\ncx q[0],q[1];\nmeasure q[0] -> c[0];\nmeasure q[1] -> c[1];\n"
    )

    QASM_1Q_PLUS = (
        'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\ncreg c[1];\n'
        "h q[0];\nmeasure q[0] -> c[0];\n"
    )

    def test_expval_known_value(self):
        """H|0> = |+>, so <Z> = 0 and <X> = 1 on a single qubit."""
        sim = QiskitSimulator(shots=5000)
        result = sim.submit_circuits({"c0": self.QASM_1Q_PLUS}, ham_ops="Z;X")
        assert [entry["label"] for entry in result.results] == ["c0"]
        expvals = result.results[0]["results"]
        assert expvals == {
            "Z": pytest.approx(0.0, abs=1e-10),
            "X": pytest.approx(1.0, abs=1e-10),
        }
        assert all(isinstance(value, float) for value in expvals.values())

    def test_expval_bell_state_zz(self):
        """Bell state |00>+|11>: <ZZ> = 1, <ZI> = 0."""
        sim = QiskitSimulator(shots=5000)
        result = sim.submit_circuits({"c0": self.QASM_2Q}, ham_ops="ZZ;ZI")
        expvals = result.results[0]["results"]
        assert expvals["ZZ"] == pytest.approx(1.0, abs=1e-10)
        assert expvals["ZI"] == pytest.approx(0.0, abs=1e-10)

    def test_short_observables_act_on_first_qubits(self):
        sim = QiskitSimulator(shots=5000)
        with pytest.warns(UserWarning, match=exact_match(padding_warning("Z", "ZI"))):
            result = sim.submit_circuits({"c0": QASM_X_ON_FIRST_QUBIT}, ham_ops="Z")
        assert result.results[0]["results"] == {"ZI": pytest.approx(-1.0)}

    def test_sampling_not_affected(self, mocker):
        """Sampling path unchanged when ham_ops=None."""
        sim = QiskitSimulator(shots=100)

        mock_aer = mocker.Mock()
        mock_aer.run.return_value.result.return_value = _mock_aer_result(
            mocker, {"00": 50, "11": 50}
        )
        mocker.patch(
            "divi.backends.runners._qiskit.AerSimulator",
            return_value=mock_aer,
        )
        mocker.patch(
            "divi.backends.runners._qiskit.transpile",
            return_value=[QuantumCircuit(2)],
        )

        result = sim.submit_circuits({"c0": self.QASM_2Q})
        assert result.results[0]["results"] == {"00": 50, "11": 50}

    def test_each_observable_group_is_padded_to_its_own_circuits(self):
        sim = QiskitSimulator(shots=5000)
        with pytest.warns(UserWarning, match=exact_match(padding_warning("Z", "ZI"))):
            result = sim.submit_circuits(
                {"c0": self.QASM_1Q_PLUS, "c1": QASM_X_ON_FIRST_QUBIT},
                ham_ops="X|Z",
                circuit_ham_map=[[0, 1], [1, 2]],
            )
        assert result.results[0]["results"] == {"X": pytest.approx(1.0, abs=1e-10)}
        assert result.results[1]["results"] == {"ZI": pytest.approx(-1.0)}

    def test_prepare_expval_circuit_strips_measurements(self):
        """_prepare_expval_circuit removes measurements and adds save instructions."""
        qc = QuantumCircuit.from_qasm_str(self.QASM_2Q)
        prepared = QiskitSimulator._prepare_expval_circuit(qc, ["ZI", "IZ"])
        # No measure gates
        op_names = [inst.operation.name for inst in prepared.data]
        assert "measure" not in op_names
        # Has save_expectation_value instructions
        assert "save_expval" in op_names or any("save" in name for name in op_names)

    def test_prepare_expval_circuit_preserves_gates(self):
        """Gate instructions survive measurement stripping."""
        qc = QuantumCircuit.from_qasm_str(self.QASM_2Q)
        prepared = QiskitSimulator._prepare_expval_circuit(qc, ["ZZ"])
        op_names = [inst.operation.name for inst in prepared.data]
        assert "h" in op_names
        assert "cx" in op_names


class TestQiskitSimulatorRuntimeEstimation:
    """Tests for QiskitSimulator runtime estimation methods."""

    def test_estimate_run_time_single_circuit(self):
        """The estimate sums the backend's durations, in seconds, along the
        longest path of the circuit transpiled with the given options."""
        backend = _calibrated_backend(
            1, _gate("x", (0,), length_ns=35.0), readout_length_ns=1000.0
        )

        estimated_time = QiskitSimulator.estimate_run_time_single_circuit(
            _DOUBLE_X_QASM, qiskit_backend=backend, optimization_level=0
        )

        assert estimated_time == pytest.approx(2 * 35e-9 + 1000e-9)

    def test_estimate_run_time_single_circuit_auto_backend(self):
        """'auto' estimates on the newest fake backend wide enough for the circuit."""
        estimate = partial(
            QiskitSimulator.estimate_run_time_single_circuit,
            _DOUBLE_X_QASM,
            seed_transpiler=0,
        )

        assert estimate(qiskit_backend="auto") == estimate(
            qiskit_backend=FAKE_BACKENDS[5][-1]()
        )

    def test_estimate_run_time_single_circuit_rejects_circuits_too_wide(self):
        with pytest.raises(ValueError, match=exact_match(_NO_FAKE_BACKEND_MESSAGE)):
            QiskitSimulator.estimate_run_time_single_circuit(
                _WIDE_QASM, qiskit_backend="auto"
            )

    def test_barriers_take_no_time_and_do_not_warn(self):
        backend = _calibrated_backend(1, _gate("x", (0,), length_ns=35.0))
        qasm = _DOUBLE_X_QASM.replace("x q[0];\nx q[0];", "x q[0];\nbarrier q[0];")

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            estimated_time = QiskitSimulator.estimate_run_time_single_circuit(
                qasm, qiskit_backend=backend, optimization_level=0
            )

        assert estimated_time == pytest.approx(35e-9)

    def test_instructions_without_a_duration_warn(self):
        backend = _calibrated_backend(1, _gate("x", (0,)))

        with pytest.warns(
            UserWarning, match=exact_match("Instruction duration not found: x")
        ):
            QiskitSimulator.estimate_run_time_single_circuit(
                _DOUBLE_X_QASM, qiskit_backend=backend, optimization_level=0
            )

    def test_a_backend_without_a_target_is_rejected(self, mocker):
        mocker.patch(
            "divi.backends.runners._qiskit.transpile",
            side_effect=lambda circuit, *args, **kwargs: circuit,
        )

        with pytest.raises(RuntimeError, match="has no transpiler target"):
            QiskitSimulator.estimate_run_time_single_circuit(
                _DOUBLE_X_QASM, qiskit_backend=mocker.Mock(target=None)
            )

    @pytest.mark.parametrize(
        "durations, qpus, expected",
        [
            ([3.0, 5.0, 3.0, 4.0, 3.0], {"n_qpus": 2}, 10.0),
            ([2.0, 7.0, 1.0], {"n_qpus": 5}, 7.0),
            ([], {"n_qpus": 5}, 0.0),
            ([1.0] * 6, {}, 2.0),
        ],
        ids=[
            "longest_first_scheduling",
            "one_circuit_per_qpu",
            "empty",
            "defaults_to_five_qpus",
        ],
    )
    def test_estimate_run_time_batch_schedules_precomputed_durations(
        self, durations, qpus, expected
    ):
        assert QiskitSimulator.estimate_run_time_batch(
            precomputed_durations=durations, **qpus
        ) == pytest.approx(expected)

    def test_estimate_run_time_batch_requires_an_input(self):
        message = (
            "estimate_run_time_batch requires either ``circuits`` or "
            "``precomputed_durations`` to be provided."
        )
        with pytest.raises(ValueError, match=exact_match(message)):
            QiskitSimulator.estimate_run_time_batch()

    def test_estimate_run_time_batch_schedules_auto_estimates(self, mocker):
        """Circuit estimates are computed on 'auto' backends with the given
        transpile options."""
        mocker.patch(
            "divi.backends.runners._qiskit.Pool", return_value=_InProcessPool()
        )
        circuits = [
            _DOUBLE_X_QASM,
            _DOUBLE_X_QASM.replace("x q[0];\n", "", 1),
            QASM_X_ON_FIRST_QUBIT,
        ]
        singles = [
            QiskitSimulator.estimate_run_time_single_circuit(
                qasm, "auto", optimization_level=0, seed_transpiler=0
            )
            for qasm in circuits
        ]

        estimate = QiskitSimulator.estimate_run_time_batch(
            circuits=circuits, n_qpus=2, optimization_level=0, seed_transpiler=0
        )

        assert estimate == pytest.approx(
            QiskitSimulator.estimate_run_time_batch(
                precomputed_durations=singles, n_qpus=2
            )
        )

    def test_estimate_run_time_batch_pins_pool_processes(self, mocker):
        """``estimate_run_time_batch`` must construct
        ``multiprocessing.Pool`` with ``processes=_default_n_processes()``
        instead of relying on ``os.cpu_count()`` defaults — otherwise it
        oversubscribes when called from inside a worker thread.
        """
        # Capture how Pool was instantiated.
        pool_kwargs: dict = {}
        mock_pool_cm = mocker.MagicMock()
        mock_pool_cm.__enter__.return_value.map.return_value = [0.1, 0.2, 0.3]
        mock_pool_cm.__exit__.return_value = False

        def fake_pool(*args, **kwargs):
            pool_kwargs.update(kwargs)
            return mock_pool_cm

        mocker.patch("divi.backends.runners._qiskit.Pool", side_effect=fake_pool)
        mocker.patch(
            "divi.backends.runners._qiskit._default_n_processes",
            return_value=7,
        )

        QiskitSimulator.estimate_run_time_batch(circuits=["a", "b", "c"], n_qpus=2)

        assert pool_kwargs.get("processes") == 7


class TestDefaultNProcesses:
    """Tests for _default_n_processes CPU-count logic."""

    def test_non_main_thread_returns_two(self, mocker):
        """Running in a worker thread limits to 2 cores."""
        mock_thread = mocker.Mock()
        mocker.patch(
            "divi.backends.runners._qiskit.threading.current_thread",
            return_value=mock_thread,
        )
        mocker.patch(
            "divi.backends.runners._qiskit.threading.main_thread",
            return_value=mocker.Mock(),
        )
        assert _default_n_processes() == 2

    def test_non_main_process_returns_two(self, mocker):
        """Running in a subprocess limits to 2 cores."""
        mock_process = mocker.Mock()
        mock_process.name = "SpawnProcess-1"
        mocker.patch(
            "divi.backends.runners._qiskit.current_process",
            return_value=mock_process,
        )
        assert _default_n_processes() == 2

    @pytest.mark.parametrize(
        "cpu_count, expected",
        [
            (2, 2),  # max(2, 2-1) = 2
            (3, 2),  # max(2, 3-1) = 2
            (4, 3),  # max(2, 4-1) = 3
            (8, 7),  # 8-1 = 7 (medium)
            (16, 15),  # 16-1 = 15 (medium)
            (32, 16),  # min(16, 32*0.75=24) = 16 (large, capped)
            (20, 15),  # min(16, 20*0.75=15) = 15 (large)
            (17, 12),  # min(16, 17*0.75=12.75) = 12 (smallest large)
        ],
    )
    def test_cpu_count_scaling(self, mocker, cpu_count, expected):
        """Correct scaling: small <= 4, medium <= 16, large > 16."""
        mocker.patch(
            "divi.backends.runners._qiskit.os.cpu_count", return_value=cpu_count
        )
        assert _default_n_processes() == expected

    def test_cpu_count_none_falls_back_to_four(self, mocker):
        """When os.cpu_count() returns None, defaults to cpu_count=4 logic."""
        mocker.patch("divi.backends.runners._qiskit.os.cpu_count", return_value=None)
        # cpu_count=4 -> max(2, 4-1) = 3
        assert _default_n_processes() == 3


_IDENTITY_QASM = (
    "OPENQASM 2.0;\n"
    'include "qelib1.inc";\n'
    "qreg q[1];\n"
    "creg c[1];\n"
    "measure q[0] -> c[0];\n"
)


@pytest.mark.parametrize(
    "sim_kwargs, submit_kwargs",
    [
        ({}, {}),
        ({}, {"ham_ops": "Z"}),
        ({}, {"shot_groups": [[0, 1, 50], [1, 2, 80]]}),
        ({"_deterministic_execution": True}, {}),
    ],
    ids=["sampling", "expval", "shot_groups", "deterministic"],
)
def test_submission_reports_aers_time_taken(mocker, sim_kwargs, submit_kwargs):
    # Compare with Aer's own figure; a coarse clock (e.g. on Windows) can
    # report 0.0 for circuits this small.
    spy = mocker.spy(AerJob, "result")
    sim = QiskitSimulator(shots=100, **sim_kwargs)
    circuits = {"c0": _IDENTITY_QASM, "c1": _IDENTITY_QASM}

    run_time = sim.submit_circuits(circuits, **submit_kwargs).run_time

    assert spy.spy_return_list
    assert run_time == sum(result.time_taken for result in spy.spy_return_list)


def test_both_provided_raises_value_error():
    """Spec: ham_ops + shot_groups together is rejected at the API boundary."""
    sim = QiskitSimulator(shots=100)
    with pytest.raises(ValueError, match=exact_match(SHOT_GROUPS_WITH_HAM_OPS_MESSAGE)):
        sim.submit_circuits(
            {"c0": "OPENQASM 2.0;\nqreg q[1];\n"},
            ham_ops="Z",
            shot_groups=[[0, 1, 100]],
        )


def test_deterministic_path_honors_per_group_shots():
    """Spec: shot_groups is respected even with _deterministic_execution=True.

    Without this, the deterministic short-circuit would silently use
    self.shots for every circuit and ignore the per-group allocation."""
    sim = QiskitSimulator(shots=999, _deterministic_execution=True)
    circuits = {
        "c0": _IDENTITY_QASM,
        "c1": _IDENTITY_QASM,
        "c2": _IDENTITY_QASM,
    }
    result = sim.submit_circuits(circuits, shot_groups=[[0, 1, 50], [1, 3, 200]])
    per_circuit_totals = [sum(r["results"].values()) for r in result.results]
    assert per_circuit_totals == [50, 200, 200]


def test_returned_counts_sum_matches_per_group_shots():
    """Spec: QiskitSimulator runs each range with the assigned shot count."""
    sim = QiskitSimulator(shots=100)
    circuits = {
        "c0": _IDENTITY_QASM,
        "c1": _IDENTITY_QASM,
        "c2": _IDENTITY_QASM,
    }
    result = sim.submit_circuits(circuits, shot_groups=[[0, 1, 50], [1, 3, 200]])
    per_circuit_totals = [sum(r["results"].values()) for r in result.results]
    assert per_circuit_totals == [50, 200, 200]


def test_partial_coverage_raises():
    """Implementation detail: shot_groups must cover every circuit."""
    sim = QiskitSimulator(shots=100)
    circuits = {
        "c0": _IDENTITY_QASM,
        "c1": _IDENTITY_QASM,
        "c2": _IDENTITY_QASM,
    }
    with pytest.raises(ValueError, match=exact_match(uncovered_circuits_message([2]))):
        sim.submit_circuits(circuits, shot_groups=[[0, 2, 100]])


class TestQiskitSimulatorShotGroupsBatching:
    """Spec: one Aer run per distinct shot count."""

    def test_single_aer_run_when_all_shots_equal(self, mocker):
        sim = QiskitSimulator(shots=100)
        circuits = {f"c{i}": _IDENTITY_QASM for i in range(4)}
        spy = mocker.spy(AerSimulator, "run")
        sim.submit_circuits(
            circuits,
            shot_groups=[[0, 1, 50], [1, 2, 50], [2, 3, 50], [3, 4, 50]],
        )
        assert spy.call_count == 1

    def test_distinct_shot_counts_get_distinct_runs(self, mocker):
        sim = QiskitSimulator(shots=100)
        circuits = {f"c{i}": _IDENTITY_QASM for i in range(3)}
        spy = mocker.spy(AerSimulator, "run")
        sim.submit_circuits(
            circuits,
            shot_groups=[[0, 1, 50], [1, 2, 100], [2, 3, 50]],
        )
        assert spy.call_count == 2

    def test_results_returned_in_original_circuit_order(self):
        sim = QiskitSimulator(shots=100)
        circuits = {f"c{i}": _IDENTITY_QASM for i in range(4)}
        result = sim.submit_circuits(
            circuits,
            shot_groups=[[0, 1, 50], [1, 2, 200], [2, 3, 50], [3, 4, 200]],
        )
        labels = [r["label"] for r in result.results]
        per_circuit_totals = [sum(r["results"].values()) for r in result.results]
        assert labels == ["c0", "c1", "c2", "c3"]
        assert per_circuit_totals == [50, 200, 50, 200]
