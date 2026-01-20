# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import warnings

import pytest
from qiskit import QuantumCircuit
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider import FakeQuitoV2

from divi.backends import ExecutionResult
from divi.backends._parallel_simulator import (
    FAKE_BACKENDS,
    ParallelSimulator,
    _find_best_fake_backend,
)


class TestFindBestFakeBackend:
    """Tests for _find_best_fake_backend function."""

    def test_find_backend_for_small_circuit(self):
        """Test finding backend for a small circuit."""
        circuit = QuantumCircuit(3)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.measure_all()

        result = _find_best_fake_backend(circuit)
        assert result is not None
        # Should find a backend with at least 3 qubits (5-qubit backend)
        assert 5 in FAKE_BACKENDS
        assert result == FAKE_BACKENDS[5]

    def test_find_backend_for_large_circuit(self):
        """Test finding backend for a large circuit."""
        circuit = QuantumCircuit(20)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.measure_all()

        result = _find_best_fake_backend(circuit)
        assert result is not None
        # Should find a backend with at least 20 qubits
        assert result == FAKE_BACKENDS[20]

    def test_find_backend_edge_case_exceeds_all(self):
        """Test edge case where circuit exceeds all fake backend sizes (lines 56-58)."""
        # Create a circuit with more qubits than the largest fake backend (27)
        circuit = QuantumCircuit(100)
        circuit.h(0)
        circuit.measure_all()

        result = _find_best_fake_backend(circuit)
        # Should return None when circuit exceeds all backend sizes
        assert result is None


class TestParallelSimulatorInit:
    """Tests for ParallelSimulator initialization."""

    def test_init_with_backend_and_noise_model_warns(self):
        """Test that warning is issued when both backend and noise_model are provided (line 88)."""
        backend = FakeQuitoV2()
        noise_model = NoiseModel()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            simulator = ParallelSimulator(
                qiskit_backend=backend, noise_model=noise_model
            )

            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)
            assert "Both `qiskit_backend` and `noise_model`" in str(w[0].message)
            assert "`noise_model` will be ignored" in str(w[0].message)

        # Verify simulator was still created
        assert simulator.qiskit_backend == backend
        assert simulator.noise_model == noise_model

    def test_init_with_backend_only(self):
        """Test initialization with backend only."""
        backend = FakeQuitoV2()
        simulator = ParallelSimulator(qiskit_backend=backend)

        assert simulator.qiskit_backend == backend
        assert simulator.noise_model is None

    def test_init_with_noise_model_only(self):
        """Test initialization with noise model only."""
        noise_model = NoiseModel()
        simulator = ParallelSimulator(noise_model=noise_model)

        assert simulator.qiskit_backend is None
        assert simulator.noise_model == noise_model


class TestParallelSimulatorProperties:
    """Tests for ParallelSimulator properties and methods."""

    def test_set_seed(self):
        """Test set_seed method (line 107)."""
        simulator = ParallelSimulator()
        assert simulator.simulation_seed is None

        simulator.set_seed(42)
        assert simulator.simulation_seed == 42

        simulator.set_seed(100)
        assert simulator.simulation_seed == 100

    def test_supports_expval(self):
        """Test supports_expval property (line 114)."""
        simulator = ParallelSimulator()
        assert simulator.supports_expval is False

    def test_is_async(self):
        """Test is_async property (line 121)."""
        simulator = ParallelSimulator()
        assert simulator.is_async is False


class TestParallelSimulatorSubmitCircuits:
    """Tests for ParallelSimulator.submit_circuits method."""

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
        mock_result = mocker.Mock()
        if counts is None:
            counts = {"0": 50, "1": 50}
        if isinstance(counts, list):
            mock_result.get_counts.side_effect = counts
        else:
            mock_result.get_counts.return_value = counts
        if metadata is None:
            metadata = {"parallel_experiments": 1, "omp_nested": False}
        mock_result.metadata = metadata
        mock_aer.run.return_value.result.return_value = mock_result

        if use_from_backend:
            return (
                mocker.patch(
                    "divi.backends._parallel_simulator.AerSimulator.from_backend",
                    return_value=mock_aer,
                ),
                mock_aer,
            )
        else:
            mocker.patch(
                "divi.backends._parallel_simulator.AerSimulator",
                return_value=mock_aer,
            )
            return mock_aer

    def _setup_mock_transpile(self, mocker, n_qubits=2, num_circuits=1):
        """Helper to set up mock transpile."""
        mock_transpiled = QuantumCircuit(n_qubits)
        mocker.patch(
            "divi.backends._parallel_simulator.transpile",
            return_value=[mock_transpiled] * num_circuits,
        )

    def test_submit_circuits_with_auto_backend(self, mocker):
        """Test submit_circuits with 'auto' backend selection (lines 191-192)."""
        simulator = ParallelSimulator(qiskit_backend="auto", shots=100)

        qasm = self._create_qasm_circuit(n_qubits=3)
        qasm = qasm.replace("h q[0];", "h q[0];\n        cx q[0], q[1];")
        circuits = {"test_circuit": qasm}

        # Mock the fake backend to avoid actual backend creation
        mock_backend_class = mocker.Mock()
        mocker.patch(
            "divi.backends._parallel_simulator._find_best_fake_backend",
            return_value=[mock_backend_class],
        )

        mock_from_backend = self._setup_mock_aer_simulator(
            mocker, use_from_backend=True
        )[0]
        self._setup_mock_transpile(mocker, n_qubits=3)

        result = simulator.submit_circuits(circuits)

        assert isinstance(result, ExecutionResult)
        assert result.results is not None
        assert len(result.results) == 1
        assert result.results[0]["label"] == "test_circuit"
        assert "results" in result.results[0]
        # Verify from_backend was called (line 200)
        mock_from_backend.assert_called_once()

    def test_submit_circuits_with_explicit_backend(self, mocker):
        """Test submit_circuits with explicit backend provided (line 194)."""
        backend = FakeQuitoV2()
        simulator = ParallelSimulator(qiskit_backend=backend, shots=100)

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
        simulator = ParallelSimulator(shots=100, _deterministic_execution=False)

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
        simulator = ParallelSimulator(
            shots=100, simulation_seed=42, _deterministic_execution=False
        )

        qasm = self._create_qasm_circuit()
        circuits = {"circuit1": qasm, "circuit2": qasm}

        self._setup_mock_aer_simulator(
            mocker,
            metadata={"parallel_experiments": 2, "omp_nested": True},
        )
        self._setup_mock_transpile(mocker, num_circuits=2)

        mock_logger = mocker.patch("divi.backends._parallel_simulator.logger")

        simulator.submit_circuits(circuits)

        # Verify warning was logged about non-determinism
        mock_logger.warning.assert_called_once()
        warning_msg = str(mock_logger.warning.call_args[0][0])
        assert "Parallel execution detected" in warning_msg
        assert "parallel_experiments=2" in warning_msg
        assert "omp_nested=True" in warning_msg
        assert "may not be deterministic" in warning_msg

    def test_submit_circuits_deterministic_with_backend(self, mocker):
        """Test deterministic execution with backend (line 147)."""
        backend = FakeQuitoV2()
        simulator = ParallelSimulator(
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
        # Verify deterministic execution path was used (from_backend called in _execute_circuits_deterministically)
        # It's called once per circuit in deterministic mode
        assert mock_from_backend.call_count >= 1

    def test_submit_circuits_deterministic_without_backend(self, mocker):
        """Test deterministic execution without backend (noise_model path)."""
        noise_model = NoiseModel()
        simulator = ParallelSimulator(
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


class TestParallelSimulatorRuntimeEstimation:
    """Tests for ParallelSimulator runtime estimation methods."""

    def test_estimate_run_time_single_circuit(self, mocker):
        """Test estimate_run_time_single_circuit method."""
        # Mock backend with instruction durations
        mock_backend = mocker.Mock()
        mock_target = mocker.Mock()
        mock_durations = mocker.Mock()

        # Mock durations.get() method to return duration in seconds
        def mock_get(inst_name, qubits, unit="s"):
            durations_map = {
                ("h", (0,)): 1.6e-7,
                ("cx", (0, 1)): 3.2e-7,
                ("measure", (0,)): 5.0e-7,
                ("measure", (1,)): 5.0e-7,
            }
            return durations_map.get((inst_name, qubits), 0.0)

        mock_durations.get = mock_get
        mock_target.durations.return_value = mock_durations
        mock_backend.target = mock_target

        # Create a circuit matching the mocked durations
        qasm = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        h q[0];
        cx q[0], q[1];
        measure q[0] -> c[0];
        measure q[1] -> c[1];
        """

        # Mock transpile to return a circuit that preserves structure
        # We need a real circuit for circuit_to_dag to work
        transpiled_circuit = QuantumCircuit.from_qasm_str(qasm)
        mocker.patch(
            "divi.backends._parallel_simulator.transpile",
            return_value=transpiled_circuit,
        )

        # We also need to mock _find_best_fake_backend if we were using "auto",
        # but here we pass explicit backend

        estimated_time = ParallelSimulator.estimate_run_time_single_circuit(
            qasm, qiskit_backend=mock_backend
        )

        # Expected time: h(0) + cx(0,1) + max(measure(0), measure(1))?
        # The logic in ParallelSimulator sums up durations of longest path?
        # Code: for node in dag.longest_path(): total += duration
        # Longest path in this circuit:
        # q[0]: h -> cx -> measure
        # q[1]: cx -> measure
        # Path 0: h(1.6) + cx(3.2) + measure(5.0) = 9.8e-7
        # Path 1: cx(3.2) + measure(5.0) = 8.2e-7
        # So it should be roughly 9.8e-7

        assert estimated_time == pytest.approx(9.8e-7)

    def test_estimate_run_time_single_circuit_auto_backend(self, mocker):
        """Test estimate_run_time_single_circuit with 'auto' backend."""
        qasm = 'OPENQASM 2.0; include "qelib1.inc"; qreg q[1]; h q[0];'

        mock_backend_cls = mocker.Mock()
        mock_backend_instance = mock_backend_cls.return_value
        mock_target = mocker.Mock()
        mock_durations = mocker.Mock()

        # Mock durations.get() method to return duration in seconds
        def mock_get(inst_name, qubits, unit="s"):
            if inst_name == "h" and qubits == (0,):
                return 1.0
            return 0.0

        mock_durations.get = mock_get
        mock_target.durations.return_value = mock_durations
        mock_backend_instance.target = mock_target

        mocker.patch(
            "divi.backends._parallel_simulator._find_best_fake_backend",
            return_value=[mock_backend_cls],
        )

        transpiled_circuit = QuantumCircuit.from_qasm_str(qasm)
        mocker.patch(
            "divi.backends._parallel_simulator.transpile",
            return_value=transpiled_circuit,
        )

        estimated_time = ParallelSimulator.estimate_run_time_single_circuit(
            qasm, qiskit_backend="auto"
        )

        assert estimated_time == 1.0
