# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pennylane as qml
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

from divi.qprog import CustomVQA
from divi.qprog.checkpointing import CheckpointConfig
from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer
from tests.conftest import CHECKPOINTING_OPTIMIZERS
from tests.qprog.qprog_contracts import (
    OPTIMIZERS_TO_TEST,
    verify_correct_circuit_count,
    verify_metacircuit_dict,
)

pytestmark = pytest.mark.algo


@pytest.fixture
def simple_quantum_script():
    """Fixture for a simple parameterized QuantumScript."""
    ops = [qml.RX(0.0, wires=0), qml.RZ(0.0, wires=0)]
    measurements = [qml.expval(qml.Z(0))]
    return qml.tape.QuantumScript(ops=ops, measurements=measurements)


@pytest.fixture
def qiskit_circuit_with_measurements():
    """Fixture for a Qiskit QuantumCircuit with measurements."""
    theta = Parameter("theta")
    phi = Parameter("phi")
    qc = QuantumCircuit(1, 1)
    qc.rx(theta, 0)
    qc.rz(phi, 0)
    qc.measure(0, 0)
    return qc


@pytest.fixture
def qiskit_circuit_no_measurements():
    """Fixture for a Qiskit QuantumCircuit without measurements."""
    theta = Parameter("theta")
    phi = Parameter("phi")
    qc = QuantumCircuit(1)
    qc.rx(theta, 0)
    qc.rz(phi, 0)
    return qc


@pytest.fixture
def qiskit_circuit_multi_qubit_all_measured():
    """Fixture for a multi-qubit Qiskit circuit with all qubits measured."""
    theta = Parameter("theta")
    phi = Parameter("phi")
    alpha = Parameter("alpha")
    qc = QuantumCircuit(3, 3)
    qc.rx(theta, 0)
    qc.ry(phi, 1)
    qc.rz(alpha, 2)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.measure(0, 0)
    qc.measure(1, 1)
    qc.measure(2, 2)
    return qc


@pytest.fixture
def qiskit_circuit_multi_qubit_partial_measured():
    """Fixture for a multi-qubit Qiskit circuit with only some qubits measured."""
    theta = Parameter("theta")
    phi = Parameter("phi")
    alpha = Parameter("alpha")
    qc = QuantumCircuit(3, 2)
    qc.rx(theta, 0)
    qc.ry(phi, 1)
    qc.rz(alpha, 2)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.measure(0, 0)  # Only qubit 0 measured
    qc.measure(2, 1)  # Only qubit 2 measured (qubit 1 not measured)
    return qc


@pytest.fixture
def qiskit_circuit_complex_gates():
    """Fixture for a Qiskit circuit with complex gate sequences."""
    theta = Parameter("theta")
    phi = Parameter("phi")
    beta = Parameter("beta")
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.rx(theta, 0)
    qc.ry(phi, 1)
    qc.cx(0, 1)
    qc.rz(beta, 0)
    qc.measure(0, 0)
    qc.measure(1, 1)
    return qc


class TestInitialization:
    """Test suite for CustomVQA initialization."""

    @pytest.mark.parametrize(
        "circuit_fixture,expected_n_qubits",
        [
            ("simple_quantum_script", 1),
            ("qiskit_circuit_with_measurements", 1),
        ],
    )
    def test_basic_initialization(
        self, circuit_fixture, expected_n_qubits, dummy_simulator, request
    ):
        """Test basic initialization with both QuantumScript and Qiskit inputs."""
        circuit = request.getfixturevalue(circuit_fixture)
        program = CustomVQA(qscript=circuit, backend=dummy_simulator)

        assert isinstance(program.qscript, qml.tape.QuantumScript)
        assert program.n_qubits == expected_n_qubits
        assert program.n_layers == 1
        assert program.n_params == 2
        assert program.param_shape == (2,)
        assert isinstance(program.cost_hamiltonian, qml.operation.Operator)
        verify_metacircuit_dict(program, ["cost_circuit"])

    @pytest.mark.parametrize(
        "param_shape,expected_shape",
        [
            (None, (2,)),
            (2, (2,)),
            ((2,), (2,)),
            ((1, 2), (1, 2)),
        ],
    )
    def test_param_shape_variations(
        self, param_shape, expected_shape, simple_quantum_script, dummy_simulator
    ):
        """Test initialization with various parameter shape formats."""
        program = CustomVQA(
            qscript=simple_quantum_script,
            param_shape=param_shape,
            backend=dummy_simulator,
        )

        assert program.param_shape == expected_shape
        assert program.n_params == 2

    def test_qiskit_parameter_names_preserved(
        self, qiskit_circuit_with_measurements, dummy_simulator
    ):
        """Test that Qiskit parameter names are preserved in sympy symbols."""
        program = CustomVQA(
            qscript=qiskit_circuit_with_measurements,
            backend=dummy_simulator,
        )

        symbols = program._param_symbols.flatten()
        symbol_names = [str(sym) for sym in symbols]
        assert "theta" in symbol_names
        assert "phi" in symbol_names

    def test_qiskit_circuit_no_measurements_warns(
        self, qiskit_circuit_no_measurements, dummy_simulator
    ):
        """Test that Qiskit circuit without measurements warns and defaults to all wires."""
        with pytest.warns(UserWarning, match="no measurement operations"):
            program = CustomVQA(
                qscript=qiskit_circuit_no_measurements,
                backend=dummy_simulator,
            )

        assert program.n_qubits == 1
        assert isinstance(program.cost_hamiltonian, qml.operation.Operator)


class TestErrorCases:
    """Test suite for error handling."""

    def test_invalid_input_type(self, dummy_simulator):
        """Test that invalid input type raises TypeError."""
        with pytest.raises(TypeError, match="must be a PennyLane QuantumScript"):
            CustomVQA(qscript="not a circuit", backend=dummy_simulator)

    def test_multiple_measurements_fails(self, dummy_simulator):
        """Test that QuantumScript with multiple measurements fails."""
        ops = [qml.RX(0.0, wires=0)]
        measurements = [qml.expval(qml.Z(0)), qml.expval(qml.Z(0))]
        qscript = qml.tape.QuantumScript(ops=ops, measurements=measurements)

        with pytest.raises(ValueError, match="exactly one measurement"):
            CustomVQA(qscript=qscript, backend=dummy_simulator)

    def test_no_measurement_fails(self, dummy_simulator):
        """Test that QuantumScript without measurement fails."""
        ops = [qml.RX(0.0, wires=0)]
        qscript = qml.tape.QuantumScript(ops=ops, measurements=[])

        with pytest.raises(ValueError, match="exactly one measurement"):
            CustomVQA(qscript=qscript, backend=dummy_simulator)

    def test_non_expval_measurement_fails(self, dummy_simulator):
        """Test that non-expectation-value measurement fails."""
        ops = [qml.RX(0.0, wires=0)]
        measurements = [qml.probs(wires=0)]
        qscript = qml.tape.QuantumScript(ops=ops, measurements=measurements)

        with pytest.raises(ValueError, match="expectation-value measurement"):
            CustomVQA(qscript=qscript, backend=dummy_simulator)

    def test_constant_only_hamiltonian_fails(self, dummy_simulator):
        """Test that constant-only Hamiltonian fails."""
        ops = [qml.RX(0.0, wires=0)]
        measurements = [qml.expval(qml.Identity(0) * 5.0)]
        qscript = qml.tape.QuantumScript(ops=ops, measurements=measurements)

        with pytest.raises(ValueError, match="only constant terms"):
            CustomVQA(qscript=qscript, backend=dummy_simulator)

    def test_no_trainable_parameters_fails(self, dummy_simulator):
        """Test that QuantumScript without trainable parameters fails."""
        # Create a script with no operations (empty circuit)
        measurements = [qml.expval(qml.Z(0))]
        qscript = qml.tape.QuantumScript(ops=[], measurements=measurements)

        with pytest.raises(ValueError, match="trainable parameters"):
            CustomVQA(qscript=qscript, backend=dummy_simulator)

    @pytest.mark.parametrize(
        "param_shape,error_match",
        [
            ((3,), "does not match"),
            ((-1,), "must be positive"),
            ((0,), "must be positive"),
        ],
    )
    def test_invalid_param_shape_fails(
        self, param_shape, error_match, simple_quantum_script, dummy_simulator
    ):
        """Test that invalid param_shape values raise appropriate errors."""
        with pytest.raises(ValueError, match=error_match):
            CustomVQA(
                qscript=simple_quantum_script,
                param_shape=param_shape,
                backend=dummy_simulator,
            )


class TestParameterHandling:
    """Test suite for parameter handling."""

    def test_initial_params_setter(self, simple_quantum_script, dummy_simulator):
        """Test setting initial parameters."""
        initial_params = np.array([[0.1, 0.2]])
        program = CustomVQA(
            qscript=simple_quantum_script,
            initial_params=initial_params,
            backend=dummy_simulator,
        )

        np.testing.assert_array_equal(program.curr_params, initial_params)

    def test_param_shape_inference(self, simple_quantum_script, dummy_simulator):
        """Test that param_shape is inferred when None."""
        program = CustomVQA(
            qscript=simple_quantum_script,
            param_shape=None,
            backend=dummy_simulator,
        )

        assert program.param_shape == (2,)

    def test_multiple_wire_observable(self, dummy_simulator):
        """Test with observable on multiple wires."""
        ops = [qml.RX(0.0, wires=0), qml.RY(0.0, wires=1)]
        measurements = [qml.expval(qml.Z(0) @ qml.Z(1))]
        qscript = qml.tape.QuantumScript(ops=ops, measurements=measurements)

        program = CustomVQA(qscript=qscript, backend=dummy_simulator)
        assert program.n_qubits == 2
        assert program.n_params == 2


class TestQiskitConversion:
    """Test suite for Qiskit QuantumCircuit conversion."""

    def test_qiskit_measurements_removed(
        self, qiskit_circuit_with_measurements, dummy_simulator
    ):
        """Test that Qiskit measurements are removed before conversion."""
        program = CustomVQA(
            qscript=qiskit_circuit_with_measurements,
            backend=dummy_simulator,
        )

        # The converted script should not have MidMeasureMP operations
        assert len(program.qscript.measurements) == 1
        assert hasattr(program.qscript.measurements[0], "obs")

    @pytest.mark.parametrize(
        "circuit_fixture,expected_n_qubits,expected_n_params,expected_measured_wires",
        [
            ("qiskit_circuit_multi_qubit_all_measured", 3, 3, {0, 1, 2}),
            ("qiskit_circuit_multi_qubit_partial_measured", 3, 3, {0, 2}),
            ("qiskit_circuit_complex_gates", 2, 3, {0, 1}),
        ],
    )
    def test_qiskit_measurement_mapping(
        self,
        circuit_fixture,
        expected_n_qubits,
        expected_n_params,
        expected_measured_wires,
        dummy_simulator,
        request,
    ):
        """Test that measured qubits are correctly mapped to observables."""
        circuit = request.getfixturevalue(circuit_fixture)
        program = CustomVQA(qscript=circuit, backend=dummy_simulator)

        assert program.n_qubits == expected_n_qubits
        assert program.n_params == expected_n_params
        assert isinstance(program.cost_hamiltonian, qml.operation.Operator)

        if len(expected_measured_wires) > 1:
            assert isinstance(program.cost_hamiltonian, qml.ops.Sum)
            ops = program.cost_hamiltonian.operands
            measured_wires = {op.wires[0] for op in ops if isinstance(op, qml.Z)}
            assert measured_wires == expected_measured_wires
        else:
            assert isinstance(program.cost_hamiltonian, qml.Z)
            assert program.cost_hamiltonian.wires[0] in expected_measured_wires


class TestOptimization:
    """Test suite for optimization functionality."""

    @pytest.mark.parametrize("optimizer", **OPTIMIZERS_TO_TEST)
    def test_optimization_runs(self, optimizer, simple_quantum_script, dummy_simulator):
        """Test that optimization runs with various optimizers."""
        optimizer = optimizer()
        # Skip L_BFGS_B as it may not run iterations with dummy simulator
        if (
            isinstance(optimizer, ScipyOptimizer)
            and optimizer.method == ScipyMethod.L_BFGS_B
        ):
            pytest.skip("L_BFGS_B may not run iterations with dummy simulator")

        program = CustomVQA(
            qscript=simple_quantum_script,
            optimizer=optimizer,
            max_iterations=1,
            backend=dummy_simulator,
        )

        program.run(perform_final_computation=False)
        verify_correct_circuit_count(program)

    @pytest.mark.parametrize("optimizer", **OPTIMIZERS_TO_TEST)
    @pytest.mark.parametrize(
        "circuit_fixture",
        [
            "qiskit_circuit_with_measurements",
            "qiskit_circuit_multi_qubit_all_measured",
            "qiskit_circuit_multi_qubit_partial_measured",
        ],
    )
    def test_qiskit_optimization_runs(
        self, optimizer, circuit_fixture, dummy_simulator, request
    ):
        """Test that optimization runs with various Qiskit circuit configurations."""
        optimizer = optimizer()
        circuit = request.getfixturevalue(circuit_fixture)
        program = CustomVQA(
            qscript=circuit,
            optimizer=optimizer,
            max_iterations=1,
            backend=dummy_simulator,
        )

        program.run(perform_final_computation=False)
        verify_correct_circuit_count(program)

    @pytest.mark.e2e
    def test_e2e_optimization(self, simple_quantum_script, default_test_simulator):
        """Test end-to-end optimization."""
        default_test_simulator.set_seed(1997)
        program = CustomVQA(
            qscript=simple_quantum_script,
            optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
            max_iterations=5,
            backend=default_test_simulator,
            seed=1997,
        )

        program.run(perform_final_computation=False)

        assert len(program.losses_history) == 5
        assert isinstance(program.best_loss, float)
        assert isinstance(program.best_params, np.ndarray)
        assert program.best_params.shape == (program.n_params,)

    @pytest.mark.e2e
    def test_e2e_qiskit_optimization(
        self, qiskit_circuit_with_measurements, default_test_simulator
    ):
        """Test end-to-end optimization with Qiskit input."""
        default_test_simulator.set_seed(1997)
        program = CustomVQA(
            qscript=qiskit_circuit_with_measurements,
            optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
            max_iterations=5,
            backend=default_test_simulator,
            seed=1997,
        )

        program.run(perform_final_computation=False)

        assert len(program.losses_history) == 5
        assert isinstance(program.best_loss, float)
        assert isinstance(program.best_params, np.ndarray)
        assert program.best_params.shape == (program.n_params,)


class TestCheckpointing:
    """Test suite for checkpointing functionality."""

    @pytest.mark.e2e
    @pytest.mark.parametrize("optimizer", **CHECKPOINTING_OPTIMIZERS)
    def test_checkpointing_resume(
        self, optimizer, simple_quantum_script, default_test_simulator, tmp_path
    ):
        """Test checkpointing and resume functionality."""
        optimizer = optimizer()
        checkpoint_dir = tmp_path / "checkpoint_test"
        default_test_simulator.set_seed(1997)

        # First run: iterations 1-2
        program1 = CustomVQA(
            qscript=simple_quantum_script,
            optimizer=optimizer,
            max_iterations=2,
            backend=default_test_simulator,
            seed=1997,
        )
        program1.run(checkpoint_config=CheckpointConfig(checkpoint_dir=checkpoint_dir))
        assert program1.current_iteration == 2

        # Verify checkpoint was created
        checkpoint_path = checkpoint_dir / "checkpoint_002"
        assert checkpoint_path.exists()
        assert (checkpoint_path / "program_state.json").exists()

        # Store state from first run
        first_run_iteration = program1.current_iteration
        first_run_losses_count = len(program1.losses_history)
        first_run_best_loss = program1.best_loss

        # Second run: resume and run iterations 3-4
        program2 = CustomVQA.load_state(
            checkpoint_dir,
            backend=default_test_simulator,
            qscript=simple_quantum_script,
        )

        # Verify loaded state matches first run
        assert program2.current_iteration == first_run_iteration
        assert len(program2.losses_history) == first_run_losses_count
        assert program2.best_loss == pytest.approx(first_run_best_loss)

        program2.max_iterations = 4
        program2.run(checkpoint_config=CheckpointConfig(checkpoint_dir=checkpoint_dir))
        assert program2.current_iteration == 4
        assert (checkpoint_dir / "checkpoint_004").exists()
