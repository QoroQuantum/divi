# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pennylane as qp
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp

from divi.pipeline.stages import DataBindingStage
from divi.qprog import CustomVQA
from divi.qprog.checkpointing import CheckpointConfig
from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer
from tests.qprog._program_contracts import (
    ObservableMeasuringContractsBase,
    verify_correct_circuit_count,
    verify_metacircuit_dict,
)


@pytest.fixture
def simple_quantum_script():
    """Fixture for a simple parameterized QuantumScript."""
    ops = [qp.RX(0.0, wires=0), qp.RZ(0.0, wires=0)]
    measurements = [qp.expval(qp.Z(0))]
    return qp.tape.QuantumScript(ops=ops, measurements=measurements)


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

        assert program.qscript is circuit
        assert program.n_qubits == expected_n_qubits
        assert program.n_layers == 1
        assert program.n_params_per_layer == 2
        assert program.param_shape == (2,)
        assert isinstance(program.cost_hamiltonian, SparsePauliOp)
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
        assert program.n_params_per_layer == 2

    def test_qnode_input_is_converted_to_quantum_script(self, dummy_simulator):
        """QNode inputs are accepted and converted via PennyLaneSpecStage."""
        dev = qp.device("default.qubit", wires=1)

        @qp.qnode(dev)
        def circuit(theta, phi):
            qp.RX(theta, wires=0)
            qp.RZ(phi, wires=0)
            return qp.expval(qp.Z(0))

        program = CustomVQA(qscript=circuit, backend=dummy_simulator)
        assert isinstance(program.qscript, qp.tape.QuantumScript)
        assert program.n_qubits == 1
        assert program.n_params_per_layer == 2

    def test_reused_qnode_arg_ties_to_one_parameter(self, default_test_simulator):
        """A function argument reused across gates is ONE optimizer knob.

        ``def c(x): RX(x); RX(x); RX(x)`` is a single trainable parameter in
        PennyLane (<Z> = cos(3x)); divi must tie the three gates to one
        parameter rather than exposing three independent ones.
        """
        dev = qp.device("default.qubit", wires=1)

        @qp.qnode(dev)
        def circuit(x):
            for _ in range(3):
                qp.RX(x, wires=0)
            return qp.expval(qp.Z(0))

        program = CustomVQA(qscript=circuit, backend=default_test_simulator)
        assert program.param_shape == (1,)
        # <Z>([0.3]) should match cos(3 * 0.3) = cos(0.9), not be untied.
        loss = program._evaluate_cost_param_sets(np.array([[0.3]]))[0]
        assert loss == pytest.approx(np.cos(0.9), abs=0.05)

    def test_qnode_arg_expression_preserves_coefficient(self, default_test_simulator):
        """A ``ParameterExpression`` arg (``RX(2*theta)``) keeps its coefficient.

        <Z> at theta=0.4 is cos(2*0.4) = cos(0.8), not cos(0.4).
        """
        dev = qp.device("default.qubit", wires=1)

        @qp.qnode(dev)
        def circuit(theta):
            qp.RX(2 * theta, wires=0)
            return qp.expval(qp.Z(0))

        program = CustomVQA(qscript=circuit, backend=default_test_simulator)
        assert program.param_shape == (1,)
        loss = program._evaluate_cost_param_sets(np.array([[0.4]]))[0]
        assert loss == pytest.approx(np.cos(0.8), abs=0.05)

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
        assert isinstance(program.cost_hamiltonian, SparsePauliOp)

    @pytest.mark.parametrize(
        "observable",
        [0.5 * qp.Z(0), qp.Z(0)],
        ids=["sprod", "bare_pauli"],
    )
    def test_single_term_observable_succeeds(self, dummy_simulator, observable):
        """Single-term observables (SProd, bare Pauli) initialize without operands error."""
        ops = [qp.RX(0.0, wires=0)]
        measurements = [qp.expval(observable)]
        qscript = qp.tape.QuantumScript(ops=ops, measurements=measurements)

        vqa = CustomVQA(qscript=qscript, backend=dummy_simulator)
        assert vqa.cost_hamiltonian is not None
        assert vqa.n_qubits == 1


class TestConstructionValidation:
    """Test suite for error handling."""

    def test_invalid_input_type(self, dummy_simulator):
        """Test that invalid input type raises TypeError."""
        with pytest.raises(TypeError, match="must be a PennyLane QuantumScript"):
            CustomVQA(qscript="not a circuit", backend=dummy_simulator)

    def test_multiple_measurements_fails(self, dummy_simulator):
        """Test that QuantumScript with multiple measurements fails."""
        ops = [qp.RX(0.0, wires=0)]
        measurements = [qp.expval(qp.Z(0)), qp.expval(qp.Z(0))]
        qscript = qp.tape.QuantumScript(ops=ops, measurements=measurements)

        with pytest.raises(ValueError, match="exactly one measurement"):
            CustomVQA(qscript=qscript, backend=dummy_simulator)

    def test_no_measurement_fails(self, dummy_simulator):
        """Test that QuantumScript without measurement fails."""
        ops = [qp.RX(0.0, wires=0)]
        qscript = qp.tape.QuantumScript(ops=ops, measurements=[])

        with pytest.raises(ValueError, match="exactly one measurement"):
            CustomVQA(qscript=qscript, backend=dummy_simulator)

    def test_non_expval_measurement_fails(self, dummy_simulator):
        """Test that non-expectation-value measurement fails."""
        ops = [qp.RX(0.0, wires=0)]
        measurements = [qp.probs(wires=0)]
        qscript = qp.tape.QuantumScript(ops=ops, measurements=measurements)

        with pytest.raises(ValueError, match="expval"):
            CustomVQA(qscript=qscript, backend=dummy_simulator)

    def test_constant_only_hamiltonian_fails(self, dummy_simulator):
        """Test that constant-only Hamiltonian fails."""
        ops = [qp.RX(0.0, wires=0)]
        measurements = [qp.expval(qp.Identity(0) * 5.0)]
        qscript = qp.tape.QuantumScript(ops=ops, measurements=measurements)

        with pytest.raises(ValueError, match="only constant terms"):
            CustomVQA(qscript=qscript, backend=dummy_simulator)

    def test_pennylane_observable_with_coefficients_does_not_crash(
        self, dummy_simulator
    ):
        """An observable with non-Identity coefficients (e.g. ``0.5 * Z(0)``)
        must not have its coefficients substituted into the ``ParameterVector``
        via ``bind_new_parameters``. Regression for the
        ``Parameter expression with unbound parameters`` crash.
        """
        ops = [qp.RX(0.0, wires=0)]
        obs = 0.5 * qp.Z(0)
        qscript = qp.tape.QuantumScript(ops=ops, measurements=[qp.expval(obs)])
        program = CustomVQA(qscript=qscript, backend=dummy_simulator)
        # One operation parameter; the observable coefficient is excluded.
        assert program.n_params_per_layer == 1

    @pytest.mark.e2e
    def test_pennylane_mixed_constant_observable_does_not_double_count(
        self, default_test_simulator
    ):
        """An observable like ``0.5 * Z(0) + 5.0 * Identity`` must yield
        the loss ``0.5 * <Z> + 5.0`` (constant added once), not
        ``0.5 * <Z> + 10.0`` (constant double-counted by including it
        in both the measured observable and ``loss_constant``).
        """
        default_test_simulator.set_seed(1997)
        ops = [qp.RX(0.0, wires=0)]  # theta=0 → state stays |0> → <Z>=1
        obs = 0.5 * qp.Z(0) + 5.0 * qp.Identity(0)
        qscript = qp.tape.QuantumScript(ops=ops, measurements=[qp.expval(obs)])
        program = CustomVQA(
            qscript=qscript,
            backend=default_test_simulator,
            seed=1997,
        )
        losses = program._evaluate_cost_param_sets(np.array([[0.0]]))
        # Correct: 0.5 * 1.0 + 5.0 = 5.5. Buggy double-count would be 10.5.
        assert losses[0] == pytest.approx(5.5, abs=0.05)

    def test_pennylane_respects_user_set_trainable_params_filter(self, dummy_simulator):
        """When the user pins ``qs.trainable_params`` to operation indices,
        observable-coefficient indices stay excluded even if they appear
        in the user's list."""
        ops = [qp.RX(0.0, wires=0), qp.RZ(0.0, wires=0)]
        obs = 0.5 * qp.Z(0)
        qscript = qp.tape.QuantumScript(ops=ops, measurements=[qp.expval(obs)])
        # User sets trainable_params to include observable index too.
        qscript.trainable_params = [0, 1, 2]
        program = CustomVQA(qscript=qscript, backend=dummy_simulator)
        # Two operation parameters; observable coefficient at index 2 is filtered out.
        assert program.n_params_per_layer == 2

    def test_no_trainable_parameters_fails(self, dummy_simulator):
        """Test that QuantumScript without trainable parameters fails."""
        # Create a script with no operations (empty circuit)
        measurements = [qp.expval(qp.Z(0))]
        qscript = qp.tape.QuantumScript(ops=[], measurements=measurements)

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

    def test_run_accepts_initial_params(self, simple_quantum_script, dummy_simulator):
        """Test passing initial parameters through run()."""
        initial_params = np.array([[0.1, 0.2]])
        program = CustomVQA(
            qscript=simple_quantum_script,
            max_iterations=1,
            optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
            backend=dummy_simulator,
        )

        program.run(initial_params=initial_params, perform_final_computation=False)
        assert program.total_circuit_count >= 0

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
        ops = [qp.RX(0.0, wires=0), qp.RY(0.0, wires=1)]
        measurements = [qp.expval(qp.Z(0) @ qp.Z(1))]
        qscript = qp.tape.QuantumScript(ops=ops, measurements=measurements)

        program = CustomVQA(qscript=qscript, backend=dummy_simulator)
        assert program.n_qubits == 2
        assert program.n_params_per_layer == 2


class TestQiskitConversion:
    """Test suite for Qiskit QuantumCircuit conversion."""

    def test_qiskit_measurements_removed(
        self, qiskit_circuit_with_measurements, dummy_simulator
    ):
        """Qiskit measurements are stripped before circuit construction."""
        program = CustomVQA(
            qscript=qiskit_circuit_with_measurements,
            backend=dummy_simulator,
        )

        # Measurements drive the observable, not the cost circuit; the
        # stripped Qiskit circuit must carry no `measure` instructions.
        assert program._qiskit_circuit is not None
        op_names = {instr.operation.name for instr in program._qiskit_circuit.data}
        assert "measure" not in op_names
        assert program.measured_wires

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
        assert program.n_params_per_layer == expected_n_params
        assert isinstance(program.cost_hamiltonian, SparsePauliOp)
        assert set(program.measured_wires) == expected_measured_wires


class TestOptimization:
    """Test suite for optimization functionality."""

    def test_optimization_runs(self, optimizer, simple_quantum_script, dummy_simulator):
        """Test that optimization runs with various optimizers."""
        program = CustomVQA(
            qscript=simple_quantum_script,
            optimizer=optimizer,
            max_iterations=1,
            backend=dummy_simulator,
        )

        program.run(perform_final_computation=False)
        verify_correct_circuit_count(program)

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
        assert program.best_params.shape == (program.n_params_per_layer,)

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
        assert program.best_params.shape == (program.n_params_per_layer,)


@pytest.mark.e2e
def test_checkpointing_resume(
    checkpointing_optimizer, simple_quantum_script, default_test_simulator, tmp_path
):
    """Test checkpointing and resume functionality."""
    checkpoint_dir = tmp_path / "checkpoint_test"
    default_test_simulator.set_seed(1997)

    # First run: iterations 1-2
    program1 = CustomVQA(
        qscript=simple_quantum_script,
        optimizer=checkpointing_optimizer,
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


class TestObservableMeasuringContracts(ObservableMeasuringContractsBase):
    @pytest.fixture
    def make_program(self, simple_quantum_script, dummy_simulator):
        def _make(**kwargs):
            return CustomVQA(
                qscript=simple_quantum_script,
                backend=dummy_simulator,
                **kwargs,
            )

        return _make


# ---------------------------------------------------------------------------
# Data-binding path
# ---------------------------------------------------------------------------


@pytest.fixture
def qiskit_data_circuit():
    """Qiskit circuit with one data parameter and two trainable weights."""
    x = Parameter("x")
    w0 = Parameter("w0")
    w1 = Parameter("w1")
    qc = QuantumCircuit(2, 2)
    qc.ry(x, 0)
    qc.ry(w0, 1)
    qc.cx(0, 1)
    qc.rz(w1, 0)
    qc.measure(0, 0)
    qc.measure(1, 1)
    return qc, x, (w0, w1)


@pytest.fixture
def feature_batch_4x1():
    return np.array([[0.1], [0.3], [0.5], [0.7]])


@pytest.fixture
def angle_bel_qnode():
    """A 2-qubit ``AngleEmbedding(inputs)`` + ``BasicEntanglerLayers(weights)``
    QNode — the common data/weight skeleton used by the data-arg error-path
    tests."""
    dev = qp.device("default.qubit", wires=2)

    @qp.qnode(dev)
    def circuit(inputs, weights):
        qp.AngleEmbedding(inputs, wires=range(2))
        qp.BasicEntanglerLayers(weights, wires=range(2))
        return qp.expval(qp.Z(0))

    return circuit


class TestDataBindingConstruction:
    def test_partitions_data_and_weight_params(
        self, qiskit_data_circuit, feature_batch_4x1, dummy_simulator
    ):
        qc, x, _ = qiskit_data_circuit
        data_idx = [list(qc.parameters).index(x)]
        program = CustomVQA(
            qscript=qc,
            data_param_indices=data_idx,
            feature_batch=feature_batch_4x1,
            backend=dummy_simulator,
        )
        assert program.feature_batch is not None
        # DataBindingStage is wired into the cost pipeline when active.
        assert DataBindingStage in [type(s) for s in program._pipelines["cost"].stages]
        # Two weight parameters survive, exposed flat to the optimizer.
        assert program.n_params_per_layer == 2
        assert program.param_shape == (2,)
        # MetaCircuit still carries the full (data + weights) parameter tuple.
        params = program.meta_circuit_factories["cost_circuit"].parameters
        assert len(params) == 3

    def test_supervised_labels_stored(
        self, qiskit_data_circuit, feature_batch_4x1, dummy_simulator
    ):
        qc, x, _ = qiskit_data_circuit
        data_idx = [list(qc.parameters).index(x)]
        program = CustomVQA(
            qscript=qc,
            data_param_indices=data_idx,
            feature_batch=feature_batch_4x1,
            labels=[1.0, -1.0, 1.0, -1.0],
            backend=dummy_simulator,
        )
        np.testing.assert_array_equal(program.labels, [1.0, -1.0, 1.0, -1.0])
        assert callable(program._sample_loss_fn)

    def test_supervised_labels_default_none(
        self, qiskit_data_circuit, feature_batch_4x1, dummy_simulator
    ):
        qc, x, _ = qiskit_data_circuit
        data_idx = [list(qc.parameters).index(x)]
        program = CustomVQA(
            qscript=qc,
            data_param_indices=data_idx,
            feature_batch=feature_batch_4x1,
            backend=dummy_simulator,
        )
        assert program.labels is None
        assert program._sample_loss_fn is None

    def test_labels_without_data_binding_rejected(
        self, simple_quantum_script, dummy_simulator
    ):
        with pytest.raises(ValueError, match="labels require data binding"):
            CustomVQA(
                qscript=simple_quantum_script,
                param_shape=(2,),
                labels=[1.0, -1.0],
                backend=dummy_simulator,
            )

    def test_supervised_labels_length_mismatch_rejected(
        self, qiskit_data_circuit, feature_batch_4x1, dummy_simulator
    ):
        qc, x, _ = qiskit_data_circuit
        data_idx = [list(qc.parameters).index(x)]
        with pytest.raises(ValueError, match="labels has 2 entries but feature_batch"):
            CustomVQA(
                qscript=qc,
                data_param_indices=data_idx,
                feature_batch=feature_batch_4x1,
                labels=[1.0, -1.0],
                backend=dummy_simulator,
            )

    def test_loss_fn_without_labels_warns(self, simple_quantum_script, dummy_simulator):
        with pytest.warns(UserWarning, match="loss_fn"):
            CustomVQA(
                qscript=simple_quantum_script,
                param_shape=(2,),
                loss_fn=lambda pred, label: abs(pred - label),
                backend=dummy_simulator,
            )

    def test_multiarg_qnode_with_arg_shapes_and_data_arg(self, default_test_simulator):
        """A two-argument template QNode (data + weights) ingests via arg_shapes
        + data_arg, optimizes only the weights, and matches PennyLane physics."""
        n = 3
        dev = qp.device("default.qubit", wires=n)

        @qp.qnode(dev)
        def circuit(inputs, weights):
            qp.AngleEmbedding(inputs, wires=range(n), rotation="Y")
            qp.StronglyEntanglingLayers(weights, wires=range(n))
            return qp.expval(qp.Z(0))

        X = np.array([[0.1, 0.2, 0.3], [0.7, 0.8, 0.9]])
        program = CustomVQA(
            qscript=circuit,
            arg_shapes={"weights": (1, n, 3)},
            data_arg="inputs",
            feature_batch=X,
            backend=default_test_simulator,
        )
        # Only the 9 SEL weights are exposed to the optimizer; data is bound.
        assert program.param_shape == (9,)
        assert program.feature_batch.shape == (2, n)
        assert DataBindingStage in [type(s) for s in program._pipelines["cost"].stages]

        # Physics: batched mean loss equals the mean of per-sample PennyLane.
        w = np.arange(9, dtype=float) / 5
        divi_loss = program._evaluate_cost_param_sets(w.reshape(1, -1))[0]
        per_sample = [
            float(
                qp.execute(
                    [qp.tape.make_qscript(circuit.func)(x, w.reshape(1, n, 3))], dev
                )[0]
            )
            for x in X
        ]
        assert divi_loss == pytest.approx(float(np.mean(per_sample)), abs=0.05)

    def test_batch_input_decorator_auto_detects_data_arg(self, dummy_simulator):
        """A ``@qml.batch_input(argnum=...)`` QNode supplies the data axis; divi
        reads it so ``data_arg`` need not be passed."""
        n = 3

        @qp.batch_input(argnum=0)
        @qp.qnode(qp.device("default.qubit", wires=n))
        def circuit(inputs, weights):
            qp.AngleEmbedding(inputs, wires=range(n), rotation="Y")
            qp.StronglyEntanglingLayers(weights, wires=range(n))
            return qp.expval(qp.Z(0))

        program = CustomVQA(
            qscript=circuit,
            arg_shapes={"weights": (1, n, 3)},
            feature_batch=np.zeros((2, n)),  # data_arg NOT passed — auto-detected
            backend=dummy_simulator,
        )
        # "inputs" (argnum 0) was detected as data → only the 9 weights remain.
        assert program.param_shape == (9,)

    def test_explicit_data_arg_overrides_batch_input_detection(self, dummy_simulator):
        # An explicit data_arg is honored even when batch_input is present.
        n = 2

        @qp.batch_input(argnum=0)
        @qp.qnode(qp.device("default.qubit", wires=n))
        def circuit(inputs, weights):
            qp.AngleEmbedding(inputs, wires=range(n), rotation="Y")
            qp.BasicEntanglerLayers(weights, wires=range(n))
            return qp.expval(qp.Z(0))

        program = CustomVQA(
            qscript=circuit,
            arg_shapes={"weights": (1, n)},
            data_arg="inputs",
            feature_batch=np.zeros((2, n)),
            backend=dummy_simulator,
        )
        assert program.param_shape == (2,)

    def test_arg_shapes_rejects_non_qnode(self, simple_quantum_script, dummy_simulator):
        with pytest.raises(ValueError, match="only valid for QNode"):
            CustomVQA(
                qscript=simple_quantum_script,
                arg_shapes={"w": (2,)},
                backend=dummy_simulator,
            )

    def test_data_arg_without_feature_batch_raises(
        self, angle_bel_qnode, dummy_simulator
    ):
        with pytest.raises(ValueError, match="data_arg requires feature_batch"):
            CustomVQA(
                qscript=angle_bel_qnode,
                arg_shapes={"weights": (1, 2)},
                data_arg="inputs",
                backend=dummy_simulator,
            )

    def test_data_arg_contributing_no_params_raises(self, dummy_simulator):
        # `inputs` is declared the data arg but never reaches a gate, so it
        # contributes no trainable parameters.
        n = 2
        dev = qp.device("default.qubit", wires=n)

        @qp.qnode(dev)
        def circuit(inputs, weights):
            qp.BasicEntanglerLayers(weights, wires=range(n))
            return qp.expval(qp.Z(0))

        with pytest.raises(ValueError, match="contributed no trainable parameters"):
            CustomVQA(
                qscript=circuit,
                arg_shapes={"inputs": (n,), "weights": (1, n)},
                data_arg="inputs",
                feature_batch=np.zeros((2, n)),
                backend=dummy_simulator,
            )

    def test_multiple_batch_input_args_raise(self, dummy_simulator):
        n = 2

        @qp.batch_input(argnum=[0, 1])
        @qp.qnode(qp.device("default.qubit", wires=n))
        def circuit(a, b):
            qp.RX(a, wires=0)
            qp.RX(b, wires=1)
            return qp.expval(qp.Z(0))

        with pytest.raises(ValueError, match="multiple batched arguments"):
            CustomVQA(
                qscript=circuit,
                feature_batch=np.zeros((2, 1)),
                backend=dummy_simulator,
            )

    def test_rejects_zero_row_feature_batch(self, qiskit_data_circuit, dummy_simulator):
        qc, x, _ = qiskit_data_circuit
        with pytest.raises(ValueError, match="at least one sample"):
            CustomVQA(
                qscript=qc,
                data_param_indices=[list(qc.parameters).index(x)],
                feature_batch=np.zeros((0, 1)),
                backend=dummy_simulator,
            )

    def test_rejects_1d_feature_batch(self, qiskit_data_circuit, dummy_simulator):
        qc, x, _ = qiskit_data_circuit
        with pytest.raises(ValueError, match="2D"):
            CustomVQA(
                qscript=qc,
                data_param_indices=[list(qc.parameters).index(x)],
                feature_batch=np.array([0.1, 0.3]),
                backend=dummy_simulator,
            )

    def test_rejects_bool_data_param_index(self, qiskit_data_circuit, dummy_simulator):
        qc, _, _ = qiskit_data_circuit
        with pytest.raises(TypeError):
            CustomVQA(
                qscript=qc,
                data_param_indices=[True],
                feature_batch=np.zeros((1, 1)),
                backend=dummy_simulator,
            )

    def test_data_arg_and_data_param_indices_are_mutually_exclusive(
        self, angle_bel_qnode, dummy_simulator
    ):
        with pytest.raises(
            ValueError, match="either data_arg .* or data_param_indices"
        ):
            CustomVQA(
                qscript=angle_bel_qnode,
                arg_shapes={"weights": (1, 2)},
                data_arg="inputs",
                data_param_indices=[0],
                feature_batch=np.zeros((1, 2)),
                backend=dummy_simulator,
            )

    def test_unknown_data_arg_lists_valid_argument_names(
        self, angle_bel_qnode, dummy_simulator
    ):
        with pytest.raises(ValueError, match=r"not a QNode argument.*inputs.*weights"):
            CustomVQA(
                qscript=angle_bel_qnode,
                arg_shapes={"weights": (1, 2)},
                data_arg="inputz",  # typo
                feature_batch=np.zeros((1, 2)),
                backend=dummy_simulator,
            )

    def test_unknown_arg_shapes_key_lists_valid_argument_names(
        self, angle_bel_qnode, dummy_simulator
    ):
        with pytest.raises(ValueError, match=r"not QNode arguments.*inputs.*weights"):
            CustomVQA(
                qscript=angle_bel_qnode,
                arg_shapes={"wieghts": (1, 2)},  # typo
                data_arg="inputs",
                feature_batch=np.zeros((1, 2)),
                backend=dummy_simulator,
            )

    def test_unset_data_param_indices_keeps_original_behaviour(
        self, qiskit_circuit_with_measurements, dummy_simulator
    ):
        """Omitting data_param_indices yields the pre-feature-batch behaviour.

        Also guards that no ``DataBindingStage`` slipped into the pipeline
        when data binding is inactive — a regression that accidentally
        inserted it would silently inflate runtime cost.
        """
        program = CustomVQA(
            qscript=qiskit_circuit_with_measurements, backend=dummy_simulator
        )
        assert program.feature_batch is None

        stage_types = [type(s) for s in program._pipelines["cost"].stages]
        assert DataBindingStage not in stage_types

    @pytest.mark.parametrize(
        "data_param_indices,feature_batch",
        [
            ([0], None),
            (None, np.zeros((1, 1))),
        ],
    )
    def test_requires_both_or_neither(
        self,
        data_param_indices,
        feature_batch,
        qiskit_data_circuit,
        dummy_simulator,
    ):
        qc, _, _ = qiskit_data_circuit
        with pytest.raises(
            ValueError, match="data_param_indices and feature_batch must both"
        ):
            CustomVQA(
                qscript=qc,
                data_param_indices=data_param_indices,
                feature_batch=feature_batch,
                backend=dummy_simulator,
            )

    def test_rejects_param_shape_when_data_active(
        self, qiskit_data_circuit, feature_batch_4x1, dummy_simulator
    ):
        qc, x, _ = qiskit_data_circuit
        data_idx = [list(qc.parameters).index(x)]
        with pytest.raises(
            ValueError,
            match="param_shape is not supported when data_param_indices",
        ):
            CustomVQA(
                qscript=qc,
                data_param_indices=data_idx,
                feature_batch=feature_batch_4x1,
                param_shape=(2,),
                backend=dummy_simulator,
            )

    def test_rejects_all_params_as_data(self, qiskit_data_circuit, dummy_simulator):
        qc, _, _ = qiskit_data_circuit
        with pytest.raises(ValueError, match="no trainable weights left"):
            CustomVQA(
                qscript=qc,
                data_param_indices=list(range(len(qc.parameters))),
                feature_batch=np.zeros((1, len(qc.parameters))),
                backend=dummy_simulator,
            )

    @pytest.mark.parametrize(
        "bad_indices,match",
        [
            ([0, 0], "duplicate"),
            ([5], "out of range"),
            ([-1], "out of range"),
        ],
    )
    def test_rejects_bad_data_indices(
        self,
        bad_indices,
        match,
        qiskit_data_circuit,
        dummy_simulator,
    ):
        qc, _, _ = qiskit_data_circuit
        with pytest.raises(ValueError, match=match):
            CustomVQA(
                qscript=qc,
                data_param_indices=bad_indices,
                feature_batch=np.zeros((1, len(bad_indices))),
                backend=dummy_simulator,
            )

    def test_accepts_numpy_integer_data_indices(
        self, qiskit_data_circuit, dummy_simulator
    ):
        """numpy integer indices (np.arange / np.where outputs) are valid —
        they must not be rejected by an ``isinstance(idx, int)`` check."""
        qc, _, _ = qiskit_data_circuit
        program = CustomVQA(
            qscript=qc,
            data_param_indices=np.array([0]),
            feature_batch=np.zeros((1, 1)),
            backend=dummy_simulator,
        )
        assert len(program._data_symbols) == 1

    def test_rejects_empty_data_param_indices(
        self, qiskit_data_circuit, dummy_simulator
    ):
        """Empty list is its own error; separate so the feature_batch shape
        is unambiguous and the assertion isn't coupled to validation order."""
        qc, _, _ = qiskit_data_circuit
        with pytest.raises(ValueError, match="at least one index"):
            CustomVQA(
                qscript=qc,
                data_param_indices=[],
                feature_batch=np.zeros((1, 1)),
                backend=dummy_simulator,
            )

    def test_pennylane_data_binding_partitions_synthesized_parameters(
        self, simple_quantum_script, dummy_simulator
    ):
        """Regression guard: a refactor of ``_prepare_pennylane_input`` that
        permuted the synthetic ``ParameterVector`` would silently bind the
        wrong parameter without changing the program's outward shape."""
        program = CustomVQA(
            qscript=simple_quantum_script,
            data_param_indices=[0],
            feature_batch=np.array([[1.5]]),
            backend=dummy_simulator,
        )
        assert [str(p) for p in program._data_symbols] == ["p0"]
        assert [str(p) for p in program._weight_symbols] == ["p1"]
        # The composed circuit retains both parameters; data substitution
        # happens per-sample inside DataBindingStage rather than up front.
        assert set(program._composed_circuit.parameters) == {
            program._data_symbols[0],
            program._weight_symbols[0],
        }
        # The optimizer view (n_params_per_layer × n_layers) exposes only
        # the weight parameter.
        assert program.n_params_per_layer == 1

    def test_pennylane_data_binding_respects_trainable_params_order(
        self, dummy_simulator
    ):
        """``data_param_indices`` always refers to the trainable-param
        ordering, not the qscript's gate-construction order. With a
        permuted ``trainable_params``, index 0 still maps to ``p0``, but
        the underlying gate is the first entry of ``trainable_params``."""
        ops = [
            qp.RX(0.0, wires=0),  # qscript-index 0
            qp.RY(0.0, wires=0),  # qscript-index 1
            qp.RZ(0.0, wires=0),  # qscript-index 2
        ]
        measurements = [qp.expval(qp.Z(0))]
        qs = qp.tape.QuantumScript(ops=ops, measurements=measurements)
        qs.trainable_params = [2, 0, 1]

        program = CustomVQA(
            qscript=qs,
            data_param_indices=[0],
            feature_batch=np.array([[1.5]]),
            backend=dummy_simulator,
        )
        assert [str(p) for p in program._data_symbols] == ["p0"]
        assert [str(p) for p in program._weight_symbols] == ["p1", "p2"]

    def test_rejects_feature_batch_column_mismatch(
        self, qiskit_data_circuit, dummy_simulator
    ):
        qc, x, _ = qiskit_data_circuit
        data_idx = [list(qc.parameters).index(x)]
        with pytest.raises(ValueError, match="data_param_indices declares 1"):
            CustomVQA(
                qscript=qc,
                data_param_indices=data_idx,
                feature_batch=np.zeros((4, 2)),  # only 1 data index declared
                backend=dummy_simulator,
            )

    @pytest.mark.e2e
    def test_predict_returns_class_labels(
        self, qiskit_data_circuit, feature_batch_4x1, default_test_simulator
    ):
        """The shared DataBindingMixin.predict works through CustomVQA too."""
        qc, x, _ = qiskit_data_circuit
        data_idx = [list(qc.parameters).index(x)]
        default_test_simulator.set_seed(1997)
        program = CustomVQA(
            qscript=qc,
            data_param_indices=data_idx,
            feature_batch=feature_batch_4x1,
            backend=default_test_simulator,
        )
        weights = np.array([0.5, 1.0])  # two trainable weights (w0, w1)
        labels = program.predict(feature_batch_4x1, params=weights)
        readout = program.predict_readout(feature_batch_4x1, params=weights)
        assert labels.shape == (feature_batch_4x1.shape[0],)
        assert set(np.unique(labels)).issubset({-1.0, 1.0})
        np.testing.assert_array_equal(labels, np.where(readout >= 0.0, 1.0, -1.0))


class TestDataBindingDryRun:
    def test_data_axis_factor_matches_sample_count(
        self, qiskit_data_circuit, feature_batch_4x1, dummy_simulator
    ):
        qc, x, _ = qiskit_data_circuit
        data_idx = [list(qc.parameters).index(x)]
        program = CustomVQA(
            qscript=qc,
            data_param_indices=data_idx,
            feature_batch=feature_batch_4x1,
            backend=dummy_simulator,
        )
        reports = program.dry_run()
        data_stage = next(s for s in reports["cost"].stages if s.axis == "data_sample")
        assert data_stage.factor == feature_batch_4x1.shape[0]
        assert data_stage.metadata["n_samples"] == feature_batch_4x1.shape[0]

    def test_total_circuits_matches_samples_times_param_sets(
        self, qiskit_data_circuit, feature_batch_4x1, dummy_simulator
    ):
        """For a QWC-groupable single-output observable, the dry-run total
        equals ``n_samples × n_param_sets``."""
        qc, x, _ = qiskit_data_circuit
        data_idx = [list(qc.parameters).index(x)]
        program = CustomVQA(
            qscript=qc,
            data_param_indices=data_idx,
            feature_batch=feature_batch_4x1,
            backend=dummy_simulator,
        )
        reports = program.dry_run()
        expected = feature_batch_4x1.shape[0] * program.optimizer.n_param_sets
        assert reports["cost"].total_circuits == expected


@pytest.mark.e2e
def test_data_binding_runs_end_to_end(
    qiskit_data_circuit, feature_batch_4x1, default_test_simulator
):
    qc, x, _ = qiskit_data_circuit
    data_idx = [list(qc.parameters).index(x)]
    default_test_simulator.set_seed(1997)
    program = CustomVQA(
        qscript=qc,
        data_param_indices=data_idx,
        feature_batch=feature_batch_4x1,
        optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
        max_iterations=3,
        backend=default_test_simulator,
        seed=1997,
    )
    program.run(perform_final_computation=False)
    assert len(program.losses_history) == 3
    # best_params reports only the weight parameters, not data.
    assert program.best_params.shape == (2,)


@pytest.mark.e2e
def test_data_binding_pennylane_input_runs(
    simple_quantum_script, default_test_simulator
):
    """The PennyLane path also accepts data indices, mapped against the
    synthesized parameter vector."""
    default_test_simulator.set_seed(1997)
    program = CustomVQA(
        qscript=simple_quantum_script,
        data_param_indices=[0],  # bind the first trainable param from data
        feature_batch=np.array([[0.1], [0.3], [0.5]]),
        optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
        max_iterations=2,
        backend=default_test_simulator,
        seed=1997,
    )
    program.run(perform_final_computation=False)
    assert len(program.losses_history) == 2
    # 2 params total, 1 is data → 1 weight.
    assert program.best_params.shape == (1,)
