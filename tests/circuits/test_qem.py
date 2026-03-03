# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from functools import partial

import cirq
import pennylane as qml
import pytest
import sympy as sp
from mitiq.zne.inference import Factory
from sympy import Symbol

from divi.circuits import qem
from divi.circuits._qasm_conversion import to_openqasm
from divi.circuits.qem import (
    ZNE,
    QEMProtocol,
    _inject_input_declarations,
    _NoMitigation,
    apply_protocol_to_qasm,
    normalize_qasm_after_cirq,
)


class TestQEMProtocol:
    """Test suite for the abstract QEMProtocol base class."""

    def test_abstract_class_cannot_be_instantiated(self):
        """Test that the abstract QEMProtocol class cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            QEMProtocol()

    def test_concrete_implementations_can_be_instantiated(self, mocker):
        """Test that concrete implementations of QEMProtocol can be instantiated."""
        # Test _NoMitigation
        no_mitigation = _NoMitigation()
        assert isinstance(no_mitigation, QEMProtocol)

        # Test ZNE (with valid parameters)
        mock_factory = mocker.Mock(spec=Factory)

        def dummy_folding(circuit, scale_factor):
            return circuit

        folding_fn = partial(dummy_folding)

        zne = ZNE(
            scale_factors=[1.0, 2.0],
            folding_fn=folding_fn,
            extrapolation_factory=mock_factory,
        )
        assert isinstance(zne, QEMProtocol)


class TestNoMitigation:
    """Test suite for the _NoMitigation protocol."""

    @pytest.fixture
    def protocol(self):
        """Returns an instance of _NoMitigation."""
        return _NoMitigation()

    @pytest.fixture
    def circuit(self):
        """Returns a simple Cirq circuit."""
        return cirq.Circuit(cirq.X(cirq.LineQubit(0)))

    def test_name_property(self, protocol):
        """Test that the name property is correct."""
        assert protocol.name == "NoMitigation"

    def test_modify_circuit_is_identity(self, protocol, circuit):
        """Test that modify_circuit returns the original circuit in a list."""
        modified_circuits = protocol.modify_circuit(circuit)
        assert modified_circuits == [circuit]
        assert modified_circuits[0] is circuit  # Check it's the same object

    def test_postprocess_results_returns_single_value(self, protocol):
        """Test postprocess_results correctly returns the single result."""
        assert protocol.postprocess_results([1.23]) == 1.23
        assert protocol.postprocess_results([-0.5]) == -0.5

    def test_postprocess_results_raises_error_for_multiple_values(self, protocol):
        """Test postprocess_results raises RuntimeError for more than one result."""
        with pytest.raises(
            RuntimeError, match="NoMitigation class received multiple partial results."
        ):
            protocol.postprocess_results([1.0, 2.0])


class TestZNE:
    """Test suite for the ZNE (Zero Noise Extrapolation) class."""

    @pytest.fixture
    def mock_factory(self, mocker):
        """Create a mock Factory instance with a mock extrapolate method."""
        factory = mocker.Mock(spec=Factory)
        factory.extrapolate = mocker.Mock(return_value=0.85)
        return factory

    @pytest.fixture
    def mock_folding_fn(self):
        """Create a mock folding function as a partial."""

        def dummy_folding(circuit, scale_factor, some_param=None):
            return circuit

        return partial(dummy_folding, some_param="test")

    def test_initialization_valid(self, mock_folding_fn, mock_factory):
        """Test valid ZNE initialization."""
        zne_instance = ZNE(
            scale_factors=[1.0, 2.0, 3.0],
            folding_fn=mock_folding_fn,
            extrapolation_factory=mock_factory,
        )
        assert isinstance(zne_instance, ZNE)
        # Test with empty but valid scale factors
        ZNE(
            scale_factors=[],
            folding_fn=mock_folding_fn,
            extrapolation_factory=mock_factory,
        )

    def test_properties(self, mock_folding_fn, mock_factory):
        """Test that ZNE properties return the correct values."""
        scale_factors = [1.0, 3.0]
        zne = ZNE(
            scale_factors=scale_factors,
            folding_fn=mock_folding_fn,
            extrapolation_factory=mock_factory,
        )
        assert zne.name == "zne"
        assert zne.scale_factors == scale_factors
        assert zne.folding_fn == mock_folding_fn
        assert zne.extrapolation_factory == mock_factory

    @pytest.mark.parametrize(
        "invalid_factors",
        [
            pytest.param(1.0, id="not_a_sequence"),
            pytest.param([1.0, "two", 3.0], id="contains_non_numeric"),
            pytest.param([1.0, -2.0, 3.0], id="contains_negative_value"),
            pytest.param([1.0, 0.5, 2.0], id="contains_value_less_than_1"),
        ],
    )
    def test_initialization_invalid_scale_factors(
        self, invalid_factors, mock_folding_fn, mock_factory
    ):
        """Test ZNE initialization with various invalid scale_factors."""
        with pytest.raises(
            ValueError,
            match="scale_factors is expected to be a sequence of real numbers >=1",
        ):
            ZNE(
                scale_factors=invalid_factors,
                folding_fn=mock_folding_fn,
                extrapolation_factory=mock_factory,
            )

    @pytest.mark.parametrize(
        "invalid_fn",
        [
            pytest.param(lambda: None, id="not_a_partial"),
            pytest.param(None, id="is_None"),
            pytest.param("not_callable", id="is_string"),
        ],
    )
    def test_initialization_invalid_folding_fn(self, invalid_fn, mock_factory):
        """Test ZNE initialization with an invalid folding_fn."""
        with pytest.raises(
            ValueError, match="folding_fn is expected to be of type partial"
        ):
            ZNE(
                scale_factors=[1.0, 2.0],
                folding_fn=invalid_fn,
                extrapolation_factory=mock_factory,
            )

    def test_initialization_invalid_factory(self, mock_folding_fn):
        """Test ZNE initialization with an invalid extrapolation_factory."""
        with pytest.raises(
            ValueError, match="extrapolation_fn is expected to be of Factory"
        ):
            ZNE(
                scale_factors=[1.0, 2.0],
                folding_fn=mock_folding_fn,
                extrapolation_factory="not_a_factory_object",
            )

    def test_modify_circuit_calls_mitiq_construct_circuits(
        self, mocker, mock_folding_fn, mock_factory
    ):
        """modify_circuit delegates to mitiq with the configured parameters.

        Note: this is an implementation-coupling test. The method is a thin
        wrapper around mitiq's ``construct_circuits``, so we verify the
        delegation contract (correct kwargs) rather than the output, which
        is owned by mitiq.
        """
        mock_construct = mocker.patch(f"{qem.__name__}.construct_circuits")
        circuit = cirq.Circuit()
        scale_factors = [1.0, 2.0, 3.0]

        zne = ZNE(
            scale_factors=scale_factors,
            folding_fn=mock_folding_fn,
            extrapolation_factory=mock_factory,
        )
        zne.modify_circuit(circuit)

        mock_construct.assert_called_once_with(
            circuit,
            scale_factors=scale_factors,
            scale_method=mock_folding_fn,
        )

    def test_postprocess_results_calls_mitiq_combine_results(
        self, mocker, mock_folding_fn, mock_factory
    ):
        """postprocess_results delegates to mitiq with the configured parameters.

        Note: this is an implementation-coupling test. The method is a thin
        wrapper around mitiq's ``combine_results``, so we verify the
        delegation contract (correct kwargs) rather than the output, which
        is owned by mitiq.
        """
        mock_combine = mocker.patch(f"{qem.__name__}.combine_results")
        mock_combine.return_value = 0.95  # Set a return value to check

        scale_factors = [1.0, 2.0, 3.0]
        results = [0.9, 0.8, 0.7]

        zne = ZNE(
            scale_factors=scale_factors,
            folding_fn=mock_folding_fn,
            extrapolation_factory=mock_factory,
        )
        final_result = zne.postprocess_results(results)

        mock_combine.assert_called_once_with(
            scale_factors=scale_factors,
            results=results,
            extrapolation_method=mock_factory.extrapolate,
        )
        assert final_result == 0.95


class TestApplyProtocolToQasm:
    """Test suite for the apply_protocol_to_qasm orchestration function."""

    @pytest.fixture
    def simple_circuit(self):
        """Fixture for a simple single-qubit circuit with RX(0.5) and expval(PauliZ(0))."""
        ops = [qml.RX(0.5, wires=0)]
        return qml.tape.QuantumScript(ops=ops, measurements=[qml.expval(qml.PauliZ(0))])

    @pytest.fixture
    def default_measurement_group(self):
        """Fixture for the default measurement group with expval(PauliZ(0))."""
        return [[qml.expval(qml.PauliZ(0))]]

    def test_symbolic_qasm_without_symbols_raises(self, default_measurement_group):
        """Symbolic QASM should fail parsing if symbols are not declared."""
        theta = Symbol("theta")
        symbolic_circuit = qml.tape.QuantumScript(
            ops=[qml.RX(theta, wires=0)], measurements=[qml.expval(qml.PauliZ(0))]
        )
        circuits, _ = to_openqasm(
            symbolic_circuit,
            default_measurement_group,
            return_measurements_separately=True,
        )
        qasm_body = circuits[0]

        with pytest.raises(Exception):
            apply_protocol_to_qasm(qasm_body, _NoMitigation())

    def test_with_symbols(self, default_measurement_group):
        """apply_protocol_to_qasm works on symbolic circuits when symbols are provided."""
        theta = Symbol("theta")
        symbolic_circuit = qml.tape.QuantumScript(
            ops=[qml.RX(theta, wires=0)], measurements=[qml.expval(qml.PauliZ(0))]
        )
        circuits, _ = to_openqasm(
            symbolic_circuit,
            default_measurement_group,
            return_measurements_separately=True,
        )
        qasm_body = circuits[0]

        result = apply_protocol_to_qasm(
            qasm_body,
            _NoMitigation(),
            symbols=[theta],
        )

        assert isinstance(result, tuple) and len(result) == 1
        assert result[0][0] == (("qem", 0),)  # single (tag, body) pair
        assert "OPENQASM 2.0;" in result[0][1]

    def test_with_empty_symbols_list(self, simple_circuit, default_measurement_group):
        """Empty symbols list should be fine for non-symbolic QASM."""
        circuits, _ = to_openqasm(
            simple_circuit,
            default_measurement_group,
            return_measurements_separately=True,
        )
        qasm_body = circuits[0]

        result = apply_protocol_to_qasm(
            qasm_body,
            _NoMitigation(),
            symbols=[],
        )

        assert isinstance(result, tuple) and len(result) == 1
        assert isinstance(result[0], tuple) and len(result[0]) == 2
        assert isinstance(result[0][1], str)

    def test_with_sympy_array_symbols(self, default_measurement_group):
        """Numpy array of sympy symbols (from sp.symarray) are properly flattened and declared."""
        beta = sp.Symbol("beta")
        thetas = sp.symarray("theta", 3)

        ops = [
            qml.RX(beta, wires=0),
            qml.U3(*thetas, wires=1),
        ]
        circuit = qml.tape.QuantumScript(
            ops=ops, measurements=[qml.expval(qml.PauliZ(0))]
        )

        symbols = [beta, thetas]

        circuits, _ = to_openqasm(
            circuit,
            default_measurement_group,
            return_measurements_separately=True,
        )
        qasm_body = circuits[0]

        result = apply_protocol_to_qasm(
            qasm_body,
            _NoMitigation(),
            symbols=symbols,
        )

        assert isinstance(result, tuple) and len(result) == 1
        assert isinstance(result[0][1], str)

    def test_multiple_modified_circuits(
        self, simple_circuit, default_measurement_group
    ):
        """apply_protocol_to_qasm returns one QASM per modified cirq circuit."""

        class MockQEMProtocol:
            name = "mock"

            def modify_circuit(self, cirq_circuit):
                return [cirq_circuit, cirq_circuit]

        circuits, _ = to_openqasm(
            simple_circuit,
            default_measurement_group,
            return_measurements_separately=True,
        )
        qasm_body = circuits[0]

        result = apply_protocol_to_qasm(
            qasm_body,
            MockQEMProtocol(),
            symbols=[Symbol("theta")],
        )

        assert isinstance(result, tuple) and len(result) == 2
        for tag, qasm_str in result:
            assert isinstance(tag, tuple)
            assert isinstance(qasm_str, str)
            assert "OPENQASM 2.0;" in qasm_str

    def test_normalize_qasm_after_cirq_cleans_output(self):
        """normalize_qasm_after_cirq strips comments, collapses blank lines, and adds creg."""
        dirty = (
            "// Comment\n"
            "OPENQASM 2.0;\n"
            "\n"
            "\n"
            "qreg q[2];\n"
            "// Another comment\n"
            "rx(0.5) q[0];\n"
        )
        cleaned = normalize_qasm_after_cirq(dirty)

        assert "//" not in cleaned
        assert "\n\n" not in cleaned
        assert "qreg q[2];" in cleaned
        assert "creg c[2];" in cleaned
        assert "rx(0.5) q[0];" in cleaned


class TestInjectInputDeclarations:
    """Tests for _inject_input_declarations."""

    QASM_TEMPLATE = (
        'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\nrx(theta) q[0];\n'
    )

    def test_single_symbol_inserted_after_include(self):
        """A single symbol gets an 'input angle[32]' declaration after the include line."""
        theta = sp.Symbol("theta")
        result = _inject_input_declarations(self.QASM_TEMPLATE, [theta])
        assert "input angle[32] theta;" in result
        # Declaration must come before qreg
        assert result.index("input angle[32] theta;") < result.index("qreg")

    def test_multiple_symbols(self):
        """Multiple symbols each get their own declaration."""
        alpha, beta = sp.Symbol("alpha"), sp.Symbol("beta")
        result = _inject_input_declarations(self.QASM_TEMPLATE, [alpha, beta])
        assert "input angle[32] alpha;" in result
        assert "input angle[32] beta;" in result

    def test_numpy_array_of_symbols_flattened(self):
        """A numpy array of symbols is flattened before injection."""
        arr = sp.symarray("x", (2, 2))  # shape (2,2)
        result = _inject_input_declarations(self.QASM_TEMPLATE, [arr])
        for sym in arr.flatten():
            assert f"input angle[32] {sym};" in result

    def test_empty_symbols_returns_unchanged(self):
        """An empty symbol list returns the QASM body unchanged."""
        result = _inject_input_declarations(self.QASM_TEMPLATE, [])
        assert result == self.QASM_TEMPLATE

    def test_missing_include_marker_returns_unchanged(self):
        """If the include line is missing, the body is returned unchanged."""
        qasm_no_include = "OPENQASM 2.0;\nqreg q[1];\n"
        theta = sp.Symbol("theta")
        result = _inject_input_declarations(qasm_no_include, [theta])
        assert result == qasm_no_include

    def test_mixed_symbols_and_arrays(self):
        """A mix of bare symbols and numpy arrays are all injected."""
        alpha = sp.Symbol("alpha")
        betas = sp.symarray("beta", 2)
        result = _inject_input_declarations(self.QASM_TEMPLATE, [alpha, betas])
        assert "input angle[32] alpha;" in result
        assert "input angle[32] beta_0;" in result
        assert "input angle[32] beta_1;" in result
