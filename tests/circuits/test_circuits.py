# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import re
from functools import partial

import dill
import numpy as np
import pennylane as qml
import pytest
import sympy as sp
from mitiq.zne.inference import ExpFactory
from mitiq.zne.scaling import fold_global

from divi.circuits import CircuitBundle, ExecutableQASMCircuit, MetaCircuit, to_openqasm
from divi.circuits.qem import ZNE, _NoMitigation


class TestCircuitBundle:
    def test_bundle_creation(self):
        ops = [
            qml.RX(0.5, wires=0),
            qml.CNOT(wires=[0, 1]),
        ]
        qscript = qml.tape.QuantumScript(
            ops=ops, measurements=[qml.expval(qml.PauliZ(0))]
        )
        qasm_list = to_openqasm(
            qscript,
            measurement_groups=[qscript.measurements],
            return_measurements_separately=False,
        )
        assert len(qasm_list) == 1

        tag = "test_bundle"
        executables = (ExecutableQASMCircuit(tag=tag, qasm=qasm_list[0]),)
        bundle = CircuitBundle(executables=executables)

        # Check basic attributes
        assert bundle.tags == [tag]
        assert bundle.qasm_circuits == qasm_list
        assert len(bundle.executables) == 1


class TestMetaCircuit:
    @pytest.fixture
    def weights_syms(self):
        return sp.symarray("w", 4)

    @pytest.fixture
    def sample_circuit(self, weights_syms):
        ops = [
            qml.AngleEmbedding(weights_syms, wires=range(4), rotation="Y"),
            qml.AngleEmbedding(weights_syms, wires=range(4), rotation="X"),
        ]
        return qml.tape.QuantumScript(ops=ops, measurements=[qml.probs()])

    @pytest.fixture
    def expval_circuit(self, weights_syms):
        ops = [
            qml.AngleEmbedding(weights_syms, wires=range(4), rotation="Y"),
        ]
        return qml.tape.QuantumScript(ops=ops, measurements=[qml.expval(qml.PauliZ(0))])

    @pytest.fixture
    def no_measurement_circuit(self, weights_syms):
        ops = [
            qml.AngleEmbedding(weights_syms, wires=range(4), rotation="Y"),
        ]
        return qml.tape.QuantumScript(ops=ops, measurements=[])

    def test_metacircuit_valid_measurement(self, expval_circuit, weights_syms):
        """Tests that MetaCircuit initializes correctly with a valid expval measurement."""
        # This should execute without raising an exception
        try:
            MetaCircuit(source_circuit=expval_circuit, symbols=weights_syms)
        except ValueError:
            pytest.fail("MetaCircuit initialization failed with a valid measurement.")

    def test_metacircuit_raises_on_no_measurements(
        self, no_measurement_circuit, weights_syms
    ):
        """Tests that MetaCircuit raises ValueError for a circuit with no measurements."""
        with pytest.raises(
            ValueError,
            match="MetaCircuit requires a circuit with exactly one measurement, but 0 were found.",
        ):
            MetaCircuit(source_circuit=no_measurement_circuit, symbols=weights_syms)

    def test_correct_symbolization(self, sample_circuit, weights_syms):

        meta_circuit = MetaCircuit(source_circuit=sample_circuit, symbols=weights_syms)

        assert meta_circuit.source_circuit == sample_circuit

        # Ensure we have all the symbols
        np.testing.assert_equal(meta_circuit.symbols, weights_syms)

        # Make sure the compiled circuit is correct
        circ_pattern = r"w_(\d+)"
        assert (
            len(set(re.findall(circ_pattern, meta_circuit._compiled_circuit_bodies[0])))
            == 4
        )
        assert (
            len(re.findall(circ_pattern, meta_circuit._compiled_circuit_bodies[0])) == 8
        )

        # Make sure the measurement qasm is correct
        assert len(meta_circuit._measurements) == 1
        meas_pattern = r"measure q\[(\d+)\] -> c\[(\d+)\];"
        assert len(re.findall(meas_pattern, meta_circuit._measurements[0])) == 4

    def test_correct_initialization_no_mitigation(
        self, mocker, sample_circuit, weights_syms
    ):

        meta_circuit = MetaCircuit(
            source_circuit=sample_circuit,
            symbols=weights_syms,
            qem_protocol=_NoMitigation(),
        )

        param_list = [0.123456789, 0.212345678, 0.312345678, 0.412345678]
        tag_prefix = "test"
        precision = 8

        method_mock = mocker.patch("divi.circuits.to_openqasm")

        circuit = meta_circuit.initialize_circuit_from_params(
            param_list, tag_prefix=tag_prefix, precision=precision
        )

        # Ensure converter wasn't called since
        # we are already providing the qasm
        method_mock.assert_not_called()

        # Check the new Circuit object
        assert circuit.tags == [f"{tag_prefix}_NoMitigation:0_0"]
        assert len(circuit.qasm_circuits) == 1

        # Ensure no more symbols exist
        symbols_pattern = r"w_(\d+)"
        assert len(re.findall(symbols_pattern, circuit.qasm_circuits[0])) == 0

        # Ensure the symbols are correctly replaced
        params_pattern = r"r[yx]\(([-+]?\d*\.?\d+)\)"
        actual_params = re.findall(params_pattern, circuit.qasm_circuits[0])

        for actual, expected in zip(actual_params, param_list * 2):
            assert round(expected, precision) == float(actual)

    scale_factors = [1, 3, 5]

    @pytest.mark.parametrize(
        "qem_protocol,expected_tags,expected_n_circuits",
        [
            (_NoMitigation(), ["test_NoMitigation:0_0"], 1),
            (
                ZNE(
                    folding_fn=partial(fold_global),
                    scale_factors=scale_factors,
                    extrapolation_factory=ExpFactory(scale_factors=scale_factors),
                ),
                [f"test_zne:{i}_0" for i in range(len(scale_factors))],
                3,
            ),
        ],
    )
    def test_correct_initialization(
        self,
        mocker,
        sample_circuit,
        weights_syms,
        qem_protocol,
        expected_tags,
        expected_n_circuits,
    ):
        meta_circuit = MetaCircuit(
            source_circuit=sample_circuit,
            symbols=weights_syms,
            qem_protocol=qem_protocol,
        )

        param_list = [0.123456789, 0.212345678, 0.312345678, 0.412345678]
        tag_prefix = "test"
        precision = 8

        method_mock = mocker.patch("divi.circuits.to_openqasm")

        circuit = meta_circuit.initialize_circuit_from_params(
            param_list, tag_prefix=tag_prefix, precision=precision
        )

        # Ensure converter wasn't called since
        # we are already providing the qasm
        method_mock.assert_not_called()

        # Check the new Circuit object
        assert circuit.tags == expected_tags
        assert len(circuit.qasm_circuits) == expected_n_circuits

        # Ensure no more symbols exist
        symbols_pattern = r"w_(\d+)"
        assert all(
            len(re.findall(symbols_pattern, curr_qasm)) == 0
            for curr_qasm in circuit.qasm_circuits
        )

        # Ensure the symbols are correctly replaced
        params_pattern = r"r[yx]\(([-+]?\d*\.?\d+)\)"
        for curr_qasm in circuit.qasm_circuits:
            actual_params = re.findall(params_pattern, curr_qasm)

            for actual, expected in zip(actual_params, param_list * 2):
                assert round(expected, precision) == float(actual)

    def test_metacircuit_with_qem_is_serializable(self, expval_circuit, weights_syms):
        """
        Ensures MetaCircuit with a QEM protocol can be pickled and unpickled correctly.
        """
        scale_factors = [1, 3, 5]
        qem_protocol = ZNE(
            folding_fn=partial(fold_global),
            scale_factors=scale_factors,
            extrapolation_factory=ExpFactory(scale_factors=scale_factors),
        )

        meta_circuit_original = MetaCircuit(
            source_circuit=expval_circuit,
            symbols=weights_syms,
            qem_protocol=qem_protocol,
        )

        # Serialize and deserialize the object
        pickled_mc = dill.dumps(meta_circuit_original)
        meta_circuit_unpickled = dill.loads(pickled_mc)

        # Assert that key attributes are preserved
        assert meta_circuit_unpickled.qem_protocol.name == "zne"
        assert len(meta_circuit_unpickled._compiled_circuit_bodies) == len(
            scale_factors
        )
        np.testing.assert_equal(meta_circuit_unpickled.symbols, weights_syms)

        # Crucially, test that the unpickled function works
        # The expval_circuit has one measurement group.
        mock_results = [0.5]  # Mock result for the single measurement group
        # The specific value doesn't matter, just that it doesn't raise an error
        assert isinstance(
            meta_circuit_unpickled.postprocessing_fn(mock_results), np.floating
        )

    def test_metacircuit_precision_default(self, expval_circuit, weights_syms):
        """Test that MetaCircuit defaults to precision=8 when not specified."""
        meta_circuit = MetaCircuit(
            source_circuit=expval_circuit,
            symbols=weights_syms,
        )
        assert meta_circuit.precision == 8

    def test_metacircuit_precision_custom(self, expval_circuit, weights_syms):
        """Test that MetaCircuit accepts a custom precision value."""
        meta_circuit = MetaCircuit(
            source_circuit=expval_circuit,
            symbols=weights_syms,
            precision=12,
        )
        assert meta_circuit.precision == 12

    def test_metacircuit_precision_used_in_qasm_conversion(
        self, mocker, expval_circuit, weights_syms
    ):
        """Test that precision is passed to to_openqasm during MetaCircuit initialization."""
        mock_to_openqasm = mocker.patch("divi.circuits._core.to_openqasm")
        mock_to_openqasm.return_value = (["circuit_body"], ["measurement"])

        MetaCircuit(
            source_circuit=expval_circuit,
            symbols=weights_syms,
            precision=6,
        )

        # Verify to_openqasm was called with precision=6
        assert mock_to_openqasm.called
        call_kwargs = mock_to_openqasm.call_args[1]
        assert call_kwargs["precision"] == 6

    def test_initialize_circuit_from_params_uses_metacircuit_precision(
        self, sample_circuit, weights_syms
    ):
        """Test that initialize_circuit_from_params uses MetaCircuit precision when not specified."""
        meta_circuit = MetaCircuit(
            source_circuit=sample_circuit,
            symbols=weights_syms,
            precision=4,
        )

        param_list = [0.123456789, 0.212345678, 0.312345678, 0.412345678]
        circuit = meta_circuit.initialize_circuit_from_params(param_list)

        # Check that parameters are formatted with precision=4
        params_pattern = r"r[yx]\(([-+]?\d*\.?\d+)\)"
        actual_params = re.findall(params_pattern, circuit.qasm_circuits[0])

        for actual, expected in zip(actual_params, param_list * 2):
            # Should be rounded to 4 decimal places
            assert len(actual.split(".")[1]) == 4 if "." in actual else True
            assert round(expected, 4) == float(actual)

    def test_initialize_circuit_from_params_overrides_precision(
        self, sample_circuit, weights_syms
    ):
        """Test that initialize_circuit_from_params can override MetaCircuit precision."""
        meta_circuit = MetaCircuit(
            source_circuit=sample_circuit,
            symbols=weights_syms,
            precision=4,  # MetaCircuit precision
        )

        param_list = [0.123456789, 0.212345678, 0.312345678, 0.412345678]
        # Override with precision=6
        circuit = meta_circuit.initialize_circuit_from_params(param_list, precision=6)

        # Check that parameters are formatted with precision=6 (overridden value)
        params_pattern = r"r[yx]\(([-+]?\d*\.?\d+)\)"
        actual_params = re.findall(params_pattern, circuit.qasm_circuits[0])

        for actual, expected in zip(actual_params, param_list * 2):
            # Should be rounded to 6 decimal places
            assert len(actual.split(".")[1]) == 6 if "." in actual else True
            assert round(expected, 6) == float(actual)
