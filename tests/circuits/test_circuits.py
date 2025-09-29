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

from divi.circuits import Circuit, MetaCircuit
from divi.circuits.qem import ZNE, _NoMitigation


class TestCircuit:
    def test_pennylane_circuit_initialization(self, mocker):

        def test_circuit():
            qml.RX(0.5, wires=0)
            qml.RX(0.5, wires=1)
            qml.RY(0.5, wires=2)
            qml.RZ(0.25, wires=3)

            return qml.expval(
                qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliX(2) @ qml.PauliY(3)
            )

        qscript = qml.tape.make_qscript(test_circuit)()

        Circuit._id_counter = 0
        circ_1 = Circuit(qscript, tags=["test_circ"])

        # Check basic attributes
        assert circ_1.main_circuit == qscript
        assert circ_1.tags == ["test_circ"]
        assert circ_1.circuit_id == 0
        assert len(circ_1.qasm_circuits) == 1

        # Ensure converter was called
        method_mock = mocker.patch("divi.circuits._core.to_openqasm")

        Circuit(qscript, tags=["test_circ"])

        method_mock.assert_called_once()


class TestMetaCircuit:
    @pytest.fixture
    def weights_syms(self):
        return sp.symarray("w", 4)

    @pytest.fixture
    def sample_circuit(self, weights_syms):
        def circ(weights):
            qml.AngleEmbedding(weights, wires=range(4), rotation="Y")
            qml.AngleEmbedding(weights, wires=range(4), rotation="X")

            return qml.probs()

        return qml.tape.make_qscript(circ)(weights_syms)

    def test_correct_symbolization(self, sample_circuit, weights_syms):

        meta_circuit = MetaCircuit(sample_circuit, weights_syms)

        assert meta_circuit.main_circuit == sample_circuit

        # Ensure we have all the symbols
        np.testing.assert_equal(meta_circuit.symbols, weights_syms)

        # Make sure the compiled circuit is correct
        circ_pattern = r"w_(\d+)"
        assert (
            len(set(re.findall(circ_pattern, meta_circuit.compiled_circuits_bodies[0])))
            == 4
        )
        assert (
            len(re.findall(circ_pattern, meta_circuit.compiled_circuits_bodies[0])) == 8
        )

        # Make sure the measurement qasm is correct
        assert len(meta_circuit.measurements) == 1
        meas_pattern = r"measure q\[(\d+)\] -> c\[(\d+)\];"
        assert len(re.findall(meas_pattern, meta_circuit.measurements[0])) == 4

    def test_correct_initialization_no_mitigation(
        self, mocker, sample_circuit, weights_syms
    ):

        meta_circuit = MetaCircuit(
            sample_circuit, weights_syms, qem_protocol=_NoMitigation()
        )

        param_list = [0.123456789, 0.212345678, 0.312345678, 0.412345678]
        tag_prefix = "test"
        precision = 8

        method_mock = mocker.patch("divi.circuits.qasm.to_openqasm")

        circuit = meta_circuit.initialize_circuit_from_params(
            param_list, tag_prefix=tag_prefix, precision=precision
        )

        # Ensure converter wasn't called since
        # we are already providing the qasm
        method_mock.assert_not_called()

        # Check the new Circuit object
        assert circuit.main_circuit == sample_circuit
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
            sample_circuit, weights_syms, qem_protocol=qem_protocol
        )

        param_list = [0.123456789, 0.212345678, 0.312345678, 0.412345678]
        tag_prefix = "test"
        precision = 8

        method_mock = mocker.patch("divi.circuits.qasm.to_openqasm")

        circuit = meta_circuit.initialize_circuit_from_params(
            param_list, tag_prefix=tag_prefix, precision=precision
        )

        # Ensure converter wasn't called since
        # we are already providing the qasm
        method_mock.assert_not_called()

        # Check the new Circuit object
        assert circuit.main_circuit == sample_circuit
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

    def test_metacircuit_with_qem_is_serializable(self, sample_circuit, weights_syms):
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
            sample_circuit, weights_syms, qem_protocol=qem_protocol
        )

        # Serialize and deserialize the object
        pickled_mc = dill.dumps(meta_circuit_original)
        meta_circuit_unpickled = dill.loads(pickled_mc)

        # Assert that key attributes are preserved
        assert meta_circuit_unpickled.qem_protocol.name == "zne"
        assert len(meta_circuit_unpickled.compiled_circuits_bodies) == len(
            scale_factors
        )
        np.testing.assert_equal(meta_circuit_unpickled.symbols, weights_syms)

        # Crucially, test that the unpickled function works
        mock_results = [[0.5], [0.4], [0.3]]  # Mock results for 3 measurement groups
        # The specific value doesn't matter, just that it doesn't raise an error
        assert isinstance(meta_circuit_unpickled.postprocessing_fn(mock_results), tuple)
