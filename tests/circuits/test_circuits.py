# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import re

import pennylane as qml
import pytest
import sympy as sp

from divi.circuits import (
    MetaCircuit,
)


class TestMetaCircuit:
    """Tests for MetaCircuit (source_circuit, symbols, precision, circuit_body_qasms, setters)."""

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
        """MetaCircuit initializes correctly with a valid expval measurement."""
        meta = MetaCircuit(source_circuit=expval_circuit, symbols=weights_syms)
        assert meta.source_circuit is expval_circuit
        assert len(meta.circuit_body_qasms) == 1
        assert meta.circuit_body_qasms[0][0] == ()
        assert "OPENQASM 2.0" in meta.circuit_body_qasms[0][1]

    def test_metacircuit_raises_on_no_measurements(
        self, no_measurement_circuit, weights_syms
    ):
        """MetaCircuit raises ValueError for a circuit with no measurements."""
        with pytest.raises(
            ValueError,
            match="MetaCircuit requires a circuit with exactly one measurement, but 0 were found.",
        ):
            MetaCircuit(source_circuit=no_measurement_circuit, symbols=weights_syms)

    def test_metacircuit_post_init_sets_circuit_body_qasms_and_measurement_groups(
        self, sample_circuit, weights_syms
    ):
        """__post_init__ sets circuit_body_qasms (body contains symbol names) and measurement_groups."""
        meta = MetaCircuit(source_circuit=sample_circuit, symbols=weights_syms)
        assert meta.measurement_groups == ()
        assert len(meta.circuit_body_qasms) == 1
        body = meta.circuit_body_qasms[0][1]
        assert "OPENQASM 2.0" in body
        # Symbol names appear in body (e.g. w_0, w_1)
        sym_pattern = r"w_(\d+)"
        found = re.findall(sym_pattern, body)
        assert len(set(found)) == 4
        assert len(found) == 8  # two AngleEmbeddings × 4 params

    def test_metacircuit_precision_default(self, expval_circuit, weights_syms):
        """MetaCircuit defaults to precision=8 when not specified."""
        meta = MetaCircuit(
            source_circuit=expval_circuit,
            symbols=weights_syms,
        )
        assert meta.precision == 8

    def test_metacircuit_precision_custom(self, expval_circuit, weights_syms):
        """MetaCircuit accepts a custom precision value."""
        meta = MetaCircuit(
            source_circuit=expval_circuit,
            symbols=weights_syms,
            precision=12,
        )
        assert meta.precision == 12

    def test_metacircuit_precision_passed_to_circuit_body_to_qasm(
        self, mocker, expval_circuit, weights_syms
    ):
        """MetaCircuit __post_init__ calls circuit_body_to_qasm with precision."""
        mock_body = mocker.patch("divi.circuits._core.circuit_body_to_qasm")
        mock_body.return_value = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[4];creg c[4];\nry(w_0) q[0];\n'

        meta = MetaCircuit(
            source_circuit=expval_circuit,
            symbols=weights_syms,
            precision=6,
        )

        assert mock_body.called
        assert mock_body.call_args[1]["precision"] == 6

    def test_set_circuit_bodies_overrides_body(self, expval_circuit, weights_syms):
        """set_circuit_bodies overwrites circuit_body_qasms and returns self."""
        meta = MetaCircuit(source_circuit=expval_circuit, symbols=weights_syms)
        new_bodies = (((("qem", 0),), "custom_body_qasm"),)
        out = meta.set_circuit_bodies(new_bodies)
        assert out is meta
        assert meta.circuit_body_qasms == new_bodies

    def test_set_measurement_bodies_sets_measurement_qasms(
        self, expval_circuit, weights_syms
    ):
        """set_measurement_bodies sets measurement_qasms (e.g. for pipeline stages)."""
        meta = MetaCircuit(source_circuit=expval_circuit, symbols=weights_syms)
        meas_bodies = (((("obs_group", 0),), "measure q[0] -> c[0];\n"),)
        out = meta.set_measurement_bodies(meas_bodies)
        assert out is meta
        assert meta.measurement_qasms == meas_bodies

    def test_set_measurement_groups(self, expval_circuit, weights_syms):
        """set_measurement_groups overwrites measurement_groups and returns self."""
        meta = MetaCircuit(source_circuit=expval_circuit, symbols=weights_syms)
        groups = ((qml.PauliZ(0),),)
        out = meta.set_measurement_groups(groups)
        assert out is meta
        assert meta.measurement_groups == groups
