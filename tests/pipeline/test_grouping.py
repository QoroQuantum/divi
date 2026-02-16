# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pennylane as qml
import pytest

from divi.circuits import MetaCircuit, measurements_to_qasm
from divi.pipeline._grouping import compute_measurement_groups


class TestComputeMeasurementGroups:
    """Tests for compute_measurement_groups."""

    def test_probs_returns_single_group_and_identity(self):
        """Probs measurement returns ((),), [[0]], and identity postprocessing."""
        circuit = qml.tape.QuantumScript(
            ops=[qml.Hadamard(0)], measurements=[qml.probs()]
        )
        measurement = circuit.measurements[0]

        groups, partition, postproc = compute_measurement_groups(measurement, "wires")

        assert groups == ((),)
        assert partition == [[0]]
        result = postproc(["dummy"])
        assert result == ["dummy"]  # identity returns input as-is

    def test_probs_with_none_strategy(self):
        """Probs with None strategy also returns single group."""
        circuit = qml.tape.QuantumScript(
            ops=[qml.Hadamard(0)], measurements=[qml.probs()]
        )
        measurement = circuit.measurements[0]

        groups, partition, postproc = compute_measurement_groups(measurement, None)

        assert groups == ((),)
        assert partition == [[0]]
        assert postproc(1.0) == 1.0

    def test_expval_single_term_wires(self):
        """Single-term expval with wires strategy."""
        obs = qml.PauliZ(0)
        circuit = qml.tape.QuantumScript(
            ops=[qml.Hadamard(0)], measurements=[qml.expval(obs)]
        )
        measurement = circuit.measurements[0]

        groups, partition, postproc = compute_measurement_groups(measurement, "wires")

        assert len(groups) == 1
        assert len(groups[0]) == 1
        assert partition == [[0]]
        energy = postproc([0.5])
        assert energy == 0.5

    def test_expval_single_term_qwc(self):
        """Single-term expval with qwc strategy."""
        obs = qml.PauliZ(0)
        circuit = qml.tape.QuantumScript(
            ops=[qml.Hadamard(0)], measurements=[qml.expval(obs)]
        )
        measurement = circuit.measurements[0]

        groups, partition, postproc = compute_measurement_groups(measurement, "qwc")

        assert len(groups) == 1
        assert len(groups[0]) == 1
        assert partition == [[0]]

    def test_expval_single_term_default(self):
        """Single-term expval with default strategy."""
        obs = qml.PauliZ(0)
        circuit = qml.tape.QuantumScript(
            ops=[qml.Hadamard(0)], measurements=[qml.expval(obs)]
        )
        measurement = circuit.measurements[0]

        groups, partition, _ = compute_measurement_groups(measurement, "default")

        assert len(groups) == 1
        assert len(groups[0]) == 1
        assert partition == [[0]]

    def test_expval_single_term_backend_expval(self):
        """Single-term expval with _backend_expval strategy."""
        obs = qml.PauliZ(0)
        circuit = qml.tape.QuantumScript(
            ops=[qml.Hadamard(0)], measurements=[qml.expval(obs)]
        )
        measurement = circuit.measurements[0]

        groups, partition, _ = compute_measurement_groups(
            measurement, "_backend_expval"
        )

        assert groups == ((),)
        assert partition == [[0]]

    def test_expval_single_term_none_strategy(self):
        """Single-term expval with None strategy (each obs own group)."""
        obs = qml.PauliZ(0)
        circuit = qml.tape.QuantumScript(
            ops=[qml.Hadamard(0)], measurements=[qml.expval(obs)]
        )
        measurement = circuit.measurements[0]

        groups, partition, _ = compute_measurement_groups(measurement, None)

        assert len(groups) == 1
        assert len(groups[0]) == 1
        assert partition == [[0]]

    def test_expval_multi_term_postprocessing(self):
        """Multi-term Hamiltonian postprocessing computes weighted sum."""
        obs = 0.5 * qml.PauliZ(0) + 0.3 * qml.PauliZ(1)
        circuit = qml.tape.QuantumScript(
            ops=[qml.Hadamard(0), qml.Hadamard(1)],
            measurements=[qml.expval(obs)],
        )
        measurement = circuit.measurements[0]

        groups, partition, postproc = compute_measurement_groups(measurement, None)

        # Each term is its own group with None strategy.
        assert len(groups) == 2
        assert len(partition) == 2
        # Simulated grouped results: [0.5, 0.3] for Z0 and Z1
        energy = postproc([0.5, 0.3])
        assert abs(energy - (0.5 * 0.5 + 0.3 * 0.3)) < 1e-10

    def test_unknown_strategy_raises(self):
        """Unknown strategy raises ValueError."""
        circuit = qml.tape.QuantumScript(
            ops=[qml.Hadamard(0)], measurements=[qml.expval(qml.PauliZ(0))]
        )
        measurement = circuit.measurements[0]

        with pytest.raises(ValueError, match="Unknown grouping strategy"):
            compute_measurement_groups(measurement, "invalid_strategy")


class TestMetaCircuitWithGrouping:
    """Tests for MetaCircuit + compute_measurement_groups + measurements_to_qasm (no compile_metacircuit)."""

    def test_wires_and_explicit_empty_group_produce_same_measurement_qasm(self):
        """measurement_groups from 'wires' vs explicit ((),) produce same measurement QASM for probs."""
        ops = [qml.Hadamard(0), qml.CNOT(wires=[0, 1])]
        circuit = qml.tape.QuantumScript(ops=ops, measurements=[qml.probs()])
        symbols = np.array([])
        meta = MetaCircuit(source_circuit=circuit, symbols=symbols)

        meas_groups_wires, _, _ = compute_measurement_groups(
            circuit.measurements[0], "wires"
        )
        meas_qasms_wires = measurements_to_qasm(
            circuit, list(meas_groups_wires), precision=meta.precision
        )
        meas_qasms_explicit = measurements_to_qasm(
            circuit, [()], precision=meta.precision
        )

        assert len(meas_qasms_wires) == len(meas_qasms_explicit) == 1
        assert meas_qasms_wires[0] == meas_qasms_explicit[0]
        assert "measure" in meas_qasms_wires[0]


class TestParameterFreeMetaCircuit:
    """Tests for parameter-free MetaCircuit with set_measurement_bodies (no compile_metacircuit)."""

    def test_meta_with_measurement_bodies_produces_valid_full_qasm(self):
        """MetaCircuit with set_measurement_bodies yields valid body+meas QASM."""
        ops = [qml.Hadamard(0), qml.CNOT(wires=[0, 1])]
        circuit = qml.tape.QuantumScript(ops=ops, measurements=[qml.probs()])
        symbols = np.array([], dtype=object)

        meta = MetaCircuit(source_circuit=circuit, symbols=symbols)
        meas_qasms = measurements_to_qasm(
            meta.source_circuit, [()], precision=meta.precision
        )
        meta.set_measurement_bodies((((), meas_qasms[0]),))

        body = meta.circuit_body_qasms[0][1]
        meas = meta.measurement_qasms[0][1]
        full_qasm = body + meas

        assert "OPENQASM" in full_qasm
        assert "qreg" in full_qasm
        assert "h " in full_qasm or "h(" in full_qasm
        assert "cx " in full_qasm or "cx(" in full_qasm
        assert "measure" in full_qasm

    def test_parameter_free_qasm_not_corrupted(self):
        """Parameter-free QASM must not be corrupted by empty regex substitution."""
        ops = [qml.Hadamard(0), qml.RZ(0.5, wires=0)]
        circuit = qml.tape.QuantumScript(ops=ops, measurements=[qml.probs()])
        symbols = np.array([], dtype=object)

        meta = MetaCircuit(source_circuit=circuit, symbols=symbols)
        meas_qasms = measurements_to_qasm(
            meta.source_circuit, [()], precision=meta.precision
        )
        meta.set_measurement_bodies((((), meas_qasms[0]),))

        full_qasm = meta.circuit_body_qasms[0][1] + meta.measurement_qasms[0][1]

        assert full_qasm.count("h ") + full_qasm.count("h(") >= 1
        assert "rz(0.5" in full_qasm or "rz(0.50000000" in full_qasm
        assert len(full_qasm) < 500
