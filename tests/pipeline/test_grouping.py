# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pennylane as qml
import pytest

from divi.circuits import MetaCircuit, measurements_to_qasm
from divi.pipeline._grouping import (
    _create_final_postprocessing_fn,
    _extract_coeffs,
    _wire_grouping,
    compute_measurement_groups,
)


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


class TestExtractCoeffs:
    """Tests for _extract_coeffs covering nested SProd (L17-19)."""

    def test_single_operator_returns_one(self):
        """A plain PauliZ (no SProd, no Hamiltonian) should return [1.0]."""
        obs = qml.PauliZ(0)
        assert _extract_coeffs(obs) == [1.0]

    def test_sprod_single_operator(self):
        """SProd wrapping a single operator extracts the scalar."""
        obs = qml.s_prod(3.0, qml.PauliZ(0))
        assert _extract_coeffs(obs) == pytest.approx([3.0])

    def test_nested_sprod_with_hamiltonian(self):
        """Nested SProd wrapping a Hamiltonian multiplies through all scalars."""
        inner_ham = qml.Hamiltonian([0.5, 0.3], [qml.PauliZ(0), qml.PauliX(1)])
        # 2.0 * (0.5 Z0 + 0.3 X1) → coeffs should be [1.0, 0.6]
        obs = qml.s_prod(2.0, inner_ham)
        result = _extract_coeffs(obs)
        assert result == pytest.approx([1.0, 0.6])

    def test_double_nested_sprod(self):
        """Double nested SProd: 2.0 * (3.0 * Z0) → [6.0]."""
        obs = qml.s_prod(2.0, qml.s_prod(3.0, qml.PauliZ(0)))
        assert _extract_coeffs(obs) == pytest.approx([6.0])

    def test_hamiltonian_without_sprod(self):
        """Hamiltonian without SProd wrapping returns its own coefficients."""
        ham = qml.Hamiltonian([0.7, -0.4], [qml.PauliZ(0), qml.PauliZ(1)])
        assert _extract_coeffs(ham) == pytest.approx([0.7, -0.4])


class TestWireGrouping:
    """Tests for _wire_grouping covering the grouping loop (L48-54)."""

    def test_non_overlapping_measurements_grouped_together(self):
        """Measurements on non-overlapping wires should be in the same group."""
        mps = [
            qml.expval(qml.PauliZ(0)),
            qml.expval(qml.PauliZ(1)),
            qml.expval(qml.PauliZ(2)),
        ]
        partition_indices, mp_groups = _wire_grouping(mps)

        # All three on different wires → should fit in one group
        assert len(mp_groups) == 1
        assert len(mp_groups[0]) == 3
        assert partition_indices == [[0, 1, 2]]

    def test_overlapping_measurements_split_into_groups(self):
        """Measurements on overlapping wires should be in separate groups."""
        mps = [
            qml.expval(qml.PauliZ(0)),
            qml.expval(qml.PauliZ(0)),  # same wire as first
        ]
        partition_indices, mp_groups = _wire_grouping(mps)

        assert len(mp_groups) == 2
        assert partition_indices == [[0], [1]]

    def test_mixed_overlapping_and_non_overlapping(self):
        """Mix of overlapping and non-overlapping wires."""
        mps = [
            qml.expval(qml.PauliZ(0)),  # Group 0
            qml.expval(qml.PauliZ(1)),  # Group 0 (non-overlapping with wire 0)
            qml.expval(qml.PauliZ(0) @ qml.PauliZ(1)),  # Group 1 (overlaps both)
        ]
        partition_indices, mp_groups = _wire_grouping(mps)

        assert len(mp_groups) == 2
        # First two non-overlapping → same group; third overlaps → new group
        assert 0 in partition_indices[0]
        assert 1 in partition_indices[0]
        assert 2 in partition_indices[1]


class TestPostprocessingFnErrors:
    """Tests for _create_final_postprocessing_fn error paths (L74-88)."""

    def test_missing_indices_raises(self):
        """partition_indices that don't cover all observables raises RuntimeError."""
        # 3 total observables but only indices 0, 1 covered
        with pytest.raises(RuntimeError, match="Missing indices"):
            _create_final_postprocessing_fn(
                coefficients=[1.0, 1.0, 1.0],
                partition_indices=[[0, 1]],  # missing index 2
                num_total_obs=3,
            )

    def test_wrong_group_count_raises(self):
        """Wrong number of grouped results raises RuntimeError."""
        postproc = _create_final_postprocessing_fn(
            coefficients=[1.0, 1.0],
            partition_indices=[[0], [1]],
            num_total_obs=2,
        )
        with pytest.raises(RuntimeError, match="Expected 2 grouped results"):
            postproc([0.5])  # only 1 group instead of 2

    def test_correct_postprocessing(self):
        """A correct setup computes the dot product properly."""
        postproc = _create_final_postprocessing_fn(
            coefficients=[2.0, 3.0],
            partition_indices=[[0], [1]],
            num_total_obs=2,
        )
        result = postproc([0.5, -1.0])
        # 2.0 * 0.5 + 3.0 * (-1.0) = 1.0 - 3.0 = -2.0
        assert result == pytest.approx(-2.0)
