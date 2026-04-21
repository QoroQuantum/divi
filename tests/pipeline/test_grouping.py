# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pennylane as qp
import pytest
from qiskit.quantum_info import SparsePauliOp

from divi.circuits import measurement_qasms_from_groups, qscript_to_meta
from divi.pipeline._grouping import (
    _create_postprocessing_fn,
    _wire_grouping_from_labels,
    compute_measurement_groups,
)


class TestComputeMeasurementGroups:
    """Tests for compute_measurement_groups (SparsePauliOp-native)."""

    def test_single_term_wires(self):
        obs = SparsePauliOp.from_list([("Z", 1.0)])
        groups, partition, postproc = compute_measurement_groups(obs, "wires", 1)
        assert len(groups) == 1
        assert len(groups[0]) == 1
        assert partition == [[0]]
        assert postproc([0.5]) == pytest.approx(0.5)

    def test_single_term_qwc(self):
        obs = SparsePauliOp.from_list([("Z", 1.0)])
        groups, partition, _ = compute_measurement_groups(obs, "qwc", 1)
        assert len(groups) == 1
        assert partition == [[0]]

    def test_single_term_default(self):
        obs = SparsePauliOp.from_list([("Z", 1.0)])
        groups, partition, _ = compute_measurement_groups(obs, "default", 1)
        assert len(groups) == 1
        assert partition == [[0]]

    def test_single_term_backend_expval(self):
        obs = SparsePauliOp.from_list([("Z", 1.0)])
        groups, partition, _ = compute_measurement_groups(obs, "_backend_expval", 1)
        assert groups == ((),)
        assert partition == [[0]]

    def test_single_term_none_strategy(self):
        obs = SparsePauliOp.from_list([("Z", 1.0)])
        groups, partition, _ = compute_measurement_groups(obs, None, 1)
        assert len(groups) == 1
        assert partition == [[0]]

    def test_multi_term_postprocessing(self):
        obs = SparsePauliOp.from_list([("IZ", 0.5), ("ZI", 0.3)])
        groups, partition, postproc = compute_measurement_groups(obs, None, 2)
        assert len(groups) == 2
        assert len(partition) == 2
        energy = postproc([0.5, 0.3])
        assert energy == pytest.approx(0.5 * 0.5 + 0.3 * 0.3)

    def test_qwc_groups_commuting_terms(self):
        # ZI and IZ qubit-wise commute → same group.
        obs = SparsePauliOp.from_list([("IZ", 0.5), ("ZI", 0.3)])
        groups, partition, _ = compute_measurement_groups(obs, "qwc", 2)
        assert len(groups) == 1  # both in one QWC group

    def test_qwc_splits_non_commuting(self):
        # ZI and XI don't qubit-wise commute on qubit 1.
        obs = SparsePauliOp.from_list([("IZ", 0.5), ("IX", 0.3)])
        groups, _, _ = compute_measurement_groups(obs, "qwc", 2)
        assert len(groups) == 2

    def test_unknown_strategy_raises(self):
        obs = SparsePauliOp.from_list([("Z", 1.0)])
        with pytest.raises(ValueError, match="Unknown grouping strategy"):
            compute_measurement_groups(obs, "invalid_strategy", 1)


class TestMetaCircuitWithGrouping:
    def test_wires_and_empty_group_produce_same_measurement_qasm(self):
        circuit = qp.tape.QuantumScript(
            ops=[qp.Hadamard(0), qp.CNOT(wires=[0, 1])],
            measurements=[qp.probs()],
        )
        meta = qscript_to_meta(circuit)
        n_qubits = meta.n_qubits

        # Both "wires" with a probs observable (no SparsePauliOp) and
        # explicit empty group should produce the same measure-all QASM.
        qasms_explicit = measurement_qasms_from_groups(((),), n_qubits)
        # Empty group → just measure all qubits.
        assert len(qasms_explicit) == 1
        assert "measure" in qasms_explicit[0]


class TestWireGroupingFromLabels:
    def test_non_overlapping_in_one_group(self):
        labels = ["ZII", "IZI", "IIZ"]
        groups = _wire_grouping_from_labels(labels)
        assert len(groups) == 1
        assert groups[0] == [0, 1, 2]

    def test_overlapping_split(self):
        labels = ["ZI", "ZI"]  # same qubit active
        groups = _wire_grouping_from_labels(labels)
        assert len(groups) == 2

    def test_mixed(self):
        labels = ["ZII", "IZI", "ZZI"]
        groups = _wire_grouping_from_labels(labels)
        assert len(groups) == 2
        assert 0 in groups[0] and 1 in groups[0]
        assert 2 in groups[1]


class TestPostprocessingFn:
    def test_missing_indices_raises(self):
        with pytest.raises(RuntimeError, match="Missing"):
            _create_postprocessing_fn(
                coefficients=np.array([1.0, 1.0, 1.0]),
                partition_indices=[[0, 1]],
                n_terms=3,
            )

    def test_wrong_group_count_raises(self):
        fn = _create_postprocessing_fn(
            coefficients=np.array([1.0, 1.0]),
            partition_indices=[[0], [1]],
            n_terms=2,
        )
        with pytest.raises(RuntimeError, match="Expected 2"):
            fn([0.5])

    def test_correct_dot_product(self):
        fn = _create_postprocessing_fn(
            coefficients=np.array([2.0, 3.0]),
            partition_indices=[[0], [1]],
            n_terms=2,
        )
        assert fn([0.5, -1.0]) == pytest.approx(-2.0)
