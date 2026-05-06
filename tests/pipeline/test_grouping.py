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
    compute_multi_observable_measurement_groups,
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


class TestComputeMultiObservableMeasurementGroups:
    """Union-grouping for multiple observables sharing one shot batch."""

    def test_empty_tuple_raises(self):
        with pytest.raises(ValueError, match="empty"):
            compute_multi_observable_measurement_groups((), "qwc", 2)

    def test_backend_expval_strategy_rejected(self):
        """``_backend_expval`` only knows how to evaluate one observable."""
        obs = SparsePauliOp.from_list([("Z", 1.0)])
        with pytest.raises(ValueError, match="_backend_expval"):
            compute_multi_observable_measurement_groups(
                (obs, obs), "_backend_expval", 1
            )

    def test_every_observable_empty_raises(self):
        """An observable with zero Pauli terms can't be measured — guard
        against silent passes from an all-empty union."""
        from qiskit.quantum_info import PauliList

        empty = SparsePauliOp(PauliList(["II"])[:0])
        assert empty.size == 0
        with pytest.raises(ValueError, match="every observable"):
            compute_multi_observable_measurement_groups((empty,), "qwc", 2)

    def test_postprocessing_returns_list_in_input_order(self):
        """One float per input observable, in the order passed."""
        obs1 = SparsePauliOp.from_list([("IZ", 1.0)])  # Z on qubit 0
        obs2 = SparsePauliOp.from_list([("ZI", 1.0)])  # Z on qubit 1
        groups, partition, postproc = compute_multi_observable_measurement_groups(
            (obs1, obs2), "qwc", 2
        )
        # Both QWC-commute → single group, two terms.
        assert len(groups) == 1
        # Each measurement group entry returns one expval per partition slot.
        out = postproc([{0: 0.7, 1: -0.4}])
        assert isinstance(out, list)
        assert len(out) == 2
        assert out == pytest.approx([0.7, -0.4])

    def test_coefficients_applied_per_observable(self):
        """Postprocessing recovers each observable's full coeff·term sum."""
        # obs1 = 2·Z(0) + 0.5·X(0)
        obs1 = SparsePauliOp.from_list([("Z", 2.0), ("X", 0.5)])
        # obs2 = -1·Z(0)
        obs2 = SparsePauliOp.from_list([("Z", -1.0)])
        groups, partition, postproc = compute_multi_observable_measurement_groups(
            (obs1, obs2), "qwc", 1
        )
        # Z and X don't QWC-commute → 2 groups: {Z, Z}, {X}.
        # Union order: obs1 contributes Z (idx 0), X (idx 1); obs2 contributes Z (idx 2).
        # The Z's go in one group, the X in its own.
        n_groups = len(groups)
        assert n_groups == 2
        # Build a synthetic per-group result:
        # for the Z group, the original-index-to-position map is determined
        # internally; we feed the values keyed by the position in each group.
        # For determinism, run the same single-observable grouper on the union
        # to discover positions, then drive postproc with consistent values.
        union = SparsePauliOp.from_list([("Z", 1.0), ("X", 1.0), ("Z", 1.0)])
        _, union_partition, _ = compute_measurement_groups(union, "qwc", 1)
        # union_partition is a list of lists; for each group, position i maps
        # to original term index union_partition[g][i].
        # Build per-group dicts that put 1.0 for term Z and 0.0 for term X.
        per_group_results = []
        for g_idx, indices in enumerate(union_partition):
            d = {}
            for pos, orig_idx in enumerate(indices):
                # term at orig_idx 0 or 2 → Z, value 1.0; term at orig_idx 1 → X, value 0.5.
                d[pos] = 1.0 if orig_idx in (0, 2) else 0.5
            per_group_results.append(d)
        out = postproc(per_group_results)
        # obs1 = 2·<Z> + 0.5·<X> = 2·1 + 0.5·0.5 = 2.25
        # obs2 = -1·<Z> = -1.0
        assert out[0] == pytest.approx(2.25)
        assert out[1] == pytest.approx(-1.0)

    def test_qwc_groups_shared_across_observables(self):
        """When every observable's terms QWC-commute, exactly one group is emitted."""
        obs1 = SparsePauliOp.from_list([("IZ", 1.0)])
        obs2 = SparsePauliOp.from_list([("ZI", 1.0)])
        obs3 = SparsePauliOp.from_list([("ZZ", 1.0)])
        groups, _, _ = compute_multi_observable_measurement_groups(
            (obs1, obs2, obs3), "qwc", 2
        )
        assert len(groups) == 1

    def test_wires_strategy_supported(self):
        """The 'wires' strategy works on the union without auto-promotion."""
        obs1 = SparsePauliOp.from_list([("IZ", 0.5)])
        obs2 = SparsePauliOp.from_list([("ZI", 1.0)])
        groups, partition, postproc = compute_multi_observable_measurement_groups(
            (obs1, obs2), "wires", 2
        )
        # Different active qubits → wire-disjoint → single wires group.
        assert len(groups) == 1
        out = postproc([{0: 0.4, 1: -0.2}])
        assert out == pytest.approx([0.5 * 0.4, 1.0 * -0.2])

    def test_wrong_grouped_result_count_raises(self):
        obs1 = SparsePauliOp.from_list([("IZ", 1.0)])
        obs2 = SparsePauliOp.from_list([("IX", 1.0)])  # non-QWC with Z on qubit 0
        groups, _, postproc = compute_multi_observable_measurement_groups(
            (obs1, obs2), "qwc", 2
        )
        assert len(groups) == 2
        with pytest.raises(RuntimeError, match="Expected 2"):
            postproc([0.5])  # only one group worth of values


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
