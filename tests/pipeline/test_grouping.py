# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pennylane as qp
import pytest
from qiskit.quantum_info import SparsePauliOp

from divi.circuits import measurement_qasms_from_groups, qscript_to_meta
from divi.circuits._core import flatten_observable_tuple
from divi.pipeline._grouping import (
    _compute_measurement_groups,
    _wire_grouping_from_labels,
)


def _wrap(obs: SparsePauliOp) -> tuple[SparsePauliOp, ...]:
    """Wrap a single observable in the canonical 1-tuple form."""
    return (obs,)


class TestComputeMeasurementGroupsSingle:
    """Single-observable cases (length-1 tuple)."""

    def test_single_term_wires(self):
        obs = SparsePauliOp.from_list([("Z", 1.0)])
        groups, partition, postproc = _compute_measurement_groups(
            _wrap(obs), "wires", 1
        )
        assert len(groups) == 1
        assert len(groups[0]) == 1
        assert partition == [[0]]
        assert postproc([0.5]) == pytest.approx([0.5])

    def test_single_term_qwc(self):
        obs = SparsePauliOp.from_list([("Z", 1.0)])
        groups, partition, _ = _compute_measurement_groups(_wrap(obs), "qwc", 1)
        assert len(groups) == 1
        assert partition == [[0]]

    def test_single_term_default(self):
        obs = SparsePauliOp.from_list([("Z", 1.0)])
        groups, partition, _ = _compute_measurement_groups(_wrap(obs), "default", 1)
        assert len(groups) == 1
        assert partition == [[0]]

    def test_single_term_backend_expval(self):
        obs = SparsePauliOp.from_list([("Z", 1.0)])
        groups, partition, _ = _compute_measurement_groups(
            _wrap(obs), "_backend_expval", 1
        )
        assert groups == ((),)
        assert partition == [[0]]

    def test_single_term_none_strategy(self):
        obs = SparsePauliOp.from_list([("Z", 1.0)])
        groups, partition, _ = _compute_measurement_groups(_wrap(obs), None, 1)
        assert len(groups) == 1
        assert partition == [[0]]

    def test_multi_term_postprocessing(self):
        obs = SparsePauliOp.from_list([("IZ", 0.5), ("ZI", 0.3)])
        groups, partition, postproc = _compute_measurement_groups(_wrap(obs), None, 2)
        assert len(groups) == 2
        assert len(partition) == 2
        out = postproc([0.5, 0.3])
        assert out == pytest.approx([0.5 * 0.5 + 0.3 * 0.3])

    def test_qwc_groups_commuting_terms(self):
        obs = SparsePauliOp.from_list([("IZ", 0.5), ("ZI", 0.3)])
        groups, _, _ = _compute_measurement_groups(_wrap(obs), "qwc", 2)
        assert len(groups) == 1

    def test_qwc_splits_non_commuting(self):
        obs = SparsePauliOp.from_list([("IZ", 0.5), ("IX", 0.3)])
        groups, _, _ = _compute_measurement_groups(_wrap(obs), "qwc", 2)
        assert len(groups) == 2

    def test_unknown_strategy_raises(self):
        obs = SparsePauliOp.from_list([("Z", 1.0)])
        with pytest.raises(ValueError, match="Unknown grouping strategy"):
            _compute_measurement_groups(_wrap(obs), "invalid_strategy", 1)


class TestComputeMeasurementGroupsMulti:
    """Cross-observable grouping (length > 1 tuple)."""

    def test_empty_tuple_raises(self):
        with pytest.raises(ValueError, match="empty"):
            _compute_measurement_groups((), "qwc", 2)

    def test_backend_expval_strategy_rejected_for_multi(self):
        obs = SparsePauliOp.from_list([("Z", 1.0)])
        with pytest.raises(ValueError, match="_backend_expval"):
            _compute_measurement_groups((obs, obs), "_backend_expval", 1)

    def test_postprocessing_returns_list_in_input_order(self):
        obs1 = SparsePauliOp.from_list([("IZ", 1.0)])
        obs2 = SparsePauliOp.from_list([("ZI", 1.0)])
        groups, partition, postproc = _compute_measurement_groups(
            (obs1, obs2), "qwc", 2
        )
        assert len(groups) == 1
        out = postproc([{0: 0.7, 1: -0.4}])
        assert isinstance(out, list)
        assert len(out) == 2
        assert out == pytest.approx([0.7, -0.4])

    def test_coefficients_applied_per_observable(self):
        obs1 = SparsePauliOp.from_list([("Z", 2.0), ("X", 0.5)])
        obs2 = SparsePauliOp.from_list([("Z", -1.0)])
        groups, partition, postproc = _compute_measurement_groups(
            (obs1, obs2), "qwc", 1
        )
        assert len(groups) == 2
        slot_value: dict[int, float] = {}
        for g_idx, indices in enumerate(partition):
            for pos, slot in enumerate(indices):
                slot_value[slot] = 1.0 if "Z" in groups[g_idx][pos] else 0.5
        per_group_results = [
            {pos: slot_value[slot] for pos, slot in enumerate(indices)}
            for indices in partition
        ]
        out = postproc(per_group_results)
        # obs1 = 2·<Z> + 0.5·<X> = 2 + 0.25 = 2.25
        # obs2 = -1·<Z> = -1.0
        assert out[0] == pytest.approx(2.25)
        assert out[1] == pytest.approx(-1.0)

    def test_duplicate_paulis_dedup_to_one_union_slot(self):
        obs1 = SparsePauliOp.from_list([("IZ", 1.0)])
        obs2 = SparsePauliOp.from_list([("IZ", 1.0)])
        groups, partition, postproc = _compute_measurement_groups(
            (obs1, obs2), "qwc", 2
        )
        assert len(groups) == 1
        assert len(groups[0]) == 1
        assert sum(len(p) for p in partition) == 1
        out = postproc([{0: 0.4}])
        assert out == pytest.approx([0.4, 0.4])

    def test_within_observable_duplicates_collapse(self):
        obs = SparsePauliOp.from_list([("Z", 0.5), ("Z", 0.25)])
        groups, partition, postproc = _compute_measurement_groups((obs,), "qwc", 1)
        assert sum(len(p) for p in partition) == 1
        out = postproc([{0: 0.8}])
        assert out == pytest.approx([(0.5 + 0.25) * 0.8])

    def test_qwc_groups_shared_across_observables(self):
        obs1 = SparsePauliOp.from_list([("IZ", 1.0)])
        obs2 = SparsePauliOp.from_list([("ZI", 1.0)])
        obs3 = SparsePauliOp.from_list([("ZZ", 1.0)])
        groups, _, _ = _compute_measurement_groups((obs1, obs2, obs3), "qwc", 2)
        assert len(groups) == 1

    def test_wires_strategy_supported(self):
        obs1 = SparsePauliOp.from_list([("IZ", 0.5)])
        obs2 = SparsePauliOp.from_list([("ZI", 1.0)])
        groups, _, postproc = _compute_measurement_groups((obs1, obs2), "wires", 2)
        assert len(groups) == 1
        out = postproc([{0: 0.4, 1: -0.2}])
        assert out == pytest.approx([0.5 * 0.4, 1.0 * -0.2])

    def test_wrong_grouped_result_count_raises(self):
        obs1 = SparsePauliOp.from_list([("IZ", 1.0)])
        obs2 = SparsePauliOp.from_list([("IX", 1.0)])
        groups, _, postproc = _compute_measurement_groups((obs1, obs2), "qwc", 2)
        assert len(groups) == 2
        with pytest.raises(RuntimeError, match="Expected 2"):
            postproc([0.5])

    def test_sign_canceling_observables_preserve_per_obs_values(self):
        obs1 = SparsePauliOp.from_list([("Z", 1.0)])
        obs2 = SparsePauliOp.from_list([("Z", -1.0)])
        union, _ = flatten_observable_tuple((obs1, obs2))
        assert float(np.real(union.coeffs[0])) == pytest.approx(2.0)

        _, _, postproc = _compute_measurement_groups((obs1, obs2), "qwc", 1)
        out = postproc([{0: 0.7}])
        assert out == pytest.approx([0.7, -0.7])

    def test_purely_imaginary_coeff_yields_zero_union_weight(self):
        """A non-Hermitian observable with purely imaginary coefficients is
        coerced to zero real part; the union slot weight is 0.0."""
        obs = SparsePauliOp.from_list([("Y", 1j)])
        union, _ = flatten_observable_tuple((obs,))
        assert float(np.real(union.coeffs[0])) == pytest.approx(0.0)

    def test_cross_obs_qwc_grouping_collapses_diagonal_observables_to_one_group(
        self,
    ):
        """All-Z observables share a single basis — QWC must produce exactly
        one measurement group across all observables AND the postprocessor
        must reconstruct each observable's value from the shared shots."""
        observables = [
            SparsePauliOp.from_list([("ZIZI", 1.0), ("ZZII", 1.0)]),
            SparsePauliOp.from_list([("IZIZ", 1.0), ("IIZZ", 1.0)]),
            SparsePauliOp.from_list([("ZZZZ", 1.0), ("ZIIZ", 1.0)]),
            SparsePauliOp.from_list([("IZZI", 1.0), ("ZZIZ", 1.0)]),
            SparsePauliOp.from_list([("ZIZZ", 1.0), ("IZZZ", 1.0)]),
        ]
        groups, partition, postproc = _compute_measurement_groups(
            tuple(observables), "qwc", 4
        )
        assert len(groups) == 1
        # All Pauli slots in the union assigned to the single group.
        # Feed each slot value 1.0 through postproc; expected per-observable
        # value is the sum of its coefficients (each 1.0, two terms each).
        slot_values = {pos: 1.0 for pos in range(len(groups[0]))}
        out = postproc([slot_values])
        assert isinstance(out, list)
        assert len(out) == len(observables)
        for v in out:
            assert v == pytest.approx(2.0)


class TestMetaCircuitWithGrouping:
    def test_wires_and_empty_group_produce_same_measurement_qasm(self):
        circuit = qp.tape.QuantumScript(
            ops=[qp.Hadamard(0), qp.CNOT(wires=[0, 1])],
            measurements=[qp.probs()],
        )
        meta = qscript_to_meta(circuit)
        n_qubits = meta.n_qubits

        qasms_explicit = measurement_qasms_from_groups(((),), n_qubits)
        assert len(qasms_explicit) == 1
        assert "measure" in qasms_explicit[0]


class TestWireGroupingFromLabels:
    def test_non_overlapping_in_one_group(self):
        labels = ["ZII", "IZI", "IIZ"]
        groups = _wire_grouping_from_labels(labels)
        assert len(groups) == 1
        assert groups[0] == [0, 1, 2]

    def test_overlapping_split(self):
        labels = ["ZI", "ZI"]
        groups = _wire_grouping_from_labels(labels)
        assert len(groups) == 2

    def test_mixed(self):
        labels = ["ZII", "IZI", "ZZI"]
        groups = _wire_grouping_from_labels(labels)
        assert len(groups) == 2
        assert 0 in groups[0] and 1 in groups[0]
        assert 2 in groups[1]
