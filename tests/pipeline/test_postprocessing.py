# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from divi.pipeline import CircuitPipeline
from divi.pipeline._compilation import batch_lineage
from divi.pipeline._postprocessing import (
    _batched_expectation,
    _counts_to_cost_variance,
    _counts_to_expvals,
    _counts_to_probs,
    _expval_dicts_to_indexed,
)
from divi.pipeline.abc import ChildResults
from divi.pipeline.stages import MeasurementStage
from tests.pipeline._helpers import DummySpecStage, meta_from_circuit


class TestCountsToExpvals:
    """Spec: _counts_to_expvals converts shot counts to expvals as a post-processing step."""

    @pytest.mark.parametrize("obs_qubit, expected", [(0, -1.0), (1, -1.0), (2, 1.0)])
    def test_single_qubit_term_maps_to_correct_qubit(
        self, dummy_pipeline_env, obs_qubit, expected
    ):
        # Little-endian backend counts (qubit 0 rightmost): "011" -> q0=1, q1=1,
        # q2=0, so <Z0>=<Z1>=-1 and <Z2>=+1. Three qubits (not two) so a partial
        # reversal is distinguishable from a full one: a 0<->2 swap would flip
        # <Z0> and <Z2> and be caught here.
        observable = SparsePauliOp.from_sparse_list(
            [("Z", [obs_qubit], 1.0)], num_qubits=3
        )
        pipeline = CircuitPipeline(
            stages=[
                DummySpecStage(
                    meta=meta_from_circuit(QuantumCircuit(3), observable=observable)
                ),
                MeasurementStage(grouping_strategy="wires"),
            ]
        )
        trace = pipeline.run_forward_pass("x", dummy_pipeline_env)
        lineage_by_label = batch_lineage(trace.final_batch)
        raw: ChildResults = {bk: {"011": 100} for bk in lineage_by_label.values()}

        result = _counts_to_expvals(raw, trace.final_batch)

        assert result
        for value in result.values():
            assert value == pytest.approx(expected)

    def test_multi_obs_group_returns_dict_in_term_order(self, dummy_pipeline_env):
        # Three wire-disjoint terms -> per-term <Z_q> keyed by term index.
        observable = SparsePauliOp.from_sparse_list(
            [("Z", [0], 0.5), ("Z", [1], 0.3), ("Z", [2], 0.2)], num_qubits=3
        )
        meta = meta_from_circuit(QuantumCircuit(3), observable=observable)

        pipeline = CircuitPipeline(
            stages=[
                DummySpecStage(meta=meta),
                MeasurementStage(grouping_strategy="wires"),
            ]
        )
        trace = pipeline.run_forward_pass("x", dummy_pipeline_env)
        lineage_by_label = batch_lineage(trace.final_batch)

        # Little-endian backend counts giving distinct <Z0>=0.4, <Z1>=0.6,
        # <Z2>=0.8, so any qubit permutation (not just a full reversal) is caught
        # by an order-sensitive (per-index) assertion.
        raw: ChildResults = {
            bk: {"000": 40, "001": 30, "010": 20, "100": 10}
            for bk in lineage_by_label.values()
        }

        result = _counts_to_expvals(raw, trace.final_batch)

        dict_results = [v for v in result.values() if isinstance(v, dict)]
        assert len(dict_results) >= 1

        for d in dict_results:
            assert len(d) == 3
            assert d[0] == pytest.approx(0.4)  # <Z0>
            assert d[1] == pytest.approx(0.6)  # <Z1>
            assert d[2] == pytest.approx(0.8)  # <Z2>


class TestCountsToCostVariance:
    """Spec: _counts_to_cost_variance estimates shot-noise variance of the cost
    Var(<H>) = Σ_i c_i²(1 − <P_i>²)/M_g from raw counts."""

    @pytest.fixture
    def single_z_trace(self, dummy_pipeline_env):
        """Forward pass for a single-qubit ``<Z0>`` cost (one measurement group).

        Uses ``wires`` grouping so a real counts-measured group is produced; the
        default would promote to the backend-native expval path (no counts), on
        which ``_counts_to_cost_variance`` is never invoked.
        """
        observable = SparsePauliOp.from_list([("Z", 1.0)])
        pipeline = CircuitPipeline(
            stages=[
                DummySpecStage(
                    meta=meta_from_circuit(QuantumCircuit(1), observable=observable)
                ),
                MeasurementStage(grouping_strategy="wires"),
            ]
        )
        trace = pipeline.run_forward_pass("x", dummy_pipeline_env)
        return trace, batch_lineage(trace.final_batch)

    def test_matches_analytic_formula(self, single_z_trace):
        # <Z> = (75 - 25)/100 = 0.5, coeff = 1, M = 100
        # Var = 1²·(1 - 0.5²)/100 = 0.0075
        trace, lineage_by_label = single_z_trace
        raw: ChildResults = {bk: {"0": 75, "1": 25} for bk in lineage_by_label.values()}
        result = _counts_to_cost_variance(raw, trace.final_batch)
        assert result
        for v in result.values():
            assert v == pytest.approx(0.0075)

    def test_zero_shots_returns_nan(self, single_z_trace):
        trace, lineage_by_label = single_z_trace
        raw: ChildResults = {bk: {} for bk in lineage_by_label.values()}
        result = _counts_to_cost_variance(raw, trace.final_batch)
        assert result
        assert all(np.isnan(v) for v in result.values())

    def test_saturated_pauli_clamps_to_nonnegative_zero(self, single_z_trace):
        # All shots in one eigenstate → <Z> = 1 → 1 − <Z>² = 0 (never negative).
        trace, lineage_by_label = single_z_trace
        raw: ChildResults = {bk: {"0": 100} for bk in lineage_by_label.values()}
        result = _counts_to_cost_variance(raw, trace.final_batch)
        assert result
        for v in result.values():
            assert v == pytest.approx(0.0)
            assert v >= 0.0

    def test_variance_scales_inversely_with_shots(self, single_z_trace):
        trace, lineage_by_label = single_z_trace
        raw_low: ChildResults = {
            bk: {"0": 75, "1": 25} for bk in lineage_by_label.values()
        }
        raw_high: ChildResults = {
            bk: {"0": 750, "1": 250} for bk in lineage_by_label.values()
        }
        var_low = next(
            iter(_counts_to_cost_variance(raw_low, trace.final_batch).values())
        )
        var_high = next(
            iter(_counts_to_cost_variance(raw_high, trace.final_batch).values())
        )
        # Same <Z>=0.5, 10× shots → exactly 10× smaller variance.
        assert var_low == pytest.approx(10.0 * var_high)


class TestBatchedExpectation:
    """Tests for _batched_expectation with big-endian Pauli label strings."""

    def test_single_z_observable(self):
        """Z on qubit 0 (big-endian "ZII"); position 0 must map to qubit 0."""
        histogram = {"000": 70, "100": 30}  # qubit 0 = 1 only in "100"
        result = _batched_expectation([histogram], ["ZII"], n_qubits=3)
        expected = (70 * 1 + 30 * (-1)) / 100  # 0.4
        assert result[0, 0] == pytest.approx(expected)

    def test_narrowed_bitstrings_raise(self):
        """A backend that returns keys narrower than n_qubits (only measured
        clbits) would misalign positional decoding — must fail loudly."""
        histogram = {"01": 100}  # 2-bit keys for a 3-qubit circuit
        with pytest.raises(ValueError, match="full-width keys"):
            _batched_expectation([histogram], ["ZII"], n_qubits=3)

    @pytest.mark.parametrize("n_qubits", [3, 70])
    def test_ragged_bitstrings_raise(self, n_qubits):
        """One wrong-width key among correct ones must be caught, not regrouped.

        Widths that happen to sum to the right total would otherwise reshape
        into a matrix whose rows straddle two keys.
        """
        good = "1" + "0" * (n_qubits - 1)
        histogram = {good: 50, good[:-1]: 30, good + "0": 20}
        with pytest.raises(ValueError, match="full-width keys"):
            _batched_expectation([histogram], ["Z" * n_qubits], n_qubits=n_qubits)

    @pytest.mark.parametrize(
        ("malformed", "n_qubits"),
        [("0_1", 3), (" 1", 2), ("+1", 2), ("2" + "0" * 64, 65)],
    )
    def test_non_binary_bitstring_raises(self, malformed, n_qubits):
        """Both integer and packed paths accept only literal zero and one."""
        with pytest.raises(ValueError, match="non-binary"):
            _batched_expectation(
                [{malformed: 1}],
                ["Z" + "I" * (n_qubits - 1)],
                n_qubits=n_qubits,
            )

    def test_non_pauli_label_character_raises(self):
        """Only IXYZ have the ±1 spectrum the parity form assumes."""
        with pytest.raises(ValueError, match="outside IXYZ"):
            _batched_expectation([{"010": 5, "111": 5}], ["ZAZ"], n_qubits=3)

    @pytest.mark.parametrize("label", ["Z", "IIZ"])
    def test_pauli_label_width_must_match_register(self, label):
        """Every observable label must address exactly the declared register."""
        with pytest.raises(ValueError, match="2-character"):
            _batched_expectation([{"00": 1}], [label], n_qubits=2)

    def test_product_observable(self):
        """ZIZ on 3 qubits: product of the qubit-0 and qubit-2 Z eigenvalues.

        Acting on qubits 0 and 2 (not 0 and 1) so a reversal that swaps them is
        distinguishable from the identity.
        """
        # "000" → (+1)(+1) = +1
        assert _batched_expectation([{"000": 100}], ["ZIZ"], n_qubits=3)[
            0, 0
        ] == pytest.approx(1.0)
        # "001" → qubit 0 = 0 (+1), qubit 2 = 1 (-1) → -1
        assert _batched_expectation([{"001": 100}], ["ZIZ"], n_qubits=3)[
            0, 0
        ] == pytest.approx(-1.0)

    def test_multiple_histograms(self):
        hist_1 = {"000": 100}
        hist_2 = {"101": 50}  # qubit 0 = 1, qubit 2 = 1
        hist_3 = {"001": 25, "100": 75}  # qubit 2 = 1 (25%); qubit 0 = 1 (75%)
        labels = ["ZII", "IIZ", "ZIZ"]  # <Z0>, <Z2>, <Z0 Z2>

        result = _batched_expectation([hist_1, hist_2, hist_3], labels, n_qubits=3)

        assert result.shape == (3, 3)
        # hist_1 "000": all eigenvalues +1
        np.testing.assert_allclose(result[:, 0], [1.0, 1.0, 1.0])
        # hist_2 "101": Z0→-1, Z2→-1, Z0Z2→+1
        np.testing.assert_allclose(result[:, 1], [-1.0, -1.0, 1.0])
        # hist_3:
        #   Z0: q0=0→+1(25%), q0=1→-1(75%) = -0.5
        #   Z2: q2=1→-1(25%), q2=0→+1(75%) = +0.5
        #   Z0Z2: (+1)(-1)(25%), (-1)(+1)(75%) = -1.0
        np.testing.assert_allclose(result[:, 2], [-0.5, 0.5, -1.0])

    @pytest.mark.parametrize(
        "n_qubits",
        [1, 10, 32, 64, 65, 100, 200, 500, 1000],
        ids=lambda n: f"{n}q",
    )
    def test_qubit_counts_no_overflow(self, n_qubits):
        """Works across qubit counts including the 64-bit boundary."""
        bitstrings = [
            "0" * n_qubits,
            "1" * n_qubits,
            "1" + "0" * (n_qubits - 1),
            "0" * (n_qubits - 1) + "1",
        ]
        histogram = {bs: 100 for bs in bitstrings}

        # Z on first qubit (big-endian position 0)
        label = "Z" + "I" * (n_qubits - 1)
        result = _batched_expectation([histogram], [label], n_qubits=n_qubits)

        assert result.shape == (1, 1)
        assert not np.isnan(result).any()
        assert not np.isinf(result).any()
        assert -1.0 <= result[0, 0] <= 1.0

    def test_boundary_64_vs_65(self):
        """Both integer (<=64) and char-array (>64) paths produce same results."""
        for nq in (64, 65):
            histogram = {"0" * nq: 100, "1" * nq: 100}
            label = "Z" + "I" * (nq - 1)
            result = _batched_expectation([histogram], [label], n_qubits=nq)
            # Half +1, half -1 → 0
            assert result[0, 0] == pytest.approx(0.0)

    @pytest.mark.parametrize(
        ("n_qubits", "n_active"),
        [(4, 1), (32, 2), (64, 3), (65, 2), (100, 3), (150, 2), (500, 3), (1000, 2)],
        ids=lambda x: f"{x[0]}q_{x[1]}w" if isinstance(x, tuple) else str(x),
    )
    def test_product_observables_large_qubit_counts(self, n_qubits, n_active):
        """Product observables (ZZ, ZZZ) work at large qubit counts."""
        histogram = {"0" * n_qubits: 100}
        label_chars = ["I"] * n_qubits
        for i in range(n_active):
            label_chars[i * (n_qubits // n_active)] = "Z"
        label = "".join(label_chars)

        result = _batched_expectation([histogram], [label], n_qubits=n_qubits)
        # All-zero bitstring → all Z eigenvalues +1 → product = +1
        assert result[0, 0] == pytest.approx(1.0)

    def test_identity_expectation_is_exact(self):
        """Integer counts are contracted before division, so identity stays exact."""
        result = _batched_expectation([{"00": 1, "01": 2, "10": 3}], ["II"], n_qubits=2)

        assert result[0, 0] == 1.0

    def test_empty_histogram_stays_zero(self):
        """An empty histogram is a finite zero row rather than ``nan``."""
        result = _batched_expectation([{}], ["I"], n_qubits=1)

        assert np.isfinite(result[0, 0])
        assert result[0, 0] == 0.0

    @pytest.mark.parametrize("n_qubits", [64, 65])
    def test_dense_label_is_a_parity_not_a_lookup_table(self, n_qubits):
        """A label active on every qubit must stay cheap and give ±1 by parity.

        Eigenvalues of X, Y and Z are ±1, so the observable's value is the
        parity of the measured ones — enumerating 2**n_active eigenvalues would
        exhaust memory here.
        """
        label = "Z" * (n_qubits - 1) + "X"
        odd_parity = "1" + "0" * (n_qubits - 1)
        even_parity = "11" + "0" * (n_qubits - 2)

        result = _batched_expectation(
            [{odd_parity: 40, even_parity: 60}], [label], n_qubits=n_qubits
        )
        assert result[0, 0] == pytest.approx((60 - 40) / 100)


class TestCountsToProbs:
    """Tests for _counts_to_probs."""

    def test_reverses_bitstrings_and_normalises(self):
        raw = {("obs",): {"100": 30, "010": 70}}
        result = _counts_to_probs(raw, shots=100)
        assert result[("obs",)] == {"001": 0.3, "010": 0.7}

    def test_non_dict_values_pass_through(self):
        # Asymmetric 3-qubit bitstrings so the endianness reversal is visible.
        raw = {("a",): 0.42, ("b",): {"110": 5, "001": 5}}
        result = _counts_to_probs(raw, shots=10)
        assert result[("a",)] == 0.42
        assert result[("b",)] == {"011": 0.5, "100": 0.5}

    def test_empty_input(self):
        assert _counts_to_probs({}, shots=100) == {}

    def test_multiple_branch_keys(self):
        raw = {
            ("x",): {"100": 4, "001": 6},
            ("y",): {"110": 2, "001": 8},
        }
        result = _counts_to_probs(raw, shots=10)
        assert result[("x",)] == {"001": 0.4, "100": 0.6}
        assert result[("y",)] == {"011": 0.2, "100": 0.8}


class TestExpvalDictsToIndexed:
    """Tests for _expval_dicts_to_indexed."""

    def test_multi_op_returns_indexed_dict(self):
        raw = {("k",): {"XII": 0.5, "IZI": -0.3, "IIX": 0.2}}
        result = _expval_dicts_to_indexed(raw, "XII;IZI;IIX")
        assert result[("k",)] == {0: 0.5, 1: -0.3, 2: 0.2}

    def test_single_op_returns_float(self):
        raw = {("k",): {"ZII": 0.7}}
        result = _expval_dicts_to_indexed(raw, "ZII")
        assert result[("k",)] == pytest.approx(0.7)

    def test_preserves_ham_ops_ordering(self):
        raw = {("k",): {"IZI": -0.3, "IIX": 0.2, "XII": 0.5}}
        result = _expval_dicts_to_indexed(raw, "XII;IZI;IIX")
        assert result[("k",)] == {0: 0.5, 1: -0.3, 2: 0.2}

    def test_non_dict_passthrough(self):
        raw = {("k",): 1.5}
        result = _expval_dicts_to_indexed(raw, "XII;IZI")
        assert result[("k",)] == 1.5
