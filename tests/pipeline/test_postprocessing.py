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
    _counts_to_wrs_cost_variance,
    _expval_dicts_to_indexed,
)
from divi.pipeline.abc import ChildResults
from divi.pipeline.stages import MeasurementStage
from tests._helpers import exact_match
from tests.pipeline._helpers import DummySpecStage, meta_from_circuit


def _measured_trace(env, observable, n_qubits=None, **stage_kw):
    """Forward pass of one counts-measured observable, with its lineage and node.

    ``wires`` grouping keeps a real counts-measured group; the default would
    promote to the backend-native expval path, which carries no counts.
    """
    if n_qubits is None:
        n_qubits = observable.num_qubits
    pipeline = CircuitPipeline(
        stages=[
            DummySpecStage(
                meta=meta_from_circuit(QuantumCircuit(n_qubits), observable=observable)
            ),
            MeasurementStage(grouping_strategy="wires", **stage_kw),
        ]
    )
    trace = pipeline.run_forward_pass("x", env)
    return (
        trace,
        batch_lineage(trace.final_batch),
        next(iter(trace.final_batch.values())),
    )


def _wrs_trace(env, observable):
    """``_measured_trace`` under weighted-random sampling, with group probabilities."""
    trace, lineage, _ = _measured_trace(
        env, observable, shot_distribution="weighted_random"
    )
    token = trace.stage_tokens[1]
    probabilities = {
        key: plan.probabilities_by_group
        for key, plan in token.group_shot_plans_by_spec.items()
    }
    return trace, lineage, probabilities


def _with_param_sets(lineage, counts_by_param_set) -> ChildResults:
    """Raw results with every branch repeated once per parameter set."""
    return {
        (*bk, ("param_set", idx)): counts
        for idx, counts in enumerate(counts_by_param_set)
        for bk in lineage.values()
    }


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
        trace, lineage_by_label, _ = _measured_trace(dummy_pipeline_env, observable)
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
        trace, lineage_by_label, _ = _measured_trace(dummy_pipeline_env, observable)

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
        """Forward pass for a single-qubit ``<Z0>`` cost (one measurement group)."""
        trace, lineage, _ = _measured_trace(dummy_pipeline_env, SparsePauliOp("Z"))
        return trace, lineage

    @pytest.mark.parametrize(
        ("observable", "n_qubits", "counts", "expected"),
        [
            (SparsePauliOp("Z"), 1, {"0": 75, "1": 25}, 0.0075),
            (SparsePauliOp("Z"), 1, {"0": 100}, 0.0),
            (
                SparsePauliOp.from_list([("Z", 0.5), ("Z", 0.5)]),
                1,
                {"0": 75, "1": 25},
                0.0075,
            ),
            (
                SparsePauliOp.from_list([("IZ", 0.0), ("ZI", 1.0)]),
                2,
                {"00": 75, "10": 25},
                0.0075,
            ),
            (SparsePauliOp("Z"), 2, {"00": 75, "01": 25}, 0.0075),
            (SparsePauliOp("Z"), 1, {"0": 1}, 0.0),
        ],
        ids=[
            "analytic-formula",
            "saturated-pauli",
            "duplicate-terms",
            "zero-coefficient-first",
            "narrow-observable",
            "one-shot",
        ],
    )
    def test_single_group_matches_analytic_formula(
        self, dummy_pipeline_env, observable, n_qubits, counts, expected
    ):
        trace, lineage, node = _measured_trace(dummy_pipeline_env, observable, n_qubits)
        raw: ChildResults = {bk: counts for bk in lineage.values()}

        variances = list(_counts_to_cost_variance(raw, trace.final_batch).values())

        assert sum(len(group) for group in node.measurement_groups) == len(
            set(observable.paulis.to_labels())
        )
        assert variances == [pytest.approx(expected)]
        assert variances[0] >= 0.0

    def test_sums_weighted_terms_within_and_across_groups(self, dummy_pipeline_env):
        # Little-endian counts: <Z_q0>=0.4 and <Z_q1>=0.6 in the Z group, <XX>=0.4.
        observable = SparsePauliOp.from_list([("ZI", 2.0), ("IZ", 0.5), ("XX", 1.5)])
        trace, lineage, node = _measured_trace(dummy_pipeline_env, observable)
        z_counts = {"00": 50, "01": 30, "10": 20}
        xx_counts = {"00": 70, "01": 30}
        raw: ChildResults = {
            bk: (
                xx_counts
                if "XX" in node.measurement_groups[dict(bk)["obs_group"]]
                else z_counts
            )
            for bk in lineage.values()
        }

        expected = (
            4.0 * (1 - 0.6**2) + 0.25 * (1 - 0.4**2) + 2.25 * (1 - 0.4**2)
        ) / 100
        assert list(_counts_to_cost_variance(raw, trace.final_batch).values()) == [
            pytest.approx(expected)
        ]

    def test_multi_observable_cost_variance_is_nan_for_every_key(
        self, dummy_pipeline_env
    ):
        trace, lineage, _ = _measured_trace(
            dummy_pipeline_env, (SparsePauliOp("Z"), SparsePauliOp("X")), 1
        )
        counts = {"0": 75, "1": 25}
        raw = _with_param_sets(lineage, [counts, counts])

        result = _counts_to_cost_variance(raw, trace.final_batch)

        assert len(result) == 2
        assert all(np.isnan(v) for v in result.values())

    def test_zero_shot_key_leaves_other_keys_intact(self, single_z_trace):
        trace, lineage = single_z_trace
        raw = _with_param_sets(lineage, [{}, {"0": 75, "1": 25}])
        (base_key,) = {
            tuple(ax for ax in bk if ax[0] != "obs_group") for bk in lineage.values()
        }

        result = _counts_to_cost_variance(raw, trace.final_batch)

        assert set(result) == {
            (*base_key, ("param_set", 0)),
            (*base_key, ("param_set", 1)),
        }
        assert np.isnan(result[(*base_key, ("param_set", 0))])
        assert result[(*base_key, ("param_set", 1))] == pytest.approx(0.0075)

    def test_zero_shots_returns_nan(self, single_z_trace):
        trace, lineage_by_label = single_z_trace
        raw: ChildResults = {bk: {} for bk in lineage_by_label.values()}
        result = _counts_to_cost_variance(raw, trace.final_batch)
        assert result
        assert all(np.isnan(v) for v in result.values())

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


class TestCountsToWRSCostVariance:
    @pytest.fixture
    def wrs_trace(self, dummy_pipeline_env):
        return _wrs_trace(
            dummy_pipeline_env, SparsePauliOp.from_list([("Z", 2.0), ("X", 1.0)])
        )

    @pytest.fixture
    def zz_wrs_trace(self, dummy_pipeline_env):
        """One commuting group whose per-shot value is ``z_q0 + z_q1``."""
        return _wrs_trace(
            dummy_pipeline_env, SparsePauliOp.from_list([("ZI", 1.0), ("IZ", 1.0)])
        )

    def test_single_shot_keys_are_each_undefined(self, wrs_trace):
        trace, lineage, probabilities = wrs_trace
        raw = _with_param_sets(dict(list(lineage.items())[:1]), [{"0": 1}, {"1": 1}])

        result = _counts_to_wrs_cost_variance(raw, trace.final_batch, probabilities)

        assert len(result) == 2
        assert all(np.isnan(v) for v in result.values())

    @pytest.mark.parametrize(
        ("counts", "expected"),
        [
            ({"00": 1, "11": 1}, 4.0),
            (
                {"00": 4, "01": 3, "10": 2, "11": 1},
                np.repeat([2.0, 0.0, 0.0, -2.0], [4, 3, 2, 1]).var(ddof=1) / 10,
            ),
        ],
        ids=["covariance-within-group", "several-bins"],
    )
    def test_matches_sample_variance_of_the_weighted_shots(
        self, zz_wrs_trace, counts, expected
    ):
        # Per-shot values are z_q0 + z_q1; counting the two Z terms
        # independently would halve the covariance case's variance.
        trace, lineage, probabilities = zz_wrs_trace
        branch_key = next(iter(lineage.values()))

        result = _counts_to_wrs_cost_variance(
            {branch_key: counts}, trace.final_batch, probabilities
        )

        assert next(iter(result.values())) == pytest.approx(expected)

    def test_a_group_without_shots_invalidates_its_cost(self, wrs_trace):
        trace, lineage, probabilities = wrs_trace
        raw: ChildResults = {
            bk: {} if dict(bk)["obs_group"] == 0 else {"0": 3, "1": 2}
            for bk in sorted(lineage.values(), key=lambda bk: dict(bk)["obs_group"])
        }

        result = _counts_to_wrs_cost_variance(raw, trace.final_batch, probabilities)

        assert len(result) == 1
        assert np.isnan(next(iter(result.values())))


@pytest.mark.parametrize(
    "payload,found",
    [
        (0.5, "float"),
        ([0.5], "list"),
        ({0: 0.5}, "a dict not keyed by bitstrings"),
    ],
    ids=["scalar", "list", "expval-dict"],
)
@pytest.mark.parametrize(
    "consume",
    [
        lambda raw, trace, _: _counts_to_expvals(raw, trace.final_batch),
        lambda raw, trace, _: _counts_to_cost_variance(raw, trace.final_batch),
        lambda raw, trace, probs: _counts_to_wrs_cost_variance(
            raw, trace.final_batch, probs
        ),
    ],
    ids=["expvals", "cost-variance", "wrs-cost-variance"],
)
def test_counts_consumers_reject_non_histogram_results(
    dummy_pipeline_env, consume, payload, found
):
    trace, lineage, probabilities = _wrs_trace(
        dummy_pipeline_env, SparsePauliOp.from_list([("Z", 2.0), ("X", 1.0)])
    )
    raw: ChildResults = {bk: payload for bk in lineage.values()}
    first_key = next(iter(raw))
    message = (
        "Expected a bitstring→count histogram from the backend for branch "
        f"{first_key!r}, got {found}."
    )
    with pytest.raises(TypeError, match=exact_match(message)):
        consume(raw, trace, probabilities)


class TestBatchedExpectation:
    """Tests for _batched_expectation with big-endian Pauli label strings."""

    @pytest.mark.parametrize(
        ("histogram", "label", "expected"),
        [
            ({"000": 70, "100": 30}, "ZII", 0.4),
            ({"000": 100}, "ZIZ", 1.0),
            ({"001": 100}, "ZIZ", -1.0),
            ({"0000": 1}, "IXYZ", 1.0),
        ],
        ids=["z-on-qubit-0", "product-even", "product-odd", "every-pauli"],
    )
    def test_single_label_expectation(self, histogram, label, expected):
        """Big-endian labels: position 0 is qubit 0, and ``ZIZ`` acts on qubits
        0 and 2 so a reversal that swaps them is distinguishable."""
        result = _batched_expectation([histogram], [label], n_qubits=len(label))
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

    def test_identity_label_does_not_stop_later_labels(self):
        result = _batched_expectation([{"10": 100}], ["II", "ZI"], n_qubits=2)

        np.testing.assert_allclose(result[:, 0], [1.0, -1.0])

    def test_identity_expectation_is_exact(self):
        """Integer counts are contracted before division, so identity stays exact."""
        result = _batched_expectation([{"00": 1, "01": 2, "10": 3}], ["II"], n_qubits=2)

        assert result[0, 0] == 1.0

    def test_empty_histogram_stays_zero(self):
        """An empty histogram is a finite zero row rather than ``nan``."""
        histograms = [{format(i, "065b"): 1} for i in range(1001)] + [{}]
        result = _batched_expectation(histograms, ["I" * 65], n_qubits=65)

        assert result.shape == (1, 1002)
        assert np.all(result[0, :-1] == 1.0)
        assert np.isfinite(result[0, -1])
        assert result[0, -1] == 0.0

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

    @pytest.mark.parametrize(
        "expvals",
        [
            pytest.param({"XII": 0.5, "IZI": -0.3, "IIX": 0.2}, id="matching_order"),
            pytest.param({"IZI": -0.3, "IIX": 0.2, "XII": 0.5}, id="shuffled_order"),
        ],
    )
    def test_multi_op_indexed_in_ham_ops_order(self, expvals):
        result = _expval_dicts_to_indexed({("k",): expvals}, "XII;IZI;IIX")
        assert result[("k",)] == {0: 0.5, 1: -0.3, 2: 0.2}

    def test_single_op_returns_float(self):
        raw = {("k",): {"ZII": 0.7}}
        result = _expval_dicts_to_indexed(raw, "ZII")
        assert result[("k",)] == pytest.approx(0.7)

    @pytest.mark.parametrize(
        "raw", [{("k",): 1.5}, {}], ids=["non-dict-values", "empty"]
    )
    def test_non_pauli_dict_input_passes_through(self, raw):
        assert _expval_dicts_to_indexed(raw, "XII;IZI") == raw
