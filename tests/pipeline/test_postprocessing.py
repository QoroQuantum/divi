# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pennylane as qml
import pytest

from divi.circuits import qscript_to_meta
from divi.pipeline import CircuitPipeline
from divi.pipeline._compilation import _compile_batch
from divi.pipeline._postprocessing import (
    _batched_expectation,
    _counts_to_expvals,
    _counts_to_probs,
    _expval_dicts_to_indexed,
    _find_batch_key,
)
from divi.pipeline.abc import ChildResults
from divi.pipeline.stages import MeasurementStage
from tests.pipeline.helpers import DummySpecStage, two_group_meta


class TestCountsToExpvals:
    """Spec: _counts_to_expvals converts shot counts to expvals as a post-processing step."""

    def test_converts_counts_dict_to_float_expval(self, dummy_pipeline_env):
        pipeline = CircuitPipeline(
            stages=[
                DummySpecStage(meta=two_group_meta()),
                MeasurementStage(),
            ]
        )
        trace = pipeline.run_forward_pass("x", dummy_pipeline_env)
        _, lineage_by_label = _compile_batch(trace.final_batch)
        raw: ChildResults = {bk: {"0": 50, "1": 50} for bk in lineage_by_label.values()}

        result = _counts_to_expvals(raw, trace.final_batch)

        assert len(result) == len(raw)
        for v in result.values():
            if isinstance(v, dict):
                assert all(isinstance(x, (int, float)) for x in v.values())
            else:
                assert isinstance(v, (int, float))

    def test_multi_obs_group_returns_dict(self, dummy_pipeline_env):
        qscript = qml.tape.QuantumScript(
            ops=[qml.Hadamard(0), qml.Hadamard(1)],
            measurements=[qml.expval(0.5 * qml.Z(0) + 0.3 * qml.Z(1))],
        )
        meta = qscript_to_meta(qscript)

        pipeline = CircuitPipeline(
            stages=[
                DummySpecStage(meta=meta),
                MeasurementStage(grouping_strategy="wires"),
            ]
        )
        trace = pipeline.run_forward_pass("x", dummy_pipeline_env)
        _, lineage_by_label = _compile_batch(trace.final_batch)

        raw: ChildResults = {
            bk: {"00": 70, "01": 5, "10": 15, "11": 10}
            for bk in lineage_by_label.values()
        }

        result = _counts_to_expvals(raw, trace.final_batch)

        dict_results = [v for v in result.values() if isinstance(v, dict)]
        assert len(dict_results) >= 1

        for d in dict_results:
            assert len(d) == 2
            assert all(isinstance(k, int) for k in d.keys())
            assert sorted(d.values()) == pytest.approx([0.5, 0.7])


class TestBatchedExpectation:
    """Tests for _batched_expectation with big-endian Pauli label strings."""

    def test_single_z_observable(self):
        """Z on qubit 0 (big-endian "ZI...I"): all-0 → +1, all-1 → -1."""
        histogram = {"00": 70, "11": 30}
        # big-endian: "ZI" means Z on qubit 0 (leftmost bit in bitstring)
        result = _batched_expectation([histogram], ["ZI"], n_qubits=2)
        # bitstring "00": qubit 0 = 0 → eigenval +1; "11": qubit 0 = 1 → eigenval -1
        expected = (70 * 1 + 30 * (-1)) / 100
        assert result[0, 0] == pytest.approx(expected)

    def test_identity_returns_one(self):
        histogram = {"0": 60, "1": 40}
        result = _batched_expectation([histogram], ["I"], n_qubits=1)
        assert result[0, 0] == pytest.approx(1.0)

    def test_product_observable(self):
        """ZZ on 2 qubits: eigenvalue is product of individual Z eigenvalues."""
        histogram = {"00": 100}
        result = _batched_expectation([histogram], ["ZZ"], n_qubits=2)
        # "00" → both qubits 0 → (+1)(+1) = +1
        assert result[0, 0] == pytest.approx(1.0)

        histogram2 = {"01": 100}
        result2 = _batched_expectation([histogram2], ["ZZ"], n_qubits=2)
        # "01" → qubit 0 = 0 (+1), qubit 1 = 1 (-1) → -1
        assert result2[0, 0] == pytest.approx(-1.0)

    def test_multiple_histograms(self):
        hist_1 = {"00": 100}
        hist_2 = {"11": 50}
        hist_3 = {"01": 25, "10": 75}
        labels = ["ZI", "IZ", "ZZ"]

        result = _batched_expectation([hist_1, hist_2, hist_3], labels, n_qubits=2)

        assert result.shape == (3, 3)
        # hist_1 "00": all eigenvalues +1
        np.testing.assert_allclose(result[:, 0], [1.0, 1.0, 1.0])
        # hist_2 "11": ZI→-1, IZ→-1, ZZ→+1
        np.testing.assert_allclose(result[:, 1], [-1.0, -1.0, 1.0])
        # hist_3 "01"(25%) "10"(75%):
        #   ZI: q0=0→+1(25%), q0=1→-1(75%) = -0.5
        #   IZ: q1=1→-1(25%), q1=0→+1(75%) = +0.5
        #   ZZ: (+1)(-1)(25%), (-1)(+1)(75%) = -1.0
        np.testing.assert_allclose(result[:, 2], [-0.5, 0.5, -1.0])

    def test_multiple_histograms_exact(self):
        """Carefully hand-computed multi-histogram test."""
        hist_1 = {"00": 100}  # all +1 for any Z observable
        hist_2 = {"11": 100}  # Z on any qubit → -1
        labels = ["ZI", "IZ"]

        result = _batched_expectation([hist_1, hist_2], labels, n_qubits=2)
        assert result.shape == (2, 2)
        np.testing.assert_allclose(result[:, 0], [1.0, 1.0])  # hist_1
        np.testing.assert_allclose(result[:, 1], [-1.0, -1.0])  # hist_2

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


class TestFindBatchKey:
    """Tests for _find_batch_key."""

    def test_exact_match(self):
        batch_keys = {("a", "b"), ("c",)}
        assert _find_batch_key(("a", "b"), batch_keys) == ("a", "b")

    def test_subset_match(self):
        batch_keys = {("x",)}
        result = _find_batch_key(("x", "y", "z"), batch_keys)
        assert result == ("x",)

    def test_empty_batch_key_matches_anything(self):
        batch_keys = {()}
        assert _find_batch_key(("a", "b"), batch_keys) == ()

    def test_no_match_raises_key_error(self):
        batch_keys = {("x", "y")}
        with pytest.raises(KeyError, match="No batch key matches branch key"):
            _find_batch_key(("a", "b"), batch_keys)

    def test_empty_batch_keys_set_raises_key_error(self):
        with pytest.raises(KeyError, match="No batch key matches branch key"):
            _find_batch_key(("a",), set())


class TestCountsToProbs:
    """Tests for _counts_to_probs."""

    def test_reverses_bitstrings_and_normalises(self):
        raw = {("obs",): {"100": 30, "010": 70}}
        result = _counts_to_probs(raw, shots=100)
        assert result[("obs",)] == {"001": 0.3, "010": 0.7}

    def test_non_dict_values_pass_through(self):
        raw = {("a",): 0.42, ("b",): {"11": 5, "00": 5}}
        result = _counts_to_probs(raw, shots=10)
        assert result[("a",)] == 0.42
        assert result[("b",)] == {"11": 0.5, "00": 0.5}

    def test_empty_input(self):
        assert _counts_to_probs({}, shots=100) == {}

    def test_multiple_branch_keys(self):
        raw = {
            ("x",): {"10": 4, "01": 6},
            ("y",): {"110": 2, "001": 8},
        }
        result = _counts_to_probs(raw, shots=10)
        assert result[("x",)] == {"01": 0.4, "10": 0.6}
        assert result[("y",)] == {"011": 0.2, "100": 0.8}


class TestExpvalDictsToIndexed:
    """Tests for _expval_dicts_to_indexed."""

    def test_multi_op_returns_indexed_dict(self):
        raw = {("k",): {"XI": 0.5, "IZ": -0.3, "XZ": 0.2}}
        result = _expval_dicts_to_indexed(raw, "XI;IZ;XZ")
        assert result[("k",)] == {0: 0.5, 1: -0.3, 2: 0.2}

    def test_single_op_returns_float(self):
        raw = {("k",): {"ZI": 0.7}}
        result = _expval_dicts_to_indexed(raw, "ZI")
        assert result[("k",)] == pytest.approx(0.7)

    def test_preserves_ham_ops_ordering(self):
        raw = {("k",): {"IZ": -0.3, "XZ": 0.2, "XI": 0.5}}
        result = _expval_dicts_to_indexed(raw, "XI;IZ;XZ")
        assert result[("k",)] == {0: 0.5, 1: -0.3, 2: 0.2}

    def test_non_dict_passthrough(self):
        raw = {("k",): 1.5}
        result = _expval_dicts_to_indexed(raw, "XI;IZ")
        assert result[("k",)] == 1.5
