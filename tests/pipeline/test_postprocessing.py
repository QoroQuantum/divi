# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pennylane as qml
import pytest
from pennylane.measurements import ExpectationMP

from divi.circuits import MetaCircuit
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
        """_counts_to_expvals should convert counts dicts to float expvals."""
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
            # Single-obs groups → scalar, multi-obs groups → {obs_idx: float}.
            if isinstance(v, dict):
                assert all(isinstance(x, (int, float)) for x in v.values())
            else:
                assert isinstance(v, (int, float))

    def test_multi_obs_group_returns_dict(self, dummy_pipeline_env):
        """When multiple observables share a measurement group, result is a dict.

        Z(0) and Z(1) operate on different wires, so with 'wires' strategy
        they land in the same group → len(col) > 1 → dict return.

        Hand-derived expected values from counts {"00": 70, "01": 5, "10": 15, "11": 10}:
          ⟨Z(0)⟩ and ⟨Z(1)⟩ are 0.5 and 0.7 (order depends on wire ordering),
          so sorted values are [0.5, 0.7] regardless of convention.
        """
        # Build a MetaCircuit with a 2-term Hamiltonian on separate wires
        qscript = qml.tape.QuantumScript(
            ops=[qml.Hadamard(0), qml.Hadamard(1)],
            measurements=[qml.expval(0.5 * qml.Z(0) + 0.3 * qml.Z(1))],
        )
        meta = MetaCircuit(source_circuit=qscript, symbols=np.array([], dtype=object))

        pipeline = CircuitPipeline(
            stages=[
                DummySpecStage(meta=meta),
                MeasurementStage(grouping_strategy="wires"),
            ]
        )
        trace = pipeline.run_forward_pass("x", dummy_pipeline_env)
        _, lineage_by_label = _compile_batch(trace.final_batch)

        # Asymmetric counts so Z(0) and Z(1) give different expectation values
        raw: ChildResults = {
            bk: {"00": 70, "01": 5, "10": 15, "11": 10}
            for bk in lineage_by_label.values()
        }

        result = _counts_to_expvals(raw, trace.final_batch)

        # At least one result should be a dict (multi-obs group)
        dict_results = [v for v in result.values() if isinstance(v, dict)]
        assert len(dict_results) >= 1

        for d in dict_results:
            assert len(d) == 2
            assert all(isinstance(k, int) for k in d.keys())
            # Sorted expvals are [0.5, 0.7] regardless of wire ordering
            assert sorted(d.values()) == pytest.approx([0.5, 0.7])


class TestBatchedExpectation:
    """Test suite for batched expectation value calculations."""

    def test_matches_pennylane_baseline(self):
        """
        Validates that the optimized batched_expectation function produces results
        identical to PennyLane's standard ExpectationMP processing.
        """
        wire_order = (3, 2, 1, 0)
        shot_histogram = {"0000": 100, "0101": 200, "1011": 300, "1111": 400}
        observables = [
            qml.PauliZ(0),
            qml.PauliZ(2),
            qml.Identity(1),
            qml.PauliZ(1) @ qml.PauliZ(3),
        ]

        baseline_expvals = []
        for obs in observables:
            mp = ExpectationMP(obs)
            expval = mp.process_counts(counts=shot_histogram, wire_order=wire_order)
            baseline_expvals.append(expval)

        optimized_expvals_matrix = _batched_expectation(
            [shot_histogram], observables, wire_order
        )
        optimized_expvals = optimized_expvals_matrix[:, 0]

        assert isinstance(optimized_expvals, np.ndarray)
        np.testing.assert_allclose(optimized_expvals, baseline_expvals)

    def test_with_multiple_histograms(self):
        """
        Tests that batched_expectation correctly processes a list of different
        shot histograms in a single call.
        """
        hist_1 = {"00": 100}
        hist_2 = {"11": 50}
        hist_3 = {"01": 25, "10": 75}
        observables = [qml.PauliZ(0), qml.PauliZ(1), qml.PauliZ(0) @ qml.PauliZ(1)]
        wire_order = (1, 0)

        expected_1 = np.array([1.0, 1.0, 1.0])
        expected_2 = np.array([-1.0, -1.0, 1.0])
        expected_3 = np.array([0.5, -0.5, -1.0])

        result_matrix = _batched_expectation(
            [hist_1, hist_2, hist_3], observables, wire_order
        )

        assert result_matrix.shape == (3, 3)
        np.testing.assert_allclose(result_matrix[:, 0], expected_1)
        np.testing.assert_allclose(result_matrix[:, 1], expected_2)
        np.testing.assert_allclose(result_matrix[:, 2], expected_3)

    def test_raises_for_unsupported_observable(self):
        """
        Ensures that a KeyError is raised when an observable outside
        the supported set (Pauli, Identity) is provided.
        """
        shots = {"0": 100}
        wire_order = (0,)
        unsupported_observables = [qml.PauliZ(0), qml.Hadamard(0)]

        with pytest.raises(KeyError):
            _batched_expectation(
                shots_dicts=[shots],
                observables=unsupported_observables,
                wire_order=wire_order,
            )

    @pytest.mark.parametrize(
        "n_qubits",
        [1, 10, 32, 64, 65, 100, 150, 200, 500, 1000],
        ids=lambda n: f"{n}-qubits",
    )
    def test_qubit_counts_no_overflow(self, n_qubits):
        """
        Tests that _batched_expectation works correctly across a wide range of qubit counts
        without integer overflow errors, including the critical boundary at 64.

        This test would have caught the bug where 150-qubit circuits caused OverflowError
        when converting bitstrings to uint64.
        """
        wire_order = tuple(range(n_qubits - 1, -1, -1))

        # Create test bitstrings including edge cases that would overflow uint64
        test_bitstrings = [
            "0" * n_qubits,  # All 0s
            "1" * n_qubits,  # All 1s - would overflow uint64 for >64 qubits
            "1" + "0" * (n_qubits - 1),  # MSB=1, rest 0s
            "0" * (n_qubits - 1) + "1",  # LSB=1, rest 0s
        ]

        shot_histogram = {bs: 100 for bs in test_bitstrings}

        # Test observables on first, middle, and last qubits
        # Ensure mid_qubit is within valid wire range [0, n_qubits-1]
        mid_qubit = n_qubits // 2 if n_qubits > 1 else 0
        observables = [
            qml.PauliZ(0),
            qml.PauliZ(n_qubits - 1),
        ]
        # Add Identity observable if we have a valid middle qubit
        if n_qubits > 1:
            observables.append(qml.Identity(mid_qubit))
        # Add product observable for larger systems
        if n_qubits >= 50:
            observables.append(qml.PauliZ(mid_qubit // 2) @ qml.PauliZ(mid_qubit))

        # Remove duplicates
        observables = list(dict.fromkeys(observables))

        # Should not raise OverflowError
        result = _batched_expectation([shot_histogram], observables, wire_order)

        # Verify results are valid
        assert result.shape == (len(observables), 1)
        assert not np.isnan(result).any(), "Results should not contain NaN"
        assert not np.isinf(result).any(), "Results should not contain Inf"

        # Verify expectation values are in valid range
        for i, obs in enumerate(observables):
            if isinstance(obs, qml.PauliZ) or (
                hasattr(obs, "name") and obs.name == "Prod"
            ):
                assert -1.0 <= result[i, 0] <= 1.0

    def test_boundary_qubit_count_both_paths_work(self):
        """
        Tests that both code paths (<=64 and >64 qubits) work correctly at the boundary.

        This ensures the conditional logic correctly switches between integer and
        character array representations.
        """
        # Test at exactly 64 qubits (uses integer path)
        wire_order_64 = tuple(range(63, -1, -1))
        shot_histogram_64 = {"1" * 64: 100, "0" * 64: 100}
        observables_64 = [qml.PauliZ(0), qml.PauliZ(31), qml.PauliZ(63)]

        result_64 = _batched_expectation(
            [shot_histogram_64], observables_64, wire_order_64
        )

        # Test at 65 qubits (uses character array path)
        wire_order_65 = tuple(range(64, -1, -1))
        shot_histogram_65 = {"1" * 65: 100, "0" * 65: 100}
        observables_65 = [qml.PauliZ(0), qml.PauliZ(32), qml.PauliZ(64)]

        result_65 = _batched_expectation(
            [shot_histogram_65], observables_65, wire_order_65
        )

        # Both should complete without errors and produce valid results
        assert result_64.shape[0] == len(observables_64)
        assert result_65.shape[0] == len(observables_65)
        assert not np.isnan(result_64).any()
        assert not np.isnan(result_65).any()

    @pytest.mark.parametrize(
        "n_qubits",
        [4, 32, 64, 65, 100, 150, 500, 1000],
        ids=lambda n: f"{n}qubits",
    )
    def test_matches_pennylane_baseline(self, n_qubits):
        """
        Validates that results match PennyLane's baseline implementation across
        various qubit counts, ensuring correctness of both code paths.
        """
        wire_order = tuple(range(n_qubits - 1, -1, -1))
        shot_histogram = {"0" * n_qubits: 50, "1" * n_qubits: 50}
        observables = [qml.PauliZ(0)]

        # Get baseline from PennyLane
        baseline_expvals = []
        for obs in observables:
            mp = ExpectationMP(obs)
            expval = mp.process_counts(counts=shot_histogram, wire_order=wire_order)
            baseline_expvals.append(expval)

        # Get result from our optimized function
        optimized_expvals = _batched_expectation(
            [shot_histogram], observables, wire_order
        )[:, 0]

        # Should match PennyLane's results
        np.testing.assert_allclose(optimized_expvals, baseline_expvals, rtol=1e-10)

    @pytest.mark.parametrize(
        "n_qubits,observable_wires",
        [
            (100, [0, 50]),
            (150, [0, 50, 100]),
            (150, [0, 75, 125]),
            (200, [0, 100, 199]),
            (500, [0, 250, 499]),
            (1000, [0, 500, 999]),
        ],
        ids=["100q_2w", "150q_3w", "150q_3w_alt", "200q_3w", "500q_3w", "1000q_3w"],
    )
    def test_product_observables_large_qubit_counts(self, n_qubits, observable_wires):
        """
        Tests that product observables (multi-qubit) work correctly for large qubit counts.
        """
        wire_order = tuple(range(n_qubits - 1, -1, -1))
        shot_histogram = {
            "0" * n_qubits: 100,
            "1" * n_qubits: 100,
            "1" + "0" * (n_qubits - 1): 100,
        }

        # Create product observable from wire indices
        obs = qml.PauliZ(observable_wires[0])
        for wire in observable_wires[1:]:
            obs = obs @ qml.PauliZ(wire)
        observables = [obs]

        result = _batched_expectation([shot_histogram], observables, wire_order)

        # Verify results
        assert result.shape == (1, 1)
        assert not np.isnan(result).any()
        # Product observables should be in range [-1, 1]
        assert np.abs(result[0, 0]) <= 1.0 + 1e-10


class TestFindBatchKey:
    """Tests for _find_batch_key."""

    def test_exact_match(self):
        """Branch key that equals a batch key is found."""
        batch_keys = {("a", "b"), ("c",)}
        assert _find_batch_key(("a", "b"), batch_keys) == ("a", "b")

    def test_subset_match(self):
        """Batch key whose axes are a subset of the branch key is found."""
        batch_keys = {("x",)}
        result = _find_batch_key(("x", "y", "z"), batch_keys)
        assert result == ("x",)

    def test_empty_batch_key_matches_anything(self):
        """An empty batch key is a subset of every branch key."""
        batch_keys = {()}
        assert _find_batch_key(("a", "b"), batch_keys) == ()

    def test_no_match_raises_key_error(self):
        """KeyError is raised when no batch key is a subset of the branch key."""
        batch_keys = {("x", "y")}
        with pytest.raises(KeyError, match="No batch key matches branch key"):
            _find_batch_key(("a", "b"), batch_keys)

    def test_empty_batch_keys_set_raises_key_error(self):
        """KeyError is raised when batch_keys is empty."""
        with pytest.raises(KeyError, match="No batch key matches branch key"):
            _find_batch_key(("a",), set())


class TestCountsToProbs:
    """Tests for _counts_to_probs."""

    def test_reverses_bitstrings_and_normalises(self):
        """Bitstrings are reversed (PennyLane->MSB-first) and divided by shots."""
        raw = {("obs",): {"100": 30, "010": 70}}
        result = _counts_to_probs(raw, shots=100)
        # "100" reversed -> "001", "010" reversed -> "010"
        assert result[("obs",)] == {"001": 0.3, "010": 0.7}

    def test_non_dict_values_pass_through(self):
        """Non-dict results (e.g. floats from expval) are left unchanged."""
        raw = {("a",): 0.42, ("b",): {"11": 5, "00": 5}}
        result = _counts_to_probs(raw, shots=10)
        assert result[("a",)] == 0.42
        assert result[("b",)] == {"11": 0.5, "00": 0.5}

    def test_empty_input(self):
        """Empty ChildResults returns empty output."""
        assert _counts_to_probs({}, shots=100) == {}

    def test_multiple_branch_keys(self):
        """Each branch key is processed independently."""
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
        """Multiple Pauli ops produce {int: float} dicts."""
        raw = {("k",): {"XI": 0.5, "IZ": -0.3, "XZ": 0.2}}
        result = _expval_dicts_to_indexed(raw, "XI;IZ;XZ")
        assert result[("k",)] == {0: 0.5, 1: -0.3, 2: 0.2}

    def test_single_op_returns_float(self):
        """Single Pauli op produces a plain float."""
        raw = {("k",): {"ZI": 0.7}}
        result = _expval_dicts_to_indexed(raw, "ZI")
        assert result[("k",)] == pytest.approx(0.7)

    def test_preserves_ham_ops_ordering(self):
        """Output indices follow ham_ops order, not dict key order."""
        raw = {("k",): {"IZ": -0.3, "XZ": 0.2, "XI": 0.5}}
        result = _expval_dicts_to_indexed(raw, "XI;IZ;XZ")
        assert result[("k",)] == {0: 0.5, 1: -0.3, 2: 0.2}

    def test_non_dict_passthrough(self):
        """Non-dict values (e.g. already-normalised floats) pass through."""
        raw = {("k",): 1.5}
        result = _expval_dicts_to_indexed(raw, "XI;IZ")
        assert result[("k",)] == 1.5
