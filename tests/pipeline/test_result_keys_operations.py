# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for divi.pipeline._result_keys_operations."""

import numpy as np
import pytest

from divi.pipeline._result_keys_operations import (
    _collapse_to_parent_results,
    _find_batch_key,
    average_by_param_set,
    extract_param_set_idx,
    group_by_base_key,
    group_by_branch_and_param_set,
    reduce_mean,
    reduce_merge_histograms,
    reduce_postprocess_ordered,
    strip_axis_from_label,
)


class TestStripAxisFromLabel:
    def test_strips_named_axis(self):
        label = (("ham", 0), ("obs_group", 2), ("other", "x"))
        assert strip_axis_from_label(label, "ham") == (("obs_group", 2), ("other", "x"))
        assert strip_axis_from_label(label, "obs_group") == (("ham", 0), ("other", "x"))

    def test_returns_empty_when_only_axis(self):
        assert strip_axis_from_label((("ham", 0),), "ham") == ()

    def test_leaves_label_unchanged_when_axis_absent(self):
        label = (("obs_group", 1),)
        assert strip_axis_from_label(label, "ham") == (("obs_group", 1),)


class TestGroupByBaseKey:
    def test_indexed_false_collects_lists(self):
        results = {
            (("ham", 0),): 10.0,
            (("ham", 1),): 20.0,
            (("ham", 2),): 30.0,
        }
        grouped = group_by_base_key(results, "ham", indexed=False)
        assert grouped == {(): [10.0, 20.0, 30.0]}

    def test_indexed_true_keyed_by_axis_value(self):
        results = {
            (("qem_foo", 0),): "a",
            (("qem_foo", 1),): "b",
            (("qem_foo", 2),): "c",
        }
        grouped = group_by_base_key(results, "qem_foo", indexed=True)
        assert grouped == {(): {0: "a", 1: "b", 2: "c"}}

    def test_multiple_base_keys(self):
        results = {
            (("spec", "A"), ("obs_group", 0)): 1.0,
            (("spec", "A"), ("obs_group", 1)): 2.0,
            (("spec", "B"), ("obs_group", 0)): 3.0,
        }
        grouped = group_by_base_key(results, "obs_group", indexed=True)
        assert grouped == {
            (("spec", "A"),): {0: 1.0, 1: 2.0},
            (("spec", "B"),): {0: 3.0},
        }


class TestExtractParamSetIdx:
    def test_parses_param_set_axis_from_nodekey(self):
        assert extract_param_set_idx((("ham", 0), ("param_set", 2))) == 2

    def test_missing_axis_raises_keyerror(self):
        with pytest.raises(KeyError, match="param_set"):
            extract_param_set_idx((("ham", 0),))

    def test_default_returned_when_axis_absent(self):
        assert extract_param_set_idx((("ham", 0),), default=0) == 0

    def test_non_tuple_key_raises_actionable_typeerror(self):
        """Calling it on evaluate() output (int keys) gives an actionable error,
        not a bare 'int object is not iterable'."""
        with pytest.raises(TypeError, match="NodeKey tuple"):
            extract_param_set_idx(0)


class TestReduceMean:
    def test_averages_per_key(self):
        grouped = {("a",): [1.0, 2.0, 3.0], ("b",): [10.0, 20.0]}
        out = reduce_mean(grouped)
        assert out == {("a",): 2.0, ("b",): 15.0}

    def test_averages_per_observable_lists_elementwise(self):
        """When values are per-observable list[float] (multi-observable
        MeasurementStage output), each observable's mean is preserved."""
        grouped = {
            ("a",): [[1.0, 5.0], [3.0, 7.0]],
            ("b",): [[2.0, 4.0, -2.0], [4.0, 0.0, 2.0]],
        }
        out = reduce_mean(grouped)
        assert out[("a",)] == pytest.approx([2.0, 6.0])
        assert out[("b",)] == pytest.approx([3.0, 2.0, 0.0])

    def test_mixed_keys_dispatch_independently(self):
        """Per-key dispatch: scalar key averaged scalar-wise, list key elementwise."""
        grouped = {
            ("scalar",): [1.0, 3.0],
            ("list",): [[1.0, 2.0], [3.0, 4.0]],
        }
        out = reduce_mean(grouped)
        assert out[("scalar",)] == pytest.approx(2.0)
        assert out[("list",)] == pytest.approx([2.0, 3.0])

    def test_rejects_probability_dicts_with_actionable_error(self):
        """A PROBS/COUNTS histogram dict raises a TypeError naming the right
        helper, instead of a raw 'int + dict' TypeError from the sum()."""
        grouped = {(("circ", 0),): [{"00": 0.6, "11": 0.4}, {"00": 0.8, "11": 0.2}]}
        with pytest.raises(TypeError, match="reduce_merge_histograms"):
            reduce_mean(grouped)


class TestReducePostprocessOrdered:
    def test_single_postprocess_fn(self):
        grouped = {(): {0: 1, 1: 2, 2: 3}}
        out = reduce_postprocess_ordered(grouped, lambda ordered: sum(ordered))
        assert out == {(): 6}

    def test_per_key_postprocess_fn(self):
        grouped = {
            ("A",): {0: 1, 1: 2},
            ("B",): {0: 10, 1: 20},
        }
        fns = {("A",): lambda v: v[0] - v[1], ("B",): lambda v: v[0] + v[1]}
        out = reduce_postprocess_ordered(grouped, fns)
        assert out == {("A",): -1, ("B",): 30}


class TestReduceMergeHistograms:
    """Tests for reduce_merge_histograms: probability histogram averaging across ham samples."""

    def test_merges_two_histograms(self):
        """Averages probability dicts across two Hamiltonian samples."""
        grouped = {(("circ", 0),): [{"00": 0.8, "11": 0.2}, {"00": 0.6, "11": 0.4}]}

        result = reduce_merge_histograms(grouped)

        assert (("circ", 0),) in result
        assert result[(("circ", 0),)]["00"] == pytest.approx(0.7)
        assert result[(("circ", 0),)]["11"] == pytest.approx(0.3)

    def test_merged_probabilities_sum_to_one(self):
        """Merged probability distribution sums to 1.0."""
        grouped = {
            (("circ", 0),): [
                {"00": 0.5, "01": 0.3, "10": 0.15, "11": 0.05},
                {"00": 0.2, "01": 0.4, "10": 0.1, "11": 0.3},
                {"00": 0.3, "01": 0.3, "10": 0.2, "11": 0.2},
            ]
        }

        result = reduce_merge_histograms(grouped)
        prob_dict = result[(("circ", 0),)]
        total = sum(prob_dict.values())
        assert total == pytest.approx(1.0)

    def test_handles_disjoint_bitstrings(self):
        """Averaging when histograms have different bitstring keys uses 0 for missing."""
        grouped = {
            (("circ", 0),): [
                {"00": 0.8, "11": 0.2},
                {"01": 0.6, "10": 0.4},
            ]
        }

        result = reduce_merge_histograms(grouped)
        prob_dict = result[(("circ", 0),)]
        assert prob_dict["00"] == pytest.approx(0.4)
        assert prob_dict["11"] == pytest.approx(0.1)
        assert prob_dict["01"] == pytest.approx(0.3)
        assert prob_dict["10"] == pytest.approx(0.2)

    def test_empty_prob_dicts(self):
        """Empty list of prob dicts returns empty dict."""
        grouped = {(("circ", 0),): []}

        result = reduce_merge_histograms(grouped)
        assert result[(("circ", 0),)] == {}

    def test_single_histogram_identity(self):
        """Single histogram merges to itself."""
        grouped = {(("circ", 0),): [{"00": 0.7, "11": 0.3}]}

        result = reduce_merge_histograms(grouped)
        assert result[(("circ", 0),)]["00"] == pytest.approx(0.7)
        assert result[(("circ", 0),)]["11"] == pytest.approx(0.3)

    def test_multiple_base_keys(self):
        """Multiple base keys are each merged independently."""
        grouped = {
            (("circ", 0),): [{"00": 0.8, "11": 0.2}, {"00": 0.6, "11": 0.4}],
            (("circ", 1),): [{"01": 1.0}, {"01": 0.5, "10": 0.5}],
        }

        result = reduce_merge_histograms(grouped)

        assert result[(("circ", 0),)]["00"] == pytest.approx(0.7)
        assert result[(("circ", 0),)]["11"] == pytest.approx(0.3)
        assert result[(("circ", 1),)]["01"] == pytest.approx(0.75)
        assert result[(("circ", 1),)]["10"] == pytest.approx(0.25)

    def test_rejects_expval_values_with_actionable_error(self):
        """An EXPVALS float raises a TypeError naming the right helper, instead
        of a raw 'float has no attribute keys' AttributeError."""
        grouped = {(("circ", 0),): [1.0, 3.0]}
        with pytest.raises(TypeError, match="reduce_mean"):
            reduce_merge_histograms(grouped)


def test_average_by_param_set_collapses_preserved_axes():
    result = {
        (("ham", 0), ("param_set", 0)): [1.0, 3.0],
        (("ham", 1), ("param_set", 0)): [3.0, 5.0],
        (("ham", 0), ("param_set", 1)): [10.0, 12.0],
    }

    averaged = average_by_param_set(result, lambda value: np.asarray(value))

    assert set(averaged) == {0, 1}
    np.testing.assert_allclose(averaged[0], [2.0, 4.0])
    np.testing.assert_allclose(averaged[1], [10.0, 12.0])


def test_group_by_branch_and_param_set_keeps_preserved_axes():
    result = {
        (("ham", 0), ("param_set", 0)): [1.0],
        (("ham", 0), ("param_set", 1)): [2.0],
        (("ham", 1), ("param_set", 0)): [3.0],
    }

    grouped = group_by_branch_and_param_set(result, lambda value: np.asarray(value))

    assert set(grouped) == {(("ham", 0),), (("ham", 1),)}
    np.testing.assert_allclose(grouped[(("ham", 0),)][0], [1.0])
    np.testing.assert_allclose(grouped[(("ham", 0),)][1], [2.0])
    np.testing.assert_allclose(grouped[(("ham", 1),)][0], [3.0])


class TestCollapseToParentResults:
    """Spec: _collapse_to_parent_results maps backend labels back to BranchKeys."""

    def test_maps_labels_to_branch_keys(self):
        spec_circ = ("spec", "circ")
        lineage = {
            "a/obs_group:0": (spec_circ, ("obs_group", 0)),
            "b/obs_group:1": (spec_circ, ("obs_group", 1)),
        }
        raw = {"a/obs_group:0": 1.0, "b/obs_group:1": 2.0}
        out = _collapse_to_parent_results(raw, lineage)
        assert out[(spec_circ, ("obs_group", 0))] == 1.0
        assert out[(spec_circ, ("obs_group", 1))] == 2.0

    def test_ignores_unknown_labels(self):
        lineage = {"only": (("spec", "k"),)}
        raw = {"only": 1, "unknown": 2}
        out = _collapse_to_parent_results(raw, lineage)
        assert out == {(("spec", "k"),): 1}


class TestFindBatchKey:
    """Spec: _find_batch_key routes a branch key to its subset batch key."""

    def test_exact_match(self):
        batch_keys = {("a", "b"), ("c",)}
        assert _find_batch_key(("a", "b"), batch_keys) == ("a", "b")

    def test_subset_match(self):
        batch_keys = {("x",)}
        assert _find_batch_key(("x", "y", "z"), batch_keys) == ("x",)

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
