# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for divi.pipeline.transformations."""

from divi.pipeline.transformations import (
    group_by_base_key,
    reduce_mean,
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


class TestReduceMean:
    def test_averages_per_key(self):
        grouped = {("a",): [1.0, 2.0, 3.0], ("b",): [10.0, 20.0]}
        out = reduce_mean(grouped)
        assert out == {("a",): 2.0, ("b",): 15.0}


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
