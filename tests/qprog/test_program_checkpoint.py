# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the JSON coercion of checkpointed program state."""

import json

import numpy as np
import pytest

from divi.qprog._program_checkpoint import SubclassState, _to_jsonable


@pytest.mark.parametrize(
    "value, expected",
    [
        (np.int64(3), 3),
        (np.float32(0.5), 0.5),
        (np.bool_(True), True),
        (np.array([[1, 2], [3, 4]]), [[1, 2], [3, 4]]),
        ({"bits": np.array([0, 1])}, {"bits": [0, 1]}),
        ({np.int64(1): np.float64(2.5)}, {1: 2.5}),
        ({np.str_("a"): np.int32(7)}, {"a": 7}),
        ({np.bool_(False): 1}, {False: 1}),
        ({"nested": {np.int64(2): (np.int64(3), 4)}}, {"nested": {2: [3, 4]}}),
    ],
    ids=[
        "int",
        "float32",
        "bool",
        "array",
        "array-value",
        "numpy-int-key",
        "numpy-str-key",
        "numpy-bool-key",
        "nested-key",
    ],
)
def test_numpy_values_and_keys_become_plain_python(value, expected):
    result = _to_jsonable(value)

    assert result == expected
    if isinstance(result, dict):
        assert {type(key) for key in result} == {type(key) for key in expected}


def test_numpy_keyed_mapping_survives_the_checkpoint_serialiser():
    state = SubclassState(data=_to_jsonable({"solution": {np.int64(0): np.int8(1)}}))

    restored = json.loads(state.model_dump_json())

    assert restored["data"] == {"solution": {"0": 1}}


def test_tuple_keys_are_left_untouched():
    assert _to_jsonable({(1, 2): 3}) == {(1, 2): 3}
