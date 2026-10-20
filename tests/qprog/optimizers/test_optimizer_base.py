# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from divi.qprog.optimizers._base import _argmin_finite


@pytest.mark.parametrize(
    "values, expected",
    [
        ([np.nan, 2.0, 1.0], 2),
        ([2.0, np.nan, 1.0], 2),
        ([-np.inf, 1.0], 1),
        ([np.inf, 2.0], 1),
        ([0.5, 0.5, 1.0], 0),
        ([np.float32(1.0), np.float64(0.5)], 1),
        ([np.nan, np.inf], None),
        ([np.nan, -np.inf], None),
        ([np.inf, np.nan], None),
        ([], None),
    ],
    ids=[
        "leading-nan",
        "middle-nan",
        "neg-inf-excluded",
        "pos-inf-excluded",
        "tie-takes-first",
        "mixed-float-widths",
        "nan-and-inf",
        "nan-and-neg-inf",
        "inf-and-nan",
        "empty",
    ],
)
def test_argmin_finite_picks_the_smallest_finite_entry(values, expected):
    assert _argmin_finite(values) == expected
