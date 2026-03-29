# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Numba JIT-compiled kernels for PCE bitstring decoding.

Provides an accelerated popcount-parity computation using XOR-fold bit
manipulation instead of lookup tables and intermediate array allocations.
"""

import numba
import numpy as np
import numpy.typing as npt


@numba.njit(cache=True)
def _popcount_parity_jit(arr: npt.NDArray[np.uint64]) -> npt.NDArray[np.uint8]:
    """Compute ``popcount(x) % 2`` for each element via XOR-fold.

    Six bitwise operations per element with no intermediate arrays or
    lookup tables.  Operates on the flattened view and reshapes to match
    the input shape.
    """
    flat = arr.ravel()
    out = np.empty(flat.shape[0], dtype=np.uint8)
    for i in range(flat.shape[0]):
        x = flat[i]
        x ^= x >> np.uint64(32)
        x ^= x >> np.uint64(16)
        x ^= x >> np.uint64(8)
        x ^= x >> np.uint64(4)
        x ^= x >> np.uint64(2)
        x ^= x >> np.uint64(1)
        out[i] = np.uint8(x & np.uint64(1))
    return out.reshape(arr.shape)
