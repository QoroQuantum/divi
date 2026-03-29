# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Numba JIT-compiled kernel for QH1 histogram decompression.

Splits the decompression into a JIT-friendly inner kernel that operates on a
``uint8`` array (the raw blob) and returns integer index/count arrays, plus a
thin Python wrapper that converts the output to ``dict[str, int]``.
"""

import numba
import numpy as np
import numpy.typing as npt


@numba.njit(cache=True)
def _uleb128_decode_jit(data: npt.NDArray[np.uint8], pos: int) -> tuple[int, int]:
    """Decode a single ULEB128 varint from a uint8 array."""
    x = np.int64(0)
    shift = np.int64(0)
    while pos < data.shape[0]:
        b = np.int64(data[pos])
        pos += 1
        x |= (b & np.int64(0x7F)) << shift
        if (b & np.int64(0x80)) == 0:
            return int(x), pos
        shift += np.int64(7)
    return int(x), pos  # truncated — caller validates


@numba.njit(cache=True)
def _decompress_histogram_jit(
    data: npt.NDArray[np.uint8],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], int, int, int, int]:
    """JIT inner kernel: decompress a QH1 blob into (indices, counts) arrays.

    Args:
        data: Raw QH1 blob as a uint8 array.

    Returns:
        ``(indices, counts, n_bits, unique, total_shots, n_decoded)`` where
        *indices* and *counts* are ``int64`` arrays of length *n_decoded*.
        *unique* is the header-declared count (may differ from *n_decoded*
        in corrupt streams).
    """
    pos = 3  # skip "QH1" magic (validated by caller)
    n_bits = int(data[pos])
    pos += 1

    unique, pos = _uleb128_decode_jit(data, pos)
    total_shots, pos = _uleb128_decode_jit(data, pos)

    # --- Decode gaps → indices ---
    num_gaps, pos = _uleb128_decode_jit(data, pos)
    indices = np.empty(num_gaps, dtype=np.int64)
    acc = np.int64(0)
    for i in range(num_gaps):
        g, pos = _uleb128_decode_jit(data, pos)
        if i == 0:
            acc = np.int64(g)
        else:
            acc += np.int64(g)
        indices[i] = acc

    # --- RLE bool decode (compute actual length from run data) ---
    rb_len, pos = _uleb128_decode_jit(data, pos)
    rle_start = pos
    pos_rle = rle_start

    num_runs, pos_rle = _uleb128_decode_jit(data, pos_rle)

    # First pass: compute total RLE length
    rle_total = 0
    pos_scan = pos_rle
    if num_runs > 0:
        pos_scan += 1  # skip first_val byte
        for _ in range(num_runs):
            ln, pos_scan = _uleb128_decode_jit(data, pos_scan)
            rle_total += ln

    is_one = np.empty(rle_total, dtype=numba.boolean)

    # Second pass: fill is_one
    if num_runs > 0:
        first_val = data[pos_rle] != 0
        pos_rle += 1
        val = first_val
        fill_pos = 0
        for _ in range(num_runs):
            ln, pos_rle = _uleb128_decode_jit(data, pos_rle)
            for j in range(ln):
                is_one[fill_pos] = val
                fill_pos += 1
            val = not val

    pos = rle_start + rb_len
    n_decoded = rle_total

    # --- Extras ---
    extras_len, pos = _uleb128_decode_jit(data, pos)
    extras = np.empty(extras_len, dtype=np.int64)
    for i in range(extras_len):
        e, pos = _uleb128_decode_jit(data, pos)
        extras[i] = np.int64(e)

    # --- Build counts ---
    counts = np.empty(n_decoded, dtype=np.int64)
    extra_idx = 0
    for i in range(n_decoded):
        if is_one[i]:
            counts[i] = 1
        else:
            counts[i] = extras[extra_idx] + 2
            extra_idx += 1

    return indices, counts, n_bits, unique, total_shots, n_decoded
