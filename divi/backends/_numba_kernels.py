# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Numba JIT-compiled kernel for QH1 histogram decompression.

Indices are represented as little-endian ``uint64`` limb arrays so the
decoder works uniformly for arbitrary ``n_bits`` (including circuits with
more than 64 qubits).  A single ULEB128 primitive
(:func:`_uleb128_decode_limbs_jit`) is used for every varint in the stream;
callers allocate a scratch limb buffer sized for the value they expect.
The primitive raises a :class:`ValueError` when the encoded value does not
fit in the caller-supplied buffer, converting silent-corruption failure
modes into loud ones.
"""

import numba
import numpy as np
import numpy.typing as npt


@numba.njit(cache=True, nogil=True)
def _uleb128_decode_limbs_jit(
    data: npt.NDArray[np.uint8],
    pos: int,
    out: npt.NDArray[np.uint64],
) -> int:
    """Decode a single ULEB128 varint into a little-endian uint64 limb buffer.

    ``out`` is zeroed on entry and then filled with the decoded value.  The
    buffer must be large enough to hold every bit of the varint; if the
    varint would spill past ``out``, a ``ValueError`` is raised so that
    corrupt streams fail loudly instead of returning a silently-truncated
    value.

    Args:
        data: Raw byte stream as a uint8 array.
        pos: Byte offset at which the varint starts.
        out: Caller-owned uint64 limb buffer, little-endian.

    Returns:
        The byte offset immediately after the decoded varint.
    """
    for i in range(out.shape[0]):
        out[i] = np.uint64(0)

    shift = 0
    while pos < data.shape[0]:
        b = np.uint64(data[pos])
        pos += 1
        payload = b & np.uint64(0x7F)
        limb_idx = shift // 64
        bit_off = shift % 64

        if limb_idx >= out.shape[0]:
            raise ValueError("ULEB128 value exceeds destination width")

        out[limb_idx] |= payload << np.uint64(bit_off)

        # Payload straddles into the next limb when bit_off > 64 - 7 == 57.
        if bit_off > 57:
            high = payload >> np.uint64(64 - bit_off)
            if high != np.uint64(0):
                if limb_idx + 1 >= out.shape[0]:
                    raise ValueError("ULEB128 value exceeds destination width")
                out[limb_idx + 1] |= high

        if (b & np.uint64(0x80)) == np.uint64(0):
            return pos
        shift += 7

    raise ValueError("truncated ULEB128 varint")


@numba.njit(cache=True, nogil=True)
def _decompress_histogram_jit(
    data: npt.NDArray[np.uint8],
) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.int64], int, int, int, int, int]:
    """JIT inner kernel: decompress a QH1 blob into (indices, counts) arrays.

    Args:
        data: Raw QH1 blob as a uint8 array.

    Returns:
        ``(indices, counts, n_bits, L, unique, total_shots, n_decoded)`` where
        ``indices`` is a 2D ``uint64`` array of shape ``(n_decoded, L)``
        storing each bitstring index as a little-endian limb sequence, and
        ``counts`` is an ``int64`` array of length ``n_decoded``.
        ``L = ceil(n_bits / 64)``.  ``unique`` is the header-declared count
        (may differ from ``n_decoded`` in corrupt streams).
    """
    pos = 3  # skip "QH1" magic (validated by caller)
    n_bits = int(data[pos])
    pos += 1

    L = (n_bits + 63) // 64
    if L < 1:
        L = 1

    # Scratch buffers ­— allocated once, reused for every varint read.
    scratch1 = np.zeros(1, dtype=np.uint64)
    gap_scratch = np.zeros(L, dtype=np.uint64)

    pos = _uleb128_decode_limbs_jit(data, pos, scratch1)
    unique = int(scratch1[0])
    pos = _uleb128_decode_limbs_jit(data, pos, scratch1)
    total_shots = int(scratch1[0])

    # --- Decode gaps → indices ---
    pos = _uleb128_decode_limbs_jit(data, pos, scratch1)
    num_gaps = int(scratch1[0])

    indices = np.zeros((num_gaps, L), dtype=np.uint64)
    # Running accumulator (current index = sum of gaps so far).
    acc = np.zeros(L, dtype=np.uint64)

    for i in range(num_gaps):
        pos = _uleb128_decode_limbs_jit(data, pos, gap_scratch)
        # acc += gap_scratch with ripple carry across limbs.
        carry = np.uint64(0)
        for k in range(L):
            s1 = acc[k] + gap_scratch[k]
            c1 = np.uint64(1) if s1 < acc[k] else np.uint64(0)
            s2 = s1 + carry
            c2 = np.uint64(1) if s2 < s1 else np.uint64(0)
            acc[k] = s2
            carry = c1 + c2
        # A non-zero carry out of the top limb means the accumulated index
        # overflowed the declared bit width — catch it here so that a
        # malicious stream whose gap sum wraps past 2**(64*L) (the edge case
        # when n_bits is a multiple of 64 and the top-limb validator is
        # skipped) can't silently return a wrong bitstring.
        if carry != np.uint64(0):
            raise ValueError("accumulated index overflows limb buffer")
        # Copy accumulator into the indices row.
        for k in range(L):
            indices[i, k] = acc[k]

    # --- RLE bool decode (compute actual length from run data) ---
    pos = _uleb128_decode_limbs_jit(data, pos, scratch1)
    rb_len = int(scratch1[0])

    rle_start = pos
    pos_rle = rle_start

    pos_rle = _uleb128_decode_limbs_jit(data, pos_rle, scratch1)
    num_runs = int(scratch1[0])

    # First pass: compute total RLE length
    rle_total = 0
    pos_scan = pos_rle
    if num_runs > 0:
        pos_scan += 1  # skip first_val byte
        for _ in range(num_runs):
            pos_scan = _uleb128_decode_limbs_jit(data, pos_scan, scratch1)
            rle_total += int(scratch1[0])

    is_one = np.empty(rle_total, dtype=np.bool_)

    # Second pass: fill is_one
    if num_runs > 0:
        first_val = data[pos_rle] != 0
        pos_rle += 1
        val = first_val
        fill_pos = 0
        for _ in range(num_runs):
            pos_rle = _uleb128_decode_limbs_jit(data, pos_rle, scratch1)
            ln = int(scratch1[0])
            for _j in range(ln):
                is_one[fill_pos] = val
                fill_pos += 1
            val = not val

    pos = rle_start + rb_len
    n_decoded = rle_total

    # --- Extras ---
    pos = _uleb128_decode_limbs_jit(data, pos, scratch1)
    extras_len = int(scratch1[0])

    extras = np.empty(extras_len, dtype=np.int64)
    for i in range(extras_len):
        pos = _uleb128_decode_limbs_jit(data, pos, scratch1)
        extras[i] = np.int64(scratch1[0])

    # --- Build counts ---
    counts = np.empty(n_decoded, dtype=np.int64)
    extra_idx = 0
    for i in range(n_decoded):
        if is_one[i]:
            counts[i] = 1
        else:
            counts[i] = extras[extra_idx] + 2
            extra_idx += 1

    # --- Validate that every index fits in n_bits ---
    # Bits above n_bits in the top limb must all be zero.
    top_bits_used = n_bits - (L - 1) * 64  # in [1, 64]
    if top_bits_used < 64:
        top_mask = (np.uint64(1) << np.uint64(top_bits_used)) - np.uint64(1)
        for i in range(n_decoded):
            if (indices[i, L - 1] & ~top_mask) != np.uint64(0):
                raise ValueError("decoded index exceeds n_bits")

    return indices, counts, n_bits, L, unique, total_shots, n_decoded
