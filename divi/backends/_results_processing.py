# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import base64
from collections.abc import Mapping


def _decode_qh1_b64(encoded: dict) -> dict[str, int]:
    """
    Decode a {'encoding':'qh1','n_bits':N,'payload':base64} histogram
    into a dict with bitstring keys -> int counts.

    If `encoded` is None, returns None.
    If `encoded` is an empty dict or has a missing/empty payload, returns `encoded` unchanged.
    Otherwise, decodes the payload and returns a dict mapping bitstrings to counts.
    """
    if not encoded or not encoded.get("payload"):
        return encoded

    if encoded.get("encoding") != "qh1":
        raise ValueError(f"Unsupported encoding: {encoded.get('encoding')}")

    blob = base64.b64decode(encoded["payload"])
    hist_int = _decompress_histogram(blob)
    return {str(k): v for k, v in hist_int.items()}


def _uleb128_decode(data: bytes, pos: int = 0) -> tuple[int, int]:
    x = 0
    shift = 0
    while True:
        if pos >= len(data):
            raise ValueError("truncated varint")
        b = data[pos]
        pos += 1
        x |= (b & 0x7F) << shift
        if (b & 0x80) == 0:
            break
        shift += 7
    return x, pos


def _int_to_bitstr(x: int, n_bits: int) -> str:
    return format(x, f"0{n_bits}b")


def _rle_bool_decode(data: bytes, pos=0) -> tuple[list[bool], int]:
    num_runs, pos = _uleb128_decode(data, pos)
    if num_runs == 0:
        return [], pos
    first_val = data[pos] != 0
    pos += 1
    total, val = [], first_val
    for _ in range(num_runs):
        ln, pos = _uleb128_decode(data, pos)
        total.extend([val] * ln)
        val = not val
    return total, pos


def _decompress_histogram(buf: bytes) -> dict[str, int]:
    if not buf:
        return {}
    pos = 0
    if buf[pos : pos + 3] != b"QH1":
        raise ValueError("bad magic")
    pos += 3
    n_bits = buf[pos]
    pos += 1
    unique, pos = _uleb128_decode(buf, pos)
    total_shots, pos = _uleb128_decode(buf, pos)

    num_gaps, pos = _uleb128_decode(buf, pos)
    gaps = []
    for _ in range(num_gaps):
        g, pos = _uleb128_decode(buf, pos)
        gaps.append(g)

    idxs, acc = [], 0
    for i, g in enumerate(gaps):
        acc = g if i == 0 else acc + g
        idxs.append(acc)

    rb_len, pos = _uleb128_decode(buf, pos)
    is_one, _ = _rle_bool_decode(buf[pos : pos + rb_len], 0)
    pos += rb_len

    extras_len, pos = _uleb128_decode(buf, pos)
    extras = []
    for _ in range(extras_len):
        e, pos = _uleb128_decode(buf, pos)
        extras.append(e)

    counts, it = [], iter(extras)
    for flag in is_one:
        counts.append(1 if flag else next(it) + 2)

    hist = {_int_to_bitstr(i, n_bits): c for i, c in zip(idxs, counts)}

    # optional integrity check
    if sum(counts) != total_shots:
        raise ValueError("corrupt stream: shot sum mismatch")
    if len(counts) != unique:
        raise ValueError("corrupt stream: unique mismatch")
    return hist


def reverse_dict_endianness(
    probs_dict: Mapping[str, Mapping[str, float]],
) -> dict[str, dict[str, float]]:
    """Reverse endianness of all bitstrings in a dictionary of probability distributions."""
    return {
        tag: {bitstring[::-1]: prob for bitstring, prob in probs.items()}
        for tag, probs in probs_dict.items()
    }


def convert_counts_to_probs(
    counts: Mapping[str, Mapping[str, int]], shots: int
) -> dict[str, dict[str, float]]:
    """Convert raw counts to probability distributions.

    Args:
        counts (dict[str, dict[str, int]]): The counts to convert to probabilities.
        shots (int): The number of shots.

    Returns:
        dict[str, dict[str, float]]: The probability distributions.
    """
    return {
        tag: {bitstring: count / shots for bitstring, count in probs.items()}
        for tag, probs in counts.items()
    }
