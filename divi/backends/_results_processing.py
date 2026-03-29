# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import base64
from collections.abc import Mapping

import numpy as np

from divi.backends._numba_kernels import _decompress_histogram_jit


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


def _decompress_histogram(buf: bytes) -> dict[str, int]:
    if not buf:
        return {}
    if buf[:3] != b"QH1":
        raise ValueError("bad magic")

    data = np.frombuffer(buf, dtype=np.uint8)
    indices, counts, n_bits, unique, total_shots, n_decoded = _decompress_histogram_jit(
        data
    )

    # Integrity checks (order matches original: shot sum first, unique second)
    if int(counts.sum()) != total_shots:
        raise ValueError("corrupt stream: shot sum mismatch")
    if n_decoded != unique:
        raise ValueError("corrupt stream: unique mismatch")

    return {
        format(int(indices[i]), f"0{n_bits}b"): int(counts[i]) for i in range(n_decoded)
    }


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
