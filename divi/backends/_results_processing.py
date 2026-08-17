# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import base64
from collections.abc import Mapping

import numpy as np

from ._numba_kernels import _decompress_histogram_jit

#: Widest histogram this decoder accepts. QH2's ULEB128 width is unbounded and
#: every declared bit costs limbs in every decoded row.
_MAX_HISTOGRAM_WIDTH = 1 << 20

#: Maximum number of unpacked bit cells rendered per pass.
_RENDER_BLOCK_CELLS = 8 * 1024 * 1024


def _decode_histogram_b64(encoded: dict) -> dict[str, int]:
    """
    Decode a {'encoding':'qh1'|'qh2','n_bits':N,'payload':base64} histogram
    into a dict with bitstring keys -> int counts.

    ``qh2`` is the same container with a ULEB128 width, which the cloud emits
    for circuits wider than 255 qubits.

    If `encoded` is None, returns None.
    If `encoded` is an empty dict or has a missing/empty payload, returns `encoded` unchanged.
    Otherwise, decodes the payload and returns a dict mapping bitstrings to counts.
    """
    if not encoded or not encoded.get("payload"):
        return encoded

    encoding = encoded.get("encoding")
    if encoding not in ("qh1", "qh2"):
        raise ValueError(f"Unsupported encoding: {encoding}")

    blob = base64.b64decode(encoded["payload"])
    expected_magic = b"QH1" if encoding == "qh1" else b"QH2"
    return _decompress_histogram(
        blob,
        expected_n_bits=encoded.get("n_bits"),
        expected_magic=expected_magic,
    )


def _width_limit_error(width: int) -> ValueError:
    return ValueError(
        f"declared width {width} exceeds the maximum supported "
        f"{_MAX_HISTOGRAM_WIDTH} bits"
    )


def _parse_histogram_header(buf: bytes) -> tuple[bytes, int, int]:
    """Return ``(magic, n_bits, body_offset)`` after bounded header parsing."""
    magic = buf[:3]
    if magic not in (b"QH1", b"QH2"):
        raise ValueError("bad magic")

    if magic == b"QH1":
        if len(buf) < 4:
            raise ValueError("truncated QH1 width")
        return magic, buf[3], 4

    width = 0
    shift = 0
    pos = 3
    while True:
        if pos >= len(buf):
            raise ValueError("truncated QH2 width")
        byte = buf[pos]
        pos += 1
        width |= (byte & 0x7F) << shift
        if width > _MAX_HISTOGRAM_WIDTH:
            raise _width_limit_error(width)
        if byte < 0x80:
            return magic, width, pos
        shift += 7
        if shift > _MAX_HISTOGRAM_WIDTH.bit_length():
            raise _width_limit_error(width)


def _decompress_histogram(
    buf: bytes,
    expected_n_bits: int | None = None,
    expected_magic: bytes | None = None,
) -> dict[str, int]:
    if not buf:
        return {}

    magic, declared, body_pos = _parse_histogram_header(buf)
    if expected_magic is not None and magic != expected_magic:
        raise ValueError(
            f"payload magic {magic.decode('ascii')} does not match "
            f"envelope encoding {expected_magic.decode('ascii').lower()}"
        )
    data = np.frombuffer(buf, dtype=np.uint8)
    if expected_n_bits is not None and declared != expected_n_bits:
        raise ValueError(
            f"payload declares {declared} bits but the envelope declares "
            f"{expected_n_bits}"
        )

    indices, counts, n_bits, L, unique, total_shots, n_decoded = (
        _decompress_histogram_jit(data, declared, body_pos)
    )

    # Integrity checks (order matches original: shot sum first, unique second)
    if int(counts.sum()) != total_shots:
        raise ValueError("corrupt stream: shot sum mismatch")
    if n_decoded != unique:
        raise ValueError("corrupt stream: unique mismatch")

    bitstrings = _limbs_to_bitstrings(indices, n_bits, L)
    return dict(zip(bitstrings, counts.tolist()))


def _limbs_to_bitstrings(limbs: np.ndarray, n_bits: int, L: int) -> list[str]:
    """Render little-endian uint64 limb rows as ``n_bits``-wide binary strings.

    Args:
        limbs: ``(n_rows, L)`` uint64 array, limb ``k`` holding bits ``[64k, 64k+64)``.
        n_bits: Width of each rendered string.
        L: Number of limbs per row.
    """
    n_rows = limbs.shape[0]
    if n_bits < 0 or n_bits > L * 64:
        raise ValueError(f"width {n_bits} is not renderable from {L} limb(s)")
    if n_bits == 0:
        # Python's ``s[-0:] == s`` quirk has a slicing analogue here: a
        # ``[-0:]`` bit slice would keep the whole limb width.
        return [""] * n_rows

    out: list[str] = []
    rows_per_block = max(1, _RENDER_BLOCK_CELLS // (L * 64))
    for start in range(0, n_rows, rows_per_block):
        block = limbs[start : start + rows_per_block]
        # Most-significant limb first, big-endian bytes, then MSB-first bits.
        big_endian = np.ascontiguousarray(block[:, ::-1]).astype(">u8")
        bit_matrix = np.unpackbits(
            big_endian.view(np.uint8).reshape(block.shape[0], L * 8), axis=1
        )
        chars = bit_matrix[:, L * 64 - n_bits :] + np.uint8(ord("0"))
        flat = chars.tobytes().decode("ascii")
        out.extend(flat[i * n_bits : (i + 1) * n_bits] for i in range(block.shape[0]))
    return out


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
