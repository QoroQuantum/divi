# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Shared representation for per-circuit shot allocation.

Backends accept a compressed ``[start, end, shots]`` wire format for shot
allocations. This module centralises the views different code paths want —
per-circuit expansion, bucketing by shot count, chunk restriction for
paginated submission, and wire-format (de)serialisation — so every backend
can ask for the shape it needs without reimplementing iteration and
validation each time.
"""

from collections.abc import Sequence
from numbers import Integral
from typing import Any, NamedTuple


class ShotRange(NamedTuple):
    """A half-open ``[start, end)`` slice of circuits with a shared shot count."""

    start: int
    end: int
    shots: int

    def shift(self, delta: int) -> "ShotRange":
        """Return a copy with ``start`` and ``end`` offset by ``delta``."""
        return ShotRange(self.start + delta, self.end + delta, self.shots)


def validate(ranges: Sequence[ShotRange], n_circuits: int) -> None:
    """Assert that ranges tile ``[0, n_circuits)`` exactly with positive shots.

    Raises :class:`ValueError` on partial coverage, overlap, empty/reversed
    ranges, or non-positive shot counts.
    """
    covered = [False] * n_circuits
    for r in ranges:
        if r.shots <= 0:
            raise ValueError(f"ShotRange {r} has non-positive shot count.")
        if r.start < 0 or r.end > n_circuits or r.start >= r.end:
            raise ValueError(
                f"ShotRange {r} is out of bounds for n_circuits={n_circuits}."
            )
        for i in range(r.start, r.end):
            if covered[i]:
                raise ValueError(
                    f"ShotRange {r} overlaps an earlier range at index {i}."
                )
            covered[i] = True
    missing = [i for i, c in enumerate(covered) if not c]
    if missing:
        raise ValueError(
            f"Shot ranges do not cover every circuit; missing indices {missing}."
        )


def per_circuit(ranges: Sequence[ShotRange], n_circuits: int) -> list[int]:
    """Expand ranges into one shot count per circuit.

    Assumes ranges have already been validated; missing circuits yield ``0``.
    """
    out = [0] * n_circuits
    for r in ranges:
        for i in range(r.start, r.end):
            out[i] = r.shots
    return out


def per_circuit_or_none(
    shot_groups: Sequence[list[int]] | None, n_circuits: int
) -> list[int] | None:
    """Validate a wire-format allocation and expand it to one count per circuit.

    ``None`` passes through, leaving the caller on its own shot count.
    """
    if shot_groups is None:
        return None
    ranges = from_wire(shot_groups)
    validate(ranges, n_circuits)
    return per_circuit(ranges, n_circuits)


def bucket_by_shots(ranges: Sequence[ShotRange]) -> dict[int, list[int]]:
    """Return ``{shots: indices}`` — circuits sharing a shot count.

    Useful for batched backends whose ``run(circuits, shots=N)`` applies a
    single ``N`` to every circuit in the call (e.g. Qiskit Aer).
    """
    out: dict[int, list[int]] = {}
    for r in ranges:
        out.setdefault(r.shots, []).extend(range(r.start, r.end))
    return out


def restrict_to_chunk(
    ranges: Sequence[ShotRange], offset: int, size: int
) -> list[ShotRange]:
    """Re-index ranges to chunk-local coordinates.

    Each range is intersected with ``[offset, offset + size)`` and rebased
    to the chunk start. Ranges outside the chunk are dropped.
    """
    end = offset + size
    out: list[ShotRange] = []
    for r in ranges:
        inter_start = max(r.start, offset)
        inter_end = min(r.end, end)
        if inter_start < inter_end:
            out.append(ShotRange(inter_start - offset, inter_end - offset, r.shots))
    return out


def to_wire(ranges: Sequence[ShotRange]) -> list[list[int]]:
    """Serialise to the ``list[list[int]]`` format accepted by the cloud API."""
    return [[r.start, r.end, r.shots] for r in ranges]


def _plain_int(value: Any) -> Any:
    """Integral values, numpy's included, as ``int``; anything else unchanged."""
    if isinstance(value, Integral) and not isinstance(value, bool):
        return int(value)
    return value


def from_wire(wire: Sequence[list[int]] | Sequence[ShotRange]) -> list[ShotRange]:
    """Normalise a mix of raw triples and ShotRange values into ShotRanges.

    Numpy integers become ``int``, so the result serialises to JSON.
    """
    return [ShotRange(*map(_plain_int, r)) for r in wire]
