# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Pauli strings as packed bit arrays: anticommutation, products and
expectation values without building Qiskit operators on the hot path.

A string ``S(x, z)`` is the Hermitian Pauli with ``X``, ``Y`` or ``Z`` on each
qubit set in ``x``, both or ``z``; qubit ``i`` is bit ``i % 64`` of word
``i // 64``. With ``Y = iXZ``, ``S(x, z) = i^{|x & z|} X^x Z^z``.
"""

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

import numpy as np
import numpy.typing as npt
from qiskit.quantum_info import PauliList, SparsePauliOp

if TYPE_CHECKING:
    from divi.qprog.initial_states import InitialState

# Candidate pairs examined at once, which bounds the memory of every pass.
PAIR_BUDGET = 1 << 18
_I_POWERS = np.array([1, 1j, -1, -1j])

Words = npt.NDArray[np.uint64]
_T = TypeVar("_T")


@dataclass(frozen=True)
class PackedPaulis:
    """Weighted Pauli strings, ``Σ_k coeffs[k] · S(x[k], z[k])``."""

    x: Words
    z: Words
    coeffs: npt.NDArray[np.complexfloating]

    @property
    def size(self) -> int:
        return len(self.coeffs)


def _pack_bits(bits: npt.NDArray[np.bool_]) -> Words:
    n_words = max(1, -(-bits.shape[1] // 64))
    packed = np.packbits(bits, axis=1, bitorder="little")
    padded = np.zeros((bits.shape[0], 8 * n_words), dtype=np.uint8)
    padded[:, : packed.shape[1]] = packed
    return padded.view(np.dtype("<u8"))


def _unpack_bits(words: Words, n_qubits: int) -> npt.NDArray[np.bool_]:
    return np.unpackbits(
        np.ascontiguousarray(words).view(np.uint8),
        axis=1,
        count=n_qubits,
        bitorder="little",
    ).astype(bool)


def _popcount(words: Words) -> npt.NDArray[np.int64]:
    return np.bitwise_count(words).sum(axis=-1, dtype=np.int64)


def pack(operator: SparsePauliOp) -> PackedPaulis:
    """``operator`` with each term's phase folded into its coefficient."""
    paulis = operator.paulis
    return PackedPaulis(
        _pack_bits(paulis.x),
        _pack_bits(paulis.z),
        np.asarray(operator.coeffs * (-1j) ** paulis.phase, dtype=complex),
    )


def to_paulis(x: Words, z: Words, n_qubits: int) -> PauliList:
    return PauliList.from_symplectic(
        _unpack_bits(z, n_qubits), _unpack_bits(x, n_qubits)
    )


def anticommuting_pairs(
    left: PackedPaulis, right: PackedPaulis
) -> Iterator[tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]]:
    """Index pairs ``(a, b)`` whose strings anticommute, a block of ``left``
    at a time."""
    block = max(1, PAIR_BUDGET // max(right.size, 1))
    for start in range(0, left.size, block):
        lx = left.x[start : start + block, None]
        lz = left.z[start : start + block, None]
        parity = _popcount((lx & right.z[None]) ^ (lz & right.x[None])) & 1
        a, b = np.nonzero(parity)
        yield a + start, b


def commutators(
    left: PackedPaulis, right: PackedPaulis
) -> Iterator[
    tuple[
        npt.NDArray[np.intp],
        npt.NDArray[np.intp],
        Words,
        Words,
        npt.NDArray[np.complexfloating],
    ]
]:
    """``[S_a, S_b] = factor · S(x, z)`` for every anticommuting pair, a block
    at a time; pairs that commute contribute nothing and are skipped."""
    for a, b in anticommuting_pairs(left, right):
        yield a, b, *_commutator(left.x[a], left.z[a], right.x[b], right.z[b])


def shift_sums(
    left: PackedPaulis, right: PackedPaulis, weights: npt.NDArray
) -> npt.NDArray[np.complexfloating]:
    """``Σ_b weights[b] · (-2 if S_a anticommutes with S_b else 0)`` per
    ``left`` string, so ``<Δ_a X>`` for ``weights[b] = x_b <S_b>``, as
    ``Δ_a S_b = S_a S_b S_a - S_b`` is ``-2 S_b`` or ``0``. Zero weights are
    skipped."""
    kept = np.flatnonzero(weights)
    nonzero = PackedPaulis(right.x[kept], right.z[kept], weights[kept])
    out = np.zeros(left.size, dtype=complex)
    for a, b in anticommuting_pairs(left, nonzero):
        out += accumulate(a, -2 * nonzero.coeffs[b], left.size)
    return out


def _commutator(
    lx: Words, lz: Words, rx: Words, rz: Words
) -> tuple[Words, Words, npt.NDArray[np.complexfloating]]:
    """``[S(lx, lz), S(rx, rz)] = factor · S(x, z)`` for anticommuting strings."""
    x, z = lx ^ rx, lz ^ rz
    exponent = (
        _popcount(lx & lz)
        + _popcount(rx & rz)
        + 2 * _popcount(lz & rx)
        - _popcount(x & z)
    ) % 4
    return x, z, 2 * _I_POWERS[exponent]


def combine(
    x: Words, z: Words, coeffs: npt.NDArray, rtol: float = 1e-12
) -> PackedPaulis:
    """Merge repeated strings, dropping terms below ``rtol`` of the largest."""
    if len(coeffs) == 0:
        return PackedPaulis(x, z, np.asarray(coeffs, dtype=complex))
    keys = _row_keys(x, z)
    _, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
    summed = accumulate(inverse, coeffs, len(first))
    magnitude = np.abs(summed)
    keep = magnitude > rtol * magnitude.max()
    return PackedPaulis(x[first][keep], z[first][keep], summed[keep])


def concatenate(parts: list[PackedPaulis], n_words: int) -> PackedPaulis:
    if not parts:
        empty = np.zeros((0, n_words), dtype=np.uint64)
        return PackedPaulis(empty, empty.copy(), np.zeros(0, dtype=complex))
    return PackedPaulis(
        np.concatenate([p.x for p in parts]),
        np.concatenate([p.z for p in parts]),
        np.concatenate([p.coeffs for p in parts]),
    )


def accumulate(
    index: npt.NDArray[np.intp], values: npt.NDArray, size: int
) -> npt.NDArray[np.complexfloating]:
    """``out[i] = Σ values[index == i]`` for complex ``values``."""
    values = np.asarray(values, dtype=complex)
    return np.bincount(index, values.real, size) + 1j * np.bincount(
        index, values.imag, size
    )


@dataclass(frozen=True)
class _ProductMasks:
    """A product of ``Z`` and ``X`` eigenstates, packed like the strings."""

    along_z: Words
    along_x: Words
    flipped: Words

    @classmethod
    def of(cls, initial_state: "InitialState", n_qubits: int) -> "_ProductMasks | None":
        masks = initial_state.product_masks(n_qubits)
        if masks is None:
            return None
        along_z, flipped = masks
        return cls(
            _pack_bits(along_z[None]),
            _pack_bits(~along_z[None]),
            _pack_bits(flipped[None]),
        )

    def keys(self, x: Words, z: Words) -> Words:
        """Equal keys exactly when the product of two strings has a non-zero
        value: no ``X`` or ``Y`` on a ``Z`` eigenstate and no ``Z`` or ``Y`` on
        an ``X`` eigenstate. The two masks are disjoint, so one word per 64
        qubits holds both parts."""
        return (x & self.along_z) | (z & self.along_x)

    def values(self, x: Words, z: Words) -> npt.NDArray[np.floating]:
        vanishes = _popcount(x & self.along_z) + _popcount(z & self.along_x) > 0
        return self.signs(x, z) * ~vanishes

    def signs(self, x: Words, z: Words) -> npt.NDArray[np.floating]:
        return 1.0 - 2.0 * (_popcount((x | z) & self.flipped) & 1)


class _DistinctRows:
    """The distinct rows of everything added, merged whenever the rows held
    since the last merge exceed a bound, so memory follows the distinct count."""

    def __init__(self, width: int):
        self._parts = [np.zeros((0, width), dtype=np.uint64)]
        self._held = 0

    def add(self, rows: Words) -> None:
        self._parts.append(_distinct(rows))
        self._held += len(self._parts[-1])
        if self._held > 8 * PAIR_BUDGET:
            self._parts = [self.rows()]
            self._held = 0

    def rows(self) -> Words:
        return _distinct(np.concatenate(self._parts))


@dataclass(frozen=True)
class _RowIndex:
    """The position of each of a set of distinct rows, found by hash."""

    rows: Words
    order: npt.NDArray[np.intp]
    hashes: npt.NDArray[np.uint64]

    @classmethod
    def of(cls, rows: Words) -> "_RowIndex":
        hashes = _hash(rows)
        order = np.argsort(hashes, kind="stable")
        return cls(rows, order, hashes[order])

    def locate(self, rows: Words) -> npt.NDArray[np.intp]:
        hashes = _hash(rows)
        candidate = np.searchsorted(self.hashes, hashes, side="left")
        end = np.searchsorted(self.hashes, hashes, side="right")
        found = np.full(len(rows), -1, dtype=np.intp)
        pending = np.flatnonzero(candidate < end)
        while pending.size:
            positions = self.order[candidate[pending]]
            hit = np.all(self.rows[positions] == rows[pending], axis=1)
            found[pending[hit]] = positions[hit]
            pending = pending[~hit]
            candidate[pending] += 1
            pending = pending[candidate[pending] < end[pending]]
        if np.any(found < 0):
            raise KeyError("A Pauli string is missing from the index.")
        return found


@dataclass(frozen=True)
class ExpectationTable:
    """``<S>`` in a state for a fixed set of strings, read off by their bits."""

    index: _RowIndex
    known: npt.NDArray[np.floating]

    @classmethod
    def of(
        cls, rows: Words, initial_state: "InitialState", n_qubits: int
    ) -> "ExpectationTable":
        """Evaluate the distinct strings ``rows`` (``x`` words, then ``z``
        words) in one batch."""
        n_words = rows.shape[1] // 2
        known = initial_state.pauli_expectations(
            to_paulis(rows[:, :n_words], rows[:, n_words:], n_qubits), n_qubits
        )
        return cls(_RowIndex.of(rows), np.asarray(known, dtype=float))

    def values(self, x: Words, z: Words) -> npt.NDArray[np.floating]:
        return self.known[self.index.locate(_rows(x, z))]


class _Recorder:
    """Every string asked for, each read as zero."""

    def __init__(self, n_words: int):
        self.seen = _DistinctRows(2 * n_words)

    def values(self, x: Words, z: Words) -> npt.NDArray[np.floating]:
        self.seen.add(_rows(x, z))
        return np.zeros(len(x))


Values = _ProductMasks | ExpectationTable | _Recorder


def evaluate(
    initial_state: "InitialState", n_qubits: int, compute: Callable[[Values], _T]
) -> _T:
    """``compute`` given the values of ``initial_state``: from the bits for a
    product state, otherwise from one batch of every string ``compute`` reads,
    found by running it once beforehand."""
    masks = _ProductMasks.of(initial_state, n_qubits)
    if masks is not None:
        return compute(masks)
    recorder = _Recorder(max(1, -(-n_qubits // 64)))
    compute(recorder)
    return compute(ExpectationTable.of(recorder.seen.rows(), initial_state, n_qubits))


def commutator_values(
    left: PackedPaulis, right: PackedPaulis, source: Values
) -> Iterator[
    tuple[npt.NDArray[np.intp], npt.NDArray[np.intp], npt.NDArray[np.complexfloating]]
]:
    """``<[S_a, S_b]>`` for every pair whose value can be non-zero, a block at
    a time. For a product state only pairs with equal keys can, so ``left`` is
    indexed by key and each ``right`` string meets only its own key group."""
    if not isinstance(source, _ProductMasks):
        for a, b, x, z, factor in commutators(left, right):
            yield a, b, factor * source.values(x, z)
        return
    if left.size == 0 or right.size == 0:
        return
    left_keys = source.keys(left.x, left.z)
    right_keys = source.keys(right.x, right.z)
    for a, b in _equal_rows(left_keys, right_keys):
        lx, lz, rx, rz = left.x[a], left.z[a], right.x[b], right.z[b]
        keep = (_popcount((lx & rz) ^ (lz & rx)) & 1).astype(bool)
        x, z, factor = _commutator(lx[keep], lz[keep], rx[keep], rz[keep])
        yield a[keep], b[keep], factor * source.signs(x, z)


def double_commutator_sums(
    left: PackedPaulis, right: PackedPaulis, source: Values
) -> tuple[npt.NDArray[np.complexfloating], npt.NDArray[np.complexfloating]]:
    """With ``v = <[S_a, [S_a', S_b]]>`` over ``a``, ``a'`` of ``left`` and
    ``b`` of ``right``: ``Σ_{a, b} c_a r_b v`` per ``a'`` and
    ``Σ_{a, a'} c_a c_a' v`` per ``b``.

    For a product state keys combine by XOR, so a term can be non-zero only
    when ``key_a ^ key_a' = key_b``: the ``(a, a')`` pairs are indexed by that
    key a block at a time and each ``right`` string meets only its own key
    group, never forming ``[S_a', S_b]`` for every pair. Otherwise, with
    ``[S_a', S_b] = f M``, ``Σ_a c_a v = f <[L, M]>`` for ``L = Σ_a c_a S_a``,
    so ``<[L, M]>`` is evaluated once per distinct ``M``.
    """
    if isinstance(source, _ProductMasks):
        return _product_double_commutator_sums(left, right, source)
    n_words = left.x.shape[1]
    seen = _DistinctRows(2 * n_words)
    for _, _, x, z, _ in commutators(left, right):
        seen.add(_rows(x, z))
    rows = seen.rows()
    middle = PackedPaulis(
        rows[:, :n_words], rows[:, n_words:], np.ones(len(rows), dtype=complex)
    )
    with_left = np.zeros(middle.size, dtype=complex)
    for a, m, values in commutator_values(left, middle, source):
        with_left += accumulate(m, left.coeffs[a] * values, middle.size)

    index = _RowIndex.of(rows)
    per_inner = np.zeros(left.size, dtype=complex)
    per_right = np.zeros(right.size, dtype=complex)
    for a2, b, x, z, factor in commutators(left, right):
        terms = factor * with_left[index.locate(_rows(x, z))]
        per_inner += accumulate(a2, terms * right.coeffs[b], left.size)
        per_right += accumulate(b, terms * left.coeffs[a2], right.size)
    return per_inner, per_right


def _product_double_commutator_sums(
    left: PackedPaulis, right: PackedPaulis, masks: _ProductMasks
) -> tuple[npt.NDArray[np.complexfloating], npt.NDArray[np.complexfloating]]:
    per_inner = np.zeros(left.size, dtype=complex)
    per_right = np.zeros(right.size, dtype=complex)
    left_keys = masks.keys(left.x, left.z)
    right_keys = masks.keys(right.x, right.z)
    left_hashes = _hash(left_keys)
    right_hashes = _hash(right_keys)
    block = max(1, PAIR_BUDGET // max(left.size, 1))
    for start in range(0, left.size, block):
        inner = np.arange(start, min(start + block, left.size))
        outer_of = np.repeat(np.arange(left.size), len(inner))
        inner_of = np.tile(inner, left.size)
        pair_hashes = left_hashes[outer_of] ^ left_hashes[inner_of]
        for p, b in _equal_hashes(pair_hashes, right_hashes):
            a, a2 = outer_of[p], inner_of[p]
            same = np.all(left_keys[a] ^ left_keys[a2] == right_keys[b], axis=1)
            a, a2, b = a[same], a2[same], b[same]
            mx, mz, rx, rz = left.x[a2], left.z[a2], right.x[b], right.z[b]
            keep = (_popcount((mx & rz) ^ (mz & rx)) & 1).astype(bool)
            a, a2, b = a[keep], a2[keep], b[keep]
            x1, z1, factor1 = _commutator(mx[keep], mz[keep], rx[keep], rz[keep])
            ox, oz = left.x[a], left.z[a]
            keep = (_popcount((ox & z1) ^ (oz & x1)) & 1).astype(bool)
            a, a2, b = a[keep], a2[keep], b[keep]
            x2, z2, factor2 = _commutator(ox[keep], oz[keep], x1[keep], z1[keep])
            terms = left.coeffs[a] * factor1[keep] * factor2 * masks.signs(x2, z2)
            per_inner += accumulate(a2, terms * right.coeffs[b], left.size)
            per_right += accumulate(b, terms * left.coeffs[a2], right.size)
    return per_inner, per_right


def _equal_rows(
    left: Words, right: Words
) -> Iterator[tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]]:
    """Index pairs ``(a, b)`` with ``left[a] == right[b]``, at most about
    ``PAIR_BUDGET`` at a time."""
    for a, b in _equal_hashes(_hash(left), _hash(right)):
        same = np.all(left[a] == right[b], axis=1)
        yield a[same], b[same]


def _equal_hashes(
    left: npt.NDArray[np.uint64], right: npt.NDArray[np.uint64]
) -> Iterator[tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]]:
    """Index pairs ``(a, b)`` with ``left[a] == right[b]``, at most about
    ``PAIR_BUDGET`` at a time: ``left`` sorted once, each ``right`` value
    located by binary search."""
    members = np.argsort(left, kind="stable")
    ordered = left[members]
    lo = np.searchsorted(ordered, right, side="left")
    counts = np.searchsorted(ordered, right, side="right") - lo
    ends = np.cumsum(counts)
    start = 0
    while start < len(right):
        done = ends[start - 1] if start else 0
        stop = max(start + 1, int(np.searchsorted(ends, done + PAIR_BUDGET, "right")))
        n = counts[start:stop]
        b = np.repeat(np.arange(start, stop), n)
        within = np.arange(len(b)) - np.repeat(np.cumsum(n) - n, n)
        yield members[np.repeat(lo[start:stop], n) + within], b
        start = stop


def _hash(words: Words) -> npt.NDArray[np.uint64]:
    """One word per row, linear over GF(2) so that ``hash(k ^ l) = hash(k) ^
    hash(l)``: the words XOR-ed together, word ``w`` rotated by ``17 w`` bits.
    Exact up to 64 qubits; beyond, unequal rows that collide are told apart by
    comparing the rows themselves."""
    out = words[:, 0].copy()
    for w in range(1, words.shape[1]):
        shift = np.uint64(17 * w % 64)
        column = words[:, w]
        out ^= (
            column
            if shift == 0
            else (column << shift) | (column >> (np.uint64(64) - shift))
        )
    return out


def _rows(x: Words, z: Words) -> Words:
    return np.concatenate([x, z], axis=1)


def _void(rows: Words) -> npt.NDArray[np.void]:
    rows = np.ascontiguousarray(rows)
    return rows.view(np.dtype((np.void, rows.dtype.itemsize * rows.shape[1]))).ravel()


def _row_keys(x: Words, z: Words) -> npt.NDArray[np.void]:
    return _void(_rows(x, z))


def _distinct(rows: Words) -> Words:
    _, first = np.unique(_void(rows), return_index=True)
    return rows[first]
