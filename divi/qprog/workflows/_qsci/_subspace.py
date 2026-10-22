# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""A growing set of computational-basis configurations of a qubit Hamiltonian.

Configurations are bitstrings whose character ``q`` is the state of qubit ``q``.
Internally each is a row of ``uint64`` limbs, so the register width is not
bounded by a machine word. The Hamiltonian acts on them directly through its
Pauli terms: a term with flip mask ``x``, phase mask ``z`` and ``n_y`` Y factors
maps ``|ket>`` to ``i**n_y * (-1)**popcount(z & ket) |ket ^ x>``, so no
statevector is ever built.
"""

from collections.abc import Iterable, Sequence

import numpy as np
from qiskit.quantum_info import SparsePauliOp
from scipy.sparse import block_array, csc_array
from scipy.sparse.linalg import eigsh

from divi.hamiltonians._term_ops import _require_qiskit_num_qubits

#: Subspace dimension up to which the CI matrix is diagonalised densely.
_DENSE_SOLVE_THRESHOLD = 1024

#: Entries of the largest intermediate array any one pass may allocate.
_MAX_CHUNK_ENTRIES = 20_000_000

#: ``i**n`` for ``n`` taken modulo 4.
_I_POWERS = np.array([1, 1j, -1, -1j])


def _n_limbs(n_qubits: int) -> int:
    return -(-n_qubits // 64)


def pack_bits(bits: np.ndarray) -> np.ndarray:
    """``(n, q)`` boolean matrix to ``(n, limbs)`` ``uint64``, column ``c`` in
    bit ``c % 64`` of limb ``c // 64``."""
    n_rows, n_columns = bits.shape
    packed = np.packbits(bits, axis=1, bitorder="little")
    padded = np.zeros((n_rows, 8 * _n_limbs(n_columns)), dtype=np.uint8)
    padded[:, : packed.shape[1]] = packed
    return padded.view("<u8").astype(np.uint64)


def unpack_bits(limbs: np.ndarray, n_qubits: int) -> np.ndarray:
    """Inverse of :func:`pack_bits`."""
    raw = np.ascontiguousarray(limbs.astype("<u8")).view(np.uint8)
    return np.unpackbits(raw, axis=1, bitorder="little")[:, :n_qubits]


def configs_to_limbs(configs: Sequence[str], n_qubits: int) -> np.ndarray:
    """Bitstrings to limb rows.

    Raises:
        ValueError: If a configuration is not an ``n_qubits``-character 0/1
            string.
    """
    if any(len(config) != n_qubits for config in configs):
        raise ValueError(f"Every configuration must be {n_qubits} characters wide.")
    raw = np.frombuffer("".join(configs).encode("ascii"), dtype=np.uint8)
    if np.any((raw != ord("0")) & (raw != ord("1"))):
        raise ValueError("Configurations must contain only '0' and '1'.")
    return pack_bits((raw == ord("1")).reshape(len(configs), n_qubits))


def limbs_to_configs(limbs: np.ndarray, n_qubits: int) -> list[str]:
    """Limb rows to bitstrings."""
    chars = (unpack_bits(limbs, n_qubits) + ord("0")).astype(np.uint8)
    return [row.tobytes().decode("ascii") for row in chars]


def _keys(limbs: np.ndarray) -> np.ndarray:
    """One sortable, comparable key per limb row."""
    return (
        np.ascontiguousarray(limbs)
        .view(np.dtype((np.void, 8 * limbs.shape[1])))
        .ravel()
    )


def _parity(masks: np.ndarray) -> np.ndarray:
    """Parity of the set bits across the last (limb) axis."""
    return np.bitwise_count(masks).sum(axis=-1, dtype=np.int64) & 1


def _reduce_by_determinant(
    dets: np.ndarray, amplitudes: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Sum ``amplitudes`` over equal rows of ``dets``."""
    if dets.shape[0] == 0:
        return dets, amplitudes
    keys = _keys(dets)
    order = np.argsort(keys, kind="stable")
    keys, dets, amplitudes = keys[order], dets[order], amplitudes[order]
    boundary = np.ones(keys.size, dtype=bool)
    boundary[1:] = keys[1:] != keys[:-1]
    starts = np.flatnonzero(boundary)
    return dets[starts], np.add.reduceat(amplitudes, starts)


class ConfigurationSubspace:
    """A set of configurations with the Hamiltonian projected onto it.

    Args:
        hamiltonian: Hermitian qubit Hamiltonian, of any width.

    Raises:
        ValueError: If the Hamiltonian is empty or not Hermitian.
    """

    def __init__(self, hamiltonian: SparsePauliOp):
        hamiltonian = hamiltonian.simplify(atol=0)
        if hamiltonian.size == 0:
            raise ValueError("Hamiltonian has no non-zero terms.")
        if np.abs(hamiltonian.coeffs.imag).max() > 1e-12:
            raise ValueError(
                "Hamiltonian is not Hermitian: it has complex coefficients."
            )

        self.n_qubits = _require_qiskit_num_qubits(hamiltonian.num_qubits)
        self._n_limbs = _n_limbs(self.n_qubits)

        x_masks = pack_bits(hamiltonian.paulis.x)
        self._z_masks = pack_bits(hamiltonian.paulis.z)
        n_y = np.bitwise_count(x_masks & self._z_masks).sum(axis=1, dtype=np.int64)
        factors = hamiltonian.coeffs * _I_POWERS[n_y % 4]
        # An even Y count in every term keeps the CI matrix real symmetric.
        self._factors = (
            factors.real.copy() if np.abs(factors.imag).max() <= 1e-12 else factors
        )

        keys = _keys(x_masks)
        self._term_order = np.argsort(keys, kind="stable")
        sorted_keys = keys[self._term_order]
        boundary = np.ones(sorted_keys.size, dtype=bool)
        boundary[1:] = sorted_keys[1:] != sorted_keys[:-1]
        self._group_starts = np.flatnonzero(boundary)
        self._group_counts = np.diff(np.append(self._group_starts, sorted_keys.size))
        self._flips = x_masks[self._term_order][self._group_starts]
        self._flip_keys = sorted_keys[self._group_starts]

        flip_free = ~x_masks.any(axis=1)
        self._diagonal_z = self._z_masks[flip_free]
        self._diagonal_factors = self._factors[flip_free].real

        self.clear()

    def clear(self) -> None:
        """Forget every configuration and the solution built on them."""
        self.configs: list[str] = []
        self._index: dict[str, int] = {}
        self._dets = np.empty((0, self._n_limbs), dtype=np.uint64)
        self.hamiltonian_matrix: csc_array | None = None
        self._matrix_size = 0
        self.eigenvector: np.ndarray | None = None
        self._energy: float | None = None

    @property
    def energy(self) -> float:
        """Lowest eigenvalue of the projected Hamiltonian.

        Raises:
            ValueError: If the eigensystem has not been solved.
        """
        if self._energy is None:
            raise ValueError("Solve the eigensystem first.")
        return self._energy

    def add_configurations(self, configs: Iterable[str]) -> int:
        """Append the configurations not already present, returning how many."""
        fresh = [c for c in dict.fromkeys(configs) if c not in self._index]
        if not fresh:
            return 0
        limbs = configs_to_limbs(fresh, self.n_qubits)
        for config in fresh:
            self._index[config] = len(self.configs)
            self.configs.append(config)
        self._dets = np.vstack([self._dets, limbs])
        return len(fresh)

    def prune(self, threshold: float) -> int:
        """Drop configurations whose eigenvector amplitude is at most
        ``threshold``, returning how many went; never empties the subspace.

        Raises:
            ValueError: If the eigenvector does not cover the configurations.
        """
        if self.eigenvector is None or self.eigenvector.size != len(self.configs):
            raise ValueError("Solve the eigensystem over every configuration first.")
        keep = np.abs(self.eigenvector) > threshold
        if keep.all() or not keep.any():
            return 0
        kept = np.flatnonzero(keep)
        self.configs = [self.configs[i] for i in kept]
        self._index = {config: i for i, config in enumerate(self.configs)}
        self._dets = self._dets[kept]
        self.eigenvector = self.eigenvector[kept]
        matrix = self.hamiltonian_matrix
        if matrix is not None:
            entries = matrix.tocoo()
            surviving = keep[entries.row] & keep[entries.col]
            position = np.cumsum(keep) - 1
            self.hamiltonian_matrix = csc_array(
                (
                    entries.data[surviving],
                    (
                        position[entries.row[surviving]],
                        position[entries.col[surviving]],
                    ),
                ),
                shape=(kept.size, kept.size),
            )
            self._matrix_size = kept.size
        return int((~keep).sum())

    def _lookup(self, flips: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Positions in ``flips`` that some Pauli term flips by, and the term
        group each belongs to."""
        keys = _keys(flips)
        position = np.clip(
            np.searchsorted(self._flip_keys, keys), 0, self._flip_keys.size - 1
        )
        hit = np.flatnonzero(self._flip_keys[position] == keys)
        return hit, position[hit]

    def _group_terms(self, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Term indices of the given flip groups, and each group's offset into
        them."""
        counts = self._group_counts[groups]
        offsets = np.cumsum(counts) - counts
        within = np.arange(counts.sum()) - np.repeat(offsets, counts)
        return (
            self._term_order[np.repeat(self._group_starts[groups], counts) + within],
            offsets,
        )

    def _column_entries(
        self, columns: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(rows, cols, data)`` of the CI matrix columns indexed by ``columns``.

        ``rows`` index every configuration, ``cols`` are positions within
        ``columns``. The flip mask connecting a pair is ``ket ^ bra``, so one
        lookup resolves every pair at once and the cost follows the connected
        pairs rather than the term count.
        """
        n_configs = len(self.configs)
        rows, cols, data = [], [], []
        block_size = max(1, _MAX_CHUNK_ENTRIES // (n_configs * self._n_limbs))
        for start in range(0, columns.size, block_size):
            block = self._dets[columns[start : start + block_size]]
            needed = (block[:, None, :] ^ self._dets[None, :, :]).reshape(
                -1, self._n_limbs
            )
            hit, groups = self._lookup(needed)
            if hit.size == 0:
                continue
            terms, offsets = self._group_terms(groups)
            column_of = hit // n_configs
            kets = np.repeat(block[column_of], self._group_counts[groups], axis=0)
            sign = 1 - 2 * _parity(self._z_masks[terms] & kets)
            values = np.add.reduceat(self._factors[terms] * sign, offsets)

            nonzero = values != 0
            rows.append((hit % n_configs)[nonzero])
            cols.append(column_of[nonzero] + start)
            data.append(values[nonzero])

        if not rows:
            return (
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=self._factors.dtype),
            )
        return np.concatenate(rows), np.concatenate(cols), np.concatenate(data)

    def extend_hamiltonian_matrix(self) -> None:
        """Bring the CI matrix up to the current configurations.

        Only the columns of configurations added since the last call are
        computed; the matching rows follow by hermiticity.
        """
        n_configs = len(self.configs)
        matrix = self.hamiltonian_matrix
        n_old = self._matrix_size
        n_new = n_configs - n_old
        if n_new == 0:
            return

        rows, cols, data = self._column_entries(np.arange(n_old, n_configs))
        old = rows < n_old
        new_rows, new_cols, new_data = rows[~old] - n_old, cols[~old], data[~old]
        new_new = csc_array(
            (
                0.5 * np.concatenate([new_data, new_data.conj()]),
                (
                    np.concatenate([new_rows, new_cols]),
                    np.concatenate([new_cols, new_rows]),
                ),
            ),
            shape=(n_new, n_new),
            dtype=self._factors.dtype,
        )
        self._matrix_size = n_configs
        if matrix is None:
            self.hamiltonian_matrix = new_new
            return
        coupling = csc_array(
            (data[old], (rows[old], cols[old])),
            shape=(n_old, n_new),
            dtype=self._factors.dtype,
        )
        coupling_adjoint = csc_array(
            (data[old].conj(), (cols[old], rows[old])),
            shape=(n_new, n_old),
            dtype=self._factors.dtype,
        )
        self.hamiltonian_matrix = csc_array(
            block_array([[matrix, coupling], [coupling_adjoint, new_new]])
        )

    def solve_eigensystem(self, warm_start: bool = False) -> None:
        """Lowest eigenpair of the CI matrix.

        ``warm_start`` seeds the sparse solver with the previous eigenvector,
        zero-padded to the current dimension.

        Raises:
            ValueError: If the matrix has not been built.
        """
        matrix = self.hamiltonian_matrix
        if matrix is None:
            raise ValueError("Build the Hamiltonian matrix before solving.")

        n = self._matrix_size
        if n <= _DENSE_SOLVE_THRESHOLD:
            values, vectors = np.linalg.eigh(matrix.toarray())
        else:
            v0 = None
            if warm_start and self.eigenvector is not None:
                v0 = np.zeros(n, dtype=self._factors.dtype)
                v0[: self.eigenvector.size] = self.eigenvector
                if not v0.any():
                    v0 = None
            values, vectors = eigsh(matrix, k=1, which="SA", v0=v0)

        vector = np.ascontiguousarray(vectors[:, 0])
        if np.iscomplexobj(vector):
            pivot = vector[np.argmax(np.abs(vector))]
            vector = vector * (np.conj(pivot) / abs(pivot))
        self._energy = float(values[0])
        self.eigenvector = vector

    def diagonal_of(self, dets: np.ndarray) -> np.ndarray:
        """``<b|H|b>`` for each limb row of ``dets``, from the flip-free terms."""
        out = np.empty(dets.shape[0])
        n_terms = max(1, self._diagonal_z.shape[0])
        step = max(1, _MAX_CHUNK_ENTRIES // (n_terms * self._n_limbs))
        for start in range(0, dets.shape[0], step):
            parity = _parity(
                dets[start : start + step, None, :] & self._diagonal_z[None]
            )
            out[start : start + step] = ((1 - 2 * parity) * self._diagonal_factors).sum(
                axis=1
            )
        return out

    def external_contributions(
        self, amplitude_cutoff: float = 1e-12
    ) -> tuple[np.ndarray, np.ndarray]:
        """Epstein-Nesbet second-order contribution of each determinant outside
        the subspace that the Hamiltonian connects to it.

        For such a determinant ``b`` the contribution is
        ``|<b|H|psi>|**2 / (E - <b|H|b>)``; their sum is the PT2 correction.

        ``<b|H|psi>`` is accumulated block by block over the flip groups, so the
        working set stays within ``_MAX_CHUNK_ENTRIES``. Terms sharing a flip
        mask are collapsed first, and a determinant whose partial amplitude is
        at most ``amplitude_cutoff`` is dropped from its block, which
        discards ``~amplitude_cutoff**2`` Hartree per determinant. Pass ``0``
        for the exact sum.

        Returns:
            ``(dets, contributions)``: limb rows and their contributions, leaving
            out determinants within ``1e-10`` of the subspace energy.

        Raises:
            ValueError: If the eigensystem has not been solved.
        """
        if self.eigenvector is None:
            raise ValueError("Solve the eigensystem before computing contributions.")

        n_configs = len(self.configs)
        sorted_keys = np.sort(_keys(self._dets))
        cumulative = np.cumsum(self._group_counts)
        term_budget = max(1, _MAX_CHUNK_ENTRIES // (n_configs * self._n_limbs))

        merged = (
            np.empty((0, self._n_limbs), dtype=np.uint64),
            np.empty(0, dtype=self._factors.dtype),
        )
        pending: list[tuple[np.ndarray, np.ndarray]] = []
        n_pending = 0

        first_group = 0
        while first_group < self._flips.shape[0]:
            consumed = int(cumulative[first_group - 1]) if first_group else 0
            stop_group = max(
                first_group + 1,
                int(np.searchsorted(cumulative, consumed + term_budget, side="right")),
            )
            first_term = int(self._group_starts[first_group])
            last_term = int(
                self._group_starts[stop_group - 1] + self._group_counts[stop_group - 1]
            )
            terms = self._term_order[first_term:last_term]

            sign = 1 - 2 * _parity(self._dets[None] & self._z_masks[terms, None])
            collapsed = np.add.reduceat(
                self._factors[terms, None] * sign,
                self._group_starts[first_group:stop_group] - first_term,
                axis=0,
            )
            targets = (
                self._dets[None] ^ self._flips[first_group:stop_group, None]
            ).reshape(-1, self._n_limbs)
            dets, amplitudes = _reduce_by_determinant(
                targets, (collapsed * self.eigenvector[None]).ravel()
            )

            target_keys = _keys(dets)
            position = np.clip(
                np.searchsorted(sorted_keys, target_keys), 0, sorted_keys.size - 1
            )
            keep = sorted_keys[position] != target_keys
            if amplitude_cutoff > 0:
                keep &= np.abs(amplitudes) > amplitude_cutoff
            pending.append((dets[keep], amplitudes[keep]))
            n_pending += int(keep.sum())

            if n_pending > _MAX_CHUNK_ENTRIES:
                merged = _reduce_by_determinant(
                    np.concatenate([merged[0], *(d for d, _ in pending)]),
                    np.concatenate([merged[1], *(a for _, a in pending)]),
                )
                pending, n_pending = [], 0
            first_group = stop_group

        dets, amplitudes = _reduce_by_determinant(
            np.concatenate([merged[0], *(d for d, _ in pending)]),
            np.concatenate([merged[1], *(a for _, a in pending)]),
        )
        if dets.shape[0] == 0:
            return dets, np.empty(0)

        gap = self.energy - self.diagonal_of(dets)
        good = np.abs(gap) > 1e-10
        return dets[good], np.abs(amplitudes[good]) ** 2 / gap[good]
