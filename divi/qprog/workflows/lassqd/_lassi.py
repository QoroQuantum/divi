# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""State interaction over products of two fragments' states (LASSI): each
fragment's lowest spin multiplets at its own electron count and at those one or
two electrons away, mixed by the full active-space Hamiltonian."""

import bisect
import itertools
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import scipy.sparse
from pyscf import ao2mo, lib, mcscf
from pyscf.fci import cistring, direct_spin1

from ._integrals import AOIntegrals, _exact_mean_field
from ._sqd import (
    _DENSE_MAX_DIM,
    SQDResult,
    ci_string_to_int,
    compute_spatial_rdm1,
    projected_matrices,
    resolves_spin,
    spatial_to_spin_occupations,
    spin_orbital_integrals,
)
from ._state import ElectronicState, FragmentSpec

#: A fragment's ``(n_alpha, n_beta)``.
_Sector = tuple[int, int]

#: An operator string's ``(dagger, spin)`` factors, in written order.
_Ops = tuple[tuple[bool, int], ...]

_ALPHA, _BETA = 0, 1

#: Product states above which the default multiplet count is lowered.
_MAX_PRODUCT_STATES = 6000

#: Energy gap (Ha) under which fragment multiplets count as degenerate, so a
#: cut keeps or drops the whole set.
_DEGENERACY_TOL = 1e-6

#: Weight of ``S^2`` added to a fragment Hamiltonian, splitting degenerate
#: states of different spin so each eigenvector has one.
_SPIN_SPLIT = 1e-5


@dataclass(frozen=True)
class _Multiplet:
    """One fragment spin multiplet: its energy under the fragment Hamiltonian
    and the dense FCI vector of its spin projection nearest zero, in
    ``sector``."""

    energy: float
    spin: float
    sector: _Sector
    vector: np.ndarray

    @property
    def sectors(self) -> list[_Sector]:
        """The sector of each spin projection, lowest first."""
        count = sum(self.sector)
        twice = round(2 * self.spin)
        return [
            ((count + m) // 2, (count - m) // 2) for m in range(-twice, twice + 1, 2)
        ]


@dataclass(frozen=True)
class _FragmentBasis:
    """A fragment's model states, as dense FCI vectors per electron sector.

    Attributes:
        n_orbitals: The fragment's orbitals.
        states: Per sector, its states as columns of a
            ``(sector determinants, states)`` array, orthonormal.
    """

    n_orbitals: int
    states: Mapping[_Sector, np.ndarray]


def lassi_states(
    ao: AOIntegrals,
    mo_coeff: np.ndarray,
    n_core: int,
    specs: Sequence[FragmentSpec],
    results: Sequence[SQDResult],
    requested: Mapping[int, int],
    n_multiplets: int | None,
    n_hops: int,
) -> tuple[tuple[ElectronicState, ...], int | None, dict[int, tuple[int, int]]]:
    """The requested states of two fragments' state interaction, by
    multiplicity then root; the multiplets kept per electron count when that
    leaves any out (``None`` otherwise); and, per fragment whose determinant
    spaces miss some of its sectors, ``(held, full)`` determinant counts.

    ``mo_coeff``'s fragment blocks must be in the bases the strings were
    sampled in.

    Raises:
        MemoryError: If the products exceed ``mol.max_memory``.
    """
    spec_a, spec_b = specs
    n_a, n_b = spec_a.n_orbitals, spec_b.n_orbitals
    n_active = n_a + n_b
    nelec = (spec_a.n_alpha + spec_b.n_alpha, spec_a.n_beta + spec_b.n_beta)
    casci = mcscf.casci.CASCI(_exact_mean_field(ao), n_active, nelec, ncore=n_core)
    h1, energy_core = casci.get_h1eff(mo_coeff)
    h2 = ao2mo.restore(1, casci.get_h2eff(mo_coeff), n_active)
    blocks = (slice(0, n_a), slice(n_a, n_active))
    densities = [
        compute_spatial_rdm1(
            result.strings_alpha,
            result.strings_beta,
            result.amplitudes,
            spec.n_orbitals,
        )
        for spec, result in zip(specs, results)
    ]
    references = (
        spec_a.n_alpha + spec_a.n_beta,
        spec_b.n_alpha + spec_b.n_beta,
    )
    total = sum(references)
    counts_a = [
        references[0] + hop
        for hop in range(-n_hops, n_hops + 1)
        if 0 <= references[0] + hop <= 2 * n_a
        and 0 <= total - references[0] - hop <= 2 * n_b
    ]
    multiplets = []
    for index, (spec, result) in enumerate(zip(specs, results)):
        own, other = blocks[index], blocks[1 - index]
        coulomb = np.einsum(
            "pqrs,rs->pq", h2[own, own, other, other], densities[1 - index]
        )
        exchange = np.einsum(
            "psrq,rs->pq", h2[own, other, other, own], densities[1 - index]
        )
        counts = counts_a if index == 0 else [total - count for count in counts_a]
        multiplets.append(
            fragment_multiplets(
                h1[own, own] + coulomb - 0.5 * exchange,
                h2[own, own, own, own],
                list(itertools.product(result.strings_alpha, result.strings_beta)),
                references[index],
                counts,
                ao.mol.max_memory,
            )
        )

    def products(limit):
        sizes_a, sizes_b = (_sector_sizes(found, limit) for found in multiplets)
        return sum(
            size * sizes_b[(nelec[0] - a[0], nelec[1] - a[1])]
            for a, size in sizes_a.items()
        )

    limit = n_multiplets
    if n_multiplets is None:
        most = max(
            len(found) for per_fragment in multiplets for found in per_fragment.values()
        )
        if products(most) > _MAX_PRODUCT_STATES:
            fitting = bisect.bisect_right(
                range(1, most + 1), _MAX_PRODUCT_STATES, key=products
            )
            limit = max(fitting, 1)
    kept = (
        limit
        if any(
            len(_kept(found, limit)) < len(found)
            for per_fragment in multiplets
            for found in per_fragment.values()
        )
        else None
    )
    incomplete = {}
    for index, (spec, per_fragment) in enumerate(zip(specs, multiplets)):
        held = sum(len(found) for found in per_fragment.values())
        full = sum(
            _dimension(spec.n_orbitals, ((count + 1) // 2, count // 2))
            for count in per_fragment
        )
        if held < full:
            incomplete[index] = (held, full)
    n_bytes = 4 * 8 * products(limit) ** 2
    if n_bytes > (ao.mol.max_memory - lib.current_memory()[0]) * 1e6:
        raise MemoryError(
            f"The state interaction holds {products(limit)} product states "
            f"densely, {n_bytes / 1e6:g} MB, above mol.max_memory="
            f"{ao.mol.max_memory:g} MB. Lower SQDConfig.fragment_roots, or raise "
            "mol.max_memory."
        )
    basis_a = _basis(n_a, multiplets[0], limit)
    basis_b = _basis(n_b, multiplets[1], limit)
    pairs = _sector_pairs(basis_a, basis_b, nelec)
    hamiltonian, spin_square, labels = si_matrices(h1, h2, basis_a, basis_b, pairs)
    states = tuple(
        ElectronicState(
            state.multiplicity,
            state.index,
            state.energy,
            state.spin_square,
            charge_transfer=float(
                sum(
                    weight**2
                    for weight, (sector_a, _, _, _) in zip(vector, labels)
                    if sum(sector_a) != references[0]
                )
            ),
        )
        for state, vector in si_states(hamiltonian, spin_square, energy_core, requested)
    )
    return states, kept, incomplete


def fragment_multiplets(
    h1: np.ndarray,
    h2: np.ndarray,
    sampled: Sequence[tuple[str, str]],
    reference: int,
    counts: Sequence[int],
    max_memory: float,
) -> dict[int, list[_Multiplet]]:
    """Each electron count's spin multiplets of the fragment Hamiltonian
    ``h1, h2``, ascending in energy, solved in a determinant space grown from
    the ``sampled`` ``(alpha, beta)`` strings.

    At the ``reference`` count that space holds the sampled determinants and
    those one electron move from them; at another count, those reached by
    adding or removing the difference. Each space is closed under spin flips,
    so its eigenstates are pure spin states.

    Raises:
        ValueError: If a count's sector exceeds ``_DENSE_MAX_DIM``
            determinants, since its states are held as dense FCI vectors.
        MemoryError: If those vectors and the solve exceed ``max_memory`` MB.
    """
    n_orbitals = h1.shape[0]
    operators = _Operators(n_orbitals)
    h_spin, g_spin = spin_orbital_integrals(h1, h2, n_orbitals)
    sampled_bits = [
        (ci_string_to_int(alpha), ci_string_to_int(beta)) for alpha, beta in sampled
    ]
    spaces = {}
    for count in counts:
        sector = ((count + 1) // 2, count // 2)
        dimension = _dimension(n_orbitals, sector)
        if dimension > _DENSE_MAX_DIM:
            raise ValueError(
                f"A {n_orbitals}-orbital fragment holds {dimension} determinants "
                f"with {count} electrons, past the {_DENSE_MAX_DIM} the state "
                "interaction holds densely. Use smaller fragments."
            )
        spaces[count] = _spin_closed(
            _grown(sampled_bits, count - reference, n_orbitals), sector, n_orbitals
        )
    # Every count's multiplets as dense FCI vectors, plus one eigensolve.
    n_bytes = (
        8
        * sum(
            len(determinants) * _dimension(n_orbitals, ((count + 1) // 2, count // 2))
            for count, determinants in spaces.items()
        )
        + 3 * 8 * max(len(determinants) for determinants in spaces.values()) ** 2
    )
    if n_bytes > (max_memory - lib.current_memory()[0]) * 1e6:
        raise MemoryError(
            f"The state interaction holds a {n_orbitals}-orbital fragment's "
            f"multiplets densely, {n_bytes / 1e6:g} MB, above mol.max_memory="
            f"{max_memory:g} MB. Use smaller fragments, or raise mol.max_memory."
        )
    return {
        count: _multiplets_in(
            h1,
            h2,
            h_spin,
            g_spin,
            determinants,
            ((count + 1) // 2, count // 2),
            operators,
        )
        for count, determinants in spaces.items()
    }


def _grown(
    sampled: Sequence[tuple[int, int]], change: int, n_orbitals: int
) -> set[tuple[int, int]]:
    """Determinants ``change`` electrons from the sampled ones, or for no change
    the sampled ones and those one electron move away."""
    spin_orbitals = [(spin, p) for spin in (0, 1) for p in range(n_orbitals)]

    def flip(determinant, spin, p):
        alpha, beta = determinant
        return (alpha ^ (1 << p), beta) if spin == 0 else (alpha, beta ^ (1 << p))

    def occupied(determinant, spin, p):
        return bool((determinant[spin] >> p) & 1)

    reached = set(sampled)
    if change == 0:
        for determinant in sampled:
            for (s, p), (t, q) in itertools.permutations(spin_orbitals, 2):
                if occupied(determinant, s, p) and not occupied(determinant, t, q):
                    reached.add(flip(flip(determinant, s, p), t, q))
        return reached
    for _ in range(abs(change)):
        reached = {
            flip(determinant, s, p)
            for determinant in reached
            for s, p in spin_orbitals
            if occupied(determinant, s, p) == (change < 0)
        }
    return reached


def _spin_closed(
    determinants: set[tuple[int, int]], sector: _Sector, n_orbitals: int
) -> list[tuple[int, int]]:
    """Every determinant of ``sector`` sharing a doubly and singly occupied
    orbital pattern with one of ``determinants``, in address order."""
    patterns = {(alpha & beta, alpha ^ beta) for alpha, beta in determinants}
    out = set()
    for doubly, singly in patterns:
        open_shells = [p for p in range(n_orbitals) if (singly >> p) & 1]
        n_up = sector[0] - bin(doubly).count("1")
        if not 0 <= n_up <= len(open_shells):
            continue
        for up in itertools.combinations(open_shells, n_up):
            alpha_open = sum(1 << p for p in up)
            out.add((doubly | alpha_open, doubly | (singly ^ alpha_open)))
    ordered = list(out)
    return [ordered[i] for i in np.argsort(_addresses(ordered, sector, n_orbitals))]


def _addresses(
    determinants: Sequence[tuple[int, int]], sector: _Sector, n_orbitals: int
) -> np.ndarray:
    """Each determinant's address in the sector's dense FCI vector."""
    alpha, beta = (
        cistring.strs2addr(
            n_orbitals, count, np.array([det[spin] for det in determinants], np.int64)
        ).astype(np.int64)
        for spin, count in enumerate(sector)
    )
    return alpha * cistring.num_strings(n_orbitals, sector[1]) + beta


def _multiplets_in(
    h1, h2, h_spin, g_spin, determinants, sector, operators
) -> list[_Multiplet]:
    """The spin multiplets of ``h1, h2`` in the span of ``determinants`` of
    ``sector``, ascending in energy, each with every spin projection."""
    n_orbitals = h1.shape[0]
    if not determinants:
        return []
    occupations = [
        (
            tuple(p for p in range(n_orbitals) if (alpha >> p) & 1),
            tuple(p for p in range(n_orbitals) if (beta >> p) & 1),
        )
        for alpha, beta in determinants
    ]
    h_proj, s2_proj = projected_matrices(
        occupations,
        [spatial_to_spin_occupations(a, b, n_orbitals) for a, b in occupations],
        h_spin,
        g_spin,
        n_orbitals,
    )
    values, vectors = np.linalg.eigh((h_proj + _SPIN_SPLIT * s2_proj).toarray())
    addresses = _addresses(determinants, sector, n_orbitals)
    dimension = _dimension(n_orbitals, sector)
    multiplets = []
    for value, vector in zip(values, vectors.T):
        dense = np.zeros(dimension)
        dense[addresses] = vector
        spin_square = float(dense @ (operators.spin_square(sector) @ dense))
        spin = 0.5 * round(np.sqrt(4.0 * spin_square + 1.0) - 1.0)
        multiplets.append(
            _Multiplet(float(value) - _SPIN_SPLIT * spin_square, spin, sector, dense)
        )
    return sorted(multiplets, key=lambda multiplet: multiplet.energy)


def _components(
    multiplet: _Multiplet, operators: "_Operators"
) -> Iterator[tuple[_Sector, np.ndarray]]:
    """Each spin projection's sector and FCI vector, ``S+`` and ``S-``
    applied from the stored one."""
    yield multiplet.sector, multiplet.vector
    for direction in (1, -1):
        current, sector = multiplet.vector, multiplet.sector
        projection = 0.5 * (sector[0] - sector[1])
        while (projection + direction) * direction <= multiplet.spin:
            ladder = operators.ladder(sector, direction)
            if ladder is None:
                break
            current = ladder @ current
            current = current / np.linalg.norm(current)
            sector = (sector[0] + direction, sector[1] - direction)
            projection += direction
            yield sector, current


def _apply_h(h1, h2, sector, vectors):
    """``H`` applied to each column of ``vectors``, dense FCI vectors of
    ``sector``."""
    n_orbitals = h1.shape[0]
    shape = (
        cistring.num_strings(n_orbitals, sector[0]),
        cistring.num_strings(n_orbitals, sector[1]),
    )
    absorbed = direct_spin1.absorb_h1e(h1, h2, n_orbitals, sector, 0.5)
    return np.column_stack(
        [
            direct_spin1.contract_2e(
                absorbed, vector.reshape(shape), n_orbitals, sector
            ).ravel()
            for vector in vectors.T
        ]
    )


def _basis(
    n_orbitals: int, multiplets: Mapping[int, list[_Multiplet]], limit: int | None
) -> _FragmentBasis:
    """Each count's lowest ``limit`` multiplets (all for ``None``), never
    splitting a degenerate set, gathered into per-sector state arrays."""
    operators = _Operators(n_orbitals)
    columns: dict[_Sector, list[np.ndarray]] = {}
    for found in multiplets.values():
        for multiplet in _kept(found, limit):
            for sector, vector in _components(multiplet, operators):
                columns.setdefault(sector, []).append(vector)
    return _FragmentBasis(
        n_orbitals,
        {sector: np.column_stack(vectors) for sector, vectors in columns.items()},
    )


def _kept(found: Sequence[_Multiplet], limit: int | None) -> Sequence[_Multiplet]:
    if limit is None or limit >= len(found):
        return found
    cut = found[limit - 1].energy + _DEGENERACY_TOL
    return [multiplet for multiplet in found if multiplet.energy < cut]


def _sector_pairs(
    basis_a: _FragmentBasis, basis_b: _FragmentBasis, nelec: tuple[int, int]
) -> list[tuple[_Sector, _Sector]]:
    """Every pair of sectors, one per fragment, holding states and adding up
    to the active space's ``nelec``."""
    return [
        (sector_a, (nelec[0] - sector_a[0], nelec[1] - sector_a[1]))
        for sector_a in sorted(basis_a.states)
        if (nelec[0] - sector_a[0], nelec[1] - sector_a[1]) in basis_b.states
    ]


def _sector_sizes(
    multiplets: Mapping[int, list[_Multiplet]], limit: int | None
) -> Counter[_Sector]:
    """How many states each sector of ``_basis`` at ``limit`` holds."""
    return Counter(
        sector
        for found in multiplets.values()
        for multiplet in _kept(found, limit)
        for sector in multiplet.sectors
    )


def _dimension(n_orbitals: int, sector: _Sector) -> int:
    return cistring.num_strings(n_orbitals, sector[0]) * cistring.num_strings(
        n_orbitals, sector[1]
    )


def _valid(sector: _Sector, n_orbitals: int) -> bool:
    return all(0 <= count <= n_orbitals for count in sector)


class _Operators:
    """One fragment's sparse creation, annihilation and spin operators between
    sectors, built on first use and kept for one state interaction."""

    def __init__(self, n_orbitals: int):
        self.n_orbitals = n_orbitals
        self._cache: dict = {}

    def __call__(
        self, sector: _Sector, orbital: int, spin: int, dagger: bool
    ) -> scipy.sparse.csr_array | None:
        """The operator on one spin-orbital, from ``sector`` to the shifted
        sector; ``None`` past the fragment's bounds."""
        key = (sector, orbital, spin, dagger)
        if key not in self._cache:
            self._cache[key] = self._build(sector, orbital, spin, dagger)
        return self._cache[key]

    def on_every_orbital(
        self, sector: _Sector, spin: int, dagger: bool
    ) -> list[scipy.sparse.csr_array] | None:
        """The operator on each of the fragment's orbitals, in order; ``None``
        past its bounds."""
        matrices = [self(sector, p, spin, dagger) for p in range(self.n_orbitals)]
        if any(matrix is None for matrix in matrices):
            return None
        return [matrix for matrix in matrices if matrix is not None]

    def _build(self, sector, orbital, spin, dagger):
        target = _target_sector(sector, ((dagger, spin),))
        if not _valid(target, self.n_orbitals):
            return None
        n_alpha, n_beta = (
            cistring.num_strings(self.n_orbitals, count) for count in sector
        )
        links = self._links(sector[spin], dagger)
        sources, entries = np.nonzero(links[:, :, 1 - dagger] == orbital)
        signs = links[sources, entries, 3].astype(float)
        if spin == _BETA and sector[0] % 2:
            signs = -signs
        one_spin = scipy.sparse.csr_array(
            (signs, (links[sources, entries, 2], sources)),
            shape=(
                cistring.num_strings(self.n_orbitals, target[spin]),
                links.shape[0],
            ),
        )
        if spin == _ALPHA:
            return scipy.sparse.csr_array(
                scipy.sparse.kron(one_spin, scipy.sparse.identity(n_beta))
            )
        return scipy.sparse.csr_array(
            scipy.sparse.kron(scipy.sparse.identity(n_alpha), one_spin)
        )

    def _links(self, count: int, dagger: bool) -> np.ndarray:
        """The ``(string, entry, [creator, annihilator, target, sign])`` link
        table of one spin's ``count``-electron strings."""
        key = ("links", count, dagger)
        if key not in self._cache:
            make = cistring.gen_cre_str_index if dagger else cistring.gen_des_str_index
            self._cache[key] = make(range(self.n_orbitals), count)
        return self._cache[key]

    def ladder(self, sector: _Sector, direction: int):
        """``S+`` (``direction`` 1) or ``S-`` (-1) from ``sector``; ``None``
        past the fragment's bounds."""
        key = ("ladder", sector, direction)
        if key not in self._cache:
            raised, lowered = (_ALPHA, _BETA) if direction == 1 else (_BETA, _ALPHA)
            total = None
            for p in range(self.n_orbitals):
                annihilate = self(sector, p, lowered, False)
                if annihilate is None:
                    break
                create = self(
                    _target_sector(sector, ((False, lowered),)), p, raised, True
                )
                if create is None:
                    break
                term = create @ annihilate
                total = term if total is None else total + term
            self._cache[key] = total
        return self._cache[key]

    def spin_square(self, sector: _Sector):
        """``S^2 = S- S+ + Sz (Sz + 1)`` within ``sector``."""
        key = ("spin_square", sector)
        if key not in self._cache:
            sz = 0.5 * (sector[0] - sector[1])
            square = (sz * (sz + 1.0)) * scipy.sparse.identity(
                _dimension(self.n_orbitals, sector), format="csr"
            )
            raised = self.ladder(sector, 1)
            if raised is not None:
                lowered = self.ladder((sector[0] + 1, sector[1] - 1), -1)
                square = square + lowered @ raised
            self._cache[key] = scipy.sparse.csr_array(square)
        return self._cache[key]


def _target_sector(sector: _Sector, ops: _Ops) -> _Sector:
    """The sector ``ops`` lead to from ``sector``."""
    alpha, beta = sector
    for dagger, spin in ops:
        step = 1 if dagger else -1
        if spin == _ALPHA:
            alpha += step
        else:
            beta += step
    return alpha, beta


class _Densities:
    """Memoised ``<bra|o_1 ... o_m|ket>`` between one fragment's model states,
    as ``(bra states, ket states, orbital, ..., orbital)`` arrays with the
    orbital axes in operator order. The string is split in two and each half
    applied to its own side, so no array holds more than two orbital axes."""

    def __init__(self, basis: _FragmentBasis, operators: _Operators):
        self._basis = basis
        self._operators = operators
        self._cache: dict = {}

    def __call__(self, bra: _Sector, ket: _Sector, ops: _Ops) -> np.ndarray | None:
        key = (bra, ket, ops)
        if key not in self._cache:
            self._cache[key] = self._compute(bra, ket, ops)
        return self._cache[key]

    def _compute(self, bra, ket, ops):
        states = self._basis.states
        if bra not in states or ket not in states or _target_sector(ket, ops) != bra:
            return None
        split = len(ops) // 2
        middle = _target_sector(ket, ops[split:])
        right = self._apply(states[ket], ket, ops[split:])
        if right is None:
            return None
        # <bra| o_1 .. o_k is (o_k^T .. o_1^T |bra>)^T, built from the bra side.
        left = states[bra]
        sector = bra
        for dagger, spin in ops[:split]:
            source = _target_sector(sector, ((not dagger, spin),))
            matrices = self._operators.on_every_orbital(source, spin, dagger)
            if matrices is None:
                return None
            tail = left.shape[1:]
            flat = left.reshape(left.shape[0], -1)
            left = np.stack([matrix.T @ flat for matrix in matrices], axis=-1)
            left = left.reshape(left.shape[0], *tail, len(matrices))
            sector = source
        if sector != middle:
            return None
        n_left = left.ndim - 2
        joined = np.tensordot(left, right, axes=([0], [0]))
        return np.moveaxis(joined, n_left + 1, 1)

    def _apply(self, vectors, sector, ops):
        """``o_1 .. o_m`` applied to ``vectors`` as ``(determinants, states,
        orbital, ..., orbital)``, orbital axes in operator order."""
        current = vectors
        for dagger, spin in reversed(ops):
            matrices = self._operators.on_every_orbital(sector, spin, dagger)
            if matrices is None:
                return None
            flat = current.reshape(current.shape[0], -1)
            stacked = np.stack([matrix @ flat for matrix in matrices])
            current = np.moveaxis(
                stacked.reshape(len(matrices), -1, *current.shape[1:]), 0, 2
            )
            sector = _target_sector(sector, ((dagger, spin),))
        return current


def _fragment_hamiltonians(
    basis: _FragmentBasis, h1: np.ndarray, h2: np.ndarray
) -> dict[_Sector, np.ndarray]:
    """Each sector's ``<a'|H_fragment|a>`` over its model states."""
    return {
        sector: vectors.T @ _apply_h(h1, h2, sector, vectors)
        for sector, vectors in basis.states.items()
    }


def _cross_terms(
    h1: np.ndarray, h2: np.ndarray, n_a: int
) -> list[tuple[float, _Ops, tuple[int, ...], np.ndarray]]:
    """The Hamiltonian's operator strings touching both fragments, as
    ``(crossing sign, ops, fragment of each op (0 is A), coefficients)``.

    ``ops`` are ``(dagger, spin)`` in written order; the coefficient array has
    one axis per op, over that op's fragment's orbitals.
    """
    n = h1.shape[0]
    ranges = (np.arange(n_a), np.arange(n_a, n))
    # 1/2 g_pqrs a+_p a+_r a_s a_q, axes reordered to (p, r, s, q).
    two_body = 0.5 * h2.transpose(0, 2, 3, 1)
    strings: list[tuple[_Ops, np.ndarray]] = [
        (((True, s), (False, s)), h1) for s in (_ALPHA, _BETA)
    ] + [
        (((True, s), (True, t), (False, t), (False, s)), two_body)
        for s, t in itertools.product((_ALPHA, _BETA), repeat=2)
    ]
    terms = []
    for ops, coefficients in strings:
        for fragments in itertools.product((0, 1), repeat=len(ops)):
            if len(set(fragments)) == 1:
                continue
            crossings = sum(
                1
                for i, j in itertools.combinations(range(len(ops)), 2)
                if fragments[i] == 1 and fragments[j] == 0
            )
            block = coefficients[np.ix_(*(ranges[f] for f in fragments))]
            terms.append(((-1.0) ** crossings, ops, fragments, block))
    return terms


def si_matrices(
    h1: np.ndarray,
    h2: np.ndarray,
    basis_a: _FragmentBasis,
    basis_b: _FragmentBasis,
    pairs: Sequence[tuple[_Sector, _Sector]],
) -> tuple[np.ndarray, np.ndarray, list[tuple[_Sector, _Sector, int, int]]]:
    """The active-space Hamiltonian and ``S^2`` over the product states
    ``|a>|b>`` of the two fragments' model states in ``pairs``.

    ``h1, h2`` are the active space's, fragment A's orbitals first. A product
    state is A's creation string, then B's, on the vacuum.

    Returns:
        ``(hamiltonian, spin_square, labels)``, each label ``(sector_a,
        sector_b, a, b)`` naming one product state.
    """
    n_a = basis_a.n_orbitals
    a_block, b_block = slice(0, n_a), slice(n_a, None)
    operators = (_Operators(basis_a.n_orbitals), _Operators(basis_b.n_orbitals))
    bases = (basis_a, basis_b)
    local = (
        _fragment_hamiltonians(
            basis_a, h1[a_block, a_block], h2[a_block, a_block, a_block, a_block]
        ),
        _fragment_hamiltonians(
            basis_b, h1[b_block, b_block], h2[b_block, b_block, b_block, b_block]
        ),
    )
    local_s2 = tuple(
        {
            sector: vectors.T @ (operators[f].spin_square(sector) @ vectors)
            for sector, vectors in bases[f].states.items()
        }
        for f in (0, 1)
    )
    densities = (
        _Densities(basis_a, operators[0]),
        _Densities(basis_b, operators[1]),
    )
    terms = _cross_terms(h1, h2, n_a)

    labels = []
    offsets = {}
    for pair in pairs:
        offsets[pair] = len(labels)
        sector_a, sector_b = pair
        labels += [
            (sector_a, sector_b, a, b)
            for a in range(basis_a.states[sector_a].shape[1])
            for b in range(basis_b.states[sector_b].shape[1])
        ]
    hamiltonian = np.zeros((len(labels), len(labels)))
    spin_square = np.zeros_like(hamiltonian)
    up = ((True, _ALPHA), (False, _BETA))
    down = ((True, _BETA), (False, _ALPHA))

    for bra, ket in itertools.product(pairs, repeat=2):
        (bra_a, bra_b), (ket_a, ket_b) = bra, ket
        n_bra = (basis_a.states[bra_a].shape[1], basis_b.states[bra_b].shape[1])
        n_ket = (basis_a.states[ket_a].shape[1], basis_b.states[ket_b].shape[1])
        h_block = np.zeros((*n_bra, *n_ket))
        s_block = np.zeros_like(h_block)
        if bra == ket:
            identity_a, identity_b = np.eye(n_ket[0]), np.eye(n_ket[1])
            h_block += np.einsum("ik,jl->ijkl", local[0][ket_a], identity_b)
            h_block += np.einsum("ik,jl->ijkl", identity_a, local[1][ket_b])
            s_block += np.einsum("ik,jl->ijkl", local_s2[0][ket_a], identity_b)
            s_block += np.einsum("ik,jl->ijkl", identity_a, local_s2[1][ket_b])
            sz_a, sz_b = (0.5 * (s[0] - s[1]) for s in (ket_a, ket_b))
            s_block += (
                2.0 * sz_a * sz_b * np.einsum("ik,jl->ijkl", identity_a, identity_b)
            )
        # S+_A S-_B + S-_A S+_B.
        for ops_a, ops_b in ((up, down), (down, up)):
            d_a = densities[0](bra_a, ket_a, ops_a)
            d_b = densities[1](bra_b, ket_b, ops_b)
            if d_a is not None and d_b is not None:
                s_block += np.einsum("ikpp,jlqq->ijkl", d_a, d_b)
        parity_a = (ket_a[0] + ket_a[1]) % 2
        for sign, ops, fragments, coefficients in terms:
            ops_a = tuple(op for op, f in zip(ops, fragments) if f == 0)
            ops_b = tuple(op for op, f in zip(ops, fragments) if f == 1)
            d_a = densities[0](bra_a, ket_a, ops_a)
            if d_a is None:
                continue
            d_b = densities[1](bra_b, ket_b, ops_b)
            if d_b is None:
                continue
            if len(ops_b) % 2 and parity_a:
                sign = -sign
            h_block += sign * _contract(coefficients, fragments, d_a, d_b)
        rows = slice(offsets[bra], offsets[bra] + n_bra[0] * n_bra[1])
        columns = slice(offsets[ket], offsets[ket] + n_ket[0] * n_ket[1])
        hamiltonian[rows, columns] = h_block.reshape(n_bra[0] * n_bra[1], -1)
        spin_square[rows, columns] = s_block.reshape(n_bra[0] * n_bra[1], -1)
    return hamiltonian, spin_square, labels


def _contract(
    coefficients: np.ndarray,
    fragments: tuple[int, ...],
    d_a: np.ndarray,
    d_b: np.ndarray,
) -> np.ndarray:
    """``sum coefficients[o_1..o_m] d_a[..., A's o] d_b[..., B's o]`` as
    ``(bra a, bra b, ket a, ket b)``."""
    letters = "pqrs"[: len(fragments)]
    on_a = "".join(letter for letter, f in zip(letters, fragments) if f == 0)
    on_b = "".join(letter for letter, f in zip(letters, fragments) if f == 1)
    return np.einsum(
        f"{letters},ik{on_a},jl{on_b}->ijkl", coefficients, d_a, d_b, optimize=True
    )


def si_states(
    hamiltonian: np.ndarray,
    spin_square: np.ndarray,
    energy_core: float,
    requested: Mapping[int, int],
) -> list[tuple[ElectronicState, np.ndarray]]:
    """The requested states of the state-interaction Hamiltonian, each with its
    product-state coefficients, by multiplicity then root.

    One diagonalisation of ``hamiltonian`` plus a small ``spin_square`` term
    gives every multiplicity's roots. A root whose ``<S^2>`` is nearer another
    spin is skipped, never replaced, so a multiplicity can yield fewer roots
    than requested.
    """
    values, vectors = np.linalg.eigh(hamiltonian + _SPIN_SPLIT * spin_square)
    found: dict[int, list[tuple[ElectronicState, np.ndarray]]] = {
        multiplicity: [] for multiplicity in requested
    }
    for value, vector in zip(values, vectors.T):
        if all(len(found[m]) >= n for m, n in requested.items()):
            break
        s2 = float(vector @ spin_square @ vector)
        energy = float(value) - _SPIN_SPLIT * s2 + energy_core
        for multiplicity, roots in found.items():
            if len(roots) < requested[multiplicity] and resolves_spin(multiplicity, s2):
                roots.append(
                    (ElectronicState(multiplicity, len(roots), energy, s2), vector)
                )
    return [entry for _, roots in sorted(found.items()) for entry in roots]
