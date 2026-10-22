# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Automatic active-space orbital selection and localisation."""

import warnings
from collections.abc import Sequence
from functools import cache
from itertools import combinations

import networkx as nx
import numpy as np
from pyscf import lo
from pyscf.lo.pipek import atomic_pops

from ._integrals import transform_integrals
from ._state import FragmentSpec

DEFAULT_MAX_ORBITALS_PER_FRAGMENT = 4
DEFAULT_COUPLING_THRESHOLD = 1e-3


def select_frontier_orbitals(
    n_orbitals_total: int,
    n_occupied: int,
    n_active_orbitals: int,
    *,
    spin: int = 0,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Choose active orbitals around the HOMO-LUMO gap.

    Args:
        n_orbitals_total: Size of the MO register.
        n_occupied: Number of occupied spatial orbitals, doubly and singly
            occupied together.
        n_active_orbitals: Take the ``ceil(k / 2)`` highest occupied and
            ``floor(k / 2)`` lowest virtual orbitals.
        spin: The molecule's ``2S``; its ``|spin|`` singly occupied orbitals,
            the highest occupied, must all be taken.

    Returns:
        ``(occupied_indices, virtual_indices)``, each ascending.

    Raises:
        ValueError: If ``n_active_orbitals`` is not positive, asks for more
            occupied or virtual orbitals than the register holds, takes no
            virtual orbital, or takes fewer occupied orbitals than there are
            singly occupied ones.
    """
    if n_active_orbitals <= 0:
        raise ValueError(
            f"n_active_orbitals must be positive; got {n_active_orbitals}."
        )
    n_occ_take = (n_active_orbitals + 1) // 2
    n_virt_take = n_active_orbitals // 2
    if n_occ_take < abs(spin):
        raise ValueError(
            f"n_active_orbitals={n_active_orbitals} takes {n_occ_take} occupied "
            f"orbitals, fewer than the molecule's {abs(spin)} singly occupied "
            f"ones, which must all be active. Raise it to at least "
            f"{2 * abs(spin) - 1}."
        )
    n_virtual = n_orbitals_total - n_occupied
    if n_occ_take > n_occupied or n_virt_take > n_virtual:
        raise ValueError(
            f"n_active_orbitals={n_active_orbitals} takes {n_occ_take} occupied "
            f"and {n_virt_take} virtual orbitals, but the molecule has "
            f"{n_occupied} occupied and {n_virtual} virtual. Lower it, or pick "
            "the orbitals with active_orbitals."
        )
    if n_virt_take == 0:
        raise ValueError(
            "The active space must contain at least one occupied and one "
            f"virtual orbital; n_active_orbitals={n_active_orbitals} takes no "
            "virtual."
        )
    return (
        tuple(range(n_occupied - n_occ_take, n_occupied)),
        tuple(range(n_occupied, n_occupied + n_virt_take)),
    )


def _canonicalize_columns(mol, block: np.ndarray) -> np.ndarray:
    """Reorder a localised block's columns into a deterministic order.

    Columns are sorted by the index of the atom carrying their largest
    Mulliken population, tiebroken by the population-weighted centroid
    coordinate; columns tied on both keep their input order. Pipek-Mezey's
    cost function is invariant to the order of the localised columns it
    returns, so two runs that converge to the same physical solution (same
    cost) can still return it in a different column order, and the fragment
    partition built from those columns would follow it. Sorting removes that
    dependence wherever the keys differ.
    """
    n_col = block.shape[1]
    atom_population = atomic_pops(mol, block, method="mulliken", mode="pop")
    dominant_atom = np.argmax(atom_population, axis=0)

    coords = mol.atom_coords()
    total_population = atom_population.sum(axis=0)
    centroid = (atom_population.T @ coords) / total_population[:, None]

    order = sorted(
        range(n_col),
        key=lambda col: (int(dominant_atom[col]), tuple(centroid[col])),
    )
    return block[:, order]


def split_active_orbitals(
    active_orbitals: Sequence[int],
    n_occupied: int,
    n_orbitals_total: int,
    *,
    spin: int = 0,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Split explicit active MO indices into occupied and virtual, ascending.

    Args:
        active_orbitals: MO column indices forming the active space.
        n_occupied: Number of occupied spatial orbitals, doubly and singly
            occupied together; an index below this is occupied.
        n_orbitals_total: Size of the MO register.
        spin: The molecule's ``2S``; its ``|spin|`` singly occupied orbitals,
            the highest occupied, must all be active.

    Raises:
        ValueError: If ``active_orbitals`` is empty, contains duplicates or
            out-of-range indices, has no occupied or no virtual orbital, or
            leaves out a singly occupied orbital.
    """
    indices = tuple(int(orbital) for orbital in active_orbitals)
    if not indices:
        raise ValueError("active_orbitals must contain at least one orbital.")
    if len(set(indices)) != len(indices):
        raise ValueError(f"active_orbitals contains duplicates: {indices}.")
    for orbital in indices:
        if not 0 <= orbital < n_orbitals_total:
            raise ValueError(
                f"active_orbitals index {orbital} is out of range for a register "
                f"of {n_orbitals_total} orbitals."
            )

    occupied = tuple(sorted(o for o in indices if o < n_occupied))
    virtual = tuple(sorted(o for o in indices if o >= n_occupied))
    if not occupied or not virtual:
        raise ValueError(
            "The active space must contain at least one occupied and one virtual "
            f"orbital; got {len(occupied)} occupied and {len(virtual)} virtual "
            f"from active_orbitals={indices} with n_occupied={n_occupied}."
        )
    missing = [o for o in range(n_occupied - abs(spin), n_occupied) if o not in indices]
    if missing:
        raise ValueError(
            f"active_orbitals leaves out singly occupied orbitals {missing}. The "
            "core must be closed-shell, so every singly occupied orbital has to "
            "be active."
        )
    return occupied, virtual


def validate_fragment_atoms(
    fragment_atoms: Sequence[Sequence[int]], n_atoms: int
) -> dict[int, int]:
    """Map each named atom to its fragment index, rejecting overlap and range errors.

    Raises:
        ValueError: If ``fragment_atoms`` is empty, names an out-of-range atom,
            shares an atom between fragments, or contains an empty fragment.
    """
    if not fragment_atoms:
        raise ValueError("fragment_atoms must contain at least one fragment.")

    owner: dict[int, int] = {}
    for index, atoms in enumerate(fragment_atoms):
        if not atoms:
            raise ValueError(f"fragment_atoms[{index}] names no atoms.")
        for atom in atoms:
            if not 0 <= atom < n_atoms:
                raise ValueError(
                    f"fragment_atoms[{index}] names atom {atom}, out of range "
                    f"for a molecule with {n_atoms} atoms."
                )
            if atom in owner:
                raise ValueError(
                    f"Atom {atom} appears in both fragment {owner[atom]} and "
                    f"fragment {index}. Fragments must be disjoint."
                )
            owner[atom] = index
    return owner


def assign_orbitals_to_atoms(
    mol, localized: np.ndarray, fragment_atoms: Sequence[Sequence[int]]
) -> list[tuple[int, ...]]:
    """Group localised columns into fragments by the atom they sit on.

    Each column is assigned to whichever fragment owns the atom carrying its
    largest Mulliken population. This is the fragmentation a localised active
    space is usually defined by -- "these orbitals belong to this metal centre"
    -- rather than one inferred from orbital coupling.

    Args:
        mol: A PySCF ``gto.Mole``.
        localized: ``(nao, n_act)`` localised active-space coefficients.
        fragment_atoms: One sequence of atom indices per fragment.

    Returns:
        One tuple of ``localized`` column indices per fragment, in the order
        ``fragment_atoms`` was given.

    Raises:
        ValueError: If any column's dominant atom is not in ``fragment_atoms``,
            or if a fragment ends up with no orbitals. Atom range and disjointness are
            propagated from :func:`validate_fragment_atoms`.
    """
    owner = validate_fragment_atoms(fragment_atoms, mol.natm)
    dominant_atom = np.argmax(
        atomic_pops(mol, localized, method="mulliken", mode="pop"), axis=0
    )
    clusters: list[list[int]] = [[] for _ in fragment_atoms]
    for column, atom in enumerate(dominant_atom):
        atom = int(atom)
        if atom not in owner:
            raise ValueError(
                f"Localised active orbital {column} sits mainly on atom {atom} "
                f"({mol.atom_symbol(atom)}), which fragment_atoms does not list. "
                "Every active orbital must belong to some fragment; add that "
                "atom to one of them, or choose an active space localised on "
                "the atoms you listed."
            )
        clusters[owner[atom]].append(column)

    for index, cluster in enumerate(clusters):
        if not cluster:
            raise ValueError(
                f"fragment_atoms[{index}] ({list(fragment_atoms[index])}) got no "
                "active orbitals. The localised active space puts none of its "
                "orbitals on those atoms."
            )
    return [tuple(cluster) for cluster in clusters]


def localize_blocks(
    mol,
    mo_coeff: np.ndarray,
    occupied_indices,
    virtual_indices,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Pipek-Mezey localize the occupied and virtual blocks independently.

    Localising the two blocks separately keeps each localised orbital
    unambiguously occupied or virtual, which makes fragment electron counts
    exact rather than inferred from occupation thresholds. Localising them
    jointly would mix occupied and virtual character across the resulting
    orbitals and destroy that property.

    PySCF starts each localisation from its atomic initial guess, the block's
    columns rotated as close as possible to orthogonalised AOs. For a
    symmetric molecule that run can stop at a stationary point of the
    Pipek-Mezey cost rather than a maximum, and a single random restart can
    land on the same one. To make escaping likely, each block is also
    localised from several randomly rotated copies of its columns (each a
    fresh random orthogonal rotation drawn from ``rng``), and whichever run
    reaches the highest Pipek-Mezey cost is kept. This never does worse than
    the unrotated run alone, but escaping a symmetric stationary point is not
    guaranteed for every input.

    Once a restart's cost exceeds the unrotated run's by more than a small
    relative tolerance, the stationary point has been escaped and further
    restarts stop early. A restart that lands back on that cost (or close to
    it) keeps trying, up to the restart cap, which is exactly the case the
    restarts exist for.

    Args:
        mol: A PySCF ``gto.Mole``.
        mo_coeff: ``(nao, n_orb)`` MO coefficients.
        occupied_indices: Column indices of ``mo_coeff`` forming the occupied
            block.
        virtual_indices: Column indices of ``mo_coeff`` forming the virtual
            block.
        rng: Random number generator used to perturb the localisation's
            starting point. Required so that fragmentation stays
            reproducible under a caller-supplied seed rather than drawing
            from unseeded global randomness.

    Returns:
        ``(localized_occupied, localized_virtual)`` AO-basis coefficient
        blocks. A block with a single orbital is returned unchanged, since
        Pipek-Mezey localisation on one orbital is a no-op. Each returned
        block's columns are canonically ordered (see
        :func:`_canonicalize_columns`): by the index of the atom carrying
        their largest Mulliken population, tiebroken by the
        population-weighted centroid coordinate. This makes the column order
        reproducible under a fixed seed even when the optimizer reaches the
        same physical solution via a different column order on different
        runs.
    """

    return (
        _localize_block(mol, mo_coeff[:, list(occupied_indices)], rng),
        _localize_block(mol, mo_coeff[:, list(virtual_indices)], rng),
    )


def _localize_block(mol, block: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One block of :func:`localize_blocks`; a block of at most one orbital is
    returned unchanged."""
    n_restarts = 8
    n = block.shape[1]
    if n <= 1:
        return block.copy()

    unrotated_start = lo.PipekMezey(mol, block)
    best_result = unrotated_start.kernel()
    best_cost = unrotated_cost = unrotated_start.cost_function()

    escape_tolerance = 1e-6
    for _ in range(n_restarts):
        orthogonal, _ = np.linalg.qr(rng.standard_normal((n, n)))
        perturbed_start = lo.PipekMezey(mol, block @ orthogonal)
        # Passed explicitly: PySCF's atomic guess would undo the rotation.
        perturbed_result = perturbed_start.kernel(block @ orthogonal)
        perturbed_cost = perturbed_start.cost_function()
        if perturbed_cost > best_cost:
            best_result, best_cost = perturbed_result, perturbed_cost
        if best_cost > unrotated_cost * (1.0 + escape_tolerance):
            break

    return _canonicalize_columns(mol, best_result)


def build_coupling_graph(
    one_body: np.ndarray,
    two_body: np.ndarray,
    *,
    coupling_threshold: float = DEFAULT_COUPLING_THRESHOLD,
) -> nx.Graph:
    """Build the orbital coupling graph in the localised basis.

    Edge weight is ``|h_pq| + |(pq|qp)|`` — the one-electron coupling plus
    the exchange integral. ``coupling_threshold`` is relative: edges with
    weight below ``coupling_threshold * max(w)`` are dropped, which keeps the
    default scale-free across molecules and basis sets. In practice, for
    genuinely localised orbitals the weakest surviving cross-fragment
    coupling is typically tens of times larger than ``coupling_threshold``'s
    default of ``1e-3`` would require to prune, so the default rarely drops
    any edge for small active spaces; a caller who wants pruning to actually
    happen should raise this value substantially rather than relying on the
    default.

    Args:
        one_body: ``(n_orb, n_orb)`` one-electron integrals in the localised
            basis.
        two_body: ``(n_orb,) * 4`` two-electron integrals in chemist order,
            in the same basis.
        coupling_threshold: Fraction of the strongest edge weight below which
            an edge is dropped.

    Returns:
        An undirected graph with nodes ``0..n_orb - 1`` and a ``weight``
        attribute on every surviving edge.

    Raises:
        ValueError: If ``one_body`` is not square, or if ``two_body``'s shape
            does not match ``one_body``'s orbital count.
    """
    n_orb = one_body.shape[0]
    if one_body.shape != (n_orb, n_orb):
        raise ValueError(f"one_body must be square; got shape {one_body.shape}.")
    if two_body.shape != (n_orb,) * 4:
        raise ValueError(
            f"two_body must have shape {(n_orb,) * 4} to match one_body's "
            f"{n_orb} orbitals; got {two_body.shape}."
        )

    weights: dict[tuple[int, int], float] = {
        (p, q): abs(one_body[p, q]) + abs(two_body[p, q, q, p])
        for p, q in combinations(range(n_orb), 2)
    }

    graph = nx.Graph()
    graph.add_nodes_from(range(n_orb))
    if not weights:
        return graph

    cutoff = coupling_threshold * max(weights.values())
    for (p, q), weight in weights.items():
        if weight > 0.0 and weight >= cutoff:
            graph.add_edge(p, q, weight=weight)
    return graph


def _inter_cluster_weight(
    graph: nx.Graph, left: tuple[int, ...], right: tuple[int, ...]
) -> float:
    """Total surviving edge weight spanning two clusters."""
    return sum(
        graph[p][q]["weight"] for p in left for q in right if graph.has_edge(p, q)
    )


def merge_clusters(
    graph: nx.Graph,
    is_occupied: Sequence[bool],
    max_orbitals_per_fragment: int,
) -> list[tuple[int, ...]]:
    """Greedily merge coupled orbitals into fragments under a size limit.

    Starts from one singleton cluster per orbital and repeatedly merges the
    most strongly coupled pair of clusters that stays within
    ``max_orbitals_per_fragment``, using the total surviving edge weight
    between clusters as the merge criterion. Ties are broken by scan order:
    candidate pairs are visited in ascending ``(i, j)`` index-pair order
    within the current cluster list, and a merge only replaces the running
    best when its weight is strictly greater, so the earliest-scanned pair
    among ties wins. A fix-up pass then absorbs any cluster holding only
    occupied or only virtual orbitals into a neighbour, since such a fragment
    captures no correlation between occupied and virtual character. A
    neighbour with positive coupling is preferred; a neighbour with zero
    coupling is used only when no coupled neighbour fits within
    ``max_orbitals_per_fragment``, since an unmixed fragment is worse than a
    zero-coupling merge. Ties in this pass favour the smallest partner, since
    consuming a small cluster leaves more room for other clusters that still
    need a fix.

    Args:
        graph: The orbital coupling graph, as returned by
            :func:`build_coupling_graph`.
        is_occupied: Whether each orbital (indexed as in ``graph``) is
            occupied.
        max_orbitals_per_fragment: Maximum number of orbitals a single
            cluster may hold.

    Returns:
        Clusters as ascending tuples of orbital indices, sorted, pairwise
        disjoint, and covering every node in ``graph`` exactly once.

    Raises:
        ValueError: If ``max_orbitals_per_fragment`` is below 2, or if an
            occupied-only or virtual-only cluster cannot be merged with any
            neighbour without exceeding ``max_orbitals_per_fragment``.
    """
    if max_orbitals_per_fragment < 2:
        raise ValueError(
            "max_orbitals_per_fragment must be at least 2; got "
            f"{max_orbitals_per_fragment}."
        )

    clusters: list[tuple[int, ...]] = [(node,) for node in sorted(graph.nodes)]

    @cache
    def coupling(left: tuple[int, ...], right: tuple[int, ...]) -> float:
        return _inter_cluster_weight(graph, left, right)

    def best_merge(candidates):
        best, best_weight = None, 0.0
        for i, j in combinations(range(len(candidates)), 2):
            if len(candidates[i]) + len(candidates[j]) > max_orbitals_per_fragment:
                continue
            weight = coupling(candidates[i], candidates[j])
            if weight > best_weight:
                best, best_weight = (i, j), weight
        return best

    while (pair := best_merge(clusters)) is not None:
        i, j = pair
        merged = tuple(sorted(clusters[i] + clusters[j]))
        clusters = [cluster for k, cluster in enumerate(clusters) if k not in (i, j)]
        clusters.append(merged)

    def is_mixed(cluster: tuple[int, ...]) -> bool:
        occupations = {bool(is_occupied[orbital]) for orbital in cluster}
        return len(occupations) == 2

    uncoupled_merges: list[tuple[int, ...]] = []
    while len(clusters) > 1:
        unmixed = [cluster for cluster in sorted(clusters) if not is_mixed(cluster)]
        if not unmixed:
            break
        cluster = unmixed[0]

        within_limit = [
            other
            for other in sorted(clusters)
            if other != cluster
            and len(cluster) + len(other) <= max_orbitals_per_fragment
        ]
        if not within_limit:
            character = "occupied" if is_occupied[cluster[0]] else "virtual"
            raise ValueError(
                f"Orbitals {cluster} form an all-{character} fragment that "
                "captures no correlation, and cannot be merged with any "
                "neighbor without exceeding "
                f"max_orbitals_per_fragment={max_orbitals_per_fragment}. "
                "Supply explicit fragments or raise the limit."
            )

        coupled = [other for other in within_limit if coupling(cluster, other) > 0.0]
        partner = max(
            within_limit,
            key=lambda other: (coupling(cluster, other), -len(other)),
        )
        merged = tuple(sorted(cluster + partner))
        if not coupled:
            uncoupled_merges.append(merged)
        clusters = [c for c in clusters if c not in (cluster, partner)]
        clusters.append(merged)

    clusters = sorted(clusters)
    uncoupled = [
        index
        for index, cluster in enumerate(clusters)
        if any(set(merged) <= set(cluster) for merged in uncoupled_merges)
    ]
    if uncoupled:
        warnings.warn(
            f"Fragments {uncoupled} join orbitals with no coupling above "
            "coupling_threshold between them, to give each fragment occupied "
            "and virtual orbitals, so their split does not follow the "
            "molecule's couplings. Lower coupling_threshold, or set "
            "fragment_atoms or active_spaces.",
            UserWarning,
            stacklevel=2,
        )
    return clusters


def _localized_active_space_integrals(
    mol,
    mo_coeff: np.ndarray,
    occupied_indices,
    n_occupied: int,
    localized: np.ndarray,
    ao_eri: np.ndarray | None = None,
    h_ao: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build one- and two-body integrals for the localised active space.

    The one-body term carries the frozen-core mean-field potential
    ``2 * J_core - K_core`` contributed by any occupied orbital not selected
    as active; the two-body term is the bare active-space integral. Both come
    from :func:`~divi.qprog.workflows._lassqd._integrals.transform_integrals`.

    Args:
        mol: A PySCF ``gto.Mole``.
        mo_coeff: ``(nao, n_orb)`` MO coefficients.
        occupied_indices: Column indices of ``mo_coeff`` selected as the
            active occupied block.
        n_occupied: Number of occupied spatial orbitals in ``mo_coeff``.
        localized: ``(nao, n_act)`` localised active-space AO-basis
            coefficient matrix.
        ao_eri: AO-basis electron-repulsion integral; ``None`` builds it.
        h_ao: AO-basis core Hamiltonian; ``None`` uses
            :func:`~divi.qprog.workflows._lassqd._integrals.cached_h_ao`.

    Returns:
        ``(one_body, two_body)`` in the localised active basis.
    """
    core_indices = sorted(set(range(n_occupied)) - set(occupied_indices))
    integrals = transform_integrals(
        mol,
        np.hstack([mo_coeff[:, core_indices], localized]),
        len(core_indices),
        localized.shape[1],
        ao_eri,
        h_ao,
    )
    return integrals.h_eff, integrals.g_act


def auto_fragment_specs(
    mol,
    mo_coeff: np.ndarray,
    n_occupied: int,
    rng: np.random.Generator,
    *,
    n_active_orbitals: int | None = None,
    max_orbitals_per_fragment: int | None = None,
    coupling_threshold: float | None = None,
    active_orbitals: Sequence[int] | None = None,
    fragment_atoms: Sequence[Sequence[int]] | None = None,
    local_spins: Sequence[int] | None = None,
    ao_eri: np.ndarray | None = None,
    h_ao: np.ndarray | None = None,
    spin: int = 0,
) -> tuple[list[FragmentSpec], np.ndarray, tuple[int, ...]]:
    """Automatically fragment an active space from orbital coupling.

    Selects active orbitals around the HOMO-LUMO gap
    (:func:`select_frontier_orbitals`), localises the occupied and virtual
    blocks independently (:func:`localize_blocks`), builds the localised
    coupling graph (:func:`build_coupling_graph`) from integrals that include
    the frozen-core mean-field potential of any occupied orbital left out of
    the active space, and greedily merges orbitals into fragments
    (:func:`merge_clusters`).

    Args:
        mol: A PySCF ``gto.Mole``.
        mo_coeff: ``(nao, n_orb)`` MO coefficients.
        n_occupied: Number of occupied spatial orbitals, doubly and singly
            occupied together.
        rng: Random number generator, forwarded to :func:`localize_blocks`.
            Required rather than created internally, so that fragmentation
            stays reproducible under a caller-supplied seed.
        n_active_orbitals: See :func:`select_frontier_orbitals`. Mutually
            exclusive with ``active_orbitals``.
        max_orbitals_per_fragment: Maximum spatial orbitals per fragment;
            ``None`` means ``DEFAULT_MAX_ORBITALS_PER_FRAGMENT``. Ignored when
            ``fragment_atoms`` is given.
        coupling_threshold: See :func:`build_coupling_graph`; ``None`` means
            ``DEFAULT_COUPLING_THRESHOLD``. Ignored when ``fragment_atoms`` is
            given.
        active_orbitals: Explicit MO column indices forming the active space,
            replacing the frontier selection. Use it when the active space is
            defined by orbital character rather than energy -- a transition
            metal's ``d`` manifold can sit well below the HOMO and its virtual
            partners well above the LUMO, where no frontier count reaches them.
        fragment_atoms: One sequence of atom indices per fragment. Assigns each
            localised active orbital to the fragment owning the atom it sits on
            (:func:`assign_orbitals_to_atoms`), replacing the coupling-graph
            clustering. This is how a localised active space is normally
            specified -- one fragment per metal centre.
        local_spins: Per-fragment ``2S``, in the order ``fragment_atoms`` names
            them, overriding the default of one unpaired spin per singly
            occupied orbital (``S = 0`` for a closed-shell fragment). The
            fragment's electron count is fixed by its occupied orbitals, so this
            sets only the spin: ``n_alpha - n_beta = 2S``. Requires
            ``fragment_atoms``, whose order is the caller's -- coupling-graph
            fragment order depends on ``max_orbitals_per_fragment``,
            ``coupling_threshold`` and the localisation RNG, so a positional
            spin list would not name a stable fragment there.
        ao_eri: AO-basis electron-repulsion integral for the coupling-graph
            integrals; ``None`` builds it from ``mol``. Ignored when
            ``fragment_atoms`` is given.
        h_ao: AO-basis core Hamiltonian for the coupling-graph integrals;
            ``None`` builds it from ``mol``. Ignored when ``fragment_atoms``
            is given.
        spin: The molecule's non-negative ``2S``; its ``spin`` highest
            occupied orbitals are singly occupied, localised apart from the
            doubly occupied ones.

    The selector and ``local_spins`` combinations are those
    ``FragmentationConfig`` accepts; they are not re-validated here.

    Returns:
        ``(specs, localized, active_positions)``: one ``FragmentSpec`` per
        fragment with ``orbitals`` already in ``mo_coeff`` register indices, the
        localised active-space AO-basis coefficients (occupied columns first),
        and the register positions those columns belong to.

    Raises:
        ImportError: If the ``chem`` extra is not installed.
        ValueError: If a requested ``2S`` has the wrong parity for a fragment's
            electron count, exceeds it, or leaves it no partly filled spin
            channel. Also propagated from
            :func:`select_frontier_orbitals`, :func:`split_active_orbitals`,
            :func:`assign_orbitals_to_atoms`, or :func:`merge_clusters`.
    """
    if active_orbitals is not None:
        occupied_indices, virtual_indices = split_active_orbitals(
            active_orbitals, n_occupied, mo_coeff.shape[1], spin=spin
        )
    elif n_active_orbitals is not None:
        occupied_indices, virtual_indices = select_frontier_orbitals(
            mo_coeff.shape[1], n_occupied, n_active_orbitals, spin=spin
        )
    else:
        raise ValueError("Pass either active_orbitals or n_active_orbitals.")
    # Singly occupied orbitals localise apart, so each keeps its occupation.
    doubly = tuple(o for o in occupied_indices if o < n_occupied - abs(spin))
    singly = tuple(o for o in occupied_indices if o >= n_occupied - abs(spin))
    localized_doubly, localized_virt = localize_blocks(
        mol, mo_coeff, doubly, virtual_indices, rng
    )
    localized_singly = _localize_block(mol, mo_coeff[:, list(singly)], rng)
    localized = np.hstack([localized_doubly, localized_singly, localized_virt])
    occupation = [2] * len(doubly) + [1] * len(singly) + [0] * localized_virt.shape[1]
    is_occupied = [count > 0 for count in occupation]
    occupied_indices = doubly + singly

    if fragment_atoms is not None:
        clusters = assign_orbitals_to_atoms(mol, localized, fragment_atoms)
    else:
        one_body, two_body = _localized_active_space_integrals(
            mol, mo_coeff, occupied_indices, n_occupied, localized, ao_eri, h_ao
        )
        if coupling_threshold is None:
            coupling_threshold = DEFAULT_COUPLING_THRESHOLD
        if max_orbitals_per_fragment is None:
            max_orbitals_per_fragment = DEFAULT_MAX_ORBITALS_PER_FRAGMENT
        graph = build_coupling_graph(
            one_body, two_body, coupling_threshold=coupling_threshold
        )
        clusters = merge_clusters(graph, is_occupied, max_orbitals_per_fragment)

    active_positions = tuple(occupied_indices) + tuple(virtual_indices)
    specs = []
    for index, cluster in enumerate(clusters):
        n_electrons = sum(occupation[orbital] for orbital in cluster)
        if local_spins is None:
            two_s = sum(occupation[orbital] == 1 for orbital in cluster)
            source = f"Fragment {index}'s singly occupied orbitals give it"
        else:
            two_s = int(local_spins[index])
            source = f"local_spins[{index}] asks for"
        if (n_electrons + two_s) % 2:
            raise ValueError(
                f"{source} 2S={two_s} on a fragment holding {n_electrons} "
                "electrons; 2S must have the parity of the electron count."
            )
        if abs(two_s) > n_electrons:
            raise ValueError(
                f"{source} 2S={two_s} on a fragment holding {n_electrons} "
                "electrons, which cannot supply that many unpaired spins."
            )
        n_alpha = (n_electrons + two_s) // 2
        n_beta = n_electrons - n_alpha
        if two_s and not (
            n_alpha <= len(cluster)
            and any(0 < count < len(cluster) for count in (n_alpha, n_beta))
        ):
            raise ValueError(
                f"{source} 2S={two_s}, giving fragment {index} n_alpha={n_alpha} "
                f"and n_beta={n_beta} over {len(cluster)} orbitals, which leaves "
                "no spin channel partly filled to excite."
            )
        specs.append(
            FragmentSpec(
                orbitals=tuple(active_positions[o] for o in cluster),
                n_alpha=n_alpha,
                n_beta=n_beta,
            )
        )
    return specs, localized, active_positions
