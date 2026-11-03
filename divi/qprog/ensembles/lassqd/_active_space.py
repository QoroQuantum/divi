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

from ._integrals import AOIntegrals, transform_integrals
from ._state import FragmentSpec

DEFAULT_MAX_ORBITALS_PER_FRAGMENT = 4
DEFAULT_COUPLING_THRESHOLD = 1e-3

#: Jacobi stability checks, each followed by a re-localisation from the
#: rotated orbitals, before a Pipek-Mezey block is kept as it is.
_MAX_STABILITY_SWEEPS = 10


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
    """Sort a localised block's columns by the index of the atom carrying their
    largest Mulliken population, then by population-weighted centroid
    coordinate; columns tied on both keep their input order.

    Pipek-Mezey can return the same solution in different column orders, which
    would change the fragment partition.
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
            f"active_orbitals leaves out singly occupied orbitals {missing}, "
            "which must all be active."
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
                    f"Atom {atom} appears in fragments {owner[atom]} and "
                    f"{index}; fragments must be disjoint."
                )
            owner[atom] = index
    return owner


def assign_orbitals_to_atoms(
    mol, localized: np.ndarray, fragment_atoms: Sequence[Sequence[int]]
) -> list[tuple[int, ...]]:
    """Group localised columns into fragments by the atom they sit on.

    Each column goes to the fragment owning the atom carrying its largest
    Mulliken population.

    Args:
        mol: A PySCF ``gto.Mole``.
        localized: ``(nao, n_act)`` localised active-space coefficients.
        fragment_atoms: One sequence of atom indices per fragment.

    Returns:
        One tuple of ``localized`` column indices per fragment, in the order
        ``fragment_atoms`` was given.

    Raises:
        ValueError: If any column's dominant atom is not in ``fragment_atoms``,
            or if a fragment ends up with no orbitals. Atom range and
            disjointness errors come from :func:`validate_fragment_atoms`.
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
                "Add it to a fragment, or choose an active space localised on "
                "the listed atoms."
            )
        clusters[owner[atom]].append(column)

    for index, cluster in enumerate(clusters):
        if not cluster:
            raise ValueError(
                f"fragment_atoms[{index}] ({list(fragment_atoms[index])}) got no "
                "active orbitals: the localised active space has none on those "
                "atoms."
            )
    return [tuple(cluster) for cluster in clusters]


def localize_blocks(
    mol,
    mo_coeff: np.ndarray,
    occupied_indices,
    virtual_indices,
) -> tuple[np.ndarray, np.ndarray]:
    """Pipek-Mezey localise the occupied and virtual blocks independently, so
    every localised orbital is unambiguously occupied or virtual.

    Each block is localised from PySCF's atomic guess, then checked with
    PySCF's Jacobi-sweep stability analysis; while a pairwise rotation still
    raises the Pipek-Mezey cost, the block is re-localised from the rotated
    orbitals. This can escape a stationary point of a symmetric molecule, and
    is deterministic.

    Args:
        mol: A PySCF ``gto.Mole``.
        mo_coeff: ``(nao, n_orb)`` MO coefficients.
        occupied_indices: Column indices of ``mo_coeff`` forming the occupied
            block.
        virtual_indices: Column indices of ``mo_coeff`` forming the virtual
            block.

    Returns:
        ``(localized_occupied, localized_virtual)`` AO-basis coefficient
        blocks, columns ordered as in :func:`_canonicalize_columns`. A
        single-orbital block is returned unchanged.
    """
    return (
        _localize_block(mol, mo_coeff[:, list(occupied_indices)]),
        _localize_block(mol, mo_coeff[:, list(virtual_indices)]),
    )


def _localize_block(mol, block: np.ndarray) -> np.ndarray:
    """One block of :func:`localize_blocks`; a block of at most one orbital is
    returned unchanged."""
    if block.shape[1] <= 1:
        return block.copy()

    localizer = lo.PipekMezey(mol, block)
    localized = localizer.kernel()
    for _ in range(_MAX_STABILITY_SWEEPS):
        localized, stable = localizer.stability_jacobi(return_status=True)
        if stable:
            break
        localized = localizer.kernel(localized)
    return _canonicalize_columns(mol, localized)


def build_coupling_graph(
    one_body: np.ndarray,
    two_body: np.ndarray,
    *,
    coupling_threshold: float = DEFAULT_COUPLING_THRESHOLD,
) -> nx.Graph:
    """Build the orbital coupling graph in the localised basis.

    Edge weight is ``|h_pq| + |(pq|qp)|``, the one-electron coupling plus the
    exchange integral. Edges with weight below ``coupling_threshold * max(w)``
    are dropped, so the threshold is relative. The default of ``1e-3`` can
    prune few edges for small localised active spaces; raise it to force
    pruning.

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
    graph: nx.Graph, left: Sequence[int], right: Sequence[int]
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
    pair with the largest total surviving edge weight between them that stays
    within ``max_orbitals_per_fragment``; the earliest pair in ascending
    index order wins ties. A fix-up pass then merges every cluster holding
    only occupied or only virtual orbitals into a neighbour, preferring a
    coupled neighbour and, among equals, the smallest. A zero-coupling
    neighbour is used only when no coupled one fits; the caller is then
    warned.

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
    def coupling(left: Sequence[int], right: Sequence[int]) -> float:
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

    def is_mixed(cluster: Sequence[int]) -> bool:
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
                "neighbour without exceeding "
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
            "coupling_threshold, to give each fragment occupied and virtual "
            "orbitals, so they do not follow the molecule's couplings. Lower "
            "coupling_threshold, or set fragment_atoms or active_spaces.",
            UserWarning,
            stacklevel=2,
        )
    return clusters


def _localized_active_space_integrals(
    ao: AOIntegrals,
    mo_coeff: np.ndarray,
    occupied_indices,
    n_occupied: int,
    localized: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build one- and two-body integrals for the localised active space.

    The one-body term carries the frozen-core potential ``2 * J_core - K_core``
    of every occupied orbital not selected as active; the two-body term is the
    bare active-space integral.

    Args:
        ao: The molecule and its AO integrals.
        mo_coeff: ``(nao, n_orb)`` MO coefficients.
        occupied_indices: Column indices of ``mo_coeff`` selected as the
            active occupied block.
        n_occupied: Number of occupied spatial orbitals in ``mo_coeff``.
        localized: ``(nao, n_act)`` localised active-space AO-basis
            coefficient matrix.

    Returns:
        ``(one_body, two_body)`` in the localised active basis.
    """
    core_indices = sorted(set(range(n_occupied)) - set(occupied_indices))
    integrals = transform_integrals(
        ao,
        np.hstack([mo_coeff[:, core_indices], localized]),
        len(core_indices),
        localized.shape[1],
    )
    return integrals.h_eff, integrals.g_act


def auto_fragment_specs(
    ao: AOIntegrals,
    mo_coeff: np.ndarray,
    n_occupied: int,
    *,
    n_active_orbitals: int | None = None,
    max_orbitals_per_fragment: int | None = None,
    coupling_threshold: float | None = None,
    active_orbitals: Sequence[int] | None = None,
    fragment_atoms: Sequence[Sequence[int]] | None = None,
    local_spins: Sequence[int] | None = None,
    spin: int = 0,
) -> tuple[list[FragmentSpec], np.ndarray, tuple[int, ...]]:
    """Fragment an active space from orbital coupling.

    Selects active orbitals around the HOMO-LUMO gap
    (:func:`select_frontier_orbitals`), localises the occupied and virtual
    blocks independently (:func:`localize_blocks`), builds the coupling graph
    (:func:`build_coupling_graph`) from integrals that include the frozen-core
    potential of any occupied orbital left out of the active space, and
    greedily merges orbitals into fragments (:func:`merge_clusters`).

    The selector and ``local_spins`` combinations are those
    ``FragmentationConfig`` accepts; they are not re-validated here.

    Args:
        ao: The molecule and its AO integrals.
        mo_coeff: ``(nao, n_orb)`` MO coefficients.
        n_occupied: Number of occupied spatial orbitals, doubly and singly
            occupied together.
        n_active_orbitals: See :func:`select_frontier_orbitals`. Mutually
            exclusive with ``active_orbitals``.
        max_orbitals_per_fragment: See :func:`merge_clusters`; ``None`` means
            ``DEFAULT_MAX_ORBITALS_PER_FRAGMENT``. Ignored when
            ``fragment_atoms`` is given.
        coupling_threshold: See :func:`build_coupling_graph`; ``None`` means
            ``DEFAULT_COUPLING_THRESHOLD``. Ignored when ``fragment_atoms`` is
            given.
        active_orbitals: Explicit MO column indices forming the active space,
            replacing the frontier selection.
        fragment_atoms: One sequence of atom indices per fragment. Assigns each
            localised active orbital to the fragment owning the atom it sits on
            (:func:`assign_orbitals_to_atoms`), replacing the coupling-graph
            clustering.
        local_spins: Per-fragment ``2S``, in the order of ``fragment_atoms``,
            overriding the default of one unpaired spin per singly occupied
            orbital. It sets only the spin (``n_alpha - n_beta = 2S``); the
            electron count is fixed by the occupied orbitals. Requires
            ``fragment_atoms``.
        spin: The molecule's non-negative ``2S``; its ``spin`` highest
            occupied orbitals are singly occupied, localised apart from the
            doubly occupied ones.

    Returns:
        ``(specs, localized, active_positions)``: one ``FragmentSpec`` per
        fragment with ``orbitals`` in ``mo_coeff`` register indices, the
        localised active-space AO-basis coefficients (occupied columns first),
        and the register positions of those columns.

    Raises:
        ImportError: If the ``chem`` extra is not installed.
        ValueError: If a requested ``2S`` has the wrong parity for a fragment's
            electron count, exceeds it, or leaves it no partly filled spin
            channel. Also raised by :func:`select_frontier_orbitals`,
            :func:`split_active_orbitals`, :func:`assign_orbitals_to_atoms`
            and :func:`merge_clusters`.
    """
    mol = ao.mol
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
        mol, mo_coeff, doubly, virtual_indices
    )
    localized_singly = _localize_block(mol, mo_coeff[:, list(singly)])
    localized = np.hstack([localized_doubly, localized_singly, localized_virt])
    occupation = [2] * len(doubly) + [1] * len(singly) + [0] * localized_virt.shape[1]
    is_occupied = [count > 0 for count in occupation]
    occupied_indices = doubly + singly

    if fragment_atoms is not None:
        clusters = assign_orbitals_to_atoms(mol, localized, fragment_atoms)
    else:
        one_body, two_body = _localized_active_space_integrals(
            ao, mo_coeff, occupied_indices, n_occupied, localized
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
