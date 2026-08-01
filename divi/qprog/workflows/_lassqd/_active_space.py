# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Automatic active-space orbital selection and localization."""

import warnings
from collections.abc import Sequence

import networkx as nx
import numpy as np

from ._state import FragmentSpec


def select_frontier_orbitals(
    mo_energy: np.ndarray,
    n_occupied: int,
    *,
    n_active_orbitals: int | None = None,
    energy_window: float | None = None,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Choose active orbitals around the HOMO-LUMO gap.

    Args:
        mo_energy: Molecular-orbital energies in ascending order.
        n_occupied: Number of occupied spatial orbitals.
        n_active_orbitals: Take the ``ceil(k / 2)`` highest occupied and
            ``floor(k / 2)`` lowest virtual orbitals, clamped at the register
            edges. Mutually exclusive with ``energy_window``.
        energy_window: Take occupied orbitals with ``eps >= eps_HOMO - w`` and
            virtual orbitals with ``eps <= eps_LUMO + w``, in Hartree. Must be
            non-negative. Mutually exclusive with ``n_active_orbitals``.

    Returns:
        ``(occupied_indices, virtual_indices)``, each ascending.

    Raises:
        ValueError: If not exactly one of ``n_active_orbitals`` and
            ``energy_window`` is given, if ``n_active_orbitals`` is not
            positive, if ``energy_window`` is negative, or if the molecule's
            orbital register has no occupied or no virtual orbitals to
            select from.
    """
    if (n_active_orbitals is None) == (energy_window is None):
        raise ValueError("Pass exactly one of n_active_orbitals or energy_window.")

    n_total = len(mo_energy)
    if n_active_orbitals is not None:
        if n_active_orbitals <= 0:
            raise ValueError(
                f"n_active_orbitals must be positive; got {n_active_orbitals}."
            )
        n_occ_take = min(-(-n_active_orbitals // 2), n_occupied)
        n_virt_take = min(n_active_orbitals // 2, n_total - n_occupied)
        occupied = tuple(range(n_occupied - n_occ_take, n_occupied))
        virtual = tuple(range(n_occupied, n_occupied + n_virt_take))
    else:
        assert energy_window is not None
        if energy_window < 0:
            raise ValueError(
                f"energy_window must be non-negative; got {energy_window}."
            )
        homo = mo_energy[n_occupied - 1]
        lumo = mo_energy[n_occupied]
        occupied = tuple(
            i for i in range(n_occupied) if mo_energy[i] >= homo - energy_window
        )
        virtual = tuple(
            i
            for i in range(n_occupied, n_total)
            if mo_energy[i] <= lumo + energy_window
        )

    if not occupied or not virtual:
        raise ValueError(
            "The active space must contain at least one occupied and one "
            f"virtual orbital; selected {len(occupied)} occupied and "
            f"{len(virtual)} virtual. The molecule's orbital register has "
            "no occupied orbitals, or no virtual orbitals, to select from."
        )
    return occupied, virtual


def _canonicalize_columns(mol, block: np.ndarray) -> np.ndarray:
    """Reorder a localized block's columns into a deterministic order.

    Columns are sorted by the index of the atom carrying their largest
    Mulliken population, tiebroken by the population-weighted centroid
    coordinate. Pipek-Mezey's cost function is invariant to the order of the
    localized columns it returns, so two runs that converge to the same
    physical solution (same cost) can still return it in a different column
    order; without canonicalizing that order, the fragment partition built
    from those columns would not be reproducible under a fixed seed even
    though the physical solution, and its energy, are unchanged.
    """
    n_col = block.shape[1]
    if n_col <= 1:
        return block

    overlap = mol.intor("int1e_ovlp")
    ao_slices = mol.aoslice_by_atom()
    per_ao_population = block * (overlap @ block)

    atom_population = np.array(
        [per_ao_population[start:stop].sum(axis=0) for _, _, start, stop in ao_slices]
    )
    dominant_atom = np.argmax(atom_population, axis=0)

    coords = mol.atom_coords()
    total_population = atom_population.sum(axis=0)
    centroid = (atom_population.T @ coords) / total_population[:, None]

    order = sorted(
        range(n_col),
        key=lambda col: (int(dominant_atom[col]), tuple(centroid[col])),
    )
    return block[:, order]


def localize_blocks(
    mol,
    mo_coeff: np.ndarray,
    occupied_indices,
    virtual_indices,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Pipek-Mezey localize the occupied and virtual blocks independently.

    Localizing the two blocks separately keeps each localized orbital
    unambiguously occupied or virtual, which makes fragment electron counts
    exact rather than inferred from occupation thresholds. Localizing them
    jointly would mix occupied and virtual character across the resulting
    orbitals and destroy that property.

    For a symmetric molecule the canonical MOs can themselves be a stationary
    point of the Pipek-Mezey cost function, so localizing from the canonical
    (identity-rotation) starting point can converge back to the delocalized
    input without moving at all — and a single random restart can also land
    back on that same stationary point. To make escaping likely, each block
    is localized from several randomly rotated starting points (each a fresh
    random orthogonal rotation of the block's columns, drawn from ``rng``) in
    addition to the canonical start, and whichever run reaches the highest
    Pipek-Mezey cost is kept. This never does worse than localizing from the
    canonical start alone, but escaping a symmetric stationary point is not
    guaranteed for every input.

    Once a restart's cost exceeds the canonical cost by more than a small
    relative tolerance, the stationary point has been escaped and further
    restarts stop early. A restart that lands back on the canonical cost (or
    close to it) keeps trying, up to the restart cap, which is exactly the
    case the restarts exist for.

    Args:
        mol: A PySCF ``gto.Mole``.
        mo_coeff: ``(nao, n_orb)`` MO coefficients.
        occupied_indices: Column indices of ``mo_coeff`` forming the occupied
            block.
        virtual_indices: Column indices of ``mo_coeff`` forming the virtual
            block.
        rng: Random number generator used to perturb the localization's
            starting point. Required so that fragmentation stays
            reproducible under a caller-supplied seed rather than drawing
            from unseeded global randomness.

    Returns:
        ``(localized_occupied, localized_virtual)`` AO-basis coefficient
        blocks. A block with a single orbital is returned unchanged, since
        Pipek-Mezey localization on one orbital is a no-op. Each returned
        block's columns are canonically ordered (see
        :func:`_canonicalize_columns`): by the index of the atom carrying
        their largest Mulliken population, tiebroken by the
        population-weighted centroid coordinate. This makes the column order
        reproducible under a fixed seed even when the optimizer reaches the
        same physical solution via a different column order on different
        runs.

    Raises:
        ImportError: If the ``chem`` extra is not installed.
    """
    try:
        # pyrefly: ignore[missing-import]  # optional ``chem`` extra
        from pyscf import lo
    except ImportError as exc:
        raise ImportError(
            "LASSQD active-space localization requires the 'chem' extra; "
            "install it with `pip install qoro-divi[chem]`."
        ) from exc

    n_restarts = 8
    localized = []
    for indices in (list(occupied_indices), list(virtual_indices)):
        block = mo_coeff[:, indices]
        n = block.shape[1]
        if n == 1:
            localized.append(block.copy())
            continue

        canonical_start = lo.PipekMezey(mol, block)
        best_result = canonical_start.kernel()
        best_cost = canonical_cost = canonical_start.cost_function()

        escape_tolerance = 1e-6
        for _ in range(n_restarts):
            orthogonal, _ = np.linalg.qr(rng.standard_normal((n, n)))
            perturbed_start = lo.PipekMezey(mol, block @ orthogonal)
            perturbed_result = perturbed_start.kernel()
            perturbed_cost = perturbed_start.cost_function()
            if perturbed_cost > best_cost:
                best_result, best_cost = perturbed_result, perturbed_cost
            if best_cost > canonical_cost * (1.0 + escape_tolerance):
                break

        localized.append(_canonicalize_columns(mol, best_result))
    return localized[0], localized[1]


def build_coupling_graph(
    one_body: np.ndarray,
    two_body: np.ndarray,
    *,
    coupling_threshold: float = 1e-3,
) -> nx.Graph:
    """Build the orbital coupling graph in the localized basis.

    Edge weight is ``|h_pq| + |(pq|qp)|`` — the one-electron coupling plus
    the exchange integral. ``coupling_threshold`` is relative: edges with
    weight below ``coupling_threshold * max(w)`` are dropped, which keeps the
    default scale-free across molecules and basis sets. In practice, for
    genuinely localized orbitals the weakest surviving cross-fragment
    coupling is typically tens of times larger than ``coupling_threshold``'s
    default of ``1e-3`` would require to prune, so the default rarely drops
    any edge for small active spaces; a caller who wants pruning to actually
    happen should raise this value substantially rather than relying on the
    default.

    Args:
        one_body: ``(n_orb, n_orb)`` one-electron integrals in the localized
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

    weights: dict[tuple[int, int], float] = {}
    for p in range(n_orb):
        for q in range(p + 1, n_orb):
            weights[(p, q)] = abs(one_body[p, q]) + abs(two_body[p, q, q, p])

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
    occupied or only virtual orbitals into a neighbor, since such a fragment
    captures no correlation between occupied and virtual character. A
    neighbor with positive coupling is preferred; a neighbor with zero
    coupling is used only when no coupled neighbor fits within
    ``max_orbitals_per_fragment``, since an unmixed fragment is worse than a
    zero-coupling merge. Ties in this pass favor the smallest partner, since
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
        ValueError: If ``max_orbitals_per_fragment`` is below 1, or if an
            occupied-only or virtual-only cluster cannot be merged with any
            neighbor without exceeding ``max_orbitals_per_fragment``.
    """
    if max_orbitals_per_fragment < 1:
        raise ValueError(
            "max_orbitals_per_fragment must be at least 1; got "
            f"{max_orbitals_per_fragment}."
        )

    clusters = [(node,) for node in sorted(graph.nodes)]

    def best_merge(candidates):
        best, best_weight = None, 0.0
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                if len(candidates[i]) + len(candidates[j]) > max_orbitals_per_fragment:
                    continue
                weight = _inter_cluster_weight(graph, candidates[i], candidates[j])
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

        coupled = [
            other
            for other in within_limit
            if _inter_cluster_weight(graph, cluster, other) > 0.0
        ]
        # A zero-coupling partner is used only as a fallback.
        candidates = coupled if coupled else within_limit
        partner = max(
            candidates,
            key=lambda other: (
                _inter_cluster_weight(graph, cluster, other),
                -len(other),
            ),
        )
        if not coupled:
            warnings.warn(
                f"Fragment {cluster} shares no positive coupling with any "
                f"candidate and is being absorbed into {partner} regardless. "
                "It will contribute no correlation to that fragment; "
                "consider lowering coupling_threshold or widening the "
                "active space.",
                UserWarning,
                stacklevel=2,
            )
        clusters = [c for c in clusters if c not in (cluster, partner)]
        clusters.append(tuple(sorted(cluster + partner)))

    return sorted(clusters)


def _localized_active_space_integrals(
    mol,
    mo_coeff: np.ndarray,
    occupied_indices,
    n_occupied: int,
    localized: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build one- and two-body integrals for the localized active space.

    The one-body term carries the frozen-core mean-field potential
    ``2 * J_core - K_core`` contributed by any occupied orbital not selected
    as active, matching the convention used by
    :func:`divi.qprog.workflows._lassqd._integrals.transform_integrals`. The
    two-body term is the bare active-space integral, local to the active
    orbitals.

    Args:
        mol: A PySCF ``gto.Mole``.
        mo_coeff: ``(nao, n_orb)`` MO coefficients.
        occupied_indices: Column indices of ``mo_coeff`` selected as the
            active occupied block.
        n_occupied: Number of occupied spatial orbitals in ``mo_coeff``.
        localized: ``(nao, n_act)`` localized active-space AO-basis
            coefficient matrix.

    Returns:
        ``(one_body, two_body)`` in the localized active basis.

    Raises:
        ImportError: If the ``chem`` extra is not installed.
    """
    try:
        # pyrefly: ignore[missing-import]  # optional ``chem`` extra
        from pyscf import ao2mo, scf
    except ImportError as exc:
        raise ImportError(
            "auto_fragment_specs requires the 'chem' extra; "
            "install it with `pip install qoro-divi[chem]`."
        ) from exc

    n_act = localized.shape[1]
    h_ao = mol.intor("int1e_kin") + mol.intor("int1e_nuc")

    core_indices = sorted(set(range(n_occupied)) - set(occupied_indices))
    if core_indices:
        c_core = mo_coeff[:, core_indices]
        dm_core = c_core @ c_core.T
        j_core_ao, k_core_ao = scf.hf.get_jk(mol, dm_core)
        # pyrefly: ignore[unsupported-operation]  # get_jk is untyped in pyscf
        h_ao = h_ao + 2.0 * j_core_ao - k_core_ao

    one_body = localized.T @ h_ao @ localized
    two_body = ao2mo.restore(1, ao2mo.kernel(mol, localized), n_act)
    return one_body, two_body


def auto_fragment_specs(
    mol,
    mo_coeff: np.ndarray,
    mo_energy: np.ndarray,
    n_occupied: int,
    rng: np.random.Generator,
    *,
    n_active_orbitals: int | None = None,
    energy_window: float | None = None,
    max_orbitals_per_fragment: int = 4,
    coupling_threshold: float = 1e-3,
) -> tuple[list[FragmentSpec], np.ndarray]:
    """Automatically fragment an active space from orbital coupling.

    Selects active orbitals around the HOMO-LUMO gap
    (:func:`select_frontier_orbitals`), localizes the occupied and virtual
    blocks independently (:func:`localize_blocks`), builds the localized
    coupling graph (:func:`build_coupling_graph`) from integrals that include
    the frozen-core mean-field potential of any occupied orbital left out of
    the active space, and greedily merges orbitals into fragments
    (:func:`merge_clusters`).

    Args:
        mol: A PySCF ``gto.Mole``.
        mo_coeff: ``(nao, n_orb)`` MO coefficients.
        mo_energy: Molecular-orbital energies in ascending order.
        n_occupied: Number of occupied spatial orbitals.
        rng: Random number generator, forwarded to :func:`localize_blocks`.
            Required rather than created internally, so that fragmentation
            stays reproducible under a caller-supplied seed.
        n_active_orbitals: See :func:`select_frontier_orbitals`.
        energy_window: See :func:`select_frontier_orbitals`.
        max_orbitals_per_fragment: Maximum spatial orbitals per fragment.
        coupling_threshold: See :func:`build_coupling_graph`.

    Returns:
        ``(specs, localized)``: one ``FragmentSpec`` per fragment, with
        ``orbitals`` indexing the columns of ``localized``, and the
        localized active-space AO-basis coefficient matrix, occupied columns
        first.

    Raises:
        ImportError: If the ``chem`` extra is not installed.
        ValueError: Propagated from :func:`select_frontier_orbitals` or
            :func:`merge_clusters`.
    """
    occupied_indices, virtual_indices = select_frontier_orbitals(
        mo_energy,
        n_occupied,
        n_active_orbitals=n_active_orbitals,
        energy_window=energy_window,
    )
    localized_occ, localized_virt = localize_blocks(
        mol, mo_coeff, occupied_indices, virtual_indices, rng
    )
    localized = np.hstack([localized_occ, localized_virt])
    is_occupied = [True] * localized_occ.shape[1] + [False] * localized_virt.shape[1]

    one_body, two_body = _localized_active_space_integrals(
        mol, mo_coeff, occupied_indices, n_occupied, localized
    )

    graph = build_coupling_graph(
        one_body, two_body, coupling_threshold=coupling_threshold
    )
    clusters = merge_clusters(graph, is_occupied, max_orbitals_per_fragment)

    specs = [
        FragmentSpec(
            orbitals=cluster,
            n_alpha=sum(is_occupied[orbital] for orbital in cluster),
            n_beta=sum(is_occupied[orbital] for orbital in cluster),
        )
        for cluster in clusters
    ]
    return specs, localized
