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
    n_orbitals_total: int, n_occupied: int, n_active_orbitals: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Choose active orbitals around the HOMO-LUMO gap.

    Args:
        n_orbitals_total: Size of the MO register.
        n_occupied: Number of occupied spatial orbitals.
        n_active_orbitals: Take the ``ceil(k / 2)`` highest occupied and
            ``floor(k / 2)`` lowest virtual orbitals, clamped at the register
            edges.

    Returns:
        ``(occupied_indices, virtual_indices)``, each ascending.

    Raises:
        ValueError: If ``n_active_orbitals`` is not positive, or if the
            selection yields no occupied or no virtual orbital.
    """
    if n_active_orbitals <= 0:
        raise ValueError(
            f"n_active_orbitals must be positive; got {n_active_orbitals}."
        )
    n_occ_take = min(-(-n_active_orbitals // 2), n_occupied)
    n_virt_take = min(n_active_orbitals // 2, n_orbitals_total - n_occupied)
    occupied = tuple(range(n_occupied - n_occ_take, n_occupied))
    virtual = tuple(range(n_occupied, n_occupied + n_virt_take))

    if not occupied or not virtual:
        raise ValueError(
            "The active space must contain at least one occupied and one "
            f"virtual orbital; got {len(occupied)} occupied and {len(virtual)} "
            f"virtual from n_active_orbitals={n_active_orbitals} with "
            f"n_occupied={n_occupied} of {n_orbitals_total} orbitals."
        )
    return occupied, virtual


def _atom_populations(mol, block: np.ndarray) -> np.ndarray:
    """``(n_atom, n_col)`` Mulliken population of each column on each atom."""
    overlap = mol.intor("int1e_ovlp")
    per_ao_population = block * (overlap @ block)
    return np.array(
        [
            per_ao_population[start:stop].sum(axis=0)
            for _, _, start, stop in mol.aoslice_by_atom()
        ]
    )


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

    atom_population = _atom_populations(mol, block)
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
    active_orbitals: Sequence[int], n_occupied: int, n_orbitals_total: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Split explicit active MO indices into occupied and virtual, ascending.

    Args:
        active_orbitals: MO column indices forming the active space.
        n_occupied: Number of occupied spatial orbitals; an index below this is
            occupied.
        n_orbitals_total: Size of the MO register.

    Raises:
        ValueError: If ``active_orbitals`` is empty, contains duplicates or
            out-of-range indices, or has no occupied or no virtual orbital.
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
    """Group localized columns into fragments by the atom they sit on.

    Each column is assigned to whichever fragment owns the atom carrying its
    largest Mulliken population. This is the fragmentation a localized active
    space is usually defined by -- "these orbitals belong to this metal centre"
    -- rather than one inferred from orbital coupling.

    Args:
        mol: A PySCF ``gto.Mole``.
        localized: ``(nao, n_act)`` localized active-space coefficients.
        fragment_atoms: One sequence of atom indices per fragment.

    Returns:
        One tuple of ``localized`` column indices per fragment, in the order
        ``fragment_atoms`` was given.

    Raises:
        ValueError: If any column's dominant atom belongs to no fragment, or if
            a fragment ends up with no orbitals. Atom range and disjointness are
            propagated from :func:`validate_fragment_atoms`.
    """
    owner = validate_fragment_atoms(fragment_atoms, mol.natm)
    dominant_atom = np.argmax(_atom_populations(mol, localized), axis=0)
    clusters: list[list[int]] = [[] for _ in fragment_atoms]
    for column, atom in enumerate(dominant_atom):
        atom = int(atom)
        if atom not in owner:
            raise ValueError(
                f"Localized active orbital {column} sits mainly on atom {atom} "
                f"({mol.atom_symbol(atom)}), which no fragment claims. Every "
                "active orbital must belong to some fragment; add that atom to "
                "one of them, or choose an active space localized on the atoms "
                "you listed."
            )
        clusters[owner[atom]].append(column)

    for index, cluster in enumerate(clusters):
        if not cluster:
            raise ValueError(
                f"Fragment {index} (atoms {list(fragment_atoms[index])}) got no "
                "active orbitals. The localized active space puts none of its "
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
    n_occupied: int,
    rng: np.random.Generator,
    *,
    n_active_orbitals: int | None = None,
    max_orbitals_per_fragment: int = 4,
    coupling_threshold: float = 1e-3,
    active_orbitals: Sequence[int] | None = None,
    fragment_atoms: Sequence[Sequence[int]] | None = None,
    local_spins: Sequence[int] | None = None,
) -> tuple[list[FragmentSpec], np.ndarray, tuple[int, ...]]:
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
        n_occupied: Number of occupied spatial orbitals.
        rng: Random number generator, forwarded to :func:`localize_blocks`.
            Required rather than created internally, so that fragmentation
            stays reproducible under a caller-supplied seed.
        n_active_orbitals: See :func:`select_frontier_orbitals`. Mutually
            exclusive with ``active_orbitals``.
        max_orbitals_per_fragment: Maximum spatial orbitals per fragment.
            Ignored when ``fragment_atoms`` is given.
        coupling_threshold: See :func:`build_coupling_graph`. Ignored when
            ``fragment_atoms`` is given.
        active_orbitals: Explicit MO column indices forming the active space,
            replacing the frontier selection. Use it when the active space is
            defined by orbital character rather than energy -- a transition
            metal's ``d`` manifold can sit well below the HOMO and its virtual
            partners well above the LUMO, where no frontier count reaches them.
        fragment_atoms: One sequence of atom indices per fragment. Assigns each
            localized active orbital to the fragment owning the atom it sits on
            (:func:`assign_orbitals_to_atoms`), replacing the coupling-graph
            clustering. This is how a localized active space is normally
            specified -- one fragment per metal centre.
        local_spins: Per-fragment ``2S``, in the order ``fragment_atoms`` names
            them, overriding the closed-shell default of ``S = 0``. The
            fragment's electron count is fixed by its occupied orbitals, so this
            sets only the spin: ``n_alpha - n_beta = 2S``. Requires
            ``fragment_atoms``, whose order is the caller's -- coupling-graph
            fragment order depends on ``max_orbitals_per_fragment``,
            ``coupling_threshold`` and the localization RNG, so a positional
            spin list would not name a stable fragment there.

    Returns:
        ``(specs, localized, active_positions)``: one ``FragmentSpec`` per
        fragment with ``orbitals`` already in ``mo_coeff`` register indices, the
        localized active-space AO-basis coefficients (occupied columns first),
        and the register positions those columns belong to.

    Raises:
        ImportError: If the ``chem`` extra is not installed.
        ValueError: If ``local_spins`` is given without ``fragment_atoms``, if
            its length differs from the fragment count, or if a requested ``2S``
            exceeds the fragment's electron count. Also propagated from
            :func:`select_frontier_orbitals`, :func:`split_active_orbitals`,
            :func:`assign_orbitals_to_atoms`, or :func:`merge_clusters`.
    """
    if local_spins is not None and fragment_atoms is None:
        raise ValueError(
            "local_spins requires fragment_atoms: coupling-graph fragment order "
            "depends on max_orbitals_per_fragment, coupling_threshold and the "
            "localization RNG, so a positional spin list would not name a "
            "stable fragment. Name the fragments by atom to assign their spins."
        )

    if active_orbitals is not None:
        occupied_indices, virtual_indices = split_active_orbitals(
            active_orbitals, n_occupied, mo_coeff.shape[1]
        )
    else:
        if n_active_orbitals is None:
            raise ValueError(
                "Pass exactly one of n_active_orbitals or active_orbitals."
            )
        occupied_indices, virtual_indices = select_frontier_orbitals(
            mo_coeff.shape[1], n_occupied, n_active_orbitals
        )
    localized_occ, localized_virt = localize_blocks(
        mol, mo_coeff, occupied_indices, virtual_indices, rng
    )
    localized = np.hstack([localized_occ, localized_virt])
    is_occupied = [True] * localized_occ.shape[1] + [False] * localized_virt.shape[1]

    if fragment_atoms is not None:
        clusters = assign_orbitals_to_atoms(mol, localized, fragment_atoms)
    else:
        one_body, two_body = _localized_active_space_integrals(
            mol, mo_coeff, occupied_indices, n_occupied, localized
        )
        graph = build_coupling_graph(
            one_body, two_body, coupling_threshold=coupling_threshold
        )
        clusters = merge_clusters(graph, is_occupied, max_orbitals_per_fragment)

    if local_spins is not None and len(local_spins) != len(clusters):
        raise ValueError(
            f"local_spins has {len(local_spins)} entries but fragment_atoms "
            f"names {len(clusters)} fragments."
        )

    active_positions = tuple(occupied_indices) + tuple(virtual_indices)
    specs = []
    for index, cluster in enumerate(clusters):
        n_electrons = 2 * sum(is_occupied[orbital] for orbital in cluster)
        two_s = 0 if local_spins is None else int(local_spins[index])
        if abs(two_s) > n_electrons:
            raise ValueError(
                f"local_spins[{index}] asks for 2S={two_s} on a fragment "
                f"holding {n_electrons} electrons, which cannot supply that "
                "many unpaired spins."
            )
        if (n_electrons - two_s) % 2:
            raise ValueError(
                f"local_spins[{index}]=2S={two_s} has the wrong parity for a "
                f"fragment holding {n_electrons} electrons; n_alpha - n_beta "
                "must match the electron count's parity."
            )
        n_alpha = (n_electrons + two_s) // 2
        specs.append(
            FragmentSpec(
                orbitals=tuple(active_positions[o] for o in cluster),
                n_alpha=n_alpha,
                n_beta=n_electrons - n_alpha,
            )
        )
    return specs, localized, active_positions
