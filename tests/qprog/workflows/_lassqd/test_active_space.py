# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for automatic active-space selection and localization."""

import networkx as nx
import numpy as np
import pytest

pytest.importorskip("pyscf")

from pyscf import mcscf, scf

from divi.qprog.workflows._lassqd import _active_space as _active_space_module
from divi.qprog.workflows._lassqd._active_space import (
    _canonicalize_columns,
    _localized_active_space_integrals,
    assign_orbitals_to_atoms,
    auto_fragment_specs,
    build_coupling_graph,
    localize_blocks,
    merge_clusters,
    select_frontier_orbitals,
    split_active_orbitals,
    validate_fragment_atoms,
)
from tests.qprog.workflows._lassqd._helpers import (  # noqa: F401
    h4_chain,
    h4_chain_mean_field,
    h4_localized_blocks_seed0,
    h8_chain,
)

# 6 orbitals, 3 occupied (indices 0-2), HOMO index 2, LUMO index 3.
_N_ORBITALS = 6


@pytest.mark.parametrize(
    "n_occupied, n_active, expected_occupied, expected_virtual",
    [
        pytest.param(3, 4, (1, 2), (3, 4), id="even_splits_evenly"),
        pytest.param(3, 3, (1, 2), (3,), id="odd_favours_occupied"),
        pytest.param(3, 12, (0, 1, 2), (3, 4, 5), id="clamps_at_register_edges"),
        pytest.param(1, 6, (0,), (1, 2, 3), id="asymmetric_clamps_occupied_only"),
    ],
)
def test_select_frontier_orbitals(
    n_occupied, n_active, expected_occupied, expected_virtual
):
    """ceil(k/2) occupied, floor(k/2) virtual, each side clamped independently at
    the register edges."""
    occupied, virtual = select_frontier_orbitals(_N_ORBITALS, n_occupied, n_active)
    assert occupied == expected_occupied
    assert virtual == expected_virtual


@pytest.mark.parametrize(
    "n_occupied, n_active, match",
    [
        # An all-occupied register leaves no virtual orbital to select.
        pytest.param(_N_ORBITALS, 2, "at least one occupied", id="all-occupied"),
        pytest.param(3, 0, "n_active_orbitals", id="non-positive"),
    ],
)
def test_select_frontier_orbitals_rejects(n_occupied, n_active, match):
    with pytest.raises(ValueError, match=match):
        select_frontier_orbitals(_N_ORBITALS, n_occupied, n_active)


def test_localization_preserves_the_occupied_subspace(
    h4_chain_mean_field, h4_localized_blocks_seed0
):
    """Localizing occupied and virtual separately must not mix them."""
    mol = h4_chain_mean_field.mol
    mo_coeff = np.asarray(h4_chain_mean_field.mo_coeff)
    overlap = mol.intor("int1e_ovlp")

    occupied_indices = (0, 1)
    localized_occ, localized_virt = h4_localized_blocks_seed0

    original_occ = mo_coeff[:, list(occupied_indices)]
    # Projector onto the original occupied space must leave the localized
    # occupied block invariant.
    projector = original_occ @ original_occ.T @ overlap
    np.testing.assert_allclose(projector @ localized_occ, localized_occ, atol=1e-8)
    # And must annihilate the localized virtual block.
    np.testing.assert_allclose(
        projector @ localized_virt, np.zeros_like(localized_virt), atol=1e-8
    )


def test_localized_blocks_stay_orthonormal(
    h4_chain_mean_field, h4_localized_blocks_seed0
):
    mol = h4_chain_mean_field.mol
    overlap = mol.intor("int1e_ovlp")

    localized_occ, localized_virt = h4_localized_blocks_seed0
    np.testing.assert_allclose(
        localized_occ.T @ overlap @ localized_occ, np.eye(2), atol=1e-8
    )
    np.testing.assert_allclose(
        localized_virt.T @ overlap @ localized_virt, np.eye(2), atol=1e-8
    )


def _max_single_unit_population(mol, mo_coeff):
    """Per orbital, the larger of its Mulliken population on the two H2 units.

    ``h4_chain`` places atoms 0-1 on one H2 unit and atoms 2-3 on the other;
    a genuinely localized orbital should sit almost entirely on one unit.
    """
    overlap = mol.intor("int1e_ovlp")
    ao_slices = mol.aoslice_by_atom()
    per_ao_population = mo_coeff * (overlap @ mo_coeff)

    unit_one = per_ao_population[ao_slices[0, 2] : ao_slices[1, 3]].sum(axis=0)
    unit_two = per_ao_population[ao_slices[2, 2] : ao_slices[3, 3]].sum(axis=0)
    return np.maximum(unit_one, unit_two)


@pytest.mark.parametrize("seed", [0, 7, 11, 12])
def test_localize_blocks_finds_atom_localized_orbitals_despite_symmetry(
    seed, h4_chain_mean_field
):
    """The canonical H4 chain is a Pipek-Mezey stationary point at the
    identity rotation. Seeds 7, 11, and 12 are the specific regression
    cases: with a single perturbed restart, each converges back to that same
    symmetric stationary point instead of escaping it. Seed 0 is a control
    that already passed before the multi-restart fix."""
    mol = h4_chain_mean_field.mol
    mo_coeff = np.asarray(h4_chain_mean_field.mo_coeff)

    localized_occ, localized_virt = localize_blocks(
        mol, mo_coeff, (0, 1), (2, 3), np.random.default_rng(seed)
    )

    assert np.all(_max_single_unit_population(mol, localized_occ) > 0.9)
    assert np.all(_max_single_unit_population(mol, localized_virt) > 0.9)


def _two_block_integrals():
    """4 orbitals: {0,1} and {2,3} strongly coupled internally, weakly across."""
    n = 4
    one_body = np.zeros((n, n))
    two_body = np.zeros((n,) * 4)
    strong, weak = 0.5, 1e-6
    for p, q in [(0, 1), (2, 3)]:
        one_body[p, q] = one_body[q, p] = strong
    for p, q in [(0, 2), (0, 3), (1, 2), (1, 3)]:
        one_body[p, q] = one_body[q, p] = weak
    return one_body, two_body


def test_coupling_graph_drops_sub_threshold_edges():
    one_body, two_body = _two_block_integrals()
    graph = build_coupling_graph(one_body, two_body, coupling_threshold=1e-3)

    assert set(graph.nodes) == {0, 1, 2, 3}
    assert graph.has_edge(0, 1)
    assert graph.has_edge(2, 3)
    assert not graph.has_edge(0, 2)


def test_coupling_threshold_is_relative_to_the_strongest_edge():
    one_body, two_body = _two_block_integrals()
    # Scaling every integral must not change which edges survive.
    graph_small = build_coupling_graph(one_body, two_body)
    graph_large = build_coupling_graph(one_body * 1000.0, two_body)
    assert set(graph_small.edges) == set(graph_large.edges) == {(0, 1), (2, 3)}


def test_merge_clusters_recovers_the_two_blocks():
    one_body, two_body = _two_block_integrals()
    graph = build_coupling_graph(one_body, two_body)
    is_occupied = [True, False, True, False]
    clusters = merge_clusters(graph, is_occupied, max_orbitals_per_fragment=2)

    assert sorted(clusters) == [(0, 1), (2, 3)]


def test_merge_clusters_respects_the_size_limit():
    """All-equal weights: only size and full coverage are guaranteed, not the
    exact partition — which pair merges first is an artifact of iteration
    order when every candidate weight ties."""
    n = 4
    one_body = np.full((n, n), 0.5)
    np.fill_diagonal(one_body, 0.0)
    graph = build_coupling_graph(one_body, np.zeros((n,) * 4))
    clusters = merge_clusters(
        graph, [True, False, True, False], max_orbitals_per_fragment=2
    )

    assert all(len(cluster) <= 2 for cluster in clusters)
    assert sorted(orbital for cluster in clusters for orbital in cluster) == [
        0,
        1,
        2,
        3,
    ]


def test_merge_clusters_absorbs_all_occupied_clusters():
    """A cluster with no virtual orbital captures no correlation."""
    one_body, two_body = _two_block_integrals()
    graph = build_coupling_graph(one_body, two_body)
    # Make {0,1} both occupied and {2,3} both virtual so neither is mixed.
    with pytest.warns(UserWarning, match="no positive coupling"):
        clusters = merge_clusters(
            graph, [True, True, False, False], max_orbitals_per_fragment=4
        )

    assert len(clusters) == 1
    assert clusters[0] == (0, 1, 2, 3)


def test_merge_clusters_diagnostic_when_size_limit_blocks_the_fix():
    one_body, two_body = _two_block_integrals()
    graph = build_coupling_graph(one_body, two_body)
    with pytest.raises(ValueError, match="max_orbitals_per_fragment"):
        merge_clusters(graph, [True, True, False, False], max_orbitals_per_fragment=2)


def test_merge_clusters_returns_disjoint_complete_clusters():
    """Regression test: a stale cluster snapshot in the fix-up pass could
    revisit an orbital already absorbed elsewhere and duplicate it."""
    graph = nx.Graph()
    graph.add_nodes_from(range(5))
    graph.add_edge(0, 2, weight=0.765)
    graph.add_edge(0, 4, weight=0.786)

    with pytest.warns(UserWarning, match="no positive coupling"):
        clusters = merge_clusters(
            graph, [True, True, False, False, True], max_orbitals_per_fragment=4
        )

    covered = [orbital for cluster in clusters for orbital in cluster]
    assert sorted(covered) == list(range(5))
    assert len(covered) == len(set(covered))


def _weighted_graph(n_nodes, edges, scale=1.0):
    graph = nx.Graph()
    graph.add_nodes_from(range(n_nodes))
    for p, q, weight in edges:
        graph.add_edge(p, q, weight=scale * weight)
    return graph


_ALTERNATING_OCCUPATION = [True, False] * 4


@pytest.mark.filterwarnings("error")
def test_merge_clusters_fully_coupled_graph_within_limit_is_one_cluster():
    graph = _weighted_graph(
        4, [(p, q, 1.0 + p + q) for p in range(4) for q in range(p + 1, 4)]
    )
    clusters = merge_clusters(
        graph, _ALTERNATING_OCCUPATION[:4], max_orbitals_per_fragment=4
    )
    assert clusters == [(0, 1, 2, 3)]


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("scale", [1.0, 1e-3, 1e-9])
def test_merge_clusters_skips_an_oversized_pair_and_merges_a_later_fitting_one(
    scale,
):
    """Merges go 0-1, 2-3, (2,3)-4, 5-6, leaving ``(0, 1)`` with an oversized
    candidate ``(2, 3, 4)`` scanned before the fitting ``(5, 6)``. The partition
    depends only on the weights' order, not their scale."""
    graph = _weighted_graph(
        7,
        [(0, 1, 10.0), (2, 3, 9.0), (3, 4, 8.0), (5, 6, 7.0), (1, 2, 5.0), (1, 5, 1.0)],
        scale=scale,
    )
    clusters = merge_clusters(
        graph, _ALTERNATING_OCCUPATION[:7], max_orbitals_per_fragment=4
    )
    assert clusters == [(0, 1, 5, 6), (2, 3, 4)]


def test_merge_clusters_fix_up_ties_go_to_the_smallest_partner():
    """Occupied ``(0,)`` has two uncoupled partners within the limit; absorbing
    the smaller, virtual ``(3,)`` mixes it, where absorbing ``(1, 2)`` would leave
    ``(3,)`` with no room."""
    graph = _weighted_graph(4, [(1, 2, 1.0)])
    with pytest.warns(UserWarning, match="no positive coupling"):
        clusters = merge_clusters(
            graph, [True, True, False, False], max_orbitals_per_fragment=3
        )
    assert clusters == [(0, 3), (1, 2)]


@pytest.mark.parametrize("limit", [0, 1])
def test_merge_clusters_rejects_a_limit_below_two(limit):
    graph = _weighted_graph(2, [(0, 1, 1.0)])
    with pytest.raises(ValueError, match="at least 2"):
        merge_clusters(graph, [True, False], max_orbitals_per_fragment=limit)


def test_split_active_orbitals_splits_on_the_occupied_count():
    assert split_active_orbitals((4, 1, 3, 2), 3, 6) == ((1, 2), (3, 4))


@pytest.mark.parametrize(
    "fragment_atoms, match",
    [
        ((), "at least one fragment"),
        (([0], []), "names no atoms"),
        (([0], [4]), "out of range"),
        (([0, 1], [1]), "disjoint"),
    ],
)
def test_validate_fragment_atoms_rejects(fragment_atoms, match):
    with pytest.raises(ValueError, match=match):
        validate_fragment_atoms(fragment_atoms, 4)


# One sto-3g AO per hydrogen, so an identity column's Mulliken population sits
# entirely on its own atom.
_ATOM_COLUMNS = np.eye(4)


def test_assign_orbitals_to_atoms_groups_columns_by_dominant_atom():
    assert assign_orbitals_to_atoms(h4_chain(), _ATOM_COLUMNS, ([0, 2], [1, 3])) == [
        (0, 2),
        (1, 3),
    ]


@pytest.mark.parametrize(
    "columns, fragment_atoms, match",
    [
        ([0, 1], ([0],), "which no fragment claims"),
        ([0, 1], ([0, 1], [2, 3]), "got no active orbitals"),
    ],
)
def test_assign_orbitals_to_atoms_rejects(columns, fragment_atoms, match):
    with pytest.raises(ValueError, match=match):
        assign_orbitals_to_atoms(h4_chain(), _ATOM_COLUMNS[:, columns], fragment_atoms)


class _ScriptedPipekMezey:
    """Stand-in whose ``kernel`` returns its start and whose cost is scripted."""

    costs: list[float] = []
    starts: list[np.ndarray] = []

    def __init__(self, mol, block):
        self._block = block
        type(self).starts.append(block)

    def kernel(self):
        return self._block

    def cost_function(self):
        return type(self).costs.pop(0)


def _localize_with_scripted_costs(mocker, mean_field, costs):
    mocker.patch.object(_ScriptedPipekMezey, "costs", list(costs))
    mocker.patch.object(_ScriptedPipekMezey, "starts", [])
    mocker.patch.object(_active_space_module.lo, "PipekMezey", _ScriptedPipekMezey)
    return localize_blocks(
        mean_field.mol,
        np.asarray(mean_field.mo_coeff),
        (0, 1),
        (2, 3),
        np.random.default_rng(0),
    )


@pytest.mark.parametrize(
    "occupied_costs, kept_start",
    [
        pytest.param([1.0] * 9, 0, id="never_escapes_runs_every_restart"),
        pytest.param([1.0, 2.0], 1, id="escape_stops_restarts"),
        pytest.param(
            [1.0, 1.0 + 1e-8, 1.0 + 1e-9] + [1.0] * 6,
            1,
            id="sub_tolerance_gain_kept_without_stopping",
        ),
    ],
)
def test_localize_blocks_restart_policy(
    mocker, h4_chain_mean_field, occupied_costs, kept_start
):
    """Up to eight random restarts follow the canonical start; the
    highest-cost run is kept, and a gain beyond the relative tolerance stops
    the restarts early."""
    virtual_costs = [1.0, 2.0]
    localized_occ, _ = _localize_with_scripted_costs(
        mocker, h4_chain_mean_field, occupied_costs + virtual_costs
    )

    starts = _ScriptedPipekMezey.starts
    assert len(starts) == len(occupied_costs) + len(virtual_costs)
    np.testing.assert_array_equal(
        localized_occ,
        _canonicalize_columns(h4_chain_mean_field.mol, starts[kept_start]),
    )


def test_localize_blocks_passes_a_single_orbital_block_through(
    mocker, h4_chain_mean_field
):
    spy = mocker.patch.object(_active_space_module.lo, "PipekMezey")
    mo_coeff = np.asarray(h4_chain_mean_field.mo_coeff)
    localized_occ, localized_virt = localize_blocks(
        h4_chain_mean_field.mol, mo_coeff, (1,), (2,), np.random.default_rng(0)
    )
    spy.assert_not_called()
    np.testing.assert_array_equal(localized_occ, mo_coeff[:, [1]])
    np.testing.assert_array_equal(localized_virt, mo_coeff[:, [2]])


def test_auto_fragment_specs_on_h4_finds_two_fragments(h4_chain_mean_field):
    specs, localized, active_positions = auto_fragment_specs(
        h4_chain_mean_field.mol,
        np.asarray(h4_chain_mean_field.mo_coeff),
        n_occupied=2,
        rng=np.random.default_rng(0),
        n_active_orbitals=4,
        max_orbitals_per_fragment=2,
    )

    assert len(specs) == 2
    for spec in specs:
        assert spec.n_orbitals == 2
        # Closed-shell fragment populations come from the localized occupied
        # count, so alpha and beta must agree.
        assert spec.n_alpha == spec.n_beta == 1
    assert localized.shape[1] == 4
    # ``orbitals`` are register indices, not localized-column indices.
    assert sorted(o for spec in specs for o in spec.orbitals) == sorted(
        active_positions
    )


def _h8_auto_specs(**overrides):
    """H8 fragmented one half-chain per fragment.

    H8 rather than H4 because a polarized split of a 2-orbital fragment fills
    one spin channel and empties the other, leaving no excitation at all.
    """
    mol = h8_chain()
    mean_field = scf.RHF(mol).run(verbose=0)
    kwargs = dict(
        n_occupied=4,
        rng=np.random.default_rng(0),
        n_active_orbitals=8,
        fragment_atoms=([0, 1, 2, 3], [4, 5, 6, 7]),
    )
    kwargs.update(overrides)
    return auto_fragment_specs(
        mol,
        np.asarray(mean_field.mo_coeff),
        **kwargs,
    )


def test_auto_fragment_specs_applies_local_spins():
    """``local_spins`` sets 2S per fragment, leaving each fragment's electron
    count alone. This is the antiferromagnetic layout the closed-shell default
    cannot express."""
    specs, _, _ = _h8_auto_specs(local_spins=[2, -2])

    assert [(spec.n_alpha, spec.n_beta) for spec in specs] == [(3, 1), (1, 3)]
    assert sum(spec.n_alpha for spec in specs) == sum(spec.n_beta for spec in specs)


def test_auto_fragment_specs_rejects_unreachable_local_spin():
    """A fragment cannot supply more unpaired spins than it has electrons."""
    with pytest.raises(ValueError, match="cannot supply that many unpaired"):
        _h8_auto_specs(local_spins=[8, -8])


def _canonical_partition_key(mol, mo_coeff, seed, **selector):
    specs, _, _ = auto_fragment_specs(
        mol,
        mo_coeff,
        n_occupied=2,
        rng=np.random.default_rng(seed),
        max_orbitals_per_fragment=2,
        **(selector or {"n_active_orbitals": 4}),
    )
    return tuple(sorted((spec.orbitals, spec.n_alpha, spec.n_beta) for spec in specs))


def test_auto_fragment_specs_explicit_active_orbitals_match_frontier_selection(
    h4_chain_mean_field,
):
    """Naming the four frontier orbitals explicitly gives the frontier partition."""
    mol = h4_chain_mean_field.mol
    mo_coeff = np.asarray(h4_chain_mean_field.mo_coeff)
    assert _canonical_partition_key(
        mol, mo_coeff, 0, active_orbitals=(0, 1, 2, 3)
    ) == _canonical_partition_key(mol, mo_coeff, 0)


def test_auto_fragment_specs_partition_is_seed_independent(h4_chain_mean_field):
    """Different seeds must converge to the same fragment partition.

    Different random restarts can localize to the same physical orbitals in
    a different column order (Pipek-Mezey's cost is order-independent), so
    without column canonicalization the resulting fragment partition (which
    orbitals end up clustered together) can differ across seeds even though
    the underlying physical solution, and its energy, does not.
    """
    mol = h4_chain_mean_field.mol
    mo_coeff = np.asarray(h4_chain_mean_field.mo_coeff)

    partitions = {_canonical_partition_key(mol, mo_coeff, seed) for seed in range(8)}
    assert len(partitions) == 1


@pytest.mark.parametrize("relativistic", [False, True], ids=["rhf", "sfx2c1e"])
def test_auto_fragment_specs_active_integrals_include_frozen_core(
    h4_chain_mean_field, relativistic
):
    """With an occupied orbital left out of the active space, the one-body
    integrals feeding the coupling graph must carry its mean-field
    potential, matching a CASCI effective core Hamiltonian built the same
    way. A scalar-relativistic mean field's own core Hamiltonian must reach
    them too, since CASCI takes it from ``get_hcore``."""
    mean_field = (
        scf.RHF(h4_chain()).sfx2c1e().run(verbose=0)
        if relativistic
        else h4_chain_mean_field
    )
    mol = mean_field.mol
    mo_coeff = np.asarray(mean_field.mo_coeff)

    # n_occupied=2 but only 1 active occupied orbital selected: orbital 0
    # is a frozen core orbital, orbital 1 is active occupied, orbital 2 is
    # active virtual.
    occupied_indices, virtual_indices = select_frontier_orbitals(
        mo_coeff.shape[1], 2, 2
    )
    localized_occ, localized_virt = localize_blocks(
        mol, mo_coeff, occupied_indices, virtual_indices, np.random.default_rng(0)
    )
    localized = np.hstack([localized_occ, localized_virt])

    one_body, _ = _localized_active_space_integrals(
        mol,
        mo_coeff,
        occupied_indices,
        n_occupied=2,
        localized=localized,
        h_ao=mean_field.get_hcore(),
    )

    mc = mcscf.CASCI(mean_field, 2, 2)
    mc.mo_coeff = mo_coeff
    h1eff, _ = mc.get_h1eff()

    np.testing.assert_allclose(one_body, h1eff, atol=1e-10)
