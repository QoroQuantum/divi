# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for automatic active-space selection and localization."""

import networkx as nx
import numpy as np
import pytest
from pyscf import mcscf, scf

from divi.qprog.workflows._lassqd._active_space import (
    _localized_active_space_integrals,
    auto_fragment_specs,
    build_coupling_graph,
    localize_blocks,
    merge_clusters,
    select_frontier_orbitals,
)
from tests.qprog.workflows._lassqd._helpers import (  # noqa: F401
    h4_chain,
    h4_chain_mean_field,
    h4_localized_blocks_seed0,
)

# 6 orbitals, 3 occupied (indices 0-2), HOMO index 2, LUMO index 3.
_ENERGIES = np.array([-2.0, -1.0, -0.4, 0.3, 1.2, 2.5])


def test_even_n_active_orbitals_splits_evenly():
    occupied, virtual = select_frontier_orbitals(
        _ENERGIES, n_occupied=3, n_active_orbitals=4
    )
    assert occupied == (1, 2)
    assert virtual == (3, 4)


def test_odd_n_active_orbitals_favors_occupied():
    """ceil(k/2) occupied, floor(k/2) virtual."""
    occupied, virtual = select_frontier_orbitals(
        _ENERGIES, n_occupied=3, n_active_orbitals=3
    )
    assert occupied == (1, 2)
    assert virtual == (3,)


def test_n_active_orbitals_clamps_at_register_edges():
    occupied, virtual = select_frontier_orbitals(
        _ENERGIES, n_occupied=3, n_active_orbitals=12
    )
    assert occupied == (0, 1, 2)
    assert virtual == (3, 4, 5)


def test_asymmetric_clamping_only_clamps_the_occupied_side():
    """A one-orbital occupied register clamps while the virtual side does not."""
    occupied, virtual = select_frontier_orbitals(
        _ENERGIES, n_occupied=1, n_active_orbitals=6
    )
    assert occupied == (0,)
    assert virtual == (1, 2, 3)


def test_energy_window_includes_occupied_boundary_exactly():
    """Occupied qualifies when eps >= eps_HOMO - w, pinned at the exact edge."""
    occupied, virtual = select_frontier_orbitals(
        _ENERGIES, n_occupied=3, energy_window=0.6
    )
    # eps_HOMO = -0.4, so the occupied cutoff is exactly -1.0: orbital 1 sits
    # exactly on the boundary and must be included, orbital 0 (-2.0) must not.
    assert occupied == (1, 2)
    # eps_LUMO = 0.3, cutoff 0.9: orbital 4 (1.2) sits strictly outside it.
    assert virtual == (3,)


def test_energy_window_includes_virtual_boundary_exactly():
    """Virtual qualifies when eps <= eps_LUMO + w, pinned at the exact edge."""
    occupied, virtual = select_frontier_orbitals(
        _ENERGIES, n_occupied=3, energy_window=0.9
    )
    # eps_HOMO = -0.4, cutoff -1.3: orbital 0 (-2.0) sits strictly outside it.
    assert occupied == (1, 2)
    # eps_LUMO = 0.3, so the virtual cutoff is exactly 1.2: orbital 4 sits
    # exactly on the boundary and must be included, orbital 5 (2.5) must not.
    assert virtual == (3, 4)


def test_requires_exactly_one_selection_mode():
    with pytest.raises(ValueError, match="exactly one"):
        select_frontier_orbitals(
            _ENERGIES, n_occupied=3, n_active_orbitals=2, energy_window=0.5
        )
    with pytest.raises(ValueError, match="exactly one"):
        select_frontier_orbitals(_ENERGIES, n_occupied=3)


def test_rejects_selection_without_both_occupied_and_virtual():
    """An all-occupied register leaves no virtual orbital to select."""
    with pytest.raises(ValueError, match="at least one occupied"):
        select_frontier_orbitals(
            _ENERGIES, n_occupied=len(_ENERGIES), n_active_orbitals=2
        )


def test_rejects_non_positive_n_active_orbitals():
    with pytest.raises(ValueError, match="n_active_orbitals"):
        select_frontier_orbitals(_ENERGIES, n_occupied=3, n_active_orbitals=0)


def test_rejects_negative_energy_window():
    with pytest.raises(ValueError, match="energy_window"):
        select_frontier_orbitals(_ENERGIES, n_occupied=3, energy_window=-0.1)


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
    assert set(graph_small.edges) == set(graph_large.edges)


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


def test_auto_fragment_specs_on_h4_finds_two_fragments():
    mol = h4_chain()
    mean_field = scf.RHF(mol).run(verbose=0)
    specs, localized = auto_fragment_specs(
        mol,
        np.asarray(mean_field.mo_coeff),
        np.asarray(mean_field.mo_energy),
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


def _canonical_partition_key(mol, mo_coeff, mo_energy, seed):
    specs, _ = auto_fragment_specs(
        mol,
        mo_coeff,
        mo_energy,
        n_occupied=2,
        rng=np.random.default_rng(seed),
        n_active_orbitals=4,
        max_orbitals_per_fragment=2,
    )
    return tuple(sorted((spec.orbitals, spec.n_alpha, spec.n_beta) for spec in specs))


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
    mo_energy = np.asarray(h4_chain_mean_field.mo_energy)

    partitions = {
        _canonical_partition_key(mol, mo_coeff, mo_energy, seed) for seed in range(8)
    }
    assert len(partitions) == 1


def test_auto_fragment_specs_active_integrals_include_frozen_core():
    """With an occupied orbital left out of the active space, the one-body
    integrals feeding the coupling graph must carry its mean-field
    potential, matching a CASCI effective core Hamiltonian built the same
    way."""
    mol = h4_chain()
    mean_field = scf.RHF(mol).run(verbose=0)
    mo_coeff = np.asarray(mean_field.mo_coeff)

    # n_occupied=2 but only 1 active occupied orbital selected: orbital 0
    # is a frozen core orbital, orbital 1 is active occupied, orbital 2 is
    # active virtual.
    occupied_indices, virtual_indices = select_frontier_orbitals(
        np.asarray(mean_field.mo_energy), n_occupied=2, n_active_orbitals=2
    )
    localized_occ, localized_virt = localize_blocks(
        mol, mo_coeff, occupied_indices, virtual_indices, np.random.default_rng(0)
    )
    localized = np.hstack([localized_occ, localized_virt])

    one_body, _ = _localized_active_space_integrals(
        mol, mo_coeff, occupied_indices, n_occupied=2, localized=localized
    )

    mc = mcscf.CASCI(mean_field, 2, 2)
    mc.mo_coeff = mo_coeff
    h1eff, _ = mc.get_h1eff()

    np.testing.assert_allclose(one_body, h1eff, atol=1e-10)
