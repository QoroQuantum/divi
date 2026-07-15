# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for signed multi-view spectral QUBO partitioning."""

import dimod
import numpy as np
import pytest
import scipy.sparse as sps

from divi.qprog.problems import GraphPartitioningConfig
from divi.qprog.problems._qubo_partitioning_utils import (
    bqm_to_sparse,
    louvain_partition,
    signed_multiview_partition,
)


def _block_matrix(blocks, intra, bridges=()):
    """Symmetric matrix with strong intra-block couplings and optional weak bridges."""
    n = sum(len(b) for b in blocks)
    s = np.zeros((n, n))
    for blk in blocks:
        for a in range(len(blk)):
            for b in range(a + 1, len(blk)):
                s[blk[a], blk[b]] = s[blk[b], blk[a]] = intra
    for i, j, w in bridges:
        s[i, j] = s[j, i] = w
    return sps.csr_matrix(s)


def _planted_two_block_sigma(intra_a=5.0, intra_b=5.0, bridge_w=0.05):
    """Two strongly-coupled blocks joined by one weak bridge (single component).

    Block membership is *interleaved* (``[0,2,4,6]`` / ``[1,3,5,7]``) so a naive
    contiguous hard-split fallback cannot reproduce the blocks — recovering them
    forces the spectral clustering to actually do the work. ``intra_a``/``intra_b``
    take independent signs so a mixed-sign choice exercises both spectral views.
    """
    block_a, block_b = [0, 2, 4, 6], [1, 3, 5, 7]
    s = np.zeros((8, 8))
    for blk, w in ((block_a, intra_a), (block_b, intra_b)):
        for x in range(len(blk)):
            for y in range(x + 1, len(blk)):
                s[blk[x], blk[y]] = s[blk[y], blk[x]] = w
    s[6, 7] = s[7, 6] = bridge_w  # weak inter-block bridge
    return sps.csr_matrix(s), set(block_a), set(block_b)


def _assert_recovers_blocks(clusters, block_a, block_b):
    assert len(clusters) == 2
    assert sorted(int(i) for cl in clusters for i in cl) == list(range(8))  # cover
    for cl in clusters:
        members = {int(i) for i in cl}
        assert members == block_a or members == block_b


def test_negative_sign_block_recovers_communities():
    # All-NEGATIVE couplings exercise the negative spectral view (positive view
    # empty), mirroring the positive-sign case; blocks must be recovered exactly.
    sigma, block_a, block_b = _planted_two_block_sigma(
        intra_a=-5.0, intra_b=-5.0, bridge_w=-0.05
    )
    config = GraphPartitioningConfig(max_n_nodes_per_cluster=4)

    clusters = signed_multiview_partition(sigma, config, seed=0)

    _assert_recovers_blocks(clusters, block_a, block_b)


def test_component_presplit_separates_disconnected_blocks():
    sigma = _block_matrix([[0, 1, 2], [3, 4, 5]], intra=1.0)
    # No budget pressure: only the connected-component pre-split should act.
    config = GraphPartitioningConfig(max_n_nodes_per_cluster=100)

    clusters = signed_multiview_partition(sigma, config, seed=0)

    got = sorted(sorted(int(i) for i in cl) for cl in clusters)
    assert got == [[0, 1, 2], [3, 4, 5]]


def test_multiview_respects_budget_and_covers_all():
    rng = np.random.default_rng(0)
    a = np.triu(rng.normal(0, 1, (20, 20)), 1)
    sigma = sps.csr_matrix(a + a.T)
    config = GraphPartitioningConfig(max_n_nodes_per_cluster=5)

    clusters = signed_multiview_partition(sigma, config, seed=0)

    assert all(len(cl) <= 5 for cl in clusters)
    flat = [int(i) for cl in clusters for i in cl]
    assert sorted(flat) == list(range(20))  # exact cover, no duplicates


def test_minimum_n_clusters_floor():
    dense = np.ones((6, 6))
    np.fill_diagonal(dense, 0)
    sigma = sps.csr_matrix(dense)
    config = GraphPartitioningConfig(minimum_n_clusters=3)

    clusters = signed_multiview_partition(sigma, config, seed=0)

    assert len(clusters) >= 3
    flat = [int(i) for cl in clusters for i in cl]
    assert sorted(flat) == list(range(6))


def test_partition_is_exact_cover_with_mixed_signs():
    rng = np.random.default_rng(1)
    a = np.triu(rng.uniform(-3, 3, (15, 15)), 1)
    sigma = sps.csr_matrix(a + a.T)
    config = GraphPartitioningConfig(max_n_nodes_per_cluster=4)

    clusters = signed_multiview_partition(sigma, config, seed=0)

    flat = [int(i) for cl in clusters for i in cl]
    assert sorted(flat) == list(range(15))
    assert len(flat) == len(set(flat))


def test_single_sign_block_recovers_communities():
    # All-POSITIVE couplings (single-sign): the negative view must be genuinely
    # empty so no garbage identity eigenvectors are fed to k-means. The blocks
    # (interleaved) must still be recovered exactly.
    sigma, block_a, block_b = _planted_two_block_sigma(intra_a=5.0, intra_b=5.0)
    config = GraphPartitioningConfig(max_n_nodes_per_cluster=4)

    clusters = signed_multiview_partition(sigma, config, seed=0)

    _assert_recovers_blocks(clusters, block_a, block_b)


def test_all_positive_partition_is_seed_deterministic():
    sigma, _a, _b = _planted_two_block_sigma(intra_a=5.0, intra_b=5.0)
    config = GraphPartitioningConfig(max_n_nodes_per_cluster=4)

    first = signed_multiview_partition(sigma, config, seed=0)
    second = signed_multiview_partition(sigma, config, seed=0)

    assert [sorted(map(int, c)) for c in first] == [sorted(map(int, c)) for c in second]


def test_min_clusters_exceeding_n_raises():
    sigma = _block_matrix([[0, 1, 2, 3]], intra=1.0)
    config = GraphPartitioningConfig(minimum_n_clusters=10)  # > 4 variables

    with pytest.raises(ValueError, match="larger than the number of variables"):
        signed_multiview_partition(sigma, config, seed=0)


def test_large_instance_respects_budget_without_recursion_error():
    rng = np.random.default_rng(3)
    a = np.triu(rng.normal(0, 1, (200, 200)), 1)
    sigma = sps.csr_matrix(a + a.T)
    config = GraphPartitioningConfig(max_n_nodes_per_cluster=6)

    clusters = signed_multiview_partition(sigma, config, seed=0)

    assert all(len(cl) <= 6 for cl in clusters)
    assert sorted(int(i) for cl in clusters for i in cl) == list(range(200))


def test_bqm_to_sparse_handles_string_labels():
    bqm = dimod.BinaryQuadraticModel(
        {"a": 1.0, "b": -2.0}, {("a", "b"): 3.0}, 0.0, dimod.Vartype.BINARY
    )

    variables, h, j = bqm_to_sparse(bqm)
    idx = {v: i for i, v in enumerate(variables)}

    assert set(variables) == {"a", "b"}
    assert h[idx["a"]] == 1.0 and h[idx["b"]] == -2.0
    assert j[idx["a"], idx["b"]] == 3.0 and j[idx["b"], idx["a"]] == 3.0


def test_louvain_recovers_planted_communities():
    sigma, block_a, block_b = _planted_two_block_sigma(intra_a=5.0, intra_b=5.0)
    config = GraphPartitioningConfig(max_n_nodes_per_cluster=4)

    clusters = louvain_partition(sigma, config, seed=0)

    _assert_recovers_blocks(clusters, block_a, block_b)


def test_louvain_respects_budget_and_covers_all():
    rng = np.random.default_rng(1)
    a = np.triu(rng.normal(0, 1, (20, 20)), 1)
    sigma = sps.csr_matrix(a + a.T)
    config = GraphPartitioningConfig(max_n_nodes_per_cluster=5)

    clusters = louvain_partition(sigma, config, seed=0)

    assert all(len(cl) <= 5 for cl in clusters)
    assert sorted(int(i) for cl in clusters for i in cl) == list(range(20))


def test_louvain_uniform_complete_block_does_not_fragment():
    # A near-uniform complete (rank-1) block has no community structure, so Louvain
    # returns a single community and the balanced-split fallback yields sized
    # clusters rather than a swarm of singletons. (Strongly non-uniform complete
    # QUBOs can still fragment under modularity — polish equalizes the outcome.)
    rng = np.random.default_rng(0)
    a = rng.uniform(1.0, 5.0, 12)
    q = np.outer(a, a)
    np.fill_diagonal(q, 0.0)
    sigma = sps.csr_matrix(np.triu(q, 1) + np.triu(q, 1).T)
    config = GraphPartitioningConfig(max_n_nodes_per_cluster=6)

    clusters = louvain_partition(sigma, config, seed=0)

    assert all(len(c) <= 6 for c in clusters)
    assert sum(1 for c in clusters if len(c) == 1) <= 1
    assert sorted(int(i) for c in clusters for i in c) == list(range(12))
