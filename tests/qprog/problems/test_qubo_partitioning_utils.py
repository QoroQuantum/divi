# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for signed multi-view spectral QUBO partitioning."""

import dimod
import numpy as np
import pytest
import scipy.sparse as sps
from networkx.algorithms.community import louvain_communities
from sklearn.cluster import KMeans

from divi.qprog.problems import GraphPartitioningConfig, QUBOPartitioningConfig
from divi.qprog.problems import _qubo_partitioning_utils as qpu
from divi.qprog.problems._qubo_partitioning_utils import (
    _ensure_min_clusters,
    _louvain_labels,
    _multiview_labels,
    _partition,
    _view_features,
    bqm_to_sparse,
    louvain_partition,
    partition_by_method,
    signed_multiview_partition,
)
from tests._helpers import exact_match


def _adjacency(n, edges, weight=1.0):
    s = np.zeros((n, n))
    for i, j in edges:
        s[i, j] = s[j, i] = weight
    return sps.csr_matrix(s)


def _id_groups(clusters):
    return [[int(i) for i in cl] for cl in clusters]


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


def _planted_two_block_sigma(
    intra_a=5.0, intra_b=5.0, bridge_w=0.05, cross_w=0.0, diagonal=0.0
):
    """Two strongly-coupled blocks joined by one weak bridge (single component).

    Block membership is *interleaved* (``[0,2,4,6]`` / ``[1,3,5,7]``) so a naive
    contiguous hard-split fallback cannot reproduce the blocks — recovering them
    forces the spectral clustering to actually do the work. ``intra_a``/``intra_b``
    take independent signs so a mixed-sign choice exercises both spectral views.
    ``cross_w`` couples every inter-block pair and ``diagonal`` fills the diagonal.
    """
    block_a, block_b = [0, 2, 4, 6], [1, 3, 5, 7]
    s = np.zeros((8, 8))
    s[np.ix_(block_a, block_b)] = cross_w
    s[np.ix_(block_b, block_a)] = cross_w
    np.fill_diagonal(s, diagonal)
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


def test_component_presplit_separates_disconnected_blocks():
    sigma = _block_matrix([[0, 1, 2], [3, 4, 5]], intra=1.0)
    # No budget pressure: only the connected-component pre-split should act.
    config = GraphPartitioningConfig(max_n_nodes_per_cluster=100)

    clusters = signed_multiview_partition(sigma, config, seed=0)

    got = sorted(sorted(int(i) for i in cl) for cl in clusters)
    assert got == [[0, 1, 2], [3, 4, 5]]


@pytest.mark.parametrize(
    ("partition", "rng_seed", "n", "budget"),
    [
        (signed_multiview_partition, 0, 20, 5),
        # Deep recursive bisection must not hit the recursion limit.
        (signed_multiview_partition, 3, 200, 6),
        (louvain_partition, 1, 20, 5),
    ],
    ids=["multiview", "multiview-large", "louvain"],
)
def test_partition_respects_budget_and_covers_all(partition, rng_seed, n, budget):
    rng = np.random.default_rng(rng_seed)
    a = np.triu(rng.normal(0, 1, (n, n)), 1)
    sigma = sps.csr_matrix(a + a.T)
    config = GraphPartitioningConfig(max_n_nodes_per_cluster=budget)

    clusters = partition(sigma, config, seed=0)

    assert all(len(cl) <= budget for cl in clusters)
    flat = [int(i) for cl in clusters for i in cl]
    assert sorted(flat) == list(range(n))  # exact cover, no duplicates


@pytest.mark.parametrize(
    ("n", "minimum_n_clusters"), [(6, 3), (4, 4)], ids=["floor", "singletons"]
)
def test_minimum_n_clusters_splits_complete_block(n, minimum_n_clusters):
    sigma = _block_matrix([list(range(n))], intra=1.0)
    config = GraphPartitioningConfig(minimum_n_clusters=minimum_n_clusters)

    clusters = signed_multiview_partition(sigma, config, seed=0)

    assert len(clusters) == minimum_n_clusters
    assert sorted(i for cl in _id_groups(clusters) for i in cl) == list(range(n))


@pytest.mark.parametrize(
    ("intra", "bridge_w"),
    [(5.0, 0.05), (0.5, 0.005), (-5.0, -0.05)],
    ids=["strong", "weak", "negative"],
)
def test_single_sign_block_recovers_communities(intra, bridge_w):
    # A single-sign problem leaves the opposite spectral view empty.
    sigma, block_a, block_b = _planted_two_block_sigma(
        intra_a=intra, intra_b=intra, bridge_w=bridge_w
    )
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

    with pytest.raises(
        ValueError,
        match=exact_match("minimum_n_clusters is larger than the number of variables."),
    ):
        signed_multiview_partition(sigma, config, seed=0)


def test_bqm_to_sparse_handles_string_labels():
    bqm = dimod.BinaryQuadraticModel(
        {"a": 1.0, "b": -2.0}, {("a", "b"): 3.0}, 0.0, dimod.Vartype.BINARY
    )

    variables, h, j = bqm_to_sparse(bqm)
    idx = {v: i for i, v in enumerate(variables)}

    assert set(variables) == {"a", "b"}
    assert h[idx["a"]] == 1.0 and h[idx["b"]] == -2.0
    assert j[idx["a"], idx["b"]] == 3.0 and j[idx["b"], idx["a"]] == 3.0


@pytest.mark.parametrize(
    "planted_kwargs",
    [
        {},
        {"diagonal": 50.0},
        {"intra_a": 1.0, "intra_b": 1.0},
        {"cross_w": 0.05},
    ],
    ids=["planted", "heavy-diagonal", "unit-intra", "all-cross-pairs"],
)
def test_louvain_recovers_planted_communities(planted_kwargs):
    sigma, block_a, block_b = _planted_two_block_sigma(**planted_kwargs)
    config = GraphPartitioningConfig(max_n_nodes_per_cluster=4)

    clusters = louvain_partition(sigma, config, seed=0)

    _assert_recovers_blocks(clusters, block_a, block_b)


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


def _path(n):
    return _adjacency(n, [(i, i + 1) for i in range(n - 1)])


def test_view_features_of_empty_view_is_none():
    assert _view_features(sps.csr_matrix((4, 4)), 2) is None


def test_view_features_of_star_is_leading_laplacian_eigenvector():
    star = _adjacency(4, [(0, 1), (0, 2), (0, 3)])

    features = _view_features(star, 1)

    assert features.shape == (4, 1)
    np.testing.assert_allclose(
        np.abs(features[:, 0]), [1 / np.sqrt(2), *[1 / np.sqrt(6)] * 3]
    )


@pytest.mark.parametrize("k", [2, 5])
def test_view_features_caps_k_below_view_size(k):
    assert _view_features(_path(3), k).shape == (3, 2)


@pytest.mark.parametrize("eigsh_fails", [False, True], ids=["sparse", "fallback"])
def test_view_features_sparse_path_matches_dense(eigsh_fails, mocker):
    view = _path(4)
    dense = _view_features(view, 2)
    mocker.patch.object(qpu, "_DENSE_EIGH_MAX", 2)
    if eigsh_fails:
        mocker.patch.object(qpu, "eigsh", side_effect=np.linalg.LinAlgError)

    features = _view_features(view, 2)

    assert features.shape == (4, 2)
    np.testing.assert_allclose(np.abs(features), np.abs(dense), atol=1e-8)


def test_view_features_at_dense_limit_uses_dense_solver(mocker):
    mocker.patch.object(qpu, "_DENSE_EIGH_MAX", 4)
    sparse_solver = mocker.patch.object(qpu, "eigsh")

    assert _view_features(_path(4), 2).shape == (4, 2)
    sparse_solver.assert_not_called()


@pytest.mark.parametrize(
    ("sigma", "k", "expected"),
    [
        (_path(3), 3, [0, 1, 2]),
        (sps.csr_matrix((4, 4)), 2, [0, 0, 0, 0]),
    ],
    ids=["k-equals-n", "no-couplings"],
)
def test_multiview_labels_trivial_cases(sigma, k, expected):
    labels = _multiview_labels(sigma, k, 0)

    assert labels.tolist() == expected
    assert labels.dtype.kind == "i"


@pytest.mark.parametrize(
    ("sigma", "expected"),
    [
        (sps.csr_matrix((2, 2)), [0, 1]),
        (_adjacency(2, [(0, 1)]), [0, 0]),
    ],
    ids=["two-uncoupled", "two-coupled"],
)
def test_louvain_labels_trivial_cases(sigma, expected):
    labels = _louvain_labels(sigma, 2, 0)

    assert sorted(labels.tolist()) == expected
    assert labels.dtype.kind == "i"


@pytest.mark.parametrize("sign", [1.0, -1.0], ids=["positive", "negative"])
def test_multiview_stored_zeros_leave_opposite_view_empty(sign, mocker):
    dense = _planted_two_block_sigma(
        intra_a=5.0 * sign, intra_b=5.0 * sign, bridge_w=0.05 * sign
    )[0].toarray()
    rows, cols = np.nonzero(~np.eye(8, dtype=bool))
    sigma = sps.csr_matrix((dense[rows, cols], (rows, cols)), shape=(8, 8))
    assert sigma.nnz == 56
    spy = mocker.spy(qpu, "_view_features")

    _multiview_labels(sigma, 2, 0)

    opposite = 1 if sign > 0 else 0
    assert spy.call_args_list[opposite].args[0].nnz == 0
    assert spy.spy_return_list[opposite] is None


def test_multiview_labels_forward_seed_to_kmeans(mocker):
    kmeans = mocker.patch.object(qpu, "KMeans", wraps=KMeans)
    sigma = _planted_two_block_sigma()[0]

    _multiview_labels(sigma, 2, 11)

    kmeans.assert_called_once()
    assert kmeans.call_args.kwargs["n_clusters"] == 2
    assert kmeans.call_args.kwargs["random_state"] == 11


def test_louvain_labels_forward_seed(mocker):
    communities = mocker.patch.object(
        qpu, "louvain_communities", wraps=louvain_communities
    )
    sigma = _planted_two_block_sigma()[0]

    _louvain_labels(sigma, 2, 13)

    assert communities.call_args.kwargs["seed"] == 13


def test_ensure_min_clusters_bisects_largest_without_couplings():
    clusters = [np.arange(0, 2), np.arange(2, 6), np.arange(6, 8)]

    result = _ensure_min_clusters(sps.csr_matrix((8, 8)), clusters, 4, 0)

    assert _id_groups(result) == [[0, 1], [6, 7], [2, 3], [4, 5]]


def test_ensure_min_clusters_stops_when_only_singletons_remain():
    clusters = [np.array([0]), np.array([1])]

    result = _ensure_min_clusters(sps.csr_matrix((2, 2)), clusters, 3, 0)

    assert _id_groups(result) == [[0], [1]]


def test_ensure_min_clusters_keeps_louvain_split():
    sigma = _adjacency(4, [(0, 1), (1, 2), (0, 2)])

    result = _ensure_min_clusters(sigma, [np.arange(4)], 2, 0, labeler=_louvain_labels)

    assert _id_groups(result) == [[0, 1, 2], [3]]


def test_minimum_n_clusters_alone_recovers_planted_blocks():
    sigma, block_a, block_b = _planted_two_block_sigma()

    clusters = signed_multiview_partition(
        sigma, GraphPartitioningConfig(minimum_n_clusters=2)
    )

    _assert_recovers_blocks(clusters, block_a, block_b)


def test_louvain_minimum_n_clusters_keeps_natural_communities():
    blocks = [[0, 3, 6], [1, 4, 7], [2, 5, 8]]
    sigma = _block_matrix(blocks, intra=5.0, bridges=[(6, 7, 0.05), (7, 8, 0.05)])

    clusters = louvain_partition(sigma, GraphPartitioningConfig(minimum_n_clusters=2))

    assert sorted(sorted(cl) for cl in _id_groups(clusters)) == blocks


def test_partition_forwards_seed_to_every_labeler_call(mocker):
    labeler = mocker.Mock(wraps=_multiview_labels)
    sigma = _planted_two_block_sigma()[0]
    config = GraphPartitioningConfig(max_n_nodes_per_cluster=4, minimum_n_clusters=3)

    clusters = _partition(sigma, config, seed=7, labeler=labeler)

    assert len(clusters) == 3
    sizes = [c.args[0].shape[0] for c in labeler.call_args_list]
    assert sizes[0] == 8 and 4 in sizes[1:]
    assert all(c.args[2] == 7 for c in labeler.call_args_list)


@pytest.mark.parametrize(
    ("method", "chosen", "other"),
    [
        ("spectral", "signed_multiview_partition", "louvain_partition"),
        ("modularity", "louvain_partition", "signed_multiview_partition"),
    ],
)
def test_partition_by_method_dispatch(method, chosen, other, mocker):
    chosen_mock = mocker.patch.object(qpu, chosen)
    other_mock = mocker.patch.object(qpu, other)
    sigma = _planted_two_block_sigma()[0]
    config = QUBOPartitioningConfig(
        max_n_variables_per_cluster=4, method=method, seed=7
    )

    result = partition_by_method(sigma, config)

    assert result is chosen_mock.return_value
    chosen_mock.assert_called_once_with(sigma, config, seed=7)
    other_mock.assert_not_called()


@pytest.mark.parametrize(
    ("wrapper", "labeler"),
    [
        (signed_multiview_partition, _multiview_labels),
        (louvain_partition, _louvain_labels),
    ],
    ids=["multiview", "louvain"],
)
def test_public_wrappers_default_seed_and_labeler(wrapper, labeler, mocker):
    partition = mocker.patch.object(qpu, "_partition")
    sigma = _planted_two_block_sigma()[0]
    config = GraphPartitioningConfig(max_n_nodes_per_cluster=4)

    result = wrapper(sigma, config)

    assert result is partition.return_value
    partition.assert_called_once_with(sigma, config, 0, labeler)
