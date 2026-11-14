# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Structure-aware QUBO partitioning by community detection or spectral clustering.

Operates directly on the sparse *signed interaction matrix* ``Sigma`` of a QUBO
(``Sigma[i, j]`` = quadratic coefficient of variables ``i, j``), so no graph
object is constructed. Strongly-coupled variables are grouped together and weak
inter-cluster couplings are cut, which minimises the energy lost when a QUBO is
decomposed into independently-solved sub-problems.

Two clustering methods are provided. The default *modularity* method runs Louvain
community detection on the coupling-magnitude graph, auto-selecting the community
count. The *signed multi-view spectral* method splits ``Sigma`` into a positive
("repulsive") and negative ("attractive") view, builds the normalised Laplacian of
each, concatenates their leading spectral features, and runs k-means — respecting
the sign/frustration structure of the couplings.

These routines back :class:`~divi.qprog.problems.CommunityDecomposer`, a drop-in
``hybrid`` decomposer for
:class:`~divi.qprog.problems.BinaryOptimizationProblem` (used exactly like
D-Wave's ``EnergyImpactDecomposer``), and the asset clustering of the portfolio
problems.
"""

import heapq
from typing import cast

import numpy as np
import scipy.sparse as sps
from networkx import Graph
from networkx.algorithms.community import louvain_communities
from scipy.sparse.csgraph import connected_components, laplacian
from scipy.sparse.linalg import ArpackError, eigsh
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

from ._partitioning_config import (
    GraphPartitioningConfig,
    QUBOPartitioningConfig,
)

_Config = GraphPartitioningConfig | QUBOPartitioningConfig

_DENSE_EIGH_MAX = 800


def bqm_to_sparse(bqm):
    """Return ``(variables, h, J)``: variable order, linear vector, symmetric coupling.

    ``J[i, j] == J[j, i]`` is the quadratic coefficient of variables ``i, j`` in the
    ``variables`` order; ``h`` holds the linear biases.
    """
    variables = list(bqm.variables)
    idx = {v: i for i, v in enumerate(variables)}
    n = len(variables)
    h = np.zeros(n)
    for v, bias in bqm.linear.items():
        h[idx[v]] = bias
    rows, cols, data = [], [], []
    for (u, v), q in bqm.quadratic.items():
        i, j = idx[u], idx[v]
        rows += [i, j]
        cols += [j, i]
        data += [q, q]
    return variables, h, sps.csr_matrix((data, (rows, cols)), shape=(n, n))


def _view_features(W: sps.csr_matrix, k: int) -> np.ndarray | None:
    """Leading ``k`` eigenvectors of a view's normalised Laplacian.

    Returns ``None`` for an empty view (no couplings of that sign). Uses dense
    ``eigh`` for moderate sizes and sparse shift-invert ``eigsh`` above
    :data:`_DENSE_EIGH_MAX`, falling back to dense on solver failure. Repeated
    calls return identical features.
    """
    if W.nnz == 0:
        return None
    # scipy's ``laplacian`` is typed as possibly returning a (matrix, diag) tuple;
    # with the default return_diag=False it returns just the sparse matrix.
    L = cast(sps.csr_matrix, laplacian(W, normed=True))
    m = L.shape[0]
    k = min(k, m - 1)
    if k < 1:
        return None

    if m <= _DENSE_EIGH_MAX:
        _vals, vecs = np.linalg.eigh(np.asarray(L.todense()))
        return vecs[:, :k]

    try:
        v0 = np.random.default_rng(0).uniform(-1.0, 1.0, m)
        vals, vecs = eigsh(L.tocsc(), k=k, sigma=-1e-6, which="LM", v0=v0)
        return vecs[:, np.argsort(vals)]
    except (ArpackError, np.linalg.LinAlgError):
        _vals, vecs = np.linalg.eigh(np.asarray(L.todense()))
        return vecs[:, :k]


def _multiview_labels(sigma: sps.csr_matrix, k: int, seed: int) -> np.ndarray:
    """Signed multi-view spectral clustering into ``k`` groups (arXiv 2502.16212).

    Repeated calls return identical labels.
    """
    n = sigma.shape[0]
    if k >= n:
        return np.arange(n)

    # Split into positive ("repulsive") and negative ("attractive") views,
    # keeping only the sign-matching nonzero entries so an all-one-sign block
    # yields a genuinely empty opposite view (no explicit zeros, which would
    # otherwise feed a degenerate all-zero Laplacian's identity eigenvectors
    # into k-means as garbage features).
    s = sigma.tocoo()
    pos, neg = s.data > 0, s.data < 0
    w_pos = sps.csr_matrix((s.data[pos], (s.row[pos], s.col[pos])), shape=sigma.shape)
    w_neg = sps.csr_matrix((-s.data[neg], (s.row[neg], s.col[neg])), shape=sigma.shape)

    features = [
        f for f in (_view_features(w_pos, k), _view_features(w_neg, k)) if f is not None
    ]
    if not features:
        return np.zeros(n, dtype=int)

    with threadpool_limits(limits=1, user_api="openmp"):
        return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(
            np.hstack(features)
        )


def _louvain_labels(sigma: sps.csr_matrix, k: int, seed: int) -> np.ndarray:
    """Community labels from Louvain modularity maximisation.

    ``k`` is ignored — Louvain derives the community count from the graph itself.
    Edge weights are coupling *magnitudes* (Louvain requires non-negative weights),
    so strongly-coupled variables of either sign are grouped together. A uniform
    complete (structureless) block collapses to one community, which the caller
    then cuts along its Fiedler vector; non-uniform complete blocks (e.g. rank-1
    number-partitioning) can still fragment.
    """
    n = sigma.shape[0]
    s = sigma.tocoo()
    g = Graph()
    g.add_nodes_from(range(n))
    for i, j, v in zip(s.row.tolist(), s.col.tolist(), s.data.tolist()):
        if i < j and v != 0.0:
            g.add_edge(i, j, weight=abs(v))

    labels = np.empty(n, dtype=int)
    for lbl, community in enumerate(louvain_communities(g, weight="weight", seed=seed)):
        for node in community:
            labels[node] = lbl
    return labels


def _split_to_budget(
    sigma: sps.csr_matrix,
    idx: np.ndarray,
    budget: int,
    seed: int,
    labeler=_multiview_labels,
) -> list[np.ndarray]:
    """Cluster ``idx`` until every group has at most ``budget`` members.

    Iterative (worklist) bisection-to-budget, mirroring the graph partitioner's
    size predicate, using ``labeler`` for the clustering step. When the labeler
    makes no progress, the group is cut into ``k`` equal runs along its Fiedler
    vector.
    """
    out: list[np.ndarray] = []
    stack: list[np.ndarray] = [idx]
    while stack:
        cur = stack.pop()
        if len(cur) <= budget:
            out.append(cur)
            continue

        k = int(np.ceil(len(cur) / budget))
        labels = labeler(cast(sps.csr_matrix, sigma[np.ix_(cur, cur)]), k, seed)
        groups = [cur[labels == lbl] for lbl in np.unique(labels)]

        if len(groups) < 2:
            groups = _split_along_fiedler(sigma, cur, k)

        stack.extend(groups)
    return out


def sort_by_fiedler_vector(sigma: sps.csr_matrix, idx: np.ndarray) -> np.ndarray:
    """``idx`` sorted along the Fiedler vector of their coupling-magnitude graph.

    Strongly coupled variables end up next to each other, so cutting the result
    into contiguous runs groups them. The vector's sign is fixed by its largest
    entry, so the order does not depend on the eigensolver. ``idx`` is returned unchanged when its
    variables share no coupling.
    """
    sub = abs(sps.csr_matrix(sigma[np.ix_(idx, idx)]))
    sub.setdiag(0)
    sub.eliminate_zeros()
    features = _view_features(sub, 2)
    if features is None or features.shape[1] < 2:
        return idx
    fiedler = features[:, 1]
    fiedler = fiedler * np.sign(fiedler[np.argmax(np.abs(fiedler))])
    return idx[np.argsort(fiedler, kind="stable")]


def _split_along_fiedler(
    sigma: sps.csr_matrix, idx: np.ndarray, k: int
) -> list[np.ndarray]:
    """``idx`` cut into ``k`` runs of near-equal size along its Fiedler vector,
    each run ascending."""
    return [
        np.sort(run) for run in np.array_split(sort_by_fiedler_vector(sigma, idx), k)
    ]


def _merge_clusters_to_budget(
    coupling: sps.csr_matrix,
    clusters: list[np.ndarray],
    budget: int,
    minimum: int | None,
) -> list[np.ndarray]:
    """Merge clusters into the neighbour they couple to most while the union
    has at most ``budget`` members.

    Coupling between clusters is the summed ``|coupling|`` across them. The
    smallest cluster with a fitting coupled neighbour merges first; ties go to
    the smaller neighbour, then the earlier one. Merging stops at ``minimum``
    clusters. Surviving clusters keep their order, merged ones are ascending.
    """
    if not clusters:
        return []
    n_clusters = len(clusters)
    members = np.concatenate(clusters)
    owner = np.repeat(np.arange(n_clusters), [len(c) for c in clusters])
    membership = sps.csr_matrix(
        (np.ones(len(members)), (members, owner)),
        shape=(coupling.shape[0], n_clusters),
    )
    between = (membership.T @ abs(coupling) @ membership).tocoo()
    links: list[dict[int, float]] = [{} for _ in range(n_clusters)]
    for i, j, weight in zip(
        between.row.tolist(), between.col.tolist(), between.data.tolist()
    ):
        if i != j and weight > 0:
            links[i][j] = links[i].get(j, 0.0) + weight

    clusters = list(clusters)
    sizes = [len(c) for c in clusters]
    queue = [(size, i) for i, size in enumerate(sizes)]
    heapq.heapify(queue)
    alive = n_clusters
    while queue and alive > max(minimum or 1, 1):
        size, small = heapq.heappop(queue)
        if size != sizes[small]:
            continue
        fits = [j for j in links[small] if size + sizes[j] <= budget]
        if not fits:
            continue
        big = max(fits, key=lambda j: (links[small][j], -sizes[j], -j))
        for j, weight in links[small].items():
            del links[j][small]
            if j != big:
                links[big][j] = links[big].get(j, 0.0) + weight
                links[j][big] = links[j].get(big, 0.0) + weight
        links[small] = {}
        clusters[big] = np.sort(np.concatenate((clusters[big], clusters[small])))
        sizes[big] += size
        sizes[small] = -1
        heapq.heappush(queue, (sizes[big], big))
        alive -= 1
    return [clusters[i] for i in range(n_clusters) if sizes[i] >= 0]


def _ensure_min_clusters(
    sigma: sps.csr_matrix,
    clusters: list[np.ndarray],
    min_clusters: int,
    seed: int,
    labeler=_multiview_labels,
) -> list[np.ndarray]:
    """Bisect the largest clusters until at least ``min_clusters`` exist."""
    clusters = list(clusters)
    while len(clusters) < min_clusters:
        splittable = [i for i, c in enumerate(clusters) if len(c) > 1]
        if not splittable:
            break
        i = max(splittable, key=lambda j: len(clusters[j]))
        idx = clusters.pop(i)
        labels = labeler(cast(sps.csr_matrix, sigma[np.ix_(idx, idx)]), 2, seed)
        groups = [idx[labels == lbl] for lbl in np.unique(labels)]
        if len(groups) < 2:
            groups = _split_along_fiedler(sigma, idx, 2)
        clusters.extend(groups)
    return clusters


def _partition(
    sigma: sps.csr_matrix, config: _Config, seed: int, labeler
) -> list[np.ndarray]:
    """Partition a QUBO's interaction matrix into variable-index clusters.

    Connected components are separated first (a zero-cost exact reduction), then
    each component is clustered with ``labeler`` to honour ``config``'s
    ``max_cluster_size`` budget and ``minimum_n_clusters`` floor, and coupled
    clusters that fit the budget together are merged.
    """
    if (
        config.minimum_n_clusters is not None
        and config.minimum_n_clusters > sigma.shape[0]
    ):
        raise ValueError("minimum_n_clusters is larger than the number of variables.")

    n_comp, comp_labels = connected_components(sigma != 0, directed=False)

    clusters: list[np.ndarray] = []
    for c in range(n_comp):
        idx = np.where(comp_labels == c)[0]
        if config.max_cluster_size is not None:
            clusters.extend(
                _split_to_budget(sigma, idx, config.max_cluster_size, seed, labeler)
            )
        else:
            clusters.append(idx)

    if config.max_cluster_size is not None:
        clusters = _merge_clusters_to_budget(
            sigma, clusters, config.max_cluster_size, config.minimum_n_clusters
        )
    if config.minimum_n_clusters is not None:
        clusters = _ensure_min_clusters(
            sigma, clusters, config.minimum_n_clusters, seed, labeler
        )
    return clusters


def partition_spectral(
    sigma: sps.csr_matrix, config: _Config, *, seed: int = 0
) -> list[np.ndarray]:
    """Partition via signed multi-view spectral clustering (arXiv 2502.16212).

    Args:
        sigma: Symmetric sparse interaction matrix over variables ``0..n-1``.
        config: Cluster-size limit and cluster-count floor.
        seed: Seed for the k-means step.

    Returns:
        Integer-index arrays partitioning ``0..n-1``, one per cluster.
    """
    return _partition(sigma, config, seed, _multiview_labels)


def partition_by_method(
    sigma: sps.csr_matrix, config: QUBOPartitioningConfig
) -> list[np.ndarray]:
    """Partition with ``config.method`` and ``config.seed``."""
    if config.method == "spectral":
        return partition_spectral(sigma, config, seed=config.seed)
    return partition_louvain(sigma, config, seed=config.seed)


def partition_louvain(
    sigma: sps.csr_matrix, config: _Config, *, seed: int = 0
) -> list[np.ndarray]:
    """Partition via Louvain modularity community detection on the interaction graph.

    Same component pre-split and budget/floor handling as
    :func:`partition_spectral`, but clusters each over-budget component by
    modularity maximisation (coupling-magnitude weights) instead of spectral views.

    Returns:
        Integer-index arrays partitioning ``0..n-1``, one per cluster.
    """
    return _partition(sigma, config, seed, _louvain_labels)
