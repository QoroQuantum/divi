# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Structure-aware QUBO partitioning via signed multi-view spectral clustering.

Operates directly on the sparse *signed interaction matrix* ``Sigma`` of a QUBO
(``Sigma[i, j]`` = quadratic coefficient of variables ``i, j``), so no graph
object is constructed. Strongly-coupled variables are grouped together and weak
inter-cluster couplings are cut, which minimizes the energy lost when a QUBO is
decomposed into independently-solved sub-problems.

Two clustering methods are provided. The default *modularity* method runs Louvain
community detection on the coupling-magnitude graph, auto-selecting the community
count. The *signed multi-view spectral* method splits ``Sigma`` into a positive
("repulsive") and negative ("attractive") view, builds the normalized Laplacian of
each, concatenates their leading spectral features, and runs k-means — respecting
the sign/frustration structure of the couplings.

Exposed as :class:`~divi.qprog.problems.CommunityDecomposer`, a drop-in
``hybrid`` decomposer for
:class:`~divi.qprog.problems.BinaryOptimizationProblem` (used exactly like
D-Wave's ``EnergyImpactDecomposer``).
"""

from typing import Literal, cast

import numpy as np
import scipy.sparse as sps
from hybrid import traits
from hybrid.core import Runnable
from hybrid.exceptions import EndOfStream
from hybrid.utils import bqm_induced_by
from networkx import Graph
from networkx.algorithms.community import louvain_communities
from scipy.sparse.csgraph import connected_components, laplacian
from scipy.sparse.linalg import ArpackError, ArpackNoConvergence, eigsh
from sklearn.cluster import KMeans

from divi.qprog.problems._graph_partitioning_utils import GraphPartitioningConfig

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
    """Leading ``k`` eigenvectors of a view's normalized Laplacian.

    Returns ``None`` for an empty view (no couplings of that sign). Uses dense
    ``eigh`` for moderate sizes and sparse shift-invert ``eigsh`` above
    :data:`_DENSE_EIGH_MAX`, falling back to dense on solver failure.
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
        vals, vecs = eigsh(L.tocsc(), k=k, sigma=-1e-6, which="LM")
        return vecs[:, np.argsort(vals)]
    except (ArpackError, ArpackNoConvergence, np.linalg.LinAlgError):
        _vals, vecs = np.linalg.eigh(np.asarray(L.todense()))
        return vecs[:, :k]


def _multiview_labels(sigma: sps.csr_matrix, k: int, seed: int) -> np.ndarray:
    """Signed multi-view spectral clustering into ``k`` groups (arXiv 2502.16212)."""
    n = sigma.shape[0]
    if k <= 1:
        return np.zeros(n, dtype=int)
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

    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(
        np.hstack(features)
    )


def _louvain_labels(sigma: sps.csr_matrix, k: int, seed: int) -> np.ndarray:
    """Community labels from Louvain modularity maximization.

    ``k`` is ignored — Louvain derives the community count from the graph itself.
    Edge weights are coupling *magnitudes* (Louvain requires non-negative weights),
    so strongly-coupled variables of either sign are grouped together. A uniform
    complete (structureless) block collapses to one community, which the caller's
    balanced-split fallback then handles; non-uniform complete blocks (e.g. rank-1
    number-partitioning) can still fragment.
    """
    n = sigma.shape[0]
    if n <= 1:
        return np.zeros(n, dtype=int)

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
    size predicate, using ``labeler`` for the clustering step. Falls back to a
    balanced hard split when the labeler fails to separate or makes no progress.
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
        groups = [g for g in groups if len(g) > 0]

        # Degenerate (single group) or no-progress (a group as large as the
        # input) → balanced hard split, which strictly shrinks and respects the
        # budget. Iterative worklist avoids any recursion-depth limit. (A tighter
        # anti-"singleton-peeling" guard was tried and reverted: forcing balance on
        # near-degenerate/rank-1 single-sign blocks measurably worsened unpolished
        # number-partitioning results, and local_search erases the difference.)
        if len(groups) < 2 or max(len(g) for g in groups) == len(cur):
            step = int(np.ceil(len(cur) / k))
            groups = [cur[i : i + step] for i in range(0, len(cur), step)]

        stack.extend(groups)
    return out


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
        groups = [g for g in groups if len(g) > 0]
        if len(groups) < 2:
            mid = len(idx) // 2
            groups = [idx[:mid], idx[mid:]]
        clusters.extend(groups)
    return clusters


def _partition(
    sigma: sps.csr_matrix, config: GraphPartitioningConfig, seed: int, labeler
) -> list[np.ndarray]:
    """Partition a QUBO's interaction matrix into variable-index clusters.

    Connected components are separated first (a zero-cost exact reduction), then
    each component is clustered with ``labeler`` to honor ``config``'s
    ``max_n_nodes_per_cluster`` budget and ``minimum_n_clusters`` floor.
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
        if config.max_n_nodes_per_cluster is not None:
            clusters.extend(
                _split_to_budget(
                    sigma, idx, config.max_n_nodes_per_cluster, seed, labeler
                )
            )
        else:
            clusters.append(idx)

    if config.minimum_n_clusters is not None:
        clusters = _ensure_min_clusters(
            sigma, clusters, config.minimum_n_clusters, seed, labeler
        )
    return clusters


def signed_multiview_partition(
    sigma: sps.csr_matrix, config: GraphPartitioningConfig, *, seed: int = 0
) -> list[np.ndarray]:
    """Partition via signed multi-view spectral clustering (arXiv 2502.16212).

    Args:
        sigma: Symmetric sparse interaction matrix over variables ``0..n-1``.
        config: Size/cluster-count constraints (``partitioning_algorithm`` ignored).
        seed: Seed for the k-means step.

    Returns:
        Integer-index arrays partitioning ``0..n-1``, one per cluster.
    """
    return _partition(sigma, config, seed, _multiview_labels)


def louvain_partition(
    sigma: sps.csr_matrix, config: GraphPartitioningConfig, *, seed: int = 0
) -> list[np.ndarray]:
    """Partition via Louvain modularity community detection on the interaction graph.

    Same component pre-split and budget/floor handling as
    :func:`signed_multiview_partition`, but clusters each over-budget component by
    modularity maximization (coupling-magnitude weights) instead of spectral views.

    Returns:
        Integer-index arrays partitioning ``0..n-1``, one per cluster.
    """
    return _partition(sigma, config, seed, _louvain_labels)


class CommunityDecomposer(traits.ProblemDecomposer, traits.SISO, Runnable):
    """Structure-aware QUBO decomposer that partitions by community structure.

    A drop-in ``hybrid`` decomposer — like D-Wave's ``EnergyImpactDecomposer`` or
    ``ComponentDecomposer`` — that groups strongly-coupled variables and cuts weak
    couplings, so little energy is lost at partition boundaries. Connected components
    are separated first, then each component is clustered to honor the size budget.
    Successive calls roll through the resulting clusters, one subproblem per iteration.

    Two clustering methods are available via ``method``:

    - ``"modularity"`` (default): Louvain community detection on the
      coupling-magnitude graph. Auto-picks the community count and is the strongest
      general-purpose choice across structured, dense, and constrained QUBOs.
    - ``"spectral"``: signed multi-view spectral clustering (arXiv 2502.16212),
      which respects coupling signs. It can degenerately peel most variables into
      singletons on dense/rank-structured inputs, so prefer it mainly for
      sparse-geometric instances.

    Best on problems with community structure; for featureless (dense, unstructured)
    QUBOs, D-Wave's ``EnergyImpactDecomposer`` is also a reasonable choice.

    Args:
        max_cluster_size: Maximum number of variables per subproblem (the qubit
            budget). At least one of ``max_cluster_size`` / ``min_clusters`` required.
        min_clusters: Minimum number of clusters (partitions) to produce.
        method: ``"modularity"`` (default) or ``"spectral"``.
        seed: Seed for the clustering step (Louvain / k-means).
        silent_rewind: If ``False``, raise ``hybrid.exceptions.EndOfStream`` once
            all clusters are exhausted (used by ``hybrid.Unwind``, which is how
            :meth:`BinaryOptimizationProblem.decompose` drives this decomposer).
    """

    def __init__(
        self,
        *,
        max_cluster_size: int | None = None,
        min_clusters: int | None = None,
        method: Literal["spectral", "modularity"] = "modularity",
        seed: int = 0,
        silent_rewind: bool = True,
        **runopts,
    ):
        super().__init__(**runopts)
        if max_cluster_size is None and min_clusters is None:
            raise ValueError(
                "Provide at least one of 'max_cluster_size' or 'min_clusters'."
            )
        if method not in ("spectral", "modularity"):
            raise ValueError(
                f"method must be 'spectral' or 'modularity', got {method!r}."
            )
        self.max_cluster_size = max_cluster_size
        self.min_clusters = min_clusters
        self.method = method
        self.seed = seed
        self.silent_rewind = silent_rewind
        self._rolling_bqm = None
        self._iter_clusters = None

    def __repr__(self):
        return (
            f"{self}(max_cluster_size={self.max_cluster_size!r}, "
            f"min_clusters={self.min_clusters!r}, method={self.method!r}, "
            f"seed={self.seed!r}, silent_rewind={self.silent_rewind!r})"
        )

    def _get_iter_clusters(self, bqm):
        variables, _h, sigma = bqm_to_sparse(bqm)
        config = GraphPartitioningConfig(
            max_n_nodes_per_cluster=self.max_cluster_size,
            minimum_n_clusters=self.min_clusters,
        )
        partition = (
            signed_multiview_partition
            if self.method == "spectral"
            else louvain_partition
        )
        clusters = partition(sigma, config, seed=self.seed)
        return iter([[variables[i] for i in cl] for cl in clusters])

    def next(self, state, **runopts):
        """Emit the next cluster as the subproblem (one hybrid decomposition step)."""
        silent_rewind = runopts.get("silent_rewind", self.silent_rewind)
        bqm = state.problem

        if bqm.num_variables <= 1:
            return state.updated(subproblem=bqm)

        # Roll through the clusters, one subproblem per call. Detect a new problem
        # by CONTENT equality, not identity: ``hybrid.State.updated()`` deep-copies
        # the (non-overridden) ``problem`` each iteration, so ``Unwind`` hands us a
        # fresh-but-equal object every call — an identity check would rebuild the
        # cluster iterator forever and never reach EndOfStream. On exhaustion,
        # either rewind or signal EndOfStream.
        if bqm != self._rolling_bqm:
            self._rolling_bqm = bqm
            self._iter_clusters = self._get_iter_clusters(bqm)
        assert self._iter_clusters is not None  # set above or on a prior call
        try:
            cluster = next(self._iter_clusters)
        except StopIteration:
            if not silent_rewind:
                self._rolling_bqm = None
                raise EndOfStream
            self._iter_clusters = self._get_iter_clusters(bqm)
            cluster = next(self._iter_clusters)

        sample = state.samples.change_vartype(bqm.vartype).first.sample
        return state.updated(subproblem=bqm_induced_by(bqm, cluster, sample))
