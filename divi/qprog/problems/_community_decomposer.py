# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""``hybrid`` decomposer that partitions a QUBO by community structure."""

from typing import Literal

from hybrid import traits
from hybrid.core import Runnable
from hybrid.exceptions import EndOfStream
from hybrid.utils import bqm_induced_by

from divi.qprog.problems._graph_partitioning_utils import GraphPartitioningConfig
from divi.qprog.problems._qubo_partitioning_utils import (
    bqm_to_sparse,
    louvain_partition,
    signed_multiview_partition,
)


class CommunityDecomposer(traits.ProblemDecomposer, traits.SISO, Runnable):
    """Structure-aware QUBO decomposer that partitions by community structure.

    A drop-in ``hybrid`` decomposer — like D-Wave's ``EnergyImpactDecomposer`` or
    ``ComponentDecomposer`` — that groups strongly-coupled variables and cuts weak
    couplings, so little energy is lost at partition boundaries. Connected components
    are separated first, then each component is clustered to honour the size budget.
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

    Raises:
        ImportError: If the ``qubo-decompose`` extra is not installed.
    """

    _reproducible = True

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

        # Content equality, not identity: hybrid.State.updated() deep-copies
        # ``problem`` each call, so an identity check would never reach EndOfStream.
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
