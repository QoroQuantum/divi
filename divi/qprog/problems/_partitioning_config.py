# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Cluster-size limits and algorithm choices for partitioned solving."""

from dataclasses import dataclass
from typing import Literal


def _check_sizes(max_size: int | None, min_clusters: int | None, max_name: str):
    if max_size is None and min_clusters is None:
        raise ValueError("At least one constraint must be specified.")
    if min_clusters is not None and min_clusters < 1:
        raise ValueError("'minimum_n_clusters' must be a positive integer.")
    if max_size is not None and max_size < 1:
        raise ValueError(f"'{max_name}' must be a positive number.")


@dataclass(frozen=True, eq=True)
class GraphPartitioningConfig:
    """Configuration for graph partitioning algorithms.

    This class defines the parameters and constraints for partitioning large graphs
    into smaller subgraphs for quantum algorithm execution. It supports multiple
    partitioning algorithms and allows specification of size constraints.

    Attributes:
        max_n_nodes_per_cluster: Maximum number of nodes allowed in each cluster.
            If None, no upper limit is enforced. Must be a positive integer.
        minimum_n_clusters: Minimum number of clusters to create. If None, no
            lower limit is enforced. Must be a positive integer.
        partitioning_algorithm: Algorithm to use for partitioning. Options are:
            - "spectral": Spectral partitioning using Fiedler vector (default)
            - "metis": METIS graph partitioning library
            - "kernighan_lin": Kernighan-Lin algorithm

    Note:
        At least one of `max_n_nodes_per_cluster` or `minimum_n_clusters` must be
        specified. Both constraints cannot be None.

    Examples:
        >>> # Partition into clusters of at most 10 nodes
        >>> config = GraphPartitioningConfig(max_n_nodes_per_cluster=10)

        >>> # Create at least 5 clusters using METIS
        >>> config = GraphPartitioningConfig(
        ...     minimum_n_clusters=5,
        ...     partitioning_algorithm="metis"
        ... )

        >>> # Both constraints: clusters of max 8 nodes, min 3 clusters
        >>> config = GraphPartitioningConfig(
        ...     max_n_nodes_per_cluster=8,
        ...     minimum_n_clusters=3
        ... )
    """

    max_n_nodes_per_cluster: int | None = None
    minimum_n_clusters: int | None = None
    partitioning_algorithm: Literal["spectral", "metis", "kernighan_lin"] = "spectral"

    @property
    def max_cluster_size(self) -> int | None:
        return self.max_n_nodes_per_cluster

    def __post_init__(self):
        _check_sizes(
            self.max_n_nodes_per_cluster,
            self.minimum_n_clusters,
            "max_n_nodes_per_cluster",
        )

        if self.partitioning_algorithm not in ("spectral", "metis", "kernighan_lin"):
            raise ValueError(
                f"Unsupported partitioning algorithm: {self.partitioning_algorithm}. "
                "Use 'spectral', 'metis' or 'kernighan_lin'."
            )


@dataclass(frozen=True, eq=True)
class QUBOPartitioningConfig:
    """Configuration for clustering QUBO variables by their couplings.

    Used by :class:`~divi.qprog.problems.CommunityDecomposer`, and by the
    portfolio problems, whose variables are assets clustered by covariance.

    Attributes:
        max_n_variables_per_cluster: Maximum number of variables per cluster.
        minimum_n_clusters: Minimum number of clusters.
        method: ``"modularity"`` (Louvain, default) or ``"spectral"`` (signed
            multi-view spectral clustering, arXiv 2502.16212).
        seed: Seed for the clustering step.

    At least one of ``max_n_variables_per_cluster`` or ``minimum_n_clusters`` is
    required.
    """

    max_n_variables_per_cluster: int | None = None
    minimum_n_clusters: int | None = None
    method: Literal["modularity", "spectral"] = "modularity"
    seed: int = 0

    @property
    def max_cluster_size(self) -> int | None:
        return self.max_n_variables_per_cluster

    def __post_init__(self):
        _check_sizes(
            self.max_n_variables_per_cluster,
            self.minimum_n_clusters,
            "max_n_variables_per_cluster",
        )

        if self.method not in ("modularity", "spectral"):
            raise ValueError(
                f"method must be 'modularity' or 'spectral', got {self.method!r}."
            )
