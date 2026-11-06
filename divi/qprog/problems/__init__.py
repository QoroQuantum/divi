# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Problem classes for quantum optimisation and ground-state estimation."""

from typing import Any

from divi._optional import import_optional

from ._base import QAOAProblem
from ._binary import BinaryOptimizationProblem
from ._constraints import LinearConstraint
from ._graph_partitioning_utils import draw_partitions
from ._graphs import (
    GraphProblemBase,
    MaxCliqueProblem,
    MaxCutProblem,
    MaxIndependentSetProblem,
    MaxWeightCycleProblem,
    MinVertexCoverProblem,
    draw_graph_solution_nodes,
)
from ._hamiltonian import HamiltonianProblem, MolecularProblem
from ._partitioning_config import GraphPartitioningConfig, QUBOPartitioningConfig
from ._portfolio import PortfolioAllocationProblem, PortfolioSelectionProblem
from ._matching import (
    MaxWeightMatchingProblem,
    check_matching_matrix,
    is_valid_matching,
)
from ._types import GraphProblemTypes
from ._routing import (
    BinaryBlockConfig,
    CVRPProblem,
    RoutingInstance,
    TSPProblem,
    compute_binary_block_config,
    create_tsp_qubo,
    build_cvrp_block_structure,
    is_valid_tsp_tour,
    parse_tsplib_file,
    parse_vrp_solution,
    compute_tour_cost,
)

__all__ = [
    "BinaryBlockConfig",
    "BinaryOptimizationProblem",
    "CVRPProblem",
    "CommunityDecomposer",
    "MolecularProblem",
    "GraphPartitioningConfig",
    "GraphProblemBase",
    "GraphProblemTypes",
    "HamiltonianProblem",
    "LinearConstraint",
    "MaxCliqueProblem",
    "MaxCutProblem",
    "MaxIndependentSetProblem",
    "MaxWeightCycleProblem",
    "MaxWeightMatchingProblem",
    "MinVertexCoverProblem",
    "PortfolioAllocationProblem",
    "PortfolioSelectionProblem",
    "QAOAProblem",
    "QUBOPartitioningConfig",
    "RoutingInstance",
    "TSPProblem",
    "compute_binary_block_config",
    "check_matching_matrix",
    "create_tsp_qubo",
    "build_cvrp_block_structure",
    "draw_graph_solution_nodes",
    "draw_partitions",
    "is_valid_matching",
    "is_valid_tsp_tour",
    "parse_tsplib_file",
    "parse_vrp_solution",
    "compute_tour_cost",
]


def __getattr__(name: str) -> Any:
    """Resolve :class:`CommunityDecomposer` on first access."""
    if name == "CommunityDecomposer":
        import_optional(
            "hybrid", extra="qubo-decompose", capability="CommunityDecomposer"
        )
        from ._community_decomposer import CommunityDecomposer

        globals()[name] = CommunityDecomposer
        return CommunityDecomposer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
