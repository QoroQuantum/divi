# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Problem classes for quantum optimisation and ground-state estimation."""

from typing import Any

from divi._optional import import_optional

from ._base import QAOAProblem
from ._binary import BinaryOptimizationProblem
from ._graph_partitioning_utils import GraphPartitioningConfig, draw_partitions
from ._graphs import (
    MaxCliqueProblem,
    MaxCutProblem,
    MaxIndependentSetProblem,
    MaxWeightCycleProblem,
    MinVertexCoverProblem,
    draw_graph_solution_nodes,
)
from ._hamiltonian import HamiltonianProblem, MolecularProblem
from ._matching import (
    MaxWeightMatchingProblem,
    check_matching_matrix,
    is_valid_matching,
)
from ._routing import (
    BinaryBlockConfig,
    CVRPProblem,
    RoutingInstance,
    TSPProblem,
    binary_block_config,
    create_tsp_qubo,
    cvrp_block_structure,
    is_valid_tsp_tour,
    parse_tsplib_file,
    parse_vrp_solution,
    tour_cost,
)

__all__ = [
    "BinaryBlockConfig",
    "BinaryOptimizationProblem",
    "CVRPProblem",
    "CommunityDecomposer",
    "MolecularProblem",
    "GraphPartitioningConfig",
    "HamiltonianProblem",
    "MaxCliqueProblem",
    "MaxCutProblem",
    "MaxIndependentSetProblem",
    "MaxWeightCycleProblem",
    "MaxWeightMatchingProblem",
    "MinVertexCoverProblem",
    "QAOAProblem",
    "RoutingInstance",
    "TSPProblem",
    "binary_block_config",
    "check_matching_matrix",
    "create_tsp_qubo",
    "cvrp_block_structure",
    "draw_graph_solution_nodes",
    "draw_partitions",
    "is_valid_matching",
    "is_valid_tsp_tour",
    "parse_tsplib_file",
    "parse_vrp_solution",
    "tour_cost",
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
