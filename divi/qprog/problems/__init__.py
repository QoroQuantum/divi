# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Problem classes for QAOA-compatible quantum optimization."""

from divi.qprog.problems._base import QAOAProblem
from divi.qprog.problems._binary import BinaryOptimizationProblem
from divi.qprog.problems._graphs import (
    MaxCliqueProblem,
    MaxCutProblem,
    MaxIndependentSetProblem,
    MaxWeightCycleProblem,
    MinVertexCoverProblem,
    draw_graph_solution_nodes,
)
from divi.qprog.problems._matching import (
    MaxWeightMatchingProblem,
    check_matching_matrix,
    is_valid_matching,
)
from divi.qprog.problems._graph_partitioning_utils import (
    GraphPartitioningConfig,
    draw_partitions,
)
from divi.qprog.problems._routing import (
    BinaryBlockConfig,
    CVRPProblem,
    TSPProblem,
    VRPInstance,
    binary_block_config,
    cvrp_block_structure,
    is_valid_tsp_tour,
    parse_vrp_file,
    parse_vrp_solution,
    tour_cost,
)

__all__ = [
    "BinaryBlockConfig",
    "BinaryOptimizationProblem",
    "CVRPProblem",
    "GraphPartitioningConfig",
    "MaxCliqueProblem",
    "MaxCutProblem",
    "MaxIndependentSetProblem",
    "MaxWeightCycleProblem",
    "MaxWeightMatchingProblem",
    "MinVertexCoverProblem",
    "QAOAProblem",
    "TSPProblem",
    "VRPInstance",
    "binary_block_config",
    "check_matching_matrix",
    "cvrp_block_structure",
    "draw_graph_solution_nodes",
    "draw_partitions",
    "is_valid_matching",
    "is_valid_tsp_tour",
    "parse_vrp_file",
    "parse_vrp_solution",
    "tour_cost",
]
