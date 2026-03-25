# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Problem classes for QAOA-compatible quantum optimization."""

from divi.qprog.problems._base import (
    QAOAProblem,
    draw_graph_solution_nodes,
)
from divi.qprog.problems._binary import BinaryOptimizationProblem
from divi.qprog.problems._graphs import (
    EdgePartitioningProblem,
    MaxCliqueProblem,
    MaxCutProblem,
    MaxIndependentSetProblem,
    MaxWeightCycleProblem,
    MinVertexCoverProblem,
)
from divi.qprog.problems._routing import (
    CVRPProblem,
    TSPProblem,
    binary_block_config,
    cvrp_block_structure,
    is_valid_tsp_tour,
    parse_vrp_file,
    parse_vrp_solution,
    tour_cost,
)
