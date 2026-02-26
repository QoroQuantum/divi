# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Shared problem constants and helpers for QAOA/PCE e2e-style tests."""

import dimod
import networkx as nx
import numpy as np

QUBO_MATRIX = np.array(
    [
        [-3.0, 4.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, -3.0],
    ]
)
# 3-variable QUBO. Optimal solution: [1, 0, 1] with energy -6.0.

QUBO_SOLUTION = np.array([1, 0, 1], dtype=np.int32)
# Known optimal assignment for ``QUBO_MATRIX``.

PCE_QUBO_MATRIX = np.array(
    [
        [1.0, -0.2, 0.0],
        [-0.2, 0.5, 0.0],
        [0.0, 0.0, -1.0],
    ]
)
# 3-variable QUBO compatible with PCE dense encoding (masks [1,2,3]).
# Optimal solution: [0, 0, 1] with energy -1.0. Representable by parity
# encoding (state "11" → parities [1,1,0] → x = [0,0,1]).
# QUBO_MATRIX cannot be used for PCE because its optimum [1, 0, 1] requires
# parity vector [0, 1, 0] which violates the XOR constraint (mask 3 = 1 XOR 2).

PCE_QUBO_SOLUTION = np.array([0, 0, 1], dtype=np.int32)
# Known optimal assignment for ``PCE_QUBO_MATRIX``.

HUBO_CUBIC = {
    (0,): -3.0,
    (1,): -3.0,
    (2,): -3.0,
    (0, 1, 2): 2.0,
}
# 3-variable cubic HUBO. Optimal solutions found via brute-force exact_hubo_minima.


def exact_hubo_minima(hubo: dict[tuple[int, ...], float], n_vars: int):
    """Brute-force exact minima for small HUBO fixtures."""
    best_energy = float("inf")
    best_assignments = []
    for mask in range(1 << n_vars):
        assignment = np.array([(mask >> i) & 1 for i in range(n_vars)], dtype=np.int32)
        energy = 0.0
        for term, coeff in hubo.items():
            if len(term) == 0:
                energy += coeff
                continue
            prod = 1.0
            for idx in term:
                prod *= assignment[idx]
            energy += coeff * prod

        if energy < best_energy - 1e-12:
            best_energy = energy
            best_assignments = [assignment]
        elif abs(energy - best_energy) <= 1e-12:
            best_assignments.append(assignment)

    return best_energy, best_assignments


def make_bull_graph() -> nx.Graph:
    """Canonical small graph used in QAOA e2e tests."""
    return nx.bull_graph()


def make_string_node_graph() -> nx.Graph:
    """Simple string-labeled graph for QAOA string-label e2e test."""
    graph = nx.Graph()
    graph.add_nodes_from(["node0", "node1", "node2"])
    graph.add_edges_from([("node0", "node1"), ("node1", "node2")])
    return graph


def make_bqm_minimize() -> dimod.BinaryQuadraticModel:
    """BQM for minimization checks: x=1, y=-2, z=3, w=-1."""
    return dimod.BinaryQuadraticModel(
        {"x": 1, "y": -2, "z": 3, "w": -1}, {}, 0.0, dimod.Vartype.BINARY
    )


def make_bqm_maximize() -> dimod.BinaryQuadraticModel:
    """BQM used via minimization of negated maximize objective."""
    return dimod.BinaryQuadraticModel(
        {"x": -1, "y": 2, "z": -3, "w": 1}, {}, 0.0, dimod.Vartype.BINARY
    )
