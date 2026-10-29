# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""MaxCut QUBO — Characterize & Solve
======================================

Characterizes a weighted MaxCut QUBO with the Qoro Characterization Service,
then runs QAOA once at the angles the service optimised, skipping the
optimizer's search.

Requirements:
    - ``QORO_API_KEY`` in ``.env`` or environment variable.
"""

import networkx as nx
import numpy as np

from divi.backends import QoroService
from divi.backends.characterization import (
    CharacterizationOptions,
    characterize_and_validate,
)
from divi.qprog import QAOA
from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer
from divi.qprog.problems import BinaryOptimizationProblem
from tutorials._backend import get_backend

SHOTS = 20000


def main() -> None:
    # Weighted Petersen graph. MaxCut as a QUBO to minimise: E(x) = -cut(x),
    # each edge contributing -w(x_u + x_v - 2 x_u x_v).
    graph = nx.petersen_graph()
    rng = np.random.default_rng(42)
    n = graph.number_of_nodes()
    Q = np.zeros((n, n))
    for u, v in graph.edges():
        w = round(rng.uniform(0.5, 3.0), 2)
        Q[u, v] = Q[v, u] = w
        Q[u, u] -= w
        Q[v, v] -= w
    problem = BinaryOptimizationProblem(Q)

    report = characterize_and_validate(
        problem,
        reference_states=["0" * n],
        service=QoroService(),
        options=CharacterizationOptions(ansatz={"layers": 1}),
    )
    report.display()

    # One evaluation at the service's p=1 angles: no optimizer iterations.
    qaoa = QAOA(
        problem,
        n_layers=1,
        optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
        max_iterations=1,
        backend=get_backend(shots=SHOTS),
    )
    qaoa.run(initial_params=report.qaoa_initial_params(layers=1))

    best = report.classical["best_energy"]
    found = problem.compute_energy(qaoa.solution_bitstring)
    print(f"QAOA found energy {found:.2f}; the classical optimum is {best:.2f}.")


if __name__ == "__main__":
    main()
