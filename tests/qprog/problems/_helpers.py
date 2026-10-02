# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Shared problem constants and helpers for QAOA/PCE e2e-style tests."""

import contextlib
import itertools
import signal

import dimod
import networkx as nx
import numpy as np
import pytest
import rustworkx as rx

from divi.qprog.problems import BinaryOptimizationProblem


def nx_twin(graph: rx.PyGraph | rx.PyDiGraph) -> nx.Graph:
    """networkx graph labelled by ``graph``'s node indices, with the same edges."""
    twin = nx.DiGraph() if isinstance(graph, rx.PyDiGraph) else nx.Graph()
    twin.add_nodes_from(graph.node_indexes())
    for left, right, payload in graph.weighted_edge_list():
        attrs = payload if isinstance(payload, dict) else {"weight": payload}
        twin.add_edge(left, right, **{k: v for k, v in attrs.items() if v is not None})
    return twin


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

KNOWN_QUBO = {
    (0, 0): -0.5,
    (1, 1): 1,
    (0, 1): -2,
    (2, 2): 1,
    (3, 3): 1,
    (2, 3): 2,
}
# 4-variable QUBO. Optimal solution [1, 1, 0, 0] with energy -1.5.


def make_known_qubo_bqm() -> dimod.BinaryQuadraticModel:
    """BQM for :data:`KNOWN_QUBO` (optimal [1, 1, 0, 0], energy -1.5)."""
    return dimod.BinaryQuadraticModel.from_qubo(KNOWN_QUBO)


ZERO_OFFSET_QUBO = {(0, 0): -0.5, (1, 1): 1, (0, 1): -2}
# 2-variable QUBO with no constant offset; the all-zeros solution has energy 0.


def make_zero_offset_bqm() -> dimod.BinaryQuadraticModel:
    """BQM for :data:`ZERO_OFFSET_QUBO` (all-zeros has energy 0)."""
    return dimod.BinaryQuadraticModel.from_qubo(ZERO_OFFSET_QUBO)


@contextlib.contextmanager
def fail_after(seconds: int):
    """Fail the enclosing test instead of hanging if its body runs too long."""
    if not hasattr(signal, "SIGALRM"):
        pytest.skip("SIGALRM is unavailable on this platform")

    def _expire(_signum, _frame):
        pytest.fail(f"timed out after {seconds} s")

    previous = signal.signal(signal.SIGALRM, _expire)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def make_decomposed_problem(source, *, decomposer_size: int = 2):
    """A ``BinaryOptimizationProblem`` wired with an ``EnergyImpactDecomposer``.

    ``source`` may be any input the problem accepts (BQM, QUBO matrix, …).
    Used by partition-aggregation tests that need a decomposable problem.
    """
    hybrid = pytest.importorskip("hybrid")
    return BinaryOptimizationProblem(
        source, decomposer=hybrid.EnergyImpactDecomposer(size=decomposer_size)
    )


def seed_zero_one_best_probs(ensemble, zeros_prob, ones_prob=None):
    """Seed every program in ``ensemble`` with all-zeros (and optional all-ones)
    ``best_probs``, bypassing circuit execution for aggregation tests.
    """
    for program in ensemble.programs.values():
        n_qubits = program.n_qubits
        probs = {"0" * n_qubits: zeros_prob}
        if ones_prob is not None:
            probs["1" * n_qubits] = ones_prob
        program._results["best_probs"] = {"tag": probs}
        program._losses_history = [{"dummy_loss": 0.0}]


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


def polynomial_value(terms: dict[tuple, float], assignment: dict) -> float:
    """Evaluate HUBO ``terms`` on a ``{variable: bit}`` assignment."""
    return sum(
        coeff * np.prod([assignment[v] for v in key]) for key, coeff in terms.items()
    )


def penalty_at(problem, bitstring: str) -> float:
    """Penalty of ``problem`` at a full bitstring over its canonical variables."""
    variables = problem.canonical_problem.variable_order
    assignment = dict(zip(variables, map(int, bitstring)))
    return polynomial_value(problem.penalty_canonical_problem.terms, assignment)


def n_slack(problem, n_decision: int) -> int:
    """Number of qubits beyond the first ``n_decision`` decision qubits."""
    return problem.cost_hamiltonian.num_qubits - n_decision


def iter_assignments(problem):
    """Yield ``(bitstring, assignment)`` for every basis state of ``problem``."""
    variables = problem.canonical_problem.variable_order
    for bits in itertools.product((0, 1), repeat=len(variables)):
        yield "".join(map(str, bits)), dict(zip(variables, bits))


def assert_penalty_zero_iff_feasible(
    problem, n_decision: int, *, require_feasible: bool = True
) -> None:
    """Every feasible decision setting admits zero penalty; every other costs more.

    The first ``n_decision`` bitstring positions are the decision qubits; the
    penalty of each decision setting is its minimum over the remaining slack.
    """
    variables = problem.canonical_problem.variable_order
    samples = np.array(list(itertools.product((0, 1), repeat=len(variables))))
    column = {v: i for i, v in enumerate(variables)}
    terms = problem.penalty_canonical_problem.terms
    penalty = np.zeros(len(samples))
    for key, coeff in terms.items():
        penalty += coeff * samples[:, [column[v] for v in key]].prod(axis=1)
    zero = 1e-13 * max(map(abs, terms.values()), default=1.0)
    heads = samples[:, :n_decision] @ (1 << np.arange(n_decision)[::-1])
    best = np.full(2**n_decision, np.inf)
    np.minimum.at(best, heads, penalty)
    decisions = [format(h, f"0{n_decision}b") for h in range(2**n_decision)]
    tail = "0" * (len(variables) - n_decision)
    feasible = [problem.is_feasible(d + tail) for d in decisions]
    assert any(feasible) or not require_feasible
    for decision, value, ok in zip(decisions, best, feasible):
        assert (abs(value) <= zero) == ok, decision
        assert value >= -zero


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
