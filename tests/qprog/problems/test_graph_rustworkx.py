# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Rustworkx graph problems identify each node by its node index.

Every rustworkx graph is checked against its networkx twin: a graph whose
integer node labels are the rustworkx node indices, inserted in the same order.
The twin must give the same Hamiltonians, wire labels, decoded solutions and
energies, whatever the rustworkx node payloads are.
"""

import importlib.util
import itertools

import networkx as nx
import numpy as np
import pytest
import rustworkx as rx
from qiskit.quantum_info import SparsePauliOp

from divi.qprog.problems import (
    GraphPartitioningConfig,
    MaxCliqueProblem,
    MaxCutProblem,
    MaxIndependentSetProblem,
    MaxWeightCycleProblem,
    MinVertexCoverProblem,
    draw_graph_solution_nodes,
)
from divi.qprog.problems._graph_hamiltonians import edges_to_wires
from divi.qprog.workflows import PartitioningProgramEnsemble
from tests._helpers import exact_match
from tests.qprog.problems._helpers import nx_twin as _nx_twin

_NODE_PROBLEMS = [
    MaxCutProblem,
    MaxCliqueProblem,
    MaxIndependentSetProblem,
    MinVertexCoverProblem,
]
_FEASIBILITY_PROBLEMS = [
    MaxCliqueProblem,
    MaxIndependentSetProblem,
    MinVertexCoverProblem,
]


def _generator_graph() -> rx.PyGraph:
    """Generator output: every node payload is ``None``."""
    graph = rx.generators.cycle_graph(5)
    graph.add_edge(0, 2, None)
    return graph


def _removed_node_graph() -> rx.PyGraph:
    """Node indices ``0, 1, 3, 4, 5, 6`` after removing node 2."""
    graph = rx.PyGraph()
    graph.add_nodes_from(range(7))
    graph.add_edges_from_no_data(
        [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 0), (1, 4)]
    )
    graph.remove_node(2)
    return graph


def _duplicate_payload_graph() -> rx.PyGraph:
    """Every node carries the same payload."""
    graph = rx.PyGraph()
    graph.add_nodes_from(["same"] * 5)
    graph.add_edges_from_no_data([(0, 1), (1, 2), (2, 3), (3, 4), (0, 3)])
    return graph


def _mixed_edge_payload_graph() -> rx.PyGraph:
    """Edges carry numeric, dict and ``None`` payloads."""
    graph = rx.PyGraph()
    graph.add_nodes_from(list("abcde"))
    graph.add_edges_from(
        [(4, 3, 2.0), (3, 0, {"weight": 3.0}), (0, 1, None), (2, 4, 1.5)]
    )
    return graph


_RX_GRAPHS = [
    pytest.param(_generator_graph, id="none-payloads"),
    pytest.param(_removed_node_graph, id="removed-node"),
    pytest.param(_duplicate_payload_graph, id="duplicate-payloads"),
    pytest.param(_mixed_edge_payload_graph, id="mixed-edge-payloads"),
]


def _canonical(spo: SparsePauliOp) -> SparsePauliOp:
    return spo.simplify(atol=1e-12).sort()


def _assert_same_operator(actual: SparsePauliOp, expected: SparsePauliOp) -> None:
    actual, expected = _canonical(actual), _canonical(expected)
    assert actual.paulis == expected.paulis
    np.testing.assert_allclose(actual.coeffs, expected.coeffs, atol=1e-12)


def _bitstrings(n: int, count: int = 16, seed: int = 0) -> list[str]:
    rng = np.random.default_rng(seed)
    return ["".join(map(str, rng.integers(0, 2, n))) for _ in range(count)]


def _assert_problems_match(rx_problem, nx_problem) -> None:
    _assert_same_operator(rx_problem.cost_hamiltonian, nx_problem.cost_hamiltonian)
    _assert_same_operator(rx_problem.mixer_hamiltonian, nx_problem.mixer_hamiltonian)
    assert rx_problem.loss_constant == pytest.approx(nx_problem.loss_constant)
    assert rx_problem.wire_labels == nx_problem.wire_labels
    for bitstring in _bitstrings(len(rx_problem.wire_labels)):
        assert rx_problem.decode_fn(bitstring) == nx_problem.decode_fn(bitstring)
        solution = [int(bit) for bit in bitstring]
        assert rx_problem.evaluate_global_solution(solution) == pytest.approx(
            nx_problem.evaluate_global_solution(solution)
        )


@pytest.mark.parametrize("problem_cls", _NODE_PROBLEMS)
@pytest.mark.parametrize("make_graph", _RX_GRAPHS)
def test_rx_nodes_are_identified_by_index(problem_cls, make_graph):
    graph = make_graph()
    indices = list(graph.node_indexes())

    problem = problem_cls(graph)

    assert problem.cost_hamiltonian.num_qubits == len(indices)
    assert problem.wire_labels == tuple(sorted(indices))
    assert problem.decode_fn("1" * len(indices)) == sorted(indices)
    for qubit, node in enumerate(sorted(indices)):
        one_hot = "".join("1" if i == qubit else "0" for i in range(len(indices)))
        assert problem.decode_fn(one_hot) == [node]


@pytest.mark.parametrize("use_constrained_mixer", [True, False])
@pytest.mark.parametrize("problem_cls", _NODE_PROBLEMS)
@pytest.mark.parametrize("make_graph", _RX_GRAPHS)
def test_rx_graph_matches_its_nx_twin(problem_cls, make_graph, use_constrained_mixer):
    graph = make_graph()

    _assert_problems_match(
        problem_cls(graph, use_constrained_mixer=use_constrained_mixer),
        problem_cls(_nx_twin(graph), use_constrained_mixer=use_constrained_mixer),
    )


@pytest.mark.parametrize("problem_cls", _FEASIBILITY_PROBLEMS)
@pytest.mark.parametrize("make_graph", _RX_GRAPHS)
def test_rx_feasibility_and_energy_match_nx_twin(problem_cls, make_graph):
    graph = make_graph()
    rx_problem, nx_problem = problem_cls(graph), problem_cls(_nx_twin(graph))

    for bits in itertools.product("01", repeat=graph.num_nodes()):
        bitstring = "".join(bits)
        assert rx_problem.is_feasible(bitstring) == nx_problem.is_feasible(bitstring)
        assert rx_problem.compute_energy(bitstring) == nx_problem.compute_energy(
            bitstring
        )


_PARALLEL_EDGES = [(0, 1), (1, 0), (0, 1), (1, 2)]


def _rx_multigraph() -> rx.PyGraph:
    multi = rx.PyGraph()
    multi.add_nodes_from(range(3))
    multi.add_edges_from_no_data(_PARALLEL_EDGES)
    return multi


@pytest.mark.parametrize(
    "make_multi",
    [_rx_multigraph, lambda: nx.MultiGraph(_PARALLEL_EDGES)],
    ids=["rx", "nx-multigraph"],
)
@pytest.mark.parametrize("problem_cls", _NODE_PROBLEMS)
def test_parallel_edges_count_once(problem_cls, make_multi):
    simple = nx.Graph([(0, 1), (1, 2)])

    _assert_problems_match(problem_cls(make_multi()), problem_cls(simple))


def _weighted_digraph(payload_kind: str, *, remove: int | None = None) -> rx.PyDiGraph:
    """Weighted digraph whose edges are inserted out of source order."""
    edges = [(2, 0, 1.3), (0, 1, 1.5), (1, 2, 0.7), (0, 2, 2.0), (2, 3, 1.1)]
    edges += [(3, 0, 0.9), (1, 0, 0.5), (3, 4, 1.2), (4, 0, 1.4)]
    graph = rx.PyDiGraph()
    graph.add_nodes_from([None] * 5)
    wrap = {"numeric": lambda w: w, "dict": lambda w: {"weight": w, "colour": "red"}}
    graph.add_edges_from([(u, v, wrap[payload_kind](w)) for u, v, w in edges])
    if remove is not None:
        graph.remove_node(remove)
    return graph


@pytest.mark.parametrize("remove", [None, 1], ids=["all-nodes", "removed-node"])
@pytest.mark.parametrize("payload_kind", ["numeric", "dict"])
@pytest.mark.parametrize("use_constrained_mixer", [True, False])
def test_rx_max_weight_cycle_matches_its_nx_twin(
    payload_kind, remove, use_constrained_mixer
):
    graph = _weighted_digraph(payload_kind, remove=remove)
    twin = _nx_twin(graph)

    rx_problem = MaxWeightCycleProblem(
        graph, use_constrained_mixer=use_constrained_mixer
    )
    nx_problem = MaxWeightCycleProblem(
        twin, use_constrained_mixer=use_constrained_mixer
    )

    _assert_problems_match(rx_problem, nx_problem)
    assert rx_problem.metadata == nx_problem.metadata
    assert set(rx_problem.metadata.values()) == set(graph.edge_list())


def test_parallel_rx_digraph_edges_keep_the_last_weight():
    multi = rx.PyDiGraph()
    multi.add_nodes_from([None] * 3)
    multi.add_edges_from([(0, 1, 5.0), (1, 2, 1.5), (2, 0, 1.2), (0, 1, 2.0)])
    simple = nx.DiGraph()
    simple.add_weighted_edges_from([(0, 1, 5.0), (1, 2, 1.5), (2, 0, 1.2)])
    simple.add_edge(0, 1, weight=2.0)

    _assert_problems_match(MaxWeightCycleProblem(multi), MaxWeightCycleProblem(simple))


def _directed_cycle(backend: str):
    edges = [(i, (i + 1) % 4, 1.5) for i in range(4)]
    if backend == "rx":
        graph = rx.PyDiGraph()
        graph.add_nodes_from([None] * 4)
        graph.add_edges_from(edges)
        return graph
    graph = nx.DiGraph()
    graph.add_weighted_edges_from(edges)
    return graph


@pytest.mark.parametrize("backend", ["rx", "nx"])
def test_max_weight_cycle_rejects_partitioning_config(backend):
    with pytest.raises(
        ValueError,
        match=exact_match(
            "MaxWeightCycleProblem does not support graph partitioning: its "
            "variables are edges, while partitioning splits the graph by node. "
            "Build it without 'config'."
        ),
    ):
        MaxWeightCycleProblem(
            _directed_cycle(backend),
            config=GraphPartitioningConfig(max_n_nodes_per_cluster=2),
        )


@pytest.mark.parametrize(
    "graph", [nx.DiGraph([(0, 1)]), rx.generators.directed_path_graph(2)]
)
@pytest.mark.parametrize("problem_cls", _NODE_PROBLEMS)
def test_node_problems_reject_directed_graphs(problem_cls, graph):
    with pytest.raises(
        TypeError, match=r"^Expected an undirected graph \(nx.Graph or rx.PyGraph\)"
    ):
        problem_cls(graph)


@pytest.mark.parametrize("graph", [nx.cycle_graph(3), rx.generators.cycle_graph(3)])
def test_max_weight_cycle_rejects_undirected_graphs(graph):
    with pytest.raises(
        TypeError, match=r"^Expected a directed graph \(nx.DiGraph or rx.PyDiGraph\)"
    ):
        MaxWeightCycleProblem(graph)


@pytest.mark.parametrize("backend", ["rx", "nx"])
def test_max_weight_cycle_rejects_self_loops(backend):
    graph = _directed_cycle(backend)
    if backend == "rx":
        graph.add_edge(0, 0, 2.0)
    else:
        graph.add_edge(0, 0, weight=2.0)

    with pytest.raises(ValueError, match=exact_match("Graph contains self-loops")):
        MaxWeightCycleProblem(graph)


@pytest.mark.parametrize("graph", [nx.empty_graph(3), rx.generators.empty_graph(3)])
def test_edgeless_graphs_are_rejected_alike(graph):
    with pytest.raises(ValueError, match="Hamiltonian contains only constant terms"):
        MaxCutProblem(graph)


def test_solution_drawing_highlights_rx_nodes_by_index(mocker):
    mocker.patch("matplotlib.pyplot.show")
    draw_nodes = mocker.spy(nx, "draw_networkx_nodes")
    graph = _removed_node_graph()

    draw_graph_solution_nodes(graph, [0, 3])
    draw_graph_solution_nodes(_nx_twin(graph), [0, 3])

    rx_call, nx_call = draw_nodes.call_args_list
    assert rx_call.kwargs["node_color"] == nx_call.kwargs["node_color"]
    assert rx_call.kwargs["node_color"].count("red") == 2


def test_rx_max_weight_cycle_wires_name_edges_by_index():
    graph = _weighted_digraph("numeric", remove=1)

    wires = edges_to_wires(graph)

    assert set(wires) == set(graph.edge_list())
    assert sorted(wires.values()) == list(range(graph.num_edges()))


def _gnp_graph_with_hole(seed: int) -> rx.PyGraph:
    """String-payload random graph with one node removed."""
    graph = rx.undirected_gnp_random_graph(11, 0.5, seed=seed)
    for index in graph.node_indexes():
        graph[index] = f"n{index}"
    graph.remove_node(3)
    return graph


_ALGORITHMS = [
    "spectral",
    "kernighan_lin",
    pytest.param(
        "metis",
        marks=pytest.mark.skipif(
            importlib.util.find_spec("pymetis") is None, reason="pymetis not installed"
        ),
    ),
]


def _partitioned(problem_cls, graph, algorithm):
    problem = problem_cls(
        graph,
        config=GraphPartitioningConfig(
            max_n_nodes_per_cluster=4, partitioning_algorithm=algorithm
        ),
    )
    return problem, problem.decompose()


@pytest.mark.filterwarnings("ignore:Heuristic-risk graph partitioning objective")
@pytest.mark.parametrize("seed", [2, 5])
@pytest.mark.parametrize("algorithm", _ALGORITHMS)
@pytest.mark.parametrize("problem_cls", [MaxCutProblem, MaxIndependentSetProblem])
def test_partitioned_rx_problem_extends_and_scores_like_nx_twin(
    problem_cls, algorithm, seed
):
    """Each sub-problem decodes the candidate's local bits to local indices, and
    stitching them back rebuilds the candidate, scored as on the nx twin."""
    graph = _gnp_graph_with_hole(seed)
    problem, sub_problems = _partitioned(problem_cls, graph, algorithm)
    nx_problem = problem_cls(_nx_twin(graph))
    position = {node: i for i, node in enumerate(graph.node_indexes())}
    candidate = [
        int(bit) for bit in _bitstrings(graph.num_nodes(), count=1, seed=seed)[0]
    ]

    solution = [0] * problem.initial_solution_size()
    for prog_id, sub in sub_problems.items():
        assert isinstance(sub.graph, rx.PyGraph)
        reverse_map = problem._reverse_index_maps[prog_id]
        assert sub.wire_labels == tuple(range(len(reverse_map)))
        local_bits = "".join(
            str(candidate[position[reverse_map[local]]]) for local in sub.wire_labels
        )
        solution = problem.extend_solution(solution, prog_id, sub.decode_fn(local_bits))

    assert solution == candidate
    assert problem.evaluate_global_solution(solution) == pytest.approx(
        nx_problem.evaluate_global_solution(candidate)
    )
    selected, _ = problem.postprocess_candidates([(0.0, solution)])[0]
    assert selected == [
        node for node, bit in zip(graph.node_indexes(), candidate) if bit
    ]


def test_partitioned_rx_ensemble_returns_node_indices(
    default_test_simulator, default_optimizer
):
    graph = _gnp_graph_with_hole(5)
    problem, _ = _partitioned(MaxCutProblem, graph, "spectral")
    ensemble = PartitioningProgramEnsemble(
        problem=problem,
        n_layers=1,
        backend=default_test_simulator,
        optimizer=default_optimizer,
        max_iterations=2,
        seed=1997,
    )

    ensemble.run()
    solution, energy = ensemble.aggregate_results()

    assert set(solution) <= set(graph.node_indexes())
    assert 3 not in solution
    cut_vector = [int(node in solution) for node in graph.node_indexes()]
    assert energy == pytest.approx(problem.evaluate_global_solution(cut_vector))
    assert energy < 0
