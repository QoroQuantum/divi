# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import networkx as nx
import pytest
from qprog_contracts import verify_basic_program_batch_behaviour

from divi.parallel_simulator import ParallelSimulator
from divi.qprog import GraphPartitioningQAOA, GraphProblem, Optimizers

problem_args = {
    "graph": nx.erdos_renyi_graph(15, 0.2, seed=1997),
    "graph_problem": GraphProblem.MAXCUT,
    "n_layers": 1,
    "n_clusters": 2,
    "optimizer": Optimizers.NELDER_MEAD,
    "max_iterations": 10,
    "backend": ParallelSimulator(shots=5000),
}


@pytest.fixture
def node_partitioning_qaoa():
    return GraphPartitioningQAOA(**problem_args)


def test_verify_basic_behaviour(mocker, node_partitioning_qaoa):
    verify_basic_program_batch_behaviour(mocker, node_partitioning_qaoa)

    mock_program = mocker.MagicMock()
    mock_program.losses = [{0: -1.0}]

    node_partitioning_qaoa.programs = {"dummy": mock_program}

    with pytest.raises(RuntimeError, match="Not all final probabilities"):
        node_partitioning_qaoa.aggregate_results()


def test_correct_initialization(node_partitioning_qaoa):
    assert node_partitioning_qaoa.main_graph == problem_args["graph"]
    assert node_partitioning_qaoa.is_edge_problem == False
    assert node_partitioning_qaoa.n_clusters == problem_args["n_clusters"]


def test_fail_if_no_qubits_or_clusters_provided():
    with pytest.raises(
        ValueError, match="One of `n_qubits` and `n_clusters` must be provided."
    ):
        GraphPartitioningQAOA(
            graph_problem=GraphProblem.MAXCUT,
            graph=None,
            n_layers=1,
            optimizer=Optimizers.NELDER_MEAD,
            max_iterations=10,
            backend=ParallelSimulator(),
        )


def test_correct_number_of_programs_created(mocker, node_partitioning_qaoa):
    mocker.patch("divi.qprog.QAOA")

    node_partitioning_qaoa.create_programs()

    assert len(node_partitioning_qaoa.programs) == problem_args["n_clusters"]

    # Assert common values propagated to all programs
    for program in node_partitioning_qaoa.programs.values():
        assert program.optimizer == Optimizers.NELDER_MEAD
        assert program.max_iterations == 10
        assert isinstance(program.backend, ParallelSimulator)
        assert program.backend.shots == 5000

    # Need to clean up at the end of the test
    node_partitioning_qaoa._live.stop()


def test_results_aggregated_correctly(node_partitioning_qaoa):
    node_partitioning_qaoa.create_programs()

    mock_program_1_nodes = node_partitioning_qaoa.programs[0].problem.number_of_nodes()
    node_partitioning_qaoa.programs[0].losses = [{0: -1.0}]
    node_partitioning_qaoa.programs[0].probs = {
        "0_0": {"0" * mock_program_1_nodes: 0.9, "1" * mock_program_1_nodes: 0.1}
    }

    mock_program_2_nodes = node_partitioning_qaoa.programs[1].problem.number_of_nodes()
    node_partitioning_qaoa.programs[1].losses = [{0: -2.0, 1: -3.0}]
    node_partitioning_qaoa.programs[1].probs = {
        "0_0": {"0" * mock_program_2_nodes: 0.9, "1" * mock_program_2_nodes: 0.1},
        "1_0": {"0" * mock_program_2_nodes: 0.2, "1" * mock_program_2_nodes: 0.8},
    }

    solution = node_partitioning_qaoa.aggregate_results()

    assert solution.count(0) == mock_program_1_nodes
    assert solution.count(1) == mock_program_2_nodes
    assert len(solution) == node_partitioning_qaoa.main_graph.number_of_nodes()

    # Need to clean up at the end of the test
    node_partitioning_qaoa._live.stop()
