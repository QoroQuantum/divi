# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import networkx as nx
import pytest

from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer
from divi.qprog.problems import MaxWeightMatchingProblem, is_valid_matching
from divi.qprog.variational_quantum_algorithm import (
    SolutionEntry,
    VariationalQuantumAlgorithm,
)
from divi.qprog.workflows import PartitioningProgramEnsemble
from tests.qprog.qprog_contracts import verify_basic_program_ensemble_behaviour

_DEFAULT_OPTIMIZER = ScipyOptimizer(method=ScipyMethod.NELDER_MEAD)


def _make_stub_problem(mocker, solution_size=4):
    """Create a mock QAOAProblem that supports the decomposition protocol."""
    problem = mocker.MagicMock()
    problem.decompose.return_value = {}
    problem.initial_solution_size.return_value = solution_size
    problem.extend_solution.side_effect = lambda current, prog_id, decoded: [
        int(decoded[i]) if i < len(decoded) else current[i] for i in range(len(current))
    ]
    problem.evaluate_global_solution.side_effect = lambda sol: -sum(sol)
    problem.postprocess_candidates.side_effect = lambda candidates, *, strict=False: [
        (list(sol), strict if strict else score) for score, sol in candidates
    ]
    return problem


def _make_mock_program(mocker, best_probs, top_solutions):
    """Create a mock VQA program with the given best_probs and top_solutions."""
    prog = mocker.MagicMock(spec=VariationalQuantumAlgorithm)
    prog.best_probs = best_probs
    prog.losses_history = [1.0]
    prog.results = {"some": "result"}
    prog.has_results.return_value = True
    prog.get_top_solutions.return_value = top_solutions
    return prog


class TestPartitioningProgramEnsemble:
    def test_aggregate_results_calls_problem_hooks(self, mocker, dummy_simulator):
        problem = _make_stub_problem(mocker, solution_size=2)
        ensemble = PartitioningProgramEnsemble(
            problem=problem,
            n_layers=1,
            backend=dummy_simulator,
            optimizer=_DEFAULT_OPTIMIZER,
        )
        ensemble.create_programs()

        candidate = SolutionEntry(bitstring="11", prob=0.8, decoded=[1, 1])
        prog = _make_mock_program(
            mocker, best_probs={"11": 0.8}, top_solutions=[candidate]
        )
        ensemble._programs["A"] = prog

        result = ensemble.aggregate_results()

        assert result == ([1, 1], -2)
        problem.evaluate_global_solution.assert_called()
        problem.postprocess_candidates.assert_called_once()

    def test_get_top_solutions_calls_problem_hooks(self, mocker, dummy_simulator):
        problem = _make_stub_problem(mocker, solution_size=2)
        ensemble = PartitioningProgramEnsemble(
            problem=problem,
            n_layers=1,
            backend=dummy_simulator,
            optimizer=_DEFAULT_OPTIMIZER,
        )
        ensemble.create_programs()

        candidate = SolutionEntry(bitstring="11", prob=0.8, decoded=[1, 1])
        prog = _make_mock_program(
            mocker, best_probs={"11": 0.8}, top_solutions=[candidate]
        )
        ensemble._programs["A"] = prog

        results = ensemble.get_top_solutions(n=1)

        assert len(results) == 1
        assert results[0][0] == [1, 1]
        assert results[0][1] == -2.0

    def test_aggregate_results_raises_if_no_programs(self, mocker, dummy_simulator):
        problem = _make_stub_problem(mocker)
        ensemble = PartitioningProgramEnsemble(
            problem=problem,
            n_layers=1,
            backend=dummy_simulator,
            optimizer=_DEFAULT_OPTIMIZER,
        )
        with pytest.raises(RuntimeError, match="No programs to aggregate"):
            ensemble.aggregate_results()

    def test_aggregate_results_raises_if_not_run(self, mocker, dummy_simulator):
        problem = _make_stub_problem(mocker)
        ensemble = PartitioningProgramEnsemble(
            problem=problem,
            n_layers=1,
            backend=dummy_simulator,
            optimizer=_DEFAULT_OPTIMIZER,
        )
        ensemble.create_programs()

        prog = mocker.MagicMock(spec=VariationalQuantumAlgorithm)
        prog.has_results.return_value = False
        ensemble._programs["A"] = prog

        with pytest.raises(RuntimeError, match="Some/All programs have no results"):
            ensemble.aggregate_results()

    def test_aggregate_raises_if_no_best_probs(self, mocker, dummy_simulator):
        problem = _make_stub_problem(mocker)
        ensemble = PartitioningProgramEnsemble(
            problem=problem,
            n_layers=1,
            backend=dummy_simulator,
            optimizer=_DEFAULT_OPTIMIZER,
        )
        ensemble.create_programs()

        prog = mocker.MagicMock(spec=VariationalQuantumAlgorithm)
        prog.has_results.return_value = True
        prog.best_probs = {}
        ensemble._programs["A"] = prog

        with pytest.raises(
            RuntimeError,
            match="Not all final probabilities computed yet",
        ):
            ensemble.aggregate_results()

    def test_get_top_solutions_raises_on_invalid_n(self, mocker, dummy_simulator):
        problem = _make_stub_problem(mocker)
        ensemble = PartitioningProgramEnsemble(
            problem=problem,
            n_layers=1,
            backend=dummy_simulator,
            optimizer=_DEFAULT_OPTIMIZER,
        )
        ensemble.create_programs()

        with pytest.raises(ValueError, match="n must be >= 1"):
            ensemble.get_top_solutions(n=0)

    def test_get_top_solutions_raises_if_not_run(self, mocker, dummy_simulator):
        problem = _make_stub_problem(mocker)
        ensemble = PartitioningProgramEnsemble(
            problem=problem,
            n_layers=1,
            backend=dummy_simulator,
            optimizer=_DEFAULT_OPTIMIZER,
        )
        ensemble.create_programs()

        prog = mocker.MagicMock(spec=VariationalQuantumAlgorithm)
        prog.has_results.return_value = False
        ensemble._programs["A"] = prog

        with pytest.raises(RuntimeError, match="Some/All programs have no results"):
            ensemble.get_top_solutions(n=1)

    def test_verify_basic_behaviour(self, mocker, dummy_simulator):
        problem = _make_stub_problem(mocker)
        ensemble = PartitioningProgramEnsemble(
            problem=problem,
            n_layers=1,
            backend=dummy_simulator,
            optimizer=_DEFAULT_OPTIMIZER,
        )
        verify_basic_program_ensemble_behaviour(mocker, ensemble)


def _attach_program_with_candidates(ensemble, mocker, decoded_candidates):
    """Wire a single mock program emitting the given decoded candidates."""
    candidates = [
        SolutionEntry(
            bitstring="".join(str(b) for b in decoded),
            prob=1.0 / (rank + 1),
            decoded=list(decoded),
        )
        for rank, decoded in enumerate(decoded_candidates)
    ]
    prog = _make_mock_program(
        mocker,
        best_probs={c.bitstring: c.prob for c in candidates},
        top_solutions=candidates,
    )
    ensemble._programs["A"] = prog
    return prog


class TestStrictAggregation:
    """Strict mode is passed to problem-specific post-processing only."""

    def test_aggregate_results_uses_default_postprocessing(
        self, mocker, dummy_simulator
    ):
        problem = _make_stub_problem(mocker, solution_size=2)
        ensemble = PartitioningProgramEnsemble(
            problem=problem,
            n_layers=1,
            backend=dummy_simulator,
            optimizer=_DEFAULT_OPTIMIZER,
        )
        ensemble.create_programs()
        _attach_program_with_candidates(ensemble, mocker, [[1, 1]])

        result = ensemble.aggregate_results()

        assert result == ([1, 1], -2)
        problem.postprocess_candidates.assert_called_once_with([(-2, [1, 1])])

    def test_aggregate_results_warns_when_postprocessing_rejects_everything(
        self, mocker, dummy_simulator
    ):
        problem = _make_stub_problem(mocker, solution_size=2)
        problem.postprocess_candidates.side_effect = None
        problem.postprocess_candidates.return_value = []
        ensemble = PartitioningProgramEnsemble(
            problem=problem,
            n_layers=1,
            backend=dummy_simulator,
            optimizer=_DEFAULT_OPTIMIZER,
        )
        ensemble.create_programs()
        _attach_program_with_candidates(ensemble, mocker, [[1, 1]])

        with pytest.warns(
            UserWarning,
            match=(
                r"beam_width=3 and n_partition_candidates=5.*"
                r"Pass wider beam search parameters to "
                r"get_top_solutions\(\.\.\., strict=True\)"
            ),
        ):
            result = ensemble.aggregate_results(
                beam_width=3,
                n_partition_candidates=5,
            )

        assert result is None

    def test_aggregate_results_warning_explains_default_candidate_count(
        self, mocker, dummy_simulator
    ):
        problem = _make_stub_problem(mocker, solution_size=2)
        problem.postprocess_candidates.side_effect = None
        problem.postprocess_candidates.return_value = []
        ensemble = PartitioningProgramEnsemble(
            problem=problem,
            n_layers=1,
            backend=dummy_simulator,
            optimizer=_DEFAULT_OPTIMIZER,
        )
        ensemble.create_programs()
        _attach_program_with_candidates(ensemble, mocker, [[1, 1]])

        with pytest.warns(
            UserWarning,
            match=(
                r"beam_width=3 and n_partition_candidates=None.*"
                r"Pass wider beam search parameters to "
                r"get_top_solutions\(\.\.\., strict=True\)"
            ),
        ):
            result = ensemble.aggregate_results(beam_width=3)

        assert result is None

    def test_get_top_solutions_passes_strict_to_postprocess(
        self, mocker, dummy_simulator
    ):
        problem = _make_stub_problem(mocker, solution_size=2)
        ensemble = PartitioningProgramEnsemble(
            problem=problem,
            n_layers=1,
            backend=dummy_simulator,
            optimizer=_DEFAULT_OPTIMIZER,
        )
        ensemble.create_programs()
        _attach_program_with_candidates(ensemble, mocker, [[1, 1], [1, 0]])

        ensemble.get_top_solutions(n=2, beam_width=None, strict=True)

        problem.postprocess_candidates.assert_called_once()
        assert problem.postprocess_candidates.call_args.kwargs == {"strict": True}

    def test_get_top_solutions_requests_exact_top_n_without_overfetch(
        self, mocker, dummy_simulator
    ):
        problem = _make_stub_problem(mocker, solution_size=3)
        ensemble = PartitioningProgramEnsemble(
            problem=problem,
            n_layers=1,
            backend=dummy_simulator,
            optimizer=_DEFAULT_OPTIMIZER,
        )
        ensemble.create_programs()
        prog = _attach_program_with_candidates(ensemble, mocker, [[1, 1, 1], [1, 1, 0]])

        ensemble.get_top_solutions(n=1, beam_width=None, strict=True)

        prog.get_top_solutions.assert_called_once_with(n=2**20, include_decoded=True)
        candidates = problem.postprocess_candidates.call_args.args[0]
        assert len(candidates) == 1
        assert candidates[0][1] == [1, 1, 1]

    def test_matching_strict_filters_cross_partition_conflict(
        self, mocker, dummy_simulator
    ):
        graph = nx.Graph()
        graph.add_weighted_edges_from([(0, 1, 10.0), (1, 2, 10.0)])
        problem = MaxWeightMatchingProblem(graph)
        problem._edge_index_maps = {
            "A": [problem._edge_to_qubit[(0, 1)]],
            "B": [problem._edge_to_qubit[(1, 2)]],
        }
        ensemble = PartitioningProgramEnsemble(
            problem=problem,
            n_layers=1,
            backend=dummy_simulator,
            optimizer=_DEFAULT_OPTIMIZER,
        )
        ensemble._programs["A"] = _make_mock_program(
            mocker,
            best_probs={"1": 0.9, "0": 0.1},
            top_solutions=[
                SolutionEntry(bitstring="1", prob=0.9, decoded=[1]),
                SolutionEntry(bitstring="0", prob=0.1, decoded=[0]),
            ],
        )
        ensemble._programs["B"] = _make_mock_program(
            mocker,
            best_probs={"1": 0.9, "0": 0.1},
            top_solutions=[
                SolutionEntry(bitstring="1", prob=0.9, decoded=[1]),
                SolutionEntry(bitstring="0", prob=0.1, decoded=[0]),
            ],
        )

        with pytest.warns(UserWarning, match="No valid matching candidates"):
            assert ensemble.get_top_solutions(n=1, beam_width=None, strict=True) == []

        results = ensemble.get_top_solutions(n=3, beam_width=None, strict=True)
        assert results == [([(0, 1)], 10.0), ([(1, 2)], 10.0)]

    def test_matching_aggregate_warns_when_raw_candidate_is_repaired(
        self, mocker, dummy_simulator
    ):
        graph = nx.Graph()
        graph.add_weighted_edges_from([(0, 1, 10.0), (1, 2, 10.0)])
        problem = MaxWeightMatchingProblem(graph)
        problem._edge_index_maps = {
            "A": [problem._edge_to_qubit[(0, 1)]],
            "B": [problem._edge_to_qubit[(1, 2)]],
        }
        ensemble = PartitioningProgramEnsemble(
            problem=problem,
            n_layers=1,
            backend=dummy_simulator,
            optimizer=_DEFAULT_OPTIMIZER,
        )
        ensemble._programs["A"] = _make_mock_program(
            mocker,
            best_probs={"1": 1.0},
            top_solutions=[SolutionEntry(bitstring="1", prob=1.0, decoded=[1])],
        )
        ensemble._programs["B"] = _make_mock_program(
            mocker,
            best_probs={"1": 1.0},
            top_solutions=[SolutionEntry(bitstring="1", prob=1.0, decoded=[1])],
        )

        with pytest.warns(UserWarning, match="was not a valid matching"):
            matching, weight = ensemble.aggregate_results(beam_width=None)

        assert is_valid_matching(matching)
        assert weight == pytest.approx(10.0)
