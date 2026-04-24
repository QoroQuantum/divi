# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer
from divi.qprog.variational_quantum_algorithm import (
    SolutionEntry,
    VariationalQuantumAlgorithm,
)
from divi.qprog.workflows._partitioning_ensemble import (
    PartitioningProgramEnsemble,
)
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
    problem.finalize_solution.side_effect = lambda score, sol: sol
    problem.format_top_solutions.side_effect = lambda results: [
        (list(sol), sc) for sc, sol in results
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

        assert result == [1, 1]
        problem.evaluate_global_solution.assert_called()
        problem.finalize_solution.assert_called_once()

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
