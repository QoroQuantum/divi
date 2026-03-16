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


class StubPartitioningEnsemble(PartitioningProgramEnsemble):
    """Minimal concrete subclass for testing the template methods."""

    def __init__(self, backend, solution_size=4):
        super().__init__(backend=backend, optimizer=_DEFAULT_OPTIMIZER)
        self._solution_size = solution_size
        self.finalized_args = None

    def create_programs(self):
        super().create_programs()

    def _initial_solution(self):
        return [0] * self._solution_size

    def _extend_solution(self, current_solution, prog_id, candidate):
        extended = list(current_solution)
        for i, val in enumerate(candidate.decoded):
            if i < len(extended):
                extended[i] = int(val)
        return extended

    def _evaluate_global_solution(self, solution):
        return -sum(solution)

    def _finalize_best(self, score, solution):
        self.finalized_args = (score, solution)
        return solution

    def _format_top_results(self, results):
        return [(list(sol), sc) for sc, sol in results]


def _make_mock_program(mocker, best_probs, top_solutions):
    """Create a mock VQA program with the given best_probs and top_solutions."""
    prog = mocker.MagicMock(spec=VariationalQuantumAlgorithm)
    prog.best_probs = best_probs
    prog.losses_history = [1.0]
    prog.results = {"some": "result"}
    prog.get_top_solutions.return_value = top_solutions
    return prog


class TestPartitioningProgramEnsemble:
    def test_aggregate_results_calls_hooks(self, mocker, dummy_simulator):
        stub = StubPartitioningEnsemble(dummy_simulator, solution_size=2)
        stub.create_programs()

        candidate = SolutionEntry(bitstring="11", prob=0.8, decoded=[1, 1])
        prog = _make_mock_program(
            mocker, best_probs={"11": 0.8}, top_solutions=[candidate]
        )
        stub._programs["A"] = prog

        result = stub.aggregate_results()

        assert result == [1, 1]
        assert stub.finalized_args is not None
        score, solution = stub.finalized_args
        assert solution == [1, 1]
        assert score == -2.0

    def test_get_top_solutions_calls_hooks(self, mocker, dummy_simulator):
        stub = StubPartitioningEnsemble(dummy_simulator, solution_size=2)
        stub.create_programs()

        candidate = SolutionEntry(bitstring="11", prob=0.8, decoded=[1, 1])
        prog = _make_mock_program(
            mocker, best_probs={"11": 0.8}, top_solutions=[candidate]
        )
        stub._programs["A"] = prog

        results = stub.get_top_solutions(n=1)

        assert len(results) == 1
        assert results[0][0] == [1, 1]
        assert results[0][1] == -2.0

    def test_aggregate_results_raises_if_no_programs(self, dummy_simulator):
        stub = StubPartitioningEnsemble(dummy_simulator)
        with pytest.raises(RuntimeError, match="No programs to aggregate"):
            stub.aggregate_results()

    def test_aggregate_results_raises_if_not_run(self, mocker, dummy_simulator):
        stub = StubPartitioningEnsemble(dummy_simulator)
        stub.create_programs()

        prog = mocker.MagicMock(spec=VariationalQuantumAlgorithm)
        prog.losses_history = []
        stub._programs["A"] = prog

        with pytest.raises(RuntimeError, match="Some/All programs have empty losses"):
            stub.aggregate_results()

    def test_aggregate_raises_if_no_best_probs(self, mocker, dummy_simulator):
        stub = StubPartitioningEnsemble(dummy_simulator)
        stub.create_programs()

        prog = mocker.MagicMock(spec=VariationalQuantumAlgorithm)
        prog.losses_history = [1.0]
        prog.results = {"some": "result"}
        prog.best_probs = {}
        stub._programs["A"] = prog

        with pytest.raises(
            RuntimeError,
            match="Not all final probabilities computed yet",
        ):
            stub.aggregate_results()

    def test_get_top_solutions_raises_on_invalid_n(self, dummy_simulator):
        stub = StubPartitioningEnsemble(dummy_simulator)
        stub.create_programs()

        with pytest.raises(ValueError, match="n must be >= 1"):
            stub.get_top_solutions(n=0)

    def test_get_top_solutions_raises_if_not_run(self, mocker, dummy_simulator):
        stub = StubPartitioningEnsemble(dummy_simulator)
        stub.create_programs()

        prog = mocker.MagicMock(spec=VariationalQuantumAlgorithm)
        prog.losses_history = []
        stub._programs["A"] = prog

        with pytest.raises(RuntimeError, match="Some/All programs have empty losses"):
            stub.get_top_solutions(n=1)

    def test_verify_basic_behaviour(self, mocker, dummy_simulator):
        stub = StubPartitioningEnsemble(dummy_simulator)
        verify_basic_program_ensemble_behaviour(mocker, stub)
