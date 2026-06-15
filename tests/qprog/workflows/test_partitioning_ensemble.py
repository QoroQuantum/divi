# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import networkx as nx
import pytest

from divi.qprog import QAOA, VariationalQuantumAlgorithm
from divi.qprog._solution_sampling_mixin import SolutionEntry
from divi.qprog.aggregation import BeamSearchStrategy, HierarchicalStrategy
from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer
from divi.qprog.problems import MaxWeightMatchingProblem, is_valid_matching
from divi.qprog.workflows import PartitioningProgramEnsemble
from tests.qprog._program_contracts import verify_basic_program_ensemble_behaviour

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
    prog = mocker.MagicMock(spec=QAOA)
    prog.best_probs = best_probs
    prog.losses_history = [1.0]
    prog.results = {"some": "result"}
    prog.has_results.return_value = True
    prog.get_top_solutions.return_value = top_solutions
    return prog


def _make_ensemble(problem, backend, **overrides):
    """Construct a PartitioningProgramEnsemble with the common test defaults."""
    kwargs = {
        "problem": problem,
        "n_layers": 1,
        "backend": backend,
        "optimizer": _DEFAULT_OPTIMIZER,
        **overrides,
    }
    return PartitioningProgramEnsemble(**kwargs)


_CONFLICT_CANDIDATES = [
    SolutionEntry(bitstring="1", prob=0.9, decoded=[1]),
    SolutionEntry(bitstring="0", prob=0.1, decoded=[0]),
]


def _make_matching_conflict_ensemble(mocker, backend, best_probs, top_solutions):
    """A two-edge matching whose partitions A/B share node 1 (a conflict).

    Both partitions are wired with the same ``best_probs``/``top_solutions``.
    """
    graph = nx.Graph()
    graph.add_weighted_edges_from([(0, 1, 10.0), (1, 2, 10.0)])
    problem = MaxWeightMatchingProblem(graph)
    problem._edge_index_maps = {
        "A": [problem._edge_to_qubit[(0, 1)]],
        "B": [problem._edge_to_qubit[(1, 2)]],
    }
    ensemble = _make_ensemble(problem, backend)
    for pid in ("A", "B"):
        ensemble._programs[pid] = _make_mock_program(
            mocker, best_probs=best_probs, top_solutions=top_solutions
        )
    return ensemble


def _attach_program_with_candidates(ensemble, mocker, decoded_candidates):
    """Wire a single mock program ``"A"`` emitting the given decoded candidates.

    Probabilities are assigned by rank (``1/(rank+1)``), so a single-candidate
    list yields ``prob=1.0``; the exact values are irrelevant to the aggregation
    routing these tests exercise.
    """
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


class TestPartitioningProgramEnsemble:
    def test_aggregate_results_calls_problem_hooks(self, mocker, dummy_simulator):
        problem = _make_stub_problem(mocker, solution_size=2)
        ensemble = _make_ensemble(problem, dummy_simulator)
        ensemble.create_programs()

        _attach_program_with_candidates(ensemble, mocker, [[1, 1]])

        result = ensemble.aggregate_results()

        assert result == ([1, 1], -2)
        problem.evaluate_global_solution.assert_called()
        problem.postprocess_candidates.assert_called_once()

    def test_get_top_solutions_calls_problem_hooks(self, mocker, dummy_simulator):
        problem = _make_stub_problem(mocker, solution_size=2)
        ensemble = _make_ensemble(problem, dummy_simulator)
        ensemble.create_programs()

        _attach_program_with_candidates(ensemble, mocker, [[1, 1]])

        results = ensemble.get_top_solutions(n=1)

        assert len(results) == 1
        assert results[0][0] == [1, 1]
        assert results[0][1] == -2.0

    def test_aggregate_raises_if_no_best_probs(self, mocker, dummy_simulator):
        problem = _make_stub_problem(mocker)
        ensemble = _make_ensemble(problem, dummy_simulator)
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
        ensemble = _make_ensemble(problem, dummy_simulator)
        ensemble.create_programs()

        with pytest.raises(ValueError, match="n must be >= 1"):
            ensemble.get_top_solutions(n=0)

    def test_get_top_solutions_raises_if_not_run(self, mocker, dummy_simulator):
        problem = _make_stub_problem(mocker)
        ensemble = _make_ensemble(problem, dummy_simulator)
        ensemble.create_programs()

        prog = mocker.MagicMock(spec=VariationalQuantumAlgorithm)
        prog.has_results.return_value = False
        ensemble._programs["A"] = prog

        with pytest.raises(RuntimeError, match="Some/All programs have no results"):
            ensemble.get_top_solutions(n=1)

    def test_verify_basic_behaviour(self, mocker, dummy_simulator):
        problem = _make_stub_problem(mocker)
        ensemble = _make_ensemble(problem, dummy_simulator)
        verify_basic_program_ensemble_behaviour(ensemble, mocker)


class TestStrictAggregation:
    """Strict mode is passed to problem-specific post-processing only."""

    def test_aggregate_results_uses_default_postprocessing(
        self, mocker, dummy_simulator
    ):
        problem = _make_stub_problem(mocker, solution_size=2)
        ensemble = _make_ensemble(problem, dummy_simulator)
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
        ensemble = _make_ensemble(problem, dummy_simulator)
        ensemble.create_programs()
        _attach_program_with_candidates(ensemble, mocker, [[1, 1]])

        with pytest.warns(
            UserWarning,
            match=(
                r"BeamSearchStrategy.*"
                r"Try wider parameters or "
                r"get_top_solutions\(\.\.\., strict=True\)"
            ),
        ):
            result = ensemble.aggregate_results(
                strategy=BeamSearchStrategy(beam_width=3, n_partition_candidates=5),
            )

        assert result is None

    def test_get_top_solutions_passes_strict_to_postprocess(
        self, mocker, dummy_simulator
    ):
        problem = _make_stub_problem(mocker, solution_size=2)
        ensemble = _make_ensemble(problem, dummy_simulator)
        ensemble.create_programs()
        _attach_program_with_candidates(ensemble, mocker, [[1, 1], [1, 0]])

        ensemble.get_top_solutions(
            n=2, strategy=BeamSearchStrategy(beam_width=None), strict=True
        )

        problem.postprocess_candidates.assert_called_once()
        assert problem.postprocess_candidates.call_args.kwargs == {"strict": True}

    def test_get_top_solutions_requests_exact_top_n_without_overfetch(
        self, mocker, dummy_simulator
    ):
        problem = _make_stub_problem(mocker, solution_size=3)
        ensemble = _make_ensemble(problem, dummy_simulator)
        ensemble.create_programs()
        prog = _attach_program_with_candidates(ensemble, mocker, [[1, 1, 1], [1, 1, 0]])

        ensemble.get_top_solutions(
            n=1, strategy=BeamSearchStrategy(beam_width=None), strict=True
        )

        prog.get_top_solutions.assert_called_once_with(n=2**20, include_decoded=True)
        candidates = problem.postprocess_candidates.call_args.args[0]
        assert len(candidates) == 1
        assert candidates[0][1] == [1, 1, 1]

    def test_matching_strict_filters_cross_partition_conflict(
        self, mocker, dummy_simulator
    ):
        ensemble = _make_matching_conflict_ensemble(
            mocker,
            dummy_simulator,
            best_probs={"1": 0.9, "0": 0.1},
            top_solutions=_CONFLICT_CANDIDATES,
        )

        with pytest.warns(UserWarning, match="No valid matching candidates"):
            assert (
                ensemble.get_top_solutions(
                    n=1, strategy=BeamSearchStrategy(beam_width=None), strict=True
                )
                == []
            )

        results = ensemble.get_top_solutions(
            n=3, strategy=BeamSearchStrategy(beam_width=None), strict=True
        )
        assert results == [([(0, 1)], 10.0), ([(1, 2)], 10.0)]

    def test_matching_aggregate_warns_when_raw_candidate_is_repaired(
        self, mocker, dummy_simulator
    ):
        ensemble = _make_matching_conflict_ensemble(
            mocker,
            dummy_simulator,
            best_probs={"1": 1.0},
            top_solutions=[SolutionEntry(bitstring="1", prob=1.0, decoded=[1])],
        )

        with pytest.warns(UserWarning, match="was not a valid matching"):
            matching, weight = ensemble.aggregate_results(
                strategy=BeamSearchStrategy(beam_width=None)
            )

        assert is_valid_matching(matching)
        assert weight == pytest.approx(10.0)


class TestHierarchicalStrategy:
    """Test HierarchicalStrategy through the PartitioningProgramEnsemble API."""

    def test_aggregate_results_hierarchical(self, mocker, dummy_simulator):
        problem = _make_stub_problem(mocker, solution_size=2)
        ensemble = _make_ensemble(problem, dummy_simulator)
        ensemble.create_programs()

        _attach_program_with_candidates(ensemble, mocker, [[1, 1]])

        result = ensemble.aggregate_results(strategy=HierarchicalStrategy())

        assert result == ([1, 1], -2)
        problem.evaluate_global_solution.assert_called()
        problem.postprocess_candidates.assert_called_once()

    def test_get_top_solutions_hierarchical(self, mocker, dummy_simulator):
        problem = _make_stub_problem(mocker, solution_size=2)
        ensemble = _make_ensemble(problem, dummy_simulator)
        ensemble.create_programs()

        _attach_program_with_candidates(ensemble, mocker, [[1, 1]])

        results = ensemble.get_top_solutions(n=1, strategy=HierarchicalStrategy())

        assert len(results) == 1
        assert results[0][0] == [1, 1]
        assert results[0][1] == -2.0

    def test_get_top_solutions_hierarchical_passes_strict(
        self, mocker, dummy_simulator
    ):
        problem = _make_stub_problem(mocker, solution_size=2)
        ensemble = _make_ensemble(problem, dummy_simulator)
        ensemble.create_programs()
        _attach_program_with_candidates(ensemble, mocker, [[1, 1], [1, 0]])

        ensemble.get_top_solutions(n=2, strategy=HierarchicalStrategy(), strict=True)

        problem.postprocess_candidates.assert_called_once()
        assert problem.postprocess_candidates.call_args.kwargs == {"strict": True}

    def test_aggregate_results_hierarchical_warns_on_empty(
        self, mocker, dummy_simulator
    ):
        problem = _make_stub_problem(mocker, solution_size=2)
        problem.postprocess_candidates.side_effect = None
        problem.postprocess_candidates.return_value = []
        ensemble = _make_ensemble(problem, dummy_simulator)
        ensemble.create_programs()
        _attach_program_with_candidates(ensemble, mocker, [[1, 1]])

        with pytest.warns(
            UserWarning,
            match=(
                r"HierarchicalStrategy.*"
                r"Try wider parameters or "
                r"get_top_solutions\(\.\.\., strict=True\)"
            ),
        ):
            result = ensemble.aggregate_results(strategy=HierarchicalStrategy())

        assert result is None

    def test_matching_hierarchical_strict_filters_conflict(
        self, mocker, dummy_simulator
    ):
        """Hierarchical strategy with strict=True filters cross-partition conflicts."""
        ensemble = _make_matching_conflict_ensemble(
            mocker,
            dummy_simulator,
            best_probs={"1": 0.9, "0": 0.1},
            top_solutions=_CONFLICT_CANDIDATES,
        )

        results = ensemble.get_top_solutions(
            n=3, strategy=HierarchicalStrategy(), strict=True
        )
        assert results == [([(0, 1)], 10.0), ([(1, 2)], 10.0)]
