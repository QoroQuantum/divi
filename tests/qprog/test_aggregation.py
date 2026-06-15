# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from divi.qprog.aggregation import BeamSearchStrategy, HierarchicalStrategy
from divi.qprog.variational_quantum_algorithm import SolutionEntry

# ──────────────────────────────────────────────────────────────────────
#  Helpers: lightweight mock VQA programs for testing
# ──────────────────────────────────────────────────────────────────────


class _MockProgram:
    """Minimal mock that implements get_top_solutions."""

    def __init__(self, candidates: list[SolutionEntry]):
        self._candidates = candidates

    def get_top_solutions(self, n=10, *, include_decoded=False):
        return self._candidates[:n]


# ──────────────────────────────────────────────────────────────────────
#  Simple extend / evaluate functions for testing
# ──────────────────────────────────────────────────────────────────────


def _sum_evaluate(solution):
    """Simple evaluator: sum of the solution vector (lower is better)."""
    return sum(solution)


def _neg_sum_evaluate(solution):
    """Evaluator where higher sums are better: returns -sum (lower is better)."""
    return -sum(solution)


def _write_extend(variable_maps):
    """Returns an extend_fn that writes decoded bits into global positions."""

    def extend(current, prog_id, candidate):
        result = list(current)
        for local_idx, global_idx in enumerate(variable_maps[prog_id]):
            result[global_idx] = int(candidate.decoded[local_idx])
        return result

    return extend


class _CountingEval:
    """Wraps an evaluate_fn and counts how many times it is called."""

    def __init__(self, fn):
        self._fn = fn
        self.calls = 0

    def __call__(self, solution):
        self.calls += 1
        return self._fn(solution)


# ──────────────────────────────────────────────────────────────────────
#  Program / candidate builders
# ──────────────────────────────────────────────────────────────────────


def _mock_programs(candidates_by_id):
    """Build a ``{id: _MockProgram(candidates)}`` mapping from candidate lists."""
    return {
        pid: _MockProgram(candidates) for pid, candidates in candidates_by_id.items()
    }


def _binary_candidates(prob_one=0.9, prob_zero=0.1):
    """Two single-bit candidates: decoded ``[1]`` (prob_one) and ``[0]`` (prob_zero)."""
    return [
        SolutionEntry(bitstring="1", prob=prob_one, decoded=[1]),
        SolutionEntry(bitstring="0", prob=prob_zero, decoded=[0]),
    ]


def _single_var_programs(n):
    """``n`` single-variable partitions, each offering candidates [1] and [0]."""
    programs = _mock_programs({f"P{i}": _binary_candidates(0.6, 0.4) for i in range(n)})
    var_maps = {f"P{i}": [i] for i in range(n)}
    return programs, var_maps


# ──────────────────────────────────────────────────────────────────────
#  Strategy wrappers: drive a strategy's ``aggregate`` from test inputs
# ──────────────────────────────────────────────────────────────────────


def beam_search_top_n(
    programs,
    initial_solution,
    extend_fn,
    evaluate_fn,
    beam_width=None,
    n_partition_candidates=None,
    top_n=1,
):
    """Test-local shorthand wrapping BeamSearchStrategy.aggregate."""
    strategy = BeamSearchStrategy(
        beam_width=beam_width, n_partition_candidates=n_partition_candidates
    )
    return strategy.aggregate(
        programs, initial_solution, extend_fn, evaluate_fn, top_n=top_n
    )


def beam_search_aggregate(*args, **kwargs):
    """``beam_search_top_n`` reduced to the single best solution vector."""
    return beam_search_top_n(*args, **kwargs)[0][1]


def hierarchical_top_n(
    programs,
    initial_solution,
    extend_fn,
    evaluate_fn,
    top_n=1,
    group_size=4,
    k_per_partition=20,
    max_per_group=200,
    merge_width=None,
):
    """Test-local shorthand wrapping HierarchicalStrategy.aggregate."""
    strategy = HierarchicalStrategy(
        group_size=group_size,
        k_per_partition=k_per_partition,
        max_per_group=max_per_group,
        merge_width=merge_width,
    )
    return strategy.aggregate(
        programs, initial_solution, extend_fn, evaluate_fn, top_n=top_n
    )


def hierarchical_aggregate(*args, **kwargs):
    """``hierarchical_top_n`` reduced to the single best solution vector."""
    return hierarchical_top_n(*args, **kwargs)[0][1]


# ──────────────────────────────────────────────────────────────────────
#  Tests
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"beam_width": 0}, "beam_width must be >= 1"),
        ({"beam_width": -1}, "beam_width must be >= 1"),
        (
            {"beam_width": 1, "n_partition_candidates": 0},
            "n_partition_candidates must be >= 1",
        ),
        (
            {"beam_width": 1, "n_partition_candidates": -3},
            "n_partition_candidates must be >= 1",
        ),
        (
            {"beam_width": 5, "n_partition_candidates": 2},
            "n_partition_candidates.*must be >= beam_width",
        ),
        # n_partition_candidates must be >= beam_width *after* the top_n bump
        # (beam_width 2 -> 5, so 3 < 5 raises). The message must surface the
        # original beam_width and the bump so it is actionable.
        (
            {"beam_width": 2, "n_partition_candidates": 3, "top_n": 5},
            r"bumped from 2 to 5 to satisfy top_n=5",
        ),
    ],
    ids=[
        "beam_width_zero",
        "beam_width_negative",
        "n_partition_candidates_zero",
        "n_partition_candidates_negative",
        "n_partition_candidates_lt_beam_width",
        "n_partition_candidates_lt_bumped_beam_width",
    ],
)
def test_beam_search_invalid_params_raise(kwargs, match):
    with pytest.raises(ValueError, match=match):
        beam_search_aggregate(
            programs={},
            initial_solution=[0],
            extend_fn=lambda c, p, s: c,
            evaluate_fn=lambda s: 0.0,
            **kwargs,
        )


class TestBeamSearchAggregateGreedy:
    """Test greedy mode (beam_width=1)."""

    def test_default_strategy_is_greedy(self):
        """The shipped default ``BeamSearchStrategy()`` is greedy (beam_width=1).

        The other helpers default ``beam_width=None`` (exhaustive); this pins the
        production default directly so it can't silently change.
        """
        assert BeamSearchStrategy().beam_width == 1

        # Greedy fetches only 1 candidate per partition, so it commits to A's
        # first candidate ([1,0], the lower-prob-but-first) and never sees [0,1].
        candidates_a = [
            SolutionEntry(bitstring="10", prob=0.6, decoded=[1, 0]),
            SolutionEntry(bitstring="01", prob=0.4, decoded=[0, 1]),
        ]
        programs = _mock_programs({"A": candidates_a})
        var_maps = {"A": [0, 1]}

        result = BeamSearchStrategy().aggregate(
            programs=programs,
            initial_solution=[0, 0],
            extend_fn=_write_extend(var_maps),
            evaluate_fn=_neg_sum_evaluate,
        )

        assert result == [(pytest.approx(-1.0), [1, 0])]

    def test_single_partition_single_candidate(self):
        """Greedy with one partition and one candidate returns that candidate."""
        candidates = [SolutionEntry(bitstring="10", prob=0.8, decoded=[1, 0])]
        programs = _mock_programs({"A": candidates})
        var_maps = {"A": [0, 1]}

        result = beam_search_aggregate(
            programs=programs,
            initial_solution=[0, 0],
            extend_fn=_write_extend(var_maps),
            evaluate_fn=_neg_sum_evaluate,
            beam_width=1,
        )

        assert result == [1, 0]

    def test_two_partitions_greedy_picks_best_per_partition(self):
        """Greedy picks the single best candidate from each partition."""
        # Partition A: variables 0,1 — candidates: [1,0] (prob=0.9)
        candidates_a = [
            SolutionEntry(bitstring="10", prob=0.9, decoded=[1, 0]),
        ]
        # Partition B: variables 2,3 — candidates: [1,1] (prob=0.7)
        candidates_b = [
            SolutionEntry(bitstring="11", prob=0.7, decoded=[1, 1]),
        ]
        programs = _mock_programs({"A": candidates_a, "B": candidates_b})
        var_maps = {"A": [0, 1], "B": [2, 3]}

        result = beam_search_aggregate(
            programs=programs,
            initial_solution=[0, 0, 0, 0],
            extend_fn=_write_extend(var_maps),
            evaluate_fn=_neg_sum_evaluate,
            beam_width=1,
        )

        # Greedy: After A, only [1,0,0,0]. After B, only [1,0,1,1].
        assert result == [1, 0, 1, 1]


class TestBeamSearchAggregateBeam:
    """Test standard beam search (beam_width > 1)."""

    def test_beam_width_2_explores_combinations(self):
        """Beam width 2 keeps two partial solutions and finds the best."""
        # Partition A (vars 0,1):
        candidates_a = [
            SolutionEntry(bitstring="10", prob=0.6, decoded=[1, 0]),
            SolutionEntry(bitstring="01", prob=0.4, decoded=[0, 1]),
        ]
        # Partition B (vars 2,3):
        candidates_b = [
            SolutionEntry(bitstring="10", prob=0.7, decoded=[1, 0]),
            SolutionEntry(bitstring="01", prob=0.3, decoded=[0, 1]),
        ]
        programs = _mock_programs({"A": candidates_a, "B": candidates_b})
        var_maps = {"A": [0, 1], "B": [2, 3]}

        result = beam_search_aggregate(
            programs=programs,
            initial_solution=[0, 0, 0, 0],
            extend_fn=_write_extend(var_maps),
            evaluate_fn=_neg_sum_evaluate,
            beam_width=2,
        )

        # With beam_width=2, both A candidates are kept.
        # Expanding B: 4 combinations. All have sum=2, so any is optimal.
        assert sum(result) == 2

    def test_beam_finds_better_than_greedy(self):
        """Beam search can find a solution that greedy misses."""
        weights = [10, 1, 1, 10]

        def weighted_evaluate(solution):
            return sum(s * w for s, w in zip(solution, weights))

        # Partition A (vars 0,1): beam_width=1 only sees first candidate
        candidates_a = [
            SolutionEntry(bitstring="10", prob=0.9, decoded=[1, 0]),
            SolutionEntry(bitstring="01", prob=0.1, decoded=[0, 1]),
        ]
        # Partition B (vars 2,3):
        candidates_b = [
            SolutionEntry(bitstring="01", prob=0.9, decoded=[0, 1]),
            SolutionEntry(bitstring="10", prob=0.1, decoded=[1, 0]),
        ]
        programs = _mock_programs({"A": candidates_a, "B": candidates_b})
        var_maps = {"A": [0, 1], "B": [2, 3]}

        # Greedy (beam_width=1): only sees 1 candidate per partition
        greedy_result = beam_search_aggregate(
            programs=programs,
            initial_solution=[0, 0, 0, 0],
            extend_fn=_write_extend(var_maps),
            evaluate_fn=weighted_evaluate,
            beam_width=1,
        )

        # Beam (beam_width=2): sees 2 candidates per partition
        beam_result = beam_search_aggregate(
            programs=programs,
            initial_solution=[0, 0, 0, 0],
            extend_fn=_write_extend(var_maps),
            evaluate_fn=weighted_evaluate,
            beam_width=2,
        )

        greedy_cost = weighted_evaluate(greedy_result)
        beam_cost = weighted_evaluate(beam_result)

        # Beam should find an equal or better solution
        assert beam_cost <= greedy_cost


class TestBeamSearchAggregateExhaustive:
    """Test exhaustive mode (beam_width=None)."""

    def test_exhaustive_explores_all_combinations(self):
        """With beam_width=None, all combinations are evaluated."""
        candidates_a = _binary_candidates(0.6, 0.4)
        candidates_b = _binary_candidates(0.7, 0.3)
        programs = _mock_programs({"A": candidates_a, "B": candidates_b})
        var_maps = {"A": [0], "B": [1]}

        result = beam_search_aggregate(
            programs=programs,
            initial_solution=[0, 0],
            extend_fn=_write_extend(var_maps),
            evaluate_fn=_sum_evaluate,
            beam_width=None,
        )

        # _sum_evaluate minimizes sum → best is [0, 0]
        assert result == [0, 0]

    def test_exhaustive_finds_global_optimum(self):
        """Exhaustive must find the true global optimum."""

        def tricky_evaluate(solution):
            """Only [0,1,0] has cost -100, everything else is >= 0."""
            if solution == [0, 1, 0]:
                return -100.0
            return sum(solution)

        programs = _mock_programs(
            {pid: _binary_candidates() for pid in ("A", "B", "C")}
        )
        var_maps = {"A": [0], "B": [1], "C": [2]}

        result = beam_search_aggregate(
            programs=programs,
            initial_solution=[0, 0, 0],
            extend_fn=_write_extend(var_maps),
            evaluate_fn=tricky_evaluate,
            beam_width=None,
        )

        assert result == [0, 1, 0]


class TestBeamSearchAggregateEdgeCases:
    """Test edge cases."""

    def test_single_partition(self):
        """Single partition still works correctly."""
        candidates = [
            SolutionEntry(bitstring="01", prob=0.9, decoded=[0, 1]),
            SolutionEntry(bitstring="10", prob=0.1, decoded=[1, 0]),
        ]
        programs = _mock_programs({"A": candidates})
        var_maps = {"A": [0, 1]}

        # beam_width=2 sees both candidates, picks highest sum
        result = beam_search_aggregate(
            programs=programs,
            initial_solution=[0, 0],
            extend_fn=_write_extend(var_maps),
            evaluate_fn=_neg_sum_evaluate,
            beam_width=2,
        )

        assert result == [0, 1]

    def test_empty_programs_returns_initial(self):
        """No programs at all returns the initial solution."""
        result = beam_search_aggregate(
            programs={},
            initial_solution=[0, 0, 0],
            extend_fn=lambda c, p, s: c,
            evaluate_fn=_sum_evaluate,
            beam_width=1,
        )

        assert result == [0, 0, 0]

    def test_program_with_no_candidates_skipped(self):
        """A program returning no candidates is skipped without error."""
        programs = _mock_programs({"A": []})

        result = beam_search_aggregate(
            programs=programs,
            initial_solution=[0, 0],
            extend_fn=lambda c, p, s: c,
            evaluate_fn=_sum_evaluate,
            beam_width=1,
        )

        assert result == [0, 0]

    def test_beam_width_limits_extraction(self):
        """beam_width limits both candidates extracted and beam size."""
        many_candidates = _binary_candidates()
        programs = _mock_programs({"A": many_candidates})
        var_maps = {"A": [0]}

        # beam_width=1 should only consider 1 candidate: [1]
        result = beam_search_aggregate(
            programs=programs,
            initial_solution=[0],
            extend_fn=_write_extend(var_maps),
            evaluate_fn=_sum_evaluate,
            beam_width=1,
        )

        # Only candidate [1] is considered (beam_width=1), so result is [1]
        assert result == [1]

    def test_non_zero_initial_solution_preserved(self):
        """Positions not touched by any partition retain their initial values."""
        # Partition only covers position 1; positions 0 and 2 should stay as-is
        candidates = [SolutionEntry(bitstring="1", prob=0.9, decoded=[1])]
        programs = _mock_programs({"A": candidates})
        var_maps = {"A": [1]}

        result = beam_search_aggregate(
            programs=programs,
            initial_solution=[1, 0, 1],
            extend_fn=_write_extend(var_maps),
            evaluate_fn=_neg_sum_evaluate,
            beam_width=1,
        )

        assert result == [1, 1, 1]

    def test_overlapping_partitions(self):
        """Partitions writing to overlapping global positions work correctly."""
        # Both partitions write to position 0; partition B overwrites A's value
        candidates_a = [SolutionEntry(bitstring="1", prob=0.9, decoded=[1])]
        candidates_b = [SolutionEntry(bitstring="0", prob=0.9, decoded=[0])]
        programs = _mock_programs({"A": candidates_a, "B": candidates_b})
        var_maps = {"A": [0], "B": [0]}

        result = beam_search_aggregate(
            programs=programs,
            initial_solution=[0],
            extend_fn=_write_extend(var_maps),
            evaluate_fn=_sum_evaluate,
            beam_width=1,
        )

        # B runs after A and overwrites position 0 → final value is 0
        assert result == [0]

    def test_tie_breaking_is_stable(self):
        """When candidates have identical scores, a valid result is returned."""
        candidates_a = _binary_candidates(0.5, 0.5)
        programs = _mock_programs({"A": candidates_a})
        var_maps = {"A": [0]}

        # Both candidates have |sum|=1 or 0 under _sum_evaluate — either is valid
        result = beam_search_aggregate(
            programs=programs,
            initial_solution=[0],
            extend_fn=_write_extend(var_maps),
            evaluate_fn=_sum_evaluate,
            beam_width=2,
        )

        assert result in ([0], [1])


class TestBeamSearchAggregatePruning:
    """Test that beam pruning behaves correctly."""

    def test_beam_prunes_to_width(self):
        """After each partition step, at most beam_width solutions are kept."""
        # 3 partitions × 3 candidates each.  With beam_width=2, the beam
        # should never exceed 2 partial solutions between steps, which limits
        # the total work and may exclude some global combinations.
        candidates = [
            SolutionEntry(bitstring="1", prob=0.5, decoded=[1]),
            SolutionEntry(bitstring="0", prob=0.3, decoded=[0]),
            SolutionEntry(bitstring="1", prob=0.2, decoded=[1]),
        ]
        programs = _mock_programs({pid: candidates for pid in ("A", "B", "C")})
        var_maps = {"A": [0], "B": [1], "C": [2]}

        result = beam_search_aggregate(
            programs=programs,
            initial_solution=[0, 0, 0],
            extend_fn=_write_extend(var_maps),
            evaluate_fn=_sum_evaluate,
            beam_width=2,
        )

        # With _sum_evaluate (minimize sum), the optimal is [0,0,0]
        assert result == [0, 0, 0]

    def test_monotonicity_exhaustive_leq_beam_leq_greedy(self):
        """Wider beam should always find equal or better solutions.

        Verifies the invariant: cost(exhaustive) <= cost(beam) <= cost(greedy).
        """
        weights = [10, 1, 1, 10, 5, 2]

        def weighted_evaluate(solution):
            return sum(s * w for s, w in zip(solution, weights))

        candidates_a = [
            SolutionEntry(bitstring="10", prob=0.8, decoded=[1, 0]),
            SolutionEntry(bitstring="01", prob=0.15, decoded=[0, 1]),
            SolutionEntry(bitstring="00", prob=0.05, decoded=[0, 0]),
        ]
        candidates_b = [
            SolutionEntry(bitstring="01", prob=0.7, decoded=[0, 1]),
            SolutionEntry(bitstring="10", prob=0.2, decoded=[1, 0]),
            SolutionEntry(bitstring="00", prob=0.1, decoded=[0, 0]),
        ]
        candidates_c = [
            SolutionEntry(bitstring="11", prob=0.6, decoded=[1, 1]),
            SolutionEntry(bitstring="10", prob=0.3, decoded=[1, 0]),
            SolutionEntry(bitstring="00", prob=0.1, decoded=[0, 0]),
        ]
        programs = _mock_programs(
            {"A": candidates_a, "B": candidates_b, "C": candidates_c}
        )
        var_maps = {"A": [0, 1], "B": [2, 3], "C": [4, 5]}

        kwargs = dict(
            programs=programs,
            initial_solution=[0] * 6,
            extend_fn=_write_extend(var_maps),
            evaluate_fn=weighted_evaluate,
        )

        greedy_cost = weighted_evaluate(beam_search_aggregate(**kwargs, beam_width=1))
        beam_cost = weighted_evaluate(beam_search_aggregate(**kwargs, beam_width=2))
        exhaustive_cost = weighted_evaluate(
            beam_search_aggregate(**kwargs, beam_width=None)
        )

        assert exhaustive_cost <= beam_cost <= greedy_cost

    def test_n_partition_candidates_widens_search(self):
        """More candidates per partition can find better solutions with narrow beam."""

        def weighted_evaluate(solution):
            return sum(solution) * 10

        # 3 candidates; greedy (beam_width=1) only sees the first one ([1])
        candidates = [
            SolutionEntry(bitstring="1", prob=0.6, decoded=[1]),
            SolutionEntry(bitstring="0", prob=0.3, decoded=[0]),
            SolutionEntry(bitstring="1", prob=0.1, decoded=[1]),
        ]
        programs = _mock_programs({"A": candidates})
        var_maps = {"A": [0]}

        # beam_width=1, default n_partition_candidates (=1): only sees [1]
        greedy_result = beam_search_aggregate(
            programs=programs,
            initial_solution=[0],
            extend_fn=_write_extend(var_maps),
            evaluate_fn=weighted_evaluate,
            beam_width=1,
        )

        # beam_width=1, n_partition_candidates=3: sees all 3, picks best ([0])
        wider_result = beam_search_aggregate(
            programs=programs,
            initial_solution=[0],
            extend_fn=_write_extend(var_maps),
            evaluate_fn=weighted_evaluate,
            beam_width=1,
            n_partition_candidates=3,
        )

        assert weighted_evaluate(wider_result) <= weighted_evaluate(greedy_result)


class TestBeamSearchAggregateTopN:
    """Test BeamSearchStrategy returning multiple ranked solutions."""

    def _make_two_partition_setup(self):
        candidates_a = [
            SolutionEntry(bitstring="10", prob=0.6, decoded=[1, 0]),
            SolutionEntry(bitstring="01", prob=0.3, decoded=[0, 1]),
            SolutionEntry(bitstring="00", prob=0.1, decoded=[0, 0]),
        ]
        candidates_b = [
            SolutionEntry(bitstring="10", prob=0.7, decoded=[1, 0]),
            SolutionEntry(bitstring="01", prob=0.2, decoded=[0, 1]),
            SolutionEntry(bitstring="00", prob=0.1, decoded=[0, 0]),
        ]
        programs = _mock_programs({"A": candidates_a, "B": candidates_b})
        var_maps = {"A": [0, 1], "B": [2, 3]}
        return programs, var_maps

    def test_top_n_returns_n_results(self):
        programs, var_maps = self._make_two_partition_setup()
        results = beam_search_top_n(
            programs=programs,
            initial_solution=[0, 0, 0, 0],
            extend_fn=_write_extend(var_maps),
            evaluate_fn=_sum_evaluate,
            beam_width=3,
            top_n=3,
        )
        assert len(results) == 3
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)

    def test_top_n_sorted_ascending(self):
        programs, var_maps = self._make_two_partition_setup()
        results = beam_search_top_n(
            programs=programs,
            initial_solution=[0, 0, 0, 0],
            extend_fn=_write_extend(var_maps),
            evaluate_fn=_sum_evaluate,
            beam_width=3,
            top_n=3,
        )
        scores = [score for score, _sol in results]
        assert scores == sorted(scores)

    def test_top_n_1_matches_original(self):
        programs, var_maps = self._make_two_partition_setup()
        kwargs = dict(
            programs=programs,
            initial_solution=[0, 0, 0, 0],
            extend_fn=_write_extend(var_maps),
            evaluate_fn=_sum_evaluate,
            beam_width=2,
        )
        original = beam_search_aggregate(**kwargs)
        top_1 = beam_search_top_n(**kwargs, top_n=1)
        assert top_1[0][1] == original

    def test_beam_width_bumped_to_n(self):
        """top_n=3 with beam_width=1 still returns 3 results."""
        programs, var_maps = self._make_two_partition_setup()
        results = beam_search_top_n(
            programs=programs,
            initial_solution=[0, 0, 0, 0],
            extend_fn=_write_extend(var_maps),
            evaluate_fn=_sum_evaluate,
            beam_width=1,
            top_n=3,
        )
        assert len(results) == 3

    def test_top_n_greater_than_beam_capped(self):
        """When fewer solutions exist than top_n, returns all available."""
        candidates = [SolutionEntry(bitstring="1", prob=0.9, decoded=[1])]
        programs = _mock_programs({"A": candidates})
        var_maps = {"A": [0]}
        results = beam_search_top_n(
            programs=programs,
            initial_solution=[0],
            extend_fn=_write_extend(var_maps),
            evaluate_fn=_sum_evaluate,
            beam_width=1,
            top_n=10,
        )
        # 1 candidate × 1 partition = exactly 1 reachable solution, even though
        # beam_width is bumped to top_n=10.
        assert len(results) == 1

    def test_top_n_bump_accepts_n_partition_candidates_at_bumped_width(self):
        """The companion to the bumped-beam validation: when n_partition_candidates
        satisfies the *bumped* beam_width it must NOT raise, and returns top_n.
        """
        programs, var_maps = self._make_two_partition_setup()
        results = beam_search_top_n(
            programs=programs,
            initial_solution=[0, 0, 0, 0],
            extend_fn=_write_extend(var_maps),
            evaluate_fn=_sum_evaluate,
            beam_width=2,
            n_partition_candidates=5,  # >= bumped beam_width (top_n=5)
            top_n=5,
        )
        assert len(results) == 5


# ──────────────────────────────────────────────────────────────────────
#  Hierarchical aggregation tests
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"top_n": 0}, "top_n must be >= 1"),
        ({"group_size": 0}, "group_size must be >= 1"),
        ({"k_per_partition": 0}, "k_per_partition must be >= 1"),
        ({"max_per_group": 0}, "max_per_group must be >= 1"),
    ],
)
def test_hierarchical_invalid_params_raise(kwargs, match):
    with pytest.raises(ValueError, match=match):
        hierarchical_top_n(
            programs={},
            initial_solution=[0],
            extend_fn=lambda c, p, s: c,
            evaluate_fn=lambda s: 0.0,
            **kwargs,
        )


class TestHierarchicalAggregateBasic:
    """Test basic hierarchical aggregation functionality."""

    def test_single_partition_single_candidate(self):
        """Single partition with one candidate returns that candidate."""
        candidates = [SolutionEntry(bitstring="10", prob=0.8, decoded=[1, 0])]
        programs = _mock_programs({"A": candidates})
        var_maps = {"A": [0, 1]}

        result = hierarchical_aggregate(
            programs=programs,
            initial_solution=[0, 0],
            extend_fn=_write_extend(var_maps),
            evaluate_fn=_neg_sum_evaluate,
        )

        assert result == [1, 0]

    def test_two_partitions_picks_best(self):
        """Two partitions should combine to produce best solution."""
        candidates_a = [
            SolutionEntry(bitstring="10", prob=0.9, decoded=[1, 0]),
        ]
        candidates_b = [
            SolutionEntry(bitstring="11", prob=0.7, decoded=[1, 1]),
        ]
        programs = _mock_programs({"A": candidates_a, "B": candidates_b})
        var_maps = {"A": [0, 1], "B": [2, 3]}

        result = hierarchical_aggregate(
            programs=programs,
            initial_solution=[0, 0, 0, 0],
            extend_fn=_write_extend(var_maps),
            evaluate_fn=_neg_sum_evaluate,
        )

        assert result == [1, 0, 1, 1]

    def test_empty_programs_returns_initial(self):
        """No programs returns the initial solution."""
        result = hierarchical_aggregate(
            programs={},
            initial_solution=[0, 0, 0],
            extend_fn=lambda c, p, s: c,
            evaluate_fn=_sum_evaluate,
        )

        assert result == [0, 0, 0]

    def test_program_with_no_candidates_skipped(self):
        """A program returning no candidates is effectively skipped."""
        programs = _mock_programs({"A": []})

        result = hierarchical_aggregate(
            programs=programs,
            initial_solution=[0, 0],
            extend_fn=lambda c, p, s: c,
            evaluate_fn=_sum_evaluate,
        )

        assert result == [0, 0]


class TestHierarchicalAggregateGrouping:
    """Test the grouping and pairwise merge logic."""

    def test_group_size_1_processes_each_partition_separately(self):
        """group_size=1 creates one group per partition, then merges pairwise."""
        candidates_a = [SolutionEntry(bitstring="1", prob=0.9, decoded=[1])]
        candidates_b = [SolutionEntry(bitstring="1", prob=0.7, decoded=[1])]
        programs = _mock_programs({"A": candidates_a, "B": candidates_b})
        var_maps = {"A": [0], "B": [1]}

        result = hierarchical_aggregate(
            programs=programs,
            initial_solution=[0, 0],
            extend_fn=_write_extend(var_maps),
            evaluate_fn=_neg_sum_evaluate,
            group_size=1,
        )

        assert result == [1, 1]

    def test_group_size_larger_than_partitions(self):
        """group_size larger than the number of partitions is fine."""
        candidates_a = [SolutionEntry(bitstring="1", prob=0.9, decoded=[1])]
        candidates_b = [SolutionEntry(bitstring="1", prob=0.7, decoded=[1])]
        programs = _mock_programs({"A": candidates_a, "B": candidates_b})
        var_maps = {"A": [0], "B": [1]}

        result = hierarchical_aggregate(
            programs=programs,
            initial_solution=[0, 0],
            extend_fn=_write_extend(var_maps),
            evaluate_fn=_neg_sum_evaluate,
            group_size=100,
        )

        assert result == [1, 1]

    def test_odd_number_of_groups_last_carried_forward(self):
        """An odd number of groups carries the last group forward unpaired."""
        candidates_a = [SolutionEntry(bitstring="1", prob=0.9, decoded=[1])]
        candidates_b = [SolutionEntry(bitstring="1", prob=0.7, decoded=[1])]
        candidates_c = [SolutionEntry(bitstring="1", prob=0.5, decoded=[1])]
        programs = _mock_programs(
            {"A": candidates_a, "B": candidates_b, "C": candidates_c}
        )
        var_maps = {"A": [0], "B": [1], "C": [2]}

        result = hierarchical_aggregate(
            programs=programs,
            initial_solution=[0, 0, 0],
            extend_fn=_write_extend(var_maps),
            evaluate_fn=_neg_sum_evaluate,
            group_size=1,
        )

        assert result == [1, 1, 1]


class TestHierarchicalAggregateFindsOptimal:
    """Test that hierarchical aggregation finds global optima."""

    def test_finds_global_optimum(self):
        """Should find the true global optimum across all combinations."""

        def tricky_evaluate(solution):
            if solution == [0, 1, 0]:
                return -100.0
            return sum(solution)

        programs = _mock_programs(
            {pid: _binary_candidates() for pid in ("A", "B", "C")}
        )
        var_maps = {"A": [0], "B": [1], "C": [2]}

        result = hierarchical_aggregate(
            programs=programs,
            initial_solution=[0, 0, 0],
            extend_fn=_write_extend(var_maps),
            evaluate_fn=tricky_evaluate,
            group_size=4,
        )

        assert result == [0, 1, 0]

    def test_finds_better_than_greedy_beam(self):
        """Hierarchical finds optima that beam_width=1 misses.

        With a *coupled* objective (not separable across partitions) greedy beam
        commits to the locally-cheapest A prefix ``[1, 0]`` and is stuck; the full
        per-group enumeration reaches the global optimum ``[0, 1]``.
        """
        # Objective couples the two partitions: A=1 looks best in isolation,
        # but the global optimum needs A=0.
        cost_table = {(0, 0): 5.0, (1, 0): 0.0, (0, 1): -10.0, (1, 1): 4.0}

        def coupled_evaluate(solution):
            return cost_table[tuple(solution)]

        candidates_a = _binary_candidates()
        candidates_b = _binary_candidates(0.5, 0.5)
        programs = _mock_programs({"A": candidates_a, "B": candidates_b})
        var_maps = {"A": [0], "B": [1]}

        greedy_result = beam_search_aggregate(
            programs=programs,
            initial_solution=[0, 0],
            extend_fn=_write_extend(var_maps),
            evaluate_fn=coupled_evaluate,
            beam_width=1,
            n_partition_candidates=2,
        )

        hierarchical_result = hierarchical_aggregate(
            programs=programs,
            initial_solution=[0, 0],
            extend_fn=_write_extend(var_maps),
            evaluate_fn=coupled_evaluate,
        )

        assert greedy_result == [1, 0]
        assert hierarchical_result == [0, 1]
        assert coupled_evaluate(hierarchical_result) < coupled_evaluate(greedy_result)


class TestHierarchicalAggregateTopN:
    """Test HierarchicalStrategy returning multiple ranked solutions."""

    def test_returns_multiple_solutions(self):
        """Should return up to top_n solutions."""
        candidates = _binary_candidates(0.6, 0.4)
        programs = _mock_programs({"A": candidates})
        var_maps = {"A": [0]}

        results = hierarchical_top_n(
            programs=programs,
            initial_solution=[0],
            extend_fn=_write_extend(var_maps),
            evaluate_fn=_sum_evaluate,
            top_n=3,
        )

        assert len(results) >= 1
        assert len(results) <= 3
        # Results sorted ascending (best first)
        for i in range(len(results) - 1):
            assert results[i][0] <= results[i + 1][0]

    def test_top_n_caps_output(self):
        """Even with many combinations, only top_n are returned."""
        candidates_a = _binary_candidates(0.6, 0.4)
        candidates_b = _binary_candidates(0.7, 0.3)
        programs = _mock_programs({"A": candidates_a, "B": candidates_b})
        var_maps = {"A": [0], "B": [1]}

        results = hierarchical_top_n(
            programs=programs,
            initial_solution=[0, 0],
            extend_fn=_write_extend(var_maps),
            evaluate_fn=_sum_evaluate,
            top_n=2,
        )

        assert len(results) == 2
        # Best (lowest sum) first
        assert results[0][0] <= results[1][0]

    def test_non_zero_initial_solution_preserved(self):
        """Positions not touched by any partition keep initial values."""
        candidates = [SolutionEntry(bitstring="1", prob=0.9, decoded=[1])]
        programs = _mock_programs({"A": candidates})
        var_maps = {"A": [1]}

        result = hierarchical_aggregate(
            programs=programs,
            initial_solution=[1, 0, 1],
            extend_fn=_write_extend(var_maps),
            evaluate_fn=_neg_sum_evaluate,
        )

        assert result == [1, 1, 1]


class TestHierarchicalAggregateMerge:
    """Robustness of the merge primitive: it rebuilds combined solutions by
    replaying selections through ``extend_fn`` rather than overlaying bits.

    These exercise the primitive on inputs a hardcoded bit-overlay would mishandle
    (overlapping indices, non-zero "unset" values). divi's own decomposers always
    produce disjoint partitions, so these are robustness guards on the merge logic,
    not coverage of a reachable overlapping decomposition.
    """

    def test_overlapping_maps_later_group_wins(self):
        """On a shared index, the later group's value wins (incl. an explicit 0).

        A sets index 1 to 1; B owns the same index 1 and assigns it 0. A bit-OR
        merge would keep A's 1; replaying B's selection correctly overwrites it.
        ``group_size=1`` forces the two partitions into separate groups so the
        merge path runs.
        """
        # A -> indices [0, 1] = [1, 1]; B -> indices [1, 2] = [0, 1] (overlap at 1)
        programs = _mock_programs(
            {
                "A": [SolutionEntry(bitstring="11", prob=1.0, decoded=[1, 1])],
                "B": [SolutionEntry(bitstring="01", prob=1.0, decoded=[0, 1])],
            }
        )
        var_maps = {"A": [0, 1], "B": [1, 2]}

        result = hierarchical_aggregate(
            programs=programs,
            initial_solution=[0, 0, 0],
            extend_fn=_write_extend(var_maps),
            evaluate_fn=_sum_evaluate,
            group_size=1,
        )

        assert result == [1, 0, 1]

    def test_non_zero_initial_preserved_across_merge(self):
        """An initial bit owned by no partition survives a multi-group merge.

        ``group_size=1`` puts A and B in separate groups; index 0 is set in the
        initial solution and owned by neither, so the merge must carry it through.
        """
        one = [SolutionEntry(bitstring="1", prob=1.0, decoded=[1])]
        programs = _mock_programs({"A": one, "B": one})
        var_maps = {"A": [1], "B": [2]}

        result = hierarchical_aggregate(
            programs=programs,
            initial_solution=[1, 0, 0, 0],
            extend_fn=_write_extend(var_maps),
            evaluate_fn=_neg_sum_evaluate,
            group_size=1,
        )

        assert result == [1, 1, 1, 0]


class TestHierarchicalAggregateCost:
    """Search cost is driven by max_per_group / merge_width, not by top_n."""

    def test_merge_width_zero_raises(self):
        with pytest.raises(ValueError, match="merge_width must be >= 1"):
            HierarchicalStrategy(merge_width=0)

    def test_top_n_does_not_inflate_search_cost(self):
        """A larger top_n must not increase the number of evaluate_fn calls.

        ``top_n`` only widens the final slice (a sort+slice), so the scoring work
        is identical whether the caller asks for 1 solution or 50. ``group_size=2``
        makes the within-group pool grow to 4 candidates (2 partitions × 2) which
        ``max_per_group=2`` then prunes — so the search-cap pruning is load-bearing
        and a regression that let ``top_n`` widen intermediate pools would change
        the count.
        """
        programs, var_maps = _single_var_programs(4)

        def calls_for(top_n):
            ev = _CountingEval(_sum_evaluate)
            HierarchicalStrategy(
                group_size=2, k_per_partition=2, max_per_group=2
            ).aggregate(programs, [0, 0, 0, 0], _write_extend(var_maps), ev, top_n)
            return ev.calls

        assert calls_for(1) == calls_for(50)

    def test_top_n_exceeds_max_per_group_returns_more(self):
        """``top_n`` larger than ``max_per_group`` still returns ``top_n`` results.

        The final stage retains ``max(max_per_group, top_n)``, so asking for more
        solutions than the search width widens the *returned* set even though it
        does not widen the search.
        """
        programs, var_maps = _single_var_programs(2)

        results = hierarchical_top_n(
            programs,
            [0, 0],
            _write_extend(var_maps),
            _sum_evaluate,
            top_n=3,
            group_size=2,
            max_per_group=2,
        )

        # 4 combinations exist; max_per_group=2 would cap at 2 without the
        # top_n-driven final widening.
        assert len(results) == 3
        scores = [score for score, _ in results]
        assert scores == sorted(scores)

    def test_merge_width_reduces_evaluate_calls(self):
        """Capping the per-merge fan-in lowers the evaluate_fn call count."""
        programs, var_maps = _single_var_programs(4)

        def calls_for(merge_width):
            ev = _CountingEval(_sum_evaluate)
            HierarchicalStrategy(
                group_size=1,
                k_per_partition=2,
                max_per_group=5,
                merge_width=merge_width,
            ).aggregate(programs, [0, 0, 0, 0], _write_extend(var_maps), ev, top_n=1)
            return ev.calls

        assert calls_for(1) < calls_for(None)

    def test_merge_width_trades_quality_for_cost(self):
        """merge_width caps the merge fan-in, which can miss the global optimum.

        Coupled cost (optimum ``[0, 1]`` = -10) is reachable only if each group's
        *non-greedy* prefix survives into the merge. Full fan-in finds it; a
        ``merge_width=1`` merge keeps only each group's local best and misses it —
        confirming the knob changes *which* solution is found, not just the count.
        """
        cost_table = {(0, 0): 5.0, (1, 0): 0.0, (0, 1): -10.0, (1, 1): 4.0}

        def coupled_evaluate(solution):
            return cost_table[tuple(solution)]

        programs, var_maps = _single_var_programs(2)
        kwargs = dict(
            programs=programs,
            initial_solution=[0, 0],
            extend_fn=_write_extend(var_maps),
            evaluate_fn=coupled_evaluate,
            group_size=1,
            max_per_group=2,
        )

        full = hierarchical_aggregate(**kwargs, merge_width=2)
        narrow = hierarchical_aggregate(**kwargs, merge_width=1)

        assert full == [0, 1]  # global optimum, cost -10
        assert narrow == [1, 1]  # local-best prefixes only, cost 4
