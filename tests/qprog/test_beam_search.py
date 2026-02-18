# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from divi.qprog.batch import beam_search_aggregate
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


# ──────────────────────────────────────────────────────────────────────
#  Tests
# ──────────────────────────────────────────────────────────────────────


class TestBeamSearchAggregateValidation:
    def test_beam_width_zero_raises(self):
        with pytest.raises(ValueError, match="beam_width must be >= 1"):
            beam_search_aggregate(
                programs={},
                initial_solution=[0],
                extend_fn=lambda c, p, s: c,
                evaluate_fn=lambda s: 0.0,
                beam_width=0,
            )

    def test_beam_width_negative_raises(self):
        with pytest.raises(ValueError, match="beam_width must be >= 1"):
            beam_search_aggregate(
                programs={},
                initial_solution=[0],
                extend_fn=lambda c, p, s: c,
                evaluate_fn=lambda s: 0.0,
                beam_width=-1,
            )

    def test_n_partition_candidates_zero_raises(self):
        with pytest.raises(ValueError, match="n_partition_candidates must be >= 1"):
            beam_search_aggregate(
                programs={},
                initial_solution=[0],
                extend_fn=lambda c, p, s: c,
                evaluate_fn=lambda s: 0.0,
                beam_width=1,
                n_partition_candidates=0,
            )

    def test_n_partition_candidates_negative_raises(self):
        with pytest.raises(ValueError, match="n_partition_candidates must be >= 1"):
            beam_search_aggregate(
                programs={},
                initial_solution=[0],
                extend_fn=lambda c, p, s: c,
                evaluate_fn=lambda s: 0.0,
                beam_width=1,
                n_partition_candidates=-3,
            )

    def test_n_partition_candidates_less_than_beam_width_raises(self):
        with pytest.raises(
            ValueError, match="n_partition_candidates.*must be >= beam_width"
        ):
            beam_search_aggregate(
                programs={},
                initial_solution=[0],
                extend_fn=lambda c, p, s: c,
                evaluate_fn=lambda s: 0.0,
                beam_width=5,
                n_partition_candidates=2,
            )


class TestBeamSearchAggregateGreedy:
    """Test greedy mode (beam_width=1)."""

    def test_single_partition_single_candidate(self):
        """Greedy with one partition and one candidate returns that candidate."""
        candidates = [SolutionEntry(bitstring="10", prob=0.8, decoded=[1, 0])]
        programs = {"A": _MockProgram(candidates)}
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
        programs = {"A": _MockProgram(candidates_a), "B": _MockProgram(candidates_b)}
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
        programs = {"A": _MockProgram(candidates_a), "B": _MockProgram(candidates_b)}
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
        programs = {"A": _MockProgram(candidates_a), "B": _MockProgram(candidates_b)}
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
        candidates_a = [
            SolutionEntry(bitstring="1", prob=0.6, decoded=[1]),
            SolutionEntry(bitstring="0", prob=0.4, decoded=[0]),
        ]
        candidates_b = [
            SolutionEntry(bitstring="1", prob=0.7, decoded=[1]),
            SolutionEntry(bitstring="0", prob=0.3, decoded=[0]),
        ]
        programs = {"A": _MockProgram(candidates_a), "B": _MockProgram(candidates_b)}
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

        candidates_a = [
            SolutionEntry(bitstring="1", prob=0.9, decoded=[1]),
            SolutionEntry(bitstring="0", prob=0.1, decoded=[0]),
        ]
        candidates_b = [
            SolutionEntry(bitstring="1", prob=0.9, decoded=[1]),
            SolutionEntry(bitstring="0", prob=0.1, decoded=[0]),
        ]
        candidates_c = [
            SolutionEntry(bitstring="1", prob=0.9, decoded=[1]),
            SolutionEntry(bitstring="0", prob=0.1, decoded=[0]),
        ]
        programs = {
            "A": _MockProgram(candidates_a),
            "B": _MockProgram(candidates_b),
            "C": _MockProgram(candidates_c),
        }
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
        programs = {"A": _MockProgram(candidates)}
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
        programs = {"A": _MockProgram([])}

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
        many_candidates = [
            SolutionEntry(bitstring="1", prob=0.9, decoded=[1]),
            SolutionEntry(bitstring="0", prob=0.1, decoded=[0]),
        ]
        programs = {"A": _MockProgram(many_candidates)}
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
        programs = {"A": _MockProgram(candidates)}
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
        programs = {"A": _MockProgram(candidates_a), "B": _MockProgram(candidates_b)}
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
        candidates_a = [
            SolutionEntry(bitstring="1", prob=0.5, decoded=[1]),
            SolutionEntry(bitstring="0", prob=0.5, decoded=[0]),
        ]
        programs = {"A": _MockProgram(candidates_a)}
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
        programs = {
            "A": _MockProgram(candidates),
            "B": _MockProgram(candidates),
            "C": _MockProgram(candidates),
        }
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
        programs = {
            "A": _MockProgram(candidates_a),
            "B": _MockProgram(candidates_b),
            "C": _MockProgram(candidates_c),
        }
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
        programs = {"A": _MockProgram(candidates)}
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
