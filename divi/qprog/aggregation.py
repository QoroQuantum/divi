# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Strategies for aggregating per-partition solutions into a global solution.

A :class:`~divi.qprog.workflows.PartitioningProgramEnsemble` solves a decomposed
problem by running one
quantum program per partition. Each program yields a ranked list of candidate
bitstrings for its own variables; an :class:`AggregationStrategy` stitches those
per-partition candidates into scored global solutions.

Two strategies are provided:

- :class:`BeamSearchStrategy` — left-to-right beam search over partitions.
- :class:`HierarchicalStrategy` — divide-and-conquer beam that delays cross-group
  commitment via a pairwise merge tree.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from divi.qprog.variational_quantum_algorithm import (
    SolutionEntry,
    VariationalQuantumAlgorithm,
)

__all__ = [
    "AggregationStrategy",
    "BeamSearchStrategy",
    "HierarchicalStrategy",
]

ExtendFn = Callable[[list[int], Any, SolutionEntry], list[int]]
EvaluateFn = Callable[[list[int]], float]


class AggregationStrategy(ABC):
    """Combines per-partition candidates into scored global solutions.

    Subclasses implement :meth:`aggregate`, which receives the executed programs
    plus the problem-specific hooks needed to build and score global solutions,
    and returns the top-N ``(score, solution)`` pairs.
    """

    @abstractmethod
    def aggregate(
        self,
        programs: dict[Any, VariationalQuantumAlgorithm],
        initial_solution: Sequence[int],
        extend_fn: ExtendFn,
        evaluate_fn: EvaluateFn,
        top_n: int = 1,
    ) -> list[tuple[float, list[int]]]:
        """Aggregate per-partition candidates into the top-N global solutions.

        Args:
            programs: Mapping of program IDs to executed
                :class:`~divi.qprog.VariationalQuantumAlgorithm` instances.
            initial_solution: Starting global solution vector (typically all zeros).
            extend_fn: ``(current_solution, prog_id, candidate) -> extended_solution``;
                splices a partition's candidate into the global vector.
            evaluate_fn: ``(solution) -> float``. Lower is better.
            top_n: Number of top solutions to return.

        Returns:
            List of ``(score, solution)`` tuples sorted ascending by score
            (best first), with at most ``top_n`` entries.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement aggregate()."
        )


@dataclass(frozen=True)
class BeamSearchStrategy(AggregationStrategy):
    """Left-to-right beam search over partitions.

    At each partition step, every retained partial solution is extended by every
    fetched candidate, all extensions are scored, and only the best ``beam_width``
    are kept. ``beam_width=1`` is greedy; ``beam_width=None`` is exhaustive.

    Args:
        beam_width: Maximum candidates to retain per partition step. ``None`` keeps
            all extensions (exhaustive). Internally bumped to at least ``top_n`` so
            the beam can return enough solutions.
        n_partition_candidates: Candidates to fetch from each partition. Defaults to
            ``beam_width`` (or all when exhaustive). Must be ``>= beam_width``.
    """

    beam_width: int | None = 1
    n_partition_candidates: int | None = None

    def __post_init__(self):
        if self.beam_width is not None and self.beam_width < 1:
            raise ValueError(f"beam_width must be >= 1 or None, got {self.beam_width}")
        if self.n_partition_candidates is not None and self.n_partition_candidates < 1:
            raise ValueError(
                "n_partition_candidates must be >= 1 or None, got "
                f"{self.n_partition_candidates}"
            )

    def aggregate(
        self,
        programs: dict[Any, VariationalQuantumAlgorithm],
        initial_solution: Sequence[int],
        extend_fn: ExtendFn,
        evaluate_fn: EvaluateFn,
        top_n: int = 1,
    ) -> list[tuple[float, list[int]]]:
        if top_n < 1:
            raise ValueError(f"top_n must be >= 1, got {top_n}")

        # Ensure the beam retains enough candidates for top_n.
        beam_width = self.beam_width
        bumped = beam_width is not None and beam_width < top_n
        if bumped:
            beam_width = top_n

        n_partition_candidates = self.n_partition_candidates
        if (
            beam_width is not None
            and n_partition_candidates is not None
            and n_partition_candidates < beam_width
        ):
            beam_width_detail = (
                f"beam_width (bumped from {self.beam_width} to {beam_width} "
                f"to satisfy top_n={top_n})"
                if bumped
                else f"beam_width ({beam_width})"
            )
            raise ValueError(
                f"n_partition_candidates ({n_partition_candidates}) must be >= "
                f"{beam_width_detail}. Extracting fewer candidates than the "
                f"beam width wastes beam capacity."
            )

        if n_partition_candidates is not None:
            n_fetch = n_partition_candidates
        elif beam_width is not None:
            n_fetch = beam_width
        else:
            n_fetch = 2**20  # exhaustive

        initial_list = list(initial_solution)
        beam: list[tuple[float, list[int]]] = [
            (evaluate_fn(initial_list), initial_list)
        ]

        for prog_id, program in programs.items():
            candidates = program.get_top_solutions(n=n_fetch, include_decoded=True)
            if not candidates:
                continue

            new_beam: list[tuple[float, list[int]]] = []
            for _, partial_solution in beam:
                for candidate in candidates:
                    extended = extend_fn(partial_solution, prog_id, candidate)
                    new_beam.append((evaluate_fn(extended), extended))

            new_beam.sort(key=lambda entry: entry[0])
            beam = new_beam[:beam_width] if beam_width is not None else new_beam

        beam.sort(key=lambda entry: entry[0])
        return beam[:top_n]


# (score, solution, selections) — selections records the (prog_id, candidate)
# choices that built the solution, so a merge can rebuild it through extend_fn
# rather than assuming anything about how candidates map onto the solution vector.
_PoolEntry = tuple[float, list[int], list[tuple[Any, SolutionEntry]]]


@dataclass(frozen=True)
class HierarchicalStrategy(AggregationStrategy):
    """Divide-and-conquer beam that delays cross-group commitment.

    Partitions are split into groups of ``group_size``. Each group is solved
    independently with a beam of width ``max_per_group`` (so within a group this is
    *not* an exhaustive product — it is a pruned beam). The resulting group pools are
    then combined in a pairwise merge tree, re-scoring and pruning at each level. A
    merge rebuilds each combined solution by replaying both groups' candidate
    selections through ``extend_fn``, so combination semantics stay entirely with
    the problem (any ``extend_fn`` encoding works; nothing here assumes a particular
    "unset" value or that partitions own disjoint indices).

    Compared to a single left-to-right beam, deferring cross-group commitment lets
    each group retain prefixes that a global beam would prune early.

    Cost: a merge scores every pair of participating pool entries, so its work
    grows with the *square* of the per-merge fan-in. ``max_per_group`` is held
    fixed across all stages and is **not** inflated by ``top_n`` — ``top_n`` only
    widens the final slice, which adds no extra scoring. Use ``merge_width`` to cap
    the per-merge fan-in below ``max_per_group``.

    Limitations:

    - Groups are formed in decomposition order (consecutive program IDs); no
      coupling-aware grouping is performed. The benefit over a wide beam only
      materializes when strongly-coupled partitions happen to land in the same
      group.
    - Cross-group interactions in ``evaluate_fn`` (e.g. penalties spanning
      partitions in different groups) are invisible during per-group pruning and
      only surface at merge time.

    The output format matches :class:`BeamSearchStrategy`: ``(score, solution)``
    pairs sorted ascending (lower is better).

    Args:
        group_size: Maximum partitions per group.
        k_per_partition: Candidates to fetch from each partition.
        max_per_group: Maximum solutions retained per group and per merge level;
            the main quality/cost dial. Drives the per-merge cost; not inflated by
            ``top_n``.
        merge_width: Maximum entries from each pool that participate in a merge's
            Cartesian product, bounding per-merge cost to ``merge_width`` squared.
            ``None`` (default) uses all ``max_per_group`` entries. Setting it below
            ``top_n`` can limit how many distinct solutions are returnable.
    """

    group_size: int = 4
    k_per_partition: int = 20
    max_per_group: int = 200
    merge_width: int | None = None

    def __post_init__(self):
        if self.group_size < 1:
            raise ValueError(f"group_size must be >= 1, got {self.group_size}")
        if self.k_per_partition < 1:
            raise ValueError(
                f"k_per_partition must be >= 1, got {self.k_per_partition}"
            )
        if self.max_per_group < 1:
            raise ValueError(f"max_per_group must be >= 1, got {self.max_per_group}")
        if self.merge_width is not None and self.merge_width < 1:
            raise ValueError(
                f"merge_width must be >= 1 or None, got {self.merge_width}"
            )

    def aggregate(
        self,
        programs: dict[Any, VariationalQuantumAlgorithm],
        initial_solution: Sequence[int],
        extend_fn: ExtendFn,
        evaluate_fn: EvaluateFn,
        top_n: int = 1,
    ) -> list[tuple[float, list[int]]]:
        if top_n < 1:
            raise ValueError(f"top_n must be >= 1, got {top_n}")

        # Search width is fixed; top_n only widens the *final* pool (a sort+slice
        # that adds no scoring), so it never inflates the per-stage cost.
        search_cap = self.max_per_group
        result_cap = max(self.max_per_group, top_n)
        merge_width = self.merge_width if self.merge_width is not None else search_cap
        initial_list = list(initial_solution)

        prog_ids = list(programs.keys())
        if not prog_ids:
            return [(evaluate_fn(initial_list), initial_list)][:top_n]

        candidates_by_prog: dict[Any, list[SolutionEntry]] = {
            pid: programs[pid].get_top_solutions(
                n=self.k_per_partition, include_decoded=True
            )
            for pid in prog_ids
        }

        groups = [
            prog_ids[i : i + self.group_size]
            for i in range(0, len(prog_ids), self.group_size)
        ]

        def _extend_pool(
            base_pool: list[_PoolEntry], pid: Any, cap: int
        ) -> list[_PoolEntry]:
            """Extend each base entry with every candidate from *pid*."""
            cands = candidates_by_prog[pid]
            if not cands:
                return base_pool

            extended: list[_PoolEntry] = []
            for _, solution, selections in base_pool:
                for candidate in cands:
                    new_sol = extend_fn(solution, pid, candidate)
                    extended.append(
                        (
                            evaluate_fn(new_sol),
                            new_sol,
                            selections + [(pid, candidate)],
                        )
                    )

            extended.sort(key=lambda entry: entry[0])
            return extended[:cap]

        # Solve each group independently with a beam of width `search_cap`. When a
        # single group produces the final result (no merges), its last step keeps
        # `result_cap` so the call can still serve `top_n`.
        single_group = len(groups) == 1
        group_pools: list[list[_PoolEntry]] = []
        for group in groups:
            pool: list[_PoolEntry] = [(evaluate_fn(initial_list), initial_list, [])]
            for step, pid in enumerate(group):
                last_step = single_group and step == len(group) - 1
                pool = _extend_pool(pool, pid, result_cap if last_step else search_cap)
                if not pool:
                    break
            group_pools.append(pool)

        # Pairwise merge tree. A merge rebuilds each combined solution by replaying
        # pool B's selections through `extend_fn` onto each of pool A's solutions,
        # so the combination obeys whatever encoding `extend_fn` uses rather than a
        # hardcoded bit overlay. (Overlapping indices, if a problem ever produces
        # them, resolve to last-writer-wins in merge order.) Only the top
        # `merge_width` entries from each pool enter the Cartesian product; the
        # final level retains `result_cap` so the call can serve `top_n`.
        while len(group_pools) > 1:
            final_level = len(group_pools) <= 2
            out_cap = result_cap if final_level else search_cap
            next_level: list[list[_PoolEntry]] = []
            for i in range(0, len(group_pools), 2):
                if i + 1 >= len(group_pools):
                    next_level.append(group_pools[i])
                    continue

                pool_a, pool_b = group_pools[i], group_pools[i + 1]
                if not pool_a or not pool_b:
                    next_level.append(pool_a or pool_b)
                    continue

                merged: list[_PoolEntry] = []
                for _, sol_a, sel_a in pool_a[:merge_width]:
                    for _, _, sel_b in pool_b[:merge_width]:
                        combined = list(sol_a)
                        for pid, candidate in sel_b:
                            combined = extend_fn(combined, pid, candidate)
                        merged.append((evaluate_fn(combined), combined, sel_a + sel_b))

                merged.sort(key=lambda entry: entry[0])
                next_level.append(merged[:out_cap])

            group_pools = next_level

        final = group_pools[0] if group_pools else []
        final.sort(key=lambda entry: entry[0])
        return [(score, solution) for score, solution, _ in final[:top_n]]
