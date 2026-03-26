# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Compare standard QAOA vs Iterative QAOA with parameter interpolation.

Both approaches optimize at every depth from 1 to MAX_DEPTH with the same
per-depth iteration budget. The only difference is initialization:

- **Standard QAOA** uses random initialization at each depth.
- **Iterative QAOA** warm-starts each depth by interpolating the optimal
  parameters from the previous depth.

This isolates the effect of parameter interpolation: any improvement must
come from better initialization, not from more compute.
"""

import networkx as nx
from rich.console import Console
from rich.table import Table

from divi.qprog import QAOA, InterpolationStrategy, IterativeQAOA
from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer
from divi.qprog.problems import MaxCutProblem
from tutorials._backend import get_backend

MAX_DEPTH = 8
ITERS_PER_DEPTH = 15
SHOTS = 10000
SEED = 42

if __name__ == "__main__":
    # ── Problem setup: MaxCut on a random 3-regular graph ─────────────
    graph = nx.random_regular_graph(3, 16, seed=SEED)

    backend = get_backend(shots=SHOTS)
    console = Console()

    # ── 1) Standard QAOA: random init at each depth ───────────────────
    console.rule("[bold]Standard QAOA (random init)")

    problem = MaxCutProblem(graph)

    standard_results = []
    for depth in range(1, MAX_DEPTH + 1):
        qaoa = QAOA(
            problem,
            n_layers=depth,
            optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
            max_iterations=ITERS_PER_DEPTH,
            backend=backend,
            seed=SEED,
        )
        qaoa.run(perform_final_computation=False)

        standard_results.append(
            {
                "depth": depth,
                "best_loss": qaoa.best_loss,
                "iterations": qaoa.current_iteration,
            }
        )
        console.print(
            f"  p={depth}  loss={qaoa.best_loss:.6f}  iters={qaoa.current_iteration}"
        )

    # ── 2) Iterative QAOA: warm-started init at each depth ────────────
    strategies = [
        InterpolationStrategy.INTERP,
        InterpolationStrategy.FOURIER,
        InterpolationStrategy.CHEBYSHEV,
    ]

    iterative_results = {}
    for strategy in strategies:
        console.rule(f"[bold]Iterative QAOA — {strategy.value}")

        iterative = IterativeQAOA(
            problem,
            max_depth=MAX_DEPTH,
            strategy=strategy,
            max_iterations_per_depth=ITERS_PER_DEPTH,
            backend=backend,
            optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
            seed=SEED,
        )
        iterative.run(perform_final_computation=False)

        iterative_results[strategy.value] = {
            "best_depth": iterative.best_depth,
            "best_loss": iterative.best_loss,
            "depth_history": iterative.depth_history,
        }

        for entry in iterative.depth_history:
            console.print(
                f"  p={entry['depth']}  loss={entry['best_loss']:.6f}  "
                f"iters={entry['n_iterations']}"
            )

    # ── 3) Head-to-head: loss at each depth ───────────────────────────
    console.print()
    console.rule(
        f"[bold]Loss by Depth ({ITERS_PER_DEPTH} iters/depth, "
        f"{graph.number_of_nodes()}-node graph)"
    )

    table = Table(show_header=True)
    table.add_column("Depth", justify="right")
    table.add_column("Standard", justify="right")
    for name in iterative_results:
        table.add_column(f"Iterative ({name})", justify="right")
    table.add_column("Best improvement", justify="right")

    for depth in range(1, MAX_DEPTH + 1):
        std_loss = standard_results[depth - 1]["best_loss"]
        row = [str(depth), f"{std_loss:.4f}"]

        iter_losses = []
        for name, result in iterative_results.items():
            entry = next(
                (e for e in result["depth_history"] if e["depth"] == depth), None
            )
            if entry:
                row.append(f"{entry['best_loss']:.4f}")
                iter_losses.append(entry["best_loss"])
            else:
                row.append("-")

        # Improvement: how much better is the best iterative vs standard?
        if iter_losses:
            best_iter = min(iter_losses)
            improvement = std_loss - best_iter  # positive = iterative is better
            if improvement > 0.01:
                row.append(f"[green]+{improvement:.4f}[/green]")
            elif improvement < -0.01:
                row.append(f"[red]{improvement:.4f}[/red]")
            else:
                row.append("~0")
        else:
            row.append("-")

        table.add_row(*row)

    console.print(table)

    # ── 4) Overall best ───────────────────────────────────────────────
    console.print()
    best_std = min(standard_results, key=lambda r: r["best_loss"])
    console.print(
        f"Standard best:  p={best_std['depth']}  loss={best_std['best_loss']:.6f}"
    )
    for name, result in iterative_results.items():
        console.print(
            f"Iterative ({name}) best:  p={result['best_depth']}  "
            f"loss={result['best_loss']:.6f}"
        )
