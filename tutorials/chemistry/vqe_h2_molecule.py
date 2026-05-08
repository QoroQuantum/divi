# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""VQE on the H2 molecule.

Three parts, all driven through :class:`~divi.qprog.algorithms.VQE`:

1. Basic VQE — minimal setup with :class:`~divi.qprog.HartreeFockAnsatz` and
   a ``dry_run()`` preview of the pipeline's circuit fan-out before execution.
2. Grouping strategy comparison — how the ``grouping_strategy`` argument
   (``None`` / ``"wires"`` / ``"qwc"``) changes the number of measurement
   groups and the resulting loss.
3. Shot allocation — within a fixed grouping strategy, how
   ``shot_distribution`` (``None`` / ``"uniform"`` / ``"weighted"``) splits
   the backend's shot budget across the groups produced by QWC grouping.
"""

import time

import numpy as np
import pennylane as qp

from divi.pipeline import format_dry_run
from divi.qprog import VQE, HartreeFockAnsatz
from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer
from tutorials._backend import get_backend

if __name__ == "__main__":
    mol = qp.qchem.Molecule(
        symbols=["H", "H"], coordinates=np.array([(0, 0, 0), (0, 0, 0.5)])
    )

    # ------------------------------------------------------------------ #
    # Part 1 — Basic VQE run with a dry-run preview.
    # ------------------------------------------------------------------ #
    print("Part 1 — Basic VQE on H2")
    print("-" * 40)

    vqe_problem = VQE(
        molecule=mol,
        ansatz=HartreeFockAnsatz(),
        n_layers=1,
        optimizer=ScipyOptimizer(method=ScipyMethod.L_BFGS_B),
        max_iterations=3,
        backend=get_backend(),
    )

    # --- Dry run: inspect pipeline fan-out before executing ---
    format_dry_run(vqe_problem.dry_run())

    t1 = time.time()
    vqe_problem.run()

    print(f"Minimum Energy Achieved: {vqe_problem.best_loss:.4f}")
    print(f"Eigenstate: {vqe_problem.eigenstate}")
    print(f"Total circuits: {vqe_problem.total_circuit_count}")
    print(f"Time taken: {round(time.time() - t1, 5)} seconds")

    # ------------------------------------------------------------------ #
    # Part 2 — How does the choice of grouping strategy affect the run?
    # ------------------------------------------------------------------ #
    print("\nPart 2 — grouping strategy comparison")
    print("-" * 40)

    backend = get_backend(shots=500, force_sampling=True)

    vqe_input = dict(
        molecule=mol,
        n_layers=1,
        ansatz=HartreeFockAnsatz(),
        optimizer=ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
        max_iterations=1,
        seed=2000,
        backend=backend,
    )

    strategies = [
        ("None", VQE(**vqe_input, grouping_strategy=None)),
        ("Wires", VQE(**vqe_input, grouping_strategy="wires")),
        ("QWC", VQE(**vqe_input, grouping_strategy="qwc")),
    ]

    n_groups_by_strategy: dict[str, int] = {}
    for name, problem in strategies:
        cost_report = problem.dry_run()["cost"]
        measurement_info = next(
            s for s in cost_report.stages if s.name == "MeasurementStage"
        )
        n_groups_by_strategy[name] = measurement_info.metadata["n_groups"]

    for name, problem in strategies:
        backend.set_seed(1997)
        problem.run()

    header = f"{'Strategy':<10} {'Groups':>8} {'Loss':>14} {'Circuits':>10}"
    print(header)
    print("-" * len(header))
    for name, problem in strategies:
        print(
            f"{name:<10} {n_groups_by_strategy[name]:>8} "
            f"{problem.best_loss:>14.6f} {problem.total_circuit_count:>10}"
        )

    losses = [p.best_loss for _, p in strategies]
    max_diff = max(losses) - min(losses)
    print(f"\nMax loss difference: {max_diff:.6f}")
    assert max_diff < 0.5, f"Losses diverged too much: {losses}"

    # ------------------------------------------------------------------ #
    # Part 3 — Adaptive shot allocation within a fixed grouping strategy.
    #
    # With QWC grouping on H2 the Hamiltonian splits into a handful of
    # measurement groups whose coefficient L1 norms span more than an order
    # of magnitude.  ``shot_distribution`` controls how the backend's total
    # shot budget is split across those groups.  The three paths differ in
    # how many total shots they spend AND how those shots are allocated:
    #
    # * ``None`` (default) — every group sampled with ``backend.shots``;
    #   total shots scale with the number of groups.
    # * ``"uniform"`` — the backend budget is split evenly across groups.
    # * ``"weighted"`` — the same budget, split proportional to each group's
    #   coefficient L1 norm so dominant terms receive more shots.
    #
    # Rather than running a full optimization per strategy, we inspect the
    # allocation directly by running the cost pipeline's forward pass once
    # and reading ``env.artifacts["per_group_shots"]``.
    # ------------------------------------------------------------------ #
    print(f"\nPart 3 — per-group shot allocation (backend.shots = {backend.shots})")
    print("-" * 40)
    header2 = f"{'shot_distribution':<18} {'per-group shots':<40} {'total':>10}"
    print(header2)
    print("-" * len(header2))

    allocations: dict[str, list[int]] = {}
    for shot_dist in ("uniform", "weighted"):
        vqe = VQE(**vqe_input, grouping_strategy="qwc", shot_distribution=shot_dist)
        cost_report = vqe.dry_run()["cost"]
        spec_alloc = next(iter(cost_report.env_artifacts["per_group_shots"].values()))
        n_groups = max(spec_alloc.keys()) + 1
        allocations[shot_dist] = [spec_alloc.get(i, 0) for i in range(n_groups)]

    # With shot_distribution=None, MeasurementStage submits each group with
    # the backend's full shot count — derive the per-group view from the
    # uniform run's group count for a clean comparison.
    n_groups = len(allocations["uniform"])
    allocations["None (default)"] = [backend.shots] * n_groups

    for name in ("None (default)", "uniform", "weighted"):
        alloc = allocations[name]
        print(f"{name:<18} {str(alloc):<40} {sum(alloc):>10}")

    print(
        "\nWith ``None``, the backend pays ``shots × n_groups`` total; with\n"
        "``uniform`` or ``weighted`` the same total is capped at ``shots`` and\n"
        "spread differently. ``weighted`` concentrates samples on the groups\n"
        "that dominate the Hamiltonian's L1 norm — reducing estimator variance\n"
        "on skewed chemistry Hamiltonians at no extra cost."
    )
