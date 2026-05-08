# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tutorial: compare QAOA and PCE loss landscapes for the same QUBO.

Demonstrates 1D scans, 2D scans, and PCA scans with periodic trajectory
wrapping and trajectory overlay.
"""

import matplotlib.pyplot as plt
import numpy as np
import pennylane as qp

from divi.qprog import PCE, QAOA, GenericLayerAnsatz
from divi.qprog.optimizers import PymooMethod, PymooOptimizer
from divi.qprog.problems import BinaryOptimizationProblem
from tutorials._backend import get_backend


def _print_scan_summary(label, best_loss, line_scan, plane_scan, pca_scan):
    print(label)
    print(f"  Best loss after optimization: {best_loss:.6f}")
    print(f"  1D scan minimum: {line_scan.values.min():.6f}")
    print(f"  2D scan minimum: {plane_scan.values.min():.6f}")
    print(f"  PCA scan minimum: {pca_scan.values.min():.6f}")


def _plot_landscape_comparison(
    qaoa_line, pce_line, qaoa_plane, pce_plane, qaoa_pca, pce_pca
):
    fig, axes = plt.subplots(3, 2, figsize=(12, 12), constrained_layout=True)
    qaoa_line.plot(ax=axes[0, 0], color="tab:blue")
    pce_line.plot(ax=axes[0, 1], color="tab:orange")
    qaoa_plane.plot(ax=axes[1, 0], cmap="viridis")
    pce_plane.plot(ax=axes[1, 1], cmap="magma")
    # PCA scans with trajectory overlay showing the optimization path.
    qaoa_pca.plot(ax=axes[2, 0], cmap="cividis", show_trajectory=True)
    pce_pca.plot(ax=axes[2, 1], cmap="plasma", show_trajectory=True)
    fig.suptitle("QAOA vs PCE Loss-Landscape Comparison")

    plt.show()


if __name__ == "__main__":
    problem = np.array([[-1.0, 2.0], [0.0, 1.0]])
    # CMAES is population-based, so every iteration records a full generation
    # in ``param_history`` — ``scan_pca`` always has a rank-2 sample cloud
    # to fit. Scipy optimizers (COBYLA / Nelder-Mead) only record one accepted
    # iterate per step and collapse to a single point on flat landscapes.
    optimizer = PymooOptimizer(method=PymooMethod.CMAES, population_size=8)
    backend = get_backend(shots=4000)
    n_opt_iters = 12

    qaoa = QAOA(
        BinaryOptimizationProblem(problem),
        n_layers=1,
        optimizer=optimizer,
        max_iterations=n_opt_iters,
        backend=backend,
    )
    qaoa.run()

    # CMAES caches a per-dimension algorithm object on the optimizer instance;
    # reset before reusing across programs with different param counts.
    optimizer.reset()

    pce = PCE(
        problem=problem,
        ansatz=GenericLayerAnsatz(
            gate_sequence=[qp.RY, qp.RZ],
            entangler=qp.CNOT,
            entangling_layout="all-to-all",
        ),
        n_layers=1,
        optimizer=optimizer,
        max_iterations=n_opt_iters,
        alpha=1.0,
        backend=backend,
    )
    pce.run()

    # 1D scans via program.viz:
    qaoa_line = qaoa.viz.scan_1d(
        direction=np.array([1.0, -1.0]),
        n_points=31,
        span=(-1.0, 1.0),
    )
    pce_line = pce.viz.scan_1d(
        direction=np.ones_like(pce.best_params),
        n_points=31,
        span=(-0.6, 0.6),
    )

    # 2D scans:
    qaoa_plane = qaoa.viz.scan_2d(
        grid_shape=(15, 15),
        span_x=(-0.8, 0.8),
        span_y=(-0.8, 0.8),
        rng=0,
    )
    pce_plane = pce.viz.scan_2d(
        grid_shape=(15, 15),
        span_x=(-0.5, 0.5),
        span_y=(-0.5, 0.5),
        rng=1,
    )

    # PCA scans using the fluent .viz wrapper with auto-collected samples.
    # wrap_periodic=True ensures continuity across 2*pi boundaries for PCA.
    qaoa_pca = qaoa.viz.scan_pca(
        wrap_periodic=True,
        center=qaoa.best_params,
        grid_shape=(15, 15),
        span_x=(-0.8, 0.8),
        span_y=(-0.8, 0.8),
    )
    pce_pca = pce.viz.scan_pca(
        wrap_periodic=True,
        center=pce.best_params,
        grid_shape=(15, 15),
        span_x=(-0.5, 0.5),
        span_y=(-0.5, 0.5),
    )

    _print_scan_summary("QAOA", qaoa.best_loss, qaoa_line, qaoa_plane, qaoa_pca)
    _print_scan_summary("PCE", pce.best_loss, pce_line, pce_plane, pce_pca)

    _plot_landscape_comparison(
        qaoa_line, pce_line, qaoa_plane, pce_plane, qaoa_pca, pce_pca
    )
