# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tutorial: compare QAOA and PCE loss landscapes for the same QUBO."""

import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml

from divi.qprog import PCE, QAOA, GenericLayerAnsatz
from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer
from divi.qprog.problems import BinaryOptimizationProblem
from divi.viz import scan_1d, scan_pca
from tutorials._backend import get_backend

if __name__ == "__main__":
    problem = np.array([[-1.0, 2.0], [0.0, 1.0]])
    optimizer = ScipyOptimizer(method=ScipyMethod.COBYLA)
    backend = get_backend(shots=4000)
    # PCA fits the plane from the best-so-far iterate each iteration (not the full
    # population). Still need enough steps that those points span two directions;
    # COBYLA often repeats the same iterate, so keep max_iterations > 3.
    n_opt_iters = 12

    qaoa = QAOA(
        BinaryOptimizationProblem(problem),
        n_layers=1,
        optimizer=optimizer,
        max_iterations=n_opt_iters,
        backend=backend,
    )
    qaoa.run()

    pce = PCE(
        problem=problem,
        ansatz=GenericLayerAnsatz(
            gate_sequence=[qml.RY, qml.RZ],
            entangler=qml.CNOT,
            entangling_layout="all-to-all",
        ),
        n_layers=1,
        optimizer=optimizer,
        max_iterations=n_opt_iters,
        alpha=1.0,
        backend=backend,
    )
    pce.run()

    qaoa_line = scan_1d(
        qaoa,
        center=qaoa.best_params,
        direction=np.array([1.0, -1.0]),
        n_points=31,
        span=(-1.0, 1.0),
    )
    pce_line = scan_1d(
        pce,
        center=pce.best_params,
        direction=np.ones_like(pce.best_params),
        n_points=31,
        span=(-0.6, 0.6),
    )

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

    # PCA plane from the best member each callback (see param_history modes).
    qaoa_pca_samples = np.vstack(qaoa.param_history(mode="best_per_iteration"))
    pce_pca_samples = np.vstack(pce.param_history(mode="best_per_iteration"))

    qaoa_pca = scan_pca(
        qaoa,
        samples=qaoa_pca_samples,
        center=qaoa.best_params,
        grid_shape=(15, 15),
        span_x=(-0.8, 0.8),
        span_y=(-0.8, 0.8),
    )
    pce_pca = pce.viz.scan_pca(
        samples=pce_pca_samples,
        center=pce.best_params,
        grid_shape=(15, 15),
        span_x=(-0.5, 0.5),
        span_y=(-0.5, 0.5),
    )

    print("QAOA")
    print(f"  Best loss after optimization: {qaoa.best_loss:.6f}")
    print(f"  1D scan minimum: {qaoa_line.values.min():.6f}")
    print(f"  2D scan minimum: {qaoa_plane.values.min():.6f}")
    print(f"  PCA scan minimum: {qaoa_pca.values.min():.6f}")
    print("PCE")
    print(f"  Best loss after optimization: {pce.best_loss:.6f}")
    print(f"  1D scan minimum: {pce_line.values.min():.6f}")
    print(f"  2D scan minimum: {pce_plane.values.min():.6f}")
    print(f"  PCA scan minimum: {pce_pca.values.min():.6f}")

    fig, axes = plt.subplots(3, 2, figsize=(12, 12), constrained_layout=True)
    qaoa_line.plot(ax=axes[0, 0], color="tab:blue")
    pce_line.plot(ax=axes[0, 1], color="tab:orange")
    qaoa_plane.plot(ax=axes[1, 0], cmap="viridis")
    pce_plane.plot(ax=axes[1, 1], cmap="magma")
    qaoa_pca.plot(ax=axes[2, 0], cmap="cividis")
    pce_pca.plot(ax=axes[2, 1], cmap="plasma")
    fig.suptitle("QAOA vs PCE Loss-Landscape Comparison")

    plt.show()
