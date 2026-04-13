# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tutorial: advanced loss-landscape analysis for a QAOA program.

Demonstrates interpolation scans, Hessian eigenvalue analysis, Fourier power
spectra, gradient overlays, 3D surface rendering, and the Nudged Elastic Band
(NEB) minimum-energy path finder.
"""

import matplotlib.pyplot as plt
import numpy as np

from divi.qprog import QAOA
from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer
from divi.qprog.problems import BinaryOptimizationProblem
from divi.viz import (
    GradientMethod,
    compute_hessian,
    fourier_analysis_2d,
    run_neb,
    scan_2d,
    scan_interp_1d,
    scan_interp_2d,
)
from tutorials._backend import get_backend

if __name__ == "__main__":
    problem = np.array([[-1.0, 2.0], [0.0, 1.0]])
    backend = get_backend(shots=4000)

    qaoa = QAOA(
        BinaryOptimizationProblem(problem),
        n_layers=1,
        optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
        max_iterations=10,
        backend=backend,
    )
    qaoa.run()

    best = qaoa.best_params
    n_params = len(best)
    other_point = best + np.array([0.5, -0.3])

    # ---- 1. Interpolation scans ----
    # 1D interpolation: objective along the line from best_params to another point.
    interp_1d = scan_interp_1d(qaoa, theta_1=best, theta_2=other_point, n_points=31)

    # 2D interpolation: plane with x = interpolation direction, y = orthogonal.
    interp_2d = scan_interp_2d(
        qaoa, theta_1=best, theta_2=other_point, grid_shape=(15, 15), rng=0
    )

    print("1D interpolation scan:")
    print(f"  Min value at t={interp_1d.offsets[np.argmin(interp_1d.values)]:.2f}:")
    print(f"  {interp_1d.values.min():.6f}")

    # ---- 2. Hessian eigenvalue analysis ----
    # GradientMethod.PARAMETER_SHIFT (default) gives exact derivatives for
    # standard quantum gates; use FINITE_DIFFERENCE as a universal fallback.
    hess = compute_hessian(qaoa, center=best)
    print(f"\nHessian at best_params (parameter-shift):")
    print(f"  Eigenvalues: {hess.eigenvalues}")
    ratio = (
        hess.eigenvalues[0] / hess.eigenvalues[-1] if hess.eigenvalues[-1] != 0 else 0
    )
    print(f"  Eigenvalue ratio (min/max): {ratio:.4f}")

    # Compare with finite-difference Hessian.
    hess_fd = compute_hessian(
        qaoa,
        center=best,
        gradient_method=GradientMethod.FINITE_DIFFERENCE,
        eps=1e-3,
    )
    print(f"\nHessian at best_params (finite-difference):")
    print(f"  Eigenvalues: {hess_fd.eigenvalues}")

    # Use steepest-curvature eigenvectors as scan directions.
    d1, d2 = hess.top_eigenvectors(k=2)
    hess_scan = scan_2d(
        qaoa,
        direction_x=d1,
        direction_y=d2,
        grid_shape=(31, 31),
        span_x=(-0.6, 0.6),
        span_y=(-0.6, 0.6),
    )

    # ---- 3. Fourier analysis ----
    spectrum = fourier_analysis_2d(hess_scan)
    print(f"\nFourier power spectrum shape: {spectrum.power_spectrum.shape}")

    # ---- 4. Nudged Elastic Band ----
    neb_result = run_neb(
        qaoa,
        theta_1=best,
        theta_2=other_point,
        n_pivots=8,
        n_steps=20,
        learning_rate=0.05,
    )
    print(f"\nNEB path barrier: {neb_result.energies.max():.6f}")
    print(
        f"  Endpoint energies: {neb_result.energies[0]:.6f}, {neb_result.energies[-1]:.6f}"
    )

    # ---- Plotting ----
    fig, axes = plt.subplots(3, 2, figsize=(14, 16), constrained_layout=True)

    # Row 1: interpolation scans
    interp_1d.plot(ax=axes[0, 0], color="tab:green")
    axes[0, 0].set_title("1D Interpolation")
    interp_2d.plot(ax=axes[0, 1], cmap="viridis")
    axes[0, 1].set_title("2D Interpolation")

    # Row 2: Hessian-directed scan with gradient overlay + Fourier spectrum
    hess_scan.plot(ax=axes[1, 0], cmap="coolwarm", show_gradients=True)
    axes[1, 0].set_title("Hessian Eigenvector Scan + Gradients")
    spectrum.plot(ax=axes[1, 1])

    # Row 3: 3D surface + NEB energy profile
    ax3d = fig.add_subplot(3, 2, 5, projection="3d")
    axes[2, 0].set_visible(False)
    hess_scan.plot_3d(ax=ax3d, cmap="coolwarm", alpha=0.8)
    ax3d.set_title("3D Surface")
    neb_result.plot(ax=axes[2, 1], color="tab:red")

    fig.suptitle("Advanced Loss-Landscape Analysis", fontsize=14)
    plt.show()
