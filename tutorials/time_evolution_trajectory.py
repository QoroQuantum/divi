# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""
Time Evolution Trajectory with Divi.

This tutorial demonstrates how to use TimeEvolutionTrajectory to simulate
quantum dynamics at multiple time points in parallel. The results are
aggregated into a time-ordered trajectory and plotted.

Examples:
1. Expectation-value trajectory: <Z> under H = X (Rabi oscillation).
2. Eigenstate trajectory: <Z> under H = Z₀ + Z₁ (constant eigenvalue).
"""

import math

import numpy as np
import pennylane as qml

from divi.qprog import TimeEvolutionTrajectory
from divi.qprog.ensemble import BatchConfig, BatchMode
from tutorials._backend import get_backend

if __name__ == "__main__":
    backend = get_backend(shots=5000)

    # ── Example 1: Rabi oscillation ──────────────────────────────────
    # H = X on a single qubit, observable = Z, |0⟩ initial state.
    # Exact: ⟨Z⟩(t) = cos(2t)
    time_points = np.linspace(0.01, math.pi, 20).tolist()

    trajectory = TimeEvolutionTrajectory(
        hamiltonian=qml.PauliX(0),
        time_points=time_points,
        observable=qml.PauliZ(0),
        initial_state="Zeros",
        backend=backend,
    )
    trajectory.create_programs()
    trajectory.run(blocking=True, batch_config=BatchConfig(mode=BatchMode.OFF))

    results = trajectory.aggregate_results()

    print("Example 1 — Rabi oscillation ⟨Z⟩(t) under H = X")
    print("-" * 50)
    exact_values = {t: math.cos(2 * t) for t in time_points}
    for t in time_points[::4]:  # Print every 4th point
        measured = results[t]
        exact = exact_values[t]
        print(f"  t={t:.3f}:  measured={measured:+.4f}  exact={exact:+.4f}")

    print(f"\n  Total circuits: {trajectory.total_circuit_count}")

    trajectory.visualize_results()

    # ── Example 2: Eigenstate (constant ⟨Z⟩) ────────────────────────
    # H = Z₀ + Z₁, observable = Z₀, |00⟩ initial state.
    # |00⟩ is an eigenstate of H, so ⟨Z₀⟩ = 1 at all times.
    time_points_2 = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    trajectory2 = TimeEvolutionTrajectory(
        hamiltonian=qml.PauliZ(0) + qml.PauliZ(1),
        time_points=time_points_2,
        observable=qml.PauliZ(0),
        initial_state="Zeros",
        backend=backend,
    )
    trajectory2.create_programs()
    trajectory2.run(blocking=True, batch_config=BatchConfig(mode=BatchMode.OFF))

    results2 = trajectory2.aggregate_results()

    print("\nExample 2 — Eigenstate: ⟨Z₀⟩ under H = Z₀ + Z₁")
    print("-" * 50)
    for t, expval in results2.items():
        print(f"  t={t:.1f}:  ⟨Z₀⟩ = {expval:+.4f}  (exact: +1.0000)")

    print(f"\n  Total circuits: {trajectory2.total_circuit_count}")

    trajectory2.visualize_results()
