# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""
Time evolution with Divi TimeEvolution.

This tutorial demonstrates:
1. Probability-mode simulation for a known analytic case.
2. Expectation-value mode with a user-defined observable (compared to exact).
3. QDrift multi-sample averaging for randomized Trotterization.
"""

import math

import pennylane as qml

from divi.hamiltonians import QDrift
from divi.qprog import TimeEvolution
from tutorials._backend import get_backend


def _print_sorted_probs(title: str, probs: dict[str, float], top_k: int = 8) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    sorted_items = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    for bitstring, prob in sorted_items[:top_k]:
        print(f"  {bitstring}: {prob:.4f}")
    print(f"  Total probability: {sum(probs.values()):.6f}")


if __name__ == "__main__":
    backend = get_backend(shots=5000)

    # ── Example 1: Exact state transfer ──────────────────────────────
    # H = X₀ + X₁ with |00⟩ at t = π/2.
    # Because X₀ and X₁ commute, each qubit evolves independently:
    #   e^{-iπ/2 X} |0⟩ = -i|1⟩
    # so the final state is |11⟩ with certainty.
    h1 = qml.PauliX(0) + qml.PauliX(1)
    te_probs = TimeEvolution(
        hamiltonian=h1,
        time=math.pi / 2,
        initial_state="Zeros",
        backend=backend,
    )
    te_probs.run()
    _print_sorted_probs("Example 1 — Probability mode", te_probs.results["probs"])
    print(f"  Analytical: |11⟩ with probability 1.0")
    print(f"  Circuits executed: {te_probs.total_circuit_count}")

    # ── Example 2: Observable estimation vs. exact ───────────────────
    # H = X₀ + Z₀ (single-qubit), observable = Z₀, t = 0.6.
    # Exact: ⟨Z₀⟩ = cos²(√2 · t) ≈ 0.435 (derivable from H = √2 · n̂·σ̂).
    # Trotter error + shot noise cause a small deviation.
    h2 = qml.PauliX(0) + qml.PauliZ(0)
    t2 = 0.6
    observable = qml.PauliZ(0)
    te_expval = TimeEvolution(
        hamiltonian=h2,
        time=t2,
        n_steps=8,
        initial_state="Zeros",
        observable=observable,
        backend=backend,
    )
    te_expval.run()

    exact_z = math.cos(math.sqrt(2) * t2) ** 2

    print("\nExample 2 — Observable mode")
    print("---------------------------")
    print(f"  Estimated ⟨Z₀⟩:  {te_expval.results['expval']:.6f}")
    print(f"  Exact     ⟨Z₀⟩:  {exact_z:.6f}")
    print(f"  Error:            {abs(te_expval.results['expval'] - exact_z):.6f}")
    print(f"  Circuits executed: {te_expval.total_circuit_count}")

    # ── Example 3: QDrift randomized Trotterization ──────────────────
    # H = X₀ + 0.5·Z₀ + X₁, |00⟩, t = 1.0.
    # QDrift stochastically samples Hamiltonian terms each Trotter step.
    # Starting from |00⟩ the X terms rotate both qubits, producing a
    # structured (non-uniform) probability distribution.
    qdrift = QDrift(
        keep_fraction=0.5,
        sampling_budget=4,
        n_hamiltonians_per_iteration=4,
        sampling_strategy="weighted",
        seed=7,
    )
    h3 = qml.PauliX(0) + 0.5 * qml.PauliZ(0) + qml.PauliX(1)
    te_qdrift = TimeEvolution(
        hamiltonian=h3,
        trotterization_strategy=qdrift,
        time=1.0,
        n_steps=1,
        initial_state="Zeros",
        backend=backend,
    )
    te_qdrift.run()
    _print_sorted_probs(
        "Example 3 — QDrift probability mode", te_qdrift.results["probs"]
    )
    print(f"  Circuits executed: {te_qdrift.total_circuit_count}")
