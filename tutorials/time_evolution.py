# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""
Time evolution with Divi TimeEvolution.

This tutorial demonstrates:
1. Probability-mode simulation for a known analytic case.
2. Expectation-value mode with a user-defined observable (compared to exact).
3. Multi-observable mode: several expectation values from one evolution.
4. QDrift multi-sample averaging for randomized Trotterization.
"""

import math

import pennylane as qp

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


def _print_observable_result(
    estimated: float, exact: float, circuit_count: int
) -> None:
    title = "Example 2 — Observable mode"
    print(f"\n{title}")
    print("-" * len(title))
    print(f"  Estimated ⟨Z₀⟩:  {estimated:.6f}")
    print(f"  Exact     ⟨Z₀⟩:  {exact:.6f}")
    print(f"  Error:            {abs(estimated - exact):.6f}")
    print(f"  Circuits executed: {circuit_count}")


def _print_multi_observable_result(
    labels: list[str],
    estimated: list[float],
    exact: list[float],
    multi_circuit_count: int,
    solo_circuit_count: int,
) -> None:
    title = "Example 3 — Multi-observable mode"
    print(f"\n{title}")
    print("-" * len(title))
    for label, est, ex in zip(labels, estimated, exact):
        print(
            f"  {label:<8} estimated: {est:+.6f}   "
            f"exact: {ex:+.6f}   error: {abs(est - ex):.6f}"
        )
    print(f"  Multi-observable run circuits: {multi_circuit_count}")
    print(f"  Three solo runs would use:     {solo_circuit_count}")
    saved = 1 - multi_circuit_count / solo_circuit_count
    print(f"  Saved by sharing the evolution: {saved:.0%}")


if __name__ == "__main__":
    backend = get_backend(shots=5000)

    # ── Example 1: Exact state transfer ──────────────────────────────
    # H = X₀ + X₁ with |00⟩ at t = π/2.
    # Because X₀ and X₁ commute, each qubit evolves independently:
    #   e^{-iπ/2 X} |0⟩ = -i|1⟩
    # so the final state is |11⟩ with certainty.
    h1 = qp.PauliX(0) + qp.PauliX(1)
    te_probs = TimeEvolution(
        hamiltonian=h1,
        time=math.pi / 2,
        backend=backend,
    )
    te_probs.run()
    _print_sorted_probs("Example 1 — Probability mode", te_probs.probabilities())
    print(f"  Analytical: |11⟩ with probability 1.0")
    print(f"  Circuits executed: {te_probs.total_circuit_count}")

    # ── Example 2: Observable estimation vs. exact ───────────────────
    # H = X₀ + Z₀ (single-qubit), observable = Z₀, t = 0.6.
    # Exact: ⟨Z₀⟩ = cos²(√2 · t) ≈ 0.435 (derivable from H = √2 · n̂·σ̂).
    # Trotter error + shot noise cause a small deviation.
    h2 = qp.PauliX(0) + qp.PauliZ(0)
    t2 = 0.6
    observable = qp.PauliZ(0)
    te_expval = TimeEvolution(
        hamiltonian=h2,
        time=t2,
        n_steps=8,
        observable=observable,
        backend=backend,
    )
    te_expval.run()

    exact_z = math.cos(math.sqrt(2) * t2) ** 2
    _print_observable_result(te_expval.expval(), exact_z, te_expval.total_circuit_count)

    # ── Example 3: Multi-observable from one evolution ───────────────
    # H = X₀ + X₁ at t = π/2, |00⟩ → |11⟩ (Example 1's analytic case).
    # Read three observables from the same evolved state:
    #   ⟨Z₀⟩ = -1, ⟨Z₁⟩ = -1, ⟨Z₀Z₁⟩ = +1
    # All three are diagonal in the Z basis and qubit-wise commute, so
    # MeasurementStage groups them into a single shot batch — the cost
    # is roughly that of measuring one observable, not three.
    h3 = qp.PauliX(0) + qp.PauliX(1)
    observables = [qp.PauliZ(0), qp.PauliZ(1), qp.PauliZ(0) @ qp.PauliZ(1)]
    te_multi = TimeEvolution(
        hamiltonian=h3,
        time=math.pi / 2,
        observable=observables,
        backend=backend,
    )
    te_multi.run()

    # Apples-to-apples baseline: measure each observable in its own run.
    solo_circuit_count = 0
    for obs in observables:
        te_solo = TimeEvolution(
            hamiltonian=h3,
            time=math.pi / 2,
            observable=obs,
            backend=backend,
        )
        te_solo.run()
        solo_circuit_count += te_solo.total_circuit_count

    _print_multi_observable_result(
        labels=["⟨Z₀⟩", "⟨Z₁⟩", "⟨Z₀Z₁⟩"],
        estimated=te_multi.expval(),
        exact=[-1.0, -1.0, +1.0],
        multi_circuit_count=te_multi.total_circuit_count,
        solo_circuit_count=solo_circuit_count,
    )

    # ── Example 4: QDrift randomized Trotterization ──────────────────
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
    h3 = qp.PauliX(0) + 0.5 * qp.PauliZ(0) + qp.PauliX(1)
    te_qdrift = TimeEvolution(
        hamiltonian=h3,
        trotterization_strategy=qdrift,
        time=1.0,
        n_steps=1,
        backend=backend,
    )
    te_qdrift.run()
    _print_sorted_probs(
        "Example 4 — QDrift probability mode", te_qdrift.probabilities()
    )
    print(f"  Circuits executed: {te_qdrift.total_circuit_count}")
