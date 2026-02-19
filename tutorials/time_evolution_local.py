# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""
Time evolution with Divi TimeEvolution.

This tutorial demonstrates:
1. Probability-mode simulation for a known analytic case.
2. Expectation-value mode with a user-defined observable.
3. QDrift multi-sample averaging for randomized Trotterization.
"""

import math

import pennylane as qml

from divi.backends import ParallelSimulator
from divi.qprog import QDrift, TimeEvolution


def _print_sorted_probs(title: str, probs: dict[str, float], top_k: int = 8) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    sorted_items = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    for bitstring, prob in sorted_items[:top_k]:
        print(f"{bitstring}: {prob:.4f}")
    print(f"Total probability: {sum(probs.values()):.6f}")


if __name__ == "__main__":
    backend = ParallelSimulator(shots=5000)

    # Example 1: H = X0 + X1, |00>, t = pi/2
    # e^{-i t (X0 + X1)}|00> = |11> up to a global phase.
    h1 = qml.PauliX(0) + qml.PauliX(1)
    te_probs = TimeEvolution(
        hamiltonian=h1,
        time=math.pi / 2,
        initial_state="Zeros",
        backend=backend,
    )
    te_probs.run()
    _print_sorted_probs("Example 1 - Probability mode", te_probs.results["probs"])
    print(
        f"Circuits executed: {te_probs.total_circuit_count}, runtime: {te_probs.total_run_time:.4f}s"
    )

    # Example 2: expectation value of Z0 + Z1 after evolution under H = X0 + Z0.
    h2 = qml.PauliX(0) + qml.PauliZ(0)
    observable = qml.PauliZ(0)
    te_expval = TimeEvolution(
        hamiltonian=h2,
        time=0.6,
        n_steps=8,
        initial_state="Zeros",
        observable=observable,
        backend=backend,
    )
    te_expval.run()
    print("\nExample 2 - Observable mode")
    print("---------------------------")
    print(f"Estimated <Z0>: {te_expval.results['expval']:.6f}")
    print(
        f"Circuits executed: {te_expval.total_circuit_count}, runtime: {te_expval.total_run_time:.4f}s"
    )

    # Example 3: QDrift multi-sample averaging in probability mode.
    qdrift = QDrift(
        keep_fraction=0.5,
        sampling_budget=2,
        n_hamiltonians_per_iteration=4,
        sampling_strategy="weighted",
        seed=7,
    )
    te_qdrift = TimeEvolution(
        hamiltonian=0.7 * qml.PauliZ(0) + 0.4 * qml.PauliX(0) + 0.2 * qml.PauliZ(1),
        trotterization_strategy=qdrift,
        time=0.8,
        n_steps=1,
        initial_state="Superposition",
        backend=backend,
    )
    te_qdrift.run()
    _print_sorted_probs(
        "Example 3 - QDrift probability mode", te_qdrift.results["probs"]
    )
    print(
        f"Circuits executed: {te_qdrift.total_circuit_count}, runtime: {te_qdrift.total_run_time:.4f}s"
    )
