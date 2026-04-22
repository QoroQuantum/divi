# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Diagnostic plots for expert review of the photonic PoC.

Produces a two-panel figure:

  Left  — QUBO energy convergence (4 configs) with brute-force optimum.
  Right — Born machine final distribution: probability mass per bitstring,
          target patterns highlighted, random baseline shown.

Run::

    .venv/bin/python -m divi.photonic.tutorials.plot_diagnostics
"""

import itertools
import time

import matplotlib.pyplot as plt
import numpy as np

from divi.photonic import (
    BarsAndStripes,
    SimulatedTBISampler,
    TBIBornMachine,
    TBIVariationalQUBO,
)


def brute_force_qubo(Q: np.ndarray) -> tuple[tuple[int, ...], float]:
    M = Q.shape[0]
    best_bitstring = (0,) * M
    best_energy = float("inf")
    for bits in itertools.product([0, 1], repeat=M):
        b = np.asarray(bits, dtype=np.float64)
        energy = float(b @ Q @ b)
        if energy < best_energy:
            best_energy = energy
            best_bitstring = bits
    return best_bitstring, best_energy


def run_qubo():
    """Run QUBO and return per-config energy histories + brute-force optimum."""
    rng = np.random.default_rng(42)
    M = 4
    Q = rng.standard_normal((M, M))
    Q = 0.5 * (Q + Q.T)

    classical_bits, classical_energy = brute_force_qubo(Q)
    print(f"Brute-force optimum: {classical_bits} → {classical_energy:.4f}")

    sampler = SimulatedTBISampler(seed=0)
    solver = TBIVariationalQUBO(Q, loop_lengths=(1, 2), sampler=sampler)

    t0 = time.time()
    result = solver.run(
        updates=20,
        shots=200,
        seed=0,
        progress=lambda cfg, i, e: print(f"  {cfg} step {i:2d}: E={e:.3f}") if i % 5 == 0 else None,
    )
    elapsed = time.time() - t0
    print(f"QUBO done in {elapsed:.0f}s — found {result.best_bitstring} → {result.best_value:.4f}")

    return result, classical_energy


def run_born_machine():
    """Run BSBM and return the result + dataset."""
    dataset = BarsAndStripes(n=3)
    sampler = SimulatedTBISampler(seed=0)
    bsbm = TBIBornMachine(
        n_bits=dataset.n_bits,
        target_bits=dataset.patterns,
        n_modes=dataset.n_bits,
        n_photons=3,
        loop_lengths=(1, 2, 3),
        sampler=sampler,
    )

    t0 = time.time()
    result = bsbm.run(
        updates=30,
        shots=200,
        seed=0,
        learning_rate=3e-2,
        progress=lambda i, mmd: print(f"  step {i:2d}: MMD={mmd:.4f}") if i % 5 == 0 else None,
    )
    elapsed = time.time() - t0
    print(f"BSBM done in {elapsed:.0f}s — final MMD={result.history_mmd[-1]:.4f}")

    return result, dataset


def plot(qubo_result, classical_energy, bsbm_result, dataset, save_path):
    """Generate the two-panel diagnostic figure."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ── Left panel: QUBO convergence ────────────────────────────────
    colors = ["#2563eb", "#7c3aed", "#db2777", "#ea580c"]
    for idx, (cfg_label, cfg_data) in enumerate(qubo_result.per_config.items()):
        history = cfg_data["history_energy"]
        ax1.plot(history, label=cfg_label, color=colors[idx], linewidth=1.5, alpha=0.85)

    ax1.axhline(
        classical_energy, color="#16a34a", linestyle="--", linewidth=1.5,
        label=f"brute-force optimum ({classical_energy:.2f})",
    )
    ax1.set_xlabel("Adam update", fontsize=11)
    ax1.set_ylabel("Mean QUBO energy  $\\langle b^T Q b \\rangle$", fontsize=11)
    ax1.set_title("Variational QUBO convergence (4-config sweep)", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9, loc="upper right")
    ax1.grid(True, alpha=0.3)

    # ── Right panel: Born machine distribution ──────────────────────
    n_bits = dataset.n_bits
    target_set = {tuple(int(x) for x in p) for p in dataset.patterns}

    # Tally final model samples into bitstring counts.
    from collections import Counter
    sample_tuples = [tuple(int(x) for x in row) for row in bsbm_result.final_samples_bits]
    counts = Counter(sample_tuples)
    total = len(sample_tuples)

    # Sort bitstrings: targets first (sorted), then non-targets (sorted).
    all_observed = sorted(counts.keys())
    targets_observed = sorted(b for b in all_observed if b in target_set)
    others_observed = sorted(b for b in all_observed if b not in target_set)
    ordered = targets_observed + others_observed

    probs = [counts[b] / total for b in ordered]
    is_target = [b in target_set for b in ordered]
    bar_colors = ["#2563eb" if t else "#d1d5db" for t in is_target]

    x = np.arange(len(ordered))
    ax2.bar(x, probs, color=bar_colors, edgecolor="none", width=0.8)

    # Random baseline.
    baseline = 1.0 / (2 ** n_bits)
    ax2.axhline(baseline, color="#ef4444", linestyle="--", linewidth=1.2,
                label=f"random baseline ({baseline:.1%})")

    # Mark the boundary between target and non-target bitstrings.
    if targets_observed and others_observed:
        ax2.axvline(len(targets_observed) - 0.5, color="#94a3b8", linestyle=":",
                    linewidth=1, alpha=0.7)
        ax2.text(len(targets_observed) / 2, max(probs) * 0.92, "target\npatterns",
                 ha="center", va="top", fontsize=8, color="#2563eb", fontweight="bold")
        ax2.text(len(targets_observed) + len(others_observed) / 2, max(probs) * 0.92,
                 "other\nbitstrings", ha="center", va="top", fontsize=8, color="#6b7280")

    ax2.set_xlabel("Bitstring index (targets left, others right)", fontsize=11)
    ax2.set_ylabel("Empirical probability", fontsize=11)
    ax2.set_title("Born machine output distribution (3×3 bars & stripes)", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9, loc="upper right")
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.set_xticks([])

    # Inset: MMD convergence.
    ax_inset = ax2.inset_axes([0.55, 0.45, 0.4, 0.4])
    ax_inset.plot(bsbm_result.history_mmd, color="#7c3aed", linewidth=1.5)
    ax_inset.set_xlabel("update", fontsize=8)
    ax_inset.set_ylabel("MMD", fontsize=8)
    ax_inset.set_title("MMD convergence", fontsize=9, fontweight="bold")
    ax_inset.tick_params(labelsize=7)
    ax_inset.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"\nSaved → {save_path}")
    plt.close(fig)


if __name__ == "__main__":
    print("=" * 60)
    print("Running QUBO …")
    print("=" * 60)
    qubo_result, classical_energy = run_qubo()

    print()
    print("=" * 60)
    print("Running Born Machine …")
    print("=" * 60)
    bsbm_result, dataset = run_born_machine()

    save_path = str(
        __import__("pathlib").Path(__file__).resolve().parent / "photonic_diagnostics.png"
    )
    plot(qubo_result, classical_energy, bsbm_result, dataset, save_path)
