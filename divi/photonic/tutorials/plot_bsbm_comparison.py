# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Compare Born Machine configurations: baseline vs. tuned vs. more loops.

Produces a 3-panel figure comparing output distributions side by side,
with MMD convergence curves overlaid.
"""

import time
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

from divi.photonic import (
    BarsAndStripes,
    SimulatedTBISampler,
    TBIBornMachine,
    count_beamsplitters,
)


def run_bsbm(dataset, loop_lengths, shots, updates, lr, seed=0, label=""):
    n_bs = count_beamsplitters(dataset.n_bits, loop_lengths)
    print(f"\n  [{label}] loop_lengths={loop_lengths}, {n_bs} beamsplitters, "
          f"shots={shots}, updates={updates}, lr={lr}")

    sampler = SimulatedTBISampler(seed=seed)
    bsbm = TBIBornMachine(
        n_bits=dataset.n_bits,
        target_bits=dataset.patterns,
        n_modes=dataset.n_bits,
        n_photons=3,
        loop_lengths=loop_lengths,
        sampler=sampler,
    )

    t0 = time.time()
    result = bsbm.run(updates=updates, shots=shots, seed=seed, learning_rate=lr)
    elapsed = time.time() - t0

    target_set = {tuple(int(x) for x in p) for p in dataset.patterns}
    matches = sum(
        1 for row in result.final_samples_bits
        if tuple(int(x) for x in row) in target_set
    )
    fraction = matches / len(result.final_samples_bits)
    baseline = len(dataset.patterns) / (2 ** dataset.n_bits)
    uplift = fraction / baseline

    print(f"  [{label}] Done in {elapsed:.0f}s — final MMD={result.history_mmd[-1]:.4f}, "
          f"target fraction={fraction:.1%}, uplift={uplift:.1f}×")

    return result, fraction, uplift


def plot_panel(ax, result, dataset, title, fraction, uplift):
    target_set = {tuple(int(x) for x in p) for p in dataset.patterns}
    sample_tuples = [tuple(int(x) for x in row) for row in result.final_samples_bits]
    counts = Counter(sample_tuples)
    total = len(sample_tuples)

    all_observed = sorted(counts.keys())
    targets_observed = sorted(b for b in all_observed if b in target_set)
    others_observed = sorted(b for b in all_observed if b not in target_set)
    ordered = targets_observed + others_observed

    probs = [counts[b] / total for b in ordered]
    is_target = [b in target_set for b in ordered]
    bar_colors = ["#2563eb" if t else "#d1d5db" for t in is_target]

    x = np.arange(len(ordered))
    ax.bar(x, probs, color=bar_colors, edgecolor="none", width=0.8)

    baseline = 1.0 / (2 ** dataset.n_bits)
    ax.axhline(baseline, color="#ef4444", linestyle="--", linewidth=1.2, alpha=0.7)

    if targets_observed and others_observed:
        ax.axvline(len(targets_observed) - 0.5, color="#94a3b8",
                   linestyle=":", linewidth=1, alpha=0.6)

    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("Bitstring index", fontsize=9)
    ax.set_ylabel("Probability", fontsize=9)
    ax.set_xticks([])

    # Stats box.
    ax.text(
        0.97, 0.95,
        f"target: {fraction:.1%}\nuplift: {uplift:.1f}×\nmin MMD: {min(result.history_mmd):.3f}",
        transform=ax.transAxes, ha="right", va="top", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cbd5e1", alpha=0.9),
    )

    # MMD inset.
    ax_ins = ax.inset_axes([0.55, 0.42, 0.38, 0.38])
    ax_ins.plot(result.history_mmd, color="#7c3aed", linewidth=1.2)
    ax_ins.set_xlabel("update", fontsize=7)
    ax_ins.set_ylabel("MMD", fontsize=7)
    ax_ins.tick_params(labelsize=6)
    ax_ins.grid(True, alpha=0.3)


if __name__ == "__main__":
    dataset = BarsAndStripes(n=3)
    print(f"Dataset: {len(dataset.patterns)} target patterns, {dataset.n_bits} bits")

    configs = [
        {
            "loop_lengths": (1, 2),
            "shots": 80,
            "updates": 20,
            "lr": 5e-2,
            "label": "Baseline",
            "title": "Baseline: (1,2), 15 BS\n80 shots, 20 updates, lr=0.05",
        },
        {
            "loop_lengths": (1, 2),
            "shots": 200,
            "updates": 30,
            "lr": 3e-2,
            "label": "Tuned",
            "title": "Tuned: (1,2), 15 BS\n200 shots, 30 updates, lr=0.03",
        },
        {
            "loop_lengths": (1, 2, 3),
            "shots": 200,
            "updates": 30,
            "lr": 3e-2,
            "label": "More loops",
            "title": "More loops: (1,2,3), 21 BS\n200 shots, 30 updates, lr=0.03",
        },
    ]

    results = []
    for cfg in configs:
        r, frac, uplift = run_bsbm(
            dataset,
            loop_lengths=cfg["loop_lengths"],
            shots=cfg["shots"],
            updates=cfg["updates"],
            lr=cfg["lr"],
            label=cfg["label"],
        )
        results.append((r, frac, uplift, cfg))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)

    for ax, (r, frac, uplift, cfg) in zip(axes, results):
        plot_panel(ax, r, dataset, cfg["title"], frac, uplift)

    axes[0].set_ylabel("Empirical probability", fontsize=10)
    fig.suptitle(
        "Born Machine comparison — 3×3 bars & stripes (12 targets / 512 bitstrings)",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()

    save_path = str(
        __import__("pathlib").Path(__file__).resolve().parent / "bsbm_comparison.png"
    )
    fig.savefig(save_path, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"\nSaved → {save_path}")
    plt.close(fig)
