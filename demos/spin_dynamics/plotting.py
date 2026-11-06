# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Plotting helpers for the spin-dynamics demonstrations."""

import matplotlib.pyplot as plt
import numpy as np


def plot_dynamics(series, title, filename):
    """Plot magnetisation trajectories.

    Args:
        series: ``(label, times, values, fmt, colour)`` entries, one line each.
        title: Figure title.
        filename: Output image path.
    """
    plt.figure(figsize=(10, 6))
    for label, times, values, fmt, colour in series:
        plt.plot(
            times, values, fmt, color=colour, label=label, linewidth=2, markersize=8
        )
    plt.axhline(0, color="black", linestyle="-", alpha=0.2)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Time (t)")
    plt.ylabel(r"Magnetisation $\langle Z_0 \rangle$")
    plt.ylim(-1.1, 1.1)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"  Plot saved to {filename}")
    plt.close()


def compute_trajectory_error(reference, estimate):
    """Return the pointwise absolute error between aligned trajectories."""
    return np.abs(
        np.asarray(reference, dtype=float) - np.asarray(estimate, dtype=float)
    )


def plot_trajectory_error(times, errors, title, filename):
    """Plot QDrift's absolute magnetisation error against the reference."""
    plt.figure(figsize=(10, 4))
    plt.plot(times, errors, "o-", color="#dc2626", linewidth=2)
    plt.title(f"QDrift error: {title}", fontsize=14, fontweight="bold")
    plt.xlabel("Time (t)")
    plt.ylabel(
        r"$|\langle Z_0\rangle_\mathrm{ExactTrotterization}"
        r" - \langle Z_0\rangle_\mathrm{QDrift}|$"
    )
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"  Error plot saved to {filename}")
    plt.close()
