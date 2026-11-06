# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Visualisation utilities for the quantum-guided cluster algorithm benchmark.

Plots: approximation-ratio bars, correlation heatmaps and energy-distribution
violins.
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

COLORS = {
    "bg": "#0f0f1a",
    "panel": "#1a1a2e",
    "text": "#e0e0f0",
    "accent": "#00d4ff",
    "sa": "#ff6b6b",
    "cc": "#ffa502",
    "grid": "#2d2d44",
}
QAOA_COLORS = ["#7bed9f", "#70a1ff", "#a29bfe", "#fd79a8", "#dfe6e9"]


def apply_dark_style():
    """Apply a dark matplotlib style."""
    plt.rcParams.update(
        {
            "figure.facecolor": COLORS["bg"],
            "axes.facecolor": COLORS["panel"],
            "axes.edgecolor": COLORS["grid"],
            "axes.labelcolor": COLORS["text"],
            "text.color": COLORS["text"],
            "xtick.color": COLORS["text"],
            "ytick.color": COLORS["text"],
            "grid.color": COLORS["grid"],
            "grid.alpha": 0.3,
            "font.family": "sans-serif",
            "font.size": 12,
        }
    )


def plot_approximation_ratios(
    results: dict, save_path: str = "approximation_ratios.png"
):
    """Plot 1: Bar chart of the mean of r = E / E₀ over restarts, ± std, per method."""
    apply_dark_style()
    fig, ax = plt.subplots(figsize=(12, 7))

    E_ground = results["E_ground"]
    if E_ground is None or abs(E_ground) < 1e-10:
        print("Cannot plot approximation ratios without ground state energy.")
        return

    methods = []
    means = []
    stds = []
    colors = []
    edge_colors = []

    sa_ratios = [e / E_ground for e in results["sa_result"].energy_history]
    methods.append("Simulated\nAnnealing")
    means.append(np.mean(sa_ratios))
    stds.append(np.std(sa_ratios))
    colors.append(COLORS["sa"])
    edge_colors.append(COLORS["sa"])

    cc_ratios = [e / E_ground for e in results["cc_result"].energy_history]
    methods.append("Coupling\nConstants")
    means.append(np.mean(cc_ratios))
    stds.append(np.std(cc_ratios))
    colors.append(COLORS["cc"])
    edge_colors.append(COLORS["cc"])

    for i, (label, res) in enumerate(results["quantum_results"].items()):
        ratios = [e / E_ground for e in res.energy_history]
        methods.append(label.replace(" ", "\n"))
        means.append(np.mean(ratios))
        stds.append(np.std(ratios))
        c = QAOA_COLORS[i % len(QAOA_COLORS)]
        colors.append(c)
        edge_colors.append(c)

    x = np.arange(len(methods))
    bars = ax.bar(
        x,
        means,
        yerr=stds,
        width=0.6,
        color=[c + "40" for c in colors],
        edgecolor=colors,
        linewidth=2,
        capsize=5,
        error_kw={"elinewidth": 2, "capthick": 2, "ecolor": COLORS["text"]},
    )

    for bar, mean, std in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std + 0.005,
            f"{mean:.3f}",
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
            color=COLORS["text"],
        )

    ax.set_ylabel("Approximation Ratio  r = E / E₀", fontsize=14, fontweight="bold")
    ax.set_title(
        "Quantum-Guided Cluster Algorithm: Mean Approximation Ratio\n"
        "(error bars = ±1σ over random restarts)",
        fontsize=15,
        fontweight="bold",
        pad=15,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=12)
    ax.set_ylim(min(0.7, min(means) - 0.05), 1.02)
    ax.axhline(
        y=1.0, color=COLORS["accent"], linestyle="--", alpha=0.5, label="r = 1 (E = E₀)"
    )
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(axis="y", alpha=0.2)

    plt.tight_layout()
    plt.savefig(
        save_path,
        dpi=200,
        bbox_inches="tight",
        facecolor=COLORS["bg"],
        edgecolor="none",
    )
    print(f"Saved: {save_path}")
    plt.close()


def plot_correlation_heatmaps(
    G: nx.Graph,
    correlations: dict[str, np.ndarray],
    save_path: str = "correlation_heatmaps.png",
):
    """Plot 2: Side-by-side heatmaps of Z for each correlation source."""
    apply_dark_style()
    n_plots = len(correlations)
    # tight_layout warns when a colourbar spans multiple axes.
    fig, axes = plt.subplots(
        1, n_plots, figsize=(5 * n_plots + 1, 5), layout="constrained"
    )
    if n_plots == 1:
        axes = [axes]

    cmap = plt.cm.RdBu_r
    vmax = max(np.abs(Z).max() for Z in correlations.values())

    for ax, (label, Z) in zip(axes, correlations.items()):
        im = ax.imshow(Z, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="equal")
        ax.set_title(label, fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel("Variable j", fontsize=11)
        ax.set_ylabel("Variable i", fontsize=11)

    fig.suptitle(
        "Correlation matrices by source",
        fontsize=15,
        fontweight="bold",
    )

    cbar = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cbar.set_label("Z_ij", fontsize=12)

    plt.savefig(
        save_path,
        dpi=200,
        bbox_inches="tight",
        facecolor=COLORS["bg"],
        edgecolor="none",
    )
    print(f"Saved: {save_path}")
    plt.close()


def plot_energy_distribution(
    results: dict, save_path: str = "energy_distributions.png"
):
    """Plot 3: Violin plot of r over restarts, per method."""
    apply_dark_style()
    fig, ax = plt.subplots(figsize=(12, 7))

    E_ground = results["E_ground"]
    if E_ground is None or abs(E_ground) < 1e-10:
        print("Cannot plot distributions without ground state energy.")
        return

    all_data = []
    labels = []
    colors_list = []

    sa_ratios = [e / E_ground for e in results["sa_result"].energy_history]
    all_data.append(sa_ratios)
    labels.append("Simulated\nAnnealing")
    colors_list.append(COLORS["sa"])

    cc_ratios = [e / E_ground for e in results["cc_result"].energy_history]
    all_data.append(cc_ratios)
    labels.append("Coupling\nConstants")
    colors_list.append(COLORS["cc"])

    for i, (label, res) in enumerate(results["quantum_results"].items()):
        ratios = [e / E_ground for e in res.energy_history]
        all_data.append(ratios)
        labels.append(label.replace(" ", "\n"))
        colors_list.append(QAOA_COLORS[i % len(QAOA_COLORS)])

    positions = np.arange(len(all_data))
    parts = ax.violinplot(
        all_data,
        positions=positions,
        showmeans=True,
        showmedians=True,
        showextrema=True,
    )

    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors_list[i] + "40")
        pc.set_edgecolor(colors_list[i])
        pc.set_linewidth(2)
        pc.set_alpha(0.7)

    for key in ["cmeans", "cmedians", "cmins", "cmaxes", "cbars"]:
        if key in parts:
            parts[key].set_color(COLORS["text"])
            parts[key].set_linewidth(1.5)

    ax.axhline(
        y=1.0, color=COLORS["accent"], linestyle="--", alpha=0.5, label="r = 1 (E = E₀)"
    )
    ax.set_ylabel("Approximation Ratio  r = E / E₀", fontsize=14, fontweight="bold")
    ax.set_title(
        "Energy Distribution Across Random Restarts",
        fontsize=15,
        fontweight="bold",
        pad=15,
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=12)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(axis="y", alpha=0.2)

    plt.tight_layout()
    plt.savefig(
        save_path,
        dpi=200,
        bbox_inches="tight",
        facecolor=COLORS["bg"],
        edgecolor="none",
    )
    print(f"Saved: {save_path}")
    plt.close()
