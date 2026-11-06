# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Minimum Birkhoff Decomposition of a doubly stochastic matrix.

Runs the standalone-pipeline Birkhoff decomposition on a problem instance from
the Quantum Optimization Benchmarking Library and prints the reconstruction.

Usage:
    python main.py                          # defaults (n=4, k=2, sparse)
    python main.py -n 4 -k 2 -inst 5 -it 20
    python main.py --help                   # see all options
"""

import argparse
import json
import os

import numpy as np
from birkhoff import encode_combination, run_birkhoff
from qiskit.circuit.library import CZGate, RYGate

from divi.backends import MaestroSimulator
from divi.qprog.algorithms import GenericLayerAnsatz
from divi.qprog.optimizers import MonteCarloOptimizer, ScipyMethod, ScipyOptimizer


def parse_arguments():
    """Parse command-line arguments for the experiment."""
    parser = argparse.ArgumentParser(
        description=(
            "Decompose a doubly stochastic matrix from QOBLIB (Quantum "
            "Optimization Benchmarking Library): a parameterised circuit is "
            "sampled and CPLEX models score the decoded permutation combinations."
        )
    )
    parser.add_argument(
        "-n",
        "--dim",
        type=int,
        default=4,
        choices=range(3, 5),
        metavar="[3-4]",
        help="Matrix dimension (default: 4).",
    )
    parser.add_argument(
        "-k",
        "--comb",
        type=int,
        default=2,
        help="Number of permutations in the combination (default: 2).",
    )
    parser.add_argument(
        "-m",
        "--matrix_type",
        type=str,
        choices=["sparse", "dense"],
        default="sparse",
        help="Type of matrix example to use (default: sparse).",
    )
    parser.add_argument(
        "-inst",
        "--instance",
        type=int,
        default=1,
        choices=range(1, 11),
        metavar="[1-10]",
        help="Problem instance to load (default: 1).",
    )
    parser.add_argument(
        "-it",
        "--iterations",
        type=int,
        default=10,
        help="Max optimiser iterations (default: 10).",
    )
    parser.add_argument(
        "-opt",
        "--optimizer",
        type=str,
        choices=["Cobyla", "MonteCarlo"],
        default="Cobyla",
        help="Optimiser to use (default: Cobyla).",
    )
    return parser.parse_args()


class BColors:
    """ANSI escape sequences for colouring terminal text."""

    OKGREEN = "\033[92m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"


def parse_instance(
    json_data: dict, instance_idx: str
) -> tuple[np.ndarray, int, np.ndarray, np.ndarray]:
    """Parse a specific instance from the problem data."""
    instance_data = json_data[instance_idx]
    n = instance_data["n"]
    D = np.array(instance_data["scaled_doubly_stochastic_matrix"]).reshape((n, n))
    perms = np.eye(n, dtype=int)[
        np.array(instance_data["permutations"]).reshape(-1, n) - 1
    ]
    return D, instance_data["scale"], perms, np.array(instance_data["weights"])


def print_matrix_with_highlights(title: str, matrix: np.ndarray, highlights: dict):
    """Print a matrix with specified entries highlighted in colour."""
    print(f"\n{BColors.BOLD}{title}{BColors.ENDC}")
    for r, row_vals in enumerate(matrix):
        line = []
        for c, val in enumerate(row_vals):
            val_str = f"{val: .4f}"
            if (r, c) in highlights:
                color = highlights.get((r, c))
                val_str = f"{color}{val_str}{BColors.ENDC}"
            line.append(val_str)
        print("  [" + " ".join(line) + "]")


def present_final_results(
    original_matrix_scaled: np.ndarray,
    scale: int,
    found_combination: list[int],
    found_weights: list[float],
    all_perms_matrix: np.ndarray,
    solution_perms: np.ndarray,
    solution_weights: np.ndarray,
    final_histogram: dict[str, int],
    k: int,
):
    """Print the final results against the original matrix and known solution."""
    print("\n--- Final Birkhoff Decomposition Analysis ---")

    print(f"{BColors.BOLD}Decomposition (Found):{BColors.ENDC}")
    for perm_idx, weight in zip(found_combination, found_weights):
        if weight > 1e-6:
            print(f"  - Permutation Index {perm_idx}: Weight = {weight:.6f}")

    # QOBLIB instances may list the same permutation twice with split weights.
    print(f"\n{BColors.BOLD}Decomposition (Known Solution):{BColors.ENDC}")
    aggregated: dict[int | None, float] = {}
    for sol_perm_matrix, weight in zip(solution_perms, solution_weights):
        if weight <= 1e-6:
            continue
        match = next(
            (
                i
                for i, p in enumerate(all_perms_matrix)
                if np.array_equal(sol_perm_matrix, p)
            ),
            None,
        )
        aggregated[match] = aggregated.get(match, 0.0) + float(weight)
    for idx, weight in aggregated.items():
        label = f"Index {idx}" if idx is not None else "(unindexed)"
        print(f"  - Permutation {label}: Weight = {weight:.6f}")

    reconstructed = sum(
        weight * all_perms_matrix[perm_idx]
        for perm_idx, weight in zip(found_combination, found_weights)
        if weight > 1e-6
    )
    original_unscaled = original_matrix_scaled / scale

    print_matrix_with_highlights("Original Matrix (Unscaled)", original_unscaled, {})
    print_matrix_with_highlights("Reconstructed Matrix", reconstructed, {})

    error_matrix = original_unscaled - reconstructed
    abs_error_matrix = np.abs(error_matrix)
    max_error_pos = np.unravel_index(
        np.argmax(abs_error_matrix), abs_error_matrix.shape
    )
    zero_error_positions = np.argwhere(abs_error_matrix < 1e-9)

    error_highlights = {tuple(pos): BColors.OKGREEN for pos in zero_error_positions}
    error_highlights[max_error_pos] = BColors.FAIL

    print_matrix_with_highlights(
        "Error Matrix (Original - Reconstructed)", error_matrix, error_highlights
    )

    final_error_norm = np.linalg.norm(original_unscaled - reconstructed, ord=2)
    print(
        f"\n{BColors.BOLD}Final Error (spectral norm of the error matrix): "
        f"{final_error_norm:.6f}{BColors.ENDC}"
    )

    # Share of shots on the bitstring that encodes the selected combination.
    total_shots = sum(final_histogram.values())
    solution_integer = encode_combination(found_combination, k)
    bitstring_width = len(next(iter(final_histogram)))
    solution_bitstring = format(solution_integer, f"0{bitstring_width}b")
    solution_shots = final_histogram.get(solution_bitstring, 0)
    if total_shots > 0:
        probability = (solution_shots / total_shots) * 100
        print(
            f"  Selected-combination bitstring '{solution_bitstring}': "
            f"{solution_shots}/{total_shots} shots ({probability:.2f}%)"
        )
    print("---------------------------------------------")


def main(args):
    """Set up and run the experiment."""
    N_DIM = args.dim
    K_COMBINATIONS = args.comb
    INSTANCE_ID = str(args.instance)
    MAX_ITERATIONS = args.iterations
    MATRIX_TYPE = args.matrix_type
    OPTIMIZER_TYPE = args.optimizer

    print(f"--- Running Birkhoff Decomposition for n={N_DIM}, k={K_COMBINATIONS} ---")
    print(
        f"Instance: {INSTANCE_ID}, Matrix: {MATRIX_TYPE}, "
        f"Optimiser: {OPTIMIZER_TYPE}, Iterations: {MAX_ITERATIONS}"
    )

    dirname = os.path.dirname(__file__)
    mat_file = os.path.join(dirname, f"qbench_0{N_DIM}_{MATRIX_TYPE}.json")
    perm_file = os.path.join(dirname, f"p{N_DIM}.dat")

    with open(mat_file) as f:
        problem_data = json.load(f)
    permutations = np.loadtxt(perm_file, dtype=int)

    all_permutation_matrices = np.eye(N_DIM, dtype=int)[permutations - 1]

    matrix, scale, sol_perms, sol_weights = parse_instance(problem_data, INSTANCE_ID)

    if OPTIMIZER_TYPE == "Cobyla":
        optimizer = ScipyOptimizer(ScipyMethod.COBYLA)
    elif OPTIMIZER_TYPE == "MonteCarlo":
        optimizer = MonteCarloOptimizer(population_size=20, n_best_sets=5)
    else:
        raise ValueError(f"Unsupported optimiser type: {OPTIMIZER_TYPE}")

    backend = MaestroSimulator(shots=5000)
    ansatz = GenericLayerAnsatz([RYGate], entangler=CZGate, entangling_layout="brick")

    print("Starting optimisation...")
    result = run_birkhoff(
        matrix=matrix,
        scale=scale,
        k=K_COMBINATIONS,
        all_perms_matrix=all_permutation_matrices,
        backend=backend,
        optimizer=optimizer,
        max_iterations=MAX_ITERATIONS,
        ansatz=ansatz,
        n_layers=3,
    )

    if result.combination is None:
        print("No decomposition was obtained from the top measurement outcomes.")
        return

    print("\nMost frequent measured outcomes:")
    for row in result.top_outcomes[:5]:
        print(
            f"  {row['bitstring']}: {row['shots']} shots → "
            f"permutations {row['combination']}, error={row['error']:.6f}"
        )

    present_final_results(
        original_matrix_scaled=matrix,
        scale=scale,
        found_combination=result.combination,
        found_weights=result.weights,
        all_perms_matrix=all_permutation_matrices,
        solution_perms=sol_perms,
        solution_weights=sol_weights / scale,
        final_histogram=result.final_histogram,
        k=K_COMBINATIONS,
    )


if __name__ == "__main__":
    cli_args = parse_arguments()
    main(cli_args)
