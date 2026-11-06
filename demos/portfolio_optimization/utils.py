# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""
Utility functions for portfolio optimisation.

This module provides helper functions for building QUBO matrices
and other common operations in portfolio optimisation.
"""

import dimod
import numpy as np
import numpy.typing as npt


def _extract_dimod_solution(sample_set: dimod.SampleSet) -> npt.NDArray[np.integer]:
    """Return the lowest-energy sample of a ``SampleSet`` as a numpy int array."""
    return np.asarray(sample_set.lowest().record.sample[0], dtype=np.int_)


def build_full_portfolio_qubo(
    returns: npt.NDArray[np.floating],
    covariance_matrix: npt.NDArray[np.floating],
    lambda_param: float = 0.75,
) -> npt.NDArray[np.floating]:
    """
    Build a single Markowitz QUBO matrix for the full portfolio.

    QUBO formulation: Minimise Risk - λ·Return  =  x^T Σ x - λ μ^T x

    - Diagonal: Q_ii = Σ_ii - λ·μ_i  (variance minus lambda-weighted return)
    - Off-diagonal: Q_ij = Σ_ij  (covariance)

    Args:
        returns: Expected returns vector. Shape: (n,)
        covariance_matrix: Covariance matrix. Shape: (n, n)
        lambda_param: Risk-return trade-off parameter. Default 0.75.

    Returns:
        QUBO matrix Q where the objective is to minimise x^T Q x.
    """
    qubo_matrix = covariance_matrix.copy()
    np.fill_diagonal(qubo_matrix, np.diag(qubo_matrix) - lambda_param * returns)
    if not np.allclose(qubo_matrix, qubo_matrix.T):
        qubo_matrix = (qubo_matrix + qubo_matrix.T) / 2
    return qubo_matrix


def _compute_portfolio_metrics(
    returns: np.ndarray,
    covariance_matrix: np.ndarray,
    bitstring: np.ndarray,
) -> tuple[float, float, float]:
    """
    Compute portfolio return, risk, and Sharpe ratio for a given bitstring.

    Sharpe Ratio = Return / sqrt(Risk)
    (Using variance as risk, so we take square root to get standard deviation)
    """
    portfolio_return = np.dot(returns, bitstring)
    portfolio_risk = bitstring @ covariance_matrix @ bitstring
    # Returns inf when the variance is not positive.
    sharpe_ratio = (
        portfolio_return / np.sqrt(portfolio_risk)
        if portfolio_risk > 0
        else float("inf")
    )
    return portfolio_return, portfolio_risk, sharpe_ratio


def evaluate_solution(
    qubo_matrix: np.ndarray,
    qaoa_solution: np.ndarray,
    returns: np.ndarray,
    covariance_matrix: np.ndarray,
    partition_id: int | None = None,
):
    """
    Evaluate QAOA solution by comparing it to the optimal classical solution.

    Computes both QUBO energies and actual financial metrics (returns and risk).

    Args:
        qubo_matrix: The QUBO matrix for this partition (already built with lambda)
        qaoa_solution: The bitstring solution from QAOA
        returns: Expected returns vector for the assets in this partition
        covariance_matrix: Covariance matrix for the assets in this partition
        partition_id: Optional ID of the cluster/partition (for logging only)
    """
    if qubo_matrix.shape[0] != qubo_matrix.shape[1]:
        raise ValueError(f"QUBO matrix must be square, got shape {qubo_matrix.shape}")

    if len(qaoa_solution) != qubo_matrix.shape[0]:
        raise ValueError(
            f"Solution length {len(qaoa_solution)} does not match QUBO size {qubo_matrix.shape[0]}"
        )

    if len(returns) != qubo_matrix.shape[0]:
        raise ValueError(
            f"Returns vector length {len(returns)} does not match QUBO size {qubo_matrix.shape[0]}"
        )

    if covariance_matrix.shape != qubo_matrix.shape:
        raise ValueError(
            f"Covariance matrix shape {covariance_matrix.shape} does not match QUBO shape {qubo_matrix.shape}"
        )

    bqm = dimod.BinaryQuadraticModel(qubo_matrix, "BINARY")
    sample_set = dimod.ExactSolver().sample(bqm)
    best_classical_bitstring = _extract_dimod_solution(sample_set)
    best_classical_energy = sample_set.lowest().record[0].energy

    qaoa_energy = bqm.energy(qaoa_solution)
    is_correct = np.array_equal(best_classical_bitstring, qaoa_solution)

    partition_label = (
        f"Partition {partition_id}" if partition_id is not None else "Partition"
    )
    print(f"{partition_label} Evaluation:")
    print()
    print("QUBO Energy Metrics:")
    print(f"  Optimal Classical Energy: {best_classical_energy:.6f}")
    print(f"  QAOA Energy: {qaoa_energy:.6f}")
    print(f"  Energy Difference: {qaoa_energy - best_classical_energy:.6f}")
    print(f"  Matches Optimal: {is_correct}")

    compare_portfolio_solutions(
        qaoa_solution,
        best_classical_bitstring,
        returns,
        covariance_matrix,
    )

    print()
    print("Solutions:")
    print(f"  Optimal Bitstring: {best_classical_bitstring}")
    print(f"  QAOA Bitstring:    {str(qaoa_solution).replace(',', '')}")


def compare_portfolio_solutions(
    qaoa_solution: npt.NDArray[np.integer],
    exact_solution: npt.NDArray[np.integer],
    returns: npt.NDArray[np.floating],
    covariance_matrix: npt.NDArray[np.floating],
) -> None:
    """
    Compare QAOA and exact solver portfolio solutions by computing and displaying financial metrics.

    Args:
        qaoa_solution: QAOA portfolio bitstring
        exact_solution: Exact solver portfolio bitstring
        returns: Expected returns vector for all assets
        covariance_matrix: Covariance matrix for all assets
    """
    qaoa_return, qaoa_risk, qaoa_sharpe = _compute_portfolio_metrics(
        returns, covariance_matrix, qaoa_solution
    )
    exact_return, exact_risk, exact_sharpe = _compute_portfolio_metrics(
        returns, covariance_matrix, exact_solution
    )

    qaoa_n_assets = np.sum(qaoa_solution)
    exact_n_assets = np.sum(exact_solution)

    print("\nFinancial Metrics:")
    print("  QAOA Portfolio:")
    print(f"    Return: {qaoa_return:.6f}")
    print(f"    Risk (Variance): {qaoa_risk:.6f}")
    print(f"    Sharpe Ratio: {qaoa_sharpe:.6f}")
    print(f"    Number of Assets: {int(qaoa_n_assets)}")
    print("  ExactSolver Portfolio:")
    print(f"    Return: {exact_return:.6f}")
    print(f"    Risk (Variance): {exact_risk:.6f}")
    print(f"    Sharpe Ratio: {exact_sharpe:.6f}")
    print(f"    Number of Assets: {int(exact_n_assets)}")
    print("  Difference:")
    print(f"    Return Difference: {qaoa_return - exact_return:.6f}")
    print(f"    Risk Difference: {qaoa_risk - exact_risk:.6f}")
    print(f"    Sharpe Ratio Difference: {qaoa_sharpe - exact_sharpe:.6f}")

    solutions_match = np.array_equal(qaoa_solution, exact_solution)
    print(f"\nSolutions Match: {solutions_match}")
    if not solutions_match:
        diff_count = np.sum(qaoa_solution != exact_solution)
        print(
            f"  Number of differing positions: {diff_count} out of {len(qaoa_solution)}"
        )
        qaoa_selected = np.where(qaoa_solution == 1)[0]
        exact_selected = np.where(exact_solution == 1)[0]
        print(f"  QAOA selected assets ({len(qaoa_selected)}): {qaoa_selected}")
        print(f"  Exact selected assets ({len(exact_selected)}): {exact_selected}")
