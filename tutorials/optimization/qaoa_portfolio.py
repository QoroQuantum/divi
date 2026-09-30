# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Mean-variance portfolio optimisation with QAOA and PCE.

* ``PortfolioSelectionProblem`` picks ``K`` of ``N`` assets at equal weight,
  here with an ESG floor on the chosen assets. QAOA runs with a ring XY mixer,
  which conserves the number of held assets, from a Dicke state of weight
  ``K``; the ESG floor is enforced by a penalty and by repair.
* ``PortfolioAllocationProblem`` chooses weights on a grid of step ``1/L``.
  PCE runs on it, and ``feasibility="repair"`` repairs its samples to fully
  invested portfolios ranked by objective.

Both results are compared against an exhaustive classical search.
"""

import itertools

import numpy as np

from divi.qprog import PCE, QAOA
from divi.qprog.optimizers import MonteCarloOptimizer, ScipyMethod, ScipyOptimizer
from divi.qprog.problems import (
    LinearConstraint,
    PortfolioAllocationProblem,
    PortfolioSelectionProblem,
)
from tutorials._backend import get_backend

TICKERS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
MU = np.array([0.10, 0.12, 0.08, 0.15, 0.09, 0.11])
ESG = np.array([70.0, 55.0, 80.0, 60.0, 75.0, 65.0])
RISK_TOLERANCE = 0.5


def _covariance() -> np.ndarray:
    vol = np.array([0.20, 0.22, 0.17, 0.30, 0.19, 0.21])
    corr = np.full((6, 6), 0.3) + 0.7 * np.eye(6)
    return corr * np.outer(vol, vol)


def _print_metrics(label: str, holdings: str, metrics: dict) -> None:
    print(
        f"  {label:<10} {holdings:<32} return {metrics['expected_return']:.2%}  "
        f"volatility {metrics['volatility']:.2%}  Sharpe {metrics['sharpe']:.2f}"
    )


def select_assets(backend, sigma: np.ndarray) -> None:
    k = 3
    problem = PortfolioSelectionProblem(
        MU,
        sigma,
        n_holdings=k,
        risk_tolerance=RISK_TOLERANCE,
        constraints=[LinearConstraint(ESG, ">=", 68)],  # mean ESG >= 68
        # ESG sums move in steps of 5, so the smallest shortfall costs 0.05 * 5**2.
        penalty_weight=0.05,
        use_constrained_mixer=True,
    )
    n_slack = problem.cost_hamiltonian.num_qubits - len(TICKERS)
    print(f"Selection: {k} of {len(TICKERS)} assets, mean ESG >= 68")
    print(f"  qubits: {len(TICKERS)} assets + {n_slack} slack")

    # Exhaustive search over the C(6, 3) = 20 equal-weight portfolios.
    def objective(held):
        w = np.zeros(len(MU))
        w[list(held)] = 1 / k
        return w @ sigma @ w - RISK_TOLERANCE * MU @ w

    subsets = list(itertools.combinations(range(len(MU)), k))
    feasible = [h for h in subsets if ESG[list(h)].mean() >= 68]
    exact = min(feasible, key=objective)

    qaoa = QAOA(
        problem,
        n_layers=2,
        optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
        max_iterations=15,
        backend=backend,
    )
    qaoa.run()
    measured = sum(s.prob for s in qaoa.get_top_solutions(n=0, feasibility="filter"))
    print(
        f"  feasible before repair: {measured:.0%} of samples "
        f"({len(feasible) / len(subsets):.0%} of all {k}-asset portfolios)"
    )
    best = qaoa.get_top_solutions(n=1, include_decoded=True, feasibility="repair")[0]
    names = ", ".join(TICKERS[i] for i in best.decoded)
    _print_metrics("QAOA", names, problem.metrics(best.bitstring))

    x = "".join("1" if i in exact else "0" for i in range(len(MU)))
    exact_bits = problem.repair_infeasible_bitstring(x + "0" * n_slack)[0]
    _print_metrics(
        "Exhaustive", ", ".join(TICKERS[i] for i in exact), problem.metrics(exact_bits)
    )
    print(f"  objective gap: {best.energy - objective(exact):.1e}\n")


def allocate_weights(backend, sigma: np.ndarray) -> None:
    n, n_steps = 3, 7
    problem = PortfolioAllocationProblem(
        MU[:n], sigma[:n, :n], n_steps=n_steps, risk_tolerance=RISK_TOLERANCE
    )
    print(f"Allocation: weights of {TICKERS[:n]} in steps of 1/{n_steps}")
    pce = PCE(
        problem,
        encoding_type="poly",
        n_layers=2,
        optimizer=MonteCarloOptimizer(population_size=10),
        max_iterations=10,
        backend=backend,
        seed=7,
    )
    print(f"  {pce.n_vars} binary variables (3 per asset) on {pce.n_qubits} PCE qubits")
    pce.run()
    feasible = pce.get_top_solutions(n=0, feasibility="filter")
    print(f"  feasible before repair: {sum(s.prob for s in feasible):.0%} of samples")
    best = pce.get_top_solutions(n=1, include_decoded=True, feasibility="repair")[0]
    shown = ", ".join(f"{t} {w:.0%}" for t, w in zip(TICKERS, best.decoded))
    _print_metrics("PCE", shown, problem.metrics(best.bitstring))
    energy = best.energy

    grid = [
        np.array(k) / n_steps for k in itertools.product(range(n_steps + 1), repeat=n)
    ]
    grid = [w for w in grid if np.isclose(w.sum(), 1.0)]
    exact = min(grid, key=lambda w: w @ sigma[:n, :n] @ w - RISK_TOLERANCE * MU[:n] @ w)
    exact_energy = exact @ sigma[:n, :n] @ exact - RISK_TOLERANCE * MU[:n] @ exact
    print(
        "  Exhaustive "
        + ", ".join(f"{t} {w:.0%}" for t, w in zip(TICKERS, exact))
        + f"  objective gap: {energy - exact_energy:.1e}"
    )


if __name__ == "__main__":
    backend = get_backend(shots=5000)
    sigma = _covariance()
    select_assets(backend, sigma)
    allocate_weights(backend, sigma)
