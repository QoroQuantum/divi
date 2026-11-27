# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Sector-balanced portfolios: one XY ring for all assets, or one XY mixer per sector.

Pick 6 of 12 US industries, exactly 2 from each of three sectors, so that the
holdings are as little correlated as possible (``PortfolioSelectionProblem``
with the correlation matrix and ``risk_tolerance=0``: every industry scaled to
the same volatility). With the sector rule, adding the least correlated name
at each step no longer finds the best basket, which makes it a fair test
case.

Two ways to encode it with an XY mixer:

* **Global ring + penalty.** ``use_constrained_mixer=True`` keeps exactly 6
  names, and ``LinearConstraint`` adds a penalty for each sector that is not
  at 2. The mixer happily moves names across sectors, so many samples break
  the rule.
* **One mixer per sector.** Start from a Dicke state of weight 2 inside each
  sector and mix with a complete XY mixer inside each sector
  (``build_block_xy_mixer_edges`` with its default ``connectivity="complete"``).
  The circuit never leaves the sector-balanced portfolios. Every Dicke state
  is an eigenstate of the complete mixer, and the extra XY terms (6 per sector
  of 4 instead of 4 for a ring) add little depth next to the cost layer.
  ``connectivity="ring"`` also works, with fewer gates and somewhat worse
  results.

Post-processing matters too: ``feasibility="repair"`` fixes broken samples
with a greedy local search, which is strong on its own, so the script also
reports ``"filter"`` and a baseline that repairs random bitstrings. On this
small instance a single repaired random bitstring already finds the best
basket, so the fair comparison between the two encodings is the share of
balanced samples and how often the best basket is sampled, not the best
repaired sample.

Data: daily returns of the Kenneth R. French 49 industry portfolios,
30 June 2025 to 30 June 2026 (Kenneth R. French Data Library).
"""

import itertools

import numpy as np
from qiskit.circuit import QuantumCircuit

from divi.hamiltonians import xy_mixer
from divi.qprog.algorithms import QAOA
from divi.qprog.initial_states import (
    DickeState,
    InitialState,
    build_block_xy_mixer_edges,
)
from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer
from divi.qprog.problems import LinearConstraint, PortfolioSelectionProblem
from tutorials._backend import get_backend

SECTORS = {
    "Technology": ["Hardw", "Softw", "Chips", "LabEq"],
    "Finance": ["Banks", "Insur", "RlEst", "Fin"],
    "Staples": ["Food", "Beer", "Smoke", "Hshld"],
}
TICKERS = [t for names in SECTORS.values() for t in names]
PER_SECTOR = 2
K = PER_SECTOR * len(SECTORS)
BLOCKS = [list(range(4 * s, 4 * s + 4)) for s in range(len(SECTORS))]

CORR = np.array(
    [
        [1.00, 0.44, 0.68, 0.58, 0.31, -0.01, 0.17, 0.47, -0.17, -0.16, -0.14, -0.14],
        [0.44, 1.00, 0.52, 0.31, 0.32, 0.12, 0.35, 0.54, -0.17, -0.22, -0.16, -0.09],
        [0.68, 0.52, 1.00, 0.62, 0.25, -0.10, 0.19, 0.49, -0.29, -0.20, -0.20, -0.14],
        [0.58, 0.31, 0.62, 1.00, 0.43, 0.13, 0.44, 0.53, 0.02, 0.02, -0.20, 0.11],
        [0.31, 0.32, 0.25, 0.43, 1.00, 0.52, 0.57, 0.75, 0.09, 0.04, -0.04, 0.25],
        [-0.01, 0.12, -0.10, 0.13, 0.52, 1.00, 0.37, 0.39, 0.26, 0.12, 0.11, 0.34],
        [0.17, 0.35, 0.19, 0.44, 0.57, 0.37, 1.00, 0.58, 0.22, 0.12, -0.02, 0.30],
        [0.47, 0.54, 0.49, 0.53, 0.75, 0.39, 0.58, 1.00, -0.03, -0.08, -0.06, 0.10],
        [-0.17, -0.17, -0.29, 0.02, 0.09, 0.26, 0.22, -0.03, 1.00, 0.66, 0.32, 0.61],
        [-0.16, -0.22, -0.20, 0.02, 0.04, 0.12, 0.12, -0.08, 0.66, 1.00, 0.22, 0.55],
        [-0.14, -0.16, -0.20, -0.20, -0.04, 0.11, -0.02, -0.06, 0.32, 0.22, 1.00, 0.29],
        [-0.14, -0.09, -0.14, 0.11, 0.25, 0.34, 0.30, 0.10, 0.61, 0.55, 0.29, 1.00],
    ]
)


class SectorDickeState(InitialState):
    """A Dicke state of the same weight inside each block of qubits."""

    def __init__(self, hamming_weight: int, blocks: list[list[int]]):
        self.hamming_weight = hamming_weight
        self.blocks = blocks

    def build(self, wires) -> QuantumCircuit:
        qc = QuantumCircuit(len(wires))
        for block in self.blocks:
            qc.compose(
                DickeState(self.hamming_weight).build(block), qubits=block, inplace=True
            )
        return qc


class SectorBalancedProblem(PortfolioSelectionProblem):
    """Portfolio selection with a complete XY mixer inside each sector instead of one global ring."""

    @property
    def mixer_hamiltonian(self):
        edges = build_block_xy_mixer_edges(4, len(BLOCKS), list(range(self.n_assets)))
        return xy_mixer(edges, n_qubits=self.n_assets)

    @property
    def recommended_initial_state(self) -> InitialState:
        return SectorDickeState(PER_SECTOR, BLOCKS)


def avg_correlation(held) -> float:
    """The objective x^T C x / K^2 of an equal-weight basket."""
    x = np.zeros(len(TICKERS))
    x[list(held)] = 1
    return float(x @ CORR @ x) / K**2


def balanced(held) -> bool:
    return all(sum(i in block for i in held) == PER_SECTOR for block in BLOCKS)


def make_problem(cls):
    constraints = [
        LinearConstraint({i: 1 for i in block}, "==", PER_SECTOR / K)
        for block in BLOCKS
    ]
    # One name moved across sectors breaks two sectors by one name: penalty 2 * weight.
    return cls(
        np.zeros(len(TICKERS)),
        CORR,
        K,
        constraints=constraints,
        penalty_weight=0.1,
        use_constrained_mixer=True,
    )


def classical_references():
    baskets = [h for h in itertools.combinations(range(len(TICKERS)), K) if balanced(h)]
    values = [avg_correlation(h) for h in baskets]
    best, worst = baskets[int(np.argmin(values))], max(values)

    held: list[int] = []
    for _ in range(K):
        options = [
            i
            for i in range(len(TICKERS))
            if i not in held and sum(j in BLOCKS[i // 4] for j in held) < PER_SECTOR
        ]
        held.append(min(options, key=lambda i: avg_correlation(held + [i])))
    return baskets, best, worst, sorted(held)


def report(label, held, best, worst):
    ar = (worst - avg_correlation(held)) / (worst - avg_correlation(best))
    names = ", ".join(TICKERS[i] for i in sorted(held))
    print(
        f"  {label:<22} {names:<42} avg corr {avg_correlation(held):+.4f}  AR {ar:.3f}"
    )


def linear_ramp(n_layers: int, spread: float) -> np.ndarray:
    """LR-QAOA start (Montanez-Barrera and Michielsen, arXiv:2405.09169), gamma scaled to the objective."""
    i = np.arange(n_layers)
    params = np.empty((1, 2 * n_layers))
    params[0, 0::2] = (i + 1) / n_layers * 6.0 / spread
    params[0, 1::2] = -(1 - i / n_layers) * 0.4
    return params


def run_qaoa(problem, backend, label, best, worst, spread):
    qaoa = QAOA(
        problem,
        n_layers=2,
        optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
        max_iterations=60,
        backend=backend,
        seed=1,
    )
    qaoa.run(initial_params=linear_ramp(2, spread))
    samples = qaoa.get_top_solutions(n=0, include_decoded=True)
    ok = [s for s in samples if balanced(s.decoded)]
    share = sum(s.prob for s in ok)
    hit = sum(s.prob for s in ok if tuple(s.decoded) == best)
    print(f"{label}: {problem.cost_hamiltonian.num_qubits} qubits")
    if ok:
        mean_ar = (
            sum(s.prob * (worst - avg_correlation(s.decoded)) for s in ok)
            / share
            / (worst - avg_correlation(best))
        )
        print(
            f"  sector-balanced samples {share:.0%}, best basket {hit:.1%} of samples "
            f"(uniform: {1 / 216:.1%}), mean AR of a balanced sample {mean_ar:.3f}"
        )
        report(
            "best sample (filter)",
            min((s.decoded for s in ok), key=avg_correlation),
            best,
            worst,
        )
    else:
        print("  no sector-balanced sample, so filter has nothing to keep")
    repaired = qaoa.get_top_solutions(n=1, include_decoded=True, feasibility="repair")[
        0
    ]
    report("best sample (repair)", repaired.decoded, best, worst)


def random_repair(problem, n_samples, best, worst, seed=0):
    rng = np.random.default_rng(seed)
    held = [
        problem.repair_infeasible_bitstring(
            "".join(map(str, rng.integers(0, 2, len(TICKERS))))
        )[1]
        for _ in range(n_samples)
    ]
    report(f"{n_samples} random + repair", min(held, key=avg_correlation), best, worst)


if __name__ == "__main__":
    backend = get_backend(shots=2000)
    baskets, best, worst, greedy = classical_references()
    print(
        f"{len(baskets)} sector-balanced baskets of {K} out of {len(TICKERS)} industries"
    )
    report("exhaustive search", best, best, worst)
    report("greedy", greedy, best, worst)
    print()

    ring = make_problem(PortfolioSelectionProblem)
    # Scale for the angles: the spread of the objective. Here the exhaustive search
    # gives it; on larger problems a few hundred random baskets estimate it well.
    spread = avg_correlation(max(baskets, key=avg_correlation)) - avg_correlation(best)
    run_qaoa(ring, backend, "Global ring + sector penalty", best, worst, spread)
    print()
    block = make_problem(SectorBalancedProblem)
    run_qaoa(block, backend, "Complete XY per sector", best, worst, spread)
    print()
    print("Classical baseline, Divi's repair on random bitstrings:")
    for n_starts in (1, 3, 10):
        random_repair(ring, n_starts, best, worst)
