# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""ADAPT-VQE on LiH.

LiH in STO-3G, four electrons in four orbitals (8 qubits).

1. :class:`~divi.qprog.algorithms.AdaptVQE` with each operator pool, growing
   one operator per step and, with TETRIS, one per disjoint set of qubits:
   energy error against CASCI, steps and parameters.
2. The circuit grown at 1.6 Å reused as a fixed
   :class:`~divi.qprog.algorithms.AdaptAnsatz` along the bond, each geometry
   started from the previous one's parameters, against a fresh ADAPT-VQE run.
"""

from pyscf import gto, mcscf, scf

from divi.qprog.algorithms import VQE, AdaptAnsatz, AdaptVQE
from divi.qprog.algorithms.adapt import CEOPool, PauliPool, QubitExcitationPool
from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer
from divi.qprog.problems import MolecularProblem
from tutorials._backend import get_backend

N_ORBITALS, N_ELECTRONS = 4, 4
BOND_LENGTHS = (1.6, 2.0, 2.4, 2.8)


def build_problem(bond_length: float) -> tuple[MolecularProblem, float]:
    """The CAS(4,4) problem at ``bond_length`` and its exact energy."""
    mean_field = scf.RHF(
        gto.M(atom=f"Li 0 0 0; H 0 0 {bond_length}", basis="sto-3g", verbose=0)
    ).run()
    exact = mcscf.CASCI(mean_field, N_ORBITALS, N_ELECTRONS).kernel()[0]
    return MolecularProblem.from_casci(mean_field, N_ORBITALS, N_ELECTRONS), exact


if __name__ == "__main__":
    backend = get_backend()
    problems = {bond_length: build_problem(bond_length) for bond_length in BOND_LENGTHS}

    print("Part 1: operator pools and growth at 1.6 Å")
    problem, exact = problems[1.6]
    print(f"  {'pool':<18} {'growth':<7} {'error':>12} {'steps':>6} {'params':>7}")
    pools = {
        "qubit excitation": QubitExcitationPool(),
        "Pauli": PauliPool(),
        "CEO": CEOPool(),
    }
    grown = None
    for name, pool in pools.items():
        for growth in ("single", "tetris"):
            adapt = AdaptVQE(
                problem,
                pool=pool,
                growth=growth,
                optimizer=ScipyOptimizer(ScipyMethod.BFGS),
                backend=backend,
            )
            adapt.run(perform_final_computation=False)
            print(
                f"  {name:<18} {growth:<7} "
                f"{1e3 * (adapt.best_loss - exact):>8.4f} mHa "
                f"{len(adapt.step_history):>6} {len(adapt.best_params):>7}"
            )
            if name == "CEO" and growth == "tetris":
                grown = adapt

    print("\nPart 2: the grown circuit along the bond")
    print(f"  {'R (Å)':<6} {'reused error':>14} {'fresh error':>14}")
    params = grown.best_params
    for bond_length, (problem, exact) in problems.items():
        reused = VQE(
            problem,
            ansatz=AdaptAnsatz(grown.operators),
            optimizer=ScipyOptimizer(ScipyMethod.BFGS),
            max_iterations=100,
            backend=backend,
        )
        reused.run(initial_params=params[None, :], perform_final_computation=False)
        params = reused.best_params
        fresh = AdaptVQE(
            problem,
            pool=CEOPool(),
            growth="tetris",
            optimizer=ScipyOptimizer(ScipyMethod.BFGS),
            backend=backend,
        )
        fresh.run(perform_final_computation=False)
        print(
            f"  {bond_length:<6} {1e3 * (reused.best_loss - exact):>10.4f} mHa "
            f"{1e3 * (fresh.best_loss - exact):>10.4f} mHa"
        )
