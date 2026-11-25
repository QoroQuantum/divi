# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""QSCI as PySCF's active-space solver, on N2 dissociation.

:class:`~divi.qprog.ensembles.qsci.QSCISolver` replaces PySCF's exact CI
solver: it samples a UCCSD VQE, grows the sampled subspace by second-order
contribution with a spin penalty, and diagonalises the Hamiltonian in it. The
active space is N2 in 6-31G, six electrons in eight orbitals (16 qubits).

Per bond length: the sampled and grown subspaces, each with its size and error
against exact CASCI, and ``<S^2>`` of the final state.
"""

from pyscf import gto, mcscf, scf

from divi.qprog.algorithms import UCCSDAnsatz
from divi.qprog.ensembles import ReportingLevel
from divi.qprog.ensembles.qsci import QSCISolver
from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer
from tutorials._backend import get_backend

N_ORBITALS, N_ELECTRONS = 8, 6
BOND_LENGTHS = (1.1, 1.6, 2.2, 2.7)


def run_casci(
    mean_field: scf.hf.RHF, solver: QSCISolver | None = None
) -> mcscf.casci.CASCI:
    casci = mcscf.CASCI(mean_field, N_ORBITALS, N_ELECTRONS)
    if solver is None:
        casci.fcisolver.conv_tol, casci.fcisolver.max_cycle = 1e-12, 1000
    else:
        casci.fcisolver = solver
    casci.kernel()
    return casci


if __name__ == "__main__":
    solver = QSCISolver(
        backend=get_backend(shots=10000),
        optimizer=ScipyOptimizer(ScipyMethod.L_BFGS_B),
        ansatz=UCCSDAnsatz(),
        max_iterations=30,
        qsci_options={"max_seed_configs": 200, "reporting_level": ReportingLevel.OFF},
    )

    print(
        f"{'R (Å)':>6} {'Sampled':>8} {'Error (mHa)':>12} "
        f"{'Grown':>6} {'Error (mHa)':>12} {'<S^2>':>9}"
    )
    for bond_length in BOND_LENGTHS:
        mol = gto.M(atom=f"N 0 0 0; N 0 0 {bond_length}", basis="6-31g", verbose=0)
        mean_field = scf.RHF(mol).run()
        exact = run_casci(mean_field).e_tot
        casci = run_casci(mean_field, solver)
        seed, final = casci.ci.records[0], casci.ci.records[-1]
        spin_square, _ = solver.spin_square(casci.ci, casci.ncas, casci.nelecas)
        print(
            f"{bond_length:>6.1f} {seed.subspace_size:>8} "
            f"{1e3 * (seed.energy - exact):>12.3f} {final.subspace_size:>6} "
            f"{1e3 * (final.energy - exact):>12.3f} {spin_square:>9.1e}"
        )
