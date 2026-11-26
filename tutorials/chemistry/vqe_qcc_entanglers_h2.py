# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Choosing qubit coupled-cluster entanglers for H2.

H2 in STO-3G, two electrons in two orbitals (4 qubits), with
:class:`~divi.qprog.algorithms.QCCAnsatz`.

1. The default entanglers, one per spin-conserving excitation.
2. The energy gradient at Hartree-Fock of every Pauli word on the four qubits
   with an odd number of ``Y``, measured with the parameter-shift rule in one
   batched evaluation.
3. One of the words with the largest gradient alone as ``entanglers``, and an
   even-``Y`` word, which the ansatz rejects.
"""

import itertools

import numpy as np
from pyscf import gto, mcscf, scf

from divi.backends import CircuitRunner
from divi.qprog import EarlyStopping
from divi.qprog.algorithms import VQE, QCCAnsatz
from divi.qprog.optimizers import RotosolveOptimizer
from divi.qprog.problems import MolecularProblem
from tutorials._backend import get_backend


def make_vqe(
    problem: MolecularProblem, ansatz: QCCAnsatz, backend: CircuitRunner
) -> VQE:
    return VQE(
        problem,
        ansatz=ansatz,
        optimizer=RotosolveOptimizer(),
        max_iterations=30,
        early_stopping=EarlyStopping(patience=1, min_delta=1e-7),
        backend=backend,
    )


def measure_gradients_at_reference(
    problem: MolecularProblem, words: list[str], backend: CircuitRunner
) -> np.ndarray:
    """``dE/dtheta`` at zero for each word, by ``[E(pi/2) - E(-pi/2)] / 2``."""
    n_words = len(words)
    vqe = make_vqe(problem, QCCAnsatz(words), backend)
    shifts = (np.pi / 2) * np.vstack([np.eye(n_words), -np.eye(n_words)])
    energies = vqe.evaluate(shifts, vqe.cost_preprocessor())
    values = np.ravel([energies[index] for index in range(2 * n_words)])
    return (values[:n_words] - values[n_words:]) / 2


if __name__ == "__main__":
    backend = get_backend()
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    mean_field = scf.RHF(mol).run()
    fci_energy = mcscf.CASCI(mean_field, 2, 2).kernel()[0]
    problem = MolecularProblem.from_casci(mean_field, 2, 2)
    print(f"Hartree-Fock is {1e3 * (mean_field.e_tot - fci_energy):.3f} mHa above FCI")

    print("\nPart 1: default entanglers")
    vqe = make_vqe(problem, QCCAnsatz(), backend)
    vqe.run(perform_final_computation=False)
    print(
        f"  {vqe.n_params} entanglers, "
        f"{1e3 * (vqe.best_loss - fci_energy):.4f} mHa above FCI"
    )

    print("\nPart 2: energy gradients of odd-Y words at Hartree-Fock")
    pool = [
        "".join(letters)
        for letters in itertools.product("IXYZ", repeat=4)
        if letters.count("Y") % 2 == 1
    ]
    magnitudes = np.round(
        np.abs(measure_gradients_at_reference(problem, pool, backend)), 6
    )
    nonzero = [word for word, size in zip(pool, magnitudes) if size > 0]
    largest = [word for word, size in zip(pool, magnitudes) if size == magnitudes.max()]
    print(f"  {len(nonzero)} of {len(pool)} words have a nonzero gradient")
    print(
        f"  {len(largest)} share the largest, {magnitudes.max():.4f}: {' '.join(largest)}"
    )

    print("\nPart 3: a single entangler")
    vqe = make_vqe(problem, QCCAnsatz([largest[0]]), backend)
    vqe.run(perform_final_computation=False)
    print(f"  {largest[0]}: {1e3 * (vqe.best_loss - fci_energy):.4f} mHa above FCI")
    try:
        QCCAnsatz(["XXYY"])
    except ValueError as error:
        print(f"  XXYY: {error}")
