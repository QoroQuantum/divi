# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Molecular-Hamiltonian dispatch across optional chemistry frontends."""

from typing import Any

from qiskit.quantum_info import SparsePauliOp

from divi._optional import optional_module
from divi.hamiltonians._chem import molecular_hamiltonian_from_pyscf
from divi.hamiltonians._term_ops import to_spo


def molecular_hamiltonian(molecule: Any) -> tuple[SparsePauliOp, int]:
    """Build a Qiskit Hamiltonian from a PennyLane or PySCF molecule.

    Returns ``(hamiltonian, n_electrons)`` for both input families.
    """
    gto = optional_module("pyscf.gto")
    scf = optional_module("pyscf.scf")
    if (
        gto is not None
        and scf is not None
        and isinstance(molecule, (gto.Mole, scf.hf.SCF))
    ):
        return molecular_hamiltonian_from_pyscf(molecule)

    qp = optional_module("pennylane")
    if qp is not None and isinstance(molecule, qp.qchem.Molecule):
        hamiltonian, _ = qp.qchem.molecular_hamiltonian(molecule)
        return to_spo(hamiltonian), int(molecule.n_electrons)

    raise TypeError(
        "molecular_hamiltonian expects a PennyLane Molecule or PySCF Mole/"
        f"mean-field object, got {type(molecule).__name__}."
    )
