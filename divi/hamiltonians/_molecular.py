# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Molecule frontends (PySCF and PennyLane) and the dispatch across them."""

from typing import TYPE_CHECKING, Any, TypeGuard, TypeVar

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from divi._optional import module_if_imported
from divi.hamiltonians._chem import _spo_from_integrals
from divi.hamiltonians._term_ops import to_spo

if TYPE_CHECKING:
    from pennylane.qchem import Molecule as _PennyLaneMolecule
    from pyscf.gto import Mole as _PySCFMole
    from pyscf.scf.hf import SCF as _PySCFMeanField
else:
    _PennyLaneMolecule = _PySCFMole = _PySCFMeanField = Any

_PySCFInput = _PySCFMole | _PySCFMeanField
_MeanField = TypeVar("_MeanField", bound=_PySCFMeanField)

#: Every molecule representation :meth:`MolecularProblem.from_molecule` accepts.
MoleculeLike = _PennyLaneMolecule | _PySCFInput


def is_pyscf_mole(molecule: object) -> TypeGuard[_PySCFMole]:
    gto = module_if_imported("pyscf.gto")
    return gto is not None and isinstance(molecule, gto.Mole)


def is_pyscf_mean_field(molecule: object) -> TypeGuard[_PySCFMeanField]:
    scf = module_if_imported("pyscf.scf")
    return scf is not None and isinstance(molecule, scf.hf.SCF)


def is_pyscf_input(molecule: object) -> TypeGuard[_PySCFInput]:
    return is_pyscf_mole(molecule) or is_pyscf_mean_field(molecule)


def is_pennylane_molecule(molecule: object) -> TypeGuard[_PennyLaneMolecule]:
    qchem = module_if_imported("pennylane.qchem")
    return qchem is not None and isinstance(molecule, qchem.Molecule)


def split_pyscf_input(
    molecule: _PySCFInput, *, open_shell: bool = False
) -> tuple[_PySCFMole, _PySCFMeanField | None]:
    """Validate a PySCF input and return ``(mol, mean_field or None)``.

    Args:
        molecule: A ``gto.Mole`` or mean-field object.
        open_shell: Accept a non-zero ``mol.spin``, with a restricted
            open-shell (ROHF/ROKS) mean field.

    Raises:
        TypeError: If ``molecule`` is neither a ``Mole`` nor a mean-field.
        NotImplementedError: If the system is open-shell (non-zero spin)
            without ``open_shell``, or the mean-field is not restricted (e.g.
            UHF, UKS or GHF), whether or not it has been run.
    """
    if is_pyscf_mole(molecule):
        mean_field = None
        mol = molecule
    elif is_pyscf_mean_field(molecule):
        mean_field = molecule
        mol = mean_field.mol
    else:
        raise TypeError(
            "Expected a pyscf Mole or mean-field object, "
            f"got {type(molecule).__name__}."
        )

    if mol.spin != 0 and not open_shell:
        raise NotImplementedError(
            "Only closed-shell (RHF) systems are supported here; got an "
            f"open-shell molecule with spin={mol.spin}. LASSQD accepts "
            "open-shell molecules."
        )

    if mean_field is not None:
        from pyscf.scf import hf

        if not isinstance(mean_field, hf.RHF):
            raise NotImplementedError(
                "Only restricted mean-fields (RHF/RKS, or ROHF/ROKS for an "
                f"open-shell molecule) are supported; got "
                f"{type(mean_field).__name__}."
            )
    return mol, mean_field


def without_chkfile(mean_field: _MeanField) -> _MeanField:
    """``mean_field`` with the temporary checkpoint file PySCF holds open for
    it closed and no longer written, for a solve nothing restarts from."""
    if mean_field.chkfile is not None:
        mean_field._chkfile.close()
        mean_field.chkfile = None
    return mean_field


def pyscf_mole_at(mol: _PySCFMole, coordinates: np.ndarray) -> _PySCFMole:
    """A copy of ``mol`` at ``coordinates`` (Bohr), keeping ``mol``'s unit."""
    from pyscf.data.nist import BOHR
    from pyscf.gto.mole import is_au

    to_unit = 1.0 if is_au(mol.unit) else BOHR
    return mol.set_geom_(
        np.asarray(coordinates) * to_unit, unit=mol.unit, inplace=False
    )


def _pyscf_integrals(
    mol: _PySCFMole, mean_field: _PySCFMeanField | None
) -> tuple[np.ndarray, np.ndarray, float]:
    """RHF molecular-orbital integrals of a validated PySCF input.

    Args:
        mol: The ``gto.Mole``, as returned by :func:`split_pyscf_input`.
        mean_field: Its restricted mean-field, or ``None`` to run RHF on
            ``mol``; an unconverged mean-field is run first.

    Returns:
        ``(one_body, two_body, constant)``: the ``(n, n)`` one-electron
        integrals, the ``(n,) * 4`` two-electron integrals in chemist order
        ``(pq|rs)``, and the mean field's nuclear repulsion energy, which
        includes any external charges its core Hamiltonian sees.
    """
    from pyscf import ao2mo, scf

    if mean_field is None:
        mean_field = without_chkfile(scf.RHF(mol)).run(verbose=0)
    elif not mean_field.converged:
        # A copy, so the caller's object, settings and checkpoint file stay as given.
        mean_field = mean_field.copy()
        mean_field.chkfile = None
        mean_field.run(verbose=0)

    mo_coeff = np.asarray(mean_field.mo_coeff)
    n_orbitals = mo_coeff.shape[1]
    one_body = mo_coeff.T @ mean_field.get_hcore() @ mo_coeff
    two_body = ao2mo.restore(1, ao2mo.kernel(mol, mo_coeff), n_orbitals)
    return one_body, two_body, float(mean_field.energy_nuc())


def molecular_hamiltonian_from_pyscf(
    molecule: _PySCFInput,
) -> tuple[SparsePauliOp, int]:
    """Build a molecular electronic-structure Hamiltonian from a PySCF input.

    Extracts the RHF molecular-orbital integrals, expands them to spin-orbitals,
    and applies the Jordan-Wigner transform (via OpenFermion), returning a
    ``SparsePauliOp`` that retains its identity (nuclear-repulsion + core)
    constant term.

    Note the return contract differs from PennyLane's identically-purposed
    ``qml.qchem.molecular_hamiltonian``, which returns ``(hamiltonian, n_qubits)``.

    Args:
        molecule: A PySCF ``gto.Mole`` (an RHF calculation is run on it) or a
            restricted mean-field object (e.g. ``scf.RHF``); an unconverged
            mean-field is run first. Closed-shell (RHF) only.

    Returns:
        ``(hamiltonian, n_electrons)``.

    Raises:
        ImportError: If the ``chem`` extra is not installed.
        TypeError: If ``molecule`` is neither a ``Mole`` nor a mean-field.
        NotImplementedError: If the system is open-shell (non-zero spin) or the
            mean-field is not restricted (e.g. UHF, UKS or GHF).
    """
    mol, mean_field = split_pyscf_input(molecule)
    one_body, two_body, constant = _pyscf_integrals(mol, mean_field)
    return _spo_from_integrals(one_body, two_body, constant), int(mol.nelectron)


def _unsupported_molecule(molecule: object) -> TypeError:
    return TypeError(
        "Expected a PennyLane Molecule or PySCF Mole/mean-field object, "
        f"got {type(molecule).__name__}."
    )


def molecular_hamiltonian(molecule: MoleculeLike) -> tuple[SparsePauliOp, int]:
    """Build a Qiskit Hamiltonian from a PennyLane or PySCF molecule.

    Returns ``(hamiltonian, n_electrons)`` for both input families.
    """
    if is_pyscf_input(molecule):
        return molecular_hamiltonian_from_pyscf(molecule)

    if is_pennylane_molecule(molecule):
        from pennylane import qchem

        hamiltonian, _ = qchem.molecular_hamiltonian(molecule)
        return to_spo(hamiltonian), int(molecule.n_electrons)

    raise _unsupported_molecule(molecule)


def molecule_spin_counts(molecule: MoleculeLike) -> tuple[int, int]:
    """``(n_alpha, n_beta)`` of a molecule, validated without solving it.

    PySCF inputs may be open-shell; PennyLane molecules must be closed-shell.

    Raises:
        TypeError: If ``molecule`` is neither a PennyLane nor a PySCF input.
        NotImplementedError: If a PennyLane molecule is open-shell, or a PySCF
            mean field is not restricted.
    """
    if is_pyscf_input(molecule):
        mol, _ = split_pyscf_input(molecule, open_shell=True)
        n_alpha, n_beta = mol.nelec
        return int(n_alpha), int(n_beta)
    n_electrons = molecule_electron_count(molecule)
    return n_electrons // 2, n_electrons // 2


def molecule_electron_count(molecule: MoleculeLike) -> int:
    """Electron count of a closed-shell molecule, validated without solving it.

    Raises:
        TypeError: If ``molecule`` is neither a PennyLane nor a PySCF input.
        NotImplementedError: If the molecule is open-shell.
    """
    if is_pyscf_input(molecule):
        mol, _ = split_pyscf_input(molecule)
        return int(mol.nelectron)

    if is_pennylane_molecule(molecule):
        if molecule.mult != 1 or molecule.n_electrons % 2:
            raise NotImplementedError(
                "Only closed-shell systems are supported; got a PennyLane "
                f"molecule with mult={molecule.mult} and "
                f"{molecule.n_electrons} electrons."
            )
        return int(molecule.n_electrons)

    raise _unsupported_molecule(molecule)


def molecule_integrals(molecule: MoleculeLike) -> tuple[np.ndarray, np.ndarray, float]:
    """Hartree-Fock molecular-orbital integrals of a PennyLane or PySCF molecule.

    Returns:
        ``(one_body, two_body, constant)`` with ``two_body`` in chemist order
        ``(pq|rs)`` and ``constant`` the nuclear repulsion energy.
    """
    if is_pyscf_input(molecule):
        return _pyscf_integrals(*split_pyscf_input(molecule))

    if is_pennylane_molecule(molecule):
        from pennylane import qchem

        # A copy with plain coordinates; differentiable ones must be passed as args.
        frozen = qchem.Molecule(
            molecule.symbols,
            np.asarray(molecule.coordinates, dtype=float),
            charge=molecule.charge,
            mult=molecule.mult,
            basis_name=molecule.basis_name,
            load_data=molecule.load_data,
            l=molecule.l,
            alpha=molecule.alpha,
            coeff=molecule.coeff,
        )
        constant, one_body, two_body = qchem.electron_integrals(frozen)()
        # PennyLane's two[p, q, r, s] is (ps|qr).
        return (
            np.asarray(one_body, dtype=float),
            np.einsum("prsq->pqrs", np.asarray(two_body, dtype=float)),
            float(np.squeeze(constant)),
        )

    raise _unsupported_molecule(molecule)
