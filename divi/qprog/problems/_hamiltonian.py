# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Hamiltonian problems: qubit Hamiltonians and molecular integrals."""

from functools import cached_property
from pathlib import Path
from typing import Self

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from divi._optional import import_optional
from divi.hamiltonians._chem import _spo_from_integrals
from divi.hamiltonians._molecular import (
    MoleculeLike,
    _PySCFMeanField,
    compute_molecule_integrals,
    is_pennylane_molecule,
    molecular_hamiltonian,
    molecule_spin_counts,
)
from divi.hamiltonians._term_ops import ObservableLike, to_spo


def _electron_counts(
    n_electrons: int | None, n_alpha: int | None, n_beta: int | None
) -> tuple[int | None, int | None, int | None]:
    if (n_alpha is None) != (n_beta is None):
        raise ValueError(
            "n_alpha and n_beta must be given together; got "
            f"n_alpha={n_alpha}, n_beta={n_beta}."
        )
    if n_alpha is not None and n_beta is not None:
        if n_alpha < 0 or n_beta < 0:
            raise ValueError(
                f"n_alpha and n_beta must be non-negative; got {n_alpha}, {n_beta}."
            )
        if n_electrons is not None and n_electrons != n_alpha + n_beta:
            raise ValueError(
                f"n_electrons={n_electrons} disagrees with n_alpha + n_beta = "
                f"{n_alpha + n_beta}."
            )
        return n_alpha + n_beta, n_alpha, n_beta
    if n_electrons is not None and n_electrons < 0:
        raise ValueError(f"n_electrons must be non-negative; got {n_electrons}.")
    return n_electrons, None, None


def _read_only(
    integrals: tuple[np.ndarray, np.ndarray, float, np.ndarray | None],
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray | None]:
    for array in integrals:
        if isinstance(array, np.ndarray):
            array.setflags(write=False)
    return integrals


class HamiltonianProblem:
    """A qubit Hamiltonian and the electron counts of its reference state.

    Args:
        hamiltonian: The Hamiltonian — a PennyLane operator, a Qiskit
            ``SparsePauliOp``, a divi Pauli-string dict, or an OpenFermion
            ``QubitOperator`` (requires the ``chem`` extra).
        n_electrons: Electrons in the reference state, for ansätze that
            prepare one (e.g. :class:`~divi.qprog.algorithms.HartreeFockAnsatz`).
            Inferred from ``n_alpha + n_beta`` when those are given.
        n_alpha: Alpha electrons, for a spin-polarised reference. Must be given
            together with ``n_beta``; without both, ansätze that need a
            reference determinant assume the closed-shell split.
        n_beta: Beta electrons, under the same convention.

    Raises:
        ValueError: If only one of ``n_alpha`` and ``n_beta`` is given, if any
            count is negative, or if ``n_electrons`` disagrees with
            ``n_alpha + n_beta``.
    """

    def __init__(
        self,
        hamiltonian: ObservableLike,
        *,
        n_electrons: int | None = None,
        n_alpha: int | None = None,
        n_beta: int | None = None,
    ):
        self._hamiltonian = to_spo(hamiltonian)
        self._n_electrons, self._n_alpha, self._n_beta = _electron_counts(
            n_electrons, n_alpha, n_beta
        )

    @property
    def hamiltonian(self) -> SparsePauliOp:
        """The qubit Hamiltonian, including its identity (constant) term."""
        return self._hamiltonian

    @property
    def n_electrons(self) -> int | None:
        """Electrons in the reference state, or ``None`` if unspecified."""
        return self._n_electrons

    @property
    def n_alpha(self) -> int | None:
        """Alpha electrons, or ``None`` for the closed-shell split."""
        return self._n_alpha

    @property
    def n_beta(self) -> int | None:
        """Beta electrons, or ``None`` for the closed-shell split."""
        return self._n_beta


class MolecularProblem(HamiltonianProblem):
    """An electronic-structure problem in spatial-orbital integral form.

    Holds the one- and two-electron integrals and the electron counts. The
    qubit :attr:`hamiltonian` is their Jordan-Wigner transform on
    ``2 * n_orbitals`` interleaved spin-orbital qubits (qubit ``2p`` is the
    alpha spin-orbital of spatial orbital ``p``, qubit ``2p + 1`` its beta
    partner), built on first access. For a PennyLane molecule, PennyLane's own
    Jordan-Wigner builder produces it, in the same qubit order.

    Construct it from integrals directly, from a molecule with
    :meth:`from_molecule`, from a PySCF mean field's active space with
    :meth:`from_casci`, or from an FCIDUMP file with :meth:`from_fcidump`. The electron counts may differ, as for a
    spin-polarised fragment; :meth:`from_molecule` builds integrals for
    closed-shell molecules only, and keeps an open-shell PySCF molecule for
    :class:`~divi.qprog.ensembles.lassqd.LASSQD`.

    Args:
        one_body: ``(n_orbitals, n_orbitals)`` one-electron integrals. The
            alpha channel when ``one_body_beta`` is given, both channels
            otherwise.
        two_body: ``(n_orbitals,) * 4`` two-electron integrals in chemist
            order ``(pq|rs)``.
        n_alpha: Alpha electrons.
        n_beta: Beta electrons.
        constant: Scalar energy offset, e.g. nuclear repulsion plus any
            frozen-core energy.
        one_body_beta: Beta-channel one-electron integrals, for a
            spin-dependent one-body potential.

    Raises:
        ValueError: If the integral shapes are inconsistent, or if ``n_alpha``
            or ``n_beta`` falls outside ``[0, n_orbitals]``.
    """

    def __init__(
        self,
        one_body: np.ndarray,
        two_body: np.ndarray,
        *,
        n_alpha: int,
        n_beta: int,
        constant: float = 0.0,
        one_body_beta: np.ndarray | None = None,
    ):
        one_body = np.array(one_body, dtype=float)
        two_body = np.array(two_body, dtype=float)
        n_orbitals = one_body.shape[0]
        if one_body.shape != (n_orbitals, n_orbitals):
            raise ValueError(f"one_body must be square; got {one_body.shape}.")
        if two_body.shape != (n_orbitals,) * 4:
            raise ValueError(
                f"two_body must have shape {(n_orbitals,) * 4}; got {two_body.shape}."
            )
        if one_body_beta is not None:
            one_body_beta = np.array(one_body_beta, dtype=float)
            if one_body_beta.shape != one_body.shape:
                raise ValueError(
                    f"one_body_beta must have shape {one_body.shape}; "
                    f"got {one_body_beta.shape}."
                )
        for name, count in (("n_alpha", n_alpha), ("n_beta", n_beta)):
            if not 0 <= count <= n_orbitals:
                raise ValueError(
                    f"{name} must be between 0 and {n_orbitals}; got {count}."
                )
        self._assign(
            (one_body, two_body, float(constant), one_body_beta),
            n_alpha,
            n_beta,
            molecule=None,
        )

    @classmethod
    def from_molecule(cls, molecule: MoleculeLike) -> Self:
        """Build the problem from a molecule.

        The integrals and the Hamiltonian are computed on first access, so a
        consumer that reads neither (e.g. :class:`~divi.qprog.ensembles.lassqd.LASSQD`,
        which needs a PySCF input) never pays for them.

        Args:
            molecule: A PennyLane ``qp.qchem.Molecule``, or a PySCF ``gto.Mole``
                or restricted mean-field object (requires the ``chem`` extra).
                Integrals come from a Hartree-Fock solve: PennyLane's own
                differentiable one, or PySCF's RHF. An open-shell PySCF input
                (non-zero ``spin``, ROHF/ROKS mean field) is accepted for
                :class:`~divi.qprog.ensembles.lassqd.LASSQD`, which builds its own
                integrals; reading this problem's integrals or Hamiltonian then
                raises ``NotImplementedError``.

        Raises:
            TypeError: If ``molecule`` is neither a PennyLane nor a PySCF input.
            NotImplementedError: If a PennyLane molecule is open-shell, or a
                PySCF mean-field is not restricted.
        """
        n_alpha, n_beta = molecule_spin_counts(molecule)
        problem = cls.__new__(cls)
        problem._assign(None, n_alpha, n_beta, molecule=molecule)
        return problem

    @classmethod
    def from_casci(
        cls,
        mean_field: _PySCFMeanField,
        n_orbitals: int,
        n_electrons: int | tuple[int, int],
    ) -> Self:
        """Build the active space of a converged PySCF mean field.

        The active space is PySCF's ``mcscf.CASCI`` one: the ``n_orbitals``
        orbitals around the Fermi level, with the doubly occupied core folded
        into :attr:`one_body` and :attr:`constant`. Requires the ``chem``
        extra.

        Args:
            mean_field: A converged PySCF mean field.
            n_orbitals: Active spatial orbitals.
            n_electrons: Active electrons, as a total split by the mean
                field's spin, or as ``(n_alpha, n_beta)``.
        """
        mcscf = import_optional(
            "pyscf.mcscf", extra="chem", capability="MolecularProblem.from_casci"
        )
        ao2mo = import_optional(
            "pyscf.ao2mo", extra="chem", capability="MolecularProblem.from_casci"
        )
        casci = mcscf.CASCI(mean_field, n_orbitals, n_electrons)
        one_body, constant = casci.get_h1eff()
        n_alpha, n_beta = casci.nelecas
        return cls(
            one_body,
            ao2mo.restore(1, casci.get_h2eff(), n_orbitals),
            n_alpha=int(n_alpha),
            n_beta=int(n_beta),
            constant=float(constant),
        )

    @classmethod
    def from_fcidump(cls, path: str | Path) -> Self:
        """Read a problem from an FCIDUMP file. Requires the ``chem`` extra.

        ``MS2`` splits the electrons between the spins, and ``ECORE`` becomes
        :attr:`constant` (0 when the file has no core-energy line).

        Raises:
            ValueError: If the header has no ``MS2``, or its parity disagrees
                with ``NELEC``.
        """
        fcidump = import_optional(
            "pyscf.tools.fcidump",
            extra="chem",
            capability="MolecularProblem.from_fcidump",
        )
        ao2mo = import_optional(
            "pyscf.ao2mo", extra="chem", capability="MolecularProblem.from_fcidump"
        )
        data = fcidump.read(str(path), molpro_orbsym=False, verbose=False)
        n_orbitals, n_electrons = data["NORB"], data["NELEC"]
        if "MS2" not in data:
            raise ValueError(
                f"{path} has no MS2 in its header, so its spin is unknown."
            )
        spin = data["MS2"]
        if (n_electrons - spin) % 2:
            raise ValueError(
                f"{path} has NELEC={n_electrons} and MS2={spin}, whose parities "
                "disagree."
            )
        return cls(
            data["H1"],
            ao2mo.restore(1, data["H2"], n_orbitals),
            n_alpha=(n_electrons + spin) // 2,
            n_beta=(n_electrons - spin) // 2,
            constant=float(data.get("ECORE", 0.0)),
        )

    def to_fcidump(self, path: str | Path) -> None:
        """Write the problem as an FCIDUMP file. Requires the ``chem`` extra.

        Raises:
            ValueError: If the alpha and beta one-electron integrals differ,
                which FCIDUMP cannot hold, or if beta electrons outnumber alpha
                ones, which would write a negative ``MS2``.
        """
        fcidump = import_optional(
            "pyscf.tools.fcidump",
            extra="chem",
            capability="MolecularProblem.to_fcidump",
        )
        ao2mo = import_optional(
            "pyscf.ao2mo", extra="chem", capability="MolecularProblem.to_fcidump"
        )
        if not np.array_equal(self.one_body_beta, self.one_body):
            raise ValueError(
                "FCIDUMP holds one set of one-electron integrals, but this "
                "problem's alpha and beta channels differ."
            )
        if self.n_beta > self.n_alpha:
            raise ValueError(
                f"FCIDUMP's MS2 must be non-negative, but this problem has "
                f"n_alpha={self.n_alpha} < n_beta={self.n_beta}. Swap the counts; "
                "with spin-independent integrals the energies are the same."
            )
        fcidump.from_integrals(
            str(path),
            self.one_body,
            ao2mo.restore(8, self.two_body, self.n_orbitals),
            self.n_orbitals,
            self.n_electrons,
            nuc=self.constant,
            ms=self.n_alpha - self.n_beta,
        )

    def rotate_orbitals(self, orbitals: np.ndarray) -> Self:
        """The same problem in another orbital basis.

        Args:
            orbitals: ``(n_orbitals, n_orbitals)`` matrix whose columns are the
                new orbitals in the current basis.
        """
        one_body, two_body, constant, one_body_beta = self._loaded_integrals()

        def rotate_one_body(matrix: np.ndarray) -> np.ndarray:
            return orbitals.T @ matrix @ orbitals

        return type(self)(
            rotate_one_body(one_body),
            np.einsum(
                "pi,qj,pqrs,rk,sl->ijkl",
                orbitals,
                orbitals,
                two_body,
                orbitals,
                orbitals,
                optimize=True,
            ),
            n_alpha=self.n_alpha,
            n_beta=self.n_beta,
            constant=constant,
            one_body_beta=(
                None if one_body_beta is None else rotate_one_body(one_body_beta)
            ),
        )

    def _assign(
        self,
        integrals: tuple[np.ndarray, np.ndarray, float, np.ndarray | None] | None,
        n_alpha: int,
        n_beta: int,
        *,
        molecule: MoleculeLike | None,
    ) -> None:
        self._integrals = None if integrals is None else _read_only(integrals)
        _electron_counts(None, n_alpha, n_beta)
        self._alpha, self._beta = n_alpha, n_beta
        self._molecule = molecule

    def _loaded_integrals(
        self,
    ) -> tuple[np.ndarray, np.ndarray, float, np.ndarray | None]:
        if self._integrals is None:
            if self._molecule is None:
                raise RuntimeError(
                    "MolecularProblem holds neither integrals nor a molecule."
                )
            self._integrals = _read_only(
                (*compute_molecule_integrals(self._molecule), None)
            )
        return self._integrals

    @property
    def molecule(self) -> MoleculeLike | None:
        """The molecule given to :meth:`from_molecule`, or ``None``."""
        return self._molecule

    @cached_property
    def hamiltonian(self) -> SparsePauliOp:
        """The Jordan-Wigner qubit Hamiltonian, including its constant term."""
        if is_pennylane_molecule(self._molecule):
            # PennyLane's own mapping, so it needs no OpenFermion.
            return molecular_hamiltonian(self._molecule)[0]
        one_body, two_body, constant, one_body_beta = self._loaded_integrals()
        return _spo_from_integrals(
            one_body, two_body, constant, one_body_beta=one_body_beta
        )

    @property
    def n_electrons(self) -> int:
        """Total electrons, ``n_alpha + n_beta``."""
        return self._alpha + self._beta

    @property
    def n_alpha(self) -> int:
        """Alpha electrons."""
        return self._alpha

    @property
    def n_beta(self) -> int:
        """Beta electrons."""
        return self._beta

    @property
    def one_body(self) -> np.ndarray:
        """``(n_orbitals, n_orbitals)`` one-electron integrals (alpha channel)."""
        return self._loaded_integrals()[0]

    @property
    def one_body_beta(self) -> np.ndarray:
        """``(n_orbitals, n_orbitals)`` beta-channel one-electron integrals."""
        one_body, _, _, one_body_beta = self._loaded_integrals()
        return one_body if one_body_beta is None else one_body_beta

    @property
    def two_body(self) -> np.ndarray:
        """``(n_orbitals,) * 4`` two-electron integrals, chemist order ``(pq|rs)``."""
        return self._loaded_integrals()[1]

    @property
    def constant(self) -> float:
        """Scalar energy offset carried by the integrals."""
        return self._loaded_integrals()[2]

    @property
    def n_orbitals(self) -> int:
        """Number of spatial orbitals."""
        return self.one_body.shape[0]
