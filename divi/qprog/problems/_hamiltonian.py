# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Hamiltonian problems: qubit Hamiltonians and molecular integrals."""

from typing import Self

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from divi.hamiltonians._chem import _spo_from_integrals
from divi.hamiltonians._molecular import (
    MoleculeLike,
    is_pennylane_molecule,
    molecular_hamiltonian,
    molecule_electron_count,
    molecule_integrals,
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

    Construct it from integrals directly, or from a molecule with
    :meth:`from_molecule`. The electron counts may differ, as for a
    spin-polarised fragment; :meth:`from_molecule` itself accepts closed-shell
    molecules only.

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
        """Build the problem from a closed-shell molecule.

        The integrals and the Hamiltonian are computed on first access, so a
        consumer that reads neither (e.g. :class:`~divi.qprog.workflows.LASSQD`,
        which needs a PySCF input) never pays for them.

        Args:
            molecule: A PennyLane ``qp.qchem.Molecule``, or a PySCF ``gto.Mole``
                or restricted mean-field object (requires the ``chem`` extra).
                Integrals come from a Hartree-Fock solve: PennyLane's own
                differentiable one, or PySCF's RHF.

        Raises:
            TypeError: If ``molecule`` is neither a PennyLane nor a PySCF input.
            NotImplementedError: If the molecule is open-shell, or a PySCF
                mean-field is not restricted.
        """
        n_electrons = molecule_electron_count(molecule)
        problem = cls.__new__(cls)
        problem._assign(None, n_electrons // 2, n_electrons // 2, molecule=molecule)
        return problem

    def _assign(
        self,
        integrals: tuple[np.ndarray, np.ndarray, float, np.ndarray | None] | None,
        n_alpha: int,
        n_beta: int,
        *,
        molecule: MoleculeLike | None,
    ) -> None:
        self._integrals = None if integrals is None else _read_only(integrals)
        self._n_electrons, self._n_alpha, self._n_beta = _electron_counts(
            None, n_alpha, n_beta
        )
        self._molecule = molecule
        self._qubit_hamiltonian: SparsePauliOp | None = None

    def _loaded_integrals(
        self,
    ) -> tuple[np.ndarray, np.ndarray, float, np.ndarray | None]:
        if self._integrals is None:
            assert self._molecule is not None
            self._integrals = _read_only((*molecule_integrals(self._molecule), None))
        return self._integrals

    @property
    def molecule(self) -> MoleculeLike | None:
        """The molecule given to :meth:`from_molecule`, or ``None``."""
        return self._molecule

    @property
    def hamiltonian(self) -> SparsePauliOp:
        """The Jordan-Wigner qubit Hamiltonian, including its constant term."""
        if self._qubit_hamiltonian is None:
            if is_pennylane_molecule(self._molecule):
                # PennyLane's own mapping, so it needs no OpenFermion.
                self._qubit_hamiltonian, _ = molecular_hamiltonian(self._molecule)
            else:
                one_body, two_body, constant, one_body_beta = self._loaded_integrals()
                self._qubit_hamiltonian = _spo_from_integrals(
                    one_body, two_body, constant, one_body_beta=one_body_beta
                )
        return self._qubit_hamiltonian

    @property
    def n_electrons(self) -> int:
        """Total electrons, ``n_alpha + n_beta``."""
        assert self._n_electrons is not None
        return self._n_electrons

    @property
    def n_alpha(self) -> int:
        """Alpha electrons."""
        assert self._n_alpha is not None
        return self._n_alpha

    @property
    def n_beta(self) -> int:
        """Beta electrons."""
        assert self._n_beta is not None
        return self._n_beta

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
