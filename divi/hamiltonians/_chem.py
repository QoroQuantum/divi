# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Adapters from external chemistry stacks to ``SparsePauliOp``.

Both live behind the ``chem`` extra (``pyscf`` + ``openfermion``) and are
lazily imported, so importing this module never requires them.
"""

from typing import Any

import numpy as np
from qiskit.quantum_info import SparsePauliOp


def qubit_operator_to_spo(
    qubit_operator: Any, n_qubits: int | None = None
) -> SparsePauliOp:
    """Convert an OpenFermion ``QubitOperator`` to a Qiskit ``SparsePauliOp``.

    OpenFermion terms are ``{((qubit, "X"), (qubit, "Z"), ...): coeff}`` with an
    implicit identity on unlisted qubits and qubit ``q`` at index ``q``. Qiskit
    labels are MSB-first, so the per-qubit characters are reversed when joined;
    the qubit numbering is preserved (OpenFermion qubit ``q`` maps to circuit
    qubit ``q``).

    Args:
        qubit_operator: An OpenFermion ``QubitOperator``.
        n_qubits: Register width. Defaults to one past the highest qubit index
            appearing in the operator, which drops any idle trailing qubits;
            pass an explicit width to embed the operator in a wider register.

    Raises:
        ValueError: If ``n_qubits`` is smaller than the operator's support, or
            the operator is empty.
    """
    terms = qubit_operator.terms
    if not terms:
        raise ValueError("QubitOperator has no terms.")

    max_index = max((idx for term in terms for idx, _ in term), default=-1)
    support = max_index + 1
    if n_qubits is None:
        n_qubits = max(support, 1)
    elif n_qubits < support:
        raise ValueError(
            f"n_qubits ({n_qubits}) is smaller than the operator's support ({support})."
        )

    labels: list[str] = []
    coeffs: list[complex] = []
    for term, coeff in terms.items():
        chars = ["I"] * n_qubits
        for idx, pauli in term:
            chars[idx] = pauli
        labels.append("".join(chars[::-1]))
        coeffs.append(complex(coeff))
    return SparsePauliOp(labels, np.array(coeffs, dtype=complex)).simplify()


def _spo_from_integrals(
    one_body: np.ndarray,
    two_body: np.ndarray,
    constant: float,
    one_body_beta: np.ndarray | None = None,
) -> SparsePauliOp:
    """Jordan-Wigner a set of spatial-MO integrals into a ``SparsePauliOp``.

    Args:
        one_body: ``(n_orb, n_orb)`` one-electron integrals in the MO basis.
            Applies to both spin channels unless ``one_body_beta`` is given, in
            which case this is the alpha channel.
        two_body: ``(n_orb,) * 4`` two-electron integrals in PySCF chemist
            order ``(pq|rs)``.
        constant: Scalar offset retained as the operator's identity term
            (nuclear repulsion plus any frozen-core energy).
        one_body_beta: Beta-channel one-electron integrals, for a
            spin-dependent one-body potential such as the exchange term of a
            spin-polarised mean-field embedding. The two-body integrals are
            spin-free regardless, being spatial-orbital integrals.

    Returns:
        A ``SparsePauliOp`` on ``2 * n_orb`` qubits. Spin-orbitals are
        interleaved: qubit ``2p`` is the alpha spin-orbital of spatial orbital
        ``p``, qubit ``2p + 1`` is its beta partner.

    Raises:
        ImportError: If the ``chem`` extra is not installed.
        ValueError: If the integral shapes are inconsistent.
    """
    try:
        # pyrefly: ignore[missing-import]  # optional ``chem`` extra
        from openfermion import InteractionOperator, jordan_wigner

        # pyrefly: ignore[missing-import]  # optional ``chem`` extra
        from openfermion.chem.molecular_data import spinorb_from_spatial

        # pyrefly: ignore[missing-import]  # optional ``chem`` extra
        from openfermion.config import EQ_TOLERANCE
    except ImportError as exc:
        raise ImportError(
            "_spo_from_integrals requires the 'chem' extra; "
            "install it with `pip install qoro-divi[chem]`."
        ) from exc

    one_body = np.asarray(one_body, dtype=float)
    two_body = np.asarray(two_body, dtype=float)

    n_orb = one_body.shape[0]
    if one_body.shape != (n_orb, n_orb):
        raise ValueError(f"one_body must be square; got {one_body.shape}.")
    if two_body.shape != (n_orb,) * 4:
        raise ValueError(
            f"two_body must have shape {(n_orb,) * 4}; got {two_body.shape}."
        )

    # PySCF eri is chemist-order (pq|rs); OpenFermion wants (0, 2, 3, 1).
    one_body_coeffs, two_body_coeffs = spinorb_from_spatial(
        one_body, two_body.transpose(0, 2, 3, 1)
    )
    if one_body_beta is not None:
        one_body_beta = np.asarray(one_body_beta, dtype=float)
        if one_body_beta.shape != (n_orb, n_orb):
            raise ValueError(
                f"one_body_beta must have shape {(n_orb, n_orb)}; "
                f"got {one_body_beta.shape}."
            )
        # ``spinorb_from_spatial`` interleaves alpha on even and beta on odd
        # indices, writing ``one_body`` into both; replace the beta block. It
        # also zeroes coefficients below its own tolerance, so apply the same
        # mask here or the two channels differ at that threshold even when the
        # inputs are identical.
        one_body_coeffs[1::2, 1::2] = np.where(
            np.abs(one_body_beta) < EQ_TOLERANCE, 0.0, one_body_beta
        )

    interaction = InteractionOperator(
        float(constant), one_body_coeffs, 0.5 * two_body_coeffs
    )
    return qubit_operator_to_spo(jordan_wigner(interaction), 2 * n_orb)


def molecular_hamiltonian_from_pyscf(molecule: Any) -> tuple[SparsePauliOp, int]:
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
            mean-field is not restricted (2D ``mo_coeff``).
    """
    try:
        # pyrefly: ignore[missing-import]  # optional ``chem`` extra
        from pyscf import ao2mo, gto, scf
    except ImportError as exc:
        raise ImportError(
            "molecular_hamiltonian_from_pyscf requires the 'chem' extra; "
            "install it with `pip install qoro-divi[chem]`."
        ) from exc

    if isinstance(molecule, gto.Mole):
        mean_field = None
        mol = molecule
    elif isinstance(molecule, scf.hf.SCF):
        mean_field = molecule
        mol = mean_field.mol
    else:
        raise TypeError(
            "molecular_hamiltonian_from_pyscf expects a pyscf Mole or mean-field "
            f"object, got {type(molecule).__name__}."
        )

    if mol.spin != 0:
        raise NotImplementedError(
            "Only closed-shell (RHF) systems are supported; got an open-shell "
            f"molecule with spin={mol.spin}."
        )

    if mean_field is None:
        mean_field = scf.RHF(mol).run(verbose=0)
    elif getattr(mean_field, "mo_coeff", None) is None:
        mean_field.run(verbose=0)

    mo_coeff = np.asarray(mean_field.mo_coeff)
    if mo_coeff.ndim != 2:
        raise NotImplementedError(
            "Only restricted (closed-shell) mean-fields are supported; "
            f"got mo_coeff with {mo_coeff.ndim} dimensions."
        )

    n_orbitals = mo_coeff.shape[1]
    one_body = mo_coeff.T @ mean_field.get_hcore() @ mo_coeff
    eri = ao2mo.restore(1, ao2mo.kernel(mol, mo_coeff), n_orbitals)

    spo = _spo_from_integrals(one_body, eri, float(mol.energy_nuc()))
    return spo, int(mol.nelectron)
