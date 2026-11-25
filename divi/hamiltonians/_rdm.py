# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Spin-resolved reduced density matrices measured as qubit observables.

Each symmetry-unique RDM element becomes the Hermitian part ``(O + O†) / 2`` of its
fermionic operator, Jordan-Wigner mapped onto interleaved spin-orbital qubits
(qubit ``2p`` alpha, ``2p + 1`` beta). Its expectation value is the real part of
the element, so the RDMs are exact for real wave functions. The assembled layout
is PySCF's ``make_rdm12s``: ``dm1[p, q] = <q† p>`` and
``dm2[p, q, r, s] = <p† r† s q>``.
"""

from collections.abc import Sequence
from itertools import product

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from divi.hamiltonians._openfermion import import_openfermion, qubit_operator_to_spo

#: Spin of each one-body RDM block, and of each pair in a two-body block.
_ONE_BODY_SPINS = {"a": 0, "b": 1}
_TWO_BODY_SPINS = {"aa": (0, 0), "ab": (0, 1), "bb": (1, 1)}

#: Index permutations and signs mapping a two-body element onto its equal
#: entries: Hermitian conjugation always, antisymmetry within one spin.
_PAIR_IMAGES = (((0, 1, 2, 3), 1.0), ((1, 0, 3, 2), 1.0))
_SAME_SPIN_IMAGES = (
    ((2, 1, 0, 3), -1.0),
    ((0, 3, 2, 1), -1.0),
    ((2, 3, 0, 1), 1.0),
    ((3, 0, 1, 2), -1.0),
    ((1, 2, 3, 0), -1.0),
    ((3, 2, 1, 0), 1.0),
)

SpinRDMs = tuple[
    tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]
]


def _list_rdm_elements(n_orbitals: int) -> list[tuple[str, tuple[int, ...]]]:
    """Symmetry-unique ``(block, indices)`` RDM elements, in measurement order.

    One-body elements have ``p <= q``; two-body ones have ``(p, r) <= (q, s)``,
    and ``p < r`` and ``q < s`` within one spin.
    """
    orbitals = range(n_orbitals)
    elements: list[tuple[str, tuple[int, ...]]] = [
        (block, (p, q))
        for block in _ONE_BODY_SPINS
        for p, q in product(orbitals, repeat=2)
        if p <= q
    ]
    for block, (first, second) in _TWO_BODY_SPINS.items():
        elements += [
            (block, (p, q, r, s))
            for p, q, r, s in product(orbitals, repeat=4)
            if (p, r) <= (q, s) and (first != second or (p < r and q < s))
        ]
    return elements


def _build_ladder_term(
    block: str, indices: tuple[int, ...]
) -> tuple[tuple[int, int], ...]:
    """The OpenFermion ladder term of one element on interleaved spin-orbitals."""
    if block in _ONE_BODY_SPINS:
        spin = _ONE_BODY_SPINS[block]
        p, q = indices
        return ((2 * q + spin, 1), (2 * p + spin, 0))
    first, second = _TWO_BODY_SPINS[block]
    p, q, r, s = indices
    return (
        (2 * p + first, 1),
        (2 * r + second, 1),
        (2 * s + second, 0),
        (2 * q + first, 0),
    )


def build_rdm_observables(n_orbitals: int) -> tuple[SparsePauliOp, ...]:
    """Hermitian RDM-element observables on ``2 * n_orbitals`` qubits.

    Their expectation values, in order, are what :func:`assemble_spin_rdms` reads.
    """
    openfermion = import_openfermion(
        "openfermion", "Measuring reduced density matrices"
    )
    observables = []
    for block, indices in _list_rdm_elements(n_orbitals):
        operator = openfermion.FermionOperator(_build_ladder_term(block, indices))
        hermitian = openfermion.jordan_wigner(
            (operator + openfermion.hermitian_conjugated(operator)) / 2
        )
        hermitian.compress()
        observables.append(qubit_operator_to_spo(hermitian, 2 * n_orbitals))
    return tuple(observables)


def assemble_spin_rdms(expectations: Sequence[float], n_orbitals: int) -> SpinRDMs:
    """Spin RDMs from the expectation values of :func:`build_rdm_observables`.

    Returns ``((dm1a, dm1b), (dm2aa, dm2ab, dm2bb))`` in the module's layout.
    """
    one_body = {block: np.zeros((n_orbitals,) * 2) for block in _ONE_BODY_SPINS}
    two_body = {block: np.zeros((n_orbitals,) * 4) for block in _TWO_BODY_SPINS}
    for (block, indices), value in zip(
        _list_rdm_elements(n_orbitals), expectations, strict=True
    ):
        if block in _ONE_BODY_SPINS:
            p, q = indices
            one_body[block][p, q] = one_body[block][q, p] = value
            continue
        first, second = _TWO_BODY_SPINS[block]
        images = _PAIR_IMAGES + (_SAME_SPIN_IMAGES if first == second else ())
        for permutation, sign in images:
            two_body[block][tuple(indices[axis] for axis in permutation)] = sign * value
    return (one_body["a"], one_body["b"]), (
        two_body["aa"],
        two_body["ab"],
        two_body["bb"],
    )


def sum_spin_rdms(rdms: SpinRDMs) -> tuple[np.ndarray, np.ndarray]:
    """Spin-summed ``(dm1, dm2)`` from spin-resolved RDMs, such as those
    :meth:`~divi.qprog.algorithms.VQE.measure_rdms` returns, in PySCF's
    ``make_rdm12`` layout."""
    (dm1a, dm1b), (dm2aa, dm2ab, dm2bb) = rdms
    return dm1a + dm1b, dm2aa + dm2bb + dm2ab + dm2ab.transpose(2, 3, 0, 1)
