# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Low-level manipulation primitives for PennyLane Hamiltonian operators."""

from functools import reduce
from typing import Literal

import pennylane as qml


def _is_multi_term_sum(op: qml.operation.Operator) -> bool:
    """True if op is a multi-term Sum or Hamiltonian (has operands and len)."""
    return isinstance(op, (qml.Hamiltonian, qml.ops.Sum))


def _get_terms_iterable(op: qml.operation.Operator) -> list:
    """Return terms as a list for iteration. Works for Sum/Hamiltonian and single-term."""
    return op.operands if _is_multi_term_sum(op) else [op]


def _is_empty_hamiltonian(op: qml.operation.Operator) -> bool:
    """True if op is an empty Sum/Hamiltonian (only constant terms)."""
    return _is_multi_term_sum(op) and len(op) == 0


def _clean_hamiltonian(
    hamiltonian: qml.operation.Operator,
) -> tuple[qml.operation.Operator, float]:
    """Separate constant and non-constant terms in a Hamiltonian.

    This function processes a PennyLane Hamiltonian to separate out any terms
    that are constant (i.e. proportional to the identity operator). The sum
    of these constant terms is returned, along with a new Hamiltonian containing
    only the non-constant terms.

    Args:
        hamiltonian: The Hamiltonian operator to process.

    Returns:
        tuple[qml.operation.Operator, float]: A tuple containing:
            - The Hamiltonian without the constant (identity) component.
            - The summed value of all constant terms.
    """

    terms = _get_terms_iterable(hamiltonian)

    constant = 0.0
    non_id_terms = []

    for term in terms:
        coeff = 1.0
        base_op = term
        if isinstance(term, qml.ops.SProd):
            coeff = term.scalar
            base_op = term.base

        # Check for Identity term
        is_constant = False
        if isinstance(base_op, qml.Identity):
            is_constant = True
        elif isinstance(base_op, qml.ops.Prod) and all(
            isinstance(op, qml.Identity) for op in base_op.operands
        ):
            is_constant = True

        if is_constant:
            constant += coeff
        else:
            non_id_terms.append(term)

    if not non_id_terms:
        return qml.Hamiltonian([], []), float(constant)

    # Reconstruct the Hamiltonian from non-constant terms
    if len(non_id_terms) > 1:
        new_hamiltonian = qml.sum(*non_id_terms)
    else:
        new_hamiltonian = non_id_terms[0]

    return new_hamiltonian.simplify(), float(constant)


def _hamiltonian_term_count(hamiltonian: qml.operation.Operator) -> int:
    """Return the number of terms in a Hamiltonian.

    Works for qml.Hamiltonian, qml.ops.Sum (multi-term), and single-term operators
    such as SProd or bare Pauli operators, which do not implement __len__.
    """
    return len(hamiltonian) if _is_multi_term_sum(hamiltonian) else 1


def _sort_hamiltonian_terms(
    hamiltonian: qml.operation.Operator,
    order: Literal["absolute", "magnitude"] = "absolute",
) -> qml.operation.Operator:
    """Sort the terms of a Hamiltonian by their coefficient magnitude."""
    if not _is_multi_term_sum(hamiltonian):
        return hamiltonian
    coeffs, terms = hamiltonian.terms()
    sorted_coeffs, sorted_terms = zip(
        *sorted(
            zip(coeffs, terms), key=lambda x: x[0] if order == "absolute" else abs(x[0])
        )
    )
    weighted_terms = [cf * trm for cf, trm in zip(sorted_coeffs, sorted_terms)]
    # Avoid Sum construction for single term; preserves original operator type.
    if len(weighted_terms) == 1:
        return weighted_terms[0]
    return qml.sum(*weighted_terms).simplify()


def _z_product(indices: tuple[int, ...]) -> qml.operation.Operator:
    """Build a Z-product operator for the given wire indices."""
    if len(indices) == 1:
        return qml.Z(indices[0])
    return reduce(lambda left, right: left @ right, (qml.Z(i) for i in indices))
