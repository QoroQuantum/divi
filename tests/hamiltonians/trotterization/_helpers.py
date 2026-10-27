# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Shared inputs and warning patterns for trotterization tests."""

import pytest
from qiskit.quantum_info import SparsePauliOp

from tests._helpers import exact_match

KEEP_TOP_N_NO_OP = exact_match(
    "keep_top_n is greater than or equal to the number of terms; "
    "returning the full Hamiltonian."
)
SAMPLING_BUDGET_NOT_SET = exact_match(
    "sampling_budget is not set; only the kept terms will be applied, "
    "equivalent to ExactTrotterization."
)

# Qiskit big-endian: rightmost char is qubit 0.
SIMPLE_HAMILTONIAN = SparsePauliOp.from_list([("IZ", 1.0), ("ZI", 2.0), ("ZZ", 3.0)])

single_term_hamiltonians = pytest.mark.parametrize(
    "single_term_spo",
    [
        0.5 * SparsePauliOp("Z"),
        SparsePauliOp("Z"),
    ],
    ids=["scaled", "unit"],
)
