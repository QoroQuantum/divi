# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

# TODO: delete whole module once Cirq properly supports parameters in openqasm 3.0
from . import _qasm_export  # Does nothing, just initiates the patch
from ._qasm_import import cirq_circuit_from_qasm
from ._validator import is_valid_qasm
