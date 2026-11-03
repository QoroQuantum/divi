# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Quantum programs and the machinery around them.

This package exports only the core contract. Everything else is imported from
the module that owns it:

* :mod:`divi.qprog.algorithms`: VQE, QAOA, PCE, QNN, time evolution, ansatzes.
* :mod:`divi.qprog.optimizers`: classical optimisers;
  :mod:`divi.qprog.optimizers.metrics`: their metric estimators.
* :mod:`divi.qprog.problems`: problem definitions and partitioning configs.
* :mod:`divi.qprog.ensembles`: :class:`ProgramEnsemble`, its reporting types, the
  concrete ensembles and the strategies that aggregate results;
  :mod:`divi.qprog.ensembles.lassqd`: LASSQD, which needs the ``chem`` extra.
* :mod:`divi.qprog.mixins`: capability mixins and :class:`SolutionEntry`.
* :mod:`divi.qprog.initial_states`, :mod:`divi.qprog.checkpointing`.
"""

from ._early_stopping import EarlyStopping, StopReason
from .quantum_program import QuantumProgram
from .variational_quantum_algorithm import (
    ParamHistoryMode,
    VariationalQuantumAlgorithm,
)

__all__ = [
    "EarlyStopping",
    "ParamHistoryMode",
    "QuantumProgram",
    "StopReason",
    "VariationalQuantumAlgorithm",
]
