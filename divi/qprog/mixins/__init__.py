# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Capability mixins composed into :class:`~divi.qprog.QuantumProgram` subclasses.

Each mixin adds one orthogonal capability — observable-measurement configuration,
discrete-solution sampling, or a classical data axis — and is listed before the
host program in the base list so its cooperative ``super()`` calls resolve.
"""

from ._data_binding import DataBindingMixin
from ._observable_measuring import ObservableMeasuringMixin
from ._solution_sampling import SolutionEntry, SolutionSamplingMixin

__all__ = [
    "DataBindingMixin",
    "ObservableMeasuringMixin",
    "SolutionEntry",
    "SolutionSamplingMixin",
]
