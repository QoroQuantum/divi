# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Harmonic structure of a variational program's trainable parameters."""

from dataclasses import dataclass
from numbers import Integral
from typing import cast

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class ParameterSpace:
    """Per-parameter frequencies of a program's cost.

    Entry ``i`` is ``(omega, order)`` when the cost holds frequencies
    ``{omega, ..., order * omega}`` in parameter ``i``, or ``None`` when they
    are unknown.

    Args:
        frequencies: One ``(omega, order)`` pair, or ``None``, per parameter.

    Raises:
        ValueError: If an ``omega`` is not positive or an ``order`` is not a
            positive integer.
    """

    frequencies: tuple[tuple[float, int] | None, ...]

    def __post_init__(self):
        for entry in self.frequencies:
            if entry is None:
                continue
            omega, order = entry
            if not omega > 0:
                raise ValueError(f"omega must be positive; got {omega}.")
            if isinstance(order, bool) or not isinstance(order, Integral) or order < 1:
                raise ValueError(f"order must be a positive integer; got {order}.")

    @property
    def periods(self) -> npt.NDArray[np.float64]:
        """Each parameter's period ``2 * pi / omega``; NaN where unknown."""
        return np.array(
            [
                np.nan if entry is None else 2 * np.pi / entry[0]
                for entry in self.frequencies
            ]
        )

    def compute_shift_rule(
        self,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """The exact parameter-shift rule, ``2 * order`` evaluations per parameter.

        Returns:
            ``(shifts, weights)``: a ``(n_evaluations, n_params)`` array of
            offsets to add to the parameter vector, and a
            ``(n_params, n_evaluations)`` array whose product with the evaluated
            values is the derivative.

        Raises:
            ValueError: If a parameter's frequencies are unknown.
        """
        missing = [
            index for index, entry in enumerate(self.frequencies) if entry is None
        ]
        if missing:
            raise ValueError(
                f"Parameters {missing} have no known frequencies, so no "
                "parameter-shift rule exists for them."
            )
        families = cast(tuple[tuple[float, int], ...], self.frequencies)
        n_params = len(families)
        n_evaluations = sum(2 * order for _, order in families)
        shifts_out = np.zeros((n_evaluations, n_params))
        weights_out = np.zeros((n_params, n_evaluations))
        offset = 0
        for index, (omega, order) in enumerate(families):
            term = np.arange(1, 2 * order + 1)
            shifts = (2 * term - 1) * np.pi / (2 * order * omega)
            # Fold the shifts into one period, centred on zero.
            period = 2.0 * np.pi / omega
            shifts = (shifts + 0.5 * period) % period - 0.5 * period
            coefficients = (
                (-1.0) ** (term - 1)
                * omega
                / (4 * order * np.sin((2 * term - 1) * np.pi / (4 * order)) ** 2)
            )
            stop = offset + len(shifts)
            shifts_out[offset:stop, index] = shifts
            weights_out[index, offset:stop] = coefficients
            offset = stop
        return shifts_out, weights_out
