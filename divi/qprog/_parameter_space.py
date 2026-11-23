# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Harmonic structure of a variational program's trainable parameters."""

from dataclasses import dataclass
from numbers import Integral
from typing import cast

import numpy as np
import numpy.typing as npt

#: Longest period a search spans; longer and unknown periods fall back to
#: ``2 pi``.
_MAX_SEARCH_PERIOD = 4 * np.pi
_FALLBACK_WIDTH = 2 * np.pi
#: Standard deviation of a warm-start population, as a fraction of each
#: parameter's search width over ``2 pi``.
_WARM_START_SPREAD = 0.05


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

    @property
    def search_widths(self) -> npt.NDArray[np.float64]:
        """The width a search spans in each parameter, which is its period when
        known and at most ``4 * pi``, and ``2 * pi`` otherwise."""
        periods = self.periods
        return np.where(periods <= _MAX_SEARCH_PERIOD, periods, _FALLBACK_WIDTH)

    def wrap(
        self, points: npt.ArrayLike, center: npt.ArrayLike
    ) -> npt.NDArray[np.float64]:
        """``points`` moved by whole periods to within half a period of
        ``center``; parameters with unknown periods are left unchanged."""
        points = np.asarray(points, dtype=np.float64)
        center = np.asarray(center, dtype=np.float64)
        periods = self.periods
        known = np.isfinite(periods)
        period = np.where(known, periods, 1.0)
        wrapped = center + (points - center + period / 2) % period - period / 2
        return np.where(known, wrapped, points)

    def build_population(
        self, center: npt.ArrayLike, n_sets: int, rng: np.random.Generator
    ) -> npt.NDArray[np.float64]:
        """``n_sets`` starting points: ``center``, then Gaussian perturbations of
        it scaled to each parameter's search width."""
        center = np.asarray(center, dtype=np.float64)
        scale = _WARM_START_SPREAD * self.search_widths / (2 * np.pi)
        perturbed = center + rng.normal(0.0, scale, (n_sets - 1, center.size))
        return np.vstack([center, perturbed])

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
