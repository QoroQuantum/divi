# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Sequential minimisation along one parameter at a time."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Self, cast

import numpy as np
import numpy.typing as npt
from scipy.optimize import OptimizeResult

from divi.pipeline import ContractViolation
from divi.qprog._parameter_space import ParameterSpace
from divi.qprog.checkpointing.names import OPTIMIZER_STATE_FILE
from divi.qprog.checkpointing.write import atomic_write

from ._base import Optimizer

#: Relative improvement below which a step is rounding noise on a flat direction.
_FLAT_TOLERANCE = 1e-12


def _find_trigonometric_minimum(samples: npt.NDArray[np.float64]) -> float:
    """Find the global minimiser in ``(-pi, pi]`` of the degree-``R``
    trigonometric polynomial through ``2R + 1`` equidistant samples on ``[0, 2pi)``.

    Candidates are ``u = 0`` and the stationary points, the roots of the
    derivative's companion polynomial in ``exp(iu)``. A candidate that does not
    lower the polynomial below its value at ``u = 0`` returns ``0``.
    """
    order = (len(samples) - 1) // 2
    coefficients = np.fft.rfft(samples) / len(samples)
    harmonics = np.arange(1, order + 1)
    derivative = np.zeros(2 * order + 1, dtype=np.complex128)
    derivative[order + harmonics] = 1j * harmonics * coefficients[1:]
    derivative[order - harmonics] = -1j * harmonics * np.conj(coefficients[1:])
    candidates = np.concatenate(([0.0], np.angle(np.roots(derivative[::-1]))))
    values = coefficients[0].real + 2 * np.real(
        np.exp(1j * np.outer(candidates, harmonics)) @ coefficients[1:]
    )
    best = int(np.argmin(values))
    if values[best] >= values[0] - _FLAT_TOLERANCE * max(1.0, abs(values[0])):
        return 0.0
    return float(candidates[best])


def _require_known_frequencies(
    space: ParameterSpace,
) -> tuple[tuple[float, int], ...]:
    """Return ``space.frequencies``, raising ``ContractViolation`` for any unknown."""
    unknown = [index for index, entry in enumerate(space.frequencies) if entry is None]
    if unknown:
        raise ContractViolation(
            f"RotosolveOptimizer needs the frequencies of every parameter, but "
            f"those of parameters {unknown} are unknown. Use a gradient-free "
            "optimizer or SPSA."
        )
    return cast(tuple[tuple[float, int], ...], space.frequencies)


class RotosolveOptimizer(Optimizer):
    """Rotosolve (Ostaszewski et al.; Nakanishi et al.).

    With every other parameter fixed, the cost is a trigonometric polynomial in
    the remaining one. For a parameter with frequencies
    ``{omega, ..., order * omega}``, one batch of ``2 * order + 1`` equidistant
    evaluations determines that polynomial, and the parameter moves to its global
    minimum. One iteration sweeps every parameter once, then measures the cost at
    the result; the optimizer returns the best of these measurements.

    Frequencies come from the program's
    :attr:`~divi.qprog.VariationalQuantumAlgorithm.parameter_space`; without a
    program every parameter is taken to have frequency 1. See
    :ref:`rotosolve-optimizer` for references.

    Args:
        shuffle: Sweep the parameters in a fresh random order each iteration,
            drawn from the run's random number generator. A resumed run draws a
            new order.
    """

    def __init__(self, *, shuffle: bool = False):
        self.shuffle = shuffle
        self._x: npt.NDArray[np.float64] | None = None
        self._fun = np.inf
        self._best_x: npt.NDArray[np.float64] | None = None
        self._best_fun = np.inf
        self._nit = 0

    @property
    def n_param_sets(self) -> int:
        """Number of independent iterates optimised at once."""
        return 1

    def validate_program(self, program) -> None:
        """Require a known frequency for every parameter."""
        _require_known_frequencies(program.parameter_space)

    def build_evaluators(self, program) -> dict[str, Callable[..., Any]]:
        """Bind the program's parameter space, which fixes each parameter's
        frequencies."""
        return {"parameter_space": lambda: program.parameter_space}

    def optimize(
        self,
        cost_fn: Callable[[npt.NDArray[np.float64]], float | npt.NDArray[np.float64]],
        initial_params: npt.NDArray[np.float64] | None = None,
        callback_fn: Callable[[OptimizeResult], Any] | None = None,
        **kwargs,
    ) -> OptimizeResult:
        """Run ``max_iterations`` sweeps (default 50).

        A parameter whose batch holds a non-finite cost keeps its value for that
        sweep.
        """
        max_iterations = self._resolve_max_iterations(kwargs)
        parameter_space = kwargs.pop("parameter_space", None)
        rng = kwargs.pop("rng", None)
        if not self.shuffle:
            rng = None
        elif rng is None:
            rng = np.random.default_rng()

        if self._x is None or self._best_x is None:
            if initial_params is None:
                raise ValueError("RotosolveOptimizer requires initial_params.")
            x = np.array(initial_params, dtype=np.float64)
            if x.ndim == 2 and x.shape[0] == 1:
                x = x[0]
            if x.ndim != 1:
                raise ValueError(
                    "initial_params must contain exactly one parameter set."
                )
            if x.size == 0:
                raise ValueError("initial_params must contain at least one parameter.")
            fun, best_x, best_fun, nit = np.inf, x.copy(), np.inf, 0
        else:
            x, best_x = self._x.copy(), self._best_x.copy()
            fun, best_fun, nit = self._fun, self._best_fun, self._nit
        frequencies = _require_known_frequencies(
            ParameterSpace(((1.0, 1),) * x.size)
            if parameter_space is None
            else parameter_space()
        )
        if len(frequencies) != x.size:
            raise ValueError(
                f"The parameter space describes {len(frequencies)} parameters, "
                f"but the iterate has {x.size}."
            )

        end_iteration = nit + max_iterations
        while nit < end_iteration:
            sweep_order = range(x.size) if rng is None else rng.permutation(x.size)
            for index in sweep_order:
                omega, order = frequencies[index]
                n_samples = 2 * order + 1
                rows = np.tile(x, (n_samples, 1))
                rows[:, index] += 2 * np.pi * np.arange(n_samples) / (n_samples * omega)
                samples = np.asarray(cost_fn(rows), dtype=np.float64).reshape(-1)
                if samples.size != n_samples:
                    raise ValueError(
                        f"The cost function returned {samples.size} values for "
                        f"{n_samples} parameter sets."
                    )
                if np.isfinite(samples).all():
                    x[index] += _find_trigonometric_minimum(samples) / omega

            fun = float(
                np.asarray(cost_fn(x[None, :]), dtype=np.float64).reshape(-1)[0]
            )
            nit += 1
            if fun < best_fun:
                best_x, best_fun = x.copy(), fun
            self._x, self._fun, self._nit = x.copy(), fun, nit
            self._best_x, self._best_fun = best_x.copy(), best_fun

            if callback_fn is not None:
                callback_fn(
                    OptimizeResult(
                        x=np.atleast_2d(x.copy()), fun=np.atleast_1d(fun), nit=nit
                    )
                )

        return OptimizeResult(
            x=best_x.copy(),
            fun=np.atleast_1d(best_fun),
            nit=nit,
            success=np.isfinite(best_fun),
            message="Optimisation terminated: reached max_iterations.",
        )

    @property
    def has_resume_state(self) -> bool:
        """Whether an iterate from an earlier run is held."""
        return self._x is not None

    def get_config(self) -> dict[str, Any]:
        """Return constructor configuration for checkpoint reconstruction."""
        return {"type": type(self).__name__, "shuffle": self.shuffle}

    def save_state(self, checkpoint_dir: Path | str) -> None:
        """Persist the configuration, the current and best iterates and the
        iteration count."""
        if self._x is None or self._best_x is None:
            raise RuntimeError(
                "Cannot save checkpoint: Rotosolve optimisation has not been run."
            )
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        state = {
            "shuffle": self.shuffle,
            "x": self._x.tolist(),
            "fun": self._fun,
            "best_x": self._best_x.tolist(),
            "best_fun": self._best_fun,
            "nit": self._nit,
        }
        atomic_write(
            checkpoint_path / OPTIMIZER_STATE_FILE, json.dumps(state, indent=2)
        )

    @classmethod
    def load_state(cls, checkpoint_dir: Path | str) -> Self:
        """Restore a Rotosolve optimizer from a checkpoint directory."""
        with open(Path(checkpoint_dir) / OPTIMIZER_STATE_FILE) as file:
            state = json.load(file)

        optimizer = cls(shuffle=state["shuffle"])
        optimizer._x = np.asarray(state["x"], dtype=np.float64)
        optimizer._fun = float(state["fun"])
        optimizer._best_x = np.asarray(state["best_x"], dtype=np.float64)
        optimizer._best_fun = float(state["best_fun"])
        optimizer._nit = int(state["nit"])
        return optimizer

    def reset(self) -> None:
        """Clear the iterates and iteration count."""
        self._x = None
        self._fun = np.inf
        self._best_x = None
        self._best_fun = np.inf
        self._nit = 0

    def copy(self) -> Self:
        """Return a fresh optimizer with the same configuration."""
        return type(self)(shuffle=self.shuffle)
