# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Variational QUBO on a multi-loop TBI.

Ports the algorithm from Orca's ``orcacomputing/quantumqubo`` (stale since
2021, Apache-2.0, with attribution): Adam over beamsplitter angles,
parameter-shift gradient at ``±π/6`` on the readout, and the four-
configuration ``(n_photons × parity)`` sweep that uniformises bitstring
support. Generalised to multi-loop TBI via the vendored
``loop-progressive-simulator``.

One caveat preserved from the plan: the bitstring-coverage argument for
the four-config sweep was originally derived for single-loop TBI. For
multi-loop we re-verify empirically — :meth:`TBIVariationalQUBO.run`
records the fraction of the ``2^M`` bitstrings observed at the end of
training in ``TBIQUBOResult.bitstring_coverage``, and the caller can
check this is not pathologically small.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from divi.photonic._adam import AdamState, adam_step
from divi.photonic._ir import (
    PhotonicProgram,
    PhotonicSamples,
    count_beamsplitters,
)
from divi.photonic._observables import ObservableResult, ParityQUBO
from divi.photonic._samplers import PhotonicSampler, SimulatedTBISampler


@dataclass
class TBIQUBOConfig:
    """One of four configurations in the quantumqubo sweep.

    Args:
        label: Human-readable label (``"config1"`` ... ``"config4"``).
        n_modes: Total TBI modes in this configuration.
        n_photons: Total injected photons.
        input_state: Fock occupation (length = ``n_modes``).
        parity: 0 or 1, parity offset in the bit mapping.
        mode_offset: Leading modes dropped before the parity mapping.
            1 when the first beamsplitter is fixed to 0 (fully reflective
            → mode 0 always empty on measurement), else 0.
        n_fixed_bs: Number of leading beamsplitters whose angle is
            frozen to 0. 1 or 0.
    """

    label: str
    n_modes: int
    n_photons: int
    input_state: tuple[int, ...]
    parity: int
    mode_offset: int
    n_fixed_bs: int


def build_four_configs(M: int) -> list[TBIQUBOConfig]:
    """Build the quantumqubo four-configuration sweep for an ``M``-variable QUBO.

    * Configs 1 & 2: ``(M+1)`` modes, ``(M+1)`` photons, one fixed
      beamsplitter at angle 0 → the first output mode is always empty
      → the remaining M modes parity-map to M bits.
    * Configs 3 & 4: ``M`` modes, ``(M-1)`` photons (last mode is empty in
      the input), no fixed beamsplitter, parity-map all M modes to M bits.

    Each pair differs only in parity (0 vs 1) to sweep the parity offset.
    The argument for uniform bitstring coverage is exact in single-loop
    TBI; for multi-loop it's approximate and must be checked empirically.
    """
    return [
        TBIQUBOConfig(
            label="config1",
            n_modes=M + 1,
            n_photons=M + 1,
            input_state=(1,) * (M + 1),
            parity=0,
            mode_offset=1,
            n_fixed_bs=1,
        ),
        TBIQUBOConfig(
            label="config2",
            n_modes=M + 1,
            n_photons=M + 1,
            input_state=(1,) * (M + 1),
            parity=1,
            mode_offset=1,
            n_fixed_bs=1,
        ),
        TBIQUBOConfig(
            label="config3",
            n_modes=M,
            n_photons=M - 1,
            input_state=(1,) * (M - 1) + (0,),
            parity=0,
            mode_offset=0,
            n_fixed_bs=0,
        ),
        TBIQUBOConfig(
            label="config4",
            n_modes=M,
            n_photons=M - 1,
            input_state=(1,) * (M - 1) + (0,),
            parity=1,
            mode_offset=0,
            n_fixed_bs=0,
        ),
    ]


@dataclass
class TBIQUBOResult:
    """Outcome of :meth:`TBIVariationalQUBO.run`.

    ``best_bitstring`` / ``best_value`` are the global minimum across all
    four configurations. ``per_config`` carries the energy trajectories
    and per-config optima for diagnostics / plotting.
    """

    best_bitstring: tuple[int, ...]
    best_value: float
    per_config: dict[str, dict[str, Any]]
    bitstring_coverage: dict[str, float] = field(default_factory=dict)
    total_shots: int = 0


class _ConfigRunner:
    """Internal: runs the Adam loop for one of the four configurations."""

    def __init__(
        self,
        config: TBIQUBOConfig,
        Q: np.ndarray,
        sampler: PhotonicSampler,
        loop_lengths: tuple[int, ...],
        learning_rate: float,
        parameter_shift: float,
        rng: np.random.Generator,
    ):
        self.config = config
        self.loop_lengths = loop_lengths
        total_bs = count_beamsplitters(config.n_modes, loop_lengths)
        self.n_variable = total_bs - config.n_fixed_bs

        # Initialise θ like quantumqubo: randn with small spread.
        self.theta = rng.standard_normal(self.n_variable).astype(np.float64)

        self.sampler = sampler
        self.observable = ParityQUBO(Q, config.parity, config.mode_offset)
        self.parameter_shift = parameter_shift
        self.adam = AdamState(learning_rate=learning_rate)

        self.history_energy: list[float] = []
        self.best_bitstring: tuple[int, ...] | None = None
        self.best_value = float("inf")
        self._shot_counter = 0
        self._observed_bitstrings: set[tuple[int, ...]] = set()

    def _full_thetas(self, theta: np.ndarray) -> list[float]:
        """Prepend the frozen fixed-reflective angle(s), if any."""
        if self.config.n_fixed_bs > 0:
            return [0.0] * self.config.n_fixed_bs + list(theta)
        return list(theta)

    def _evaluate(self, theta: np.ndarray, shots: int) -> ObservableResult:
        program = PhotonicProgram(
            n_modes=self.config.n_modes,
            input_state=self.config.input_state,
            parameters=np.asarray(self._full_thetas(theta), dtype=np.float64),
            loop_lengths=self.loop_lengths,
        )
        samples = self.sampler.submit(program, shots=shots)
        self._shot_counter += shots
        self._track_bitstrings(samples)
        return self.observable.evaluate(samples)

    def _track_bitstrings(self, samples: PhotonicSamples) -> None:
        from divi.photonic._observables import parity_bitstring

        bits = parity_bitstring(
            samples.samples, self.config.parity, self.config.mode_offset
        )
        for row in bits:
            self._observed_bitstrings.add(tuple(int(x) for x in row))

    def step(self, shots: int) -> float:
        # Current-point expectation.
        result = self._evaluate(self.theta, shots)
        current = result.value
        self.history_energy.append(current)
        if (
            result.best_bitstring is not None
            and result.best_value is not None
            and result.best_value < self.best_value
        ):
            self.best_value = result.best_value
            self.best_bitstring = result.best_bitstring

        # Parameter-shift gradient at ±parameter_shift.
        grad = np.zeros_like(self.theta)
        for j in range(len(self.theta)):
            theta_plus = self.theta.copy()
            theta_plus[j] += self.parameter_shift
            theta_minus = self.theta.copy()
            theta_minus[j] -= self.parameter_shift

            E_plus = self._evaluate(theta_plus, shots).value
            E_minus = self._evaluate(theta_minus, shots).value
            grad[j] = (E_plus - E_minus) / np.sin(2.0 * self.parameter_shift)

        self.theta = adam_step(self.adam, self.theta, grad)
        return current


class TBIVariationalQUBO:
    """Variational QUBO solver on a multi-loop TBI.

    Usage::

        M = 4
        Q = np.random.default_rng(0).standard_normal((M, M))
        solver = TBIVariationalQUBO(Q, loop_lengths=(1, 2))
        result = solver.run(updates=30, shots=200, seed=0)
        print(result.best_bitstring, result.best_value)

    The optimiser runs four Adam loops in sequence (one per
    configuration); divi's ensemble scheduling could parallelise them in
    future, but the PoC keeps the loop sequential so the per-config
    trajectories are easy to read.
    """

    def __init__(
        self,
        Q: np.ndarray,
        loop_lengths: tuple[int, ...] = (1,),
        sampler: PhotonicSampler | None = None,
    ):
        Q = np.asarray(Q, dtype=np.float64)
        if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
            raise ValueError(f"Q must be square, got shape {Q.shape}")
        if Q.shape[0] < 2:
            raise ValueError(f"QUBO dimension must be >= 2, got {Q.shape[0]}")

        self.Q = Q
        self.M = Q.shape[0]
        self.loop_lengths = tuple(loop_lengths)
        self.sampler = sampler if sampler is not None else SimulatedTBISampler()

    def run(
        self,
        updates: int = 50,
        shots: int = 200,
        learning_rate: float = 5e-2,
        parameter_shift: float = np.pi / 6,
        seed: int | None = None,
        progress: Callable[[str, int, float], None] | None = None,
    ) -> TBIQUBOResult:
        """Run the four-config sweep and return the best bitstring found.

        Args:
            updates: Number of Adam updates per configuration.
            shots: Photonic shots per expectation-value evaluation. Each
                Adam update costs ``(2*n_variable + 1) * shots`` photonic
                shots (parameter-shift rule + one center evaluation).
            learning_rate: Adam learning rate.
            parameter_shift: Parameter-shift rule offset.
            seed: RNG seed for θ initialisation. Note: the sampler's own
                seed is independent; pass one via the sampler constructor
                if you want fully-deterministic shots too.
            progress: Optional callback ``(config_label, update_idx, energy)``
                invoked after each update.
        """
        rng = np.random.default_rng(seed)
        configs = build_four_configs(self.M)

        per_config: dict[str, dict[str, Any]] = {}
        coverage: dict[str, float] = {}
        best_value = float("inf")
        best_bitstring: tuple[int, ...] = (0,) * self.M
        total_shots = 0

        for cfg in configs:
            runner = _ConfigRunner(
                cfg,
                self.Q,
                self.sampler,
                self.loop_lengths,
                learning_rate,
                parameter_shift,
                rng,
            )
            for i in range(updates):
                energy = runner.step(shots)
                if progress is not None:
                    progress(cfg.label, i, energy)

            per_config[cfg.label] = {
                "history_energy": runner.history_energy,
                "best_bitstring": runner.best_bitstring,
                "best_value": runner.best_value,
                "theta": runner.theta,
                "shots": runner._shot_counter,
            }
            total_shots += runner._shot_counter
            coverage[cfg.label] = len(runner._observed_bitstrings) / (2**self.M)

            if runner.best_bitstring is not None and runner.best_value < best_value:
                best_value = runner.best_value
                best_bitstring = runner.best_bitstring

        return TBIQUBOResult(
            best_bitstring=best_bitstring,
            best_value=best_value,
            per_config=per_config,
            bitstring_coverage=coverage,
            total_shots=total_shots,
        )
