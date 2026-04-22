# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Boson Sampling Born Machine (BSBM) training on a multi-loop TBI.

Trains the beamsplitter angles of a TBI so the induced distribution over
parity-mapped bitstrings matches a target distribution, under a multi-
bandwidth Gaussian MMD loss. Follows the "train classically, deploy
quantumly" paradigm — the MMD is evaluated on classical-simulator shots
here, but the same θ would drive a photonic chip at inference.

Ships with a bars-and-stripes dataset helper as the canonical BSBM toy
problem.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from divi.photonic._adam import AdamState, adam_step
from divi.photonic._ir import PhotonicProgram, count_beamsplitters
from divi.photonic._observables import MMDSampleLoss
from divi.photonic._samplers import PhotonicSampler, SimulatedTBISampler

# --------------------------------------------------------------------------
# Dataset helper
# --------------------------------------------------------------------------


class BarsAndStripes:
    """Canonical Born-machine toy dataset.

    A bars-and-stripes image of size ``n × n`` is either all rows the same
    (horizontal bars) or all columns the same (vertical stripes). The
    dataset is the union of the two sets of patterns, flattened to a
    bit string of length ``n²``.

    Args:
        n: Side length of the grid.
        exclude_all_constant: If True (default), drop the all-0 and
            all-1 images that appear in both families.
    """

    def __init__(self, n: int = 3, exclude_all_constant: bool = True):
        self.n = n
        self.n_bits = n * n
        self._patterns = self._build(n, exclude_all_constant)

    @staticmethod
    def _build(n: int, exclude_all_constant: bool) -> np.ndarray:
        patterns = set()
        for k in range(2**n):
            mask = np.array([(k >> i) & 1 for i in range(n)], dtype=np.int64)
            bars = np.tile(mask[:, None], (1, n)).reshape(-1)
            stripes = np.tile(mask[None, :], (n, 1)).reshape(-1)
            patterns.add(tuple(bars.tolist()))
            patterns.add(tuple(stripes.tolist()))
        if exclude_all_constant:
            patterns.discard((0,) * (n * n))
            patterns.discard((1,) * (n * n))
        return np.array(sorted(patterns), dtype=np.int64)

    @property
    def patterns(self) -> np.ndarray:
        return self._patterns

    def sample(self, n_samples: int, rng: np.random.Generator) -> np.ndarray:
        idx = rng.integers(0, len(self._patterns), size=n_samples)
        return self._patterns[idx].copy()


# --------------------------------------------------------------------------
# Algorithm
# --------------------------------------------------------------------------


@dataclass
class TBIBornMachineResult:
    theta: np.ndarray
    history_mmd: list[float]
    final_samples_bits: np.ndarray
    total_shots: int = 0
    per_step_metrics: dict[str, list[float]] = field(default_factory=dict)


class TBIBornMachine:
    """Train a multi-loop TBI as a Boson Sampling Born Machine.

    The model parametrises the output distribution of a TBI with
    beamsplitter angles ``θ``; a parity mapping from photon-count tuples
    to bit strings makes the induced distribution live on ``{0, 1}^n_bits``.
    Training minimises the MMD between model samples and target samples
    using the same numpy Adam + parameter-shift machinery as
    :class:`TBIVariationalQUBO`.

    Args:
        n_bits: Dimension of the bit-string space (e.g. ``9`` for
            3×3 bars-and-stripes).
        target_bits: Target dataset, shape ``(n_target, n_bits)``.
        n_modes: Number of TBI modes. Must be ``>= n_bits``.
        n_photons: Number of photons injected. Defaults to
            ``min(n_modes - 1, n_modes // 2 + 1)`` (a reasonable default
            for loop-based TBI: enough photons for interesting statistics
            but below saturation).
        loop_lengths: Multi-loop geometry.
        parity: Parity offset for the photon-count → bitstring mapping.
        mode_offset: Leading modes to drop before the parity mapping
            (0 by default — BSBM doesn't need a fixed-reflective first BS).
        kernel_bandwidths: Gaussian-kernel bandwidths for the MMD.
        sampler: Optional custom :class:`PhotonicSampler`; defaults to the
            vendored progressive-simulation-backed simulator.
    """

    def __init__(
        self,
        n_bits: int,
        target_bits: np.ndarray,
        n_modes: int,
        n_photons: int | None = None,
        loop_lengths: tuple[int, ...] = (1, 2),
        parity: int = 0,
        mode_offset: int = 0,
        kernel_bandwidths: tuple[float, ...] = (2.0, 5.0, 10.0),
        sampler: PhotonicSampler | None = None,
    ):
        if n_modes - mode_offset != n_bits:
            raise ValueError(
                f"After mode_offset={mode_offset}, the TBI produces "
                f"{n_modes - mode_offset}-bit strings but target has "
                f"{n_bits} bits."
            )

        if n_photons is None:
            n_photons = min(n_modes - 1, n_modes // 2 + 1)
        if n_photons <= 0 or n_photons >= n_modes * 4:
            raise ValueError(f"unreasonable n_photons={n_photons}")

        # Input state: first n_photons modes occupied (keeps input
        # consistent with quantumqubo's (1,)*k convention; BSBM is not
        # sensitive to this choice but we pin it for reproducibility).
        self.n_bits = n_bits
        self.n_modes = n_modes
        self.n_photons = n_photons
        self.loop_lengths = tuple(loop_lengths)
        self.parity = parity
        self.mode_offset = mode_offset
        self.input_state: tuple[int, ...] = tuple(
            1 if i < n_photons else 0 for i in range(n_modes)
        )

        self.observable = MMDSampleLoss(
            target_bits=target_bits,
            kernel_bandwidths=kernel_bandwidths,
            parity=parity,
            mode_offset=mode_offset,
        )

        self.sampler = sampler if sampler is not None else SimulatedTBISampler()
        self.n_beamsplitters = count_beamsplitters(n_modes, self.loop_lengths)

    def _evaluate(self, theta: np.ndarray, shots: int):
        program = PhotonicProgram(
            n_modes=self.n_modes,
            input_state=self.input_state,
            parameters=theta,
            loop_lengths=self.loop_lengths,
        )
        samples = self.sampler.submit(program, shots=shots)
        return samples, self.observable.evaluate(samples).value

    def run(
        self,
        updates: int = 200,
        shots: int = 200,
        learning_rate: float = 5e-2,
        parameter_shift: float = np.pi / 6,
        seed: int | None = None,
        progress: Callable[[int, float], None] | None = None,
    ) -> TBIBornMachineResult:
        """Train the BSBM and return the final model samples + MMD history."""
        rng = np.random.default_rng(seed)
        theta = rng.standard_normal(self.n_beamsplitters).astype(np.float64)
        adam = AdamState(learning_rate=learning_rate)

        history: list[float] = []
        total_shots = 0

        for step in range(updates):
            _, mmd = self._evaluate(theta, shots)
            total_shots += shots
            history.append(mmd)
            if progress is not None:
                progress(step, mmd)

            grad = np.zeros_like(theta)
            for j in range(len(theta)):
                theta_plus = theta.copy()
                theta_plus[j] += parameter_shift
                theta_minus = theta.copy()
                theta_minus[j] -= parameter_shift
                _, mmd_plus = self._evaluate(theta_plus, shots)
                _, mmd_minus = self._evaluate(theta_minus, shots)
                total_shots += 2 * shots
                grad[j] = (mmd_plus - mmd_minus) / np.sin(2.0 * parameter_shift)

            theta = adam_step(adam, theta, grad)

        from divi.photonic._observables import parity_bitstring

        final_samples, final_mmd = self._evaluate(theta, shots * 5)
        total_shots += shots * 5
        history.append(final_mmd)
        final_bits = parity_bitstring(
            final_samples.samples, self.parity, self.mode_offset
        )
        return TBIBornMachineResult(
            theta=theta,
            history_mmd=history,
            final_samples_bits=final_bits,
            total_shots=total_shots,
        )
