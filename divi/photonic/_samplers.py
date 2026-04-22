# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Sampler protocol and a simulated TBI adapter.

The adapter wraps the vendored subset of Orca's
`loop-progressive-simulator <https://github.com/orcacomputing/loop-progressive-simulator>`_
(arXiv:2411.16873). It accepts a :class:`~divi.photonic.PhotonicProgram`
and produces :class:`~divi.photonic.PhotonicSamples` (dense integer arrays
of photon counts per mode).
"""

from typing import Protocol, runtime_checkable

import numpy as np

from divi.photonic._ir import PhotonicProgram, PhotonicSamples
from divi.photonic._vendor.loop_progressive_simulator import (
    SparseBosonicFockState,
    multiple_loops,
    progressive_simulation,
)


@runtime_checkable
class PhotonicSampler(Protocol):
    """Protocol implemented by photonic samplers.

    A sampler takes a :class:`PhotonicProgram` and a shot count and returns
    :class:`PhotonicSamples` containing the photon-count outcomes. The
    intent is that future adapters — e.g. a ``RemoteTBISampler`` for the
    Orca cloud, or a CUDA-Q-routed variant — satisfy the same protocol.
    """

    def submit(self, program: PhotonicProgram, shots: int) -> PhotonicSamples: ...


class SimulatedTBISampler:
    """A classical simulator for multi-loop TBI programs.

    Uses the progressive-sampling algorithm from arXiv:2411.16873 (via the
    vendored ``loop-progressive-simulator`` subset). Practical on a laptop
    up to ~12-16 modes; scaling beyond that requires real hardware or
    approximate samplers.

    Args:
        seed: Optional RNG seed. When set, the sampler re-seeds
            :func:`numpy.random.seed` before each ``submit`` call. The
            upstream simulator uses :mod:`numpy.random` under the hood, so
            this is the simplest way to get deterministic shots.
    """

    def __init__(self, seed: int | None = None):
        self._seed = seed

    def submit(self, program: PhotonicProgram, shots: int) -> PhotonicSamples:
        if shots <= 0:
            raise ValueError(f"shots must be positive, got {shots}")

        if self._seed is not None:
            np.random.seed(self._seed)

        circuit = multiple_loops(
            *program.loop_lengths,
            modes=program.n_modes,
            thetas=list(program.parameters),
        )

        input_state = SparseBosonicFockState({tuple(program.input_state): 1.0})

        samples = np.zeros((shots, program.n_modes), dtype=np.int64)
        for i, sample in enumerate(
            progressive_simulation(input_state, circuit, number_samples=shots)
        ):
            # The vendored simulator returns tuples with trailing zeros
            # truncated (sparse Fock convention). Pad back to n_modes.
            L = len(sample)
            samples[i, :L] = sample

        return PhotonicSamples(
            samples=samples,
            n_modes=program.n_modes,
            n_photons=program.n_photons,
        )
