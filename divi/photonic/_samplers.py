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

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from typing import Protocol, runtime_checkable

import numpy as np

from divi.photonic._ir import PhotonicProgram, PhotonicSamples
from divi.photonic._vendor.loop_progressive_simulator import (
    SparseBosonicFockState,
    multiple_loops,
    progressive_simulation,
)

# Minimum shots per worker before parallelism is worthwhile.  Below this
# the process dispatch overhead dominates.
_MIN_SHOTS_PER_WORKER = 4


def _simulate_chunk(args: tuple) -> np.ndarray:
    """Worker function: simulate a chunk of shots in a subprocess.

    Reconstructs the circuit from raw parameters so we never pickle
    vendored simulator objects (which may not be picklable across spawn).
    """
    loop_lengths, n_modes, thetas, input_state_tuple, chunk_shots, worker_seed = args

    # Each spawned worker re-imports everything (including numba JIT on first
    # call).  The import cost is paid once per worker process and amortised
    # across all subsequent submit() calls when the pool is reused.
    import numpy as np

    from divi.photonic._vendor.loop_progressive_simulator import (
        SparseBosonicFockState,
        multiple_loops,
        progressive_simulation,
    )

    if worker_seed is not None:
        np.random.seed(worker_seed)

    circuit = multiple_loops(*loop_lengths, modes=n_modes, thetas=thetas)
    input_state = SparseBosonicFockState({input_state_tuple: 1.0})

    samples = np.zeros((chunk_shots, n_modes), dtype=np.int64)
    for i, sample in enumerate(
        progressive_simulation(input_state, circuit, number_samples=chunk_shots)
    ):
        L = len(sample)
        samples[i, :L] = sample

    return samples


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
        seed: Optional RNG seed. When set, deterministic (but different)
            per-worker seeds are derived so results are reproducible.
            Note: parallel results are deterministic but not identical to
            the sequential path (different shot-to-worker assignment).
        n_workers: Number of worker processes for parallel shot simulation.
            Defaults to ``min(os.cpu_count(), 8)``. Set to ``1`` to force
            sequential execution (useful for debugging or profiling).
    """

    def __init__(self, seed: int | None = None, n_workers: int | None = None):
        self._seed = seed
        if n_workers is None:
            n_workers = min(os.cpu_count() or 1, 8)
        self._n_workers = max(1, n_workers)
        self._pool: ProcessPoolExecutor | None = None

    def submit(self, program: PhotonicProgram, shots: int) -> PhotonicSamples:
        if shots <= 0:
            raise ValueError(f"shots must be positive, got {shots}")

        # Decide whether parallelism is worthwhile for this call.
        effective_workers = min(
            self._n_workers, max(1, shots // _MIN_SHOTS_PER_WORKER)
        )

        if effective_workers > 1:
            samples = self._submit_parallel(program, shots, effective_workers)
        else:
            samples = self._submit_sequential(program, shots)

        return PhotonicSamples(
            samples=samples,
            n_modes=program.n_modes,
            n_photons=program.n_photons,
        )

    # ── Sequential (original) path ──────────────────────────────────

    def _submit_sequential(
        self, program: PhotonicProgram, shots: int
    ) -> np.ndarray:
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
            L = len(sample)
            samples[i, :L] = sample

        return samples

    # ── Parallel path ───────────────────────────────────────────────

    def _submit_parallel(
        self, program: PhotonicProgram, shots: int, n_workers: int
    ) -> np.ndarray:
        # Split shots evenly across workers.
        base, remainder = divmod(shots, n_workers)
        chunk_sizes = [base + (1 if i < remainder else 0) for i in range(n_workers)]

        # Derive deterministic per-worker seeds from the master seed.
        if self._seed is not None:
            rng = np.random.default_rng(self._seed)
            worker_seeds = [int(rng.integers(0, 2**31)) for _ in range(n_workers)]
        else:
            # No master seed → each worker gets a random seed from OS entropy.
            worker_seeds = [
                int(np.random.default_rng().integers(0, 2**31))
                for _ in range(n_workers)
            ]

        args_list = [
            (
                program.loop_lengths,
                program.n_modes,
                list(program.parameters),
                tuple(program.input_state),
                chunk,
                seed,
            )
            for chunk, seed in zip(chunk_sizes, worker_seeds)
        ]

        # Lazily create (and reuse) the process pool.  Workers stay alive
        # across submit() calls so the numba JIT cost is paid only once.
        if self._pool is None:
            self._pool = ProcessPoolExecutor(max_workers=n_workers)

        futures = [self._pool.submit(_simulate_chunk, a) for a in args_list]
        chunks = [f.result() for f in futures]

        return np.vstack(chunks)

    def close(self) -> None:
        """Shut down the worker pool, if any."""
        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None

    def __del__(self) -> None:
        self.close()
