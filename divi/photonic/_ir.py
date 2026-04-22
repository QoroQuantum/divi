# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Photonic IR: program (submitted) + samples (returned) dataclasses.

Distinct from ``divi.backends.ExecutionResult`` because photonic samples
carry photon-count tuples per mode, not bitstrings with Pauli-eigenvalue
semantics. The program IR is intentionally *not* QASM / Blackbird: it is
the minimal description needed by a multi-loop TBI sampler, with
``loop_lengths`` as the discriminator that covers single-loop PT-1-style
and nested-loop PT-2-style architectures.
"""

from collections import Counter
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PhotonicProgram:
    """A parametrised loop-based TBI program.

    The fields map one-to-one onto
    :func:`loop_progressive_simulator.bscircuits.multiple_loops` — choose
    ``loop_lengths=(1,)`` for a single-loop PT-1-style TBI, or multiple
    loop lengths (e.g. ``(1, 2)``) for a PT-2-style multi-loop TBI.

    Attributes:
        n_modes: Total number of photonic modes in the TBI (``M``).
        input_state: Fock occupation of each input mode, length ``n_modes``.
            The total photon count is ``sum(input_state)``.
        parameters: Beamsplitter angles ``θ``. The expected length is
            ``sum(n_modes - L for L in loop_lengths)`` — one angle per
            beamsplitter in the composed multi-loop circuit.
        loop_lengths: Loop lengths for the multi-loop TBI architecture.
    """

    n_modes: int
    input_state: tuple[int, ...]
    parameters: np.ndarray
    loop_lengths: tuple[int, ...] = (1,)

    def __post_init__(self):
        if len(self.input_state) != self.n_modes:
            raise ValueError(
                f"input_state length {len(self.input_state)} does not match "
                f"n_modes {self.n_modes}"
            )

        expected_n_bs = sum(self.n_modes - L for L in self.loop_lengths)
        if len(self.parameters) != expected_n_bs:
            raise ValueError(
                f"parameters length {len(self.parameters)} does not match "
                f"expected number of beamsplitters {expected_n_bs} for "
                f"n_modes={self.n_modes}, loop_lengths={self.loop_lengths}"
            )

        if any(L >= self.n_modes for L in self.loop_lengths):
            raise ValueError(
                f"every loop length must be < n_modes; got "
                f"loop_lengths={self.loop_lengths}, n_modes={self.n_modes}"
            )

    @property
    def n_photons(self) -> int:
        return sum(self.input_state)

    @property
    def n_beamsplitters(self) -> int:
        return len(self.parameters)


def count_beamsplitters(n_modes: int, loop_lengths: tuple[int, ...]) -> int:
    """Number of beamsplitters in a multi-loop TBI with the given geometry."""
    return sum(n_modes - L for L in loop_lengths)


@dataclass(frozen=True)
class PhotonicSamples:
    """Photon-count samples from a TBI (or any Fock-measurement sampler).

    Attributes:
        samples: Integer array of shape ``(n_shots, n_modes)``. Entry
            ``samples[s, m]`` is the number of photons detected in mode
            ``m`` for shot ``s``.
        n_modes: Number of modes.
        n_photons: Total photons in the input state (conserved per shot).
    """

    samples: np.ndarray
    n_modes: int
    n_photons: int

    def __post_init__(self):
        if self.samples.ndim != 2 or self.samples.shape[1] != self.n_modes:
            raise ValueError(
                f"expected samples of shape (n_shots, {self.n_modes}); "
                f"got shape {self.samples.shape}"
            )
        if not np.issubdtype(self.samples.dtype, np.integer):
            raise ValueError(f"samples must be integer-typed, got {self.samples.dtype}")

    @property
    def n_shots(self) -> int:
        return self.samples.shape[0]

    def as_counts(self) -> dict[tuple[int, ...], int]:
        """Aggregate samples into a ``{photon_count_tuple: shots}`` dict."""
        return dict(Counter(map(tuple, self.samples.tolist())))
