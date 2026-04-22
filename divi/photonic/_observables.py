# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Observables / losses for TBI samples.

Deliberately separate from divi's Pauli-based observable machinery: TBI
output is a photon-count tuple per mode, not a computational-basis
bitstring, so Pauli decomposition / eigenvalue maps do not apply.
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from divi.photonic._ir import PhotonicSamples


@dataclass(frozen=True)
class ObservableResult:
    """Scalar value (and optionally a best-candidate bitstring) from a loss.

    ``value`` is the expectation under the sample distribution — used by
    the optimiser. ``best_bitstring`` and ``best_value`` are the single
    best candidate encountered, used to report the solution at the end of
    a QUBO run (irrelevant for distribution-matching losses like MMD,
    where they stay ``None``).
    """

    value: float
    best_bitstring: tuple[int, ...] | None = None
    best_value: float | None = None


@runtime_checkable
class PhotonicObservable(Protocol):
    """Protocol for things that reduce :class:`PhotonicSamples` to a scalar."""

    def evaluate(self, samples: PhotonicSamples) -> ObservableResult: ...


def parity_bitstring(
    photon_counts: np.ndarray, parity: int, mode_offset: int = 0
) -> np.ndarray:
    """Mode-local parity mapping ``b_i = (n_i + parity) mod 2``.

    Args:
        photon_counts: Integer array of shape ``(n_shots, n_modes)`` or
            ``(n_modes,)``.
        parity: 0 or 1.
        mode_offset: Drop this many leading modes before the mapping. Used
            by the QUBO "fixed first beamsplitter" configurations where
            mode 0 is forced to zero occupation and contributes no bit.

    Returns:
        Integer array of bits, shape
        ``(n_shots, n_modes - mode_offset)`` (or 1D accordingly).
    """
    if parity not in (0, 1):
        raise ValueError(f"parity must be 0 or 1, got {parity}")
    sliced = photon_counts[..., mode_offset:]
    return (sliced + parity) % 2


class ParityQUBO:
    """QUBO loss ``b^T Q b`` under a parity mapping of photon counts.

    Mirrors the readout convention of
    `orcacomputing/quantumqubo <https://github.com/orcacomputing/quantumqubo>`_
    (``QuboOneConfiguration.readout``) with attribution: takes photon-count
    samples, parity-maps them to bitstrings, evaluates the QUBO quadratic
    form, and returns the sample-averaged energy plus the best bitstring
    encountered.

    Args:
        Q: Symmetric (or upper-triangular; we symmetrise) QUBO matrix of
            shape ``(n_bits, n_bits)``.
        parity: 0 or 1, picks which parity offset is applied.
        mode_offset: Number of leading modes to drop before mapping (used
            for QUBO configurations with a fixed-reflective first
            beamsplitter).
    """

    def __init__(self, Q: np.ndarray, parity: int, mode_offset: int = 0):
        Q = np.asarray(Q, dtype=np.float64)
        if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
            raise ValueError(f"Q must be square, got shape {Q.shape}")
        # QUBO energy is invariant under Q -> (Q + Q^T) / 2 on {0,1} vectors,
        # but we symmetrise once to simplify downstream vectorised evaluation.
        self.Q = 0.5 * (Q + Q.T)
        self.parity = parity
        self.mode_offset = mode_offset
        self.n_bits = Q.shape[0]

    def evaluate(self, samples: PhotonicSamples) -> ObservableResult:
        bits = parity_bitstring(samples.samples, self.parity, self.mode_offset)
        if bits.shape[1] != self.n_bits:
            raise ValueError(
                f"QUBO dimension {self.n_bits} does not match bitstring "
                f"length {bits.shape[1]} (n_modes={samples.n_modes}, "
                f"mode_offset={self.mode_offset})"
            )

        # Vectorised b^T Q b per shot: einsum 'si,ij,sj->s'
        energies = np.einsum("si,ij,sj->s", bits, self.Q, bits)

        mean_energy = float(energies.mean())
        best_idx = int(np.argmin(energies))
        best_bits = tuple(int(x) for x in bits[best_idx])
        best_val = float(energies[best_idx])

        return ObservableResult(
            value=mean_energy,
            best_bitstring=best_bits,
            best_value=best_val,
        )


class MMDSampleLoss:
    """Maximum-mean-discrepancy (MMD) between TBI samples and target samples.

    Used to train Boson Sampling Born Machines ("train classically, deploy
    quantumly" — the MMD evaluation runs on classical simulator samples,
    and the same parameters would drive the photonic hardware at inference).

    Uses a mixture of Gaussian kernels in the *bitstring parity space* of
    the photon counts — the model defines a distribution over bit strings,
    and we match it against a target distribution over bit strings. The
    kernel bandwidths are a small fixed set (2, 5, 10) by default; see the
    Rahimi-Recht / Gretton et al. usage of multi-bandwidth Gaussian kernels
    for QCBM/BSBM training.

    Args:
        target_bits: Target sample array, shape ``(n_target, n_bits)``,
            entries in ``{0, 1}``.
        kernel_bandwidths: Gaussian bandwidths ``σ``; the kernel is
            ``k(x, y) = Σ_σ exp(-||x-y||² / (2 σ²))``.
        parity: Parity offset applied when mapping photon counts to bits.
        mode_offset: Leading modes to drop before the mapping.
    """

    def __init__(
        self,
        target_bits: np.ndarray,
        kernel_bandwidths: tuple[float, ...] = (2.0, 5.0, 10.0),
        parity: int = 0,
        mode_offset: int = 0,
    ):
        target_bits = np.asarray(target_bits, dtype=np.int64)
        if target_bits.ndim != 2:
            raise ValueError(f"target_bits must be 2D, got shape {target_bits.shape}")
        self.target_bits = target_bits
        self.kernel_bandwidths = tuple(kernel_bandwidths)
        self.parity = parity
        self.mode_offset = mode_offset
        self.n_bits = target_bits.shape[1]

        # Precompute target-target kernel mean (constant across training).
        self._kyy = self._kernel_mean(target_bits, target_bits)

    def _kernel_matrix(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        # Squared L2 distance between all rows of A and B.
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a . b
        a2 = np.sum(A * A, axis=1, keepdims=True)
        b2 = np.sum(B * B, axis=1, keepdims=True).T
        d2 = a2 + b2 - 2.0 * A @ B.T
        d2 = np.maximum(d2, 0.0)

        K = np.zeros_like(d2, dtype=np.float64)
        for sigma in self.kernel_bandwidths:
            K += np.exp(-d2 / (2.0 * sigma * sigma))
        return K

    def _kernel_mean(self, A: np.ndarray, B: np.ndarray) -> float:
        return float(self._kernel_matrix(A, B).mean())

    def evaluate(self, samples: PhotonicSamples) -> ObservableResult:
        bits = parity_bitstring(samples.samples, self.parity, self.mode_offset)
        if bits.shape[1] != self.n_bits:
            raise ValueError(
                f"target has {self.n_bits} bits but sample bitstrings are "
                f"{bits.shape[1]}-long"
            )

        kxx = self._kernel_mean(bits, bits)
        kxy = self._kernel_mean(bits, self.target_bits)
        mmd2 = kxx + self._kyy - 2.0 * kxy

        return ObservableResult(value=float(mmd2))
