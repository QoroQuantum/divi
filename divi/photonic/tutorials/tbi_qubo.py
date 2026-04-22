# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Variational QUBO on a multi-loop photonic time-bin interferometer.

This tutorial demonstrates divi's photonic middleware on an Orca-style
loop-based TBI. A small QUBO is solved via the four-configuration
parameter-shift sweep ported from ``orcacomputing/quantumqubo`` (Apache-2.0),
running against a multi-loop TBI simulator. Switching to Orca cloud
hardware in the future is a single-line ``sampler=`` swap, using any
backend that satisfies ``divi.photonic.PhotonicSampler``.
"""

import itertools
import time

import numpy as np

from divi.photonic import SimulatedTBISampler, TBIVariationalQUBO


def brute_force_qubo(Q: np.ndarray) -> tuple[tuple[int, ...], float]:
    M = Q.shape[0]
    best_bitstring = (0,) * M
    best_energy = float("inf")
    for bits in itertools.product([0, 1], repeat=M):
        b = np.asarray(bits, dtype=np.float64)
        energy = float(b @ Q @ b)
        if energy < best_energy:
            best_energy = energy
            best_bitstring = bits
    return best_bitstring, best_energy


if __name__ == "__main__":
    # A 4-variable QUBO we can brute-force for a sanity check.
    rng = np.random.default_rng(42)
    M = 4
    Q = rng.standard_normal((M, M))
    Q = 0.5 * (Q + Q.T)

    classical_bits, classical_energy = brute_force_qubo(Q)
    print(f"Brute-force optimum: {classical_bits} with energy {classical_energy:.4f}")

    # Multi-loop TBI matching a PT-2-style (nested-loop) architecture.
    # loop_lengths=(1, 2) gives two loop layers of different delays.
    # Swap in ``sampler=RemoteTBISampler(endpoint=...)`` to target real hardware.
    sampler = SimulatedTBISampler(seed=0)

    print(f"Running QUBO on {M} qubits with Time Bin Interferometer Simulator, takes ~2 minutes.")

    solver = TBIVariationalQUBO(Q, loop_lengths=(1, 2), sampler=sampler)
    t0 = time.time()
    result = solver.run(updates=20, shots=200, seed=0)
    elapsed = time.time() - t0

    print(
        f"TBI variational QUBO found: {result.best_bitstring} "
        f"with energy {result.best_value:.4f}"
    )
    print(f"Ran 4 configurations, {result.total_shots} total shots, {elapsed:.1f}s")
    print("Bitstring coverage per config (fraction of 2^M observed):")
    for cfg, cov in result.bitstring_coverage.items():
        print(f"  {cfg}: {cov:.2f}")

    # Coverage check: the four-config sweep's uniform-support argument was
    # derived for single-loop TBI. For multi-loop we verify empirically.
    if min(result.bitstring_coverage.values()) < 0.25:
        print(
            "WARNING: at least one configuration saw less than 25% of all "
            "bit strings. Consider tightening loop_lengths or using more "
            "shots before relying on convergence."
        )
