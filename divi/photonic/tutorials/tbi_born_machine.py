# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Boson Sampling Born Machine on a multi-loop TBI: bars and stripes.

Trains a multi-loop photonic TBI as a generative model — the parity-mapped
TBI output distribution is pushed (via a classical MMD loss) to match a
target over bit strings. The target here is the 3×3 bars-and-stripes
dataset, the canonical QCBM / BSBM toy problem: 12 target patterns in a
2^9 = 512-bitstring search space, so the random baseline is ~2.3% and a
trained model is easy to distinguish from uniform.

This follows the "train classically, deploy quantumly" paradigm: MMD is
evaluated on classical-simulator samples during training, and the same
beamsplitter angles ``θ`` would drive a physical Orca PT-series chip at
inference via a ``RemoteTBISampler`` swap.
"""

import time

from divi.photonic import (
    BarsAndStripes,
    SimulatedTBISampler,
    TBIBornMachine,
)

if __name__ == "__main__":
    # 3x3 bars-and-stripes: 12 patterns in a 512-bitstring search space.
    # Random baseline: ~2.3%. Canonical BSBM / QCBM toy.
    dataset = BarsAndStripes(n=3)
    baseline = len(dataset.patterns) / (2**dataset.n_bits)
    print(
        f"Dataset: {len(dataset.patterns)} patterns of length {dataset.n_bits} "
        f"(random baseline: {baseline:.1%})"
    )

    # Multi-loop TBI: 9 modes, nested loops (1, 2).
    sampler = SimulatedTBISampler(seed=0)
    bsbm = TBIBornMachine(
        n_bits=dataset.n_bits,
        target_bits=dataset.patterns,
        n_modes=dataset.n_bits,
        n_photons=3,
        loop_lengths=(1, 2),
        sampler=sampler,
    )

    t0 = time.time()
    result = bsbm.run(updates=20, shots=80, seed=0)
    elapsed = time.time() - t0

    print(
        f"\nMMD went from {result.history_mmd[0]:.3f} "
        f"to {result.history_mmd[-1]:.3f} "
        f"over {len(result.history_mmd) - 1} updates "
        f"({result.total_shots} total shots, {elapsed:.1f}s)."
    )

    # Empirical probability mass the trained model places on target patterns.
    target_set = {tuple(int(x) for x in p) for p in dataset.patterns}
    matches = sum(
        1
        for row in result.final_samples_bits
        if tuple(int(x) for x in row) in target_set
    )
    fraction = matches / len(result.final_samples_bits)
    print(
        f"Fraction of final model samples on target patterns: "
        f"{fraction:.1%} (random baseline: {baseline:.1%}, "
        f"uplift: {fraction / baseline:.1f}x)"
    )
