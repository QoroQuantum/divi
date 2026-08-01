# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""LASSQD on a stretched H4 chain.

:class:`~divi.qprog.workflows.LASSQD` estimates a molecule's ground-state
energy by splitting its active space into fragments, running one VQE per
fragment, recovering each fragment's ground state from the sampled bitstring
distribution via sample-based quantum diagonalization (SQD), reassembling
the fragment reduced density matrices (RDMs) into an active-space RDM, and
re-optimizing the molecular orbitals against it. One round of
``LASSQD.run()`` is one full pass through this cycle (a "macro-cycle");
``run()`` repeats it until the total energy converges (or ``max_rounds`` is
reached).

This tutorial builds a linear H4 chain and fragments its active space into
two fragments of two orbitals and two electrons each: ``(0, 1)`` (the two
occupied canonical RHF MOs) and ``(2, 3)`` (the two virtual MOs) — a split
along orbital occupancy, not spatial position. Splitting the active space
this way is an approximation, not a shortcut to the same answer as an
unfragmented calculation: reassembling the fragments' RDMs zeroes every
cross-fragment 2-RDM block, so the reported energy is **not** comparable to
a CASCI/FCI reference on the same active space, and can fall substantially
below it. The gap measured here (about 1.47 Hartree) mostly comes from each
macro-cycle re-optimizing the orbitals against that non-N-representable
RDM, compounding round over round until convergence — not from the zeroed
blocks alone (which account for only about 0.37 Ha at a fixed orbital
basis). A single fragment spanning the *whole* active space has no
cross-fragment blocks to zero and is variational, matching FCI to about
``2e-16`` Ha on H2/STO-3G with a large enough sampling budget — see the
user guide's
`Localized Active-Space SQD (LASSQD)
<https://divi.readthedocs.io/en/latest/user_guide/localized_active_space_sqd.html>`_
page for that comparison and the full accuracy discussion.
"""

import time

from pyscf import gto, mcscf, scf

from divi.qprog import LASSQD, FragmentSpec
from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer
from tutorials._backend import get_backend


def main() -> None:
    # Linear H4, two well-separated H2 pairs: 4 spatial orbitals, 4 electrons.
    h4 = gto.M(
        atom="H 0 0 0; H 0 0 0.74; H 0 0 2.0; H 0 0 2.74",
        basis="sto-3g",
        verbose=0,
    )

    ensemble = LASSQD(
        h4,
        active_spaces=[
            FragmentSpec(orbitals=(0, 1), n_alpha=1, n_beta=1),
            FragmentSpec(orbitals=(2, 3), n_alpha=1, n_beta=1),
        ],
        optimizer=ScipyOptimizer(ScipyMethod.COBYLA),
        max_iterations=60,
        n_batches=6,
        batch_size=16,
        n_sqd_iterations=3,
        seed=7,
        backend=get_backend(shots=5000),
    )

    t0 = time.time()
    ensemble.run(max_rounds=5)
    elapsed = time.time() - t0

    mean_field = scf.RHF(h4).run(verbose=0)
    casci_energy = mcscf.CASCI(mean_field, 4, 4).kernel()[0]

    print(f"Stop reason: {ensemble.stop_reason}")
    print(f"Rounds run: {len(ensemble.round_history)}")
    print(f"LASSQD energy:  {ensemble.energy:.6f} Ha")
    print(f"CASCI energy:   {casci_energy:.6f} Ha")
    print(f"Difference:     {ensemble.energy - casci_energy:+.6f} Ha")
    print(f"Time taken: {elapsed:.1f}s")
    print(
        "\nThe difference above is expected to be large and negative — see this "
        "file's module docstring for why a multi-fragment LASSQD energy is not "
        "comparable to CASCI."
    )


if __name__ == "__main__":
    main()
