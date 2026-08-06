# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""LASSQD on a stretched H4 chain, and why the fragmentation matters.

:class:`~divi.qprog.workflows.LASSQD` estimates a molecule's ground-state
energy by splitting its active space into fragments, running one VQE per
fragment, recovering each fragment's ground state from the sampled bitstring
distribution via sample-based quantum diagonalization (SQD), reassembling the
fragment reduced density matrices (RDMs) into an active-space RDM, and
re-optimizing the molecular orbitals against it. One round of ``LASSQD.run()``
is one such macro-cycle; ``run()`` repeats until the energy converges or
``max_rounds`` is reached.

The reassembled RDM is that of a *product* of fragment states, so the energy is
a genuine expectation value and sits **above** a CASCI/FCI reference on the same
active space. What fragmenting costs is the correlation *between* fragments,
which a product state cannot represent -- so the split you choose is the main
thing determining accuracy.

This tutorial makes that concrete on a linear H4 chain built as two
well-separated H2 pairs, comparing two fragmentations of the same 4-orbital
active space:

* **Automatic** -- localize the frontier orbitals and cut along the weakest
  coupling, so the correlation dropped is the weak inter-pair correlation.
  Fragment indices it reports are positions in the localized basis, not
  canonical MO labels.
* **By orbital occupancy** -- one fragment of the occupied MOs ``(0, 1)`` and
  one of the virtuals ``(2, 3)``. Every H2 unit straddles both fragments, so
  this cuts through the correlation that matters.

Both give two fragments over the same active space, and both are valid upper
bounds -- but the automatic split recovers most of the correlation energy where
the occupancy split recovers a fraction of it. See the
user guide's `Localized Active-Space SQD (LASSQD)
<https://divi.readthedocs.io/en/latest/user_guide/localized_active_space_sqd.html>`_
page for the accuracy discussion and for tuning the sampling budget.
"""

import time

from pyscf import gto, mcscf, scf

from divi.qprog import (
    LASSQD,
    FragmentationConfig,
    FragmentSpec,
    ReportingLevel,
    SQDConfig,
)
from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer
from tutorials._backend import get_backend


def main() -> None:
    # Linear H4, two well-separated H2 pairs: 4 spatial orbitals, 4 electrons.
    h4 = gto.M(
        atom="H 0 0 0; H 0 0 0.74; H 0 0 2.0; H 0 0 2.74",
        basis="sto-3g",
        verbose=0,
    )

    mean_field = scf.RHF(h4).run(verbose=0)
    casci_energy = mcscf.CASCI(mean_field, 4, 4).kernel()[0]
    correlation = casci_energy - mean_field.e_tot

    settings = dict(
        optimizer=ScipyOptimizer(ScipyMethod.COBYLA),
        sqd=SQDConfig(n_batches=6, batch_size=32, n_recovery_iterations=3),
        max_iterations=60,
        seed=7,
        backend=get_backend(shots=5000),
        reporting_level=ReportingLevel.OFF,
    )

    layouts = {
        "automatic (cuts along weakest coupling)": FragmentationConfig(
            n_active_orbitals=4, max_orbitals_per_fragment=2
        ),
        "by occupancy (cuts through both H2 units)": FragmentationConfig(
            active_spaces=[
                FragmentSpec(orbitals=(0, 1), n_alpha=1, n_beta=1),
                FragmentSpec(orbitals=(2, 3), n_alpha=1, n_beta=1),
            ]
        ),
    }

    print(f"RHF:   {mean_field.e_tot:.6f} Ha")
    print(f"CASCI: {casci_energy:.6f} Ha  (correlation {correlation:+.6f} Ha)\n")

    for label, fragmentation in layouts.items():
        ensemble = LASSQD(h4, fragmentation=fragmentation, **settings)
        start = time.time()
        ensemble.run(max_rounds=5)
        elapsed = time.time() - start

        # best_energy, not energy: the macro-cycle need not be monotone, and
        # every round is a valid upper bound.
        gap = ensemble.best_energy - casci_energy
        recovered = (ensemble.best_energy - mean_field.e_tot) / correlation
        print(f"{label}")
        print(
            f"  fragments:   {[f.spec.orbitals for f in ensemble.workflow_state.fragments]}"
        )
        print(f"  energy:      {ensemble.best_energy:.6f} Ha")
        print(f"  above CASCI: {gap:+.6f} Ha")
        print(f"  correlation recovered: {recovered:.0%}")
        print(f"  {len(ensemble.round_history)} rounds, {elapsed:.1f}s\n")

    print(
        "Both energies are upper bounds, so both sit above CASCI. The gap is the "
        "inter-fragment correlation each split discards -- which is why cutting "
        "along a weak interaction beats cutting through a bond. The occupancy "
        "split also warns that a fragment collapsed to one determinant, which is "
        "the same problem showing up in the sampling."
    )


if __name__ == "__main__":
    main()
