# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Calibration on a real chemistry landscape.

The synthetic bowls in ``_landscapes.py`` have a curvature scale the test picks,
which is exactly what calibration exists to discover — so the case for it has to
be made against a loss nobody tuned. H2/sto-3g is small enough to run in the
suite and has an exact reference energy from the Hamiltonian's own spectrum.
"""

import warnings

import numpy as np
import pytest
from qiskit.circuit.library import RYGate, RZGate

from divi.qprog import VQE
from divi.qprog.algorithms import GenericLayerAnsatz
from divi.qprog.optimizers import SPSAOptimizer
from divi.qprog.problems import MolecularProblem

pytest.importorskip("pyscf", reason="H2 reference needs the chem extra")

#: Upper end of "tens of milli-Hartree", in Hartree.
TENS_OF_MILLIHARTREE = 0.1


@pytest.fixture(scope="module")
def h2_molecule():
    from pyscf import gto

    return gto.M(atom="H 0 0 -0.6614; H 0 0 0.6614", basis="sto-3g", unit="Bohr")


def _run_vqe(backend, molecule, optimizer, ansatz, n_layers, seed, max_iterations=40):
    """Energy error against the exact ground state on a shot-sampling backend.

    The sampling RNG belongs to the backend, so it is seeded there; seeding only
    the program leaves the run irreproducible.
    """
    backend.set_seed(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vqe = VQE(
            MolecularProblem.from_molecule(molecule),
            ansatz=ansatz,
            n_layers=n_layers,
            optimizer=optimizer,
            max_iterations=max_iterations,
            backend=backend,
            seed=seed,
        )
        vqe.run()
    exact = float(np.linalg.eigvalsh(vqe.cost_hamiltonian.to_matrix())[0]) + float(
        vqe.loss_constant
    )
    return float(vqe.best_loss) - exact


def _median_error(backend, molecule, factory, seeds=8):
    """Median energy error over ``seeds`` runs of the 16-parameter ansatz."""
    errors = [
        _run_vqe(
            backend,
            molecule,
            factory(),
            GenericLayerAnsatz([RYGate, RZGate]),
            n_layers=2,
            seed=1000 + seed,
        )
        for seed in range(seeds)
    ]
    return float(np.median(errors))


def test_calibrated_spsa_beats_a_fixed_gain_under_shot_noise(
    sampling_test_simulator, h2_molecule
):
    """A fixed gain cannot match a loss scale it was not chosen for and lands
    hundreds of milli-Hartree out; calibrating against the loss reaches tens.

    Both the margin and the absolute number are asserted — a regression that made
    both runs equally bad would otherwise pass the ratio.
    """
    fixed = _median_error(
        sampling_test_simulator,
        h2_molecule,
        lambda: SPSAOptimizer(learning_rate=0.2, c=0.2),
    )
    calibrated = _median_error(
        sampling_test_simulator, h2_molecule, lambda: SPSAOptimizer(c=0.2)
    )

    assert calibrated < 0.3 * fixed
    assert calibrated < TENS_OF_MILLIHARTREE
