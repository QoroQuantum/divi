# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for LASSQD's continuum solvent field."""

import numpy as np
import pytest
from pyscf import gto, mcscf, scf, solvent

from divi.qprog import LASSQD, WorkflowStatus
from divi.qprog.problems import MolecularProblem
from divi.qprog.workflows.lassqd import FixedOrbitals, SecondOrderOrbitalUpdate
from divi.qprog.workflows.lassqd._state import FragmentSpec
from tests._helpers import exact_match
from tests.qprog.workflows.lassqd._helpers import (  # noqa: F401
    _solvated_h2o_lassqd,
    build_exact_sampler_lassqd,
    exported_h2o_states,
    h2o_molecule,
    lassqd_kwargs,
)


def test_a_solvent_field_converges_to_ddcosmo_casci(dummy_expval_backend, mocker):
    ensemble, mean_field = _solvated_h2o_lassqd(
        dummy_expval_backend, mocker, FixedOrbitals(), {1: 2}
    )
    reference = solvent.ddCOSMO(mcscf.CASCI(mean_field, 4, 4))
    reference.with_solvent.conv_tol = 1e-12
    reference.verbose = 0
    reference.kernel(mean_field.mo_coeff)

    ensemble.run(max_rounds=30)

    assert ensemble.stop_reason is WorkflowStatus.COMPLETE
    assert ensemble.electronic_states[(1, 0)].energy == pytest.approx(
        reference.e_tot, abs=1e-8
    )


@pytest.mark.e2e
def test_a_solvent_field_converges_with_the_orbitals_to_self_consistent_casscf(
    dummy_expval_backend, mocker
):
    """PySCF's own ddCOSMO-CASSCF lags its field behind its density, so the
    reference is CASSCF iterated to self-consistency with the field."""
    ensemble, mean_field = _solvated_h2o_lassqd(
        dummy_expval_backend, mocker, SecondOrderOrbitalUpdate(), {1: 1}
    )
    model = solvent.ddCOSMO(mean_field.mol)
    hcore = mean_field.get_hcore()
    field_energy, potential, mo_coeff = 0.0, np.zeros_like(hcore), mean_field.mo_coeff
    for _ in range(30):
        fielded = scf.RHF(mean_field.mol)
        fielded.get_hcore = lambda *args, potential=potential: hcore + potential
        casscf = mcscf.CASSCF(fielded, 4, 4)
        casscf.conv_tol = 1e-12
        casscf.verbose = 0
        casscf.kernel(mo_coeff)
        mo_coeff, density = casscf.mo_coeff, casscf.make_rdm1()
        reference = (
            casscf.e_tot - np.einsum("ij,ji->", potential, density) + field_energy
        )
        field_energy, potential = model.kernel(density)

    ensemble.run(max_rounds=40)

    assert ensemble.stop_reason is WorkflowStatus.COMPLETE
    assert ensemble.energy == pytest.approx(reference, abs=1e-7)


def test_nonequilibrium_corrections_are_the_fast_response_to_each_density_change(
    dummy_expval_backend, mocker
):
    ensemble, mean_field = _solvated_h2o_lassqd(
        dummy_expval_backend, mocker, FixedOrbitals(), {1: 3}
    )
    ensemble.run(max_rounds=30)
    fast = solvent.ddCOSMO(mean_field.mol)
    fast.eps = 1.78
    fast.build()
    densities = {
        label: 2 * w.core_orbitals @ w.core_orbitals.T
        + w.active_orbitals @ w.rdm1 @ w.active_orbitals.T
        for label, w in ensemble.state_wavefunctions().items()
    }
    ground = next(iter(densities.values()))

    corrections = ensemble.nonequilibrium_corrections()

    assert list(corrections) == list(ensemble.electronic_states)
    for label, density in densities.items():
        change = density - ground
        assert corrections[label] == pytest.approx(
            0.5 * np.einsum("ij,ji->", fast._B_dot_x(change), change), abs=1e-10
        )
    assert all(correction <= 0 for correction in corrections.values())
    assert all(
        correction == pytest.approx(0, abs=1e-12)
        for correction in ensemble.nonequilibrium_corrections(optical_eps=1).values()
    )


def test_nonequilibrium_corrections_need_a_solvent(exported_h2o_states):
    ensemble, _ = exported_h2o_states
    message = (
        "nonequilibrium_corrections needs LASSQD built with a solvent, e.g. "
        "solvent=pyscf.solvent.ddCOSMO(mol)."
    )

    with pytest.raises(RuntimeError, match=exact_match(message)):
        ensemble.nonequilibrium_corrections()


def _h2o_ddcosmo(eps):
    model = solvent.ddCOSMO(h2o_molecule())
    model.eps = eps
    return model


def _unusable_eps(value):
    return (
        "solvent's dielectric constant must be a finite number of at least 1; "
        f"got {value!r}. Set solvent.eps, or for SMD its solvent, e.g. "
        "smd.solvent = 'water'."
    )


@pytest.mark.parametrize(
    "model, error, message",
    [
        pytest.param(
            "water",
            TypeError,
            "solvent must be a PySCF ddCOSMO, ddPCM, PCM or SMD model, such as "
            "pyscf.solvent.ddCOSMO(mol); got str.",
            id="not-a-model",
        ),
        pytest.param(
            _h2o_ddcosmo(0.5), ValueError, _unusable_eps(0.5), id="eps-below-1"
        ),
        pytest.param(
            _h2o_ddcosmo(float("nan")),
            ValueError,
            _unusable_eps(float("nan")),
            id="eps-nan",
        ),
        pytest.param(
            solvent.SMD(h2o_molecule()), ValueError, _unusable_eps(0), id="smd-unset"
        ),
        pytest.param(
            solvent.ddCOSMO(gto.M(atom="H 0 0 0; H 0 0 0.74", verbose=0)),
            ValueError,
            "solvent was built on a different molecule; build it on the molecule "
            "the problem was made from.",
            id="other-molecule",
        ),
    ],
)
def test_lassqd_rejects_an_unusable_solvent(
    dummy_expval_backend, model, error, message
):
    with pytest.raises(error, match=exact_match(message)):
        LASSQD(
            MolecularProblem.from_molecule(h2o_molecule()),
            backend=dummy_expval_backend,
            solvent=model,
            **lassqd_kwargs(
                active_spaces=[FragmentSpec(orbitals=(3, 4, 5, 6), n_alpha=2, n_beta=2)]
            ),
        )


@pytest.mark.parametrize(
    "optical_eps, message",
    [
        (
            True,
            "optical_eps must be a finite number of at least 1; got True.",
        ),
        (
            "1.78",
            "optical_eps must be a finite number of at least 1; got '1.78'.",
        ),
        (
            100.0,
            "optical_eps=100.0 exceeds the solvent's static eps=78.3553: the fast "
            "response cannot outweigh the full one. Use the solvent's refractive "
            "index squared.",
        ),
    ],
    ids=["bool", "string", "above-static"],
)
def test_nonequilibrium_corrections_reject_an_unphysical_optical_eps(
    dummy_expval_backend, mocker, optical_eps, message
):
    ensemble, _ = _solvated_h2o_lassqd(
        dummy_expval_backend, mocker, FixedOrbitals(), {1: 2}
    )
    ensemble.run(max_rounds=30)

    with pytest.raises(ValueError, match=exact_match(message)):
        ensemble.nonequilibrium_corrections(optical_eps=optical_eps)


def test_a_solvated_run_reports_its_converged_energy_as_best_and_leaves_the_model(
    dummy_expval_backend, mocker
):
    """Early rounds sit in a field not yet consistent with their density, so
    they are no better than the converged one."""
    model = solvent.ddCOSMO(h2o_molecule())
    ensemble, _ = build_exact_sampler_lassqd(
        dummy_expval_backend,
        mocker,
        problem=MolecularProblem.from_molecule(h2o_molecule()),
        active_spaces=[FragmentSpec(orbitals=(3, 4, 5, 6), n_alpha=2, n_beta=2)],
        extension="doubles",
        n_batches=1,
        n_recovery_iterations=1,
        energy_tol=1e-10,
        solvent=model,
    )

    ensemble.run(max_rounds=30)

    assert ensemble.best_energy == ensemble.energy
    assert model.e is None and model.v is None


def test_a_mean_field_solved_in_a_solvent_warns_without_one(dummy_expval_backend):
    mean_field = solvent.ddCOSMO(scf.RHF(h2o_molecule())).run(verbose=0)
    message = (
        "The mean field passed in was solved in a solvent, but LASSQD has none, so "
        "it runs in gas phase from those orbitals. Pass "
        "solvent=mean_field.with_solvent to solvate it."
    )

    with pytest.warns(UserWarning, match=exact_match(message)):
        LASSQD(
            MolecularProblem.from_molecule(mean_field),
            backend=dummy_expval_backend,
            **lassqd_kwargs(
                active_spaces=[FragmentSpec(orbitals=(3, 4, 5, 6), n_alpha=2, n_beta=2)]
            ),
        )
