# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for CCSD-seeded LUCJ circuit preparation."""

import hashlib
import itertools
from pathlib import Path

import numpy as np
import pytest
from qiskit import ClassicalRegister, QuantumCircuit

ffsim = pytest.importorskip("ffsim")

from pyscf import cc, gto, lib, scf

from divi.circuits._conversions import lower_to_basis_gates
from divi.qprog.ensembles.lassqd import _preparation as preparation
from divi.qprog.ensembles.lassqd._preparation import (
    LUCJFragmentProgram,
    LUCJPreparation,
    _solve_fragment_rohf,
    build_lucj_circuit,
    build_lucj_interaction_pairs,
    prepare_lucj,
    prepare_lucj_fragment,
    rotate_rdms_to_fragment_basis,
    run_fragment_coupled_cluster,
)
from divi.qprog.ensembles.lassqd._state import FragmentSpec
from divi.qprog.problems import MolecularProblem
from tests._helpers import exact_match
from tests.qprog.ensembles.lassqd._helpers import (
    dense_fci_energy,
    fragment_problem,
    h2o_molecule,
)

_PREPARATION = "divi.qprog.ensembles.lassqd._preparation"


def _patch_linear_method(mocker, optimum):
    """Stub the linear method to return ``optimum(x0)`` as its parameters."""
    return mocker.patch(
        "ffsim.optimize.minimize_linear_method",
        side_effect=lambda _params_to_vec, _hamiltonian, x0, **_options: mocker.Mock(
            x=optimum(x0)
        ),
    )


def _patch_mean_field_and_ccsd(mocker, mo_coeff, t1, t2, mo_occ):
    """Stub the fragment ROHF and CCSD solves with fixed results."""
    mocker.patch(
        f"{_PREPARATION}._solve_fragment_rohf",
        return_value=mocker.Mock(
            mo_coeff=mo_coeff, mo_occ=mo_occ, mo_energy=np.zeros(len(mo_occ)), e_tot=0.0
        ),
    )
    mocker.patch(
        f"{_PREPARATION}.run_fragment_coupled_cluster",
        return_value=mocker.Mock(t1=t1, t2=t2, e_tot=0.0),
    )


def _one_pair_amplitudes(mixed_double=0.0):
    """CCSD amplitudes for one occupied and one virtual orbital per spin."""
    t1 = (np.zeros((1, 1)), np.zeros((1, 1)))
    t2 = (
        np.zeros((1, 1, 1, 1)),
        np.full((1, 1, 1, 1), mixed_double),
        np.zeros((1, 1, 1, 1)),
    )
    return t1, t2


def _two_orbital_problem(spec):
    """A two-orbital fragment problem with distinct alpha and beta one-body terms."""
    return fragment_problem(
        np.diag([-1.0, 0.5]), np.diag([-0.8, 0.7]), np.zeros((2, 2, 2, 2)), spec
    )


@pytest.mark.parametrize(
    "interaction, opposite_spin",
    [
        ("heavy-hex", [(0, 0), (4, 4)]),
        ("all", [(p, p) for p in range(6)]),
    ],
    ids=["heavy-hex", "all"],
)
def test_lucj_interaction_pairs_match_the_reference_topology(
    interaction, opposite_spin
):
    neighbours = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]

    assert build_lucj_interaction_pairs(6, interaction) == (
        neighbours,
        opposite_spin,
        neighbours,
    )


def test_lucj_interaction_pairs_reject_an_unknown_topology():
    with pytest.raises(
        ValueError,
        match=exact_match("interaction must be 'heavy-hex' or 'all'; got 'heavyhex'."),
    ):
        build_lucj_interaction_pairs(6, "heavyhex")


def _identity_lucj_operator(norb):
    """The heavy-hex LUCJ operator at zero parameters, i.e. the identity."""
    pairs = build_lucj_interaction_pairs(norb)
    n_params = ffsim.UCJOpSpinUnbalanced.n_params(
        norb,
        1,
        interaction_pairs=pairs,
        with_final_orbital_rotation=True,
    )
    return ffsim.UCJOpSpinUnbalanced.from_parameters(
        np.zeros(n_params),
        norb=norb,
        n_reps=1,
        interaction_pairs=pairs,
        with_final_orbital_rotation=True,
    )


def _reference_probabilities(simulator, circuit):
    probabilities = np.round(np.abs(simulator.statevector(circuit)) ** 2, 12)
    return {
        format(index, f"0{circuit.num_qubits}b"): float(probabilities[index])
        for index in np.flatnonzero(probabilities)
    }


@pytest.mark.parametrize(
    "nelec, expected",
    [
        pytest.param((2, 0), "000101", id="alpha-only"),
        pytest.param((2, 1), "000111", id="beta-on-odd-wires"),
    ],
)
def test_lucj_circuit_maps_grouped_spin_gates_to_interleaved_wires(
    nelec, expected, default_test_simulator
):
    """Qubit ``2p`` holds orbital ``p``'s alpha electron and ``2p + 1`` its beta
    one; the key is little-endian, so qubit 0 is the rightmost bit."""
    norb = 3

    circuit = build_lucj_circuit(_identity_lucj_operator(norb), norb, nelec)

    assert _reference_probabilities(default_test_simulator, circuit) == {expected: 1.0}
    assert circuit.num_clbits == 2 * norb
    assert {instruction.operation.name for instruction in circuit.data} <= {
        "cx",
        "measure",
        "u",
        "x",
    }


def _unmerged_lucj_circuit(gate, norb, nelec):
    """The LUCJ circuit lowered gate by gate, without merging its rotations."""
    circuit = QuantumCircuit(2 * norb)
    wires = [
        *(circuit.qubits[2 * p] for p in range(norb)),
        *(circuit.qubits[2 * p + 1] for p in range(norb)),
    ]
    circuit.append(ffsim.qiskit.PrepareHartreeFockJW(norb, nelec), wires)
    circuit.append(gate, wires)
    return lower_to_basis_gates(circuit, ("x", "u", "cx"))


@pytest.mark.parametrize(
    "operator_type, gate_type, n_pair_lists, nelec",
    [
        (ffsim.UCJOpSpinUnbalanced, ffsim.qiskit.UCJOpSpinUnbalancedJW, 3, (3, 1)),
        (ffsim.UCJOpSpinBalanced, ffsim.qiskit.UCJOpSpinBalancedJW, 2, (2, 2)),
    ],
    ids=["spin-unbalanced", "spin-balanced"],
)
def test_lucj_circuit_merges_orbital_rotations_without_changing_the_state(
    default_test_simulator, operator_type, gate_type, n_pair_lists, nelec
):
    norb = 4
    pairs = build_lucj_interaction_pairs(norb)[:n_pair_lists]
    n_params = operator_type.n_params(
        norb, 1, interaction_pairs=pairs, with_final_orbital_rotation=True
    )
    operator = operator_type.from_parameters(
        np.random.default_rng(0).normal(scale=0.3, size=n_params),
        norb=norb,
        n_reps=1,
        interaction_pairs=pairs,
        with_final_orbital_rotation=True,
    )
    unmerged = _unmerged_lucj_circuit(gate_type(operator), norb, nelec)

    circuit = build_lucj_circuit(operator, norb, nelec)

    np.testing.assert_allclose(
        np.abs(
            default_test_simulator.statevector(
                circuit.remove_final_measurements(inplace=False)
            )
        )
        ** 2,
        np.abs(default_test_simulator.statevector(unmerged)) ** 2,
        atol=1e-12,
    )
    assert circuit.count_ops()["cx"] < unmerged.count_ops()["cx"]


def test_prepare_lucj_fragment_uses_ccsd_seed_and_linear_method(mocker):
    spec = FragmentSpec(orbitals=(0, 1), n_alpha=1, n_beta=1)

    minimize = _patch_linear_method(mocker, lambda x0: x0)
    from_amplitudes = mocker.spy(ffsim.UCJOpSpinUnbalanced, "from_t_amplitudes")

    result = prepare_lucj_fragment(_two_orbital_problem(spec), spec)

    assert from_amplitudes.call_count == 1
    call = from_amplitudes.call_args
    assert isinstance(call.args[0], tuple)
    assert isinstance(call.kwargs["t1"], tuple)
    assert call.kwargs["n_reps"] == 1
    assert call.kwargs["interaction_pairs"] == build_lucj_interaction_pairs(2)
    assert call.kwargs["optimize"] is True
    assert minimize.call_count == 1
    np.testing.assert_allclose(result.params, minimize.call_args.kwargs["x0"])
    assert result.circuit.num_qubits == 4
    assert result.orbital_rotation.shape == (2, 2)
    assert result.problem.one_body.shape == (2, 2)
    np.testing.assert_allclose(result.problem.one_body_beta, np.diag([-0.8, 0.7]))
    assert result.problem.two_body.shape == (2, 2, 2, 2)


def test_prepare_lucj_fragment_samples_the_ccsd_seed_without_the_linear_method(
    mocker,
):
    minimize = _patch_linear_method(mocker, lambda x0: x0 + 1.0)
    seed = mocker.spy(ffsim.UCJOpSpinUnbalanced, "to_parameters")
    stages = []
    spec = FragmentSpec(orbitals=(0, 1), n_alpha=1, n_beta=1)

    result = prepare_lucj_fragment(
        _two_orbital_problem(spec),
        spec,
        report=stages.append,
        run_linear_method=False,
    )

    minimize.assert_not_called()
    np.testing.assert_array_equal(result.params, seed.spy_return)
    assert stages == ["Fragment ROHF", "CCSD seed"]


def _antisymmetric_t2(rng):
    t2 = rng.normal(scale=0.05, size=(2, 2, 2, 2))
    t2 = t2 - t2.transpose(1, 0, 2, 3)
    return t2 - t2.transpose(0, 1, 3, 2)


def test_prepare_lucj_fragment_refits_nonzero_seed_to_the_paper_topology(mocker):
    rng = np.random.default_rng(17)
    t1 = (
        rng.normal(scale=0.05, size=(2, 2)),
        rng.normal(scale=0.05, size=(2, 2)),
    )
    t2 = (
        _antisymmetric_t2(rng),
        rng.normal(scale=0.05, size=(2, 2, 2, 2)),
        _antisymmetric_t2(rng),
    )
    spec = FragmentSpec(orbitals=(0, 1, 2, 3), n_alpha=2, n_beta=2)
    pairs = build_lucj_interaction_pairs(4)
    dense_masked = ffsim.UCJOpSpinUnbalanced.from_t_amplitudes(
        t2,
        t1=t1,
        n_reps=1,
    ).to_parameters(interaction_pairs=pairs)
    _patch_mean_field_and_ccsd(
        mocker, np.eye(4), t1, t2, mo_occ=np.array([2.0, 2.0, 0.0, 0.0])
    )
    _patch_linear_method(mocker, lambda x0: x0)

    result = prepare_lucj_fragment(
        fragment_problem(np.zeros((4, 4)), np.zeros((4, 4)), np.zeros((4,) * 4), spec),
        spec,
    )

    assert np.max(np.abs(result.params - dense_masked)) > 1e-3


def test_prepare_lucj_fragment_rotates_real_beta_integrals_for_sqd(mocker):
    rotation = np.array([[0.0, -1.0], [1.0, 0.0]])
    _patch_mean_field_and_ccsd(
        mocker, rotation, *_one_pair_amplitudes(), mo_occ=np.array([2.0, 0.0])
    )
    _patch_linear_method(mocker, lambda x0: x0)
    spec = FragmentSpec(orbitals=(0, 1), n_alpha=1, n_beta=1)

    result = prepare_lucj_fragment(_two_orbital_problem(spec), spec)

    np.testing.assert_allclose(result.problem.one_body_beta, np.diag([0.7, -0.8]))


@pytest.fixture(scope="module")
def water_active_space():
    """Water's (4e,4o) CASCI active space."""
    mean_field = scf.RHF(h2o_molecule(basis="sto-3g")).run(verbose=0)
    return MolecularProblem.from_casci(mean_field, 4, 4)


def _with_electrons(problem, n_alpha, n_beta, **kwargs):
    return MolecularProblem(
        problem.one_body,
        problem.two_body,
        n_alpha=n_alpha,
        n_beta=n_beta,
        constant=problem.constant,
        **kwargs,
    )


def _state_energy(problem, operator=None):
    """Energy on ``problem`` of its reference determinant, after ``operator``."""
    norb, nelec = problem.n_orbitals, (problem.n_alpha, problem.n_beta)
    state = ffsim.hartree_fock_state(norb, nelec)
    if operator is not None:
        state = ffsim.apply_unitary(state, operator, norb=norb, nelec=nelec)
    hamiltonian = ffsim.linear_operator(
        ffsim.MolecularHamiltonian(
            problem.one_body, problem.two_body, problem.constant
        ),
        norb=norb,
        nelec=nelec,
    )
    return np.vdot(state, hamiltonian @ state).real


@pytest.mark.parametrize(
    "molecule, mean_field_type, coupled_cluster",
    [
        (h2o_molecule(basis="sto-3g"), scf.RHF, cc.CCSD),
        (
            gto.M(atom="N 0 0 0; H 0 0 1.036", basis="sto-3g", spin=2, verbose=0),
            scf.ROHF,
            lambda rohf: cc.UCCSD(rohf.to_uhf()),
        ),
    ],
    ids=["closed-shell", "triplet"],
)
def test_prepare_lucj_matches_the_molecular_mean_field_and_ccsd_with_every_orbital_active(
    molecule, mean_field_type, coupled_cluster
):
    """With every orbital active, the active-space mean field and CCSD are the
    molecule's own."""
    mean_field = mean_field_type(molecule).run(verbose=0)
    ccsd = coupled_cluster(mean_field).run(verbose=0)

    result = prepare_lucj(
        MolecularProblem.from_casci(mean_field, molecule.nao, molecule.nelec)
    )

    assert result.mean_field_energy == pytest.approx(mean_field.e_tot, abs=1e-8)
    assert result.ccsd_energy == pytest.approx(ccsd.e_tot, abs=1e-6)
    assert result.ccsd_converged


def test_prepare_lucj_reports_an_unconverged_ccsd(water_active_space, mocker):
    mocker.patch(f"{_PREPARATION}._CC_MAX_CYCLE", 1)

    with pytest.warns(UserWarning, match="CCSD did not converge"):
        result = prepare_lucj(water_active_space)

    assert not result.ccsd_converged


@pytest.mark.parametrize("n_alpha, n_beta", [(2, 2), (3, 1)])
def test_the_lucj_reference_is_the_mean_field_determinant(
    water_active_space, n_alpha, n_beta
):
    """The circuit starts from the first ``n_alpha`` / ``n_beta`` orbitals of
    ``result.problem``, so that determinant must carry the mean-field energy."""
    result = prepare_lucj(_with_electrons(water_active_space, n_alpha, n_beta))

    assert _state_energy(result.problem) == pytest.approx(
        result.mean_field_energy, abs=1e-9
    )


def test_the_linear_method_lowers_the_spin_balanced_seed_energy(water_active_space):
    pairs = build_lucj_interaction_pairs(4)[:2]

    def prepared_energy(result):
        operator = ffsim.UCJOpSpinBalanced.from_parameters(
            result.params,
            norb=4,
            n_reps=1,
            interaction_pairs=pairs,
            with_final_orbital_rotation=True,
        )
        return _state_energy(result.problem, operator)

    seed = prepare_lucj(water_active_space)
    optimised = prepare_lucj(water_active_space, run_linear_method=True)

    exact = dense_fci_energy(
        water_active_space.one_body,
        water_active_space.two_body,
        2,
        2,
        water_active_space.constant,
    )
    assert exact - 1e-9 <= prepared_energy(optimised)
    assert prepared_energy(optimised) < prepared_energy(seed) - 1e-4


def test_spin_balanced_preparation_reports_an_unconverged_rhf(
    water_active_space, mocker
):
    rohf = mocker.Mock(
        make_rdm1=mocker.Mock(return_value=np.zeros((2, 4, 4))),
        mo_occ=np.array([2.0, 2.0, 0.0, 0.0]),
        mo_energy=np.zeros(4),
    )
    mocker.patch(f"{_PREPARATION}._solve_fragment_rohf", return_value=rohf)
    unconverged = mocker.Mock(converged=False)
    unconverged.newton.return_value = mocker.Mock(converged=False)
    mocker.patch(f"{_PREPARATION}.run_fragment_mean_field", return_value=unconverged)

    with pytest.raises(
        RuntimeError,
        match=exact_match("RHF did not converge for fragment (0, 1, 2, 3)."),
    ):
        prepare_lucj(water_active_space)

    unconverged.newton.assert_called_once_with()


@pytest.mark.parametrize(
    "n_alpha, n_beta, beta_shift, options, operator_type, n_layers",
    [
        (2, 2, 0.0, {}, ffsim.UCJOpSpinBalanced, 1),
        (2, 2, 0.1, {}, ffsim.UCJOpSpinUnbalanced, 1),
        (2, 2, 0.0, {"spin_balanced": False}, ffsim.UCJOpSpinUnbalanced, 1),
        (3, 1, 0.0, {}, ffsim.UCJOpSpinUnbalanced, 1),
        (2, 2, 0.0, {"interaction": "all"}, ffsim.UCJOpSpinBalanced, 1),
        (2, 2, 0.0, {"n_reps": 2}, ffsim.UCJOpSpinBalanced, 2),
        (3, 1, 0.0, {"n_reps": (1, 1)}, ffsim.UCJOpSpinUnbalanced, 2),
    ],
    ids=[
        "closed-shell",
        "spin-dependent",
        "forced-unbalanced",
        "triplet",
        "all-pairs",
        "two-layers",
        "layer-pair",
    ],
)
def test_prepare_lucj_builds_the_requested_operator(
    water_active_space, n_alpha, n_beta, beta_shift, options, operator_type, n_layers
):
    problem = _with_electrons(
        water_active_space,
        n_alpha,
        n_beta,
        one_body_beta=water_active_space.one_body + beta_shift * np.eye(4),
    )
    n_pair_lists = 2 if operator_type is ffsim.UCJOpSpinBalanced else 3
    pairs = build_lucj_interaction_pairs(4, options.get("interaction", "heavy-hex"))

    result = prepare_lucj(problem, **options)

    assert len(result.params) == operator_type.n_params(
        4,
        n_layers,
        interaction_pairs=pairs[:n_pair_lists],
        with_final_orbital_rotation=True,
    )


def test_prepare_lucj_refuses_a_spin_balanced_open_shell(water_active_space):
    with pytest.raises(
        ValueError,
        match=exact_match(
            "spin_balanced=True needs equal alpha and beta electron counts and "
            "one-electron integrals."
        ),
    ):
        prepare_lucj(_with_electrons(water_active_space, 3, 1), spin_balanced=True)


@pytest.mark.parametrize(
    "n_alpha, n_beta", [(2, 2), (3, 1)], ids=["spin-balanced", "spin-unbalanced"]
)
def test_refitting_changes_the_seeded_parameters(water_active_space, n_alpha, n_beta):
    problem = _with_electrons(water_active_space, n_alpha, n_beta)

    refitted, truncated = (
        prepare_lucj(problem, refit=refit).params for refit in (True, False)
    )

    assert np.max(np.abs(refitted - truncated)) > 1e-3


def test_a_layer_pair_chooses_the_spin_channel_of_each_layer():
    """Water's (4e,6o) space, which has same-spin doubles on neighbouring
    orbitals, prepared spin-unbalanced with one layer of either kind."""
    mean_field = scf.RHF(h2o_molecule(basis="6-31g")).run(verbose=0)
    problem = MolecularProblem.from_casci(mean_field, 6, 4)
    pairs = build_lucj_interaction_pairs(6)

    def layer(n_reps):
        result = prepare_lucj(problem, n_reps=n_reps, spin_balanced=False)
        return ffsim.UCJOpSpinUnbalanced.from_parameters(
            result.params,
            norb=6,
            n_reps=1,
            interaction_pairs=pairs,
            with_final_orbital_rotation=True,
        ).diag_coulomb_mats[0]

    alpha_beta, same_spin = layer((1, 0)), layer((0, 1))

    assert np.abs(alpha_beta[1]).max() > 1e-6
    np.testing.assert_array_equal(same_spin[1], 0.0)
    assert np.abs(same_spin[[0, 2]]).max() > 1e-6


@pytest.mark.parametrize(
    "n_alpha, n_beta, options, message",
    [
        (2, 2, {"n_reps": 0}, "n_reps must be at least 1; got 0."),
        (
            3,
            1,
            {"n_reps": (0, 0)},
            "n_reps pair entries must be at least 0 and sum to at least 1; got (0, 0).",
        ),
        (
            3,
            1,
            {"n_reps": (-1, 2)},
            "n_reps pair entries must be at least 0 and sum to at least 1; "
            "got (-1, 2).",
        ),
        (
            2,
            2,
            {"n_reps": (1, 1)},
            "n_reps as an (alpha-beta, same-spin) pair needs the spin-unbalanced "
            "operator. Pass spin_balanced=False.",
        ),
        (
            3,
            1,
            {"n_reps": (1, 1, 1)},
            "n_reps must be an integer or a pair of integers; got (1, 1, 1).",
        ),
        (
            3,
            1,
            {"n_reps": (0.5, 0.5)},
            "n_reps must be an integer or a pair of integers; got (0.5, 0.5).",
        ),
        (
            2,
            2,
            {"n_reps": 1.5},
            "n_reps must be an integer or a pair of integers; got 1.5.",
        ),
        (
            3,
            1,
            {"n_reps": [1, 1]},
            "n_reps must be an integer or a pair of integers; got [1, 1].",
        ),
        (
            2,
            2,
            {"regularization": -0.5},
            "regularization must be at least 0; got -0.5.",
        ),
        (
            2,
            2,
            {"regularization": float("nan")},
            "regularization must be at least 0; got nan.",
        ),
        (
            2,
            2,
            {"regularization": 1e-3, "refit": False},
            "regularization needs refit=True.",
        ),
    ],
    ids=[
        "no-layers",
        "empty-pair",
        "negative-pair-entry",
        "pair-on-a-balanced-operator",
        "three-entries",
        "fractional-pair",
        "fractional-count",
        "list",
        "negative-regularization",
        "nan-regularization",
        "regularization-without-refit",
    ],
)
def test_prepare_lucj_rejects_invalid_settings_before_any_mean_field(
    water_active_space, mocker, n_alpha, n_beta, options, message
):
    solve = mocker.spy(preparation, "_solve_fragment_rohf")

    with pytest.raises(ValueError, match=exact_match(message)):
        prepare_lucj(_with_electrons(water_active_space, n_alpha, n_beta), **options)

    solve.assert_not_called()


@pytest.mark.parametrize(
    "n_alpha, n_beta", [(2, 2), (3, 1)], ids=["spin-balanced", "spin-unbalanced"]
)
def test_regularization_changes_the_refitted_parameters(
    water_active_space, n_alpha, n_beta
):
    problem = _with_electrons(water_active_space, n_alpha, n_beta)

    plain, regularised = (
        prepare_lucj(problem, regularization=weight).params for weight in (0.0, 0.1)
    )

    assert np.max(np.abs(plain - regularised)) > 1e-6


_EMPTY_BEFORE_SINGLY_OCCUPIED = [0, 1, 3, 2]


def _swap_orbitals(mean_field, order):
    """Permute an ROHF solution's orbitals, as PySCF would have listed them."""
    energies = mean_field.mo_energy
    mean_field.mo_coeff = mean_field.mo_coeff[:, order]
    mean_field.mo_occ = mean_field.mo_occ[order]
    mean_field.mo_energy = lib.tag_array(
        energies[order], mo_ea=energies.mo_ea[order], mo_eb=energies.mo_eb[order]
    )
    return mean_field


def test_reordering_by_occupation_permutes_every_orbital_array(water_active_space):
    problem = _with_electrons(water_active_space, 3, 1)
    spec = FragmentSpec(orbitals=(0, 1, 2, 3), n_alpha=3, n_beta=1)
    mean_field = _solve_fragment_rohf(problem.one_body, problem.two_body, spec)
    energies = mean_field.mo_energy
    expected = {
        "mo_coeff": mean_field.mo_coeff.copy(),
        "mo_occ": mean_field.mo_occ.copy(),
        "mo_energy": np.asarray(energies).copy(),
        "mo_ea": energies.mo_ea.copy(),
        "mo_eb": energies.mo_eb.copy(),
    }
    _swap_orbitals(mean_field, _EMPTY_BEFORE_SINGLY_OCCUPIED)

    preparation._order_orbitals_by_occupation(mean_field)

    reordered = mean_field.mo_energy
    for name, value in {
        "mo_coeff": mean_field.mo_coeff,
        "mo_occ": mean_field.mo_occ,
        "mo_energy": np.asarray(reordered),
        "mo_ea": reordered.mo_ea,
        "mo_eb": reordered.mo_eb,
    }.items():
        np.testing.assert_array_equal(value, expected[name], err_msg=name)


@pytest.mark.parametrize(
    "n_alpha, n_beta", [(3, 1), (1, 3)], ids=["alpha-majority", "beta-majority"]
)
def test_lucj_preparation_orders_the_rohf_orbitals_by_occupation(
    water_active_space, mocker, n_alpha, n_beta
):
    """An ROHF solution with an empty orbital before a singly occupied one still
    gives a circuit that starts from the mean-field determinant."""
    problem = _with_electrons(water_active_space, n_alpha, n_beta)
    in_order = prepare_lucj(problem)
    solve = preparation._solve_fragment_rohf
    mocker.patch(
        f"{_PREPARATION}._solve_fragment_rohf",
        side_effect=lambda *args: _swap_orbitals(
            solve(*args), _EMPTY_BEFORE_SINGLY_OCCUPIED
        ),
    )

    result = prepare_lucj(problem)

    assert _state_energy(result.problem) == pytest.approx(
        result.mean_field_energy, abs=1e-9
    )
    assert result.ccsd_energy == pytest.approx(in_order.ccsd_energy, abs=1e-8)


def test_fragment_rohf_reaches_the_ground_state_of_a_spin_polarised_iron_fragment():
    """An open-shell iron fragment's ROHF reaches its stable ground state."""
    data = np.load(Path(__file__).parent / "data" / "fefe_fragment_round1.npz")
    n_alpha, n_beta = (int(count) for count in data["nelec"])
    spec = FragmentSpec(orbitals=tuple(range(5)), n_alpha=n_alpha, n_beta=n_beta)

    mean_field = _solve_fragment_rohf(data["one_body"], data["two_body"], spec)

    _, stable = preparation.stability.rohf_internal(
        mean_field, with_symmetry=False, return_status=True
    )
    assert stable
    assert mean_field.e_tot == pytest.approx(-18.158417, abs=1e-6)


_ONE_START = np.array([np.diag([1.0, 0.0]), np.diag([1.0, 0.0])])


def _report_stable(mean_field, **_):
    return mean_field.mo_coeff, True


def _patch_rohf_starts(mocker, starts=(_ONE_START,), stability=_report_stable):
    """Run the fragment ROHF from ``starts``, its stability check answered by
    ``stability`` (every solution stable by default)."""
    mocker.patch(f"{_PREPARATION}._rohf_starts", return_value=list(starts))
    return mocker.patch(
        f"{_PREPARATION}.stability.rohf_internal", side_effect=stability
    )


@pytest.mark.parametrize("newton_converges", [True, False])
def test_fragment_rohf_retries_unconverged_scf_with_newton(mocker, newton_converges):
    direct_solver = mocker.Mock(converged=False)
    newton_solver = mocker.Mock(converged=newton_converges)
    direct_solver.newton.return_value = newton_solver
    rohf = mocker.patch("pyscf.scf.ROHF", return_value=direct_solver)
    _patch_rohf_starts(mocker)
    spec = FragmentSpec(orbitals=(3, 4), n_alpha=1, n_beta=1)

    def solve():
        return _solve_fragment_rohf(np.diag([-1.0, 0.5]), np.zeros((2, 2, 2, 2)), spec)

    if newton_converges:
        assert solve() is newton_solver
    else:
        with pytest.raises(RuntimeError, match=r"ROHF did not converge.*\(3, 4\)"):
            solve()

    rohf.assert_called_once()
    direct_solver.kernel.assert_called_once()
    assert direct_solver.kernel.call_args.kwargs["dm0"] is _ONE_START
    direct_solver.newton.assert_called_once_with()
    newton_solver.kernel.assert_called_once_with()


def test_rohf_starts_cover_every_representable_occupation_aufbau_first():
    spec = FragmentSpec(orbitals=(0, 1, 2), n_alpha=1, n_beta=2)

    starts = preparation._rohf_starts(np.diag([-1.0, -0.5, 0.5]), spec)

    # Two singly-or-doubly occupied orbitals of three, one of them doubly.
    assert len(starts) == 3 * 2
    np.testing.assert_allclose(
        np.abs(starts[0]), [np.diag([1.0, 1.0, 0.0]), np.diag([1.0, 0.0, 0.0])]
    )


def test_rohf_starts_are_a_fixed_sample_once_the_cap_binds():
    spec = FragmentSpec(orbitals=tuple(range(10)), n_alpha=6, n_beta=2)
    one_body = np.diag(np.arange(10.0))

    starts = preparation._rohf_starts(one_body, spec)

    assert len(starts) == preparation._ROHF_STARTS
    for again, start in zip(preparation._rohf_starts(one_body, spec), starts):
        np.testing.assert_array_equal(again, start)


def test_capped_rohf_starts_can_select_the_first_non_aufbau_occupation():
    spec = FragmentSpec(orbitals=tuple(range(7)), n_alpha=3, n_beta=1)
    occupations = (
        (singles, doubles)
        for singles in itertools.combinations(range(7), 3)
        for doubles in itertools.combinations(singles, 1)
    )
    next(occupations)
    singles, doubles = next(occupations)
    expected = np.array(
        [
            np.diag(np.isin(range(7), occupied).astype(float))
            for occupied in (singles, doubles)
        ]
    )

    starts = preparation._rohf_starts(np.diag(np.arange(7.0)), spec)

    assert any(np.array_equal(start, expected) for start in starts)


@pytest.mark.parametrize(
    "n_orbitals, n_alpha, n_beta",
    [(40, 20, 18), (70, 35, 35)],
    ids=["beyond-enumeration", "beyond-ssize_t"],
)
def test_capped_rohf_starts_sample_a_space_too_large_to_enumerate(
    n_orbitals, n_alpha, n_beta
):
    """Sizes with about 2.6e13 and 3.9e21 occupations: finishing at all shows
    the sample never enumerates them."""
    spec = FragmentSpec(
        orbitals=tuple(range(n_orbitals)), n_alpha=n_alpha, n_beta=n_beta
    )

    starts = preparation._rohf_starts(np.diag(np.arange(float(n_orbitals))), spec)

    assert len(starts) == preparation._ROHF_STARTS
    assert len({start.tobytes() for start in starts}) == len(starts)
    np.testing.assert_allclose(
        np.abs(starts[0]),
        [
            np.diag(np.arange(n_orbitals) < n_alpha).astype(float),
            np.diag(np.arange(n_orbitals) < n_beta).astype(float),
        ],
    )
    for singles, doubles in starts:
        np.testing.assert_allclose(singles @ singles, singles, atol=1e-12)
        np.testing.assert_allclose(doubles @ doubles, doubles, atol=1e-12)
        assert np.trace(singles) == pytest.approx(n_alpha)
        assert np.trace(doubles) == pytest.approx(n_beta)


def test_a_capped_rohf_sample_is_distinct_and_keeps_aufbau_first():
    """90 occupations exceed the cap, so the sample must still lead with the
    aufbau occupation and never repeat one."""
    spec = FragmentSpec(orbitals=tuple(range(6)), n_alpha=4, n_beta=2)

    starts = preparation._rohf_starts(np.diag(np.arange(6.0)), spec)

    assert len(starts) == preparation._ROHF_STARTS
    np.testing.assert_allclose(
        np.abs(starts[0]),
        [np.diag([1.0] * 4 + [0.0] * 2), np.diag([1.0] * 2 + [0.0] * 4)],
    )
    for first, second in itertools.combinations(starts, 2):
        assert not np.allclose(first, second)


def test_fragment_rohf_keeps_the_lowest_converged_solution(mocker):
    low = mocker.Mock(converged=True, e_tot=-2.0)
    high = mocker.Mock(converged=True, e_tot=-1.0)
    failed = mocker.Mock(converged=False)
    failed.newton.return_value = mocker.Mock(converged=False)
    mocker.patch("pyscf.scf.ROHF", side_effect=[high, failed, low])
    _patch_rohf_starts(mocker, starts=[_ONE_START] * 3)
    spec = FragmentSpec(orbitals=(0, 1), n_alpha=1, n_beta=1)

    assert (
        _solve_fragment_rohf(np.diag([-1.0, 0.5]), np.zeros((2, 2, 2, 2)), spec) is low
    )


def _descending_solver(mocker):
    """A converged ROHF mock whose every ``kernel`` call lowers its energy."""
    solver = mocker.Mock(converged=True, e_tot=0.0, mo_occ=np.array([2.0, 0.0]))

    def kernel(*args, **kwargs):
        solver.e_tot -= 1.0

    solver.kernel.side_effect = kernel
    return solver


def test_fragment_rohf_reoptimises_from_an_unstable_solution(mocker):
    solver = _descending_solver(mocker)
    mocker.patch("pyscf.scf.ROHF", return_value=solver)
    rotated = np.array([[0.0, 1.0], [1.0, 0.0]])
    _patch_rohf_starts(mocker, stability=[(rotated, False), (rotated, True)])
    spec = FragmentSpec(orbitals=(0, 1), n_alpha=1, n_beta=1)

    _solve_fragment_rohf(np.diag([-1.0, 0.5]), np.zeros((2, 2, 2, 2)), spec)

    assert solver.kernel.call_count == 2
    solver.make_rdm1.assert_called_once_with(rotated, solver.mo_occ)


@pytest.mark.parametrize("restart_converges", [False, True])
def test_fragment_rohf_rejects_an_unstable_or_unconverged_restart(
    mocker, restart_converges
):
    solver = _descending_solver(mocker)
    lower_energy = solver.kernel.side_effect

    def kernel(*args, **kwargs):
        lower_energy()
        if solver.kernel.call_count > 1:
            solver.converged = restart_converges

    solver.kernel.side_effect = kernel
    mocker.patch("pyscf.scf.ROHF", return_value=solver)
    _patch_rohf_starts(mocker, stability=[(np.eye(2), False)] * 4)
    spec = FragmentSpec(orbitals=(0, 1), n_alpha=1, n_beta=1)

    with pytest.raises(RuntimeError, match="ROHF did not converge"):
        _solve_fragment_rohf(np.diag([-1.0, 0.5]), np.zeros((2, 2, 2, 2)), spec)


def test_fragment_rohf_accepts_an_instability_that_lowers_nothing(mocker):
    """A degenerate pair's flat direction keeps being reported as unstable."""
    solver = mocker.Mock(converged=True, e_tot=-1.0, mo_occ=np.array([2.0, 0.0]))
    mocker.patch("pyscf.scf.ROHF", return_value=solver)
    _patch_rohf_starts(mocker, stability=lambda mean_field, **_: (np.eye(2), False))
    spec = FragmentSpec(orbitals=(0, 1), n_alpha=1, n_beta=1)

    assert (
        _solve_fragment_rohf(np.diag([-1.0, 0.5]), np.zeros((2, 2, 2, 2)), spec)
        is solver
    )
    assert solver.kernel.call_count == 2


def test_fragment_rohf_uses_positive_local_spin_for_beta_majority(mocker):
    solver = mocker.Mock(converged=True)
    rohf = mocker.patch("pyscf.scf.ROHF", return_value=solver)
    _patch_rohf_starts(mocker)
    spec = FragmentSpec(orbitals=(0, 1, 2), n_alpha=1, n_beta=2)

    _solve_fragment_rohf(
        np.diag([-1.0, -0.5, 0.5]),
        np.zeros((3, 3, 3, 3)),
        spec,
    )

    assert rohf.call_args.args[0].spin == 1


def test_beta_majority_ccsd_amplitudes_are_relabelled_to_physical_spin_channels(
    mocker,
):
    t1_majority = np.array([[11.0], [12.0]])
    t1_minority = np.array([[21.0, 22.0]])
    t2_majority = np.arange(4.0).reshape(2, 2, 1, 1) + 100.0
    t2_mixed = np.arange(4.0).reshape(2, 1, 1, 2) + 200.0
    t2_minority = np.arange(4.0).reshape(1, 1, 2, 2) + 300.0
    coupled_cluster = mocker.Mock(
        t1=(t1_majority, t1_minority),
        t2=(t2_majority, t2_mixed, t2_minority),
    )
    spec = FragmentSpec(orbitals=(0, 1, 2), n_alpha=1, n_beta=2)

    t1, t2 = preparation._physical_spin_amplitudes(coupled_cluster, spec)

    np.testing.assert_array_equal(t1[0], t1_minority)
    np.testing.assert_array_equal(t1[1], t1_majority)
    np.testing.assert_array_equal(t2[0], t2_minority)
    np.testing.assert_array_equal(t2[1], t2_mixed.transpose(1, 0, 3, 2))
    np.testing.assert_array_equal(t2[2], t2_majority)


def test_non_finite_ccsd_amplitudes_fail_before_factorization(mocker):
    spec = FragmentSpec(orbitals=(3, 4), n_alpha=1, n_beta=1)
    _patch_mean_field_and_ccsd(
        mocker, np.eye(2), *_one_pair_amplitudes(np.nan), mo_occ=np.array([2.0, 0.0])
    )

    with pytest.raises(RuntimeError, match=r"fragment \(3, 4\).*non-finite CCSD"):
        prepare_lucj_fragment(_two_orbital_problem(spec), spec)


def test_non_finite_linear_method_result_fails_before_circuit_construction(mocker):
    spec = FragmentSpec(orbitals=(0, 1), n_alpha=1, n_beta=1)
    _patch_linear_method(mocker, lambda x0: np.full_like(x0, np.nan))

    with pytest.raises(
        RuntimeError, match=r"fragment \(0, 1\).*non-finite linear-method"
    ):
        prepare_lucj_fragment(_two_orbital_problem(spec), spec)


def test_fragment_ccsd_warns_and_keeps_best_unconverged_amplitudes(mocker):
    coupled_cluster = mocker.Mock(converged=False)
    ccsd = mocker.Mock(return_value=coupled_cluster)
    mean_field = mocker.Mock()
    spec = FragmentSpec(orbitals=(3, 4), n_alpha=1, n_beta=1)
    message = (
        "UCCSD did not converge for fragment (3, 4) after 500 cycles; using "
        "its best available amplitudes."
    )

    with pytest.warns(UserWarning, match=exact_match(message)):
        result = run_fragment_coupled_cluster(ccsd, "UCCSD", mean_field, spec)

    ccsd.assert_called_once_with(mean_field)
    assert coupled_cluster.max_cycle == 500
    coupled_cluster.kernel.assert_called_once_with()
    assert result is coupled_cluster


def _two_orbital_program(backend, **kwargs):
    problem = MolecularProblem(np.eye(2), np.zeros((2, 2, 2, 2)), n_alpha=1, n_beta=1)
    return LUCJFragmentProgram(
        problem,
        FragmentSpec(orbitals=(0, 1), n_alpha=1, n_beta=1),
        backend=backend,
        **kwargs,
    )


def _measured_x0_preparation():
    circuit = QuantumCircuit(4)
    circuit.x(0)
    classical_bits = ClassicalRegister(4)
    circuit.add_register(classical_bits)
    for index, qubit in enumerate(circuit.qubits):
        circuit.measure(qubit, classical_bits[index])
    return LUCJPreparation(
        circuit=circuit,
        params=np.array([0.1, -0.2]),
        problem=MolecularProblem(
            np.eye(2),
            np.zeros((2, 2, 2, 2)),
            n_alpha=1,
            n_beta=1,
            one_body_beta=2 * np.eye(2),
        ),
        orbital_rotation=np.eye(2),
        mean_field_energy=0.0,
        ccsd_energy=0.0,
        ccsd_converged=True,
    )


@pytest.fixture
def prepared_program(dummy_simulator, mocker):
    """A LUCJ fragment program run once against a patched classical preparation.

    Returns ``(program, preparation, prepare, submit)``.
    """
    preparation = _measured_x0_preparation()
    prepare = mocker.patch(
        f"{_PREPARATION}.prepare_lucj_fragment", return_value=preparation
    )
    submit = mocker.spy(dummy_simulator, "submit_circuits")
    program = _two_orbital_program(dummy_simulator, seed=7)
    assert program.run() is program
    return program, preparation, prepare, submit


def test_lucj_fragment_program_prepares_classically_then_samples_once(
    prepared_program,
):
    program, preparation, prepare, submit = prepared_program

    prepare.assert_called_once()
    assert submit.call_count == 1
    assert program.total_circuit_count == 1
    assert program.has_results()
    assert set(program.best_probs) == {0}
    np.testing.assert_allclose(program.best_params, preparation.params)
    assert program.sampled_problem is preparation.problem
    assert program.orbital_rotation is preparation.orbital_rotation


def test_lucj_fragment_program_rejects_an_unclaimed_run_keyword(
    dummy_simulator, mocker
):
    prepare = mocker.patch(f"{_PREPARATION}.prepare_lucj_fragment")
    program = _two_orbital_program(dummy_simulator)

    with pytest.raises(
        TypeError,
        match=exact_match(
            "LUCJFragmentProgram.run() got unexpected keyword argument(s): bogus."
        ),
    ):
        program.run(bogus=1)

    prepare.assert_not_called()


def test_lucj_fragment_checkpoint_restores_without_preparing_or_sampling(
    prepared_program, dummy_simulator, tmp_path
):
    program, _, prepare, submit = prepared_program
    checkpoint = program._make_checkpoint(tmp_path)
    restored = _two_orbital_program(dummy_simulator, seed=7)
    prepare.reset_mock()
    submit.reset_mock()

    assert restored._restore_checkpoint(checkpoint.model_dump_json(), tmp_path) is True

    assert "phase" not in checkpoint.model_dump()
    prepare.assert_not_called()
    submit.assert_not_called()
    assert restored.best_probs == program.best_probs
    np.testing.assert_allclose(restored.best_params, program.best_params)
    np.testing.assert_allclose(restored.orbital_rotation, program.orbital_rotation)
    restored_problem, problem = restored.sampled_problem, program.sampled_problem
    np.testing.assert_allclose(restored_problem.one_body, problem.one_body)
    np.testing.assert_allclose(restored_problem.one_body_beta, problem.one_body_beta)
    np.testing.assert_allclose(restored_problem.two_body, problem.two_body)
    assert (restored_problem.n_alpha, restored_problem.n_beta) == (1, 1)


def test_lucj_fragment_checkpoint_rejects_a_mismatched_digest(
    prepared_program, dummy_simulator, tmp_path
):
    program, *_ = prepared_program
    checkpoint = program._make_checkpoint(tmp_path).model_copy(
        update={"state_sha256": "0" * 64}
    )
    mismatched = _two_orbital_program(dummy_simulator, seed=7)

    with pytest.raises(
        ValueError,
        match=exact_match("Completed fragment state digest does not match metadata"),
    ):
        mismatched._restore_checkpoint(checkpoint.model_dump_json(), tmp_path)
    assert not mismatched.has_results()


def test_lucj_fragment_checkpoint_writes_its_temporary_state_beside_the_artifact(
    prepared_program, mocker, tmp_path
):
    """The state is staged in the checkpoint directory so the final rename stays
    on one filesystem."""
    program, *_ = prepared_program
    temporary = mocker.spy(preparation.tempfile, "NamedTemporaryFile")

    program._make_checkpoint(tmp_path)

    assert Path(temporary.spy_return.name).parent == tmp_path
    assert [path.name for path in tmp_path.iterdir()] == ["completed_state.npz"]


def test_a_failed_checkpoint_write_leaves_no_temporary_state(
    prepared_program, mocker, tmp_path
):
    program, *_ = prepared_program
    mocker.patch(f"{_PREPARATION}.np.savez", side_effect=OSError("disk full"))

    with pytest.raises(OSError, match=exact_match("disk full")):
        program._make_checkpoint(tmp_path)

    assert list(tmp_path.iterdir()) == []


def test_a_checkpoint_into_a_missing_directory_raises_file_not_found(
    prepared_program, tmp_path
):
    program, *_ = prepared_program

    with pytest.raises(FileNotFoundError):
        program._make_checkpoint(tmp_path / "missing")


def _completed_state_arrays():
    return {
        "params": np.array([0.1]),
        "h_alpha": np.eye(2),
        "h_beta": np.eye(2),
        "two_body": np.zeros((2, 2, 2, 2)),
        "orbital_rotation": np.eye(2),
    }


def _restore_completed_state(
    program, directory, arrays, program_type="LUCJFragmentProgram"
):
    """Write ``arrays`` as a completed fragment state and restore it."""
    state_path = directory / "completed_state.npz"
    np.savez(state_path, **arrays)
    with state_path.open("rb") as handle:
        state_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
    checkpoint = preparation._LUCJFragmentCheckpoint(
        program_type=program_type,
        total_circuit_count=0,
        total_run_time=0.0,
        state_file="completed_state.npz",
        state_sha256=state_sha256,
        best_probs={0: {"0011": 1.0}},
    )
    return program._restore_checkpoint(checkpoint.model_dump_json(), directory)


def test_completed_fragment_state_is_refused_for_another_program_type(
    dummy_simulator, tmp_path
):
    program = _two_orbital_program(dummy_simulator)

    with pytest.raises(
        ValueError, match=exact_match("Checkpoint is for a different program type.")
    ):
        _restore_completed_state(
            program, tmp_path, _completed_state_arrays(), program_type="VQE"
        )

    assert not program.has_results()


@pytest.mark.parametrize(
    "missing",
    ["params", "h_alpha", "h_beta", "two_body", "orbital_rotation"],
)
def test_completed_fragment_state_requires_every_array(
    missing, dummy_simulator, tmp_path
):
    arrays = _completed_state_arrays()
    arrays.pop(missing)
    program = _two_orbital_program(dummy_simulator)

    with pytest.raises(
        ValueError,
        match=exact_match("Completed fragment state has missing or extra arrays"),
    ):
        _restore_completed_state(program, tmp_path, arrays)

    assert not program.has_results()


_INCOMPATIBLE_SHAPES = "Completed fragment state has incompatible array shapes"
_NON_FINITE = "Completed fragment state arrays must be finite numeric data"


@pytest.mark.parametrize(
    "name, value, message",
    [
        ("h_alpha", np.eye(3), _INCOMPATIBLE_SHAPES),
        ("params", np.zeros((1, 1)), _INCOMPATIBLE_SHAPES),
        ("two_body", np.full((2, 2, 2, 2), np.nan), _NON_FINITE),
        ("params", np.array(["a"]), _NON_FINITE),
    ],
)
def test_completed_fragment_state_rejects_corrupt_arrays(
    name, value, message, dummy_simulator, tmp_path
):
    arrays = _completed_state_arrays()
    arrays[name] = value
    program = _two_orbital_program(dummy_simulator)

    with pytest.raises(ValueError, match=exact_match(message)):
        _restore_completed_state(program, tmp_path, arrays)

    assert not program.has_results()


def test_completed_fragment_state_requires_an_orthonormal_rotation(
    dummy_simulator, tmp_path
):
    arrays = _completed_state_arrays()
    arrays["orbital_rotation"] = 2.0 * np.eye(2)
    program = _two_orbital_program(dummy_simulator)

    with pytest.raises(
        ValueError,
        match=r"^Completed fragment state orbital_rotation is not orthonormal",
    ):
        _restore_completed_state(program, tmp_path, arrays)

    assert not program.has_results()


def test_completed_fragment_state_refuses_an_object_array(dummy_simulator, tmp_path):
    arrays = _completed_state_arrays()
    arrays["params"] = np.array([{"payload": 1}], dtype=object)
    program = _two_orbital_program(dummy_simulator)

    with pytest.raises(ValueError, match="allow_pickle=False"):
        _restore_completed_state(program, tmp_path, arrays)

    assert not program.has_results()


@pytest.mark.parametrize(
    "read",
    [
        lambda program: program.best_params,
        lambda program: program.sampled_problem,
        lambda program: program.orbital_rotation,
        lambda program: program._initial_spec(),
    ],
    ids=["best_params", "sampled_problem", "orbital_rotation", "initial_spec"],
)
def test_lucj_fragment_results_are_unavailable_before_run(dummy_simulator, read):
    program = _two_orbital_program(dummy_simulator)

    with pytest.raises(
        RuntimeError,
        match=exact_match("The fragment has not been prepared; call run() first."),
    ):
        read(program)


def test_rotates_sqd_rdms_back_to_the_workflow_fragment_basis():
    angle = 0.37
    cosine = np.cos(angle)
    sine = np.sin(angle)
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    rdm1_alpha = np.array([[0.8, 0.1], [0.1, 0.2]])
    rdm1_beta = np.array([[0.3, -0.05], [-0.05, 0.7]])
    rdm1 = rdm1_alpha + rdm1_beta
    rdm2 = np.zeros((2, 2, 2, 2))
    rdm2[0, 1, 0, 1] = 2.5

    actual = rotate_rdms_to_fragment_basis(rdm1, rdm2, rdm1_alpha, rdm1_beta, rotation)

    cosine_squared = cosine**2
    sine_squared = sine**2
    sine_cosine = sine * cosine
    expected_rdm1_alpha = np.array(
        [
            [
                0.8 * cosine_squared - 0.2 * sine_cosine + 0.2 * sine_squared,
                0.6 * sine_cosine + 0.1 * (cosine_squared - sine_squared),
            ],
            [
                0.6 * sine_cosine + 0.1 * (cosine_squared - sine_squared),
                0.8 * sine_squared + 0.2 * sine_cosine + 0.2 * cosine_squared,
            ],
        ]
    )
    expected_rdm1_beta = np.array(
        [
            [
                0.3 * cosine_squared + 0.1 * sine_cosine + 0.7 * sine_squared,
                -0.4 * sine_cosine - 0.05 * (cosine_squared - sine_squared),
            ],
            [
                -0.4 * sine_cosine - 0.05 * (cosine_squared - sine_squared),
                0.3 * sine_squared - 0.1 * sine_cosine + 0.7 * cosine_squared,
            ],
        ]
    )
    expected_rdm1 = expected_rdm1_alpha + expected_rdm1_beta
    transformed_pair = np.array(
        [
            [-sine_cosine, cosine_squared],
            [-sine_squared, sine_cosine],
        ]
    )
    expected_rdm2 = 2.5 * np.einsum("ij,kl->ijkl", transformed_pair, transformed_pair)
    for received, expected in zip(
        actual,
        (expected_rdm1, expected_rdm2, expected_rdm1_alpha, expected_rdm1_beta),
    ):
        np.testing.assert_allclose(received, expected)
