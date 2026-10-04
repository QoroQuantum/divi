# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for LASSQD's VQE fragment preparation: the fragment VQE and its CCSD seed."""

import itertools

import numpy as np
import pytest
from pyscf import cc, fci
from pyscf.cc import addons as cc_addons
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from divi.hamiltonians._chem import _spo_from_integrals
from divi.qprog.algorithms import LUCJAnsatz, QCCAnsatz, UCCSDAnsatz
from divi.qprog.algorithms._ansatze import (
    _uccsd_excitations,
    lucj_jastrow_pairs,
    n_rotation_params,
)
from divi.qprog.optimizers import GridSearchOptimizer
from divi.qprog.problems import HamiltonianProblem
from divi.qprog.workflows._lassqd import _vqe_preparation
from divi.qprog.workflows._lassqd._integrals import (
    fragment_effective_integrals,
    transform_integrals,
)
from divi.qprog.workflows._lassqd._state import FragmentSpec, FragmentState
from tests._helpers import exact_match
from tests.qprog.workflows._lassqd._helpers import (  # noqa: F401
    H4_WHOLE_SPACE,
    H8_HALF_CHAINS,
    ansatz_energy,
    cobyla,
    embedded_fragment_ccsd,
    fragment_integrals,
    fragment_problem,
    h4_chain_mean_field,
    h8_frontier_lassqd,
)


def test_fragment_vqe_rejects_mismatched_seed_params_length(dummy_expval_backend):
    problem = HamiltonianProblem(SparsePauliOp(["ZZ"], [1.0]), n_electrons=2)
    baseline = _vqe_preparation._FragmentVQE(
        problem,
        ansatz=LUCJAnsatz(),
        optimizer=cobyla(),
        backend=dummy_expval_backend,
    )
    bad_params = np.ones(baseline.n_params + 1)

    with pytest.raises(ValueError, match="seed_params"):
        _vqe_preparation._FragmentVQE(
            problem,
            ansatz=LUCJAnsatz(),
            optimizer=cobyla(),
            backend=dummy_expval_backend,
            seed_params=bad_params,
        )


def test_fragment_vqe_exposes_its_problem_integrals_in_its_own_basis(
    dummy_expval_backend,
):
    """SQD reduces every fragment program through the same four attributes; a
    VQE samples in the fragment's own orbitals, so its rotation is the identity."""
    spec = FragmentSpec(orbitals=(0, 1), n_alpha=1, n_beta=1)
    h_alpha = np.array([[-1.0, 0.1], [0.1, 0.5]])
    h_beta = np.array([[-0.9, 0.2], [0.2, 0.4]])
    two_body = np.full((2,) * 4, 0.05)
    program = _vqe_preparation._FragmentVQE(
        fragment_problem(h_alpha, h_beta, two_body, spec),
        ansatz=UCCSDAnsatz(),
        optimizer=cobyla(),
        backend=dummy_expval_backend,
    )

    np.testing.assert_array_equal(program.h_alpha, h_alpha)
    np.testing.assert_array_equal(program.h_beta, h_beta)
    np.testing.assert_array_equal(program.two_body, two_body)
    np.testing.assert_array_equal(program.orbital_rotation, np.eye(2))


_FALLBACK = "Falling back to the optimizer's own initialisation."


@pytest.mark.parametrize(
    "n_beta, n_params, ansatz, message",
    [
        # Neither the t1/t2 read-off nor the factorization means anything for an
        # ansatz whose parameters are unrelated to coupled-cluster amplitudes.
        pytest.param(
            1,
            6,
            QCCAnsatz(),
            "CCSD seeding skipped for fragment (0, 1): no correspondence is defined "
            f"between CCSD amplitudes and QCCAnsatz's parameters. {_FALLBACK}",
            id="qcc",
        ),
        # Restricted CCSD cannot represent a polarized fragment, which is
        # reachable whenever the fragments together still sum to Sz = 0.
        pytest.param(
            0,
            4,
            UCCSDAnsatz(),
            "CCSD seeding skipped for fragment (0, 1): restricted CCSD requires "
            f"equal alpha/beta electron counts, got n_alpha=1, n_beta=0. {_FALLBACK}",
            id="spin-imbalanced",
        ),
    ],
)
def test_ccsd_seed_params_warns_and_skips(n_beta, n_params, ansatz, message):
    spec = FragmentSpec(orbitals=(0, 1), n_alpha=1, n_beta=n_beta)

    with pytest.warns(UserWarning, match=exact_match(message)):
        result = _vqe_preparation._ccsd_seed_params(
            np.eye(2), np.zeros((2,) * 4), spec, n_params, ansatz
        )

    assert result is None


def _h4_as_one_fragment(mean_field):
    """``(h_eff, g_frag)`` for H4's whole canonical-MO space as one fragment."""
    integrals = transform_integrals(
        mean_field.mol, np.asarray(mean_field.mo_coeff), n_core=0, n_act=4
    )
    placeholder = FragmentState(
        spec=H4_WHOLE_SPACE, rdm1=np.zeros((4, 4)), rdm2=np.zeros((4, 4, 4, 4))
    )
    h_eff, _, g_frag = fragment_effective_integrals(integrals, [placeholder], 0)
    return h_eff, g_frag


def test_ccsd_seed_params_uses_amplitude_correspondence_for_uccsd(h4_chain_mean_field):
    """The seed's exact-statevector energy must land near the fragment's own
    CCSD energy, and clearly beats a permutation of the same values.

    A fragment with only one double excitation (e.g. a minimal H2 fragment)
    cannot exercise ordering, sign, or magnitude errors: with a single value
    there is nothing to permute, no crossed-spin case, and no scale to get
    wrong. This uses a single 4-orbital fragment (8 singles, 18 doubles of
    differing sign) so a scrambled index map, a wrong spin pairing, or an
    un-doubled amplitude all produce a detectably worse energy.
    """
    spec = H4_WHOLE_SPACE
    h_eff, g_frag = _h4_as_one_fragment(h4_chain_mean_field)
    n_params = UCCSDAnsatz.n_params_per_layer(
        2 * spec.n_orbitals, n_electrons=spec.n_alpha + spec.n_beta
    )

    seed = _vqe_preparation._ccsd_seed_params(
        h_eff, g_frag, spec, n_params, UCCSDAnsatz()
    )

    assert seed is not None
    coupled_cluster = cc.CCSD(h4_chain_mean_field)
    coupled_cluster.kernel()
    ccsd_electronic_energy = (
        coupled_cluster.e_tot - h4_chain_mean_field.mol.energy_nuc()
    )

    seed_energy = ansatz_energy(seed, h_eff, g_frag, spec)
    permuted_energy = ansatz_energy(
        np.random.default_rng(0).permutation(seed), h_eff, g_frag, spec
    )

    assert seed_energy == pytest.approx(ccsd_electronic_energy, abs=1e-3)
    assert seed_energy < permuted_energy - 0.05


@pytest.mark.parametrize(
    "ansatz, min_recovered_fraction",
    [(UCCSDAnsatz(), 0.9), (LUCJAnsatz(), 0.3)],
    ids=["uccsd", "lucj"],
)
def test_seed_beats_the_reference_determinant_in_a_localized_basis(
    dummy_expval_backend, ansatz, min_recovered_fraction
):
    """The seed must recover correlation in the basis the fragments actually use.

    The canonical-MO test above cannot see this: there the fragment basis and
    the basis an SCF on the fragment's integrals converges to coincide. Real
    fragments are localized, and an SCF rotates within their occupied and
    virtual blocks, permuting which amplitude belongs to which orbital pair.
    Amplitudes read off that rotated solution seeded a state *above* the
    reference determinant.

    LUCJ's seed runs through the double factorization, where every link is a
    sign convention -- the Jastrow's factor of ``i`` absorbed into one sector's
    rotation, the conjugate transpose of the sandwiched block, the
    ``RZZ(-J / 2)`` pair term -- and getting one wrong leaves the seed *at* the
    reference. One layer holds only the leading factorization term, so its
    recovered fraction is well short of UCCSD's (measured 0.36).
    """
    ensemble = h8_frontier_lassqd(dummy_expval_backend)
    state = ensemble.initial_state()

    for index, fragment in enumerate(state.fragments):
        spec = fragment.spec
        h_alpha, h_beta, g_frag = fragment_integrals(
            ensemble, state.mo_coeff, state.fragments, index
        )
        n_params = type(ansatz).n_params_per_layer(
            2 * spec.n_orbitals, n_electrons=spec.n_alpha + spec.n_beta
        )
        exact = fci.direct_spin1.kernel(
            h_alpha, g_frag, spec.n_orbitals, (spec.n_alpha, spec.n_beta)
        )[0]
        seed = _vqe_preparation._ccsd_seed_params(
            0.5 * (h_alpha + h_beta), g_frag, spec, n_params, ansatz, {}
        )
        assert seed is not None

        reference = ansatz_energy(np.zeros(n_params), h_alpha, g_frag, spec, ansatz)
        seeded = ansatz_energy(seed, h_alpha, g_frag, spec, ansatz)

        assert seeded > exact - 1e-8
        assert (reference - seeded) / (reference - exact) > min_recovered_fraction


def _seed_gain(
    ansatz,
    h_alpha,
    h_beta,
    g_frag,
    spec,
    seed_integrals=None,
    permute=False,
    **kwargs,
):
    """``_seed_energy_gain`` for a seed on one fragment's integrals.

    ``seed_integrals`` builds the seed from a *different* ``(h, g)`` than the
    Hamiltonian it is scored against, which is what a basis mismatch is.
    ``permute`` scores a permutation of the seed instead.
    """
    build_kwargs = {
        "n_electrons": spec.n_alpha + spec.n_beta,
        "n_alpha": spec.n_alpha,
        "n_beta": spec.n_beta,
        **kwargs,
    }
    n_params = type(ansatz).n_params_per_layer(2 * spec.n_orbitals, **build_kwargs)
    h_seed, g_seed = seed_integrals or (0.5 * (h_alpha + h_beta), g_frag)
    seed = _vqe_preparation._ccsd_seed_params(
        h_seed, g_seed, spec, n_params, ansatz, kwargs
    )
    assert seed is not None
    if permute:
        seed = np.random.default_rng(0).permutation(seed)
    hamiltonian = _spo_from_integrals(
        h_alpha, g_frag, constant=0.0, one_body_beta=h_beta
    )
    return _vqe_preparation._seed_energy_gain(
        seed, hamiltonian, ansatz, 2 * spec.n_orbitals, 1, build_kwargs
    )


@pytest.mark.filterwarnings("ignore::scipy.sparse.SparseEfficiencyWarning")
def test_seed_acceptance_rejects_amplitudes_on_the_wrong_excitations():
    """The failure class the check exists for, and the one a stationarity
    precondition provably cannot see.

    Running CCSD in a basis rotated within the occupied and virtual blocks --
    what an SCF canonicalization does -- leaves the reference determinant's
    energy identical while permuting which amplitude belongs to which orbital
    pair. ``F_ov`` transforms as ``U_o^T F_ov U_v``, so a stationarity check
    stays satisfied throughout. Permuting the seed vector is that same corruption
    applied directly, and the energy comparison catches it.
    """
    ensemble = h8_frontier_lassqd(None)
    state = ensemble.initial_state()
    spec = state.fragments[0].spec
    h_eff, _, g_frag = fragment_integrals(ensemble, state.mo_coeff, state.fragments, 0)

    def gain(permute):
        return _seed_gain(UCCSDAnsatz(), h_eff, h_eff, g_frag, spec, permute=permute)

    assert gain(permute=False) > _vqe_preparation._SEED_ACCEPTANCE_MARGIN
    assert gain(permute=True) < _vqe_preparation._SEED_ACCEPTANCE_MARGIN


@pytest.mark.filterwarnings("ignore::scipy.sparse.SparseEfficiencyWarning")
def test_seed_acceptance_admits_a_polarized_non_stationary_fragment():
    """A spin-polarized fragment has no shared spatial basis making both spin
    channels Hartree-Fock stationary, so the old precondition refused it
    outright -- which is what left the diiron benchmark unseeded. The seed is
    worth keeping there."""
    ensemble = h8_frontier_lassqd(local_spins=[2, -2], fragment_atoms=H8_HALF_CHAINS)
    state = ensemble.initial_state()
    spec = state.fragments[0].spec
    assert spec.n_alpha != spec.n_beta
    h_alpha, h_beta, g_frag = fragment_integrals(
        ensemble, state.mo_coeff, state.fragments, 0
    )

    gain = _seed_gain(
        LUCJAnsatz(), h_alpha, h_beta, g_frag, spec, trailing_rotation=True
    )

    assert gain > _vqe_preparation._SEED_ACCEPTANCE_MARGIN


def test_seed_acceptance_skips_a_fragment_too_wide_to_check():
    """Above the exact-check width the seed is accepted unchecked rather than
    discarded: refusing a good seed is the failure this replaced."""
    spec = FragmentSpec(orbitals=tuple(range(11)), n_alpha=2, n_beta=2)
    n_qubits = 2 * spec.n_orbitals
    assert n_qubits > _vqe_preparation._SEED_CHECK_MAX_QUBITS

    gain = _vqe_preparation._seed_energy_gain(
        np.zeros(4), object(), LUCJAnsatz(), n_qubits, 1, {}
    )

    assert gain is None


def test_seed_acceptance_checks_a_fragment_at_the_exact_check_width(mocker):
    ansatz = mocker.Mock(build=mocker.Mock(return_value=QuantumCircuit(1)))
    n_qubits = _vqe_preparation._SEED_CHECK_MAX_QUBITS

    gain = _vqe_preparation._seed_energy_gain(
        np.ones(3), SparsePauliOp("Z"), ansatz, n_qubits, 1, {}
    )

    assert gain == 0.0
    assert [call.args[1] for call in ansatz.build.call_args_list] == [n_qubits] * 2


def test_uccsd_seed_singles_match_the_ccsd_t1_amplitudes(h4_chain_mean_field):
    """The energy assertion above cannot cover the singles, so pin their values.

    That fragment's orbitals are canonical MOs, so ``t1`` is
    Brillouin-suppressed: measured ``max|t1|`` is 1.25e-03 against ``max|t2|``
    of 8.2e-02. Zeroing every single moves the seeded energy by 7.5e-06 and
    sign-flipping them by 2.4e-05 -- both far inside that test's 1e-3 tolerance,
    so the whole singles block could be scrambled or deleted unnoticed. The
    doubles are caught there with 40x margin.
    """
    spec = H4_WHOLE_SPACE
    h_eff, g_frag = _h4_as_one_fragment(h4_chain_mean_field)
    n_params = UCCSDAnsatz.n_params_per_layer(8, n_electrons=4)

    seed = _vqe_preparation._ccsd_seed_params(
        h_eff, g_frag, spec, n_params, UCCSDAnsatz()
    )
    assert seed is not None

    embedded = embedded_fragment_ccsd(h_eff, g_frag, spec)
    t1_spin = cc_addons.spatial2spin(embedded.t1)

    n_spatial, n_occupied = spec.n_orbitals, spec.n_alpha
    singles_seen = 0
    for index, (occupied, unoccupied) in enumerate(
        _uccsd_excitations(n_spatial, (n_occupied, n_occupied))
    ):
        if len(occupied) != 1:
            continue
        spin_o, spatial_o = divmod(occupied[0], n_spatial)
        spin_u, spatial_u = divmod(unoccupied[0], n_spatial)
        expected = -t1_spin[
            2 * spatial_o + spin_o, 2 * (spatial_u - n_occupied) + spin_u
        ]
        assert seed[index] == pytest.approx(expected, abs=1e-12)
        singles_seen += 1

    assert singles_seen == 8
    # Guards against the whole block being zero, which the loop above would
    # otherwise accept if t1 itself came back empty.
    assert np.abs(t1_spin).max() > 1e-6


_ONE_PAIR_SPEC = FragmentSpec(orbitals=(0, 1), n_alpha=1, n_beta=1)

#: The coupled-cluster solver each seeded ansatz runs.
_SEED_SOLVERS = [
    pytest.param(UCCSDAnsatz(), "CCSD", id="uccsd"),
    pytest.param(LUCJAnsatz(), "UCCSD", id="lucj"),
]


@pytest.mark.parametrize("ansatz, solver", _SEED_SOLVERS)
def test_an_unconverged_seed_calculation_warns_but_still_seeds(
    h4_chain_mean_field, mocker, ansatz, solver
):
    mocker.patch.object(_vqe_preparation, "_SEED_CC_MAX_CYCLE", 1)
    h_eff, g_frag = _h4_as_one_fragment(h4_chain_mean_field)
    n_params = type(ansatz).n_params_per_layer(8, n_electrons=4, n_alpha=2, n_beta=2)
    message = (
        f"{solver} did not converge for fragment (0, 1, 2, 3); seeding from its "
        "amplitudes anyway, since the seed is accepted on the energy it delivers "
        "rather than on the solver's own criterion."
    )

    with pytest.warns(UserWarning, match=exact_match(message)):
        seed = _vqe_preparation._ccsd_seed_params(
            h_eff, g_frag, H4_WHOLE_SPACE, n_params, ansatz, {}
        )

    assert seed is not None
    assert seed.shape == (n_params,)


@pytest.mark.parametrize("ansatz, solver", _SEED_SOLVERS)
def test_a_failed_seed_calculation_falls_back_with_a_warning(mocker, ansatz, solver):
    mocker.patch.object(_vqe_preparation.cc, solver, side_effect=RuntimeError("boom"))
    message = f"CCSD seeding failed for fragment (0, 1): boom. {_FALLBACK}"

    with pytest.warns(UserWarning, match=exact_match(message)):
        seed = _vqe_preparation._ccsd_seed_params(
            np.eye(2), np.zeros((2,) * 4), _ONE_PAIR_SPEC, 6, ansatz, {}
        )

    assert seed is None


def test_lucj_seed_runs_uccsd_on_the_fragment_reference(h4_chain_mean_field, mocker):
    """On a closed-shell fragment the unrestricted solve must reproduce
    restricted CCSD on the same integrals and reference determinant."""
    solver = mocker.spy(_vqe_preparation.cc, "UCCSD")
    h_eff, g_frag = _h4_as_one_fragment(h4_chain_mean_field)
    n_params = LUCJAnsatz.n_params_per_layer(8, n_electrons=4, n_alpha=2, n_beta=2)

    _vqe_preparation._lucj_seed_params(h_eff, g_frag, H4_WHOLE_SPACE, n_params, {})

    restricted = embedded_fragment_ccsd(h_eff, g_frag, H4_WHOLE_SPACE)
    assert solver.spy_return.e_corr == pytest.approx(restricted.e_corr, abs=1e-6)


def _coupled_cluster(mocker, t1_alpha, t1_beta, t2_opposite):
    """UCCSD amplitudes with the given singles and opposite-spin doubles, and no
    same-spin doubles."""
    t2_opposite = np.asarray(t2_opposite)
    return mocker.Mock(
        t1=(np.asarray(t1_alpha), np.asarray(t1_beta)),
        t2=(np.zeros_like(t2_opposite), t2_opposite, np.zeros_like(t2_opposite)),
    )


def _one_pair_coupled_cluster(mocker, double, single=0.0):
    """UCCSD amplitudes for one occupied and one virtual orbital per spin."""
    return _coupled_cluster(
        mocker,
        np.full((1, 1), single),
        np.full((1, 1), -single),
        np.full((1, 1, 1, 1), double),
    )


def _trailing_rotation_warning(orbitals):
    """Exact pattern of the warning for an unrealizable trailing rotation."""
    return exact_match(
        f"CCSD seeding for fragment {orbitals} could not realize the trailing "
        "rotation; seeding the rest and leaving it at the identity."
    )


def test_lucj_seed_skips_a_shared_spin_params_ansatz(mocker):
    message = (
        "CCSD seeding skipped for fragment (0, 1): the double factorisation gives "
        "each spin sector its own rotation, which shared_spin_params cannot hold. "
        f"{_FALLBACK}"
    )

    with pytest.warns(UserWarning, match=exact_match(message)):
        seed = _vqe_preparation._lucj_amplitude_seed(
            _one_pair_coupled_cluster(mocker, 0.3),
            _ONE_PAIR_SPEC,
            6,
            {"shared_spin_params": True},
        )

    assert seed is None


def _lucj_n_params(**ansatz_kwargs):
    return LUCJAnsatz.n_params_per_layer(
        4, n_electrons=2, n_alpha=1, n_beta=1, **ansatz_kwargs
    )


@pytest.mark.parametrize("double", [0.3, -0.3])
@pytest.mark.parametrize(
    "ansatz_kwargs",
    [{}, {"rotation_depth": 1}, {"same_spin_pairs": [(0, 1)]}],
    ids=["default", "rotation_depth", "same_spin_pairs"],
)
def test_lucj_seed_places_the_factorised_jastrow_in_its_layout(
    mocker, ansatz_kwargs, double
):
    """With one pair per spin the doubles tensor is a single number ``t``: its
    singular value is ``|t|`` and each sector's one-body operator has
    eigenvalues ``(-1, 1)``, so every opposite-spin angle is
    ``-|t| d_p d_q / 2`` and every same-spin angle stays zero."""
    n_params = _lucj_n_params(**ansatz_kwargs)
    same_pairs, opposite_pairs = lucj_jastrow_pairs(
        2, ansatz_kwargs.get("same_spin_pairs"), None
    )
    jastrow_start = 2 * n_rotation_params(
        2, orbital_phases=False, depth=ansatz_kwargs.get("rotation_depth")
    )
    diagonal = np.array([-1.0, 1.0])

    seed = _vqe_preparation._lucj_amplitude_seed(
        _one_pair_coupled_cluster(mocker, double),
        _ONE_PAIR_SPEC,
        n_params,
        ansatz_kwargs,
    )

    assert seed is not None
    assert seed.shape == (n_params,)
    jastrow_end = jastrow_start + len(opposite_pairs)
    np.testing.assert_allclose(
        seed[jastrow_start:jastrow_end],
        [-0.5 * abs(double) * diagonal[p] * diagonal[q] for p, q in opposite_pairs],
        atol=1e-12,
    )
    assert jastrow_end + 2 * len(same_pairs) == n_params
    assert not seed[jastrow_end:].any()


def _fit_only_the_first(count):
    """``rotation_angles`` that gives up after its first ``count`` fits."""
    fit = _vqe_preparation.rotation_angles
    calls = iter(range(count))

    def first_only(target, *, depth=None):
        return fit(target, depth=depth) if next(calls, None) is not None else None

    return first_only


def test_lucj_seed_keeps_the_rest_when_the_trailing_rotation_cannot_be_fit(mocker):
    mocker.patch.object(
        _vqe_preparation, "rotation_angles", side_effect=_fit_only_the_first(2)
    )
    n_params = _lucj_n_params(trailing_rotation=True)
    n_trailing = 2 * n_rotation_params(2, orbital_phases=True)

    with pytest.warns(UserWarning, match=_trailing_rotation_warning((0, 1))):
        seed = _vqe_preparation._lucj_amplitude_seed(
            _one_pair_coupled_cluster(mocker, 0.3, single=0.05),
            _ONE_PAIR_SPEC,
            n_params,
            {"trailing_rotation": True},
        )

    assert seed is not None
    assert seed.shape == (n_params,)
    assert not seed[-n_trailing:].any()
    assert seed[:-n_trailing].any()


_THREE_ORBITAL_SPEC = FragmentSpec(orbitals=(0, 1, 2), n_alpha=1, n_beta=1)

#: Departs from every default: two rotation half-layers of three, and a single
#: same-spin pair other than the nearest-neighbour ones.
_THREE_ORBITAL_KWARGS = {
    "rotation_depth": 2,
    "same_spin_pairs": [(0, 2)],
    "trailing_rotation": True,
}

_THREE_ORBITAL_DOUBLES = np.array([0.3, 0.1, 0.05, 0.2]).reshape(1, 1, 2, 2)


def _numbered_rotation_angles(fail_on=None):
    """``rotation_angles`` stand-in returning ``1, 2, ..., n`` for each block, so
    every block's position in the seed is readable; call ``fail_on`` (counting
    from 1) gives up instead."""
    calls = itertools.count(1)

    def numbered(target, *, depth=None):
        if next(calls) == fail_on:
            return None
        n_params = n_rotation_params(target.shape[0], orbital_phases=True, depth=depth)
        return np.arange(1.0, n_params + 1)

    return numbered


def _three_orbital_lucj_seed(mocker, fail_on=None):
    """``_lucj_amplitude_seed`` on one occupied and two virtual orbitals per spin
    under :data:`_THREE_ORBITAL_KWARGS`, with numbered rotation fits."""
    fit = mocker.patch.object(
        _vqe_preparation,
        "rotation_angles",
        side_effect=_numbered_rotation_angles(fail_on),
    )
    n_params = LUCJAnsatz.n_params_per_layer(
        6, n_electrons=2, n_alpha=1, n_beta=1, **_THREE_ORBITAL_KWARGS
    )

    seed = _vqe_preparation._lucj_amplitude_seed(
        _coupled_cluster(
            mocker, [[0.04, -0.02]], [[-0.03, 0.01]], _THREE_ORBITAL_DOUBLES
        ),
        _THREE_ORBITAL_SPEC,
        n_params,
        _THREE_ORBITAL_KWARGS,
    )

    assert [call.kwargs["depth"] for call in fit.call_args_list] == [2] * 4
    return seed


def _three_orbital_layout(alpha_trailing):
    """The seed :func:`_three_orbital_lucj_seed` must produce.

    Each sector's one-body operator holds a unit vector, so its eigenvalues are
    ``(-1, 0, 1)`` and the on-site angles are ``-s / 2 * (1, 0, 1)`` for the
    leading singular value ``s``.
    """
    sandwiched = np.arange(1.0, n_rotation_params(3, orbital_phases=False, depth=2) + 1)
    trailing = np.arange(1.0, n_rotation_params(3, orbital_phases=True, depth=2) + 1)
    scale = np.linalg.svd(_THREE_ORBITAL_DOUBLES.reshape(2, 2), compute_uv=False)[0]
    jastrow = -0.5 * scale * np.array([1.0, 0.0, 1.0])
    same_spin = np.zeros(2)
    return np.concatenate(
        [
            sandwiched,
            sandwiched,
            jastrow,
            same_spin,
            trailing if alpha_trailing else np.zeros_like(trailing),
            trailing,
        ]
    )


def test_lucj_seed_layout_honours_rotation_depth_and_pairs(mocker):
    """The layout test above runs two orbitals, where depth 1 is the full schedule
    and ``(0, 1)`` is the only same-spin pair, so neither keyword can move a
    block there."""
    seed = _three_orbital_lucj_seed(mocker)

    np.testing.assert_allclose(seed, _three_orbital_layout(True), atol=1e-12)


def test_lucj_seed_keeps_the_beta_trailing_rotation_when_only_alpha_fails(mocker):
    with pytest.warns(UserWarning, match=_trailing_rotation_warning((0, 1, 2))):
        seed = _three_orbital_lucj_seed(mocker, fail_on=3)

    np.testing.assert_allclose(seed, _three_orbital_layout(False), atol=1e-12)


def _h4_fragment_vqe(backend, h_alpha, h_beta, g_frag, optimizer=None, params=None):
    """``build_fragment_vqe`` for H4's whole space as a UCCSD fragment."""
    spec = H4_WHOLE_SPACE
    return _vqe_preparation.build_fragment_vqe(
        fragment_problem(h_alpha, h_beta, g_frag, spec),
        FragmentState(
            spec=spec, rdm1=np.zeros((4, 4)), rdm2=np.zeros((4,) * 4), params=params
        ),
        ansatz=UCCSDAnsatz(),
        optimizer=cobyla() if optimizer is None else optimizer,
        max_iterations=1,
        backend=backend,
        seed=0,
        options={},
    )


@pytest.mark.filterwarnings("ignore::scipy.sparse.SparseEfficiencyWarning")
def test_seed_averages_an_asymmetric_embedding_and_warns(
    h4_chain_mean_field, dummy_expval_backend, mocker
):
    seed_params = mocker.spy(_vqe_preparation, "_ccsd_seed_params")
    h_alpha, g_frag = _h4_as_one_fragment(h4_chain_mean_field)
    h_beta = h_alpha + np.diag([0.0, 2e-3, 0.0, 0.0])
    message = (
        "CCSD seeding for fragment (0, 1, 2, 3) averages an embedding potential "
        "whose spin channels differ by 2.000e-03 Hartree, because seeding takes a "
        "single one-body matrix. The seed may sit in a different basin than the "
        "symmetry-broken solution; the Hamiltonian being optimised keeps both "
        "channels."
    )

    with pytest.warns(UserWarning, match=exact_match(message)):
        _h4_fragment_vqe(dummy_expval_backend, h_alpha, h_beta, g_frag)

    np.testing.assert_allclose(seed_params.call_args.args[0], 0.5 * (h_alpha + h_beta))


@pytest.mark.filterwarnings("ignore::scipy.sparse.SparseEfficiencyWarning")
@pytest.mark.filterwarnings("error:CCSD seeding for fragment .* averages")
def test_seed_from_a_spin_symmetric_embedding_does_not_warn(
    h4_chain_mean_field, dummy_expval_backend
):
    h_eff, g_frag = _h4_as_one_fragment(h4_chain_mean_field)

    program = _h4_fragment_vqe(dummy_expval_backend, h_eff, h_eff, g_frag)

    assert program._seed_params is not None


def test_a_seed_above_the_reference_is_rejected(
    h4_chain_mean_field, dummy_expval_backend, mocker
):
    mocker.patch.object(_vqe_preparation, "_seed_energy_gain", return_value=-2e-3)
    h_eff, g_frag = _h4_as_one_fragment(h4_chain_mean_field)
    message = (
        "CCSD seeding rejected for fragment (0, 1, 2, 3): the seed sits +2.000e-03 "
        "Hartree relative to the reference determinant, so it carries no "
        f"correlation energy. {_FALLBACK}"
    )

    with pytest.warns(UserWarning, match=exact_match(message)):
        program = _h4_fragment_vqe(dummy_expval_backend, h_eff, h_eff, g_frag)

    assert program._seed_params is None


def test_each_fragment_vqe_gets_its_own_optimizer_state(
    h4_chain_mean_field, dummy_expval_backend
):
    h_eff, g_frag = _h4_as_one_fragment(h4_chain_mean_field)
    n_params = UCCSDAnsatz.n_params_per_layer(8, n_electrons=4)
    template = GridSearchOptimizer(param_grid=np.zeros((1, n_params)))

    program = _h4_fragment_vqe(
        dummy_expval_backend,
        h_eff,
        h_eff,
        g_frag,
        optimizer=template,
        params=np.zeros(n_params),
    )

    assert program.optimizer._param_grid is not template._param_grid
    np.testing.assert_array_equal(program.optimizer._param_grid, template._param_grid)
