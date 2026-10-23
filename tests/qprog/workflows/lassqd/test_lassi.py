# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for states across two fragments: the state interaction over products
of fragment multiplets, and LASSQD runs that request it."""

import itertools
import warnings

import numpy as np
import pytest
import scipy.linalg
from pyscf import ao2mo, fci, gto, mcscf, scf
from pyscf.fci import cistring

from divi.qprog.problems import MolecularProblem
from divi.qprog.workflows.lassqd import (
    FixedOrbitals,
    SecondOrderOrbitalUpdate,
    StateAveragedOrbitalUpdate,
    _lassi,
)
from divi.qprog.workflows.lassqd._state import FragmentSpec
from tests._helpers import exact_match
from tests.qprog.workflows.lassqd._helpers import (
    assert_roots_match,
    build_exact_sampler_lassqd,
    fci_roots_by_multiplicity,
)

#: Each H2 of the dimer as one fragment: its bonding then antibonding orbital.
_H2_DIMER_FRAGMENTS = [
    FragmentSpec(orbitals=(0, 2), n_alpha=1, n_beta=1),
    FragmentSpec(orbitals=(1, 3), n_alpha=1, n_beta=1),
]


def _h2_dimer_mean_field(separation):
    """Two parallel H2 molecules ``separation`` Angstrom apart, its orbitals the
    molecules' own bonding (0, 1) then antibonding (2, 3) orbitals,
    Lowdin-orthogonalised."""

    def h2(x):
        return f"H {x} 0 0; H {x} 0 0.74"

    mean_field = scf.RHF(
        gto.M(atom=f"{h2(0.0)}; {h2(separation)}", basis="sto-3g", verbose=0)
    ).run()
    monomers = []
    for x, rows in ((0.0, slice(0, 2)), (separation, slice(2, 4))):
        orbitals = np.zeros((4, 2))
        orbitals[rows] = (
            scf.RHF(gto.M(atom=h2(x), basis="sto-3g", verbose=0)).run().mo_coeff
        )
        monomers.append(orbitals)
    first, second = monomers
    orbitals = np.column_stack([first[:, 0], second[:, 0], first[:, 1], second[:, 1]])
    overlap = mean_field.mol.intor_symmetric("int1e_ovlp")
    mean_field.mo_coeff = orbitals @ scipy.linalg.fractional_matrix_power(
        orbitals.T @ overlap @ orbitals, -0.5
    )
    return mean_field


def _fragment_ordered_integrals(mean_field):
    """The active space's ``(h1, h2, energy_core)`` with each fragment's
    orbitals together, the first fragment's first."""
    order = [orbital for spec in _H2_DIMER_FRAGMENTS for orbital in spec.orbitals]
    casci = mcscf.CASCI(mean_field, 4, 4)
    mo_coeff = mean_field.mo_coeff[:, order]
    h1, energy_core = casci.get_h1eff(mo_coeff)
    return h1, ao2mo.restore(1, casci.get_h2eff(mo_coeff), 4), energy_core


def _random_integrals(n_orbitals, seed):
    """Random real integrals with the 8-fold symmetry of electron repulsion."""
    rng = np.random.default_rng(seed)
    h1 = rng.standard_normal((n_orbitals, n_orbitals))
    n_pairs = n_orbitals * (n_orbitals + 1) // 2
    pairs = rng.standard_normal((n_pairs, n_pairs))
    return h1 + h1.T, ao2mo.restore(1, pairs + pairs.T, n_orbitals)


def _every_determinant(n_orbitals, sector):
    strings = [
        [
            format(int(string), f"0{n_orbitals}b")[::-1]
            for string in cistring.make_strings(range(n_orbitals), count)
        ]
        for count in sector
    ]
    return list(itertools.product(*strings))


def _interaction(h1, h2, shapes, references, n_hops):
    """The state-interaction Hamiltonian and ``S^2`` of two fragments of
    ``shapes`` orbitals, sampled completely at their ``references`` sectors,
    with every multiplet kept and up to ``n_hops`` electrons moved."""
    n_a, n_b = shapes
    blocks = (slice(0, n_a), slice(n_a, n_a + n_b))
    totals = [sum(reference) for reference in references]
    counts = [
        totals[0] + hop
        for hop in range(-n_hops, n_hops + 1)
        if 0 <= totals[0] + hop <= 2 * n_a and 0 <= totals[1] - hop <= 2 * n_b
    ]
    bases = [
        _lassi._basis(
            n,
            _lassi.fragment_multiplets(
                h1[block, block],
                h2[block, block, block, block],
                _every_determinant(n, reference),
                totals[index],
                counts if index == 0 else [sum(totals) - count for count in counts],
                max_memory=4000,
            ),
            None,
        )
        for index, (n, block, reference) in enumerate(zip(shapes, blocks, references))
    ]
    nelec = tuple(references[0][s] + references[1][s] for s in (0, 1))
    pairs = _lassi._sector_pairs(*bases, nelec)
    return _lassi.si_matrices(h1, h2, *bases, pairs)


@pytest.mark.parametrize(
    "shapes, references",
    [
        ((2, 2), ((1, 1), (1, 1))),
        ((3, 2), ((2, 1), (1, 1))),
        ((2, 3), ((1, 0), (2, 1))),
        ((3, 3), ((2, 2), (1, 0))),
    ],
    ids=["even", "odd-first", "odd-second", "unequal-counts"],
)
def test_mixing_every_charge_split_reproduces_fci(shapes, references):
    """With complete fragment spaces and every electron split between the
    fragments, the product states span the whole active space: the spectrum
    and every state's spin are exact, whatever the integrals."""
    n = sum(shapes)
    h1, h2 = _random_integrals(n, seed=sum(shapes) + 7)
    nelec = tuple(references[0][s] + references[1][s] for s in (0, 1))

    hamiltonian, spin_square, _ = _interaction(h1, h2, shapes, references, n_hops=2 * n)

    exact = fci_roots_by_multiplicity(h1, h2, *nelec)
    np.testing.assert_allclose(
        np.linalg.eigvalsh(hamiltonian),
        np.sort([energy for roots in exact.values() for energy in roots]),
        atol=1e-9,
    )
    np.testing.assert_allclose(
        hamiltonian @ spin_square, spin_square @ hamiltonian, atol=1e-9
    )


def test_the_interaction_does_not_depend_on_the_fragments_spin_projections():
    """Fragments described from polarised references hold the same multiplets
    as from closed-shell ones, so the one-hop spectrum is the same."""
    h1, h2 = _random_integrals(4, seed=3)

    closed, _, _ = _interaction(h1, h2, (2, 2), ((1, 1), (1, 1)), n_hops=1)
    polarised, _, _ = _interaction(h1, h2, (2, 2), ((2, 0), (0, 2)), n_hops=1)

    np.testing.assert_allclose(
        np.linalg.eigvalsh(closed), np.linalg.eigvalsh(polarised), atol=1e-9
    )


def _two_fragment_lassqd(backend, mocker, mean_field, states, **overrides):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ensemble, _ = build_exact_sampler_lassqd(
            backend,
            mocker,
            problem=MolecularProblem.from_molecule(mean_field),
            active_spaces=_H2_DIMER_FRAGMENTS,
            states=states,
            **overrides,
        )
    return ensemble


def _casci_roots(mean_field):
    h1, h2, energy_core = _fragment_ordered_integrals(mean_field)
    return fci_roots_by_multiplicity(h1, h2, 2, 2, constant=energy_core)


def test_states_across_two_fragments_match_casci_on_the_same_orbitals(
    dummy_expval_backend, mocker
):
    """Well apart, one electron hop between the fragments is all the
    interaction needs, so each state matches CASCI on the run's orbitals."""
    mean_field = _h2_dimer_mean_field(6.0)
    states = {1: 2, 3: 1, 5: 1}
    ensemble = _two_fragment_lassqd(
        dummy_expval_backend, mocker, mean_field, states, orbital_update=FixedOrbitals()
    )

    ensemble.run(max_rounds=1)

    assert_roots_match(
        ensemble.electronic_states.values(), _casci_roots(mean_field), states
    )


def test_two_hops_make_close_contact_exact(dummy_expval_backend, mocker):
    """In contact, moving two electrons between the fragments matters; with
    both hops every charge split of two-orbital fragments is present."""
    mean_field = _h2_dimer_mean_field(1.5)
    states = {1: 3, 3: 2}
    ensemble = _two_fragment_lassqd(
        dummy_expval_backend,
        mocker,
        mean_field,
        states,
        orbital_update=FixedOrbitals(),
        fragment_hops=2,
    )

    ensemble.run(max_rounds=1)

    assert_roots_match(
        ensemble.electronic_states.values(), _casci_roots(mean_field), states
    )


def test_charge_transfer_is_the_weight_off_each_fragments_own_electron_count(
    dummy_expval_backend, mocker
):
    """The weight equals CASCI's on determinants that move an electron between
    the H2 molecules, which the exact interaction reproduces."""
    mean_field = _h2_dimer_mean_field(1.5)
    states = {1: 3}
    ensemble = _two_fragment_lassqd(
        dummy_expval_backend,
        mocker,
        mean_field,
        states,
        orbital_update=FixedOrbitals(),
        fragment_hops=2,
    )
    ensemble.run(max_rounds=1)

    h1, h2, _ = _fragment_ordered_integrals(mean_field)
    energies, vectors = fci.direct_spin1.kernel(h1, h2, 4, (2, 2), nroots=36)
    strings = cistring.make_strings(range(4), 2)
    on_first = np.array([bin(int(s) & 0b0011).count("1") for s in strings])
    moved = on_first[:, None] + on_first[None, :] != 2
    singlets = [
        float(np.sum(vector[moved] ** 2))
        for vector in vectors
        if round(fci.spin_op.spin_square0(vector, 4, (2, 2))[1]) == 1
    ]
    np.testing.assert_allclose(
        [ensemble.electronic_states[(1, i)].charge_transfer for i in range(3)],
        singlets[:3],
        atol=1e-8,
    )


def test_the_interaction_lowers_the_lowest_state_below_the_fragment_product(
    dummy_expval_backend, mocker
):
    """Close together, the fragments correlate, which a product of fragment
    states misses and the interaction recovers."""
    ensemble = _two_fragment_lassqd(
        dummy_expval_backend,
        mocker,
        _h2_dimer_mean_field(2.0),
        {1: 1},
        orbital_update=FixedOrbitals(),
    )

    ensemble.run(max_rounds=1)

    assert ensemble.electronic_states[(1, 0)].energy < ensemble.energy - 1e-5


def _warnings_of_run(backend, mocker, separation, states, **overrides):
    ensemble = _two_fragment_lassqd(
        backend,
        mocker,
        _h2_dimer_mean_field(separation),
        states,
        orbital_update=FixedOrbitals(),
        **overrides,
    )
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        ensemble.run(max_rounds=1)
    return [str(warning.message) for warning in record]


def _multiplet_warnings(messages):
    return [
        message
        for message in messages
        if message.endswith("Raise SQDConfig.fragment_roots to keep more.")
    ]


def test_a_product_budget_that_leaves_out_multiplets_warns(
    dummy_expval_backend, mocker
):
    mocker.patch.object(_lassi, "_MAX_PRODUCT_STATES", 20)

    (message,) = _multiplet_warnings(
        _warnings_of_run(dummy_expval_backend, mocker, 6.0, {1: 1})
    )

    assert message.startswith(
        "To stay within its product-state budget, the state interaction keeps "
        "each fragment's 1 lowest multiplets per electron count"
    )


def test_a_fragment_roots_that_leaves_out_multiplets_warns(
    dummy_expval_backend, mocker
):
    (message,) = _multiplet_warnings(
        _warnings_of_run(dummy_expval_backend, mocker, 6.0, {1: 1}, fragment_roots=1)
    )

    assert message.startswith(
        "SQDConfig.fragment_roots=1 leaves out each fragment's higher multiplets"
    )


def test_keeping_every_multiplet_does_not_warn(dummy_expval_backend, mocker):
    messages = _warnings_of_run(dummy_expval_backend, mocker, 6.0, {1: 1})

    assert _multiplet_warnings(messages) == []


@pytest.mark.parametrize("fragment_hops, warns", [(1, True), (2, False)])
def test_one_hop_at_close_contact_suggests_two(
    dummy_expval_backend, mocker, fragment_hops, warns
):
    """At 0.8 Angstrom the lowest state's charge-transfer weight is 0.024, and
    one hop leaves it 10 mHa above CASCI."""
    messages = _warnings_of_run(
        dummy_expval_backend, mocker, 0.8, {1: 1}, fragment_hops=fragment_hops
    )

    assert (
        any(message.endswith("Try SQDConfig.fragment_hops=2.") for message in messages)
        is warns
    )


def test_well_apart_one_hop_does_not_suggest_two(dummy_expval_backend, mocker):
    messages = _warnings_of_run(dummy_expval_backend, mocker, 6.0, {1: 1})

    assert not any("fragment_hops" in message for message in messages)


def test_degenerate_states_across_fragments_do_not_warn_about_their_order(
    dummy_expval_backend, mocker
):
    """The two molecules' triplets are degenerate when far apart, but states
    across fragments are labelled by energy every round."""
    messages = _warnings_of_run(dummy_expval_backend, mocker, 10.0, {3: 2})

    assert not any("their order can swap" in message for message in messages)


@pytest.mark.parametrize(
    "max_memory, held",
    [
        (0.01, "The state interaction holds 34 product states densely"),
        (1e-5, "a 2-orbital fragment's multiplets"),
    ],
    ids=["products", "fragment"],
)
def test_the_state_interaction_refuses_to_outgrow_max_memory(
    dummy_expval_backend, mocker, max_memory, held
):
    mean_field = _h2_dimer_mean_field(6.0)
    mean_field.mol.max_memory = max_memory
    ensemble = _two_fragment_lassqd(
        dummy_expval_backend,
        mocker,
        mean_field,
        {1: 1},
        orbital_update=FixedOrbitals(),
    )

    with pytest.raises(MemoryError, match=held):
        ensemble.run(max_rounds=1)


def test_an_under_sampled_fragment_warns(dummy_expval_backend, mocker):
    """Grown from the reference determinant alone, each electron count's space
    misses a determinant, the neutral one the doubly excited one."""
    messages = _warnings_of_run(
        dummy_expval_backend,
        mocker,
        6.0,
        {1: 2},
        extension=None,
        include_reference=True,
        batch_size=1,
        n_batches=1,
    )

    assert (
        "fragment_0's states for the interaction span 5 of 8 determinants. They "
        "grow from its ground-state samples, so the states can come out too high, "
        "the lowest included. Raise the sampling budget (the backend's shots, "
        "batch_size), or set SQDConfig.extension to add the determinants it "
        "misses." in messages
    )


def test_a_fragment_whose_samples_grow_into_every_determinant_does_not_warn(
    dummy_expval_backend, mocker
):
    messages = _warnings_of_run(dummy_expval_backend, mocker, 6.0, {1: 2})

    assert not any("states for the interaction span" in message for message in messages)


def test_fragment_options_need_states_across_two_fragments(
    dummy_expval_backend, mocker
):
    with pytest.raises(
        ValueError,
        match=exact_match(
            "SQDConfig.fragment_roots and fragment_hops shape states across two "
            "fragments; this run has 2 fragment(s) and no SQDConfig.states."
        ),
    ):
        build_exact_sampler_lassqd(
            dummy_expval_backend,
            mocker,
            problem=MolecularProblem.from_molecule(_h2_dimer_mean_field(6.0)),
            active_spaces=_H2_DIMER_FRAGMENTS,
            fragment_hops=2,
        )


def test_lowest_state_orbitals_warn_for_states_across_two_fragments(
    dummy_expval_backend, mocker
):
    with pytest.warns(
        UserWarning,
        match=exact_match(
            "SecondOrderOrbitalUpdate optimises the orbitals for the lowest state "
            "only, so the others miss their orbital relaxation and gaps to them "
            "tend to come out too large. Use FixedOrbitals to compare with CASCI on the "
            "same orbitals."
        ),
    ):
        build_exact_sampler_lassqd(
            dummy_expval_backend,
            mocker,
            problem=MolecularProblem.from_molecule(_h2_dimer_mean_field(6.0)),
            active_spaces=_H2_DIMER_FRAGMENTS,
            states={1: 2},
            orbital_update=SecondOrderOrbitalUpdate(),
        )


@pytest.mark.parametrize(
    "method",
    ["state_wavefunctions", "transition_properties", "nevpt2"],
)
def test_state_exports_need_a_single_fragment(dummy_expval_backend, mocker, method):
    ensemble = _two_fragment_lassqd(
        dummy_expval_backend,
        mocker,
        _h2_dimer_mean_field(6.0),
        {1: 2},
        orbital_update=FixedOrbitals(),
    )
    ensemble.run(max_rounds=1)

    with pytest.raises(
        RuntimeError,
        match=exact_match(
            f"{method} needs the states of a single fragment; states across two "
            "fragments give only their energies, in electronic_states."
        ),
    ):
        getattr(ensemble, method)()


def test_state_averaging_rejects_states_across_two_fragments(
    dummy_expval_backend, mocker
):
    with pytest.raises(
        ValueError,
        match=exact_match(
            "StateAveragedOrbitalUpdate averages over one fragment's states; "
            "states across two fragments use the lowest-state orbitals. Use "
            "SecondOrderOrbitalUpdate, LBFGSOrbitalUpdate or FixedOrbitals."
        ),
    ):
        _two_fragment_lassqd(
            dummy_expval_backend,
            mocker,
            _h2_dimer_mean_field(6.0),
            {1: 2},
            orbital_update=StateAveragedOrbitalUpdate(),
        )


def test_states_across_two_fragments_are_checked_against_the_whole_active_space(
    dummy_expval_backend, mocker
):
    """Each fragment alone holds one triplet, but four orbitals with four
    electrons hold more, so the request is judged on the whole space."""
    ensemble = _two_fragment_lassqd(
        dummy_expval_backend,
        mocker,
        _h2_dimer_mean_field(2.0),
        {3: 3},
        orbital_update=FixedOrbitals(),
        fragment_hops=2,
    )

    ensemble.run(max_rounds=1)

    assert [key for key in ensemble.electronic_states] == [(3, 0), (3, 1), (3, 2)]
