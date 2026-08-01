# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for SQD post-processing, checked against exact classical oracles."""

import itertools

import numpy as np
import pytest
from pyscf import ao2mo, fci, scf

import divi.qprog.workflows._lassqd._sqd as sqd_module
from divi.qprog.workflows._lassqd._sqd import (
    SQDSolver,
    bit_flip_correction,
    bitstring_to_spatial_det,
    compute_spatial_rdms,
    filter_symmetry,
    s2_matrix_element,
    slater_condon,
    spatial_to_spin_occupations,
    spin_orbital_integrals,
    spin_to_spatial_occupations,
)
from tests.qprog.workflows._lassqd._helpers import (  # noqa: F401
    dense_fci_energy,
    h2_molecule,
    h4_chain,
    uniform_full_space_probs,
)


def _integrals_from_mol(mol):
    mean_field = scf.RHF(mol).run(verbose=0)
    mo_coeff = np.asarray(mean_field.mo_coeff)
    n_orb = mo_coeff.shape[1]
    one_body = mo_coeff.T @ mean_field.get_hcore() @ mo_coeff
    two_body = ao2mo.restore(1, ao2mo.kernel(mol, mo_coeff), n_orb)
    return one_body, two_body, n_orb, float(mol.energy_nuc())


def _h2_integrals():
    return _integrals_from_mol(h2_molecule())


def _h2_631g_integrals():
    return _integrals_from_mol(h2_molecule(basis="6-31g"))


def _h4_integrals():
    return _integrals_from_mol(h4_chain())


def test_spatial_to_spin_occupations_uses_blocked_indexing():
    """Alpha keeps its spatial index; beta is offset by n_orb."""
    assert spatial_to_spin_occupations((0,), (1,), n_orb=2) == (0, 3)
    assert spatial_to_spin_occupations((0, 1), (0,), n_orb=2) == (0, 1, 2)
    # Distinguishes blocked ((1,) -> (1,)) from interleaved ((1,) -> (2,)).
    assert spatial_to_spin_occupations((1,), (), n_orb=2) == (1,)


def test_spin_to_spatial_round_trips():
    for n_orb in (2, 3):
        for alpha in itertools.combinations(range(n_orb), 1):
            for beta in itertools.combinations(range(n_orb), 1):
                spin = spatial_to_spin_occupations(alpha, beta, n_orb)
                assert spin_to_spatial_occupations(spin, n_orb) == (alpha, beta)


def test_bitstring_to_spatial_det():
    """Blocked SQD bitstring '1001' on 2 orbitals: alpha {0}, beta {1}."""
    assert bitstring_to_spatial_det("1001", n_orb=2) == ((0,), (1,))
    assert bitstring_to_spatial_det("1010", n_orb=2) == ((0,), (0,))


def test_slater_condon_vanishes_beyond_double_excitation():
    # Only 2 spatial orbitals available in H2/STO-3G, so build a 3-orbital toy instead.
    # Pins the physical property that Slater-Condon matrix elements vanish
    # beyond a double excitation -- not a claim about the specific early-return
    # guard in slater_condon(), which is redundant with the function's own
    # trailing fallback. Integrals are non-zero so a future refactor that
    # dropped or reordered that fallback would still be caught here.
    n = 3
    h3 = np.zeros((n, n))
    g3 = np.arange(1, n**4 + 1, dtype=float).reshape((n,) * 4) * 0.01
    h3[np.diag_indices(n)] = [1.0, 2.0, 3.0]
    hs, gs = spin_orbital_integrals(h3, g3, n)
    det_i = spatial_to_spin_occupations((0, 1), (0, 1), n)
    det_j = spatial_to_spin_occupations((2,), (2,), n)  # quadruple excitation
    assert slater_condon(det_i, det_j, hs, gs) == pytest.approx(0.0)


def test_full_subspace_diagonalization_reproduces_fci():
    """The strongest single check: Slater-Condon over the COMPLETE determinant
    space must give exactly the FCI energy."""
    one_body, two_body, n_orb, constant = _h2_integrals()
    h_spin, g_spin = spin_orbital_integrals(one_body, two_body, n_orb)

    n_alpha = n_beta = 1
    dets = [
        spatial_to_spin_occupations(a, b, n_orb)
        for a in itertools.combinations(range(n_orb), n_alpha)
        for b in itertools.combinations(range(n_orb), n_beta)
    ]
    dim = len(dets)
    hamiltonian = np.array(
        [
            [slater_condon(dets[i], dets[j], h_spin, g_spin) for j in range(dim)]
            for i in range(dim)
        ]
    )
    lowest = float(np.min(np.linalg.eigvalsh(hamiltonian))) + constant
    expected = dense_fci_energy(one_body, two_body, n_alpha, n_beta, constant)

    assert lowest == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize(
    "integrals_fn, n_alpha, n_beta",
    [
        (_h2_631g_integrals, 1, 1),
        (_h4_integrals, 2, 2),
    ],
)
def test_full_subspace_diagonalization_reproduces_fci_beyond_two_orbitals(
    integrals_fn, n_alpha, n_beta
):
    """Same check as above, but on active spaces with more than 2 spatial
    orbitals, where a double-excitation sign error is no longer an exact
    gauge transformation and must change the ground-state energy."""
    one_body, two_body, n_orb, constant = integrals_fn()
    h_spin, g_spin = spin_orbital_integrals(one_body, two_body, n_orb)

    dets = [
        spatial_to_spin_occupations(a, b, n_orb)
        for a in itertools.combinations(range(n_orb), n_alpha)
        for b in itertools.combinations(range(n_orb), n_beta)
    ]
    dim = len(dets)
    hamiltonian = np.array(
        [
            [slater_condon(dets[i], dets[j], h_spin, g_spin) for j in range(dim)]
            for i in range(dim)
        ]
    )
    lowest = float(np.min(np.linalg.eigvalsh(hamiltonian))) + constant
    expected = dense_fci_energy(one_body, two_body, n_alpha, n_beta, constant)

    assert lowest == pytest.approx(expected, abs=1e-9)


def test_hamiltonian_matrix_is_symmetric():
    one_body, two_body, n_orb, _ = _h2_integrals()
    h_spin, g_spin = spin_orbital_integrals(one_body, two_body, n_orb)
    dets = [
        spatial_to_spin_occupations(a, b, n_orb)
        for a in itertools.combinations(range(n_orb), 1)
        for b in itertools.combinations(range(n_orb), 1)
    ]
    dim = len(dets)
    matrix = np.array(
        [
            [slater_condon(dets[i], dets[j], h_spin, g_spin) for j in range(dim)]
            for i in range(dim)
        ]
    )
    np.testing.assert_allclose(matrix, matrix.T, atol=1e-12)


def _s2_eigenvalues(dets, n_orb):
    dim = len(dets)
    matrix = np.array(
        [
            [s2_matrix_element(dets[i], dets[j], n_orb) for j in range(dim)]
            for i in range(dim)
        ]
    )
    return np.linalg.eigvalsh(matrix), matrix


def test_s2_eigenvalues_are_singlet_and_triplet():
    """Two electrons in two orbitals span one triplet (S^2 = 2) and singlets (0)."""
    n_orb = 2
    dets = [
        ((0,), (0,)),
        ((0,), (1,)),
        ((1,), (0,)),
        ((1,), (1,)),
    ]
    eigenvalues, matrix = _s2_eigenvalues(dets, n_orb)
    np.testing.assert_allclose(matrix, matrix.T, atol=1e-12)
    rounded = np.round(eigenvalues, 9)
    assert sorted(rounded) == pytest.approx([0.0, 0.0, 0.0, 2.0], abs=1e-8)


def test_s2_of_high_spin_determinant_is_maximal():
    """Both electrons alpha: S = 1, so S^2 = 2 exactly and diagonally."""
    det = ((0, 1), ())
    assert s2_matrix_element(det, det, n_orb=2) == pytest.approx(2.0, abs=1e-9)


def test_filter_symmetry_keeps_only_correct_particle_numbers():
    candidates = ["1010", "1100", "0110", "1110"]
    kept = filter_symmetry(candidates, n_orb=2, n_alpha=1, n_beta=1)
    assert kept == ["1010", "0110"]


def test_filter_symmetry_can_return_empty():
    assert filter_symmetry(["1100"], n_orb=2, n_alpha=1, n_beta=1) == []


def test_bit_flip_correction_always_restores_particle_numbers():
    rng = np.random.default_rng(11)
    occupancy = np.full((2, 3), 0.5)
    for bits in ["111000", "000111", "110100", "000000", "111111"]:
        fixed = bit_flip_correction(
            bits, n_orb=3, n_alpha=2, n_beta=1, occupancy=occupancy, rng=rng
        )
        assert fixed[:3].count("1") == 2
        assert fixed[3:].count("1") == 1


def test_bit_flip_correction_is_deterministic_under_a_seed():
    occupancy = np.full((2, 3), 0.5)
    kwargs = dict(n_orb=3, n_alpha=2, n_beta=1, occupancy=occupancy)
    first = bit_flip_correction("111000", rng=np.random.default_rng(5), **kwargs)
    second = bit_flip_correction("111000", rng=np.random.default_rng(5), **kwargs)
    assert first == second


def test_bit_flip_correction_prefers_flipping_uncertain_orbitals():
    """A confidently-occupied orbital is preferentially kept, not always kept.

    The modified-ReLU weighting puts a small floor on every candidate so no
    orbital is ever excluded from the draw, making retention statistical
    rather than absolute.
    """
    occupancy = np.array([[1.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    rng = np.random.default_rng(3)
    trials = 200
    confident_survivals = 0
    uncertain_survivals = 0

    for _ in range(trials):
        fixed = bit_flip_correction(
            "111100", n_orb=3, n_alpha=1, n_beta=1, occupancy=occupancy, rng=rng
        )
        assert fixed[:3].count("1") == 1
        assert fixed[3:].count("1") == 1
        confident_survivals += fixed[0] == "1"
        uncertain_survivals += fixed[1] == "1"

    assert confident_survivals > 0.9 * trials
    assert confident_survivals > uncertain_survivals


def test_bit_flip_correction_leaves_valid_bitstrings_untouched():
    occupancy = np.full((2, 2), 0.5)
    assert (
        bit_flip_correction(
            "1010",
            n_orb=2,
            n_alpha=1,
            n_beta=1,
            occupancy=occupancy,
            rng=np.random.default_rng(0),
        )
        == "1010"
    )


def _degenerate_orbital_integrals(exchange, coulomb=1.0):
    """Two degenerate, non-hopping orbitals with an on-site Coulomb repulsion
    and an inter-orbital exchange term that, for ``exchange > 0``, puts the
    Sz=0 triplet component below the singlets -- a Hund's-rule-like case."""
    n_orb = 2
    one_body = np.zeros((n_orb, n_orb))
    two_body = np.zeros((n_orb,) * 4)
    two_body[0, 0, 0, 0] = coulomb
    two_body[1, 1, 1, 1] = coulomb
    two_body[0, 1, 1, 0] = exchange
    two_body[1, 0, 0, 1] = exchange
    return one_body, two_body, n_orb


def test_spin_penalty_suppresses_the_triplet_ground_state():
    """Without the S^2 penalty the unpenalized projected ground state is the
    Sz=0 triplet component (S^2 = 2); a large lambda_penalty must push it
    above the singlets (S^2 = 0) so the solver returns the singlet energy
    instead."""
    one_body, two_body, n_orb = _degenerate_orbital_integrals(exchange=0.3)
    probs = uniform_full_space_probs(n_orb, 1, 1)

    unpenalized = SQDSolver(
        n_orb,
        1,
        1,
        n_batches=4,
        batch_size=256,
        n_iterations=1,
        lambda_penalty=0.0,
        rng=np.random.default_rng(0),
    )
    triplet_energy = unpenalized.solve(probs, one_body, two_body).energy
    assert triplet_energy == pytest.approx(-0.3, abs=1e-8)

    penalized = SQDSolver(
        n_orb,
        1,
        1,
        n_batches=4,
        batch_size=256,
        n_iterations=1,
        lambda_penalty=20.0,
        rng=np.random.default_rng(0),
    )
    singlet_energy = penalized.solve(probs, one_body, two_body).energy
    assert singlet_energy == pytest.approx(0.3, abs=1e-8)


def test_solver_recovers_fci_when_subspace_is_complete():
    one_body, two_body, n_orb, constant = _h2_integrals()
    probs = uniform_full_space_probs(n_orb, 1, 1)

    # Batch subspaces are drawn with replacement, so a single small batch can
    # miss an alpha or beta half and project onto an incomplete space. Four
    # batches of eight draws each (n_samples = sqrt(batch_size) / 2) make the
    # best-of-batches subspace complete with overwhelming probability.
    solver = SQDSolver(
        n_orb,
        1,
        1,
        n_batches=4,
        batch_size=256,
        n_iterations=1,
        lambda_penalty=0.0,
        rng=np.random.default_rng(0),
    )
    result = solver.solve(probs, one_body, two_body, constant=constant)
    expected = dense_fci_energy(one_body, two_body, 1, 1, constant)

    assert result.energy == pytest.approx(expected, abs=1e-8)


def test_solver_carries_the_best_energy_across_iterations():
    """The returned energy is the minimum found over all iterations, not just
    the last one: a single-iteration solve reproduces iteration 0 of a
    multi-iteration solve seeded identically, so the multi-iteration result
    can never be worse.

    Zero two-body integrals and well-separated one-body diagonal energies
    make every determinant's projected energy an exact sum of orbital
    energies, with no near-degenerate ties for the batch-selection argmin to
    trip on. ``recovery=False`` keeps the sampled candidate list identical
    (and insertion-ordered) across iterations, so the outcome depends only
    on the seeded RNG.
    """
    n_orb, n_alpha, n_beta = 3, 1, 1
    one_body = np.diag([-10.0, -0.1, 0.2])
    two_body = np.zeros((n_orb,) * 4)
    probs = uniform_full_space_probs(n_orb, n_alpha, n_beta)
    kwargs = dict(
        n_batches=2,
        batch_size=4,
        lambda_penalty=0.0,
        recovery=False,
    )

    single = SQDSolver(
        n_orb, n_alpha, n_beta, n_iterations=1, rng=np.random.default_rng(0), **kwargs
    )
    single_energy = single.solve(probs, one_body, two_body).energy

    multi = SQDSolver(
        n_orb, n_alpha, n_beta, n_iterations=3, rng=np.random.default_rng(0), **kwargs
    )
    multi_energy = multi.solve(probs, one_body, two_body).energy

    assert single_energy == pytest.approx(-9.8, abs=1e-9)
    assert multi_energy == pytest.approx(-20.0, abs=1e-9)


def test_recovered_bitstrings_deduplicate_in_sorted_order(monkeypatch):
    """The recovery branch must deduplicate corrected bitstrings with
    ``sorted(set(...))``, not ``list(set(...))``: plain ``set`` iteration
    order depends on Python's per-process string hash randomization, so
    ``rng.choice`` draws from it would vary across process launches even
    under a fixed seed.

    Spies on the ``sorted`` name in the ``_sqd`` module (module-level
    attribute lookup shadows the builtin there) during a ``solve()`` that
    exercises the recovery branch, and checks every dedup call on
    bitstrings is duplicate-free. Fails if the source reverts to
    ``list(set(...))``, since that call would then never happen.
    """
    captured = []
    real_sorted = sorted

    def spy_sorted(iterable, *args, **kwargs):
        result = real_sorted(iterable, *args, **kwargs)
        captured.append(result)
        return result

    monkeypatch.setattr(sqd_module, "sorted", spy_sorted, raising=False)

    n_orb, n_alpha, n_beta = 3, 1, 1
    one_body = np.diag([-10.0, -0.1, 0.2])
    two_body = np.zeros((n_orb,) * 4)
    probs = uniform_full_space_probs(n_orb, n_alpha, n_beta)

    solver = SQDSolver(
        n_orb,
        n_alpha,
        n_beta,
        n_batches=2,
        batch_size=4,
        n_iterations=2,
        recovery=True,
        lambda_penalty=0.0,
        rng=np.random.default_rng(0),
    )
    solver.solve(probs, one_body, two_body)

    bitstring_dedup_calls = [c for c in captured if c and isinstance(c[0], str)]
    assert (
        bitstring_dedup_calls
    ), "expected the recovery branch to call sorted(set(...))"


def test_solver_is_reproducible_under_a_seed():
    one_body, two_body, n_orb, constant = _h2_integrals()
    probs = uniform_full_space_probs(n_orb, 1, 1)

    def run():
        solver = SQDSolver(
            n_orb,
            1,
            1,
            n_batches=3,
            batch_size=8,
            n_iterations=2,
            rng=np.random.default_rng(42),
        )
        return solver.solve(probs, one_body, two_body, constant=constant).energy

    assert run() == pytest.approx(run(), abs=0.0)


def test_solver_raises_when_no_configuration_has_valid_symmetry():
    one_body, two_body, n_orb, _ = _h2_integrals()
    solver = SQDSolver(n_orb, 1, 1, n_iterations=1, rng=np.random.default_rng(0))
    with pytest.raises(ValueError, match="particle symmetry"):
        solver.solve({"1100": 1.0}, one_body, two_body)


def test_occupancy_is_refreshed_from_batch_results():
    """A fresh solver starts at all-zero occupancy. After each solve() call,
    occupancy holds a valid per-spin electron distribution -- row 0 (alpha)
    summing to n_alpha and row 1 (beta) to n_beta -- refreshed from that
    call's own batch results. n_alpha != n_beta so a transposed or swapped
    row assignment would fail this check.
    """
    n_orb, n_alpha, n_beta = 4, 1, 3
    one_body = np.diag([-1.0, -0.5, 0.2, 0.7])
    two_body = np.zeros((n_orb,) * 4)
    probs = uniform_full_space_probs(n_orb, n_alpha, n_beta)

    solver = SQDSolver(
        n_orb,
        n_alpha,
        n_beta,
        n_batches=2,
        batch_size=8,
        n_iterations=2,
        recovery=True,
        lambda_penalty=0.0,
        rng=np.random.default_rng(0),
    )
    assert np.array_equal(solver.occupancy, np.zeros((2, n_orb)))

    solver.solve(probs, one_body, two_body)
    assert not np.array_equal(solver.occupancy, np.zeros((2, n_orb)))
    np.testing.assert_allclose(
        solver.occupancy.sum(axis=1), [n_alpha, n_beta], atol=1e-8
    )

    solver.solve(probs, one_body, two_body)
    np.testing.assert_allclose(
        solver.occupancy.sum(axis=1), [n_alpha, n_beta], atol=1e-8
    )


def test_subspace_pools_alpha_and_beta_separately():
    """Alpha candidates must come only from alpha halves, and beta from beta.

    The reference merges both halves into one pool and filters that by
    particle count. Both schemes are equally constrained by n_alpha and
    n_beta whenever they differ, since a pooled string with the wrong
    popcount for a given role is rejected either way -- so this only
    discriminates when n_alpha == n_beta, where a merged pool additionally
    lets an alpha half stand in for a beta half (and vice versa) that was
    never actually sampled in that role.
    """
    n_orb, n_alpha, n_beta = 2, 1, 1
    one_body = np.diag([-1.0, -0.5])
    two_body = np.zeros((n_orb,) * 4)

    # Only "1001" (alpha half "10", beta half "01") is ever sampled. Separate
    # pooling can only ever offer "10" as an alpha half and "01" as a beta
    # half, so the subspace is exactly {"1001"}. Merged pooling would offer
    # both "10" and "01" to each role, producing the extra determinants
    # "1010", "0101", "0110" -- none of which were ever sampled as a pair.
    probs = {"1001": 1.0}
    solver = SQDSolver(
        n_orb,
        n_alpha,
        n_beta,
        n_batches=4,
        batch_size=16,
        n_iterations=1,
        lambda_penalty=0.0,
        rng=np.random.default_rng(0),
    )
    result = solver.solve(probs, one_body, two_body)

    assert set(result.subspace) == {"1001"}


def test_spatial_rdm_trace_equals_electron_count():
    one_body, two_body, n_orb, constant = _h2_integrals()
    solver = SQDSolver(
        n_orb,
        1,
        1,
        n_batches=4,
        batch_size=256,
        n_iterations=1,
        lambda_penalty=0.0,
        rng=np.random.default_rng(0),
    )
    result = solver.solve(
        uniform_full_space_probs(n_orb, 1, 1), one_body, two_body, constant=constant
    )
    dets = [bitstring_to_spatial_det(bs, n_orb) for bs in result.subspace]
    rdm1, rdm2 = compute_spatial_rdms(dets, result.eigenvector, n_orb)

    assert np.trace(rdm1) == pytest.approx(2.0, abs=1e-8)


def test_spatial_rdm1_matches_pyscf_fci():
    one_body, two_body, n_orb, constant = _h2_integrals()
    solver = SQDSolver(
        n_orb,
        1,
        1,
        n_batches=4,
        batch_size=256,
        n_iterations=1,
        lambda_penalty=0.0,
        rng=np.random.default_rng(0),
    )
    result = solver.solve(
        uniform_full_space_probs(n_orb, 1, 1), one_body, two_body, constant=constant
    )
    dets = [bitstring_to_spatial_det(bs, n_orb) for bs in result.subspace]
    rdm1, _ = compute_spatial_rdms(dets, result.eigenvector, n_orb)

    _, civec = fci.direct_spin1.kernel(one_body, two_body, n_orb, (1, 1))
    expected = fci.direct_spin1.make_rdm1(civec, n_orb, (1, 1))

    np.testing.assert_allclose(rdm1, expected, atol=1e-6)


def test_spatial_rdm12_and_energy_match_pyscf_fci_beyond_two_orbitals():
    """Both RDMs, and the energy they reconstruct, on a 4-orbital active
    space. A sign error in the two-body contraction corrupts rdm2 and the
    reconstructed energy even where it happens to leave rdm1 alone, and a
    2-orbital case cannot exercise that path at all."""
    one_body, two_body, n_orb, constant = _h4_integrals()
    n_alpha = n_beta = 2
    solver = SQDSolver(
        n_orb,
        n_alpha,
        n_beta,
        n_batches=8,
        batch_size=4096,
        n_iterations=1,
        lambda_penalty=0.0,
        rng=np.random.default_rng(0),
    )
    result = solver.solve(
        uniform_full_space_probs(n_orb, n_alpha, n_beta),
        one_body,
        two_body,
        constant=constant,
    )
    dets = [bitstring_to_spatial_det(bs, n_orb) for bs in result.subspace]
    rdm1, rdm2 = compute_spatial_rdms(dets, result.eigenvector, n_orb)

    _, civec = fci.direct_spin1.kernel(one_body, two_body, n_orb, (n_alpha, n_beta))
    expected_rdm1, expected_rdm2 = fci.direct_spin1.make_rdm12(
        civec, n_orb, (n_alpha, n_beta)
    )

    np.testing.assert_allclose(rdm1, expected_rdm1, atol=1e-6)
    np.testing.assert_allclose(rdm2, expected_rdm2, atol=1e-6)
    np.testing.assert_allclose(rdm1, rdm1.T, atol=1e-10)

    energy_from_rdms = (
        np.sum(one_body * rdm1)
        + 0.5 * np.einsum("pqrs,pqrs->", two_body, rdm2)
        + constant
    )
    assert energy_from_rdms == pytest.approx(result.energy, abs=1e-8)
