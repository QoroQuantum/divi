# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for SQD post-processing, checked against exact classical oracles."""

import functools
import itertools

import numpy as np
import pytest

pytest.importorskip("pyscf")

from pyscf import fci, scf

import divi.qprog.workflows._lassqd._sqd as sqd_module
from divi.qprog.workflows._lassqd._config import SQDConfig
from divi.qprog.workflows._lassqd._sqd import (
    SQDSolver,
    _annihilation_sign,
    _bit_matrix,
    _correct_spin_part,
    _creation_sign,
    _heaviest_strings,
    _modified_relu,
    _occupations_from_bit_matrix,
    _spatial_rdms_exact,
    bit_flip_correction,
    bitstring_to_spatial_det,
    carryover_weights,
    ci_string_to_int,
    compute_spatial_rdms,
    filter_symmetry,
    projected_matrices,
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
    mo_integrals,
    uniform_full_space_probs,
)


@functools.cache
def _mean_field(system):
    """RHF solved once per system; callers get fresh integral arrays each time."""
    molecule = {
        "h2": h2_molecule,
        "h2_631g": lambda: h2_molecule(basis="6-31g"),
        "h4": h4_chain,
    }[system]()
    return scf.RHF(molecule).run(verbose=0)


def _h2_integrals():
    return mo_integrals(_mean_field("h2"))


def _h2_631g_integrals():
    return mo_integrals(_mean_field("h2_631g"))


def _h4_integrals():
    return mo_integrals(_mean_field("h4"))


def _solver(n_orb, n_alpha, n_beta, **overrides):
    """An ``SQDSolver`` on ``SQDConfig``'s defaults, seeded, and conventional SQD
    unless ``carryover_cutoff`` is overridden."""
    config = SQDConfig()
    settings = dict(
        n_batches=config.n_batches,
        batch_size=config.batch_size,
        n_iterations=config.n_recovery_iterations,
        lambda_penalty=config.lambda_penalty,
        recovery=True,
        carryover_cutoff=None,
        max_carryover=config.max_carryover,
        max_dim=config.max_dim,
        include_reference=config.include_reference,
        symmetrize_spin=config.symmetrize_spin,
        energy_tol=config.recovery_energy_tol,
        occupancies_tol=config.recovery_occupancies_tol,
        rng=np.random.default_rng(0),
    )
    return SQDSolver(n_orb, n_alpha, n_beta, **(settings | overrides))


def _sector_occupations(strings, n_orb):
    return _occupations_from_bit_matrix(_bit_matrix(strings, n_orb))


def test_solver_settings_have_no_defaults():
    """Defaults live only in ``SQDConfig``, so a solver built without them must
    fail rather than silently diverge from the configuration."""
    with pytest.raises(TypeError, match="required keyword-only"):
        SQDSolver(2, 1, 1, rng=np.random.default_rng(0))


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


@pytest.mark.parametrize("n_orb", [2, 70])
def test_sector_occupations_matches_per_string_decode(n_orb):
    """A sector's occupations equal decoding each string on its own.

    Determinants are the product of two sectors, so the per-string decode runs
    once per sector string here instead of once per determinant.
    """
    rng = np.random.default_rng(4)
    strings = ["".join(rng.choice(["0", "1"], n_orb)) for _ in range(6)]
    strings.append("0" * n_orb)
    strings.append("1" * n_orb)

    expected = [
        bitstring_to_spatial_det(half + "0" * n_orb, n_orb)[0] for half in strings
    ]
    assert _sector_occupations(strings, n_orb) == expected


@pytest.mark.parametrize(
    ("consumer", "args"),
    [
        pytest.param(
            _sector_occupations,
            (["1010", "10", "101010"], 4),
            id="occupations",
        ),
        pytest.param(
            filter_symmetry,
            (["1010", "10100", "101"], 2, 1, 1),
            id="symmetry",
        ),
    ],
)
def test_ragged_strings_are_rejected_not_regrouped(consumer, args):
    """Mixed widths summing to the expected total must raise.

    Reshaping alone would produce rows straddling two strings, silently
    filtering or occupying the wrong orbitals.
    """
    with pytest.raises(ValueError, match="got one of width"):
        consumer(*args)


def test_non_binary_sector_string_is_rejected():
    """Packed sector decoding must not treat arbitrary character bytes as bits."""
    with pytest.raises(ValueError, match="non-binary"):
        _sector_occupations(["1200"], n_orb=4)


def test_slater_condon_vanishes_beyond_double_excitation():
    # Only 2 spatial orbitals available in H2/STO-3G, so build a 3-orbital toy instead.
    # Integrals are non-zero so dropping the beyond-double guard would surface
    # as a non-zero element (or a failed unpacking) here.
    n = 3
    h3 = np.zeros((n, n))
    g3 = np.arange(1, n**4 + 1, dtype=float).reshape((n,) * 4) * 0.01
    h3[np.diag_indices(n)] = [1.0, 2.0, 3.0]
    hs, gs = spin_orbital_integrals(h3, g3, n)
    det_i = spatial_to_spin_occupations((0, 1), (0, 1), n)
    det_j = spatial_to_spin_occupations((2,), (2,), n)  # quadruple excitation
    assert slater_condon(det_i, det_j, hs, gs) == pytest.approx(0.0)


@pytest.mark.parametrize(
    "integrals_fn, n_alpha, n_beta",
    [
        pytest.param(_h2_integrals, 1, 1, id="h2-sto3g"),
        # Beyond two spatial orbitals a double-excitation sign error is no
        # longer an exact gauge transformation and must change the energy.
        pytest.param(_h2_631g_integrals, 1, 1, id="h2-631g"),
        pytest.param(_h4_integrals, 2, 2, id="h4"),
    ],
)
def test_full_subspace_diagonalization_reproduces_fci(integrals_fn, n_alpha, n_beta):
    """The strongest single check: Slater-Condon over the COMPLETE determinant
    space is symmetric and gives exactly the FCI energy."""
    one_body, two_body, n_orb, constant = integrals_fn()
    h_spin, g_spin = spin_orbital_integrals(one_body, two_body, n_orb)
    dets = [
        spatial_to_spin_occupations(a, b, n_orb)
        for a in itertools.combinations(range(n_orb), n_alpha)
        for b in itertools.combinations(range(n_orb), n_beta)
    ]
    hamiltonian = np.array(
        [[slater_condon(d_i, d_j, h_spin, g_spin) for d_j in dets] for d_i in dets]
    )

    np.testing.assert_allclose(hamiltonian, hamiltonian.T, atol=1e-12)
    lowest = float(np.min(np.linalg.eigvalsh(hamiltonian))) + constant
    expected = dense_fci_energy(one_body, two_body, n_alpha, n_beta, constant)
    assert lowest == pytest.approx(expected, abs=1e-9)


def _scalar_spin_orbital_integrals(one_body, two_body, n_orb, one_body_beta=None):
    """The elementwise rule ``spin_orbital_integrals`` replaced with blocks."""
    if one_body_beta is None:
        one_body_beta = one_body
    n_spin = 2 * n_orb
    h_spin = np.zeros((n_spin, n_spin))
    h_spin[:n_orb, :n_orb] = one_body
    h_spin[n_orb:, n_orb:] = one_body_beta
    g_spin = np.zeros((n_spin,) * 4)
    for p, q, r, s in itertools.product(range(n_spin), repeat=4):
        p_sp, p_spin = p % n_orb, p // n_orb
        q_sp, q_spin = q % n_orb, q // n_orb
        r_sp, r_spin = r % n_orb, r // n_orb
        s_sp, s_spin = s % n_orb, s // n_orb
        coulomb = (
            two_body[p_sp, r_sp, q_sp, s_sp]
            if (p_spin == r_spin and q_spin == s_spin)
            else 0.0
        )
        exchange = (
            two_body[p_sp, s_sp, q_sp, r_sp]
            if (p_spin == s_spin and q_spin == r_spin)
            else 0.0
        )
        g_spin[p, q, r, s] = coulomb - exchange
    return h_spin, g_spin


@pytest.mark.parametrize("n_orb", [1, 2, 3])
@pytest.mark.parametrize("polarized", [True, False])
def test_spin_orbital_integrals_match_the_elementwise_definition(n_orb, polarized):
    """The block form must reproduce the elementwise spin selection exactly.

    Six block assignments stand in for the four-index loop, so a block written to
    the wrong spin sector -- or a transpose that permutes the wrong pair -- still
    produces a plausibly-shaped tensor. Integrals without the ``(pq|rs)``
    permutation symmetries are what separate the two: a physical ``two_body``
    makes several of the blocks coincide.
    """
    rng = np.random.default_rng(31)
    one_body = rng.normal(size=(n_orb, n_orb))
    one_body = 0.5 * (one_body + one_body.T)
    one_body_beta = rng.normal(size=(n_orb, n_orb)) if polarized else None
    two_body = rng.normal(size=(n_orb,) * 4)

    expected_h, expected_g = _scalar_spin_orbital_integrals(
        one_body, two_body, n_orb, one_body_beta
    )
    h_spin, g_spin = spin_orbital_integrals(one_body, two_body, n_orb, one_body_beta)

    np.testing.assert_array_equal(h_spin, expected_h)
    np.testing.assert_array_equal(g_spin, expected_g)


def _scalar_projected_matrices(dets, dets_spin, h_spin, g_spin):
    """The double loop ``projected_matrices`` replaced, kept as the reference."""
    dim = len(dets)
    h_proj = np.zeros((dim, dim))
    s2_proj = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            h_proj[i, j] = slater_condon(dets_spin[i], dets_spin[j], h_spin, g_spin)
            s2_proj[i, j] = s2_matrix_element(dets[i], dets[j])
    return h_proj, s2_proj


@pytest.mark.parametrize(
    "n_orb, n_alpha, n_beta",
    [
        (2, 1, 1),
        (3, 2, 1),
        (4, 2, 2),
        (4, 3, 1),
        (5, 4, 2),
        (5, 2, 2),
        (6, 3, 3),
        # Fully polarized: no beta electrons, so S^2 sits at its maximum and the
        # spin-exchange branch has nothing to find.
        (4, 3, 0),
        (4, 0, 3),
        # One determinant, where only the diagonal branch runs.
        (1, 1, 1),
    ],
)
@pytest.mark.parametrize("symmetrize", [True, False])
def test_projected_matrices_match_the_scalar_rules(n_orb, n_alpha, n_beta, symmetrize):
    """The vectorized build must agree element-wise with Slater-Condon.

    It reproduces the double loop by gathering each excitation rank at once,
    which means it re-derives the fermionic signs from cumulative occupations
    rather than from sequential annihilation and creation. A sign convention off
    by one still yields a symmetric matrix with plausible eigenvalues, so
    compare against the scalar routines directly over a complete determinant
    space -- every rank, both spin sectors, and both matrices.

    Run with and without the ``(pq|rs)`` permutation symmetries. Physical
    integrals always carry them, which lets a pair sum over ``p < q`` be written
    as half the sum over all pairs -- correct in production and wrong in
    general, so the asymmetric case is what catches that shortcut.
    """
    rng = np.random.default_rng(2024)
    one_body = rng.normal(size=(n_orb, n_orb))
    one_body = 0.5 * (one_body + one_body.T)
    two_body = rng.normal(size=(n_orb,) * 4)
    if symmetrize:
        two_body = 0.25 * (
            two_body
            + two_body.transpose(1, 0, 2, 3)
            + two_body.transpose(0, 1, 3, 2)
            + two_body.transpose(1, 0, 3, 2)
        )
        two_body = 0.5 * (two_body + two_body.transpose(2, 3, 0, 1))
    h_spin, g_spin = spin_orbital_integrals(one_body, two_body, n_orb)

    space = [
        (alpha, beta)
        for alpha in itertools.combinations(range(n_orb), n_alpha)
        for beta in itertools.combinations(range(n_orb), n_beta)
    ]
    dets_spin = [spatial_to_spin_occupations(a, b, n_orb) for a, b in space]

    expected_h, expected_s2 = _scalar_projected_matrices(
        space, dets_spin, h_spin, g_spin
    )
    h_proj, s2_proj = projected_matrices(space, dets_spin, h_spin, g_spin, n_orb)

    np.testing.assert_allclose(h_proj, expected_h, rtol=1e-10, atol=1e-11)
    np.testing.assert_allclose(s2_proj, expected_s2, rtol=1e-10, atol=1e-11)


@pytest.mark.parametrize("block_rows", [1, 3, 7])
def test_projected_matrices_are_independent_of_the_row_block(monkeypatch, block_rows):
    """Rows are processed in blocks to bound peak memory, which must not change
    the result: a block boundary falling inside a rank class would drop the pairs
    that straddle it. Forced small so the boundary is crossed repeatedly."""
    n_orb, n_alpha, n_beta = 4, 2, 2
    rng = np.random.default_rng(7)
    one_body = rng.normal(size=(n_orb, n_orb))
    one_body = 0.5 * (one_body + one_body.T)
    two_body = rng.normal(size=(n_orb,) * 4)
    h_spin, g_spin = spin_orbital_integrals(one_body, two_body, n_orb)

    space = [
        (alpha, beta)
        for alpha in itertools.combinations(range(n_orb), n_alpha)
        for beta in itertools.combinations(range(n_orb), n_beta)
    ]
    dets_spin = [spatial_to_spin_occupations(a, b, n_orb) for a, b in space]
    expected_h, expected_s2 = _scalar_projected_matrices(
        space, dets_spin, h_spin, g_spin
    )

    monkeypatch.setattr(sqd_module, "_PAIR_BLOCK_ROWS", block_rows)
    h_proj, s2_proj = projected_matrices(space, dets_spin, h_spin, g_spin, n_orb)

    np.testing.assert_allclose(h_proj, expected_h, rtol=1e-10, atol=1e-11)
    np.testing.assert_allclose(s2_proj, expected_s2, rtol=1e-10, atol=1e-11)


def _penalized_subspace(n_orb=4, n_alpha=2, n_beta=2, seed=17):
    """A projected Hamiltonian and its ``S^2`` deviation over a full space."""
    rng = np.random.default_rng(seed)
    one_body = rng.normal(size=(n_orb, n_orb))
    one_body = 0.5 * (one_body + one_body.T)
    # Every ``(pq|rs)`` permutation symmetry, since without them the projected
    # Hamiltonian is not symmetric and the two solvers disagree by construction:
    # LAPACK reads one triangle where the operator form reads the whole matrix.
    two_body = rng.normal(size=(n_orb,) * 4)
    two_body = 0.25 * (
        two_body
        + two_body.transpose(1, 0, 2, 3)
        + two_body.transpose(0, 1, 3, 2)
        + two_body.transpose(1, 0, 3, 2)
    )
    two_body = 0.5 * (two_body + two_body.transpose(2, 3, 0, 1))
    h_spin, g_spin = spin_orbital_integrals(one_body, two_body, n_orb)

    space = [
        (alpha, beta)
        for alpha in itertools.combinations(range(n_orb), n_alpha)
        for beta in itertools.combinations(range(n_orb), n_beta)
    ]
    dets_spin = [spatial_to_spin_occupations(a, b, n_orb) for a, b in space]
    h_proj, s2_proj = projected_matrices(space, dets_spin, h_spin, g_spin, n_orb)
    target_s = 0.5 * abs(n_alpha - n_beta)
    deviation = s2_proj - target_s * (target_s + 1.0) * np.eye(len(space))
    return h_proj, deviation


def test_ground_root_iterative_branch_matches_the_dense_solve(monkeypatch):
    """Above the threshold the penalty is applied as an operator instead of being
    squared into a dense matrix, and the root comes from Lanczos rather than
    LAPACK. Both must land on the same eigenpair, so the threshold is forced down
    onto a subspace small enough to diagonalize densely for the comparison."""
    h_proj, deviation = _penalized_subspace()
    lambda_penalty = 0.2
    expected_values, expected_vectors = np.linalg.eigh(
        h_proj + lambda_penalty * (deviation @ deviation)
    )

    monkeypatch.setattr(sqd_module, "_ITERATIVE_SUBSPACE_MIN", 1)
    energy, vector = sqd_module.ground_root(h_proj, deviation, lambda_penalty)

    assert energy == pytest.approx(expected_values[0], abs=1e-10)
    # The sign of an eigenvector is arbitrary in either solver.
    assert abs(float(vector @ expected_vectors[:, 0])) == pytest.approx(1.0, abs=1e-8)


def test_ground_root_iterative_branch_lets_the_penalty_choose_the_root(
    mocker, monkeypatch
):
    """With the penalty applied as an operator, the iterative branch must still
    move the root: unpenalised the Sz=0 triplet component is lowest, penalised
    the singlet is. Neither solve may fall back to the dense form."""
    one_body, two_body, n_orb = _degenerate_orbital_integrals(exchange=0.3)
    h_spin, g_spin = spin_orbital_integrals(one_body, two_body, n_orb)
    space = [((a,), (b,)) for a in range(n_orb) for b in range(n_orb)]
    dets_spin = [spatial_to_spin_occupations(a, b, n_orb) for a, b in space]
    h_proj, deviation = projected_matrices(space, dets_spin, h_spin, g_spin, n_orb)

    monkeypatch.setattr(sqd_module, "_ITERATIVE_SUBSPACE_MIN", 1)
    dense = mocker.spy(sqd_module.scipy.linalg, "eigh")

    triplet, _ = sqd_module.ground_root(h_proj, deviation, 0.0)
    singlet, vector = sqd_module.ground_root(h_proj, deviation, 20.0)

    dense.assert_not_called()
    assert triplet == pytest.approx(-0.3, abs=1e-8)
    assert singlet == pytest.approx(0.3, abs=1e-8)
    assert float(vector @ deviation @ vector) == pytest.approx(0.0, abs=1e-8)


def test_ground_root_falls_back_when_the_iterative_root_is_not_the_lowest(
    mocker, monkeypatch
):
    """Lanczos can converge on an interior root, which the dense solve cannot.

    The smallest diagonal entry bounds the true lowest eigenvalue from above, so
    a root above it is provably not the ground state and must be discarded
    rather than reported as the fragment's energy.
    """
    h_proj, deviation = _penalized_subspace()
    lambda_penalty = 0.2
    penalized = h_proj + lambda_penalty * (deviation @ deviation)
    expected = np.linalg.eigvalsh(penalized)[0]
    interior = float(np.diag(penalized).min()) + 1.0
    assert expected < interior

    monkeypatch.setattr(sqd_module, "_ITERATIVE_SUBSPACE_MIN", 1)
    mocker.patch.object(
        sqd_module.scipy.sparse.linalg,
        "eigsh",
        return_value=(
            np.array([interior, interior + 1.0]),
            np.eye(h_proj.shape[0], 2),
        ),
    )

    energy, _ = sqd_module.ground_root(h_proj, deviation, lambda_penalty)

    assert energy == pytest.approx(expected, abs=1e-12)


def test_ground_root_falls_back_when_the_iterative_solve_fails(mocker, monkeypatch):
    """Lanczos can exhaust its budget on a subspace whose lowest roots are nearly
    degenerate, which must cost accuracy rather than the solve."""
    h_proj, deviation = _penalized_subspace()
    lambda_penalty = 0.2
    expected = np.linalg.eigvalsh(h_proj + lambda_penalty * (deviation @ deviation))[0]

    monkeypatch.setattr(sqd_module, "_ITERATIVE_SUBSPACE_MIN", 1)
    failing = mocker.patch.object(
        sqd_module.scipy.sparse.linalg,
        "eigsh",
        side_effect=sqd_module.scipy.sparse.linalg.ArpackError(-1),
    )

    energy, vector = sqd_module.ground_root(h_proj, deviation, lambda_penalty)

    failing.assert_called_once()
    assert energy == pytest.approx(expected, abs=1e-12)
    assert vector.shape == (h_proj.shape[0],)


def test_projected_matrices_handles_an_empty_subspace():
    one_body, two_body, n_orb, _ = _h2_integrals()
    h_spin, g_spin = spin_orbital_integrals(one_body, two_body, n_orb)
    h_proj, s2_proj = projected_matrices([], [], h_spin, g_spin, n_orb)
    assert h_proj.shape == (0, 0)
    assert s2_proj.shape == (0, 0)


def _s2_eigenvalues(dets):
    dim = len(dets)
    matrix = np.array(
        [[s2_matrix_element(dets[i], dets[j]) for j in range(dim)] for i in range(dim)]
    )
    return np.linalg.eigvalsh(matrix), matrix


def test_s2_eigenvalues_are_singlet_and_triplet():
    """Two electrons in two orbitals span one triplet (S^2 = 2) and singlets (0)."""
    dets = [
        ((0,), (0,)),
        ((0,), (1,)),
        ((1,), (0,)),
        ((1,), (1,)),
    ]
    eigenvalues, matrix = _s2_eigenvalues(dets)
    np.testing.assert_allclose(matrix, matrix.T, atol=1e-12)
    rounded = np.round(eigenvalues, 9)
    assert sorted(rounded) == pytest.approx([0.0, 0.0, 0.0, 2.0], abs=1e-8)


def test_s2_of_high_spin_determinant_is_maximal():
    """Both electrons alpha: S = 1, so S^2 = 2 exactly and diagonally."""
    det = ((0, 1), ())
    assert s2_matrix_element(det, det) == pytest.approx(2.0, abs=1e-9)


@pytest.mark.parametrize(
    "candidates, kept",
    [
        (["1010", "1100", "0110", "1110"], ["1010", "0110"]),
        (["1100"], []),
    ],
)
def test_filter_symmetry_keeps_only_correct_particle_numbers(candidates, kept):
    assert filter_symmetry(candidates, n_orb=2, n_alpha=1, n_beta=1) == kept


def test_bit_flip_correction_always_restores_particle_numbers():
    rng = np.random.default_rng(11)
    occupancy = np.full((2, 3), 0.5)
    for bits in ["111000", "000111", "110100", "000000", "111111"]:
        fixed = bit_flip_correction(
            bits, n_orb=3, n_alpha=2, n_beta=1, occupancy=occupancy, rng=rng
        )
        assert set(fixed) <= {"0", "1"}
        assert len(fixed) == 6
        assert fixed[:3].count("1") == 2
        assert fixed[3:].count("1") == 1


@pytest.mark.parametrize(
    "n_alpha, n_beta, match",
    [(-1, 1, "n_alpha"), (3, 1, "n_alpha"), (1, -1, "n_beta"), (1, 3, "n_beta")],
)
def test_bit_flip_correction_rejects_impossible_sector_counts(n_alpha, n_beta, match):
    with pytest.raises(ValueError, match=f"{match} must be between 0 and n_orb"):
        bit_flip_correction(
            "1010",
            n_orb=2,
            n_alpha=n_alpha,
            n_beta=n_beta,
            occupancy=np.full((2, 2), 0.5),
            rng=np.random.default_rng(0),
        )


def test_bit_flip_correction_accepts_a_full_sector():
    """``n_orb`` electrons per sector is the bound itself, not past it."""
    fixed = bit_flip_correction(
        "0000",
        n_orb=2,
        n_alpha=2,
        n_beta=0,
        occupancy=np.full((2, 2), 0.5),
        rng=np.random.default_rng(0),
    )
    assert fixed == "1100"


@pytest.mark.parametrize(
    "distance, threshold, expected",
    [(0.2, 0.5, 0.01), (0.5, 0.5, 0.01), (0.8, 0.5, 0.31)],
)
def test_modified_relu_is_the_published_flip_weight(distance, threshold, expected):
    """arXiv:2405.05068: ``delta`` up to the threshold, then linear above it."""
    assert _modified_relu(distance, threshold, 0.01) == pytest.approx(expected)


@pytest.mark.parametrize(
    "part, target, average, candidates, weights, flipped, corrected",
    [
        # Emptying: distance is how far each occupied bit sits from full.
        (
            [1, 1, 1],
            1,
            [1.0, 0.5, 0.2],
            [0, 1, 2],
            [0.01, 0.5 - 1 / 3 + 0.01, 0.8 - 1 / 3 + 0.01],
            [0, 1],
            [0, 0, 1],
        ),
        # Filling: distance is how far each empty bit sits from empty.
        (
            [0, 0, 1],
            2,
            [0.9, 0.1, 1.0],
            [0, 1],
            [0.9 - 2 / 3 + 0.01, 0.01],
            [0],
            [1, 0, 1],
        ),
    ],
)
def test_bit_flip_draws_follow_the_modified_relu_weights(
    mocker, part, target, average, candidates, weights, flipped, corrected
):
    """The draw's probabilities are the normalised flip weights, and exactly the
    surplus or deficit is flipped."""
    rng = mocker.Mock(spec=np.random.Generator)
    rng.choice.return_value = np.array(flipped)

    result = _correct_spin_part(list(part), target, np.array(average), 3, rng)

    rng.choice.assert_called_once()
    assert list(rng.choice.call_args.args[0]) == candidates
    call = rng.choice.call_args.kwargs
    assert call["size"] == len(flipped)
    assert call["replace"] is False
    np.testing.assert_allclose(call["p"], np.array(weights) / sum(weights))
    assert result == corrected


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

    unpenalized = _solver(
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

    penalized = _solver(
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


def test_spin_penalty_targets_a_polarised_sector_s_squared():
    """A large penalty must still land on the sector's exact ground state, which
    it only does if the penalty's target spin is the sector's own ``S``."""
    one_body, two_body, n_orb, _ = _h4_integrals()
    solver = _solver(
        n_orb, 3, 1, n_batches=1, batch_size=64, n_iterations=1, lambda_penalty=50.0
    )

    result = solver.solve(uniform_full_space_probs(n_orb, 3, 1), one_body, two_body)

    assert result.energy == pytest.approx(
        dense_fci_energy(one_body, two_body, 3, 1), abs=1e-8
    )


def test_beta_one_body_applies_to_the_beta_channel():
    """``one_body_beta`` must reach the beta spin-orbitals, not the alpha ones.

    SQD blocks its spin-orbitals -- index ``p < n_orb`` is alpha -- which is the
    opposite of the circuits' interleaved order, so a mix-up here is plausible
    and silent: particle number and Sz are unchanged by swapping the channels.

    With no two-body term and diagonal one-body matrices, every determinant is
    an eigenstate, so the ``(2 alpha, 1 beta)`` sector energy is exact by hand:
    alpha must fill both orbitals (``-0.9 + 0.4``) and beta takes the lower of
    its own diagonal (``-0.6``), totalling ``-1.1``. Swapping the channels would
    give ``(-0.2 - 0.6) + (-0.9) = -1.7`` instead.
    """
    n_orb, n_alpha, n_beta = 2, 2, 1
    h_alpha = np.diag([-0.9, 0.4])
    h_beta = np.diag([-0.2, -0.6])
    two_body = np.zeros((n_orb,) * 4)
    probs = uniform_full_space_probs(n_orb, n_alpha, n_beta)

    solver = _solver(
        n_orb,
        n_alpha,
        n_beta,
        n_batches=1,
        batch_size=8,
        n_iterations=1,
        lambda_penalty=0.0,
        rng=np.random.default_rng(0),
    )
    result = solver.solve(probs, h_alpha, two_body, one_body_beta=h_beta)

    assert result.energy == pytest.approx(-1.1, abs=1e-9)


@pytest.mark.parametrize(
    "batch_size,expected_subspace",
    [(1, 1), (2, 4), (8, 4)],
)
def test_batch_size_sets_the_number_of_samples_drawn(batch_size, expected_subspace):
    """``batch_size`` is the per-batch sample count, so the recovered subspace
    grows with it until it saturates the space.

    A two-orbital ``(1, 1)`` space spans four determinants. One draw can only
    ever yield one, while two or more saturate it. This discriminates the
    documented behavior from the ``n_samples = sqrt(batch_size) / 2`` scaling
    it replaced, under which ``batch_size=8`` drew a single sample and left the
    subspace at one determinant -- the mean-field answer, for any budget.

    ``include_reference=False`` so the subspace is exactly what was drawn: the
    reference determinant would otherwise contribute a half of its own and the
    single-draw case could no longer land on one determinant.
    """
    one_body, two_body, n_orb, constant = _h2_integrals()
    probs = uniform_full_space_probs(n_orb, 1, 1)

    solver = _solver(
        n_orb,
        1,
        1,
        n_batches=1,
        batch_size=batch_size,
        n_iterations=1,
        lambda_penalty=0.0,
        include_reference=False,
        rng=np.random.default_rng(0),
    )
    result = solver.solve(probs, one_body, two_body, constant=constant)

    assert len(set(result.subspace)) == expected_subspace


def test_solver_carries_the_best_energy_across_iterations():
    """The returned energy is the minimum found over all iterations, not just
    the last one: a single-iteration solve reproduces iteration 0 of a
    multi-iteration solve seeded identically, so the multi-iteration result
    can never be worse.

    Zero two-body integrals and well-separated one-body diagonal energies
    make every determinant's projected energy an exact sum of orbital
    energies, with no near-degenerate ties for the batch-selection argmin to
    trip on. ``recovery=False`` keeps the sampled candidate list identical
    across iterations, so the outcome depends only on the seeded RNG.

    ``batch_size=1`` is load-bearing: 2 or more saturates this 9-determinant
    space in one iteration, so both solves hit the exact ground state and the
    comparison stops discriminating. So is ``include_reference=False``: the
    reference determinant here *is* the ground state, so including it would hand
    both solves the answer on iteration zero.
    """
    n_orb, n_alpha, n_beta = 3, 1, 1
    one_body = np.diag([-10.0, -0.1, 0.2])
    two_body = np.zeros((n_orb,) * 4)
    probs = uniform_full_space_probs(n_orb, n_alpha, n_beta)
    kwargs = dict(
        n_batches=2,
        batch_size=1,
        lambda_penalty=0.0,
        recovery=False,
        include_reference=False,
    )

    single = _solver(
        n_orb, n_alpha, n_beta, n_iterations=1, rng=np.random.default_rng(0), **kwargs
    )
    single_energy = single.solve(probs, one_body, two_body).energy

    multi = _solver(
        n_orb, n_alpha, n_beta, n_iterations=3, rng=np.random.default_rng(0), **kwargs
    )
    multi_energy = multi.solve(probs, one_body, two_body).energy

    # One draw lands a single determinant (orbitals 0 and 1, -10.0 + -0.1);
    # three iterations reach the true ground state with both electrons in
    # orbital 0.
    assert single_energy == pytest.approx(-10.1, abs=1e-9)
    assert multi_energy == pytest.approx(-20.0, abs=1e-9)


def _spy_on_retained(monkeypatch):
    """Capture what ``_heaviest_strings`` decides to keep, per call."""
    kept = []
    real = sqd_module._heaviest_strings

    def spy(weights, limit):
        result = real(weights, limit)
        kept.append(list(result))
        return result

    monkeypatch.setattr(sqd_module, "_heaviest_strings", spy)
    return kept


def _spy_on_dimensions(monkeypatch):
    """Capture the subspace dimension of every projected diagonalization."""
    dimensions = []
    real = sqd_module.projected_matrices

    def spy(dets, dets_spin, *args, **kwargs):
        dimensions.append(len(dets))
        return real(dets, dets_spin, *args, **kwargs)

    monkeypatch.setattr(sqd_module, "projected_matrices", spy)
    return dimensions


def _h4_exact_energy():
    one_body, two_body, _, _ = _h4_integrals()
    return dense_fci_energy(one_body, two_body, 2, 2, 0.0)


def _solve_h4(batch_size, seed, **solver_kwargs):
    """Five-iteration H4 ``(2, 2)`` solve from uniform probabilities, without
    recovery or the reference, so the subspace is only what was drawn or carried."""
    one_body, two_body, n_orb, _ = _h4_integrals()
    return _solver(
        n_orb,
        2,
        2,
        n_batches=1,
        batch_size=batch_size,
        n_iterations=5,
        lambda_penalty=0.0,
        recovery=False,
        include_reference=False,
        rng=np.random.default_rng(seed),
        **solver_kwargs,
    ).solve(uniform_full_space_probs(n_orb, 2, 2), one_body, two_body)


def test_carryover_grows_the_subspace_across_iterations(monkeypatch):
    """Retention must enlarge later subspaces and lower the energy."""
    dimensions = _spy_on_dimensions(monkeypatch)
    energy = _solve_h4(1, seed=3, carryover_cutoff=1e-2).energy
    monkeypatch.undo()

    assert len(dimensions) == 5
    assert max(dimensions) > dimensions[0]
    # Measured 87 mHa on this configuration. A bound of a millihartree or two
    # would also pass on an implementation that retained almost nothing.
    assert energy < _solve_h4(1, seed=3).energy - 0.05


def test_carryover_can_recover_the_full_determinant_space():
    """The sharpest statement of what retention buys: on H4 with a sampling
    budget that reaches a quarter of the space, accumulating across iterations
    completes it and the projected energy becomes exactly FCI.

    Runs at the default tolerances, which are zero: early stopping would end the
    run before the accumulation finishes, which is exactly why it is opt-in (see
    ``test_the_convergence_break_can_stop_carryover_short``).
    """
    exact = _h4_exact_energy()
    full_space = 36  # C(4,2) alpha strings x C(4,2) beta strings

    plain = _solve_h4(4, seed=0)
    carried = _solve_h4(4, seed=0, carryover_cutoff=1e-2)

    assert len(set(plain.subspace)) < full_space
    assert plain.energy > exact + 0.02
    assert len(set(carried.subspace)) == full_space
    assert carried.energy == pytest.approx(exact, abs=1e-9)


def test_the_convergence_break_can_stop_carryover_short():
    """Pins the cost of ending recovery early, which is why it is opt-in.

    Carryover improves the subspace non-monotonically, so the break stops 6
    determinants and 11.7 mHa short of the full space the same run reaches
    without it. Tolerances are passed explicitly since they default to zero.
    """
    stopped = _solve_h4(
        4, seed=0, carryover_cutoff=1e-2, energy_tol=1e-8, occupancies_tol=1e-5
    )

    assert stopped.amplitudes.size == 30
    assert stopped.energy - _h4_exact_energy() == pytest.approx(1.16846068e-2, abs=1e-9)


def test_carried_strings_persist_through_an_iteration_that_does_not_resample_them(
    monkeypatch,
):
    """A carried string must still be there after an iteration whose own draw
    was something else, otherwise retention spans only that one iteration.

    Asserted on the retained strings rather than subspace sizes: a
    single-iteration lookback reaches the same sizes here, which is why a size
    assertion cannot tell the two apart.
    """
    kept = _spy_on_retained(monkeypatch)

    _solve_h4(1, seed=3, carryover_cutoff=1e-3)

    # Two calls per iteration (alpha then beta), skipping the first iteration
    # where nothing has been retained yet.
    alpha_rounds = [set(k) for k in kept[0::2]]
    assert len(alpha_rounds) >= 3
    # Something retained early is still retained at the end.
    assert alpha_rounds[0] & alpha_rounds[-1]


def test_carryover_weights_rank_the_halves_by_probability():
    """The cap keeps the heaviest strings, so the weights it ranks by must
    follow the eigenvector's probability rather than the order encountered."""
    strings_alpha = ("100", "010", "001")
    strings_beta = ("010", "001", "100")
    amplitudes = np.diag([0.9, 0.4, 0.1])

    alpha_weights, _ = carryover_weights(
        strings_alpha, strings_beta, amplitudes, cutoff=1e-8
    )
    assert alpha_weights["100"] > alpha_weights["010"] > alpha_weights["001"]
    assert alpha_weights["100"] == pytest.approx(0.81)


def test_carryover_weights_rank_by_the_full_marginal_not_the_eligible_part():
    """A string's rank must come from its weight across the whole subspace, not
    only from the determinants that cleared the cutoff: summing eligible ones
    alone ranks ``01`` first and loses the heavier ``10``."""
    strings_alpha = ("10", "01")
    strings_beta = ("10", "01")
    amplitudes = np.array([[0.3, 0.9], [0.4, 0.0]])

    alpha_weights, _ = carryover_weights(
        strings_alpha, strings_beta, amplitudes, cutoff=0.4
    )

    # 0.9 is the largest, so the cutoff sits at 0.36: only 0.4 and 0.9 clear it.
    assert set(alpha_weights) == {"10", "01"}
    assert alpha_weights["10"] == pytest.approx(0.3**2 + 0.9**2)
    assert alpha_weights["01"] == pytest.approx(0.4**2)
    assert alpha_weights["10"] > alpha_weights["01"]


_NUDGED = 1.0 - 2.0**-52


@pytest.mark.parametrize(
    "weights, limit, expected",
    [
        # Ranking is by weight, not lexicographic: a lexicographic cap passes
        # every test that only inspects subspace sizes.
        ({"001": 0.9, "010": 0.05, "100": 0.05}, 1, ["001"]),
        ({"001": 0.05, "010": 0.05, "100": 0.9}, 1, ["100"]),
        ({"001": 0.05, "010": 0.05, "100": 0.9}, 2, ["100", "001"]),
        # Weights differing at the last bit resolve the same way every time:
        # threaded BLAS and set order perturb eigenvector components there.
        ({"010": 1.0, "001": _NUDGED}, 1, ["001"]),
        ({"010": _NUDGED, "001": 1.0}, 1, ["001"]),
        # Rounding to twelve places absorbs only float noise.
        ({"001": 0.5, "010": 0.5 + 1e-10}, 1, ["010"]),
        ({"001": 0.5, "010": 0.5 + 1e-14}, 1, ["001"]),
        ({"001": 0.1, "010": 0.3, "100": 0.2}, None, ["010", "100", "001"]),
    ],
)
def test_the_cap_keeps_the_heaviest_strings(weights, limit, expected):
    assert _heaviest_strings(weights, limit) == expected


def test_carryover_cutoff_prunes_relative_to_the_largest_coefficient():
    """The cutoff must drop light determinants, and must mean the same thing at
    any subspace size: a normalized eigenvector's components fall as
    1/sqrt(m), so an absolute threshold would prune almost nothing at large m.
    Scaling the whole vector therefore must not change what is retained."""
    strings_alpha = ("10", "01")
    strings_beta = ("10", "01")
    # Determinants "1010" (0.8), "0101" (0.5) and "1001" (0.05); "0110" is absent.
    amplitudes = np.array([[0.8, 0.05], [0.0, 0.5]])

    kept = carryover_weights(strings_alpha, strings_beta, amplitudes, cutoff=0.1)[1]
    scaled = carryover_weights(
        strings_alpha, strings_beta, amplitudes / 100.0, cutoff=0.1
    )[1]

    # 0.05 is below a tenth of 0.8, and it is the only determinant placing beta
    # "01" alongside alpha "10", so beta "01" survives only through "0101".
    assert set(kept) == {"10", "01"}
    assert set(kept) == set(scaled)

    # Raising the cutoff past 0.5 leaves only the "1010" determinant eligible.
    pruned = carryover_weights(strings_alpha, strings_beta, amplitudes, cutoff=0.7)[1]
    assert set(pruned) == {"10"}


@pytest.mark.parametrize("cap", [1, 2])
def test_max_carryover_bounds_the_subspace(monkeypatch, cap):
    """The cap must actually bind, holding the subspace below what uncapped
    carryover reaches -- that is the whole point of offering it."""

    def run(max_carryover):
        dimensions = _spy_on_dimensions(monkeypatch)
        _solve_h4(1, seed=3, carryover_cutoff=1e-8, max_carryover=max_carryover)
        monkeypatch.undo()
        return dimensions

    capped = run(cap)
    uncapped = run(None)

    # A batch's subspace is the product of the two pools, each holding at most
    # the cap plus this batch's own single draw.
    assert max(capped) <= (cap + 1) ** 2
    assert max(capped) < max(uncapped)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"carryover_cutoff": 0.0}, "carryover_cutoff must be positive"),
        ({"carryover_cutoff": -1e-3}, "carryover_cutoff must be positive"),
        ({"max_carryover": 5}, "needs carryover_cutoff"),
        (
            {"carryover_cutoff": 1e-3, "max_carryover": 0},
            "max_carryover must be >= 1",
        ),
    ],
)
def test_carryover_rejects_invalid_configuration(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _solver(2, 1, 1, **kwargs)


def test_recovery_accumulates_probability_onto_collapsed_bitstrings(monkeypatch):
    """Correction is many-to-one, and the collapsed multiplicity must survive it.

    The correction is pinned to a fixed mapping so the expected distribution is
    exact: deduplicating with a ``set`` and looking each survivor up in the
    original distribution would give ``1010`` nothing, and a uniform rebuild
    would give every survivor a third.
    """
    corrections = {"1111": "1010", "1100": "1010", "0011": "0101", "1001": "1001"}
    monkeypatch.setattr(
        sqd_module, "bit_flip_correction", lambda bits, *_args: corrections[bits]
    )
    solver = _solver(2, 1, 1)
    probs = {"1111": 0.4, "1100": 0.2, "0011": 0.3, "1001": 0.1}

    recovered = solver._recovered_distribution(probs, iteration=1)

    assert recovered == {
        "1010": pytest.approx(0.6),
        "0101": pytest.approx(0.3),
        "1001": pytest.approx(0.1),
    }


def test_a_zero_weight_distribution_is_recovered_as_uniform():
    """Survivors that carry no probability at all still form a distribution."""
    solver = _solver(2, 1, 1)
    probs = {"1010": 0.0, "0101": 0.0, "1100": 0.0}

    assert solver._recovered_distribution(probs, iteration=0) == {
        "1010": 0.5,
        "0101": 0.5,
    }


def test_recovery_leaves_the_first_iteration_on_the_sampled_probabilities():
    """Iteration zero postselects rather than corrects, so survivors keep their
    sampled weights renormalized."""
    solver = _solver(2, 1, 1, rng=np.random.default_rng(0))
    probs = {"1010": 0.6, "0101": 0.2, "1100": 0.2}

    recovered = solver._recovered_distribution(probs, iteration=0)

    # "1100" holds both electrons in the alpha sector, so postselection drops it.
    assert recovered == {"1010": pytest.approx(0.75), "0101": pytest.approx(0.25)}


def test_a_batch_draws_distinct_configurations():
    """A batch of size k contributes k distinct configurations. With replacement,
    4 draws from 4 equally likely candidates reach all of them ~9% of the time."""
    solver = _solver(
        2,
        1,
        1,
        n_batches=1,
        batch_size=4,
        include_reference=False,
        rng=np.random.default_rng(0),
    )
    candidates = dict.fromkeys(["1010", "1001", "0110", "0101"], 0.25)

    for _ in range(20):
        ((strings_alpha, strings_beta),) = solver._draw_batches(candidates, [], [])
        # All four determinants means both halves appeared in both sectors.
        assert set(strings_alpha) == {"10", "01"}
        assert set(strings_beta) == {"10", "01"}


def test_a_batch_cannot_ask_for_more_than_the_positive_probability_pool():
    """An oversized batch_size must clamp to the positive-probability pool rather
    than raise."""
    solver = _solver(
        2,
        1,
        1,
        n_batches=1,
        batch_size=100,
        include_reference=False,
        rng=np.random.default_rng(0),
    )
    candidates = {"1010": 0.5, "0101": 0.5, "1001": 0.0}

    ((strings_alpha, strings_beta),) = solver._draw_batches(candidates, [], [])

    assert set(strings_alpha) | set(strings_beta) == {"10", "01"}


def test_sector_strings_are_ordered_for_pyscf_addressing():
    """PySCF addresses determinants by ascending CI-string integer, which is not
    the strings' lexicographic order since bit ``p`` is character ``p``."""
    solver = _solver(3, 1, 1, rng=np.random.default_rng(0))
    counts = {"100": 1, "010": 1, "001": 1}

    strings = solver._sector_strings(counts, [], "100", target=1, max_dim=None)

    assert [ci_string_to_int(half) for half in strings] == [1, 2, 4]
    assert strings == ("100", "010", "001")
    # Lexicographic ordering would have put "001" first.
    assert strings != tuple(sorted(strings))


def test_the_reference_determinant_is_always_available():
    """The aufbau determinant must be in the subspace even when sampling never
    produced it, bounding the fragment's energy by its reference."""
    n_orb = 3
    one_body = np.diag([-10.0, -0.1, 0.2])
    two_body = np.zeros((n_orb,) * 4)
    # Only a doubly-excited determinant is ever sampled.
    probs = {"001001": 1.0}
    kwargs = dict(
        n_batches=1,
        batch_size=1,
        n_iterations=1,
        lambda_penalty=0.0,
        rng=np.random.default_rng(0),
    )

    without = _solver(n_orb, 1, 1, include_reference=False, **kwargs).solve(
        probs, one_body, two_body
    )
    with_reference = _solver(n_orb, 1, 1, include_reference=True, **kwargs).solve(
        probs, one_body, two_body
    )

    assert without.strings_alpha == ("001",)
    assert without.energy == pytest.approx(0.4, abs=1e-9)
    # The reference "100100" puts both electrons in orbital 0.
    assert "100" in with_reference.strings_alpha
    assert with_reference.energy == pytest.approx(-20.0, abs=1e-9)


@pytest.mark.parametrize("max_dim", [1, 2, (1, 3), (3, 1)])
def test_max_dim_caps_each_spin_sector(max_dim):
    """The cap binds per sector, so the subspace never exceeds their product."""
    one_body, two_body, n_orb, _ = _h4_integrals()
    probs = uniform_full_space_probs(n_orb, 2, 2)
    cap_alpha, cap_beta = max_dim if isinstance(max_dim, tuple) else (max_dim, max_dim)

    result = _solver(
        n_orb,
        2,
        2,
        n_batches=2,
        batch_size=6,
        n_iterations=4,
        lambda_penalty=0.0,
        carryover_cutoff=1e-8,
        max_dim=max_dim,
        rng=np.random.default_rng(0),
    ).solve(probs, one_body, two_body)

    assert len(result.strings_alpha) <= cap_alpha
    assert len(result.strings_beta) <= cap_beta
    assert result.amplitudes.shape == (
        len(result.strings_alpha),
        len(result.strings_beta),
    )


def test_max_dim_keeps_the_reference_when_the_cap_binds():
    """At ``max_dim=1`` the one surviving string per sector must be the reference,
    whatever was sampled."""
    n_orb = 3
    one_body = np.diag([-10.0, -0.1, 0.2])
    two_body = np.zeros((n_orb,) * 4)
    probs = uniform_full_space_probs(n_orb, 1, 1)

    result = _solver(
        n_orb,
        1,
        1,
        n_batches=1,
        batch_size=8,
        n_iterations=1,
        lambda_penalty=0.0,
        max_dim=1,
        rng=np.random.default_rng(0),
    ).solve(probs, one_body, two_body)

    assert result.strings_alpha == ("100",)
    assert result.strings_beta == ("100",)


@pytest.mark.parametrize(
    "include_reference, carried, counts, max_dim, expected",
    [
        # Sampled strings rank by how often they were drawn.
        (False, [], {"100": 1, "010": 3, "001": 2}, 2, ("010", "001")),
        # Equal counts fall back to the string itself.
        (False, [], {"010": 2, "001": 2}, 1, ("001",)),
        # Reference first, then carried, then sampled, whatever the counts.
        (True, ["001"], {"010": 5}, 2, ("100", "001")),
        (True, ["001"], {"010": 5}, 1, ("100",)),
    ],
)
def test_max_dim_keeps_strings_in_priority_order(
    include_reference, carried, counts, max_dim, expected
):
    solver = _solver(3, 1, 1, include_reference=include_reference)

    kept = solver._sector_strings(counts, carried, "100", target=1, max_dim=max_dim)

    assert kept == expected


def test_symmetrize_spin_pools_the_sectors_together():
    """Sampling only "1001" offers alpha "10" and beta "01" separately; the merged
    pool offers both to both, spanning the singlet and triplet."""
    n_orb = 2
    one_body = np.diag([-1.0, -0.5])
    two_body = np.zeros((n_orb,) * 4)
    probs = {"1001": 1.0}
    kwargs = dict(
        n_batches=1,
        batch_size=4,
        n_iterations=1,
        lambda_penalty=0.0,
        include_reference=False,
        rng=np.random.default_rng(0),
    )

    separate = _solver(n_orb, 1, 1, symmetrize_spin=False, **kwargs).solve(
        probs, one_body, two_body
    )
    merged = _solver(n_orb, 1, 1, symmetrize_spin=True, **kwargs).solve(
        probs, one_body, two_body
    )

    assert set(separate.subspace) == {"1001"}
    assert set(merged.subspace) == {"1001", "1010", "0101", "0110"}


def test_symmetrize_spin_is_inactive_on_a_polarized_fragment():
    """Exchanging the sectors is not a symmetry at unequal electron counts, so the
    flag must be ignored rather than merge pools."""
    assert _solver(3, 2, 1, symmetrize_spin=True).symmetrize_spin is False
    assert _solver(3, 2, 2, symmetrize_spin=True).symmetrize_spin is True


def _count_settled_diagonalizations(mocker, **solver_kwargs):
    """Diagonalizations in a two-orbital solve whose iterations all reproduce the
    first: zero two-body terms and one saturating batch per iteration."""
    n_orb = 2
    spy = mocker.spy(sqd_module, "projected_matrices")
    _solver(
        n_orb,
        1,
        1,
        n_batches=1,
        batch_size=16,
        lambda_penalty=0.0,
        recovery=False,
        rng=np.random.default_rng(0),
        **solver_kwargs,
    ).solve(
        uniform_full_space_probs(n_orb, 1, 1),
        np.diag([-1.0, -0.5]),
        np.zeros((n_orb,) * 4),
    )
    return spy.call_count


def test_recovery_stops_once_energy_and_occupancy_settle(mocker):
    """Both criteria are met at iteration one, so the rest are skipped."""
    count = _count_settled_diagonalizations(
        mocker, n_iterations=8, energy_tol=1e-8, occupancies_tol=1e-5
    )

    # Iterations 0 and 1 to establish that nothing moved, then the break.
    assert count == 2


def test_recovery_runs_every_iteration_at_the_default_tolerances(mocker):
    """The default tolerances are zero, so no iteration can qualify to stop."""
    assert _count_settled_diagonalizations(mocker, n_iterations=5) == 5
    assert SQDConfig().recovery_energy_tol == 0.0
    assert SQDConfig().recovery_occupancies_tol == 0.0


def test_solver_is_reproducible_under_a_seed():
    one_body, two_body, n_orb, constant = _h2_integrals()
    probs = uniform_full_space_probs(n_orb, 1, 1)

    def run():
        solver = _solver(
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
    solver = _solver(n_orb, 1, 1, n_iterations=1, rng=np.random.default_rng(0))
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

    solver = _solver(
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
        solver.occupancy.sum(axis=1), [n_alpha, n_beta], atol=1e-12
    )

    solver.solve(probs, one_body, two_body)
    np.testing.assert_allclose(
        solver.occupancy.sum(axis=1), [n_alpha, n_beta], atol=1e-12
    )


def test_batch_occupancy_is_the_eigenvector_weighted_occupation():
    """A batch's occupancy is ``sum_k |c_k|^2 n_k`` over its determinants, which
    the solver reads off the sector marginals instead of the determinants."""
    one_body, two_body, n_orb, _ = _h4_integrals()
    h_spin, g_spin = spin_orbital_integrals(one_body, two_body, n_orb)
    sectors = (("1110", "1101", "1011"), ("1000", "0100", "0001"))

    result, occupancy = _solver(n_orb, 3, 1)._diagonalize(
        sectors, h_spin, g_spin, target_s=1.0, constant=0.0
    )

    weights = result.eigenvector**2
    assert np.count_nonzero(weights > 1e-6) > 1
    bits = _bit_matrix(result.subspace, 2 * n_orb)
    expected = np.stack([weights @ bits[:, :n_orb], weights @ bits[:, n_orb:]])
    np.testing.assert_allclose(occupancy, expected, atol=1e-12)


def _rdms_from(result, n_orb):
    """Both RDM paths off one solve, asserted to agree so every assertion below
    constrains the PySCF layout as well as the definitions."""
    fast = compute_spatial_rdms(
        result.strings_alpha, result.strings_beta, result.amplitudes, n_orb
    )
    exact = _spatial_rdms_exact(
        result.strings_alpha, result.strings_beta, result.amplitudes, n_orb
    )
    for from_pyscf, from_definition in zip(fast, exact):
        np.testing.assert_allclose(from_pyscf, from_definition, atol=1e-10)
    return fast


def test_solver_reproduces_fci_energy_and_rdm1_when_subspace_is_complete():
    one_body, two_body, n_orb, constant = _h2_integrals()
    solver = _solver(
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
    rdm1, _, _, _ = _rdms_from(result, n_orb)

    _, civec = fci.direct_spin1.kernel(one_body, two_body, n_orb, (1, 1))
    expected = fci.direct_spin1.make_rdm1(civec, n_orb, (1, 1))

    assert result.energy == pytest.approx(
        dense_fci_energy(one_body, two_body, 1, 1, constant), abs=1e-8
    )
    assert np.trace(rdm1) == pytest.approx(2.0, abs=1e-12)
    np.testing.assert_allclose(rdm1, expected, atol=1e-9)


def test_spatial_rdm12_and_energy_match_pyscf_fci_beyond_two_orbitals():
    """Both RDMs, and the energy they reconstruct, on a 4-orbital active
    space. A sign error in the two-body contraction corrupts rdm2 and the
    reconstructed energy even where it happens to leave rdm1 alone, and a
    2-orbital case cannot exercise that path at all.

    Also pins rdm2's ``rspq`` and ``qpsr`` invariances, which
    ``rotation_energy_gradient_fn`` needs to collapse the two-electron
    derivative into one Fock matrix. The last assertion keeps them non-vacuous.
    """
    one_body, two_body, n_orb, constant = _h4_integrals()
    n_alpha = n_beta = 2
    solver = _solver(
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
    rdm1, rdm2, _, _ = _rdms_from(result, n_orb)

    _, civec = fci.direct_spin1.kernel(one_body, two_body, n_orb, (n_alpha, n_beta))
    expected_rdm1, expected_rdm2 = fci.direct_spin1.make_rdm12(
        civec, n_orb, (n_alpha, n_beta)
    )

    np.testing.assert_allclose(rdm1, expected_rdm1, atol=1e-9)
    np.testing.assert_allclose(rdm2, expected_rdm2, atol=1e-9)
    np.testing.assert_allclose(rdm1, rdm1.T, atol=1e-10)

    np.testing.assert_allclose(rdm2, rdm2.transpose(2, 3, 0, 1), atol=1e-12)
    np.testing.assert_allclose(rdm2, rdm2.transpose(1, 0, 3, 2), atol=1e-12)
    assert not np.allclose(rdm2, rdm2.transpose(1, 0, 2, 3), atol=1e-8)

    energy_from_rdms = (
        np.sum(one_body * rdm1)
        + 0.5 * np.einsum("pqrs,pqrs->", two_body, rdm2)
        + constant
    )
    assert energy_from_rdms == pytest.approx(result.energy, abs=1e-8)


def test_spatial_spin_rdm1s_match_pyscf_for_a_polarized_sector():
    """The per-spin halves are why this returns a 4-tuple, and a balanced sector
    cannot test them: there ``alpha == beta == rdm1 / 2``, so a swapped or
    spin-traced return would pass. Use a polarized sector, where PySCF's
    ``make_rdm1s`` gives two different matrices."""
    one_body, two_body, n_orb, constant = _h4_integrals()
    n_alpha, n_beta = 3, 1
    solver = _solver(
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
    rdm1, _, rdm1_alpha, rdm1_beta = _rdms_from(result, n_orb)

    _, civec = fci.direct_spin1.kernel(one_body, two_body, n_orb, (n_alpha, n_beta))
    expected_alpha, expected_beta = fci.direct_spin1.make_rdm1s(
        civec, n_orb, (n_alpha, n_beta)
    )

    assert not np.allclose(expected_alpha, expected_beta), "sector must be polarized"
    np.testing.assert_allclose(rdm1_alpha, expected_alpha, atol=1e-9)
    np.testing.assert_allclose(rdm1_beta, expected_beta, atol=1e-9)
    np.testing.assert_allclose(rdm1_alpha + rdm1_beta, rdm1, atol=1e-12)
    # The traces are electron counts, exact to machine precision.
    assert np.trace(rdm1_alpha) == pytest.approx(n_alpha, abs=1e-12)
    assert np.trace(rdm1_beta) == pytest.approx(n_beta, abs=1e-12)


def _spin_orbital_reference_rdms(strings_alpha, strings_beta, amplitudes, n_orb):
    """The spin-orbital formulation ``_spatial_rdms_exact`` replaced: every pair
    of determinants, every ``a+_p a+_r a_s a_q``, into a ``(2 n_orb) ** 4`` array
    spin-traced at the end. Kept as the reference definition."""
    dets = [
        spatial_to_spin_occupations(alpha, beta, n_orb)
        for alpha in _sector_occupations(strings_alpha, n_orb)
        for beta in _sector_occupations(strings_beta, n_orb)
    ]
    coefficients = np.asarray(amplitudes, dtype=float).ravel()
    n_spin = 2 * n_orb
    rdm1 = np.zeros((n_spin, n_spin))
    rdm2 = np.zeros((n_spin,) * 4)
    for det_i, c_i in zip(dets, coefficients):
        for det_j, c_j in zip(dets, coefficients):
            for q in det_j:
                sign_q, occ_1 = _annihilation_sign(det_j, q)
                for p in set(range(n_spin)) - set(occ_1):
                    sign_p, occ = _creation_sign(occ_1, p)
                    if occ == det_i:
                        rdm1[p, q] += c_i * c_j * sign_p * sign_q
                for s in occ_1:
                    sign_s, occ_2 = _annihilation_sign(occ_1, s)
                    for r in set(range(n_spin)) - set(occ_2):
                        sign_r, occ_3 = _creation_sign(occ_2, r)
                        for p in set(range(n_spin)) - set(occ_3):
                            sign_p, occ = _creation_sign(occ_3, p)
                            if occ == det_i:
                                rdm2[p, q, r, s] += (
                                    c_i * c_j * sign_q * sign_s * sign_r * sign_p
                                )
    alpha, beta = slice(None, n_orb), slice(n_orb, None)
    rdm1_alpha, rdm1_beta = rdm1[alpha, alpha], rdm1[beta, beta]
    spatial_rdm2 = (
        rdm2[alpha, alpha, alpha, alpha]
        + rdm2[alpha, alpha, beta, beta]
        + rdm2[beta, beta, alpha, alpha]
        + rdm2[beta, beta, beta, beta]
    )
    return rdm1_alpha + rdm1_beta, spatial_rdm2, rdm1_alpha, rdm1_beta


def _random_sector_state(n_orb, n_alpha, n_beta, seed):
    """A random state over a strict subset of each sector's strings, ordered for
    PySCF addressing, so the pair search sees every excitation rank and misses."""
    rng = np.random.default_rng(seed)

    def sector(n_electrons):
        strings = [
            "".join("1" if p in occ else "0" for p in range(n_orb))
            for occ in itertools.combinations(range(n_orb), n_electrons)
        ]
        if len(strings) > 2:
            strings = list(rng.choice(strings, len(strings) - 1, replace=False))
        return tuple(sorted(strings, key=ci_string_to_int))

    strings_alpha, strings_beta = sector(n_alpha), sector(n_beta)
    amplitudes = rng.normal(size=(len(strings_alpha), len(strings_beta)))
    return strings_alpha, strings_beta, amplitudes / np.linalg.norm(amplitudes)


@pytest.mark.parametrize(
    "n_orb, n_alpha, n_beta",
    [(2, 1, 1), (4, 2, 2), (4, 3, 1), (5, 2, 3), (4, 3, 0), (3, 0, 2)],
)
def test_spatial_rdms_exact_matches_the_spin_orbital_definition_and_pyscf(
    n_orb, n_alpha, n_beta
):
    """Accumulating straight into the spatial blocks must reproduce the
    spin-orbital formulation to rounding, and PySCF's contractions too."""
    state = _random_sector_state(n_orb, n_alpha, n_beta, seed=n_orb + 7 * n_alpha)

    exact = _spatial_rdms_exact(*state, n_orb)
    reference = _spin_orbital_reference_rdms(*state, n_orb)
    from_pyscf = compute_spatial_rdms(*state, n_orb)

    for got, want, pyscf_block in zip(exact, reference, from_pyscf):
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)
        np.testing.assert_allclose(got, pyscf_block, rtol=0, atol=1e-12)


@pytest.mark.parametrize("block_rows", [1, 5])
def test_spatial_rdms_exact_is_independent_of_the_row_block(monkeypatch, block_rows):
    """The pair search runs in row blocks, which must not drop the pairs that
    straddle a block boundary."""
    state = _random_sector_state(4, 2, 2, seed=3)
    expected = _spatial_rdms_exact(*state, 4)

    monkeypatch.setattr(sqd_module, "_PAIR_BLOCK_ROWS", block_rows)

    for got, want in zip(_spatial_rdms_exact(*state, 4), expected):
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)


def test_wide_fragments_reconstruct_rdms_off_the_exact_path(mocker, monkeypatch):
    """Past PySCF's 64-bit CI strings the reconstruction must switch kernels and
    still give PySCF's answer."""
    state = _random_sector_state(4, 3, 1, seed=11)
    expected = compute_spatial_rdms(*state, 4)

    monkeypatch.setattr(sqd_module, "_MAX_PYSCF_ORBITALS", 1)
    exact = mocker.spy(sqd_module, "_spatial_rdms_exact")
    pyscf_rdm2 = mocker.spy(sqd_module.selected_ci, "make_rdm2")
    wide = compute_spatial_rdms(*state, 4)

    exact.assert_called_once()
    pyscf_rdm2.assert_not_called()
    for got, want in zip(wide, expected):
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)
