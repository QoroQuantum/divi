# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Effective per-fragment integrals, the LASSQD total-energy functional, and
the orbital solves."""

import math
import os
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import accumulate, combinations, pairwise, permutations, product
from pathlib import Path
from warnings import warn

import numpy as np
import scipy.linalg
import scipy.optimize
import scipy.sparse
from pyscf import ao2mo, lib, mcscf, mrpt
from pyscf.fci import selected_ci
from pyscf.scf import hf, hf_symm
from pyscf.soscf import ciah
from scipy.optimize import minimize

from ._sqd import (
    ci_string_to_int,
    compute_spatial_rdms,
    lowest_roots,
    projected_matrices,
    resolves_spin,
    sci_vector,
    spin_orbital_integrals,
    subspace_determinants,
)
from ._state import (
    ElectronicState,
    FragmentSpec,
    FragmentState,
    StateTransition,
    StateWavefunction,
    _SampledSubspace,
)

# L-BFGS-B ftol is relative, so a tol=1e-6 shorthand halts around 1e-6 * |E|,
# coarser than LASSQD's energy_tol. Tighter than this terminates ABNORMAL.
ORBITAL_MINIMIZE_OPTIONS = {"ftol": 1e-12, "gtol": 1e-6}
# Most L-BFGS-B runs in one orbital solve, each restarted from the last one's
# orbitals.
_MAX_RESTARTS = 10
# Central-difference step of the Hessian-vector product, as the largest angle.
_HESSIAN_STEP = 1e-4
# Largest element-wise deviation from the identity that still counts as no step.
_STALLED_STEP = 1e-14
# Largest deviation of a space's irrep projection from a projector that still
# counts as symmetry-adapted; converged solves leave ~2e-5, broken ones >0.2.
_SYMMETRY_TOL = 1e-3
# A converged point this far above the best one seen still replaces it.
_ENERGY_ROUNDING = 1e-10
# Lowest orbital-Hessian eigenvalue below which a saddle is escaped, about 100x
# the central-difference noise floor.
_SADDLE_CURVATURE = -1e-3
# Eigenvalue tolerance of the stability Davidson solve.
_DAVIDSON_TOL = 1e-5
# Trial vectors the stability Davidson solve keeps before restarting.
_DAVIDSON_SPACE = 30
# Smallest magnitude the stability preconditioner divides by; many pairs have a
# zero Hessian diagonal.
_PRECONDITIONER_FLOOR = 1e-2
# First angle of the line search along a negative mode, doubled up to pi / 4.
_ESCAPE_FIRST_ANGLE = 0.05
# Frames under this prefix are divi's own, never the code a warning is for.
_DIVI_PREFIX = os.path.join(Path(__file__).parents[3], "")


@dataclass(frozen=True)
class OrbitalSolution:
    """Outcome of one round's orbital re-optimisation.

    Attributes:
        mo_coeff: The rotated MO coefficients.
        energy: Total energy at the returned orbitals.
        converged: Whether ``gradient_norm`` is within the caller's tolerance.
            A capped or stalled solve still returns its best point, so the
            energy remains an upper bound, but not a stationary point.
        n_iterations: Optimizer iterations taken.
        n_evaluations: Objective evaluations taken, each one four-index MO
            transform.
        gradient_norm: L2 norm of the orbital gradient over the rotation pairs
            at the returned orbitals.
        n_rotation_pairs: Number of orbital pairs the rotation spanned.
    """

    mo_coeff: np.ndarray
    energy: float
    converged: bool
    n_iterations: int
    n_evaluations: int
    gradient_norm: float
    n_rotation_pairs: int


@dataclass(frozen=True)
class MOIntegrals:
    """One round's active-space integrals.

    Every array is indexed over the active space alone, in the fragment-block
    order of the ``mo_coeff`` passed to :func:`transform_integrals`. The core
    and virtual blocks are not stored: the only consumer,
    :func:`fragment_effective_integrals`, reads neither, and materialising the
    full register costs ``n_orb ** 4``.

    Attributes:
        h_eff: ``(n_act, n_act)`` one-electron integrals with the frozen-core
            potential ``2 J_core - K_core`` folded in.
        g_act: ``(n_act,) * 4`` two-electron integrals in chemist order.
    """

    h_eff: np.ndarray
    g_act: np.ndarray


def fragment_blocks(specs: Sequence[FragmentSpec], offset: int = 0) -> list[slice]:
    """Each fragment's contiguous orbital span, starting at ``offset``."""
    bounds = list(accumulate((spec.n_orbitals for spec in specs), initial=offset))
    return [slice(start, stop) for start, stop in pairwise(bounds)]


def build_active_permutation(
    specs: Sequence[FragmentSpec], n_core: int, n_orbitals_total: int
) -> np.ndarray:
    """Order orbitals as ``[core | fragment blocks | virtual]``.

    Downstream code assumes the active block is contiguous. This permutation
    honours the caller's requested orbital indices (``FragmentSpec.orbitals``
    is never sorted, so orbitals within a fragment, and fragments among
    themselves, keep the caller's order) while still producing that
    contiguous active block.

    Args:
        specs: Fragment specifications, in the order their blocks should
            appear in the permuted register.
        n_core: Number of frozen-core orbitals to place first.
        n_orbitals_total: Total number of spatial orbitals in the molecule.

    Returns:
        A length-``n_orbitals_total`` index array; ``mo_coeff[:, permutation]``
        reorders the MO columns into ``[core | active | virtual]``.
    """
    active = [orbital for spec in specs for orbital in spec.orbitals]
    used = set(active)
    remaining = [o for o in range(n_orbitals_total) if o not in used]
    core, virtual = remaining[:n_core], remaining[n_core:]
    return np.array(core + active + virtual, dtype=int)


def transform_integrals(
    mol,
    mo_coeff: np.ndarray,
    n_core: int,
    n_act: int,
    ao_eri: np.ndarray | None = None,
    h_ao: np.ndarray | None = None,
) -> MOIntegrals:
    """Transform the active-space integrals, embedding the frozen core, with
    PySCF's CASCI on the exact integrals.

    Args:
        mol: A PySCF ``gto.Mole``.
        mo_coeff: ``(nao, n_orb)`` MO coefficients, already permuted so that
            columns ``[0, n_core)`` are core, ``[n_core, n_core + n_act)``
            are the active fragment blocks in spec order, and the rest are
            virtual.
        n_core: Number of frozen-core orbitals.
        n_act: Number of active orbitals.
        ao_eri: AO-basis electron-repulsion integral from :func:`cached_ao_eri`.
            Supplying it avoids rebuilding the AO integrals every round;
            ``None`` builds them here.
        h_ao: AO-basis core Hamiltonian; ``None`` uses :func:`cached_h_ao`.
    """

    if ao_eri is None:
        ao_eri = cached_ao_eri(mol)
    if h_ao is None:
        h_ao = cached_h_ao(mol)

    casci = mcscf.casci.CASCI(
        _exact_mean_field(mol, ao_eri, h_ao), n_act, mol.nelectron - 2 * n_core
    )
    casci.ncore = n_core
    h_eff, _ = casci.get_h1eff(mo_coeff)
    return MOIntegrals(
        h_eff=h_eff,
        g_act=ao2mo.restore(1, casci.get_h2eff(mo_coeff), n_act),
    )


def fragment_effective_integrals(
    integrals: MOIntegrals, fragments: Sequence[FragmentState], index: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build one fragment's effective one- and two-body integrals.

    The frozen core contributes a spin-free mean-field potential
    (``2 * J_core - K_core``) -- a doubly occupied core carries no spin density
    -- and every *other* fragment contributes its own mean-field Coulomb and
    exchange term. Coulomb sees the other fragment's total density; exchange is
    same-spin, so it contracts that fragment's alpha or beta density and the
    resulting one-body potential differs between the two spin channels:

    ``h_sigma[p,q] = h[p,q] + 2 J_core - K_core
    + sum_B ( gamma_B[r,s] (pq|rs) - gamma_B^sigma[r,s] (pr|sq) )``

    The two channels coincide only when every other fragment is closed-shell.
    Spin-averaging them would erase the ``K_alpha - K_beta`` asymmetry that
    generates inter-fragment magnetic coupling, leaving each fragment's solver
    blind to the sign of its neighbours' local moments.

    Args:
        integrals: Active-space integrals for the current round.
        fragments: Every fragment's current state, in permutation order.
        index: Position of the target fragment within ``fragments``.

    Returns:
        ``(h_alpha, h_beta, g_frag)``: the fragment's effective one-body
        integrals per spin channel, and its bare two-body integrals.

    Raises:
        ValueError: If the fragments cover a different number of orbitals than
            ``integrals`` spans, meaning the two were built from different
            fragment specs.
    """
    blocks = fragment_blocks([fragment.spec for fragment in fragments])
    running = blocks[-1].stop

    n_act = integrals.h_eff.shape[0]
    if running != n_act:
        raise ValueError(
            f"Fragments cover {running} orbitals but the active-space integrals "
            f"span {n_act}. `integrals` and `fragments` were built from "
            "different fragment specs."
        )

    target = blocks[index]
    core_embedded = integrals.h_eff[target, target].copy()
    # Separate arrays even with one fragment, where the loop below never rebinds
    # them: two names for one array would make any later in-place edit corrupt
    # both channels, and only in that configuration.
    h_alpha = core_embedded
    h_beta = core_embedded.copy()

    g_act = integrals.g_act
    for other_index, other in enumerate(fragments):
        if other_index == index:
            continue
        span = blocks[other_index]
        exchange_block = g_act[target, span, span, target]
        alpha_other, beta_other = other.spin_rdm1s()

        coulomb = np.einsum(
            "rs,pqrs->pq", other.rdm1, g_act[target, target, span, span]
        )
        h_alpha = (
            h_alpha + coulomb - np.einsum("rs,prsq->pq", alpha_other, exchange_block)
        )
        h_beta = h_beta + coulomb - np.einsum("rs,prsq->pq", beta_other, exchange_block)

    return h_alpha, h_beta, g_act[target, target, target, target].copy()


def assemble_active_rdms(
    fragments: Sequence[FragmentState],
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble the full active-space 1- and 2-RDM from per-fragment RDMs.

    Fragments are placed block-diagonally in the order supplied, purely by
    position (the running offset of ``spec.n_orbitals``). The 1-RDM has no
    cross-fragment elements; the 2-RDM does. For a product state,

    ``Gamma[p,q,r,s] = gamma[p,q] gamma[r,s] - sum_sigma gamma^sigma[p,s]
    gamma^sigma[r,q]``

    so with ``p, q`` in fragment A and ``r, s`` in fragment B only the direct
    term survives, and with ``p, s`` in A and ``q, r`` in B only the exchange
    term does. Both are filled for every ordered pair of distinct fragments.
    """
    n_act = sum(fragment.spec.n_orbitals for fragment in fragments)
    rdm1 = np.zeros((n_act, n_act))
    rdm2 = np.zeros((n_act, n_act, n_act, n_act))

    spans = fragment_blocks([fragment.spec for fragment in fragments])
    blocks = []
    for span, fragment in zip(spans, fragments):
        rdm1[span, span] = fragment.rdm1
        rdm2[span, span, span, span] = fragment.rdm2
        blocks.append((span, fragment.rdm1, *fragment.spin_rdm1s()))

    for (span_a, rdm1_a, alpha_a, beta_a), (
        span_b,
        rdm1_b,
        alpha_b,
        beta_b,
    ) in permutations(blocks, 2):
        rdm2[span_a, span_a, span_b, span_b] += np.einsum("pq,rs->pqrs", rdm1_a, rdm1_b)
        rdm2[span_a, span_b, span_b, span_a] -= np.einsum(
            "ps,rq->pqrs", alpha_a, alpha_b
        ) + np.einsum("ps,rq->pqrs", beta_a, beta_b)

    return rdm1, rdm2


def cached_ao_eri(mol) -> np.ndarray:
    """Compute the AO-basis electron-repulsion integral once per run.

    The returned array is the 8-fold symmetry-packed ``int2e`` integral,
    suitable as the ``ao_eri`` argument to :func:`_total_energy` for every
    orbital-rotation loss evaluation in a macro-cycle, avoiding a repeated
    ``ao2mo.kernel`` AO integral recomputation.
    """
    return mol.intor("int2e", aosym="s8")


def cached_h_ao(mol) -> np.ndarray:
    """Compute the AO-basis core Hamiltonian of ``mol`` once per run.

    Kinetic and nuclear-attraction integrals plus any effective core
    potential. A mean field with its own core Hamiltonian, such as a
    scalar-relativistic one, supplies it through ``get_hcore`` instead.

    The returned array is suitable as the ``h_ao`` argument to
    :func:`_total_energy` for every orbital-rotation loss evaluation in a
    macro-cycle, avoiding repeated AO integral recomputation.
    """
    return hf.get_hcore(mol)


def _total_energy(
    mol,
    mo_coeff: np.ndarray,
    n_core: int,
    rdm1_active: np.ndarray,
    rdm2_active: np.ndarray,
    ao_eri: np.ndarray,
    h_ao: np.ndarray,
) -> float:
    """Compute the total molecular energy for a given set of MO coefficients.

    The four-index MO transform runs in-core from a pre-computed AO-basis
    ``ao_eri`` (see :func:`cached_ao_eri`) via ``ao2mo.incore.full`` rather
    than calling ``ao2mo.kernel(mol, mo_coeff)`` again on every evaluation.
    This is the dominant cost of a gradient-free orbital-rotation
    optimisation and must not be paid per loss evaluation.

    Args:
        mol: A PySCF ``gto.Mole``.
        mo_coeff: ``(nao, n_orb)`` MO coefficients for this evaluation.
        n_core: Number of frozen-core orbitals.
        rdm1_active: ``(n_act, n_act)`` active-space 1-RDM.
        rdm2_active: ``(n_act,) * 4`` active-space 2-RDM.
        ao_eri: AO-basis electron-repulsion integral, as returned by
            :func:`cached_ao_eri`.
        h_ao: AO-basis one-electron integral, as returned by
            :func:`cached_h_ao`.

    Raises:
        ImportError: If the ``chem`` extra is not installed.
    """
    energy, _ = energy_and_generalized_fock(
        mol, mo_coeff, n_core, rdm1_active, rdm2_active, ao_eri, h_ao
    )
    return energy


def energy_and_generalized_fock(
    mol,
    mo_coeff: np.ndarray,
    n_core: int,
    rdm1_active: np.ndarray,
    rdm2_active: np.ndarray,
    ao_eri: np.ndarray,
    h_ao: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Energy and generalized Fock matrix without a full four-index transform.

    Equivalent to contracting the dense full-register 1- and 2-particle
    densities (doubly occupied core, the active RDMs, empty virtuals) against
    the full MO integrals, but built from the blocks those densities actually
    reach. The two-particle density vanishes whenever any index is virtual and
    is diagonal within the core, so the contraction reduces to Coulomb/exchange
    builds from the core and active densities plus one ``(all|act,act,act)``
    transform -- ``n_orb * n_act ** 3`` elements rather than ``n_orb ** 4``.

    Args:
        mol: A PySCF ``gto.Mole``.
        mo_coeff: ``(nao, n_orb)`` MO coefficients, permuted into
            ``[core | fragment blocks | virtual]`` order.
        n_core: Number of frozen-core orbitals.
        rdm1_active: ``(n_act, n_act)`` active-space 1-RDM.
        rdm2_active: ``(n_act,) * 4`` active-space 2-RDM.
        ao_eri: AO-basis electron-repulsion integral from :func:`cached_ao_eri`.
        h_ao: AO-basis one-electron integral from :func:`cached_h_ao`.

    Returns:
        ``(energy, fock)`` with ``fock`` shaped ``(n_orb, n_orb)``; its virtual
        columns are zero, since the densities do not reach them.
    """
    energy, fock, _, _ = _energy_fock_and_potentials(
        mol, mo_coeff, n_core, rdm1_active, rdm2_active, ao_eri, h_ao
    )
    return energy, fock


def _energy_fock_and_potentials(
    mol,
    mo_coeff: np.ndarray,
    n_core: int,
    rdm1_active: np.ndarray,
    rdm2_active: np.ndarray,
    ao_eri: np.ndarray,
    h_ao: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """:func:`energy_and_generalized_fock` plus the AO Coulomb and exchange
    potentials ``vj`` and ``vk`` of the unit-occupation core and the active
    density, each stacked ``[core, active]``."""
    n_orb = mo_coeff.shape[1]
    n_act = rdm1_active.shape[0]
    active = slice(n_core, n_core + n_act)
    core_coeff = mo_coeff[:, :n_core]
    active_coeff = mo_coeff[:, active]

    h_mo = mo_coeff.T @ h_ao @ mo_coeff

    # Occupation-one core density, so vj/vk come out as sum_i (mn|ii) and
    # sum_i (mi|in) directly. Both densities go in one call so the ERI is
    # traversed once rather than twice.
    dm_core = core_coeff @ core_coeff.T
    dm_act = active_coeff @ rdm1_active @ active_coeff.T
    vj, vk = hf.dot_eri_dm(ao_eri, np.stack([dm_core, dm_act]), hermi=1)

    j_core = mo_coeff.T @ vj[0] @ mo_coeff
    k_core = mo_coeff.T @ vk[0] @ mo_coeff
    j_act = mo_coeff.T @ vj[1] @ mo_coeff
    k_act = mo_coeff.T @ vk[1] @ mo_coeff

    # Both small index sets go first: ``half_e1`` transforms the leading pair, so
    # (act,act) costs n_act**2 pairs where (all,act) costs n_orb*n_act. Chemist
    # symmetry recovers the wanted ordering, (m a | b c) = g_aaaA[b, c, a, m].
    g_gaaa = (
        ao2mo.incore.general(
            ao_eri,
            (active_coeff, active_coeff, active_coeff, mo_coeff),
            compact=False,
        )
        .reshape(n_act, n_act, n_act, n_orb)
        .transpose(3, 2, 0, 1)
    )

    core_diagonal = np.arange(n_core)
    e_core = 2.0 * float(
        np.sum(
            h_mo[core_diagonal, core_diagonal]
            + j_core[core_diagonal, core_diagonal]
            - 0.5 * k_core[core_diagonal, core_diagonal]
        )
    )
    embedded_h = (
        h_mo[active, active] + 2.0 * j_core[active, active] - k_core[active, active]
    )
    e_act = float(np.sum(rdm1_active * embedded_h)) + 0.5 * float(
        np.einsum("pqrs,pqrs->", rdm2_active, g_gaaa[active], optimize=True)
    )
    energy = float(mol.energy_nuc()) + e_core + e_act

    fock = np.zeros((n_orb, n_orb))
    fock[:, :n_core] = (
        2.0 * h_mo[:, :n_core]
        + 4.0 * j_core[:, :n_core]
        - 2.0 * k_core[:, :n_core]
        + 2.0 * j_act[:, :n_core]
        - k_act[:, :n_core]
    )
    embedded_general = h_mo[:, active] + 2.0 * j_core[:, active] - k_core[:, active]
    fock[:, active] = embedded_general @ rdm1_active.T + np.einsum(
        "mqrs,pqrs->mp", g_gaaa, rdm2_active, optimize=True
    )
    return energy, fock, vj, vk


def rotation_energy_gradient_fn(
    mol,
    mo_coeff: np.ndarray,
    n_core: int,
    fragment_specs: Sequence[FragmentSpec],
    rdm1_active: np.ndarray,
    rdm2_active: np.ndarray,
    ao_eri: np.ndarray,
    h_ao: np.ndarray,
):
    """Build the orbital-rotation objective and its analytic gradient.

    The returned callable evaluates :func:`_total_energy` at
    ``mo_coeff @ expm(K(x))`` -- via
    :func:`energy_and_generalized_fock` -- together with its exact derivative
    with respect to the rotation angles ``x``, sharing the single four-index MO
    transform between the two.

    The gradient follows from the generalized Fock matrix
    ``F = h @ D.T + einsum("mqrs,nqrs->mn", g, d)``. Because the
    parameterisation is global (``expm`` of the full generator, not a step
    from the current point), the chain rule through the matrix exponential is
    a Frechet pullback, ``M = expm_frechet(K.T, U @ 2F)``, and
    ``dE/dx_i = M[p, q] - M[q, p]``. At ``x = 0`` this reduces to
    ``2 * (F[p, q] - F[q, p])``.

    The generalized Fock matrix collapses the four derivative terms of the
    two-electron energy into one, so the gradient (not the energy) requires
    the active RDMs to carry their physical permutation symmetry:
    ``rdm1_active`` symmetric, and ``rdm2_active`` invariant under
    ``pqrs -> rspq`` and ``pqrs -> qpsr``.

    Args:
        mol: A PySCF ``gto.Mole``.
        mo_coeff: ``(nao, n_orb)`` MO coefficients, permuted into
            ``[core | fragment blocks | virtual]`` order.
        n_core: Number of frozen-core orbitals.
        fragment_specs: Fragment specifications, in the same order as the
            fragment blocks in ``mo_coeff``; only each fragment's orbital
            count is used.
        rdm1_active: ``(n_act, n_act)`` active-space 1-RDM.
        rdm2_active: ``(n_act,) * 4`` active-space 2-RDM.
        ao_eri: AO-basis electron-repulsion integral, as returned by
            :func:`cached_ao_eri`.
        h_ao: AO-basis one-electron integral, as returned by
            :func:`cached_h_ao`.

    Returns:
        ``(rotation_pairs, energy_and_gradient)``: the ``(p, q)`` orbital
        pairs the rotation angles are indexed by, and a callable mapping a
        length-``len(rotation_pairs)`` angle vector to ``(energy, gradient)``.

    Raises:
        ImportError: If the ``chem`` extra is not installed.
    """
    pairs = rotation_pairs(mo_coeff.shape[1], n_core, fragment_specs)
    rows, cols = pair_indices(pairs)

    def energy_and_gradient(
        rotation_params: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        return _rotated_energy_and_gradient(
            mol,
            mo_coeff,
            rows,
            cols,
            rotation_params,
            n_core,
            rdm1_active,
            rdm2_active,
            ao_eri,
            h_ao,
        )

    return pairs, energy_and_gradient


def _rotated_energy_and_gradient(
    mol,
    mo_coeff: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    rotation_params: np.ndarray,
    n_core: int,
    rdm1_active: np.ndarray,
    rdm2_active: np.ndarray,
    ao_eri: np.ndarray,
    h_ao: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Energy at ``mo_coeff @ expm(K(x))`` and its gradient in the angles ``x``."""
    generator = rotation_generator(rows, cols, rotation_params, mo_coeff.shape[1])
    unitary = scipy.linalg.expm(generator)
    energy, fock = energy_and_generalized_fock(
        mol, mo_coeff @ unitary, n_core, rdm1_active, rdm2_active, ao_eri, h_ao
    )
    pullback = np.asarray(
        scipy.linalg.expm_frechet(
            generator.T, unitary @ (2.0 * fock), compute_expm=False
        )
    )
    return energy, pullback[rows, cols] - pullback[cols, rows]


def rotation_pairs(
    n_orbitals: int, n_core: int, fragment_specs: Sequence[FragmentSpec]
) -> list[tuple[int, int]]:
    """Orbital pairs LASSQD rotates: core-active, core-virtual, active-active
    across different fragments, and active-virtual."""
    n_act = sum(spec.n_orbitals for spec in fragment_specs)
    core = range(n_core)
    active = range(n_core, n_core + n_act)
    virtual = range(n_core + n_act, n_orbitals)

    pairs: list[tuple[int, int]] = [*product(core, active), *product(core, virtual)]
    for block_a, block_b in combinations(
        fragment_blocks(fragment_specs, offset=n_core), 2
    ):
        pairs += product(
            range(block_a.start, block_a.stop), range(block_b.start, block_b.stop)
        )
    pairs += product(active, virtual)
    return pairs


def pair_indices(
    rotation_pairs: Sequence[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    """Split rotation pairs into row and column index arrays."""
    pairs = np.asarray(rotation_pairs, dtype=int).reshape(len(rotation_pairs), 2)
    return pairs[:, 0], pairs[:, 1]


def rotation_generator(
    rows: np.ndarray, cols: np.ndarray, angles: np.ndarray, n_orbitals: int
) -> np.ndarray:
    """Antisymmetric generator ``K`` with ``K[rows, cols] = angles``."""
    generator = np.zeros((n_orbitals, n_orbitals))
    generator[rows, cols] = angles
    generator[cols, rows] = -angles
    return generator


def optimize_orbitals(
    mol,
    mo_coeff: np.ndarray,
    n_core: int,
    fragment_specs: Sequence[FragmentSpec],
    rdm1_active: np.ndarray,
    rdm2_active: np.ndarray,
    ao_eri: np.ndarray,
    h_ao: np.ndarray,
    *,
    gradient_tol: float,
    max_iterations: int | None = None,
    report: Callable[[str], None] | None = None,
) -> OrbitalSolution:
    """Optimise molecular orbitals against the current active-space RDMs.

    Parameterizes an orbital rotation as the exponential of a skew-symmetric
    generator over a fixed set of allowed rotation pairs -- core-active,
    core-virtual, active-active across different fragments, and
    active-virtual -- and minimises :func:`_total_energy` over those rotation
    angles with L-BFGS-B. The objective and its analytic gradient come from
    :func:`rotation_energy_gradient_fn`, so an iteration costs one four-index
    MO transform rather than the ``n_rot + 1`` a finite-difference gradient
    would need (both ``ao_eri`` and ``h_ao`` must still be cached before this
    call). Core-core rotations are excluded because :func:`_total_energy` has no
    core-core degree of freedom to resolve; intra-fragment active rotations
    are excluded not because they leave the energy unchanged (they do not)
    but because each fragment's RDM is only valid in that fragment's current
    orbital basis, and rotating within it would invalidate that RDM.

    The zero-rotation baseline (``mo_coeff`` unchanged) is always evaluated
    and compared against ``scipy.optimize.minimize``'s result; whichever is
    lower is returned. This makes the routine monotone by construction -- it
    can never report an energy worse than not rotating at all -- and guards
    against ``minimize`` reporting a spurious ``fun`` (e.g. L-BFGS-B returns
    ``fun=0.0`` without evaluating the objective when there are zero rotation
    parameters) instead of raising.

    The angles are measured from the orbitals a solve starts at, which turns
    them ill-conditioned after a large rotation, so while the gradient at the
    current orbitals exceeds ``gradient_tol`` the solve restarts from them, up
    to ``_MAX_RESTARTS`` times and within ``max_iterations`` in total. ``gradient_norm`` is always taken at the returned orbitals.

    That monotonicity is also why the convergence flag matters: an optimizer
    that gives up returns the baseline, so the round's energy barely moves and
    looks exactly like a converged macro-cycle. The flag lets the caller tell
    the two apart. It tests the orbital gradient itself, not L-BFGS-B's status,
    which also reports success once the relative energy change stalls.

    Args:
        mol: A PySCF ``gto.Mole``.
        mo_coeff: ``(nao, n_orb)`` MO coefficients, permuted into
            ``[core | fragment blocks | virtual]`` order.
        n_core: Number of frozen-core orbitals.
        fragment_specs: Fragment specifications, in the same order as the
            fragment blocks in ``mo_coeff``; only each fragment's orbital
            count is used.
        rdm1_active: ``(n_act, n_act)`` active-space 1-RDM.
        rdm2_active: ``(n_act,) * 4`` active-space 2-RDM.
        ao_eri: AO-basis electron-repulsion integral, as returned by
            :func:`cached_ao_eri`.
        h_ao: AO-basis one-electron integral, as returned by
            :func:`cached_h_ao`.
        gradient_tol: The solve counts as converged when the L2 norm of the
            orbital gradient at the returned orbitals is at most this.
        max_iterations: Cap on L-BFGS-B iterations for this orbital solve,
            bounding the cost of one round at the price of returning before
            convergence. ``None`` uses scipy's default.
        report: Receives a progress line (iteration, energy, gradient norm,
            elapsed time) after every iteration.

    Returns:
        An :class:`OrbitalSolution` carrying the rotated orbitals, the energy, and
        the solve's own diagnostics -- iteration and evaluation counts, the
        final gradient norm, and whether it converged.

    Raises:
        ImportError: If the ``chem`` extra is not installed, propagated from
            :func:`_total_energy`.

    Warns:
        UserWarning: If the solve ended without converging, naming the
            iteration count, scipy's reason and the gradient norm.
    """
    n_orb_total = mo_coeff.shape[1]
    fixed = (n_core, fragment_specs, rdm1_active, rdm2_active, ao_eri, h_ao)

    pairs, energy_and_gradient = rotation_energy_gradient_fn(mol, mo_coeff, *fixed)
    n_rot = len(pairs)
    best_energy, best_gradient = energy_and_gradient(np.zeros(n_rot))
    rotated_mo_coeff = mo_coeff
    n_iterations = 0
    n_evaluations = 1
    stop_reason = "no rotation freedom"
    if n_rot > 0:
        rows, cols = pair_indices(pairs)
        started = time.monotonic()
        latest_gradient_norm = float(np.linalg.norm(best_gradient))

        def tracked(rotation_params: np.ndarray) -> tuple[float, np.ndarray]:
            nonlocal latest_gradient_norm
            energy, gradient = energy_and_gradient(rotation_params)
            latest_gradient_norm = float(np.linalg.norm(gradient))
            return energy, gradient

        def on_iteration(intermediate_result) -> None:
            nonlocal n_iterations
            n_iterations += 1
            if report is not None:
                report(
                    f"Orbital solve: iteration {n_iterations}, energy "
                    f"{intermediate_result.fun:.8f} Ha, |g| "
                    f"{latest_gradient_norm:.2e}, {time.monotonic() - started:.0f} s"
                )

        for _ in range(_MAX_RESTARTS):
            if np.linalg.norm(best_gradient) <= gradient_tol:
                break
            options = dict(ORBITAL_MINIMIZE_OPTIONS)
            if max_iterations is not None:
                if n_iterations >= max_iterations:
                    break
                options["maxiter"] = max_iterations - n_iterations
            res = minimize(
                tracked,
                np.zeros(n_rot),
                method="L-BFGS-B",
                jac=True,
                options=options,
                callback=on_iteration,
            )
            n_evaluations += int(res.nfev)
            stop_reason = str(res.message).strip()
            if not (np.isfinite(res.fun) and res.fun < best_energy):
                break
            rotated_mo_coeff = rotated_mo_coeff @ scipy.linalg.expm(
                rotation_generator(rows, cols, res.x, n_orb_total)
            )
            _, energy_and_gradient = rotation_energy_gradient_fn(
                mol, rotated_mo_coeff, *fixed
            )
            best_energy, best_gradient = energy_and_gradient(np.zeros(n_rot))
            n_evaluations += 1
    return _finished_solve(
        rotated_mo_coeff,
        best_energy,
        best_gradient,
        gradient_tol,
        n_iterations,
        n_evaluations,
        n_rot,
        stop_reason,
    )


def _finished_solve(
    mo_coeff: np.ndarray,
    energy: float,
    gradient: np.ndarray,
    gradient_tol: float,
    n_iterations: int,
    n_evaluations: int,
    n_rotation_pairs: int,
    stop_reason: str,
) -> OrbitalSolution:
    """The :class:`OrbitalSolution` for a finished solve, warning if it did not
    converge."""
    gradient_norm = float(np.linalg.norm(gradient))
    converged = gradient_norm <= gradient_tol
    if not converged:
        warn(
            "Orbital optimisation ended without converging after "
            f"{n_iterations} iterations and {n_evaluations} evaluations "
            f"({stop_reason}): its orbital-gradient norm {gradient_norm:.2e} "
            f"exceeds {gradient_tol:.2e}. The returned orbitals are the best "
            "seen, so the energy is still an upper bound, but this round is not a "
            "stationary point -- a small round-to-round energy change here means "
            "the optimizer gave up, not that the macro-cycle converged.",
            UserWarning,
            stacklevel=_outside_divi_stacklevel(),
        )
    return OrbitalSolution(
        mo_coeff=mo_coeff,
        energy=float(energy),
        converged=converged,
        n_iterations=n_iterations,
        n_evaluations=n_evaluations,
        gradient_norm=gradient_norm,
        n_rotation_pairs=n_rotation_pairs,
    )


def _outside_divi_stacklevel() -> int:
    """``stacklevel`` for a warning raised by this function's caller that names
    the first frame outside divi, however deep the call path into divi is."""
    frame = sys._getframe(1)
    stacklevel = 1
    while frame is not None and frame.f_code.co_filename.startswith(_DIVI_PREFIX):
        frame = frame.f_back
        stacklevel += 1
    return stacklevel


class _LASOrbitalHessian(ciah.CIAHOptimizerMixin):
    """pyscf's CIAH optimiser contract over LASSQD's rotation pairs.

    The Hessian-vector product is a central difference of the analytic
    gradient. The preconditioning diagonal is the one-body part of pyscf
    CASSCF's (``mc1step.gen_g_hop``, parts 7 and 8), built from the inactive
    plus active Fock matrix in place of the core Hamiltonian.
    """

    # Values at the orbitals ``_move_to`` last evaluated.
    _energy: float
    _gradient: np.ndarray
    _fock: np.ndarray
    _vj: np.ndarray
    _vk: np.ndarray

    def __init__(
        self,
        mol,
        mo_coeff: np.ndarray,
        n_core: int,
        fragment_specs: Sequence[FragmentSpec],
        rdm1_active: np.ndarray,
        rdm2_active: np.ndarray,
        ao_eri: np.ndarray,
        h_ao: np.ndarray,
    ):
        super().__init__(mo_coeff.shape[1])
        self._mol = mol
        self._mo_coeff = mo_coeff
        self._n_core = n_core
        self._rdm1_active = rdm1_active
        self._rdm2_active = rdm2_active
        self._ao_eri = ao_eri
        self._h_ao = h_ao
        self._rows, self._cols = pair_indices(
            rotation_pairs(self.norb, n_core, fragment_specs)
        )
        self.evaluations = 0
        self._u: np.ndarray | None = None

    @property
    def pdim(self) -> int:
        return len(self._rows)

    def pack_uniq_var(self, mat: np.ndarray) -> np.ndarray:
        return np.asarray(mat)[self._rows, self._cols]

    def unpack_uniq_var(self, v: np.ndarray) -> np.ndarray:
        return rotation_generator(self._rows, self._cols, v, self.norb)

    def _move_to(self, u: np.ndarray) -> None:
        """Evaluate at ``mo_coeff @ u`` unless ``u`` is the array last
        evaluated."""
        if u is self._u:
            return
        self._energy, self._fock, self._vj, self._vk = _energy_fock_and_potentials(
            self._mol,
            self._mo_coeff @ u,
            self._n_core,
            self._rdm1_active,
            self._rdm2_active,
            self._ao_eri,
            self._h_ao,
        )
        self.evaluations += 1
        rows, cols = self._rows, self._cols
        self._gradient = 2.0 * (self._fock[rows, cols] - self._fock[cols, rows])
        self._u = u

    def energy_and_gradient_at(self, u: np.ndarray) -> tuple[float, np.ndarray]:
        """Energy and orbital gradient at ``mo_coeff @ u``."""
        self._move_to(u)
        return self._energy, self._gradient

    def get_grad(self, u: np.ndarray) -> np.ndarray:
        self._move_to(u)
        return self._gradient

    def gen_g_hop(self, u: np.ndarray):
        self._move_to(u)
        mo_coeff = self._mo_coeff @ u

        def gradient_at(angles: np.ndarray) -> np.ndarray:
            self.evaluations += 1
            return _rotated_energy_and_gradient(
                self._mol,
                mo_coeff,
                self._rows,
                self._cols,
                angles,
                self._n_core,
                self._rdm1_active,
                self._rdm2_active,
                self._ao_eri,
                self._h_ao,
            )[1]

        def hessian_vector(vector: np.ndarray) -> np.ndarray:
            largest = float(np.abs(vector).max())
            if largest == 0.0:
                return np.zeros(self.pdim)
            scale = _HESSIAN_STEP / largest
            return (gradient_at(scale * vector) - gradient_at(-scale * vector)) / (
                2.0 * scale
            )

        return self._gradient, hessian_vector, self._hessian_diagonal(mo_coeff)

    def _hessian_diagonal(self, mo_coeff: np.ndarray) -> np.ndarray:
        """Diagonal at the last evaluated orbitals, ``mo_coeff``."""
        n_core = self._n_core
        active = slice(n_core, n_core + self._rdm1_active.shape[0])
        (j_core, j_act), (k_core, k_act) = self._vj, self._vk
        mean_field = (
            mo_coeff.T
            @ (self._h_ao + 2.0 * j_core - k_core + j_act - 0.5 * k_act)
            @ mo_coeff
        )
        fock = self._fock
        density = np.zeros((self.norb, self.norb))
        density[np.arange(n_core), np.arange(n_core)] = 2.0
        density[active, active] = self._rdm1_active
        rows, cols = self._rows, self._cols
        # Our gradient is twice pyscf's, so the Hessian is too.
        return 2.0 * (
            mean_field[rows, rows] * density[cols, cols]
            + mean_field[cols, cols] * density[rows, rows]
            - 2.0 * mean_field[rows, cols] * density[rows, cols]
            - fock[rows, rows]
            - fock[cols, cols]
        )


def ciah_orbital_solve(
    mol,
    mo_coeff: np.ndarray,
    n_core: int,
    fragment_specs: Sequence[FragmentSpec],
    rdm1_active: np.ndarray,
    rdm2_active: np.ndarray,
    ao_eri: np.ndarray,
    h_ao: np.ndarray,
    *,
    gradient_tol: float,
    max_iterations: int,
    report: Callable[[str], None] | None = None,
) -> OrbitalSolution:
    """Optimise the orbitals with pyscf's second-order CIAH solver.

    Same objective, rotation pairs and result as :func:`optimize_orbitals`;
    each macro-iteration takes an augmented-Hessian step and re-centres the
    angles on the new orbitals. A vanishing step restarts the augmented-Hessian
    solve; a restart that also gives no step ends the solve. A step that raises
    the energy hands the rest of the budget to :func:`optimize_orbitals` from
    the lowest point. The lowest-energy orbitals seen are returned, preferring a
    converged point within rounding of them.

    Args:
        max_iterations: Macro-iterations, each one gradient at new orbitals
            plus the Hessian-vector products of its augmented-Hessian solve.
        report: Receives a progress line after every macro-iteration.
    """
    hessian = _LASOrbitalHessian(
        mol,
        mo_coeff,
        n_core,
        fragment_specs,
        rdm1_active,
        rdm2_active,
        ao_eri,
        h_ao,
    )
    identity = np.eye(mo_coeff.shape[1])
    rotation = identity
    best_energy, best_gradient = hessian.energy_and_gradient_at(rotation)
    best_mo_coeff = mo_coeff
    n_iterations = 0
    rose = False
    if not hessian.pdim:
        stop_reason = "no rotation freedom"
    elif np.linalg.norm(best_gradient) <= gradient_tol:
        stop_reason = "converged"
    else:
        stop_reason = "macro-iteration limit"
        started = time.monotonic()

        def start():
            # pyscf seeds a solve with its last step; a fresh one uses the gradient.
            steps = ciah.rotate_orb_cc(hessian, rotation, gradient_tol, verbose=0)
            return steps, next(steps)[0], True

        steps, step, fresh = start()
        while n_iterations < max_iterations:
            if np.abs(step - identity).max() <= _STALLED_STEP:
                if fresh:
                    stop_reason = "stalled"
                    break
                steps.close()
                steps, step, fresh = start()
                continue
            fresh = False
            n_iterations += 1
            rotation = rotation @ step
            energy, gradient = hessian.energy_and_gradient_at(rotation)
            gradient_norm = float(np.linalg.norm(gradient))
            converged = gradient_norm <= gradient_tol
            rose = energy > best_energy + _ENERGY_ROUNDING
            if not rose and (energy < best_energy or converged):
                best_energy, best_gradient = energy, gradient
                best_mo_coeff = mo_coeff @ rotation
            if report is not None:
                report(
                    f"Orbital solve: iteration {n_iterations}, energy {energy:.8f} "
                    f"Ha, |g| {gradient_norm:.2e}, {time.monotonic() - started:.0f} s"
                )
            if rose:
                stop_reason = "energy rose"
                break
            if converged:
                stop_reason = "converged"
                break
            if n_iterations < max_iterations:
                step = steps.send(rotation)[0]
        steps.close()

    if rose and n_iterations < max_iterations:
        # The augmented-Hessian step can climb towards a saddle; L-BFGS-B only
        # descends.
        descent = optimize_orbitals(
            mol,
            best_mo_coeff,
            n_core,
            fragment_specs,
            rdm1_active,
            rdm2_active,
            ao_eri,
            h_ao,
            gradient_tol=gradient_tol,
            max_iterations=max_iterations - n_iterations,
            report=report,
        )
        return replace(
            descent,
            n_iterations=n_iterations + descent.n_iterations,
            n_evaluations=hessian.evaluations + descent.n_evaluations,
        )
    return _finished_solve(
        best_mo_coeff,
        best_energy,
        best_gradient,
        gradient_tol,
        n_iterations,
        hessian.evaluations,
        hessian.pdim,
        stop_reason,
    )


def escape_orbital_saddle(
    mol,
    mo_coeff: np.ndarray,
    n_core: int,
    fragment_specs: Sequence[FragmentSpec],
    rdm1_active: np.ndarray,
    rdm2_active: np.ndarray,
    ao_eri: np.ndarray,
    h_ao: np.ndarray,
    *,
    min_energy_drop: float,
) -> tuple[tuple[np.ndarray, float] | None, bool]:
    """Orbitals past a saddle point of the fixed-RDM energy at ``mo_coeff``,
    with their energy, or ``None`` if none lower by ``min_energy_drop`` is
    found; and whether the check was conclusive.

    The lowest orbital-Hessian eigenpair comes from a Davidson solve
    preconditioned by the Hessian diagonal, as in pyscf's SCF stability
    analysis. Below ``_SADDLE_CURVATURE`` the orbitals are rotated along that
    mode in both directions, keeping the lowest energy. A negative eigenvalue
    proves the coupled orbital-CI energy is also at a saddle; a positive one
    does not prove a minimum. The check is inconclusive when the solve stops
    unconverged above ``_SADDLE_CURVATURE``, since a lower mode may remain.
    """
    hessian = _LASOrbitalHessian(
        mol,
        mo_coeff,
        n_core,
        fragment_specs,
        rdm1_active,
        rdm2_active,
        ao_eri,
        h_ao,
    )
    if not hessian.pdim:
        return None, True
    identity = np.eye(mo_coeff.shape[1])
    energy, _ = hessian.energy_and_gradient_at(identity)
    _, hessian_vector, diagonal = hessian.gen_g_hop(identity)

    def precondition(residual, eigenvalue, _vector):
        shifted = diagonal - eigenvalue
        shifted[np.abs(shifted) < _PRECONDITIONER_FLOOR] = _PRECONDITIONER_FLOOR
        return residual / shifted

    # A random component reaches modes the lowest-diagonal guess is blind to.
    guess = np.random.default_rng(0).standard_normal(hessian.pdim)
    guess *= 0.1 / np.linalg.norm(guess)
    guess[np.argmin(diagonal)] += 1.0
    converged, eigenvalues, modes = lib.davidson1(
        lambda vectors: [hessian_vector(vector) for vector in vectors],
        guess,
        precondition,
        tol=_DAVIDSON_TOL,
        max_space=_DAVIDSON_SPACE,
        verbose=0,
    )
    eigenvalue, mode = float(np.asarray(eigenvalues)[0]), modes[0]
    if not eigenvalue < _SADDLE_CURVATURE:
        return None, bool(converged[0])
    mode = mode / np.abs(mode).max()
    best_energy, best_rotation = energy, identity
    for sign in (1.0, -1.0):
        angle = _ESCAPE_FIRST_ANGLE
        while angle <= np.pi / 4:
            rotation = scipy.linalg.expm(hessian.unpack_uniq_var(sign * angle * mode))
            rotated_energy, _ = hessian.energy_and_gradient_at(rotation)
            if rotated_energy < best_energy:
                best_energy, best_rotation = rotated_energy, rotation
            angle *= 2.0
    if not best_energy < energy - min_energy_drop:
        return None, True
    return (mo_coeff @ best_rotation, float(best_energy)), True


class _SubspaceCI(selected_ci.SelectedCI):
    """PySCF CI solver confined to a fixed alpha x beta determinant set,
    solving for ``nroots`` states of one spin multiplicity.

    Diagonalises the spin-penalised projected Hamiltonian as SQD does; the
    returned energies are the unpenalised ``<H>``. PySCF's selected-CI
    routines supply the RDMs and ``<S^2>``.
    """

    def __init__(
        self,
        strings_alpha: Sequence[str],
        strings_beta: Sequence[str],
        multiplicity: int,
        nroots: int,
        lambda_penalty: float,
    ):
        super().__init__()
        self.verbose = 0
        n_orb = len(strings_alpha[0])
        self._strs = (
            np.array(
                [ci_string_to_int(half) for half in strings_alpha], dtype=np.int64
            ),
            np.array([ci_string_to_int(half) for half in strings_beta], dtype=np.int64),
        )
        self._shape = (len(strings_alpha), len(strings_beta))
        self._dets = subspace_determinants(strings_alpha, strings_beta, n_orb)
        self._spin = 0.5 * (multiplicity - 1)
        self.nroots = nroots
        self._lambda_penalty = lambda_penalty

    def kernel(
        self,
        h1e,
        eri,
        norb,
        nelec,
        ci0=None,
        tol=None,
        lindep=None,
        max_cycle=None,
        max_space=None,
        nroots=None,
        davidson_only=None,
        max_memory=None,
        verbose=None,
        ecore=0,
        **kwargs,
    ):
        energies, civecs = self.roots(h1e, eri, norb, ecore)
        if self.nroots == 1:
            return energies[0], civecs[0]
        return energies, civecs

    def roots(
        self, h1e, eri, norb: int, ecore: float = 0.0
    ) -> tuple[list[float], list[np.ndarray]]:
        """Every root's energy and CI vector, as lists whatever ``nroots``."""
        h_spin, g_spin = spin_orbital_integrals(
            np.asarray(h1e), ao2mo.restore(1, eri, norb), norb
        )
        h_proj, s2_proj = projected_matrices(*self._dets, h_spin, g_spin, norb)
        deviation = s2_proj - self._spin * (self._spin + 1.0) * scipy.sparse.eye_array(
            len(self._dets[0]), format="csr"
        )
        _, vectors = lowest_roots(
            h_proj,
            deviation,
            self._lambda_penalty,
            self.nroots,
            spin=self._spin,
            block=True,
        )
        energies = [float(v @ (h_proj @ v)) + ecore for v in vectors.T]
        civecs = [
            selected_ci._as_SCIvector(v.reshape(self._shape), self._strs)
            for v in vectors.T
        ]
        return energies, civecs

    approx_kernel = kernel


def abelian_molecule(mol):
    """``mol``, or for a linear molecule a copy that works in D2h or C2v.

    PySCF pairs a linear molecule's degenerate orbitals by their energies,
    which a rotation within an irrep separates.
    """
    subgroup = {"Dooh": "D2h", "Coov": "C2v"}.get(mol.groupname)
    if subgroup is None:
        return mol
    abelian = mol.copy()
    abelian.symmetry_subgroup = subgroup
    abelian.build(dump_input=False, parse_arg=False)
    return abelian


def orbital_irreps(mol, mo_coeff: np.ndarray) -> np.ndarray | None:
    """Each orbital's irrep under the abelian point group ``mol`` works in, or
    ``None`` when ``mol`` has no symmetry or an orbital mixes irreps."""
    if not mol.symmetry:
        return None
    try:
        return hf_symm.get_orbsym(
            abelian_molecule(mol), mo_coeff, mol.intor_symmetric("int1e_ovlp"), True
        )
    except ValueError:
        return None


def _broken_symmetry(
    mol, mo_coeff: np.ndarray, n_core: int, n_active: int
) -> str | None:
    """``mol``'s point group, detected even when ``mol`` was built without
    symmetry, when the core, active or virtual space of ``mo_coeff`` mixes its
    irreps; ``None`` when all three are symmetry-adapted.

    Each space's projection onto an irrep is a projector exactly when the space
    is symmetry-adapted, whatever rotation mixes orbitals within it.
    """
    symmetric = mol
    if not mol.symmetry:
        symmetric = mol.copy()
        symmetric.symmetry = True
        symmetric.build(dump_input=False, parse_arg=False)
    overlap = mol.intor_symmetric("int1e_ovlp")
    bounds = (0, n_core, n_core + n_active, mo_coeff.shape[1])
    for start, stop in pairwise(bounds):
        if start == stop:
            continue
        projected = overlap @ mo_coeff[:, start:stop]
        for irrep_basis in symmetric.symm_orb:
            component = irrep_basis.T @ projected
            weight = component.T @ scipy.linalg.solve(
                irrep_basis.T @ overlap @ irrep_basis, component, assume_a="pos"
            )
            if np.abs(weight @ weight - weight).max() > _SYMMETRY_TOL:
                return symmetric.groupname
    return None


def _exact_mean_field(mol, ao_eri: np.ndarray, h_ao: np.ndarray) -> hf.RHF:
    """An RHF object whose integrals are ``ao_eri`` and ``h_ao``, for PySCF
    CASCI and CASSCF to build their Hamiltonians from, with the
    non-symmetry-adapted classes, since LASSQD's orbitals break point-group
    symmetry."""
    mean_field = hf.RHF(mol)
    mean_field._chkfile.close()
    mean_field.chkfile = None
    mean_field._eri = ao_eri
    mean_field.get_hcore = lambda mol=None: h_ao
    return mean_field


def subspace_states_at(
    mol,
    subspace: _SampledSubspace,
    n_core: int,
    ao_eri: np.ndarray,
    h_ao: np.ndarray,
) -> tuple[ElectronicState, ...]:
    """Each requested state with the subspace re-diagonalised at fixed
    orbitals, as CASCI would, in root order.

    ``subspace.mo_coeff``'s active block must be the basis its strings refer
    to. Energies are the unpenalised ``<H>``.
    """
    return tuple(
        state for state, _ in _subspace_roots(mol, subspace, n_core, ao_eri, h_ao)
    )


def _subspace_roots(
    mol,
    subspace: _SampledSubspace,
    n_core: int,
    ao_eri: np.ndarray,
    h_ao: np.ndarray,
) -> Iterator[tuple[ElectronicState, np.ndarray]]:
    """Each requested state of ``subspace`` at its orbitals, with its CI
    vector, in root order, labelled by ``subspace.labels`` when set."""
    strings_alpha, strings_beta = subspace.strings
    n_active = len(strings_alpha[0])
    nelec = (strings_alpha[0].count("1"), strings_beta[0].count("1"))
    casci = mcscf.casci.CASCI(_exact_mean_field(mol, ao_eri, h_ao), n_active, nelec)
    casci.ncore = n_core
    h1e, energy_core = casci.get_h1eff(subspace.mo_coeff)
    eri = casci.get_h2eff(subspace.mo_coeff)
    for multiplicity, n_roots in sorted(subspace.states.items()):
        solver = _SubspaceCI(
            strings_alpha,
            strings_beta,
            multiplicity,
            min(n_roots, len(strings_alpha) * len(strings_beta)),
            subspace.lambda_penalty,
        )
        energies, civecs = solver.roots(h1e, eri, n_active, energy_core)
        index = 0
        for energy, civec in zip(energies, civecs):
            spin_square, _ = solver.spin_square(civec, n_active, nelec)
            if not resolves_spin(multiplicity, spin_square):
                continue
            _, label = (subspace.labels or {}).get(
                (multiplicity, index), (multiplicity, index)
            )
            state = ElectronicState(
                multiplicity, label, float(energy), float(spin_square)
            )
            yield state, np.asarray(civec)
            index += 1


def subspace_wavefunctions(
    mol,
    subspace: _SampledSubspace,
    n_core: int,
    ao_eri: np.ndarray,
    h_ao: np.ndarray,
) -> tuple[StateWavefunction, ...]:
    """Each requested state of ``subspace`` at its orbitals, with its CI vector
    and RDMs, in root order."""
    strings_alpha, strings_beta = (tuple(half) for half in subspace.strings)
    n_active = len(strings_alpha[0])
    core_orbitals = subspace.mo_coeff[:, :n_core]
    active_orbitals = subspace.mo_coeff[:, n_core : n_core + n_active]
    wavefunctions = []
    for state, amplitudes in _subspace_roots(mol, subspace, n_core, ao_eri, h_ao):
        rdm1, rdm2, _, _ = compute_spatial_rdms(
            strings_alpha, strings_beta, amplitudes, n_active
        )
        wavefunctions.append(
            StateWavefunction(
                state=state,
                core_orbitals=core_orbitals,
                active_orbitals=active_orbitals,
                strings_alpha=strings_alpha,
                strings_beta=strings_beta,
                amplitudes=amplitudes,
                rdm1=rdm1,
                rdm2=rdm2,
            )
        )
    return tuple(wavefunctions)


def subspace_nevpt2(
    mol,
    subspace: _SampledSubspace,
    n_core: int,
    ao_eri: np.ndarray,
    h_ao: np.ndarray,
) -> tuple[tuple[ElectronicState, float], ...]:
    """Each requested state of ``subspace`` at its orbitals with its strongly
    contracted NEVPT2 correction, in root order.

    Raises:
        MemoryError: If one full-space CI vector exceeds ``mol.max_memory``.
    """
    strings_alpha, strings_beta = subspace.strings
    n_active = len(strings_alpha[0])
    nelec = (strings_alpha[0].count("1"), strings_beta[0].count("1"))
    n_bytes = 8 * math.comb(n_active, nelec[0]) * math.comb(n_active, nelec[1])
    if n_bytes > mol.max_memory * 1e6:
        raise MemoryError(
            f"NEVPT2 needs the full-space CI vector of CAS({sum(nelec)},{n_active}), "
            f"{n_bytes / 1e6:g} MB, above mol.max_memory={mol.max_memory:g} MB. "
            "Raise mol.max_memory if the machine has room."
        )
    corrections = []
    for state, amplitudes in _subspace_roots(mol, subspace, n_core, ao_eri, h_ao):
        casci = mcscf.casci.CASCI(_exact_mean_field(mol, ao_eri, h_ao), n_active, nelec)
        casci.ncore = n_core
        casci.verbose = 0
        casci.mo_coeff = subspace.mo_coeff
        casci.ci = selected_ci.to_fci(
            sci_vector(strings_alpha, strings_beta, amplitudes), n_active, nelec
        )
        corrections.append((state, float(mrpt.NEVPT(casci).kernel())))
    return tuple(corrections)


def track_states(
    previous: _SampledSubspace,
    current: _SampledSubspace,
    n_core: int,
    mol,
    ao_eri: np.ndarray,
    h_ao: np.ndarray,
) -> dict[tuple[int, int], tuple[int, int]]:
    """Each of ``current``'s roots' label, ``(multiplicity, root index)`` to
    ``(multiplicity, index)``: that of the previous state of its multiplicity
    whose CI vector it overlaps most, or else the lowest unused index."""
    labels = {}
    for multiplicity, before, now, overlaps in _root_overlaps(
        previous, current, n_core, mol, ao_eri, h_ao
    ):
        matched: dict[int, int] = {}
        if before:
            for i, j in zip(*scipy.optimize.linear_sum_assignment(-overlaps)):
                matched[int(i)] = before[j].index
        free = (
            index
            for index in range(len(now) + len(before))
            if index not in matched.values()
        )
        for i, state in enumerate(now):
            labels[(multiplicity, state.index)] = (
                multiplicity,
                matched[i] if i in matched else next(free),
            )
    return labels


def _root_overlaps(
    previous: _SampledSubspace,
    current: _SampledSubspace,
    n_core: int,
    mol,
    ao_eri: np.ndarray,
    h_ao: np.ndarray,
) -> Iterator[tuple[int, list[ElectronicState], list[ElectronicState], np.ndarray]]:
    """Per multiplicity, the previous and current (root-order) states and
    ``|<previous_j|current_i>|`` as rows ``i``, columns ``j``.

    The previous determinants are carried into the current orbitals by the
    one-to-one orbital assignment carryover uses, with the orbitals' phases and
    the reordering's sign.
    """
    n_active = len(current.strings[0][0])
    active = slice(n_core, n_core + n_active)
    overlap = (
        current.mo_coeff[:, active].T
        @ mol.intor_symmetric("int1e_ovlp")
        @ previous.mo_coeff[:, active]
    )
    _, source = scipy.optimize.linear_sum_assignment(np.abs(overlap), maximize=True)
    target = np.argsort(source)
    phases = np.sign(overlap[np.arange(n_active), source])

    def carried(strings, positions):
        """Each previous string's row in the current string list (-1 where it
        has none) and sign."""
        rows, signs = [], []
        for string in strings:
            occupied = [int(target[q]) for q, bit in enumerate(string) if bit == "1"]
            inversions = sum(1 for a, b in combinations(occupied, 2) if a > b)
            mapped = "".join("1" if p in occupied else "0" for p in range(n_active))
            rows.append(positions.get(mapped, -1))
            signs.append((-1) ** inversions * np.prod(phases[occupied]))
        return np.array(rows), np.array(signs)

    rows, row_signs = carried(
        previous.strings[0], {s: i for i, s in enumerate(current.strings[0])}
    )
    columns, column_signs = carried(
        previous.strings[1], {s: j for j, s in enumerate(current.strings[1])}
    )
    kept_rows, kept_columns = rows >= 0, columns >= 0

    def projected(amplitudes):
        """A previous CI vector over the current determinants."""
        signed = amplitudes * np.outer(row_signs, column_signs)
        vector = np.zeros((len(current.strings[0]), len(current.strings[1])))
        vector[np.ix_(rows[kept_rows], columns[kept_columns])] = signed[
            np.ix_(kept_rows, kept_columns)
        ]
        return vector

    previous_roots = list(_subspace_roots(mol, previous, n_core, ao_eri, h_ao))
    current_roots = list(
        _subspace_roots(mol, replace(current, labels=None), n_core, ao_eri, h_ao)
    )
    for multiplicity in sorted({state.multiplicity for state, _ in current_roots}):
        now = [root for root in current_roots if root[0].multiplicity == multiplicity]
        before = [
            (state, projected(amplitudes))
            for state, amplitudes in previous_roots
            if state.multiplicity == multiplicity
        ]
        overlaps = np.array(
            [[abs(np.sum(old * new)) for _, old in before] for _, new in now]
        ).reshape(len(now), len(before))
        yield (
            multiplicity,
            [state for state, _ in before],
            [state for state, _ in now],
            overlaps,
        )


def state_transitions(
    mol, wavefunctions: Sequence[StateWavefunction]
) -> tuple[StateTransition, ...]:
    """Every pair of same-multiplicity states' one-electron transition, the
    earlier state of ``wavefunctions`` first."""
    if not wavefunctions:
        return ()
    orbitals = wavefunctions[0].active_orbitals
    n_active = orbitals.shape[1]
    dipole_mo = np.einsum(
        "xab,ap,bq->xpq", mol.intor_symmetric("int1e_r"), orbitals, orbitals
    )
    transitions = []
    for lower, upper in combinations(wavefunctions, 2):
        if lower.state.multiplicity != upper.state.multiplicity:
            continue
        nelec = (lower.strings_alpha[0].count("1"), lower.strings_beta[0].count("1"))
        transition_rdm1 = selected_ci.trans_rdm1(
            sci_vector(upper.strings_alpha, upper.strings_beta, upper.amplitudes),
            sci_vector(lower.strings_alpha, lower.strings_beta, lower.amplitudes),
            n_active,
            nelec,
        )
        dipole = -np.einsum("xpq,pq->x", dipole_mo, transition_rdm1)
        gap = upper.state.energy - lower.state.energy
        particles, singular_values, holes = np.linalg.svd(transition_rdm1)
        total = float(np.sum(singular_values**2))
        transitions.append(
            StateTransition(
                initial=lower.state,
                final=upper.state,
                transition_rdm1=transition_rdm1,
                transition_dipole=dipole,
                oscillator_strength=float(2.0 / 3.0 * gap * dipole @ dipole),
                hole_orbitals=orbitals @ holes.T,
                particle_orbitals=orbitals @ particles,
                nto_weights=(
                    singular_values**2 / total
                    if total > 0
                    else np.zeros_like(singular_values)
                ),
            )
        )
    return tuple(transitions)


def state_averaged_orbital_solve(
    mol,
    mo_coeff: np.ndarray,
    n_core: int,
    spec: FragmentSpec,
    strings: tuple[Sequence[str], Sequence[str]],
    weights: Mapping[tuple[int, int], float],
    lambda_penalty: float,
    ao_eri: np.ndarray,
    h_ao: np.ndarray,
    *,
    gradient_tol: float,
    max_iterations: int,
    report: Callable[[str], None] | None = None,
) -> tuple[OrbitalSolution, tuple[ElectronicState, ...]]:
    """State-averaged CASSCF over one fragment, its CI confined to an SQD
    subspace.

    Orbitals and CI vectors move together (PySCF's one-step SA-CASSCF), so the
    solve leaves saddle points that alternating orbital and CI updates stall
    at. ``mo_coeff``'s active block must be the basis ``strings`` refer to.
    The Hamiltonian is built from ``ao_eri`` and ``h_ao``, the integrals every
    other LASSQD energy uses, never a density-fitted approximation. Convergence
    and the reported gradient use LASSQD's own gradient at the returned
    orbitals and the averaged RDMs.

    Args:
        strings: The subspace's alpha and beta occupation strings.
        weights: Weight per state, keyed ``(multiplicity, index)``.
        max_iterations: Cap on SA-CASSCF macro-iterations.
        report: Receives a progress line after every macro-iteration.

    Returns:
        ``(solve, states)``: the solve, its energy the weighted average, and
        each state at the returned orbitals, in root order.
    """
    strings_alpha, strings_beta = strings
    roots = dict(sorted(Counter(multiplicity for multiplicity, _ in weights).items()))
    solvers = [
        _SubspaceCI(strings_alpha, strings_beta, multiplicity, n_roots, lambda_penalty)
        for multiplicity, n_roots in roots.items()
    ]
    labels = [
        (multiplicity, index)
        for multiplicity, n_roots in roots.items()
        for index in range(n_roots)
    ]
    n_active = spec.n_orbitals
    nelec = (spec.n_alpha, spec.n_beta)

    keeps_symmetry = orbital_irreps(mol, mo_coeff) is not None
    if keeps_symmetry:
        casscf = mcscf.mc1step_symm.SymAdaptedCASSCF(
            _exact_mean_field(abelian_molecule(mol), ao_eri, h_ao), n_active, nelec
        )
    else:
        casscf = mcscf.mc1step.CASSCF(
            _exact_mean_field(mol, ao_eri, h_ao), n_active, nelec
        )
    casscf.ncore = n_core
    casscf.verbose = 0
    casscf.canonicalization = False
    casscf.max_cycle_macro = max_iterations
    casscf.conv_tol = gradient_tol**2
    # PySCF's orbital gradient is half LASSQD's.
    casscf.conv_tol_grad = gradient_tol / 2
    casscf = mcscf.addons.state_average_mix(
        casscf, solvers, [weights[label] for label in labels]
    )
    started = time.monotonic()
    macro_iterations = 0

    def progress(envs):
        nonlocal macro_iterations
        macro_iterations = envs["imacro"]
        if report is not None:
            report(
                f"State-averaged orbital solve: iteration {macro_iterations}, "
                f"energy {envs['e_tot']:.8f} Ha, |g| {2 * envs['norm_gorb']:.2e}, "
                f"{time.monotonic() - started:.0f} s"
            )

    casscf.callback = progress
    casscf.kernel(mo_coeff)
    group = (
        None
        if keeps_symmetry
        else _broken_symmetry(mol, casscf.mo_coeff, n_core, n_active)
    )
    if group is not None:
        remedy = (
            "Pass a mean field with symmetry-adapted orbitals"
            if mol.symmetry
            else "Build the molecule with symmetry=True"
        )
        warn(
            f"The state-averaged solve broke the molecule's {group} symmetry, "
            "mixing states of different symmetry; which such solution a run "
            f"reaches depends on rounding. {remedy} to keep the symmetry.",
            UserWarning,
            stacklevel=_outside_divi_stacklevel(),
        )

    rdm1, rdm2 = casscf.fcisolver.make_rdm12(casscf.ci, n_active, nelec)
    pairs, energy_and_gradient = rotation_energy_gradient_fn(
        mol, casscf.mo_coeff, n_core, [spec], rdm1, rdm2, ao_eri, h_ao
    )
    energy, gradient = energy_and_gradient(np.zeros(len(pairs)))
    solve = _finished_solve(
        casscf.mo_coeff,
        energy,
        gradient,
        gradient_tol,
        macro_iterations,
        macro_iterations,
        len(pairs),
        "converged" if casscf.converged else "macro-iteration limit",
    )
    if solve.converged and not casscf.converged:
        warn(
            f"The state-averaged solve stopped at max_iterations={max_iterations} "
            "before settling, though its orbital gradient is below tolerance. "
            "Unless that cap is deliberately low, this usually means a requested "
            "state is degenerate with one left out, such as one component of a "
            "degenerate pair, so the averaged energy has a flat direction. "
            "Request every component of the degenerate set.",
            UserWarning,
            stacklevel=_outside_divi_stacklevel(),
        )
    spin_squares = casscf.fcisolver.states_spin_square(casscf.ci, n_active, nelec)[0]
    states: list[ElectronicState] = []
    for (multiplicity, _), state_energy, spin_square in zip(
        labels, casscf.e_states, spin_squares
    ):
        if resolves_spin(multiplicity, spin_square):
            index = sum(1 for state in states if state.multiplicity == multiplicity)
            states.append(
                ElectronicState(
                    multiplicity, index, float(state_energy), float(spin_square)
                )
            )
    return solve, tuple(states)
