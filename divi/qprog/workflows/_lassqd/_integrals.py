# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Effective per-fragment integrals, the LASSQD total-energy functional, and
the two orbital solves."""

import os
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import accumulate, combinations, permutations, product
from pathlib import Path
from warnings import warn

import numpy as np
import scipy.linalg
from pyscf import ao2mo
from pyscf.scf import hf
from pyscf.soscf import ciah
from scipy.optimize import minimize

from ._state import FragmentSpec, FragmentState

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
# A converged point this far above the best one seen still replaces it.
_ENERGY_ROUNDING = 1e-10
# Frames under this prefix are divi's own, never the code a warning is for.
_DIVI_PREFIX = os.path.join(Path(__file__).parents[3], "")


@dataclass(frozen=True)
class OrbitalSolve:
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
    """One round's active-space integrals and frozen-core potentials.

    Every array is indexed over the active space alone, in the fragment-block
    order of the ``mo_coeff`` passed to :func:`transform_integrals`. The core
    and virtual blocks are not stored: the only consumer,
    :func:`fragment_effective_integrals`, reads neither, and materialising the
    full register costs ``n_orb ** 4``.

    Attributes:
        h_act: ``(n_act, n_act)`` one-electron integrals.
        g_act: ``(n_act,) * 4`` two-electron integrals in chemist order.
        j_core: ``(n_act, n_act)`` frozen-core Coulomb potential.
        k_core: ``(n_act, n_act)`` frozen-core exchange potential.
    """

    h_act: np.ndarray
    g_act: np.ndarray
    j_core: np.ndarray
    k_core: np.ndarray


def fragment_blocks(specs: Sequence[FragmentSpec], offset: int = 0) -> list[slice]:
    """Each fragment's contiguous orbital span, starting at ``offset``."""
    bounds = list(accumulate((spec.n_orbitals for spec in specs), initial=offset))
    return [slice(start, stop) for start, stop in zip(bounds, bounds[1:])]


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
    remaining = [o for o in range(n_orbitals_total) if o not in set(active)]
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
    """Transform the active-space integrals and build the frozen-core potentials.

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

    core_coeff = mo_coeff[:, :n_core]
    active_coeff = mo_coeff[:, n_core : n_core + n_act]

    # Occupation-one core density, so vj/vk come out as sum_i (pq|ii) and
    # sum_i (pi|iq) directly.
    vj_core, vk_core = hf.dot_eri_dm(ao_eri, core_coeff @ core_coeff.T, hermi=1)

    return MOIntegrals(
        h_act=active_coeff.T @ h_ao @ active_coeff,
        g_act=ao2mo.incore.general(ao_eri, (active_coeff,) * 4, compact=False).reshape(
            (n_act,) * 4
        ),
        j_core=active_coeff.T @ vj_core @ active_coeff,
        k_core=active_coeff.T @ vk_core @ active_coeff,
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

    n_act = integrals.h_act.shape[0]
    if running != n_act:
        raise ValueError(
            f"Fragments cover {running} orbitals but the active-space integrals "
            f"span {n_act}. `integrals` and `fragments` were built from "
            "different fragment specs."
        )

    target = blocks[index]
    core_embedded = (
        integrals.h_act[target, target]
        + 2.0 * integrals.j_core[target, target]
        - integrals.k_core[target, target]
    )
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
) -> OrbitalSolve:
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
        An :class:`OrbitalSolve` carrying the rotated orbitals, the energy, and
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
) -> OrbitalSolve:
    """The :class:`OrbitalSolve` for a finished solve, warning if it did not
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
    return OrbitalSolve(
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
) -> OrbitalSolve:
    """Optimise the orbitals with pyscf's second-order CIAH solver.

    Same objective, rotation pairs and result as :func:`optimize_orbitals`;
    each macro-iteration takes an augmented-Hessian step and re-centres the
    angles on the new orbitals. A vanishing step restarts the augmented-Hessian
    solve; a restart that also gives no step ends the solve. The lowest-energy
    orbitals seen are returned, preferring a converged point within rounding of
    them.

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
            if energy < best_energy or (
                converged and energy <= best_energy + _ENERGY_ROUNDING
            ):
                best_energy, best_gradient = energy, gradient
                best_mo_coeff = mo_coeff @ rotation
            if report is not None:
                report(
                    f"Orbital solve: iteration {n_iterations}, energy {energy:.8f} "
                    f"Ha, |g| {gradient_norm:.2e}, {time.monotonic() - started:.0f} s"
                )
            if converged:
                stop_reason = "converged"
                break
            if n_iterations < max_iterations:
                step = steps.send(rotation)[0]
        steps.close()

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
