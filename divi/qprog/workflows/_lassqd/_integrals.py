# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Effective per-fragment integrals and the LASSQD total-energy functional.

Implements the frozen-core / active-space integral machinery for LASSQD:

1. The AO-basis electron-repulsion integral is computed once per run
   (:func:`cached_ao_eri`) and reused for every :func:`_total_energy`
   evaluation, rather than re-running a full four-index ``ao2mo.kernel``
   transform on every orbital-rotation loss evaluation.
2. :func:`build_active_permutation` honours the caller's requested orbital
   indices instead of silently discarding them for a contiguous range.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import accumulate, combinations, permutations, product
from warnings import warn

import numpy as np
import scipy.linalg
from scipy.optimize import minimize

from ._state import FragmentSpec, FragmentState

# L-BFGS-B ftol is relative, so a tol=1e-6 shorthand halts around 1e-6 * |E|,
# coarser than LASSQD's energy_tol. Tighter than this terminates ABNORMAL.
ORBITAL_MINIMIZE_OPTIONS = {"ftol": 1e-12, "gtol": 1e-6}


@dataclass(frozen=True)
class OrbitalSolve:
    """Outcome of one round's orbital re-optimisation.

    Attributes:
        mo_coeff: The rotated MO coefficients.
        energy: Total energy at the returned orbitals.
        converged: Whether the optimizer reported success rather than exhausting
            its budget. A capped solve still returns its best point, so the
            energy remains an upper bound, but not a stationary point.
        n_iterations: Optimizer iterations taken.
        n_evaluations: Objective evaluations taken, each one four-index MO
            transform.
        gradient_norm: Largest absolute orbital-gradient component at the
            returned orbitals. A solve that stops on its energy-reduction
            tolerance can report ``converged`` with this well above
            ``ORBITAL_MINIMIZE_OPTIONS``' gradient tolerance.
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
    """
    # optional ``chem`` extra
    from pyscf import ao2mo
    from pyscf.scf import hf

    if ao_eri is None:
        ao_eri = cached_ao_eri(mol)
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
    running = blocks[-1].stop if blocks else 0

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
    """Compute the AO-basis one-electron (kinetic + nuclear) integral once per run.

    The returned array is suitable as the ``h_ao`` argument to
    :func:`_total_energy` for every orbital-rotation loss evaluation in a
    macro-cycle, avoiding repeated AO integral recomputation.
    """
    return mol.intor("int1e_kin") + mol.intor("int1e_nuc")


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


def _build_energy_rdms(
    n_orb: int, n_core: int, rdm1_active: np.ndarray, rdm2_active: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Build the full-register 1- and 2-particle densities of :func:`_total_energy`.

    Returns dense ``D`` and ``d`` over the whole permuted MO register such that

    .. math::

        E = E_\\mathrm{nuc} + \\sum_{mn} h_{mn} D_{mn}
            + \\tfrac{1}{2} \\sum_{mnop} d_{mnop} g_{mnop}

    reproduces :func:`_total_energy` exactly, with ``h`` and ``g`` the MO-basis
    one- and two-electron integrals (chemist order) over the same register.
    Frozen-core orbitals occupy ``[0, n_core)``, the active space follows, and
    the virtual block is zero in both densities.

    Args:
        n_orb: Size of the full MO register.
        n_core: Number of frozen-core orbitals.
        rdm1_active: ``(n_act, n_act)`` active-space 1-RDM.
        rdm2_active: ``(n_act,) * 4`` active-space 2-RDM.

    Returns:
        ``(D, d)`` of shapes ``(n_orb, n_orb)`` and ``(n_orb,) * 4``.
    """
    n_act = rdm1_active.shape[0]
    active = slice(n_core, n_core + n_act)

    one_rdm = np.zeros((n_orb, n_orb))
    two_rdm = np.zeros((n_orb,) * 4)

    for i in range(n_core):
        one_rdm[i, i] = 2.0
    one_rdm[active, active] = rdm1_active

    for i in range(n_core):
        for j in range(n_core):
            two_rdm[i, i, j, j] += 4.0
            two_rdm[i, j, j, i] -= 2.0

    for i in range(n_core):
        two_rdm[active, active, i, i] += 2.0 * rdm1_active
        two_rdm[i, i, active, active] += 2.0 * rdm1_active
        two_rdm[active, i, i, active] -= rdm1_active
        two_rdm[i, active, active, i] -= rdm1_active

    two_rdm[active, active, active, active] += rdm2_active

    return one_rdm, two_rdm


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

    Equivalent to contracting :func:`_build_energy_rdms`' dense densities against
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
    # optional ``chem`` extra
    from pyscf import ao2mo
    from pyscf.scf import hf

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
    return energy, fock


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
    ``mo_coeff @ expm(K(x))`` -- to within floating-point round-off, via the
    contracted form of :func:`_build_energy_rdms` rather than the explicit
    loops -- together with its exact derivative with respect to the rotation
    angles ``x``, sharing the single four-index MO transform between the two.

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
    n_orb_total = mo_coeff.shape[1]
    n_act = sum(spec.n_orbitals for spec in fragment_specs)

    blocks = fragment_blocks(fragment_specs, offset=n_core)
    core = range(n_core)
    active = range(n_core, n_core + n_act)
    virtual = range(n_core + n_act, n_orb_total)

    rotation_pairs: list[tuple[int, int]] = [
        *product(core, active),
        *product(core, virtual),
    ]
    # Active <-> active, across different fragments only.
    for block_a, block_b in combinations(blocks, 2):
        rotation_pairs += product(
            range(block_a.start, block_a.stop), range(block_b.start, block_b.stop)
        )
    rotation_pairs += product(active, virtual)
    rows, cols = pair_indices(rotation_pairs)

    def energy_and_gradient(
        rotation_params: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        generator = np.zeros((n_orb_total, n_orb_total))
        generator[rows, cols] = rotation_params
        generator[cols, rows] = -rotation_params

        unitary = scipy.linalg.expm(generator)
        rotated = np.dot(mo_coeff, unitary)

        energy, fock = energy_and_generalized_fock(
            mol, rotated, n_core, rdm1_active, rdm2_active, ao_eri, h_ao
        )
        pullback = np.asarray(
            scipy.linalg.expm_frechet(
                generator.T, np.dot(unitary, 2.0 * fock), compute_expm=False
            )
        )
        gradient = pullback[rows, cols] - pullback[cols, rows]

        return energy, gradient

    return rotation_pairs, energy_and_gradient


def pair_indices(
    rotation_pairs: Sequence[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    """Split rotation pairs into row and column index arrays."""
    pairs = np.asarray(rotation_pairs, dtype=int).reshape(len(rotation_pairs), 2)
    return pairs[:, 0], pairs[:, 1]


def optimize_orbitals(
    mol,
    mo_coeff: np.ndarray,
    n_core: int,
    fragment_specs: Sequence[FragmentSpec],
    rdm1_active: np.ndarray,
    rdm2_active: np.ndarray,
    ao_eri: np.ndarray,
    h_ao: np.ndarray,
    max_orbital_iterations: int | None = None,
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

    That monotonicity is also why the third return value matters: an optimizer
    that gives up returns the baseline, so the round's energy barely moves and
    looks exactly like a converged macro-cycle. The flag lets the caller tell
    the two apart.

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
        max_orbital_iterations: Cap on L-BFGS-B iterations for this orbital
            solve, bounding the cost of one round at the price of returning
            before convergence. Unrelated to any VQE iteration budget.
            ``None`` uses scipy's default.

    Returns:
        An :class:`OrbitalSolve` carrying the rotated orbitals, the energy, and
        the solve's own diagnostics -- iteration and evaluation counts, the
        final gradient norm, and whether it converged.

    Raises:
        ImportError: If the ``chem`` extra is not installed, propagated from
            :func:`_total_energy`.

    Warns:
        UserWarning: If the optimizer stopped without converging, naming the
            iteration count and scipy's reason.
    """
    n_orb_total = mo_coeff.shape[1]

    rotation_pairs, energy_and_gradient = rotation_energy_gradient_fn(
        mol,
        mo_coeff,
        n_core,
        fragment_specs,
        rdm1_active,
        rdm2_active,
        ao_eri,
        h_ao,
    )
    n_rot = len(rotation_pairs)

    init_params = np.zeros(n_rot)
    baseline_energy, baseline_gradient = energy_and_gradient(init_params)

    best_params = init_params
    best_energy = baseline_energy
    best_gradient = baseline_gradient
    converged = True
    n_iterations = 0
    n_evaluations = 1
    if n_rot > 0:
        options = dict(ORBITAL_MINIMIZE_OPTIONS)
        if max_orbital_iterations is not None:
            options["maxiter"] = max_orbital_iterations
        res = minimize(
            energy_and_gradient,
            init_params,
            method="L-BFGS-B",
            jac=True,
            options=options,
        )
        n_iterations = int(res.nit)
        n_evaluations += int(res.nfev)
        if np.isfinite(res.fun) and (
            not np.isfinite(best_energy) or res.fun < best_energy
        ):
            best_params = res.x
            best_energy = float(res.fun)
            best_gradient = np.asarray(res.jac)
        converged = bool(res.success)
        if not converged:
            warn(
                "Orbital optimisation stopped without converging after "
                f"{res.nit} iterations and {res.nfev} evaluations: "
                f"{str(res.message).strip()}. The returned orbitals are the best "
                "seen, so the energy is still an upper bound, but this round is "
                "not a stationary point -- a small round-to-round energy change "
                "here means the optimizer gave up, not that the macro-cycle "
                "converged.",
                UserWarning,
                stacklevel=2,
            )

    generator = np.zeros((n_orb_total, n_orb_total))
    if n_rot > 0:
        rows, cols = pair_indices(rotation_pairs)
        generator[rows, cols] = best_params
        generator[cols, rows] = -best_params
    rotated_mo_coeff = np.dot(mo_coeff, scipy.linalg.expm(generator))

    return OrbitalSolve(
        mo_coeff=rotated_mo_coeff,
        energy=float(best_energy),
        converged=converged,
        n_iterations=n_iterations,
        n_evaluations=n_evaluations,
        gradient_norm=(
            float(np.max(np.abs(best_gradient))) if best_gradient.size else 0.0
        ),
        n_rotation_pairs=n_rot,
    )
