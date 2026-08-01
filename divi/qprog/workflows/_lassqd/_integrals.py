# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Effective per-fragment integrals and the LASSQD total-energy functional.

Implements the frozen-core / active-space integral machinery for LASSQD:

1. The AO-basis electron-repulsion integral is computed once per run
   (:func:`cached_ao_eri`) and reused for every :func:`total_energy`
   evaluation, rather than re-running a full four-index ``ao2mo.kernel``
   transform on every orbital-rotation loss evaluation.
2. :func:`build_active_permutation` honors the caller's requested orbital
   indices instead of silently discarding them for a contiguous range.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import scipy.linalg
from scipy.optimize import minimize

from ._state import FragmentSpec, FragmentState


@dataclass(frozen=True)
class MOIntegrals:
    """One round's MO-basis integrals and frozen-core potentials.

    All arrays are indexed over the full permuted MO register (core, then
    fragment blocks in spec order, then virtual), matching the column order
    of the ``mo_coeff`` passed to :func:`transform_integrals`.

    Attributes:
        h_mo: ``(n_orb, n_orb)`` one-electron integrals.
        g_mo: ``(n_orb,) * 4`` two-electron integrals in chemist order.
        j_core: ``(n_orb, n_orb)`` frozen-core Coulomb potential.
        k_core: ``(n_orb, n_orb)`` frozen-core exchange potential.
        n_core: Number of frozen-core orbitals these integrals were built
            with; needed to locate each fragment's active block.
    """

    h_mo: np.ndarray
    g_mo: np.ndarray
    j_core: np.ndarray
    k_core: np.ndarray
    n_core: int


def build_active_permutation(
    specs: Sequence[FragmentSpec], n_core: int, n_orbitals_total: int
) -> np.ndarray:
    """Order orbitals as ``[core | fragment blocks | virtual]``.

    Downstream code assumes the active block is contiguous. This permutation
    honors the caller's requested orbital indices (``FragmentSpec.orbitals``
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


def transform_integrals(mol, mo_coeff: np.ndarray, n_core: int) -> MOIntegrals:
    """Run one four-index MO transform and build the frozen-core potentials.

    A single ``ao2mo.kernel`` call per macro-cycle, reused by every
    fragment's effective integrals for that cycle.

    Args:
        mol: A PySCF ``gto.Mole``.
        mo_coeff: ``(nao, n_orb)`` MO coefficients, already permuted so that
            columns ``[0, n_core)`` are core, ``[n_core, n_core + n_act)``
            are the active fragment blocks in spec order, and the rest are
            virtual.
        n_core: Number of frozen-core orbitals.

    Raises:
        ImportError: If the ``chem`` extra is not installed.
    """
    try:
        # pyrefly: ignore[missing-import]  # optional ``chem`` extra
        from pyscf import ao2mo
    except ImportError as exc:
        raise ImportError(
            "transform_integrals requires the 'chem' extra; "
            "install it with `pip install qoro-divi[chem]`."
        ) from exc

    n_orb = mo_coeff.shape[1]
    h_ao = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    h_mo = np.dot(mo_coeff.T, np.dot(h_ao, mo_coeff))

    g_mo = ao2mo.kernel(mol, mo_coeff)
    g_mo = ao2mo.restore(1, g_mo, n_orb)

    j_core = np.zeros((n_orb, n_orb))
    k_core = np.zeros((n_orb, n_orb))
    for p in range(n_orb):
        for q in range(n_orb):
            for i in range(n_core):
                j_core[p, q] += g_mo[p, q, i, i]
                k_core[p, q] += g_mo[p, i, i, q]

    return MOIntegrals(
        h_mo=h_mo, g_mo=g_mo, j_core=j_core, k_core=k_core, n_core=n_core
    )


def fragment_effective_integrals(
    integrals: MOIntegrals, fragments: Sequence[FragmentState], index: int
) -> tuple[np.ndarray, np.ndarray]:
    """Build one fragment's effective one- and two-body integrals.

    The frozen core contributes a mean-field potential
    (``2 * J_core - K_core``) and every *other* fragment contributes its own
    mean-field Coulomb/exchange term through its current 1-RDM, while the
    two-body integrals stay local to this fragment.

    Fragments are indexed by their permuted contiguous positions: the
    running offset of ``spec.n_orbitals`` in the order ``fragments`` is
    supplied, offset by ``integrals.n_core``. This is valid because
    ``integrals`` was built from a ``mo_coeff`` permuted with
    :func:`build_active_permutation` using the same ``n_core``.

    Args:
        integrals: MO-basis integrals for the current round.
        fragments: Every fragment's current state, in permutation order.
        index: Position of the target fragment within ``fragments``.

    Returns:
        ``(h_eff, g_frag)``: the fragment's effective one-body integrals
        (including core and other-fragment mean-field terms) and its bare
        two-body integrals, both shaped for the fragment's own orbital
        count.

    Raises:
        ValueError: If ``integrals.n_core`` plus the total number of
            orbitals across ``fragments`` exceeds the size of ``integrals``,
            which indicates ``integrals`` was built with a different
            ``n_core`` (or a different set of fragments) than supplied here.
    """
    offsets = []
    running = 0
    for fragment in fragments:
        offsets.append(running)
        running += fragment.spec.n_orbitals

    n_orb = integrals.h_mo.shape[0]
    if integrals.n_core + running > n_orb:
        raise ValueError(
            "Fragment orbitals do not fit in the MO register: n_core="
            f"{integrals.n_core} + total fragment orbitals={running} exceeds "
            f"n_orb={n_orb}. `integrals` and `fragments` were likely built "
            "with inconsistent n_core or fragment specs."
        )

    target = fragments[index]
    n_f_orb = target.spec.n_orbitals
    n_core = integrals.n_core
    h_mo, g_mo = integrals.h_mo, integrals.g_mo

    h_eff = np.zeros((n_f_orb, n_f_orb))
    for p_idx in range(n_f_orb):
        p = n_core + offsets[index] + p_idx
        for q_idx in range(n_f_orb):
            q = n_core + offsets[index] + q_idx
            h_eff[p_idx, q_idx] = (
                h_mo[p, q] + 2.0 * integrals.j_core[p, q] - integrals.k_core[p, q]
            )
            for other_idx, other in enumerate(fragments):
                if other_idx == index:
                    continue
                n_other_orb = other.spec.n_orbitals
                for r_idx in range(n_other_orb):
                    r = n_core + offsets[other_idx] + r_idx
                    for s_idx in range(n_other_orb):
                        s = n_core + offsets[other_idx] + s_idx
                        h_eff[p_idx, q_idx] += other.rdm1[r_idx, s_idx] * (
                            2.0 * g_mo[p, q, r, s] - g_mo[p, r, s, q]
                        )

    g_frag = np.zeros((n_f_orb, n_f_orb, n_f_orb, n_f_orb))
    for p_idx in range(n_f_orb):
        p = n_core + offsets[index] + p_idx
        for q_idx in range(n_f_orb):
            q = n_core + offsets[index] + q_idx
            for r_idx in range(n_f_orb):
                r = n_core + offsets[index] + r_idx
                for s_idx in range(n_f_orb):
                    s = n_core + offsets[index] + s_idx
                    g_frag[p_idx, q_idx, r_idx, s_idx] = g_mo[p, q, r, s]

    return h_eff, g_frag


def assemble_active_rdms(
    fragments: Sequence[FragmentState],
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble the full active-space 1- and 2-RDM from per-fragment RDMs.

    Fragments are placed block-diagonally in the order supplied, purely by
    position (the running offset of ``spec.n_orbitals``); no cross-fragment
    2-RDM elements are populated.
    """
    n_act = sum(fragment.spec.n_orbitals for fragment in fragments)
    rdm1 = np.zeros((n_act, n_act))
    rdm2 = np.zeros((n_act, n_act, n_act, n_act))

    offset = 0
    for fragment in fragments:
        n_f_orb = fragment.spec.n_orbitals
        rdm1[offset : offset + n_f_orb, offset : offset + n_f_orb] = fragment.rdm1

        for p in range(n_f_orb):
            for q in range(n_f_orb):
                for r in range(n_f_orb):
                    for s in range(n_f_orb):
                        rdm2[offset + p, offset + q, offset + r, offset + s] = (
                            fragment.rdm2[p, q, r, s]
                        )

        offset += n_f_orb

    return rdm1, rdm2


def cached_ao_eri(mol) -> np.ndarray:
    """Compute the AO-basis electron-repulsion integral once per run.

    The returned array is the 8-fold symmetry-packed ``int2e`` integral,
    suitable as the ``ao_eri`` argument to :func:`total_energy` for every
    orbital-rotation loss evaluation in a macro-cycle, avoiding a repeated
    ``ao2mo.kernel`` AO integral recomputation.
    """
    return mol.intor("int2e", aosym="s8")


def cached_h_ao(mol) -> np.ndarray:
    """Compute the AO-basis one-electron (kinetic + nuclear) integral once per run.

    The returned array is suitable as the ``h_ao`` argument to
    :func:`total_energy` for every orbital-rotation loss evaluation in a
    macro-cycle, avoiding repeated AO integral recomputation.
    """
    return mol.intor("int1e_kin") + mol.intor("int1e_nuc")


def total_energy(
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
    optimization and must not be paid per loss evaluation.

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
    try:
        # pyrefly: ignore[missing-import]  # optional ``chem`` extra
        from pyscf import ao2mo
    except ImportError as exc:
        raise ImportError(
            "total_energy requires the 'chem' extra; "
            "install it with `pip install qoro-divi[chem]`."
        ) from exc

    n_orb = mo_coeff.shape[1]
    h_mo = np.dot(mo_coeff.T, np.dot(h_ao, mo_coeff))

    g_mo = ao2mo.incore.full(ao_eri, mo_coeff)
    g_mo = ao2mo.restore(1, g_mo, n_orb)

    n_act = rdm1_active.shape[0]

    e_core = mol.energy_nuc()
    for i in range(n_core):
        e_core += 2.0 * h_mo[i, i]
        for j in range(n_core):
            e_core += 2.0 * g_mo[i, i, j, j] - g_mo[i, j, j, i]

    j_core = np.zeros((n_act, n_act))
    k_core = np.zeros((n_act, n_act))
    for p in range(n_act):
        p_mo = n_core + p
        for q in range(n_act):
            q_mo = n_core + q
            for i in range(n_core):
                j_core[p, q] += g_mo[p_mo, q_mo, i, i]
                k_core[p, q] += g_mo[p_mo, i, i, q_mo]

    e_act = 0.0
    for p in range(n_act):
        p_mo = n_core + p
        for q in range(n_act):
            q_mo = n_core + q
            e_act += rdm1_active[p, q] * (
                h_mo[p_mo, q_mo] + 2.0 * j_core[p, q] - k_core[p, q]
            )
            for r in range(n_act):
                r_mo = n_core + r
                for s in range(n_act):
                    s_mo = n_core + s
                    e_act += (
                        0.5 * rdm2_active[p, q, r, s] * g_mo[p_mo, q_mo, r_mo, s_mo]
                    )

    return float(e_core + e_act)


def optimize_orbitals(
    mol,
    mo_coeff: np.ndarray,
    n_core: int,
    fragment_specs: Sequence[FragmentSpec],
    rdm1_active: np.ndarray,
    rdm2_active: np.ndarray,
    ao_eri: np.ndarray,
    h_ao: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Optimize molecular orbitals against the current active-space RDMs.

    Parameterizes an orbital rotation as the exponential of a skew-symmetric
    generator over a fixed set of allowed rotation pairs -- core-active,
    core-virtual, active-active across different fragments, and
    active-virtual -- and minimizes :func:`total_energy` over those rotation
    angles with L-BFGS-B, a quasi-Newton optimizer using finite-difference
    gradients (``n_rot + 1`` loss evaluations per iteration, which is why
    both ``ao_eri`` and ``h_ao`` must already be cached before this call).
    Core-core rotations are excluded because :func:`total_energy` has no
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
        ``(rotated_mo_coeff, energy)``: the rotated MO coefficients and the
        total energy at the optimum.

    Raises:
        ImportError: If the ``chem`` extra is not installed, propagated from
            :func:`total_energy`.
    """
    n_orb_total = mo_coeff.shape[1]
    n_act = sum(spec.n_orbitals for spec in fragment_specs)

    offsets = []
    running = 0
    for spec in fragment_specs:
        offsets.append(running)
        running += spec.n_orbitals

    rotation_pairs: list[tuple[int, int]] = []
    # Core <-> active.
    for p in range(n_core):
        for q in range(n_core, n_core + n_act):
            rotation_pairs.append((p, q))
    # Core <-> virtual.
    for p in range(n_core):
        for q in range(n_core + n_act, n_orb_total):
            rotation_pairs.append((p, q))
    # Active <-> active, across different fragments only.
    for idx_a, spec_a in enumerate(fragment_specs):
        for idx_b, spec_b in enumerate(fragment_specs):
            if idx_a < idx_b:
                for p_idx in range(spec_a.n_orbitals):
                    p = n_core + offsets[idx_a] + p_idx
                    for q_idx in range(spec_b.n_orbitals):
                        q = n_core + offsets[idx_b] + q_idx
                        rotation_pairs.append((p, q))
    # Active <-> virtual.
    for p in range(n_core, n_core + n_act):
        for q in range(n_core + n_act, n_orb_total):
            rotation_pairs.append((p, q))

    n_rot = len(rotation_pairs)

    def loss_fn(rotation_params: np.ndarray) -> float:
        generator = np.zeros((n_orb_total, n_orb_total))
        for idx, (p, q) in enumerate(rotation_pairs):
            generator[p, q] = rotation_params[idx]
            generator[q, p] = -rotation_params[idx]
        rotated = np.dot(mo_coeff, scipy.linalg.expm(generator))
        return total_energy(
            mol, rotated, n_core, rdm1_active, rdm2_active, ao_eri, h_ao
        )

    init_params = np.zeros(n_rot)
    baseline_energy = loss_fn(init_params)

    best_params = init_params
    best_energy = baseline_energy
    if n_rot > 0:
        res = minimize(loss_fn, init_params, method="L-BFGS-B", tol=1e-6)
        if np.isfinite(res.fun) and (
            not np.isfinite(best_energy) or res.fun < best_energy
        ):
            best_params = res.x
            best_energy = float(res.fun)

    generator = np.zeros((n_orb_total, n_orb_total))
    for idx, (p, q) in enumerate(rotation_pairs):
        generator[p, q] = best_params[idx]
        generator[q, p] = -best_params[idx]
    rotated_mo_coeff = np.dot(mo_coeff, scipy.linalg.expm(generator))

    return rotated_mo_coeff, float(best_energy)
