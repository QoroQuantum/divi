# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""AO and active-space integrals, effective per-fragment integrals, RDM assembly
and the LASSQD total-energy functional."""

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import accumulate, pairwise, permutations
from typing import Any

import numpy as np
from pyscf import ao2mo, mcscf
from pyscf.scf import hf

from divi.hamiltonians._molecular import without_chkfile

from ._state import FragmentSpec, FragmentState


@dataclass(frozen=True)
class MOIntegrals:
    """One round's active-space integrals.

    Arrays cover the active space alone, in the fragment-block order of the
    ``mo_coeff`` passed to :func:`transform_integrals`.

    Attributes:
        h_eff: ``(n_act, n_act)`` one-electron integrals with the frozen-core
            potential ``2 J_core - K_core`` folded in.
        g_act: ``(n_act,) * 4`` two-electron integrals in chemist order.
    """

    h_eff: np.ndarray
    g_act: np.ndarray


@dataclass(frozen=True, eq=False)
class AOIntegrals:
    """A molecule with its AO integrals.

    Attributes:
        mol: The PySCF ``gto.Mole``.
        eri: Its 8-fold symmetry-packed two-electron integrals.
        h_core: Its core Hamiltonian, plus any external potential.
    """

    mol: Any
    eri: np.ndarray
    h_core: np.ndarray

    @classmethod
    def of(cls, mol) -> "AOIntegrals":
        """``mol`` with its bare integrals."""
        return cls(mol, cached_ao_eri(mol), cached_h_ao(mol))


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
    ao: AOIntegrals,
    mo_coeff: np.ndarray,
    n_core: int,
    n_act: int,
) -> MOIntegrals:
    """Transform the active-space integrals, embedding the frozen core, with
    PySCF's CASCI on the exact integrals.

    Args:
        ao: The molecule and its AO integrals.
        mo_coeff: ``(nao, n_orb)`` MO coefficients, already permuted so that
            columns ``[0, n_core)`` are core, ``[n_core, n_core + n_act)``
            are the active fragment blocks in spec order, and the rest are
            virtual.
        n_core: Number of frozen-core orbitals.
        n_act: Number of active orbitals.
    """
    casci = mcscf.casci.CASCI(
        _exact_mean_field(ao), n_act, ao.mol.nelectron - 2 * n_core
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

    The frozen core contributes the spin-free potential ``2 J_core - K_core``,
    and every other fragment its mean-field Coulomb term from its total density
    and exchange term from its alpha or beta density, so the one-body potential
    differs between spin channels:

    ``h_sigma[p,q] = h[p,q] + 2 J_core - K_core
    + sum_B ( gamma_B[r,s] (pq|rs) - gamma_B^sigma[r,s] (pr|sq) )``

    The channels coincide only when every other fragment is closed-shell.

    Args:
        integrals: Active-space integrals for the current round.
        fragments: Every fragment's current state, in permutation order.
        index: Position of the target fragment within ``fragments``.

    Returns:
        ``(h_alpha, h_beta, g_frag)``: the fragment's effective one-body
        integrals per spin channel, and its bare two-body integrals.

    Raises:
        ValueError: If the fragments cover a different number of orbitals than
            ``integrals`` spans.
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
    # Separate arrays even with one fragment, so an in-place edit hits one channel.
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
    """The AO-basis electron-repulsion integral of ``mol``, 8-fold
    symmetry-packed ``int2e``."""
    return mol.intor("int2e", aosym="s8")


def cached_h_ao(mol) -> np.ndarray:
    """The AO-basis core Hamiltonian of ``mol``.

    Kinetic and nuclear-attraction integrals plus any effective core
    potential. A mean field with its own core Hamiltonian, such as a
    scalar-relativistic one, supplies it through ``get_hcore`` instead.
    """
    return hf.get_hcore(mol)


def _exact_mean_field(ao: AOIntegrals) -> hf.RHF:
    """An RHF object on ``ao``'s integrals, for PySCF CASCI and CASSCF to
    build their Hamiltonians from, with the non-symmetry-adapted classes,
    since LASSQD's orbitals can break point-group symmetry."""
    mean_field = without_chkfile(hf.RHF(ao.mol))
    mean_field._eri = ao.eri
    mean_field.get_hcore = lambda mol=None: ao.h_core
    return mean_field


def _total_energy(
    ao: AOIntegrals,
    mo_coeff: np.ndarray,
    n_core: int,
    rdm1_active: np.ndarray,
    rdm2_active: np.ndarray,
) -> float:
    """Total molecular energy for ``mo_coeff``, transforming in-core from the
    stored AO integrals.

    Args:
        ao: The molecule and its AO integrals.
        mo_coeff: ``(nao, n_orb)`` MO coefficients for this evaluation.
        n_core: Number of frozen-core orbitals.
        rdm1_active: ``(n_act, n_act)`` active-space 1-RDM.
        rdm2_active: ``(n_act,) * 4`` active-space 2-RDM.

    Raises:
        ImportError: If the ``chem`` extra is not installed.
    """
    energy, _ = energy_and_generalized_fock(
        ao, mo_coeff, n_core, rdm1_active, rdm2_active
    )
    return energy


def energy_and_generalized_fock(
    ao: AOIntegrals,
    mo_coeff: np.ndarray,
    n_core: int,
    rdm1_active: np.ndarray,
    rdm2_active: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Energy and generalised Fock matrix without a full four-index transform.

    Equals contracting the full-register densities (doubly occupied core, the
    active RDMs, empty virtuals) against the full MO integrals, from Coulomb and
    exchange builds plus one ``(all|act,act,act)`` transform of ``n_orb * n_act
    ** 3`` elements.

    Args:
        ao: The molecule and its AO integrals.
        mo_coeff: ``(nao, n_orb)`` MO coefficients, permuted into
            ``[core | fragment blocks | virtual]`` order.
        n_core: Number of frozen-core orbitals.
        rdm1_active: ``(n_act, n_act)`` active-space 1-RDM.
        rdm2_active: ``(n_act,) * 4`` active-space 2-RDM.

    Returns:
        ``(energy, fock)`` with ``fock`` shaped ``(n_orb, n_orb)``; its virtual
        columns are zero, since the densities do not reach them.
    """
    energy, fock, _, _ = _energy_fock_and_potentials(
        ao, mo_coeff, n_core, rdm1_active, rdm2_active
    )
    return energy, fock


def _energy_fock_and_potentials(
    ao: AOIntegrals,
    mo_coeff: np.ndarray,
    n_core: int,
    rdm1_active: np.ndarray,
    rdm2_active: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """:func:`energy_and_generalized_fock` plus the AO Coulomb and exchange
    potentials ``vj`` and ``vk`` of the unit-occupation core and the active
    density, each stacked ``[core, active]``."""
    n_orb = mo_coeff.shape[1]
    n_act = rdm1_active.shape[0]
    active = slice(n_core, n_core + n_act)
    core_coeff = mo_coeff[:, :n_core]
    active_coeff = mo_coeff[:, active]

    h_mo = mo_coeff.T @ ao.h_core @ mo_coeff

    # Occupation-one core density, so vj/vk come out as sum_i (mn|ii) and
    # sum_i (mi|in) directly. Both densities go in one call so the ERI is
    # traversed once rather than twice.
    dm_core = core_coeff @ core_coeff.T
    dm_act = active_coeff @ rdm1_active @ active_coeff.T
    vj, vk = hf.dot_eri_dm(ao.eri, np.stack([dm_core, dm_act]), hermi=1)

    j_core = mo_coeff.T @ vj[0] @ mo_coeff
    k_core = mo_coeff.T @ vk[0] @ mo_coeff
    j_act = mo_coeff.T @ vj[1] @ mo_coeff
    k_act = mo_coeff.T @ vk[1] @ mo_coeff

    # Both small index sets go first: ``half_e1`` transforms the leading pair, so
    # (act,act) costs n_act**2 pairs where (all,act) costs n_orb*n_act. Chemist
    # symmetry recovers the wanted ordering, (m a | b c) = g_aaaA[b, c, a, m].
    g_gaaa = (
        ao2mo.incore.general(
            ao.eri,
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
    energy = float(ao.mol.energy_nuc()) + e_core + e_act

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
