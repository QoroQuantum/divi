# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Configuration interaction confined to a sampled determinant subspace, and the
state properties built on it: wavefunctions, NEVPT2 corrections, state tracking
and transitions."""

import math
from collections.abc import Iterator, Sequence
from dataclasses import replace
from itertools import combinations

import numpy as np
import scipy.optimize
import scipy.sparse
from pyscf import ao2mo, lib, mcscf, mrpt
from pyscf.fci import selected_ci

from ._integrals import AOIntegrals, _exact_mean_field
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
    StateTransition,
    StateWavefunction,
    _SampledSubspace,
)


class _SubspaceCI(selected_ci.SelectedCI):
    """PySCF CI solver confined to a fixed alpha x beta determinant set,
    solving for ``nroots`` states of one spin multiplicity.

    Diagonalises the spin-penalised projected Hamiltonian as SQD does; the
    returned energies are the unpenalised ``<H>``. Roots nearer another spin
    than ``multiplicity`` can still appear when the set holds too few.
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
        return np.array(energies), civecs

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


def _subspace_casci(
    ao: AOIntegrals, subspace: _SampledSubspace, n_core: int
) -> mcscf.casci.CASCI:
    """A CASCI over ``subspace``'s active space, on the exact integrals."""
    strings_alpha, strings_beta = subspace.strings
    nelec = (strings_alpha[0].count("1"), strings_beta[0].count("1"))
    return mcscf.casci.CASCI(
        _exact_mean_field(ao), len(strings_alpha[0]), nelec, ncore=n_core
    )


def subspace_states_at(
    ao: AOIntegrals, subspace: _SampledSubspace, n_core: int
) -> tuple[ElectronicState, ...]:
    """Each requested state with the subspace re-diagonalised at fixed
    orbitals, in root order; see :func:`_subspace_roots`.

    ``subspace.mo_coeff``'s active block must be the basis its strings refer
    to. Energies are the unpenalised ``<H>``.
    """
    return tuple(state for state, _ in _subspace_roots(ao, subspace, n_core))


def _subspace_roots(
    ao: AOIntegrals, subspace: _SampledSubspace, n_core: int
) -> Iterator[tuple[ElectronicState, np.ndarray]]:
    """Each requested state of ``subspace`` at its orbitals, with its CI vector,
    in root order, labelled by ``subspace.labels`` when set.

    A root whose ``<S^2>`` is nearer another spin is dropped, not replaced, so a
    multiplicity can yield fewer roots than requested; indices count only the
    kept ones."""
    strings_alpha, strings_beta = subspace.strings
    casci = _subspace_casci(ao, subspace, n_core)
    n_active, nelec = casci.ncas, casci.nelecas
    for multiplicity, n_roots in sorted(subspace.states.items()):
        solver = _SubspaceCI(
            strings_alpha,
            strings_beta,
            multiplicity,
            min(n_roots, len(strings_alpha) * len(strings_beta)),
            subspace.lambda_penalty,
        )
        casci.fcisolver = solver
        energies, _, civecs = mcscf.casci.kernel(casci, subspace.mo_coeff, verbose=0)
        if solver.nroots == 1:
            energies, civecs = [energies], [civecs]
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
    ao: AOIntegrals, subspace: _SampledSubspace, n_core: int
) -> tuple[StateWavefunction, ...]:
    """Each requested state of ``subspace`` at its orbitals, with its CI vector
    and RDMs, in root order."""
    strings_alpha, strings_beta = (tuple(half) for half in subspace.strings)
    n_active = len(strings_alpha[0])
    core_orbitals = subspace.mo_coeff[:, :n_core]
    active_orbitals = subspace.mo_coeff[:, n_core : n_core + n_active]
    wavefunctions = []
    for state, amplitudes in _subspace_roots(ao, subspace, n_core):
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
    ao: AOIntegrals, subspace: _SampledSubspace, n_core: int
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
    max_memory = ao.mol.max_memory
    if n_bytes > (max_memory - lib.current_memory()[0]) * 1e6:
        raise MemoryError(
            f"NEVPT2 needs the full-space CI vector of CAS({sum(nelec)},{n_active}), "
            f"{n_bytes / 1e6:g} MB, above mol.max_memory={max_memory:g} MB. "
            "Raise mol.max_memory."
        )
    corrections = []
    for state, amplitudes in _subspace_roots(ao, subspace, n_core):
        casci = _subspace_casci(ao, subspace, n_core)
        casci.verbose = 0
        casci.mo_coeff = subspace.mo_coeff
        casci.ci = selected_ci.to_fci(
            sci_vector(strings_alpha, strings_beta, amplitudes), n_active, nelec
        )
        corrections.append((state, float(mrpt.NEVPT(casci).kernel())))
    return tuple(corrections)


def track_states(
    ao: AOIntegrals,
    previous: _SampledSubspace,
    current: _SampledSubspace,
    n_core: int,
) -> dict[tuple[int, int], tuple[int, int]]:
    """Each of ``current``'s roots' label, ``(multiplicity, root index)`` to
    ``(multiplicity, index)``: that of the previous state of its multiplicity
    whose CI vector it overlaps most, or else the lowest unused index."""
    labels = {}
    for multiplicity, before, now, overlaps in _root_overlaps(
        ao, previous, current, n_core
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
    ao: AOIntegrals,
    previous: _SampledSubspace,
    current: _SampledSubspace,
    n_core: int,
) -> Iterator[tuple[int, list[ElectronicState], list[ElectronicState], np.ndarray]]:
    """Per multiplicity, the previous and current (root-order) states and
    ``|<previous_j|current_i>|`` as rows ``i``, columns ``j``.

    Previous determinants are carried into the current orbitals by carryover's
    one-to-one orbital assignment, with orbital phases and reordering sign.
    """
    n_active = len(current.strings[0][0])
    active = slice(n_core, n_core + n_active)
    overlap = (
        current.mo_coeff[:, active].T
        @ ao.mol.intor_symmetric("int1e_ovlp")
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

    previous_roots = list(_subspace_roots(ao, previous, n_core))
    current_roots = list(_subspace_roots(ao, replace(current, labels=None), n_core))
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
        # Rows index the orbital the electron leaves, columns the one it enters.
        holes, singular_values, particles = np.linalg.svd(transition_rdm1)
        total = float(np.sum(singular_values**2))
        transitions.append(
            StateTransition(
                initial=lower.state,
                final=upper.state,
                transition_rdm1=transition_rdm1,
                transition_dipole=dipole,
                oscillator_strength=float(2.0 / 3.0 * gap * dipole @ dipole),
                hole_orbitals=orbitals @ holes,
                particle_orbitals=orbitals @ particles.T,
                nto_weights=(
                    singular_values**2 / total
                    if total > 0
                    else np.zeros_like(singular_values)
                ),
            )
        )
    return tuple(transitions)
