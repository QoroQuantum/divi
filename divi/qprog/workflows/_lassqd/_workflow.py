# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""The LASSQD program ensemble: construction and per-round program creation."""

import hashlib
import itertools
import os
import tempfile
import time
from collections.abc import Hashable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from math import isfinite
from pathlib import Path
from typing import Any, Literal
from warnings import warn

import numpy as np
import scipy.linalg
from pyscf import gto, scf

from divi.backends import CircuitRunner
from divi.hamiltonians._molecular import is_pyscf_input, split_pyscf_input
from divi.qprog.ensemble import (
    ProgramEnsemble,
    ReportingLevel,
    _check_circuit_runner,
)
from divi.qprog.problems import MolecularProblem

from ._active_space import (
    auto_fragment_specs,
    select_frontier_orbitals,
    split_active_orbitals,
)
from ._active_space import validate_fragment_atoms as _validate_fragment_atoms
from ._config import (
    CCSDPreparation,
    FragmentationConfig,
    OrbitalUpdate,
    Preparation,
    SecondOrderOrbitalUpdate,
    StateAveragedOrbitalUpdate,
    _checkpoint_digest,
    _OrbitalTarget,
)
from ._integrals import (
    MOIntegrals,
    _outside_divi_stacklevel,
    assemble_active_rdms,
    build_active_permutation,
    cached_ao_eri,
    cached_h_ao,
    escape_orbital_saddle,
    fragment_blocks,
    fragment_effective_integrals,
    subspace_states_at,
    transform_integrals,
)
from ._preparation import rotate_rdms_to_fragment_basis
from ._sqd import (
    SQDConfig,
    SQDResult,
    SQDSolver,
    compute_spatial_rdms,
    default_states,
    full_space_size,
    map_carried_strings,
    probs_to_sqd_bitstrings,
    spin_square_of,
    validate_sqd_for_fragment,
)
from ._state import (
    ElectronicState,
    FragmentSpec,
    FragmentState,
    LASSQDState,
    _OrbitalRound,
    _SampledSubspace,
    require_orthonormal,
    validate_fragment_specs,
)

#: ``<S^2>`` deviation from a state's target above which it is reported unresolved.
_SPIN_SQUARE_TOL = 1e-3

#: Energy gap (Ha) between same-multiplicity roots reported as near-degenerate.
_DEGENERACY_TOL = 1e-6

#: Gap (Ha) under which an unrequested root counts as degenerate with the last
#: requested one, above the ~3e-6 Ha the starting-orbital nudge splits a pair by.
_SPLIT_SET_TOL = 1e-4

#: Scale of the random rotation applied to the starting orbitals.
_ORBITAL_NUDGE = 1e-3

#: How many times ``energy_tol`` a saddle escape must lower the energy by.
_SADDLE_ESCAPE_DROP = 10.0

#: Options that callers pass to LASSQD by mistake, and where each belongs.
_MISPLACED_OPTIONS = {
    "max_rounds": "run()",
    "batch_config": "run()",
    "checkpoint_config": "run()",
    "shots": "the backend",
}


def _orbital_nudge(n_orbitals: int, seed: int | None) -> np.ndarray:
    """A small seeded rotation of the MO register.

    Point-group-symmetric orbitals have an exactly zero gradient along every
    rotation that mixes symmetries, so the macro-cycle cannot leave a symmetric
    saddle point of the orbital-CI energy without one.
    """
    generator = np.random.default_rng(seed).standard_normal((n_orbitals, n_orbitals))
    return scipy.linalg.expm(_ORBITAL_NUDGE * (generator - generator.T))


def _warn_no_round(name: str) -> None:
    warn(
        f"{name} is not available: no LASSQD round has completed yet. Call run() "
        "first.",
        UserWarning,
        stacklevel=3,
    )


def _is_off_spin(multiplicity: int, spin_square: float) -> bool:
    return abs(spin_square - spin_square_of(multiplicity)) > _SPIN_SQUARE_TOL


def _state_warnings(
    states: Sequence[ElectronicState],
) -> Iterator[tuple[Hashable, str]]:
    """``(key, message)`` for each state the subspace leaves off-spin and each
    near-degenerate pair."""
    for state in states:
        if _is_off_spin(state.multiplicity, state.spin_square):
            yield (
                ("off-spin", state.multiplicity, state.index),
                f"State ({state.multiplicity}, {state.index}) has <S^2> = "
                f"{state.spin_square:.4f}, not "
                f"{spin_square_of(state.multiplicity):.4f}: the subspace does not "
                "resolve it as that spin. Raise lambda_penalty or the sampling "
                "budget (batch_size, n_recovery_iterations).",
            )
    for lower, upper in itertools.pairwise(sorted(states)):
        gap = abs(upper.energy - lower.energy)
        if lower.multiplicity == upper.multiplicity and gap < _DEGENERACY_TOL:
            yield (
                ("degenerate", lower.multiplicity, lower.index),
                f"States ({lower.multiplicity}, {lower.index}) and "
                f"({upper.multiplicity}, {upper.index}) lie within {gap:.1e} Ha "
                "of each other, so their order can swap between rounds.",
            )


def _in_energy_order(
    states: Sequence[ElectronicState],
) -> tuple[ElectronicState, ...]:
    """Ascending in energy; states within ``_DEGENERACY_TOL`` of each other keep
    root order, so float noise cannot reorder them."""
    groups: list[list[ElectronicState]] = []
    for state in sorted(states, key=lambda state: state.energy):
        if groups and state.energy - groups[-1][0].energy < _DEGENERACY_TOL:
            groups[-1].append(state)
        else:
            groups.append([state])
    return tuple(state for group in groups for state in sorted(group))


@dataclass(frozen=True)
class LASSQDRoundReport:
    """One macro-cycle round's stage outputs.

    Appended to ``LASSQD.round_reports`` as the round completes, so an
    interrupted run keeps every finished round's numbers.

    Attributes:
        number: Round number, counting from 1.
        energy: This round's total energy of the lowest state, a variational
            upper bound.
        energy_change: Signed change from the previous round's ``energy``, or
            ``None`` for the first round, which has no predecessor.
        subspace_sizes: Distinct determinants each fragment's SQD recovery
            spanned, in fragment order. A one-determinant entry means that
            fragment captured no correlation.
        orbital_iterations: Iterations the orbital solve took.
        orbital_evaluations: Objective evaluations the orbital solve took, each
            one four-index MO transform.
        orbital_gradient_norm: L2 norm of the orbital gradient at the returned
            orbitals.
        orbital_converged: Whether that norm is at most ``sqrt(energy_tol)``,
            rather than the solve stopping on a budget or an energy-reduction
            floor.
        rotation_pairs: Orbital pairs the rotation spanned.
        recovery_seconds: Wall-clock time in SQD recovery for all fragments.
        orbital_seconds: Wall-clock time in the orbital re-optimisation.
        states: The states ``SQDConfig.states`` requested, ascending in
            energy, with states within 1e-6 Ha of each other in
            ``(multiplicity, index)`` order; empty when it requested none.
        averaged_energy: The weighted average over ``states`` that
            :class:`StateAveragedOrbitalUpdate` minimised and the macro-cycle
            converges on; ``None`` under any other orbital update.
        saddle_check: Outcome of the orbital-Hessian check run when a
            lowest-state round would end the macro-cycle: ``'escaped'`` (a
            saddle point was left and the macro-cycle continues), ``'passed'``
            or ``'inconclusive'`` (the eigensolver stopped unconverged, so the
            orbitals may still be a saddle point). ``None`` when it did not run.
    """

    number: int
    energy: float
    energy_change: float | None
    subspace_sizes: tuple[int, ...]
    orbital_iterations: int
    orbital_evaluations: int
    orbital_gradient_norm: float
    orbital_converged: bool
    rotation_pairs: int
    recovery_seconds: float
    orbital_seconds: float
    states: tuple[ElectronicState, ...] = ()
    averaged_energy: float | None = None
    saddle_check: Literal["escaped", "passed", "inconclusive"] | None = None

    def summary(self) -> str:
        """One-line human-readable digest of the round."""
        change = (
            "first round"
            if self.energy_change is None
            else f"change {self.energy_change:+.3e}"
        )
        parts = [
            f"Round {self.number} done. Energy: {self.energy:.8f} Ha ({change})",
            f"subspaces {list(self.subspace_sizes)}",
            self._orbital_summary(),
            f"SQD {self.recovery_seconds:.1f}s, orbitals {self.orbital_seconds:.1f}s",
        ]
        parts += [
            f"state ({state.multiplicity}, {state.index}) {state.energy:.8f} Ha, "
            f"<S^2> {state.spin_square:.3f}"
            for state in self.states
        ]
        if self.averaged_energy is not None:
            parts.append(f"state-averaged {self.averaged_energy:.8f} Ha")
        if self.saddle_check is not None:
            parts.append(f"saddle check {self.saddle_check}")
        return "; ".join(parts)

    def _orbital_summary(self) -> str:
        if self.rotation_pairs == 0:
            return "orbitals: none rotated"
        return (
            f"orbitals: {self.orbital_iterations} iterations over "
            f"{self.rotation_pairs} pairs, |g| {self.orbital_gradient_norm:.2e}, "
            f"converged={self.orbital_converged}"
        )


def _stored_array(
    stored: Any,
    name: str,
    shape: tuple[int, ...] | None = None,
    *,
    allow_infinite: bool = False,
) -> np.ndarray:
    """Copy one array out of a checkpoint archive, rejecting malformed data.

    Raises:
        ValueError: If the array is missing, an object array (which an archive
            opened with ``allow_pickle=False`` refuses to load), non-numeric,
            of the wrong shape, NaN, or infinite where ``allow_infinite`` is
            not set.
    """
    if name not in stored.files:
        raise ValueError(f"LASSQD checkpoint artifact is missing {name}.")
    array = np.asarray(stored[name])
    if not (np.issubdtype(array.dtype, np.number) or array.dtype == np.bool_):
        raise ValueError(
            f"LASSQD checkpoint {name} has non-numeric dtype {array.dtype}."
        )
    if shape is not None and array.shape != shape:
        raise ValueError(
            f"LASSQD checkpoint {name} has shape {array.shape}; expected {shape}."
        )
    invalid = np.isnan(array) if allow_infinite else ~np.isfinite(array)
    if np.any(invalid):
        raise ValueError(f"LASSQD checkpoint {name} contains non-finite values.")
    return array.copy()


def _carried_strings_from_metadata(
    metadata: Mapping[str, Any], name: str, spec: FragmentSpec, index: int
) -> tuple[str, ...]:
    """One sector's carried strings from checkpoint metadata, rejecting any that
    is not a 0/1 string as wide as the fragment.

    Raises:
        ValueError: If the entry is not a list of such strings.
    """
    strings = metadata.get(name)
    if not isinstance(strings, list) or any(
        not isinstance(string, str)
        or len(string) != spec.n_orbitals
        or set(string) - {"0", "1"}
        for string in strings
    ):
        raise ValueError(
            f"LASSQD checkpoint fragment {index} {name} must be a list of "
            f"{spec.n_orbitals}-character 0/1 strings."
        )
    return tuple(strings)


def _molecule_fingerprint(mol) -> str:
    """Digest of what fixes a molecule's integrals: atoms, geometry, basis,
    effective core potentials and electron count."""
    digest = hashlib.sha256()
    for array in (mol._atm, mol._bas, mol._env[gto.PTR_ENV_START :], mol._ecpbas):
        digest.update(np.ascontiguousarray(array).tobytes())
    digest.update(f"{mol.cart}:{mol.nelectron}".encode())
    return digest.hexdigest()


def _compute_n_core(specs: Sequence[FragmentSpec], n_occupied: int) -> int:
    """Frozen occupied-orbital count implied by a fragment spec list.

    ``FragmentSpec.orbitals`` always carries the molecule's original,
    pre-permutation orbital indices, so this can be recomputed from any
    fragment spec list together with the molecule's occupied-orbital count,
    independent of whether ``mo_coeff`` has already been permuted.
    """
    active_orbitals = [orbital for spec in specs for orbital in spec.orbitals]
    return n_occupied - sum(1 for orbital in active_orbitals if orbital < n_occupied)


def _diagonal_rdm_guess(
    spec: FragmentSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """RDMs of a fresh fragment's reference determinant.

    The determinant places ``n_alpha`` alpha and ``n_beta`` beta electrons on
    the fragment's lowest-indexed orbitals. Its 2-RDM carries the Coulomb term
    ``Gamma[p, p, q, q] = n_p n_q`` and the same-spin exchange term
    ``Gamma[p, q, q, p] -= sum_sigma n^sigma_p n^sigma_q``.

    Returns ``(rdm1, rdm2, rdm1_alpha, rdm1_beta)``.
    """
    n_orb = spec.n_orbitals
    occ_alpha = np.zeros(n_orb)
    occ_alpha[: spec.n_alpha] = 1.0
    occ_beta = np.zeros(n_orb)
    occ_beta[: spec.n_beta] = 1.0
    occupation = occ_alpha + occ_beta

    rdm2 = np.zeros((n_orb, n_orb, n_orb, n_orb))
    p, q = np.meshgrid(np.arange(n_orb), np.arange(n_orb), indexing="ij")
    rdm2[p, p, q, q] = occupation[p] * occupation[q]
    rdm2[p, q, q, p] -= np.outer(occ_alpha, occ_alpha) + np.outer(occ_beta, occ_beta)
    return np.diag(occupation), rdm2, np.diag(occ_alpha), np.diag(occ_beta)


class LASSQD(ProgramEnsemble):
    """Localised active-space sample-based quantum diagonalisation.

    Partitions a molecule's active space into fragments, prepares one circuit
    per fragment, and recovers each fragment state via sample-based quantum
    diagonalisation. Fragment preparation is chosen by ``preparation``:
    :class:`~divi.qprog.workflows.CCSDPreparation` (the default) samples a
    one-repetition LUCJ operator seeded from CCSD, whose classical cost is the
    fragment's CCSD, so it never builds the fragment statevector;
    :class:`~divi.qprog.workflows.LinearMethodPreparation` first optimises it
    with ffsim's exact linear method; :class:`~divi.qprog.workflows.VQEPreparation`
    optimises an ansatz against backend-estimated energies.

    :attr:`energy` is a variational upper bound -- the assembled RDM is that of
    a product of fragment states, so the energy is a genuine expectation value.
    Fragmenting nonetheless costs accuracy, and the cost grows with how
    strongly the fragments interact; see
    :ref:`lassqd-accuracy-characteristics` in the LASSQD guide.

    Args:
        problem: An :class:`~divi.qprog.problems.MolecularProblem`
            built by :meth:`~divi.qprog.problems.MolecularProblem.\
from_molecule` from a PySCF ``gto.Mole`` (an RHF calculation is run on it
            lazily, in :meth:`initial_state`) or a restricted mean-field
            object — not from a PennyLane ``qchem.Molecule`` or from bare
            integrals, since the orbital optimisation needs the atomic-orbital
            basis. Closed-shell (RHF) only. A restricted DFT object supplies
            starting orbitals only; its functional enters no energy.
        fragmentation: Which orbitals are active and how they split into
            fragments, as a
            :class:`~divi.qprog.workflows.FragmentationConfig`.
        sqd: Sampling and diagonalisation budget per fragment solve, as an
            :class:`~divi.qprog.workflows.SQDConfig`. Defaults to
            ``SQDConfig()``.
        preparation: Fragment-circuit preparation strategy.
        orbital_update: How each macro-cycle updates the orbitals.
        energy_tol: Macro-cycle stops once consecutive rounds' total energies
            differ by less than this (Hartree) and the round's orbital solve
            converged, meaning its orbital-gradient L2 norm is at most
            ``sqrt(energy_tol)``. The gradient is ``2 (F_pq - F_qp)`` over the
            generalised Fock matrix, twice PySCF CASSCF's, so this is twice
            as tight as CASSCF's ``sqrt``-of-energy-tolerance rule.
        seed: Seed for fragmentation, localisation, and SQD subsampling.
        **kwargs: ``backend`` (required), ``sampling_backend``, and
            ``reporting_level`` are consumed here; ``sampling_backend`` runs
            each fragment's final sample. Other keywords are
            forwarded to each fragment program. ``precision``,
            ``qem_protocol`` and ``suppress_performance_warnings`` work with
            any preparation; VQE options such as ``n_layers``,
            ``ansatz_kwargs``, ``grouping_strategy`` and ``early_stopping``
            require :class:`~divi.qprog.workflows.VQEPreparation`.

    Raises:
        ValueError: If ``fragmentation``'s ``active_orbitals`` has out-of-range
            indices; if its ``n_active_orbitals`` or ``active_orbitals``
            selects no occupied or no virtual orbital of this molecule; if its
            ``fragment_atoms`` names an out-of-range atom or shares one between
            fragments; if ``energy_tol`` is not positive and finite; or if any fragment leaves no
            excitation available, fragments overlap, or the fragments do not
            sum to ``Sz = 0``; if ``sqd.states`` is set with more than one
            fragment or asks for a state the fragment cannot hold; or if
            ``sqd.extension`` could outgrow the solver with ``max_dim`` unset.
            The configuration objects validate their own fields on
            construction.
        TypeError: If ``backend`` is missing or not a
            :class:`~divi.backends.CircuitRunner`, if ``sampling_backend`` is
            neither one nor ``None``, if ``problem`` was not built
            from a PySCF ``Mole`` or mean-field, if ``preparation`` or
            ``orbital_update`` is not one of its strategies, or if a preparation other than
            :class:`~divi.qprog.workflows.VQEPreparation` receives a keyword
            its fragment programs do not take.
        ImportError: If the ``chem`` extra is not installed.
    """

    def __init__(
        self,
        problem: MolecularProblem,
        *,
        fragmentation: FragmentationConfig,
        sqd: SQDConfig | None = None,
        preparation: Preparation = CCSDPreparation(),
        orbital_update: OrbitalUpdate = SecondOrderOrbitalUpdate(),
        energy_tol: float = 1e-6,
        seed: int | None = None,
        **kwargs,
    ):
        if not isinstance(preparation, Preparation):
            raise TypeError(
                "preparation must be CCSDPreparation, LinearMethodPreparation or "
                f"VQEPreparation; got {type(preparation).__name__}."
            )
        if not isinstance(orbital_update, OrbitalUpdate):
            raise TypeError(
                "orbital_update must be SecondOrderOrbitalUpdate, "
                "StateAveragedOrbitalUpdate, LBFGSOrbitalUpdate or FixedOrbitals; got "
                f"{type(orbital_update).__name__}."
            )
        if not isinstance(fragmentation, FragmentationConfig):
            raise TypeError(
                "fragmentation must be a FragmentationConfig; got "
                f"{type(fragmentation).__name__}."
            )
        if sqd is not None and not isinstance(sqd, SQDConfig):
            raise TypeError(f"sqd must be an SQDConfig; got {type(sqd).__name__}.")
        if not (isfinite(energy_tol) and energy_tol > 0):
            raise ValueError(
                f"energy_tol must be a positive finite number; got {energy_tol}."
            )
        if "backend" not in kwargs:
            raise TypeError(
                "LASSQD.__init__ missing required keyword-only argument: 'backend'."
            )

        backend = kwargs.pop("backend")
        sampling_backend = kwargs.pop("sampling_backend", None)
        # LUCJ preparations pass the base class None in its place.
        _check_circuit_runner("sampling_backend", sampling_backend, optional=True)
        reporting_level = kwargs.pop("reporting_level", ReportingLevel.COMPACT)
        for name, owner in _MISPLACED_OPTIONS.items():
            if name in kwargs:
                raise TypeError(f"{name} belongs to {owner}, not to LASSQD.")
        preparation._check_options(kwargs)
        super().__init__(
            backend=backend,
            sampling_backend=preparation._ensemble_sampling_backend(sampling_backend),
            reporting_level=reporting_level,
        )
        self._fragment_sampling_backend = sampling_backend

        if not isinstance(problem, MolecularProblem) or problem.molecule is None:
            raise TypeError(
                "LASSQD expects a MolecularProblem built with "
                "from_molecule; bare integrals carry no atomic-orbital basis. "
                f"Got {type(problem).__name__}."
            )
        molecule = problem.molecule
        if not is_pyscf_input(molecule):
            raise TypeError(
                "LASSQD needs a problem built from a pyscf Mole or restricted "
                f"mean-field object, got {type(molecule).__name__}."
            )
        self._mol, self._mean_field = split_pyscf_input(molecule)
        if (
            self._mean_field is not None
            and self._mean_field.mo_coeff is not None
            and not self._mean_field.converged
        ):
            warn(
                "The mean-field object passed in did not converge, so the "
                "orbitals the macro-cycle starts from are not its stationary "
                "point. Converge it first unless those orbitals are intended.",
                UserWarning,
                stacklevel=2,
            )

        # Validate the caller's orbital choices against this molecule here
        # rather than in ``initial_state``, so an out-of-range index fails at
        # construction.
        n_orbitals_total = self._register_size()
        n_occupied = self._mol.nelectron // 2
        if fragmentation.active_spaces is not None:
            validate_fragment_specs(
                fragmentation.active_spaces, n_orbitals_total, n_occupied
            )
        if fragmentation.n_active_orbitals is not None:
            select_frontier_orbitals(
                n_orbitals_total, n_occupied, fragmentation.n_active_orbitals
            )
        if fragmentation.active_orbitals is not None:
            split_active_orbitals(
                fragmentation.active_orbitals, n_occupied, n_orbitals_total
            )
        if fragmentation.fragment_atoms is not None:
            _validate_fragment_atoms(fragmentation.fragment_atoms, self._mol.natm)

        self._fragmentation = fragmentation
        self._sqd = SQDConfig() if sqd is None else sqd
        if isinstance(orbital_update, StateAveragedOrbitalUpdate):
            if self._sqd.states is None:
                raise ValueError(
                    "StateAveragedOrbitalUpdate averages over SQDConfig.states, "
                    "which is unset. Request the states to average, e.g. "
                    "SQDConfig(states={1: 2})."
                )
            orbital_update._weights_for(self._sqd.states)
        self._orbital_update = orbital_update
        if fragmentation.active_spaces is not None:
            self._check_fragments(fragmentation.active_spaces)
        self._preparation = preparation
        self._energy_tol = energy_tol
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._extra_kwargs = kwargs

        self._state: LASSQDState | None = None
        self._solvers: dict[int, SQDSolver] = {}
        self._energy_history: list[float] = []
        self._round_reports: list[LASSQDRoundReport] = []
        self._warned: set[Hashable] = set()
        self._ao_eri: np.ndarray | None = None
        self._h_ao: np.ndarray | None = None

    def _check_fragments(self, specs: Sequence[FragmentSpec]) -> None:
        """Reject ``SQDConfig`` settings the fragments cannot carry, and warn
        when excited states get orbitals optimised for the lowest state only.

        Warns:
            UserWarning: If more than one state is requested with
                :class:`SecondOrderOrbitalUpdate` or :class:`LBFGSOrbitalUpdate`.
        """
        for spec in specs:
            validate_sqd_for_fragment(
                self._sqd, spec.n_orbitals, spec.n_alpha, spec.n_beta
            )
        states = self._sqd.states
        if states is None:
            return
        if len(specs) != 1:
            raise ValueError(
                f"SQDConfig.states needs a single fragment; got {len(specs)}. A "
                "product of fragment states cannot represent an excitation "
                "spread over several fragments. Put the whole active space in "
                "one FragmentSpec."
            )
        if (
            sum(states.values()) > 1
            and self._orbital_update._relaxes is _OrbitalTarget.LOWEST_STATE
        ):
            warn(
                f"{type(self._orbital_update).__name__} optimises the orbitals "
                "for the lowest state only, so the others miss their orbital "
                "relaxation and gaps to them come out too large. Use "
                "StateAveragedOrbitalUpdate, or FixedOrbitals to compare with "
                "CASCI on the same orbitals.",
                UserWarning,
                stacklevel=_outside_divi_stacklevel(),
            )

    def _register_size(self) -> int:
        """Molecular-orbital count: the supplied mean field's, which drops
        linearly dependent basis functions, else the basis size."""
        mean_field = self._mean_field
        if mean_field is not None and mean_field.mo_coeff is not None:
            return np.asarray(mean_field.mo_coeff).shape[1]
        return self._mol.nao_nr()

    @property
    def sampling_backend(self) -> CircuitRunner | None:
        """Backend the fragments' final samples run on, when configured."""
        return self._fragment_sampling_backend

    @property
    def preparation(self) -> Preparation:
        """Fragment-circuit preparation strategy."""
        return self._preparation

    @property
    def orbital_update(self) -> OrbitalUpdate:
        """Per-macro-cycle orbital update strategy."""
        return self._orbital_update

    def initial_state(self) -> LASSQDState:
        """Resolve fragments and build the initial workflow state.

        Runs RHF on the molecule if no mean-field has run yet, resolves the
        fragments (validating explicit ``active_spaces``, or fragmenting
        automatically), rotates the orbitals by a small seeded amount unless
        the orbital update keeps them fixed (symmetric orbitals would otherwise
        hold the macro-cycle at a saddle point), permutes the MO register into
        ``[core | fragments | virtual]`` order, and seeds each fragment with
        its reference determinant's RDMs.

        While a program map built by :meth:`create_programs` waits to run, this
        returns the state those programs were built from instead, so the round
        that runs them is reduced against the same orbitals.

        Returns:
            A fresh :class:`~divi.qprog.workflows.LASSQDState` with ``energy``
            and ``previous_energy`` at their default (``inf``) values and
            every fragment's ``params`` set to ``None``.

        Raises:
            RuntimeError: If a mean field computed here does not converge.

        Warns:
            UserWarning: If the mean-field reference is not aufbau, i.e. its
                LUMO lies below its HOMO. Frontier selection assumes ascending
                orbital energies, so the active space would be wrong.
        """
        if self._programs_pending and self._state is not None:
            return self._state

        mean_field = self._mean_field
        if mean_field is None or mean_field.mo_coeff is None:
            mean_field = scf.RHF(self._mol) if mean_field is None else mean_field
            mean_field.run(verbose=0)
            if not mean_field.converged:
                raise RuntimeError(
                    "The mean-field reference did not converge, so every orbital "
                    "the macro-cycle starts from is meaningless. Converge it "
                    "yourself and pass the mean-field object instead of the "
                    "molecule."
                )
            self._mean_field = mean_field

        mo_coeff = np.asarray(mean_field.mo_coeff)
        mo_energy = np.asarray(mean_field.mo_energy)
        n_orbitals_total = mo_coeff.shape[1]
        n_occupied = self._mol.nelectron // 2
        # Only the frontier path reads orbital energies. Checking this on the
        # other paths would test a stale array anyway: a caller who reorders
        # ``mo_coeff`` (as AVAS does) leaves ``mo_energy`` describing the old
        # register.
        if (
            self._fragmentation.n_active_orbitals is not None
            and mo_energy[n_occupied] < mo_energy[n_occupied - 1]
        ):
            warn(
                f"The mean-field reference is not aufbau: the LUMO sits "
                f"{(mo_energy[n_occupied - 1] - mo_energy[n_occupied]) * 1000:.1f} "
                "mHa below the HOMO, so the occupied set is not the lowest "
                "orbitals. Frontier selection assumes ascending orbital "
                "energies and will pick the wrong active space.",
                UserWarning,
                stacklevel=2,
            )

        config = self._fragmentation
        if config.active_spaces is not None:
            specs = list(config.active_spaces)
        else:
            specs, localized, active_positions = auto_fragment_specs(
                self._mol,
                mo_coeff,
                n_occupied,
                self._rng,
                n_active_orbitals=config.n_active_orbitals,
                max_orbitals_per_fragment=config.max_orbitals_per_fragment,
                coupling_threshold=config.coupling_threshold,
                active_orbitals=config.active_orbitals,
                fragment_atoms=config.fragment_atoms,
                local_spins=config.local_spins,
                ao_eri=self._cached_mol_integrals()[0],
                h_ao=self._core_hamiltonian(),
            )
            mo_coeff = mo_coeff.copy()
            mo_coeff[:, active_positions] = localized

        validate_fragment_specs(specs, n_orbitals_total, n_occupied)
        if config.active_spaces is None:
            # Explicit layouts were checked, and warned about, at construction.
            self._check_fragments(specs)

        n_core = _compute_n_core(specs, n_occupied)
        if self._orbital_update._relaxes is not _OrbitalTarget.NOTHING:
            mo_coeff = mo_coeff @ _orbital_nudge(n_orbitals_total, self._seed)

        permutation = build_active_permutation(specs, n_core, n_orbitals_total)
        mo_coeff = mo_coeff[:, permutation]

        fragments = []
        for spec in specs:
            rdm1, rdm2, rdm1_alpha, rdm1_beta = _diagonal_rdm_guess(spec)
            fragments.append(
                FragmentState(
                    spec=spec,
                    rdm1=rdm1,
                    rdm2=rdm2,
                    rdm1_alpha=rdm1_alpha,
                    rdm1_beta=rdm1_beta,
                )
            )

        return LASSQDState(mo_coeff=mo_coeff, fragments=tuple(fragments))

    def create_programs(self, state: LASSQDState | None = None):
        """Create one preparation-and-sampling program per fragment in ``state``.

        Args:
            state: Workflow state to build programs from. Defaults to a
                fresh :meth:`initial_state`.

        Raises:
            RuntimeError: If an executor is already running, or if programs
                have already been created (from ``super().create_programs()``).
        """
        super().create_programs()

        if state is None:
            state = self.initial_state()
        self._state = state

        integrals, _n_core = self._active_space_integrals(state)
        fragment_seeds = self._rng.integers(0, 2**63 - 1, size=len(state.fragments))

        for index, fragment in enumerate(state.fragments):
            h_alpha, h_beta, g_frag = fragment_effective_integrals(
                integrals, state.fragments, index
            )
            fragment_problem = MolecularProblem(
                h_alpha,
                g_frag,
                n_alpha=fragment.spec.n_alpha,
                n_beta=fragment.spec.n_beta,
                one_body_beta=h_beta,
            )
            prog_id = f"fragment_{index}"
            self._programs[prog_id] = self._preparation._build_program(
                fragment_problem,
                fragment,
                backend=self.backend,
                sampling_backend=self.sampling_backend,
                seed=int(fragment_seeds[index]),
                options=self._extra_kwargs,
            )

    def aggregate_results(self) -> LASSQDState:
        """Return the workflow's current state.

        Returns the same object exposed by :attr:`~divi.qprog.ensemble.\
ProgramEnsemble.workflow_state`: the state :meth:`update_state` produced
        from the round that just ran, not the state that was used to build
        that round's programs.

        Returns:
            The latest :class:`~divi.qprog.workflows.LASSQDState`.

        Raises:
            RuntimeError: If no round has been reduced into
                :attr:`~divi.qprog.ensemble.ProgramEnsemble.workflow_state` yet.
                A round driven by hand is reduced by :meth:`update_state`, whose
                return value is that round's state. A run that fails after a
                completed round keeps that round's state.
        """
        if self.workflow_state is None or not self._energy_history:
            raise RuntimeError(
                "No LASSQD round has been reduced into workflow_state yet. Use "
                "run(), or the state update_state returns when driving rounds "
                "by hand."
            )
        return self.workflow_state

    def _reset_workflow_state(self) -> None:
        """Clear per-workflow state, also re-seeding ``_rng`` and dropping
        every fragment's cached ``SQDSolver``.

        ``run()`` calls this at the start of every invocation. Without
        re-deriving ``_rng`` from the stored seed and clearing ``_solvers``
        here, a second ``run()`` on the same instance would resume fragment
        0's SQD stream mid-sequence, draw different fragment seeds, and (in
        automatic mode) re-draw the localisation restarts from an advanced
        generator instead of reproducing the first run.
        """
        super()._reset_workflow_state()
        self._rng = np.random.default_rng(self._seed)
        self._solvers.clear()
        self._energy_history.clear()
        self._round_reports.clear()
        self._warned.clear()

    def _save_workflow_checkpoint_state(
        self, state: Any, round_dir: Path, stem: str
    ) -> dict[str, Any]:
        """Write an explicit, non-pickled snapshot of mutable LASSQD state."""
        if not isinstance(state, LASSQDState):
            raise TypeError(
                f"LASSQD checkpoint state must be LASSQDState, got "
                f"{type(state).__name__}."
            )

        arrays: dict[str, np.ndarray] = {
            "mo_coeff": state.mo_coeff,
            "energy": np.asarray(state.energy),
            "previous_energy": np.asarray(state.previous_energy),
            "orbitals_converged": np.asarray(state.orbitals_converged),
            "energy_history": np.asarray(self._energy_history, dtype=np.float64),
        }
        fragments = []
        for index, fragment in enumerate(state.fragments):
            prefix = f"fragment_{index}"
            arrays[f"{prefix}_rdm1"] = fragment.rdm1
            arrays[f"{prefix}_rdm2"] = fragment.rdm2
            params_present = fragment.params is not None
            alpha_present = fragment.rdm1_alpha is not None
            beta_present = fragment.rdm1_beta is not None
            if fragment.params is not None:
                arrays[f"{prefix}_params"] = fragment.params
            if fragment.rdm1_alpha is not None:
                arrays[f"{prefix}_rdm1_alpha"] = fragment.rdm1_alpha
            if fragment.rdm1_beta is not None:
                arrays[f"{prefix}_rdm1_beta"] = fragment.rdm1_beta
            if fragment.sampled_orbitals is not None:
                arrays[f"{prefix}_sampled_orbitals"] = fragment.sampled_orbitals
            fragments.append(
                {
                    "orbitals": list(fragment.spec.orbitals),
                    "n_alpha": fragment.spec.n_alpha,
                    "n_beta": fragment.spec.n_beta,
                    "params": params_present,
                    "rdm1_alpha": alpha_present,
                    "rdm1_beta": beta_present,
                    "sampled_orbitals": fragment.sampled_orbitals is not None,
                    "carried_alpha": list(fragment.carried_alpha),
                    "carried_beta": list(fragment.carried_beta),
                }
            )

        solvers = [
            {"index": index, "rng_state": solver._rng.bit_generator.state}
            for index, solver in sorted(self._solvers.items())
        ]

        artifact = f"{stem}.npz"
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=round_dir, suffix=".npz", delete=False
            ) as handle:
                temporary_path = Path(handle.name)
                np.savez(handle, **arrays)  # pyrefly: ignore[bad-argument-type]
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, round_dir / artifact)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()

        return {
            "artifact": artifact,
            "molecule": _molecule_fingerprint(self._mol),
            "configuration": self._configuration_record(),
            "fragments": fragments,
            "rng_state": self._rng.bit_generator.state,
            "solvers": solvers,
            "round_reports": [asdict(report) for report in self._round_reports],
        }

    def _load_workflow_checkpoint_state(
        self, payload: dict[str, Any], round_dir: Path, stem: str
    ) -> LASSQDState:
        """Load and validate an explicit LASSQD NPZ snapshot."""
        artifact = payload.get("artifact")
        if not isinstance(artifact, str):
            raise ValueError("LASSQD checkpoint is missing its NPZ artifact.")
        if artifact != f"{stem}.npz":
            raise ValueError("LASSQD checkpoint references the wrong state artifact.")
        molecule = payload.get("molecule")
        if not isinstance(molecule, str):
            raise ValueError("LASSQD checkpoint is missing its molecule fingerprint.")
        if molecule != _molecule_fingerprint(self._mol):
            raise ValueError(
                "LASSQD checkpoint was written for a different molecule: its "
                "geometry, basis or electron count differs from this one's."
            )
        configuration = payload.get("configuration")
        if not isinstance(configuration, str):
            raise ValueError("LASSQD checkpoint is missing its configuration.")
        if configuration != self._configuration_record():
            raise ValueError(
                "LASSQD checkpoint was written with a different LASSQD "
                "configuration: fragmentation, sqd, preparation or "
                "orbital_update differ."
            )
        fragment_metadata = payload.get("fragments")
        if not isinstance(fragment_metadata, list) or not fragment_metadata:
            raise ValueError("LASSQD checkpoint has no fragment metadata.")
        rng_state = payload.get("rng_state")
        if not isinstance(rng_state, dict):
            raise ValueError("LASSQD checkpoint is missing RNG state.")
        solver_metadata = payload.get("solvers")
        if not isinstance(solver_metadata, list):
            raise ValueError("LASSQD checkpoint solver metadata must be a list.")
        report_payloads = payload.get("round_reports")
        if not isinstance(report_payloads, list):
            raise ValueError("LASSQD checkpoint round reports must be a list.")

        n_orbitals_total = self._register_size()
        with np.load(round_dir / artifact, allow_pickle=False) as stored:
            fragments = []
            for index, metadata in enumerate(fragment_metadata):
                prefix = f"fragment_{index}"
                spec = FragmentSpec(
                    tuple(metadata["orbitals"]),
                    metadata["n_alpha"],
                    metadata["n_beta"],
                )
                rdm1_shape = (spec.n_orbitals,) * 2
                params = (
                    _stored_array(stored, f"{prefix}_params")
                    if metadata.get("params")
                    else None
                )
                if params is not None and params.ndim != 1:
                    raise ValueError(
                        f"LASSQD checkpoint fragment {index} parameters are not 1-D."
                    )
                rdm1_alpha, rdm1_beta = (
                    (
                        _stored_array(stored, f"{prefix}_{name}", rdm1_shape)
                        if metadata.get(name)
                        else None
                    )
                    for name in ("rdm1_alpha", "rdm1_beta")
                )
                sampled_orbitals = (
                    _stored_array(
                        stored,
                        f"{prefix}_sampled_orbitals",
                        (self._mol.nao_nr(), spec.n_orbitals),
                    )
                    if metadata.get("sampled_orbitals")
                    else None
                )
                carried_alpha, carried_beta = (
                    _carried_strings_from_metadata(metadata, name, spec, index)
                    for name in ("carried_alpha", "carried_beta")
                )
                fragments.append(
                    FragmentState(
                        spec=spec,
                        rdm1=_stored_array(stored, f"{prefix}_rdm1", rdm1_shape),
                        rdm2=_stored_array(
                            stored, f"{prefix}_rdm2", (spec.n_orbitals,) * 4
                        ),
                        params=params,
                        rdm1_alpha=rdm1_alpha,
                        rdm1_beta=rdm1_beta,
                        carried_alpha=carried_alpha,
                        carried_beta=carried_beta,
                        sampled_orbitals=sampled_orbitals,
                    )
                )
            mo_coeff = _stored_array(
                stored, "mo_coeff", (self._mol.nao_nr(), n_orbitals_total)
            )
            energy = _stored_array(stored, "energy", (), allow_infinite=True)
            previous_energy = _stored_array(
                stored, "previous_energy", (), allow_infinite=True
            )
            orbitals_converged = _stored_array(stored, "orbitals_converged", ())
            energy_history = _stored_array(stored, "energy_history")
            if energy_history.ndim != 1:
                raise ValueError("LASSQD checkpoint energy history is not 1-D.")

        require_orthonormal(
            mo_coeff,
            "LASSQD checkpoint mo_coeff",
            self._mol.intor_symmetric("int1e_ovlp"),
        )
        specs = [fragment.spec for fragment in fragments]
        validate_fragment_specs(specs, n_orbitals_total, self._mol.nelectron // 2)
        state = LASSQDState(
            mo_coeff=mo_coeff,
            fragments=tuple(fragments),
            energy=float(energy),
            previous_energy=float(previous_energy),
            orbitals_converged=bool(orbitals_converged),
        )

        reports = []
        for report in report_payloads:
            reports.append(
                LASSQDRoundReport(
                    number=int(report["number"]),
                    energy=float(report["energy"]),
                    energy_change=(
                        None
                        if report["energy_change"] is None
                        else float(report["energy_change"])
                    ),
                    subspace_sizes=tuple(
                        int(size) for size in report["subspace_sizes"]
                    ),
                    orbital_iterations=int(report["orbital_iterations"]),
                    orbital_evaluations=int(report["orbital_evaluations"]),
                    orbital_gradient_norm=float(report["orbital_gradient_norm"]),
                    orbital_converged=bool(report["orbital_converged"]),
                    rotation_pairs=int(report["rotation_pairs"]),
                    recovery_seconds=float(report["recovery_seconds"]),
                    orbital_seconds=float(report["orbital_seconds"]),
                    states=tuple(
                        ElectronicState(
                            multiplicity=int(entry["multiplicity"]),
                            index=int(entry["index"]),
                            energy=float(entry["energy"]),
                            spin_square=float(entry["spin_square"]),
                        )
                        for entry in report["states"]
                    ),
                    averaged_energy=(
                        None
                        if report["averaged_energy"] is None
                        else float(report["averaged_energy"])
                    ),
                    saddle_check=report["saddle_check"],
                )
            )

        solver_indices = [metadata.get("index") for metadata in solver_metadata]
        if (
            any(not isinstance(index, int) for index in solver_indices)
            or len(set(solver_indices)) != len(solver_indices)
            or any(not 0 <= index < len(state.fragments) for index in solver_indices)
        ):
            raise ValueError("LASSQD checkpoint has invalid solver fragment indices.")
        validation_rng = np.random.default_rng()
        validation_rng.bit_generator.state = rng_state
        for metadata in solver_metadata:
            solver_rng_state = metadata.get("rng_state")
            if not isinstance(solver_rng_state, dict):
                raise ValueError("LASSQD checkpoint is missing solver RNG state.")
            validation_rng.bit_generator.state = solver_rng_state

        self._rng.bit_generator.state = rng_state
        self._solvers.clear()
        for metadata in solver_metadata:
            index = metadata["index"]
            solver = self._solver_for(index, state.fragments[index].spec)
            solver._rng.bit_generator.state = metadata["rng_state"]
        self._rng.bit_generator.state = rng_state
        self._energy_history = energy_history.astype(float).tolist()
        self._round_reports = reports
        # Rebuilt programs keep the state they were built from.
        if not self._programs:
            self._state = state
        return state

    def _configuration_record(self) -> str:
        """The settings a checkpoint must have been written with to resume
        here: those that decide which computation a round performs."""
        return repr(
            (
                self._fragmentation,
                self._sqd,
                self._preparation._checkpoint_record(),
                _checkpoint_digest(dict(sorted(self._extra_kwargs.items()))),
                self._orbital_update,
            )
        )

    def _solver_for(self, index: int, spec: FragmentSpec) -> SQDSolver:
        """Return this fragment's cached ``SQDSolver``, building it once.

        Each fragment gets its own child generator spawned from the
        workflow's seeded RNG, so distinct fragments never share a draw
        sequence and repeated runs under the same ``seed`` stay reproducible.
        Caching avoids rebuilding the solver every round; across rounds it
        carries only its generator's position. Each ``solve`` call recovers its
        occupancies from scratch. Carryover between rounds goes through
        :class:`FragmentState`, since a retained determinant must first be
        mapped into the round's new orbital basis.
        """
        solver = self._solvers.get(index)
        if solver is None:
            solver = SQDSolver(
                spec.n_orbitals,
                spec.n_alpha,
                spec.n_beta,
                self._sqd,
                rng=self._rng.spawn(1)[0],
            )
            self._solvers[index] = solver
        return solver

    def _cached_mol_integrals(self) -> tuple[np.ndarray, np.ndarray]:
        """Return this run's AO-basis integrals, computing them once.

        Both are independent of ``mo_coeff`` and reused unchanged by every
        round's :func:`optimize_orbitals` call, which itself evaluates
        :func:`_total_energy` many times per round.
        """
        if self._ao_eri is None:
            self._ao_eri = cached_ao_eri(self._mol)
        return self._ao_eri, self._core_hamiltonian()

    def _core_hamiltonian(self) -> np.ndarray:
        """This run's AO core Hamiltonian, computed once: the mean field's when
        one is held, so a relativistic or otherwise modified one carries
        through, else built from ``mol``."""
        if self._h_ao is None:
            self._h_ao = (
                cached_h_ao(self._mol)
                if self._mean_field is None
                else np.asarray(self._mean_field.get_hcore())
            )
        return self._h_ao

    def _n_core(self, state: LASSQDState) -> int:
        """This state's frozen-core count."""
        return _compute_n_core(
            [fragment.spec for fragment in state.fragments], self._mol.nelectron // 2
        )

    def _active_space_integrals(self, state: LASSQDState) -> tuple[MOIntegrals, int]:
        """This state's active-space integrals and its frozen-core count."""
        n_core = self._n_core(state)
        ao_eri, h_ao = self._cached_mol_integrals()
        n_act = sum(fragment.spec.n_orbitals for fragment in state.fragments)
        integrals = transform_integrals(
            self._mol, state.mo_coeff, n_core, n_act, ao_eri, h_ao
        )
        return integrals, n_core

    def update_state(self, state: LASSQDState) -> LASSQDState:
        """Reduce this round's sampled distributions into the next state.

        For every fragment, converts its program's sampled distribution to
        the blocked SQD bitstring convention, recovers the ground state via
        that fragment's ``SQDSolver``, and rebuilds its spatial RDMs
        from the recovered subspace. The full active-space RDM is then
        reassembled and the molecular orbitals re-optimised against it.

        The reassembled RDM includes the cross-fragment 2-RDM blocks, so it is
        the RDM of a product of fragment states and the returned ``energy`` is a
        variational upper bound. What fragmenting costs is the inter-fragment
        *correlation* that a product state cannot represent.

        Args:
            state: The state whose fragments were used to build the
                programs currently held by this ensemble.

        Returns:
            A new :class:`~divi.qprog.workflows.LASSQDState` with updated
            ``mo_coeff``, per-fragment RDMs and parameters, ``energy`` (this
            round's optimised total energy), ``previous_energy`` (set to
            ``state.energy``), and ``orbitals_converged``. ``state`` itself is
            left unmodified.

        Raises:
            ValueError: If SQD recovery fails for some fragment (e.g. no
                sampled bitstring can be brought into agreement with that
                fragment's target particle symmetry); the message names the
                failing fragment's program ID.

        Warns:
            UserWarning: If a fragment's recovered subspace contains only one
                determinant — this round captured no correlation energy for
                that fragment, indistinguishable from convergence by
                ``stop_reason`` alone. With ``SQDConfig.states`` set, also if
                excited states come from an incomplete subspace, if a state's
                ``<S^2>`` misses its multiplicity, or if two roots of one
                multiplicity are near-degenerate.

        Raises:
            RuntimeError: If ``state`` is not the state ``create_programs``
                built this round's circuits from, which would silently reduce
                the fragment results against different orbitals.
        """
        if self._state is not None and state is not self._state:
            raise RuntimeError(
                "update_state received a different state than create_programs "
                "built this round's circuits from; the reduction would use "
                "different orbitals than the fragment circuits were prepared in."
            )

        n_core = self._n_core(state)
        blocks = fragment_blocks(
            [fragment.spec for fragment in state.fragments], offset=n_core
        )
        ao_overlap = self._mol.intor_symmetric("int1e_ovlp")

        self._emit_workflow_stage("Recovering fragment subspaces (SQD)")
        recovery_started = time.perf_counter()
        programs = self.programs

        def in_sampled_basis(mo_coeff: np.ndarray) -> np.ndarray:
            """``mo_coeff`` with fragment 0's block in the basis its strings
            were sampled in; ``states`` admits a single fragment only."""
            rotated = mo_coeff.copy()
            rotated[:, blocks[0]] = (
                mo_coeff[:, blocks[0]] @ programs["fragment_0"].orbital_rotation
            )
            return rotated

        new_fragments = []
        subspace_sizes = []
        results = []
        for index, fragment in enumerate(state.fragments):
            # Same "fragment_{index}" id create_programs() assigned.
            program_id = f"fragment_{index}"
            program = programs[program_id]
            spec = fragment.spec
            sampled_orbitals = (
                state.mo_coeff[:, blocks[index]] @ program.orbital_rotation
            )

            carried: tuple[tuple[str, ...], tuple[str, ...]] = ((), ())
            if fragment.sampled_orbitals is not None:
                overlap = sampled_orbitals.T @ ao_overlap @ fragment.sampled_orbitals
                mapping = self._sqd.resolved_carryover_mapping
                carried = (
                    map_carried_strings(fragment.carried_alpha, overlap, mapping),
                    map_carried_strings(fragment.carried_beta, overlap, mapping),
                )

            probs = next(iter(program.best_probs.values()))
            sqd_probs = probs_to_sqd_bitstrings(probs, spec.n_orbitals)

            solver = self._solver_for(index, spec)
            sampled_problem = program.sampled_problem
            try:
                result = solver.solve(
                    sqd_probs,
                    sampled_problem.one_body,
                    sampled_problem.two_body,
                    one_body_beta=sampled_problem.one_body_beta,
                    carried=carried,
                )
            except ValueError as exc:
                raise ValueError(f"SQD failed for {program_id}: {exc}") from exc

            subspace_size = result.amplitudes.size
            subspace_sizes.append(subspace_size)
            self._emit_workflow_stage(
                f"Recovered {program_id}: {subspace_size} determinants, "
                f"fragment energy {result.energy:.8f} Ha"
            )
            if subspace_size == 1:
                self._warn_once(
                    ("single determinant", program_id),
                    f"{program_id}'s recovered subspace contains only one "
                    "determinant: this round captured no correlation energy "
                    f"for this fragment. {self._single_determinant_advice()}",
                )

            results.append(result)
            rdm1, rdm2, rdm1_alpha, rdm1_beta = rotate_rdms_to_fragment_basis(
                *compute_spatial_rdms(
                    result.strings_alpha,
                    result.strings_beta,
                    result.amplitudes,
                    spec.n_orbitals,
                ),
                program.orbital_rotation,
            )
            carried_alpha, carried_beta = result.carried
            new_fragments.append(
                FragmentState(
                    spec=spec,
                    rdm1=rdm1,
                    rdm2=rdm2,
                    params=np.asarray(program.best_params).ravel(),
                    rdm1_alpha=rdm1_alpha,
                    rdm1_beta=rdm1_beta,
                    carried_alpha=carried_alpha,
                    carried_beta=carried_beta,
                    sampled_orbitals=(
                        sampled_orbitals
                        if self._sqd.carryover_cutoff is not None
                        else None
                    ),
                )
            )

        ao_eri, h_ao = self._cached_mol_integrals()
        states = self._sqd.states
        if states is not None and not self._round_reports:
            # Once per run, before an orbital update can lift the degeneracy.
            self._warn_if_a_degenerate_set_is_split(
                in_sampled_basis(state.mo_coeff),
                n_core,
                new_fragments[0].spec,
                results[0],
                states,
                ao_eri,
                h_ao,
            )
        self._emit_workflow_stage("Assembling active-space RDMs")
        orbital_round = _OrbitalRound(
            self._mol,
            state.mo_coeff,
            n_core,
            [fragment.spec for fragment in new_fragments],
            *assemble_active_rdms(new_fragments),
            ao_eri,
            h_ao,
            gradient_tol=float(np.sqrt(self._energy_tol)),
            report=self._emit_workflow_stage,
            subspace=(
                None
                if states is None
                else _SampledSubspace(
                    in_sampled_basis(state.mo_coeff),
                    (results[0].strings_alpha, results[0].strings_beta),
                    states,
                    self._sqd.lambda_penalty,
                )
            ),
        )
        self._emit_workflow_stage("Re-optimising orbitals")
        orbital_started = time.perf_counter()
        solution, relaxed_states = self._orbital_update._solve(orbital_round)
        averaged = self._orbital_update._relaxes is _OrbitalTarget.AVERAGE
        orbital_seconds = time.perf_counter() - orbital_started
        if not np.isfinite(solution.energy):
            raise ValueError(
                f"Round {len(self._energy_history) + 1} produced a non-finite "
                f"energy ({solution.energy}), so its orbitals and RDMs cannot seed "
                "another round."
            )
        electronic_states: tuple[ElectronicState, ...] = ()
        if states is not None:
            result, spec = results[0], new_fragments[0].spec
            if relaxed_states is None:
                # Re-solve the CI at the round's orbitals.
                relaxed_states = subspace_states_at(
                    self._mol,
                    in_sampled_basis(solution.mo_coeff),
                    n_core,
                    spec,
                    (result.strings_alpha, result.strings_beta),
                    states,
                    self._sqd.lambda_penalty,
                    ao_eri,
                    h_ao,
                )
            electronic_states = self._electronic_states(
                "fragment_0", result, spec, relaxed_states
            )
        # A state-averaged solve's energy is the average, not the lowest state's.
        energy = electronic_states[0].energy if averaged else solution.energy
        previous = self._energy_history[-1] if self._energy_history else None
        self._energy_history.append(energy)

        escape, saddle_check = None, None
        if (
            self._orbital_update._relaxes is _OrbitalTarget.LOWEST_STATE
            and solution.converged
            and previous is not None
            and abs(energy - previous) < self._energy_tol
        ):
            escape, conclusive = escape_orbital_saddle(
                self._mol,
                solution.mo_coeff,
                n_core,
                orbital_round.specs,
                orbital_round.rdm1_active,
                orbital_round.rdm2_active,
                ao_eri,
                h_ao,
                min_energy_drop=_SADDLE_ESCAPE_DROP * self._energy_tol,
            )
            if escape is not None:
                saddle_check = "escaped"
            else:
                saddle_check = "passed" if conclusive else "inconclusive"

        self._round_reports.append(
            LASSQDRoundReport(
                number=len(self._energy_history),
                energy=energy,
                energy_change=None if previous is None else energy - previous,
                subspace_sizes=tuple(subspace_sizes),
                orbital_iterations=solution.n_iterations,
                orbital_evaluations=solution.n_evaluations,
                orbital_gradient_norm=solution.gradient_norm,
                orbital_converged=solution.converged,
                rotation_pairs=solution.n_rotation_pairs,
                recovery_seconds=orbital_started - recovery_started,
                orbital_seconds=orbital_seconds,
                states=electronic_states,
                averaged_energy=solution.energy if averaged else None,
                saddle_check=saddle_check,
            )
        )
        self._emit_workflow_stage(self._round_reports[-1].summary(), final=True)

        next_state = LASSQDState(
            mo_coeff=solution.mo_coeff,
            fragments=tuple(new_fragments),
            energy=solution.energy,
            previous_energy=state.energy,
            orbitals_converged=solution.converged,
        )
        if escape is not None:
            mo_coeff, escaped_energy = escape
            self._emit_workflow_stage(
                "Left a saddle point of the orbital energy, "
                f"{(escaped_energy - solution.energy) * 1000:+.3f} mHa; continuing"
            )
            # The drop exceeds energy_tol, so is_complete continues.
            next_state = replace(
                next_state,
                mo_coeff=mo_coeff,
                energy=escaped_energy,
                orbitals_converged=False,
            )
        return next_state

    def _electronic_states(
        self,
        program_id: str,
        result: SQDResult,
        spec: FragmentSpec,
        states: Sequence[ElectronicState],
    ) -> tuple[ElectronicState, ...]:
        """``states``, the subspace's eigenstates at the round's orbitals, in
        energy order, warning about any the subspace does not resolve."""
        excited = self._sqd.states != default_states(spec.n_alpha, spec.n_beta)
        full_size = full_space_size(spec.n_orbitals, spec.n_alpha, spec.n_beta)
        if excited and result.amplitudes.size < full_size:
            hint = (
                " Set SQDConfig.extension to add the determinants it misses."
                if self._sqd.extension is None
                else ""
            )
            self._warn_once(
                ("incomplete subspace", program_id),
                f"{program_id}'s subspace holds {result.amplitudes.size} of "
                f"{full_size} determinants. The circuit samples the ground "
                "state, so excited states in an incomplete subspace come out "
                f"too high.{hint}",
            )

        for key, message in _state_warnings(states):
            self._warn_once(key, message)
        return _in_energy_order(states)

    def _warn_once(self, key: Hashable, message: str) -> None:
        """Warn the first time ``key`` comes up in this run."""
        if key not in self._warned:
            self._warned.add(key)
            warn(message, UserWarning, stacklevel=_outside_divi_stacklevel())

    def _warn_if_a_degenerate_set_is_split(
        self,
        mo_coeff: np.ndarray,
        n_core: int,
        spec: FragmentSpec,
        result: SQDResult,
        requested: Mapping[int, int],
        ao_eri: np.ndarray,
        h_ao: np.ndarray,
    ) -> None:
        """Warn for each multiplicity whose last requested root is degenerate
        with the next, unrequested one, solved at ``mo_coeff``."""
        probe = {
            multiplicity: n_roots + 1
            for multiplicity, n_roots in requested.items()
            if n_roots < result.amplitudes.size
        }
        solved = subspace_states_at(
            self._mol,
            mo_coeff,
            n_core,
            spec,
            (result.strings_alpha, result.strings_beta),
            probe,
            self._sqd.lambda_penalty,
            ao_eri,
            h_ao,
        )
        for multiplicity, n_roots in requested.items():
            roots = [state for state in solved if state.multiplicity == multiplicity]
            if len(roots) <= n_roots:
                continue
            last, unrequested = roots[n_roots - 1], roots[n_roots]
            gap = unrequested.energy - last.energy
            if (
                not _is_off_spin(multiplicity, unrequested.spin_square)
                and gap < _SPLIT_SET_TOL
            ):
                warn(
                    f"State ({multiplicity}, {n_roots - 1}) lies within {gap:.1e} "
                    f"Ha of ({multiplicity}, {n_roots}), which SQDConfig.states "
                    "does not request: it splits a degenerate set, so the state "
                    "reported is an arbitrary member, and state averaging over "
                    f"part of the set breaks its symmetry. Request {multiplicity}: "
                    f"{n_roots + 1} or more.",
                    UserWarning,
                    stacklevel=_outside_divi_stacklevel(),
                )

    def is_complete(self, state: LASSQDState) -> bool:
        """Stop once the macro-cycle energy change is below ``energy_tol`` and
        the round's orbital solve converged."""
        if not abs(state.energy - state.previous_energy) < self._energy_tol:
            return False
        if not state.orbitals_converged:
            self._warn_once(
                "unconverged orbitals",
                "The macro-cycle energy change is below energy_tol but the "
                "orbital optimisation did not converge, so this is not a fixed "
                "point. Continuing; raise orbital_update's max_iterations or "
                "loosen energy_tol if the run hits max_rounds.",
            )
            return False
        if self._round_reports and all(
            size == 1 for size in self._round_reports[-1].subspace_sizes
        ):
            warn(
                "The macro-cycle converged with every fragment's subspace a "
                "single determinant: the mean-field fixed point, with no "
                f"correlation energy captured. {self._single_determinant_advice()}",
                UserWarning,
                stacklevel=2,
            )
        return True

    def _single_determinant_advice(self) -> str:
        """How to grow a subspace stuck at one determinant."""
        if self._sqd.resolved_max_dim == (1, 1):
            return "SQDConfig.max_dim caps each spin sector at one string; raise it."
        extension = (
            ", SQDConfig.extension='singles'," if self._sqd.extension is None else ""
        )
        return (
            "Use a larger sampling budget (the backend's shots, SQDConfig's "
            f"batch_size){extension} or a more expressive preparation."
        )

    @property
    def round_reports(self) -> tuple[LASSQDRoundReport, ...]:
        """Each completed round's :class:`LASSQDRoundReport`, in order. Cleared
        at the start of every :meth:`~divi.qprog.ensemble.ProgramEnsemble.run`
        that does not resume a checkpoint."""
        return tuple(self._round_reports)

    @property
    def electronic_states(self) -> dict[tuple[int, int], ElectronicState]:
        """The states ``SQDConfig.states`` requested, from the last completed
        round, keyed ``(multiplicity, index)`` and ordered by ascending energy;
        empty before the first or without ``states``.

        ``index`` counts within one multiplicity from 0, so with ``{1: 2,
        3: 1}`` the first excited singlet is ``(1, 1)`` and the lowest triplet
        ``(3, 0)``.

        Each is an eigenstate of the subspace at the round's final orbitals.
        After a lowest-state orbital update, the lowest state's energy can
        therefore sit slightly off :attr:`energy`, which keeps the CI the
        orbitals were optimised for.

        Warns:
            UserWarning: If no round has completed yet.
        """
        if not self._round_reports:
            _warn_no_round("electronic_states")
            return {}
        return {
            (state.multiplicity, state.index): state
            for state in self._round_reports[-1].states
        }

    @property
    def energy_history(self) -> tuple[float, ...]:
        """Total energy of each completed round, in order."""
        return tuple(self._energy_history)

    @property
    def best_energy(self) -> float:
        """Lowest energy over all completed rounds, or ``inf`` before the first.

        Every round's energy is a variational upper bound, so the lowest is the
        tightest one this run established.

        Note that ``workflow_state`` still holds the *last* round's orbitals,
        which are not the ones that produced this energy unless the two
        coincide.

        Warns:
            UserWarning: If no round has completed yet.
        """
        if not self._energy_history:
            _warn_no_round("best_energy")
            return float("inf")
        return min(self._energy_history)

    @property
    def energy(self) -> float:
        """Total energy of the lowest state in the last completed round, or
        ``inf`` before the first.

        A variational upper bound: the assembled RDM is that of a product of
        fragment states, so this is a genuine expectation value and cannot fall
        below CASSCF with the same active-space size. Fragmenting still
        costs accuracy -- see :ref:`lassqd-accuracy-characteristics`.

        The macro-cycle is not guaranteed monotone, so a later round can report
        a higher energy than an earlier one; ``energy_history`` records each, and
        ``best_energy`` gives the lowest. Under
        :class:`~divi.qprog.workflows.StateAveragedOrbitalUpdate` the
        macro-cycle converges on the weighted average instead, which each
        round report carries as ``averaged_energy``.

        Warns:
            UserWarning: If no round has completed yet.
        """
        if self.workflow_state is None or not self._energy_history:
            _warn_no_round("energy")
            return float("inf")
        return self._energy_history[-1]
