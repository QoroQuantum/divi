# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""The LASSQD program ensemble: construction and per-round program creation."""

import time
from collections.abc import Hashable, Sequence
from dataclasses import replace
from functools import cached_property
from math import isfinite
from numbers import Integral
from pathlib import Path
from typing import Any
from warnings import warn

import numpy as np
import scipy.linalg
from pyscf import scf
from pyscf.solvent import ddcosmo, pcm

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
from ._checkpoint import _WorkflowSnapshot, load_snapshot, save_snapshot
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
from ._excited_states import _ExcitedStates
from ._integrals import (
    AOIntegrals,
    MOIntegrals,
    assemble_active_rdms,
    build_active_permutation,
    cached_ao_eri,
    cached_h_ao,
    fragment_blocks,
    fragment_effective_integrals,
    transform_integrals,
)
from ._orbital_solvers import escape_orbital_saddle, orbital_irreps
from ._preparation import rotate_rdms_to_fragment_basis
from ._solvent import _SolventField
from ._sqd import (
    SQDConfig,
    SQDSolver,
    compute_spatial_rdms,
    map_carried_strings,
    probs_to_sqd_bitstrings,
    validate_sqd_for_fragment,
)
from ._state import (
    ElectronicState,
    FragmentSpec,
    FragmentState,
    LASSQDRoundReport,
    LASSQDState,
    StateTransition,
    StateWavefunction,
    _core_orbital_count,
    _OrbitalRound,
    _SampledSubspace,
    validate_fragment_specs,
)
from ._subspace import subspace_wavefunctions
from ._warnings import _outside_divi_stacklevel

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


def _orbital_nudge(
    n_orbitals: int, seed: int | None, irreps: np.ndarray | None = None
) -> np.ndarray:
    """A small seeded rotation of the MO register, mixing only orbitals of one
    irrep when ``irreps`` labels them.

    Point-group-symmetric orbitals have an exactly zero gradient along every
    rotation that mixes symmetries, so the macro-cycle cannot leave a symmetric
    saddle point of the orbital-CI energy without one.
    """
    generator = np.random.default_rng(seed).standard_normal((n_orbitals, n_orbitals))
    if irreps is not None:
        generator *= irreps[:, None] == irreps[None, :]
    return scipy.linalg.expm(_ORBITAL_NUDGE * (generator - generator.T))


def _warn_no_round(name: str) -> None:
    warn(
        f"{name} is not available: no LASSQD round has completed yet. Call run() "
        "first.",
        UserWarning,
        stacklevel=3,
    )


def _ao_density(core: np.ndarray, active: np.ndarray, rdm1: np.ndarray) -> np.ndarray:
    """The spin-traced AO density of doubly occupied ``core`` orbitals plus
    ``active`` orbitals carrying ``rdm1``."""
    return 2.0 * core @ core.T + active @ rdm1 @ active.T


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
    :class:`~divi.qprog.workflows.lassqd.CCSDPreparation` (the default) samples a
    one-repetition LUCJ operator seeded from CCSD, whose classical cost is the
    fragment's CCSD, so it never builds the fragment statevector;
    :class:`~divi.qprog.workflows.lassqd.LinearMethodPreparation` first optimises it
    with ffsim's exact linear method; :class:`~divi.qprog.workflows.lassqd.VQEPreparation`
    optimises an ansatz against backend-estimated energies.

    :attr:`energy` is a variational upper bound -- the assembled RDM is that of
    a product of fragment states, so the energy is a genuine expectation value
    (in gas phase; a ``solvent`` adds a solvation energy that is no bound).
    Fragmenting nonetheless costs accuracy, and the cost grows with how
    strongly the fragments interact; see
    :ref:`lassqd-accuracy-characteristics` in the LASSQD guide.

    Args:
        problem: An :class:`~divi.qprog.problems.MolecularProblem`
            built by :meth:`~divi.qprog.problems.MolecularProblem.\
from_molecule` from a PySCF ``gto.Mole`` (an RHF calculation, or ROHF for
            a non-zero ``spin``, is run on it lazily, in :meth:`initial_state`)
            or a restricted (RHF, ROHF) mean-field object — not from a
            PennyLane ``qchem.Molecule`` or from bare integrals, since the
            orbital optimisation needs the atomic-orbital basis. An open-shell
            molecule keeps a closed-shell core: its singly occupied orbitals
            must all be active. A restricted DFT object supplies starting
            orbitals only; its functional enters no energy.
        fragmentation: Which orbitals are active and how they split into
            fragments, as a
            :class:`~divi.qprog.workflows.lassqd.FragmentationConfig`.
        sqd: Sampling and diagonalisation budget per fragment solve, as an
            :class:`~divi.qprog.workflows.lassqd.SQDConfig`. Defaults to
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
        solvent: A PySCF continuum solvent model built on the molecule —
            ``pyscf.solvent.ddCOSMO(mol)``, ``ddPCM``, ``PCM`` or ``SMD`` —
            with its ``eps`` set, or for SMD its ``solvent``; ``None`` (the
            default) runs in gas phase. LASSQD works on a copy. Each round adds
            the reaction field of the lowest requested state's density from the
            round before, so the macro-cycle converges it self-consistently,
            and every energy includes the solvation energy. Excited states sit
            in that equilibrium field; :meth:`nonequilibrium_corrections` adds
            the fast response to each.
        **kwargs: ``backend`` (required), ``sampling_backend``, and
            ``reporting_level`` are consumed here; ``sampling_backend`` runs
            each fragment's final sample. Other keywords are
            forwarded to each fragment program. ``precision``,
            ``qem_protocol`` and ``suppress_performance_warnings`` work with
            any preparation; VQE options such as ``n_layers``,
            ``ansatz_kwargs``, ``grouping_strategy`` and ``early_stopping``
            require :class:`~divi.qprog.workflows.lassqd.VQEPreparation`.

    Raises:
        ValueError: If ``fragmentation``'s ``active_orbitals`` has out-of-range
            indices; if its ``n_active_orbitals`` or ``active_orbitals``
            selects no occupied or no virtual orbital of this molecule; if its
            ``fragment_atoms`` names an out-of-range atom or shares one between
            fragments; if ``energy_tol`` is not positive and finite; or if any fragment leaves no
            excitation available, fragments overlap, a singly occupied orbital
            is left out, or the fragments do not sum to the molecule's ``Sz``;
            if the molecule's ``spin`` is negative; if ``local_spins`` does
            not sum to ``spin``; if ``sqd.states`` is set with more than one
            fragment or asks for a state the fragment cannot hold; or if
            ``sqd.extension`` could outgrow the solver with ``max_dim`` unset;
            or if ``solvent`` was built on a different molecule or its
            dielectric constant is not a finite number of at least 1.
            The configuration objects validate their own fields on
            construction.
        TypeError: If ``solvent`` is not a PySCF ddCOSMO, ddPCM, PCM or SMD
            model, if
            ``backend`` is missing or not a
            :class:`~divi.backends.CircuitRunner`, if ``sampling_backend`` is
            neither one nor ``None``, if ``problem`` was not built
            from a PySCF ``Mole`` or mean-field, if ``preparation`` or
            ``orbital_update`` is not one of its strategies, or if a preparation other than
            :class:`~divi.qprog.workflows.lassqd.VQEPreparation` receives a keyword
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
        solvent: ddcosmo.DDCOSMO | pcm.PCM | None = None,
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
        if seed is not None and (
            isinstance(seed, bool) or not isinstance(seed, Integral)
        ):
            raise ValueError(f"seed must be an integer or None; got {seed!r}.")
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
        self._mol, self._mean_field = split_pyscf_input(molecule, open_shell=True)
        if self._mol.spin < 0:
            raise ValueError(
                f"LASSQD needs a non-negative spin; got spin={self._mol.spin}. "
                f"Build the molecule with spin={-self._mol.spin}: the energies do "
                "not depend on the sign of Sz, and PySCF's ROHF does not converge "
                "with more beta than alpha electrons."
            )
        # ROHF ordering: the spin highest occupied orbitals are singly occupied.
        self._spin = int(self._mol.spin)
        self._n_occupied = int(self._mol.nelec[0])
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
        n_occupied, spin = self._n_occupied, self._spin
        if fragmentation.active_spaces is not None:
            validate_fragment_specs(
                fragmentation.active_spaces, n_orbitals_total, n_occupied, spin=spin
            )
        if fragmentation.n_active_orbitals is not None:
            select_frontier_orbitals(
                n_orbitals_total,
                n_occupied,
                fragmentation.n_active_orbitals,
                spin=spin,
            )
        if fragmentation.active_orbitals is not None:
            split_active_orbitals(
                fragmentation.active_orbitals, n_occupied, n_orbitals_total, spin=spin
            )
        if (
            fragmentation.local_spins is not None
            and sum(fragmentation.local_spins) != spin
        ):
            raise ValueError(
                f"local_spins must sum to the molecule's 2S, mol.spin={spin}; got "
                f"{list(fragmentation.local_spins)}, summing to "
                f"{sum(fragmentation.local_spins)}."
            )
        if fragmentation.fragment_atoms is not None:
            _validate_fragment_atoms(fragmentation.fragment_atoms, self._mol.natm)
        self._solvent: _SolventField | None = None
        if solvent is not None:
            self._solvent = _SolventField(solvent, self._mol)
        elif getattr(self._mean_field, "with_solvent", None) is not None:
            warn(
                "The mean field passed in was solved in a solvent, but LASSQD has "
                "none, so it runs in gas phase from those orbitals. Pass "
                "solvent=mean_field.with_solvent to solvate it.",
                UserWarning,
                stacklevel=2,
            )

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
        self._seed = None if seed is None else int(seed)
        self._rng = np.random.default_rng(self._seed)
        self._extra_kwargs = kwargs

        self._state: LASSQDState | None = None
        self._solvers: dict[int, SQDSolver] = {}
        self._energy_history: list[float] = []
        self._round_reports: list[LASSQDRoundReport] = []
        self._warned: set[Hashable] = set()
        self._excited = _ExcitedStates(
            self._mol,
            self._sqd,
            self._spin,
            self._ao_integrals,
            self._warn_once,
            lambda: bool(self._round_reports),
        )

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
            A fresh :class:`~divi.qprog.workflows.lassqd.LASSQDState` with ``energy``
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
            mean_field = (
                (scf.ROHF(self._mol) if self._spin else scf.RHF(self._mol))
                if mean_field is None
                else mean_field.copy()
            )
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
        n_occupied = self._n_occupied
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
                self._ao_integrals(),
                mo_coeff,
                n_occupied,
                self._rng,
                n_active_orbitals=config.n_active_orbitals,
                max_orbitals_per_fragment=config.max_orbitals_per_fragment,
                coupling_threshold=config.coupling_threshold,
                active_orbitals=config.active_orbitals,
                fragment_atoms=config.fragment_atoms,
                local_spins=config.local_spins,
                spin=self._spin,
            )
            mo_coeff = mo_coeff.copy()
            mo_coeff[:, active_positions] = localized

        validate_fragment_specs(specs, n_orbitals_total, n_occupied, spin=self._spin)
        if config.active_spaces is None:
            # Explicit layouts were checked, and warned about, at construction.
            self._check_fragments(specs)

        n_core = _core_orbital_count(
            self._mol.nelectron, sum(spec.n_alpha + spec.n_beta for spec in specs)
        )
        relaxes = self._orbital_update._relaxes
        if relaxes is not _OrbitalTarget.NOTHING:
            # A state-averaged solve keeps a symmetric molecule's symmetry.
            irreps = (
                orbital_irreps(self._mol, mo_coeff)
                if relaxes is _OrbitalTarget.AVERAGE
                else None
            )
            mo_coeff = mo_coeff @ _orbital_nudge(n_orbitals_total, self._seed, irreps)

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
            The latest :class:`~divi.qprog.workflows.lassqd.LASSQDState`.

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
        """Clear per-workflow state, re-seed ``_rng`` and drop every fragment's
        cached ``SQDSolver``, so a repeated ``run()`` reproduces the first. The
        state of a program map the caller built and has not run is kept."""
        super()._reset_workflow_state()
        if not self._programs_pending:
            self._state = None
        self._rng = np.random.default_rng(self._seed)
        self._solvers.clear()
        self._energy_history.clear()
        self._round_reports.clear()
        self._warned.clear()
        self._excited.reset()
        if self._solvent is not None:
            self._solvent.reset()

    def _save_workflow_checkpoint_state(
        self, state: Any, round_dir: Path, stem: str
    ) -> dict[str, Any]:
        """Write an explicit, non-pickled snapshot of mutable LASSQD state."""
        if not isinstance(state, LASSQDState):
            raise TypeError(
                f"LASSQD checkpoint state must be LASSQDState, got "
                f"{type(state).__name__}."
            )

        snapshot = _WorkflowSnapshot(
            state=state,
            rng_state=self._rng.bit_generator.state,
            solver_rng_states={
                index: solver._rng.bit_generator.state
                for index, solver in self._solvers.items()
            },
            energy_history=tuple(self._energy_history),
            round_reports=tuple(self._round_reports),
            final_subspace=self._excited.final_subspace,
            reaction_field=None if self._solvent is None else self._solvent.field,
        )
        return save_snapshot(
            snapshot,
            round_dir,
            stem,
            mol=self._mol,
            configuration=self._configuration_record(),
        )

    def _load_workflow_checkpoint_state(
        self, payload: dict[str, Any], round_dir: Path, stem: str
    ) -> LASSQDState:
        """Load and validate an explicit LASSQD NPZ snapshot."""
        snapshot = load_snapshot(
            payload,
            round_dir,
            stem,
            mol=self._mol,
            n_orbitals=self._register_size(),
            n_occupied=self._n_occupied,
            spin=self._spin,
            sqd=self._sqd,
            configuration=self._configuration_record(),
        )
        self._apply_snapshot(snapshot)
        return snapshot.state

    def _apply_snapshot(self, snapshot: _WorkflowSnapshot) -> None:
        """Make ``snapshot`` this workflow's state."""
        self._rng.bit_generator.state = snapshot.rng_state
        self._solvers.clear()
        for index, solver_rng_state in snapshot.solver_rng_states.items():
            solver = self._solver_for(index, snapshot.state.fragments[index].spec)
            solver._rng.bit_generator.state = solver_rng_state
        self._rng.bit_generator.state = snapshot.rng_state
        self._energy_history = list(snapshot.energy_history)
        self._round_reports = list(snapshot.round_reports)
        self._excited.final_subspace = snapshot.final_subspace
        if self._solvent is not None:
            self._solvent.field = snapshot.reaction_field
        # Rebuilt programs keep the state they were built from.
        if not self._programs:
            self._state = snapshot.state

    def _configuration_record(self) -> dict[str, str]:
        """The settings a checkpoint must have been written with to resume
        here, by name: those that decide which computation a round performs."""
        return {
            "fragmentation": repr(self._fragmentation),
            "sqd": repr(self._sqd),
            "preparation": repr(self._preparation._checkpoint_record()),
            "fragment program options": _checkpoint_digest(
                dict(sorted(self._extra_kwargs.items()))
            ),
            "orbital_update": repr(self._orbital_update),
            "solvent": (
                "None" if self._solvent is None else self._solvent.checkpoint_record()
            ),
        }

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

    def _ao_integrals(self) -> AOIntegrals:
        """The molecule with this run's AO integrals, its core Hamiltonian
        carrying the solvent's current reaction field."""
        if self._solvent is None or self._solvent.field is None:
            return AOIntegrals(self._mol, self._ao_eri, self._h_ao)
        return AOIntegrals(self._mol, self._ao_eri, self._h_ao + self._solvent.field[1])

    @cached_property
    def _ao_eri(self) -> np.ndarray:
        """This run's AO two-electron integrals, computed once."""
        return cached_ao_eri(self._mol)

    @cached_property
    def _h_ao(self) -> np.ndarray:
        """This run's AO core Hamiltonian, computed once: the mean field's when
        one is held, so a relativistic or otherwise modified one carries
        through, else built from ``mol``."""
        if self._mean_field is None:
            return cached_h_ao(self._mol)
        return np.asarray(self._mean_field.get_hcore())

    def _n_core(self, state: LASSQDState) -> int:
        """This state's frozen-core count."""
        return _core_orbital_count(
            self._mol.nelectron,
            sum(
                fragment.spec.n_alpha + fragment.spec.n_beta
                for fragment in state.fragments
            ),
        )

    def _active_space_integrals(self, state: LASSQDState) -> tuple[MOIntegrals, int]:
        """This state's active-space integrals and its frozen-core count."""
        n_core = self._n_core(state)
        n_act = sum(fragment.spec.n_orbitals for fragment in state.fragments)
        integrals = transform_integrals(
            self._ao_integrals(), state.mo_coeff, n_core, n_act
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
            A new :class:`~divi.qprog.workflows.lassqd.LASSQDState` with updated
            ``mo_coeff``, per-fragment RDMs and parameters, ``energy`` (this
            round's optimised total energy), ``previous_energy`` (set to
            ``state.energy``), and ``orbitals_converged``. ``state`` itself is
            left unmodified.

        Raises:
            ValueError: If SQD recovery fails for some fragment (e.g. no
                sampled bitstring can be brought into agreement with that
                fragment's target particle symmetry); the message names the
                failing fragment's program ID.
            RuntimeError: If ``state`` is not the state ``create_programs``
                built this round's circuits from, which would silently reduce
                the fragment results against different orbitals.

        Warns:
            UserWarning: If a fragment's recovered subspace contains only one
                determinant — this round captured no correlation energy for
                that fragment, indistinguishable from convergence by
                ``stop_reason`` alone. With ``SQDConfig.states`` set, also if
                excited states come from an incomplete subspace, if the
                subspace resolves fewer states of a multiplicity than requested,
                if a state's ``<S^2>`` misses its multiplicity, or if two roots
                of one multiplicity are near-degenerate.
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

        ao = self._ao_integrals()
        states = self._sqd.states
        if states is not None and not self._round_reports:
            # Once per run, before an orbital update can lift the degeneracy.
            self._excited.warn_if_a_degenerate_set_is_split(
                ao,
                in_sampled_basis(state.mo_coeff),
                n_core,
                results[0],
                states,
            )
        self._emit_workflow_stage("Assembling active-space RDMs")
        orbital_round = _OrbitalRound(
            ao,
            state.mo_coeff,
            n_core,
            [fragment.spec for fragment in new_fragments],
            *assemble_active_rdms(new_fragments),
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
                    labels=(
                        None
                        if self._excited.final_subspace is None
                        else self._excited.final_subspace.labels
                    ),
                )
            ),
        )
        self._emit_workflow_stage("Re-optimising orbitals")
        orbital_started = time.perf_counter()
        solution, relaxed_states = self._orbital_update._solve(orbital_round)
        if (
            isinstance(self._orbital_update, StateAveragedOrbitalUpdate)
            and self._excited.final_subspace is not None
        ):
            solution, relaxed_states = self._excited.follow_swapped_states(
                self._orbital_update,
                orbital_round,
                solution,
                relaxed_states,
                len(self._energy_history) + 1,
            )
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
            electronic_states = self._excited.record_round(
                ao,
                n_core,
                states,
                results[0],
                new_fragments[0].spec,
                solution,
                relaxed_states,
                in_sampled_basis,
            )
        solvation_offset = 0.0
        if self._solvent is not None:
            density = self._lowest_state_density(
                solution.mo_coeff, n_core, orbital_round.rdm1_active
            )
            solvation_offset = self._solvent.energy_offset(density)
            solution = replace(solution, energy=solution.energy + solvation_offset)
            electronic_states = tuple(
                replace(state, energy=state.energy + solvation_offset)
                for state in electronic_states
            )
            self._solvent.advance(density)
        # A state-averaged solve's energy is the average, not the lowest state's.
        energy = (
            electronic_states[0].energy
            if averaged and electronic_states
            else solution.energy
        )
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
                ao,
                solution.mo_coeff,
                n_core,
                orbital_round.specs,
                orbital_round.rdm1_active,
                orbital_round.rdm2_active,
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
                energy=escaped_energy + solvation_offset,
                orbitals_converged=False,
            )
        self._state = None
        return next_state

    def _lowest_state_density(
        self, mo_coeff: np.ndarray, n_core: int, rdm1_active: np.ndarray
    ) -> np.ndarray:
        """AO density of the round's lowest state, which drives the reaction
        field: the lowest requested state at the final orbitals, or else the
        fragments' assembled RDM at ``mo_coeff``."""
        final_subspace = self._excited.final_subspace
        if self._sqd.states is not None and final_subspace is not None:
            wavefunctions = subspace_wavefunctions(
                self._ao_integrals(), final_subspace, n_core
            )
            lowest = min(
                wavefunctions, key=lambda wavefunction: wavefunction.state.energy
            )
            return _ao_density(
                lowest.core_orbitals, lowest.active_orbitals, lowest.rdm1
            )
        return _ao_density(
            mo_coeff[:, :n_core],
            mo_coeff[:, n_core : n_core + rdm1_active.shape[0]],
            rdm1_active,
        )

    def _warn_once(self, key: Hashable, message: str) -> None:
        """Warn the first time ``key`` comes up in this run."""
        if key not in self._warned:
            self._warned.add(key)
            warn(message, UserWarning, stacklevel=_outside_divi_stacklevel())

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
        """Each completed round's
        :class:`~divi.qprog.workflows.lassqd.LASSQDRoundReport`, in order. Cleared
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
        orbitals were optimised for. A root whose ``<S^2>`` lies nearer another
        multiplicity is left out, so a subspace short of states of one spin
        yields fewer than were requested.

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

    def state_wavefunctions(self) -> dict[tuple[int, int], StateWavefunction]:
        """Each state ``SQDConfig.states`` requested, with its CI vector and
        RDMs, from the last completed round, keyed ``(multiplicity, index)``
        and ordered as :attr:`electronic_states`.

        The subspace is re-diagonalised at the round's final orbitals on each
        call, so the states match :attr:`electronic_states` to solver precision.

        Raises:
            RuntimeError: If no round has completed with ``SQDConfig.states``
                set.
        """
        return self._excited.state_wavefunctions()

    def transition_properties(
        self,
    ) -> dict[tuple[tuple[int, int], tuple[int, int]], StateTransition]:
        """The one-electron transition between every pair of requested states
        of one multiplicity, from the last completed round, keyed
        ``(initial, final)`` labels with the lower state first, or for a
        degenerate pair the lower label.

        Pairs of different multiplicity are left out: their spin-free
        transition dipole vanishes. With one state of each multiplicity there
        is no pair, and the result is empty.

        Raises:
            RuntimeError: If no round has completed with ``SQDConfig.states``
                set.
        """
        return self._excited.transition_properties()

    def nevpt2(self) -> dict[tuple[int, int], float]:
        """Each requested state's strongly contracted NEVPT2 correction, from
        the last completed round, keyed ``(multiplicity, index)`` and ordered
        as :attr:`electronic_states`.

        Each correction is PySCF's state-specific NEVPT2 on the state's CI
        vector at the round's final orbitals; add it to the state's energy.
        The correction can reorder states, so the keys keep the uncorrected
        order. PySCF needs the CI vector over the whole active space, so the
        active space is bounded by memory as in classical CASCI. The subspace is
        re-diagonalised on each call, so repeat calls agree to solver precision.

        Raises:
            RuntimeError: If no round has completed with ``SQDConfig.states``
                set.
            MemoryError: If one full-space CI vector exceeds the molecule's
                ``max_memory``.
        """
        return self._excited.nevpt2()

    def nonequilibrium_corrections(
        self, optical_eps: float = 1.78
    ) -> dict[tuple[int, int], float]:
        """Each requested state's non-equilibrium solvent correction, from the
        last completed round, keyed ``(multiplicity, index)`` and ordered as
        :attr:`electronic_states`.

        A vertical excitation is too fast for the solvent's slow, orientational
        part to follow, so each state keeps the lowest state's equilibrium
        field, which :attr:`electronic_states` already holds, and only the
        fast, electronic part responds to its density change. The correction
        is that response's energy, ``1/2 tr(dD v_fast[dD])``, with ``dD`` the
        state's density less the lowest state's and ``v_fast`` the solvent
        model's response at ``optical_eps``; add it to the state's energy. It
        is first order: the state does not relax in the response.

        Args:
            optical_eps: The solvent's optical dielectric constant, its
                refractive index squared. The default, 1.78, is PySCF's and
                suits water or acetonitrile; set about 2 to 3 for an organic
                host.

        Raises:
            RuntimeError: If LASSQD was built without a ``solvent``, or no
                round has completed with ``SQDConfig.states`` set.
            ValueError: If ``optical_eps`` is not a finite number of at least
                1, or exceeds the solvent's static dielectric constant.
        """
        if self._solvent is None:
            raise RuntimeError(
                "nonequilibrium_corrections needs LASSQD built with a solvent, "
                "e.g. solvent=pyscf.solvent.ddCOSMO(mol)."
            )

        wavefunctions: list[StateWavefunction] = []

        def densities() -> list[np.ndarray]:
            wavefunctions.extend(
                self._excited.wavefunctions_in_energy_order(
                    "nonequilibrium_corrections"
                )
            )
            return [
                _ao_density(
                    wavefunction.core_orbitals,
                    wavefunction.active_orbitals,
                    wavefunction.rdm1,
                )
                for wavefunction in wavefunctions
            ]

        corrections = self._solvent.nonequilibrium(densities, optical_eps)
        return {
            (wavefunction.state.multiplicity, wavefunction.state.index): correction
            for wavefunction, correction in zip(wavefunctions, corrections)
        }

    @property
    def energy_history(self) -> tuple[float, ...]:
        """Total energy of each completed round, in order."""
        return tuple(self._energy_history)

    @property
    def best_energy(self) -> float:
        """Lowest energy over all completed rounds, or ``inf`` before the first.

        Every round's energy is a variational upper bound, so the lowest is the
        tightest one this run established. With a ``solvent`` it is no bound,
        and earlier rounds sit in a field not yet consistent with their
        density, so this is the last round's energy, as :attr:`energy`.

        Note that ``workflow_state`` still holds the *last* round's orbitals,
        which are not the ones that produced this energy unless the two
        coincide.

        Warns:
            UserWarning: If no round has completed yet.
        """
        if not self._energy_history:
            _warn_no_round("best_energy")
            return float("inf")
        if self._solvent is not None:
            return self._energy_history[-1]
        return min(self._energy_history)

    @property
    def energy(self) -> float:
        """Total energy of the lowest state in the last completed round, or
        ``inf`` before the first.

        A variational upper bound: the assembled RDM is that of a product of
        fragment states, so this is a genuine expectation value and cannot fall
        below CASSCF with the same active-space size. Fragmenting still
        costs accuracy -- see :ref:`lassqd-accuracy-characteristics`. With a
        ``solvent`` it includes the solvation energy and is no bound.

        The macro-cycle is not guaranteed monotone, so a later round can report
        a higher energy than an earlier one; ``energy_history`` records each, and
        ``best_energy`` gives the lowest. Under
        :class:`~divi.qprog.workflows.lassqd.StateAveragedOrbitalUpdate` the
        macro-cycle converges on the weighted average instead, which each
        round report carries as ``averaged_energy``.

        Warns:
            UserWarning: If no round has completed yet.
        """
        if self.workflow_state is None or not self._energy_history:
            _warn_no_round("energy")
            return float("inf")
        return self._energy_history[-1]
