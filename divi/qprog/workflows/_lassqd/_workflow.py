# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""The LASSQD program ensemble: construction and per-round program creation."""

import hashlib
import os
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from warnings import warn

import numpy as np
from pyscf import gto, scf

from divi.backends import CircuitRunner
from divi.hamiltonians._molecular import is_pyscf_input, split_pyscf_input
from divi.qprog.ensemble import ProgramEnsemble, ReportingLevel
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
    SecondOrderOrbitalSolve,
    _checkpoint_digest,
)
from ._integrals import (
    MOIntegrals,
    assemble_active_rdms,
    build_active_permutation,
    cached_ao_eri,
    cached_h_ao,
    fragment_blocks,
    fragment_effective_integrals,
    transform_integrals,
)
from ._preparation import rotate_rdms_to_fragment_basis
from ._sqd import (
    SQDConfig,
    SQDSolver,
    compute_spatial_rdms,
    map_carried_strings,
    probs_to_sqd_bitstrings,
)
from ._state import (
    FragmentSpec,
    FragmentState,
    LASSQDState,
    require_orthonormal,
    validate_fragment_specs,
)


@dataclass(frozen=True)
class LASSQDRoundReport:
    """One macro-cycle round's stage outputs.

    Appended to ``LASSQD.round_reports`` as the round completes, so an
    interrupted run keeps every finished round's numbers.

    Attributes:
        number: Round number, counting from 1.
        energy: This round's total energy, a variational upper bound.
        energy_change: Signed change from the previous round's energy, or
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

    def summary(self) -> str:
        """One-line human-readable digest of the round."""
        change = (
            "first round"
            if self.energy_change is None
            else f"change {self.energy_change:+.3e}"
        )
        return (
            f"Round {self.number} done. Energy: {self.energy:.8f} Ha "
            f"({change}); subspaces "
            f"{list(self.subspace_sizes)}; orbitals: {self.orbital_iterations} "
            f"iterations over {self.rotation_pairs} pairs, "
            f"|g| {self.orbital_gradient_norm:.2e}, "
            f"converged={self.orbital_converged}; "
            f"SQD {self.recovery_seconds:.1f}s, orbitals {self.orbital_seconds:.1f}s"
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
            basis. Closed-shell (RHF) only.
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
            fragments; if ``energy_tol`` is not positive; or if any fragment leaves no
            excitation available, fragments overlap, or the fragments do not
            sum to ``Sz = 0``. The configuration objects validate their own
            fields on construction.
        TypeError: If ``backend`` is missing, if ``problem`` was not built
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
        orbital_update: OrbitalUpdate = SecondOrderOrbitalSolve(),
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
                "orbital_update must be SecondOrderOrbitalSolve or "
                "FullOrbitalSolve; got "
                f"{type(orbital_update).__name__}."
            )
        if energy_tol <= 0:
            raise ValueError(f"energy_tol must be positive; got {energy_tol}.")
        if "backend" not in kwargs:
            raise TypeError(
                "LASSQD.__init__ missing required keyword-only argument: 'backend'."
            )

        backend = kwargs.pop("backend")
        sampling_backend = kwargs.pop("sampling_backend", None)
        reporting_level = kwargs.pop("reporting_level", ReportingLevel.COMPACT)
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
        self._preparation = preparation
        self._orbital_update = orbital_update
        self._energy_tol = energy_tol
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._extra_kwargs = kwargs

        self._state: LASSQDState | None = None
        self._solvers: dict[int, SQDSolver] = {}
        self._energy_history: list[float] = []
        self._round_reports: list[LASSQDRoundReport] = []
        self._ao_eri: np.ndarray | None = None
        self._h_ao: np.ndarray | None = None

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
        automatically), permutes the MO register into
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
                h_ao=self._core_hamiltonian(),
            )
            mo_coeff = mo_coeff.copy()
            mo_coeff[:, active_positions] = localized

        validate_fragment_specs(specs, n_orbitals_total, n_occupied)

        n_core = _compute_n_core(specs, n_occupied)

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
            RuntimeError: If no programs exist, if programs haven't
                completed execution, or if no round has been reduced into
                :attr:`~divi.qprog.ensemble.ProgramEnsemble.workflow_state` yet.
                A round driven by hand is reduced by :meth:`update_state`, whose
                return value is that round's state.
        """
        super().aggregate_results()
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
                ``stop_reason`` alone.

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
        new_fragments = []
        subspace_sizes = []
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
                mapping = self._sqd.carryover_mapping
                carried = (
                    map_carried_strings(fragment.carried_alpha, overlap, mapping),
                    map_carried_strings(fragment.carried_beta, overlap, mapping),
                )

            probs = next(iter(program.best_probs.values()))
            sqd_probs = probs_to_sqd_bitstrings(probs, spec.n_orbitals)

            solver = self._solver_for(index, spec)
            try:
                result = solver.solve(
                    sqd_probs,
                    program.h_alpha,
                    program.two_body,
                    one_body_beta=program.h_beta,
                    carried=carried,
                )
            except ValueError as exc:
                raise ValueError(
                    f"SQD failed for {program_id}: {exc} Increase the "
                    "backend shot count or n_recovery_iterations."
                ) from exc

            subspace_size = result.amplitudes.size
            subspace_sizes.append(subspace_size)
            self._emit_workflow_stage(
                f"Recovered {program_id}: {subspace_size} determinants, "
                f"fragment energy {result.energy:.8f} Ha"
            )
            if subspace_size == 1:
                warn(
                    f"{program_id}'s recovered subspace contains only one "
                    "determinant: this round captured no correlation energy "
                    "for this fragment. Use a larger sampling budget "
                    "(n_batches, batch_size) or a more expressive ansatz.",
                    UserWarning,
                    stacklevel=2,
                )

            rdm1, rdm2, rdm1_alpha, rdm1_beta = compute_spatial_rdms(
                result.strings_alpha,
                result.strings_beta,
                result.amplitudes,
                spec.n_orbitals,
            )
            rdm1, rdm2, rdm1_alpha, rdm1_beta = rotate_rdms_to_fragment_basis(
                rdm1, rdm2, rdm1_alpha, rdm1_beta, program.orbital_rotation
            )
            carried_alpha, carried_beta = solver.carried_strings(result)
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

        self._emit_workflow_stage("Assembling active-space RDMs")
        rdm1_active, rdm2_active = assemble_active_rdms(new_fragments)
        ao_eri, h_ao = self._cached_mol_integrals()

        self._emit_workflow_stage("Re-optimising orbitals")
        orbital_started = time.perf_counter()
        solve = self._orbital_update._solve(
            self._mol,
            state.mo_coeff,
            n_core,
            [fragment.spec for fragment in new_fragments],
            rdm1_active,
            rdm2_active,
            ao_eri,
            h_ao,
            gradient_tol=float(np.sqrt(self._energy_tol)),
            report=self._emit_workflow_stage,
        )
        if not np.isfinite(solve.energy):
            raise ValueError(
                f"Round {len(self._energy_history) + 1} produced a non-finite "
                f"energy ({solve.energy}), so its orbitals and RDMs cannot seed "
                "another round."
            )
        self._energy_history.append(solve.energy)

        self._round_reports.append(
            LASSQDRoundReport(
                number=len(self._energy_history),
                energy=solve.energy,
                energy_change=(
                    solve.energy - state.energy if np.isfinite(state.energy) else None
                ),
                subspace_sizes=tuple(subspace_sizes),
                orbital_iterations=solve.n_iterations,
                orbital_evaluations=solve.n_evaluations,
                orbital_gradient_norm=solve.gradient_norm,
                orbital_converged=solve.converged,
                rotation_pairs=solve.n_rotation_pairs,
                recovery_seconds=orbital_started - recovery_started,
                orbital_seconds=time.perf_counter() - orbital_started,
            )
        )
        self._emit_workflow_stage(self._round_reports[-1].summary(), final=True)

        return LASSQDState(
            mo_coeff=solve.mo_coeff,
            fragments=tuple(new_fragments),
            energy=solve.energy,
            previous_energy=state.energy,
            orbitals_converged=solve.converged,
        )

    def is_complete(self, state: LASSQDState) -> bool:
        """Stop once the macro-cycle energy change is below ``energy_tol`` and
        the round's orbital solve converged."""
        if not abs(state.energy - state.previous_energy) < self._energy_tol:
            return False
        if not state.orbitals_converged:
            warn(
                "The macro-cycle energy change is below energy_tol but the "
                "orbital optimisation did not converge, so this is not a fixed "
                "point. Continuing; raise orbital_update's max_iterations or "
                "loosen energy_tol if this repeats.",
                UserWarning,
                stacklevel=2,
            )
            return False
        return True

    @property
    def round_reports(self) -> tuple[LASSQDRoundReport, ...]:
        """Each completed round's :class:`LASSQDRoundReport`, in order."""
        return tuple(self._round_reports)

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
        """
        if not self._energy_history:
            return float("inf")
        return min(self._energy_history)

    @property
    def energy(self) -> float:
        """Total energy of the last completed round, or ``inf`` before the first.

        A variational upper bound: the assembled RDM is that of a product of
        fragment states, so this is a genuine expectation value and cannot fall
        below CASSCF with the same active-space size. Fragmenting still
        costs accuracy -- see :ref:`lassqd-accuracy-characteristics`.

        The macro-cycle is not guaranteed monotone, so a later round can report
        a higher energy than an earlier one; ``energy_history`` records each, and
        ``best_energy`` gives the lowest.
        """
        if self.workflow_state is None:
            return float("inf")
        return self.workflow_state.energy
