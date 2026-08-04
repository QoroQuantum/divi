# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""The LASSQD program ensemble: construction and per-round program creation."""

import copy
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
from warnings import warn

import numpy as np

from divi.hamiltonians._chem import _spo_from_integrals
from divi.qprog.algorithms import UCCSDAnsatz
from divi.qprog.algorithms._ansatze import Ansatz, _uccsd_excitations
from divi.qprog.algorithms._vqe import VQE
from divi.qprog.ensemble import ProgramEnsemble, ReportingLevel
from divi.qprog.optimizers import Optimizer

from ._active_space import (
    auto_fragment_specs,
    split_active_orbitals,
)
from ._active_space import validate_fragment_atoms as _validate_fragment_atoms
from ._integrals import (
    assemble_active_rdms,
    build_active_permutation,
    cached_ao_eri,
    cached_h_ao,
    fragment_effective_integrals,
    optimize_orbitals,
    transform_integrals,
)
from ._sqd import (
    SQDSolver,
    bitstring_to_spatial_det,
    compute_spatial_rdms,
    probs_to_sqd_bitstrings,
)
from ._state import FragmentSpec, FragmentState, LASSQDState, validate_fragment_specs

try:
    # pyrefly: ignore[missing-import]  # optional ``chem`` extra
    from pyscf import cc
    from pyscf.cc import addons as cc_addons
except ImportError:
    cc = None
    cc_addons = None


# Below this the two spin channels of an embedding potential are the same matrix
# to seeding precision, so averaging them for restricted CCSD costs nothing.
_SEED_SPIN_ASYMMETRY_TOL = 1e-6

# Largest occupied-virtual Fock element, relative to the Fock scale, at which the
# fragment's reference determinant still counts as a Hartree-Fock solution.
_SEED_STATIONARITY_TOL = 1e-6


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
        orbital_gradient_norm: Largest orbital-gradient component at the
            returned orbitals.
        orbital_converged: Whether the orbital solve reached a stationary point
            rather than stopping on a budget or an energy-reduction floor.
        rotation_pairs: Orbital pairs the rotation spanned.
        recovery_seconds: Wall-clock time in SQD recovery for all fragments.
        orbital_seconds: Wall-clock time in the orbital re-optimization.
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
            f"|g|max {self.orbital_gradient_norm:.2e}, "
            f"converged={self.orbital_converged}; "
            f"SQD {self.recovery_seconds:.1f}s, orbitals {self.orbital_seconds:.1f}s"
        )


class _FragmentVQE(VQE):
    """A fragment VQE whose fresh parameters can be supplied by the workflow.

    Accepts an explicit ``seed_params`` vector and returns it from
    ``_initialize_param_sets`` instead of the optimizer's own random
    initialization.

    Raises:
        ValueError: If ``seed_params`` is given and its length does not match
            this VQE's parameter count.
    """

    def __init__(self, *args, seed_params: np.ndarray | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        if seed_params is None:
            self._seed_params = None
        else:
            seed_params = np.asarray(seed_params, dtype=float)
            if seed_params.shape != (self.n_params,):
                raise ValueError(
                    f"seed_params has shape {seed_params.shape}, but this "
                    f"VQE expects {self.n_params} parameters."
                )
            self._seed_params = seed_params

    def _initialize_param_sets(self):
        if self._seed_params is None:
            return super()._initialize_param_sets()
        return np.tile(self._seed_params, (self.optimizer.n_param_sets, 1))


def _uccsd_amplitude_seed(
    coupled_cluster, spec: FragmentSpec, n_params: int
) -> np.ndarray:
    """Map CCSD ``t1``/``t2`` onto :class:`~divi.qprog.algorithms.UCCSDAnsatz`'s
    first layer by direct amplitude correspondence.

    ``pyscf.cc.addons.spatial2spin`` expands the restricted ``t1``/``t2`` into
    interleaved spin-orbital tensors (even index alpha, odd beta), indexed
    separately within the occupied and virtual blocks. ``qiskit_nature``'s
    excitation list uses blocked indices over the whole register, so each is
    remapped through its ``(spatial orbital, spin)`` pair.

    The correspondence is positional, so ``coupled_cluster`` must have been
    solved in the same orbital basis the ansatz excites in -- the fragment's own
    orbitals, not a rotated set of them.

    The angles are ``theta_single = -t1`` and ``theta_double = +t2``: a unique
    excitation carries the amplitude itself, with no antisymmetrization factor
    and no same-spin/mixed-spin distinction.

    Requires a spin-balanced fragment -- one occupied count serves both spins.
    Only the first layer is seeded; further layers have no corresponding CCSD
    amplitude and stay at zero.
    """
    # pyrefly: ignore[missing-attribute]  # cc_addons is None only if pyscf is absent
    t1_full = cc_addons.spatial2spin(coupled_cluster.t1)
    # pyrefly: ignore[missing-attribute]  # cc_addons is None only if pyscf is absent
    t2_full = cc_addons.spatial2spin(coupled_cluster.t2)
    n_spatial = spec.n_orbitals
    n_occupied = spec.n_alpha

    def block_index(blocked: int) -> int:
        """Amplitude-block index for a blocked spin-orbital index."""
        spin, spatial = divmod(blocked, n_spatial)
        if spatial < n_occupied:
            return 2 * spatial + spin
        return 2 * (spatial - n_occupied) + spin

    first_layer = []
    for occupied, unoccupied in _uccsd_excitations(n_spatial, (n_occupied, n_occupied)):
        occupied_indices = [block_index(index) for index in occupied]
        virtual_indices = [block_index(index) for index in unoccupied]
        if len(occupied) == 1:
            first_layer.append(-t1_full[occupied_indices[0], virtual_indices[0]])
        else:
            first_layer.append(t2_full[tuple(occupied_indices + virtual_indices)])

    seed = np.zeros(n_params)
    take = min(n_params, len(first_layer))
    seed[:take] = first_layer[:take]
    return seed


def _ccsd_seed_params(
    h_eff: np.ndarray,
    g_frag: np.ndarray,
    spec: FragmentSpec,
    n_params: int,
    ansatz: Ansatz | None = None,
) -> np.ndarray | None:
    """Map a fragment's CCSD amplitudes onto an ansatz parameter vector.

    Optimization started from random parameters converges poorly, and SQD's
    subspace quality depends directly on the sampled distribution covering
    the right determinants, so a fresh fragment's first round is seeded from
    coupled-cluster amplitudes computed on that fragment's own effective
    integrals, instead of starting from the optimizer's random initial guess.

    Only :class:`~divi.qprog.algorithms.UCCSDAnsatz` is seeded, via
    :func:`_uccsd_amplitude_seed`: its parameters are themselves
    singles-and-doubles excitation amplitudes, so each is read off the matching
    entry of ``t1``/``t2``. No other ansatz has parameters that correspond to
    CCSD amplitudes -- LUCJ's hopping and Coulomb angles do not -- so those warn
    and defer to the optimizer's own initialization.

    The CCSD runs in the fragment's own orbital basis, on the determinant the
    ansatz's Hartree-Fock reference prepares, rather than on a self-consistent
    field's canonical orbitals. Fragment orbitals are localized, and an SCF
    would rotate within the occupied and virtual blocks -- leaving the reference
    determinant and its energy untouched while permuting which amplitude belongs
    to which orbital pair, so the resulting seed would be attached to the wrong
    excitations. That determinant must still be a Hartree-Fock solution for its
    amplitudes to mean anything, which the occupied-virtual Fock block checks.

    Args:
        h_eff: Fragment's effective one-body integrals, shape
            ``(n_orbitals, n_orbitals)``.
        g_frag: Fragment's bare two-body integrals, shape
            ``(n_orbitals,) * 4``.
        spec: Fragment specification.
        n_params: Length of the returned vector.
        ansatz: The fragment's configured ansatz.

    Returns:
        A length-``n_params`` vector, or ``None`` (with a ``UserWarning``) if
        ``ansatz`` is not a ``UCCSDAnsatz``, if the fragment is spin-imbalanced
        (``n_alpha != n_beta``, which restricted CCSD cannot represent), if the
        fragment's reference determinant is not Hartree-Fock stationary, or if
        the CCSD calculation fails to converge or raises.
    """
    if not isinstance(ansatz, UCCSDAnsatz):
        warn(
            f"CCSD seeding skipped for fragment {spec.orbitals}: only "
            "UCCSDAnsatz has parameters that correspond to CCSD amplitudes, got "
            f"{type(ansatz).__name__}. Falling back to the optimizer's own "
            "initialization.",
            UserWarning,
            stacklevel=2,
        )
        return None

    if spec.n_alpha != spec.n_beta:
        warn(
            f"CCSD seeding skipped for fragment {spec.orbitals}: restricted "
            f"CCSD requires equal alpha/beta electron counts, got n_alpha="
            f"{spec.n_alpha}, n_beta={spec.n_beta}. Falling back to the "
            "optimizer's own initialization.",
            UserWarning,
            stacklevel=2,
        )
        return None

    try:
        # pyrefly: ignore[missing-import]  # optional ``chem`` extra
        from pyscf import ao2mo, gto, scf

        n_orb = spec.n_orbitals
        n_occupied = spec.n_alpha
        fake_mol = gto.M(verbose=0)
        fake_mol.nelectron = spec.n_alpha + spec.n_beta
        fake_mol.incore_anyway = True

        mean_field = scf.RHF(fake_mol)
        # pyrefly: ignore[bad-assignment]  # overriding with fragment integrals
        mean_field.get_hcore = lambda *args: h_eff
        # pyrefly: ignore[bad-assignment]  # overriding with fragment integrals
        mean_field.get_ovlp = lambda *args: np.eye(n_orb)
        mean_field._eri = ao2mo.restore(8, g_frag, n_orb)

        occupations = np.zeros(n_orb)
        occupations[:n_occupied] = 2.0
        mean_field.mo_coeff = np.eye(n_orb)
        mean_field.mo_occ = occupations
        # pyrefly: ignore[not-callable]  # resolved on the pyscf mean-field at runtime
        density = mean_field.make_rdm1()
        fock = mean_field.get_fock(dm=density)
        off_diagonal = np.abs(fock[:n_occupied, n_occupied:]).max()
        if off_diagonal > _SEED_STATIONARITY_TOL * max(1.0, np.abs(fock).max()):
            raise RuntimeError(
                "the fragment's reference determinant is not a Hartree-Fock "
                f"solution in its own orbital basis (max |F_ov| = "
                f"{off_diagonal:.3e})"
            )
        mean_field.mo_energy = np.diag(fock)
        mean_field.e_tot = mean_field.energy_tot(dm=density)
        mean_field.converged = True

        # pyrefly: ignore[missing-attribute]  # cc is None only if pyscf is absent
        coupled_cluster = cc.CCSD(mean_field)
        coupled_cluster.kernel()
        if not coupled_cluster.converged:
            raise RuntimeError("CCSD did not converge")
    except Exception as exc:
        warn(
            f"CCSD seeding failed for fragment {spec.orbitals}: {exc}. "
            "Falling back to the optimizer's own initialization.",
            UserWarning,
            stacklevel=2,
        )
        return None

    return _uccsd_amplitude_seed(coupled_cluster, spec, n_params)


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
    """Build a diagonal RDM guess for a fresh fragment.

    Places ``n_alpha`` alpha electrons and ``n_beta`` beta electrons on the
    fragment's lowest-indexed orbitals to form the 1-RDM. Only the
    ``[p, p, q, q]`` elements of the 2-RDM are populated, each set to
    ``rdm1[p, p] * rdm1[q, q]``; every other element, including all
    off-diagonal-block elements, stays zero.

    Returns ``(rdm1, rdm2, rdm1_alpha, rdm1_beta)``.
    """
    n_orb = spec.n_orbitals
    occ_alpha = np.zeros(n_orb)
    occ_alpha[: spec.n_alpha] = 1.0
    occ_beta = np.zeros(n_orb)
    occ_beta[: spec.n_beta] = 1.0
    occupation = occ_alpha + occ_beta
    rdm1 = np.diag(occupation)

    rdm2 = np.zeros((n_orb, n_orb, n_orb, n_orb))
    p, q = np.meshgrid(np.arange(n_orb), np.arange(n_orb), indexing="ij")
    rdm2[p, p, q, q] = occupation[p] * occupation[q]
    return rdm1, rdm2, np.diag(occ_alpha), np.diag(occ_beta)


class LASSQD(ProgramEnsemble):
    """Localized active-space sample-based quantum diagonalization.

    Partitions a molecule's active space into fragments, runs one VQE per
    fragment against its own mean-field-embedded effective Hamiltonian, and
    (in later rounds) recovers the ground state via sample-based quantum
    diagonalization. This class builds the workflow state and the per-round
    VQE programs; running rounds and aggregating results are handled
    elsewhere.

    :attr:`energy` is a variational upper bound -- the assembled RDM is that of
    a product of fragment states, so the energy is a genuine expectation value.
    Fragmenting nonetheless costs accuracy, and the cost grows with how
    strongly the fragments interact; see
    :ref:`lassqd-accuracy-characteristics` in the user guide.

    Args:
        molecule: A PySCF ``gto.Mole`` (an RHF calculation is run on it lazily,
            in :meth:`initial_state`) or a restricted mean-field object —
            not a PennyLane ``qchem.Molecule``. Closed-shell (RHF) only.
        optimizer: Optimizer template, deep-copied for each fragment's VQE.
        active_spaces: Explicit fragment layout, one ``FragmentSpec`` per
            fragment. Fixes which orbitals are active, how they partition, and
            each fragment's spin at once, and skips localization entirely.
            Mutually exclusive with the automatic path below.
        n_active_orbitals: Total active orbitals to select around the HOMO-LUMO
            gap. Mutually exclusive with ``active_spaces`` and
            ``active_orbitals``.
        active_orbitals: Explicit MO column indices forming the active space,
            instead of selecting it by energy. Use it when the active space is
            defined by orbital character -- a metal ``d`` manifold can sit well
            below the HOMO with its virtual partners well above the LUMO, where
            no frontier count reaches them. Pair it with a mean field whose
            orbitals already carry that character, e.g. from PySCF's AVAS, and
            with ``fragment_atoms`` to say which centre each belongs to.
            Mutually exclusive with ``active_spaces`` and ``n_active_orbitals``.
        max_orbitals_per_fragment: Maximum spatial orbitals per automatically
            built fragment. Ignored when ``active_spaces`` or ``fragment_atoms``
            is given.
        coupling_threshold: Relative edge-pruning threshold for the automatic
            fragmentation's orbital coupling graph. Ignored when
            ``active_spaces`` or ``fragment_atoms`` is given.
        fragment_atoms: One sequence of atom indices per fragment. Assigns each
            localized active orbital to the fragment owning the atom it sits on,
            replacing the coupling-graph clustering -- one fragment per metal
            centre, for instance. Ignored when ``active_spaces`` is given.
        local_spins: Per-fragment ``2S``, in the order ``fragment_atoms`` names
            them. The fragment's electron count comes from its occupied
            orbitals, so this sets spin alone: ``n_alpha - n_beta = 2S``. Use it
            for spin-polarized fragments, e.g. ``[2, -2]`` for an
            antiferromagnetically coupled dimer of local triplets. The fragments
            must still sum to ``Sz = 0``. Requires ``fragment_atoms``.
        ansatz: Per-fragment ansatz. Defaults to ``UCCSDAnsatz()``.
        max_iterations: Max optimization iterations per fragment VQE. An
            iteration is an optimizer step, not a circuit evaluation: a
            gradient-free method spends several evaluations per step and
            ``n_params + 1`` of them building its initial simplex before the
            first step, so this has to scale with the ansatz's parameter count.
        max_orbital_iterations: Cap on L-BFGS-B iterations in each round's
            orbital re-optimization, a separate solve from the fragment VQEs and
            usually the round's dominant cost on a large register. ``None``
            leaves it uncapped. A capped round still returns its best orbitals
            but is not a stationary point, and reports as not converged.
        n_batches: Number of SQD batches per fragment recovery.
        batch_size: Configurations sampled per SQD batch, so the subspace holds
            up to ``batch_size ** 2`` determinants. The accuracy knob; a
            one-determinant subspace is the mean field.
        n_sqd_iterations: Number of SQD self-consistent recovery iterations.
        energy_tol: Macro-cycle stops once consecutive rounds' total energies
            differ by less than this (Hartree).
        lambda_penalty: Weight of the S² spin-contamination penalty added to
            each fragment's projected Hamiltonian before diagonalization.
        seed: Seed for fragmentation, localization, and SQD subsampling, also
            passed to the backend. Reproducibility is limited by the backend:
            :class:`~divi.backends.QiskitSimulator` seeds exactly, while
            :class:`~divi.backends.MaestroSimulator` cannot, so identical runs
            are not guaranteed to agree bit for bit there.
        **kwargs: ``backend`` (required) and ``reporting_level`` are consumed
            here; ``program_id`` and ``progress_queue`` are set internally and
            must not be passed here. Any other keyword is forwarded verbatim
            to every fragment's :class:`~divi.qprog.algorithms.VQE`, e.g.
            ``grouping_strategy``, ``shot_distribution``, ``precision``, or
            ``early_stopping``.

    Raises:
        ValueError: If not exactly one of ``active_spaces``,
            ``n_active_orbitals``, ``active_orbitals`` is given; if
            ``fragment_atoms`` or ``local_spins`` is combined with
            ``active_spaces``; if ``local_spins`` is given without
            ``fragment_atoms`` or does not match its length; if
            ``active_orbitals`` has duplicates, out-of-range indices, or no
            occupied or no virtual orbital; if ``fragment_atoms`` names an
            out-of-range atom or shares one between fragments; if
            ``n_active_orbitals`` is not positive; if ``max_iterations`` is below
            1; if ``max_orbital_iterations`` is given and below 1; if
            ``max_orbitals_per_fragment`` is below 1; if ``n_batches``,
            ``batch_size``, or ``n_sqd_iterations`` is below 1; if
            ``energy_tol`` is not positive; if ``coupling_threshold`` or
            ``lambda_penalty`` is negative; or if any fragment leaves no
            excitation available, fragments overlap, or the fragments do not
            sum to ``Sz = 0``.
        TypeError: If ``program_id`` or ``progress_queue`` is passed via
            ``kwargs``, if ``backend`` is missing, if ``molecule`` is
            neither a PySCF ``Mole`` nor a restricted mean-field, or if
            ``ansatz`` is not an :class:`~divi.qprog.algorithms.Ansatz`.
        NotImplementedError: If the molecule is open-shell, or its mean-field
            is not restricted (non-2D ``mo_coeff``).
        ImportError: If the ``chem`` extra is not installed.
    """

    def __init__(
        self,
        molecule: Any,
        *,
        optimizer: Optimizer,
        active_spaces: Sequence[FragmentSpec] | None = None,
        n_active_orbitals: int | None = None,
        active_orbitals: Sequence[int] | None = None,
        max_orbitals_per_fragment: int = 4,
        coupling_threshold: float = 1e-3,
        fragment_atoms: Sequence[Sequence[int]] | None = None,
        local_spins: Sequence[int] | None = None,
        ansatz: Ansatz | None = None,
        max_iterations: int = 10,
        max_orbital_iterations: int | None = None,
        n_batches: int = 15,
        batch_size: int = 170,
        n_sqd_iterations: int = 6,
        energy_tol: float = 1e-6,
        lambda_penalty: float = 0.2,
        seed: int | None = None,
        **kwargs,
    ):
        n_given = sum(
            spec is not None
            for spec in (active_spaces, n_active_orbitals, active_orbitals)
        )
        if n_given != 1:
            raise ValueError(
                "Pass exactly one of active_spaces (explicit fragment layout), "
                "n_active_orbitals (frontier selection), or active_orbitals "
                "(explicit MO indices)."
            )
        for name, value in (
            ("local_spins", local_spins),
            ("fragment_atoms", fragment_atoms),
        ):
            if value is not None and active_spaces is not None:
                raise ValueError(
                    f"{name} applies to automatic fragmentation only; "
                    "active_spaces already fixes the fragment layout."
                )
        if local_spins is not None and fragment_atoms is None:
            raise ValueError(
                "local_spins requires fragment_atoms: coupling-graph fragment "
                "order depends on max_orbitals_per_fragment, coupling_threshold "
                "and the localization RNG, so a positional spin list would not "
                "name a stable fragment."
            )
        if n_active_orbitals is not None and n_active_orbitals <= 0:
            raise ValueError(
                f"n_active_orbitals must be positive; got {n_active_orbitals}."
            )
        if max_iterations < 1:
            raise ValueError(
                f"max_iterations must be at least 1; got {max_iterations}."
            )
        if max_orbital_iterations is not None and max_orbital_iterations < 1:
            raise ValueError(
                "max_orbital_iterations must be at least 1 when given; got "
                f"{max_orbital_iterations}."
            )
        if max_orbitals_per_fragment < 1:
            raise ValueError(
                "max_orbitals_per_fragment must be at least 1; got "
                f"{max_orbitals_per_fragment}."
            )
        if n_batches < 1:
            raise ValueError(f"n_batches must be at least 1; got {n_batches}.")
        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1; got {batch_size}.")
        if n_sqd_iterations < 1:
            raise ValueError(
                f"n_sqd_iterations must be at least 1; got {n_sqd_iterations}."
            )
        if energy_tol <= 0:
            raise ValueError(f"energy_tol must be positive; got {energy_tol}.")
        if coupling_threshold < 0:
            raise ValueError(
                f"coupling_threshold must be non-negative; got {coupling_threshold}."
            )
        if lambda_penalty < 0:
            raise ValueError(
                f"lambda_penalty must be non-negative; got {lambda_penalty}."
            )
        if ansatz is not None and not isinstance(ansatz, Ansatz):
            raise TypeError(
                f"ansatz must be an Ansatz instance; got {type(ansatz).__name__}."
            )
        for reserved in ("program_id", "progress_queue"):
            if reserved in kwargs:
                raise TypeError(
                    f"LASSQD sets {reserved!r} internally; do not pass it via "
                    "kwargs."
                )
        if "backend" not in kwargs:
            raise TypeError(
                "LASSQD.__init__ missing required keyword-only argument: 'backend'."
            )

        super().__init__(
            backend=kwargs.pop("backend"),
            reporting_level=kwargs.pop("reporting_level", ReportingLevel.COMPACT),
        )

        try:
            # pyrefly: ignore[missing-import]  # optional ``chem`` extra
            from pyscf import gto, scf
        except ImportError as exc:
            raise ImportError(
                "LASSQD requires the 'chem' extra; install it with "
                "`pip install qoro-divi[chem]`."
            ) from exc

        if isinstance(molecule, gto.Mole):
            self._mol = molecule
            mean_field = None
        elif isinstance(molecule, scf.hf.SCF):
            mean_field = molecule
            self._mol = mean_field.mol
        else:
            raise TypeError(
                "LASSQD expects a pyscf Mole or restricted mean-field object, "
                f"got {type(molecule).__name__}."
            )

        if self._mol.spin != 0:
            raise NotImplementedError(
                "Only closed-shell (RHF) systems are supported; got an "
                f"open-shell molecule with spin={self._mol.spin}."
            )

        if mean_field is not None and getattr(mean_field, "mo_coeff", None) is not None:
            mo_coeff = np.asarray(mean_field.mo_coeff)
            if mo_coeff.ndim != 2:
                raise NotImplementedError(
                    "Only restricted (closed-shell) mean-fields are "
                    f"supported; got mo_coeff with {mo_coeff.ndim} dimensions."
                )
        self._mean_field = mean_field

        n_orbitals_total = self._mol.nao_nr()
        n_occupied = self._mol.nelectron // 2
        if active_spaces is not None:
            validate_fragment_specs(active_spaces, n_orbitals_total, n_occupied)

        # Materialize and validate here rather than in ``initial_state``: these
        # are consumed more than once, so a generator would arrive empty the
        # second time, and an out-of-range index should fail at construction
        # like an explicit ``active_spaces`` does.
        self._active_orbitals = (
            None if active_orbitals is None else tuple(int(o) for o in active_orbitals)
        )
        if self._active_orbitals is not None:
            split_active_orbitals(self._active_orbitals, n_occupied, n_orbitals_total)

        self._fragment_atoms = (
            None
            if fragment_atoms is None
            else tuple(tuple(int(a) for a in atoms) for atoms in fragment_atoms)
        )
        if self._fragment_atoms is not None:
            _validate_fragment_atoms(self._fragment_atoms, self._mol.natm)

        self._local_spins = (
            None if local_spins is None else tuple(int(s) for s in local_spins)
        )
        if self._local_spins is not None and self._fragment_atoms is not None:
            if len(self._local_spins) != len(self._fragment_atoms):
                raise ValueError(
                    f"local_spins has {len(self._local_spins)} entries but "
                    f"fragment_atoms names {len(self._fragment_atoms)} fragments."
                )

        self._active_spaces = None if active_spaces is None else list(active_spaces)
        self._n_active_orbitals = n_active_orbitals
        self._max_orbitals_per_fragment = max_orbitals_per_fragment
        self._coupling_threshold = coupling_threshold
        self._ansatz: Ansatz = UCCSDAnsatz() if ansatz is None else ansatz
        self._optimizer = optimizer
        self._max_iterations = max_iterations
        self._max_orbital_iterations = max_orbital_iterations
        self._n_batches = n_batches
        self._batch_size = batch_size
        self._n_sqd_iterations = n_sqd_iterations
        self._energy_tol = energy_tol
        self._lambda_penalty = lambda_penalty
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._extra_kwargs = kwargs

        self._state: LASSQDState | None = None
        self._solvers: dict[int, SQDSolver] = {}
        self._energy_history: list[float] = []
        self._round_reports: list[LASSQDRoundReport] = []
        # No round has run, so nothing has stalled yet.
        self._orbitals_converged = True
        self._ao_eri: np.ndarray | None = None
        self._h_ao: np.ndarray | None = None

    def initial_state(self) -> LASSQDState:
        """Resolve fragments and build the initial workflow state.

        Runs RHF on the molecule if no mean-field has run yet, resolves
        fragments (explicit ``active_spaces``, validated via
        ``validate_fragment_specs``; or automatic fragmentation via
        ``auto_fragment_specs``), permutes the MO register into
        ``[core | fragments | virtual]`` order via
        ``build_active_permutation``, and seeds each fragment with a diagonal
        mean-field RDM guess.

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
        # pyrefly: ignore[missing-import]  # optional ``chem`` extra
        from pyscf import scf

        mean_field = self._mean_field
        if mean_field is None or getattr(mean_field, "mo_coeff", None) is None:
            mean_field = scf.RHF(self._mol).run(verbose=0)
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
            self._n_active_orbitals is not None
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

        if self._active_spaces is not None:
            specs = list(self._active_spaces)
        else:
            specs, localized, active_positions = auto_fragment_specs(
                self._mol,
                mo_coeff,
                n_occupied,
                self._rng,
                n_active_orbitals=self._n_active_orbitals,
                max_orbitals_per_fragment=self._max_orbitals_per_fragment,
                coupling_threshold=self._coupling_threshold,
                active_orbitals=self._active_orbitals,
                fragment_atoms=self._fragment_atoms,
                local_spins=self._local_spins,
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
        """Create one fragment VQE per fragment in ``state``.

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

        n_occupied = self._mol.nelectron // 2
        n_core = _compute_n_core(
            [fragment.spec for fragment in state.fragments], n_occupied
        )
        ao_eri, _ = self._cached_mol_integrals()
        n_act = sum(fragment.spec.n_orbitals for fragment in state.fragments)
        integrals = transform_integrals(
            self._mol, state.mo_coeff, n_core, n_act, ao_eri
        )
        fragment_seeds = self._rng.integers(0, 2**63 - 1, size=len(state.fragments))

        for index, fragment in enumerate(state.fragments):
            h_alpha, h_beta, g_frag = fragment_effective_integrals(
                integrals, state.fragments, index
            )
            prog_id = f"fragment_{index}"
            self._programs[prog_id] = self._build_fragment_program(
                fragment,
                h_alpha,
                h_beta,
                g_frag,
                prog_id,
                int(fragment_seeds[index]),
            )

    def _build_fragment_program(
        self,
        fragment: FragmentState,
        h_alpha: np.ndarray,
        h_beta: np.ndarray,
        g_frag: np.ndarray,
        program_id: str,
        seed: int,
    ) -> _FragmentVQE:
        """Build one fragment's VQE program from its effective integrals.

        A fresh fragment (``fragment.params is None``) is seeded from its
        own CCSD amplitudes via :func:`_ccsd_seed_params`; a fragment
        warm-started from a previous round uses ``fragment.params`` directly
        and never calls CCSD.

        CCSD seeding is restricted, so it gets the spin-averaged one-body
        potential. That only affects the optimizer's starting point, not the
        Hamiltonian it optimizes against, which carries both channels -- but a
        spin-restricted-looking seed can still land a local optimizer in a
        different basin than the symmetry-broken solution, so a materially
        asymmetric embedding is warned about.
        """
        hamiltonian = _spo_from_integrals(
            h_alpha, g_frag, constant=0.0, one_body_beta=h_beta
        )
        n_electrons = fragment.spec.n_alpha + fragment.spec.n_beta

        if fragment.params is not None:
            seed_params = fragment.params
        else:
            n_qubits = 2 * fragment.spec.n_orbitals
            n_layers = self._extra_kwargs.get("n_layers", 1)
            n_params = n_layers * self._ansatz.n_params_per_layer(
                n_qubits,
                n_electrons=n_electrons,
                n_alpha=fragment.spec.n_alpha,
                n_beta=fragment.spec.n_beta,
                **self._extra_kwargs.get("ansatz_kwargs", {}),
            )
            spin_asymmetry = float(np.abs(h_alpha - h_beta).max())
            if spin_asymmetry > _SEED_SPIN_ASYMMETRY_TOL:
                warn(
                    f"CCSD seeding for fragment {fragment.spec.orbitals} averages "
                    f"an embedding potential whose spin channels differ by "
                    f"{spin_asymmetry:.3e} Hartree, because restricted CCSD "
                    "takes one one-body matrix. The seed may sit in a different "
                    "basin than the symmetry-broken solution; the Hamiltonian "
                    "being optimized keeps both channels.",
                    UserWarning,
                    stacklevel=2,
                )
            seed_params = _ccsd_seed_params(
                0.5 * (h_alpha + h_beta),
                g_frag,
                fragment.spec,
                n_params,
                self._ansatz,
            )

        return _FragmentVQE(
            hamiltonian=hamiltonian,
            n_electrons=n_electrons,
            n_alpha=fragment.spec.n_alpha,
            n_beta=fragment.spec.n_beta,
            ansatz=self._ansatz,
            optimizer=copy.deepcopy(self._optimizer),
            max_iterations=self._max_iterations,
            backend=self.backend,
            program_id=program_id,
            progress_queue=self._queue,
            seed=seed,
            seed_params=seed_params,
            **self._extra_kwargs,
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
            RuntimeError: If no programs exist, or if programs haven't
                completed execution.
        """
        super().aggregate_results()
        if self.workflow_state is not None:
            return self.workflow_state
        assert self._state is not None
        return self._state

    def _reset_workflow_state(self) -> None:
        """Clear per-workflow state, also re-seeding ``_rng`` and dropping
        every fragment's cached ``SQDSolver``.

        ``run()`` calls this at the start of every invocation. Without
        re-deriving ``_rng`` from the stored seed and clearing ``_solvers``
        here, a second ``run()`` on the same instance would resume fragment
        0's SQD stream mid-sequence, draw different fragment seeds, and (in
        automatic mode) re-draw the localization restarts from an advanced
        generator instead of reproducing the first run.
        """
        super()._reset_workflow_state()
        self._rng = np.random.default_rng(self._seed)
        self._solvers.clear()
        self._energy_history.clear()
        self._round_reports.clear()
        self._orbitals_converged = True
        if self._seed is not None and self.backend is not None:
            # No-op on backends that cannot seed their sampler, so a run stays
            # reproducible only as far as the backend allows.
            self.backend.set_seed(self._seed)

    def _solver_for(self, index: int, spec: FragmentSpec) -> SQDSolver:
        """Return this fragment's cached ``SQDSolver``, building it once.

        Each fragment gets its own child generator spawned from the
        workflow's seeded RNG, so distinct fragments never share a draw
        sequence and repeated runs under the same ``seed`` stay reproducible.
        Caching avoids rebuilding the solver every round; it does not carry
        any useful state across rounds by itself (``occupancy`` is
        overwritten from that round's own batch results before it is ever
        read again).
        """
        solver = self._solvers.get(index)
        if solver is None:
            solver = SQDSolver(
                spec.n_orbitals,
                spec.n_alpha,
                spec.n_beta,
                n_batches=self._n_batches,
                batch_size=self._batch_size,
                n_iterations=self._n_sqd_iterations,
                lambda_penalty=self._lambda_penalty,
                rng=self._rng.spawn(1)[0],
            )
            self._solvers[index] = solver
        return solver

    def _cached_mol_integrals(self) -> tuple[np.ndarray, np.ndarray]:
        """Return this run's AO-basis integrals, computing them once.

        Both are independent of ``mo_coeff`` and reused unchanged by every
        round's :func:`optimize_orbitals` call, which itself evaluates
        :func:`total_energy` many times per round.
        """
        if self._ao_eri is None or self._h_ao is None:
            self._ao_eri = cached_ao_eri(self._mol)
            self._h_ao = cached_h_ao(self._mol)
        return self._ao_eri, self._h_ao

    def update_state(self, state: LASSQDState) -> LASSQDState:
        """Reduce this round's sampled distributions into the next state.

        For every fragment, converts its program's sampled distribution to
        the blocked SQD bitstring convention, recovers the ground state via
        that fragment's ``SQDSolver``, and rebuilds its spatial RDMs
        from the recovered subspace. The full active-space RDM is then
        reassembled and the molecular orbitals re-optimized against it.

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
            round's optimized total energy), and ``previous_energy`` (set to
            ``state.energy``). ``state`` itself is left unmodified.

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
                "different orbitals than the VQEs optimized against."
            )

        n_occupied = self._mol.nelectron // 2
        n_core = _compute_n_core(
            [fragment.spec for fragment in state.fragments], n_occupied
        )
        ao_eri, _ = self._cached_mol_integrals()
        n_act = sum(fragment.spec.n_orbitals for fragment in state.fragments)
        integrals = transform_integrals(
            self._mol, state.mo_coeff, n_core, n_act, ao_eri
        )

        self._emit_workflow_round_stage("Recovering fragment subspaces (SQD)")
        recovery_started = time.perf_counter()
        programs = self.programs
        new_fragments = []
        subspace_sizes = []
        for index, fragment in enumerate(state.fragments):
            # Same "fragment_{index}" id create_programs() assigned.
            program_id = f"fragment_{index}"
            program = programs[program_id]
            spec = fragment.spec
            h_alpha, h_beta, g_frag = fragment_effective_integrals(
                integrals, state.fragments, index
            )

            probs = next(iter(program.best_probs.values()))
            sqd_probs = probs_to_sqd_bitstrings(probs, spec.n_orbitals)

            solver = self._solver_for(index, spec)
            try:
                result = solver.solve(sqd_probs, h_alpha, g_frag, one_body_beta=h_beta)
            except ValueError as exc:
                raise ValueError(
                    f"SQD failed for {program_id}: {exc} Increase the "
                    "backend shot count or n_sqd_iterations."
                ) from exc

            subspace_sizes.append(len(set(result.subspace)))
            if len(set(result.subspace)) == 1:
                warn(
                    f"{program_id}'s recovered subspace contains only one "
                    "determinant: this round captured no correlation energy "
                    "for this fragment. Use a larger sampling budget "
                    "(n_batches, batch_size) or a more expressive ansatz.",
                    UserWarning,
                    stacklevel=2,
                )

            dets = [
                bitstring_to_spatial_det(bs, spec.n_orbitals) for bs in result.subspace
            ]
            rdm1, rdm2, rdm1_alpha, rdm1_beta = compute_spatial_rdms(
                dets, result.eigenvector, spec.n_orbitals
            )
            new_fragments.append(
                FragmentState(
                    spec=spec,
                    rdm1=rdm1,
                    rdm2=rdm2,
                    params=np.asarray(program.best_params).ravel(),
                    rdm1_alpha=rdm1_alpha,
                    rdm1_beta=rdm1_beta,
                )
            )

        self._emit_workflow_round_stage("Assembling active-space RDMs")
        rdm1_active, rdm2_active = assemble_active_rdms(new_fragments)
        ao_eri, h_ao = self._cached_mol_integrals()

        self._emit_workflow_round_stage("Re-optimizing orbitals")
        orbital_started = time.perf_counter()
        solve = optimize_orbitals(
            self._mol,
            state.mo_coeff,
            n_core,
            [fragment.spec for fragment in new_fragments],
            rdm1_active,
            rdm2_active,
            ao_eri,
            h_ao,
            max_orbital_iterations=self._max_orbital_iterations,
        )
        self._orbitals_converged = solve.converged
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
        self._emit_workflow_round_stage(self._round_reports[-1].summary(), final=True)

        return LASSQDState(
            mo_coeff=solve.mo_coeff,
            fragments=tuple(new_fragments),
            energy=solve.energy,
            previous_energy=state.energy,
        )

    def is_complete(self, state: LASSQDState) -> bool:
        """Stop once the macro-cycle energy change falls below ``energy_tol``.

        A round whose orbital optimization gave up does not count as converged,
        however small its energy change. ``optimize_orbitals`` is monotone -- it
        falls back to the unrotated orbitals rather than returning something
        worse -- so a stalled optimizer produces a round that barely moves and
        is otherwise indistinguishable from a real fixed point. Requiring the
        inner solve to have converged is what separates the two.
        """
        if not abs(state.energy - state.previous_energy) < self._energy_tol:
            return False
        if not self._orbitals_converged:
            warn(
                "The macro-cycle energy change is below energy_tol but the "
                "orbital optimization did not converge, so this is not a fixed "
                "point. Continuing; raise the optimizer's iteration budget or "
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
        below an exact reference on the same active space. Fragmenting still
        costs accuracy -- see :ref:`lassqd-accuracy-characteristics`.

        The macro-cycle is not guaranteed monotone, so a later round can report
        a higher energy than an earlier one; ``energy_history`` records each, and
        ``best_energy`` gives the lowest.
        """
        if self.workflow_state is None:
            return float("inf")
        return self.workflow_state.energy
