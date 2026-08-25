# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""The LASSQD program ensemble: construction and per-round program creation."""

import copy
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from warnings import warn

import numpy as np
from qiskit.quantum_info import Statevector
from scipy.linalg import expm

from divi.hamiltonians._chem import _spo_from_integrals
from divi.qprog.algorithms import LUCJAnsatz, UCCSDAnsatz
from divi.qprog.algorithms._ansatze import (
    Ansatz,
    _uccsd_excitations,
    lucj_jastrow_pairs,
    n_rotation_params,
    rotation_angles,
)
from divi.qprog.algorithms._vqe import VQE
from divi.qprog.ensemble import ProgramEnsemble, ReportingLevel
from divi.qprog.optimizers import Optimizer

from ._active_space import (
    auto_fragment_specs,
    split_active_orbitals,
)
from ._active_space import validate_fragment_atoms as _validate_fragment_atoms
from ._config import FragmentationConfig, SQDConfig
from ._integrals import (
    MOIntegrals,
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
    compute_spatial_rdms,
    probs_to_sqd_bitstrings,
)
from ._state import FragmentSpec, FragmentState, LASSQDState, validate_fragment_specs

try:
    # optional ``chem`` extra
    from pyscf import cc
    from pyscf.cc import addons as cc_addons
except ImportError:
    cc = None
    cc_addons = None


# Below this the two spin channels of an embedding potential are the same matrix
# to seeding precision, so averaging them for the seed costs nothing.
_SEED_SPIN_ASYMMETRY_TOL = 1e-6

# A seed must beat the reference determinant by at least this much, in Hartree, to
# be worth using. Accepted seeds clear it by ~1e-2; a seed built in the wrong
# orbital basis lands ~3e-3 above the reference, so zero is not a safe boundary.
_SEED_ACCEPTANCE_MARGIN = 1e-4

# Coupled-cluster iterations allowed on a fragment. pyscf's default of 50 leaves
# a localised fragment short of convergence where a few hundred reach it, and the
# fragments are small enough that the extra cycles cost nothing.
_SEED_CC_MAX_CYCLE = 500

# Widest fragment whose seed is checked exactly. 20 qubits is 16 MB of amplitudes
# and covers a ten-orbital fragment, the largest either SQD paper runs.
_SEED_CHECK_MAX_QUBITS = 20


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
            f"|g|max {self.orbital_gradient_norm:.2e}, "
            f"converged={self.orbital_converged}; "
            f"SQD {self.recovery_seconds:.1f}s, orbitals {self.orbital_seconds:.1f}s"
        )


class _FragmentVQE(VQE):
    """A fragment VQE whose fresh parameters can be supplied by the workflow.

    Accepts an explicit ``seed_params`` vector and returns it from
    ``_initialize_param_sets`` instead of the optimizer's own random
    initialisation.

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


def _one_body_from_excitations(
    block: np.ndarray, n_orb: int
) -> tuple[np.ndarray, np.ndarray]:
    """Diagonalize the one-body operator an occupied-virtual block defines.

    Returns its eigenbasis, with the column signs chosen so the basis is a
    rotation: an eigenbasis is only defined up to those signs, and a determinant
    of ``-1`` is a reflection that no product of rotations can realize.
    """
    n_occupied, n_virtual = block.shape
    one_body = np.zeros((n_orb, n_orb))
    one_body[:n_occupied, n_occupied : n_occupied + n_virtual] = block
    one_body[n_occupied : n_occupied + n_virtual, :n_occupied] = block.T
    eigenvalues, eigenvectors = np.linalg.eigh(one_body)
    if np.linalg.det(eigenvectors) < 0:
        eigenvectors[:, 0] *= -1
    return eigenvectors, eigenvalues


def _lucj_amplitude_seed(
    coupled_cluster, spec: FragmentSpec, n_params: int, ansatz_kwargs: Mapping
) -> np.ndarray | None:
    """Map unrestricted CCSD amplitudes onto :class:`LUCJAnsatz`'s parameters.

    LUCJ's parameters are rotation and Coulomb angles, not amplitudes, so the
    correspondence runs through the double factorization. The opposite-spin
    doubles carry it: reshaped over ``(occupied, virtual)`` pairs, their leading
    singular triplet gives one one-body operator per spin, and the square of a
    one-body operator is a diagonal Coulomb operator in the basis that
    diagonalises it -- exactly ``exp(K) exp(iJ) exp(-K)``'s content. So each
    spin's eigenbasis is its rotation and the eigenvalues give
    ``J_pq = sigma * d_p * d_q``. ``t1`` supplies the trailing rotation.

    Opposite-spin rather than same-spin because that is what the layer's on-site
    Coulomb term represents, and because a fragment holding one electron of a
    given spin has no same-spin double excitation at all -- its ``t2`` block is
    identically zero and would seed a bare Hartree-Fock determinant.

    Approximate in three ways, all of which leave it a starting point rather than
    an encoding of CCSD: one Jastrow layer holds only the leading factorization
    term; ``J`` is projected onto the layer's own pair pattern; and the emitted
    ``RZZ`` gates carry the pair term of ``exp(i J n_p n_q)`` but not its
    one-body remainder.

    Returns ``None`` if either spin sector's rotation cannot be realized, or if
    the ansatz ties the two sectors together (``shared_spin_params``), which a
    factorisation giving each its own rotation has nothing to say about.
    """
    if ansatz_kwargs.get("shared_spin_params"):
        warn(
            f"CCSD seeding skipped for fragment {spec.orbitals}: the double "
            "factorisation gives each spin sector its own rotation, which "
            "shared_spin_params cannot hold. Falling back to the optimizer's own "
            "initialisation.",
            UserWarning,
            stacklevel=2,
        )
        return None

    n_orb = spec.n_orbitals
    depth = ansatz_kwargs.get("rotation_depth")
    same_pairs, opposite_pairs = lucj_jastrow_pairs(
        n_orb,
        ansatz_kwargs.get("same_spin_pairs"),
        ansatz_kwargs.get("opposite_spin_pairs"),
    )
    t1_alpha, t1_beta = coupled_cluster.t1
    t2_opposite = np.asarray(coupled_cluster.t2[1])

    n_occupied_alpha, n_occupied_beta, n_virtual_alpha, n_virtual_beta = (
        t2_opposite.shape
    )
    if min(t2_opposite.shape) == 0:
        # All-zero parameters are an exactly stationary point -- the Jastrow
        # generators annihilate the reference and the rotations cancel -- so a
        # gradient optimizer seeded there could not move. Random beats it.
        return None

    # Rectangular whenever the spin sectors differ in size, so a singular value
    # decomposition rather than an eigendecomposition.
    matrix = t2_opposite.transpose(0, 2, 1, 3).reshape(
        n_occupied_alpha * n_virtual_alpha, n_occupied_beta * n_virtual_beta
    )
    left, singular_values, right = np.linalg.svd(matrix)
    scale = float(singular_values[0])

    rotation_alpha, diagonal_alpha = _one_body_from_excitations(
        left[:, 0].reshape(n_occupied_alpha, n_virtual_alpha), n_orb
    )
    rotation_beta, diagonal_beta = _one_body_from_excitations(
        right[0, :].reshape(n_occupied_beta, n_virtual_beta), n_orb
    )
    # A real doubles amplitude needs the Jastrow's factor of i absorbed into a
    # rotation, or the first-order energy correction is imaginary and cancels.
    # One sector taking the anti-Hermitian embedding supplies it: the same
    # rotation times -i on its virtual orbitals.
    phase = np.ones(n_orb, dtype=complex)
    phase[n_occupied_beta:] = -1j
    rotation_beta = rotation_beta.astype(complex) * phase[:, None]

    seed = np.zeros(n_params)
    cursor = 0
    sandwiched = n_rotation_params(n_orb, orbital_phases=False, depth=depth)
    for rotation in (rotation_alpha, rotation_beta):
        # The block runs first, in exp(-K), so it realizes the eigenbasis's
        # inverse. Conjugate transpose, not transpose: with the -i above, the two
        # are not the same and each sign is only right alongside the other.
        angles = rotation_angles(rotation.conj().T, depth=depth)
        if angles is None:
            return None
        # The fit's per-orbital phases are dropped; the sandwich cancels them.
        seed[cursor : cursor + sandwiched] = angles[:sandwiched]
        cursor += sandwiched

    # exp(i J n_p n_q) contributes exp(i J Z_p Z_q / 4), which is RZZ(-J / 2).
    for p, q in opposite_pairs:
        seed[cursor] = -0.5 * scale * diagonal_alpha[p] * diagonal_beta[q]
        cursor += 1

    # The factorisation holds only the cross term between the two sectors, so it
    # says nothing about same-spin Coulomb weights; those stay at zero.
    cursor += 2 * len(same_pairs)

    if ansatz_kwargs.get("trailing_rotation"):
        for t1_block in (t1_alpha, t1_beta):
            generator = np.zeros((n_orb, n_orb))
            n_occupied, n_virtual = t1_block.shape
            generator[:n_occupied, n_occupied : n_occupied + n_virtual] = -t1_block
            generator -= generator.T
            angles = rotation_angles(expm(generator), depth=depth)
            if angles is None:
                # Only this block is lost. Returning None here would discard the
                # rotations and Jastrow too, and the fit fails most often for a
                # near-identity target -- exactly when t1 is small and the rest
                # of the seed is at its most useful.
                warn(
                    f"CCSD seeding for fragment {spec.orbitals} could not realize "
                    "the trailing rotation; seeding the rest and leaving it at "
                    "the identity.",
                    UserWarning,
                    stacklevel=2,
                )
                angles = np.zeros(
                    n_rotation_params(n_orb, orbital_phases=True, depth=depth)
                )
            seed[cursor : cursor + len(angles)] = angles
            cursor += len(angles)

    return seed


def _embedded_mean_field(
    scf_method: str,
    h_eff: np.ndarray,
    g_frag: np.ndarray,
    spec: FragmentSpec,
    mo_coeff: np.ndarray,
    occupations: np.ndarray,
):
    """A pyscf mean field carrying the fragment's integrals and reference.

    The molecule is a shell -- the integrals are supplied directly, so the only
    real inputs are the electron count and spin. Orbital energies come from the
    Fock diagonal in the given basis, which need not diagonalize it; coupled
    cluster is solved non-canonically either way.
    """
    # optional ``chem`` extra
    from pyscf import ao2mo, gto, scf

    n_orb = spec.n_orbitals
    fake_mol = gto.M(verbose=0)
    fake_mol.nelectron = spec.n_alpha + spec.n_beta
    fake_mol.spin = spec.n_alpha - spec.n_beta
    fake_mol.incore_anyway = True

    mean_field = getattr(scf, scf_method)(fake_mol)
    # overriding with fragment integrals
    mean_field.get_hcore = lambda *args: h_eff
    # overriding with fragment integrals
    mean_field.get_ovlp = lambda *args: np.eye(n_orb)
    mean_field._eri = ao2mo.restore(8, g_frag, n_orb)
    mean_field.mo_coeff = mo_coeff
    mean_field.mo_occ = occupations

    # resolved on the pyscf mean-field at runtime
    density = mean_field.make_rdm1()
    fock = np.asarray(mean_field.get_fock(dm=density))
    mean_field.mo_energy = (
        np.array([np.diag(fock[0]), np.diag(fock[1])])
        if fock.ndim == 3
        else np.diag(fock)
    )
    mean_field.e_tot = mean_field.energy_tot(dm=density)
    mean_field.converged = True
    return mean_field


def _lucj_seed_params(
    h_eff: np.ndarray,
    g_frag: np.ndarray,
    spec: FragmentSpec,
    n_params: int,
    ansatz_kwargs: Mapping,
) -> np.ndarray | None:
    """Run unrestricted CCSD on the fragment and factorize it onto LUCJ.

    Unrestricted rather than restricted because these fragments are routinely
    spin-polarised, which a restricted reference cannot represent at all. The
    one-body potential is still spin-averaged, so the reference is polarised only
    through its occupations.
    """
    try:
        n_orb = spec.n_orbitals
        occupations = np.zeros((2, n_orb))
        occupations[0, : spec.n_alpha] = 1.0
        occupations[1, : spec.n_beta] = 1.0
        mean_field = _embedded_mean_field(
            "UHF",
            h_eff,
            g_frag,
            spec,
            np.array([np.eye(n_orb), np.eye(n_orb)]),
            occupations,
        )

        # pyrefly: ignore[missing-attribute]  # cc is None only if pyscf is absent
        coupled_cluster = cc.UCCSD(mean_field)
        coupled_cluster.max_cycle = _SEED_CC_MAX_CYCLE
        coupled_cluster.kernel()
        if not coupled_cluster.converged:
            warn(
                f"UCCSD did not converge for fragment {spec.orbitals}; seeding "
                "from its amplitudes anyway, since the seed is accepted on the "
                "energy it delivers rather than on the solver's own criterion.",
                UserWarning,
                stacklevel=2,
            )
    except Exception as exc:
        warn(
            f"CCSD seeding failed for fragment {spec.orbitals}: {exc}. "
            "Falling back to the optimizer's own initialisation.",
            UserWarning,
            stacklevel=2,
        )
        return None

    return _lucj_amplitude_seed(coupled_cluster, spec, n_params, ansatz_kwargs)


def _seed_energy_gain(
    seed: np.ndarray,
    hamiltonian,
    ansatz: Ansatz,
    n_qubits: int,
    n_layers: int,
    build_kwargs: Mapping,
) -> float | None:
    """How far below the reference determinant the seed sits, in Hartree.

    Positive means the seed is an improvement. ``None`` means the fragment is too
    wide to check exactly, leaving the caller to accept the seed unchecked.

    Replaces a Hartree-Fock stationarity precondition, which rejected even an
    exact open-shell solution -- no single spatial basis makes both spin channels
    of a polarised fragment stationary -- and could not catch a seed built in the
    wrong orbital basis, since ``F_ov`` transforms as ``U_o^T F_ov U_v`` and so is
    invariant under exactly the rotation that misattaches amplitudes. Such a seed
    lands *above* the reference determinant, which this measures directly.

    All-zero parameters realize the reference determinant exactly, so it is both
    the baseline and what the caller falls back to.
    """
    if n_qubits > _SEED_CHECK_MAX_QUBITS:
        return None

    def energy(params: np.ndarray) -> float:
        circuit = ansatz.build(params, n_qubits, n_layers, **build_kwargs)
        state = Statevector.from_instruction(circuit)
        return float(np.real(state.expectation_value(hamiltonian)))

    return energy(np.zeros_like(seed)) - energy(seed)


def _ccsd_seed_params(
    h_eff: np.ndarray,
    g_frag: np.ndarray,
    spec: FragmentSpec,
    n_params: int,
    ansatz: Ansatz | None = None,
    ansatz_kwargs: Mapping | None = None,
) -> np.ndarray | None:
    """Map a fragment's CCSD amplitudes onto an ansatz parameter vector.

    Optimisation started from random parameters converges poorly, and SQD's
    subspace quality depends directly on the sampled distribution covering
    the right determinants, so a fresh fragment's first round is seeded from
    coupled-cluster amplitudes computed on that fragment's own effective
    integrals, instead of starting from the optimizer's random initial guess.

    Two ansaetze have a correspondence, by different routes.
    :class:`~divi.qprog.algorithms.UCCSDAnsatz`'s parameters *are*
    singles-and-doubles amplitudes, so :func:`_uccsd_amplitude_seed` reads each
    off the matching entry of ``t1``/``t2``.
    :class:`~divi.qprog.algorithms.LUCJAnsatz`'s are rotation and Coulomb angles
    instead, so :func:`_lucj_seed_params` goes through the doubles tensor's
    double factorization. Any other ansatz warns and defers to the optimizer's
    own initialisation.

    The CCSD runs in the fragment's own orbital basis, on the determinant the
    ansatz's Hartree-Fock reference prepares, rather than on a self-consistent
    field's canonical orbitals. Fragment orbitals are localised, and an SCF
    would rotate within the occupied and virtual blocks -- leaving the reference
    determinant and its energy untouched while permuting which amplitude belongs
    to which orbital pair, so the resulting seed would be attached to the wrong
    excitations. That determinant need not be Hartree-Fock stationary -- coupled
    cluster is solved non-canonically, so a non-stationary reference inflates
    ``t1`` rather than misattaching anything. :func:`_seed_energy_gain` is what
    rejects a seed that went wrong.

    Args:
        h_eff: Fragment's effective one-body integrals, shape
            ``(n_orbitals, n_orbitals)``.
        g_frag: Fragment's bare two-body integrals, shape
            ``(n_orbitals,) * 4``.
        spec: Fragment specification.
        n_params: Length of the returned vector.
        ansatz: The fragment's configured ansatz.
        ansatz_kwargs: The keywords the ansatz is built with, which fix the
            parameter layout the seed has to fill.

    Returns:
        A length-``n_params`` vector, or ``None`` (with a ``UserWarning``) if
        ``ansatz`` is neither a ``UCCSDAnsatz`` nor a ``LUCJAnsatz``, or if
        the coupled-cluster calculation raises. Non-convergence only warns, since
        the seed is judged on the energy it delivers. Restricted
        CCSD additionally cannot represent a spin-imbalanced fragment
        (``n_alpha != n_beta``), so ``UCCSDAnsatz`` also returns ``None`` there;
        the LUCJ path uses unrestricted CCSD and has no such limit.
    """
    if not isinstance(ansatz, (UCCSDAnsatz, LUCJAnsatz)):
        warn(
            f"CCSD seeding skipped for fragment {spec.orbitals}: no "
            f"correspondence is defined between CCSD amplitudes and "
            f"{type(ansatz).__name__}'s parameters. Falling back to the "
            "optimizer's own initialisation.",
            UserWarning,
            stacklevel=2,
        )
        return None

    if isinstance(ansatz, LUCJAnsatz):
        return _lucj_seed_params(h_eff, g_frag, spec, n_params, ansatz_kwargs or {})

    if spec.n_alpha != spec.n_beta:
        warn(
            f"CCSD seeding skipped for fragment {spec.orbitals}: restricted "
            f"CCSD requires equal alpha/beta electron counts, got n_alpha="
            f"{spec.n_alpha}, n_beta={spec.n_beta}. Falling back to the "
            "optimizer's own initialisation.",
            UserWarning,
            stacklevel=2,
        )
        return None

    try:
        n_orb = spec.n_orbitals
        occupations = np.zeros(n_orb)
        occupations[: spec.n_alpha] = 2.0
        mean_field = _embedded_mean_field(
            "RHF", h_eff, g_frag, spec, np.eye(n_orb), occupations
        )

        # pyrefly: ignore[missing-attribute]  # cc is None only if pyscf is absent
        coupled_cluster = cc.CCSD(mean_field)
        coupled_cluster.max_cycle = _SEED_CC_MAX_CYCLE
        coupled_cluster.kernel()
        if not coupled_cluster.converged:
            warn(
                f"CCSD did not converge for fragment {spec.orbitals}; seeding "
                "from its amplitudes anyway, since the seed is accepted on the "
                "energy it delivers rather than on the solver's own criterion.",
                UserWarning,
                stacklevel=2,
            )
    except Exception as exc:
        warn(
            f"CCSD seeding failed for fragment {spec.orbitals}: {exc}. "
            "Falling back to the optimizer's own initialisation.",
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
    """Localised active-space sample-based quantum diagonalisation.

    Partitions a molecule's active space into fragments, runs one VQE per
    fragment against its own mean-field-embedded effective Hamiltonian, and
    (in later rounds) recovers the ground state via sample-based quantum
    diagonalisation. This class builds the workflow state and the per-round
    VQE programs; running rounds and aggregating results are handled
    elsewhere.

    :attr:`energy` is a variational upper bound -- the assembled RDM is that of
    a product of fragment states, so the energy is a genuine expectation value.
    Fragmenting nonetheless costs accuracy, and the cost grows with how
    strongly the fragments interact; see
    :ref:`lassqd-accuracy-characteristics` in the LASSQD guide.

    Args:
        molecule: A PySCF ``gto.Mole`` (an RHF calculation is run on it lazily,
            in :meth:`initial_state`) or a restricted mean-field object —
            not a PennyLane ``qchem.Molecule``. Closed-shell (RHF) only.
        optimizer: Optimizer template, deep-copied for each fragment's VQE.
        fragmentation: Which orbitals are active and how they split into
            fragments, as a
            :class:`~divi.qprog.workflows.FragmentationConfig`.
        sqd: Sampling and diagonalisation budget per fragment solve, as an
            :class:`~divi.qprog.workflows.SQDConfig`. Defaults to
            ``SQDConfig()``.
        ansatz: Per-fragment ansatz. Defaults to ``UCCSDAnsatz()``.
        max_iterations: Max optimisation iterations per fragment VQE. An
            iteration is an optimizer step, not a circuit evaluation: a
            gradient-free method spends several evaluations per step and
            ``n_params + 1`` of them building its initial simplex before the
            first step, so this has to scale with the ansatz's parameter count.
        max_orbital_iterations: Cap on L-BFGS-B iterations in each round's
            orbital re-optimisation, a separate solve from the fragment VQEs and
            usually the round's dominant cost on a large register. ``None``
            leaves it uncapped. A capped round still returns its best orbitals
            but is not a stationary point, and reports as not converged.
        energy_tol: Macro-cycle stops once consecutive rounds' total energies
            differ by less than this (Hartree).
        seed: Seed for fragmentation, localisation, and SQD subsampling, also
            passed to the backend. Reproducibility is limited by the backend:
            :class:`~divi.backends.QiskitSimulator` seeds exactly, while
            :class:`~divi.backends.MaestroSimulator` cannot, so identical runs
            are not guaranteed to agree bit for bit there.
        **kwargs: ``backend`` (required), ``sampling_backend``, and
            ``reporting_level`` are consumed here; ``program_id`` and
            ``progress_queue`` are set internally and must not be passed here.
            Any other keyword is forwarded verbatim to every fragment's
            :class:`~divi.qprog.algorithms.VQE`, e.g. ``grouping_strategy``,
            ``shot_distribution``, ``precision``, or ``early_stopping``.

    Raises:
        ValueError: If ``fragmentation``'s ``active_orbitals`` has out-of-range
            indices or no occupied or no virtual orbital; if its
            ``fragment_atoms`` names an out-of-range atom or shares one between
            fragments; if ``max_iterations`` is below 1; if
            ``max_orbital_iterations`` is given and below 1; if ``energy_tol``
            is not positive; or if any fragment leaves no excitation available,
            fragments overlap, or the fragments do not sum to ``Sz = 0``. The
            configuration objects validate their own fields on construction.
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
        fragmentation: FragmentationConfig,
        sqd: SQDConfig | None = None,
        ansatz: Ansatz | None = None,
        max_iterations: int = 10,
        max_orbital_iterations: int | None = None,
        energy_tol: float = 1e-6,
        seed: int | None = None,
        **kwargs,
    ):
        if max_iterations < 1:
            raise ValueError(
                f"max_iterations must be at least 1; got {max_iterations}."
            )
        if max_orbital_iterations is not None and max_orbital_iterations < 1:
            raise ValueError(
                "max_orbital_iterations must be at least 1 when given; got "
                f"{max_orbital_iterations}."
            )
        if energy_tol <= 0:
            raise ValueError(f"energy_tol must be positive; got {energy_tol}.")
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
            sampling_backend=kwargs.pop("sampling_backend", None),
            reporting_level=kwargs.pop("reporting_level", ReportingLevel.COMPACT),
        )

        try:
            # optional ``chem`` extra
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

        # Validate the caller's orbital choices against this molecule here
        # rather than in ``initial_state``, so an out-of-range index fails at
        # construction.
        n_orbitals_total = self._mol.nao_nr()
        n_occupied = self._mol.nelectron // 2
        if fragmentation.active_spaces is not None:
            validate_fragment_specs(
                fragmentation.active_spaces, n_orbitals_total, n_occupied
            )
        if fragmentation.active_orbitals is not None:
            split_active_orbitals(
                fragmentation.active_orbitals, n_occupied, n_orbitals_total
            )
        if fragmentation.fragment_atoms is not None:
            _validate_fragment_atoms(fragmentation.fragment_atoms, self._mol.natm)

        self._fragmentation = fragmentation
        self._sqd = SQDConfig() if sqd is None else sqd
        self._ansatz: Ansatz = UCCSDAnsatz() if ansatz is None else ansatz
        self._optimizer = optimizer
        self._max_iterations = max_iterations
        self._max_orbital_iterations = max_orbital_iterations
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
        # optional ``chem`` extra
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

        integrals, _n_core = self._active_space_integrals(state)
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

        Seeding takes a single one-body matrix, so it gets the spin-averaged
        embedding potential. That only affects the optimizer's starting point,
        not the Hamiltonian it optimises against, which carries both channels --
        but a spin-symmetric seed can still land a local optimizer in a
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
            ansatz_kwargs = self._extra_kwargs.get("ansatz_kwargs", {})
            n_params = n_layers * self._ansatz.n_params_per_layer(
                n_qubits,
                n_electrons=n_electrons,
                n_alpha=fragment.spec.n_alpha,
                n_beta=fragment.spec.n_beta,
                **ansatz_kwargs,
            )
            spin_asymmetry = float(np.abs(h_alpha - h_beta).max())
            if spin_asymmetry > _SEED_SPIN_ASYMMETRY_TOL:
                warn(
                    f"CCSD seeding for fragment {fragment.spec.orbitals} averages "
                    f"an embedding potential whose spin channels differ by "
                    f"{spin_asymmetry:.3e} Hartree, because seeding takes a "
                    "single one-body matrix. The seed may sit in a different "
                    "basin than the symmetry-broken solution; the Hamiltonian "
                    "being optimised keeps both channels.",
                    UserWarning,
                    stacklevel=2,
                )
            seed_params = _ccsd_seed_params(
                0.5 * (h_alpha + h_beta),
                g_frag,
                fragment.spec,
                n_params,
                self._ansatz,
                ansatz_kwargs,
            )
            if seed_params is not None:
                gain = _seed_energy_gain(
                    seed_params,
                    hamiltonian,
                    self._ansatz,
                    n_qubits,
                    n_layers,
                    {
                        "n_electrons": n_electrons,
                        "n_alpha": fragment.spec.n_alpha,
                        "n_beta": fragment.spec.n_beta,
                        **ansatz_kwargs,
                    },
                )
                if gain is not None and gain < _SEED_ACCEPTANCE_MARGIN:
                    warn(
                        f"CCSD seeding rejected for fragment "
                        f"{fragment.spec.orbitals}: the seed sits "
                        f"{-gain:+.3e} Hartree relative to the reference "
                        "determinant, so it carries no correlation energy. "
                        "Falling back to the optimizer's own initialisation.",
                        UserWarning,
                        stacklevel=2,
                    )
                    seed_params = None

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
        automatic mode) re-draw the localisation restarts from an advanced
        generator instead of reproducing the first run.
        """
        super()._reset_workflow_state()
        self._rng = np.random.default_rng(self._seed)
        self._solvers.clear()
        self._energy_history.clear()
        self._round_reports.clear()
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
        read again, and carryover is scoped to one ``solve`` call because a
        retained determinant is only meaningful in the orbital basis it was
        found in).
        """
        solver = self._solvers.get(index)
        if solver is None:
            solver = SQDSolver(
                spec.n_orbitals,
                spec.n_alpha,
                spec.n_beta,
                n_batches=self._sqd.n_batches,
                batch_size=self._sqd.batch_size,
                n_iterations=self._sqd.n_recovery_iterations,
                lambda_penalty=self._sqd.lambda_penalty,
                carryover_cutoff=self._sqd.carryover_cutoff,
                max_carryover=self._sqd.max_carryover,
                max_dim=self._sqd.max_dim,
                include_reference=self._sqd.include_reference,
                symmetrize_spin=self._sqd.symmetrize_spin,
                energy_tol=self._sqd.recovery_energy_tol,
                occupancies_tol=self._sqd.recovery_occupancies_tol,
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
        if self._ao_eri is None or self._h_ao is None:
            self._ao_eri = cached_ao_eri(self._mol)
            self._h_ao = cached_h_ao(self._mol)
        return self._ao_eri, self._h_ao

    def _active_space_integrals(self, state: LASSQDState) -> tuple[MOIntegrals, int]:
        """This state's active-space integrals and its frozen-core count."""
        n_occupied = self._mol.nelectron // 2
        n_core = _compute_n_core(
            [fragment.spec for fragment in state.fragments], n_occupied
        )
        ao_eri, _ = self._cached_mol_integrals()
        n_act = sum(fragment.spec.n_orbitals for fragment in state.fragments)
        integrals = transform_integrals(
            self._mol, state.mo_coeff, n_core, n_act, ao_eri
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
                "different orbitals than the VQEs optimised against."
            )

        integrals, n_core = self._active_space_integrals(state)

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
                    "backend shot count or n_recovery_iterations."
                ) from exc

            subspace_size = int(result.amplitudes.size)
            subspace_sizes.append(subspace_size)
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

        self._emit_workflow_round_stage("Re-optimising orbitals")
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
            orbitals_converged=solve.converged,
        )

    def is_complete(self, state: LASSQDState) -> bool:
        """Stop once the macro-cycle energy change falls below ``energy_tol``.

        A round whose orbital optimisation gave up does not count as converged,
        however small its energy change. ``optimize_orbitals`` is monotone -- it
        falls back to the unrotated orbitals rather than returning something
        worse -- so a stalled optimizer produces a round that barely moves and
        is otherwise indistinguishable from a real fixed point. Requiring the
        inner solve to have converged is what separates the two.
        """
        if not abs(state.energy - state.previous_energy) < self._energy_tol:
            return False
        if not state.orbitals_converged:
            warn(
                "The macro-cycle energy change is below energy_tol but the "
                "orbital optimisation did not converge, so this is not a fixed "
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
