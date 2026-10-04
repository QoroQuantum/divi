# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""VQE fragment preparation: the fragment VQE program and its CCSD seed."""

import copy
from collections.abc import Mapping
from typing import Any
from warnings import warn

import numpy as np
from pyscf import ao2mo, cc, gto, scf
from pyscf.cc import addons as cc_addons
from qiskit.quantum_info import Statevector
from scipy.linalg import expm

from divi.backends import CircuitRunner
from divi.qprog.algorithms import LUCJAnsatz, UCCSDAnsatz
from divi.qprog.algorithms._ansatze import (
    Ansatz,
    _uccsd_excitations,
    lucj_jastrow_pairs,
    n_rotation_params,
    rotation_angles,
)
from divi.qprog.algorithms._vqe import VQE
from divi.qprog.optimizers import Optimizer
from divi.qprog.problems import MolecularProblem

from ._state import FragmentSpec, FragmentState

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


class _FragmentVQE(VQE):
    """A fragment VQE whose fresh parameters can be supplied by the workflow.

    Accepts an explicit ``seed_params`` vector and returns it from
    ``_initialize_param_sets`` instead of the optimizer's own random
    initialisation.

    Raises:
        ValueError: If ``seed_params`` is given and its length does not match
            this VQE's parameter count.
    """

    def __init__(
        self,
        problem: MolecularProblem,
        *,
        seed_params: np.ndarray | None = None,
        **kwargs,
    ):
        super().__init__(problem, **kwargs)
        self._fragment_problem = problem
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

    @property
    def h_alpha(self) -> np.ndarray:
        """Alpha one-body integrals the fragment was sampled under."""
        return self._fragment_problem.one_body

    @property
    def h_beta(self) -> np.ndarray:
        """Beta one-body integrals the fragment was sampled under."""
        return self._fragment_problem.one_body_beta

    @property
    def two_body(self) -> np.ndarray:
        """Two-body integrals the fragment was sampled under."""
        return self._fragment_problem.two_body

    @property
    def orbital_rotation(self) -> np.ndarray:
        """Identity: a fragment VQE samples in the fragment's own orbitals."""
        return np.eye(self._fragment_problem.n_orbitals)


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
    t1_full = cc_addons.spatial2spin(coupled_cluster.t1)
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
    """Diagonalise the one-body operator an occupied-virtual block defines.

    Returns ``(eigenvectors, eigenvalues)``.
    """
    n_occupied, n_virtual = block.shape
    one_body = np.zeros((n_orb, n_orb))
    one_body[:n_occupied, n_occupied : n_occupied + n_virtual] = block
    one_body[n_occupied : n_occupied + n_virtual, :n_occupied] = block.T
    eigenvalues, eigenvectors = np.linalg.eigh(one_body)
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
    h_eff: np.ndarray,
    g_frag: np.ndarray,
    spec: FragmentSpec,
    mo_coeff: np.ndarray,
    occupations: np.ndarray,
    *,
    unrestricted: bool,
):
    """A pyscf mean field carrying the fragment's integrals and reference.

    ``unrestricted`` selects UHF over RHF.

    The molecule is a shell -- the integrals are supplied directly, so the only
    real input is the spin. Coupled cluster builds its own Fock matrix from
    ``mo_coeff`` and ``mo_occ`` and is solved non-canonically, so no orbital
    energies or SCF energy are set.
    """

    n_orb = spec.n_orbitals
    fake_mol = gto.M(verbose=0)
    fake_mol.spin = spec.n_alpha - spec.n_beta
    fake_mol.incore_anyway = True

    mean_field: Any = (scf.UHF if unrestricted else scf.RHF)(fake_mol)
    mean_field.get_hcore = lambda *args: h_eff
    mean_field._eri = ao2mo.restore(8, g_frag, n_orb)
    mean_field.mo_coeff = mo_coeff
    mean_field.mo_occ = occupations
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
            h_eff,
            g_frag,
            spec,
            np.array([np.eye(n_orb), np.eye(n_orb)]),
            occupations,
            unrestricted=True,
        )

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
            h_eff, g_frag, spec, np.eye(n_orb), occupations, unrestricted=False
        )

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


def build_fragment_vqe(
    problem: MolecularProblem,
    fragment: FragmentState,
    *,
    ansatz: Ansatz,
    optimizer: Optimizer,
    max_iterations: int,
    backend: CircuitRunner,
    seed: int,
    options: Mapping[str, Any],
) -> _FragmentVQE:
    """Build one fragment's VQE from its embedded problem.

    A fresh fragment (``fragment.params is None``) is seeded from its own CCSD
    amplitudes via :func:`_ccsd_seed_params`; a fragment warm-started from a
    previous round uses ``fragment.params`` directly and never calls CCSD.

    Seeding takes a single one-body matrix, so it gets the spin-averaged
    embedding potential. That only affects the optimizer's starting point, not
    the Hamiltonian it optimises against, which carries both channels -- but a
    spin-symmetric seed can still land a local optimizer in a different basin
    than the symmetry-broken solution, so a materially asymmetric embedding is
    warned about.
    """
    h_alpha, h_beta = problem.one_body, problem.one_body_beta
    n_electrons = problem.n_electrons

    if fragment.params is not None:
        seed_params = fragment.params
    else:
        n_qubits = 2 * problem.n_orbitals
        n_layers = options.get("n_layers", 1)
        ansatz_kwargs = options.get("ansatz_kwargs", {})
        n_params = n_layers * ansatz.n_params_per_layer(
            n_qubits,
            n_electrons=n_electrons,
            n_alpha=problem.n_alpha,
            n_beta=problem.n_beta,
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
            problem.two_body,
            fragment.spec,
            n_params,
            ansatz,
            ansatz_kwargs,
        )
        if seed_params is not None:
            gain = _seed_energy_gain(
                seed_params,
                problem.hamiltonian,
                ansatz,
                n_qubits,
                n_layers,
                {
                    "n_electrons": n_electrons,
                    "n_alpha": problem.n_alpha,
                    "n_beta": problem.n_beta,
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
        problem,
        ansatz=ansatz,
        optimizer=copy.deepcopy(optimizer),
        max_iterations=max_iterations,
        backend=backend,
        seed=seed,
        seed_params=seed_params,
        **options,
    )
