# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""VQE fragment preparation: the fragment VQE program and its CCSD seed."""

import copy
from collections.abc import Mapping
from contextlib import closing
from typing import Any
from warnings import warn

import numpy as np
from pyscf import cc, scf
from pyscf.cc import addons as cc_addons
from scipy.linalg import expm

from divi.backends import CircuitRunner, MaestroConfig, MaestroSimulator
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

from ._preparation import fragment_mean_field
from ._state import FragmentSpec, FragmentState

# Below this the embedding potential's spin channels are the same to seeding precision.
_SEED_SPIN_ASYMMETRY_TOL = 1e-6

# Minimum energy gain over the reference determinant, in Hartree, for a seed to be used.
_SEED_ACCEPTANCE_MARGIN = 1e-4

# Coupled-cluster iterations allowed on a fragment.
_SEED_CC_MAX_CYCLE = 500

# What a coupled-cluster solve raises on a fragment it cannot handle;
# ``numpy.linalg.LinAlgError`` is a ``ValueError``.
_CC_FAILURES = (ArithmeticError, RuntimeError, ValueError)

# Widest fragment whose seed is checked exactly (ten orbitals).
_SEED_CHECK_MAX_QUBITS = 20


class _FragmentVQE(VQE):
    """A fragment VQE whose fresh parameters can be supplied by the workflow.

    An explicit ``seed_params`` vector replaces the optimiser's own random
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
    def sampled_problem(self) -> MolecularProblem:
        """The fragment's Hamiltonian in the basis it was sampled in."""
        return self._fragment_problem

    @property
    def orbital_rotation(self) -> np.ndarray:
        """Identity: a fragment VQE samples in the fragment's own orbitals."""
        return np.eye(self._fragment_problem.n_orbitals)


def _uccsd_amplitude_seed(
    coupled_cluster, spec: FragmentSpec, n_params: int
) -> np.ndarray:
    """Map restricted CCSD ``t1``/``t2`` onto the first layer of
    :class:`~divi.qprog.algorithms.UCCSDAnsatz`.

    Each excitation takes its own amplitude: ``-t1`` for singles, ``+t2`` for
    doubles. The correspondence is positional, so ``coupled_cluster`` must be
    solved in the fragment's own orbitals, not a rotated set. Requires a
    spin-balanced fragment. Further layers stay at zero.
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


def _unrestricted_uccsd_amplitude_seed(
    coupled_cluster, spec: FragmentSpec, n_params: int
) -> np.ndarray:
    """:func:`_uccsd_amplitude_seed` for unrestricted CCSD, for a spin-polarised
    fragment.

    The blocks become antisymmetric spin-orbital tensors, occupied spin orbitals
    before virtual ones, which are read back through the blocked register
    (alpha orbitals first, then beta).
    """
    n_spatial = spec.n_orbitals
    n_alpha, n_beta = spec.n_alpha, spec.n_beta
    n_occupied = (n_alpha, n_beta)
    n_virtual = (n_spatial - n_alpha, n_spatial - n_beta)
    orbspin = np.repeat([0, 1, 0, 1], (*n_occupied, *n_virtual))
    t1 = cc_addons.spatial2spin(coupled_cluster.t1, orbspin)
    t2 = cc_addons.spatial2spin(coupled_cluster.t2, orbspin)

    def occupied_index(blocked: int) -> int:
        spin, spatial = divmod(blocked, n_spatial)
        return spatial + spin * n_alpha

    def virtual_index(blocked: int) -> int:
        spin, spatial = divmod(blocked, n_spatial)
        return spatial - n_occupied[spin] + spin * n_virtual[0]

    first_layer = []
    for occupied, unoccupied in _uccsd_excitations(n_spatial, (n_alpha, n_beta)):
        occupied_indices = [occupied_index(index) for index in occupied]
        virtual_indices = [virtual_index(index) for index in unoccupied]
        if len(occupied) == 1:
            first_layer.append(-t1[occupied_indices[0], virtual_indices[0]])
        else:
            first_layer.append(t2[tuple(occupied_indices + virtual_indices)])
    seed = np.zeros(n_params)
    take = min(n_params, len(first_layer))
    seed[:take] = first_layer[:take]
    return seed


def _unrestricted_ccsd(h_eff: np.ndarray, g_frag: np.ndarray, spec: FragmentSpec):
    """Unrestricted CCSD on the fragment's reference determinant, or ``None``
    (with a ``UserWarning``) if PySCF fails; non-convergence only warns."""
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
    return _coupled_cluster(cc.UCCSD, "UCCSD", mean_field, spec, stacklevel=4)


def _coupled_cluster(
    solver, name: str, mean_field, spec: FragmentSpec, stacklevel: int
):
    """``solver`` run on ``mean_field``, or ``None`` (with a ``UserWarning``) if
    PySCF raises a numerical or runtime error; non-convergence only warns."""
    try:
        coupled_cluster = solver(mean_field)
        coupled_cluster.max_cycle = _SEED_CC_MAX_CYCLE
        coupled_cluster.kernel()
    except _CC_FAILURES as exc:
        warn(
            f"CCSD seeding failed for fragment {spec.orbitals}: {exc}. "
            "Falling back to the optimiser's own initialisation.",
            UserWarning,
            stacklevel=stacklevel,
        )
        return None
    if not coupled_cluster.converged:
        warn(
            f"{name} did not converge for fragment {spec.orbitals}; seeding "
            "from its amplitudes anyway, as the seed is judged on its energy.",
            UserWarning,
            stacklevel=stacklevel,
        )
    return coupled_cluster


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
    """Map unrestricted CCSD amplitudes onto :class:`LUCJAnsatz`'s parameters
    through a double factorisation.

    The leading singular triplet of the opposite-spin doubles, reshaped over
    ``(occupied, virtual)`` pairs, gives one one-body operator per spin. Each
    spin's eigenbasis is its rotation, the eigenvalues give
    ``J_pq = sigma * d_p * d_q``, and ``t1`` supplies the trailing rotation.
    Opposite-spin doubles are used because a fragment with one electron of a
    spin has no same-spin doubles; same-spin Coulomb weights stay at zero.

    The seed is a starting point, not an encoding of CCSD: one Jastrow layer
    holds only the leading factorisation term, ``J`` is projected onto the
    layer's pair pattern, and the ``RZZ`` gates omit the one-body remainder of
    ``exp(i J n_p n_q)``.

    Returns ``None`` if a spin sector's rotation cannot be realised, if the
    opposite-spin doubles block is empty, or if the ansatz sets
    ``shared_spin_params``.
    """
    if ansatz_kwargs.get("shared_spin_params"):
        warn(
            f"CCSD seeding skipped for fragment {spec.orbitals}: the double "
            "factorisation gives each spin sector its own rotation, which "
            "shared_spin_params cannot hold. Falling back to the optimiser's own "
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
        # All-zero parameters are a stationary point a gradient optimiser cannot leave.
        return None

    # Rectangular when the spin sectors differ in size.
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
    # The beta rotation takes -i on its virtual orbitals to absorb the Jastrow's factor of i.
    phase = np.ones(n_orb, dtype=complex)
    phase[n_occupied_beta:] = -1j
    rotation_beta = rotation_beta.astype(complex) * phase[:, None]

    seed = np.zeros(n_params)
    cursor = 0
    sandwiched = n_rotation_params(n_orb, orbital_phases=False, depth=depth)
    for rotation in (rotation_alpha, rotation_beta):
        # The block runs first, in exp(-K), so it realises the inverse; the
        # conjugate transpose, not the transpose, pairs with the -i above.
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

    # Same-spin Coulomb weights stay at zero.
    cursor += 2 * len(same_pairs)

    if ansatz_kwargs.get("trailing_rotation"):
        for t1_block in (t1_alpha, t1_beta):
            generator = np.zeros((n_orb, n_orb))
            n_occupied, n_virtual = t1_block.shape
            generator[:n_occupied, n_occupied : n_occupied + n_virtual] = -t1_block
            generator -= generator.T
            angles = rotation_angles(expm(generator), depth=depth)
            if angles is None:
                # Only this block is lost; the rest of the seed stays useful.
                warn(
                    f"CCSD seeding for fragment {spec.orbitals} could not realise "
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

    ``unrestricted`` selects UHF over RHF. No orbital energies or SCF energy are
    set, since coupled cluster builds its Fock matrix from ``mo_coeff`` and
    ``mo_occ``.
    """
    mean_field = fragment_mean_field(
        scf.UHF if unrestricted else scf.RHF, h_eff, g_frag, spec
    )
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
    """Run unrestricted CCSD on the fragment and factorise it onto LUCJ.

    Unrestricted so spin-polarised fragments are representable. The one-body
    potential is spin-averaged, so the reference is polarised only through its
    occupations.
    """
    coupled_cluster = _unrestricted_ccsd(h_eff, g_frag, spec)
    if coupled_cluster is None:
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
    wide to check exactly, so the caller accepts the seed unchecked. A seed built
    in the wrong orbital basis can land above the reference determinant. All-zero
    parameters realise the reference determinant exactly, so it is the baseline.
    """
    if n_qubits > _SEED_CHECK_MAX_QUBITS:
        return None

    # Exact and in double precision, whatever backend the run itself uses.
    with closing(
        MaestroSimulator(maestro_config=MaestroConfig(simulation_type="Statevector"))
    ) as simulator:
        reference, seeded = (
            simulator.expectation_value(
                ansatz.build(params, n_qubits, n_layers, **build_kwargs), hamiltonian
            )
            for params in (np.zeros_like(seed), seed)
        )
    return reference - seeded


def _ccsd_seed_params(
    h_eff: np.ndarray,
    g_frag: np.ndarray,
    spec: FragmentSpec,
    n_params: int,
    ansatz: Ansatz | None = None,
    ansatz_kwargs: Mapping | None = None,
) -> np.ndarray | None:
    """Map a fragment's CCSD amplitudes onto an ansatz parameter vector.

    A fresh fragment's first round is seeded from coupled-cluster amplitudes
    computed on its own effective integrals instead of the optimiser's random
    initial guess.

    :class:`~divi.qprog.algorithms.UCCSDAnsatz` reads each parameter off the
    matching ``t1``/``t2`` entry (:func:`_uccsd_amplitude_seed`);
    :class:`~divi.qprog.algorithms.LUCJAnsatz` goes through a double
    factorisation of the doubles tensor (:func:`_lucj_seed_params`). Any other
    ansatz warns and uses the optimiser's own initialisation.

    CCSD runs in the fragment's own orbitals on the ansatz's reference
    determinant, not on SCF canonical orbitals, which would permute the
    amplitudes onto the wrong excitations. :func:`_seed_energy_gain` rejects a
    seed that still went wrong.

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
        the coupled-cluster calculation raises. Non-convergence only warns. A
        spin-imbalanced fragment (``n_alpha != n_beta``) is seeded from
        unrestricted CCSD.
    """
    if not isinstance(ansatz, (UCCSDAnsatz, LUCJAnsatz)):
        warn(
            f"CCSD seeding skipped for fragment {spec.orbitals}: no CCSD "
            f"correspondence is defined for {type(ansatz).__name__}. Falling "
            "back to the optimiser's own initialisation.",
            UserWarning,
            stacklevel=2,
        )
        return None

    if isinstance(ansatz, LUCJAnsatz):
        return _lucj_seed_params(h_eff, g_frag, spec, n_params, ansatz_kwargs or {})

    if spec.n_alpha != spec.n_beta:
        coupled_cluster = _unrestricted_ccsd(h_eff, g_frag, spec)
        if coupled_cluster is None:
            return None
        return _unrestricted_uccsd_amplitude_seed(coupled_cluster, spec, n_params)

    n_orb = spec.n_orbitals
    occupations = np.zeros(n_orb)
    occupations[: spec.n_alpha] = 2.0
    mean_field = _embedded_mean_field(
        h_eff, g_frag, spec, np.eye(n_orb), occupations, unrestricted=False
    )
    coupled_cluster = _coupled_cluster(cc.CCSD, "CCSD", mean_field, spec, stacklevel=3)
    if coupled_cluster is None:
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
    amplitudes via :func:`_ccsd_seed_params`; a warm-started fragment uses
    ``fragment.params`` and never calls CCSD.

    Seeding uses the spin-averaged embedding potential, which affects only the
    starting point; the optimised Hamiltonian keeps both channels. A materially
    asymmetric embedding is warned about, since the seed can land in a
    different basin than the symmetry-broken solution.
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
                f"embedding-potential spin channels that differ by "
                f"{spin_asymmetry:.3e} Hartree. The seed may sit in a "
                "different basin than the symmetry-broken solution; the "
                "optimised Hamiltonian keeps both channels.",
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
                    f"{fragment.spec.orbitals}: the seed gains {gain:.3e} "
                    "Hartree over the reference determinant, below the "
                    f"{_SEED_ACCEPTANCE_MARGIN:g} required. Falling back to "
                    "the optimiser's own initialisation.",
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
