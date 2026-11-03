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

from divi.backends import CircuitRunner, MaestroConfig, MaestroSimulator
from divi.qprog.algorithms import (
    VQE,
    AmplitudeSeededAnsatz,
    AmplitudeSeedError,
    Ansatz,
    ElectronCounts,
)
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


def _unrestricted_amplitudes(
    coupled_cluster,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """A restricted CCSD solution's ``(t1, t2)`` in UCCSD's spin-blocked layout."""
    t1, t2 = np.asarray(coupled_cluster.t1), np.asarray(coupled_cluster.t2)
    t2_same_spin = t2 - t2.transpose(0, 1, 3, 2)
    return (t1, t1), (t2_same_spin, t2, t2_same_spin)


def _seed_energy_gain(
    seed: np.ndarray,
    hamiltonian,
    ansatz: Ansatz,
    n_qubits: int,
    n_layers: int,
    electron_counts: ElectronCounts,
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
                ansatz.build(params, n_qubits, n_layers, electron_counts), hamiltonian
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
) -> np.ndarray | None:
    """Map a fragment's CCSD amplitudes onto an ansatz parameter vector.

    A fresh fragment's first round is seeded from coupled-cluster amplitudes
    computed on its own effective integrals instead of the optimiser's random
    initial guess.

    The ansatz maps the amplitudes onto its own parameters
    (:meth:`~divi.qprog.algorithms.AmplitudeSeededAnsatz.params_from_amplitudes`).
    Any ansatz without that capability warns and uses the optimiser's own
    initialisation.

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
        ansatz: The fragment's configured ansatz, whose options fix the
            parameter layout the seed has to fill.

    Returns:
        A length-``n_params`` vector, or ``None`` (with a ``UserWarning``) if
        ``ansatz`` is not an
        :class:`~divi.qprog.algorithms.AmplitudeSeededAnsatz`, if the
        coupled-cluster calculation raises, or if the ansatz cannot map the
        amplitudes. Non-convergence only warns. A spin-imbalanced fragment
        (``n_alpha != n_beta``) is seeded from unrestricted CCSD, a balanced
        one from restricted CCSD.
    """
    if not isinstance(ansatz, AmplitudeSeededAnsatz):
        warn(
            f"CCSD seeding skipped for fragment {spec.orbitals}: no CCSD "
            f"correspondence is defined for {type(ansatz).__name__}. Falling "
            "back to the optimiser's own initialisation.",
            UserWarning,
            stacklevel=2,
        )
        return None

    if spec.n_alpha != spec.n_beta:
        coupled_cluster = _unrestricted_ccsd(h_eff, g_frag, spec)
        if coupled_cluster is None:
            return None
        t1, t2 = coupled_cluster.t1, coupled_cluster.t2
    else:
        n_orb = spec.n_orbitals
        occupations = np.zeros(n_orb)
        occupations[: spec.n_alpha] = 2.0
        mean_field = _embedded_mean_field(
            h_eff, g_frag, spec, np.eye(n_orb), occupations, unrestricted=False
        )
        coupled_cluster = _coupled_cluster(
            cc.CCSD, "CCSD", mean_field, spec, stacklevel=3
        )
        if coupled_cluster is None:
            return None
        t1, t2 = _unrestricted_amplitudes(coupled_cluster)

    try:
        first_layer = ansatz.params_from_amplitudes(
            t1,
            t2,
            2 * spec.n_orbitals,
            ElectronCounts(spec.n_alpha + spec.n_beta, spec.n_alpha, spec.n_beta),
        )
    except AmplitudeSeedError as error:
        warn(
            f"CCSD seeding skipped for fragment {spec.orbitals}: {error} "
            "Falling back to the optimiser's own initialisation.",
            UserWarning,
            stacklevel=2,
        )
        return None

    seed = np.zeros(n_params)
    take = min(n_params, len(first_layer))
    seed[:take] = first_layer[:take]
    return seed


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

    if fragment.params is not None:
        seed_params = fragment.params
    else:
        n_qubits = 2 * problem.n_orbitals
        n_layers = options.get("n_layers", 1)
        electron_counts = ElectronCounts(
            problem.n_electrons, problem.n_alpha, problem.n_beta
        )
        n_params = n_layers * ansatz.n_params_per_layer(n_qubits, electron_counts)
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
        )
        if seed_params is not None:
            gain = _seed_energy_gain(
                seed_params,
                problem.hamiltonian,
                ansatz,
                n_qubits,
                n_layers,
                electron_counts,
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
