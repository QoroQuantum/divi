# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Orbital re-optimisation: the L-BFGS-B, second-order (CIAH), saddle-escape and
state-averaged CASSCF solves, with the orbital rotation parameterisation and the
point-group symmetry checks they rely on."""

import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import combinations, pairwise, product
from warnings import warn

import numpy as np
import scipy.linalg
from pyscf import lib, mcscf
from pyscf.scf import hf_symm
from pyscf.soscf import ciah
from scipy.optimize import minimize

from ._integrals import (
    AOIntegrals,
    _energy_fock_and_potentials,
    _exact_mean_field,
    energy_and_generalized_fock,
    fragment_blocks,
)
from ._sqd import resolves_spin
from ._state import ElectronicState, FragmentSpec
from ._subspace import _SubspaceCI
from ._warnings import _outside_divi_stacklevel

# L-BFGS-B ftol is relative, so a tol=1e-6 shorthand halts around 1e-6 * |E|,
# coarser than LASSQD's energy_tol. Tighter than this terminates ABNORMAL.
ORBITAL_MINIMIZE_OPTIONS = {"ftol": 1e-12, "gtol": 1e-6}
# Most L-BFGS-B runs in one orbital solve, each restarted from the last one's
# orbitals.
_MAX_RESTARTS = 10
# Central-difference step of the Hessian-vector product, as the largest angle.
_HESSIAN_STEP = 1e-4
# Largest element-wise deviation from the identity that still counts as no step.
_STALLED_STEP = 1e-14
# Largest deviation of a space's irrep projection from a projector that still
# counts as symmetry-adapted.
_SYMMETRY_TOL = 1e-3
# A converged point this far above the best one seen still replaces it.
_ENERGY_ROUNDING = 1e-10
# Lowest orbital-Hessian eigenvalue below which a saddle is escaped, about 100x
# the central-difference noise floor.
_SADDLE_CURVATURE = -1e-3
# Eigenvalue tolerance of the stability Davidson solve.
_DAVIDSON_TOL = 1e-5
# Trial vectors the stability Davidson solve keeps before restarting.
_DAVIDSON_SPACE = 30
# Smallest magnitude the stability preconditioner divides by; many pairs have a
# zero Hessian diagonal.
_PRECONDITIONER_FLOOR = 1e-2
# First angle of the line search along a negative mode, doubled up to pi / 4.
_ESCAPE_FIRST_ANGLE = 0.05


@dataclass(frozen=True)
class OrbitalSolution:
    """Outcome of one round's orbital re-optimisation.

    Attributes:
        mo_coeff: The rotated MO coefficients.
        energy: Total energy at the returned orbitals.
        converged: Whether ``gradient_norm`` is within the caller's tolerance.
            A capped or stalled solve still returns its orbitals and energy,
            an upper bound but not a stationary point.
        n_iterations: Optimiser iterations taken.
        n_evaluations: Objective evaluations taken, each one four-index MO
            transform.
        gradient_norm: L2 norm of the orbital gradient over the rotation pairs
            at the returned orbitals.
        n_rotation_pairs: Number of orbital pairs the rotation spanned.
    """

    mo_coeff: np.ndarray
    energy: float
    converged: bool
    n_iterations: int
    n_evaluations: int
    gradient_norm: float
    n_rotation_pairs: int


def rotation_energy_gradient_fn(
    ao: AOIntegrals,
    mo_coeff: np.ndarray,
    n_core: int,
    fragment_specs: Sequence[FragmentSpec],
    rdm1_active: np.ndarray,
    rdm2_active: np.ndarray,
):
    """Build the orbital-rotation objective and its analytic gradient.

    The returned callable evaluates the energy at ``mo_coeff @ expm(K(x))`` and
    its derivative with respect to the rotation angles ``x``, sharing one
    four-index MO transform.

    The gradient is the Frechet pullback ``M = expm_frechet(K.T, U @ 2F)`` of
    the generalised Fock matrix ``F``, ``dE/dx_i = M[p, q] - M[q, p]``, which
    is ``2 * (F[p, q] - F[q, p])`` at ``x = 0``. It requires ``rdm1_active``
    symmetric and ``rdm2_active`` invariant under ``pqrs -> rspq`` and
    ``pqrs -> qpsr``; the energy does not.

    Args:
        ao: The molecule and its AO integrals.
        mo_coeff: ``(nao, n_orb)`` MO coefficients, permuted into
            ``[core | fragment blocks | virtual]`` order.
        n_core: Number of frozen-core orbitals.
        fragment_specs: Fragment specifications, in the same order as the
            fragment blocks in ``mo_coeff``; only each fragment's orbital
            count is used.
        rdm1_active: ``(n_act, n_act)`` active-space 1-RDM.
        rdm2_active: ``(n_act,) * 4`` active-space 2-RDM.

    Returns:
        ``(rotation_pairs, energy_and_gradient)``: the ``(p, q)`` orbital
        pairs the rotation angles are indexed by, and a callable mapping a
        length-``len(rotation_pairs)`` angle vector to ``(energy, gradient)``.

    Raises:
        ImportError: If the ``chem`` extra is not installed.
    """
    pairs = rotation_pairs(mo_coeff.shape[1], n_core, fragment_specs)
    rows, cols = pair_indices(pairs)

    def energy_and_gradient(
        rotation_params: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        return _rotated_energy_and_gradient(
            ao,
            mo_coeff,
            rows,
            cols,
            rotation_params,
            n_core,
            rdm1_active,
            rdm2_active,
        )

    return pairs, energy_and_gradient


def _rotated_energy_and_gradient(
    ao: AOIntegrals,
    mo_coeff: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    rotation_params: np.ndarray,
    n_core: int,
    rdm1_active: np.ndarray,
    rdm2_active: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Energy at ``mo_coeff @ expm(K(x))`` and its gradient in the angles ``x``."""
    generator = rotation_generator(rows, cols, rotation_params, mo_coeff.shape[1])
    unitary = scipy.linalg.expm(generator)
    energy, fock = energy_and_generalized_fock(
        ao, mo_coeff @ unitary, n_core, rdm1_active, rdm2_active
    )
    pullback = np.asarray(
        scipy.linalg.expm_frechet(
            generator.T, unitary @ (2.0 * fock), compute_expm=False
        )
    )
    return energy, pullback[rows, cols] - pullback[cols, rows]


def rotation_pairs(
    n_orbitals: int, n_core: int, fragment_specs: Sequence[FragmentSpec]
) -> list[tuple[int, int]]:
    """Orbital pairs LASSQD rotates: core-active, core-virtual, active-active
    across different fragments, and active-virtual."""
    n_act = sum(spec.n_orbitals for spec in fragment_specs)
    core = range(n_core)
    active = range(n_core, n_core + n_act)
    virtual = range(n_core + n_act, n_orbitals)

    pairs: list[tuple[int, int]] = [*product(core, active), *product(core, virtual)]
    for block_a, block_b in combinations(
        fragment_blocks(fragment_specs, offset=n_core), 2
    ):
        pairs += product(
            range(block_a.start, block_a.stop), range(block_b.start, block_b.stop)
        )
    pairs += product(active, virtual)
    return pairs


def pair_indices(
    rotation_pairs: Sequence[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    """Split rotation pairs into row and column index arrays."""
    pairs = np.asarray(rotation_pairs, dtype=int).reshape(len(rotation_pairs), 2)
    return pairs[:, 0], pairs[:, 1]


def rotation_generator(
    rows: np.ndarray, cols: np.ndarray, angles: np.ndarray, n_orbitals: int
) -> np.ndarray:
    """Antisymmetric generator ``K`` with ``K[rows, cols] = angles``."""
    generator = np.zeros((n_orbitals, n_orbitals))
    generator[rows, cols] = angles
    generator[cols, rows] = -angles
    return generator


def optimize_orbitals(
    ao: AOIntegrals,
    mo_coeff: np.ndarray,
    n_core: int,
    fragment_specs: Sequence[FragmentSpec],
    rdm1_active: np.ndarray,
    rdm2_active: np.ndarray,
    *,
    gradient_tol: float,
    max_iterations: int | None = None,
    report: Callable[[str], None] | None = None,
) -> OrbitalSolution:
    """Optimise molecular orbitals against the current active-space RDMs.

    Minimises the energy with L-BFGS-B over the angles of an orbital rotation
    on :func:`rotation_pairs` -- core-active, core-virtual, active-active across
    fragments, and active-virtual -- using the analytic gradient of
    :func:`rotation_energy_gradient_fn`, one four-index MO transform per
    iteration. Rotations within a fragment are excluded because its RDM is only
    valid in its current orbital basis.

    The unrotated orbitals are always evaluated and returned if the minimiser
    does no better, so the energy never worsens. With no rotation pairs,
    ``minimize`` is not called.

    The first pass runs even when the starting gradient is within
    ``gradient_tol``. While the gradient at the current orbitals exceeds it, the
    solve restarts from them, up to ``_MAX_RESTARTS`` times and ``max_iterations``
    in total, since angles turn ill-conditioned after a large rotation.
    ``gradient_norm`` is taken at the returned orbitals.

    Convergence is judged on that gradient, not L-BFGS-B's status: a minimiser
    that gives up returns the unrotated orbitals, which looks like a converged
    macro-cycle, and the status also reports success once the relative energy
    change stalls.

    Args:
        ao: The molecule and its AO integrals.
        mo_coeff: ``(nao, n_orb)`` MO coefficients, permuted into
            ``[core | fragment blocks | virtual]`` order.
        n_core: Number of frozen-core orbitals.
        fragment_specs: Fragment specifications, in the same order as the
            fragment blocks in ``mo_coeff``; only each fragment's orbital
            count is used.
        rdm1_active: ``(n_act, n_act)`` active-space 1-RDM.
        rdm2_active: ``(n_act,) * 4`` active-space 2-RDM.
        gradient_tol: The solve counts as converged when the L2 norm of the
            orbital gradient at the returned orbitals is at most this.
        max_iterations: Cap on L-BFGS-B iterations; ``None`` uses scipy's
            default.
        report: Receives a progress line after every iteration.

    Returns:
        The :class:`OrbitalSolution`.

    Raises:
        ImportError: If the ``chem`` extra is not installed.

    Warns:
        UserWarning: If the solve ended without converging.
    """
    n_orb_total = mo_coeff.shape[1]
    fixed = (n_core, fragment_specs, rdm1_active, rdm2_active)

    pairs, energy_and_gradient = rotation_energy_gradient_fn(ao, mo_coeff, *fixed)
    n_rot = len(pairs)
    best_energy, best_gradient = energy_and_gradient(np.zeros(n_rot))
    rotated_mo_coeff = mo_coeff
    n_iterations = 0
    n_evaluations = 1
    stop_reason = "no rotation freedom"
    if n_rot > 0:
        rows, cols = pair_indices(pairs)
        started = time.monotonic()
        latest_gradient_norm = float(np.linalg.norm(best_gradient))

        def tracked(rotation_params: np.ndarray) -> tuple[float, np.ndarray]:
            nonlocal latest_gradient_norm
            energy, gradient = energy_and_gradient(rotation_params)
            latest_gradient_norm = float(np.linalg.norm(gradient))
            return energy, gradient

        def on_iteration(intermediate_result) -> None:
            nonlocal n_iterations
            n_iterations += 1
            if report is not None:
                report(
                    f"Orbital solve: iteration {n_iterations}, energy "
                    f"{intermediate_result.fun:.8f} Ha, |g| "
                    f"{latest_gradient_norm:.2e}, {time.monotonic() - started:.0f} s"
                )

        # The first pass runs even within tolerance, as ciah_orbital_solve's
        # first step does.
        for restart in range(_MAX_RESTARTS):
            if restart and np.linalg.norm(best_gradient) <= gradient_tol:
                break
            options = dict(ORBITAL_MINIMIZE_OPTIONS)
            if max_iterations is not None:
                if n_iterations >= max_iterations:
                    break
                options["maxiter"] = max_iterations - n_iterations
            res = minimize(
                tracked,
                np.zeros(n_rot),
                method="L-BFGS-B",
                jac=True,
                options=options,
                callback=on_iteration,
            )
            n_evaluations += int(res.nfev)
            stop_reason = str(res.message).strip()
            if not (np.isfinite(res.fun) and res.fun < best_energy):
                break
            rotated_mo_coeff = rotated_mo_coeff @ scipy.linalg.expm(
                rotation_generator(rows, cols, res.x, n_orb_total)
            )
            _, energy_and_gradient = rotation_energy_gradient_fn(
                ao, rotated_mo_coeff, *fixed
            )
            best_energy, best_gradient = energy_and_gradient(np.zeros(n_rot))
            n_evaluations += 1
    return _finished_solve(
        rotated_mo_coeff,
        best_energy,
        best_gradient,
        gradient_tol,
        n_iterations,
        n_evaluations,
        n_rot,
        stop_reason,
    )


def _finished_solve(
    mo_coeff: np.ndarray,
    energy: float,
    gradient: np.ndarray,
    gradient_tol: float,
    n_iterations: int,
    n_evaluations: int,
    n_rotation_pairs: int,
    stop_reason: str,
) -> OrbitalSolution:
    """The :class:`OrbitalSolution` for a finished solve, warning if it did not
    converge."""
    gradient_norm = float(np.linalg.norm(gradient))
    converged = gradient_norm <= gradient_tol
    if not converged:
        warn(
            f"Orbital optimisation stopped after {n_iterations} iterations and "
            f"{n_evaluations} evaluations ({stop_reason}) with orbital-gradient "
            f"norm {gradient_norm:.2e} above {gradient_tol:.2e}. The energy "
            "remains an upper bound, but this round is not a stationary point, "
            "so a small energy change from the last round may mean the "
            "optimiser stopped rather than that the macro-cycle converged.",
            UserWarning,
            stacklevel=_outside_divi_stacklevel(),
        )
    return OrbitalSolution(
        mo_coeff=mo_coeff,
        energy=float(energy),
        converged=converged,
        n_iterations=n_iterations,
        n_evaluations=n_evaluations,
        gradient_norm=gradient_norm,
        n_rotation_pairs=n_rotation_pairs,
    )


class _LASOrbitalHessian(ciah.CIAHOptimizerMixin):
    """pyscf's CIAH optimiser contract over LASSQD's rotation pairs.

    The Hessian-vector product is a central difference of the analytic
    gradient. The preconditioning diagonal is the one-body part of pyscf
    CASSCF's (``mc1step.gen_g_hop``, parts 7 and 8), built from the inactive
    plus active Fock matrix in place of the core Hamiltonian.
    """

    # Values at the orbitals ``_move_to`` last evaluated.
    _energy: float
    _gradient: np.ndarray
    _fock: np.ndarray
    _vj: np.ndarray
    _vk: np.ndarray

    def __init__(
        self,
        ao: AOIntegrals,
        mo_coeff: np.ndarray,
        n_core: int,
        fragment_specs: Sequence[FragmentSpec],
        rdm1_active: np.ndarray,
        rdm2_active: np.ndarray,
    ):
        super().__init__(mo_coeff.shape[1])
        self._ao = ao
        self._mo_coeff = mo_coeff
        self._n_core = n_core
        self._rdm1_active = rdm1_active
        self._rdm2_active = rdm2_active
        self._rows, self._cols = pair_indices(
            rotation_pairs(self.norb, n_core, fragment_specs)
        )
        self.evaluations = 0
        self._u: np.ndarray | None = None

    @property
    def pdim(self) -> int:
        return len(self._rows)

    def pack_uniq_var(self, mat: np.ndarray) -> np.ndarray:
        return np.asarray(mat)[self._rows, self._cols]

    def unpack_uniq_var(self, v: np.ndarray) -> np.ndarray:
        return rotation_generator(self._rows, self._cols, v, self.norb)

    def _move_to(self, u: np.ndarray) -> None:
        """Evaluate at ``mo_coeff @ u`` unless ``u`` is the array last
        evaluated."""
        if u is self._u:
            return
        self._energy, self._fock, self._vj, self._vk = _energy_fock_and_potentials(
            self._ao,
            self._mo_coeff @ u,
            self._n_core,
            self._rdm1_active,
            self._rdm2_active,
        )
        self.evaluations += 1
        rows, cols = self._rows, self._cols
        self._gradient = 2.0 * (self._fock[rows, cols] - self._fock[cols, rows])
        self._u = u

    def energy_and_gradient_at(self, u: np.ndarray) -> tuple[float, np.ndarray]:
        """Energy and orbital gradient at ``mo_coeff @ u``."""
        self._move_to(u)
        return self._energy, self._gradient

    def get_grad(self, u: np.ndarray) -> np.ndarray:
        self._move_to(u)
        return self._gradient

    def gen_g_hop(self, u: np.ndarray):
        self._move_to(u)
        mo_coeff = self._mo_coeff @ u

        def gradient_at(angles: np.ndarray) -> np.ndarray:
            self.evaluations += 1
            return _rotated_energy_and_gradient(
                self._ao,
                mo_coeff,
                self._rows,
                self._cols,
                angles,
                self._n_core,
                self._rdm1_active,
                self._rdm2_active,
            )[1]

        def hessian_vector(vector: np.ndarray) -> np.ndarray:
            largest = float(np.abs(vector).max())
            if largest == 0.0:
                return np.zeros(self.pdim)
            scale = _HESSIAN_STEP / largest
            return (gradient_at(scale * vector) - gradient_at(-scale * vector)) / (
                2.0 * scale
            )

        return self._gradient, hessian_vector, self._hessian_diagonal(mo_coeff)

    def _hessian_diagonal(self, mo_coeff: np.ndarray) -> np.ndarray:
        """Diagonal at the last evaluated orbitals, ``mo_coeff``."""
        n_core = self._n_core
        active = slice(n_core, n_core + self._rdm1_active.shape[0])
        (j_core, j_act), (k_core, k_act) = self._vj, self._vk
        mean_field = (
            mo_coeff.T
            @ (self._ao.h_core + 2.0 * j_core - k_core + j_act - 0.5 * k_act)
            @ mo_coeff
        )
        fock = self._fock
        density = np.zeros((self.norb, self.norb))
        density[np.arange(n_core), np.arange(n_core)] = 2.0
        density[active, active] = self._rdm1_active
        rows, cols = self._rows, self._cols
        # Our gradient is twice pyscf's, so the Hessian is too.
        return 2.0 * (
            mean_field[rows, rows] * density[cols, cols]
            + mean_field[cols, cols] * density[rows, rows]
            - 2.0 * mean_field[rows, cols] * density[rows, cols]
            - fock[rows, rows]
            - fock[cols, cols]
        )


def ciah_orbital_solve(
    ao: AOIntegrals,
    mo_coeff: np.ndarray,
    n_core: int,
    fragment_specs: Sequence[FragmentSpec],
    rdm1_active: np.ndarray,
    rdm2_active: np.ndarray,
    *,
    gradient_tol: float,
    max_iterations: int,
    report: Callable[[str], None] | None = None,
) -> OrbitalSolution:
    """Optimise the orbitals with pyscf's second-order CIAH solver.

    Same objective, rotation pairs and result as :func:`optimize_orbitals`.
    Each macro-iteration takes an augmented-Hessian step, at least one even when
    the starting gradient is within tolerance. A vanishing step restarts the
    augmented-Hessian solve, and a restart that also gives none ends the solve.
    A step that raises the energy hands the rest of the budget to
    :func:`optimize_orbitals` from the lowest point. The lowest-energy orbitals
    seen are returned, preferring a converged point within rounding of them.

    Args:
        max_iterations: Macro-iterations, each one gradient at new orbitals
            plus the Hessian-vector products of its augmented-Hessian solve.
        report: Receives a progress line after every macro-iteration.
    """
    hessian = _LASOrbitalHessian(
        ao, mo_coeff, n_core, fragment_specs, rdm1_active, rdm2_active
    )
    identity = np.eye(mo_coeff.shape[1])
    rotation = identity
    best_energy, best_gradient = hessian.energy_and_gradient_at(rotation)
    best_mo_coeff = mo_coeff
    n_iterations = 0
    rose = False
    if not hessian.pdim:
        stop_reason = "no rotation freedom"
    else:
        stop_reason = "macro-iteration limit"
        started = time.monotonic()

        def start(tolerance: float = gradient_tol):
            # pyscf seeds a solve with its last step; a fresh one uses the gradient.
            steps = ciah.rotate_orb_cc(hessian, rotation, tolerance, verbose=0)
            return steps, next(steps)[0], True

        # A gradient already under the tolerance still gets one tight step:
        # stopping there can leave the macro-cycle short of the minimum.
        already_small = np.linalg.norm(best_gradient) <= gradient_tol
        steps, step, fresh = start(0.0 if already_small else gradient_tol)
        while n_iterations < max_iterations:
            if np.abs(step - identity).max() <= _STALLED_STEP:
                if fresh:
                    stop_reason = "stalled"
                    break
                steps.close()
                steps, step, fresh = start()
                continue
            fresh = False
            n_iterations += 1
            rotation = rotation @ step
            energy, gradient = hessian.energy_and_gradient_at(rotation)
            gradient_norm = float(np.linalg.norm(gradient))
            converged = gradient_norm <= gradient_tol
            rose = energy > best_energy + _ENERGY_ROUNDING
            if not rose and (energy < best_energy or converged):
                best_energy, best_gradient = energy, gradient
                best_mo_coeff = mo_coeff @ rotation
            if report is not None:
                report(
                    f"Orbital solve: iteration {n_iterations}, energy {energy:.8f} "
                    f"Ha, |g| {gradient_norm:.2e}, {time.monotonic() - started:.0f} s"
                )
            if rose:
                stop_reason = "energy rose"
                break
            if converged:
                stop_reason = "converged"
                break
            if n_iterations < max_iterations:
                step = steps.send(rotation)[0]
        steps.close()

    if rose and n_iterations < max_iterations:
        # The augmented-Hessian step can climb towards a saddle; L-BFGS-B only
        # descends.
        descent = optimize_orbitals(
            ao,
            best_mo_coeff,
            n_core,
            fragment_specs,
            rdm1_active,
            rdm2_active,
            gradient_tol=gradient_tol,
            max_iterations=max_iterations - n_iterations,
            report=report,
        )
        return replace(
            descent,
            n_iterations=n_iterations + descent.n_iterations,
            n_evaluations=hessian.evaluations + descent.n_evaluations,
        )
    return _finished_solve(
        best_mo_coeff,
        best_energy,
        best_gradient,
        gradient_tol,
        n_iterations,
        hessian.evaluations,
        hessian.pdim,
        stop_reason,
    )


def escape_orbital_saddle(
    ao: AOIntegrals,
    mo_coeff: np.ndarray,
    n_core: int,
    fragment_specs: Sequence[FragmentSpec],
    rdm1_active: np.ndarray,
    rdm2_active: np.ndarray,
    *,
    min_energy_drop: float,
) -> tuple[tuple[np.ndarray, float] | None, bool]:
    """Orbitals past a saddle point of the fixed-RDM energy at ``mo_coeff``,
    with their energy, or ``None`` if none lower by ``min_energy_drop`` is
    found; and whether the check was conclusive.

    The lowest orbital-Hessian eigenpair comes from a Davidson solve
    preconditioned by the Hessian diagonal, as in pyscf's SCF stability
    analysis. Below ``_SADDLE_CURVATURE`` the orbitals are rotated along that
    mode in both directions, keeping the lowest energy. A negative eigenvalue
    indicates the coupled orbital-CI energy is also at a saddle; a positive one
    does not prove a minimum. The check is inconclusive when the solve stops
    unconverged above ``_SADDLE_CURVATURE``, since a lower mode may remain.
    """
    hessian = _LASOrbitalHessian(
        ao, mo_coeff, n_core, fragment_specs, rdm1_active, rdm2_active
    )
    if not hessian.pdim:
        return None, True
    identity = np.eye(mo_coeff.shape[1])
    energy, _ = hessian.energy_and_gradient_at(identity)
    _, hessian_vector, diagonal = hessian.gen_g_hop(identity)

    def precondition(residual, eigenvalue, _vector):
        shifted = diagonal - eigenvalue
        shifted[np.abs(shifted) < _PRECONDITIONER_FLOOR] = _PRECONDITIONER_FLOOR
        return residual / shifted

    # A random component reaches modes the lowest-diagonal guess is blind to.
    guess = np.random.default_rng(0).standard_normal(hessian.pdim)
    guess *= 0.1 / np.linalg.norm(guess)
    guess[np.argmin(diagonal)] += 1.0
    converged, eigenvalues, modes = lib.davidson1(
        lambda vectors: [hessian_vector(vector) for vector in vectors],
        guess,
        precondition,
        tol=_DAVIDSON_TOL,
        max_space=_DAVIDSON_SPACE,
        verbose=0,
    )
    eigenvalue, mode = float(np.asarray(eigenvalues)[0]), modes[0]
    if not eigenvalue < _SADDLE_CURVATURE:
        return None, bool(converged[0])
    mode = mode / np.abs(mode).max()
    best_energy, best_rotation = energy, identity
    for sign in (1.0, -1.0):
        angle = _ESCAPE_FIRST_ANGLE
        while angle <= np.pi / 4:
            rotation = scipy.linalg.expm(hessian.unpack_uniq_var(sign * angle * mode))
            rotated_energy, _ = hessian.energy_and_gradient_at(rotation)
            if rotated_energy < best_energy:
                best_energy, best_rotation = rotated_energy, rotation
            angle *= 2.0
    if not best_energy < energy - min_energy_drop:
        return None, True
    return (mo_coeff @ best_rotation, float(best_energy)), True


def abelian_molecule(mol):
    """``mol``, or for a linear molecule a copy that works in D2h or C2v.

    PySCF pairs a linear molecule's degenerate orbitals by their energies,
    which a rotation within an irrep separates.
    """
    subgroup = {"Dooh": "D2h", "Coov": "C2v"}.get(mol.groupname)
    if subgroup is None:
        return mol
    abelian = mol.copy()
    abelian.symmetry_subgroup = subgroup
    abelian.build(dump_input=False, parse_arg=False)
    return abelian


def orbital_irreps(mol, mo_coeff: np.ndarray) -> np.ndarray | None:
    """Each orbital's irrep under the abelian point group ``mol`` works in, or
    ``None`` when ``mol`` has no symmetry or an orbital mixes irreps."""
    if not mol.symmetry:
        return None
    try:
        return hf_symm.get_orbsym(
            abelian_molecule(mol), mo_coeff, mol.intor_symmetric("int1e_ovlp"), True
        )
    except ValueError:
        return None


def _broken_symmetry(
    mol, mo_coeff: np.ndarray, n_core: int, n_active: int
) -> str | None:
    """``mol``'s point group, detected even when ``mol`` was built without
    symmetry, when the core, active or virtual space of ``mo_coeff`` mixes its
    irreps; ``None`` when all three are symmetry-adapted.

    Each space's projection onto an irrep is a projector exactly when the space
    is symmetry-adapted, whatever rotation mixes orbitals within it.
    """
    symmetric = mol
    if not mol.symmetry:
        symmetric = mol.copy()
        symmetric.symmetry = True
        symmetric.build(dump_input=False, parse_arg=False)
    overlap = mol.intor_symmetric("int1e_ovlp")
    bounds = (0, n_core, n_core + n_active, mo_coeff.shape[1])
    for start, stop in pairwise(bounds):
        if start == stop:
            continue
        projected = overlap @ mo_coeff[:, start:stop]
        for irrep_basis in symmetric.symm_orb:
            component = irrep_basis.T @ projected
            weight = component.T @ scipy.linalg.solve(
                irrep_basis.T @ overlap @ irrep_basis, component, assume_a="pos"
            )
            if np.abs(weight @ weight - weight).max() > _SYMMETRY_TOL:
                return symmetric.groupname
    return None


def state_averaged_orbital_solve(
    ao: AOIntegrals,
    mo_coeff: np.ndarray,
    n_core: int,
    spec: FragmentSpec,
    strings: tuple[Sequence[str], Sequence[str]],
    weights: Mapping[tuple[int, int], float],
    lambda_penalty: float,
    *,
    gradient_tol: float,
    max_iterations: int,
    report: Callable[[str], None] | None = None,
) -> tuple[OrbitalSolution, tuple[ElectronicState, ...]]:
    """State-averaged CASSCF over one fragment, its CI confined to an SQD
    subspace.

    Orbitals and CI vectors move together (PySCF's one-step SA-CASSCF).
    ``mo_coeff``'s active block must be the basis ``strings`` refer to. The
    Hamiltonian is built from ``ao``, never a density-fitted approximation.
    The reported gradient is LASSQD's own, at the returned orbitals and the
    averaged RDMs.

    Args:
        strings: The subspace's alpha and beta occupation strings.
        weights: Weight per state, keyed ``(multiplicity, index)``.
        max_iterations: Cap on SA-CASSCF macro-iterations.
        report: Receives a progress line after every macro-iteration.

    Returns:
        ``(solve, states)``: the solve, its energy the weighted average, and
        each state at the returned orbitals, in root order.
    """
    strings_alpha, strings_beta = strings
    roots = dict(sorted(Counter(multiplicity for multiplicity, _ in weights).items()))
    solvers = [
        _SubspaceCI(strings_alpha, strings_beta, multiplicity, n_roots, lambda_penalty)
        for multiplicity, n_roots in roots.items()
    ]
    labels = [
        (multiplicity, index)
        for multiplicity, n_roots in roots.items()
        for index in range(n_roots)
    ]
    n_active = spec.n_orbitals
    nelec = (spec.n_alpha, spec.n_beta)

    mol = ao.mol
    keeps_symmetry = orbital_irreps(mol, mo_coeff) is not None
    if keeps_symmetry:
        casscf = mcscf.mc1step_symm.SymAdaptedCASSCF(
            _exact_mean_field(replace(ao, mol=abelian_molecule(mol))),
            n_active,
            nelec,
            n_core,
        )
    else:
        casscf = mcscf.mc1step.CASSCF(_exact_mean_field(ao), n_active, nelec, n_core)
    casscf.verbose = 0
    casscf.canonicalization = False
    casscf.max_cycle_macro = max_iterations
    casscf.conv_tol = gradient_tol**2
    # PySCF's orbital gradient is half LASSQD's.
    casscf.conv_tol_grad = gradient_tol / 2
    casscf = mcscf.addons.state_average_mix(
        casscf, solvers, [weights[label] for label in labels]
    )
    started = time.monotonic()
    macro_iterations = 0

    def progress(envs):
        nonlocal macro_iterations
        # PySCF also calls back after each micro-iteration, while its orbital
        # rotation generator is still open; it is closed by the macro callback.
        if envs.get("rota") is not None:
            return
        macro_iterations = envs["imacro"]
        if report is not None:
            report(
                f"State-averaged orbital solve: iteration {macro_iterations}, "
                f"energy {envs['e_tot']:.8f} Ha, |g| {2 * envs['norm_gorb']:.2e}, "
                f"{time.monotonic() - started:.0f} s"
            )

    casscf.callback = progress
    casscf.kernel(mo_coeff)
    group = (
        None
        if keeps_symmetry
        else _broken_symmetry(mol, casscf.mo_coeff, n_core, n_active)
    )
    if group is not None:
        remedy = (
            "Pass a mean field with symmetry-adapted orbitals"
            if mol.symmetry
            else "Build the molecule with symmetry=True"
        )
        warn(
            f"The state-averaged solve broke the molecule's {group} symmetry, "
            "mixing states of different symmetry; which such solution a run "
            f"reaches can depend on rounding and the starting orbitals. {remedy} "
            "to keep it.",
            UserWarning,
            stacklevel=_outside_divi_stacklevel(),
        )

    rdm1, rdm2 = casscf.fcisolver.make_rdm12(casscf.ci, n_active, nelec)
    pairs, energy_and_gradient = rotation_energy_gradient_fn(
        ao, casscf.mo_coeff, n_core, [spec], rdm1, rdm2
    )
    energy, gradient = energy_and_gradient(np.zeros(len(pairs)))
    solve = _finished_solve(
        casscf.mo_coeff,
        energy,
        gradient,
        gradient_tol,
        macro_iterations,
        macro_iterations,
        len(pairs),
        "converged" if casscf.converged else "macro-iteration limit",
    )
    if solve.converged and not casscf.converged:
        warn(
            f"The state-averaged solve stopped at max_iterations={max_iterations} "
            "before settling, though its orbital gradient is below tolerance. "
            "Unless the cap is deliberately low, a common cause is a requested "
            "state degenerate with one left out, which leaves the averaged "
            "energy a flat direction. Request every component of the "
            "degenerate set.",
            UserWarning,
            stacklevel=_outside_divi_stacklevel(),
        )
    spin_squares = casscf.fcisolver.states_spin_square(casscf.ci, n_active, nelec)[0]
    states: list[ElectronicState] = []
    for (multiplicity, _), state_energy, spin_square in zip(
        labels, casscf.e_states, spin_squares
    ):
        if resolves_spin(multiplicity, spin_square):
            index = sum(1 for state in states if state.multiplicity == multiplicity)
            states.append(
                ElectronicState(
                    multiplicity, index, float(state_energy), float(spin_square)
                )
            )
    return solve, tuple(states)
