# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Configuration objects for the LASSQD workflow."""

import hashlib
import pickle
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

from divi.backends import CircuitRunner
from divi.qprog.algorithms import Ansatz, UCCSDAnsatz
from divi.qprog.optimizers import Optimizer
from divi.qprog.problems import MolecularProblem

from ._integrals import OrbitalSolve, ciah_orbital_solve, optimize_orbitals
from ._preparation import LUCJFragmentProgram
from ._state import FragmentSpec, FragmentState
from ._vqe_preparation import _FragmentVQE, build_fragment_vqe


def _checkpoint_digest(value: object) -> str:
    """Stable digest of a configuration template without object addresses."""
    return hashlib.sha256(pickle.dumps(value, protocol=4)).hexdigest()


# Program options an LUCJ fragment takes, beyond those LASSQD sets itself.
_LUCJ_PROGRAM_OPTIONS = (
    "precision",
    "qem_protocol",
    "suppress_performance_warnings",
)


@dataclass(frozen=True)
class _LUCJPreparation:
    """Shared behaviour of the preparations that build a CCSD-seeded LUCJ
    circuit classically and submit only its sample."""

    _run_linear_method: ClassVar[bool]

    def _check_options(self, options: Mapping[str, Any]) -> None:
        unsupported = set(options) - set(_LUCJ_PROGRAM_OPTIONS)
        if unsupported:
            raise TypeError(
                f"{type(self).__name__} does not take "
                f"{', '.join(sorted(unsupported))}; its fragment programs "
                f"accept only {', '.join(_LUCJ_PROGRAM_OPTIONS)}. "
                "VQE options need VQEPreparation."
            )

    def _ensemble_sampling_backend(
        self, sampling_backend: CircuitRunner | None
    ) -> CircuitRunner | None:
        return None

    def _build_program(
        self,
        problem: MolecularProblem,
        fragment: FragmentState,
        *,
        backend: CircuitRunner,
        sampling_backend: CircuitRunner | None,
        seed: int,
        options: Mapping[str, Any],
    ) -> LUCJFragmentProgram:
        return LUCJFragmentProgram(
            problem,
            fragment.spec,
            backend=backend,
            sampling_backend=sampling_backend,
            run_linear_method=self._run_linear_method,
            seed=seed,
            **options,
        )

    def _checkpoint_record(self) -> object:
        return self


@dataclass(frozen=True)
class CCSDPreparation(_LUCJPreparation):
    """Sample each fragment's CCSD-seeded LUCJ circuit as it is.

    Classical cost is the fragment's CCSD; one sampling job per fragment per
    macro-cycle. Never builds the fragment statevector, so it suits fragments
    too large for :class:`LinearMethodPreparation`.
    """

    _run_linear_method: ClassVar[bool] = False


@dataclass(frozen=True)
class LinearMethodPreparation(_LUCJPreparation):
    """Optimise the CCSD-seeded LUCJ circuit with ffsim's linear method first.

    The linear method works on the exact fragment statevector, so this is
    limited to fragments small enough to simulate classically. It is the
    preparation arXiv:2512.14936 uses.
    """

    _run_linear_method: ClassVar[bool] = True


@dataclass(frozen=True)
class VQEPreparation:
    """Optimise a fragment ansatz against backend-estimated energies.

    Args:
        optimizer: Optimizer template, deep-copied for each fragment.
        ansatz: Fragment ansatz.
        max_iterations: Optimisation iterations per fragment and round.

    Raises:
        TypeError: If ``ansatz`` is not an :class:`~divi.qprog.algorithms.Ansatz`.
        ValueError: If ``max_iterations`` is below 1.
    """

    optimizer: Optimizer
    ansatz: Ansatz = field(default_factory=UCCSDAnsatz)
    max_iterations: int = 10

    def __post_init__(self):
        if not isinstance(self.ansatz, Ansatz):
            raise TypeError(
                f"ansatz must be an Ansatz instance; got {type(self.ansatz).__name__}."
            )
        if self.max_iterations < 1:
            raise ValueError(
                f"max_iterations must be at least 1; got {self.max_iterations}."
            )

    def _check_options(self, options: Mapping[str, Any]) -> None:
        pass

    def _ensemble_sampling_backend(
        self, sampling_backend: CircuitRunner | None
    ) -> CircuitRunner | None:
        return sampling_backend

    def _build_program(
        self,
        problem: MolecularProblem,
        fragment: FragmentState,
        *,
        backend: CircuitRunner,
        sampling_backend: CircuitRunner | None,
        seed: int,
        options: Mapping[str, Any],
    ) -> _FragmentVQE:
        return build_fragment_vqe(
            problem,
            fragment,
            ansatz=self.ansatz,
            optimizer=self.optimizer,
            max_iterations=self.max_iterations,
            backend=backend,
            seed=seed,
            options=options,
        )

    def _checkpoint_record(self) -> object:
        return (
            type(self).__name__,
            type(self.optimizer).__name__,
            _checkpoint_digest(self.optimizer),
            type(self.ansatz).__name__,
            _checkpoint_digest(self.ansatz),
            self.max_iterations,
        )


Preparation = CCSDPreparation | LinearMethodPreparation | VQEPreparation


@dataclass(frozen=True)
class FullOrbitalSolve:
    """Re-optimise the orbitals to convergence every macro-cycle with L-BFGS-B.

    Args:
        max_iterations: Cap on L-BFGS-B iterations per macro-cycle; ``None``
            leaves it at scipy's default. A capped round returns its best
            orbitals and reports as not converged.

    Raises:
        ValueError: If ``max_iterations`` is given and below 1.
    """

    max_iterations: int | None = None

    def __post_init__(self):
        if self.max_iterations is not None and self.max_iterations < 1:
            raise ValueError(
                f"max_iterations must be at least 1; got {self.max_iterations}."
            )

    def _solve(
        self,
        mol,
        mo_coeff: np.ndarray,
        n_core: int,
        fragment_specs: Sequence[FragmentSpec],
        rdm1_active: np.ndarray,
        rdm2_active: np.ndarray,
        ao_eri: np.ndarray,
        h_ao: np.ndarray,
        *,
        gradient_tol: float,
        report: Callable[[str], None] | None = None,
    ) -> OrbitalSolve:
        return optimize_orbitals(
            mol,
            mo_coeff,
            n_core,
            fragment_specs,
            rdm1_active,
            rdm2_active,
            ao_eri,
            h_ao,
            gradient_tol=gradient_tol,
            max_iterations=self.max_iterations,
            report=report,
        )


@dataclass(frozen=True)
class SecondOrderOrbitalSolve:
    """Re-optimise the orbitals every macro-cycle with PySCF's CIAH solver.

    Each iteration takes an augmented-Hessian step, with Hessian-vector
    products from finite differences of the analytic orbital gradient.

    Args:
        max_iterations: Cap on augmented-Hessian iterations per macro-cycle. A
            capped round returns its best orbitals and reports as not
            converged.

    Raises:
        ValueError: If ``max_iterations`` is below 1.
    """

    max_iterations: int = 50

    def __post_init__(self):
        if self.max_iterations < 1:
            raise ValueError(
                f"max_iterations must be at least 1; got {self.max_iterations}."
            )

    def _solve(
        self,
        mol,
        mo_coeff: np.ndarray,
        n_core: int,
        fragment_specs: Sequence[FragmentSpec],
        rdm1_active: np.ndarray,
        rdm2_active: np.ndarray,
        ao_eri: np.ndarray,
        h_ao: np.ndarray,
        *,
        gradient_tol: float,
        report: Callable[[str], None] | None = None,
    ) -> OrbitalSolve:
        return ciah_orbital_solve(
            mol,
            mo_coeff,
            n_core,
            fragment_specs,
            rdm1_active,
            rdm2_active,
            ao_eri,
            h_ao,
            gradient_tol=gradient_tol,
            max_iterations=self.max_iterations,
            report=report,
        )


OrbitalUpdate = SecondOrderOrbitalSolve | FullOrbitalSolve


@dataclass(frozen=True)
class FragmentationConfig:
    """How the active space is chosen and split into fragments.

    Exactly one of ``active_spaces``, ``n_active_orbitals`` or
    ``active_orbitals`` selects the active space. The first fixes the fragment
    layout outright; the other two select orbitals and leave the split to the
    coupling graph, which ``max_orbitals_per_fragment`` and
    ``coupling_threshold`` shape, or to ``fragment_atoms``.

    Args:
        active_spaces: Explicit fragment layout, one ``FragmentSpec`` per
            fragment. Fixes which orbitals are active, how they partition, and
            each fragment's spin at once, and skips localisation entirely.
        n_active_orbitals: Total active orbitals to select around the HOMO-LUMO
            gap.
        active_orbitals: Explicit MO column indices forming the active space,
            instead of selecting it by energy. Use it when the active space is
            defined by orbital character -- a metal ``d`` manifold can sit well
            below the HOMO with its virtual partners well above the LUMO, where
            no frontier count reaches them. Pair it with a mean field whose
            orbitals already carry that character, e.g. from PySCF's AVAS, and
            with ``fragment_atoms`` to say which centre each belongs to.
        max_orbitals_per_fragment: Maximum spatial orbitals per automatically
            built fragment. Ignored when ``active_spaces`` or ``fragment_atoms``
            is given.
        coupling_threshold: Relative edge-pruning threshold for the orbital
            coupling graph. Ignored when ``active_spaces`` or ``fragment_atoms``
            is given.
        fragment_atoms: One sequence of atom indices per fragment. Assigns each
            localised active orbital to the fragment owning the atom it sits on,
            replacing the coupling-graph clustering -- one fragment per metal
            centre, for instance.
        local_spins: Per-fragment ``2S``, in the order ``fragment_atoms`` names
            them. The fragment's electron count comes from its occupied
            orbitals, so this sets spin alone: ``n_alpha - n_beta = 2S``. Use it
            for spin-polarised fragments, e.g. ``[2, -2]`` for an
            antiferromagnetically coupled dimer of local triplets. The fragments
            must still sum to ``Sz = 0``.

    Raises:
        ValueError: If not exactly one active-space selector is given; if
            ``fragment_atoms`` or ``local_spins`` is combined with
            ``active_spaces``; if ``local_spins`` is given without
            ``fragment_atoms`` or does not match its length; if
            ``n_active_orbitals`` or ``max_orbitals_per_fragment`` is below 2;
            or if ``coupling_threshold`` is negative.
    """

    active_spaces: Sequence[FragmentSpec] | None = None
    n_active_orbitals: int | None = None
    active_orbitals: Sequence[int] | None = None
    max_orbitals_per_fragment: int = 4
    coupling_threshold: float = 1e-3
    fragment_atoms: Sequence[Sequence[int]] | None = None
    local_spins: Sequence[int] | None = None

    def __post_init__(self):
        if self.active_spaces is not None:
            object.__setattr__(self, "active_spaces", tuple(self.active_spaces))
        if self.active_orbitals is not None:
            object.__setattr__(
                self, "active_orbitals", tuple(int(o) for o in self.active_orbitals)
            )
        if self.fragment_atoms is not None:
            object.__setattr__(
                self,
                "fragment_atoms",
                tuple(tuple(int(a) for a in atoms) for atoms in self.fragment_atoms),
            )
        if self.local_spins is not None:
            object.__setattr__(
                self, "local_spins", tuple(int(s) for s in self.local_spins)
            )

        selectors = (self.active_spaces, self.n_active_orbitals, self.active_orbitals)
        if sum(selector is not None for selector in selectors) != 1:
            raise ValueError(
                "Pass exactly one of active_spaces (explicit fragment layout), "
                "n_active_orbitals (frontier selection), or active_orbitals "
                "(explicit MO indices)."
            )
        for name, value in (
            ("local_spins", self.local_spins),
            ("fragment_atoms", self.fragment_atoms),
        ):
            if value is not None and self.active_spaces is not None:
                raise ValueError(
                    f"{name} applies to automatic fragmentation only; "
                    "active_spaces already fixes the fragment layout."
                )
        if self.local_spins is not None:
            if self.fragment_atoms is None:
                raise ValueError(
                    "local_spins requires fragment_atoms: coupling-graph fragment "
                    "order depends on max_orbitals_per_fragment, coupling_threshold "
                    "and the localisation RNG, so a positional spin list would not "
                    "name a stable fragment."
                )
            if len(self.local_spins) != len(self.fragment_atoms):
                raise ValueError(
                    f"local_spins has {len(self.local_spins)} entries but "
                    f"fragment_atoms names {len(self.fragment_atoms)} fragments."
                )
        if self.n_active_orbitals is not None and self.n_active_orbitals < 2:
            raise ValueError(
                "n_active_orbitals must be at least 2, one occupied and one "
                f"virtual; got {self.n_active_orbitals}."
            )
        if self.max_orbitals_per_fragment < 2:
            raise ValueError(
                "max_orbitals_per_fragment must be at least 2, since a fragment "
                "needs an occupied and a virtual orbital; got "
                f"{self.max_orbitals_per_fragment}."
            )
        if self.coupling_threshold < 0:
            raise ValueError(
                "coupling_threshold must be non-negative; got "
                f"{self.coupling_threshold}."
            )
