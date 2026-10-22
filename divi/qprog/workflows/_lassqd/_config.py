# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Configuration objects for the LASSQD workflow."""

import hashlib
import math
import pickle
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import field
from enum import Enum
from typing import Any, ClassVar

from pydantic import Field, field_validator
from pydantic.dataclasses import dataclass as config_dataclass

from divi.backends import CircuitRunner
from divi.qprog.algorithms import Ansatz, UCCSDAnsatz
from divi.qprog.optimizers import Optimizer
from divi.qprog.problems import MolecularProblem
from divi.qprog.quantum_program import QuantumProgram

from ._integrals import (
    OrbitalSolution,
    _total_energy,
    ciah_orbital_solve,
    optimize_orbitals,
    state_averaged_orbital_solve,
)
from ._preparation import LUCJFragmentProgram
from ._state import ElectronicState, FragmentSpec, FragmentState, _OrbitalRound
from ._types import CONFIG, exact_ints
from ._vqe_preparation import _FragmentVQE, build_fragment_vqe

#: How far state-averaging weights may sum from 1, absorbing rounding in
#: hand-typed decimals.
_WEIGHT_SUM_ATOL = 1e-8


class _OrbitalTarget(Enum):
    """What an orbital update relaxes the orbitals for."""

    NOTHING = "nothing"
    LOWEST_STATE = "lowest state"
    AVERAGE = "average"


def _checkpoint_digest(value: object) -> str:
    """Stable digest of a configuration template without object addresses."""
    return hashlib.sha256(pickle.dumps(value, protocol=4)).hexdigest()


# Program options an LUCJ fragment takes, beyond those LASSQD sets itself.
_LUCJ_PROGRAM_OPTIONS = (
    "precision",
    "qem_protocol",
    "suppress_performance_warnings",
)


class Preparation(ABC):
    """How each fragment's circuit is prepared before it is sampled.

    Pass one of :class:`CCSDPreparation`, :class:`LinearMethodPreparation` or
    :class:`VQEPreparation` as ``LASSQD``'s ``preparation``.
    """

    @abstractmethod
    def _check_options(self, options: Mapping[str, Any]) -> None:
        """Reject program options LASSQD received that this preparation's
        fragment programs do not take."""

    def _ensemble_sampling_backend(
        self, sampling_backend: CircuitRunner | None
    ) -> CircuitRunner | None:
        """The sampling backend the ensemble itself should run with."""
        return sampling_backend

    @abstractmethod
    def _build_program(
        self,
        problem: MolecularProblem,
        fragment: FragmentState,
        *,
        backend: CircuitRunner,
        sampling_backend: CircuitRunner | None,
        seed: int,
        options: Mapping[str, Any],
    ) -> QuantumProgram:
        """One fragment's program for this round."""

    @abstractmethod
    def _checkpoint_record(self) -> object:
        """What a checkpoint compares to tell whether this preparation changed."""


@config_dataclass(frozen=True)
class _LUCJPreparation(Preparation):
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


@config_dataclass(frozen=True)
class CCSDPreparation(_LUCJPreparation):
    """Sample each fragment's CCSD-seeded LUCJ circuit as it is.

    Classical cost is the fragment's CCSD; one sampling job per fragment per
    macro-cycle. Never builds the fragment statevector, so it suits fragments
    too large for :class:`LinearMethodPreparation`.
    """

    _run_linear_method: ClassVar[bool] = False


@config_dataclass(frozen=True)
class LinearMethodPreparation(_LUCJPreparation):
    """Optimise the CCSD-seeded LUCJ circuit with ffsim's linear method first.

    The linear method works on the exact fragment statevector, so this is
    limited to fragments small enough to simulate classically. It is the
    preparation arXiv:2512.14936 uses.
    """

    _run_linear_method: ClassVar[bool] = True


@config_dataclass(frozen=True, config=CONFIG)
class VQEPreparation(Preparation):
    """Optimise a fragment ansatz against backend-estimated energies.

    Args:
        optimizer: Optimizer template, deep-copied for each fragment.
        ansatz: Fragment ansatz.
        max_iterations: Optimisation iterations per fragment and round.

    Raises:
        ValueError: If ``optimizer`` is not an
            :class:`~divi.qprog.optimizers.Optimizer`, ``ansatz`` is not an
            :class:`~divi.qprog.algorithms.Ansatz`, or ``max_iterations`` is not
            an integer of at least 1.
    """

    optimizer: Optimizer
    ansatz: Ansatz = field(default_factory=UCCSDAnsatz)
    max_iterations: int = Field(default=10, ge=1)

    _exact_ints = field_validator("max_iterations", mode="before")(exact_ints)

    def _check_options(self, options: Mapping[str, Any]) -> None:
        reserved = set(options) & {
            "ansatz",
            "optimizer",
            "max_iterations",
            "seed_params",
        }
        if reserved:
            raise TypeError(
                f"{', '.join(sorted(reserved))} cannot be passed to LASSQD with "
                "VQEPreparation: ansatz, optimizer and max_iterations are set on "
                "VQEPreparation itself, and seed_params comes from each fragment's "
                "CCSD seed."
            )

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


class OrbitalUpdate(ABC):
    """How each macro-cycle updates the orbitals.

    Pass one of :class:`SecondOrderOrbitalUpdate`, :class:`LBFGSOrbitalUpdate`,
    :class:`StateAveragedOrbitalUpdate` or :class:`FixedOrbitals` as
    ``LASSQD``'s ``orbital_update``.
    """

    #: The state the update relaxes the orbitals for.
    _relaxes: ClassVar[_OrbitalTarget]

    @abstractmethod
    def _solve(
        self, round_: _OrbitalRound
    ) -> tuple[OrbitalSolution, tuple[ElectronicState, ...] | None]:
        """The round's new orbitals, and each requested state when the update
        re-solves the CI with them."""


@config_dataclass(frozen=True)
class _LowestStateOrbitalUpdate(OrbitalUpdate):
    """Shared behaviour of the updates that optimise the orbitals for the
    round's assembled RDMs."""

    max_iterations: int | None
    _relaxes: ClassVar[_OrbitalTarget] = _OrbitalTarget.LOWEST_STATE
    _solver: ClassVar[Callable[..., OrbitalSolution]]

    _exact_ints = field_validator("max_iterations", mode="before")(exact_ints)

    def _solve(self, round_: _OrbitalRound) -> tuple[OrbitalSolution, None]:
        solution = self._solver(
            round_.mol,
            round_.mo_coeff,
            round_.n_core,
            round_.specs,
            round_.rdm1_active,
            round_.rdm2_active,
            round_.ao_eri,
            round_.h_ao,
            gradient_tol=round_.gradient_tol,
            max_iterations=self.max_iterations,
            report=round_.report,
        )
        return solution, None


@config_dataclass(frozen=True)
class LBFGSOrbitalUpdate(_LowestStateOrbitalUpdate):
    """Re-optimise the orbitals to convergence every macro-cycle with L-BFGS-B.

    Args:
        max_iterations: Cap on L-BFGS-B iterations per macro-cycle; ``None``
            leaves it at scipy's default. A capped round returns its best
            orbitals and reports as not converged.

    Raises:
        ValueError: If ``max_iterations`` is given and is not an integer of at
            least 1.
    """

    max_iterations: int | None = Field(default=None, ge=1)
    _solver = staticmethod(optimize_orbitals)


@config_dataclass(frozen=True)
class SecondOrderOrbitalUpdate(_LowestStateOrbitalUpdate):
    """Re-optimise the orbitals every macro-cycle with PySCF's CIAH solver.

    Each iteration takes an augmented-Hessian step, with Hessian-vector
    products from finite differences of the analytic orbital gradient.

    Args:
        max_iterations: Cap on augmented-Hessian iterations per macro-cycle. A
            capped round returns its best orbitals and reports as not
            converged.

    Raises:
        ValueError: If ``max_iterations`` is not an integer of at least 1.
    """

    max_iterations: int = Field(default=50, ge=1)
    _solver = staticmethod(ciah_orbital_solve)


@config_dataclass(frozen=True)
class StateAveragedOrbitalUpdate(OrbitalUpdate):
    """Re-optimise the orbitals for a weighted average of the states
    ``SQDConfig.states`` requests: state-averaged CASSCF with its CI confined
    to the round's SQD subspace.

    Orbitals and CI vectors are optimised together (PySCF's one-step solver),
    since alternating them can stall at a saddle point of the averaged energy.
    Each round's ``energy`` is still the lowest state's; the average is its
    report's ``averaged_energy``, and every state's own energy is in
    ``LASSQD.electronic_states``. Needs a single fragment.

    Args:
        max_iterations: Cap on SA-CASSCF iterations in each LASSQD round.
        weights: Weight per state, keyed ``(multiplicity, index)``, covering
            every requested state and summing to 1. ``None`` (the default)
            weights each requested root equally, so a triplet counts once, not
            once per spin component. The weights shape the shared orbitals,
            and so every gap computed from them.

    Raises:
        ValueError: If ``max_iterations`` is not an integer of at least 1, or a
            weight is negative or non-finite, or the weights do not sum to 1.
    """

    max_iterations: int = Field(default=50, ge=1)
    weights: Mapping[tuple[int, int], float] | None = field(default=None, hash=False)
    _relaxes: ClassVar[_OrbitalTarget] = _OrbitalTarget.AVERAGE

    _exact_ints = field_validator("max_iterations", "weights", mode="before")(
        exact_ints
    )

    def __post_init__(self):
        if self.weights is None:
            return
        if not all(0 <= weight < math.inf for weight in self.weights.values()):
            raise ValueError(
                "weights must be finite and non-negative; got " f"{dict(self.weights)}."
            )
        total = sum(self.weights.values())
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=_WEIGHT_SUM_ATOL):
            raise ValueError(f"weights must sum to 1; got {total}.")
        object.__setattr__(self, "weights", dict(sorted(self.weights.items())))

    def _weights_for(self, states: Mapping[int, int]) -> dict[tuple[int, int], float]:
        """Each requested state's weight, keyed ``(multiplicity, index)``.

        Raises:
            ValueError: If ``weights`` does not name exactly the requested
                states.
        """
        labels = [
            (multiplicity, index)
            for multiplicity, n_roots in states.items()
            for index in range(n_roots)
        ]
        if self.weights is None:
            return {label: 1.0 / len(labels) for label in labels}
        if set(self.weights) != set(labels):
            raise ValueError(
                "StateAveragedOrbitalUpdate.weights must name exactly the states "
                f"SQDConfig.states requests, {sorted(labels)}; got "
                f"{sorted(self.weights)}."
            )
        return dict(self.weights)

    def _solve(
        self, round_: _OrbitalRound
    ) -> tuple[OrbitalSolution, tuple[ElectronicState, ...]]:
        subspace = round_.subspace
        if subspace is None:
            raise RuntimeError("StateAveragedOrbitalUpdate needs SQDConfig.states.")
        return state_averaged_orbital_solve(
            round_.mol,
            subspace.mo_coeff,
            round_.n_core,
            round_.specs[0],
            subspace.strings,
            _by_root(self._weights_for(subspace.states), subspace.labels or {}),
            subspace.lambda_penalty,
            round_.ao_eri,
            round_.h_ao,
            gradient_tol=round_.gradient_tol,
            max_iterations=self.max_iterations,
            report=round_.report,
        )


def _by_root(
    weights: Mapping[tuple[int, int], float],
    labels: Mapping[tuple[int, int], tuple[int, int]],
) -> dict[tuple[int, int], float]:
    """``weights``, keyed by tracked label, rekeyed by root: each tracked root
    takes its state's weight, and the rest pair with the unused labels in
    order."""
    by_root = {}
    for multiplicity in {multiplicity for multiplicity, _ in weights}:
        indices = sorted(index for m, index in weights if m == multiplicity)
        label_of = {
            root: label for (m, root), (_, label) in labels.items() if m == multiplicity
        }
        label_of.update(
            zip(
                [index for index in indices if index not in label_of],
                [index for index in indices if index not in label_of.values()],
            )
        )
        for root, label in label_of.items():
            by_root[(multiplicity, root)] = weights[(multiplicity, label)]
    return by_root


@config_dataclass(frozen=True)
class FixedOrbitals(OrbitalUpdate):
    """Keep the orbitals each round starts from, with no optimisation.

    Each round's energy is then that of the recovered state on fixed orbitals,
    as a CASCI calculation would give. The orbital gradient is not computed, so
    round reports carry it as ``nan``.
    """

    _relaxes: ClassVar[_OrbitalTarget] = _OrbitalTarget.NOTHING

    def _solve(self, round_: _OrbitalRound) -> tuple[OrbitalSolution, None]:
        energy = _total_energy(
            round_.mol,
            round_.mo_coeff,
            round_.n_core,
            round_.rdm1_active,
            round_.rdm2_active,
            round_.ao_eri,
            round_.h_ao,
        )
        solution = OrbitalSolution(
            mo_coeff=round_.mo_coeff,
            energy=energy,
            converged=True,
            n_iterations=0,
            n_evaluations=1,
            gradient_norm=float("nan"),
            n_rotation_pairs=0,
        )
        return solution, None


@config_dataclass(frozen=True)
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
        max_orbitals_per_fragment: Maximum spatial orbitals per coupling-graph
            fragment; ``None`` means 4. Rejected with ``active_spaces`` or
            ``fragment_atoms``, which fix the layout themselves.
        coupling_threshold: Relative edge-pruning threshold for the orbital
            coupling graph, as a fraction of the strongest edge weight in
            ``[0, 1]``; ``None`` means ``1e-3``. Rejected with
            ``active_spaces`` or ``fragment_atoms``.
        fragment_atoms: One sequence of atom indices per fragment. Assigns each
            localised active orbital to the fragment owning the atom it sits on,
            replacing the coupling-graph clustering -- one fragment per metal
            centre, for instance.
        local_spins: Per-fragment ``2S``, in the order ``fragment_atoms`` names
            them. The fragment's electron count comes from its occupied
            orbitals, so this sets spin alone: ``n_alpha - n_beta = 2S``. Use it
            for spin-polarised fragments, e.g. ``[2, -2]`` for an
            antiferromagnetically coupled dimer of local triplets. They must
            sum to the molecule's ``2S`` (``mol.spin``), checked by ``LASSQD``.

    Raises:
        ValueError: If a field has the wrong type or is out of range:
            ``n_active_orbitals`` or ``max_orbitals_per_fragment`` below 2, a
            ``coupling_threshold`` outside ``[0, 1]``, or an ``active_spaces``
            entry that is not a ``FragmentSpec``. Also if not exactly one
            active-space selector is given; if ``fragment_atoms`` or
            ``local_spins`` is combined with ``active_spaces``; if
            ``max_orbitals_per_fragment`` or ``coupling_threshold`` is combined
            with ``active_spaces`` or ``fragment_atoms``; or if ``local_spins``
            is given without ``fragment_atoms`` or does not match its length.
    """

    active_spaces: tuple[FragmentSpec, ...] | None = None
    n_active_orbitals: int | None = Field(default=None, ge=2)
    active_orbitals: tuple[int, ...] | None = None
    max_orbitals_per_fragment: int | None = Field(default=None, ge=2)
    coupling_threshold: float | None = Field(
        default=None, ge=0, le=1, allow_inf_nan=False
    )
    fragment_atoms: tuple[tuple[int, ...], ...] | None = None
    local_spins: tuple[int, ...] | None = None

    _exact_ints = field_validator(
        "n_active_orbitals",
        "active_orbitals",
        "max_orbitals_per_fragment",
        "fragment_atoms",
        "local_spins",
        mode="before",
    )(exact_ints)

    def __post_init__(self):
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
        if self.active_spaces is not None or self.fragment_atoms is not None:
            layout = (
                "active_spaces" if self.active_spaces is not None else "fragment_atoms"
            )
            for name, value in (
                ("max_orbitals_per_fragment", self.max_orbitals_per_fragment),
                ("coupling_threshold", self.coupling_threshold),
            ):
                if value is not None:
                    raise ValueError(
                        f"{name} shapes coupling-graph fragments only; {layout} "
                        "already fixes the fragment layout."
                    )
