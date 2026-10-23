# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Immutable state carried through the LASSQD workflow."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from pydantic import field_validator
from pydantic.dataclasses import dataclass as config_dataclass

from ._types import exact_ints

if TYPE_CHECKING:
    from ._integrals import AOIntegrals

# Largest deviation of a checkpointed orbital overlap from the identity. Orbital
# rotations keep it near machine precision; a scaled or edited array does not.
_ORTHONORMALITY_TOL = 1e-8


def require_orthonormal(
    coeff: np.ndarray, label: str, overlap: np.ndarray | None = None
) -> None:
    """Raise unless ``coeff``'s columns are orthonormal under ``overlap``
    (the identity when ``None``)."""
    metric_coeff = coeff if overlap is None else overlap @ coeff
    deviation = float(np.abs(coeff.T @ metric_coeff - np.eye(coeff.shape[1])).max())
    if deviation > _ORTHONORMALITY_TOL:
        raise ValueError(
            f"{label} is not orthonormal: its overlap deviates from the identity "
            f"by {deviation:.1e}."
        )


@config_dataclass(frozen=True)
class FragmentSpec:
    """A single active-space fragment.

    Args:
        orbitals: Indices into the mean field's molecular orbitals (RHF or
            ROHF, or those supplied with it). Occupied orbitals come first, as
            the reference determinant fills them in order.
        n_alpha: Alpha electrons assigned to the fragment. May differ from
            ``n_beta`` for a spin-polarised fragment.
        n_beta: Beta electrons assigned to the fragment.

    Raises:
        ValueError: If ``orbitals`` is a set, empty or contains duplicates, if
            ``n_alpha``/``n_beta`` fall outside ``[0, n_orbitals]``, or if any
            field is not an integer. Whether the spin counts leave an excitation
            available is checked separately, during fragment validation.
    """

    orbitals: tuple[int, ...]
    n_alpha: int
    n_beta: int

    @field_validator("orbitals", mode="before")
    @classmethod
    def _ordered_exact_orbitals(cls, value: Any) -> Any:
        if isinstance(value, (set, frozenset)):
            raise ValueError(
                "orbitals must be an ordered sequence; a set loses the "
                "occupied-first order."
            )
        return exact_ints(value)

    _exact_counts = field_validator("n_alpha", "n_beta", mode="before")(exact_ints)

    def __post_init__(self):
        orbitals = self.orbitals
        if not orbitals:
            raise ValueError("A fragment must contain at least one orbital.")
        if len(set(orbitals)) != len(orbitals):
            raise ValueError(f"Fragment orbitals contain duplicates: {orbitals}.")
        if not 0 <= self.n_alpha <= len(orbitals):
            raise ValueError(
                f"n_alpha must be between 0 and {len(orbitals)}; got {self.n_alpha}."
            )
        if not 0 <= self.n_beta <= len(orbitals):
            raise ValueError(
                f"n_beta must be between 0 and {len(orbitals)}; got {self.n_beta}."
            )

    @property
    def n_orbitals(self) -> int:
        """Number of spatial orbitals in the fragment."""
        return len(self.orbitals)

    @property
    def n_qubits(self) -> int:
        """Register width of this fragment's VQE circuit."""
        return 2 * len(self.orbitals)


@dataclass(frozen=True, eq=False)
class FragmentState:
    """Per-fragment results carried between LASSQD rounds.

    Compares by identity.

    Attributes:
        spec: The fragment's active-space specification.
        rdm1: ``(n_orb, n_orb)`` fragment 1-RDM, in the fragment's own orbital
            ordering.
        rdm2: ``(n_orb,) * 4`` fragment 2-RDM, in the same ordering.
        params: The preparation's parameters from the previous round, or
            ``None`` for a fragment not yet optimised.
        rdm1_alpha: Alpha-spin half of ``rdm1``. ``None`` assumes the
            closed-shell split ``rdm1 / 2``, allowed only for equal spin
            counts.
        rdm1_beta: Beta-spin half of ``rdm1``, under the same convention.
        carried_alpha: Alpha strings carried over from the previous round's
            SQD solve, in the basis of ``sampled_orbitals``.
        carried_beta: Beta strings, likewise.
        sampled_orbitals: ``(nao, n_orb)`` AO coefficients of the basis the
            previous round sampled in, which the carried strings refer to;
            ``None`` when nothing is carried.

    Raises:
        ValueError: If exactly one of ``rdm1_alpha`` and ``rdm1_beta`` is given,
            if neither is given for a fragment with unequal spin counts, or if
            strings are carried without ``sampled_orbitals``.
    """

    spec: FragmentSpec
    rdm1: np.ndarray
    rdm2: np.ndarray
    params: np.ndarray | None = None
    rdm1_alpha: np.ndarray | None = None
    rdm1_beta: np.ndarray | None = None
    carried_alpha: tuple[str, ...] = ()
    carried_beta: tuple[str, ...] = ()
    sampled_orbitals: np.ndarray | None = None

    def __post_init__(self):
        object.__setattr__(self, "carried_alpha", tuple(self.carried_alpha))
        object.__setattr__(self, "carried_beta", tuple(self.carried_beta))
        if (self.rdm1_alpha is None) != (self.rdm1_beta is None):
            raise ValueError(
                "rdm1_alpha and rdm1_beta must be given together or not at all."
            )
        if self.rdm1_alpha is None and self.spec.n_alpha != self.spec.n_beta:
            raise ValueError(
                f"A spin-polarised fragment ({self.spec.n_alpha} alpha, "
                f"{self.spec.n_beta} beta) needs rdm1_alpha and rdm1_beta."
            )
        if (self.carried_alpha or self.carried_beta) and self.sampled_orbitals is None:
            raise ValueError(
                "Carried strings need sampled_orbitals, the basis they refer to."
            )

    def spin_rdm1s(self) -> tuple[np.ndarray, np.ndarray]:
        """``(alpha, beta)`` 1-RDM halves, splitting ``rdm1`` if not supplied."""
        if self.rdm1_alpha is None or self.rdm1_beta is None:
            half = self.rdm1 / 2.0
            return half, half.copy()
        return self.rdm1_alpha, self.rdm1_beta


@dataclass(frozen=True, eq=False)
class LASSQDState:
    """Workflow state for one LASSQD macro-cycle.

    Compares by identity.

    Attributes:
        mo_coeff: ``(nao, n_orb)`` molecular-orbital coefficients, permuted
            into ``[core | fragment blocks | virtual]`` order.
        fragments: Per-fragment state, in the order of the fragment blocks in
            ``mo_coeff``.
        energy: The energy the macro-cycle converges on, ``inf`` until
            computed: the total energy, or under
            :class:`~divi.qprog.workflows.lassqd.StateAveragedOrbitalUpdate` the
            weighted average over the requested states.
        previous_energy: The previous macro-cycle's ``energy``, ``inf`` for
            the initial state.
        orbitals_converged: Whether the orbital optimisation that produced
            ``mo_coeff`` converged; ``True`` for an initial state.
    """

    mo_coeff: np.ndarray
    fragments: tuple[FragmentState, ...]
    energy: float = float("inf")
    previous_energy: float = float("inf")
    orbitals_converged: bool = True


@dataclass(frozen=True, order=True)
class ElectronicState:
    """One eigenstate requested through ``SQDConfig.states``, after a round.

    Orders by ``(multiplicity, index)``.

    Attributes:
        multiplicity: Spin multiplicity ``2S + 1`` the state was solved for.
        index: Position among that multiplicity's roots in energy order at the
            end of the first round, ``0`` the lowest. Later rounds keep it with
            the same state, matched by CI-vector overlap, so it no longer
            follows energy order after two states cross. States across two
            fragments are not matched: their index is each round's energy
            order.
        energy: Total energy of the state at the round's orbitals.
        spin_square: ``<S^2>`` of the state.
        charge_transfer: With states across two fragments, the state's weight
            on products in which an electron has moved between the fragments:
            near 0 for a local state, near 1 for a charge-transfer state.
            Either direction counts, and between bonded fragments so does
            covalent sharing. ``None`` for a single fragment's states.
    """

    multiplicity: int
    index: int
    energy: float
    spin_square: float
    charge_transfer: float | None = field(default=None, compare=False)


@dataclass(frozen=True, eq=False)
class StateWavefunction:
    """One requested state's wave function at the final round's orbitals.

    Compares by identity.

    Attributes:
        state: The state's label, energy and ``<S^2>``.
        core_orbitals: ``(nao, n_core)`` AO coefficients of the doubly
            occupied orbitals outside the active space.
        active_orbitals: ``(nao, n_act)`` AO coefficients of the active
            orbitals the strings, amplitudes and RDMs refer to.
        strings_alpha: Alpha occupation strings indexing ``amplitudes``' rows;
            character ``p`` is orbital ``p``'s occupation.
        strings_beta: Beta occupation strings indexing its columns.
        amplitudes: CI coefficients over the product of the two string lists,
            with PySCF's determinant phase: reading character ``p`` as bit
            ``p``, ``pyscf.fci.cistring.str2addr`` places each amplitude in a
            full-space PySCF FCI vector. A state of higher spin is its
            ``M_S = (n_alpha - n_beta) / 2`` component.
        rdm1: ``(n_act, n_act)`` spin-traced 1-RDM, in PySCF's convention.
        rdm2: ``(n_act,) * 4`` spin-traced 2-RDM, in PySCF's chemist-order
            convention.
    """

    state: ElectronicState
    core_orbitals: np.ndarray
    active_orbitals: np.ndarray
    strings_alpha: tuple[str, ...]
    strings_beta: tuple[str, ...]
    amplitudes: np.ndarray
    rdm1: np.ndarray
    rdm2: np.ndarray

    def leading_determinants(
        self, count: int = 5
    ) -> tuple[tuple[str, str, float], ...]:
        """The ``count`` determinants of largest weight, as ``(alpha, beta,
        coefficient)``, by descending ``coefficient ** 2``; all of them when
        ``count`` exceeds the subspace.

        Raises:
            ValueError: If ``count`` is below 1.
        """
        if count < 1:
            raise ValueError(f"count must be at least 1; got {count}.")
        order = np.argsort(-np.abs(self.amplitudes), axis=None, kind="stable")[:count]
        return tuple(
            (
                self.strings_alpha[row],
                self.strings_beta[column],
                float(self.amplitudes[row, column]),
            )
            for row, column in zip(*np.unravel_index(order, self.amplitudes.shape))
        )


@dataclass(frozen=True, eq=False)
class StateTransition:
    """The one-electron transition between two requested states of one spin.

    Compares by identity.

    Attributes:
        initial: The lower state.
        final: The higher state.
        transition_rdm1: ``(n_act, n_act)`` spin-traced transition 1-RDM,
            ``<final| a_q^+ a_p |initial>``, over the same active orbitals as
            the states' :class:`StateWavefunction`.
        transition_dipole: Electronic transition dipole, in atomic units. Its
            sign follows the states' arbitrary phases.
        oscillator_strength: ``2/3 * (E_final - E_initial) * |dipole|^2``.
        hole_orbitals: ``(nao, n_act)`` natural transition orbitals the
            electron leaves, ordered as ``nto_weights``.
        particle_orbitals: ``(nao, n_act)`` natural transition orbitals the
            electron enters, paired column by column with ``hole_orbitals``.
        nto_weights: Each natural-transition-orbital pair's share of the
            transition, descending and summing to 1; all zero when no
            one-electron move connects the states.
    """

    initial: ElectronicState
    final: ElectronicState
    transition_rdm1: np.ndarray
    transition_dipole: np.ndarray
    oscillator_strength: float
    hole_orbitals: np.ndarray
    particle_orbitals: np.ndarray
    nto_weights: np.ndarray


_SaddleCheck = Literal["escaped", "passed", "inconclusive"]


@dataclass(frozen=True)
class LASSQDRoundReport:
    """One macro-cycle round's stage outputs.

    Appended to ``LASSQD.round_reports`` as the round completes, so an
    interrupted run keeps every finished round's numbers.

    Attributes:
        number: Round number, counting from 1.
        energy: This round's total energy of the lowest state. In gas phase an
            upper bound (see :ref:`lassqd-accuracy-characteristics`); with a
            solvent it includes the solvation energy, the first round's in no
            field yet.
        energy_change: Signed change from the previous round's ``energy``,
            ``None`` for the first round.
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
        states: The states ``SQDConfig.states`` requested, ascending in
            energy, with states within 1e-6 Ha of each other in
            ``(multiplicity, index)`` order; empty when it requested none.
        averaged_energy: The weighted average over ``states`` that
            :class:`StateAveragedOrbitalUpdate` minimised and the macro-cycle
            converges on; ``None`` under any other orbital update.
        saddle_check: Outcome of the orbital-Hessian check run when a
            lowest-state round would end the macro-cycle: ``'escaped'`` (a
            saddle point was left; the macro-cycle continues), ``'passed'``
            or ``'inconclusive'`` (the eigensolver stopped unconverged, so the
            orbitals may still be a saddle point). ``None`` when it did not run.
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
    states: tuple[ElectronicState, ...] = ()
    averaged_energy: float | None = None
    saddle_check: _SaddleCheck | None = None

    def summary(self) -> str:
        """One-line human-readable digest of the round."""
        change = (
            "first round"
            if self.energy_change is None
            else f"change {self.energy_change:+.3e}"
        )
        parts = [
            f"Round {self.number} done. Energy: {self.energy:.8f} Ha ({change})",
            f"subspaces {list(self.subspace_sizes)}",
            self._orbital_summary(),
            f"SQD {self.recovery_seconds:.1f}s, orbitals {self.orbital_seconds:.1f}s",
        ]
        parts += [
            f"state ({state.multiplicity}, {state.index}) {state.energy:.8f} Ha, "
            f"<S^2> {state.spin_square:.3f}"
            for state in self.states
        ]
        if self.averaged_energy is not None:
            parts.append(f"state-averaged {self.averaged_energy:.8f} Ha")
        if self.saddle_check is not None:
            parts.append(f"saddle check {self.saddle_check}")
        return "; ".join(parts)

    def _orbital_summary(self) -> str:
        if self.rotation_pairs == 0:
            return "orbitals: none rotated"
        return (
            f"orbitals: {self.orbital_iterations} iterations over "
            f"{self.rotation_pairs} pairs, |g| {self.orbital_gradient_norm:.2e}, "
            f"converged={self.orbital_converged}"
        )


def _core_orbital_count(n_electrons: int, n_active_electrons: int) -> int:
    """Doubly occupied core orbitals holding the electrons outside the active
    space, which is closed-shell."""
    return (n_electrons - n_active_electrons) // 2


@dataclass(frozen=True)
class _SampledSubspace:
    """Fragment 0's SQD subspace, for an update that re-solves its CI."""

    mo_coeff: np.ndarray  # the round's orbitals, fragment 0 in its sampled basis
    strings: tuple[Sequence[str], Sequence[str]]
    states: Mapping[int, int]
    lambda_penalty: float
    # Each root's tracked label, (multiplicity, root index) -> (multiplicity, index).
    labels: Mapping[tuple[int, int], tuple[int, int]] | None = None


@dataclass(frozen=True)
class _OrbitalRound:
    """What a round hands its orbital update."""

    ao: "AOIntegrals"
    mo_coeff: np.ndarray
    n_core: int
    specs: Sequence[FragmentSpec]
    rdm1_active: np.ndarray
    rdm2_active: np.ndarray
    gradient_tol: float
    report: Callable[[str], None] | None = None
    subspace: _SampledSubspace | None = None


def validate_fragment_specs(
    specs: Sequence[FragmentSpec],
    n_orbitals_total: int,
    n_occupied: int,
    *,
    spin: int = 0,
) -> None:
    """Reject fragments that overlap, index outside the orbital register, or
    carry the wrong electron count.

    Fragments must be disjoint because effective fragment integrals sum every
    *other* fragment's mean-field contribution. Their electron counts must add
    up to those of the active orbitals below ``n_occupied`` (two per doubly
    occupied orbital, one per singly occupied one), as a mismatch yields an
    energy for the wrong number of electrons. Each fragment must list its
    occupied orbitals before its virtual ones, as the reference determinant and
    initial density fill its orbitals in the order given.

    Args:
        specs: The fragment specifications to validate.
        n_orbitals_total: Size of the molecule's orbital register.
        n_occupied: Number of occupied orbitals in the reference determinant,
            doubly and singly occupied together.
        spin: The molecule's ``2S`` (``n_alpha - n_beta``); its ``|spin|``
            highest occupied orbitals are singly occupied, as in ROHF.

    Raises:
        ValueError: If any orbital index is out of range or shared between
            fragments, if ``specs`` is empty, if a fragment leaves every spin
            channel empty or full (no excitation available, so no correlation to
            capture), if a fragment lists a virtual orbital before an occupied
            one, if a singly occupied orbital is left out of every fragment, if
            the fragments' total electron count does not match the orbitals
            they cover, or if the fragments do not sum to the molecule's
            ``Sz``. Per-fragment spin-count bounds are enforced by
            :class:`FragmentSpec` itself.
    """
    if not specs:
        raise ValueError("At least one fragment is required.")

    seen: dict[int, int] = {}
    for index, spec in enumerate(specs):
        if not any(
            0 < count < spec.n_orbitals for count in (spec.n_alpha, spec.n_beta)
        ):
            raise ValueError(
                f"Fragment {index} (orbitals {spec.orbitals}) has no excitation "
                f"available: n_alpha={spec.n_alpha}, n_beta={spec.n_beta} leave "
                f"every spin channel of its {spec.n_orbitals} orbitals either "
                "empty or full."
            )
        for orbital in spec.orbitals:
            if not 0 <= orbital < n_orbitals_total:
                raise ValueError(
                    f"Fragment {index} orbital {orbital} is out of range for a "
                    f"molecule with {n_orbitals_total} orbitals."
                )
            if orbital in seen:
                raise ValueError(
                    f"Fragments {seen[orbital]} and {index} overlap on orbital "
                    f"{orbital}. Fragments must be disjoint."
                )
            seen[orbital] = index
        virtual = next((o for o in spec.orbitals if o >= n_occupied), None)
        if virtual is not None:
            position = spec.orbitals.index(virtual)
            occupied = next(
                (o for o in spec.orbitals[position:] if o < n_occupied), None
            )
            if occupied is not None:
                raise ValueError(
                    f"Fragment {index} lists virtual orbital {virtual} before "
                    f"occupied orbital {occupied}. List each fragment's occupied "
                    "orbitals first: its reference determinant fills them in order."
                )

    n_doubly = n_occupied - abs(spin)
    missing = [
        orbital for orbital in range(n_doubly, n_occupied) if orbital not in seen
    ]
    if missing:
        raise ValueError(
            f"Singly occupied orbitals {missing} are in no fragment; the core "
            "must be closed-shell, so all of them must be active."
        )
    n_held = sum(
        2 if orbital < n_doubly else 1 for orbital in seen if orbital < n_occupied
    )
    n_declared = sum(spec.n_alpha + spec.n_beta for spec in specs)
    if n_declared != n_held:
        raise ValueError(
            f"Fragments declare {n_declared} electrons but cover occupied "
            f"orbitals holding {n_held}."
        )

    total_alpha = sum(spec.n_alpha for spec in specs)
    total_beta = sum(spec.n_beta for spec in specs)
    if total_alpha - total_beta != spin:
        raise ValueError(
            f"Fragments declare {total_alpha} alpha and {total_beta} beta "
            f"electrons, a total Sz of {(total_alpha - total_beta) / 2}, but the "
            f"molecule's is {spin / 2}. Individual fragments may be polarised; "
            "their total must match."
        )
