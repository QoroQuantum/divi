# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Immutable state carried through the LASSQD workflow."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from pydantic import field_validator
from pydantic.dataclasses import dataclass as config_dataclass

from ._types import exact_ints

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
        orbitals: Canonical RHF molecular-orbital indices, in energy order —
            not the caller's own arbitrary numbering. Occupied orbitals come
            first, since the reference determinant fills them in order.
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

    Compares by identity, not value: the numpy fields make a value-based
    ``__eq__`` raise.

    Attributes:
        spec: The fragment's active-space specification.
        rdm1: ``(n_orb, n_orb)`` fragment 1-RDM, in the fragment's own local
            orbital ordering.
        rdm2: ``(n_orb,) * 4`` fragment 2-RDM, in the same ordering as
            ``rdm1``.
        params: The fragment preparation's parameters from the previous
            round, or ``None`` for a fragment that has not been optimised
            yet (e.g. a freshly built initial state).
        rdm1_alpha: Alpha-spin half of ``rdm1``, or ``None`` to assume the
            closed-shell split ``rdm1 / 2``, which only a fragment with equal
            spin counts may do. Needed for the cross-fragment exchange term,
            which contracts same-spin densities.
        rdm1_beta: Beta-spin half of ``rdm1``, under the same convention.
        carried_alpha: Alpha strings carryover retained from the previous
            round's SQD solve, in the basis of ``sampled_orbitals``.
        carried_beta: Beta strings, likewise.
        sampled_orbitals: ``(nao, n_orb)`` AO coefficients of the orbital basis
            the previous round sampled and diagonalised in, which the carried
            strings refer to; ``None`` when nothing is carried.

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

    Compares by identity, not value: the numpy fields make a value-based
    ``__eq__`` raise.

    Attributes:
        mo_coeff: ``(nao, n_orb)`` molecular-orbital coefficients, permuted
            into ``[core | fragment blocks | virtual]`` order.
        fragments: Per-fragment state, in the same order as the fragment
            blocks in ``mo_coeff``.
        energy: The energy the macro-cycle converges on, or ``inf`` if not yet
            computed: the total energy, or under
            :class:`~divi.qprog.workflows.StateAveragedOrbitalUpdate` the
            weighted average over the requested states.
        previous_energy: The previous macro-cycle's ``energy``, or ``inf`` for
            the initial state.
        orbitals_converged: Whether the orbital optimisation that produced
            ``mo_coeff`` converged. ``True`` for an initial state, which has no
            solve behind it.
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
            the same state, matched by CI-vector overlap, so after two states
            cross it no longer follows energy order.
        energy: Total energy of the state at the round's orbitals, from its
            own RDMs.
        spin_square: ``<S^2>`` of the state.
    """

    multiplicity: int
    index: int
    energy: float
    spin_square: float


@dataclass(frozen=True, eq=False)
class StateWavefunction:
    """One requested state's wave function at the final round's orbitals.

    Compares by identity, not value: the numpy fields make a value-based
    ``__eq__`` raise.

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

    Compares by identity, not value: the numpy fields make a value-based
    ``__eq__`` raise.

    Attributes:
        initial: The lower state.
        final: The higher state.
        transition_rdm1: ``(n_act, n_act)`` spin-traced transition 1-RDM,
            ``<final| a_p^+ a_q |initial>``, over the same active orbitals as
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

    mol: Any
    mo_coeff: np.ndarray
    n_core: int
    specs: Sequence[FragmentSpec]
    rdm1_active: np.ndarray
    rdm2_active: np.ndarray
    ao_eri: np.ndarray
    h_ao: np.ndarray
    gradient_tol: float
    report: Callable[[str], None] | None = None
    subspace: _SampledSubspace | None = None


def validate_fragment_specs(
    specs: Sequence[FragmentSpec], n_orbitals_total: int, n_occupied: int
) -> None:
    """Reject fragments that overlap or index outside the orbital register.

    Non-overlap matters beyond hygiene: effective fragment integrals sum the
    mean-field contribution of every *other* fragment, so shared orbitals
    would be double-counted.

    The fragments' electron counts must also add up to twice the number of
    active orbitals below ``n_occupied``. A mismatch is not caught downstream;
    it yields an energy for the wrong number of electrons.

    Args:
        specs: The fragment specifications to validate.
        n_orbitals_total: Size of the molecule's orbital register.
        n_occupied: Number of doubly occupied orbitals in the reference
            determinant (``mol.nelectron // 2``).

    Each fragment must list its occupied orbitals before its virtual ones: the
    reference determinant and the initial density fill a fragment's orbitals in
    the order given.

    Raises:
        ValueError: If any orbital index is out of range or shared between
            fragments, if ``specs`` is empty, if a fragment leaves every spin
            channel empty or full (no excitation available, so no correlation to
            capture), if a fragment lists a virtual orbital before an occupied
            one, if the fragments' total electron count does not match the
            orbitals they cover, or if the fragments do not sum to ``Sz = 0``.
            Per-fragment spin-count bounds are enforced by
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
                "empty or full, so there is no correlation for this fragment to "
                "capture."
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

    n_active_occupied = sum(1 for orbital in seen if orbital < n_occupied)
    n_declared = sum(spec.n_alpha + spec.n_beta for spec in specs)
    if n_declared != 2 * n_active_occupied:
        raise ValueError(
            f"Fragments declare {n_declared} electrons but cover "
            f"{n_active_occupied} occupied orbitals, which hold "
            f"{2 * n_active_occupied}."
        )

    total_alpha = sum(spec.n_alpha for spec in specs)
    total_beta = sum(spec.n_beta for spec in specs)
    if total_alpha != total_beta:
        raise ValueError(
            f"Fragments declare {total_alpha} alpha and {total_beta} beta "
            f"electrons, a total Sz of {(total_alpha - total_beta) / 2}. Only "
            "closed-shell molecules are supported, so the fragments must sum "
            "to Sz = 0 even where individual fragments are polarised."
        )
