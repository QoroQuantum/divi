# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Immutable state carried through the LASSQD workflow."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FragmentSpec:
    """A single active-space fragment.

    Args:
        orbitals: Canonical RHF molecular-orbital indices, in energy order —
            not the caller's own arbitrary numbering.
        n_alpha: Alpha electrons assigned to the fragment. May differ from
            ``n_beta`` for a spin-polarised fragment.
        n_beta: Beta electrons assigned to the fragment.

    Raises:
        ValueError: If ``orbitals`` is empty or contains duplicates, or if
            ``n_alpha``/``n_beta`` fall outside ``[0, n_orbitals]``. Whether the
            spin counts leave an excitation available is checked separately,
            during fragment validation.
    """

    orbitals: tuple[int, ...]
    n_alpha: int
    n_beta: int

    def __post_init__(self):
        orbitals = tuple(int(o) for o in self.orbitals)
        object.__setattr__(self, "orbitals", orbitals)

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
        params: The fragment VQE's converged parameters from the previous
            round, or ``None`` for a fragment that has not been optimised
            yet (e.g. a freshly built initial state).
        rdm1_alpha: Alpha-spin half of ``rdm1``, or ``None`` to assume the
            closed-shell split ``rdm1 / 2``. Needed for the cross-fragment
            exchange term, which contracts same-spin densities.
        rdm1_beta: Beta-spin half of ``rdm1``, under the same convention.
    """

    spec: FragmentSpec
    rdm1: np.ndarray
    rdm2: np.ndarray
    params: np.ndarray | None = None
    rdm1_alpha: np.ndarray | None = None
    rdm1_beta: np.ndarray | None = None

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
        energy: Total energy for this state, or ``inf`` if not yet computed.
        previous_energy: Total energy from the previous macro-cycle, or
            ``inf`` for the initial state.
    """

    mo_coeff: np.ndarray
    fragments: tuple[FragmentState, ...]
    energy: float = float("inf")
    previous_energy: float = float("inf")


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

    Raises:
        ValueError: If any orbital index is out of range or shared between
            fragments, if ``specs`` is empty, if a fragment leaves every spin
            channel empty or full (no excitation available, so no correlation to
            capture), if the fragments' total electron count does not match the
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
