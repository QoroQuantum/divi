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
        n_alpha: Alpha electrons assigned to the fragment. Must equal
            ``n_beta``: only closed-shell fragments are supported.
        n_beta: Beta electrons assigned to the fragment. Must equal
            ``n_alpha``: only closed-shell fragments are supported.

    Raises:
        ValueError: If ``orbitals`` is empty or contains duplicates, or if
            ``n_alpha``/``n_beta`` fall outside ``[0, n_orbitals]``.
            ``n_alpha != n_beta`` is rejected separately, during fragment
            validation.
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
            round, or ``None`` for a fragment that has not been optimized
            yet (e.g. a freshly built initial state).
    """

    spec: FragmentSpec
    rdm1: np.ndarray
    rdm2: np.ndarray
    params: np.ndarray | None = None


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
    specs: Sequence[FragmentSpec], n_orbitals_total: int
) -> None:
    """Reject fragments that overlap or index outside the orbital register.

    Non-overlap matters beyond hygiene: effective fragment integrals sum the
    mean-field contribution of every *other* fragment, so shared orbitals
    would be double-counted.

    Raises:
        ValueError: If any orbital index is out of range or shared between
            fragments, if ``specs`` is empty, if any fragment is
            spin-imbalanced (``n_alpha != n_beta``), or if any fragment is
            fully occupied (``n_alpha + n_beta >= 2 * n_orbitals``, leaving no
            correlation to capture).
    """
    if not specs:
        raise ValueError("At least one fragment is required.")

    seen: dict[int, int] = {}
    for index, spec in enumerate(specs):
        if spec.n_alpha != spec.n_beta:
            raise ValueError(
                f"Fragment {index} (orbitals {spec.orbitals}) is "
                f"spin-imbalanced: n_alpha={spec.n_alpha}, n_beta="
                f"{spec.n_beta}. Only closed-shell fragments (n_alpha == "
                "n_beta) are supported."
            )
        if spec.n_alpha + spec.n_beta >= 2 * spec.n_orbitals:
            raise ValueError(
                f"Fragment {index} (orbitals {spec.orbitals}) is fully "
                f"occupied: n_alpha={spec.n_alpha}, n_beta={spec.n_beta} "
                f"fill all {spec.n_orbitals} orbitals, leaving no "
                "correlation for this fragment to capture."
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
