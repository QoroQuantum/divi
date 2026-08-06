# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Configuration objects for the LASSQD workflow."""

from collections.abc import Sequence
from dataclasses import dataclass

from ._state import FragmentSpec

# Carryover retention threshold, relative to the winning batch's largest
# coefficient. arXiv:2512.14936 sweeps it over 1e-1 to 1e-8 and converges from
# 1e-3 down; without carryover at all their macro-cycle energies oscillate.
_DEFAULT_CARRYOVER_CUTOFF = 1e-5


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
            each fragment's spin at once, and skips localization entirely.
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
            localized active orbital to the fragment owning the atom it sits on,
            replacing the coupling-graph clustering -- one fragment per metal
            centre, for instance.
        local_spins: Per-fragment ``2S``, in the order ``fragment_atoms`` names
            them. The fragment's electron count comes from its occupied
            orbitals, so this sets spin alone: ``n_alpha - n_beta = 2S``. Use it
            for spin-polarized fragments, e.g. ``[2, -2]`` for an
            antiferromagnetically coupled dimer of local triplets. The fragments
            must still sum to ``Sz = 0``.

    Raises:
        ValueError: If not exactly one active-space selector is given; if
            ``fragment_atoms`` or ``local_spins`` is combined with
            ``active_spaces``; if ``local_spins`` is given without
            ``fragment_atoms`` or does not match its length; if
            ``n_active_orbitals`` is not positive; if
            ``max_orbitals_per_fragment`` is below 1; or if
            ``coupling_threshold`` is negative.
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
                    "and the localization RNG, so a positional spin list would not "
                    "name a stable fragment."
                )
            if len(self.local_spins) != len(self.fragment_atoms):
                raise ValueError(
                    f"local_spins has {len(self.local_spins)} entries but "
                    f"fragment_atoms names {len(self.fragment_atoms)} fragments."
                )
        if self.n_active_orbitals is not None and self.n_active_orbitals <= 0:
            raise ValueError(
                f"n_active_orbitals must be positive; got {self.n_active_orbitals}."
            )
        if self.max_orbitals_per_fragment < 1:
            raise ValueError(
                "max_orbitals_per_fragment must be at least 1; got "
                f"{self.max_orbitals_per_fragment}."
            )
        if self.coupling_threshold < 0:
            raise ValueError(
                "coupling_threshold must be non-negative; got "
                f"{self.coupling_threshold}."
            )


@dataclass(frozen=True)
class SQDConfig:
    """Sampling and diagonalization budget for each fragment's SQD solve.

    Args:
        n_batches: Subspaces diagonalized per recovery iteration; the lowest
            energy wins.
        batch_size: Configurations sampled per batch, so the subspace holds up
            to ``batch_size ** 2`` determinants. The accuracy knob; a
            one-determinant subspace is the mean field.
        n_recovery_iterations: Configuration-recovery passes per fragment solve.
            Each pass reweights the next one's sampling from the orbital
            occupancies the previous pass recovered, so these are
            self-consistent passes over the sampled distribution, not optimizer
            steps.
        lambda_penalty: Weight of the ``S^2`` spin-contamination penalty added
            to the projected Hamiltonian before diagonalization.
        carryover_cutoff: Carryover SQD's retention threshold
            (arXiv:2512.14936), on by default. Each recovery iteration retains
            the determinants whose coefficient exceeds this fraction of the
            largest coefficient in the winning batch, and extends later
            iterations' subspaces with them. ``None`` reverts to conventional
            SQD.
        max_carryover: Caps the alpha and beta strings carryover retains, *per
            spin sector*. Carried strings join each batch's own sampled halves
            rather than replacing them, so a cap of ``k`` bounds a batch's
            subspace at ``(k + batch_size) ** 2`` determinants. ``None`` leaves
            it uncapped, and since the cutoff is relative it prunes little: the
            retained set then grows every recovery iteration and the subspace
            with it, quadratically. ``max_dim`` bounds the sector outright rather
            than only the carried part.
        max_dim: Caps each spin sector, as one integer or an ``(alpha, beta)``
            pair, so the subspace never exceeds their product. When it binds,
            strings are kept in priority order: reference, then carried, then
            sampled by descending sample count.
        include_reference: Keep the aufbau reference determinant in every batch,
            bounding the fragment's energy by its reference.
        symmetrize_spin: Pool the alpha and beta halves together for a
            spin-exchange invariant subspace. Inactive unless
            ``n_alpha == n_beta``.
        recovery_energy_tol: Ends a fragment's recovery once the winning energy
            moves less than this between iterations and the occupancies have also
            settled. ``0.0`` (the default) spends every iteration, since a
            settled iteration does not mean carryover had nothing left to add.
            Not ``LASSQD``'s ``energy_tol``, which ends the macro-cycle.
        recovery_occupancies_tol: The occupancy half of that test, on the largest
            change in any orbital's average occupancy.

    Raises:
        ValueError: If ``n_batches``, ``batch_size`` or
            ``n_recovery_iterations`` is below 1; if ``lambda_penalty`` is
            negative; if ``carryover_cutoff`` is not positive; if
            ``max_carryover`` is given without a cutoff or is below 1; if
            ``max_dim`` is not a positive integer or a pair of them; or if
            ``recovery_energy_tol`` or ``recovery_occupancies_tol`` is negative.
    """

    n_batches: int = 15
    batch_size: int = 170
    n_recovery_iterations: int = 6
    lambda_penalty: float = 0.2
    carryover_cutoff: float | None = _DEFAULT_CARRYOVER_CUTOFF
    max_carryover: int | None = None
    max_dim: int | tuple[int, int] | None = None
    include_reference: bool = True
    symmetrize_spin: bool = False
    recovery_energy_tol: float = 0.0
    recovery_occupancies_tol: float = 0.0

    def __post_init__(self):
        if self.n_batches < 1:
            raise ValueError(f"n_batches must be at least 1; got {self.n_batches}.")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be at least 1; got {self.batch_size}.")
        if self.n_recovery_iterations < 1:
            raise ValueError(
                "n_recovery_iterations must be at least 1; got "
                f"{self.n_recovery_iterations}."
            )
        if self.lambda_penalty < 0:
            raise ValueError(
                f"lambda_penalty must be non-negative; got {self.lambda_penalty}."
            )
        if self.carryover_cutoff is not None and self.carryover_cutoff <= 0:
            raise ValueError(
                f"carryover_cutoff must be positive; got {self.carryover_cutoff}."
            )
        if self.max_carryover is not None:
            if self.carryover_cutoff is None:
                raise ValueError(
                    "max_carryover caps what carryover retains, so it needs "
                    "carryover_cutoff to be set."
                )
            if self.max_carryover < 1:
                raise ValueError(
                    f"max_carryover must be at least 1; got {self.max_carryover}."
                )
        if self.max_dim is not None:
            if isinstance(self.max_dim, tuple):
                object.__setattr__(
                    self, "max_dim", tuple(int(dim) for dim in self.max_dim)
                )
                if len(self.max_dim) != 2:
                    raise ValueError(
                        "max_dim takes one integer or an (alpha, beta) pair; got "
                        f"{len(self.max_dim)} entries."
                    )
                dims = self.max_dim
            else:
                dims = (self.max_dim,)
            for dim in dims:
                if dim < 1:
                    raise ValueError(f"max_dim entries must be at least 1; got {dim}.")
        for name, value in (
            ("recovery_energy_tol", self.recovery_energy_tol),
            ("recovery_occupancies_tol", self.recovery_occupancies_tol),
        ):
            if value < 0:
                raise ValueError(f"{name} must be non-negative; got {value}.")
