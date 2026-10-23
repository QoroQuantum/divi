# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Excited-state bookkeeping of a LASSQD run: the final subspace, state
tracking, the warnings about unresolved states and the state exports."""

import itertools
from collections import Counter
from collections.abc import Callable, Hashable, Iterator, Mapping, Sequence
from dataclasses import replace
from warnings import warn

import numpy as np

from ._config import StateAveragedOrbitalUpdate, _by_root
from ._integrals import AOIntegrals
from ._orbital_solvers import OrbitalSolution
from ._sqd import (
    SQDConfig,
    SQDResult,
    default_states,
    full_space_size,
    spin_square_of,
)
from ._state import (
    ElectronicState,
    FragmentSpec,
    StateTransition,
    StateWavefunction,
    _core_orbital_count,
    _OrbitalRound,
    _SampledSubspace,
)
from ._subspace import (
    state_transitions,
    subspace_nevpt2,
    subspace_states_at,
    subspace_wavefunctions,
    track_states,
)
from ._warnings import _outside_divi_stacklevel

#: ``<S^2>`` deviation from a state's target above which it is reported unresolved.
_SPIN_SQUARE_TOL = 1e-3

#: Energy gap (Ha) between same-multiplicity roots reported as near-degenerate.
_DEGENERACY_TOL = 1e-6

#: Gap (Ha) under which an unrequested root counts as degenerate with the last
#: requested one, above the ~3e-6 Ha the starting-orbital nudge splits a pair by.
_SPLIT_SET_TOL = 1e-4


def _is_off_spin(multiplicity: int, spin_square: float) -> bool:
    return abs(spin_square - spin_square_of(multiplicity)) > _SPIN_SQUARE_TOL


def _state_warnings(
    states: Sequence[ElectronicState],
) -> Iterator[tuple[Hashable, str]]:
    """``(key, message)`` for each state the subspace leaves off-spin and each
    near-degenerate pair."""
    for state in states:
        if _is_off_spin(state.multiplicity, state.spin_square):
            yield (
                ("off-spin", state.multiplicity, state.index),
                f"State ({state.multiplicity}, {state.index}) has <S^2> = "
                f"{state.spin_square:.4f}, not "
                f"{spin_square_of(state.multiplicity):.4f}: the subspace does not "
                "resolve it as that spin. Raise lambda_penalty or the sampling "
                "budget (batch_size, n_recovery_iterations).",
            )
    for lower, upper in itertools.pairwise(
        sorted(states, key=lambda state: (state.multiplicity, state.energy))
    ):
        gap = abs(upper.energy - lower.energy)
        if lower.multiplicity == upper.multiplicity and gap < _DEGENERACY_TOL:
            first, second = sorted((lower.index, upper.index))
            yield (
                ("degenerate", lower.multiplicity, first, second),
                f"States ({lower.multiplicity}, {first}) and "
                f"({lower.multiplicity}, {second}) lie within {gap:.1e} Ha "
                "of each other, so their order can swap between rounds.",
            )


def _in_energy_order(
    states: Sequence[ElectronicState],
) -> tuple[ElectronicState, ...]:
    """Ascending in energy; states within ``_DEGENERACY_TOL`` of each other keep
    root order, so float noise cannot reorder them."""
    groups: list[list[ElectronicState]] = []
    for state in sorted(states, key=lambda state: state.energy):
        if groups and state.energy - groups[-1][0].energy < _DEGENERACY_TOL:
            groups[-1].append(state)
        else:
            groups.append([state])
    return tuple(state for group in groups for state in sorted(group))


class _ExcitedStates:
    """The requested states' subspace and everything derived from it.

    ``final_subspace`` is the last completed round's SQD subspace at its final
    orbitals, ``None`` until a round with ``SQDConfig.states`` completes.
    """

    def __init__(
        self,
        mol,
        sqd: SQDConfig,
        spin: int,
        ao_integrals: Callable[[], AOIntegrals],
        warn_once: Callable[[Hashable, str], None],
        rounds_completed: Callable[[], bool],
    ):
        self._mol = mol
        self._sqd = sqd
        self._spin = spin
        self._ao_integrals = ao_integrals
        self._warn_once = warn_once
        self._rounds_completed = rounds_completed
        self.final_subspace: _SampledSubspace | None = None

    def reset(self) -> None:
        self.final_subspace = None

    def record_round(
        self,
        ao: AOIntegrals,
        n_core: int,
        states: Mapping[int, int],
        result: SQDResult,
        spec: FragmentSpec,
        solution: OrbitalSolution,
        relaxed_states: tuple[ElectronicState, ...] | None,
        in_sampled_basis: Callable[[np.ndarray], np.ndarray],
    ) -> tuple[ElectronicState, ...]:
        """Make the round's subspace the final one, following each state's
        label from the previous round's, and return its states in energy
        order."""
        # A state-averaged solve returns its orbitals in the sampled basis.
        final_subspace = _SampledSubspace(
            (
                in_sampled_basis(solution.mo_coeff)
                if relaxed_states is None
                else solution.mo_coeff
            ),
            (result.strings_alpha, result.strings_beta),
            states,
            self._sqd.lambda_penalty,
        )
        if self.final_subspace is not None:
            final_subspace = replace(
                final_subspace,
                labels=track_states(ao, self.final_subspace, final_subspace, n_core),
            )
        labels = final_subspace.labels or {}
        if relaxed_states is None:
            relaxed_states = subspace_states_at(ao, final_subspace, n_core)
        else:
            relaxed_states = tuple(
                replace(
                    state,
                    index=labels.get(
                        (state.multiplicity, state.index),
                        (state.multiplicity, state.index),
                    )[1],
                )
                for state in relaxed_states
            )
        self.final_subspace = final_subspace
        return self._resolved_states("fragment_0", result, spec, relaxed_states)

    def follow_swapped_states(
        self,
        orbital_update: StateAveragedOrbitalUpdate,
        orbital_round: _OrbitalRound,
        solution: OrbitalSolution,
        relaxed_states: tuple[ElectronicState, ...] | None,
        round_number: int,
    ) -> tuple[OrbitalSolution, tuple[ElectronicState, ...] | None]:
        """Re-solve once if the tracked states swapped during the
        state-averaged solve, which weights roots by energy rank, so the
        weights move to where the states ended up.

        Warns:
            UserWarning: If they swap again in the re-solve.
        """
        subspace = orbital_round.subspace
        if subspace is None or self.final_subspace is None:
            return solution, relaxed_states
        weights = orbital_update._weights_for(subspace.states)
        used = subspace.labels or {}
        for attempt in range(2):
            landed = track_states(
                orbital_round.ao,
                self.final_subspace,
                replace(subspace, mo_coeff=solution.mo_coeff, labels=None),
                orbital_round.n_core,
            )
            if _by_root(weights, landed) == _by_root(weights, used):
                return solution, relaxed_states
            if attempt:
                self._warn_once(
                    ("weights not followed", round_number),
                    f"Round {round_number}'s requested states "
                    "swapped order during the state-averaged solve, twice, so its "
                    "weights may sit on the other states this round.",
                )
                return solution, relaxed_states
            used = landed
            solution, relaxed_states = orbital_update._solve(
                replace(
                    orbital_round,
                    subspace=replace(
                        subspace, mo_coeff=solution.mo_coeff, labels=landed
                    ),
                )
            )
        return solution, relaxed_states

    def _resolved_states(
        self,
        program_id: str,
        result: SQDResult,
        spec: FragmentSpec,
        states: Sequence[ElectronicState],
    ) -> tuple[ElectronicState, ...]:
        """``states``, the subspace's eigenstates at the round's orbitals, in
        energy order, warning about any the subspace does not resolve."""
        excited = self._sqd.states != default_states(spec.n_alpha, spec.n_beta)
        full_size = full_space_size(spec.n_orbitals, spec.n_alpha, spec.n_beta)
        if excited and result.amplitudes.size < full_size:
            hint = (
                " Set SQDConfig.extension to add the determinants it misses."
                if self._sqd.extension is None
                else ""
            )
            self._warn_once(
                ("incomplete subspace", program_id),
                f"{program_id}'s subspace holds {result.amplitudes.size} of "
                f"{full_size} determinants. The circuit samples the ground "
                "state, so excited states in an incomplete subspace come out "
                f"too high.{hint}",
            )

        resolved = Counter(state.multiplicity for state in states)
        for multiplicity, n_roots in (self._sqd.states or {}).items():
            if resolved[multiplicity] < n_roots:
                self._warn_once(
                    ("unresolved", program_id, multiplicity),
                    f"{program_id}'s subspace resolves {resolved[multiplicity]} of "
                    f"the {n_roots} requested states of multiplicity "
                    f"{multiplicity}; the rest are left out. Raise the sampling "
                    "budget (batch_size, n_recovery_iterations) or "
                    "SQDConfig.extension.",
                )
        for key, message in _state_warnings(states):
            self._warn_once(key, message)
        return _in_energy_order(states)

    def warn_if_a_degenerate_set_is_split(
        self,
        ao: AOIntegrals,
        mo_coeff: np.ndarray,
        n_core: int,
        result: SQDResult,
        requested: Mapping[int, int],
    ) -> None:
        """Warn for each multiplicity whose last requested root is degenerate
        with the next, unrequested one, solved at ``mo_coeff``."""
        probe = {
            multiplicity: n_roots + 1
            for multiplicity, n_roots in requested.items()
            if n_roots < result.amplitudes.size
        }
        solved = subspace_states_at(
            ao,
            _SampledSubspace(
                mo_coeff,
                (result.strings_alpha, result.strings_beta),
                probe,
                self._sqd.lambda_penalty,
            ),
            n_core,
        )
        for multiplicity, n_roots in requested.items():
            roots = [state for state in solved if state.multiplicity == multiplicity]
            if len(roots) <= n_roots:
                continue
            last, unrequested = roots[n_roots - 1], roots[n_roots]
            gap = unrequested.energy - last.energy
            if (
                not _is_off_spin(multiplicity, unrequested.spin_square)
                and gap < _SPLIT_SET_TOL
            ):
                warn(
                    f"State ({multiplicity}, {n_roots - 1}) lies within "
                    f"{abs(gap):.1e} "
                    f"Ha of ({multiplicity}, {n_roots}), which SQDConfig.states "
                    "does not request: it splits a degenerate set, so the state "
                    "reported is an arbitrary member, and state averaging over "
                    f"part of the set breaks its symmetry. Request {multiplicity}: "
                    f"{n_roots + 1} or more.",
                    UserWarning,
                    stacklevel=_outside_divi_stacklevel(),
                )

    def state_wavefunctions(self) -> dict[tuple[int, int], StateWavefunction]:
        """Each requested state's wave function, keyed ``(multiplicity,
        index)`` in energy order."""
        return {
            (wavefunction.state.multiplicity, wavefunction.state.index): wavefunction
            for wavefunction in self.wavefunctions_in_energy_order(
                "state_wavefunctions"
            )
        }

    def wavefunctions_in_energy_order(self, name: str) -> tuple[StateWavefunction, ...]:
        """The final wave functions for the method ``name``, ordered as
        ``LASSQD.electronic_states``."""
        wavefunctions = self._final_wavefunctions(name)
        by_state = {wavefunction.state: wavefunction for wavefunction in wavefunctions}
        return tuple(
            by_state[state]
            for state in _in_energy_order(
                [wavefunction.state for wavefunction in wavefunctions]
            )
        )

    def transition_properties(
        self,
    ) -> dict[tuple[tuple[int, int], tuple[int, int]], StateTransition]:
        """The one-electron transition between every pair of requested states
        of one multiplicity, keyed ``(initial, final)`` labels."""
        return {
            (
                (transition.initial.multiplicity, transition.initial.index),
                (transition.final.multiplicity, transition.final.index),
            ): transition
            for transition in state_transitions(
                self._mol,
                list(self.wavefunctions_in_energy_order("transition_properties")),
            )
        }

    def nevpt2(self) -> dict[tuple[int, int], float]:
        """Each requested state's NEVPT2 correction, keyed ``(multiplicity,
        index)`` in uncorrected energy order."""
        subspace, n_core = self._final_subspace_and_core("nevpt2")
        corrections = dict(subspace_nevpt2(self._ao_integrals(), subspace, n_core))
        return {
            (state.multiplicity, state.index): corrections[state]
            for state in _in_energy_order(list(corrections))
        }

    def _final_subspace_and_core(self, name: str) -> tuple[_SampledSubspace, int]:
        """The last completed round's subspace and frozen-core count, for the
        method ``name``."""
        subspace = self.final_subspace
        if subspace is None:
            example = f"SQDConfig(states={{{self._spin + 1}: 2}})"
            remedy = (
                f"; this run requested none. Set them, e.g. {example}, and run again."
                if self._rounds_completed()
                else f". Request the states, e.g. {example}, and call run() first."
            )
            raise RuntimeError(
                f"{name} needs a completed LASSQD round with SQDConfig.states "
                f"set{remedy}"
            )
        strings_alpha, strings_beta = subspace.strings
        n_core = _core_orbital_count(
            self._mol.nelectron,
            strings_alpha[0].count("1") + strings_beta[0].count("1"),
        )
        return subspace, n_core

    def _final_wavefunctions(self, name: str) -> tuple[StateWavefunction, ...]:
        """The last completed round's requested states at its final orbitals,
        for the method ``name``."""
        subspace, n_core = self._final_subspace_and_core(name)
        return subspace_wavefunctions(self._ao_integrals(), subspace, n_core)
