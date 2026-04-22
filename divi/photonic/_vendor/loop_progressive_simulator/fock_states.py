# SPDX-FileCopyrightText: 2025 ORCA Computing Limited
# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0
#
# Vendored from orcacomputing/loop-progressive-simulator @ commit
# 941400e28594270391ec9bde14cb1bbf0af2a8a2. Only modifications: relative
# imports, SPDX header.

from collections.abc import Callable, Iterable
from copy import copy
from functools import cached_property
from typing import TypeVar

import numpy as np

from . import number_basis
from .utils import SetAsDict, abs2

EPS = 1e-14


class SparseBosonicFockState:
    """A sparse (dictionary) representation of a bosonic Fock state as an element of the symmetric algebra S(CC^M)."""

    _amplitudes = dict()

    def __init__(self, amp: dict[tuple[int, ...], complex]):
        self._amplitudes = {
            number_basis.make_basis_state(key): value for key, value in amp.items()
        }

    def basis(self):
        return self._amplitudes.keys()

    @cached_property
    def modes(self):
        return max([len(basis_state) for basis_state in self.basis()])

    @cached_property
    def dim(self):
        return len(self.basis())

    def __copy__(self):
        return self.__class__(copy(self._amplitudes))

    def __getitem__(self, basis_state: tuple[int, ...]) -> complex:
        "Get the amplitude of a number basis state."

        basis_state = number_basis.make_basis_state(basis_state)

        # try to find amplitude; default is 0
        try:
            return self._amplitudes[basis_state]
        except KeyError:
            return 0

    def __setitem__(self, basis_state: tuple[int, ...], amp: complex) -> None:
        "Set the amplitude of a number basis state, in place. No normalization is done."

        basis_state = number_basis.make_basis_state(basis_state)

        # if setting the amplitude to zero, delete the key (keep the state sparse)
        if np.isclose(amp, 0, atol=EPS):
            try:
                del self._amplitudes[basis_state]
            except KeyError:
                pass
        else:
            self._amplitudes[basis_state] = amp

    def __delitem__(self, basis_state: tuple[int, ...]) -> None:
        """Delete (set amplitude to 0) the basis state, in place. No normalization is done."""

        basis_state = number_basis.make_basis_state(basis_state)
        try:
            del self._amplitudes[basis_state]
        except KeyError:
            pass

    def items(self):
        return self._amplitudes.items()

    @property
    def norm2(self):
        return sum([abs2(amp) for _, amp in self._amplitudes.items()])

    def normalize_in_place(self):
        factor = 1 / np.sqrt(self.norm2)

        for basis_state in self._amplitudes.keys():
            self._amplitudes[basis_state] *= factor

    def __add__(self, Psi):
        amplitudes = copy(self._amplitudes)

        for basis_state, amp in Psi._amplitudes.items():
            try:
                amplitudes[basis_state] += amp
            except KeyError:
                amplitudes[basis_state] = amp

        return self.__class__(amplitudes)

    def __iadd__(self, Psi):
        for basis_state, amp in Psi._amplitudes.items():
            try:
                self[basis_state] = self._amplitudes[basis_state] + amp
            except KeyError:
                self[basis_state] = amp

    def __repr__(self):
        return repr(self._amplitudes)

    __str__ = __repr__

    @cached_property
    def grades(self):
        """Return a set of grades of the bosonic Fock space that contain the state."""
        return {sum(basis_state) for basis_state in self.basis()}

    @property
    def photons(self):
        """If the state is homogeneous (single grade), return the number of photons. Otherwise returns None."""
        match list(self.grades):
            case []:
                return 0
            case [n]:
                return n
            case _:
                return None


class AbstractBosonicFockState(SparseBosonicFockState):
    """An abstract representation of a bosonic Fock state, listing possible basis states, but not amplitudes."""

    _amplitudes = None

    def __init__(self, states: Iterable[tuple[int, ...]]):
        self._amplitudes = SetAsDict(
            number_basis.make_basis_state(key) for key in states
        )

    def basis(self):
        return self._amplitudes

    def __setitem__(self, basis_state: tuple[int, ...], amp: bool) -> None:
        basis_state = number_basis.make_basis_state(basis_state)
        if amp:
            self._amplitudes[basis_state] = True
        else:
            del self._amplitudes[basis_state]

    @property
    def norm2(self):
        if len(self._amplitudes) == 0:
            return 0
        else:
            return 1

    def normalize_in_place(self):
        pass

    def __add__(self, Psi):
        return self.__class__(SetAsDict.union(self._amplitudes, Psi._amplitudes))

    def __iadd__(self, Psi):
        self._amplitudes.update(Psi._amplitudes)


φ = TypeVar("φ", AbstractBosonicFockState, SparseBosonicFockState)


def map_fock_basis(fn: Callable[[tuple[int, ...]], tuple[int, ...]], state: φ) -> φ:

    if isinstance(state, AbstractBosonicFockState):
        return AbstractBosonicFockState(
            x for basis in state.basis() if len(x := fn(basis)) > 0
        )

    else:
        return SparseBosonicFockState(
            {x: amp for basis, amp in state.items() if len(x := fn(basis)) > 0}
        )


def filter_fock_basis(fn: Callable[[tuple[int, ...]], bool], state: φ) -> φ:

    if isinstance(state, AbstractBosonicFockState):
        return AbstractBosonicFockState(filter(fn, state.basis()))

    else:
        return SparseBosonicFockState(
            {basis: amp for basis, amp in state.items() if fn(basis)}
        )


def project_number_mode(mode: int, number: int, state: φ) -> φ:
    return filter_fock_basis(
        lambda basis: number_basis.number_in_mode(basis, mode) == number, state
    )
