# SPDX-FileCopyrightText: 2025 ORCA Computing Limited
# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0
#
# Vendored from orcacomputing/loop-progressive-simulator @ commit
# 941400e28594270391ec9bde14cb1bbf0af2a8a2. Only modifications: relative
# imports, SPDX header. See divi/photonic/_vendor/loop_progressive_simulator/
# LICENSE.upstream for the upstream Apache-2.0 text.

from copy import copy
from functools import cached_property, reduce
from io import StringIO
from itertools import chain
from typing import TypeVar

import numpy as np

from .utils import split_to_lengths

# This is backward compatibility with an older version that does symbolics with Sage
Expression = TypeVar("Expression")


class AbstractBSCircuit:
    """An abstract beamsplitter circuit, represented as sequence of beamsplitters. There are no beamsplitter angles, only connectivity."""

    def __init__(self, beamsplitters: list[tuple[int, int]]):
        self.beamsplitters = [tuple(sorted(x)) for x in beamsplitters]

    @classmethod
    def generate_loop(cls, length, modes):
        circuit_length = modes - length
        return cls([(i, i + length) for i in range(circuit_length)])

    @property
    def _abstract_beamsplitters(self):
        return self.beamsplitters

    def _diagram(self, sep_len=2, start_sep_len=None):
        gate_len = sep_len + 1
        if start_sep_len is None:
            start_sep_len = sep_len

        M = self.modes
        circ = self._abstract_beamsplitters
        diagram = np.full((M, len(circ) * gate_len + start_sep_len), "", dtype=str)

        for m in range(M):
            for i in range(start_sep_len):
                diagram[m, i] = "─"

        for t in range(len(circ)):
            t_diag = t * gate_len + start_sep_len
            i, j = sorted(circ[t])
            for a in chain(range(i), range(j + 1, M)):
                diagram[a, t_diag] = "─"
            diagram[i, t_diag] = "┰"
            diagram[j, t_diag] = "┸"
            for a in range(i + 1, j):
                diagram[a, t_diag] = "╂"

            for m in range(M):
                for q in range(1, sep_len + 1):
                    diagram[m, t_diag + q] = "─"

        return diagram

    def __str__(self):
        diagram = self._diagram()

        buffer = StringIO()
        for m in range(self.modes):
            for t in range(diagram.shape[1]):
                print(diagram[m, t], end="", file=buffer)
            print(file=buffer)

        return buffer.getvalue()

    __repr__ = __str__

    def __len__(self):
        return len(self.beamsplitters)

    def __iter__(self):
        return iter(self.beamsplitters)

    @cached_property
    def modes(self):
        return reduce(lambda x, y: max(x, y[0], y[1]), self.beamsplitters, 0) + 1

    @property
    def empty(self):
        return len(self.beamsplitters) == 0

    def __add__(self, b):
        M = self.modes
        b_shifted = [(i + M, j + M) for (i, j) in b.beamsplitters]
        return self.__class__(self.beamsplitters + b_shifted)

    def __mul__(self, b):
        # note left-to-right composition!
        return self.__class__(self.beamsplitters + b.beamsplitters)

    def __rmul__(self, a):
        # note left-to-right composition!
        return self.__class__(a.beamsplitters + self.beamsplitters)

    def __getitem__(self, index):
        return self.beamsplitters[index]

    def __copy__(self):
        return self.__class__(copy(self.beamsplitters))

    def shift_unsafe(self, modes):
        if modes == 0:
            return self

        beamsplitters = [
            (i + modes, j + modes, *th) for i, j, *th in self.beamsplitters
        ]
        return self.__class__(beamsplitters)


class ConcreteBSCircuit(AbstractBSCircuit):
    """A concrete beamsplitter circit, with beamsplitter angles."""

    def __init__(self, beamsplitters: list[tuple[int, int, float]]):
        self.beamsplitters = [tuple(sorted(x[:2])) + (x[2],) for x in beamsplitters]

    @classmethod
    def from_abstract(cls, abc: AbstractBSCircuit, thetas: list[float] | None):
        """Instantiate an abstract beamsplitter circuit with concrete angles."""
        if len(abc) != len(thetas):
            raise ValueError(
                f"Incorrect number of beamsplitter angles: expected {len(abc)}, got {len(thetas)}."
            )
        return cls(
            [(i, j, theta) for ((i, j), theta) in zip(abc.beamsplitters, thetas)]
        )

    @classmethod
    def generate_loop(cls, length, modes, thetas: list[float] | None):
        # generate an abstract loop, then add angles (super() is AbstractBSCircuit)
        abc = super().generate_loop(length, modes)
        return cls.from_abstract(abc, thetas)

    @property
    def _abstract_beamsplitters(self):
        return [(i, j) for (i, j, _) in self.beamsplitters]

    def to_abstract(self):
        return AbstractBSCircuit(self._abstract_beamsplitters)

    def __add__(self, b):
        M = self.modes
        b_shifted = [(i + M, j + M, th) for (i, j, th) in b.beamsplitters]
        return self.__class__(self.beamsplitters + b_shifted)


BSCircuit = AbstractBSCircuit | ConcreteBSCircuit


def multiple_loops(*lengths, modes, thetas: list[float] | None = None) -> BSCircuit:
    loops = [AbstractBSCircuit.generate_loop(length, modes) for length in lengths]

    if thetas is not None:
        comp_lengths = map(len, loops)
        thetas = split_to_lengths(thetas, comp_lengths)
        loops = [
            ConcreteBSCircuit.from_abstract(abc, th) for (abc, th) in zip(loops, thetas)
        ]

    return reduce(lambda a, b: a * b, loops)


def check_commute(x: tuple, y: tuple) -> bool:
    """Check that the beamsplitters x and y do not touch the same modes."""
    # use y[:2] to make this compatible with beamsplitters with or without angles
    return (x[0] not in y[:2]) and (x[1] not in y[:2])


β = TypeVar("β", AbstractBSCircuit, ConcreteBSCircuit)


def commute_pos_forward(circ: β, pos: int) -> β:
    if pos == 0:
        return circ, pos

    after = circ[pos + 1 :]
    nearest_noncommuting = pos - 1
    while nearest_noncommuting >= 0:
        if check_commute(circ[pos], circ[nearest_noncommuting]):
            nearest_noncommuting -= 1
        else:
            break

    before = circ[: nearest_noncommuting + 1]  # inclusive
    between = circ[nearest_noncommuting + 1 : pos]

    return (
        circ.__class__(before + [circ[pos]] + between + after),
        nearest_noncommuting + 1,
    )


def commute_multiple_forward(circ: β, pos: int) -> β:
    # beamsplitter[:2] to make it compatible with beamsplitters with or without angles
    modes = set(circ[pos][:2])
    positions = [pos]

    for i in range(pos - 1, -1, -1):
        if circ[i][0] in modes or circ[i][1] in modes:
            modes.update(set(circ[i][:2]))
            positions.append(i)

    new_circ = copy(circ)
    new_positions = set()
    for p in reversed(positions):
        new_circ, new_p = commute_pos_forward(new_circ, p)
        new_positions.add(new_p)
    return new_circ, (min(new_positions), max(new_positions))


def last_interaction_with(circ: BSCircuit, mode: int) -> int:
    last = -1
    for i in range(len(circ)):
        if mode in circ[i][:2]:
            last = i

    if last == -1:
        return None
    else:
        return last


def commute_mode_forward(circ: β, mode: int) -> β:
    pos = last_interaction_with(circ, mode)
    if pos is not None:
        return commute_multiple_forward(circ, pos)

    else:
        return circ, (0, len(circ) - 1)


def progressive_decomposition(circ: β) -> list[β]:
    components = []
    CircClass = circ.__class__
    rest = copy(circ)

    for m in range(circ.modes):
        new_circ, (start, stop) = commute_mode_forward(rest, m)
        assert start == 0
        if not new_circ.empty:
            components.append(CircClass(new_circ[start : stop + 1]))
        rest = CircClass(new_circ[stop + 1 :])

    return components


def split_L1_rest(circ: β, touched_modes=None) -> tuple[β, β, set[int]]:
    if touched_modes is None:
        touched_modes = set()
    else:
        touched_modes = copy(touched_modes)

    L1 = []
    rest = []
    for x in circ:
        a, b, *_th = x

        if abs(b - a) == 1:
            if a in touched_modes and b in touched_modes:
                rest.append(x)
            else:
                L1.append(x)
                touched_modes.update({a, b})

        else:
            assert a in touched_modes and b in touched_modes, ValueError(
                "split_L1_rest can only be used with a circuit starting with a loop L1."
            )
            rest.append(x)

    CircClass = circ.__class__
    return CircClass(L1), CircClass(rest), touched_modes


def split_decomposition_L1_rest(
    components: list[β], touched_modes=None
) -> list[tuple[β, β, set]]:
    comps = []

    for X in components:
        L1, rest, touched_modes = split_L1_rest(X, touched_modes)
        comps.append((L1, rest, touched_modes))

    return comps
