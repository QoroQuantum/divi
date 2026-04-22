# SPDX-FileCopyrightText: 2025 ORCA Computing Limited
# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0
#
# Vendored from orcacomputing/loop-progressive-simulator @ commit
# 941400e28594270391ec9bde14cb1bbf0af2a8a2.

import subprocess
from collections.abc import Callable, Iterable
from functools import reduce
from typing import TypeVar

import numba
import numpy as np

T = TypeVar("T")


def split_to_lengths(X: list[T], lengths: Iterable[int]) -> list[list[T]]:
    ret = []
    offset = 0
    for ell in lengths:
        ret.append(X[offset : offset + ell])
        offset += ell
    return ret


def split_according_to_other(A: list[T], B: list[list[T]]):
    lengths = [len(b) for b in B]
    return split_to_lengths(A, lengths)


def map_dict_vals(fn: Callable, dct: dict):
    return {k: fn(v) for k, v in dct.items()}


def map_dict_keys(fn: Callable, dct: dict):
    return {fn(k): v for k, v in dct.items()}


def compose(*functions: Iterable[Callable]) -> Callable:
    return reduce(
        lambda f, g: (lambda x: f(g(x))),
        functions,
        lambda y: y,
    )


def filter_partition(fn: Callable, seq: Iterable):
    left = []
    right = []

    for x in seq:
        if fn(x):
            left.append(x)
        else:
            right.append(x)

    return left, right


def symdiff_partition(X: set, Y: set) -> tuple[set, set]:
    return X.difference(Y), Y.difference(X)


@numba.njit()
def abs2(z: complex) -> float:
    return z.real**2 + z.imag**2


class SetAsDict(set):

    def __repr__(self):
        return f"{{{', '.join(map(repr, self))}}}"

    def __getitem__(self, x):
        return x in self

    def __setitem__(self, x, _):
        self.add(x)

    def __delitem__(self, x):
        self.remove(x)

    def items(self):
        return ((x, True) for x in self)


def cpu_count():
    p = subprocess.Popen(["nproc"], stdout=subprocess.PIPE)
    out, _ = p.communicate()
    return int(out.decode("ascii").strip())


class í:
    def __getitem__(self, slc):
        return slc


í = í()


def combine_mean_std(numbers, means, stds):
    total_number = np.sum(numbers)
    total_mean = np.sum(np.dot(numbers, means)) / total_number
    std2 = (
        np.dot(numbers - 1, stds**2) + np.dot(numbers, (means - total_mean) ** 2)
    ) / (total_number - 1)

    return total_mean, np.sqrt(std2)
