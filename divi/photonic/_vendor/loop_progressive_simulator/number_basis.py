# SPDX-FileCopyrightText: 2025 ORCA Computing Limited
# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0
#
# Vendored from orcacomputing/loop-progressive-simulator @ commit
# 941400e28594270391ec9bde14cb1bbf0af2a8a2.

import numba


@numba.njit()
def _make_basis_state(basis_state: tuple[int, ...]) -> list[int]:
    basis_state = list(basis_state)

    # truncate trailing zeros, if any
    for i in range(len(basis_state) - 1, -1, -1):

        # find last nonzero entry
        if basis_state[i] != 0:
            return basis_state[: i + 1]

    # no trailing zeros found
    return basis_state


def make_basis_state(basis_state: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(_make_basis_state(basis_state))


@numba.njit()
def number_in_mode(basis_state: tuple[int, ...], mode: int) -> int:
    if mode >= len(basis_state):
        return 0
    else:
        return basis_state[mode]


@numba.njit()
def _replace_two_modes(
    basis_state: tuple[int, ...],
    i: int,
    ni: int,
    j: int,
    nj: int,
) -> list[int]:

    assert i != j

    M = len(basis_state)
    new_M = max(M, max(i, j) + 1)

    new_basis_state = [0] * new_M
    for mode in range(new_M):
        if mode == i:
            new_basis_state[mode] = ni
        elif mode == j:
            new_basis_state[mode] = nj
        elif mode < M:
            new_basis_state[mode] = basis_state[mode]
        else:
            new_basis_state[mode] = 0

    return new_basis_state


def replace_two_modes(
    basis_state: tuple[int, ...],
    i: int,
    ni: int,
    j: int,
    nj: int,
) -> tuple[int, ...]:
    return tuple(_replace_two_modes(basis_state, i, ni, j, nj))
