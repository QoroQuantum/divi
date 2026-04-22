# SPDX-FileCopyrightText: 2025 ORCA Computing Limited
# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0
#
# Vendored from orcacomputing/loop-progressive-simulator @ commit
# 941400e28594270391ec9bde14cb1bbf0af2a8a2 (file: step_simulator_nosym.py).
# Modifications: relative imports, SPDX header, file rename.

from copy import copy

import numba
import numpy as np

from . import number_basis
from .bscircuits import AbstractBSCircuit, ConcreteBSCircuit, progressive_decomposition
from .factorial_table import log_factorial
from .fock_states import (
    AbstractBosonicFockState,
    SparseBosonicFockState,
    project_number_mode,
)
from .utils import abs2, map_dict_vals


@numba.njit()
def bs_amplitude(
    n1_in: int, n2_in: int, N1_out: int, N2_out: int, theta: float
) -> complex:
    """Compute the amplitude of the process |n_1, n_2>  -θ->  |N_1, N_2>. The formula is derived in the symmetric algebra S(CC^2); can be seen using action of beamsplitter on creation operators."""

    # transmission
    cos_theta = np.cos(theta)

    # reflection
    sin_theta = np.sin(theta)

    # combinatorial & normalization prefactor
    prefactor = np.exp(
        0.5
        * (
            log_factorial(N1_out)
            + log_factorial(N2_out)
            + log_factorial(n1_in)
            + log_factorial(n2_in)
        )
    )

    # sum over valid pairs (k, ℓ = ell = N1_out - k), i.e. where
    #   k + ℓ = N1_out
    #   0 <= k <= n1_in
    #   0 <= ℓ <= n2_in
    k_min = max(0, N1_out - n2_in)
    k_max = min(n1_in, N1_out)
    summation = 0
    for k in range(k_min, k_max + 1):
        ell = N1_out - k

        sign = -1 if (n1_in - k) % 2 else 1
        summation += (
            sign
            * cos_theta ** (n2_in + k - ell)
            * sin_theta ** (n1_in - k + ell)
            / np.exp(
                log_factorial(k)
                + log_factorial(n1_in - k)
                + log_factorial(ell)
                + log_factorial(n2_in - ell)
            )
        )

    return prefactor * summation


def apply_beamsplitter(
    i: int, j: int, theta: float, state: SparseBosonicFockState, **_
) -> SparseBosonicFockState:
    """Apply a beamsplitter with angle theta to modes i, j of the state."""

    Psi = SparseBosonicFockState({})

    for basis_state, incoming_amplitude in state.items():
        ni_in = number_basis.number_in_mode(basis_state, i)
        nj_in = number_basis.number_in_mode(basis_state, j)
        n_total = ni_in + nj_in

        # all possible output occupations
        for Ni_out in range(n_total + 1):
            Nj_out = n_total - Ni_out
            process_amplitude = bs_amplitude(ni_in, nj_in, Ni_out, Nj_out, theta)

            new_basis = number_basis.replace_two_modes(
                basis_state, i, Ni_out, j, Nj_out
            )
            new_amplitude = incoming_amplitude * process_amplitude
            Psi[new_basis] = Psi[new_basis] + new_amplitude

    return Psi


def apply_abstract_beamsplitter(
    i: int, j: int, state: AbstractBosonicFockState
) -> AbstractBosonicFockState:
    """Apply a beamsplitter to modes i, j of the abstract state."""

    Psi = AbstractBosonicFockState({})

    for basis_state in state.basis():
        ni_in = number_basis.number_in_mode(basis_state, i)
        nj_in = number_basis.number_in_mode(basis_state, j)
        n_total = ni_in + nj_in

        # all possible output occupations
        for Ni_out in range(n_total + 1):
            Nj_out = n_total - Ni_out

            new_basis = number_basis.replace_two_modes(
                basis_state, i, Ni_out, j, Nj_out
            )
            Psi[new_basis] = 1

    return Psi


def apply_bs_circuit(
    circuit: ConcreteBSCircuit, state: SparseBosonicFockState
) -> SparseBosonicFockState:

    for i, j, theta in circuit:
        state = apply_beamsplitter(i, j, theta, state)

    return state


def apply_abstract_bs_circuit(
    circuit: AbstractBSCircuit, state: AbstractBosonicFockState
) -> AbstractBosonicFockState:

    for i, j in circuit:
        state = apply_abstract_beamsplitter(i, j, state)

    return state


def mode_marginals_and_projections(
    state: SparseBosonicFockState, mode: int
) -> dict[int, tuple[float, SparseBosonicFockState]]:
    """Compute the marginal probabilities of sampling from `mode`.
    At the same time, for each marginal, project (without normalizing) the state to that marginal's subspace.
    """

    marginals_projections = {}

    for basis_state, amplitude in state.items():
        n_mode = number_basis.number_in_mode(basis_state, mode)
        probability = abs2(amplitude)

        try:
            marginals_projections[n_mode][0] += probability
            marginals_projections[n_mode][1][basis_state] = amplitude
        except KeyError:
            marginals_projections[n_mode] = [
                probability,
                SparseBosonicFockState({basis_state: amplitude}),
            ]

    return map_dict_vals(tuple, marginals_projections)


def mode_marginals_only(state: SparseBosonicFockState, mode: int) -> dict[int, float]:
    """Like `mode_marginals_and_projection` but without doing all the projections.
    Use this when not amortizing marginals over many samples, it should use less memory (especially memory write, hence save time).
    """

    marginals = {}

    for basis_state, amplitude in state.items():
        n_mode = number_basis.number_in_mode(basis_state, mode)
        probability = abs2(amplitude)

        try:
            marginals[n_mode] += probability
        except KeyError:
            marginals[n_mode] = probability

    return marginals


def sample_and_collapse(
    marginals_projections: dict[int, tuple[float, SparseBosonicFockState]],
) -> tuple[int, SparseBosonicFockState]:
    """Get marginals (together with their projections to subspaces) from `mode_marginals_and_projections`,
    take a sample and return a tuple (sample, projected_state). Not normalizing."""

    numbers = sorted(list(marginals_projections.keys()))
    probabilities = [marginals_projections[n][0] for n in numbers]

    sample_n = np.random.choice(numbers, p=probabilities)
    new_state = marginals_projections[sample_n][1]

    new_state.normalize_in_place()
    return sample_n, new_state


def sample_project_collapse(
    state: SparseBosonicFockState, mode: int, marginals: dict[int, float]
) -> tuple[int, SparseBosonicFockState]:
    """Get marginals (without projection) from `mode_marginals_only`, sample,
    and then project & collapse the state to the subspace of that marginal."""

    numbers = sorted(list(marginals.keys()))
    probabilities = [marginals[n] for n in numbers]

    sample_n = int(np.random.choice(numbers, p=probabilities))

    # project to subspace corresponding to the sample
    new_state = project_number_mode(mode, sample_n, state)
    new_state.normalize_in_place()

    return sample_n, new_state


def measure_whole_state(state: SparseBosonicFockState) -> tuple[int, ...]:
    basis_states, amplitudes = zip(*list(state.items()))
    probabilities = [abs2(amp) for amp in amplitudes]

    idx = np.random.choice(len(basis_states), p=probabilities)
    return basis_states[idx]


def progressive_simulation(
    input_state: tuple[int, ...] | SparseBosonicFockState,
    circuit: ConcreteBSCircuit,
    number_samples: int = 1,
):

    if isinstance(input_state, SparseBosonicFockState):
        input_state = copy(input_state)
    else:
        input_state = SparseBosonicFockState({input_state: 1})

    total_photons = max([sum(basis_state) for basis_state, _ in input_state.items()])

    components = progressive_decomposition(circuit)

    for _ in range(number_samples):

        Phi = copy(input_state)

        # In the following, simulate components corresponding to a single output mode;
        # the last component has more output modes and will be handled separately.
        for mode in range(len(components) - 1):
            X = components[mode]
            after_X = apply_bs_circuit(X, Phi)

            marginals = mode_marginals_only(after_X, mode)
            sample_in_mode, collapsed_state = sample_project_collapse(
                after_X, mode, marginals
            )

            Phi = collapsed_state
            total_photons -= sample_in_mode
            if total_photons == 0:
                break

        # after the last component, sample the whole remaining state
        X = components[-1]
        after_X = apply_bs_circuit(X, Phi)

        # in contrast to the paper, we do not delete modes here, so the entire sample is still contained as a key of the sparse vector
        final_sample = measure_whole_state(after_X)

        yield final_sample
