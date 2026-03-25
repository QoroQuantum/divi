# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Binary optimization (QUBO / HUBO) problem class for QAOA."""

from __future__ import annotations

import string
from collections.abc import Callable
from typing import Any, Literal

import dimod
import hybrid
import numpy as np
import pennylane as qml
import pennylane.qaoa as pqaoa
import scipy.sparse as sps
from dimod import BinaryQuadraticModel

from divi.hamiltonians import normalize_binary_polynomial_problem, qubo_to_ising
from divi.qprog.problems._base import QAOAProblem
from divi.typing import HUBOProblemTypes, QUBOProblemTypes


def _merge_substates(_, substates):
    """Merge two hybrid framework substates by stacking their sample sets."""
    a, b = substates
    return a.updated(subsamples=hybrid.hstack_samplesets(a.subsamples, b.subsamples))


def _sanitize_problem_input(qubo):
    """Normalize a QUBO input to (original, BinaryQuadraticModel) pair."""
    if isinstance(qubo, BinaryQuadraticModel):
        return qubo, qubo

    if isinstance(qubo, (np.ndarray, sps.spmatrix)):
        x, y = qubo.shape
        if x != y:
            raise ValueError("Only square matrices are supported.")

    if isinstance(qubo, np.ndarray):
        return qubo, dimod.BinaryQuadraticModel(qubo, vartype=dimod.Vartype.BINARY)

    if isinstance(qubo, sps.spmatrix):
        return qubo, dimod.BinaryQuadraticModel(
            {(row, col): data for row, col, data in zip(qubo.row, qubo.col, qubo.data)},
            vartype=dimod.Vartype.BINARY,
        )

    raise ValueError(f"Got an unsupported QUBO input format: {type(qubo)}")


class BinaryOptimizationProblem(QAOAProblem):
    """Generic QUBO or HUBO problem.

    Normalises the input, converts to an Ising Hamiltonian, and provides
    a standard X-mixer with equal superposition initial state.

    Args:
        problem: QUBO matrix, BinaryQuadraticModel, HUBO dict, or
            BinaryPolynomial.
        hamiltonian_builder: Ising conversion backend (``"native"`` or
            ``"quadratized"``).
        quadratization_strength: Penalty strength for quadratization.
    """

    def __init__(
        self,
        problem: QUBOProblemTypes | HUBOProblemTypes,
        *,
        hamiltonian_builder: Literal["native", "quadratized"] = "native",
        quadratization_strength: float = 10.0,
        decomposer: hybrid.traits.ProblemDecomposer | None = None,
        composer: hybrid.traits.SubsamplesComposer | None = None,
    ):
        self._raw_problem = problem
        self._canonical_problem = normalize_binary_polynomial_problem(problem)
        self._ising = qubo_to_ising(
            problem,
            hamiltonian_builder=hamiltonian_builder,
            quadratization_strength=quadratization_strength,
        )
        self._mixer_hamiltonian = pqaoa.x_mixer(range(self._ising.n_qubits))

        # Decomposition support (optional)
        self._decomposer = decomposer
        if decomposer is not None:
            _, self._bqm = _sanitize_problem_input(problem)
            self._partitioning = hybrid.Unwind(decomposer)
            self._aggregating = hybrid.Reduce(hybrid.Lambda(_merge_substates)) | (
                composer or hybrid.SplatComposer()
            )
        else:
            self._bqm = None

        self._variable_maps = {}
        self._trivial_program_ids = set()
        self._bqm_subproblem_states = {}

    @property
    def cost_hamiltonian(self) -> qml.operation.Operator:
        return self._ising.cost_hamiltonian

    @property
    def mixer_hamiltonian(self) -> qml.operation.Operator:
        return self._mixer_hamiltonian

    @property
    def loss_constant(self) -> float:
        return self._ising.loss_constant

    @property
    def decode_fn(self) -> Callable[[str], Any]:
        base_decode = self._ising.encoding.decode_fn
        vo = self._canonical_problem.variable_order

        if vo != tuple(range(self._canonical_problem.n_vars)):

            def _decode_with_names(bitstring: str) -> dict | None:
                decoded = base_decode(bitstring)
                if decoded is None:
                    return None
                return dict(zip(vo, decoded))

            return _decode_with_names
        return base_decode

    @property
    def metadata(self) -> dict[str, Any]:
        return self._ising.encoding.metadata or {}

    @property
    def canonical_problem(self):
        """The normalised ``BinaryPolynomialProblem``."""
        return self._canonical_problem

    @property
    def raw_problem(self):
        """The original QUBO/HUBO input passed at construction time."""
        return self._raw_problem

    def decompose(self) -> dict[tuple[str, int], QAOAProblem]:
        if self._decomposer is None:
            raise ValueError(
                "Cannot decompose: no decomposer was provided at construction."
            )

        self._bqm_subproblem_states = {}
        self._variable_maps = {}
        self._trivial_program_ids = set()

        init_state = hybrid.State.from_problem(self._bqm)
        _bqm_partitions = self._partitioning.run(init_state).result()

        all_variables = list(self._bqm.variables)
        var_to_global_idx = {v: i for i, v in enumerate(all_variables)}

        sub_problems = {}

        for i, partition in enumerate(_bqm_partitions):
            if i > 0:
                del partition["problem"]

            prog_id = (string.ascii_uppercase[i], len(partition.subproblem))
            self._bqm_subproblem_states[prog_id] = partition

            self._variable_maps[prog_id] = [
                var_to_global_idx[v] for v in partition.subproblem.variables
            ]

            if partition.subproblem.num_interactions == 0:
                self._trivial_program_ids.add(prog_id)
                continue

            ldata, (irow, icol, qdata), _ = partition.subproblem.to_numpy_vectors(
                partition.subproblem.variables
            )

            coo_mat = sps.coo_matrix(
                (
                    np.r_[ldata, qdata],
                    (
                        np.r_[np.arange(len(ldata)), icol],
                        np.r_[np.arange(len(ldata)), irow],
                    ),
                ),
                shape=(len(ldata), len(ldata)),
            )

            sub_problems[prog_id] = BinaryOptimizationProblem(coo_mat)

        return sub_problems

    def initial_solution_size(self) -> int:
        return len(self._bqm.variables)

    def extend_solution(
        self,
        current_solution: list[int],
        prog_id: tuple[str, int],
        candidate_decoded: list[int],
    ) -> list[int]:
        extended = list(current_solution)
        global_indices = self._variable_maps[prog_id]

        for local_idx, global_idx in enumerate(global_indices):
            extended[global_idx] = int(candidate_decoded[local_idx])

        return extended

    def evaluate_global_solution(self, solution: list[int]) -> float:
        variables = list(self._bqm.variables)
        sample = dict(zip(variables, solution))
        return self._bqm.energy(sample)

    def finalize_solution(
        self, score: float, solution: list[int]
    ) -> tuple[np.ndarray, float]:
        return self._compose_solution(solution)

    def format_top_solutions(
        self, results: list[tuple[float, list[int]]]
    ) -> list[tuple[np.ndarray, float]]:
        composed = [self._compose_solution(sol) for _score, sol in results]
        composed.sort(key=lambda entry: entry[1])
        return composed

    def _compose_solution(self, solution):
        """Run a single solution through the hybrid composer pipeline."""
        states_copy = {}
        for prog_id, bqm_subproblem_state in self._bqm_subproblem_states.items():
            if prog_id in self._trivial_program_ids:
                var_to_val = {v: 0 for v in bqm_subproblem_state.subproblem.variables}
            else:
                variables = list(bqm_subproblem_state.subproblem.variables)
                global_indices = self._variable_maps[prog_id]
                var_to_val = {
                    v: solution[gi] for v, gi in zip(variables, global_indices)
                }

            sample_set = dimod.SampleSet.from_samples(
                dimod.as_samples(var_to_val), "BINARY", 0
            )
            states_copy[prog_id] = bqm_subproblem_state.updated(subsamples=sample_set)

        states = hybrid.States(*list(states_copy.values()))
        final_state = self._aggregating.run(states).result()

        sol, energy, _ = final_state.samples.record[0]
        return np.array(sol, dtype=np.int32), float(energy)
