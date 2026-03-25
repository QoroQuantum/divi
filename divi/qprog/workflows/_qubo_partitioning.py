# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import string
from functools import partial
from typing import Literal, TypeVar

import dimod
import hybrid
import numpy as np
import scipy.sparse as sps
from dimod import BinaryQuadraticModel

from divi.backends import CircuitRunner
from divi.qprog.algorithms import PCE, QAOA, IterativeQAOA
from divi.qprog.optimizers import Optimizer
from divi.qprog.problems import BinaryOptimizationProblem
from divi.qprog.workflows._partitioning_ensemble import (
    PartitioningProgramEnsemble,
)
from divi.typing import QUBOProblemTypes


# Helper function to merge subsamples in-place
def _merge_substates(_, substates):
    a, b = substates
    return a.updated(subsamples=hybrid.hstack_samplesets(a.subsamples, b.subsamples))


T = TypeVar("T", bound=QUBOProblemTypes | BinaryQuadraticModel)


def _sanitize_problem_input(qubo: T) -> tuple[T, BinaryQuadraticModel]:
    if isinstance(qubo, BinaryQuadraticModel):
        return qubo, qubo

    if isinstance(qubo, (np.ndarray, sps.spmatrix)):
        x, y = qubo.shape
        if x != y:
            raise ValueError("Only matrices supported.")

    if isinstance(qubo, np.ndarray):
        return qubo, dimod.BinaryQuadraticModel(qubo, vartype=dimod.Vartype.BINARY)

    if isinstance(qubo, sps.spmatrix):
        return qubo, dimod.BinaryQuadraticModel(
            {(row, col): data for row, col, data in zip(qubo.row, qubo.col, qubo.data)},
            vartype=dimod.Vartype.BINARY,
        )

    raise ValueError(f"Got an unsupported QUBO input format: {type(qubo)}")


class QUBOPartitioning(PartitioningProgramEnsemble):
    def __init__(
        self,
        qubo: QUBOProblemTypes,
        decomposer: hybrid.traits.ProblemDecomposer,
        n_layers: int,
        backend: CircuitRunner,
        optimizer: Optimizer,
        quantum_routine: Literal["qaoa", "pce", "iterative_qaoa"] = "qaoa",
        composer: hybrid.traits.SubsamplesComposer = hybrid.SplatComposer(),
        max_iterations: int = 10,
        **kwargs,
    ):
        """
        Initialize a partitioning workflow for solving QUBO problems.

        Args:
            qubo (QUBOProblemTypes): The QUBO problem to solve, provided as a supported type.
                Note: Variable types are assumed to be binary (not Spin).
            decomposer (hybrid.traits.ProblemDecomposer): The decomposer used to partition the QUBO problem into subproblems.
            n_layers (int): Number of ansatz layers to use for each subproblem.
            backend (CircuitRunner): Backend responsible for running quantum circuits.
            quantum_routine (Literal["qaoa", "pce", "iterative_qaoa"], optional): Per-partition
                quantum algorithm.  Defaults to ``"qaoa"``. When ``"iterative_qaoa"`` is
                selected, ``n_layers`` is used as ``max_depth``.
            composer (hybrid.traits.SubsamplesComposer, optional): Composer to aggregate subsamples from subproblems.
                Defaults to hybrid.SplatComposer(). Only used when ``beam_width=1`` (greedy).
            optimizer (Optimizer): Optimizer to use for each sub-program.
            max_iterations (int, optional): Maximum number of optimization iterations.
                Defaults to 10.
            **kwargs: Additional keyword arguments forwarded to the selected
                engine constructor (e.g. ``encoding_type`` for PCE).

        """
        super().__init__(
            backend=backend,
            quantum_routine=quantum_routine,
            optimizer=optimizer,
            max_iterations=max_iterations,
            **kwargs,
        )

        self.main_qubo, self._bqm = _sanitize_problem_input(qubo)

        self._partitioning = hybrid.Unwind(decomposer)
        self._aggregating = hybrid.Reduce(hybrid.Lambda(_merge_substates)) | composer

        self.trivial_program_ids = set()

        routine = self.quantum_routine.lower()
        if routine == "qaoa":
            self._engine_constructor = QAOA
        elif routine == "pce":
            self._engine_constructor = PCE
        elif routine == "iterative_qaoa":
            self._engine_constructor = IterativeQAOA
        else:
            raise ValueError(
                f"Unsupported quantum_routine: {quantum_routine!r}. "
                "Supported values are 'qaoa', 'pce', and 'iterative_qaoa'."
            )

        if routine == "iterative_qaoa":
            self._constructor = partial(
                self._engine_constructor,
                max_depth=n_layers,
                backend=self.backend,
                **self._engine_kwargs,
            )
        else:
            self._constructor = partial(
                self._engine_constructor,
                max_iterations=self.max_iterations,
                backend=self.backend,
                n_layers=n_layers,
                **self._engine_kwargs,
            )

    def create_programs(self):
        """
        Partition the main QUBO problem and instantiate engine programs for each subproblem.

        This implementation:
        - Uses the configured decomposer to split the main QUBO into subproblems.
        - For each subproblem, creates a QAOA/PCE program with the specified parameters.
        - Stores each program in `self.programs` with a unique identifier.

        Unique Identifier Format:
            Each key in `self.programs` is a tuple of the form (letter, size), where:
            - letter: An uppercase letter ('A', 'B', 'C', ...) indicating the partition index.
            - size: The number of variables in the subproblem.

            Example: ('A', 5) refers to the first partition with 5 variables.
        """

        super().create_programs()

        self.prog_id_to_bqm_subproblem_states = {}
        self._variable_maps = {}  # prog_id -> list of global variable indices

        init_state = hybrid.State.from_problem(self._bqm)
        _bqm_partitions = self._partitioning.run(init_state).result()

        # Build a variable-to-index map for the full BQM
        all_variables = list(self._bqm.variables)
        var_to_global_idx = {v: i for i, v in enumerate(all_variables)}

        for i, partition in enumerate(_bqm_partitions):
            if i > 0:
                # We only need 'problem' on the first partition since
                # it will propagate to the other partitions during
                # aggregation, otherwise it's a waste of memory
                del partition["problem"]

            prog_id = (string.ascii_uppercase[i], len(partition.subproblem))
            self.prog_id_to_bqm_subproblem_states[prog_id] = partition

            # Store mapping: local position -> global variable index
            self._variable_maps[prog_id] = [
                var_to_global_idx[v] for v in partition.subproblem.variables
            ]

            if partition.subproblem.num_interactions == 0:
                # Skip creating a full QAOA program for this trivial case.
                self.trivial_program_ids.add(prog_id)
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

            # QAOA and IterativeQAOA expect Problem objects;
            # PCE takes raw QUBO matrices directly.
            if self._engine_constructor in (QAOA, IterativeQAOA):
                problem = BinaryOptimizationProblem(coo_mat)
            else:
                problem = coo_mat

            self._programs[prog_id] = self._constructor(
                problem=problem,
                **self._make_program_args(prog_id),
            )

    def _extend_solution(self, current_solution, prog_id, candidate):
        """Extend a global solution with a partition candidate's decoded bits.

        Uses ``_variable_maps`` to map the candidate's local variable positions
        to global positions in the solution vector.

        Args:
            current_solution (list[int]): Current global solution vector.
            prog_id: Program identifier (key into ``_variable_maps``).
            candidate: A ``SolutionEntry`` with ``decoded`` containing a numpy
                array of binary values for the subproblem variables.

        Returns:
            list[int]: A new solution vector with the candidate's bits applied.
        """
        extended = list(current_solution)
        global_indices = self._variable_maps[prog_id]

        for local_idx, global_idx in enumerate(global_indices):
            extended[global_idx] = int(candidate.decoded[local_idx])

        return extended

    def _evaluate_global_solution(self, solution):
        """Evaluate a QUBO solution using the BQM energy function.

        Args:
            solution (list[int]): Binary solution vector.

        Returns:
            float: The BQM energy for the given solution.
        """
        variables = list(self._bqm.variables)
        sample = dict(zip(variables, solution))
        return self._bqm.energy(sample)

    def _initial_solution(self):
        return [0] * len(self._bqm.variables)

    def _finalize_best(self, score, solution):
        self.solution, self.solution_energy = self._compose_solution(solution)
        return self.solution, self.solution_energy

    def _format_top_results(self, results):
        composed = [self._compose_solution(solution) for _score, solution in results]
        composed.sort(key=lambda entry: entry[1])
        return composed

    def _compose_solution(self, solution):
        """Run a single solution through the hybrid composer pipeline.

        Operates on a copy of the subproblem states to avoid mutating
        shared state.

        Returns:
            tuple[npt.NDArray[np.int32], float]: ``(solution_array, energy)``.
        """
        states_copy = {}
        for (
            prog_id,
            bqm_subproblem_state,
        ) in self.prog_id_to_bqm_subproblem_states.items():
            if prog_id in self.trivial_program_ids:
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
