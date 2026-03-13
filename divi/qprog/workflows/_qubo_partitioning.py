# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import copy
import string
from functools import partial
from typing import Literal, TypeVar

import dimod
import hybrid
import numpy as np
import numpy.typing as npt
import scipy.sparse as sps
from dimod import BinaryQuadraticModel

from divi.backends import CircuitRunner
from divi.qprog.algorithms import PCE, QAOA, IterativeQAOA
from divi.qprog.ensemble import ProgramEnsemble, _beam_search_aggregate_top_n
from divi.qprog.optimizers import MonteCarloOptimizer, Optimizer, copy_optimizer
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


class QUBOPartitioningQAOA(ProgramEnsemble):
    def __init__(
        self,
        qubo: QUBOProblemTypes,
        decomposer: hybrid.traits.ProblemDecomposer,
        n_layers: int,
        backend: CircuitRunner,
        engine: Literal["qaoa", "pce", "iterative_qaoa"] = "qaoa",
        composer: hybrid.traits.SubsamplesComposer = hybrid.SplatComposer(),
        optimizer: Optimizer | None = None,
        max_iterations: int = 10,
        **kwargs,
    ):
        """
        Initialize a partitioning workflow for solving QUBO problems with QAOA or PCE.

        Args:
            qubo (QUBOProblemTypes): The QUBO problem to solve, provided as a supported type.
                Note: Variable types are assumed to be binary (not Spin).
            decomposer (hybrid.traits.ProblemDecomposer): The decomposer used to partition the QUBO problem into subproblems.
            n_layers (int): Number of ansatz layers to use for each subproblem.
            backend (CircuitRunner): Backend responsible for running quantum circuits.
            engine (Literal["qaoa", "pce", "iterative_qaoa"], optional): Per-partition quantum engine.
                Defaults to ``"qaoa"``. When ``"iterative_qaoa"`` is selected,
                ``n_layers`` is used as ``max_depth``.
            composer (hybrid.traits.SubsamplesComposer, optional): Composer to aggregate subsamples from subproblems.
                Defaults to hybrid.SplatComposer(). Only used when ``beam_width=1`` (greedy).
            optimizer (Optimizer, optional): Optimizer to use for the selected engine.
                Defaults to MonteCarloOptimizer.
            max_iterations (int, optional): Maximum number of optimization iterations.
                Defaults to 10.
            **kwargs: Additional keyword arguments forwarded to the selected
                engine constructor (e.g. ``encoding_type`` for PCE).

        """
        super().__init__(backend=backend)

        self.main_qubo, self._bqm = _sanitize_problem_input(qubo)

        self._partitioning = hybrid.Unwind(decomposer)
        self._aggregating = hybrid.Reduce(hybrid.Lambda(_merge_substates)) | composer

        self.max_iterations = max_iterations

        self.trivial_program_ids = set()

        # Store the optimizer template (will be copied for each program)
        self._optimizer_template = (
            optimizer if optimizer is not None else MonteCarloOptimizer()
        )

        # Extract early_stopping so each sub-program gets its own copy
        self._early_stopping_template = kwargs.pop("early_stopping", None)

        engine_name = engine.lower()
        if engine_name == "qaoa":
            self._engine_constructor = QAOA
        elif engine_name == "pce":
            self._engine_constructor = PCE
        elif engine_name == "iterative_qaoa":
            self._engine_constructor = IterativeQAOA
        else:
            raise ValueError(
                f"Unsupported engine: {engine!r}. "
                "Supported values are 'qaoa', 'pce', and 'iterative_qaoa'."
            )

        self.engine = engine_name

        if engine_name == "iterative_qaoa":
            self._constructor = partial(
                self._engine_constructor,
                max_depth=n_layers,
                backend=self.backend,
                **kwargs,
            )
        else:
            self._constructor = partial(
                self._engine_constructor,
                max_iterations=self.max_iterations,
                backend=self.backend,
                n_layers=n_layers,
                **kwargs,
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

            self._programs[prog_id] = self._constructor(
                program_id=prog_id,
                problem=coo_mat,
                optimizer=copy_optimizer(self._optimizer_template),
                early_stopping=copy.deepcopy(self._early_stopping_template),
                progress_queue=self._queue,
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

    def _evaluate_solution(self, solution):
        """Evaluate a QUBO solution using the BQM energy function.

        Args:
            solution (list[int]): Binary solution vector.

        Returns:
            float: The BQM energy for the given solution.
        """
        variables = list(self._bqm.variables)
        sample = dict(zip(variables, solution))
        return self._bqm.energy(sample)

    def aggregate_results(
        self, beam_width=1, n_partition_candidates=None
    ) -> tuple[npt.NDArray[np.int32], float]:
        """
        Aggregate results from all QUBO subproblems into a global solution.

        Uses the hybrid framework composer to assemble per-partition solutions
        into a global result.  When ``beam_width > 1`` or ``beam_width is None``,
        beam search is used first to select the best candidate combination
        across partitions before feeding them through the composer.

        Args:
            beam_width (int | None): Width of the beam search. Defaults to ``1``
                (greedy).
            n_partition_candidates (int | None): Number of candidate bitstrings
                to extract from each partition. Defaults to ``beam_width``.

        Returns:
            tuple: A tuple containing:
                - solution (npt.NDArray[np.int32]): Binary solution vector for the QUBO problem.
                - solution_energy (float): Energy/cost of the solution.

        Raises:
            RuntimeError: If programs haven't been run or if final probabilities
                haven't been computed.
        """
        super().aggregate_results()

        if any(len(program.best_probs) == 0 for program in self.programs.values()):
            raise RuntimeError(
                "Not all final probabilities computed yet. Please call `run()` first."
            )

        n_vars = len(self._bqm.variables)
        _, best_solution = _beam_search_aggregate_top_n(
            programs=self._programs,
            initial_solution=[0] * n_vars,
            extend_fn=self._extend_solution,
            evaluate_fn=self._evaluate_solution,
            beam_width=beam_width,
            n_partition_candidates=n_partition_candidates,
        )[0]

        self.solution, self.solution_energy = self._compose_solution(best_solution)

        return self.solution, self.solution_energy

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

    def get_top_solutions(self, n=10, *, beam_width=1, n_partition_candidates=None):
        """Get the top-N global solutions as ``(solution_array, energy)`` tuples.

        Each solution is run through the hybrid composer pipeline.

        Args:
            n (int): Number of solutions to return. Must be >= 1.
            beam_width (int | None): Beam width for search.
            n_partition_candidates (int | None): Candidates per partition.

        Returns:
            list[tuple[npt.NDArray[np.int32], float]]: Each element is
                ``(solution_vector, energy)``, ordered best-first by energy.
        """
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")

        self._check_ready_for_aggregation()

        if any(len(program.best_probs) == 0 for program in self.programs.values()):
            raise RuntimeError(
                "Not all final probabilities computed yet. Please call `run()` first."
            )

        n_vars = len(self._bqm.variables)
        top_results = _beam_search_aggregate_top_n(
            programs=self._programs,
            initial_solution=[0] * n_vars,
            extend_fn=self._extend_solution,
            evaluate_fn=self._evaluate_solution,
            beam_width=beam_width,
            n_partition_candidates=n_partition_candidates,
            top_n=n,
        )

        composed = [
            self._compose_solution(solution) for _score, solution in top_results
        ]
        composed.sort(key=lambda entry: entry[1])
        return composed
