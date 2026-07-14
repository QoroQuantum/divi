# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Binary optimization (QUBO / HUBO) problem class for QAOA."""

import math
from collections.abc import Callable, Hashable
from typing import Any, Literal

import dimod
import hybrid
import numpy as np
import scipy.sparse as sps
from dimod import BinaryQuadraticModel
from qiskit.quantum_info import SparsePauliOp

from divi.hamiltonians import (
    HUBOProblemTypes,
    IsingResult,
    QUBOProblemTypes,
    normalize_binary_polynomial_problem,
    qubo_to_ising,
    x_mixer,
)
from divi.qprog.problems import QAOAProblem


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
        coo = sps.coo_matrix(qubo)
        return qubo, dimod.BinaryQuadraticModel(
            {(row, col): data for row, col, data in zip(coo.row, coo.col, coo.data)},
            vartype=dimod.Vartype.BINARY,
        )

    if isinstance(qubo, dict):
        linear = {}
        quadratic = {}
        for key, coeff in qubo.items():
            if not isinstance(key, tuple):
                raise ValueError(f"Got an unsupported QUBO input format: {type(qubo)}")
            if len(key) == 1:
                linear[key[0]] = linear.get(key[0], 0.0) + float(coeff)
            elif len(key) == 2:
                u, v = key
                if u == v:
                    linear[u] = linear.get(u, 0.0) + float(coeff)
                else:
                    quadratic[(u, v)] = quadratic.get((u, v), 0.0) + float(coeff)
            else:
                raise ValueError("Decomposition only supports quadratic problems.")
        return qubo, dimod.BinaryQuadraticModel(
            linear, quadratic, 0.0, vartype=dimod.Vartype.BINARY
        )

    raise ValueError(f"Got an unsupported QUBO input format: {type(qubo)}")


def _combine_polynomial_terms(cost_canonical, penalty_canonical, penalty_weight: float):
    """Return objective + penalty_weight * penalty in canonical term form."""
    terms = dict(cost_canonical.terms)
    for term_key, coeff in penalty_canonical.terms.items():
        combined = float(terms.get(term_key, 0.0)) + penalty_weight * float(coeff)
        if combined == 0.0:
            terms.pop(term_key, None)
        else:
            terms[term_key] = combined
    return terms


class BinaryOptimizationProblem(QAOAProblem):
    """Generic QUBO or HUBO problem for QAOA.

    Wraps a binary optimization problem expressed as either a quadratic
    form (QUBO) or higher-order polynomial (HUBO), normalizes it to a
    canonical :class:`~divi.hamiltonians.BinaryPolynomialProblem`, and
    exposes the QAOA building blocks: cost Hamiltonian (via Ising
    conversion), standard X-mixer, and ground-state initial
    superposition.

    Accepted inputs (anything :class:`dimod.BinaryQuadraticModel` or
    :class:`dimod.BinaryPolynomial` accepts, plus matrices):

    - ``np.ndarray`` / ``scipy.sparse.spmatrix`` — square QUBO matrix.
    - :class:`dimod.BinaryQuadraticModel` — quadratic form with named
      variables.
    - ``dict`` with tuple keys — HUBO terms, e.g.
      ``{(0,): -1.0, (0, 1, 2): 2.0}``.
    - :class:`dimod.BinaryPolynomial` — polynomial of arbitrary degree.

    Ising-conversion strategies selected via ``hamiltonian_builder``:

    - ``"native"`` (default): translates each polynomial term into a
      Pauli-Z product. Exact, but high-degree terms produce many-body
      interactions that some simulators handle slowly.
    - ``"quadratized"``: introduces auxiliary qubits and penalty terms
      so every interaction becomes two-body. Penalty magnitude is
      ``quadratization_strength``; ``None`` picks
      ``2 * max(|hubo coeff|)``.

    Optionally accepts a ``dimod.hybrid`` decomposer/composer pair to
    enable partitioned solving via :meth:`decompose`. Without one, the
    decomposition-related methods raise ``RuntimeError``.

    Args:
        problem: Objective/cost QUBO matrix, BQM, HUBO dict, or BinaryPolynomial.
        penalty: Optional penalty-only QUBO/HUBO component. When provided, the
            QAOA problem is the penalized objective
            ``problem + penalty_weight * penalty`` while the objective/penalty
            split remains available for characterization.
        penalty_weight: Multiplier applied to ``penalty`` when building the
            penalized QUBO/HUBO. Defaults to ``1.0``.
        hamiltonian_builder: ``"native"`` (default) or ``"quadratized"``.
        quadratization_strength: Penalty strength for the quadratized
            builder. ``None`` (default) auto-picks
            ``2 * max(|hubo coeff|)``. Ignored when
            ``hamiltonian_builder="native"``. The auto default is sized
            against a single worst-case term and may under-penalise dense
            HUBOs where many constraints can be violated simultaneously;
            pass an explicit value (or raise the multiplier on
            :class:`~divi.hamiltonians.QuadratizedIsingConverter`) for
            such instances.
        decomposer: Optional ``hybrid.traits.ProblemDecomposer`` that
            enables :meth:`decompose`.
        composer: Optional ``hybrid.traits.SubsamplesComposer`` for
            recombining sub-solutions; defaults to
            ``hybrid.SplatComposer``.

    Examples:
        >>> import numpy as np
        >>> from divi.qprog.problems import BinaryOptimizationProblem
        >>> Q = np.array([[-1.0, 2.0], [0.0, -1.0]])
        >>> problem = BinaryOptimizationProblem(Q)
        >>> problem.cost_hamiltonian  # ready for QAOA
    """

    def __init__(
        self,
        problem: QUBOProblemTypes | HUBOProblemTypes,
        *,
        penalty: QUBOProblemTypes | HUBOProblemTypes | None = None,
        penalty_weight: float = 1.0,
        hamiltonian_builder: Literal["native", "quadratized"] = "native",
        quadratization_strength: float | None = None,
        decomposer: hybrid.traits.ProblemDecomposer | None = None,
        composer: hybrid.traits.SubsamplesComposer | None = None,
    ):
        if hamiltonian_builder not in ("native", "quadratized"):
            raise ValueError(
                "hamiltonian_builder must be either 'native' or 'quadratized'."
            )
        penalty_weight = float(penalty_weight)
        if not math.isfinite(penalty_weight):
            raise ValueError("penalty_weight must be finite.")

        self._objective_problem = problem
        self._objective_canonical_problem = normalize_binary_polynomial_problem(problem)
        self._penalty_problem = penalty
        self._penalty_canonical_problem = (
            normalize_binary_polynomial_problem(penalty)
            if penalty is not None
            else None
        )
        self._penalty_weight = penalty_weight
        if self._penalty_canonical_problem is None:
            self._raw_problem = problem
            self._canonical_problem = self._objective_canonical_problem
        else:
            self._raw_problem = _combine_polynomial_terms(
                self._objective_canonical_problem,
                self._penalty_canonical_problem,
                penalty_weight,
            )
            self._canonical_problem = normalize_binary_polynomial_problem(
                self._raw_problem
            )
        self._hamiltonian_builder: Literal["native", "quadratized"] = (
            hamiltonian_builder
        )
        self._quadratization_strength = quadratization_strength
        self._ising_cache: IsingResult | None = None
        self._mixer_cache: SparsePauliOp | None = None

        # Decomposition support (optional)
        self._decomposer = decomposer
        self._bqm: dimod.BinaryQuadraticModel | None
        if decomposer is not None:
            _, self._bqm = _sanitize_problem_input(self._raw_problem)
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
    def _ising(self) -> IsingResult:
        """Cached Ising conversion of the canonical polynomial."""
        if self._ising_cache is None:
            self._ising_cache = qubo_to_ising(
                self._raw_problem,
                hamiltonian_builder=self._hamiltonian_builder,
                quadratization_strength=self._quadratization_strength,
            )
        return self._ising_cache

    @property
    def cost_hamiltonian(self) -> SparsePauliOp:
        """Cost Hamiltonian derived from the Ising conversion of the QUBO/HUBO."""
        return self._ising.cost_hamiltonian

    @property
    def mixer_hamiltonian(self) -> SparsePauliOp:
        """Standard X-mixer over all qubits in the Ising encoding."""
        if self._mixer_cache is None:
            self._mixer_cache = x_mixer(self._ising.n_qubits)
        return self._mixer_cache

    @property
    def loss_constant(self) -> float:
        """Constant offset from Ising conversion, added back to expectation values."""
        return self._ising.loss_constant

    @property
    def decode_fn(self) -> Callable[[str], Any]:
        """Decode a measurement bitstring to the original variable assignment.

        For problems built from named variables (e.g. a BQM with
        non-integer keys), the result is a ``dict`` mapping the original
        variable names to their bit values. For integer-indexed
        problems, returns the encoding's raw bitstring projection.
        """
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
        """Encoding metadata from the Ising conversion (e.g. quadratization aux info)."""
        return self._ising.encoding.metadata or {}

    @property
    def canonical_problem(self):
        """The normalized ``BinaryPolynomialProblem``."""
        return self._canonical_problem

    @property
    def objective_problem(self):
        """The objective/cost component passed as ``problem``."""
        return self._objective_problem

    @property
    def objective_canonical_problem(self):
        """The normalized objective/cost component."""
        return self._objective_canonical_problem

    @property
    def penalty_problem(self):
        """The penalty-only component, if one was provided."""
        return self._penalty_problem

    @property
    def penalty_canonical_problem(self):
        """The normalized penalty-only component, if one was provided."""
        return self._penalty_canonical_problem

    @property
    def penalty_weight(self) -> float:
        """Multiplier applied to ``penalty_problem`` in the full QUBO/HUBO."""
        return self._penalty_weight

    @property
    def raw_problem(self):
        """The QUBO/HUBO input used for QAOA.

        This is the original objective when no penalty was provided, otherwise
        the combined penalized objective.
        """
        return self._raw_problem

    def decompose(self) -> dict[Hashable, QAOAProblem]:
        """Partition the problem using the configured ``hybrid`` decomposer.

        Each non-trivial partition becomes its own
        :class:`BinaryOptimizationProblem` keyed by ``(name, size)``.
        Partitions with no interactions are tracked internally and
        skipped during composition.

        Raises:
            ValueError: If no decomposer was provided at construction.
        """
        if self._decomposer is None or self._bqm is None:
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

        sub_problems: dict[Hashable, QAOAProblem] = {}

        for i, partition in enumerate(_bqm_partitions):
            if i > 0:
                del partition["problem"]

            prog_id = (f"P{i}", len(partition.subproblem))
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
        """Number of variables in the global solution vector.

        Equals the number of variables in the underlying BQM. Only
        defined when a decomposer was provided at construction.

        Raises:
            RuntimeError: If no decomposer was provided.
        """
        if self._bqm is None:
            raise RuntimeError(
                "initial_solution_size requires a decomposer to have been "
                "provided at construction."
            )
        return len(self._bqm.variables)

    def extend_solution(
        self,
        current_solution: list[int],
        prog_id: Hashable,
        candidate_decoded: list[int],
    ) -> list[int]:
        """Splice a sub-problem's decoded bits into the global solution.

        Returns a new list with values at positions corresponding to
        ``prog_id``'s variables overwritten by ``candidate_decoded``.
        """
        extended = list(current_solution)
        global_indices = self._variable_maps[prog_id]

        for local_idx, global_idx in enumerate(global_indices):
            extended[global_idx] = int(candidate_decoded[local_idx])

        return extended

    def evaluate_global_solution(self, solution: list[int]) -> float:
        """Energy of a global bit assignment under the underlying BQM.

        Raises:
            RuntimeError: If no decomposer was provided at construction.
        """
        if self._bqm is None:
            raise RuntimeError(
                "evaluate_global_solution requires a decomposer to have been "
                "provided at construction."
            )
        variables = list(self._bqm.variables)
        sample = dict(zip(variables, solution))
        return float(self._bqm.energy(sample))

    def postprocess_candidates(
        self, candidates: list[tuple[float, list[int]]], *, strict: bool = False
    ) -> list[tuple[np.ndarray, float]]:
        """Run global candidate solutions through the configured composer.

        Returns:
            Tuples ``(composed_solution, energy)`` where
            ``composed_solution`` is an ``int32`` ndarray of bits and
            ``energy`` is the objective value computed by the composer.
        """
        composed = [self._compose_solution(sol) for _score, sol in candidates]
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
