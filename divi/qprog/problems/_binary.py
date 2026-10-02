# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Binary optimisation (QUBO / HUBO) problem class for QAOA."""

import math
from collections.abc import Callable, Hashable, Mapping, Sequence
from types import ModuleType
from typing import TYPE_CHECKING, Any, Literal

import dimod
import numpy as np
import scipy.sparse as sps
from dimod import BinaryQuadraticModel
from qiskit.quantum_info import SparsePauliOp

from divi._optional import import_optional
from divi.hamiltonians import (
    HUBOProblemTypes,
    IsingResult,
    QUBOProblemTypes,
    normalize_binary_polynomial_problem,
    qubo_to_ising,
    x_mixer,
)
from divi.qprog.problems import QAOAProblem
from divi.qprog.problems._constraints import (
    LinearConstraint,
    _encode_constraints,
    _Slack,
)
from divi.qprog.problems._qubo_partitioning_utils import bqm_to_sparse

if TYPE_CHECKING:
    # pyrefly: ignore[missing-import]
    import hybrid


def _hybrid() -> ModuleType:
    return import_optional(
        "hybrid",
        extra="qubo-decompose",
        capability="Partitioning a BinaryOptimizationProblem",
        hint="Plain QUBO/HUBO problems work without it.",
    )


def _merge_substates(_, substates):
    """Merge two hybrid framework substates by stacking their sample sets."""
    a, b = substates
    return a.updated(subsamples=_hybrid().hstack_samplesets(a.subsamples, b.subsamples))


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


def _declared_variables(problem, canonical) -> set[Hashable]:
    """Variables a QUBO/HUBO input declares, including ones with zero coefficients."""
    if isinstance(problem, dimod.BinaryQuadraticModel):
        return set(problem.variables)
    return set(canonical.variable_order)


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


def _induced_sub_qubo(bqm, variables):
    """Sub-QUBO induced on ``variables``, relabeled to local indices ``0..m-1``.

    Keeps the linear biases of ``variables`` and the quadratic biases between them;
    couplings crossing the ``variables`` boundary are dropped.
    """
    local = {v: i for i, v in enumerate(variables)}
    cset = set(variables)
    linear = {local[v]: bqm.linear[v] for v in variables}
    quadratic = {
        (local[u], local[v]): q
        for (u, v), q in bqm.quadratic.items()
        if u in cset and v in cset
    }
    return dimod.BinaryQuadraticModel(linear, quadratic, 0.0, dimod.Vartype.BINARY)


class BinaryOptimizationProblem(QAOAProblem):
    """Generic QUBO or HUBO problem for QAOA.

    Wraps a binary optimisation problem expressed as either a quadratic
    form (QUBO) or higher-order polynomial (HUBO), normalises it to a
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

    Optionally accepts a ``dimod.hybrid`` decomposer/composer pair to enable
    partitioned solving via :meth:`decompose`. For structure-aware decomposition of
    QUBOs with community structure, pass
    :class:`~divi.qprog.problems.CommunityDecomposer` as the ``decomposer``.
    Without a decomposer, the decomposition-related methods raise ``RuntimeError``.

    ``constraints`` (:class:`~divi.qprog.problems.LinearConstraint`) join the
    penalty component. Slack qubits follow the decision variables when those
    have integer labels.
    :attr:`decode_fn` drops the slack, :meth:`is_feasible` checks the
    constraints and :meth:`compute_energy` returns the objective alone.
    Constraints cannot be combined with a ``decomposer``.

    Args:
        problem: Objective/cost QUBO matrix, BQM, HUBO dict, or BinaryPolynomial.
        constraints: Optional linear constraints over the problem's variables.
        penalty: Optional penalty-only QUBO/HUBO component. When provided, the
            QAOA problem is the penalised objective
            ``problem + penalty_weight * penalty`` while the objective/penalty
            split remains available for characterisation.
        penalty_weight: Multiplier applied to ``penalty`` and to the encoded
            ``constraints`` when building the penalised QUBO/HUBO. Defaults to
            ``1.0``.
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
        local_search: If ``True``, refine each aggregated candidate to a local
            minimum by greedy single-bit-flip descent against the full QUBO.
            Works with any ``decomposer``.

    Raises:
        ImportError: If ``decomposer`` is given but the ``qubo-decompose``
            extra is not installed.
        ValueError: If a constraint is infeasible, its slack range cannot be
            encoded, or ``constraints`` is combined with a ``decomposer``.

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
        constraints: Sequence[LinearConstraint] | None = None,
        penalty: QUBOProblemTypes | HUBOProblemTypes | None = None,
        penalty_weight: float = 1.0,
        hamiltonian_builder: Literal["native", "quadratized"] = "native",
        quadratization_strength: float | None = None,
        decomposer: "hybrid.traits.ProblemDecomposer | None" = None,
        composer: "hybrid.traits.SubsamplesComposer | None" = None,
        local_search: bool = False,
    ):
        if constraints and decomposer is not None:
            raise ValueError(
                "constraints cannot be combined with a decomposer: partitioning "
                "would split each constraint's penalty across sub-problems."
            )
        hybrid = _hybrid() if decomposer is not None else None
        if hamiltonian_builder not in ("native", "quadratized"):
            raise ValueError(
                "hamiltonian_builder must be either 'native' or 'quadratized'."
            )
        penalty_weight = float(penalty_weight)
        if not math.isfinite(penalty_weight) or penalty_weight <= 0:
            raise ValueError("penalty_weight must be finite and positive.")
        constraints = tuple(constraints or ())
        for constraint in constraints:
            if not isinstance(constraint, LinearConstraint):
                raise TypeError(
                    "constraints must be LinearConstraint objects, got "
                    f"{type(constraint).__name__}."
                )

        self._objective_problem = problem
        self._objective_canonical_problem = normalize_binary_polynomial_problem(problem)
        self._constraints = constraints
        self._slacks: tuple[_Slack, ...] = ()
        if constraints:
            variables = _declared_variables(problem, self._objective_canonical_problem)
            penalty_terms: dict[tuple, float] = {}
            if penalty is not None:
                penalty_canonical = normalize_binary_polynomial_problem(penalty)
                variables |= _declared_variables(penalty, penalty_canonical)
                penalty_terms = penalty_canonical.terms
            self._slacks, penalty = _encode_constraints(
                constraints, variables, penalty_terms, self._activity_bounds
            )
        self._slack_variables = tuple(
            label for slack in self._slacks for label, _ in slack.terms
        )
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
            # A declared variable keeps its qubit even if its terms cancel.
            declared = (
                *self._objective_canonical_problem.variable_order,
                *self._penalty_canonical_problem.variable_order,
                *(var for constraint in constraints for var in constraint.coefficients),
            )
            for var in declared:
                self._raw_problem.setdefault((var,), 0.0)
            self._canonical_problem = normalize_binary_polynomial_problem(
                self._raw_problem
            )
        slack = set(self._slack_variables)
        self._decision_vars = tuple(
            v for v in self._canonical_problem.variable_order if v not in slack
        )
        self._hamiltonian_builder: Literal["native", "quadratized"] = (
            hamiltonian_builder
        )
        self._quadratization_strength = quadratization_strength
        self._ising_cache: IsingResult | None = None
        self._mixer_cache: SparsePauliOp | None = None

        # Decomposition support (optional)
        if local_search and decomposer is None:
            raise ValueError(
                "local_search requires a decomposer; it polishes the aggregated "
                "solution produced during partitioned solving."
            )
        self._decomposer = decomposer
        self._local_search = bool(local_search)
        self._polish_cache: tuple | None = None
        self._bqm: dimod.BinaryQuadraticModel | None
        if hybrid is not None:
            _, self._bqm = _sanitize_problem_input(self._raw_problem)
            self._partitioning = hybrid.Unwind(decomposer)
            self._aggregating = hybrid.Reduce(hybrid.Lambda(_merge_substates)) | (
                composer or hybrid.SplatComposer()
            )
        else:
            self._bqm = None

        self._variable_maps = {}
        self._bqm_subproblem_states = {}

    def _activity_bounds(
        self, coefficients: Mapping[Hashable, float]
    ) -> tuple[float, float]:
        """Minimum and maximum of ``Σ aᵢxᵢ`` over the assignments the problem allows.

        Sizes constraint slack. Defaults to every binary assignment.
        """
        values = list(coefficients.values())
        lo = math.fsum(v for v in values if v < 0)
        hi = math.fsum(v for v in values if v > 0)
        return lo, hi

    @property
    def hamiltonian_builder(self) -> Literal["native", "quadratized"]:
        """Ising-conversion strategy passed at construction."""
        return self._hamiltonian_builder

    @property
    def constraints(self) -> tuple[LinearConstraint, ...]:
        """The linear constraints passed at construction, keyed by the problem's variables."""
        return self._constraints

    def _assignment(self, bitstring: str) -> dict[Hashable, int]:
        """Values of every canonical variable (decision and slack) in ``bitstring``."""
        if len(bitstring) != self._ising.n_qubits:
            raise ValueError(
                f"Expected a bitstring of {self._ising.n_qubits} bits, one per "
                f"qubit including slack, got {len(bitstring)}."
            )
        values = self._ising.encoding.decode_fn(bitstring)
        return dict(zip(self._canonical_problem.variable_order, values.tolist()))

    def _complete_bitstring(self, decision: Mapping[Hashable, int]) -> str:
        """Bitstring for a decision assignment, with the slack that zeroes its penalty."""
        values: dict[Hashable, int] = {v: int(decision[v]) for v in self._decision_vars}
        for slack in self._slacks:
            values.update(slack.bits(values))
        idx = self._canonical_problem.variable_to_idx
        if self._ising.n_qubits != len(idx):
            raise NotImplementedError(
                "Completing a bitstring requires the native Hamiltonian builder."
            )
        bits = ["0"] * len(idx)
        for var, value in values.items():
            bits[idx[var]] = str(value)
        return "".join(bits)

    def _solution_key(self, bitstring: str) -> Hashable:
        """The decision-variable values, ignoring slack and quadratization ancillas."""
        assignment = self._assignment(bitstring)
        return tuple(assignment[v] for v in self._decision_vars)

    def is_feasible(self, bitstring: str) -> bool:
        """Whether ``bitstring`` satisfies every constraint, before rounding.

        Always ``True`` when the problem has no ``constraints``.
        """
        if not self._constraints:
            return True
        assignment = self._assignment(bitstring)
        return all(c.is_satisfied(assignment) for c in self._constraints)

    def compute_energy(self, bitstring: str) -> float:
        """Objective value of ``bitstring``, excluding penalties and constraints."""
        assignment = self._assignment(bitstring)
        return math.fsum(
            coeff * math.prod(assignment[v] for v in term)
            for term, coeff in self._objective_canonical_problem.terms.items()
        )

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
        Constraint slack variables are omitted in both cases.
        """
        base_decode = self._ising.encoding.decode_fn
        labels = self._decision_vars
        decode = base_decode
        if self._slack_variables:
            idx = self._canonical_problem.variable_to_idx
            positions = np.array([idx[v] for v in labels], dtype=int)

            def decode(bitstring: str) -> np.ndarray:
                return base_decode(bitstring)[positions]

        if labels == tuple(range(len(labels))):
            return decode
        return lambda bitstring: dict(zip(labels, decode(bitstring)))

    @property
    def metadata(self) -> dict[str, Any]:
        """Encoding metadata from the Ising conversion (e.g. quadratization aux info)."""
        return self._ising.encoding.metadata or {}

    @property
    def canonical_problem(self):
        """The normalised ``BinaryPolynomialProblem``."""
        return self._canonical_problem

    @property
    def objective_problem(self):
        """The objective/cost component passed as ``problem``."""
        return self._objective_problem

    @property
    def objective_canonical_problem(self):
        """The normalised objective/cost component."""
        return self._objective_canonical_problem

    @property
    def penalty_problem(self):
        """The penalty-only component, including encoded ``constraints``, if any."""
        return self._penalty_problem

    @property
    def penalty_canonical_problem(self):
        """The normalised penalty-only component, including encoded ``constraints``, if any."""
        return self._penalty_canonical_problem

    @property
    def penalty_weight(self) -> float:
        """Multiplier applied to ``penalty_problem`` in the full QUBO/HUBO."""
        return self._penalty_weight

    @property
    def raw_problem(self):
        """The QUBO/HUBO input used for QAOA.

        This is the original objective when no penalty was provided, otherwise
        the combined penalised objective.
        """
        return self._raw_problem

    def decompose(self) -> dict[Hashable, QAOAProblem]:
        """Partition the problem using the configured ``hybrid`` decomposer.

        Each non-trivial partition becomes its own
        :class:`BinaryOptimizationProblem` keyed by ``(name, size)``. Partitions
        with no interactions get no sub-problem; composition keeps the
        candidate's values for their variables.

        Raises:
            ValueError: If no decomposer was provided at construction.
        """
        if self._decomposer is None or self._bqm is None:
            raise ValueError(
                "Cannot decompose: no decomposer was provided at construction."
            )

        self._bqm_subproblem_states = {}
        self._variable_maps = {}

        init_state = _hybrid().State.from_problem(self._bqm)
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
                continue

            sub_problems[prog_id] = BinaryOptimizationProblem(
                _induced_sub_qubo(
                    partition.subproblem, list(partition.subproblem.variables)
                )
            )

        return sub_problems

    def _has_reproducible_decomposition(self) -> bool:
        """Whether repeated decomposition preserves program-slot meaning."""
        return bool(getattr(self._decomposer, "_reproducible", False))

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
        """Compose global candidate solutions into ``(solution, energy)`` pairs.

        With ``local_search=True`` each candidate is refined to a local minimum by
        greedy single-bit-flip descent against the full QUBO; otherwise the
        dwave-hybrid composer is run.

        Returns:
            Tuples ``(solution, energy)`` where ``solution`` is an ``int32`` ndarray
            of bits, sorted ascending by ``energy``.
        """
        if self._local_search:
            results = [self._greedy_bit_flip(sol) for _score, sol in candidates]
        else:
            results = [self._compose_solution(sol) for _score, sol in candidates]

        results.sort(key=lambda entry: entry[1])
        return results

    def _compose_solution(self, solution):
        """Run a single solution through the hybrid composer pipeline."""
        states_copy = {}
        for prog_id, bqm_subproblem_state in self._bqm_subproblem_states.items():
            variables = list(bqm_subproblem_state.subproblem.variables)
            global_indices = self._variable_maps[prog_id]
            var_to_val = {v: solution[gi] for v, gi in zip(variables, global_indices)}

            sample_set = dimod.SampleSet.from_samples(
                dimod.as_samples(var_to_val), "BINARY", 0
            )
            states_copy[prog_id] = bqm_subproblem_state.updated(subsamples=sample_set)

        states = _hybrid().States(*list(states_copy.values()))
        final_state = self._aggregating.run(states).result()

        sol, energy, _ = final_state.samples.record[0]
        return np.array(sol, dtype=np.int32), float(energy)

    def _polish_fields(self):
        """Cache ``(h, J, J_csc, tol)`` for the incremental single-bit-flip polish.

        ``tol`` is a PER-NODE improvement threshold, ``1e-12 * (|h_i| + sum_j
        |J[i,j]|)``. The incremental field's floating-point drift scales with each
        node's *local* field magnitude, so a per-node bound stays above the drift
        where couplings are large without desensitizing low-scale nodes elsewhere
        (a single global bound would let one huge outlier coupling reject genuine
        improvements across the whole problem).
        """
        if self._polish_cache is None:
            _variables, h, j = bqm_to_sparse(self._bqm)
            j_abs = j.copy()
            j_abs.data = np.abs(j_abs.data)
            node_scale = np.abs(h) + np.asarray(j_abs.sum(axis=1)).ravel()
            tol = 1e-12 * np.maximum(1.0, node_scale)
            self._polish_cache = (h, j, j.tocsc(), tol)
        return self._polish_cache

    def _greedy_bit_flip(self, solution) -> tuple[np.ndarray, float]:
        """Greedy single-bit-flip descent to a local minimum of the full BQM.

        Maintains an incremental effective field ``g = h + J @ x`` and updates only a
        flipped variable's neighbours, so cost scales with the number of couplings. The
        per-node improvement threshold (see :meth:`_polish_fields`) stays above the
        incremental field's floating-point drift at any coefficient scale.
        """
        h, j, jc, tol = self._polish_fields()
        x = np.array(solution, dtype=np.float64)
        g = h + j.dot(x)
        indptr, indices, data = jc.indptr, jc.indices, jc.data

        improved = True
        while improved:
            improved = False
            for i in range(len(x)):
                if (1 - 2 * x[i]) * g[i] < -tol[i]:
                    dx = 1 - 2 * x[i]
                    x[i] = 1 - x[i]
                    s, e = indptr[i], indptr[i + 1]
                    g[indices[s:e]] += data[s:e] * dx
                    improved = True

        bits = x.astype(np.int32)
        return bits, self.evaluate_global_solution(bits.tolist())
