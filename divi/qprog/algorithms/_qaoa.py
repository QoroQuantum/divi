# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import logging
from fractions import Fraction
from math import gcd, lcm
from typing import Any, Literal, Self

import numpy as np
import numpy.typing as npt
from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import SparsePauliOp

from divi.circuits import MetaCircuit
from divi.hamiltonians import (
    ExactTrotterization,
    TrotterizationResult,
    TrotterizationStrategy,
)
from divi.hamiltonians._term_ops import (
    _spo_to_qiskit_basis_gates,
    _spo_wires,
    to_spo,
)
from divi.pipeline import Stage
from divi.pipeline.stages import TrotterSpecStage
from divi.qprog._program_checkpoint import _to_jsonable
from divi.qprog.algorithms import InitialState
from divi.qprog.mixins import SolutionEntry, SolutionSamplingMixin
from divi.qprog.mixins._solution_sampling import _SAMPLE_HINT
from divi.qprog.problems import QAOAProblem
from divi.qprog.variational_quantum_algorithm import VariationalQuantumAlgorithm
from divi.reporting._events import ProgressEvent

logger = logging.getLogger(__name__)

_MAX_PARAMETER_SHIFT_EVALUATIONS = 256


def _hamiltonian_parameter_frequency(
    hamiltonian: SparsePauliOp,
) -> tuple[float, int]:
    """Return a harmonic frequency superset for one shared evolution angle."""
    active_terms = np.any(hamiltonian.paulis.x | hamiltonian.paulis.z, axis=1)
    frequencies = 2 * np.abs(hamiltonian.coeffs.real[active_terms])
    frequencies = frequencies[frequencies > 1e-12]
    if frequencies.size == 0:
        return 1.0, 1

    rational = [
        Fraction(float(value)).limit_denominator(10_000) for value in frequencies
    ]
    if any(
        not np.isclose(float(value), frequency, rtol=1e-10, atol=1e-12)
        for value, frequency in zip(rational, frequencies)
    ):
        raise NotImplementedError(
            "QAOA parameter-shift gradients require commensurate Hamiltonian "
            "coefficients. Use a gradient-free optimizer or SPSA for this problem."
        )

    common_denominator = lcm(*(value.denominator for value in rational))
    integer_frequencies = [
        value.numerator * (common_denominator // value.denominator)
        for value in rational
    ]
    divisor = gcd(*integer_frequencies)
    omega = divisor / common_denominator
    order = sum(value // divisor for value in integer_frequencies)
    return float(omega), order


class QAOA(SolutionSamplingMixin, VariationalQuantumAlgorithm):
    """Quantum Approximate Optimisation Algorithm (QAOA) implementation.

    QAOA is a hybrid quantum-classical algorithm designed to solve combinatorial
    optimisation problems. It alternates between applying a cost Hamiltonian
    (encoding the problem) and a mixer Hamiltonian (enabling exploration).

    The problem is provided as a :class:`~divi.qprog.problems.QAOAProblem` instance that supplies the
    cost Hamiltonian, mixer Hamiltonian, initial state, loss constant, and
    decode function.

    **Warm-starting via** ``run(initial_params=...)``: each parameter set is a
    flat row of length ``2 * n_layers``, ordered **per layer, cost angle then
    mixer angle**: ``[γ_0, β_0, γ_1, β_1, ..., γ_{p-1}, β_{p-1}]`` (``γ`` drives
    the cost layer, ``β`` the mixer). Shape is ``(n_param_sets, 2 * n_layers)``;
    a single set may be passed as a 1-D array of length ``2 * n_layers``. To
    warm-start from a characterisation result's ``ar_vs_depth[p]`` entry, zip its
    ``gammas`` and ``betas`` in this interleaved order — e.g. for ``p`` layers::

        pt = result.ar_vs_depth[p - 1]
        initial_params = [x for g, b in zip(pt["gammas"], pt["betas"]) for x in (g, b)]

    Only the shape is validated; an interleave in the wrong order starts the
    optimizer from silently wrong angles, so match this layout exactly.

    Args:
        problem: A :class:`~divi.qprog.problems.QAOAProblem` instance providing the QAOA ingredients.
        initial_state: Override the problem's recommended initial state.
        trotterization_strategy: The trotterization strategy. Defaults to ExactTrotterization.
        max_iterations: Maximum number of optimisation iterations. Defaults to 10.
        n_layers: Number of QAOA layers. Defaults to 1.
        max_shift_evaluations_per_parameter: Safety limit for generalised
            parameter-shift evaluations per parameter. Set to ``None`` to opt out.
        **kwargs: Additional keyword arguments passed to
            :class:`~divi.qprog.variational_quantum_algorithm.VariationalQuantumAlgorithm`, including ``optimizer``
            and ``backend``.
    """

    def __init__(
        self,
        problem: QAOAProblem,
        *,
        initial_state: InitialState | None = None,
        trotterization_strategy: TrotterizationStrategy | None = None,
        max_iterations: int = 10,
        n_layers: int = 1,
        max_shift_evaluations_per_parameter: int | None = (
            _MAX_PARAMETER_SHIFT_EVALUATIONS
        ),
        **kwargs,
    ):
        """Initialise the QAOA algorithm.

        Args:
            problem: A :class:`~divi.qprog.problems.QAOAProblem` instance that provides cost/mixer
                Hamiltonians, loss constant, decode function, and
                recommended initial state.
            initial_state: Override the problem's recommended initial state.
                If ``None``, uses ``problem.recommended_initial_state``.
            trotterization_strategy: Strategy for Hamiltonian evolution.
                Defaults to :class:`~divi.hamiltonians.ExactTrotterization`.
            max_iterations: Maximum number of optimisation iterations.
                Defaults to 10.
            n_layers: Number of QAOA layers (circuit depth). Defaults to 1.
            max_shift_evaluations_per_parameter: Safety limit for generalised
                parameter-shift evaluations per parameter. Set to ``None`` to
                permit arbitrarily large rules.
            **kwargs: Passed to :class:`~divi.qprog.variational_quantum_algorithm.VariationalQuantumAlgorithm`,
                including ``optimizer`` and ``backend``.
        """
        if initial_state is not None and not isinstance(initial_state, InitialState):
            raise TypeError(
                f"initial_state must be an InitialState instance or None, "
                f"got {type(initial_state).__name__}"
            )
        if max_shift_evaluations_per_parameter is not None and (
            isinstance(max_shift_evaluations_per_parameter, bool)
            or not isinstance(max_shift_evaluations_per_parameter, int)
            or max_shift_evaluations_per_parameter < 1
        ):
            raise ValueError(
                "max_shift_evaluations_per_parameter must be a positive integer "
                "or None."
            )

        super().__init__(**kwargs)

        # Coerce both Hamiltonians to ``SparsePauliOp`` at the input boundary.
        self.problem = problem
        self.cost_hamiltonian: SparsePauliOp = to_spo(problem.cost_hamiltonian)
        self.mixer_hamiltonian: SparsePauliOp = to_spo(problem.mixer_hamiltonian)
        self._decode_solution_fn = problem.decode_fn
        self.loss_constant = problem.loss_constant
        self.initial_state = initial_state or problem.recommended_initial_state
        self.problem_metadata = getattr(problem, "metadata", {})
        self.max_shift_evaluations_per_parameter = max_shift_evaluations_per_parameter

        # Canonical wire mapping aligned with the cost SPO; problems may
        # surface domain-level labels (e.g. graph node names) via
        # ``wire_labels``, otherwise we fall back to dense qubit indices.
        self._circuit_wires = tuple(
            problem.wire_labels or _spo_wires(self.cost_hamiltonian)
        )
        self.n_qubits = len(self._circuit_wires)

        cost_n = self.cost_hamiltonian.num_qubits
        mixer_n = self.mixer_hamiltonian.num_qubits
        if cost_n != self.n_qubits or mixer_n != self.n_qubits:
            raise ValueError(
                f"wire_labels has {self.n_qubits} entries but "
                f"cost_hamiltonian.num_qubits is {cost_n} and "
                f"mixer_hamiltonian.num_qubits is {mixer_n}. Each must "
                f"equal len(wire_labels)."
            )

        # Algorithm parameters
        self.n_layers = n_layers
        self.max_iterations = max_iterations
        self.current_iteration = 0
        self.trotterization_strategy = trotterization_strategy or ExactTrotterization()
        # Circuit parameters — Qiskit ParameterVector, no sympy.
        betas = ParameterVector("β", self.n_layers)
        gammas = ParameterVector("γ", self.n_layers)
        self._params = np.array([[b, g] for b, g in zip(betas, gammas)], dtype=object)

    @property
    def n_params_per_layer(self) -> int:
        return 2

    def _parameter_frequencies(self):
        """Hamiltonian-derived frequency families for each shared layer angle."""
        if not isinstance(self.trotterization_strategy, ExactTrotterization):
            raise NotImplementedError(
                "QAOA has no parameter-shift gradient for stochastic or "
                "approximate trotterization. Use a gradient-free optimizer or SPSA."
            )
        per_layer = [
            _hamiltonian_parameter_frequency(self.cost_hamiltonian),
            _hamiltonian_parameter_frequency(self.mixer_hamiltonian),
        ]
        evaluation_counts = [2 * order for _frequency, order in per_layer]
        limit = self.max_shift_evaluations_per_parameter
        if limit is not None and any(count > limit for count in evaluation_counts):
            total_evaluations = self.n_layers * sum(evaluation_counts)
            raise NotImplementedError(
                "QAOA parameter-shift gradients require "
                f"{evaluation_counts[0]} cost and {evaluation_counts[1]} mixer "
                "circuit evaluations per parameter; the full "
                f"{self.n_params}-parameter gradient requires "
                f"{total_evaluations} evaluations, and the per-parameter limit "
                f"is {limit}. Increase max_shift_evaluations_per_parameter, "
                "set it to None to opt out, or use a gradient-free optimizer "
                "or SPSA for this problem."
            )
        return per_layer * self.n_layers

    def _spec_stage(self) -> Stage:
        # QAOA trotterizes the cost Hamiltonian into the ansatz: seeded with
        # the Hamiltonian, not a pre-built MetaCircuit.
        def _make_meta(result: TrotterizationResult, _ham_id: int) -> MetaCircuit:
            spo = result.effective_hamiltonian
            dag = circuit_to_dag(self._build_qaoa_qiskit_circuit(spo))
            return MetaCircuit(
                circuit_bodies=(((), dag),),
                parameters=tuple(self._params.flatten()),
                observable=spo,
                precision=self._precision,
            )

        return TrotterSpecStage(
            trotterization_strategy=self.trotterization_strategy,
            meta_circuit_factory=_make_meta,
        )

    def _initial_spec(self) -> SparsePauliOp:
        return self.cost_hamiltonian

    def _save_subclass_state(self) -> dict[str, Any]:
        """Save QAOA-specific runtime state."""
        state = {
            **super()._save_subclass_state(),
            "problem_metadata": _to_jsonable(self.problem_metadata),
            "loss_constant": self.loss_constant,
            "max_shift_evaluations_per_parameter": (
                self.max_shift_evaluations_per_parameter
            ),
        }
        if "solution_bitstring" in self._results:
            state["solution_bitstring"] = self._results["solution_bitstring"]
        return state

    def _load_subclass_state(self, state: dict[str, Any]) -> None:
        """Load QAOA-specific state.

        Raises:
            KeyError: If any required state key is missing (indicates checkpoint corruption).
        """
        super()._load_subclass_state(state)
        required_keys = ["problem_metadata", "loss_constant"]
        missing_keys = [key for key in required_keys if key not in state]
        if missing_keys:
            raise KeyError(
                f"Corrupted checkpoint: missing required state keys: {missing_keys}"
            )

        self.problem_metadata = state["problem_metadata"]
        if "solution_bitstring" in state:
            bitstring = state["solution_bitstring"]
            self._results["solution_bitstring"] = bitstring
            # Decoded by this program's problem, whose labels may differ.
            self._results["decoded_solution"] = self._decode_solution_fn(bitstring)
        self.loss_constant = state["loss_constant"]
        self.max_shift_evaluations_per_parameter = state.get(
            "max_shift_evaluations_per_parameter",
            self.max_shift_evaluations_per_parameter,
        )

    @property
    def solution(self):
        """Get the solution found by QAOA optimisation.

        The return type depends on the Problem's decode function; ``None``
        is a legitimate decoded value after ``.run()``.

        Raises:
            RuntimeError: If no solution has been sampled yet.
        """
        if "decoded_solution" not in self._results:
            raise RuntimeError(f"QAOA.solution is not available yet. {_SAMPLE_HINT}")
        return self._results["decoded_solution"]

    @property
    def solution_bitstring(self) -> str:
        """Most-probable bitstring measured at the optimised parameters.

        Always a string of ``0``/``1`` characters of length ``n_qubits``,
        regardless of how the problem's decode function shapes :attr:`solution`.

        Raises:
            RuntimeError: If no solution has been sampled yet.
        """
        if "solution_bitstring" not in self._results:
            raise RuntimeError(
                f"QAOA.solution_bitstring is not available yet. {_SAMPLE_HINT}"
            )
        return self._results["solution_bitstring"]

    def _build_qaoa_qiskit_circuit(self, cost_spo: SparsePauliOp) -> QuantumCircuit:
        """Build the QAOA ansatz directly as a qiskit ``QuantumCircuit``.

        Wire labels (which may be graph node strings) are flattened to
        ``range(n_qubits)`` indices via ``_circuit_wires``' positional
        mapping — qubit ``i`` ↔ ``self._circuit_wires[i]``.
        """
        n_qubits = self.n_qubits
        cost_qubits = list(range(n_qubits))
        # The mixer SPO is built over the same dense qubit space as the cost
        # SPO; use straight 0..n-1 indexing.
        mixer_qubits = list(range(n_qubits))

        qc = QuantumCircuit(n_qubits)
        qc.compose(self.initial_state.build(self._circuit_wires), inplace=True)

        for layer_params in self._params:
            gamma, beta = layer_params
            _spo_to_qiskit_basis_gates(qc, cost_spo, gamma, cost_qubits)
            _spo_to_qiskit_basis_gates(qc, self.mixer_hamiltonian, beta, mixer_qubits)

        return qc

    def _create_cost_circuit(self) -> MetaCircuit:
        """Generate the cost MetaCircuit for the QAOA problem.

        Executed circuits come from the cost pipeline's cached spec-stage cohort.
        This template supports callers that inspect ``cost_circuit`` without
        driving a pipeline.
        """
        result = self.trotterization_strategy.process_hamiltonian(self.cost_hamiltonian)
        spo = result.effective_hamiltonian
        dag = circuit_to_dag(self._build_qaoa_qiskit_circuit(spo))
        return MetaCircuit(
            circuit_bodies=(((), dag),),
            parameters=tuple(self._params.flatten()),
            observable=spo,
            precision=self._precision,
        )

    def sample_solution(
        self,
        params: npt.NDArray[np.float64] | None = None,
        **kwargs,
    ) -> Self:
        """Run measurement circuits with the given parameters and decode the solution."""
        with self._ensure_progress_session(label="QAOA solution sampling", total=None):
            self._progress_emitter(
                ProgressEvent.show(self._progress_key, "🏁 Computing Final Solution 🏁")
            )

            super().sample_solution(self._resolve_sample_params(params), **kwargs)

            best_probs = next(iter(self._results["best_probs"].values()))
            best_bitstring = max(best_probs, key=best_probs.__getitem__)
            self._results["solution_bitstring"] = best_bitstring
            self._results["decoded_solution"] = self._decode_solution_fn(best_bitstring)

            self._progress_emitter(
                ProgressEvent.show(self._progress_key, "🏁 Computed Final Solution! 🏁")
            )
            return self

    def get_top_solutions(
        self,
        n: int = 10,
        *,
        min_prob: float = 0.0,
        include_decoded: bool = False,
        feasibility: Literal["ignore", "filter", "repair"] = "ignore",
    ) -> list[SolutionEntry]:
        """Get top-N solutions with optional feasibility filtering and repair.

        Args:
            n: Number of top solutions to return (0 = all). Defaults to 10.
            min_prob: Minimum probability threshold. Defaults to 0.0.
            include_decoded: Include decoded representations. Defaults to False.
            feasibility: How to handle infeasible solutions:

                - ``"ignore"`` (default): return every measured bitstring,
                  slack qubits included, ranked by probability, with
                  ``energy`` left as ``None``.
                - ``"filter"``: drop infeasible solutions, rank by objective
                  energy.  This implements the **PHQC** (Polynomial-time
                  Hybrid Quantum-Classical) post-processing from
                  `arXiv:2511.14296 <https://arxiv.org/abs/2511.14296>`_
                  (Algorithm 4): every sampled bitstring is checked for
                  feasibility and scored by ``compute_energy`` (the true
                  objective, not the penalty Hamiltonian), and the feasible
                  ones are returned from lowest energy up.
                - ``"repair"``: repair infeasible solutions via the Problem's
                  ``repair_infeasible_bitstring`` method, rank by energy.
                  Repairs that stay infeasible are dropped.

                In ``"filter"`` and ``"repair"``, samples of the same solution
                (ignoring slack) are merged into one entry with their
                probabilities summed, and ``energy`` is the objective without
                penalties.

        Returns:
            List of :class:`~divi.qprog.SolutionEntry`.
        """
        if n < 0:
            raise ValueError(f"n must be non-negative, got {n}")
        if feasibility not in ("ignore", "filter", "repair"):
            raise ValueError(
                "feasibility must be 'ignore', 'filter' or 'repair', "
                f"got '{feasibility}'"
            )
        fetch_n = n or 2**self.n_qubits

        # No feasibility handling — just return by probability
        if feasibility == "ignore":
            return super().get_top_solutions(
                n=fetch_n, min_prob=min_prob, include_decoded=include_decoded
            )

        # Retrieve every measured bitstring so we can filter/repair
        n_measured = len(self._single_distribution())
        all_solutions = super().get_top_solutions(
            n=n_measured, min_prob=min_prob, include_decoded=include_decoded
        )

        result = self.problem._rank_feasible(
            ((sol.bitstring, sol.prob) for sol in all_solutions),
            feasibility,
            self._decode_solution_fn if include_decoded else None,
        )
        return result[:fetch_n]
