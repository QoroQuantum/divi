# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import logging
import pickle
from typing import Any, Literal

import numpy as np
import pennylane as qml
import pennylane.qaoa as pqaoa
import sympy as sp

from divi.circuits import MetaCircuit
from divi.hamiltonians import (
    ExactTrotterization,
    TrotterizationStrategy,
)
from divi.pipeline.stages import TrotterSpecStage
from divi.qprog.algorithms._initial_state import InitialState
from divi.qprog.problems import QAOAProblem
from divi.qprog.variational_quantum_algorithm import (
    SolutionEntry,
    VariationalQuantumAlgorithm,
    _extract_param_set_idx,
)

logger = logging.getLogger(__name__)


class QAOA(VariationalQuantumAlgorithm):
    """Quantum Approximate Optimization Algorithm (QAOA) implementation.

    QAOA is a hybrid quantum-classical algorithm designed to solve combinatorial
    optimization problems. It alternates between applying a cost Hamiltonian
    (encoding the problem) and a mixer Hamiltonian (enabling exploration).

    The problem is provided as a :class:`QAOAProblem` instance that supplies the
    cost Hamiltonian, mixer Hamiltonian, initial state, loss constant, and
    decode function.

    Args:
        problem: A :class:`QAOAProblem` instance providing the QAOA ingredients.
        initial_state: Override the problem's recommended initial state.
        trotterization_strategy: The trotterization strategy. Defaults to ExactTrotterization.
        max_iterations: Maximum number of optimization iterations. Defaults to 10.
        n_layers: Number of QAOA layers. Defaults to 1.
        **kwargs: Additional keyword arguments passed to
            :class:`VariationalQuantumAlgorithm`, including ``optimizer``
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
        **kwargs,
    ):
        """Initialize the QAOA algorithm.

        Args:
            problem: A :class:`QAOAProblem` instance that provides cost/mixer
                Hamiltonians, loss constant, decode function, and
                recommended initial state.
            initial_state: Override the problem's recommended initial state.
                If ``None``, uses ``problem.recommended_initial_state``.
            trotterization_strategy: Strategy for Hamiltonian evolution.
                Defaults to :class:`ExactTrotterization`.
            max_iterations: Maximum number of optimization iterations.
                Defaults to 10.
            n_layers: Number of QAOA layers (circuit depth). Defaults to 1.
            **kwargs: Passed to :class:`VariationalQuantumAlgorithm`,
                including ``optimizer``, ``backend``, ``shots``, etc.
        """
        if initial_state is not None and not isinstance(initial_state, InitialState):
            raise TypeError(
                f"initial_state must be an InitialState instance or None, "
                f"got {type(initial_state).__name__}"
            )

        super().__init__(**kwargs)

        # Problem provides all domain-specific ingredients
        self.problem = problem
        self._cost_hamiltonian = problem.cost_hamiltonian
        self._decode_solution_fn = problem.decode_fn
        self.loss_constant = problem.loss_constant
        self.initial_state = initial_state or problem.recommended_initial_state
        self.problem_metadata = getattr(problem, "metadata", {})

        # Derived from cost Hamiltonian
        self.n_qubits = len(self._cost_hamiltonian.wires)
        self._circuit_wires = tuple(self._cost_hamiltonian.wires)

        # Algorithm parameters
        self.n_layers = n_layers
        self.max_iterations = max_iterations
        self.current_iteration = 0
        self.trotterization_strategy = trotterization_strategy or ExactTrotterization()
        self._n_params_per_layer = 2
        self._decoded_solution = None

        # Symbolic parameters for the ansatz
        betas = sp.symarray("β", self.n_layers)
        gammas = sp.symarray("γ", self.n_layers)
        self._sym_params = np.vstack((betas, gammas)).transpose()

        self._build_pipelines()

    def _build_pipelines(self) -> None:
        self._cost_pipeline = self._build_cost_pipeline(
            TrotterSpecStage(
                trotterization_strategy=self.trotterization_strategy,
                meta_circuit_factory=self._cost_meta_circuit_factory,
            )
        )
        self._measurement_pipeline = self._build_measurement_pipeline()

    def _save_subclass_state(self) -> dict[str, Any]:
        """Save QAOA-specific runtime state."""
        return {
            "problem_metadata": self.problem_metadata,
            "decoded_solution": self._decoded_solution,
            "loss_constant": self.loss_constant,
            "trotterization_strategy": pickle.dumps(
                self.trotterization_strategy, protocol=pickle.HIGHEST_PROTOCOL
            ).hex(),
        }

    def _load_subclass_state(self, state: dict[str, Any]) -> None:
        """Load QAOA-specific state.

        Raises:
            KeyError: If any required state key is missing (indicates checkpoint corruption).
        """
        required_keys = [
            "problem_metadata",
            "decoded_solution",
            "loss_constant",
        ]
        missing_keys = [key for key in required_keys if key not in state]
        if missing_keys:
            raise KeyError(
                f"Corrupted checkpoint: missing required state keys: {missing_keys}"
            )

        self.problem_metadata = state["problem_metadata"]
        self._decoded_solution = state["decoded_solution"]
        self.loss_constant = state["loss_constant"]
        self.trotterization_strategy = pickle.loads(
            bytes.fromhex(state["trotterization_strategy"])
        )

    @property
    def solution(self):
        """Get the solution found by QAOA optimization.

        The return type depends on the Problem's decode function.
        """
        return self._decoded_solution

    def _build_qaoa_ops(self, cost_hamiltonian: qml.operation.Operator) -> list:
        """Build QAOA layer ops for a given cost Hamiltonian."""
        ops = self.initial_state.build(self._circuit_wires)

        for layer_params in self._sym_params:
            gamma, beta = layer_params
            ops.append(pqaoa.cost_layer(gamma, cost_hamiltonian))
            ops.append(pqaoa.mixer_layer(beta, self.problem.mixer_hamiltonian))

        return ops

    def _cost_meta_circuit_factory(
        self, processed_ham: qml.operation.Operator, ham_id: int
    ) -> MetaCircuit:
        """Build a cost MetaCircuit for a given (possibly QDrift-sampled) Hamiltonian."""
        return MetaCircuit(
            source_circuit=qml.tape.QuantumScript(
                ops=self._build_qaoa_ops(processed_ham),
                measurements=[qml.expval(processed_ham)],
            ),
            symbols=self._sym_params.flatten(),
            precision=self._precision,
        )

    def _create_meta_circuit_factories(self) -> dict[str, MetaCircuit]:
        """Generate meta-circuit factories for the QAOA problem."""
        ops = self._build_qaoa_ops(self._cost_hamiltonian)

        return {
            "cost_circuit": MetaCircuit(
                source_circuit=qml.tape.QuantumScript(
                    ops=ops, measurements=[qml.expval(self._cost_hamiltonian)]
                ),
                symbols=self._sym_params.flatten(),
                precision=self._precision,
            ),
            "meas_circuit": MetaCircuit(
                source_circuit=qml.tape.QuantumScript(
                    ops=ops, measurements=[qml.probs()]
                ),
                symbols=self._sym_params.flatten(),
                precision=self._precision,
            ),
        }

    def _evaluate_cost_param_sets(
        self, param_sets: np.ndarray, **kwargs
    ) -> dict[int, float]:
        """Evaluate the cost pipeline for the provided parameter sets."""

        env = self._build_pipeline_env(param_sets=np.atleast_2d(param_sets))
        result = self._cost_pipeline.run(
            initial_spec=self._cost_hamiltonian,
            env=env,
        )
        self._total_circuit_count += env.artifacts.get("circuit_count", 0)
        self._total_run_time += env.artifacts.get("run_time", 0.0)
        self._current_execution_result = env.artifacts.get("_current_execution_result")

        return {
            _extract_param_set_idx(key): value + self.loss_constant
            for key, value in result.items()
        }

    def _perform_final_computation(self, **kwargs):
        """Run measurement circuits with the best parameters and decode the solution.

        Returns:
            tuple[int, float]: Total circuit count and total runtime.
        """
        self.reporter.info(message="🏁 Computing Final Solution 🏁", overwrite=True)

        self._run_solution_measurement_for(np.atleast_2d(self._best_params))
        best_probs = next(iter(self._best_probs.values()))
        best_bitstring = max(best_probs, key=best_probs.get)
        self._decoded_solution = self._decode_solution_fn(best_bitstring)

        self.reporter.info(message="🏁 Computed Final Solution! 🏁")
        return self._total_circuit_count, self._total_run_time

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

                - ``"ignore"`` (default): return all solutions, ranked by
                  probability.
                - ``"filter"``: drop infeasible solutions, rank by objective
                  energy.  This implements the **PHQC** (Polynomial-time
                  Hybrid Quantum-Classical) post-processing from
                  `arXiv:2511.14296 <https://arxiv.org/abs/2511.14296>`_
                  (Algorithm 4): every sampled bitstring is checked for
                  feasibility and scored by ``compute_energy`` (the true
                  objective, not the penalty Hamiltonian), then the
                  lowest-energy feasible solution is returned.
                - ``"repair"``: repair infeasible solutions via the Problem's
                  ``repair_infeasible_bitstring`` method, rank by energy.

        Returns:
            List of :class:`SolutionEntry`.
        """
        fetch_n = n if n > 0 else 2**self.n_qubits

        # No feasibility handling — just return by probability
        if feasibility == "ignore":
            return super().get_top_solutions(
                n=fetch_n, min_prob=min_prob, include_decoded=include_decoded
            )

        # Retrieve every measured bitstring so we can filter/repair
        n_measured = len(next(iter(self._best_probs.values())))
        all_solutions = super().get_top_solutions(
            n=n_measured, min_prob=min_prob, include_decoded=include_decoded
        )

        # Walk each solution: keep feasible ones, repair or skip infeasible
        p = self.problem
        result: list[SolutionEntry] = []
        for sol in all_solutions:
            bs = sol.bitstring

            if p.is_feasible(bs):
                energy = p.compute_energy(bs)
                decoded = self._decode_solution_fn(bs) if include_decoded else None
            elif feasibility == "repair":
                bs, repaired_decoded, energy = p.repair_infeasible_bitstring(bs)
                decoded = repaired_decoded if include_decoded else None
            else:  # "filter" — drop infeasible
                continue

            result.append(SolutionEntry(bs, sol.prob, decoded, energy))

        # Rank by energy (lower is better), break ties by higher probability
        result.sort(
            key=lambda s: (s.energy if s.energy is not None else float("inf"), -s.prob)
        )
        return result[:fetch_n]
