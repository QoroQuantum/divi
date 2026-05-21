# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import logging
import pickle
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import SparsePauliOp

from divi.circuits import MetaCircuit
from divi.hamiltonians import (
    ExactTrotterization,
    TrotterizationStrategy,
)
from divi.hamiltonians._term_ops import (
    _spo_to_qiskit_basis_gates,
    _spo_wires,
    to_spo,
)
from divi.pipeline.stages import TrotterSpecStage
from divi.qprog.algorithms import InitialState
from divi.qprog.problems import QAOAProblem
from divi.qprog.variational_quantum_algorithm import (
    SolutionEntry,
    VariationalQuantumAlgorithm,
)

logger = logging.getLogger(__name__)

# Sentinel distinguishing 'run() not yet called' from a decoded solution of ``None``.
_UNSET: Any = object()


class QAOA(VariationalQuantumAlgorithm):
    """Quantum Approximate Optimization Algorithm (QAOA) implementation.

    QAOA is a hybrid quantum-classical algorithm designed to solve combinatorial
    optimization problems. It alternates between applying a cost Hamiltonian
    (encoding the problem) and a mixer Hamiltonian (enabling exploration).

    The problem is provided as a :class:`~divi.qprog.problems.QAOAProblem` instance that supplies the
    cost Hamiltonian, mixer Hamiltonian, initial state, loss constant, and
    decode function.

    Args:
        problem: A :class:`~divi.qprog.problems.QAOAProblem` instance providing the QAOA ingredients.
        initial_state: Override the problem's recommended initial state.
        trotterization_strategy: The trotterization strategy. Defaults to ExactTrotterization.
        max_iterations: Maximum number of optimization iterations. Defaults to 10.
        n_layers: Number of QAOA layers. Defaults to 1.
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
        **kwargs,
    ):
        """Initialize the QAOA algorithm.

        Args:
            problem: A :class:`~divi.qprog.problems.QAOAProblem` instance that provides cost/mixer
                Hamiltonians, loss constant, decode function, and
                recommended initial state.
            initial_state: Override the problem's recommended initial state.
                If ``None``, uses ``problem.recommended_initial_state``.
            trotterization_strategy: Strategy for Hamiltonian evolution.
                Defaults to :class:`~divi.hamiltonians.ExactTrotterization`.
            max_iterations: Maximum number of optimization iterations.
                Defaults to 10.
            n_layers: Number of QAOA layers (circuit depth). Defaults to 1.
            **kwargs: Passed to :class:`~divi.qprog.variational_quantum_algorithm.VariationalQuantumAlgorithm`,
                including ``optimizer``, ``backend``, ``shots``, etc.
        """
        if initial_state is not None and not isinstance(initial_state, InitialState):
            raise TypeError(
                f"initial_state must be an InitialState instance or None, "
                f"got {type(initial_state).__name__}"
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

        # Canonical wire mapping aligned with the cost SPO; problems may
        # surface domain-level labels (e.g. graph node names) via
        # ``wire_labels``, otherwise we fall back to dense qubit indices.
        self._circuit_wires = tuple(
            getattr(problem, "wire_labels", None) or _spo_wires(self.cost_hamiltonian)
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
        self._decoded_solution: Any = _UNSET
        self._solution_bitstring: Any = _UNSET
        self._cost_meta_cache: dict[tuple[int, int], MetaCircuit] = {}

        # Circuit parameters — Qiskit ParameterVector, no sympy.
        betas = ParameterVector("β", self.n_layers)
        gammas = ParameterVector("γ", self.n_layers)
        self._params = np.array([[b, g] for b, g in zip(betas, gammas)], dtype=object)

        self._pipelines = self._build_pipelines()
        self._meta_circuit_factories: dict[str, MetaCircuit] | None = None

    @property
    def n_params_per_layer(self) -> int:
        return 2

    def _build_pipelines(self) -> dict:
        return {
            "cost": self._build_cost_pipeline(
                TrotterSpecStage(
                    trotterization_strategy=self.trotterization_strategy,
                    meta_circuit_factory=self._cost_meta_circuit_factory,
                )
            ),
            "measurement": self._build_measurement_pipeline(),
        }

    def _get_initial_spec(self, name: str) -> Any:
        # QAOA's cost pipeline is driven by a TrotterSpecStage, which expects
        # a Hamiltonian (not a MetaCircuit) as its initial spec.  Measurement
        # keeps the default (a pre-built MetaCircuit).
        if name == "cost":
            return self.cost_hamiltonian
        return super()._get_initial_spec(name)

    def _save_subclass_state(self) -> dict[str, Any]:
        """Save QAOA-specific runtime state."""
        decoded = None if self._decoded_solution is _UNSET else self._decoded_solution
        bitstring = (
            None if self._solution_bitstring is _UNSET else self._solution_bitstring
        )
        return {
            "problem_metadata": self.problem_metadata,
            "decoded_solution": decoded,
            "solution_bitstring": bitstring,
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
        loaded_bitstring = state.get("solution_bitstring")
        self._solution_bitstring = (
            _UNSET if loaded_bitstring is None else loaded_bitstring
        )
        self.loss_constant = state["loss_constant"]
        self.trotterization_strategy = pickle.loads(
            bytes.fromhex(state["trotterization_strategy"])
        )

    @property
    def solution(self):
        """Get the solution found by QAOA optimization.

        The return type depends on the Problem's decode function; ``None``
        is a legitimate decoded value after ``.run()``.

        Raises:
            RuntimeError: If ``.run()`` has not yet been called.
        """
        if self._decoded_solution is _UNSET:
            raise RuntimeError("QAOA.solution is not available. Call .run() first.")
        return self._decoded_solution

    @property
    def solution_bitstring(self) -> str:
        """Most-probable bitstring measured at the optimized parameters.

        Always a string of ``0``/``1`` characters of length ``n_qubits``,
        regardless of how the problem's decode function shapes :attr:`solution`.

        Raises:
            RuntimeError: If ``.run()`` has not yet been called.
        """
        if self._solution_bitstring is _UNSET:
            raise RuntimeError(
                "QAOA.solution_bitstring is not available. Call .run() first."
            )
        return self._solution_bitstring

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

    def _cost_meta_circuit_factory(
        self, processed_spo: SparsePauliOp, ham_id: int
    ) -> MetaCircuit:
        """Build a cost MetaCircuit for a given (possibly QDrift-sampled) SPO."""
        stateless = not self.trotterization_strategy.stateful
        # Cache key includes the parameter count so a depth change
        # (IterativeQAOA) self-invalidates without external bookkeeping.
        cache_key = (ham_id, self._params.size)
        if stateless and cache_key in self._cost_meta_cache:
            return self._cost_meta_cache[cache_key]

        qc = self._build_qaoa_qiskit_circuit(processed_spo)
        meta = MetaCircuit(
            circuit_bodies=(((), circuit_to_dag(qc)),),
            parameters=tuple(self._params.flatten()),
            observable=processed_spo,
            precision=self._precision,
        )
        if stateless:
            self._cost_meta_cache[cache_key] = meta
        return meta

    def _create_meta_circuit_factories(self) -> dict[str, MetaCircuit]:
        """Generate meta-circuit factories for the QAOA problem.

        The cost circuit is built via :meth:`_cost_meta_circuit_factory` so
        the cost and measurement pipelines share one
        ``_build_qaoa_qiskit_circuit`` pass through :attr:`_cost_meta_cache`.
        Stateful strategies (e.g. QDrift) bypass the cache and resample per
        call; the cost circuit built here serves only as the structural
        template for the measurement DAG.
        """
        flat_params = tuple(self._params.flatten())
        cost_circuit = self._cost_meta_circuit_factory(self.cost_hamiltonian, 0)
        meas_circuit = MetaCircuit(
            circuit_bodies=cost_circuit.circuit_bodies,
            parameters=flat_params,
            measured_wires=tuple(range(self.n_qubits)),
            precision=self._precision,
        )
        return {"cost_circuit": cost_circuit, "meas_circuit": meas_circuit}

    def compute_solution(
        self,
        params: npt.NDArray[np.float64] | None = None,
        **kwargs,
    ) -> "QAOA":
        """Run measurement circuits with the given parameters and decode the solution."""
        self.reporter.info(message="🏁 Computing Final Solution 🏁", overwrite=True)

        super().compute_solution(params, **kwargs)

        best_probs = next(iter(self._best_probs.values()))
        best_bitstring = max(best_probs, key=best_probs.get)
        self._solution_bitstring = best_bitstring
        self._decoded_solution = self._decode_solution_fn(best_bitstring)

        self.reporter.info(message="🏁 Computed Final Solution! 🏁")
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
            List of :class:`~divi.qprog.SolutionEntry`.
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
