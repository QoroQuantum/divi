# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import logging
from collections.abc import Mapping
from functools import cached_property
from typing import Any, Literal, Self

import numpy as np
import numpy.typing as npt
from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import SparsePauliOp

from divi.circuits import MetaCircuit
from divi.hamiltonians._term_ops import (
    ObservableLike,
    _spo_to_qiskit_basis_gates,
    _spo_wires,
    to_spo,
)
from divi.hamiltonians.trotterization import (
    ExactTrotterization,
    TrotterizationResult,
    TrotterizationStrategy,
    TrotterSpec,
)
from divi.pipeline import Stage
from divi.pipeline.stages import TrotterSpecStage
from divi.qprog import ParameterSpace, VariationalQuantumAlgorithm
from divi.qprog._program_checkpoint import _to_jsonable
from divi.qprog.initial_states import InitialState
from divi.qprog.mixins import SolutionEntry, SolutionSamplingMixin
from divi.qprog.problems import QAOAProblem
from divi.reporting._events import ProgressEvent

logger = logging.getLogger(__name__)


class QAOA(SolutionSamplingMixin, VariationalQuantumAlgorithm):
    """Quantum Approximate Optimisation Algorithm (QAOA) implementation.

    QAOA is a hybrid quantum-classical algorithm designed to solve combinatorial
    optimisation problems. It alternates between applying a cost Hamiltonian
    (encoding the problem) and a mixer Hamiltonian (enabling exploration).

    The problem is provided as a :class:`~divi.qprog.problems.QAOAProblem` instance that supplies the
    cost Hamiltonian, mixer Hamiltonian, initial state and decode function.

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
        """Initialise the QAOA algorithm.

        Args:
            problem: A :class:`~divi.qprog.problems.QAOAProblem` instance that provides cost/mixer
                Hamiltonians, decode function, and recommended initial state.
            initial_state: Override the problem's recommended initial state.
                If ``None``, uses ``problem.recommended_initial_state``.
            trotterization_strategy: Strategy for Hamiltonian evolution.
                Defaults to :class:`~divi.hamiltonians.trotterization.ExactTrotterization`.
            max_iterations: Maximum number of optimisation iterations.
                Defaults to 10.
            n_layers: Number of QAOA layers (circuit depth). Defaults to 1.
            **kwargs: Passed to :class:`~divi.qprog.variational_quantum_algorithm.VariationalQuantumAlgorithm`,
                including ``optimizer`` and ``backend``.
        """
        super().__init__(**kwargs)

        self.problem = problem
        cost_hamiltonian = to_spo(problem.cost_hamiltonian)
        self._decode_solution_fn = problem.decode_fn
        self.initial_state = initial_state
        self.problem_metadata = getattr(problem, "metadata", {})

        # Canonical wire mapping aligned with the cost SPO; problems may
        # surface domain-level labels (e.g. graph node names) via
        # ``wire_labels``, otherwise we fall back to dense qubit indices.
        self._circuit_wires = tuple(problem.wire_labels or _spo_wires(cost_hamiltonian))
        self.n_qubits = len(self._circuit_wires)

        self.cost_hamiltonian = cost_hamiltonian
        self._mixer_override: SparsePauliOp | None = None
        if not problem._applies_mixer_as_gates:
            self._check_width("mixer_hamiltonian", to_spo(problem.mixer_hamiltonian))

        # Algorithm parameters
        self.n_layers = n_layers
        self.max_iterations = max_iterations
        self.current_iteration = 0
        self.trotterization_strategy = trotterization_strategy

    @cached_property
    def _trotter_spec(self) -> TrotterSpec:
        """The spec the cost pipeline trotterizes, built once per cost Hamiltonian."""
        return TrotterSpec(self.cost_hamiltonian)

    def _drop_built_circuits(self) -> None:
        super()._drop_built_circuits()
        self.__dict__.pop("_trotter_spec", None)

    @property
    def mixer_hamiltonian(self) -> SparsePauliOp:
        """The mixer Hamiltonian, the problem's unless one was assigned.

        An assigned Hamiltonian is validated, coerced to ``SparsePauliOp`` and
        replaces the problem's mixer layer, gate-level mixers included.
        """
        if self._mixer_override is not None:
            return self._mixer_override
        return to_spo(self.problem.mixer_hamiltonian)

    @mixer_hamiltonian.setter
    def mixer_hamiltonian(self, hamiltonian: ObservableLike) -> None:
        self._mixer_override = self._check_width(
            "mixer_hamiltonian", to_spo(hamiltonian)
        )
        self._drop_built_circuits()

    @property
    def n_layers(self) -> int:
        """Number of QAOA layers; assigning it rebuilds the layer parameters."""
        return self._n_layers

    @n_layers.setter
    def n_layers(self, n_layers: int) -> None:
        self._n_layers = n_layers
        gammas = ParameterVector("γ", n_layers)
        betas = ParameterVector("β", n_layers)
        self._params = np.array([[g, b] for g, b in zip(gammas, betas)], dtype=object)
        self._drop_built_circuits()

    @property
    def initial_state(self) -> InitialState:
        """Initial state; ``None`` selects the problem's recommended one."""
        return self._initial_state

    @initial_state.setter
    def initial_state(self, state: InitialState | None) -> None:
        if state is None:
            state = self.problem.recommended_initial_state
        elif not isinstance(state, InitialState):
            raise TypeError(
                f"initial_state must be an InitialState instance or None, "
                f"got {type(state).__name__}"
            )
        self._initial_state = state
        self._drop_built_circuits()

    @property
    def trotterization_strategy(self) -> TrotterizationStrategy:
        """Trotterization strategy; ``None`` selects exact trotterization."""
        return self._trotterization_strategy

    @trotterization_strategy.setter
    def trotterization_strategy(self, strategy: TrotterizationStrategy | None) -> None:
        self._trotterization_strategy = (
            ExactTrotterization() if strategy is None else strategy
        )
        self._drop_built_circuits()

    @property
    def n_params_per_layer(self) -> int:
        return 2

    def _build_parameter_space(self) -> ParameterSpace:
        """The circuit's derived frequencies, with the problem's mixer
        declaration if any; cost angles are unknown under non-exact
        trotterization."""
        frequencies = list(super()._build_parameter_space().frequencies)
        if not isinstance(self.trotterization_strategy, ExactTrotterization):
            frequencies[0::2] = [None] * self.n_layers
        mixer = (
            self.problem.mixer_parameter_frequency()
            if self._mixer_override is None
            else None
        )
        if mixer is not None:
            frequencies[1::2] = [mixer] * self.n_layers
        return ParameterSpace(tuple(frequencies))

    @cached_property
    def _grad_shift_rule(
        self,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """The exact rule, refused under stochastic or approximate trotterization."""
        if not isinstance(self.trotterization_strategy, ExactTrotterization):
            raise NotImplementedError(
                "QAOA has no parameter-shift gradient for stochastic or "
                "approximate trotterization. Use a gradient-free optimizer or SPSA."
            )
        return super()._grad_shift_rule

    def _spec_stage(self) -> Stage:
        # QAOA trotterizes the cost Hamiltonian into the ansatz: seeded with
        # the Hamiltonian, not a pre-built MetaCircuit.
        return TrotterSpecStage(
            trotterization_strategy=self.trotterization_strategy,
            meta_circuit_factory=self._build_meta_circuit,
        )

    def _build_meta_circuit(
        self, result: TrotterizationResult, ham_id: int
    ) -> MetaCircuit:
        """The QAOA circuit measuring one trotterization result's effective
        Hamiltonian."""
        spo = result.effective_hamiltonian
        dag = circuit_to_dag(self._build_qaoa_qiskit_circuit(spo))
        return MetaCircuit(
            circuit_bodies=(((), dag),),
            parameters=tuple(self._params.flatten()),
            observable=spo,
            precision=self._precision,
        )

    def _initial_spec(self) -> TrotterSpec:
        return self._trotter_spec

    def _save_subclass_state(self) -> dict[str, Any]:
        """Save QAOA-specific runtime state."""
        state = {
            **super()._save_subclass_state(),
            "problem_metadata": _to_jsonable(self.problem_metadata),
        }
        if "solution_bitstring" in self._results:
            state["solution_bitstring"] = self._results["solution_bitstring"]
        return state

    def _load_subclass_state(self, state: Mapping[str, Any]) -> None:
        """Load QAOA-specific state.

        Raises:
            KeyError: If any required state key is missing (indicates checkpoint corruption).
        """
        super()._load_subclass_state(state)
        required_keys = ["problem_metadata"]
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

    @property
    def solution(self):
        """Get the solution found by QAOA optimisation.

        The return type depends on the Problem's decode function; ``None``
        is a legitimate decoded value after ``.run()``.

        Raises:
            RuntimeError: If no solution has been sampled yet.
        """
        if "decoded_solution" not in self._results:
            raise RuntimeError(
                f"QAOA.solution is not available yet. {self._SAMPLE_HINT}"
            )
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
                f"QAOA.solution_bitstring is not available yet. {self._SAMPLE_HINT}"
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
        mixer_qubits = list(range(n_qubits))

        qc = QuantumCircuit(n_qubits)
        qc.compose(self.initial_state.build(self._circuit_wires), inplace=True)

        for layer_params in self._params:
            gamma, beta = layer_params
            _spo_to_qiskit_basis_gates(qc, cost_spo, gamma, cost_qubits)
            if self._mixer_override is None:
                self.problem.apply_mixer(qc, beta, mixer_qubits)
            else:
                _spo_to_qiskit_basis_gates(qc, self._mixer_override, beta, mixer_qubits)

        return qc

    def _create_cost_circuit(self) -> MetaCircuit:
        """Generate the cost MetaCircuit for the QAOA problem.

        Executed circuits come from the cost pipeline's cached spec-stage cohort.
        This template supports callers that inspect ``cost_circuit`` without
        driving a pipeline.
        """
        return self._build_meta_circuit(
            self.trotterization_strategy.process_hamiltonian(self._trotter_spec),
            ham_id=0,
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
            List of :class:`~divi.qprog.mixins.SolutionEntry`.
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
