# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Literal, get_args

import numpy as np
import pennylane as qml

from divi.circuits import MetaCircuit
from divi.hamiltonians import (
    ExactTrotterization,
    TrotterizationStrategy,
    _clean_hamiltonian,
    _get_terms_iterable,
    _is_empty_hamiltonian,
    _is_multi_term_sum,
)
from divi.pipeline import CircuitPipeline
from divi.pipeline.stages import MeasurementStage, TrotterSpecStage
from divi.qprog.quantum_program import QuantumProgram

_INITIAL_STATE_LITERAL = Literal["Zeros", "Superposition", "Ones"]


class TimeEvolution(QuantumProgram):
    """Quantum program for Hamiltonian time evolution.

    Simulates the evolution of a quantum state under a Hamiltonian using
    Trotter-Suzuki decomposition. Uses Divi's TrotterizationStrategy
    (ExactTrotterization, QDrift) for term selection and approximation.
    """

    def __init__(
        self,
        hamiltonian: qml.operation.Operator,
        trotterization_strategy: TrotterizationStrategy | None = None,
        time: float = 1.0,
        n_steps: int = 1,
        order: int = 1,
        initial_state: _INITIAL_STATE_LITERAL = "Zeros",
        observable: qml.operation.Operator | None = None,
        **kwargs,
    ):
        """Initialize TimeEvolution.

        Args:
            hamiltonian: Hamiltonian to evolve under.
            trotterization_strategy: Strategy for term selection (ExactTrotterization, QDrift).
                Defaults to ExactTrotterization().
            time: Evolution time t (e^(-iHt)).
            n_steps: Number of Trotter steps.
            order: Suzuki-Trotter order (1 or even).
            initial_state: One of ``"Zeros"`` (``|0...0>``), ``"Superposition"``
                (``|+...+>``), or ``"Ones"`` (``|1...1>``).
            observable: If None, measure qml.probs(); else qml.expval(observable).
            **kwargs: Passed to QuantumProgram (backend, seed, progress_queue, etc.).
        """
        super().__init__(**kwargs)

        if trotterization_strategy is None:
            trotterization_strategy = ExactTrotterization()

        hamiltonian_clean, _ = _clean_hamiltonian(hamiltonian)
        if _is_empty_hamiltonian(hamiltonian_clean):
            raise ValueError("Hamiltonian contains only constant terms.")

        if initial_state not in get_args(_INITIAL_STATE_LITERAL):
            raise ValueError(
                f"initial_state must be one of {get_args(_INITIAL_STATE_LITERAL)}, got {initial_state!r}"
            )

        self._hamiltonian = hamiltonian_clean
        self.trotterization_strategy = trotterization_strategy
        self.time = time
        self.n_steps = n_steps
        self.order = order
        self.initial_state = initial_state
        self.observable = observable
        self._circuit_wires = tuple(hamiltonian_clean.wires)
        self.n_qubits = len(self._circuit_wires)

        self.results: dict[str, Any] = {}

        self._build_pipelines()

    def _build_pipelines(self) -> None:
        trotter = TrotterSpecStage(
            trotterization_strategy=self.trotterization_strategy,
            meta_circuit_factory=self._meta_circuit_factory,
        )
        self._pipeline = CircuitPipeline(stages=[trotter, MeasurementStage()])

    def _meta_circuit_factory(
        self, hamiltonian: qml.operation.Operator, ham_id: int
    ) -> MetaCircuit:
        """Factory for TrotterSpecStage: build a MetaCircuit for one Hamiltonian sample."""
        ops = self._build_ops(hamiltonian)
        use_probs = self.observable is None

        if use_probs:
            measurement = qml.probs()
            return MetaCircuit(
                source_circuit=qml.tape.QuantumScript(
                    ops=ops, measurements=[measurement]
                ),
                symbols=np.array([], dtype=object),
            )

        measurement = qml.expval(self.observable)
        return MetaCircuit(
            source_circuit=qml.tape.QuantumScript(ops=ops, measurements=[measurement]),
            symbols=np.array([], dtype=object),
        )

    def run(self, **kwargs) -> tuple[int, float]:
        """Execute time evolution.

        Returns:
            tuple[int, float]: (total_circuit_count, total_run_time).
        """
        n_samples = getattr(
            self.trotterization_strategy, "n_hamiltonians_per_iteration", 1
        )

        if (
            n_samples > 1
            and self.observable is not None
            and self.backend.supports_expval
        ):
            raise ValueError(
                "Multi-sample QDrift with observable and expval backend is not supported. "
                "Use a shot-based backend or set observable=None for probs."
            )

        env = self._build_pipeline_env()

        result = self._pipeline.run(initial_spec=self._hamiltonian, env=env)
        self._total_circuit_count += env.artifacts.get("circuit_count", 0)
        self._total_run_time += env.artifacts.get("run_time", 0.0)
        self._current_execution_result = env.artifacts.get("_current_execution_result")

        if self.observable is None:
            self.results = {"probs": next(iter(result.values()))}
        else:
            self.results = {"expval": float(next(iter(result.values())))}

        return self.total_circuit_count, self.total_run_time

    def _build_ops(self, hamiltonian: qml.operation.Operator) -> list:
        """Build circuit ops: initial state, evolution, measurement."""
        ops = []

        # Initial state
        if self.initial_state == "Ones":
            for wire in self._circuit_wires:
                ops.append(qml.PauliX(wires=wire))
        elif self.initial_state == "Superposition":
            for wire in self._circuit_wires:
                ops.append(qml.Hadamard(wires=wire))

        # Evolution: e^(-iHt)
        n_terms = len(hamiltonian) if _is_multi_term_sum(hamiltonian) else 1
        if n_terms >= 2:
            evo = qml.adjoint(
                qml.TrotterProduct(
                    hamiltonian, time=self.time, n=self.n_steps, order=self.order
                )
            )
            ops.append(evo)
        else:
            term = (
                hamiltonian
                if not _is_multi_term_sum(hamiltonian)
                else _get_terms_iterable(hamiltonian)[0]
            )
            ops.append(qml.evolve(term, coeff=self.time))

        return ops
