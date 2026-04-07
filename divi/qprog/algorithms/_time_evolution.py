# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pennylane as qml

from divi.circuits import MetaCircuit
from divi.circuits.qem import _NoMitigation
from divi.hamiltonians import (
    ExactTrotterization,
    TrotterizationStrategy,
    _clean_hamiltonian,
    _get_terms_iterable,
    _is_empty_hamiltonian,
    _is_multi_term_sum,
)
from divi.pipeline import CircuitPipeline
from divi.pipeline.stages import (
    MeasurementStage,
    PauliTwirlStage,
    QEMStage,
    TrotterSpecStage,
)
from divi.qprog.algorithms._initial_state import InitialState, ZerosState
from divi.qprog.quantum_program import QuantumProgram


class TimeEvolution(QuantumProgram):
    """Quantum program for Hamiltonian time evolution.

    Simulates the evolution of a quantum state under a Hamiltonian using
    Trotter-Suzuki decomposition. Uses Divi's TrotterizationStrategy
    (``ExactTrotterization``, ``QDrift``) for term selection and approximation.
    """

    def __init__(
        self,
        hamiltonian: qml.operation.Operator,
        trotterization_strategy: TrotterizationStrategy | None = None,
        time: float = 1.0,
        n_steps: int = 1,
        order: int = 1,
        initial_state: InitialState | None = None,
        observable: qml.operation.Operator | None = None,
        **kwargs,
    ):
        """Initialize TimeEvolution.

        Args:
            hamiltonian: Hamiltonian to evolve under.
            trotterization_strategy: Strategy for term selection (``ExactTrotterization``, ``QDrift``).
                Defaults to ExactTrotterization().
            time: Evolution time t (e^(-iHt)).
            n_steps: Number of Trotter steps.
            order: Suzuki-Trotter order (1 or even).
            initial_state: Initial state preparation. Pass an :class:`InitialState`
                instance (e.g. ``ZerosState()``, ``SuperpositionState()``).
                Defaults to ``ZerosState()`` if None.
            observable: If None, measure ``qml.probs()``; else ``qml.expval(observable)``.
            **kwargs: Passed to QuantumProgram (backend, seed, progress_queue, etc.).
                Accepts ``qem_protocol`` for quantum error mitigation (requires
                ``observable`` to be set, since QEM operates on expectation values).
        """
        super().__init__(**kwargs)

        if trotterization_strategy is None:
            trotterization_strategy = ExactTrotterization()

        hamiltonian_clean, _ = _clean_hamiltonian(hamiltonian)
        if _is_empty_hamiltonian(hamiltonian_clean):
            raise ValueError("Hamiltonian contains only constant terms.")

        self._hamiltonian = hamiltonian_clean
        self.trotterization_strategy = trotterization_strategy
        self.time = time
        self.n_steps = n_steps
        self.order = order
        self.observable = observable
        self._circuit_wires = tuple(hamiltonian_clean.wires)
        self.n_qubits = len(self._circuit_wires)

        if initial_state is None:
            initial_state = ZerosState()
        if not isinstance(initial_state, InitialState):
            raise TypeError(
                f"initial_state must be an InitialState instance, got {type(initial_state).__name__}"
            )
        self.initial_state = initial_state

        self.results: dict[str, float] | float | None = None

        self._build_pipelines()

    def _build_pipelines(self) -> None:
        trotter = TrotterSpecStage(
            trotterization_strategy=self.trotterization_strategy,
            meta_circuit_factory=self._meta_circuit_factory,
        )
        stages: list = [trotter]

        if not isinstance(self._qem_protocol, _NoMitigation):
            stages.append(QEMStage(protocol=self._qem_protocol))
            n_twirls = getattr(self._qem_protocol, "n_twirls", 0)
            if n_twirls > 0:
                stages.append(PauliTwirlStage(n_twirls=n_twirls))

        stages.append(MeasurementStage())
        self._pipeline = CircuitPipeline(stages=stages)

    def _get_dry_run_pipelines(self) -> dict[str, tuple]:
        return {"evolution": (self._pipeline, self._hamiltonian)}

    def _meta_circuit_factory(
        self, hamiltonian: qml.operation.Operator, ham_id: int
    ) -> MetaCircuit:
        """Factory for TrotterSpecStage: build a MetaCircuit for one Hamiltonian sample."""
        ops = self._build_ops(hamiltonian)
        # Ensure canonical wire ordering matches the Hamiltonian,
        # regardless of which subset of terms QDrift sampled.
        ops = [qml.Identity(w) for w in self._circuit_wires] + ops
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
        env = self._build_pipeline_env()

        result = self._pipeline.run(initial_spec=self._hamiltonian, env=env)
        self._total_circuit_count += env.artifacts.get("circuit_count", 0)
        self._total_run_time += env.artifacts.get("run_time", 0.0)
        self._current_execution_result = env.artifacts.get("_current_execution_result")

        if len(result) != 1:
            raise RuntimeError(
                f"Expected exactly 1 pipeline result, got {len(result)}."
            )
        (raw,) = result.values()
        self.results = raw if self.observable is None else float(raw)

        self.reporter.info(message="Finished successfully!")

        return self.total_circuit_count, self.total_run_time

    def _build_ops(self, hamiltonian: qml.operation.Operator) -> list:
        """Build circuit ops: initial state, evolution, measurement."""
        ops = self.initial_state.build(self._circuit_wires)

        # Campbell's faithful QDrift: individual evolution gates per sampled term.
        # This avoids Trotter error from feeding a resampled Hamiltonian to
        # TrotterProduct, which is incorrect for non-commuting Hamiltonians.
        # Each Trotter step repeats the sampled terms with time/n_steps, giving
        # sampling_budget * n_steps total gates (matching old circuit depth).
        sampled_terms = getattr(
            self.trotterization_strategy, "_last_sampled_terms", None
        )
        if sampled_terms is not None:
            step_time = self.time / self.n_steps
            for _ in range(self.n_steps):
                ops.extend(
                    qml.adjoint(qml.evolve(term, coeff=step_time))
                    for term in sampled_terms
                )
            return ops

        # Standard Trotter-Suzuki for ExactTrotterization
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
