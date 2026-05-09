# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import numpy as np
import pennylane as qp
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp

from divi.circuits import MetaCircuit, qscript_to_meta, sparse_pauli_op_to_pl_observable
from divi.circuits.qem import _NoMitigation
from divi.hamiltonians import (
    ExactTrotterization,
    TrotterizationStrategy,
)
from divi.hamiltonians._term_ops import (
    _clean_hamiltonian_via_spo,
    _get_terms_iterable,
    _is_empty_hamiltonian,
    _is_multi_term_sum,
    _spo_wires,
)
from divi.pipeline import CircuitPipeline
from divi.pipeline.stages import (
    MeasurementStage,
    ParameterBindingStage,
    PauliTwirlStage,
    QEMStage,
    TrotterSpecStage,
)
from divi.qprog.algorithms import InitialState, ZerosState
from divi.qprog.quantum_program import QuantumProgram
from divi.reporting import TerminalStatus


class TimeEvolution(QuantumProgram):
    """Quantum program for Hamiltonian time evolution.

    Simulates the evolution of a quantum state under a Hamiltonian using
    Trotter-Suzuki decomposition. Uses Divi's TrotterizationStrategy
    (``ExactTrotterization``, ``QDrift``) for term selection and approximation.
    """

    def __init__(
        self,
        hamiltonian: qp.operation.Operator,
        trotterization_strategy: TrotterizationStrategy | None = None,
        time: float = 1.0,
        n_steps: int = 1,
        order: int = 1,
        initial_state: InitialState | None = None,
        observable: (
            qp.operation.Operator
            | tuple[qp.operation.Operator, ...]
            | list[qp.operation.Operator]
            | None
        ) = None,
        _template_meta: MetaCircuit | None = None,
        _template_param: Parameter | None = None,
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
            initial_state: Initial state preparation. Pass an :class:`~divi.qprog.algorithms.InitialState`
                instance (e.g. ``ZerosState()``, ``SuperpositionState()``).
                Defaults to ``ZerosState()`` if None.
            observable: One of:

                * ``None`` — measure ``qp.probs()``.
                * Single :class:`~pennylane.operation.Operator` — one
                  ``qp.expval(observable)`` measurement; ``self.results`` is
                  a ``float``.
                * ``Sequence[Operator]`` — multiple ``qp.expval(O_i)``
                  measurements from the same circuit; ``self.results`` is a
                  ``list[float]`` (one mitigated value per observable).
                  Commuting observables are measured from a shared shot
                  batch via
                  :class:`~divi.pipeline.stages.MeasurementStage`'s QWC
                  grouping; QuEPP shares the target circuit and dedupes
                  path DAGs across observables.
            **kwargs: Passed to QuantumProgram (backend, seed, progress_queue, etc.).
                Accepts ``qem_protocol`` for quantum error mitigation (requires
                ``observable`` to be set, since QEM operates on expectation values).
        """
        super().__init__(**kwargs)

        if trotterization_strategy is None:
            trotterization_strategy = ExactTrotterization()

        hamiltonian_clean, _ = _clean_hamiltonian_via_spo(hamiltonian)
        if _is_empty_hamiltonian(hamiltonian_clean):
            raise ValueError("Hamiltonian contains only constant terms.")

        self._hamiltonian = hamiltonian_clean
        self.trotterization_strategy = trotterization_strategy
        self.time = time
        self.n_steps = n_steps
        self.order = order
        if isinstance(observable, list):
            observable = tuple(observable)
        self.observable = observable
        self._circuit_wires = _spo_wires(hamiltonian_clean)
        self.n_qubits = len(self._circuit_wires)

        if initial_state is None:
            initial_state = ZerosState()
        if not isinstance(initial_state, InitialState):
            raise TypeError(
                f"initial_state must be an InitialState instance, got {type(initial_state).__name__}"
            )
        self.initial_state = initial_state

        if (_template_meta is None) != (_template_param is None):
            raise ValueError(
                "_template_meta and _template_param must be provided together."
            )
        self._template_meta = _template_meta
        self._template_param = _template_param

        self._results: dict[str, float] | float | list[float] | None = None

        self._pipelines = self._build_pipelines()

    def has_results(self) -> bool:
        return self._results is not None

    @property
    def results(self) -> dict[str, float] | float | list[float]:
        """Get the final results.

        Returns one of:

        * ``dict[str, float]`` — probability distribution when no
          ``observable`` was provided.
        * ``float`` — expectation value for a single ``observable``.
        * ``list[float]`` — per-observable expectation values when
          ``observable`` is a list/tuple.

        Raises:
            RuntimeError: If ``.run()`` has not yet been called.
        """
        if self._results is None:
            raise RuntimeError(
                "TimeEvolution.results is not available. Call .run() first."
            )
        return self._results

    def probabilities(self) -> dict[str, float]:
        """Return probability-mode results.

        Raises:
            RuntimeError: If ``.run()`` has not yet been called, or if this
                instance was constructed with an ``observable`` (expectation
                value mode). Use :meth:`expval` instead.
        """
        results = self.results
        if not isinstance(results, dict):
            raise RuntimeError(
                "TimeEvolution was run in expectation-value mode; use "
                ".expval() instead of .probabilities()."
            )
        return results

    def expval(self) -> float | list[float]:
        """Return expectation-value-mode results.

        Returns a ``float`` when ``observable`` was a single operator, or a
        ``list[float]`` (one entry per observable, in input order) when
        ``observable`` was a list/tuple.

        Raises:
            RuntimeError: If ``.run()`` has not yet been called, or if this
                instance was constructed without an ``observable`` (probability
                mode). Use :meth:`probabilities` instead.
        """
        results = self.results
        if isinstance(results, dict):
            raise RuntimeError(
                "TimeEvolution was run in probability mode; use "
                ".probabilities() instead of .expval()."
            )
        return results

    def _build_pipelines(self) -> dict:
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
        if self._template_meta is not None:
            # ParameterBindingStage binds the trajectory template's
            # ``t`` parameter to ``self.time`` via ``env.param_sets``.
            stages.append(ParameterBindingStage())
        return {"evolution": CircuitPipeline(stages=stages)}

    def _build_pipeline_env(self, **overrides):
        if self._template_meta is not None and "param_sets" not in overrides:
            overrides["param_sets"] = np.array([[float(self.time)]])
        return super()._build_pipeline_env(**overrides)

    @property
    def _pipeline(self) -> CircuitPipeline:
        """The evolution pipeline (thin accessor over ``self._pipelines``)."""
        return self._pipelines["evolution"]

    def _get_initial_spec(self, name: str) -> Any:
        if name == "evolution":
            return self._hamiltonian
        raise KeyError(f"No initial spec registered for pipeline {name!r}.")

    def _meta_circuit_factory(
        self, processed_spo: SparsePauliOp, ham_id: int
    ) -> MetaCircuit:
        """Factory for TrotterSpecStage: build a MetaCircuit for one Hamiltonian sample (SPO)."""
        if self._template_meta is not None:
            return self._template_meta
        # ``qp.TrotterProduct`` / ``qp.evolve`` need a PennyLane operator.
        hamiltonian = sparse_pauli_op_to_pl_observable(
            processed_spo, self._circuit_wires
        )
        ops = self._build_ops(hamiltonian)
        # Ensure canonical wire ordering matches the Hamiltonian,
        # regardless of which subset of terms QDrift sampled.
        ops = [qp.Identity(w) for w in self._circuit_wires] + ops

        if self.observable is None:
            measurements = [qp.probs()]
        elif isinstance(self.observable, tuple):
            measurements = [qp.expval(o) for o in self.observable]
        else:
            measurements = [qp.expval(self.observable)]
        return qscript_to_meta(
            qp.tape.QuantumScript(ops=ops, measurements=measurements),
            was_multi_obs=isinstance(self.observable, tuple),
        )

    def run(self, **kwargs) -> "TimeEvolution":
        """Execute time evolution.

        Returns:
            TimeEvolution: Returns ``self`` for method chaining.
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
        if self.observable is None:
            self._results = raw
        elif isinstance(self.observable, tuple):
            self._results = [float(v) for v in raw]
        else:
            (single,) = raw
            self._results = float(single)

        self.reporter.info(
            message="Finished successfully!", final_status=TerminalStatus.SUCCESS
        )

        return self

    def _build_ops(self, hamiltonian: qp.operation.Operator) -> list:
        """Build circuit ops: initial state, evolution, measurement."""
        ops = self.initial_state.build(self._circuit_wires)

        # Campbell's faithful QDrift: one evolution gate per sampled term,
        # repeated ``n_steps`` times with ``time/n_steps`` per step.
        sampled_spo = getattr(self.trotterization_strategy, "_last_sampled_spo", None)
        if sampled_spo is not None:
            # Skip simplify so sampling-with-replacement duplicates stay split.
            sampled_pl_unsimplified = sparse_pauli_op_to_pl_observable(
                sampled_spo, self._circuit_wires
            )
            sampled_terms = list(_get_terms_iterable(sampled_pl_unsimplified))
            step_time = self.time / self.n_steps
            for _ in range(self.n_steps):
                ops.extend(
                    qp.adjoint(qp.evolve(term, coeff=step_time))
                    for term in sampled_terms
                )
            return ops

        # Standard Trotter-Suzuki for ExactTrotterization
        n_terms = len(hamiltonian) if _is_multi_term_sum(hamiltonian) else 1
        if n_terms >= 2:
            evo = qp.adjoint(
                qp.TrotterProduct(
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
            ops.append(qp.evolve(term, coeff=self.time))

        return ops
