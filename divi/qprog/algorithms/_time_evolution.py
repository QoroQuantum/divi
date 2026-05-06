# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import numpy as np
import pennylane as qp
from qiskit.circuit import Parameter

from divi.circuits import MetaCircuit, qscript_to_meta
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

        hamiltonian_clean, _ = _clean_hamiltonian(hamiltonian)
        if _is_empty_hamiltonian(hamiltonian_clean):
            raise ValueError("Hamiltonian contains only constant terms.")

        self._hamiltonian = hamiltonian_clean
        self.trotterization_strategy = trotterization_strategy
        self.time = time
        self.n_steps = n_steps
        self.order = order
        # Normalise multi-observable input to a tuple so downstream
        # ``isinstance(..., tuple)`` checks fire regardless of whether the
        # caller passed a list or a tuple.  Single observables and ``None``
        # are passed through unchanged so the standard scalar code path is
        # bit-for-bit unchanged.
        if isinstance(observable, list):
            observable = tuple(observable)
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

    def expval(self) -> float:
        """Return expectation-value-mode results.

        Raises:
            RuntimeError: If ``.run()`` has not yet been called, or if this
                instance was constructed without an ``observable`` (probability
                mode). Use :meth:`probabilities` instead.
        """
        results = self.results
        if not isinstance(results, float):
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
        self, hamiltonian: qp.operation.Operator, ham_id: int
    ) -> MetaCircuit:
        """Factory for TrotterSpecStage: build a MetaCircuit for one Hamiltonian sample."""
        if self._template_meta is not None:
            return self._template_meta
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
            # Multi-observable: pipeline yields a per-observable list.
            self._results = [float(v) for v in raw]
        else:
            self._results = float(raw)

        self.reporter.info(
            message="Finished successfully!", final_status=TerminalStatus.SUCCESS
        )

        return self

    def _build_ops(self, hamiltonian: qp.operation.Operator) -> list:
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
