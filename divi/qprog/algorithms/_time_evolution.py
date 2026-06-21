# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from typing import Any

import numpy as np
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import SparsePauliOp

from divi.circuits import MetaCircuit
from divi.circuits._conversions import _QISKIT_TO_QASM2
from divi.hamiltonians import (
    ExactTrotterization,
    TrotterizationResult,
    TrotterizationStrategy,
)
from divi.hamiltonians._term_ops import (
    _clean_hamiltonian_spo,
    _require_qiskit_num_qubits,
    to_spo,
)
from divi.pipeline import CircuitPipeline, CircuitPreprocessor, ResultFormat, Stage
from divi.pipeline.stages import ParameterBindingStage, TrotterSpecStage
from divi.qprog import ObservableMeasuringMixin
from divi.qprog.algorithms import InitialState, ZerosState
from divi.qprog.quantum_program import QuantumProgram
from divi.reporting import TerminalStatus


class TimeEvolution(ObservableMeasuringMixin, QuantumProgram):
    """Quantum program for Hamiltonian time evolution.

    Simulates the evolution of a quantum state under a Hamiltonian using
    Trotter-Suzuki decomposition. Uses Divi's TrotterizationStrategy
    (``ExactTrotterization``, ``QDrift``) for term selection and approximation.
    """

    def __init__(
        self,
        hamiltonian: SparsePauliOp,
        trotterization_strategy: TrotterizationStrategy | None = None,
        time: float = 1.0,
        n_steps: int = 1,
        order: int = 1,
        initial_state: InitialState | None = None,
        observable: SparsePauliOp | Sequence[SparsePauliOp] | None = None,
        _template_meta: MetaCircuit | None = None,
        _template_param: Parameter | None = None,
        **kwargs,
    ):
        """Initialize TimeEvolution.

        Args:
            hamiltonian: Hamiltonian to evolve under. Accepts anything
                :func:`~divi.hamiltonians.to_spo` consumes (``SparsePauliOp``,
                PennyLane operator, or a divi-convention Pauli-string dict).
            trotterization_strategy: Strategy for term selection (``ExactTrotterization``, ``QDrift``).
                Defaults to ExactTrotterization().
            time: Evolution time t (e^(-iHt)).
            n_steps: Number of Trotter steps.
            order: Suzuki-Trotter order (1 or even).
            initial_state: Initial state preparation. Pass an :class:`~divi.qprog.algorithms.InitialState`
                instance (e.g. ``ZerosState()``, ``SuperpositionState()``).
                Defaults to ``ZerosState()`` if None.
            observable: One of:

                * ``None`` — measure computational-basis probabilities over
                  all qubits.
                * Single observable accepted by
                  :func:`~divi.hamiltonians.to_spo` — one expectation-value
                  measurement; ``self.results`` is a ``float``.
                * Sequence of such observables — multiple expectation-value
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

        if not isinstance(n_steps, int) or n_steps < 1:
            raise ValueError(f"n_steps must be a positive integer, got {n_steps!r}.")
        if order != 1 and (not isinstance(order, int) or order < 2 or order % 2 != 0):
            raise ValueError(f"order must be 1 or an even integer >= 2, got {order!r}.")

        if trotterization_strategy is None:
            trotterization_strategy = ExactTrotterization()

        hamiltonian_spo = to_spo(hamiltonian)
        hamiltonian_clean, _ = _clean_hamiltonian_spo(hamiltonian_spo)
        if hamiltonian_clean.size == 0:
            raise ValueError("Hamiltonian contains only constant terms.")

        self._hamiltonian = hamiltonian_clean
        self.trotterization_strategy = trotterization_strategy
        self.time = time
        self.n_steps = n_steps
        self.order = order
        if isinstance(observable, Sequence) and not isinstance(observable, tuple):
            observable = tuple(observable)
        self.n_qubits = _require_qiskit_num_qubits(hamiltonian_clean.num_qubits)
        self._circuit_wires = tuple(range(self.n_qubits))
        # Normalise observables to SparsePauliOp at the input boundary, aligning
        # qubit count with the cost circuit (a 1-qubit ``Z`` on qubit 0 against
        # a multi-qubit evolution must lift to ``Z ⊗ I ⊗ … ⊗ I``).
        self.observable = self._normalise_observable(observable)

        if initial_state is None:
            initial_state = ZerosState()
        if not isinstance(initial_state, InitialState):
            raise TypeError(
                f"initial_state must be an InitialState instance, got {type(initial_state).__name__}"
            )
        self.initial_state = initial_state

        if (_template_meta is None) != (_template_param is None):
            missing = (
                "_template_param" if _template_meta is not None else "_template_meta"
            )
            raise ValueError(
                f"_template_meta and _template_param must be provided together; "
                f"got {missing}=None."
            )
        self._template_meta = _template_meta
        self._template_param = _template_param

        self._results: dict[str, float] | float | list[float] | None = None

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

    def _spec_stage(self) -> Stage:
        # TimeEvolution trotterizes the Hamiltonian into the evolution circuit.
        return TrotterSpecStage(
            trotterization_strategy=self.trotterization_strategy,
            meta_circuit_factory=self._meta_circuit_factory,
        )

    def _initial_spec(self) -> SparsePauliOp:
        return self._hamiltonian

    def _evolution_preprocessor(self) -> CircuitPreprocessor:
        """Measure the evolved state — expectation values when an observable was
        given, otherwise the computational-basis distribution."""
        result_format = (
            ResultFormat.EXPVALS if self.observable is not None else ResultFormat.PROBS
        )
        return CircuitPreprocessor(
            "evolution", result_format=result_format, cache_key="evolution"
        )

    def _preprocessors(self) -> tuple[CircuitPreprocessor, ...]:
        return (*super()._preprocessors(), self._evolution_preprocessor())

    def _evolution_params(self) -> np.ndarray:
        """One parameter set: the trajectory template binds ``t`` to ``self.time``;
        the direct path has no free parameters."""
        if self._template_meta is not None:
            return np.array([[float(self.time)]])
        return np.empty((1, 0))

    def _assemble_pipeline(
        self,
        spec_stage: Stage,
        terminal_stage: Stage,
        *,
        result_format: ResultFormat,
        extra_stages: tuple[Stage, ...] = (),
    ) -> CircuitPipeline:
        # Parameter binding is needed only for the trajectory-template path
        # (binding the template's ``t`` to ``self.time``).
        if self._template_meta is None:
            return super()._assemble_pipeline(
                spec_stage,
                terminal_stage,
                result_format=result_format,
                extra_stages=extra_stages,
            )
        mitigation_stages = self._mitigation_stages(result_format)
        bind_early = (
            bool(mitigation_stages) and self._qem_protocol.requires_bound_params
        )
        stages: list[Stage] = [spec_stage, *extra_stages]
        if bind_early:
            stages.append(ParameterBindingStage())
        stages.extend(mitigation_stages)
        stages.append(terminal_stage)
        if not bind_early:
            stages.append(ParameterBindingStage())
        return CircuitPipeline(
            stages=stages,
            suppress_performance_warnings=self._suppress_performance_warnings,
        )

    def _build_pipeline_env(self, **overrides):
        if self._template_meta is not None and "param_sets" not in overrides:
            overrides["param_sets"] = np.array([[float(self.time)]])
        return super()._build_pipeline_env(**overrides)

    def _meta_circuit_factory(
        self, result: TrotterizationResult, ham_id: int
    ) -> MetaCircuit:
        """Build a MetaCircuit from one explicit trotterization result."""
        if self._template_meta is not None:
            return self._template_meta

        qc = QuantumCircuit(self.n_qubits)
        qc.compose(self.initial_state.build(self._circuit_wires), inplace=True)
        dag = circuit_to_dag(
            result.synthesize_evolution(
                qc,
                time=self.time,
                n_steps=self.n_steps,
                order=self.order,
                qubits=list(range(self.n_qubits)),
                basis_gates=list(_QISKIT_TO_QASM2.keys()),
            )
        )

        readout: dict[str, Any]
        if self.observable is None:
            readout = {"measured_wires": tuple(range(self.n_qubits))}
        elif isinstance(self.observable, tuple):
            readout = {"observable": self.observable, "_was_multi_obs": True}
        else:
            readout = {"observable": (self.observable,)}

        return MetaCircuit(
            circuit_bodies=(((), dag),), precision=self._precision, **readout
        )

    def run(self, **kwargs) -> "TimeEvolution":
        """Execute time evolution.

        Returns:
            TimeEvolution: Returns ``self`` for method chaining.
        """
        result = self.evaluate(self._evolution_params(), self._evolution_preprocessor())

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

    def _normalise_observable(self, observable):
        """Convert observables to ``SparsePauliOp`` on the cost-circuit wires."""
        if observable is None:
            return None
        if isinstance(observable, tuple):
            return tuple(self._lift_observable(o) for o in observable)
        return self._lift_observable(observable)

    def _lift_observable(self, op) -> SparsePauliOp:
        # ``to_spo`` lifts PL-operator / dict inputs onto the cost wires and
        # validates Hermiticity (a ``SparsePauliOp`` passes through unchanged);
        # the width guard then rejects an SPO that targets the wrong register.
        spo = to_spo(op, wires=self._circuit_wires)
        if spo.num_qubits != self.n_qubits:
            raise ValueError(
                f"Observable has {spo.num_qubits} qubits but the cost circuit "
                f"has {self.n_qubits}."
            )
        return spo
