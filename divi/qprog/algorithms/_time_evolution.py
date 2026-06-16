# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence

import numpy as np
from qiskit import transpile
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import SparsePauliOp
from qiskit.synthesis import LieTrotter, SuzukiTrotter

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
    _spo_to_qiskit_basis_gates,
    to_spo,
)
from divi.pipeline import CircuitPipeline, PipelineSet, ResultFormat, Stage
from divi.pipeline.stages import (
    MeasurementStage,
    ParameterBindingStage,
    TrotterSpecStage,
)
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

    def _build_pipelines(self) -> PipelineSet:
        # Non-variational program: parameter binding is needed only for the
        # trajectory-template path (binding the template's ``t`` to ``self.time``).
        result_format = (
            ResultFormat.EXPVALS if self.observable is not None else ResultFormat.PROBS
        )
        spec_stage = TrotterSpecStage(
            trotterization_strategy=self.trotterization_strategy,
            meta_circuit_factory=self._meta_circuit_factory,
        )
        terminal_stage = MeasurementStage(
            grouping_strategy=self._grouping_strategy,
            shot_distribution=self._shot_distribution,
        )

        if self._template_meta is None:
            evolution = self._assemble_pipeline(
                spec_stage,
                terminal_stage,
                result_format=result_format,
            )
        else:
            mitigation_stages = self._mitigation_stages(result_format)
            bind_early = (
                bool(mitigation_stages) and self._qem_protocol.requires_bound_params
            )
            stages: list[Stage] = [spec_stage]
            if bind_early:
                stages.append(ParameterBindingStage())
            stages.extend(mitigation_stages)
            stages.append(terminal_stage)
            if not bind_early:
                stages.append(ParameterBindingStage())
            evolution = CircuitPipeline(stages=stages)

        return PipelineSet({"evolution": (evolution, lambda: self._hamiltonian)})

    def _build_pipeline_env(self, **overrides):
        if self._template_meta is not None and "param_sets" not in overrides:
            overrides["param_sets"] = np.array([[float(self.time)]])
        return super()._build_pipeline_env(**overrides)

    @property
    def _pipeline(self) -> CircuitPipeline:
        """The evolution pipeline (thin accessor over ``self._pipelines``)."""
        return self._pipelines["evolution"]

    def _meta_circuit_factory(
        self, result: TrotterizationResult, ham_id: int
    ) -> MetaCircuit:
        """Build a MetaCircuit from one explicit trotterization result."""
        if self._template_meta is not None:
            return self._template_meta

        qc = self._build_qiskit_circuit(
            result.effective_hamiltonian,
            sampled_terms=result.sampled_terms,
        )
        dag = circuit_to_dag(qc)

        if self.observable is None:
            return MetaCircuit(
                circuit_bodies=(((), dag),),
                measured_wires=tuple(range(self.n_qubits)),
                precision=self._precision,
            )
        if isinstance(self.observable, tuple):
            return MetaCircuit(
                circuit_bodies=(((), dag),),
                observable=self.observable,
                precision=self._precision,
                _was_multi_obs=True,
            )
        return MetaCircuit(
            circuit_bodies=(((), dag),),
            observable=(self.observable,),
            precision=self._precision,
        )

    def run(self, **kwargs) -> "TimeEvolution":
        """Execute time evolution.

        Returns:
            TimeEvolution: Returns ``self`` for method chaining.
        """
        result = self._run_pipeline("evolution")

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
        if isinstance(op, SparsePauliOp):
            if op.num_qubits != self.n_qubits:
                raise ValueError(
                    f"Observable has {op.num_qubits} qubits but the cost circuit "
                    f"has {self.n_qubits}."
                )
            return op
        # Non-SPO input (PL operator, Pauli-string dict): lift onto the
        # full wire register so a narrow observable matches the cost circuit.
        return to_spo(op, wires=self._circuit_wires)

    def _build_qiskit_circuit(
        self,
        processed_spo: SparsePauliOp,
        *,
        sampled_terms: SparsePauliOp | None = None,
    ) -> QuantumCircuit:
        """Build initial-state preparation + Trotter evolution as a ``QuantumCircuit``.

        Adjoint evolution is realized via negative time. Single-term
        Hamiltonians use positive time to preserve the standard
        ``exp(-i t H)`` sign convention even when ``H`` carries its own
        coefficient sign.

        QDrift sampled-term evolution uses
        :func:`_spo_to_qiskit_basis_gates` directly so each sampled term
        keeps its multiplicity. Standard Trotter uses
        :class:`PauliEvolutionGate` with :class:`LieTrotter` /
        :class:`SuzukiTrotter` synthesis; the circuit is then transpiled
        down to the basis-gate set the QASM body emitter understands.
        """
        qc = QuantumCircuit(self.n_qubits)
        qc.compose(self.initial_state.build(self._circuit_wires), inplace=True)
        qubits = list(range(self.n_qubits))

        if sampled_terms is not None:
            # Faithful QDrift: one evolution gate per sampled term (preserving
            # sampling-with-replacement multiplicities), repeated ``n_steps``
            # times with ``time / n_steps`` per step. Adjoint via negative time.
            step_time = -self.time / self.n_steps
            for _ in range(self.n_steps):
                _spo_to_qiskit_basis_gates(qc, sampled_terms, step_time, qubits)
            return qc

        # Standard Trotter-Suzuki for ExactTrotterization.
        if processed_spo.size >= 2:
            evolution_time = -self.time
            synthesis = (
                LieTrotter(reps=self.n_steps, preserve_order=True)
                if self.order == 1
                else SuzukiTrotter(
                    order=self.order, reps=self.n_steps, preserve_order=True
                )
            )
            qc.append(
                PauliEvolutionGate(
                    processed_spo, time=evolution_time, synthesis=synthesis
                ),
                qubits,
            )
            # Lower to the gate set ``dag_to_qasm_body`` accepts.  Qiskit's
            # Trotter synthesis can emit ``rxx``/``ryy``/``rzz``-style compound
            # rotations that the QASM2 emitter raises on; ``optimization_level=0``
            # keeps it to a cheap gate-by-gate substitution.
            try:
                qc = transpile(
                    qc,
                    basis_gates=list(_QISKIT_TO_QASM2.keys()),
                    optimization_level=0,
                )
            except Exception as exc:
                raise RuntimeError(
                    "TimeEvolution failed to lower the Trotter-synthesised "
                    "circuit to Divi's supported basis-gate set. This usually "
                    "means PauliEvolutionGate synthesis emitted a gate Divi's "
                    "QASM2 emitter does not handle. Supported gates: "
                    f"{sorted(_QISKIT_TO_QASM2.keys())}."
                ) from exc
        else:
            # Single-term Hamiltonian — positive-time convention.
            _spo_to_qiskit_basis_gates(qc, processed_spo, self.time, qubits)
        return qc
