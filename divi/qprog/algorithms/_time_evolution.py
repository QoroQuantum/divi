# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import numpy as np
import pennylane as qp
from qiskit import transpile
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import SparsePauliOp
from qiskit.synthesis import LieTrotter, SuzukiTrotter

from divi.circuits import MetaCircuit
from divi.circuits._conversions import _QISKIT_TO_QASM2, _observable_to_sparse_pauli_op
from divi.circuits.qem import _NoMitigation
from divi.hamiltonians import (
    ExactTrotterization,
    TrotterizationStrategy,
)
from divi.hamiltonians._term_ops import (
    _clean_hamiltonian_spo,
    _n_qubits,
    _spo_to_qiskit_basis_gates,
    to_spo,
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
        hamiltonian: qp.operation.Operator | SparsePauliOp,
        trotterization_strategy: TrotterizationStrategy | None = None,
        time: float = 1.0,
        n_steps: int = 1,
        order: int = 1,
        initial_state: InitialState | None = None,
        observable: (
            qp.operation.Operator
            | SparsePauliOp
            | tuple[qp.operation.Operator | SparsePauliOp, ...]
            | list[qp.operation.Operator | SparsePauliOp]
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

        hamiltonian_spo = to_spo(hamiltonian)
        hamiltonian_clean, _ = _clean_hamiltonian_spo(hamiltonian_spo)
        if hamiltonian_clean.size == 0:
            raise ValueError("Hamiltonian contains only constant terms.")

        self._hamiltonian = hamiltonian_clean
        self.trotterization_strategy = trotterization_strategy
        self.time = time
        self.n_steps = n_steps
        self.order = order
        if isinstance(observable, list):
            observable = tuple(observable)
        self.n_qubits = _n_qubits(hamiltonian_clean)
        self._circuit_wires = tuple(range(self.n_qubits))
        # Normalise observables to SparsePauliOp at the input boundary, aligning
        # qubit count with the cost circuit (a 1-qubit ``qp.PauliZ(0)`` against
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

        qc = self._build_qiskit_circuit(processed_spo)
        dag = circuit_to_dag(qc)

        if self.observable is None:
            return MetaCircuit(
                circuit_bodies=(((), dag),),
                measured_wires=tuple(range(self.n_qubits)),
                precision=8,
            )
        if isinstance(self.observable, tuple):
            return MetaCircuit(
                circuit_bodies=(((), dag),),
                observable=self.observable,
                precision=8,
                _was_multi_obs=True,
            )
        return MetaCircuit(
            circuit_bodies=(((), dag),),
            observable=(self.observable,),
            precision=8,
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
        # PennyLane operator: lift onto the full wire register.
        return _observable_to_sparse_pauli_op(op, self._circuit_wires)

    def _build_qiskit_circuit(self, processed_spo: SparsePauliOp) -> QuantumCircuit:
        """Build initial-state preparation + Trotter evolution as a ``QuantumCircuit``.

        Adjoint evolution is realized via negative time, matching the prior
        PennyLane ``adjoint(TrotterProduct/evolve)`` path. Single-term
        Hamiltonians use positive time to preserve ``qp.evolve(term, coeff=t)``
        semantics.

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

        sampled_spo = getattr(self.trotterization_strategy, "_last_sampled_spo", None)
        if sampled_spo is not None:
            # Faithful QDrift: one evolution gate per sampled term (preserving
            # sampling-with-replacement multiplicities), repeated ``n_steps``
            # times with ``time / n_steps`` per step. Adjoint via negative time.
            step_time = -self.time / self.n_steps
            for _ in range(self.n_steps):
                _spo_to_qiskit_basis_gates(qc, sampled_spo, step_time, qubits)
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
            # Transpile down to basis gates so the QASM body emitter and
            # Clifford-only stages (e.g. QuEPP) can consume the result.
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
