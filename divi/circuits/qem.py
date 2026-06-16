# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from qiskit.dagcircuit import DAGCircuit
from qiskit.quantum_info import SparsePauliOp

from divi.pipeline.abc import ResultFormat

__all__ = [
    "QEMProtocol",
    "_NoMitigation",
]

#: Type alias for QEM context data passed between expand and reduce.
#: A plain dict carrying protocol-specific side-channel information.
QEMContext = dict


class QEMProtocol(ABC):
    """Abstract base class for Quantum Error Mitigation protocols.

    Subclasses implement two methods that mirror the pipeline's
    expand/reduce lifecycle:

    * ``expand`` — given a Qiskit :class:`~qiskit.dagcircuit.DAGCircuit`
      and the observable tuple being measured, return the DAGs to execute
      on quantum hardware and a ``QEMContext`` carrying any classically-
      computed side-channel data needed during postprocessing.
    * ``reduce`` — given the context from ``expand`` and the quantum
      results, produce a ``list[float]`` of per-observable mitigated
      expectation values.

    The observable flows through as a ``tuple[SparsePauliOp, ...]`` and is
    forwarded unchanged to whichever stage needs its structure.
    """

    #: Number of Pauli-twirling samples requested by the protocol; ``0`` means
    #: no twirling.
    n_twirls: int = 0

    #: Whether ``expand`` needs concrete parameter values rather than symbolic
    #: circuit parameters.
    requires_bound_params: bool = False

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def expand(
        self,
        dag: DAGCircuit,
        observable: tuple[SparsePauliOp, ...] | None = None,
    ) -> tuple[tuple[DAGCircuit, ...], QEMContext]:
        """Generate DAGs and classical context for error mitigation.

        The input ``dag`` is consumed by this method: implementations may
        mutate it, and callers must not retain it expecting the original
        state.

        Args:
            dag: Circuit to mitigate.
            observable: ``tuple[SparsePauliOp, ...]`` (one entry per
                expectation value being measured), or ``None``.
        """

    def dry_expand(
        self,
        dag: DAGCircuit,
        observable: tuple[SparsePauliOp, ...] | None = None,
    ) -> tuple[tuple[DAGCircuit, ...], QEMContext]:
        """Analytic counterpart to :meth:`expand` used by dry-run pipelines.

        Must emit the **same number of DAGs** as :meth:`expand` would on the
        same input and populate any context keys that
        :meth:`~divi.pipeline.stages.QEMStage.introspect` inspects
        (``n_rotations``, ``n_paths``, ``symbolic``) so dry-run reports
        render correctly. Implementations
        should skip any computation that only matters at reduction time —
        classical simulation, weight evaluation, deep-copying the DAG for
        each scale factor, etc.

        The default implementation falls back to :meth:`expand`, which is
        correct but not necessarily fast; override on expensive protocols
        (e.g. QuEPP's Clifford simulation).
        """
        return self.expand(dag, observable)

    @abstractmethod
    def reduce(
        self,
        quantum_results: Sequence[Any],
        context: QEMContext,
    ) -> list[float]:
        """Combine quantum results with classical context into mitigated values.

        Returns a ``list[float]`` of per-observable mitigated values.
        ``quantum_results`` is ordered along the QEM axis; each entry is
        itself a ``list[float]`` of per-observable expectation values from
        :class:`~divi.pipeline.stages.MeasurementStage`.

        Implementations may use ``context["dag_indices"]`` (when present)
        to select the relevant positions in ``quantum_results``.
        """

    def post_reduce(self, contexts: Sequence[QEMContext]) -> None:
        """Hook called after all per-group ``reduce`` calls in an evaluation.

        Protocols can override this to inspect the collected contexts and
        emit summary diagnostics (e.g. signal-destruction warnings).
        """

    def applies_to(self, result_format: ResultFormat) -> bool:
        """Whether this protocol is applicable to ``result_format`` (``True`` for
        every format by default; protocols meaningful only for some override).

        Applicability is distinct from doing work: a no-op protocol is vacuously
        applicable, while pipeline assembly decides whether a protocol requires
        a QEM stage.
        """
        return True


class _NoMitigation(QEMProtocol):
    """A dummy default mitigation protocol — pass the circuit through."""

    @property
    def name(self) -> str:
        return "NoMitigation"

    def expand(
        self,
        dag: DAGCircuit,
        observable: tuple[SparsePauliOp, ...] | None = None,
    ) -> tuple[tuple[DAGCircuit, ...], QEMContext]:
        return (dag,), {"dag_indices": [0]}

    def reduce(
        self,
        quantum_results: Sequence[Any],
        context: QEMContext,
    ) -> list[float]:
        indices = context.get("dag_indices")
        selected = (
            [quantum_results[i] for i in indices]
            if indices is not None
            else list(quantum_results)
        )
        if len(selected) == 0:
            raise RuntimeError("NoMitigation received an empty results sequence.")
        if len(selected) > 1:
            raise RuntimeError("NoMitigation class received multiple partial results.")
        only = selected[0]
        if isinstance(only, list):
            return [float(v) for v in only]
        return [float(only)]
