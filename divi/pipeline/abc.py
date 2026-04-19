# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import warnings
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from threading import Event
from typing import Any, Generic, TypeVar

import numpy as np
import numpy.typing as npt

from divi.backends import CircuitRunner
from divi.circuits import MetaCircuit
from divi.reporting import ProgressReporter
from divi.typing import AxisLabel

__all__ = [
    "BundleStage",
    "ContractViolation",
    "DiviPerformanceWarning",
    "ExpansionResult",
    "NodeKey",
    "PipelineEnv",
    "PipelineResult",
    "PipelineTrace",
    "ResultFormat",
    "SpecStage",
    "Stage",
]

NodeKey = tuple[AxisLabel, ...]  # Batch key: sequence of (axis_name, value) pairs.

MetaCircuitBatch = dict[NodeKey, MetaCircuit]
BranchKey = tuple[AxisLabel, ...]  # Full branch key: (axis_name, value) pairs.
ParentBranchResults = dict[NodeKey, dict[BranchKey, Any]]
ChildResults = dict[Any, Any]

StageToken = Any


class PipelineResult(dict):
    """Pipeline result dict with convenience access for single-result pipelines.

    Behaves exactly like a regular ``dict`` keyed by ``NodeKey`` tuples.
    For the common single-circuit case, use the :attr:`value` property
    instead of ``result[()]``.
    """

    @property
    def value(self):
        """Return the single result value.

        Equivalent to ``result[()]`` for single-circuit pipelines.

        Raises:
            ValueError: If the result contains more than one key.
        """
        if len(self) != 1:
            raise ValueError(
                f".value requires exactly one result key, got {len(self)}. "
                f"Keys: {list(self.keys())}. "
                f"Use result[key] to access specific results."
            )
        return next(iter(self.values()))


InT = TypeVar("InT")  # Generic input type consumed by Stage.expand.

OutT = TypeVar("OutT")  # Generic output type produced by Stage.expand.


class ResultFormat(Enum):
    """Canonical format that raw backend results should be converted into.

    Set by a measurement stage during ``expand``; read by ``pipeline.run()``
    to apply the correct conversion between execute and reduce.
    """

    COUNTS = "counts"
    """Raw shot counts — no conversion. Used by PCE (nonlinear reduce)."""

    PROBS = "probs"
    """Probability distributions (``{bitstring: probability}``)."""

    EXPVALS = "expvals"
    """Expectation values (``{observable_key: float}`` mapping per branch key)."""


@dataclass(frozen=True)
class ExpansionResult:
    """Bundle-stage expansion output."""

    batch: MetaCircuitBatch
    stage_name: str | None = None
    """Stage name attached by planner for forward-pass traceability."""


@dataclass(frozen=True)
class PipelineTrace:
    """Forward-pass pipeline trace for fan-out verification before execution."""

    initial_batch: MetaCircuitBatch
    """The batch of MetaCircuits before any stage expansion."""

    final_batch: MetaCircuitBatch
    """The fully-expanded batch after all stages have run."""

    stage_expansions: tuple[ExpansionResult, ...]
    """Per-stage expansion results, one entry per BundleStage in expand order."""

    stage_tokens: tuple[StageToken, ...]
    """Per-stage opaque tokens returned by each BundleStage's expand."""

    result_format: "ResultFormat | None" = None
    """Result format declared by the measurement stage during expand."""

    env_artifacts: dict = field(default_factory=dict)
    """Stage-produced artifacts (e.g. ham_ops) captured for cache restore."""


@dataclass
class PipelineEnv:
    """Per-run context for the circuit pipeline.

    The client passes the backend and any stage-specific data
    when constructing the env for a pipeline run.
    """

    backend: CircuitRunner
    """Backend used to run circuits (e.g. simulator or cloud service)."""

    param_sets: Sequence[Sequence[float]] | npt.NDArray[np.floating] = ()
    """Parameter sets for binding — strictly 2D (list-of-lists or 2D ndarray)."""

    artifacts: dict = field(default_factory=dict)
    """Mutable output dict populated during execution (e.g. ``circuit_count``)."""

    result_format: ResultFormat | None = None
    """Canonical result format, set by the measurement stage during expand."""

    reporter: ProgressReporter | None = None
    """Progress reporter for async polling feedback."""

    cancellation_event: Event | None = None
    """Threading event signalling cancellation (set by ProgramEnsemble)."""


class ContractViolation(ValueError):
    """Raised when a stage's positional requirements are not met."""


class DiviPerformanceWarning(UserWarning):
    """Emitted when a pipeline configuration is known to be slow.

    Raised from :class:`~divi.pipeline.CircuitPipeline` at construction
    time for configurations that are legal but suboptimal (e.g. exhaustive
    QuEPP sampling, ParameterBindingStage placed before QEMStage).
    Suppress by passing ``suppress_performance_warnings=True`` to the
    pipeline constructor, or by filtering on this class at module level.
    """


class Stage(ABC, Generic[InT, OutT]):
    """Abstract base for pipeline stages."""

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def axis_name(self) -> str:
        """Axis name introduced by this stage."""
        return self._name

    @property
    def stateful(self) -> bool:
        """Whether this stage invalidates forward-pass reuse from this point."""
        return False

    def validate(self, before: tuple["Stage", ...], after: tuple["Stage", ...]) -> None:
        """Check this stage's position in the pipeline.

        Called by :class:`~divi.pipeline.CircuitPipeline` at construction
        time after structural validation.  Override to inspect neighboring
        stages and either:

        * raise :class:`~divi.pipeline.abc.ContractViolation` if
          preconditions are not met, or
        * emit :class:`~divi.pipeline.DiviPerformanceWarning` for
          legal-but-slow configurations (e.g. expensive internal options,
          known-bad neighboring stages).  Suppressed at the pipeline level
          via ``CircuitPipeline(..., suppress_performance_warnings=True)``.

        Args:
            before: Stages before this one in expand order.
            after: Stages after this one in expand order.
        """

    @abstractmethod
    def expand(self, items: InT, env: PipelineEnv) -> tuple[OutT, StageToken]:
        """Transform input for the forward pass and return a reduction token."""
        ...

    @abstractmethod
    def reduce(
        self, results: ChildResults, env: PipelineEnv, token: StageToken
    ) -> ChildResults:
        """Transform results in the backward pass using the forward-pass token."""
        ...

    def introspect(
        self, batch: MetaCircuitBatch, env: PipelineEnv, token: StageToken
    ) -> dict[str, Any]:
        """Return stage-specific metadata for dry-run reporting.

        Override in subclasses to provide richer introspection data.
        Called by the dry-run tool after ``expand`` with the post-expand
        batch, the pipeline env, and the stage's token.
        """
        return {}


class SpecStage(Stage[InT, MetaCircuitBatch], ABC):
    """First stage in every pipeline: converts an arbitrary spec into a keyed MetaCircuit batch.

    Examples:
        - ``CircuitSpecStage``: wraps one or more pre-built ``MetaCircuit``
          instances into a batch (by position or by name).
        - ``TrotterSpecStage``: takes a Hamiltonian and decomposes it into one
          or more ``MetaCircuit`` entries via a trotterization factory.
    """

    @abstractmethod
    def expand(
        self, items: InT, env: PipelineEnv
    ) -> tuple[MetaCircuitBatch, StageToken]:
        """Transform input (e.g. Hamiltonian) into a keyed batch of MetaCircuits."""
        ...

    def reduce(
        self, results: ChildResults, env: PipelineEnv, token: StageToken
    ) -> ChildResults:
        """Identity by default; override if this stage reduces results."""
        return results


class BundleStage(Stage[MetaCircuitBatch, ExpansionResult], ABC):
    """Abstract stage that transforms a keyed MetaCircuit batch.

    Subclasses declare two orthogonal contracts via class properties:

    - :attr:`handles_measurement` — this stage emits measurement QASMs and
      sets :attr:`~divi.pipeline.PipelineEnv.result_format`.
    - :attr:`consumes_dag_bodies` — this stage reads (and typically mutates)
      ``meta.circuit_bodies`` during ``expand``.

    The pipeline is transformative by design: every ``BundleStage`` is
    expected to either handle measurement or consume body DAGs (or both).
    Declaring neither is almost always a misuse of the abstraction —
    metadata-only or logging passes belong outside the ``Stage`` ABC —
    so constructing such a stage emits a ``UserWarning`` at instantiation
    time.
    """

    def __init__(self, name: str) -> None:
        super().__init__(name=name)
        if not self.handles_measurement and not self.consumes_dag_bodies:
            warnings.warn(
                f"BundleStage {type(self).__name__!r} declares neither "
                "measurement handling nor DAG consumption; it is a no-op "
                "in the pipeline. If this is intentional, set one of "
                "handles_measurement / consumes_dag_bodies to True; "
                "otherwise use a non-Stage mechanism (hook, middleware).",
                UserWarning,
                stacklevel=3,
            )

    @property
    def handles_measurement(self) -> bool:
        """Whether this stage sets up measurement circuits and result format.

        Pipelines must contain at least one stage with this property True.
        """
        return False

    @property
    def consumes_dag_bodies(self) -> bool:
        """Whether this stage reads ``meta.circuit_bodies`` during ``expand``.

        Default ``True`` — the safe assumption. Override with ``False`` on
        stages that only inspect measurement/observable metadata
        (e.g. :class:`~divi.pipeline.stages.MeasurementStage`,
        :class:`~divi.pipeline.stages.PCECostStage`).
        Used by
        :class:`~divi.pipeline.stages.ParameterBindingStage` to decide
        whether it can stay on the fast QASM-template render path.
        """
        return True

    @abstractmethod
    def expand(
        self, batch: MetaCircuitBatch, env: PipelineEnv
    ) -> tuple[ExpansionResult, StageToken]:
        """Transform keyed MetaCircuit batch and return expansion lineage plus token."""
        ...

    def reduce(
        self, results: ChildResults, env: PipelineEnv, token: StageToken
    ) -> ChildResults:
        """Identity by default; override if this stage reduces results."""
        return results
