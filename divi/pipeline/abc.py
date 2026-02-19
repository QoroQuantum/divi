# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

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

NodeKey = tuple[AxisLabel, ...]  # Batch key: sequence of (axis_name, value) pairs.

MetaCircuitBatch = dict[NodeKey, MetaCircuit]
BranchKey = tuple[AxisLabel, ...]  # Full branch key: (axis_name, value) pairs.
ParentBranchResults = dict[NodeKey, dict[BranchKey, Any]]
ChildResults = dict[Any, Any]

StageToken = Any

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
    """Probability distributions: ``{bitstring: probability}``."""

    EXPVALS = "expvals"
    """Expectation values: ``{observable_key: float}`` mapping per branch key."""


@dataclass(frozen=True)
class ExpansionResult:
    """Bundle-stage expansion output."""

    batch: MetaCircuitBatch
    stage_name: str | None = None
    """Stage name attached by planner for forward-pass observability."""


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
    """Parameter sets for binding: strictly 2D (list-of-lists or 2D ndarray)."""

    artifacts: dict = field(default_factory=dict)
    """Mutable output dict populated during execution (e.g. ``circuit_count``)."""

    result_format: ResultFormat | None = None
    """Canonical result format, set by the measurement stage during expand."""

    reporter: ProgressReporter | None = None
    """Progress reporter for async polling feedback."""

    cancellation_event: Event | None = None
    """Threading event signalling cancellation (set by ProgramBatch)."""


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


class SpecStage(Stage[InT, MetaCircuitBatch], ABC):
    """First stage in every pipeline: converts an arbitrary spec into a keyed MetaCircuit batch.

    Examples:
        - ``CircuitSpecStage``: wraps one or more pre-built ``MetaCircuit``
          instances into a batch (by position or by name).
        - ``TrotterSpecStage``: takes a Hamiltonian and decomposes it into one
          or more ``MetaCircuit`` entries via a trotterisation factory.
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
    """Abstract stage that transforms a keyed MetaCircuit batch."""

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
