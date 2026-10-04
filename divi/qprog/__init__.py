# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Any

from ._metrics import (
    FubiniStudyMetricEstimator,
    MetricEstimator,
    PullbackMetricEstimator,
    StochasticFidelityMetricEstimator,
)
from ._types import GraphProblemTypes

# isort: off
from .mixins import (
    DataBindingMixin,
    ObservableMeasuringMixin,
    SolutionEntry,
    SolutionSamplingMixin,
)
from .aggregation import (
    AggregationStrategy,
    BeamSearchStrategy,
    HierarchicalStrategy,
)
from .algorithms import (
    PCE,
    QAOA,
    QNN,
    VQE,
    AngleEmbedding,
    Ansatz,
    CustomVQA,
    FeatureMap,
    GenericLayerAnsatz,
    HartreeFockAnsatz,
    InterpolationStrategy,
    IterativeQAOA,
    LUCJAnsatz,
    QAOAAnsatz,
    QCCAnsatz,
    TimeEvolution,
    UCCSDAnsatz,
    ZZFeatureMap,
)
from .initial_states import (
    CustomPerQubitState,
    InitialState,
    OnesState,
    SuperpositionState,
    WState,
    ZerosState,
)
from .early_stopping import EarlyStopping
from .ensemble import (
    BatchConfig,
    BatchMode,
    ProgramEnsemble,
    ReportingLevel,
    RoundRecord,
    WorkflowStatus,
)
from .optimizers import (
    GridSearchOptimizer,
    MonteCarloOptimizer,
    QNGOptimizer,
    QNSPSAOptimizer,
    QUIVEROptimizer,
    RosalinOptimizer,
    ScipyMethod,
    ScipyOptimizer,
    SPSAOptimizer,
)

# isort: on
from .quantum_program import QuantumProgram
from .variational_quantum_algorithm import (
    ParamHistoryMode,
    VariationalQuantumAlgorithm,
)
from .workflows import (
    MoleculeTransformer,
    PartitioningProgramEnsemble,
    TimeEvolutionTrajectory,
    VQEHyperparameterSweep,
)
from .workflows import _LASSQD_EXPORTS

if TYPE_CHECKING:
    from .workflows import (
        LASSQD,
        CCSDPreparation,
        FragmentationConfig,
        FragmentSpec,
        FragmentState,
        FullOrbitalSolve,
        LASSQDRoundReport,
        LASSQDState,
        LinearMethodPreparation,
        SecondOrderOrbitalSolve,
        SQDConfig,
        VQEPreparation,
    )


__all__ = [
    *_LASSQD_EXPORTS,
    "AggregationStrategy",
    "AngleEmbedding",
    "Ansatz",
    "BatchConfig",
    "BatchMode",
    "BeamSearchStrategy",
    "CustomPerQubitState",
    "CustomVQA",
    "DataBindingMixin",
    "EarlyStopping",
    "FeatureMap",
    "FubiniStudyMetricEstimator",
    "GenericLayerAnsatz",
    "GraphProblemTypes",
    "GridSearchOptimizer",
    "HartreeFockAnsatz",
    "HierarchicalStrategy",
    "InitialState",
    "InterpolationStrategy",
    "IterativeQAOA",
    "LUCJAnsatz",
    "MetricEstimator",
    "MoleculeTransformer",
    "MonteCarloOptimizer",
    "ObservableMeasuringMixin",
    "OnesState",
    "PCE",
    "ParamHistoryMode",
    "PartitioningProgramEnsemble",
    "ProgramEnsemble",
    "PullbackMetricEstimator",
    "QAOA",
    "QAOAAnsatz",
    "QCCAnsatz",
    "QNGOptimizer",
    "QNN",
    "QNSPSAOptimizer",
    "QUIVEROptimizer",
    "QuantumProgram",
    "ReportingLevel",
    "RosalinOptimizer",
    "RoundRecord",
    "SPSAOptimizer",
    "ScipyMethod",
    "ScipyOptimizer",
    "SolutionEntry",
    "SolutionSamplingMixin",
    "StochasticFidelityMetricEstimator",
    "SuperpositionState",
    "TimeEvolution",
    "TimeEvolutionTrajectory",
    "UCCSDAnsatz",
    "VQE",
    "VQEHyperparameterSweep",
    "VariationalQuantumAlgorithm",
    "WState",
    "WorkflowStatus",
    "ZZFeatureMap",
    "ZerosState",
]


def __getattr__(name: str) -> Any:
    """Resolve the LASSQD exports on first access."""
    if name in _LASSQD_EXPORTS:
        from . import workflows

        return getattr(workflows, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
