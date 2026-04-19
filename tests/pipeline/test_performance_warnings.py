# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for pipeline performance warnings and the no-op-stage guard."""

import warnings

import pytest

from divi.circuits.quepp import QuEPP
from divi.pipeline import CircuitPipeline, DiviPerformanceWarning
from divi.pipeline.abc import (
    BundleStage,
    ExpansionResult,
    MetaCircuitBatch,
    PipelineEnv,
    StageToken,
)
from divi.pipeline.stages import (
    MeasurementStage,
    ParameterBindingStage,
    PauliTwirlStage,
    QEMStage,
)
from tests.pipeline.helpers import DummySpecStage, two_group_meta


def _pipeline_stages_exhaustive_quepp() -> list:
    return [
        DummySpecStage(meta=two_group_meta()),
        QEMStage(
            protocol=QuEPP(
                sampling="exhaustive",
                truncation_order=1,
                n_twirls=1,
            )
        ),
        PauliTwirlStage(n_twirls=1, seed=0),
        ParameterBindingStage(),
        MeasurementStage(),
    ]


def _pipeline_stages_param_bind_before_qem() -> list:
    return [
        DummySpecStage(meta=two_group_meta()),
        ParameterBindingStage(),
        QEMStage(
            protocol=QuEPP(
                sampling="montecarlo",
                truncation_order=1,
                n_twirls=1,
            )
        ),
        PauliTwirlStage(n_twirls=1, seed=0),
        MeasurementStage(),
    ]


class TestExhaustiveQuEPPWarning:
    """Spec: QuEPP with sampling='exhaustive' emits DiviPerformanceWarning."""

    def test_exhaustive_sampling_warns(self):
        with pytest.warns(DiviPerformanceWarning, match="exhaustive"):
            CircuitPipeline(stages=_pipeline_stages_exhaustive_quepp())

    def test_montecarlo_sampling_does_not_warn(self):
        stages = [
            DummySpecStage(meta=two_group_meta()),
            QEMStage(
                protocol=QuEPP(
                    sampling="montecarlo",
                    truncation_order=1,
                    n_twirls=1,
                )
            ),
            PauliTwirlStage(n_twirls=1, seed=0),
            ParameterBindingStage(),
            MeasurementStage(),
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("error", DiviPerformanceWarning)
            CircuitPipeline(stages=stages)


class TestParamBindBeforeQEMWarning:
    """Spec: ParameterBindingStage placed before QEMStage emits DiviPerformanceWarning."""

    def test_param_bind_before_qem_warns(self):
        with pytest.warns(DiviPerformanceWarning, match="ParameterBindingStage"):
            CircuitPipeline(stages=_pipeline_stages_param_bind_before_qem())

    def test_param_bind_after_qem_does_not_warn(self):
        stages = [
            DummySpecStage(meta=two_group_meta()),
            QEMStage(
                protocol=QuEPP(
                    sampling="montecarlo",
                    truncation_order=1,
                    n_twirls=1,
                )
            ),
            PauliTwirlStage(n_twirls=1, seed=0),
            ParameterBindingStage(),
            MeasurementStage(),
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("error", DiviPerformanceWarning)
            CircuitPipeline(stages=stages)

    def test_no_mitigation_qem_does_not_warn(self):
        """QEMStage with default _NoMitigation protocol is benign regardless of ordering."""
        stages = [
            DummySpecStage(meta=two_group_meta()),
            ParameterBindingStage(),
            QEMStage(),  # _NoMitigation — pass-through
            MeasurementStage(),
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("error", DiviPerformanceWarning)
            CircuitPipeline(stages=stages)


class TestMultipleWarnings:
    """Both footguns at once emit both warnings."""

    def test_exhaustive_and_param_bind_before_qem(self):
        stages = [
            DummySpecStage(meta=two_group_meta()),
            ParameterBindingStage(),
            QEMStage(
                protocol=QuEPP(
                    sampling="exhaustive",
                    truncation_order=1,
                    n_twirls=1,
                )
            ),
            PauliTwirlStage(n_twirls=1, seed=0),
            MeasurementStage(),
        ]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DiviPerformanceWarning)
            CircuitPipeline(stages=stages)

        messages = [
            str(w.message)
            for w in caught
            if issubclass(w.category, DiviPerformanceWarning)
        ]
        assert any("exhaustive" in m for m in messages)
        assert any("ParameterBindingStage" in m for m in messages)


class TestSuppressionKwarg:
    """Spec: suppress_performance_warnings=True silences all DiviPerformanceWarnings."""

    def test_kwarg_silences_exhaustive_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", DiviPerformanceWarning)
            CircuitPipeline(
                stages=_pipeline_stages_exhaustive_quepp(),
                suppress_performance_warnings=True,
            )

    def test_kwarg_silences_ordering_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", DiviPerformanceWarning)
            CircuitPipeline(
                stages=_pipeline_stages_param_bind_before_qem(),
                suppress_performance_warnings=True,
            )


class TestNoOpStageGuard:
    """Spec: a BundleStage with no work declared emits a UserWarning at instantiation."""

    def test_no_op_stage_warns_on_init(self):
        class NoOpStage(BundleStage):
            @property
            def handles_measurement(self) -> bool:
                return False

            @property
            def consumes_dag_bodies(self) -> bool:
                return False

            def expand(
                self, batch: MetaCircuitBatch, env: PipelineEnv
            ) -> tuple[ExpansionResult, StageToken]:
                return ExpansionResult(batch=batch), None

        with pytest.warns(UserWarning, match="no-op"):
            NoOpStage(name="NoOpStage")

    def test_measurement_stage_does_not_warn(self):
        """MeasurementStage declares handles_measurement=True — no no-op warning."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            MeasurementStage()

    def test_dag_consuming_stage_does_not_warn(self):
        """ParameterBindingStage keeps consumes_dag_bodies=True by default — no no-op warning."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            ParameterBindingStage()
