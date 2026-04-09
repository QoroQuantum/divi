# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for pipeline-level design-by-contract: measurement exclusivity and default validate."""

import pytest

from divi.pipeline import CircuitPipeline
from divi.pipeline.stages import MeasurementStage, PauliTwirlStage

from .helpers import DummySpecStage, FanoutAndSumStage, two_group_meta


class TestMeasurementExclusivity:
    """Spec: at most one measurement-handling stage per pipeline."""

    def test_single_measurement_stage_passes(self):
        CircuitPipeline(
            stages=[DummySpecStage(meta=two_group_meta()), MeasurementStage()]
        )

    def test_duplicate_measurement_stages_raises(self):
        with pytest.raises(
            ValueError,
            match="Multiple measurement-handling stages",
        ):
            CircuitPipeline(
                stages=[
                    DummySpecStage(meta=two_group_meta()),
                    MeasurementStage(),
                    MeasurementStage(),
                ]
            )


class TestPauliTwirlStandalone:
    """Spec: PauliTwirlStage works without QEMStage."""

    def test_pauli_twirl_without_qem_passes(self):
        CircuitPipeline(
            stages=[
                DummySpecStage(meta=two_group_meta()),
                PauliTwirlStage(n_twirls=5),
                MeasurementStage(),
            ]
        )


class TestDefaultValidateIsNoop:
    """Spec: stages without validate overrides do not block pipeline construction."""

    def test_plain_bundle_stages_pass(self):
        CircuitPipeline(
            stages=[
                DummySpecStage(meta=two_group_meta()),
                FanoutAndSumStage("x", 2),
                MeasurementStage(),
            ]
        )
