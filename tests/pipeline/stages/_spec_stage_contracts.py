# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Shared behavioural checks for circuit spec stages."""

import pytest

from divi.pipeline import CircuitPipeline, PipelineEnv
from divi.pipeline.abc import SpecStage
from divi.pipeline.stages import MeasurementStage


def verify_expand_sequence(stage: SpecStage, bell, parametric, env: PipelineEnv):
    batch, _ = stage.expand([bell, parametric], env)

    assert len(batch) == 2
    assert (("circuit", 0),) in batch
    assert (("circuit", 1),) in batch
    assert len(batch[(("circuit", 0),)].parameters) == 0
    assert len(batch[(("circuit", 1),)].parameters) == 2


def verify_expand_mapping(stage: SpecStage, bell, parametric, env: PipelineEnv):
    batch, _ = stage.expand({"bell": bell, "param": parametric}, env)

    assert len(batch) == 2
    assert (("circuit", "bell"),) in batch
    assert (("circuit", "param"),) in batch


def verify_bell_probs(stage: SpecStage, bell, simulator):
    """A non-parametric Bell state produces ~50/50 probabilities."""
    pipeline = CircuitPipeline(stages=[stage, MeasurementStage()])
    result = pipeline.run(initial_spec=bell, env=PipelineEnv(backend=simulator))

    probs = result[()]
    assert "00" in probs
    assert "11" in probs
    assert probs["00"] + probs["11"] == pytest.approx(1.0, abs=1e-12)
    # σ = sqrt(0.25 / 5000) ≈ 0.007 at the fixture's 5000 shots, so this is 5σ.
    assert probs["00"] == pytest.approx(0.5, abs=0.035)
