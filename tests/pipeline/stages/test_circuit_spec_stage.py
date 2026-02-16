# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for CircuitSpecStage: single, sequence, and mapping inputs."""

import numpy as np
import pennylane as qml
import pytest

from divi.circuits import MetaCircuit
from divi.pipeline.abc import PipelineEnv
from divi.pipeline.stages import CircuitSpecStage


def _make_meta(n_wires: int = 1) -> MetaCircuit:
    """Create a minimal MetaCircuit for testing."""
    qscript = qml.tape.QuantumScript(
        ops=[qml.Hadamard(i) for i in range(n_wires)],
        measurements=[qml.expval(qml.Z(0))],
    )
    return MetaCircuit(source_circuit=qscript, symbols=np.array([], dtype=object))


class TestCircuitSpecStageExpand:
    """Expand contract: each input shape produces correctly keyed batches."""

    def test_single_meta_circuit(self, dummy_expval_backend):
        stage = CircuitSpecStage()
        env = PipelineEnv(backend=dummy_expval_backend)
        meta = _make_meta()

        batch, token = stage.expand(meta, env)

        assert len(batch) == 1
        key = next(iter(batch))
        assert key == (("circuit", 0),)
        assert batch[key] is meta

    def test_sequence_of_meta_circuits(self, dummy_expval_backend):
        stage = CircuitSpecStage()
        env = PipelineEnv(backend=dummy_expval_backend)
        metas = [_make_meta(1), _make_meta(2)]

        batch, token = stage.expand(metas, env)

        assert len(batch) == 2
        assert (("circuit", 0),) in batch
        assert (("circuit", 1),) in batch
        assert batch[(("circuit", 0),)] is metas[0]
        assert batch[(("circuit", 1),)] is metas[1]

    def test_mapping_of_meta_circuits(self, dummy_expval_backend):
        stage = CircuitSpecStage()
        env = PipelineEnv(backend=dummy_expval_backend)
        cost = _make_meta(1)
        meas = _make_meta(2)
        spec = {"cost": cost, "meas": meas}

        batch, token = stage.expand(spec, env)

        assert len(batch) == 2
        assert (("circuit", "cost"),) in batch
        assert (("circuit", "meas"),) in batch
        assert batch[(("circuit", "cost"),)] is cost
        assert batch[(("circuit", "meas"),)] is meas

    def test_invalid_input_raises_type_error(self, dummy_expval_backend):
        stage = CircuitSpecStage()
        env = PipelineEnv(backend=dummy_expval_backend)

        with pytest.raises(TypeError, match="CircuitSpecStage expects"):
            stage.expand(42, env)


class TestCircuitSpecStageReduce:
    """Reduce contract: circuit axis is stripped from results."""

    def test_reduce_single_circuit(self, dummy_expval_backend):
        stage = CircuitSpecStage()
        env = PipelineEnv(backend=dummy_expval_backend)
        results = {(("circuit", 0), ("meas", 0)): 1.5}

        reduced = stage.reduce(results, env, token="single")

        assert (("meas", 0),) in reduced
        assert reduced[(("meas", 0),)] == 1.5

    def test_reduce_multiple_circuits(self, dummy_expval_backend):
        stage = CircuitSpecStage()
        env = PipelineEnv(backend=dummy_expval_backend)
        results = {
            (("circuit", "cost"), ("meas", 0)): 1.0,
            (("circuit", "meas"), ("meas", 0)): 2.0,
        }

        reduced = stage.reduce(results, env, token="mapping")

        # Both circuits share the same downstream key ("meas", 0), so they
        # are grouped under (("meas", 0),) as a list.
        assert (("meas", 0),) in reduced
        assert reduced[(("meas", 0),)] == [1.0, 2.0]
