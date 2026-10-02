# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for CircuitSpecStage: single, sequence, and mapping inputs."""

import pytest
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag

from divi.circuits import MetaCircuit
from divi.pipeline.abc import PipelineEnv
from divi.pipeline.stages import CircuitSpecStage


def _make_meta(n_wires: int = 1) -> MetaCircuit:
    """Create a minimal MetaCircuit for testing."""
    qc = QuantumCircuit(n_wires)
    for i in range(n_wires):
        qc.h(i)
    return MetaCircuit(circuit_bodies=(((), circuit_to_dag(qc)),))


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
        assert token == "single"

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
        assert token == "sequence"

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
        assert token == "mapping"

    def test_invalid_input_raises_type_error(self, dummy_expval_backend):
        stage = CircuitSpecStage()
        env = PipelineEnv(backend=dummy_expval_backend)

        with pytest.raises(
            TypeError,
            match="^CircuitSpecStage expects a MetaCircuit, sequence, or mapping, "
            "got int$",
        ):
            stage.expand(42, env)


def test_stage_is_named_after_its_class():
    assert CircuitSpecStage().name == "CircuitSpecStage"


@pytest.mark.parametrize(
    "items, type_name",
    [("abc", "str"), (42, "int")],
    ids=["string", "unsupported"],
)
def test_convert_by_shape_rejects_non_container_inputs(items, type_name):
    with pytest.raises(TypeError, match=f"^Expected a widget, got {type_name}$"):
        CircuitSpecStage._convert_by_shape(
            items, str.upper, single_type=bytes, expected="Expected a widget"
        )


def test_convert_by_shape_maps_over_each_shape():
    def convert(items):
        return CircuitSpecStage._convert_by_shape(
            items, str.upper, single_type=str, expected="Expected a string"
        )

    assert convert("ab") == "AB"
    assert convert(["a", "b"]) == ["A", "B"]
    assert convert({"x": "a"}) == {"x": "A"}


def test_introspect_of_an_empty_batch_is_empty(dummy_expval_backend):
    env = PipelineEnv(backend=dummy_expval_backend)
    assert CircuitSpecStage().introspect({}, env, token=None) == {}


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


def test_depth_and_depth_2q_in_metadata(dummy_expval_backend):
    """Spec: ``introspect()`` reports gate counts plus depth and depth_2q
    for the sample (first) DAG body — the same surface ``CircuitRunner``
    exposes at runtime via ``depth_history``."""
    # H, then CX, then H — three gates in three layers, exactly one is 2q.
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.h(0)
    meta = MetaCircuit(circuit_bodies=(((), circuit_to_dag(qc)),))

    stage = CircuitSpecStage()
    env = PipelineEnv(backend=dummy_expval_backend)
    batch, token = stage.expand(meta, env)

    info = stage.introspect(batch, env, token)
    assert info == {
        "n_qubits": 2,
        "n_gates": 3,
        "n_1q_gates": 2,
        "n_2q_gates": 1,
        "depth": circuit_to_dag(qc).depth(),
        "depth_2q": 1,
    }
