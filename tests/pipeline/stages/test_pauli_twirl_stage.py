# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for divi.pipeline.stages._pauli_twirl_stage."""

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.quantum_info import Operator, SparsePauliOp

from divi.circuits import MetaCircuit
from divi.pipeline import CircuitPipeline, PipelineEnv
from divi.pipeline.stages import (
    MeasurementStage,
    ParameterBindingStage,
    PauliTwirlStage,
)
from tests.pipeline._helpers import (
    DummySpecStage,
    meta_from_circuit,
    parametric_twirlable_meta,
    stage_body_tags,
    stage_output,
)

_BASE_KEY = (("spec", "circ"),)


def _twirl_results(values) -> dict:
    return {(*_BASE_KEY, ("twirl", idx)): value for idx, value in enumerate(values)}


def _two_cx_meta():
    """Two CX gates on disjoint qubit pairs, so each gate's twirl is visible."""
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.h(2)
    qc.cx(0, 1)
    qc.cx(2, 3)
    return meta_from_circuit(qc, observable=SparsePauliOp.from_list([("ZZZZ", 1.0)]))


def _structural_twirl_bodies(env, meta=None, n_twirls=20):
    pipeline = CircuitPipeline(
        stages=[
            DummySpecStage(meta=meta or _two_cx_meta()),
            PauliTwirlStage(n_twirls=n_twirls, seed=0),
            MeasurementStage(),
        ]
    )
    trace = pipeline.run_forward_pass("x", env)
    return stage_output(trace, "PauliTwirlStage").circuit_bodies


def _op_sequence(dag) -> tuple:
    return tuple(
        (node.op.name, tuple(dag.find_bit(q).index for q in node.qargs))
        for node in dag.topological_op_nodes()
    )


def _mixed_cx_cz_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cz(1, 2)
    qc.h(1)
    qc.cx(2, 0)
    return qc


def test_structural_twirls_preserve_the_unitary_and_vary(dummy_pipeline_env):
    qc = _mixed_cx_cz_circuit()
    meta = meta_from_circuit(qc, observable=SparsePauliOp.from_list([("ZZZ", 1.0)]))
    bodies = _structural_twirl_bodies(dummy_pipeline_env, meta=meta)

    original = Operator(qc)
    assert all(Operator(dag_to_circuit(dag)).equiv(original) for _, dag in bodies)
    assert len({_op_sequence(dag) for _, dag in bodies}) > 1


def test_structural_twirl_leaves_circuits_without_two_qubit_cliffords(
    dummy_pipeline_env,
):
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.rz(0.5, 1)
    qc.ry(0.3, 0)
    meta = meta_from_circuit(qc, observable=SparsePauliOp.from_list([("ZZ", 1.0)]))
    bodies = _structural_twirl_bodies(dummy_pipeline_env, meta=meta, n_twirls=3)

    original = _op_sequence(circuit_to_dag(qc))
    assert [_op_sequence(dag) for _, dag in bodies] == [original] * 3


def test_bodies_share_twirl_labels_for_each_twirl_index(dummy_pipeline_env):
    """Common random numbers: every body draws the same Pauli vector for a given
    ``twirl_idx``, so differences between bodies are not twirl noise."""
    dag = circuit_to_dag(_mixed_cx_cz_circuit())
    meta = MetaCircuit(
        circuit_bodies=(((("body", 0),), dag), ((("body", 1),), dag)),
        observable=SparsePauliOp.from_list([("ZZZ", 1.0)]),
    )
    sequences = {
        tag: _op_sequence(twirled)
        for tag, twirled in _structural_twirl_bodies(
            dummy_pipeline_env, meta=meta, n_twirls=5
        )
    }

    for twirl_idx in range(5):
        assert (
            sequences[(("body", 0), ("twirl", twirl_idx))]
            == sequences[(("body", 1), ("twirl", twirl_idx))]
        )
    assert len(set(sequences.values())) > 1


@pytest.mark.parametrize(
    "values, expected",
    [
        pytest.param([1.0, 2.0, 6.0], 3.0, id="scalar"),
        pytest.param([[1.0, 4.0], [2.0, 5.0], [6.0, 9.0]], [3.0, 6.0], id="list"),
        pytest.param(
            [{0: 1.0, 1: 4.0}, {0: 2.0, 1: 5.0}, {0: 6.0, 1: 9.0}],
            {0: 3.0, 1: 6.0},
            id="dict",
        ),
    ],
)
def test_reduce_averages_twirls_per_base_key(dummy_pipeline_env, values, expected):
    reduced = PauliTwirlStage(n_twirls=3).reduce(
        _twirl_results(values), dummy_pipeline_env, token=None
    )
    assert reduced == {_BASE_KEY: pytest.approx(expected)}


def test_structural_twirl_puts_real_paulis_on_every_two_qubit_gate(
    dummy_pipeline_env,
):
    bodies = _structural_twirl_bodies(dummy_pipeline_env)
    twirled_qubits = set()
    for _, dag in bodies:
        for node in dag.op_nodes():
            if node.op.name in ("x", "y", "z"):
                twirled_qubits.update(dag.find_bit(q).index for q in node.qargs)
    assert twirled_qubits == {0, 1, 2, 3}
    assert all(node.op.name != "id" for _, dag in bodies for node in dag.op_nodes())


@pytest.mark.parametrize(
    "meta, leading_stages, param_sets",
    [
        pytest.param(_two_cx_meta(), [], ((),), id="structural"),
        pytest.param(
            parametric_twirlable_meta(),
            [ParameterBindingStage()],
            np.array([[0.1, 0.2], [0.3, 0.4]]),
            id="fast",
        ),
    ],
)
def test_dry_expand_labels_match_real_expand(
    dummy_pipeline_env, meta, leading_stages, param_sets
):
    pipeline = CircuitPipeline(
        stages=[
            DummySpecStage(meta=meta),
            *leading_stages,
            PauliTwirlStage(n_twirls=3, seed=0),
            MeasurementStage(),
        ]
    )
    env = PipelineEnv(backend=dummy_pipeline_env.backend, param_sets=param_sets)
    real = pipeline.run_forward_pass("x", env)
    dry = pipeline.run_forward_pass("x", env, dry=True)
    assert stage_body_tags(dry, "PauliTwirlStage") == stage_body_tags(
        real, "PauliTwirlStage"
    )
