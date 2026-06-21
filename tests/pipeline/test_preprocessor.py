# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``CircuitPreprocessor`` and its factory functions."""

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import SparsePauliOp

from divi.circuits import MetaCircuit
from divi.pipeline import (
    CircuitPreprocessor,
    ResultFormat,
    cost_preprocessor,
    sample_preprocessor,
)
from divi.pipeline._preprocessor import _clear_observable, _identity
from divi.pipeline.stages import PreprocessStage


def test_cost_preprocessor_is_expval_identity():
    p = cost_preprocessor()
    assert p.name == "cost"
    assert p.result_format is ResultFormat.EXPVALS
    assert p.preprocess is _identity  # measures the seed's observable as-is
    assert p.terminal_stage is None  # program supplies its default MeasurementStage
    assert p.consumes_dag_bodies is False


def test_sample_preprocessor_clears_observable_as_probs():
    p = sample_preprocessor()
    assert p.name == "sample"
    assert p.result_format is ResultFormat.PROBS
    assert p.preprocess is _clear_observable
    assert p.terminal_stage is None
    assert p.consumes_dag_bodies is False


def test_protocol_is_hashable_and_value_equal():
    # Repeated factory calls compare and hash equal (and distinct routines must
    # not collide); the frozen dataclass stays hashable for use as a value.
    assert cost_preprocessor() == cost_preprocessor()
    assert hash(cost_preprocessor()) == hash(cost_preprocessor())
    assert cost_preprocessor() != sample_preprocessor()


def test_factory_preprocessors_declare_stable_cache_keys():
    # The pipeline cache keys on ``cache_key``, so the program-facing routines
    # carry stable, distinct keys while a bare preprocessor stays uncacheable
    # (its transform may carry per-call state, as the metric estimators' do).
    assert cost_preprocessor().cache_key == "cost"
    assert sample_preprocessor().cache_key == "sample"
    assert cost_preprocessor().cache_key != sample_preprocessor().cache_key
    assert CircuitPreprocessor("metric").cache_key is None


def test_preprocess_stage_delegates_dag_consumption_flag():
    assert PreprocessStage(CircuitPreprocessor("metadata")).consumes_dag_bodies is False
    assert (
        PreprocessStage(
            CircuitPreprocessor("body-transform", consumes_dag_bodies=True)
        ).consumes_dag_bodies
        is True
    )


def test_clear_observable_turns_expval_seed_into_all_wires_probs():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    meta = MetaCircuit(
        circuit_bodies=(((), circuit_to_dag(qc)),),
        observable=SparsePauliOp("ZZ"),
    )

    out = _clear_observable(meta)

    assert out.observable is None
    assert out.measured_wires == (0, 1)
    assert out.measurement_qasms == ()
    assert out.measurement_groups == ()
