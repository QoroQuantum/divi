# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the program pipeline assembler and the per-protocol pipelines."""

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import RYGate
from qiskit.quantum_info import SparsePauliOp

from divi.circuits.zne import ZNE
from divi.pipeline import CircuitPreprocessor, ResultFormat
from divi.pipeline.stages import CircuitSpecStage, MeasurementStage
from divi.qprog import PCE, VQE, CustomVQA
from divi.qprog.algorithms import GenericLayerAnsatz
from divi.qprog.problems import BinaryOptimizationProblem


def _stage_types(pipeline):
    return [type(stage).__name__ for stage in pipeline.stages]


def _protocol_names(program):
    return [protocol.name for protocol in program._preprocessors()]


def _protocol_pipeline(program, protocol):
    return program._build_preprocessor_pipeline(protocol)


def _metric_pipeline(program):
    """The expval pipeline a natural-gradient estimator drives."""
    return program._build_preprocessor_pipeline(CircuitPreprocessor("metric"))


@pytest.fixture
def vqe(dummy_simulator, default_optimizer):
    return VQE(
        hamiltonian=SparsePauliOp.from_list([("ZI", 0.5), ("IZ", 0.5)]),
        ansatz=GenericLayerAnsatz([RYGate]),
        n_layers=1,
        backend=dummy_simulator,
        optimizer=default_optimizer,
    )


@pytest.fixture
def mitigated_vqe(dummy_simulator, default_optimizer):
    return VQE(
        hamiltonian=SparsePauliOp.from_list([("ZI", 0.5), ("IZ", 0.5)]),
        ansatz=GenericLayerAnsatz([RYGate]),
        n_layers=1,
        backend=dummy_simulator,
        optimizer=default_optimizer,
        qem_protocol=ZNE(scale_factors=[1.0, 3.0]),
    )


def test_vqe_exposes_cost_and_sample(vqe):
    # The metric routine is not exposed — it is driven on demand by the
    # natural-gradient estimator, not enumerated for introspection.
    assert _protocol_names(vqe) == ["cost", "sample"]
    # Default protocol is NoMitigation, so the assembler omits QEM everywhere.
    for protocol in vqe._preprocessors():
        types = _stage_types(_protocol_pipeline(vqe, protocol))
        assert "QEMStage" not in types
        assert "MeasurementStage" in types
        assert "ParameterBindingStage" in types


def test_metric_pipeline_is_a_bound_expval_measurement(vqe):
    types = _stage_types(_metric_pipeline(vqe))
    assert "QEMStage" not in types  # default NoMitigation
    assert "MeasurementStage" in types
    assert "ParameterBindingStage" in types


def test_mitigated_vqe_rides_qem_on_expval_pipelines_only(mitigated_vqe):
    # ZNE applies to expectation values, so it rides cost and the metric measurement...
    assert "QEMStage" in _stage_types(
        _protocol_pipeline(mitigated_vqe, mitigated_vqe.cost_preprocessor())
    )
    assert "QEMStage" in _stage_types(_metric_pipeline(mitigated_vqe))
    # ...but not the probability-sampling pipeline.
    assert "QEMStage" not in _stage_types(
        _protocol_pipeline(mitigated_vqe, mitigated_vqe._sample_preprocessor())
    )


def test_assembled_stage_order(mitigated_vqe):
    """spec → QEM → terminal (measurement) → parameter binding."""
    types = _stage_types(
        _protocol_pipeline(mitigated_vqe, mitigated_vqe.cost_preprocessor())
    )
    assert (
        types.index("QEMStage")
        < types.index("MeasurementStage")
        < types.index("ParameterBindingStage")
    )


def test_assemble_pipeline_qem_inclusion_is_protocol_driven(mitigated_vqe):
    """The QEM protocol decides applicability per result format — not the recipe."""
    expval = mitigated_vqe._assemble_pipeline(
        CircuitSpecStage(), MeasurementStage(), result_format=ResultFormat.EXPVALS
    )
    probs = mitigated_vqe._assemble_pipeline(
        CircuitSpecStage(), MeasurementStage(), result_format=ResultFormat.PROBS
    )
    assert "QEMStage" in _stage_types(expval)
    assert "QEMStage" not in _stage_types(probs)
    # Variational assembly always binds the trainable parameters.
    assert "ParameterBindingStage" in _stage_types(expval)
    assert "ParameterBindingStage" in _stage_types(probs)


def test_custom_vqa_has_no_sample_pipeline(dummy_simulator, default_optimizer):
    weight = Parameter("w")
    qc = QuantumCircuit(1, 1)
    qc.ry(weight, 0)
    qc.measure(0, 0)
    program = CustomVQA(
        qscript=qc, backend=dummy_simulator, optimizer=default_optimizer
    )
    # No bitstring extraction, and the metric pipeline is built on demand.
    assert "sample" not in _protocol_names(program)
    assert "MeasurementStage" in _stage_types(_metric_pipeline(program))


def test_pce_cost_uses_pce_cost_stage_without_mitigation(
    dummy_simulator, default_optimizer
):
    pce = PCE(
        problem=BinaryOptimizationProblem(np.array([[1.0, 0.2], [0.2, 2.0]])),
        ansatz=GenericLayerAnsatz([RYGate]),
        n_layers=1,
        backend=dummy_simulator,
        optimizer=default_optimizer,
    )
    cost_types = _stage_types(_protocol_pipeline(pce, pce.cost_preprocessor()))
    assert "PCECostStage" in cost_types
    assert "QEMStage" not in cost_types  # COUNTS is outside the QEM protocol's remit
    # The metric measures plain expectation values, not PCE's COUNTS objective.
    metric_types = _stage_types(_metric_pipeline(pce))
    assert "MeasurementStage" in metric_types
    assert "PCECostStage" not in metric_types
