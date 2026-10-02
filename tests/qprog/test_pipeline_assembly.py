# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the program pipeline assembler and the per-protocol pipelines."""

import warnings

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import RYGate
from qiskit.quantum_info import SparsePauliOp

from divi.circuits.quepp import QuEPP
from divi.circuits.zne import ZNE
from divi.pipeline import CircuitPreprocessor, DiviPerformanceWarning, ResultFormat
from divi.pipeline.stages import (
    CircuitSpecStage,
    MeasurementStage,
    PauliTwirlStage,
    QEMStage,
)
from divi.qprog import PCE, VQE, CustomVQA
from divi.qprog.algorithms import GenericLayerAnsatz, TimeEvolution
from divi.qprog.problems import BinaryOptimizationProblem, HamiltonianProblem


def _stage_types(pipeline):
    return [type(stage).__name__ for stage in pipeline.stages]


def _protocol_names(program):
    return [protocol.name for protocol in program._preprocessors()]


def _protocol_pipeline(program, protocol):
    return program._build_preprocessor_pipeline(protocol)


def _metric_pipeline(program):
    """The expval pipeline a natural-gradient estimator drives."""
    return program._build_preprocessor_pipeline(CircuitPreprocessor("metric"))


def _vqe_with(backend, optimizer, qem_protocol=None, **kwargs):
    return VQE(
        HamiltonianProblem(SparsePauliOp.from_list([("ZI", 0.5), ("IZ", 0.5)])),
        ansatz=GenericLayerAnsatz([RYGate]),
        n_layers=1,
        backend=backend,
        optimizer=optimizer,
        qem_protocol=qem_protocol,
        **kwargs,
    )


@pytest.fixture
def vqe(dummy_simulator, default_optimizer):
    return _vqe_with(dummy_simulator, default_optimizer)


@pytest.fixture
def mitigated_vqe(dummy_simulator, default_optimizer):
    return _vqe_with(dummy_simulator, default_optimizer, ZNE(scale_factors=[1.0, 3.0]))


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


def test_twirl_stage_draws_from_the_program_seed(dummy_simulator, default_optimizer):
    """An unseeded twirl stage redraws its labels on every call, so the same
    program produces different circuits each time it assembles a pipeline."""

    def twirled(seed):
        program = _vqe_with(
            dummy_simulator,
            default_optimizer,
            QuEPP(truncation_order=1, n_twirls=3),
            seed=seed,
        )
        (stage,) = _stages_of(
            _protocol_pipeline(program, program.cost_preprocessor()), PauliTwirlStage
        )
        return program._base_seed, stage._seed

    base_seed, stage_seed = twirled(1234)
    assert stage_seed == base_seed
    # Two programs given the same seed twirl identically; a different seed does not.
    assert twirled(1234)[1] == stage_seed
    assert twirled(4321)[1] != stage_seed


def _stages_of(pipeline, stage_type):
    return [stage for stage in pipeline.stages if isinstance(stage, stage_type)]


@pytest.mark.parametrize("n_twirls", [0, 1, 3])
def test_mitigation_stages_carry_the_programs_protocol(
    dummy_simulator, default_optimizer, n_twirls
):
    protocol = QuEPP(truncation_order=1, n_twirls=n_twirls)
    program = _vqe_with(dummy_simulator, default_optimizer, protocol)
    pipeline = _protocol_pipeline(program, program.cost_preprocessor())

    (qem_stage,) = _stages_of(pipeline, QEMStage)
    assert qem_stage.protocol is program._qem_protocol
    twirls = _stages_of(pipeline, PauliTwirlStage)
    assert [stage._n_twirls for stage in twirls] == ([n_twirls] if n_twirls else [])


def _exhaustive_quepp():
    return QuEPP(truncation_order=1, sampling="exhaustive", n_twirls=0)


def test_exhaustive_quepp_binds_parameters_before_mitigation(
    dummy_simulator, default_optimizer
):
    program = _vqe_with(
        dummy_simulator,
        default_optimizer,
        _exhaustive_quepp(),
        suppress_performance_warnings=True,
    )
    types = _stage_types(_protocol_pipeline(program, program.cost_preprocessor()))

    assert types.index("ParameterBindingStage") < types.index("QEMStage")
    assert types.count("ParameterBindingStage") == 1


def _time_evolution_with(backend, qem_protocol, **kwargs):
    return TimeEvolution(
        hamiltonian=SparsePauliOp.from_list([("X", 1.0), ("Z", 1.0)]),
        observable=SparsePauliOp("Z"),
        backend=backend,
        qem_protocol=qem_protocol,
        **kwargs,
    )


def _vqe_cost_pipeline(backend, optimizer, **kwargs):
    program = _vqe_with(backend, optimizer, _exhaustive_quepp(), **kwargs)
    return _protocol_pipeline(program, program.cost_preprocessor())


def _time_evolution_pipeline(backend, optimizer, **kwargs):
    program = _time_evolution_with(backend, _exhaustive_quepp(), **kwargs)
    return _protocol_pipeline(program, program._evolution_preprocessor())


@pytest.mark.parametrize(
    "build", [_vqe_cost_pipeline, _time_evolution_pipeline], ids=["vqa", "base"]
)
def test_exhaustive_quepp_warns_unless_suppressed(
    dummy_simulator, default_optimizer, build
):
    """Both the variational assembler and the shared one honour the program's
    ``suppress_performance_warnings``."""
    with pytest.warns(DiviPerformanceWarning) as record:
        build(dummy_simulator, default_optimizer)
    assert any("sampling='exhaustive'" in str(w.message) for w in record)

    with warnings.catch_warnings():
        warnings.simplefilter("error", DiviPerformanceWarning)
        build(dummy_simulator, default_optimizer, suppress_performance_warnings=True)


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
