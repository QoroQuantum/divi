# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for divi.pipeline.stages._trotter_spec_stage."""

import numpy as np
import pennylane as qml
import pytest

from divi.circuits import MetaCircuit
from divi.hamiltonians import ExactTrotterization
from divi.pipeline import CircuitPipeline, PipelineEnv, PipelineTrace
from divi.pipeline._compilation import _compile_batch
from divi.pipeline.abc import ChildResults
from divi.pipeline.stages import MeasurementStage, TrotterSpecStage


class TestTrotterSpecStage:
    """Spec: TrotterSpecStage expand turns Hamiltonian into keyed MetaCircuit batch; reduce averages over ham axis."""

    def test_reduce_averages_sample_results(self, dummy_expval_backend):
        class _DummyTrotterizationStrategy:
            n_hamiltonians_per_iteration = 3
            stateful = False

            def process_hamiltonian(self, hamiltonian):
                return hamiltonian

        def _meta_factory(hamiltonian, ham_id):
            qscript = qml.tape.QuantumScript(
                ops=[qml.Hadamard(0)], measurements=[qml.expval(hamiltonian)]
            )
            return MetaCircuit(
                source_circuit=qscript, symbols=np.array([], dtype=object)
            )

        env = PipelineEnv(backend=dummy_expval_backend)

        pipeline = CircuitPipeline(
            stages=[
                TrotterSpecStage(
                    _DummyTrotterizationStrategy(), meta_circuit_factory=_meta_factory
                ),
                MeasurementStage(),
            ]
        )

        def _execute_fn(trace: PipelineTrace, env: PipelineEnv) -> ChildResults:
            _, lineage_by_label = _compile_batch(trace.final_batch)
            branch_keys = sorted(lineage_by_label.values(), key=str)
            return {bk: 1.0 + i for i, bk in enumerate(branch_keys)}

        reduced = pipeline.run(initial_spec=qml.Z(0), env=env, execute_fn=_execute_fn)
        assert len(reduced) >= 1
        assert any(v == pytest.approx(2.0) for v in reduced.values())

    def test_raises_empty_hamiltonian(self, dummy_expval_backend):
        def _meta_factory(hamiltonian, ham_id):
            qscript = qml.tape.QuantumScript(
                ops=[qml.Hadamard(0)], measurements=[qml.expval(hamiltonian)]
            )
            return MetaCircuit(
                source_circuit=qscript, symbols=np.array([], dtype=object)
            )

        env = PipelineEnv(backend=dummy_expval_backend)
        pipeline = CircuitPipeline(
            stages=[
                TrotterSpecStage(
                    ExactTrotterization(), meta_circuit_factory=_meta_factory
                ),
                MeasurementStage(),
            ]
        )
        with pytest.raises(ValueError, match="only constant terms|empty"):
            pipeline.run_forward_pass(initial_spec=qml.I(0), env=env)
