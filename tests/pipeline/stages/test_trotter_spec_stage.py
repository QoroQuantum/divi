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
from divi.pipeline.transformations import reduce_merge_histograms


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


class TestReduceMergeHistograms:
    """Tests for reduce_merge_histograms: probability histogram averaging across ham samples."""

    def test_merges_two_histograms(self):
        """Averages probability dicts across two Hamiltonian samples."""
        grouped = {(("circ", 0),): [{"00": 0.8, "11": 0.2}, {"00": 0.6, "11": 0.4}]}

        result = reduce_merge_histograms(grouped)

        assert (("circ", 0),) in result
        assert result[(("circ", 0),)]["00"] == pytest.approx(0.7)
        assert result[(("circ", 0),)]["11"] == pytest.approx(0.3)

    def test_merged_probabilities_sum_to_one(self):
        """Merged probability distribution sums to 1.0."""
        grouped = {
            (("circ", 0),): [
                {"00": 0.5, "01": 0.3, "10": 0.15, "11": 0.05},
                {"00": 0.2, "01": 0.4, "10": 0.1, "11": 0.3},
                {"00": 0.3, "01": 0.3, "10": 0.2, "11": 0.2},
            ]
        }

        result = reduce_merge_histograms(grouped)
        prob_dict = result[(("circ", 0),)]
        total = sum(prob_dict.values())
        assert total == pytest.approx(1.0)

    def test_handles_disjoint_bitstrings(self):
        """Averaging when histograms have different bitstring keys uses 0 for missing."""
        grouped = {
            (("circ", 0),): [
                {"00": 0.8, "11": 0.2},
                {"01": 0.6, "10": 0.4},
            ]
        }

        result = reduce_merge_histograms(grouped)
        prob_dict = result[(("circ", 0),)]
        assert prob_dict["00"] == pytest.approx(0.4)
        assert prob_dict["11"] == pytest.approx(0.1)
        assert prob_dict["01"] == pytest.approx(0.3)
        assert prob_dict["10"] == pytest.approx(0.2)

    def test_empty_prob_dicts(self):
        """Empty list of prob dicts returns empty dict."""
        grouped = {(("circ", 0),): []}

        result = reduce_merge_histograms(grouped)
        assert result[(("circ", 0),)] == {}

    def test_single_histogram_identity(self):
        """Single histogram merges to itself."""
        grouped = {(("circ", 0),): [{"00": 0.7, "11": 0.3}]}

        result = reduce_merge_histograms(grouped)
        assert result[(("circ", 0),)]["00"] == pytest.approx(0.7)
        assert result[(("circ", 0),)]["11"] == pytest.approx(0.3)

    def test_multiple_base_keys(self):
        """Multiple base keys are each merged independently."""
        grouped = {
            (("circ", 0),): [{"00": 0.8, "11": 0.2}, {"00": 0.6, "11": 0.4}],
            (("circ", 1),): [{"01": 1.0}, {"01": 0.5, "10": 0.5}],
        }

        result = reduce_merge_histograms(grouped)

        assert result[(("circ", 0),)]["00"] == pytest.approx(0.7)
        assert result[(("circ", 0),)]["11"] == pytest.approx(0.3)
        assert result[(("circ", 1),)]["01"] == pytest.approx(0.75)
        assert result[(("circ", 1),)]["10"] == pytest.approx(0.25)
