# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the shot-allocation plumbing between MeasurementStage,
_default_execute_fn, and CircuitRunner backends.

Covers:
- ``_build_shot_groups`` translating per-spec/per-group dicts into the
  contiguous ``[start, end, shots]`` payload format.
- The full pipeline forwarding ``shot_groups`` to the backend.
"""

from collections.abc import Mapping
from typing import Any

import pennylane as qp
import pytest
from qiskit_aer import AerSimulator

from divi.backends import CircuitRunner, ExecutionResult
from divi.backends._qiskit_simulator import QiskitSimulator
from divi.circuits import MetaCircuit
from divi.circuits._conversions import qscript_to_meta
from divi.pipeline import CircuitPipeline, PipelineEnv
from divi.pipeline._core import _build_shot_groups
from divi.pipeline.stages import MeasurementStage

from .helpers import DummySpecStage


def _three_group_meta() -> MetaCircuit:
    obs = 10.0 * qp.Z(0) + 1.0 * qp.X(0) + 0.1 * qp.Y(0)
    qscript = qp.tape.QuantumScript(ops=[qp.Hadamard(0)], measurements=[qp.expval(obs)])
    return qscript_to_meta(qscript)


class _RecordingBackend(CircuitRunner):
    """Captures the kwargs that ``_default_execute_fn`` passes to submit_circuits."""

    def __init__(self, shots: int = 1000) -> None:
        super().__init__(shots=shots)
        self.last_circuits: dict[str, str] | None = None
        self.last_kwargs: dict[str, Any] = {}

    @property
    def is_async(self) -> bool:
        return False

    @property
    def supports_expval(self) -> bool:
        return False

    def submit_circuits(
        self, circuits: Mapping[str, str], **kwargs: Any
    ) -> ExecutionResult:
        self.last_circuits = dict(circuits)
        self.last_kwargs = dict(kwargs)
        # Return a single-bitstring result for each circuit so the postprocessing
        # (counts -> expval -> weighted sum) does not fail.
        results = [
            {"label": label, "results": {"0": kwargs.get("shots_for_label", 100)}}
            for label in circuits
        ]
        return ExecutionResult(results=results)


# --------------------------------------------------------------------------- #
# Pure-function tests for _build_shot_groups
# --------------------------------------------------------------------------- #


class TestBuildShotGroupsPure:
    """Spec: _build_shot_groups maps lineage + per-spec shot dicts to ranges."""

    def test_returns_none_when_no_circuits_match(self):
        circuits = {"a": "qasm", "b": "qasm"}
        lineage = {
            "a": (("circuit", 0), ("obs_group", 0)),
            "b": (("circuit", 0), ("obs_group", 1)),
        }
        # per_group_shots has a different spec key -> nothing matches.
        per_group = {(("other", 0),): {0: 100, 1: 200}}
        assert _build_shot_groups(circuits, lineage, per_group) is None

    def test_single_spec_consecutive_groups_collapsed(self):
        circuits = {"a": "x", "b": "x", "c": "x"}
        lineage = {
            "a": (("circuit", 0), ("obs_group", 0)),
            "b": (("circuit", 0), ("obs_group", 1)),
            "c": (("circuit", 0), ("obs_group", 2)),
        }
        per_group = {(("circuit", 0),): {0: 50, 1: 50, 2: 200}}
        assert _build_shot_groups(circuits, lineage, per_group) == [
            [0, 2, 50],
            [2, 3, 200],
        ]

    def test_distinct_shots_create_separate_ranges(self):
        circuits = {"a": "x", "b": "x", "c": "x"}
        lineage = {
            "a": (("circuit", 0), ("obs_group", 0)),
            "b": (("circuit", 0), ("obs_group", 1)),
            "c": (("circuit", 0), ("obs_group", 2)),
        }
        per_group = {(("circuit", 0),): {0: 100, 1: 200, 2: 300}}
        assert _build_shot_groups(circuits, lineage, per_group) == [
            [0, 1, 100],
            [1, 2, 200],
            [2, 3, 300],
        ]

    def test_two_specs_independent_allocations(self):
        circuits = {"a": "x", "b": "x", "c": "x", "d": "x"}
        lineage = {
            "a": (("circuit", 0), ("obs_group", 0)),
            "b": (("circuit", 0), ("obs_group", 1)),
            "c": (("circuit", 1), ("obs_group", 0)),
            "d": (("circuit", 1), ("obs_group", 1)),
        }
        per_group = {
            (("circuit", 0),): {0: 100, 1: 200},
            (("circuit", 1),): {0: 300, 1: 400},
        }
        assert _build_shot_groups(circuits, lineage, per_group) == [
            [0, 1, 100],
            [1, 2, 200],
            [2, 3, 300],
            [3, 4, 400],
        ]

    def test_missing_obs_group_for_a_circuit_raises(self):
        circuits = {"a": "x", "b": "x"}
        lineage = {
            "a": (("circuit", 0), ("obs_group", 0)),
            "b": (("circuit", 0), ("obs_group", 1)),
        }
        # per_group only has shots for obs_group 0; obs_group 1 missing.
        per_group = {(("circuit", 0),): {0: 100}}
        with pytest.raises(ValueError, match="no per-group shot allocation"):
            _build_shot_groups(circuits, lineage, per_group)

    def test_extra_axes_in_branch_key_dont_break_matching(self):
        """Branch keys with QEM/param_set axes still match a spec key prefix."""
        circuits = {"a": "x", "b": "x"}
        lineage = {
            "a": (("circuit", 0), ("param_set", 5), ("obs_group", 0)),
            "b": (("circuit", 0), ("param_set", 5), ("obs_group", 1)),
        }
        per_group = {(("circuit", 0),): {0: 100, 1: 200}}
        assert _build_shot_groups(circuits, lineage, per_group) == [
            [0, 1, 100],
            [1, 2, 200],
        ]


# --------------------------------------------------------------------------- #
# End-to-end pipeline -> backend wiring
# --------------------------------------------------------------------------- #


class TestExecuteFnForwardsShotGroups:
    """Spec: when MeasurementStage produces per_group_shots, the execute fn
    forwards a correctly-formed shot_groups list to the backend."""

    def _build_pipeline_with_distribution(
        self, backend: CircuitRunner, distribution
    ) -> tuple[CircuitPipeline, PipelineEnv]:
        env = PipelineEnv(backend=backend)
        pipeline = CircuitPipeline(
            stages=[
                DummySpecStage(meta=_three_group_meta()),
                MeasurementStage(shot_distribution=distribution),
            ],
        )
        return pipeline, env

    def test_uniform_forwards_evenly_split_shot_groups(self):
        backend = _RecordingBackend(shots=300)
        pipeline, env = self._build_pipeline_with_distribution(backend, "uniform")
        pipeline.run(initial_spec="ignored", env=env)

        assert "shot_groups" in backend.last_kwargs
        # Three groups, all 100 shots -> collapses to one range [0, 3, 100].
        assert backend.last_kwargs["shot_groups"] == [[0, 3, 100]]

    def test_weighted_produces_distinct_ranges(self):
        backend = _RecordingBackend(shots=1000)
        pipeline, env = self._build_pipeline_with_distribution(backend, "weighted")
        pipeline.run(initial_spec="ignored", env=env)

        groups = backend.last_kwargs["shot_groups"]
        assert sum(end - start for start, end, _ in groups) == 3
        total_shots = sum((end - start) * shots for start, end, shots in groups)
        assert total_shots == 1000
        # The dominant group (norm 10) should get a strictly larger allocation
        # than the smallest group (norm 0.1).
        per_circuit = []
        for start, end, shots in groups:
            per_circuit.extend([shots] * (end - start))
        assert per_circuit[0] > per_circuit[2]

    def test_dropped_groups_not_submitted(self):
        backend = _RecordingBackend(shots=11)  # weighted 10:1:0.1 -> 10:1:0
        pipeline, env = self._build_pipeline_with_distribution(backend, "weighted")
        with pytest.warns(UserWarning, match="zero shots"):
            pipeline.run(initial_spec="ignored", env=env)

        # Only 2 circuits should have been submitted.
        assert len(backend.last_circuits) == 2
        groups = backend.last_kwargs["shot_groups"]
        assert sum(end - start for start, end, _ in groups) == 2

    def test_no_distribution_means_no_shot_groups_in_kwargs(self):
        backend = _RecordingBackend(shots=300)
        env = PipelineEnv(backend=backend)
        pipeline = CircuitPipeline(
            stages=[
                DummySpecStage(meta=_three_group_meta()),
                MeasurementStage(),  # no shot_distribution
            ],
        )
        pipeline.run(initial_spec="ignored", env=env)
        assert "shot_groups" not in backend.last_kwargs


# --------------------------------------------------------------------------- #
# QiskitSimulator e2e: shot_groups actually drive per-range execution shots
# --------------------------------------------------------------------------- #


_IDENTITY_QASM = (
    "OPENQASM 2.0;\n"
    'include "qelib1.inc";\n'
    "qreg q[1];\n"
    "creg c[1];\n"
    "measure q[0] -> c[0];\n"
)


class TestQiskitSimulatorShotGroups:
    """Spec: QiskitSimulator runs each [start, end, shots] range with that
    shot count, and the returned counts reflect the per-range budget."""

    def test_returned_counts_sum_matches_per_group_shots(self):
        """Each circuit's total counts equals the shot count assigned to its range."""
        sim = QiskitSimulator(shots=100)
        circuits = {"c0": _IDENTITY_QASM, "c1": _IDENTITY_QASM, "c2": _IDENTITY_QASM}
        result = sim.submit_circuits(circuits, shot_groups=[[0, 1, 50], [1, 3, 200]])
        per_circuit_totals = [sum(r["results"].values()) for r in result.results]
        assert per_circuit_totals == [50, 200, 200]

    def test_full_pipeline_e2e_with_qiskit_simulator(self):
        """End-to-end: pipeline + MeasurementStage(shot_distribution) + QiskitSimulator.

        Use ``grouping_strategy="wires"`` to bypass the auto-fallback to
        ``_backend_expval`` (QiskitSimulator's expval-native mode).
        """
        sim = QiskitSimulator(shots=900)
        env = PipelineEnv(backend=sim)
        pipeline = CircuitPipeline(
            stages=[
                DummySpecStage(meta=_three_group_meta()),
                MeasurementStage(
                    grouping_strategy="wires", shot_distribution="weighted"
                ),
            ],
        )
        result = pipeline.run(initial_spec="ignored", env=env)
        assert len(result) == 1
        energy = list(result.values())[0]
        # Energy is a real, finite scalar within the range of |c_i| sums (= 11.1).
        assert isinstance(energy, float)
        assert -11.2 <= energy <= 11.2


class TestQiskitSimulatorShotGroupsValidation:
    """Implementation detail: shot_groups must cover every circuit."""

    def test_partial_coverage_raises(self):
        sim = QiskitSimulator(shots=100)
        circuits = {"c0": _IDENTITY_QASM, "c1": _IDENTITY_QASM, "c2": _IDENTITY_QASM}
        # Only first 2 circuits covered.
        with pytest.raises(ValueError, match="do not cover every circuit"):
            sim.submit_circuits(circuits, shot_groups=[[0, 2, 100]])


class TestQiskitSimulatorShotGroupsBatching:
    """Spec: one Aer run per distinct shot count, regardless of how many
    contiguous ranges are involved. Preserves Aer's batched parallelism."""

    def test_single_aer_run_when_all_shots_equal(self, mocker):
        """All circuits get the same shot count -> exactly one aer run."""
        sim = QiskitSimulator(shots=100)
        circuits = {f"c{i}": _IDENTITY_QASM for i in range(4)}
        # 4 ranges all with shots=50 -> still one aer call.
        spy = mocker.spy(AerSimulator, "run")
        sim.submit_circuits(
            circuits,
            shot_groups=[[0, 1, 50], [1, 2, 50], [2, 3, 50], [3, 4, 50]],
        )
        assert spy.call_count == 1

    def test_distinct_shot_counts_get_distinct_runs(self, mocker):
        sim = QiskitSimulator(shots=100)
        circuits = {f"c{i}": _IDENTITY_QASM for i in range(3)}
        spy = mocker.spy(AerSimulator, "run")
        sim.submit_circuits(
            circuits,
            shot_groups=[[0, 1, 50], [1, 2, 100], [2, 3, 50]],
        )
        # 50 and 100 -> two runs, even though 50 appears in two non-contiguous ranges.
        assert spy.call_count == 2

    def test_results_returned_in_original_circuit_order(self):
        """When non-contiguous ranges are batched together, results must
        still be re-ordered back to the original circuit positions."""
        sim = QiskitSimulator(shots=100)
        circuits = {f"c{i}": _IDENTITY_QASM for i in range(4)}
        # Interleave shot counts across non-contiguous ranges.
        result = sim.submit_circuits(
            circuits,
            shot_groups=[[0, 1, 50], [1, 2, 200], [2, 3, 50], [3, 4, 200]],
        )
        labels = [r["label"] for r in result.results]
        per_circuit_totals = [sum(r["results"].values()) for r in result.results]
        assert labels == ["c0", "c1", "c2", "c3"]
        assert per_circuit_totals == [50, 200, 50, 200]
