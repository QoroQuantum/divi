# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for the multi-round ``ProgramEnsemble`` workflow loop.

Covers the ``initial_state`` → ``create_programs`` → dispatch → ``update_state``
→ ``is_complete`` loop driven by :meth:`ProgramEnsemble.run`, and the
:class:`ReportingLevel` display behavior layered on top of it. Single-dispatch
mechanics live in ``test_ensemble_dispatch.py``.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Any

import pytest

import divi.qprog.ensemble as ensemble_module
from divi.qprog._ensemble_checkpoint import (
    ProgramRoundRecord,
    RoundCheckpoint,
    _encode_program_id,
)
from divi.qprog._program_checkpoint import ProgramCheckpoint
from divi.qprog.checkpointing import PROGRAM_COMPLETION_FILE, CheckpointConfig
from divi.qprog.ensemble import (
    BatchConfig,
    BatchMode,
    ProgramEnsemble,
    ReportingLevel,
    WorkflowStatus,
)
from divi.qprog.quantum_program import QuantumProgram
from divi.qprog.variational_quantum_algorithm import VariationalQuantumAlgorithm
from divi.reporting._events import (
    EventKind,
    ProgressEvent,
    ProgressScope,
    TerminalStatus,
)
from divi.reporting._logging import log_progress_event
from divi.reporting._session import ProgressSession
from divi.reporting._state import ProgressState
from tests.qprog._helpers import (
    FailingTestProgram,
    SimpleTestProgram,
    _RecordingSession,
)

# Each round of _LifecycleEnsemble contributes these totals via
# SimpleTestProgram(10, 5.5) + SimpleTestProgram(5, 10.0).
_CIRCUITS_PER_ROUND = 15
_RUNTIME_PER_ROUND = 15.5


class _TerminalTestCheckpoint(ProgramCheckpoint):
    value: int


class _TerminalTestProgram(QuantumProgram):
    def __init__(self, slot, tracker, completed, *, fail, backend):
        super().__init__(backend=backend)
        self.slot = slot
        self.tracker = tracker
        self.completed = completed
        self.fail = fail
        self.value = None

    def run(self):
        self.tracker[self.slot] += 1
        if self.fail:
            assert self.completed.wait(timeout=2)
            raise RuntimeError("planned child failure")
        self.value = self.slot + 10
        self._total_circuit_count = self.slot + 1
        self._total_run_time = self.slot + 0.5
        self.completed.set()
        return self

    def has_results(self):
        return self.value is not None

    def _make_checkpoint(self, checkpoint_dir: Path):
        return _TerminalTestCheckpoint(
            program_type=type(self).__name__,
            total_circuit_count=self.total_circuit_count,
            total_run_time=self.total_run_time,
            value=self.value,
        )

    def _restore_checkpoint(self, checkpoint_json: str, checkpoint_dir: Path) -> bool:
        checkpoint = _TerminalTestCheckpoint.model_validate_json(checkpoint_json)
        self.value = checkpoint.value
        self.restored_from = checkpoint_dir
        return True


class _TerminalTestEnsemble(ProgramEnsemble):
    def __init__(self, backend, tracker, *, fail_second):
        super().__init__(backend=backend)
        self.tracker = tracker
        self.fail_second = fail_second
        self.completed = Event()

    def initial_state(self):
        return 0

    def create_programs(self, state=None):
        super().create_programs()
        self.programs = {
            "first": _TerminalTestProgram(
                0, self.tracker, self.completed, fail=False, backend=self.backend
            ),
            "second": _TerminalTestProgram(
                1,
                self.tracker,
                self.completed,
                fail=self.fail_second,
                backend=self.backend,
            ),
        }

    def update_state(self, state):
        return state + 1

    def is_complete(self, state):
        return state >= 1

    def aggregate_results(self):
        return self.workflow_state

    def _save_workflow_checkpoint_state(self, state, round_dir, stem):
        return {"value": state}

    def _load_workflow_checkpoint_state(self, payload, round_dir, stem):
        return payload["value"]


class _LifecycleEnsemble(ProgramEnsemble):
    """Records every lifecycle hook call so ordering can be asserted.

    The workflow state is a round counter; ``n_rounds`` sets convergence and
    ``fail_on_round`` makes that round's programs raise. ``programs_per_round``
    accepts a callable of the state to vary the count across rounds.
    """

    def __init__(
        self,
        backend,
        *,
        n_rounds: int = 2,
        programs_per_round=2,
        fail_on_round: int | None = None,
        **kwargs,
    ):
        super().__init__(backend=backend, **kwargs)
        self.max_iterations = 1
        self._n_rounds = n_rounds
        self._programs_per_round = programs_per_round
        self._fail_on_round = fail_on_round
        self.calls: list[str] = []
        self.states_seen: list[int] = []
        self.program_ids_per_round: list[list[str]] = []

    def _n_programs_for(self, state: int) -> int:
        if callable(self._programs_per_round):
            return self._programs_per_round(state)
        return self._programs_per_round

    def initial_state(self):
        self.calls.append("initial_state")
        return 0

    def create_programs(self, state=None):
        self.calls.append(f"create_programs({state})")
        super().create_programs()
        self.states_seen.append(state)
        round_number = state + 1
        failing = round_number == self._fail_on_round
        programs = {}
        for idx in range(self._n_programs_for(state)):
            prog_id = f"r{round_number}p{idx}"
            programs[prog_id] = (
                FailingTestProgram(backend=self.backend)
                if failing
                else SimpleTestProgram(
                    10 if idx == 0 else 5,
                    5.5 if idx == 0 else 10.0,
                    backend=self.backend,
                )
            )
        self.programs = programs
        self.program_ids_per_round.append(sorted(programs))

    def update_state(self, state):
        self.calls.append(f"update_state({state})")
        return state + 1

    def is_complete(self, state):
        self.calls.append(f"is_complete({state})")
        return state >= self._n_rounds

    def aggregate_results(self):
        return self.workflow_state

    def _save_workflow_checkpoint_state(self, state, round_dir, stem):
        return {"value": state}

    def _load_workflow_checkpoint_state(self, payload, round_dir, stem):
        return payload["value"]


class _ArtifactEnsemble(_LifecycleEnsemble):
    """Persists its state as a stem-named file inside the round directory."""

    def _save_workflow_checkpoint_state(self, state, round_dir, stem):
        self.calls.append(f"save({stem})")
        artifact = f"{stem}.json"
        (round_dir / artifact).write_text(json.dumps(state))
        return {"artifact": artifact}

    def _load_workflow_checkpoint_state(self, payload, round_dir, stem):
        if payload["artifact"] != f"{stem}.json":
            raise ValueError(f"Expected the {stem} artifact.")
        return json.loads((round_dir / payload["artifact"]).read_text())


class _StatefulLoadEnsemble(_ArtifactEnsemble):
    """Restores side state from whichever snapshot it loads, as LASSQD's loader
    does with its generators; ``fail_on_stem`` makes that load raise."""

    def __init__(self, backend, *, fail_on_stem=None, **kwargs):
        super().__init__(backend, **kwargs)
        self.fail_on_stem = fail_on_stem
        self.restored_from = None

    def _load_workflow_checkpoint_state(self, payload, round_dir, stem):
        if stem == self.fail_on_stem:
            raise ValueError(f"{stem} boom")
        self.restored_from = stem
        return super()._load_workflow_checkpoint_state(payload, round_dir, stem)


class _MixedCheckpointEnsemble(_TerminalTestEnsemble):
    """Puts a program without checkpoint support ahead of checkpointing ones."""

    def create_programs(self, state=None):
        super().create_programs(state)
        self.programs = {
            "plain": SimpleTestProgram(1, 0.5, backend=self.backend),
            **self.programs,
        }


def _fail_reduction_of_round(ensemble, round_number):
    """Make ``update_state`` raise when reducing ``round_number``."""
    normal_update = ensemble.update_state

    def fail_reduction(state):
        if state == round_number - 1:
            raise RuntimeError("reduction boom")
        return normal_update(state)

    ensemble.update_state = fail_reduction


def _interrupt_round_two(ensemble, checkpoint_dir):
    """Run ``ensemble`` until its second round fails, checkpointing as it goes."""
    _fail_reduction_of_round(ensemble, 2)
    with pytest.raises(RuntimeError, match="reduction boom"):
        ensemble.run(checkpoint_config=CheckpointConfig(checkpoint_dir=checkpoint_dir))


def _rename_programs(ensemble):
    return {f"other_{pid}": program for pid, program in ensemble.programs.items()}


def _retype_programs(ensemble):
    return {
        pid: FailingTestProgram(backend=ensemble.backend) for pid in ensemble.programs
    }


def _edit_child_round_start(checkpoint_dir, **fields):
    """Overwrite the first child's round-start accounting fields."""
    path = checkpoint_dir / "round_001" / "round_start.json"
    checkpoint = json.loads(path.read_text())
    checkpoint["programs"][0].update(fields)
    path.write_text(json.dumps(checkpoint))


class _OneShotEnsemble(ProgramEnsemble):
    """Overrides only ``create_programs``, leaving every other hook default."""

    def create_programs(self, state=None):
        super().create_programs()
        self.programs = {
            "prog1": SimpleTestProgram(10, 5.5, backend=self.backend),
            "prog2": SimpleTestProgram(5, 10.0, backend=self.backend),
        }

    def aggregate_results(self):
        return None


class _FailingMaterialisationEnsemble(_LifecycleEnsemble):
    """Raises from ``create_programs`` on ``fail_on_round``, before building any."""

    def create_programs(self, state=None):
        if state + 1 == self._fail_on_round:
            raise ValueError("materialisation boom")
        super().create_programs(state)


@pytest.fixture
def lifecycle_ensemble(dummy_simulator):
    """Factory for ``_LifecycleEnsemble``; progress output is off by default."""

    def _make(cls=_LifecycleEnsemble, **kwargs):
        kwargs.setdefault("reporting_level", ReportingLevel.OFF)
        ensemble = cls(backend=dummy_simulator, **kwargs)
        made.append(ensemble)
        return ensemble

    made: list[_LifecycleEnsemble] = []
    yield _make
    for ensemble in made:
        try:
            ensemble.reset()
        except Exception:
            pass  # Don't break teardown on a race condition


def _interrupt_first_dispatch(mocker):
    """Raise KeyboardInterrupt from the first ``as_completed`` call only.

    Interrupting every call would escape via ``_stop_remaining_programs``,
    which calls ``as_completed`` again outside ``join()``'s try block.
    """
    real_as_completed = ensemble_module.as_completed
    calls = {"n": 0}

    def _side_effect(futures, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise KeyboardInterrupt
        return real_as_completed(futures, *args, **kwargs)

    mocker.patch("divi.qprog.ensemble.as_completed", side_effect=_side_effect)


def _interrupt_dispatch_then_cleanup(mocker):
    """Interrupt the dispatch, then interrupt again during its cleanup.

    The second ``as_completed`` call is the one inside
    ``_stop_remaining_programs``, which runs from ``join()``'s
    ``except KeyboardInterrupt`` handler and so cannot be caught by it.
    """
    calls = {"n": 0}

    def _side_effect(futures, *args, **kwargs):
        calls["n"] += 1
        raise KeyboardInterrupt

    mocker.patch("divi.qprog.ensemble.as_completed", side_effect=_side_effect)
    return calls


class TestLifecycleHookContract:
    def test_hooks_fire_in_documented_order(self, lifecycle_ensemble):
        ensemble = lifecycle_ensemble(n_rounds=2)
        ensemble.run()

        assert ensemble.calls == [
            "initial_state",
            "is_complete(0)",
            "create_programs(0)",
            "update_state(0)",
            "is_complete(1)",
            "create_programs(1)",
            "update_state(1)",
            "is_complete(2)",
        ]

    def test_state_propagates_into_each_round(self, lifecycle_ensemble):
        ensemble = lifecycle_ensemble(n_rounds=3)
        ensemble.run()

        assert ensemble.states_seen == [0, 1, 2]
        assert ensemble.workflow_state == 3

    def test_each_round_gets_a_fresh_program_map(self, lifecycle_ensemble):
        ensemble = lifecycle_ensemble(n_rounds=3)
        ensemble.run()

        assert ensemble.program_ids_per_round == [
            ["r1p0", "r1p1"],
            ["r2p0", "r2p1"],
            ["r3p0", "r3p1"],
        ]
        # Only the final round's programs remain addressable.
        assert sorted(ensemble.programs) == ["r3p0", "r3p1"]

    def test_convergence_sets_complete_stop_reason(self, lifecycle_ensemble):
        ensemble = lifecycle_ensemble(n_rounds=2)
        ensemble.run()

        assert ensemble.stop_reason == WorkflowStatus.COMPLETE
        assert len(ensemble.round_history) == 2

    def test_returns_self_for_chaining(self, lifecycle_ensemble):
        ensemble = lifecycle_ensemble(n_rounds=1)
        assert ensemble.run() is ensemble

    def test_default_hooks_run_exactly_one_round(self, dummy_simulator):
        """The one-shot built-in contract: no hook overrides needed."""
        ensemble = _OneShotEnsemble(
            backend=dummy_simulator, reporting_level=ReportingLevel.OFF
        )
        ensemble.run()

        assert ensemble.stop_reason == WorkflowStatus.COMPLETE
        assert len(ensemble.round_history) == 1
        assert ensemble.workflow_state is None


def _prepare_round(
    program, checkpoint_dir, interrupted_checkpoint=None, program_id="first"
):
    return ensemble_module._RoundCheckpointSession.prepare(
        checkpoint_config=CheckpointConfig(checkpoint_dir=checkpoint_dir),
        round_path=checkpoint_dir / "round_001",
        ensemble_type="tests.TerminalTestEnsemble",
        round_index=1,
        ensemble_state={"value": 0},
        programs=[program],
        child_recovery_states=[
            ProgramRoundRecord(
                program_id=_encode_program_id(program_id),
                program_type="tests.TerminalTestProgram",
                circuit_count_at_round_start=0,
                run_time_at_round_start=0.0,
            )
        ],
        interrupted_checkpoint=interrupted_checkpoint,
    )


@dataclass
class _IterativeChild:
    program: Any
    path: Path
    record: ProgramRoundRecord
    checkpoint: Any


def _iterative_child(mocker, path, totals, at_round_start):
    """A spec'd VQA child whose iterative checkpoint reports ``totals``."""
    path.mkdir(parents=True)
    program = mocker.MagicMock(spec=VariationalQuantumAlgorithm)
    program._training_finished = False
    checkpoint = mocker.Mock(total_circuit_count=totals[0], total_run_time=totals[1])
    mocker.patch.object(
        type(program),
        "_load_checkpoint_state",
        create=True,
        return_value=(path, checkpoint),
    )
    record = ProgramRoundRecord(
        program_id=_encode_program_id(path.name),
        program_type="tests.IterativeChild",
        circuit_count_at_round_start=at_round_start[0],
        run_time_at_round_start=at_round_start[1],
    )
    return _IterativeChild(program, path, record, checkpoint)


def _iterative_session(children):
    return ensemble_module._RoundCheckpointSession(
        checkpoint_path_by_program={child.program: child.path for child in children},
        iterative_config_by_program={
            child.program: CheckpointConfig(checkpoint_dir=child.path)
            for child in children
        },
    )


class TestEnsembleCheckpointing:
    def test_interrupted_round_reuses_its_child_directory(
        self, dummy_simulator, tmp_path
    ):
        program = _TerminalTestProgram(
            0, [0], Event(), fail=False, backend=dummy_simulator
        )
        round_path = tmp_path / "round_001"

        first = _prepare_round(program, tmp_path)
        active = RoundCheckpoint.model_validate_json(
            (round_path / "round_start.json").read_text()
        )
        second = _prepare_round(program, tmp_path, active)

        assert first.checkpoint_path_by_program[program] == round_path / "program_000"
        assert second.checkpoint_path_by_program[program] == round_path / "program_000"
        assert (
            RoundCheckpoint.model_validate_json(
                (round_path / "round_start.json").read_text()
            )
            == active
        )

    def test_interrupted_round_names_the_mismatched_program(
        self, dummy_simulator, tmp_path
    ):
        program = _TerminalTestProgram(
            0, [0], Event(), fail=False, backend=dummy_simulator
        )
        _prepare_round(program, tmp_path)
        active = RoundCheckpoint.model_validate_json(
            (tmp_path / "round_001" / "round_start.json").read_text()
        )

        with pytest.raises(ValueError, match="Program slot 0 .* program_id="):
            _prepare_round(program, tmp_path, active, program_id="second")

    def test_fresh_run_rejects_a_checkpoint_root_from_an_older_run(
        self, lifecycle_ensemble, tmp_path
    ):
        first = lifecycle_ensemble(n_rounds=1)
        first.run(checkpoint_config=CheckpointConfig(checkpoint_dir=tmp_path))
        second = lifecycle_ensemble(n_rounds=1)

        with pytest.raises(
            RuntimeError, match="already contains an ensemble checkpoint"
        ):
            second.run(checkpoint_config=CheckpointConfig(checkpoint_dir=tmp_path))

        assert second.workflow_state is None
        assert second.round_history == ()

    def test_completed_round_restores_and_continues_cumulative_limit(
        self, lifecycle_ensemble, tmp_path
    ):
        original = lifecycle_ensemble(n_rounds=3)
        original.run(
            max_rounds=1,
            checkpoint_config=CheckpointConfig(checkpoint_dir=tmp_path),
        )

        restored = lifecycle_ensemble(n_rounds=3)
        assert restored.restore_state(tmp_path) is restored

        assert restored.workflow_state == 1
        assert restored._round_index == 1
        assert restored.round_history == original.round_history
        assert restored.total_circuit_count == original.total_circuit_count
        assert restored.stop_reason is None

        restored.run(max_rounds=2)

        assert restored.workflow_state == 2
        assert restored.stop_reason is WorkflowStatus.MAX_ROUNDS
        assert [record.number for record in restored.round_history] == [1, 2]
        assert restored.program_ids_per_round == [["r1p0", "r1p1"], ["r2p0", "r2p1"]]

    def test_completed_round_restore_leaves_the_output_state_applied(
        self, lifecycle_ensemble, tmp_path
    ):
        """Rebuilding the round's programs loads its input snapshot too; the
        output snapshot must be the last one applied, or a loader with side
        effects is left at the round's start."""
        lifecycle_ensemble(cls=_StatefulLoadEnsemble, n_rounds=3).run(
            max_rounds=1, checkpoint_config=CheckpointConfig(checkpoint_dir=tmp_path)
        )

        restored = lifecycle_ensemble(cls=_StatefulLoadEnsemble, n_rounds=3)
        restored.restore_state(tmp_path)

        assert restored.restored_from == "output_state"
        assert restored.workflow_state == 1

    def test_completed_round_restore_keeps_no_programs_when_the_state_fails_to_load(
        self, lifecycle_ensemble, tmp_path
    ):
        lifecycle_ensemble(cls=_StatefulLoadEnsemble, n_rounds=3).run(
            max_rounds=1, checkpoint_config=CheckpointConfig(checkpoint_dir=tmp_path)
        )
        restoring = lifecycle_ensemble(
            cls=_StatefulLoadEnsemble, n_rounds=3, fail_on_stem="output_state"
        )

        with pytest.raises(ValueError, match="output_state boom"):
            restoring.restore_state(tmp_path)

        assert restoring.programs == {}
        assert restoring.workflow_state is None

    def test_one_shot_completed_checkpoint_stays_complete(
        self, dummy_simulator, tmp_path
    ):
        original = _OneShotEnsemble(
            backend=dummy_simulator, reporting_level=ReportingLevel.OFF
        )
        original.run(checkpoint_config=CheckpointConfig(checkpoint_dir=tmp_path))

        restored = _OneShotEnsemble(
            backend=dummy_simulator, reporting_level=ReportingLevel.FULL
        )
        restored.restore_state(tmp_path).run()

        assert len(restored.round_history) == 1
        assert restored.stop_reason is WorkflowStatus.COMPLETE
        assert restored.reporting_level is ReportingLevel.FULL

    def test_failed_round_restores_its_input_and_reenters_same_round(
        self, lifecycle_ensemble, tmp_path
    ):
        failed = lifecycle_ensemble(n_rounds=2)
        failed._total_circuit_count = 100
        failed._total_run_time = 50.0
        _fail_reduction_of_round(failed, 2)
        with pytest.raises(RuntimeError, match="reduction boom"):
            failed.run(checkpoint_config=CheckpointConfig(checkpoint_dir=tmp_path))

        resumed = lifecycle_ensemble(n_rounds=2)
        resumed.restore_state(tmp_path)

        assert resumed.workflow_state == 1
        assert resumed._round_index == 1
        assert resumed.total_circuit_count == 115
        assert resumed.total_run_time == 65.5

        resumed.run()

        assert resumed.states_seen == [1]
        assert resumed.workflow_state == 2
        assert [record.number for record in resumed.round_history] == [1, 2]
        assert (tmp_path / "round_002" / "round_completion.json").is_file()

    def test_interrupted_restore_rejects_previous_round_from_another_ensemble(
        self, lifecycle_ensemble, tmp_path
    ):
        _interrupt_round_two(lifecycle_ensemble(n_rounds=2), tmp_path)

        previous_marker = tmp_path / "round_001" / "round_completion.json"
        previous = json.loads(previous_marker.read_text())
        previous["ensemble_type"] = "tests.OtherEnsemble"
        previous_marker.write_text(json.dumps(previous))

        with pytest.raises(ValueError, match="earlier completed round"):
            lifecycle_ensemble(n_rounds=2).restore_state(tmp_path)

    def test_restore_rejects_pre_materialized_programs(
        self, lifecycle_ensemble, tmp_path
    ):
        original = lifecycle_ensemble(n_rounds=1)
        original.run(checkpoint_config=CheckpointConfig(checkpoint_dir=tmp_path))
        target = lifecycle_ensemble(n_rounds=1)
        target.create_programs(0)

        with pytest.raises(RuntimeError, match="pre-materialized"):
            target.restore_state(tmp_path)

    def test_completed_marker_failure_records_failed_round(
        self, lifecycle_ensemble, tmp_path
    ):
        ensemble = lifecycle_ensemble(n_rounds=1)
        save_state = ensemble._save_workflow_checkpoint_state

        def fail_output(state, round_dir, stem):
            if stem == "output_state":
                raise OSError("disk full")
            return save_state(state, round_dir, stem)

        ensemble._save_workflow_checkpoint_state = fail_output

        with pytest.raises(OSError, match="disk full"):
            ensemble.run(checkpoint_config=CheckpointConfig(checkpoint_dir=tmp_path))

        assert ensemble.stop_reason is WorkflowStatus.FAILED
        assert ensemble.round_history[-1].status is WorkflowStatus.FAILED
        assert (tmp_path / "round_001" / "round_start.json").is_file()
        assert not (tmp_path / "round_001" / "round_completion.json").exists()

    def test_completed_child_is_not_reexecuted_after_round_failure(
        self, make_dummy_simulator, tmp_path
    ):
        tracker = [0, 0]
        first_backend = make_dummy_simulator(100)
        first = _TerminalTestEnsemble(first_backend, tracker, fail_second=True)
        config = CheckpointConfig(checkpoint_dir=tmp_path)

        with pytest.raises(RuntimeError, match="Ensemble execution failed"):
            first.run(checkpoint_config=config)

        marker = json.loads(
            (
                tmp_path / "round_001" / "program_000" / "program_completion.json"
            ).read_text()
        )
        assert "phase" not in marker
        assert marker["value"] == 10
        assert "payload" not in marker

        replacement_backend = make_dummy_simulator(100)
        restored = _TerminalTestEnsemble(
            replacement_backend, tracker, fail_second=False
        ).restore_state(tmp_path)
        restored.run()

        assert tracker == [1, 2]
        assert restored.programs["first"].value == 10
        assert restored.programs["first"].restored_from == (
            tmp_path / "round_001" / "program_000"
        )
        assert restored.programs["second"].value == 11
        assert restored.programs["first"].backend is replacement_backend
        assert restored.total_circuit_count == 3
        assert restored.total_run_time == 2.0

    def test_corrupt_completed_child_restarts_from_round_input(
        self, dummy_simulator, tmp_path
    ):
        tracker = [0, 0]
        first = _TerminalTestEnsemble(dummy_simulator, tracker, fail_second=True)
        config = CheckpointConfig(checkpoint_dir=tmp_path)

        with pytest.raises(RuntimeError, match="Ensemble execution failed"):
            first.run(checkpoint_config=config)

        marker = tmp_path / "round_001" / "program_000" / "program_completion.json"
        marker.write_text("not json")
        restored = _TerminalTestEnsemble(
            dummy_simulator, tracker, fail_second=False
        ).restore_state(tmp_path)
        restored.run()

        assert tracker == [2, 2]
        assert restored.programs["first"].value == 10

    def test_completed_child_with_negative_accounting_restarts_before_mutation(
        self, dummy_simulator, tmp_path
    ):
        tracker = [0, 0]
        first = _TerminalTestEnsemble(dummy_simulator, tracker, fail_second=True)
        config = CheckpointConfig(checkpoint_dir=tmp_path)

        with pytest.raises(RuntimeError, match="Ensemble execution failed"):
            first.run(checkpoint_config=config)

        _edit_child_round_start(tmp_path, circuit_count_at_round_start=2)

        restored = _TerminalTestEnsemble(
            dummy_simulator, tracker, fail_second=False
        ).restore_state(tmp_path)
        restored.run()

        assert tracker == [2, 2]
        assert restored.programs["first"].value == 10

    def test_completed_child_with_zero_new_work_is_not_reexecuted(
        self, dummy_simulator, tmp_path
    ):
        tracker = [0, 0]
        first = _TerminalTestEnsemble(dummy_simulator, tracker, fail_second=True)
        with pytest.raises(RuntimeError, match="Ensemble execution failed"):
            first.run(checkpoint_config=CheckpointConfig(checkpoint_dir=tmp_path))
        _edit_child_round_start(
            tmp_path, circuit_count_at_round_start=1, run_time_at_round_start=0.5
        )

        restored = _TerminalTestEnsemble(
            dummy_simulator, tracker, fail_second=False
        ).restore_state(tmp_path)
        restored.run()

        assert tracker == [1, 2]

    def test_completed_round_rebuild_skips_programs_without_checkpoints(
        self, dummy_simulator, tmp_path
    ):
        tracker = [0, 0]
        _MixedCheckpointEnsemble(dummy_simulator, tracker, fail_second=False).run(
            checkpoint_config=CheckpointConfig(checkpoint_dir=tmp_path)
        )

        restored = _MixedCheckpointEnsemble(
            dummy_simulator, tracker, fail_second=False
        ).restore_state(tmp_path)

        assert restored.programs["second"].value == 11

    def test_fresh_run_reuses_an_empty_round_directory(
        self, lifecycle_ensemble, tmp_path
    ):
        (tmp_path / "round_001").mkdir()

        lifecycle_ensemble(n_rounds=1).run(
            checkpoint_config=CheckpointConfig(checkpoint_dir=tmp_path)
        )

        assert (tmp_path / "round_001" / "round_completion.json").is_file()

    def test_checkpointed_run_after_a_plain_run(self, lifecycle_ensemble, tmp_path):
        ensemble = lifecycle_ensemble(n_rounds=1)
        ensemble.run()

        ensemble.run(checkpoint_config=CheckpointConfig(checkpoint_dir=tmp_path))

        assert ensemble.stop_reason is WorkflowStatus.COMPLETE
        assert (tmp_path / "round_001" / "round_completion.json").is_file()

    @pytest.mark.parametrize("checkpointed", [False, True], ids=["plain", "fresh-dir"])
    def test_run_after_a_resumed_run_starts_a_fresh_workflow(
        self, lifecycle_ensemble, tmp_path, checkpointed
    ):
        source, fresh = tmp_path / "source", tmp_path / "fresh"
        _interrupt_round_two(lifecycle_ensemble(n_rounds=2), source)
        resumed = lifecycle_ensemble(n_rounds=2).restore_state(source)
        resumed.run()
        resumed.calls.clear()

        resumed.run(
            checkpoint_config=CheckpointConfig(
                checkpoint_dir=fresh if checkpointed else None
            )
        )

        assert resumed.calls[0] == "initial_state"
        assert resumed.stop_reason is WorkflowStatus.COMPLETE
        assert [record.number for record in resumed.round_history] == [1, 2]
        if checkpointed:
            assert (fresh / "round_002" / "round_completion.json").is_file()

    def test_explicit_checkpoint_dir_overrides_the_restored_root(
        self, lifecycle_ensemble, tmp_path
    ):
        source, target = tmp_path / "source", tmp_path / "target"
        lifecycle_ensemble(n_rounds=2).run(
            max_rounds=1, checkpoint_config=CheckpointConfig(checkpoint_dir=source)
        )

        restored = lifecycle_ensemble(n_rounds=2).restore_state(source)
        restored.run(checkpoint_config=CheckpointConfig(checkpoint_dir=target))

        assert (target / "round_002" / "round_completion.json").is_file()
        assert not (source / "round_002").exists()

    def test_restore_targets_an_explicit_round(self, lifecycle_ensemble, tmp_path):
        lifecycle_ensemble(n_rounds=3).run(
            max_rounds=2, checkpoint_config=CheckpointConfig(checkpoint_dir=tmp_path)
        )

        restored = lifecycle_ensemble(n_rounds=3).restore_state(
            tmp_path, subdirectory="round_001"
        )

        assert restored.workflow_state == 1
        assert [record.number for record in restored.round_history] == [1]

    def test_restore_rejects_another_ensemble_type(
        self, lifecycle_ensemble, dummy_simulator, tmp_path
    ):
        lifecycle_ensemble(n_rounds=1).run(
            checkpoint_config=CheckpointConfig(checkpoint_dir=tmp_path)
        )
        other = _OneShotEnsemble(
            backend=dummy_simulator, reporting_level=ReportingLevel.OFF
        )

        with pytest.raises(ValueError, match="_LifecycleEnsemble"):
            other.restore_state(tmp_path)

    @pytest.mark.parametrize(
        "rebuild", [_rename_programs, _retype_programs], ids=["renamed", "retyped"]
    )
    def test_resume_rejects_a_round_rebuilt_with_other_programs(
        self, lifecycle_ensemble, tmp_path, rebuild
    ):
        _interrupt_round_two(lifecycle_ensemble(n_rounds=2), tmp_path)
        resumed = lifecycle_ensemble(n_rounds=2).restore_state(tmp_path)
        create_programs = resumed.create_programs

        def create_other_programs(state=None):
            create_programs(state)
            resumed.programs = rebuild(resumed)

        resumed.create_programs = create_other_programs

        with pytest.raises(ValueError, match="slot 0"):
            resumed.run()

    def test_completed_round_reloads_its_output_state_artifact(
        self, lifecycle_ensemble, tmp_path
    ):
        lifecycle_ensemble(cls=_ArtifactEnsemble, n_rounds=2).run(
            max_rounds=1, checkpoint_config=CheckpointConfig(checkpoint_dir=tmp_path)
        )

        restored = lifecycle_ensemble(cls=_ArtifactEnsemble, n_rounds=2).restore_state(
            tmp_path
        )

        round_path = tmp_path / "round_001"
        assert json.loads((round_path / "input_state.json").read_text()) == 0
        assert json.loads((round_path / "output_state.json").read_text()) == 1
        assert restored.states_seen == [0]
        assert restored.workflow_state == 1

    def test_interrupted_round_reloads_its_input_state_artifact(
        self, lifecycle_ensemble, tmp_path
    ):
        _interrupt_round_two(
            lifecycle_ensemble(cls=_ArtifactEnsemble, n_rounds=2), tmp_path
        )

        resumed = lifecycle_ensemble(cls=_ArtifactEnsemble, n_rounds=2).restore_state(
            tmp_path
        )
        assert resumed.workflow_state == 1
        resumed.run()

        assert resumed.states_seen == [1]
        assert resumed.workflow_state == 2

    def test_round_input_is_snapshotted_once_before_materialisation(
        self, lifecycle_ensemble, tmp_path
    ):
        ensemble = lifecycle_ensemble(cls=_ArtifactEnsemble, n_rounds=1)

        ensemble.run(checkpoint_config=CheckpointConfig(checkpoint_dir=tmp_path))

        assert [
            call for call in ensemble.calls if call.startswith(("save", "create"))
        ] == ["save(input_state)", "create_programs(0)", "save(output_state)"]

    def test_iterative_child_accounting_is_validated_before_restore(
        self, tmp_path, mocker
    ):
        child = _iterative_child(mocker, tmp_path / "child", (1, 0.5), (2, 1.0))
        session = _iterative_session([child])

        session._recover([child.program], {child.program: child.record})

        child.program._restore_loaded_checkpoint.assert_not_called()
        assert child.path.is_dir()
        assert session.recovered_circuit_count == 0
        assert session.recovered_run_time == 0.0

    def test_iterative_children_resume_and_sum_their_accounting(self, tmp_path, mocker):
        children = [
            _iterative_child(mocker, tmp_path / "a", (5, 3.0), (2, 1.0)),
            _iterative_child(mocker, tmp_path / "b", (7, 4.5), (3, 0.5)),
        ]
        session = _iterative_session(children)

        session._recover(
            [child.program for child in children],
            {child.program: child.record for child in children},
        )

        for child in children:
            type(child.program)._load_checkpoint_state.assert_called_once_with(
                child.path
            )
            child.program._restore_loaded_checkpoint.assert_called_once_with(
                child.path, child.checkpoint
            )
        assert session.recovered_circuit_count == 7
        assert session.recovered_run_time == pytest.approx(6.0)
        assert not session.unfinalised_programs

    def test_completed_children_sum_their_recovered_accounting(self, tmp_path, mocker):
        children = [
            _iterative_child(mocker, tmp_path / name, (1, 1.0), (0, 0.0))
            for name in ("a", "b")
        ]
        for child in children:
            (child.path / PROGRAM_COMPLETION_FILE).write_text("{}")
        mocker.patch.object(
            ensemble_module,
            "_restore_completion",
            return_value=mocker.Mock(total_circuit_count=1, total_run_time=1.0),
        )
        session = _iterative_session(children)

        session._recover(
            [child.program for child in children],
            {child.program: child.record for child in children},
        )

        assert session.completed_programs == {child.program for child in children}
        assert session.recovered_circuit_count == 2
        assert session.recovered_run_time == 2.0

    def test_vqa_child_without_iterative_support_is_not_recovered(
        self, tmp_path, mocker
    ):
        child = _iterative_child(mocker, tmp_path / "child", (1, 0.5), (0, 0.0))
        session = ensemble_module._RoundCheckpointSession(
            checkpoint_path_by_program={child.program: child.path},
            iterative_config_by_program={},
        )

        session._recover([child.program], {child.program: child.record})

        type(child.program)._load_checkpoint_state.assert_not_called()

    def test_completed_child_checkpoint_failure_fails_round(
        self, dummy_simulator, tmp_path, mocker
    ):
        tracker = [0, 0]
        ensemble = _TerminalTestEnsemble(dummy_simulator, tracker, fail_second=False)
        mocker.patch.object(
            _TerminalTestProgram,
            "_make_checkpoint",
            side_effect=OSError("disk full"),
        )

        with pytest.raises(RuntimeError, match="Ensemble execution failed"):
            ensemble.run(checkpoint_config=CheckpointConfig(checkpoint_dir=tmp_path))

        assert ensemble.stop_reason is WorkflowStatus.FAILED


class TestAdaptiveRounds:
    """The pattern the loop exists for: each round shaped by the last one."""

    def test_update_state_reads_the_finished_round_programs(self, dummy_simulator):
        """``self.programs`` must still hold the round that just completed."""

        class _SummingEnsemble(_LifecycleEnsemble):
            def __init__(self, backend, **kwargs):
                super().__init__(backend, **kwargs)
                self.circuits_seen: list[int] = []

            def update_state(self, state):
                # Read results off the round that just finished.
                self.circuits_seen.append(
                    sum(p.circ_count for p in self.programs.values())
                )
                return state + 1

        ensemble = _SummingEnsemble(
            dummy_simulator, n_rounds=3, reporting_level=ReportingLevel.OFF
        )
        try:
            ensemble.run()
        finally:
            ensemble.reset()

        # Two programs per round contributing 10 + 5 circuits each round.
        assert ensemble.circuits_seen == [15, 15, 15]

    def test_program_count_may_vary_per_round(self, lifecycle_ensemble):
        """A shrinking ensemble must be accounted per round, not cached."""
        # state 0 -> 3 programs, state 1 -> 2, state 2 -> 1.
        ensemble = lifecycle_ensemble(
            n_rounds=3, programs_per_round=lambda state: 3 - state
        )
        ensemble.run()

        assert [len(ids) for ids in ensemble.program_ids_per_round] == [3, 2, 1]
        assert [r.program_count for r in ensemble.round_history] == [3, 2, 1]
        # SimpleTestProgram circuit counts are 10 for idx 0 and 5 for the rest.
        assert [r.circuit_count for r in ensemble.round_history] == [20, 15, 10]
        assert ensemble.total_circuit_count == 45

    def test_state_drives_the_next_round_program_ids(self, lifecycle_ensemble):
        """Program identities come from the state, proving it threads through."""
        ensemble = lifecycle_ensemble(n_rounds=3)
        ensemble.run()

        assert ensemble.program_ids_per_round[0] == ["r1p0", "r1p1"]
        assert ensemble.program_ids_per_round[-1] == ["r3p0", "r3p1"]


class TestMaxRoundsTermination:
    def test_max_rounds_stops_before_convergence(self, lifecycle_ensemble):
        ensemble = lifecycle_ensemble(n_rounds=10)

        assert ensemble.run(max_rounds=3) is ensemble
        assert ensemble.stop_reason == WorkflowStatus.MAX_ROUNDS
        assert len(ensemble.round_history) == 3
        # The unconverged state is preserved for inspection.
        assert ensemble.workflow_state == 3

    @pytest.mark.parametrize(
        "n_rounds, max_rounds",
        [
            pytest.param(1, 5, id="reached_first"),
            pytest.param(2, 2, id="exact_tie"),
        ],
    )
    def test_convergence_wins(self, lifecycle_ensemble, n_rounds, max_rounds):
        """is_complete is checked before the round cap, so a tie is COMPLETE."""
        ensemble = lifecycle_ensemble(n_rounds=n_rounds)
        ensemble.run(max_rounds=max_rounds)

        assert ensemble.stop_reason == WorkflowStatus.COMPLETE
        assert len(ensemble.round_history) == n_rounds

    def test_already_complete_initial_state_runs_zero_rounds(self, lifecycle_ensemble):
        """is_complete is evaluated before the first round, not after it."""
        ensemble = lifecycle_ensemble(n_rounds=0)
        ensemble.run()

        assert ensemble.stop_reason == WorkflowStatus.COMPLETE
        assert ensemble.round_history == ()
        assert not any(call.startswith("create_programs") for call in ensemble.calls)
        assert ensemble.total_circuit_count == 0

    @pytest.mark.parametrize("max_rounds", [0, -1])
    def test_non_positive_max_rounds_rejected(self, lifecycle_ensemble, max_rounds):
        ensemble = lifecycle_ensemble()
        with pytest.raises(ValueError, match="max_rounds must be >= 1"):
            ensemble.run(max_rounds=max_rounds)


class TestRoundAccounting:
    def test_history_records_per_round_deltas(self, lifecycle_ensemble):
        ensemble = lifecycle_ensemble(n_rounds=3)
        ensemble.run()

        assert [record.number for record in ensemble.round_history] == [1, 2, 3]
        for record in ensemble.round_history:
            assert record.program_count == 2
            assert record.status is WorkflowStatus.COMPLETE
            assert record.error is None
            assert record.circuit_count == _CIRCUITS_PER_ROUND
            assert record.run_time == pytest.approx(_RUNTIME_PER_ROUND)

    def test_totals_accumulate_across_rounds(self, lifecycle_ensemble):
        ensemble = lifecycle_ensemble(n_rounds=3)
        ensemble.run()

        assert ensemble.total_circuit_count == 3 * _CIRCUITS_PER_ROUND
        assert ensemble.total_run_time == pytest.approx(3 * _RUNTIME_PER_ROUND)

    def test_round_history_is_an_immutable_snapshot(self, lifecycle_ensemble):
        ensemble = lifecycle_ensemble(n_rounds=1)
        ensemble.run()
        history = ensemble.round_history
        (first_record,) = history

        ensemble.run()

        assert history == (first_record,)
        assert history[0] is first_record
        assert ensemble.round_history[0] is not first_record


class TestRepeatedWorkflowRuns:
    def test_second_run_executes_rounds_again(self, lifecycle_ensemble):
        """Regression: a stale round index used to make run() a silent no-op."""
        ensemble = lifecycle_ensemble(n_rounds=2)
        ensemble.run()
        assert len(ensemble.round_history) == 2

        ensemble.run()

        assert ensemble.stop_reason == WorkflowStatus.COMPLETE
        assert len(ensemble.round_history) == 2
        assert ensemble.total_circuit_count == 4 * _CIRCUITS_PER_ROUND

    def test_second_run_rematerializes_from_initial_state(self, lifecycle_ensemble):
        """Regression: the prior round's spent programs used to be reused as
        the new run's first round, skipping one create_programs()."""
        ensemble = lifecycle_ensemble(n_rounds=2)
        ensemble.run()
        ensemble.calls.clear()

        ensemble.run()

        # State restarts from initial_state(), totals do not.
        assert ensemble.calls[0] == "initial_state"
        assert ensemble.states_seen == [0, 1, 0, 1]
        assert ensemble.total_circuit_count == 4 * _CIRCUITS_PER_ROUND

    def test_caller_materialized_programs_used_for_first_round(
        self, lifecycle_ensemble
    ):
        """The legacy create_programs(); run() pattern skips one materialization."""
        ensemble = lifecycle_ensemble(n_rounds=2)
        ensemble.create_programs(0)
        ensemble.calls.clear()

        ensemble.run()

        # Round 1 reuses the prepared map, so only round 2 materializes.
        assert [call for call in ensemble.calls if call.startswith("create")] == [
            "create_programs(1)"
        ]
        assert len(ensemble.round_history) == 2

    def test_prepared_then_two_runs_rematerializes_the_second(self, lifecycle_ensemble):
        """The prepared map is consumed once, not reused by a later run()."""
        ensemble = lifecycle_ensemble(n_rounds=2)
        ensemble.create_programs(0)
        ensemble.run()
        ensemble.calls.clear()

        ensemble.run()

        assert ensemble.calls[0] == "initial_state"
        assert "create_programs(0)" in ensemble.calls
        assert ensemble.total_circuit_count == 4 * _CIRCUITS_PER_ROUND

    def test_spent_programs_are_not_treated_as_prepared(self, lifecycle_ensemble):
        """A standalone round consumes the pending map, so run() starts fresh."""
        ensemble = lifecycle_ensemble(n_rounds=1)
        ensemble.create_programs(0)
        ensemble.run_one_round(blocking=True)
        ensemble.calls.clear()

        ensemble.run()

        assert "create_programs(0)" in ensemble.calls
        assert ensemble.total_circuit_count == 2 * _CIRCUITS_PER_ROUND

    def test_reset_between_runs_clears_history_but_keeps_totals(
        self, lifecycle_ensemble
    ):
        ensemble = lifecycle_ensemble(n_rounds=2)
        ensemble.run()
        ensemble.reset()

        assert ensemble.round_history == ()
        assert ensemble.stop_reason is None
        assert ensemble.workflow_state is None
        assert ensemble.total_circuit_count == 2 * _CIRCUITS_PER_ROUND

        ensemble.run()
        assert len(ensemble.round_history) == 2
        assert ensemble.total_circuit_count == 4 * _CIRCUITS_PER_ROUND


class TestRoundFailureHandling:
    def test_failed_round_is_recorded_and_raises(self, lifecycle_ensemble):
        ensemble = lifecycle_ensemble(n_rounds=3, fail_on_round=2)

        with pytest.raises(RuntimeError, match="Ensemble execution failed"):
            ensemble.run()

        assert ensemble.stop_reason == WorkflowStatus.FAILED
        assert ensemble.workflow_state == 1
        assert [record.status for record in ensemble.round_history] == [
            WorkflowStatus.COMPLETE,
            WorkflowStatus.FAILED,
        ]
        failed = ensemble.round_history[-1]
        assert failed.number == 2
        assert failed.program_count == 2
        # FailingTestProgram raises before touching its counters.
        assert failed.circuit_count == 0
        assert failed.run_time == 0.0
        assert failed.error is not None
        assert failed.error.startswith("RuntimeError: Ensemble execution failed")
        # The program's own exception, not only the ensemble's wrapper.
        assert failed.error.endswith(" Caused by RuntimeError: program boom")

    def test_failed_materialisation_records_its_own_program_count(
        self, lifecycle_ensemble
    ):
        ensemble = lifecycle_ensemble(
            cls=_FailingMaterialisationEnsemble,
            n_rounds=3,
            programs_per_round=3,
            fail_on_round=2,
        )

        with pytest.raises(ValueError, match="materialisation boom"):
            ensemble.run()

        assert [
            (record.number, record.program_count, record.status)
            for record in ensemble.round_history
        ] == [(1, 3, WorkflowStatus.COMPLETE), (2, 0, WorkflowStatus.FAILED)]

    def test_failed_first_round_leaves_the_initial_state(self, lifecycle_ensemble):
        ensemble = lifecycle_ensemble(n_rounds=2, fail_on_round=1)

        with pytest.raises(RuntimeError, match="Ensemble execution failed"):
            ensemble.run()

        assert ensemble.workflow_state == 0

    def test_update_state_failure_is_recorded_as_a_failed_round(self, dummy_simulator):
        """A reducer bug fails the round even though its circuits ran."""

        class _BadReducer(_LifecycleEnsemble):
            def update_state(self, state):
                raise ValueError("reducer boom")

        ensemble = _BadReducer(
            dummy_simulator, n_rounds=3, reporting_level=ReportingLevel.OFF
        )
        with pytest.raises(ValueError, match="reducer boom"):
            ensemble.run()

        assert ensemble.stop_reason == WorkflowStatus.FAILED
        failed = ensemble.round_history[-1]
        assert failed.status is WorkflowStatus.FAILED
        assert "ValueError" in failed.error
        # The circuits did execute, so the delta is real.
        assert failed.circuit_count == _CIRCUITS_PER_ROUND

    def test_create_programs_failure_is_recorded_as_a_failed_round(
        self, dummy_simulator
    ):
        """A materialization bug must fail the round, not escape unrecorded."""

        class _HalfBuilder(_LifecycleEnsemble):
            fail_materialization = True

            def create_programs(self, state=None):
                super().create_programs(state)
                if self.fail_materialization:
                    raise ValueError("materialization boom")

        ensemble = _HalfBuilder(
            dummy_simulator, n_rounds=2, reporting_level=ReportingLevel.OFF
        )
        with pytest.raises(ValueError, match="materialization boom"):
            ensemble.run()

        assert ensemble.stop_reason == WorkflowStatus.FAILED
        failed = ensemble.round_history[-1]
        assert failed.number == 1
        assert failed.status is WorkflowStatus.FAILED
        assert "ValueError" in failed.error
        assert failed.circuit_count == 0

    def test_failed_materialization_is_not_reused_by_the_next_run(
        self, dummy_simulator
    ):
        """Its half-built map must not be adopted as the next run's round 1."""

        class _HalfBuilder(_LifecycleEnsemble):
            fail_materialization = True

            def create_programs(self, state=None):
                super().create_programs(state)
                if self.fail_materialization:
                    raise ValueError("materialization boom")

        ensemble = _HalfBuilder(
            dummy_simulator, n_rounds=1, reporting_level=ReportingLevel.OFF
        )
        with pytest.raises(ValueError):
            ensemble.run()

        ensemble.fail_materialization = False
        ensemble.calls.clear()
        ensemble.run()

        assert "create_programs(0)" in ensemble.calls
        assert ensemble.stop_reason == WorkflowStatus.COMPLETE

    def test_keyboard_interrupt_in_update_state_cancels_the_workflow(
        self, dummy_simulator
    ):
        """Ctrl-C in the classical reduction stops the loop instead of escaping."""

        class _InterruptedReducer(_LifecycleEnsemble):
            def update_state(self, state):
                raise KeyboardInterrupt

        ensemble = _InterruptedReducer(
            dummy_simulator, n_rounds=3, reporting_level=ReportingLevel.OFF
        )

        assert ensemble.run() is ensemble
        assert ensemble.stop_reason == WorkflowStatus.CANCELLED
        assert [r.status for r in ensemble.round_history] == [WorkflowStatus.CANCELLED]
        assert ensemble.round_history[0].program_count == 2
        # The interrupted round's results never reach the state.
        assert ensemble.workflow_state == 0
        assert len(ensemble.program_ids_per_round) == 1

        ensemble.run()

        assert len(ensemble.program_ids_per_round) == 2

    def test_failure_tears_down_round_machinery(self, lifecycle_ensemble):
        ensemble = lifecycle_ensemble(n_rounds=2, fail_on_round=1)

        with pytest.raises(RuntimeError):
            ensemble.run()

        assert ensemble._executor is None
        assert ensemble._coordinator is None
        assert ensemble._round_context is None
        assert all(
            program._progress_emitter is log_progress_event
            for program in ensemble.programs.values()
        )

    def test_keyboard_interrupt_aborts_the_workflow(self, lifecycle_ensemble, mocker):
        """Ctrl-C during a round must stop run(), not start the next round."""
        ensemble = lifecycle_ensemble(n_rounds=5)
        _interrupt_first_dispatch(mocker)

        assert ensemble.run(max_rounds=5) is ensemble
        assert ensemble.stop_reason == WorkflowStatus.CANCELLED
        assert [r.status for r in ensemble.round_history] == [WorkflowStatus.CANCELLED]
        assert ensemble.round_history[0].program_count == 2
        # Only round 1 was materialized; no further round started.
        assert ensemble.program_ids_per_round == [["r1p0", "r1p1"]]

    def test_cancelled_round_does_not_reduce_state(self, lifecycle_ensemble, mocker):
        """update_state must not fold partial results from a cancelled round."""
        ensemble = lifecycle_ensemble(n_rounds=5)
        _interrupt_first_dispatch(mocker)

        ensemble.run(max_rounds=5)

        assert not any(call.startswith("update_state") for call in ensemble.calls)
        assert ensemble.workflow_state == 0

    def test_second_interrupt_during_cleanup_still_tears_down(
        self, lifecycle_ensemble, mocker
    ):
        """A Ctrl-C while cancelling must not leave the round half-dismantled.

        The cleanup path runs from ``join()``'s own interrupt handler, so a
        second interrupt there escapes it. ``join()``'s ``finally`` must still
        shut the executor down and release the display, and ``run()`` must
        record the round as cancelled rather than propagating.
        """
        ensemble = lifecycle_ensemble(n_rounds=5)
        _interrupt_dispatch_then_cleanup(mocker)

        ensemble.run(max_rounds=5)

        assert ensemble.stop_reason == WorkflowStatus.CANCELLED
        assert [r.status for r in ensemble.round_history] == [WorkflowStatus.CANCELLED]
        assert ensemble._executor is None
        assert ensemble._coordinator is None
        assert all(
            program._progress_emitter is log_progress_event
            for program in ensemble.programs.values()
        )
        # The interrupted round's results never reach the state.
        assert ensemble.workflow_state == 0

    def test_ensemble_is_reusable_after_a_failed_round(self, lifecycle_ensemble):
        ensemble = lifecycle_ensemble(n_rounds=2, fail_on_round=1)
        with pytest.raises(RuntimeError):
            ensemble.run()

        ensemble._fail_on_round = None
        ensemble.reset()
        ensemble.run()

        assert ensemble.stop_reason == WorkflowStatus.COMPLETE
        assert len(ensemble.round_history) == 2


class TestRunOneRoundInteropWithRun:
    def test_single_round_dispatch_records_no_history(self, lifecycle_ensemble):
        ensemble = lifecycle_ensemble(n_rounds=5)
        ensemble.create_programs(0)

        ensemble.run_one_round(blocking=True)

        assert ensemble.total_circuit_count == _CIRCUITS_PER_ROUND
        assert ensemble.round_history == ()
        assert ensemble.stop_reason is None

    def test_batch_config_forwarded_through_run(self, lifecycle_ensemble, mocker):
        """Every round must receive the caller's BatchConfig, not the default."""
        ensemble = lifecycle_ensemble(n_rounds=2)
        config = BatchConfig(mode=BatchMode.OFF)
        spy = mocker.spy(ensemble, "run_one_round")

        ensemble.run(batch_config=config)

        assert spy.call_count == 2
        for call in spy.call_args_list:
            assert call.kwargs["batch_config"] is config
        assert ensemble.total_circuit_count == 2 * _CIRCUITS_PER_ROUND

    def test_run_rejects_reentry_while_running(self, lifecycle_ensemble):
        ensemble = lifecycle_ensemble(n_rounds=1)
        ensemble.create_programs(0)
        ensemble.run_one_round(blocking=False)
        try:
            with pytest.raises(RuntimeError, match="already being run"):
                ensemble.run()
        finally:
            ensemble.join()


def test_programs_assigned_before_run_are_its_first_round(dummy_simulator):
    ensemble = _OneShotEnsemble(
        backend=dummy_simulator, reporting_level=ReportingLevel.OFF
    )
    assigned = {"mine": SimpleTestProgram(3, 1.0, backend=dummy_simulator)}

    ensemble.programs = assigned
    assigned["stray"] = SimpleTestProgram(100, 1.0, backend=dummy_simulator)
    ensemble.run(max_rounds=1)

    assert ensemble.total_circuit_count == 3
    assert [record.program_count for record in ensemble.round_history] == [1]


class TestReportingLevels:
    """Visibility and standing-row lifecycles are reducer state."""

    def test_workflow_stage_after_dispatch_is_logged(self, mocker, dummy_simulator):
        ensemble = _OneShotEnsemble(
            backend=dummy_simulator,
            reporting_level=ReportingLevel.COMPACT,
        )
        events: list[ProgressEvent] = []
        ensemble._progress_emitter = events.append
        info = mocker.spy(ensemble_module.logger, "info")

        ensemble._emit_workflow_stage("Reducing samples")

        assert events == []
        info.assert_called_once_with("Reducing samples")

    @staticmethod
    def _scope_targets(ensemble, scope):
        session = ensemble._progress_session
        assert session is not None
        return [
            target for target in session.state.targets.values() if target.scope is scope
        ]

    def test_off_creates_no_session_but_keeps_history(self, lifecycle_ensemble):
        ensemble = lifecycle_ensemble(n_rounds=1, reporting_level=ReportingLevel.OFF)
        ensemble.run()

        assert ensemble._progress_session is None
        assert len(ensemble.round_history) == 1

    def test_compact_hides_program_rows(self, lifecycle_ensemble):
        ensemble = lifecycle_ensemble(
            n_rounds=1, reporting_level=ReportingLevel.COMPACT
        )
        ensemble.run()

        program_keys = self._scope_targets(ensemble, ProgressScope.PROGRAM)
        assert program_keys
        assert all(not target.visible for target in program_keys)

    @pytest.mark.parametrize("level", [ReportingLevel.FULL, "full"])
    def test_full_shows_program_rows(self, lifecycle_ensemble, level):
        """A string level shows rows exactly like the enum member."""
        ensemble = lifecycle_ensemble(n_rounds=1, reporting_level=level)
        ensemble.run()

        program_keys = self._scope_targets(ensemble, ProgressScope.PROGRAM)
        assert program_keys
        assert all(target.visible for target in program_keys)

    def test_full_shows_program_rows_for_a_large_ensemble(self, lifecycle_ensemble):
        """Row visibility is driven only by the level, not the program count."""
        ensemble = lifecycle_ensemble(
            n_rounds=1,
            programs_per_round=70,
            reporting_level=ReportingLevel.FULL,
        )
        ensemble.run(batch_config=BatchConfig(max_batch_size=8))

        program_keys = self._scope_targets(ensemble, ProgressScope.PROGRAM)
        assert len(program_keys) == 70
        assert all(target.visible for target in program_keys)

    @pytest.mark.parametrize("level", [ReportingLevel.COMPACT, ReportingLevel.FULL])
    def test_workflow_round_row_is_rendered(self, lifecycle_ensemble, level):
        ensemble = lifecycle_ensemble(n_rounds=1, reporting_level=level)
        ensemble.run(max_rounds=1)

        workflow_targets = self._scope_targets(ensemble, ProgressScope.WORKFLOW)
        assert len(workflow_targets) == 1
        target = workflow_targets[0]
        assert target.label == "Workflow"
        assert "Round 1/1" in target.detail
        assert "2 programs" in target.detail

    def test_round_row_reports_round_number_without_a_limit(self, lifecycle_ensemble):
        ensemble = lifecycle_ensemble(
            n_rounds=2, reporting_level=ReportingLevel.COMPACT
        )
        ensemble.run()

        target = self._scope_targets(ensemble, ProgressScope.WORKFLOW)[0]
        assert target.detail.startswith("Round 2 ")

    def test_round_row_marked_successful(self, lifecycle_ensemble):
        ensemble = lifecycle_ensemble(
            n_rounds=1, reporting_level=ReportingLevel.COMPACT
        )
        ensemble.run()

        target = self._scope_targets(ensemble, ProgressScope.WORKFLOW)[0]
        assert target.terminal_status is TerminalStatus.SUCCESS

    def test_round_row_marked_failed(self, lifecycle_ensemble):
        ensemble = lifecycle_ensemble(
            n_rounds=1, fail_on_round=1, reporting_level=ReportingLevel.COMPACT
        )
        with pytest.raises(RuntimeError):
            ensemble.run()

        target = self._scope_targets(ensemble, ProgressScope.WORKFLOW)[0]
        assert target.terminal_status is TerminalStatus.FAILED

    def test_no_round_row_for_standalone_run_one_round(self, lifecycle_ensemble):
        ensemble = lifecycle_ensemble(
            n_rounds=1, reporting_level=ReportingLevel.COMPACT
        )
        ensemble.create_programs(0)
        ensemble.run_one_round(blocking=True)

        assert self._scope_targets(ensemble, ProgressScope.WORKFLOW) == []

    def test_no_prep_row_without_merged_batching(self, lifecycle_ensemble):
        """Regression: the prep row can only advance under merged dispatch."""
        ensemble = lifecycle_ensemble(
            n_rounds=1, reporting_level=ReportingLevel.COMPACT
        )
        ensemble.run(batch_config=BatchConfig(mode=BatchMode.OFF))

        assert self._scope_targets(ensemble, ProgressScope.PREPARATION) == []

    def test_compact_reveals_a_failed_program_row(self, lifecycle_ensemble):
        ensemble = lifecycle_ensemble(
            n_rounds=1, fail_on_round=1, reporting_level=ReportingLevel.COMPACT
        )
        with pytest.raises(RuntimeError):
            ensemble.run()

        program_keys = self._scope_targets(ensemble, ProgressScope.PROGRAM)
        assert any(
            target.visible for target in program_keys
        ), "a failed program row must be revealed even in COMPACT"

    def test_workflow_and_preparation_use_typed_target_lifecycles(
        self, lifecycle_ensemble, mocker
    ):
        ensemble = lifecycle_ensemble(
            n_rounds=1, reporting_level=ReportingLevel.COMPACT
        )
        session = _RecordingSession(ProgressState(hide_successful_programs=True))
        mocker.patch.object(ProgressSession, "queued", return_value=session)

        ensemble.run(max_rounds=1)

        workflow_key = ("workflow", id(ensemble))
        preparation_key = ("preparation", id(ensemble))
        workflow_events = [
            event for event in session.events if event.progress_key == workflow_key
        ]
        preparation_events = [
            event for event in session.events if event.progress_key == preparation_key
        ]
        assert [event.kind for event in workflow_events] == [
            EventKind.REGISTER,
            EventKind.SHOW,
            EventKind.FINISH,
        ]
        assert preparation_events[0].kind is EventKind.REGISTER
        # These stub programs do not submit circuits. Coordinator coverage
        # separately proves first submissions emit preparation advances.
        assert preparation_events[1:-1] == []
        assert preparation_events[-1].kind is EventKind.FINISH
        assert preparation_events[-1].terminal_status is TerminalStatus.SUCCESS

    def test_default_level_is_compact(self, dummy_simulator):
        ensemble = _LifecycleEnsemble(backend=dummy_simulator, n_rounds=1)
        assert ensemble.reporting_level is ReportingLevel.COMPACT

    @pytest.mark.parametrize(
        "value, expected",
        [
            ("full", ReportingLevel.FULL),
            ("compact", ReportingLevel.COMPACT),
            ("off", ReportingLevel.OFF),
        ],
    )
    def test_string_level_is_coerced_to_the_enum(
        self, dummy_simulator, value, expected
    ):
        """A plain string must behave like the member; dispatch compares by identity."""
        ensemble = _LifecycleEnsemble(
            backend=dummy_simulator, n_rounds=1, reporting_level=value
        )
        assert ensemble.reporting_level is expected

    def test_unknown_level_is_rejected(self, dummy_simulator):
        with pytest.raises(ValueError, match="is not a valid ReportingLevel"):
            _LifecycleEnsemble(
                backend=dummy_simulator, n_rounds=1, reporting_level="verbose"
            )
