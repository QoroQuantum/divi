# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for the multi-round ``ProgramEnsemble`` workflow loop.

Covers the ``initial_state`` → ``create_programs`` → dispatch → ``update_state``
→ ``is_complete`` loop driven by :meth:`ProgramEnsemble.run`, and the
:class:`ReportingLevel` display behavior layered on top of it. Single-dispatch
mechanics live in ``test_ensemble_dispatch.py``.
"""

import pytest

import divi.qprog.ensemble as ensemble_module
from divi.qprog.ensemble import (
    BatchConfig,
    BatchMode,
    ProgramEnsemble,
    ReportingLevel,
    RoundRecord,
    WorkflowStatus,
)
from divi.reporting import TerminalStatus
from tests.qprog._helpers import FailingTestProgram, SimpleTestProgram

# Each round of _LifecycleEnsemble contributes these totals via
# SimpleTestProgram(10, 5.5) + SimpleTestProgram(5, 10.0).
_CIRCUITS_PER_ROUND = 15
_RUNTIME_PER_ROUND = 15.5


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
                FailingTestProgram(backend=self.backend, program_id=prog_id)
                if failing
                else SimpleTestProgram(
                    10 if idx == 0 else 5,
                    5.5 if idx == 0 else 10.0,
                    backend=self.backend,
                    program_id=prog_id,
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


class _OneShotEnsemble(ProgramEnsemble):
    """Overrides only ``create_programs``, leaving every other hook default."""

    def create_programs(self, state=None):
        super().create_programs()
        self.programs = {
            "prog1": SimpleTestProgram(
                10, 5.5, backend=self.backend, program_id="prog1"
            ),
            "prog2": SimpleTestProgram(
                5, 10.0, backend=self.backend, program_id="prog2"
            ),
        }

    def aggregate_results(self):
        return None


@pytest.fixture
def lifecycle_ensemble(dummy_simulator):
    """Factory for ``_LifecycleEnsemble``; progress output is off by default."""

    def _make(**kwargs):
        kwargs.setdefault("reporting_level", ReportingLevel.OFF)
        ensemble = _LifecycleEnsemble(backend=dummy_simulator, **kwargs)
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
        ensemble.run(max_rounds=3)

        assert ensemble.stop_reason == WorkflowStatus.MAX_ROUNDS
        assert len(ensemble.round_history) == 3
        # The unconverged state is preserved for inspection.
        assert ensemble.workflow_state == 3

    def test_convergence_wins_when_reached_first(self, lifecycle_ensemble):
        ensemble = lifecycle_ensemble(n_rounds=1)
        ensemble.run(max_rounds=5)

        assert ensemble.stop_reason == WorkflowStatus.COMPLETE
        assert len(ensemble.round_history) == 1

    def test_convergence_wins_on_an_exact_tie(self, lifecycle_ensemble):
        """is_complete is checked before the round cap, so a tie is COMPLETE."""
        ensemble = lifecycle_ensemble(n_rounds=2)
        ensemble.run(max_rounds=2)

        assert ensemble.stop_reason == WorkflowStatus.COMPLETE
        assert len(ensemble.round_history) == 2

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
        assert isinstance(history, tuple)
        assert isinstance(history[0], RoundRecord)
        with pytest.raises(AttributeError):
            history[0].number = 99


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
        assert "RuntimeError" in failed.error

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

    def test_failure_tears_down_round_machinery(self, lifecycle_ensemble):
        ensemble = lifecycle_ensemble(n_rounds=2, fail_on_round=1)

        with pytest.raises(RuntimeError):
            ensemble.run()

        assert ensemble._executor is None
        assert ensemble._coordinator is None
        assert ensemble._round_context is None

    def test_keyboard_interrupt_aborts_the_workflow(self, lifecycle_ensemble, mocker):
        """Ctrl-C during a round must stop run(), not start the next round."""
        ensemble = lifecycle_ensemble(n_rounds=5)
        _interrupt_first_dispatch(mocker)

        ensemble.run(max_rounds=5)

        assert ensemble.stop_reason == WorkflowStatus.CANCELLED
        assert [r.status for r in ensemble.round_history] == [WorkflowStatus.CANCELLED]
        # Only round 1 was materialized; no further round started.
        assert ensemble.program_ids_per_round == [["r1p0", "r1p1"]]

    def test_cancelled_round_does_not_reduce_state(self, lifecycle_ensemble, mocker):
        """update_state must not fold partial results from a cancelled round."""
        ensemble = lifecycle_ensemble(n_rounds=5)
        _interrupt_first_dispatch(mocker)

        ensemble.run(max_rounds=5)

        assert not any(call.startswith("update_state") for call in ensemble.calls)
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


class TestReportingLevels:
    """Row visibility and round-row rendering per :class:`ReportingLevel`."""

    def _program_rows(self, progress):
        """Per-program rows only — the prep row is also tagged ``program``."""
        return [
            task
            for task in progress.tasks
            if task.fields.get("row_kind") == "program"
            and task.fields.get("job_name", "").startswith("Program ")
        ]

    def _workflow_rows(self, progress):
        return [
            task for task in progress.tasks if task.fields.get("row_kind") == "workflow"
        ]

    def test_off_creates_no_display_but_keeps_history(self, lifecycle_ensemble):
        ensemble = lifecycle_ensemble(n_rounds=1, reporting_level=ReportingLevel.OFF)
        ensemble.run()

        assert ensemble._progress_bar is None
        assert ensemble._listener_thread is None
        assert len(ensemble.round_history) == 1

    def test_compact_hides_program_rows(self, lifecycle_ensemble):
        ensemble = lifecycle_ensemble(
            n_rounds=1, reporting_level=ReportingLevel.COMPACT
        )
        ensemble.run()

        program_rows = self._program_rows(ensemble._progress_bar)
        assert program_rows, "expected per-program rows to exist"
        assert all(not row.visible for row in program_rows)

    def test_full_shows_program_rows(self, lifecycle_ensemble):
        ensemble = lifecycle_ensemble(n_rounds=1, reporting_level=ReportingLevel.FULL)
        ensemble.run()

        program_rows = self._program_rows(ensemble._progress_bar)
        assert program_rows
        assert all(row.visible for row in program_rows)

    def test_full_shows_program_rows_for_a_large_ensemble(self, lifecycle_ensemble):
        """Row visibility is driven only by the level, not the program count."""
        ensemble = lifecycle_ensemble(
            n_rounds=1,
            programs_per_round=70,
            reporting_level=ReportingLevel.FULL,
        )
        ensemble.run(batch_config=BatchConfig(max_batch_size=8))

        program_rows = self._program_rows(ensemble._progress_bar)
        assert len(program_rows) == 70
        assert all(row.visible for row in program_rows)

    @pytest.mark.parametrize("level", [ReportingLevel.COMPACT, ReportingLevel.FULL])
    def test_workflow_round_row_is_rendered(self, lifecycle_ensemble, level):
        ensemble = lifecycle_ensemble(n_rounds=1, reporting_level=level)
        ensemble.run(max_rounds=1)

        workflow_rows = self._workflow_rows(ensemble._progress_bar)
        assert len(workflow_rows) == 1
        row = workflow_rows[0]
        assert row.fields["job_name"] == "Workflow"
        assert "Round 1/1" in row.fields["message"]
        assert "2 programs" in row.fields["message"]

    def test_round_row_reports_round_number_without_a_limit(self, lifecycle_ensemble):
        ensemble = lifecycle_ensemble(
            n_rounds=2, reporting_level=ReportingLevel.COMPACT
        )
        ensemble.run()

        # Only the last round's display survives teardown.
        row = self._workflow_rows(ensemble._progress_bar)[0]
        assert row.fields["message"].startswith("Round 2 ")

    def test_round_row_marked_successful(self, lifecycle_ensemble):
        ensemble = lifecycle_ensemble(
            n_rounds=1, reporting_level=ReportingLevel.COMPACT
        )
        ensemble.run()

        row = self._workflow_rows(ensemble._progress_bar)[0]
        assert row.fields["final_status"] == TerminalStatus.SUCCESS

    def test_round_row_marked_failed(self, lifecycle_ensemble):
        ensemble = lifecycle_ensemble(
            n_rounds=1, fail_on_round=1, reporting_level=ReportingLevel.COMPACT
        )
        with pytest.raises(RuntimeError):
            ensemble.run()

        row = self._workflow_rows(ensemble._progress_bar)[0]
        assert row.fields["final_status"] == TerminalStatus.FAILED

    def test_no_round_row_for_standalone_run_one_round(self, lifecycle_ensemble):
        ensemble = lifecycle_ensemble(
            n_rounds=1, reporting_level=ReportingLevel.COMPACT
        )
        ensemble.create_programs(0)
        ensemble.run_one_round(blocking=True)

        assert self._workflow_rows(ensemble._progress_bar) == []

    def test_no_prep_row_without_merged_batching(self, lifecycle_ensemble):
        """Regression: the prep row can only advance under merged dispatch."""
        ensemble = lifecycle_ensemble(
            n_rounds=1, reporting_level=ReportingLevel.COMPACT
        )
        ensemble.run(batch_config=BatchConfig(mode=BatchMode.OFF))

        prep_rows = [
            task
            for task in ensemble._progress_bar.tasks
            if task.fields.get("job_name") == "Submitting circuits"
        ]
        assert prep_rows == []

    def test_compact_reveals_a_failed_program_row(self, lifecycle_ensemble):
        ensemble = lifecycle_ensemble(
            n_rounds=1, fail_on_round=1, reporting_level=ReportingLevel.COMPACT
        )
        with pytest.raises(RuntimeError):
            ensemble.run()

        program_rows = self._program_rows(ensemble._progress_bar)
        assert any(
            row.visible for row in program_rows
        ), "a failed program row must be revealed even in COMPACT"

    def test_env_var_suppresses_display_at_every_level(
        self, lifecycle_ensemble, monkeypatch
    ):
        monkeypatch.setenv("DIVI_DISABLE_PROGRESS", "1")
        ensemble = lifecycle_ensemble(n_rounds=1, reporting_level=ReportingLevel.FULL)
        ensemble.run()

        assert ensemble._progress_bar is None
        assert len(ensemble.round_history) == 1

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
        with pytest.raises(ValueError):
            _LifecycleEnsemble(
                backend=dummy_simulator, n_rounds=1, reporting_level="verbose"
            )

    def test_string_full_shows_program_rows(self, lifecycle_ensemble):
        """Regression: a string level used to silently degrade to COMPACT."""
        ensemble = lifecycle_ensemble(n_rounds=1, reporting_level="full")
        ensemble.run()

        program_rows = self._program_rows(ensemble._progress_bar)
        assert program_rows
        assert all(row.visible for row in program_rows)
